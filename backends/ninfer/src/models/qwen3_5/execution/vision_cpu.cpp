#include "models/qwen3_5/execution/vision_cpu.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

constexpr float kNormEpsilon = 1.0e-6F;
// The two-axis rotary geometry of the Qwen3.5 tower: 72-wide heads rotate 36 pairs, the first 18
// by the patch row and the rest by its column, at 10000^(-2i/36) for i = pair % 18 -- the device
// kernel's constant table (ops/kernel/rope.cuh), so both encoders rotate by the same angles.
constexpr std::int32_t kHeadDim                   = 72;
constexpr std::int32_t kAxisPairs                 = 18;
constexpr std::int32_t kRotaryPairs               = 2 * kAxisPairs;
constexpr float kRopeInverseFrequency[kAxisPairs] = {
    1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F, 1.291549665e-01F,
    7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F, 1.668100537e-02F, 1.000000000e-02F,
    5.994842503e-03F, 3.593813664e-03F, 2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F,
    4.641588834e-04F, 2.782559402e-04F, 1.668100537e-04F,
};

float bf16_to_f32(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

std::uint16_t f32_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if (std::isnan(value)) { return static_cast<std::uint16_t>((bits >> 16U) | 0x40U); }
    const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

// A fork-join team kept for one encode, so the few hundred parallel sections of an item do not
// each create threads.
class Team {
public:
    explicit Team(unsigned threads) : size_(std::max(1U, threads)) {
        workers_.reserve(size_ - 1);
        for (unsigned index = 1; index < size_; ++index) {
            workers_.emplace_back([this, index] { serve(index); });
        }
    }

    ~Team() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (std::thread& worker : workers_) { worker.join(); }
    }

    Team(const Team&)            = delete;
    Team& operator=(const Team&) = delete;

    // Runs fn(begin, end) over [0, count) in contiguous slices of at least `grain`, one per member.
    template <class Fn>
    void run(std::size_t count, std::size_t grain, Fn&& fn) {
        if (count == 0) { return; }
        const std::size_t slices =
            std::max<std::size_t>(1, std::min<std::size_t>(size_, (count + grain - 1) / grain));
        const std::size_t step = (count + slices - 1) / slices;
        const auto slice       = [&](unsigned index) {
            const std::size_t begin = std::min(count, index * step);
            const std::size_t end   = std::min(count, begin + step);
            if (begin < end) { fn(begin, end); }
        };
        if (slices == 1) {
            slice(0);
            return;
        }
        {
            const std::lock_guard lock(mutex_);
            job_       = slice;
            active_    = static_cast<unsigned>(slices);
            remaining_ = static_cast<unsigned>(slices - 1);
            ++generation_;
        }
        wake_.notify_all();
        slice(0);
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
    }

private:
    void serve(unsigned index) {
        std::uint64_t seen = 0;
        for (;;) {
            std::function<void(unsigned)> job;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) { return; }
                seen = generation_;
                if (index >= active_) { continue; }
                job = job_;
            }
            job(index);
            {
                const std::lock_guard lock(mutex_);
                --remaining_;
            }
            done_.notify_one();
        }
    }

    unsigned size_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::function<void(unsigned)> job_;
    std::uint64_t generation_ = 0;
    unsigned active_          = 0;
    unsigned remaining_       = 0;
    bool stopping_            = false;
};

// Activations are feature-major: row r holds feature r across every token, so each op below
// streams contiguous token runs.
struct Matrix {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::vector<float> data;

    Matrix() = default;

    Matrix(std::int32_t r, std::int32_t c)
        : rows(r), cols(c), data(static_cast<std::size_t>(r) * static_cast<std::size_t>(c)) {}

    [[nodiscard]] float* row(std::int32_t r) {
        return data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    }

    [[nodiscard]] const float* row(std::int32_t r) const {
        return data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    }
};

constexpr std::int32_t kTileRows   = 4;
constexpr std::int32_t kTileTokens = 16;

// y rows [row, row + kTileRows) of W x + bias over one packed token panel ([k, kTileTokens],
// contiguous), accumulated as a register tile over the whole inner dimension.
void linear_tile(const float* w, std::int32_t k, const float* bias, const float* panel,
                 std::int32_t width, Matrix& y, std::int32_t row, std::int32_t token) {
    float acc[kTileRows][kTileTokens];
    for (int r = 0; r < kTileRows; ++r) {
        for (int t = 0; t < kTileTokens; ++t) { acc[r][t] = bias[row + r]; }
    }
    const float* rows[kTileRows];
    for (int r = 0; r < kTileRows; ++r) { rows[r] = w + static_cast<std::size_t>(row + r) * k; }
    for (std::int32_t inner = 0; inner < k; ++inner) {
        const float* xs = panel + static_cast<std::size_t>(inner) * kTileTokens;
        for (int r = 0; r < kTileRows; ++r) {
            const float weight = rows[r][inner];
            for (int t = 0; t < kTileTokens; ++t) { acc[r][t] += weight * xs[t]; }
        }
    }
    for (int r = 0; r < kTileRows; ++r) {
        float* out = y.row(row + r) + token;
        for (std::int32_t t = 0; t < width; ++t) { out[t] = acc[r][t]; }
    }
}

void linear_row(const float* w, std::int32_t k, const float* bias, const float* panel,
                std::int32_t width, Matrix& y, std::int32_t row, std::int32_t token) {
    float acc[kTileTokens];
    for (float& value : acc) { value = bias[row]; }
    const float* weights = w + static_cast<std::size_t>(row) * k;
    for (std::int32_t inner = 0; inner < k; ++inner) {
        const float* xs = panel + static_cast<std::size_t>(inner) * kTileTokens;
        for (int t = 0; t < kTileTokens; ++t) { acc[t] += weights[inner] * xs[t]; }
    }
    float* out = y.row(row) + token;
    for (std::int32_t t = 0; t < width; ++t) { out[t] = acc[t]; }
}

// y = W x + bias, W row-major [n, k], x feature-major [k, T], y feature-major [n, T]. Each member
// copies one token panel at a time into contiguous storage: rows of x sit a power-of-two stride
// apart at typical token counts, which would fold a whole panel onto a few cache sets.
void linear(const std::vector<float>& w, const std::vector<float>& bias, std::int32_t n,
            const Matrix& x, Matrix& y, Team& team) {
    const std::int32_t k = x.rows;
    if (w.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(k) ||
        bias.size() != static_cast<std::size_t>(n) || y.rows != n || y.cols != x.cols) {
        throw std::logic_error("CPU Vision: projection extents do not match");
    }
    const std::int32_t tokens = x.cols;
    const std::int32_t groups = (n + kTileRows - 1) / kTileRows;
    team.run(static_cast<std::size_t>(groups), 4, [&](std::size_t begin, std::size_t end) {
        std::vector<float> panel(static_cast<std::size_t>(k) * kTileTokens);
        for (std::int32_t token = 0; token < tokens; token += kTileTokens) {
            const std::int32_t width = std::min(kTileTokens, tokens - token);
            for (std::int32_t inner = 0; inner < k; ++inner) {
                float* target = panel.data() + static_cast<std::size_t>(inner) * kTileTokens;
                std::copy_n(x.row(inner) + token, width, target);
                std::fill(target + width, target + kTileTokens, 0.0F);
            }
            for (auto group = static_cast<std::int32_t>(begin);
                 group < static_cast<std::int32_t>(end); ++group) {
                const std::int32_t row = group * kTileRows;
                if (row + kTileRows <= n) {
                    linear_tile(w.data(), k, bias.data(), panel.data(), width, y, row, token);
                } else {
                    for (std::int32_t r = row; r < n; ++r) {
                        linear_row(w.data(), k, bias.data(), panel.data(), width, y, r, token);
                    }
                }
            }
        }
    });
}

// Per-token LayerNorm over the feature rows, eps inside the square root as on the device.
void layer_norm(const Matrix& x, const std::vector<float>& weight, const std::vector<float>& bias,
                Matrix& y, Team& team) {
    const std::int32_t features = x.rows;
    team.run(static_cast<std::size_t>(x.cols), 64, [&](std::size_t begin, std::size_t end) {
        const auto first = static_cast<std::int32_t>(begin);
        const auto count = static_cast<std::int32_t>(end - begin);
        std::vector<float> mean(static_cast<std::size_t>(count), 0.0F);
        std::vector<float> variance(static_cast<std::size_t>(count), 0.0F);
        for (std::int32_t r = 0; r < features; ++r) {
            const float* xs = x.row(r) + first;
            for (std::int32_t t = 0; t < count; ++t) { mean[t] += xs[t]; }
        }
        for (float& value : mean) { value /= static_cast<float>(features); }
        for (std::int32_t r = 0; r < features; ++r) {
            const float* xs = x.row(r) + first;
            for (std::int32_t t = 0; t < count; ++t) {
                const float centered = xs[t] - mean[t];
                variance[t] += centered * centered;
            }
        }
        for (float& value : variance) {
            value = 1.0F / std::sqrt(value / static_cast<float>(features) + kNormEpsilon);
        }
        for (std::int32_t r = 0; r < features; ++r) {
            const float* xs = x.row(r) + first;
            float* ys       = y.row(r) + first;
            for (std::int32_t t = 0; t < count; ++t) {
                ys[t] = (xs[t] - mean[t]) * variance[t] * weight[r] + bias[r];
            }
        }
    });
}

void add_into(Matrix& x, const Matrix& delta, Team& team) {
    team.run(x.data.size(), 1U << 16U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) { x.data[i] += delta.data[i]; }
    });
}

void gelu_tanh(Matrix& x, Team& team) {
    constexpr float kScale = 0.7978845608028654F; // sqrt(2 / pi)
    team.run(x.data.size(), 1U << 16U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const float value = x.data[i];
            x.data[i] = 0.5F * value *
                        (1.0F + std::tanh(kScale * (value + 0.044715F * value * value * value)));
        }
    });
}

void gelu_exact(Matrix& x, Team& team) {
    constexpr float kInverseSqrt2 = 0.7071067811865476F;
    team.run(x.data.size(), 1U << 16U, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const float value = x.data[i];
            x.data[i]         = 0.5F * value * (1.0F + std::erf(value * kInverseSqrt2));
        }
    });
}

// Rotates the query and key rows of every head in place: element i of a head pairs with i + 36,
// by the patch row for i < 18 and by its column otherwise.
void rope(Matrix& qkv, std::int32_t hidden, std::int32_t heads,
          const std::vector<std::int32_t>& ids, Team& team) {
    const std::int32_t tokens = qkv.cols;
    std::vector<float> cosine(static_cast<std::size_t>(kRotaryPairs) * tokens);
    std::vector<float> sine(cosine.size());
    for (std::int32_t pair = 0; pair < kRotaryPairs; ++pair) {
        const std::int32_t axis = pair / kAxisPairs;
        const float frequency   = kRopeInverseFrequency[pair % kAxisPairs];
        for (std::int32_t t = 0; t < tokens; ++t) {
            const float angle =
                static_cast<float>(ids[static_cast<std::size_t>(axis) * tokens + t]) * frequency;
            cosine[static_cast<std::size_t>(pair) * tokens + t] = std::cos(angle);
            sine[static_cast<std::size_t>(pair) * tokens + t]   = std::sin(angle);
        }
    }
    // Queries occupy rows [0, hidden) and keys [hidden, 2 * hidden).
    team.run(static_cast<std::size_t>(2 * heads), 1, [&](std::size_t begin, std::size_t end) {
        for (auto head = static_cast<std::int32_t>(begin); head < static_cast<std::int32_t>(end);
             ++head) {
            const std::int32_t base = (head / heads) * hidden + (head % heads) * kHeadDim;
            for (std::int32_t pair = 0; pair < kRotaryPairs; ++pair) {
                float* first         = qkv.row(base + pair);
                float* second        = qkv.row(base + pair + kRotaryPairs);
                const float* cosines = cosine.data() + static_cast<std::size_t>(pair) * tokens;
                const float* sines   = sine.data() + static_cast<std::size_t>(pair) * tokens;
                for (std::int32_t t = 0; t < tokens; ++t) {
                    const float a = first[t];
                    const float b = second[t];
                    first[t]      = a * cosines[t] - b * sines[t];
                    second[t]     = b * cosines[t] + a * sines[t];
                }
            }
        }
    });
}

// Dense bidirectional attention inside each segment of `segment` consecutive tokens (one video
// frame group or one image), per head, over the packed query/key/value rows of `qkv`. Both passes
// walk the keys in blocks so a query block's scores stay in the first-level cache.
void segment_attention(const Matrix& qkv, std::int32_t hidden, std::int32_t heads,
                       std::int32_t segment, Matrix& out, Team& team) {
    constexpr std::int32_t kQueries = 32;
    constexpr std::int32_t kKeys    = 256;
    const std::int32_t tokens       = qkv.cols;
    const std::int32_t segments     = tokens / segment;
    const std::int32_t blocks       = (segment + kQueries - 1) / kQueries;
    const float scale       = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));
    const std::size_t tasks = static_cast<std::size_t>(heads) * segments * blocks;
    team.run(tasks, 1, [&](std::size_t begin, std::size_t end) {
        std::vector<float> scores(static_cast<std::size_t>(kQueries) * segment);
        std::vector<float> result(static_cast<std::size_t>(kQueries) * kHeadDim);
        for (std::size_t task = begin; task < end; ++task) {
            const auto head        = static_cast<std::int32_t>(task / (segments * blocks));
            const auto rest        = static_cast<std::int32_t>(task % (segments * blocks));
            const std::int32_t s0  = (rest / blocks) * segment;
            const std::int32_t q0  = s0 + (rest % blocks) * kQueries;
            const std::int32_t qn  = std::min(kQueries, s0 + segment - q0);
            const std::int32_t row = head * kHeadDim;
            for (std::int32_t k0 = 0; k0 < segment; k0 += kKeys) {
                const std::int32_t kn = std::min(kKeys, segment - k0);
                for (std::int32_t q = 0; q < qn; ++q) {
                    std::fill_n(scores.data() + static_cast<std::size_t>(q) * segment + k0, kn,
                                0.0F);
                }
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const float* queries = qkv.row(row + d) + q0;
                    const float* keys    = qkv.row(hidden + row + d) + s0 + k0;
                    for (std::int32_t q = 0; q < qn; ++q) {
                        const float coefficient = queries[q] * scale;
                        float* line = scores.data() + static_cast<std::size_t>(q) * segment + k0;
                        for (std::int32_t key = 0; key < kn; ++key) {
                            line[key] += coefficient * keys[key];
                        }
                    }
                }
            }
            for (std::int32_t q = 0; q < qn; ++q) {
                float* line   = scores.data() + static_cast<std::size_t>(q) * segment;
                const float m = *std::max_element(line, line + segment);
                float sum     = 0.0F;
                for (std::int32_t key = 0; key < segment; ++key) {
                    line[key] = std::exp(line[key] - m);
                    sum += line[key];
                }
                const float inverse = 1.0F / sum;
                for (std::int32_t key = 0; key < segment; ++key) { line[key] *= inverse; }
            }
            std::fill(result.begin(), result.end(), 0.0F);
            for (std::int32_t k0 = 0; k0 < segment; k0 += kKeys) {
                const std::int32_t kn = std::min(kKeys, segment - k0);
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const float* values = qkv.row(2 * hidden + row + d) + s0 + k0;
                    for (std::int32_t q = 0; q < qn; ++q) {
                        const float* line =
                            scores.data() + static_cast<std::size_t>(q) * segment + k0;
                        float lanes[8]{};
                        std::int32_t key = 0;
                        for (; key + 8 <= kn; key += 8) {
                            for (int lane = 0; lane < 8; ++lane) {
                                lanes[lane] += line[key + lane] * values[key + lane];
                            }
                        }
                        float sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                                    ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
                        for (; key < kn; ++key) { sum += line[key] * values[key]; }
                        result[static_cast<std::size_t>(q) * kHeadDim + d] += sum;
                    }
                }
            }
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                float* target = out.row(row + d) + q0;
                for (std::int32_t q = 0; q < qn; ++q) {
                    target[q] = result[static_cast<std::size_t>(q) * kHeadDim + d];
                }
            }
        }
    });
}

// Each encode spreads over every core, so concurrent sessions take turns rather than oversubscribe.
std::mutex& encode_turn() {
    static std::mutex turn;
    return turn;
}

} // namespace

std::vector<std::uint16_t> encode_vision_on_cpu(const CpuVisionWeights& weights,
                                                std::span<const std::uint16_t> patches,
                                                const VisionItemControl& control, unsigned threads,
                                                const std::atomic<bool>* cancelled) {
    const auto count   = static_cast<std::int32_t>(control.patch_count);
    const auto merged  = static_cast<std::int32_t>(control.merged_count);
    const auto hidden  = weights.hidden;
    const auto segment = control.segment_length;
    if (count <= 0 || merged <= 0 || count != merged * weights.merge_unit || segment <= 0 ||
        count % segment != 0 ||
        patches.size() != static_cast<std::size_t>(count) * weights.patch_width ||
        control.position_ids.size() != static_cast<std::size_t>(count) * 2 ||
        control.position_table_indices.size() != static_cast<std::size_t>(count) * 4 ||
        control.position_table_weights.size() != static_cast<std::size_t>(count) * 4) {
        throw std::invalid_argument("CPU Vision item does not match its control");
    }
    Team team(threads != 0 ? threads : std::max(1U, std::thread::hardware_concurrency()));
    const auto stop = [&] { return cancelled != nullptr && cancelled->load(); };

    Matrix input(weights.patch_width, count);
    team.run(static_cast<std::size_t>(count), 64, [&](std::size_t begin, std::size_t end) {
        for (std::size_t p = begin; p < end; ++p) {
            const std::uint16_t* source = patches.data() + p * weights.patch_width;
            for (std::int32_t f = 0; f < weights.patch_width; ++f) {
                input.row(f)[p] = bf16_to_f32(source[f]);
            }
        }
    });
    Matrix x(hidden, count);
    linear(weights.patch_embedding, weights.patch_embedding_bias, hidden, input, x, team);
    input = {};
    team.run(static_cast<std::size_t>(hidden), 16, [&](std::size_t begin, std::size_t end) {
        for (std::size_t h = begin; h < end; ++h) {
            float* xs = x.row(static_cast<std::int32_t>(h));
            for (std::int32_t p = 0; p < count; ++p) {
                float sum = 0.0F;
                for (int corner = 0; corner < 4; ++corner) {
                    const std::int32_t index = control.position_table_indices[p * 4 + corner];
                    if (index >= 0 && index < weights.position_rows) {
                        sum +=
                            weights
                                .position_embedding[static_cast<std::size_t>(index) * hidden + h] *
                            control.position_table_weights[p * 4 + corner];
                    }
                }
                xs[p] += sum;
            }
        }
    });

    Matrix normed(hidden, count);
    Matrix qkv(3 * hidden, count);
    Matrix attended(hidden, count);
    Matrix projected(hidden, count);
    Matrix up(weights.intermediate, count);
    for (const CpuVisionWeights::Layer& layer : weights.layers) {
        if (stop()) { return {}; }
        layer_norm(x, layer.norm1_weight, layer.norm1_bias, normed, team);
        linear(layer.qkv, layer.qkv_bias, 3 * hidden, normed, qkv, team);
        rope(qkv, hidden, weights.heads, control.position_ids, team);
        segment_attention(qkv, hidden, weights.heads, segment, attended, team);
        linear(layer.output, layer.output_bias, hidden, attended, projected, team);
        add_into(x, projected, team);
        layer_norm(x, layer.norm2_weight, layer.norm2_bias, normed, team);
        linear(layer.fc1, layer.fc1_bias, weights.intermediate, normed, up, team);
        gelu_tanh(up, team);
        linear(layer.fc2, layer.fc2_bias, hidden, up, projected, team);
        add_into(x, projected, team);
    }
    if (stop()) { return {}; }

    layer_norm(x, weights.merger_norm_weight, weights.merger_norm_bias, normed, team);
    // The merger reads merge_unit consecutive patches' features as one merged token: feature
    // c * hidden + h of token v is feature h of patch v * merge_unit + c.
    Matrix gathered(weights.merger_width(), merged);
    for (std::int32_t c = 0; c < weights.merge_unit; ++c) {
        for (std::int32_t h = 0; h < hidden; ++h) {
            const float* source = normed.row(h);
            float* target       = gathered.row(c * hidden + h);
            for (std::int32_t v = 0; v < merged; ++v) {
                target[v] = source[v * weights.merge_unit + c];
            }
        }
    }
    Matrix merger_hidden(weights.merger_width(), merged);
    linear(weights.merger_fc1, weights.merger_fc1_bias, weights.merger_width(), gathered,
           merger_hidden, team);
    gelu_exact(merger_hidden, team);
    Matrix result(weights.output_hidden, merged);
    linear(weights.merger_fc2, weights.merger_fc2_bias, weights.output_hidden, merger_hidden,
           result, team);

    std::vector<std::uint16_t> out(static_cast<std::size_t>(merged) * weights.output_hidden);
    for (std::int32_t o = 0; o < weights.output_hidden; ++o) {
        const float* source = result.row(o);
        for (std::int32_t v = 0; v < merged; ++v) {
            out[static_cast<std::size_t>(v) * weights.output_hidden + o] = f32_to_bf16(source[v]);
        }
    }
    return out;
}

CpuVisionSession::CpuVisionSession(std::shared_ptr<const CpuVisionWeights> weights)
    : weights_(std::move(weights)) {
    if (!weights_) { throw std::invalid_argument("CPU Vision session has no weights"); }
}

CpuVisionSession::~CpuVisionSession() {
    if (worker_.joinable()) {
        cancelled_.store(true, std::memory_order_release);
        worker_.join();
    }
}

void CpuVisionSession::submit_item(std::shared_ptr<const PreparedMediaPayload> payload,
                                   const VisionItemControl& control) {
    if (worker_.joinable()) { throw std::logic_error("CPU Vision item is already in flight"); }
    if (!payload) { throw std::invalid_argument("CPU Vision item has no patches"); }
    finished_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    slot_ ^= 1U;
    worker_ = std::thread([this, payload = std::move(payload), control, slot = slot_] {
        try {
            const std::lock_guard turn(encode_turn());
            const auto started = std::chrono::steady_clock::now();
            results_[slot] =
                encode_vision_on_cpu(*weights_, payload->span(), control, 0, &cancelled_);
            worker_seconds_ =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        } catch (...) { error_ = std::current_exception(); }
        finished_.store(true, std::memory_order_release);
    });
}

std::span<const std::byte> CpuVisionSession::complete_item() {
    if (!worker_.joinable()) { throw std::logic_error("no CPU Vision item is in flight"); }
    worker_.join();
    if (error_) { std::rethrow_exception(std::exchange(error_, nullptr)); }
    encode_seconds_ += worker_seconds_;
    return std::as_bytes(std::span<const std::uint16_t>(results_[slot_]));
}

std::span<const std::byte>
CpuVisionSession::encode_item(std::shared_ptr<const PreparedMediaPayload> payload,
                              const VisionItemControl& control) {
    submit_item(std::move(payload), control);
    return complete_item();
}

} // namespace ninfer::models::qwen3_5::execution
