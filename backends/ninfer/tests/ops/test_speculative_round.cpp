#include "ninfer/ops/speculative_round.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include <string_view>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

template <typename T>
void initialize(GuardedDeviceBuffer& buffer, const std::vector<T>& values) {
    buffer.copy_from_host(values.data(), values.size() * sizeof(T));
}

template <typename T>
std::vector<T> read(const GuardedDeviceBuffer& buffer, std::size_t count) {
    return from_device<T>(buffer.data(), count);
}

DeviceBuffer device_config(const ops::SamplingConfig& config) {
    return to_device(std::vector<ops::SamplingConfig>{config});
}

int prepare_verify_case(int k, int batch) {
    const int width = k + 1;
    std::vector<std::int32_t> anchors(batch), lengths(batch), extents(batch), drafts(k * batch);
    for (int b = 0; b < batch; ++b) {
        anchors[b] = 70000 + 11 * b;
        lengths[b] = 131072 + 1009 * b;
        for (int j = 0; j < k; ++j) drafts[b * k + j] = 37 + 257 * b + 7919 * j;
    }
    DeviceBuffer d_anchors = to_device(anchors), d_lengths = to_device(lengths),
                 d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device(extents);
    GuardedDeviceBuffer d_full(width * batch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_ids(d_full.bytes()), d_positions(d_full.bytes());
    Tensor a(d_anchors.p, DType::I32, {batch}), l(d_lengths.p, DType::I32, {batch});
    Tensor d(d_drafts.p, DType::I32, {k, batch}), e(d_extents.p, DType::I32, {batch});
    Tensor full(d_full.data(), DType::I32, {width, batch}),
        ids(d_ids.data(), DType::I32, {width, batch});
    Tensor positions(d_positions.data(), DType::I32, {width, batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] {
        ops::speculative_prepare_verify_inputs(a, d, l, e, full, positions, context.stream);
        ops::speculative_prepare_verify_ids(a, d, e, ids, context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (k == 15 && batch == 8) {
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase <= k; ++phase) {
        for (int b = 0; b < batch; ++b) extents[b] = (phase + 3 * b) % width;
        CUDA_CHECK(cudaMemcpyAsync(d_extents.p, extents.data(), d_extents.bytes,
                                   cudaMemcpyHostToDevice, context.stream));
        if (graph.ready())
            graph.launch(context.stream);
        else
            launch();
        context.synchronize();
        std::vector<std::int32_t> expected_ids(width * batch), expected_positions(width * batch);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < width; ++j) {
                expected_ids[b * width + j] =
                    j > 0 && j <= extents[b] ? drafts[b * k + j - 1] : anchors[b];
                expected_positions[b * width + j] = lengths[b] + std::min(j, extents[b]);
            }
        failures += verify_exact("verify ids", read<std::int32_t>(d_ids, expected_ids.size()),
                                 expected_ids);
        failures += verify_exact("verify full ids", read<std::int32_t>(d_full, expected_ids.size()),
                                 expected_ids);
        failures += verify_exact("verify positions",
                                 read<std::int32_t>(d_positions, expected_positions.size()),
                                 expected_positions);
        failures +=
            verify_exact("extents unchanged", from_device<std::int32_t>(d_extents, batch), extents);
    }
    failures +=
        verify_exact("anchors unchanged", from_device<std::int32_t>(d_anchors, batch), anchors);
    failures +=
        verify_exact("lengths unchanged", from_device<std::int32_t>(d_lengths, batch), lengths);
    failures += verify_exact("drafts unchanged", from_device<std::int32_t>(d_drafts, drafts.size()),
                             drafts);
    failures += d_full.verify_guards("verify full ids");
    failures += d_ids.verify_guards("verify ids");
    failures += d_positions.verify_guards("verify positions");
    return failures;
}

// Exact oracle: a flagged row takes every copy element, an unflagged row keeps its own. One graph
// capture replays every row mixture because the selection is read on the device.
int overlay_case(int k, int batch, bool sparse) {
    const int slots = ops::kSparseSpeculativeCandidates;
    const int plane = sparse ? slots * k * batch : 0;
    std::vector<std::int32_t> copy_drafts(k * batch), own_drafts(k * batch);
    std::vector<std::int32_t> copy_candidates(plane), own_candidates(plane);
    std::vector<float> copy_q(plane), own_q(plane);
    for (int i = 0; i < k * batch; ++i) {
        copy_drafts[i] = 100000 + 7 * i;
        own_drafts[i]  = 200000 + 13 * i;
    }
    for (int i = 0; i < plane; ++i) {
        copy_candidates[i] = 300000 + 3 * i;
        own_candidates[i]  = 400000 + 5 * i;
        copy_q[i]          = i % slots == 0 ? 1.0F : 0.0F;
        own_q[i]           = 0.25F + static_cast<float>(i) * 1.0e-4F;
    }
    std::vector<std::int32_t> rows(batch, 0);
    DeviceBuffer d_rows = to_device(rows), d_copy_drafts = to_device(copy_drafts);
    DeviceBuffer d_copy_candidates, d_copy_q;
    GuardedDeviceBuffer d_drafts(own_drafts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer d_candidates(std::max<std::size_t>(1, plane) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_q(std::max<std::size_t>(1, plane) * sizeof(float));
    if (sparse) {
        d_copy_candidates = to_device(copy_candidates);
        d_copy_q          = to_device(copy_q);
    }
    Tensor t_rows(d_rows.p, DType::I32, {batch});
    Tensor t_copy_drafts(d_copy_drafts.p, DType::I32, {k, batch});
    Tensor t_drafts(d_drafts.data(), DType::I32, {k, batch});
    Tensor t_copy_candidates, t_copy_q, t_candidates, t_q;
    if (sparse) {
        t_copy_candidates = Tensor(d_copy_candidates.p, DType::I32, {slots, k, batch});
        t_copy_q          = Tensor(d_copy_q.p, DType::FP32, {slots, k, batch});
        t_candidates      = Tensor(d_candidates.data(), DType::I32, {slots, k, batch});
        t_q               = Tensor(d_q.data(), DType::FP32, {slots, k, batch});
    }
    DeviceContext context;
    const auto launch = [&] {
        ops::speculative_overlay_copy_proposals(t_rows, t_copy_drafts, t_copy_candidates, t_copy_q,
                                                t_drafts, t_candidates, t_q, context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(context.stream, launch);
    graph.instantiate(definition);

    int failures = 0;
    for (int mask = 0; mask < (1 << batch); mask += batch == 8 ? 37 : 1) {
        d_drafts.copy_from_host(own_drafts.data(), own_drafts.size() * sizeof(std::int32_t));
        if (sparse) {
            d_candidates.copy_from_host(own_candidates.data(), plane * sizeof(std::int32_t));
            d_q.copy_from_host(own_q.data(), plane * sizeof(float));
        }
        for (int b = 0; b < batch; ++b) rows[b] = (mask >> b) & 1 ? (b + 1) : 0;
        CUDA_CHECK(cudaMemcpyAsync(d_rows.p, rows.data(), d_rows.bytes, cudaMemcpyHostToDevice,
                                   context.stream));
        graph.launch(context.stream);
        context.synchronize();
        std::vector<std::int32_t> expected_drafts     = own_drafts;
        std::vector<std::int32_t> expected_candidates = own_candidates;
        std::vector<float> expected_q                 = own_q;
        for (int b = 0; b < batch; ++b) {
            if (rows[b] == 0) continue;
            for (int j = 0; j < k; ++j) expected_drafts[b * k + j] = copy_drafts[b * k + j];
            for (int i = 0; sparse && i < slots * k; ++i) {
                expected_candidates[b * slots * k + i] = copy_candidates[b * slots * k + i];
                expected_q[b * slots * k + i]          = copy_q[b * slots * k + i];
            }
        }
        const std::string label = "overlay K=" + std::to_string(k) + " B=" + std::to_string(batch) +
                                  (sparse ? " sparse" : " tokens") +
                                  " mask=" + std::to_string(mask);
        failures +=
            verify_exact((label + " drafts").c_str(),
                         read<std::int32_t>(d_drafts, expected_drafts.size()), expected_drafts);
        if (sparse) {
            failures += verify_exact((label + " candidates").c_str(),
                                     read<std::int32_t>(d_candidates, expected_candidates.size()),
                                     expected_candidates);
            failures += verify_exact((label + " q").c_str(), read<float>(d_q, expected_q.size()),
                                     expected_q);
        }
    }
    failures +=
        verify_exact("overlay copy drafts unchanged",
                     from_device<std::int32_t>(d_copy_drafts, copy_drafts.size()), copy_drafts);
    failures += d_drafts.verify_guards("overlay drafts");
    failures += d_candidates.verify_guards("overlay candidates");
    failures += d_q.verify_guards("overlay q");
    return failures;
}

struct AcceptExpected {
    std::vector<std::int32_t> sampled;
    std::int32_t num_sampled;
    std::int32_t accepted;
    std::int32_t length;
    std::int32_t token;
};

constexpr int kSparsePhysicalRows = 248320;
constexpr int kSparseTokenDomain  = 248077;
constexpr int kSparseCandidates   = 16;
constexpr int kTargetCandidateCap = 20;

struct SparseAcceptSuite {
    const int kSparseDrafts;
    const int kSparseColumns;
    const int kSparseBatch;
    std::size_t observed_workspace = 0;
    std::vector<std::uint32_t> masks;

    SparseAcceptSuite(int drafts, int batch)
        : kSparseDrafts(drafts), kSparseColumns(drafts + 1), kSparseBatch(batch) {}

    std::size_t sparse_logit_index(int row, int column, int token) {
        return (static_cast<std::size_t>(row) * kSparseColumns + column) * kSparsePhysicalRows +
               token;
    }

    std::size_t sparse_candidate_index(int row, int draft, int candidate) {
        return (static_cast<std::size_t>(row) * kSparseDrafts + draft) * kSparseCandidates +
               candidate;
    }

    std::uint64_t oracle_splitmix64(std::uint64_t value) {
        value += 0x9E3779B97F4A7C15ull;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
        return value ^ (value >> 31);
    }

    float oracle_uniform(std::uint64_t seed, int position, int purpose) {
        std::uint64_t key = seed;
        key               = oracle_splitmix64(key ^
                                              (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                                 0xD1B54A32D192ED03ull));
        key               = oracle_splitmix64(
            key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) << 21));
        const std::uint32_t bits = static_cast<std::uint32_t>(key >> 40);
        return static_cast<float>(bits) * (1.0f / 16777216.0f);
    }

    std::uint64_t find_seed_for_uniform(int position, int purpose, float lower, float upper) {
        for (std::uint64_t seed = 1; seed < 1000000; ++seed) {
            const float value = oracle_uniform(seed, position, purpose);
            if (value > lower && value < upper) { return seed; }
        }
        throw std::runtime_error("failed to find deterministic sparse-accept seed");
    }

    struct TargetDistribution {
        std::vector<std::int32_t> ids;
        std::vector<double> probabilities;
    };

    TargetDistribution sparse_target_distribution(const std::vector<std::uint16_t>& logits, int row,
                                                  int column, const ops::SamplingConfig& config,
                                                  const std::vector<std::int32_t>& token_counts,
                                                  const std::vector<std::int32_t>& drafts) {
        const auto adjusted = [&](int token) {
            const int words = (kSparseTokenDomain + 31) / 32;
            if (!masks.empty() && (masks[(row * kSparseColumns + column) * words + token / 32] &
                                   (1U << (token % 32))) == 0) {
                return -std::numeric_limits<double>::infinity();
            }
            double value = bf16_to_f32(logits[sparse_logit_index(row, column, token)]);
            int count    = token_counts[static_cast<std::size_t>(row) * kSparseTokenDomain + token];
            for (int previous = 0; previous < column; ++previous) {
                if (drafts[static_cast<std::size_t>(row) * kSparseDrafts + previous] == token) {
                    ++count;
                }
            }
            if (count > 0) value -= config.presence_penalty;
            value -= config.frequency_penalty * static_cast<double>(count);
            return value;
        };
        const auto better = [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
        };

        if (!(config.temperature > 0.0f)) {
            std::pair<double, std::int32_t> best{-std::numeric_limits<double>::infinity(), 0};
            for (int token = 0; token < kSparseTokenDomain; ++token) {
                const std::pair<double, std::int32_t> candidate{adjusted(token), token};
                if (better(candidate, best)) best = candidate;
            }
            return {{best.second}, {1.0}};
        }

        int cap = kTargetCandidateCap;
        if (config.top_k > 0 && config.top_k < cap) cap = config.top_k;
        cap = std::min(cap, kSparseTokenDomain);
        std::vector<std::pair<double, std::int32_t>> ranked(
            static_cast<std::size_t>(kSparseTokenDomain));
        for (int token = 0; token < kSparseTokenDomain; ++token) {
            ranked[static_cast<std::size_t>(token)] = {adjusted(token), token};
        }
        std::partial_sort(ranked.begin(), ranked.begin() + cap, ranked.end(), better);
        ranked.resize(static_cast<std::size_t>(cap));

        std::vector<double> weights(static_cast<std::size_t>(cap));
        const double inverse_temperature = 1.0 / static_cast<double>(config.temperature);
        const double maximum             = ranked.front().first * inverse_temperature;
        double total                     = 0.0;
        for (int item = 0; item < cap; ++item) {
            weights[static_cast<std::size_t>(item)] = std::exp(
                ranked[static_cast<std::size_t>(item)].first * inverse_temperature - maximum);
            total += weights[static_cast<std::size_t>(item)];
        }

        const double min_p_threshold = config.min_p > 0.0f ? config.min_p * weights.front() : -1.0;
        const bool top_p_active      = config.top_p < 1.0f;
        const double top_p_target    = config.top_p * total;
        double cumulative            = 0.0;
        int support                  = 0;
        for (int item = 0; item < cap; ++item) {
            if (min_p_threshold >= 0.0 &&
                weights[static_cast<std::size_t>(item)] < min_p_threshold) {
                break;
            }
            cumulative += weights[static_cast<std::size_t>(item)];
            support = item + 1;
            if (top_p_active && cumulative >= top_p_target) break;
        }
        support            = std::max(1, support);
        double support_sum = 0.0;
        for (int item = 0; item < support; ++item) {
            support_sum += weights[static_cast<std::size_t>(item)];
        }

        TargetDistribution distribution;
        distribution.ids.resize(static_cast<std::size_t>(support));
        distribution.probabilities.resize(static_cast<std::size_t>(support));
        for (int item = 0; item < support; ++item) {
            distribution.ids[static_cast<std::size_t>(item)] =
                ranked[static_cast<std::size_t>(item)].second;
            distribution.probabilities[static_cast<std::size_t>(item)] =
                weights[static_cast<std::size_t>(item)] / support_sum;
        }
        return distribution;
    }

    double distribution_probability(const TargetDistribution& distribution, int token) {
        for (std::size_t item = 0; item < distribution.ids.size(); ++item) {
            if (distribution.ids[item] == token) return distribution.probabilities[item];
        }
        return 0.0;
    }

    double sparse_proposal_probability(const std::vector<std::int32_t>& candidate_ids,
                                       const std::vector<float>& proposal_q, int row, int draft,
                                       int token) {
        for (int candidate = 0; candidate < kSparseCandidates; ++candidate) {
            const std::size_t index = sparse_candidate_index(row, draft, candidate);
            if (candidate_ids[index] == token) return proposal_q[index];
        }
        return 0.0;
    }

    int sample_target_distribution(const TargetDistribution& distribution, double uniform) {
        double cumulative = 0.0;
        for (std::size_t item = 0; item < distribution.ids.size(); ++item) {
            cumulative += distribution.probabilities[item];
            if (uniform < cumulative) return distribution.ids[item];
        }
        return distribution.ids.back();
    }

    int sample_sparse_residual(const TargetDistribution& target,
                               const std::vector<std::int32_t>& candidate_ids,
                               const std::vector<float>& proposal_q, int row, int draft,
                               double uniform) {
        double mass = 0.0;
        for (std::size_t item = 0; item < target.ids.size(); ++item) {
            const double q = sparse_proposal_probability(candidate_ids, proposal_q, row, draft,
                                                         target.ids[item]);
            mass += std::max(target.probabilities[item] - q, 0.0);
        }
        if (!(mass > 0.0)) return target.ids.front();
        const double goal = uniform * mass;
        double cumulative = 0.0;
        int selected      = target.ids.back();
        for (std::size_t item = 0; item < target.ids.size(); ++item) {
            const double q = sparse_proposal_probability(candidate_ids, proposal_q, row, draft,
                                                         target.ids[item]);
            const double residual = std::max(target.probabilities[item] - q, 0.0);
            if (!(residual > 0.0)) continue;
            cumulative += residual;
            selected = target.ids[item];
            if (goal < cumulative) return selected;
        }
        return selected;
    }

    struct SparseExpected {
        std::vector<std::int32_t> licensed_tokens;
        std::vector<std::int32_t> licensed_counts;
        std::vector<std::int32_t> accepted;
        std::vector<std::int32_t> lengths;
        std::vector<std::int32_t> anchors;
    };

    SparseExpected sparse_accept_oracle(const std::vector<std::uint16_t>& logits,
                                        const std::vector<std::int32_t>& drafts,
                                        const std::vector<std::int32_t>& candidate_ids,
                                        const std::vector<float>& proposal_q,
                                        const std::vector<std::int32_t>& extents,
                                        const std::vector<std::int32_t>& initial_lengths,
                                        const std::vector<ops::SamplingConfig>& configs,
                                        const std::vector<std::int32_t>& token_counts) {
        SparseExpected expected{
            .licensed_tokens = std::vector<std::int32_t>(kSparseColumns * kSparseBatch, 0),
            .licensed_counts = std::vector<std::int32_t>(kSparseBatch),
            .accepted        = std::vector<std::int32_t>(kSparseBatch),
            .lengths         = initial_lengths,
            .anchors         = std::vector<std::int32_t>(kSparseBatch),
        };
        for (int row = 0; row < kSparseBatch; ++row) {
            const int extent = std::clamp(extents[static_cast<std::size_t>(row)], 0, kSparseDrafts);
            int accepted_count = 0;
            int terminal_token = 0;
            for (int column = 0; column <= extent; ++column) {
                const TargetDistribution target = sparse_target_distribution(
                    logits, row, column, configs[static_cast<std::size_t>(row)], token_counts,
                    drafts);
                if (column == extent) {
                    const double uniform =
                        oracle_uniform(configs[static_cast<std::size_t>(row)].seed,
                                       initial_lengths[static_cast<std::size_t>(row)] + extent + 1,
                                       ops::kSamplePurposeSpeculativeBonus);
                    terminal_token = sample_target_distribution(target, uniform);
                    break;
                }

                const int draft = drafts[static_cast<std::size_t>(row) * kSparseDrafts + column];
                if (!(configs[static_cast<std::size_t>(row)].temperature > 0.0f)) {
                    if (target.ids.front() == draft) {
                        ++accepted_count;
                        continue;
                    }
                    terminal_token = target.ids.front();
                    break;
                }

                const double p = distribution_probability(target, draft);
                const double q =
                    sparse_proposal_probability(candidate_ids, proposal_q, row, column, draft);
                const double uniform =
                    oracle_uniform(configs[static_cast<std::size_t>(row)].seed,
                                   initial_lengths[static_cast<std::size_t>(row)] + column + 1,
                                   ops::kSamplePurposeSpeculativeAccept);
                if (p >= q || uniform * q < p) {
                    ++accepted_count;
                    continue;
                }
                const double correction_uniform =
                    oracle_uniform(configs[static_cast<std::size_t>(row)].seed,
                                   initial_lengths[static_cast<std::size_t>(row)] + column + 1,
                                   ops::kSamplePurposeSpeculativeCorrection);
                terminal_token = sample_sparse_residual(target, candidate_ids, proposal_q, row,
                                                        column, correction_uniform);
                break;
            }

            const std::size_t row_base   = static_cast<std::size_t>(row) * kSparseColumns;
            const std::size_t draft_base = static_cast<std::size_t>(row) * kSparseDrafts;
            for (int item = 0; item < accepted_count; ++item) {
                expected.licensed_tokens[row_base + item] = drafts[draft_base + item];
            }
            expected.licensed_tokens[row_base + accepted_count]     = terminal_token;
            expected.licensed_counts[static_cast<std::size_t>(row)] = accepted_count + 1;
            expected.accepted[static_cast<std::size_t>(row)]        = accepted_count;
            expected.lengths[static_cast<std::size_t>(row)] += accepted_count + 1;
            expected.anchors[static_cast<std::size_t>(row)] = terminal_token;
        }
        return expected;
    }

    int execute_sparse_accept_case(
        const std::string& label, const std::vector<std::int32_t>& target_tokens,
        const std::vector<std::uint16_t>& logits, const std::vector<std::int32_t>& drafts,
        const std::vector<std::int32_t>& candidate_ids, const std::vector<float>& proposal_q,
        const std::vector<std::int32_t>& extents, const std::vector<std::int32_t>& initial_lengths,
        const std::vector<std::int32_t>& initial_anchors,
        const std::vector<ops::SamplingConfig>& host_configs,
        const std::vector<std::int32_t>& token_counts,
        ops::SpeculativeAcceptExecutionEnvelope envelope,
        const SparseExpected* expected_override = nullptr, bool mtp_onehot = false) {
        // The oracle derives the greedy terminal from the truncated logit distribution. The
        // non-finite fixture deliberately poisons that same logit, so the oracle cannot describe
        // it -- those cases state their expectation directly instead. Graph replay is skipped with
        // an override for the same reason: it re-runs the oracle against mutated extents.
        const SparseExpected expected =
            expected_override != nullptr
                ? *expected_override
                : sparse_accept_oracle(logits, drafts, candidate_ids, proposal_q, extents,
                                       initial_lengths, host_configs, token_counts);

        DeviceBuffer d_target_tokens                    = to_device(target_tokens);
        DeviceBuffer d_logits                           = to_device(logits);
        DeviceBuffer d_drafts                           = to_device(drafts);
        DeviceBuffer d_candidate_ids                    = to_device(candidate_ids);
        DeviceBuffer d_proposal_q                       = to_device(proposal_q);
        DeviceBuffer d_extents                          = to_device(extents);
        DeviceBuffer d_token_counts                     = to_device(token_counts);
        std::vector<ops::SamplingConfig> device_configs = host_configs;
        DeviceBuffer d_masks = to_device(masks.empty() ? std::vector<std::uint32_t>{~0U} : masks);
        if (!masks.empty()) {
            for (int row = 0; row < kSparseBatch; ++row) {
                device_configs[row].token_mask_stride = (kSparseTokenDomain + 31) / 32;
                device_configs[row].token_mask =
                    static_cast<const std::uint32_t*>(d_masks.p) +
                    row * kSparseColumns * device_configs[row].token_mask_stride;
            }
        }
        for (int row = 0; row < kSparseBatch; ++row) {
            device_configs[static_cast<std::size_t>(row)].token_counts =
                static_cast<std::int32_t*>(d_token_counts.p) +
                static_cast<std::size_t>(row) * kSparseTokenDomain;
        }
        DeviceBuffer d_configs = to_device(device_configs);

        GuardedDeviceBuffer d_lengths(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_anchors(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_licensed(kSparseColumns * kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_licensed_counts(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_accepted(kSparseBatch * sizeof(std::int32_t));
        initialize(d_lengths, initial_lengths);
        initialize(d_anchors, initial_anchors);
        d_licensed.fill(0xcd);
        d_licensed_counts.fill(0xef);
        d_accepted.fill(0xab);

        Tensor target_tensor(d_target_tokens.p, DType::I32, {kSparseColumns, kSparseBatch});
        Tensor logits_tensor(d_logits.p, DType::BF16,
                             {kSparsePhysicalRows, kSparseColumns, kSparseBatch});
        Tensor drafts_tensor(d_drafts.p, DType::I32, {kSparseDrafts, kSparseBatch});
        Tensor candidate_tensor(d_candidate_ids.p, DType::I32,
                                {kSparseCandidates, kSparseDrafts, kSparseBatch});
        Tensor q_tensor(d_proposal_q.p, DType::FP32,
                        {kSparseCandidates, kSparseDrafts, kSparseBatch});
        Tensor extent_tensor(d_extents.p, DType::I32, {kSparseBatch});
        Tensor lengths_tensor(d_lengths.data(), DType::I32, {kSparseBatch});
        Tensor anchors_tensor(d_anchors.data(), DType::I32, {kSparseBatch});
        Tensor licensed_tensor(d_licensed.data(), DType::I32, {kSparseColumns, kSparseBatch});
        Tensor licensed_counts_tensor(d_licensed_counts.data(), DType::I32, {kSparseBatch});
        Tensor accepted_tensor(d_accepted.data(), DType::I32, {kSparseBatch});
        const std::size_t workspace_bytes =
            mtp_onehot
                ? ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                      kSparseTokenDomain, kSparseDrafts, kSparseDrafts, kSparseBatch, kSparseBatch)
                : ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                      kSparseTokenDomain, envelope, kSparseDrafts, kSparseDrafts, kSparseBatch,
                      kSparseBatch);
        GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 1));
        WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
        const auto launch = [&](cudaStream_t stream) {
            if (mtp_onehot) {
                ops::speculative_accept_greedy_drafts(
                    target_tensor, logits_tensor, drafts_tensor, extent_tensor, lengths_tensor,
                    anchors_tensor, licensed_tensor, licensed_counts_tensor, accepted_tensor,
                    kSparseTokenDomain, static_cast<const ops::SamplingConfig*>(d_configs.p),
                    workspace, stream);
                return;
            }
            ops::speculative_accept_sparse_drafts(
                target_tensor, logits_tensor, drafts_tensor, candidate_tensor, q_tensor,
                extent_tensor, lengths_tensor, anchors_tensor, licensed_tensor,
                licensed_counts_tensor, accepted_tensor, kSparseTokenDomain,
                static_cast<const ops::SamplingConfig*>(d_configs.p), envelope, workspace, stream);
        };
        launch(nullptr);
        cuda_synchronize();

        const auto check_result = [&](const SparseExpected& wanted, const std::string& phase) {
            int checks = verify_exact((label + phase + " licensed tokens").c_str(),
                                      read<std::int32_t>(d_licensed, wanted.licensed_tokens.size()),
                                      wanted.licensed_tokens);
            checks += verify_exact((label + phase + " licensed counts").c_str(),
                                   read<std::int32_t>(d_licensed_counts, kSparseBatch),
                                   wanted.licensed_counts);
            checks += verify_exact((label + phase + " accepted drafts").c_str(),
                                   read<std::int32_t>(d_accepted, kSparseBatch), wanted.accepted);
            checks += verify_exact((label + phase + " provisional lengths").c_str(),
                                   read<std::int32_t>(d_lengths, kSparseBatch), wanted.lengths);
            checks += verify_exact((label + phase + " provisional anchors").c_str(),
                                   read<std::int32_t>(d_anchors, kSparseBatch), wanted.anchors);
            if (mtp_onehot) {
                auto committed_counts = token_counts;
                for (int row = 0; row < kSparseBatch; ++row) {
                    for (int col = 0; col < wanted.licensed_counts[row]; ++col) {
                        ++committed_counts[static_cast<std::size_t>(row) * kSparseTokenDomain +
                                           wanted.licensed_tokens[row * kSparseColumns + col]];
                    }
                }
                checks += verify_exact((label + phase + " licensed history only").c_str(),
                                       from_device<int>(d_token_counts, committed_counts.size()),
                                       committed_counts);
            }
            return checks;
        };
        int failures = check_result(expected, " eager");
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) ++failures;
        failures += scratch.verify_guards(label + " workspace");
        observed_workspace = std::max(observed_workspace, workspace.peak_used());
        if (expected_override == nullptr && (kSparseBatch == 1 || kSparseBatch == 8)) {
            cudaStream_t stream = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "sparse graph stream");
            {
                DecodeGraphDefinition definition;
                DecodeGraphExecutable executable;
                definition.capture(stream, [&] { launch(stream); });
                executable.instantiate(definition);
                for (int pass = 0; pass < 3; ++pass) {
                    auto next_extents = extents;
                    auto next_lengths = initial_lengths;
                    if (pass == 1)
                        for (int row = 0; row < kSparseBatch; ++row) {
                            next_extents[row] = (row + kSparseDrafts / 2) % (kSparseDrafts + 1);
                            next_lengths[row] += 4096;
                        }
                    const bool change_inputs =
                        !mtp_onehot && !envelope.all_rows_greedy_without_penalties &&
                        kSparseBatch == 8 &&
                        (kSparseDrafts == 1 || kSparseDrafts == 7 || kSparseDrafts == 15);
                    std::vector<std::uint16_t> replay_logits;
                    std::vector<float> replay_q;
                    if (change_inputs) {
                        replay_logits = logits;
                        replay_q      = proposal_q;
                        if (pass == 1) {
                            for (auto& value : replay_logits)
                                value = f32_to_bf16(2.0f * bf16_to_f32(value));
                            for (int row = 0; row < kSparseBatch; ++row)
                                if (host_configs[row].temperature > 0.0f)
                                    for (int column = 0; column < kSparseDrafts; ++column) {
                                        const int base = sparse_candidate_index(row, column, 0);
                                        int rank       = 0;
                                        while (rank < 16 &&
                                               candidate_ids[base + rank] !=
                                                   drafts[row * kSparseDrafts + column])
                                            ++rank;
                                        if (rank == 16)
                                            throw std::runtime_error("invalid proposal fixture");
                                        std::fill_n(replay_q.begin() + base, 16, 0.0f);
                                        replay_q[base + rank]            = 0.75f;
                                        replay_q[base + (rank + 1) % 16] = 0.25f;
                                    }
                        }
                        d_logits.copy_from_host(replay_logits.data(),
                                                replay_logits.size() * sizeof(std::uint16_t));
                        d_proposal_q.copy_from_host(replay_q.data(),
                                                    replay_q.size() * sizeof(float));
                    }
                    const auto& current_logits = change_inputs ? replay_logits : logits;
                    const auto& current_q      = change_inputs ? replay_q : proposal_q;
                    initialize(d_lengths, next_lengths);
                    initialize(d_anchors, initial_anchors);
                    if (mtp_onehot) {
                        d_token_counts.copy_from_host(token_counts.data(),
                                                      token_counts.size() * sizeof(int));
                    }
                    d_extents.copy_from_host(next_extents.data(),
                                             next_extents.size() * sizeof(int));
                    cuda_synchronize();
                    executable.launch(stream);
                    cuda_synchronize(stream);
                    const auto wanted = sparse_accept_oracle(current_logits, drafts, candidate_ids,
                                                             current_q, next_extents, next_lengths,
                                                             host_configs, token_counts);
                    failures += check_result(wanted, " replay " + std::to_string(pass));
                    if (change_inputs) {
                        failures += verify_exact(
                            "replay logits readonly",
                            from_device<std::uint16_t>(d_logits, current_logits.size()),
                            current_logits);
                        failures += verify_exact("replay q readonly",
                                                 from_device<float>(d_proposal_q, current_q.size()),
                                                 current_q);
                    }
                    failures += verify_exact("sparse extent readonly",
                                             from_device<int>(d_extents, next_extents.size()),
                                             next_extents);
                    failures += scratch.verify_guards(label + " replay workspace");
                    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes)
                        ++failures;
                }
            }
            cuda_check(cudaStreamDestroy(stream), "destroy sparse graph stream");
        }
        failures += verify_exact("sparse logits readonly",
                                 from_device<std::uint16_t>(d_logits, logits.size()), logits);
        failures +=
            verify_exact("sparse targets readonly",
                         from_device<int>(d_target_tokens, target_tokens.size()), target_tokens);
        failures += verify_exact("sparse drafts readonly",
                                 from_device<int>(d_drafts, drafts.size()), drafts);
        failures +=
            verify_exact("sparse candidate ids readonly",
                         from_device<int>(d_candidate_ids, candidate_ids.size()), candidate_ids);
        failures += verify_exact("sparse actual q readonly",
                                 from_device<float>(d_proposal_q, proposal_q.size()), proposal_q);
        const auto* config_bytes = reinterpret_cast<const std::uint8_t*>(device_configs.data());
        failures += verify_exact(
            "sparse configs readonly", from_device<std::uint8_t>(d_configs, d_configs.bytes),
            std::vector<std::uint8_t>(config_bytes, config_bytes + d_configs.bytes));
        if (!mtp_onehot) {
            failures += verify_exact((label + " token counts read-only").c_str(),
                                     from_device<std::int32_t>(d_token_counts, token_counts.size()),
                                     token_counts);
        }
        failures += d_lengths.verify_guards((label + " lengths guards").c_str());
        failures += d_anchors.verify_guards((label + " anchors guards").c_str());
        failures += d_licensed.verify_guards((label + " licensed guards").c_str());
        failures += d_licensed_counts.verify_guards((label + " licensed-count guards").c_str());
        failures += d_accepted.verify_guards((label + " accepted guards").c_str());
        return failures;
    }

    int sparse_greedy_direct_case(int pattern = 0, bool general = false, int fixed_reject = -1,
                                  int fixed_extent = -1) {
        std::vector<std::int32_t> targets(kSparseColumns * kSparseBatch),
            drafts(kSparseDrafts * kSparseBatch);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<int> ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch), lengths(kSparseBatch), anchors(kSparseBatch, -1);
        std::vector<float> q(ids.size(), 0.0f);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 3);
        for (int row = 0; row < kSparseBatch; ++row) {
            const int kind = (row + pattern) % 7;
            extents[row]   = kind == 0   ? kSparseDrafts
                             : kind == 1 ? 0
                             : kind == 5 ? -2
                             : kind == 6 ? kSparseDrafts + 3
                                         : kSparseDrafts;
            if (fixed_extent >= 0) extents[row] = fixed_extent;
            const int extent = std::clamp(extents[row], 0, kSparseDrafts);
            const int reject = fixed_reject >= 0 ? fixed_reject
                               : kind == 2       ? 0
                               : kind == 3       ? kSparseDrafts / 2
                               : kind == 4       ? kSparseDrafts - 1
                                                 : extent;
            lengths[row]     = 4096 + row * 13;
            for (int col = 0; col < kSparseColumns; ++col) {
                const int target                             = 100000 + row * 128 + col;
                targets[row * kSparseColumns + col]          = target;
                logits[sparse_logit_index(row, col, target)] = f32_to_bf16(20.0f);
                for (int v = kSparseTokenDomain; v < kSparsePhysicalRows; ++v)
                    logits[sparse_logit_index(row, col, v)] = f32_to_bf16(100.0f);
            }
            for (int col = 0; col < kSparseDrafts; ++col) {
                const int base = sparse_candidate_index(row, col, 0);
                for (int rank = 0; rank < kSparseCandidates; ++rank)
                    ids[base + rank] = 30000 + row * 1024 + col * 16 + rank;
                ids[base]                         = targets[row * kSparseColumns + col];
                const int rank                    = col == reject && reject < extent ? 15 : 0;
                drafts[row * kSparseDrafts + col] = ids[base + rank];
                q[base + rank]                    = 1.0f;
            }
        }
        return execute_sparse_accept_case("sparse greedy K=" + std::to_string(kSparseDrafts) +
                                              " B=" + std::to_string(kSparseBatch),
                                          targets, logits, drafts, ids, q, extents, lengths,
                                          anchors, configs, history, {!general});
    }

    // Regression for the gap CodePulse raised on PR #16: a greedy sparse round used to license
    // whatever token sat at the divergence column even when the verify pass had produced an
    // all-NaN row there, advancing the sequence with an arbitrary token instead of returning the
    // non-finite sentinel. `poison` selects which rows get the NaN column, so a mixed batch proves
    // the refusal is per row rather than a blanket one -- the unpoisoned rows must still commit
    // normally in the same launch. Runs on both greedy sparse routes: raw (the dedicated warp
    // kernel) and the multiblock group-finalize kernel's SparseProposal branch.
    // `penalized` selects the greedy routes that presence/frequency penalties force down the
    // group-finalize path: dense speculative_store_accept_result and sparse
    // speculative_sparse_warp_accept's greedy branch. Those pick the terminal from the penalized
    // top-k rather than reading target_tokens, and a NaN does not sort predictably, so the poison
    // there is the whole column -- which is what a diverged forward pass actually produces. The
    // raw route reads target_tokens directly, so poisoning that one logit is exact.
    // `poison_matched` moves the NaN off the divergence column and onto a column the round
    // *accepts*. Both sparse greedy routes read target_tokens directly, so the match still
    // succeeds -- the ids say "the target picked what the draft proposed" while the logits behind
    // that verdict are NaN, which is precisely the case a terminal-only guard waves through: it
    // licenses the accepted draft and then commits on a finite terminal further along. Only
    // meaningful for the non-penalized routes; the penalized ones select from the column's own
    // ranking, so poisoning a column there changes what is selected and the divergence moves.
    int sparse_non_finite_terminal_case(bool raw_greedy, int poison, bool penalized = false,
                                        bool poison_matched = false) {
        std::vector<std::int32_t> targets(kSparseColumns * kSparseBatch),
            drafts(kSparseDrafts * kSparseBatch);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<int> ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch, kSparseDrafts), lengths(kSparseBatch), anchors(kSparseBatch, -1);
        std::vector<float> q(ids.size(), 0.0f);
        // temperature 0 everywhere: this is the greedy contract, and the raw route requires it.
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 3);
        if (penalized) {
            // Uniform history, so the penalty shifts every candidate equally and the argmax of a
            // clean column is still its target. What changes is the route, not the answer.
            for (auto& cfg : configs) {
                cfg.presence_penalty  = 1.0f;
                cfg.frequency_penalty = 0.5f;
            }
        }

        SparseExpected expected{
            .licensed_tokens = std::vector<std::int32_t>(kSparseColumns * kSparseBatch, 0),
            .licensed_counts = std::vector<std::int32_t>(kSparseBatch),
            .accepted        = std::vector<std::int32_t>(kSparseBatch),
            .lengths         = std::vector<std::int32_t>(kSparseBatch),
            .anchors         = std::vector<std::int32_t>(kSparseBatch),
        };

        const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
        for (int row = 0; row < kSparseBatch; ++row) {
            // Diverge at a different column per row so the poisoned column is not always the same
            // one, and full acceptance (reject == kSparseDrafts) is covered too.
            const int reject = (row + poison) % (kSparseDrafts + 1);
            lengths[row]     = 4096 + row * 13;
            for (int col = 0; col < kSparseColumns; ++col) {
                const int target                             = 100000 + row * 128 + col;
                targets[row * kSparseColumns + col]          = target;
                logits[sparse_logit_index(row, col, target)] = f32_to_bf16(20.0f);
                for (int v = kSparseTokenDomain; v < kSparsePhysicalRows; ++v)
                    logits[sparse_logit_index(row, col, v)] = f32_to_bf16(100.0f);
            }
            for (int col = 0; col < kSparseDrafts; ++col) {
                const int base = sparse_candidate_index(row, col, 0);
                for (int rank = 0; rank < kSparseCandidates; ++rank)
                    ids[base + rank] = 30000 + row * 1024 + col * 16 + rank;
                ids[base]                         = targets[row * kSparseColumns + col];
                const int rank                    = col == reject ? 15 : 0;
                drafts[row * kSparseDrafts + col] = ids[base + rank];
                q[base + rank]                    = 1.0f;
            }
            // Poison every other row. The NaN goes on the exact logit the kernel tests.
            // poison_matched needs an accepted column to exist, so a row that diverges at 0 has
            // none and is left clean -- expected_refusal tracks that rather than assuming.
            const int matched_column = reject / 2;
            const bool poisoned = ((row & 1) == (poison & 1)) && (!poison_matched || reject > 0);
            if (poisoned) {
                const int column = poison_matched ? matched_column : reject;
                if (penalized) {
                    for (int v = 0; v < kSparseTokenDomain; ++v)
                        logits[sparse_logit_index(row, column, v)] = f32_to_bf16(quiet_nan);
                } else {
                    logits[sparse_logit_index(row, column,
                                              targets[row * kSparseColumns + column])] =
                        f32_to_bf16(quiet_nan);
                }
            }

            const std::size_t row_base = static_cast<std::size_t>(row) * kSparseColumns;
            if (poisoned) {
                // The sentinel shape from speculative_sparse_warp_store: one licensed token,
                // nothing accepted, and the length still advances by one so the caller's frontier
                // stays consistent.
                expected.licensed_tokens[row_base] = ops::kSamplerNonFiniteToken;
                expected.licensed_counts[row]      = 1;
                expected.accepted[row]             = 0;
                expected.anchors[row]              = ops::kSamplerNonFiniteToken;
                expected.lengths[row]              = lengths[row] + 1;
            } else {
                for (int item = 0; item < reject; ++item)
                    expected.licensed_tokens[row_base + item] = drafts[row * kSparseDrafts + item];
                expected.licensed_tokens[row_base + reject] =
                    targets[row * kSparseColumns + reject];
                expected.licensed_counts[row] = reject + 1;
                expected.accepted[row]        = reject;
                expected.anchors[row]         = targets[row * kSparseColumns + reject];
                expected.lengths[row]         = lengths[row] + reject + 1;
            }
        }
        return execute_sparse_accept_case(
            "sparse non-finite " + std::string(poison_matched ? "matched" : "terminal") +
                " K=" + std::to_string(kSparseDrafts) + " B=" + std::to_string(kSparseBatch) +
                (raw_greedy ? " raw" : " general") + (penalized ? " penalized" : "") + " p" +
                std::to_string(poison),
            targets, logits, drafts, ids, q, extents, lengths, anchors, configs, history,
            {raw_greedy}, &expected);
    }

    int generated_general_case(bool masked = false, bool one_hot = false, bool mtp_onehot = false,
                               int variant = 0) {
        masks.clear();
        std::vector<int> targets(kSparseColumns * kSparseBatch),
            drafts(kSparseDrafts * kSparseBatch);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<int> ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch), lengths(kSparseBatch), anchors(kSparseBatch, -1);
        std::vector<float> q(ids.size(), 0.0f);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 0);
        for (int row = 0; row < kSparseBatch; ++row) {
            const int kind        = (row + kSparseDrafts + variant) % 4;
            auto& cfg             = configs[row];
            cfg.temperature       = (kind == 1 || kind == 2) ? 0.0f : 0.75f;
            cfg.top_k             = kind == 3 ? 13 : 20;
            cfg.top_p             = kind == 3 ? 0.78f : 1.0f;
            cfg.min_p             = kind == 3 ? 0.3f : 0.0f;
            cfg.presence_penalty  = kind == 1 || kind == 3 ? 0.5f : 0.0f;
            cfg.frequency_penalty = kind == 1 || kind == 3 ? 0.125f : 0.0f;
            if (variant >= 4) {
                cfg.presence_penalty *= -1.0f;
                cfg.frequency_penalty *= -1.0f;
            }
            cfg.seed         = 10007 + 31 * row + 17 * kSparseDrafts;
            lengths[row]     = 9000 + 37 * row;
            const int extent = row == kSparseBatch - 1 ? kSparseDrafts
                               : row % 4 == 0          ? 0
                                                       : std::max(1, kSparseDrafts - row % 3);
            extents[row]     = extent;
            const int style  = (2 * row + kSparseDrafts + variant) % 5;
            const int reject = style == 0 || style == 4 ? extent
                               : style == 1             ? 0
                               : style == 2             ? extent / 2
                                                        : std::max(0, extent - 1);
            for (int col = 0; col < kSparseDrafts; ++col) {
                const int base = sparse_candidate_index(row, col, 0);
                for (int rank = 0; rank < 16; ++rank) {
                    ids[base + rank] = 10000 + row * 4096 + col * 32 + rank;
                    q[base + rank]   = cfg.temperature > 0.0f && !one_hot ? 1.0f / 16.0f
                                       : rank == 0                        ? 1.0f
                                                                          : 0.0f;
                    history[row * kSparseTokenDomain + ids[base + rank]] = (col + rank) % 3;
                }
                drafts[row * kSparseDrafts + col] = ids[base];
                if (!one_hot && kind == 0 && col == reject && reject < extent)
                    drafts[row * kSparseDrafts + col] = ids[base + 9];
            }
            for (int col = 0; col < kSparseColumns; ++col) {
                const int base = 10000 + row * 4096 + col * 32;
                if (col < reject && col < kSparseDrafts) {
                    targets[row * kSparseColumns + col] = drafts[row * kSparseDrafts + col];
                    logits[sparse_logit_index(row, col, targets[row * kSparseColumns + col])] =
                        f32_to_bf16(16.0f);
                } else {
                    for (int rank = 0; rank < 20; ++rank)
                        logits[sparse_logit_index(row, col, base + 1 + rank)] =
                            f32_to_bf16(2.0f - rank * 0.0625f);
                    targets[row * kSparseColumns + col] = base + 1;
                }
                for (int v = kSparseTokenDomain; v < kSparsePhysicalRows; ++v)
                    logits[sparse_logit_index(row, col, v)] = f32_to_bf16(100.0f);
            }
        }
        if (masked) {
            const int words = (kSparseTokenDomain + 31) / 32;
            masks.assign(kSparseColumns * kSparseBatch * words, 0);
            for (int row = 0; row < kSparseBatch; ++row) {
                for (int col = 0; col < kSparseColumns; ++col) {
                    auto allow = [&](int token) {
                        masks[(row * kSparseColumns + col) * words + token / 32] |= 1U
                                                                                    << (token % 32);
                    };
                    const int base = 10000 + row * 4096 + col * 32;
                    for (int rank = 1; rank <= 20; ++rank) {
                        if ((rank + row + col) % 3 == 0) { allow(base + rank); }
                    }
                    const int target = targets[row * kSparseColumns + col];
                    if (col != row % kSparseDrafts) { allow(target); }
                    // A forbidden raw winner must be removed BEFORE target truncation.
                    logits[sparse_logit_index(row, col, kSparseTokenDomain - 1)] =
                        f32_to_bf16(1000.0f);
                    targets[row * kSparseColumns + col] = kSparseTokenDomain - 1;
                }
            }
        }
        return execute_sparse_accept_case(
            std::string(masked ? "masked " : "") + (mtp_onehot ? "MTP onehot" : "sparse") +
                " general K=" + std::to_string(kSparseDrafts) +
                " B=" + std::to_string(kSparseBatch) + " variant=" + std::to_string(variant),
            targets, logits, drafts, ids, q, extents, lengths, anchors, configs, history, {false},
            nullptr, mtp_onehot);
    }

    int repeated_history_case(bool stochastic, bool mtp_onehot = false) {
        std::vector<int> targets(kSparseColumns * kSparseBatch, 100),
            drafts(kSparseDrafts * kSparseBatch, 100);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<int> ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch, kSparseDrafts), lengths(kSparseBatch, 7000),
            anchors(kSparseBatch, -1);
        std::vector<float> q(ids.size(), 0.0f);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 0);
        for (int row = 0; row < kSparseBatch; ++row) {
            auto& cfg                               = configs[row];
            cfg.temperature                         = stochastic ? 1.0f : 0.0f;
            cfg.top_k                               = 2;
            cfg.frequency_penalty                   = 1.0f;
            cfg.seed                                = 523 + row;
            history[row * kSparseTokenDomain + 100] = 2;
            for (int col = 0; col < kSparseDrafts; ++col) {
                const int base = sparse_candidate_index(row, col, 0);
                for (int rank = 0; rank < 16; ++rank) ids[base + rank] = 100 + rank;
                q[base] = 1.0f;
            }
            for (int col = 0; col < kSparseColumns; ++col) {
                logits[sparse_logit_index(row, col, 100)] = f32_to_bf16(5.0f);
                logits[sparse_logit_index(row, col, 101)] = f32_to_bf16(0.5f);
            }
        }
        return execute_sparse_accept_case(
            std::string(mtp_onehot ? "MTP onehot" : "sparse") + " repeated-history K=" +
                std::to_string(kSparseDrafts) + (stochastic ? " stochastic" : " greedy"),
            targets, logits, drafts, ids, q, extents, lengths, anchors, configs, history, {false},
            nullptr, mtp_onehot);
    }

    int sparse_general_mixed_case() {
        std::vector<std::int32_t> targets(kSparseColumns * kSparseBatch);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<std::int32_t> drafts(kSparseDrafts * kSparseBatch);
        std::vector<std::int32_t> candidate_ids(kSparseCandidates * kSparseDrafts * kSparseBatch);
        std::vector<float> proposal_q(candidate_ids.size(), 0.0f);
        std::vector<std::int32_t> extents{2, 1, 1, kSparseDrafts, 0, 3, 2, 4};
        for (auto& extent : extents) extent = std::min(extent, kSparseDrafts);
        std::vector<std::int32_t> lengths(kSparseBatch);
        std::vector<std::int32_t> anchors(kSparseBatch, -1);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<std::int32_t> token_counts(
            static_cast<std::size_t>(kSparseTokenDomain) * kSparseBatch, 0);

        for (int row = 0; row < kSparseBatch; ++row) {
            lengths[static_cast<std::size_t>(row)]             = 4000 + 31 * row;
            configs[static_cast<std::size_t>(row)].temperature = 1.0f;
            configs[static_cast<std::size_t>(row)].top_k       = 1;
            configs[static_cast<std::size_t>(row)].top_p       = 1.0f;
            configs[static_cast<std::size_t>(row)].seed        = 100 + row;
            for (int draft = 0; draft < kSparseDrafts; ++draft) {
                for (int candidate = 0; candidate < kSparseCandidates; ++candidate) {
                    candidate_ids[sparse_candidate_index(row, draft, candidate)] =
                        1000 + row * 2048 + draft * kSparseCandidates + candidate;
                }
                const std::size_t base = sparse_candidate_index(row, draft, 0);
                drafts[static_cast<std::size_t>(row) * kSparseDrafts + draft] = candidate_ids[base];
                proposal_q[base]                                              = 0.5f;
                proposal_q[base + 1]                                          = 0.5f;
            }

            const int extent = extents[static_cast<std::size_t>(row)];
            for (int column = 0; column < kSparseColumns; ++column) {
                const int token =
                    column < extent ? drafts[static_cast<std::size_t>(row) * kSparseDrafts + column]
                                    : 200000 + row * 8 + column;
                targets[static_cast<std::size_t>(row) * kSparseColumns + column] = token;
                logits[sparse_logit_index(row, column, token)] = f32_to_bf16(12.0f);
                if (column > extent) {
                    const int poison = 240000 + row * kSparseColumns + column;
                    logits[sparse_logit_index(row, column, poison)] = f32_to_bf16(24.0f);
                    targets[static_cast<std::size_t>(row) * kSparseColumns + column] = poison;
                }
            }
        }

        // A committed-history penalty changes row 0's first target argmax.
        configs[0].temperature      = 0.0f;
        configs[0].presence_penalty = 2.0f;
        const int row0_draft        = drafts[0];
        const int row0_alternative  = candidate_ids[sparse_candidate_index(0, 0, 1)];
        logits[sparse_logit_index(0, 0, row0_draft)]       = f32_to_bf16(5.0f);
        logits[sparse_logit_index(0, 0, row0_alternative)] = f32_to_bf16(4.0f);
        targets[0]                                         = row0_draft;
        token_counts[static_cast<std::size_t>(row0_draft)] = 1;

        // p(d)=1/4, q(d)=1/2, and u is inside (p,p/q): row 1 must accept by p/q.
        configs[1].top_k         = 4;
        const int previous_draft = candidate_ids[sparse_candidate_index(1, 0, 0)];
        logits[sparse_logit_index(1, 0, previous_draft)] = f32_to_bf16(-20.0f);
        candidate_ids[sparse_candidate_index(1, 0, 0)]   = kSparseTokenDomain - 1;
        drafts[kSparseDrafts]                            = kSparseTokenDomain - 1;
        targets[kSparseColumns] = candidate_ids[sparse_candidate_index(1, 0, 1)];
        for (int candidate = 0; candidate < 4; ++candidate) {
            const int token = candidate_ids[sparse_candidate_index(1, 0, candidate)];
            logits[sparse_logit_index(1, 0, token)] = f32_to_bf16(0.0f);
        }
        configs[1].seed = find_seed_for_uniform(lengths[1] + 1,
                                                ops::kSamplePurposeSpeculativeAccept, 0.30f, 0.45f);

        // Row 2's draft has p=0. Sparse q removes the two larger target masses, so candidate 3 is
        // the only positive residual correction.
        configs[2].top_k     = 3;
        const int row2_draft = candidate_ids[sparse_candidate_index(2, 0, 0)];
        const int row2_a     = candidate_ids[sparse_candidate_index(2, 0, 1)];
        const int row2_b     = candidate_ids[sparse_candidate_index(2, 0, 2)];
        const int row2_c     = candidate_ids[sparse_candidate_index(2, 0, 3)];
        logits[sparse_logit_index(2, 0, row2_draft)] = f32_to_bf16(-20.0f);
        logits[sparse_logit_index(2, 0, row2_a)]     = f32_to_bf16(3.0f);
        logits[sparse_logit_index(2, 0, row2_b)]     = f32_to_bf16(2.0f);
        logits[sparse_logit_index(2, 0, row2_c)]     = f32_to_bf16(1.0f);
        targets[2 * kSparseColumns]                  = row2_a;
        const std::size_t row2_q                     = sparse_candidate_index(2, 0, 0);
        std::fill_n(proposal_q.begin() + static_cast<std::ptrdiff_t>(row2_q), kSparseCandidates,
                    0.0f);
        proposal_q[row2_q]     = 0.05f;
        proposal_q[row2_q + 1] = 0.70f;
        proposal_q[row2_q + 2] = 0.25f;

        // Row 5 is raw greedy inside the general mixed-batch route.
        configs[5].temperature = 0.0f;

        return execute_sparse_accept_case(
            "sparse speculative general mixed B=8", targets, logits, drafts, candidate_ids,
            proposal_q, extents, lengths, anchors, configs, token_counts,
            ops::SpeculativeAcceptExecutionEnvelope{.all_rows_greedy_without_penalties = false});
    }
};

AcceptExpected accept_state_oracle(const std::vector<std::int32_t>& drafts, std::int32_t accepted,
                                   std::int32_t terminal_token, std::int32_t initial_length) {
    const int k = static_cast<int>(drafts.size());
    AcceptExpected expected{
        .sampled     = std::vector<std::int32_t>(static_cast<std::size_t>(k + 1), 0),
        .num_sampled = accepted + 1,
        .accepted    = accepted,
        .length      = initial_length + accepted + 1,
        .token       = terminal_token,
    };
    for (int i = 0; i < accepted; ++i) {
        expected.sampled[static_cast<std::size_t>(i)] = drafts[static_cast<std::size_t>(i)];
    }
    expected.sampled[static_cast<std::size_t>(accepted)] = terminal_token;
    return expected;
}

// The refusal shape from speculative_store_accept_result's non-finite branch: one licensed
// sentinel token, nothing accepted, length still advances by one so the caller's frontier stays
// consistent.
AcceptExpected non_finite_accept_oracle(int k, std::int32_t initial_length) {
    AcceptExpected expected{
        .sampled     = std::vector<std::int32_t>(static_cast<std::size_t>(k + 1), 0),
        .num_sampled = 1,
        .accepted    = 0,
        .length      = initial_length + 1,
        .token       = ops::kSamplerNonFiniteToken,
    };
    expected.sampled[0] = ops::kSamplerNonFiniteToken;
    return expected;
}

int execute_accept_case(const std::string& label, const std::vector<std::int32_t>& target_tokens,
                        const std::vector<std::uint16_t>& logits_bits, int physical_rows,
                        const std::vector<std::int32_t>& drafts, std::int32_t initial_length,
                        int token_domain, ops::SamplingConfig config,
                        const std::vector<std::int32_t>& initial_token_counts,
                        const AcceptExpected& expected) {
    const int k              = static_cast<int>(drafts.size());
    DeviceBuffer d_targets   = to_device(target_tokens);
    DeviceBuffer d_logits    = to_device(logits_bits);
    DeviceBuffer d_drafts    = to_device(drafts);
    DeviceBuffer d_counts    = to_device(initial_token_counts);
    config.token_counts      = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config    = device_config(config);
    const auto config_before = from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig));

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_token(sizeof(std::int32_t));
    GuardedDeviceBuffer d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_num(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    DeviceBuffer d_extent = to_device<std::int32_t>({k});
    initialize(d_length, std::vector<std::int32_t>{initial_length});
    initialize(d_token, std::vector<std::int32_t>{-1234567});
    d_sampled.fill(0x9d);
    initialize(d_num, std::vector<std::int32_t>{-11});
    initialize(d_accepted, std::vector<std::int32_t>{-13});

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {physical_rows, k + 1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor length(d_length.data(), DType::I32, {1});
    Tensor token(d_token.data(), DType::I32, {1});
    Tensor sampled(d_sampled.data(), DType::I32, {k + 1});
    Tensor num_sampled(d_num.data(), DType::I32, {1});
    Tensor accepted(d_accepted.data(), DType::I32, {1});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, 1, 1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::speculative_accept_greedy_drafts(
        targets, logits, draft_tensor, extent, length, token, sampled, num_sampled, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact((label + " sampled").c_str(), read<std::int32_t>(d_sampled, k + 1),
                                expected.sampled);
    failures += verify_exact((label + " num sampled").c_str(), read<std::int32_t>(d_num, 1),
                             {expected.num_sampled});
    failures += verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                             {expected.accepted});
    failures += verify_exact((label + " length").c_str(), read<std::int32_t>(d_length, 1),
                             {expected.length});
    failures +=
        verify_exact((label + " token").c_str(), read<std::int32_t>(d_token, 1), {expected.token});

    failures +=
        verify_exact((label + " target tokens unchanged").c_str(),
                     from_device<std::int32_t>(d_targets, target_tokens.size()), target_tokens);
    failures += verify_exact((label + " logits unchanged").c_str(),
                             from_device<std::uint16_t>(d_logits, logits_bits.size()), logits_bits);
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " config unchanged").c_str(),
                             from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig)),
                             config_before);

    auto expected_counts = initial_token_counts;
    // The non-finite sentinel branch returns before speculative_store_accept_result's
    // UpdateTokenCounts block, so counts must stay untouched -- indexing token_counts by
    // kSamplerNonFiniteToken (-1) would be out of bounds.
    if (expected.token != ops::kSamplerNonFiniteToken) {
        for (int i = 0; i < expected.num_sampled; ++i) {
            ++expected_counts[static_cast<std::size_t>(
                expected.sampled[static_cast<std::size_t>(i)])];
        }
    }
    failures +=
        verify_exact((label + " token counts").c_str(),
                     from_device<std::int32_t>(d_counts, expected_counts.size()), expected_counts);

    failures += d_length.verify_guards((label + " length guards").c_str());
    failures += d_token.verify_guards((label + " token guards").c_str());
    failures += d_sampled.verify_guards((label + " sampled guards").c_str());
    failures += d_num.verify_guards((label + " num guards").c_str());
    failures += d_accepted.verify_guards((label + " accepted guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int greedy_accept_case(int k, int accepted_count, int token_domain = 64) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i <= k; ++i) {
        targets[static_cast<std::size_t>(i)] = 3 + 2 * i;
        if (i < k) drafts[static_cast<std::size_t>(i)] = targets[static_cast<std::size_t>(i)];
    }
    if (accepted_count < k) {
        drafts[static_cast<std::size_t>(accepted_count)] =
            targets[static_cast<std::size_t>(accepted_count)] + 1;
    }
    const std::int32_t initial_length = 200 + k;
    const auto expected               = accept_state_oracle(
        drafts, accepted_count, targets[static_cast<std::size_t>(accepted_count)], initial_length);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(token_domain) * (k + 1));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<std::uint16_t>(0x3f00u + (i % 127u));
    }
    std::vector<std::int32_t> token_counts(token_domain);
    for (int i = 0; i < token_domain; ++i) token_counts[static_cast<std::size_t>(i)] = i % 5;
    return execute_accept_case("speculative greedy K=" + std::to_string(k) +
                                   " A=" + std::to_string(accepted_count),
                               targets, logits, token_domain, drafts, initial_length, token_domain,
                               ops::SamplingConfig{}, token_counts, expected);
}

int masked_greedy_drafts_case(int k, int domain, bool stochastic, int rejection) {
    const int words = (domain + 31) / 32;
    std::vector<std::uint32_t> mask(words * (k + 1), 0);
    std::vector<int> targets(k + 1, 0), drafts(k), counts(domain, 0);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(domain) * (k + 1),
                                      f32_to_bf16(1000.0f));
    for (int col = 0; col <= k; ++col) {
        const int allowed                                        = domain - 1 - col;
        mask[col * words + allowed / 32]                         = 1U << (allowed % 32);
        logits[static_cast<std::size_t>(col) * domain + allowed] = f32_to_bf16(-2.0f);
        if (col < k) { drafts[col] = col == rejection ? 0 : allowed; }
    }
    auto d_mask = to_device(mask);
    ops::SamplingConfig config;
    config.temperature       = stochastic ? 0.8f : 0.0f;
    config.top_k             = 20;
    config.top_p             = 1.0f;
    config.token_mask        = static_cast<const std::uint32_t*>(d_mask.p);
    config.token_mask_stride = words;
    const auto expected      = accept_state_oracle(drafts, rejection, domain - 1 - rejection, 256);
    return execute_accept_case("per-position grammar mask", targets, logits, domain, drafts, 256,
                               domain, config, counts, expected);
}

int deterministic_sampling_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 5;
    constexpr int accepted      = 2;
    const std::vector<std::int32_t> drafts{17, 7919, 65537, 131071, 200003};
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) targets[static_cast<std::size_t>(i)] = 101 + i;
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        const int winner =
            col < accepted ? drafts[static_cast<std::size_t>(col)] : correction + col - accepted;
        logits[base + static_cast<std::size_t>(winner)] = 20.0f;
        logits[base + token_domain]                     = 100.0f;
        logits[base + physical_rows - 1]                = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) logits_bits[i] = f32_to_bf16(logits[i]);

    const std::int32_t initial_length = 4093;
    const auto expected = accept_state_oracle(drafts, accepted, correction, initial_length);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    token_counts[static_cast<std::size_t>(drafts[0])]  = 3;
    token_counts[static_cast<std::size_t>(drafts[1])]  = 5;
    token_counts[static_cast<std::size_t>(correction)] = 7;

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    config.top_p       = 0.9f;
    config.min_p       = 0.5f;
    config.seed        = 0x123456789abcdef0ull;
    return execute_accept_case("speculative sampling deterministic support", targets, logits_bits,
                               physical_rows, drafts, initial_length, token_domain, config,
                               token_counts, expected);
}

int greedy_penalty_case(int token_domain) {
    constexpr int k = 2;
    const std::vector<std::int32_t> drafts{1, 1};
    const std::vector<std::int32_t> raw_targets{1, 1, 3};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0F);
    logits[1]                    = 5.0F;
    logits[token_domain + 1]     = 5.0F;
    logits[token_domain + 4]     = 4.5F;
    logits[2 * token_domain + 3] = 5.0F;
    std::vector<std::uint16_t> bits(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        bits[index] = f32_to_bf16(logits[index]);
    }

    ops::SamplingConfig config{};
    config.temperature      = 0.0F;
    config.presence_penalty = 1.0F;
    const std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    const auto expected = accept_state_oracle(drafts, 1, 4, 100);
    return execute_accept_case(
        "speculative greedy penalty token-domain=" + std::to_string(token_domain), raw_targets,
        bits, token_domain, drafts, 100, token_domain, config, counts, expected);
}

// Regression for the gap CodePulse raised on PR #18: the sparse suite below covers the sparse
// greedy non-finite guards, but nothing exercised the dense guards at
// speculative_sampling_group_finalize_kernel<false> -- reachable only when token_domain exceeds
// kSamplerTileItems, which routes execution off the single-block kernel above and onto the
// partial/group top-k pipeline. `poison` off reruns the ordinary accept/divergence shape (a
// cross-check that the harness change here does not itself alter a passing case); `poison` on
// makes the licensed terminal's own verify logit NaN and expects the refusal sentinel instead.
int greedy_multiblock_non_finite_case(int accepted_count, int token_domain, bool poison) {
    constexpr int k = 3;
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i <= k; ++i) {
        targets[static_cast<std::size_t>(i)] = 3 + 2 * i;
        if (i < k) drafts[static_cast<std::size_t>(i)] = targets[static_cast<std::size_t>(i)];
    }
    if (accepted_count < k) {
        drafts[static_cast<std::size_t>(accepted_count)] =
            targets[static_cast<std::size_t>(accepted_count)] + 1;
    }
    const std::int32_t initial_length = 500;
    const std::int32_t terminal       = targets[static_cast<std::size_t>(accepted_count)];
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0f);
    if (poison) {
        logits[static_cast<std::size_t>(accepted_count) * static_cast<std::size_t>(token_domain) +
               static_cast<std::size_t>(terminal)] = std::numeric_limits<float>::quiet_NaN();
    }
    std::vector<std::uint16_t> bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) bits[i] = f32_to_bf16(logits[i]);
    std::vector<std::int32_t> token_counts(static_cast<std::size_t>(token_domain), 0);
    const auto expected =
        poison ? non_finite_accept_oracle(k, initial_length)
               : accept_state_oracle(drafts, accepted_count, terminal, initial_length);
    return execute_accept_case(
        "speculative greedy multiblock no-penalty token-domain=" + std::to_string(token_domain) +
            " A=" + std::to_string(accepted_count) + " poison=" + std::to_string(poison), targets,
        bits, token_domain, drafts, initial_length, token_domain, ops::SamplingConfig{},
        token_counts, expected);
}

// Same regression, for the penalized dense multiblock branch fixed alongside its sparse
// counterpart in 7107528f. That route picks the terminal from the *penalized* top-k rather than
// reading target_tokens, and a NaN does not sort predictably against finite candidates, so poison
// the entire divergence column (index 1, where drafts[1] fails to survive the presence penalty)
// rather than one logit -- an all-NaN column is also what a diverged forward pass actually
// produces. This mirrors sparse_non_finite_terminal_case's penalized poisoning exactly.
int greedy_penalty_non_finite_case(int token_domain, bool poison) {
    constexpr int k = 2;
    const std::vector<std::int32_t> drafts{1, 1};
    const std::vector<std::int32_t> raw_targets{1, 1, 3};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0F);
    logits[1]                    = 5.0F;
    logits[token_domain + 1]     = 5.0F;
    logits[token_domain + 4]     = 4.5F;
    logits[2 * token_domain + 3] = 5.0F;
    if (poison) {
        for (int v = 0; v < token_domain; ++v) {
            logits[static_cast<std::size_t>(token_domain) + static_cast<std::size_t>(v)] =
                std::numeric_limits<float>::quiet_NaN();
        }
    }
    std::vector<std::uint16_t> bits(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        bits[index] = f32_to_bf16(logits[index]);
    }

    ops::SamplingConfig config{};
    config.temperature      = 0.0F;
    config.presence_penalty = 1.0F;
    const std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    const auto expected =
        poison ? non_finite_accept_oracle(k, 100) : accept_state_oracle(drafts, 1, 4, 100);
    return execute_accept_case(
        "speculative greedy penalty multiblock non-finite token-domain=" +
            std::to_string(token_domain) + " poison=" + std::to_string(poison),
        raw_targets, bits, token_domain, drafts, 100, token_domain, config, counts, expected);
}

int batched_sampling_workspace_stride_case() {
    constexpr int physical_rows = 257;
    constexpr int token_domain  = 257;
    constexpr int k             = 3;
    constexpr int batch         = 2;
    constexpr int columns       = k + 1;

    const std::vector<std::int32_t> drafts{10, 11, 12, 30, 31, 32};
    const std::vector<std::int32_t> winners{10, 20, 21, 22, 30, 31, 32, 33};
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns * batch,
                                      f32_to_bf16(-20.0f));
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < columns; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * columns + col) * physical_rows;
            logits[base + static_cast<std::size_t>(winners[row * columns + col])] =
                f32_to_bf16(20.0f);
        }
    }

    DeviceBuffer d_targets = to_device(winners);
    DeviceBuffer d_logits  = to_device(logits);
    DeviceBuffer d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device<std::int32_t>({k, k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({100, 200});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1, -1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    const std::vector<ops::SamplingConfig> configs{config, config};
    DeviceBuffer d_configs = to_device(configs);

    Tensor targets(d_targets.p, DType::I32, {columns, batch});
    Tensor logits_tensor(d_logits.p, DType::BF16, {physical_rows, columns, batch});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor lengths(d_lengths.p, DType::I32, {batch});
    Tensor anchors(d_anchors.p, DType::I32, {batch});
    Tensor licensed(d_licensed.p, DType::I32, {columns, batch});
    Tensor counts(d_counts.p, DType::I32, {batch});
    Tensor accepted(d_accepted.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch,
                                                                       batch);
    WorkspaceArena workspace(workspace_bytes);
    ops::speculative_accept_greedy_drafts(
        targets, logits_tensor, draft_tensor, extents, lengths, anchors, licensed, counts, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact("speculative sampling B=2 licensed",
                                from_device<std::int32_t>(d_licensed, columns * batch),
                                {10, 20, 0, 0, 30, 31, 32, 33});
    failures += verify_exact("speculative sampling B=2 counts",
                             from_device<std::int32_t>(d_counts, batch), {2, 4});
    failures += verify_exact("speculative sampling B=2 accepted",
                             from_device<std::int32_t>(d_accepted, batch), {1, 3});
    failures += verify_exact("speculative sampling B=2 lengths",
                             from_device<std::int32_t>(d_lengths, batch), {102, 204});
    failures += verify_exact("speculative sampling B=2 anchors",
                             from_device<std::int32_t>(d_anchors, batch), {20, 33});
    return failures;
}

int select_hidden_case(int rows, int columns, int accepted_value) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * columns);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            hidden[static_cast<std::size_t>(col) * rows + row] =
                static_cast<std::uint16_t>(0x0100u + ((col * 257 + row * 13) & 0x7fffu));
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows));
    std::copy_n(hidden.begin() + static_cast<std::ptrdiff_t>(accepted_value) * rows, rows,
                expected.begin());

    DeviceBuffer d_hidden   = to_device(hidden);
    DeviceBuffer d_accepted = to_device<std::int32_t>({accepted_value});
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(rows) * sizeof(std::uint16_t));
    d_out.fill(0xcd);
    Tensor hidden_tensor(d_hidden.p, DType::BF16, {rows, columns});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor out(d_out.data(), DType::BF16, {rows, 1});
    ops::speculative_select_accepted_hidden(hidden_tensor, accepted, out, nullptr);
    cuda_synchronize();

    const std::string label =
        "speculative select D=" + std::to_string(rows) + " A=" + std::to_string(accepted_value);
    int failures = verify_exact((label + " output").c_str(),
                                read<std::uint16_t>(d_out, expected.size()), expected);
    failures += verify_exact((label + " hidden unchanged").c_str(),
                             from_device<std::uint16_t>(d_hidden, hidden.size()), hidden);
    failures += verify_exact((label + " accepted unchanged").c_str(),
                             from_device<std::int32_t>(d_accepted, 1), {accepted_value});
    failures += d_out.verify_guards((label + " output guards").c_str());
    return failures;
}

int batched_select_hidden_case(int width, int batch) {
    constexpr int rows = 5120;
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * width * batch);
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = static_cast<std::uint16_t>(i * 257 + i / rows * 6113);
    std::vector<std::int32_t> selectors(batch);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * batch);
    DeviceBuffer input = to_device(hidden), indices = to_device(selectors);
    GuardedDeviceBuffer output(expected.size() * sizeof(std::uint16_t));
    output.fill(0xcd);
    Tensor h(input.p, DType::BF16, {rows, width, batch}), i(indices.p, DType::I32, {batch});
    Tensor o(output.data(), DType::BF16, {rows, batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] { ops::speculative_select_accepted_hidden(h, i, o, context.stream); };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (width == 16 && batch == 8) {
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    // Final commit N is positive here; a zero commit never issues this gather with selector -1.
    for (int phase = 0; phase < width; ++phase) {
        for (int b = 0; b < batch; ++b) {
            selectors[b] = (phase + 3 * b) % width;
            std::copy_n(hidden.begin() + static_cast<std::size_t>(b * width + selectors[b]) * rows,
                        rows, expected.begin() + b * rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(indices.p, selectors.data(), indices.bytes,
                                   cudaMemcpyHostToDevice, context.stream));
        if (graph.ready())
            graph.launch(context.stream);
        else
            launch();
        context.synchronize();
        failures += verify_exact("committed hidden N-1",
                                 read<std::uint16_t>(output, expected.size()), expected);
        failures += verify_exact("selectors unchanged", from_device<std::int32_t>(indices, batch),
                                 selectors);
    }
    failures +=
        verify_exact("hidden unchanged", from_device<std::uint16_t>(input, hidden.size()), hidden);
    failures += output.verify_guards("committed hidden");
    return failures;
}

int remap_case(int token_count) {
    constexpr int map_size = 131072;
    std::vector<std::int32_t> id_map(map_size);
    for (int i = 0; i < map_size; ++i) {
        const auto value                    = 65537u * static_cast<std::uint32_t>(i) + 17u;
        id_map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(value & (map_size - 1u));
    }
    std::vector<std::int32_t> proposals(static_cast<std::size_t>(token_count));
    for (int i = 0; i < token_count; ++i) {
        proposals[static_cast<std::size_t>(i)] =
            i == 0 ? 0 : (i == token_count - 1 ? map_size - 1 : (7919 * i) & (map_size - 1));
    }
    std::vector<std::int32_t> expected(proposals.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        expected[i] = id_map[static_cast<std::size_t>(proposals[i])];
    }

    DeviceBuffer d_map = to_device(id_map);
    GuardedDeviceBuffer d_proposals(proposals.size() * sizeof(std::int32_t));
    initialize(d_proposals, proposals);
    Tensor proposal_tensor(d_proposals.data(), DType::I32, {token_count});
    ops::proposal_remap_token_ids(proposal_tensor, static_cast<const std::int32_t*>(d_map.p),
                                  map_size, nullptr);
    cuda_synchronize();

    const std::string label = "proposal remap T=" + std::to_string(token_count);
    int failures            = verify_exact((label + " in-place output").c_str(),
                                           read<std::int32_t>(d_proposals, proposals.size()), expected);
    failures += verify_exact((label + " map unchanged").c_str(),
                             from_device<std::int32_t>(d_map, id_map.size()), id_map);
    failures += d_proposals.verify_guards((label + " guards").c_str());
    return failures;
}

int transforms_conformance() {
    int failures = 0;
    for (int k = 1; k <= 31; ++k)
        for (int batch : {1, 8}) failures += prepare_verify_case(k, batch);
    for (int k : {1, 7, 15})
        for (int batch : {1, 2, 8})
            for (bool sparse : {false, true}) failures += overlay_case(k, batch, sparse);
    for (int width = 2; width <= 32; ++width)
        for (int batch : {1, 8}) failures += batched_select_hidden_case(width, batch);
    failures += select_hidden_case(5120, 6, 0);
    failures += select_hidden_case(5120, 6, 5);
    failures += select_hidden_case(2048, 16, 7);
    failures += remap_case(1);
    failures += remap_case(15);
    failures += remap_case(120);
    return failures;
}

int onehot_distribution_case(int k, bool graph, bool mtp_onehot = false, int forced_prefix = 0) {
    if (forced_prefix < 0 || forced_prefix >= k) {
        throw std::invalid_argument("distribution prefix must leave at least one stochastic draft");
    }
    const int batch  = k > 31 ? 1 : 8;
    const int trials = 8192 / batch;
    const int width  = k + 1;
    const std::array<float, 4> requested{0.6F, 0.25F, 0.1F, 0.05F};
    std::array<double, 4> probabilities{};
    double total = 0;
    for (int id = 0; id < 4; ++id) {
        probabilities[id] = std::exp(bf16_to_f32(f32_to_bf16(std::log(requested[id]))));
        total += probabilities[id];
    }
    for (auto& probability : probabilities) { probability /= total; }
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) * width * batch,
                                      f32_to_bf16(-std::numeric_limits<float>::infinity()));
    for (int row = 0; row < width * batch; ++row) {
        if (row % width < forced_prefix) {
            logits[static_cast<std::size_t>(row) * kSparsePhysicalRows] = f32_to_bf16(0);
            continue;
        }
        for (int id = 0; id < 4; ++id) {
            logits[static_cast<std::size_t>(row) * kSparsePhysicalRows + id] =
                f32_to_bf16(std::log(requested[id]));
        }
    }
    std::vector<int> ids(16 * k * batch), zero(width * batch, 0);
    std::vector<float> q(ids.size(), 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<int>(i % 16);
        q[i]   = i % 16 == 0 ? 1.0F : 0.0F;
    }
    auto d_logits = to_device(logits);
    auto d_ids    = to_device(ids);
    auto d_q      = to_device(q);
    auto d_zero   = to_device(zero);
    auto d_drafts = to_device(std::vector<int>(k * batch, 0));
    const std::vector<int> initial_anchors(batch, 0), initial_lengths(batch, 100);
    auto d_anchors = to_device(initial_anchors);
    auto d_extents = to_device(std::vector<int>(batch, k));
    auto d_lengths = to_device(initial_lengths);
    auto d_history =
        to_device(std::vector<int>(static_cast<std::size_t>(batch) * kSparseTokenDomain, 0));
    std::vector<ops::SamplingConfig> configs(batch);
    for (int row = 0; row < batch; ++row) {
        configs[row].temperature  = 1;
        configs[row].top_k        = 4;
        configs[row].top_p        = 1;
        configs[row].token_counts = static_cast<int*>(d_history.p) + row * kSparseTokenDomain;
    }
    auto d_configs = to_device(configs);
    GuardedDeviceBuffer licensed(width * batch * sizeof(int)), counts(batch * sizeof(int)),
        accepted(batch * sizeof(int));
    licensed.fill(0xcd);
    counts.fill(0);
    accepted.fill(0);
    Tensor target(d_zero.p, DType::I32, {width, batch});
    Tensor target_logits(d_logits.p, DType::BF16, {kSparsePhysicalRows, width, batch});
    Tensor drafts(d_drafts.p, DType::I32, {k, batch});
    Tensor candidates(d_ids.p, DType::I32, {16, k, batch});
    Tensor proposal_q(d_q.p, DType::FP32, {16, k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor lengths(d_lengths.p, DType::I32, {batch});
    Tensor anchors(d_anchors.p, DType::I32, {batch});
    Tensor output(licensed.data(), DType::I32, {width, batch});
    Tensor output_counts(counts.data(), DType::I32, {batch});
    Tensor output_accepted(accepted.data(), DType::I32, {batch});
    const auto bytes = mtp_onehot ? ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                        kSparseTokenDomain, k, k, batch, batch)
                                  : ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                                        kSparseTokenDomain, {false}, k, k, batch, batch);
    GuardedDeviceBuffer scratch(bytes);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "distribution stream");
    int failures = 0;
    {
        const auto launch = [&] {
            if (mtp_onehot) {
                ops::speculative_accept_greedy_drafts(
                    target, target_logits, drafts, extents, lengths, anchors, output, output_counts,
                    output_accepted, kSparseTokenDomain,
                    static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, stream);
                return;
            }
            ops::speculative_accept_sparse_drafts(
                target, target_logits, drafts, candidates, proposal_q, extents, lengths, anchors,
                output, output_counts, output_accepted, kSparseTokenDomain,
                static_cast<const ops::SamplingConfig*>(d_configs.p), {false}, workspace, stream);
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        launch();
        cuda_synchronize(stream);
        if (graph) {
            definition.capture(stream, launch);
            executable.instantiate(definition);
        }
        std::array<std::uint64_t, 4> first{}, conditional_second{};
        std::uint64_t second_trials = 0;
        // Penalties are zero in this fixture; cumulative history must not affect sampling.
        std::vector<int> expected_history(static_cast<std::size_t>(batch) * kSparseTokenDomain, 0);
        cuda_check(cudaMemsetAsync(d_history.p, 0, d_history.bytes, stream),
                   "reset distribution history");
        cuda_synchronize(stream);
        SparseAcceptSuite oracle(k, batch);
        for (int trial = 0; trial < trials; ++trial) {
            for (int row = 0; row < batch; ++row) {
                configs[row].seed = 555001ULL + trial * batch + row;
            }
            d_configs.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
            d_lengths.copy_from_host(initial_lengths.data(), initial_lengths.size() * sizeof(int));
            d_anchors.copy_from_host(initial_anchors.data(), initial_anchors.size() * sizeof(int));
            if (graph) {
                executable.launch(stream);
            } else {
                launch();
            }
            cuda_synchronize(stream);
            const auto tokens          = read<int>(licensed, width * batch);
            const auto sizes           = read<int>(counts, batch);
            const auto accepted_counts = read<int>(accepted, batch);
            for (int row = 0; row < batch; ++row) {
                if (sizes[row] < 1 || sizes[row] > width ||
                    accepted_counts[row] != sizes[row] - 1) {
                    ++failures;
                    continue;
                }
                std::vector<int> expected_tokens;
                for (int column = 0; column <= k; ++column) {
                    if (column < forced_prefix) {
                        expected_tokens.push_back(0);
                        continue;
                    }
                    const int position = 101 + column;
                    if (column < k && oracle.oracle_uniform(configs[row].seed, position,
                                                            ops::kSamplePurposeSpeculativeAccept) <
                                          probabilities[0]) {
                        expected_tokens.push_back(0);
                        continue;
                    }
                    const bool correction = column < k;
                    const double uniform =
                        oracle.oracle_uniform(configs[row].seed, position,
                                              correction ? ops::kSamplePurposeSpeculativeCorrection
                                                         : ops::kSamplePurposeSpeculativeBonus);
                    const double normalizer = correction ? 1 - probabilities[0] : 1;
                    double cumulative       = 0;
                    int sampled             = 3;
                    for (int id = correction ? 1 : 0; id < 4; ++id) {
                        cumulative += probabilities[id] / normalizer;
                        if (uniform < cumulative) {
                            sampled = id;
                            break;
                        }
                    }
                    expected_tokens.push_back(sampled);
                    break;
                }
                if (mtp_onehot) {
                    for (const int token : expected_tokens) {
                        ++expected_history[static_cast<std::size_t>(row) * kSparseTokenDomain +
                                           token];
                    }
                }
                if (sizes[row] != static_cast<int>(expected_tokens.size()) ||
                    !std::equal(expected_tokens.begin(), expected_tokens.end(),
                                tokens.begin() + row * width)) {
                    ++failures;
                    if (failures < 4) {
                        std::cerr << "onehot per-seed oracle mismatch seed=" << configs[row].seed
                                  << '\n';
                    }
                }
                if (sizes[row] <= forced_prefix) {
                    ++failures;
                    continue;
                }
                const int id = tokens[row * width + forced_prefix];
                if (id < 0 || id >= 4) {
                    ++failures;
                    continue;
                }
                ++first[id];
                if (id == 0) {
                    const int next = tokens[row * width + forced_prefix + 1];
                    if (sizes[row] < forced_prefix + 2 || next < 0 || next >= 4) {
                        ++failures;
                        continue;
                    }
                    ++second_trials;
                    ++conditional_second[next];
                } else if (sizes[row] != forced_prefix + 1) {
                    ++failures;
                }
            }
        }
        const auto check = [&](const auto& observed, std::uint64_t n) {
            for (int id = 0; id < 4; ++id) {
                const double expected = n * probabilities[id];
                const double bound =
                    6 * std::sqrt(n * probabilities[id] * (1 - probabilities[id])) + 2;
                if (std::abs(static_cast<double>(observed[id]) - expected) > bound) { ++failures; }
            }
        };
        check(first, trials * batch);
        check(conditional_second, second_trials);
        failures += scratch.verify_guards("onehot distribution workspace");
        failures += licensed.verify_guards("onehot distribution output");
        failures += counts.verify_guards("onehot distribution counts");
        failures += accepted.verify_guards("onehot distribution accepted");
        failures +=
            verify_exact("onehot distribution history",
                         from_device<int>(d_history, expected_history.size()), expected_history);
        std::cout << (mtp_onehot ? "MTP" : "sparse") << " onehot distribution K=" << k
                  << " graph=" << graph << " forced_prefix=" << forced_prefix
                  << " trials=" << trials * batch << " first=";
        for (const auto count : first) { std::cout << count << ','; }
        std::cout << " conditional_second=";
        for (const auto count : conditional_second) { std::cout << count << ','; }
        std::cout << " failures=" << failures << '\n';
    }
    cuda_check(cudaStreamDestroy(stream), "distribution stream cleanup");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "speculative_round: SKIP (CUDA unavailable)\n";
        return 77;
    }

    if (argc == 2 && std::string_view(argv[1]) == "--wide-accept") {
        int failures     = 0;
        std::size_t peak = 0;
        for (const int k : {31, 32, 33, 47, 63}) {
            SparseAcceptSuite suite(k, 1);
            for (int reject = 0; reject <= k; ++reject)
                failures += suite.sparse_greedy_direct_case(0, false, reject, k);
            for (const int extent : {0, 1, 30, 31, 32, k, k + 2})
                failures += suite.sparse_greedy_direct_case(0, false, k, extent);
            for (int variant = 0; variant < 8; ++variant) {
                failures += suite.generated_general_case(false, false, false, variant);
                failures += suite.generated_general_case(false, true, false, variant);
                failures += suite.generated_general_case(false, true, true, variant);
            }
            failures += suite.repeated_history_case(false);
            failures += suite.repeated_history_case(true);
            peak = std::max(peak, suite.observed_workspace);
        }
        if (ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                kSparseTokenDomain, {false}, 1, 63, 1, 1) != peak) {
            std::cerr << "wide workspace interval mismatch\n";
            ++failures;
        }
        for (const bool greedy : {false, true}) {
            for (const auto [k, batch] : {std::pair{32, 2}, std::pair{63, 8}, std::pair{64, 1}}) {
                try {
                    (void)ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                        kSparseTokenDomain, {greedy}, 1, k, 1, batch);
                    std::cerr << "wide unsupported profile admitted\n";
                    ++failures;
                } catch (const std::invalid_argument&) {}
            }
        }
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " wide acceptance and graph replay\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--wide-distribution") {
        int failures = 0;
        for (const int prefix : {31, 62}) {
            for (const bool mtp : {false, true}) {
                for (const bool graph : {false, true}) {
                    failures += onehot_distribution_case(63, graph, mtp, prefix);
                }
            }
        }
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--ngram-only") {
        int failures = transforms_conformance();
        for (const int k : {1, 5, 15, 16, 31}) {
            SparseAcceptSuite suite(k, 1);
            failures += suite.generated_general_case(false, true);
            failures += suite.repeated_history_case(false);
            failures += suite.repeated_history_case(true);
            failures += greedy_accept_case(k, k / 2, 257);
        }
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " ngram acceptance paths\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--onehot-distribution") {
        int failures = 0;
        for (const int k : {1, 15, 31}) {
            for (const bool graph : {false, true}) {
                failures += onehot_distribution_case(k, graph);
            }
        }
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--mtp-onehot") {
        int failures = 0;
        for (const int k : {1, 5, 15, 16, 31}) {
            for (const int batch : {1, 8}) {
                SparseAcceptSuite suite(k, batch);
                for (int variant = 0; variant < 4; ++variant) {
                    failures += suite.generated_general_case(false, true, true, variant);
                }
                failures += suite.repeated_history_case(false, true);
                failures += suite.repeated_history_case(true, true);
            }
        }
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " MTP onehot oracle and graph replay\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--ngram-negative-penalties") {
        int failures = 0;
        for (const int k : {1, 5, 15, 16, 31}) {
            for (const int batch : {1, 8}) {
                SparseAcceptSuite suite(k, batch);
                for (const bool mtp : {false, true}) {
                    for (int variant = 4; variant < 8; ++variant) {
                        failures += suite.generated_general_case(false, true, mtp, variant);
                    }
                }
            }
        }
        std::cout << (failures == 0 ? "PASS" : "FAIL")
                  << " ngram negative-penalty oracle and graph replay\n";
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--mtp-distribution") {
        int failures = 0;
        for (const int k : {1, 15, 31}) {
            for (const bool graph : {false, true}) {
                failures += onehot_distribution_case(k, graph, true);
            }
        }
        return failures == 0 ? 0 : 1;
    }
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--transforms-only")) {
        std::cerr << "usage: ninfer_speculative_round_test "
                     "[--transforms-only|--ngram-only|--onehot-distribution|--mtp-onehot|--mtp-"
                     "distribution|--ngram-negative-penalties|--wide-accept|--wide-distribution]\n";
        return 2;
    }
    int failures = transforms_conformance();
    if (argc == 2) {
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " speculative transforms\n";
        return failures == 0 ? 0 : 1;
    }
    const std::size_t k15 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 1);
    const std::size_t k31 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 31, 31, 1, 1);
    const std::size_t k63 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 63, 63, 1, 1);
    if (k15 == 0 || k15 != ops::sampling_workspace_capacity_bytes(257, 16, 16) || k31 <= k15 ||
        k63 <= k31 || ops::sampling_workspace_capacity_bytes(257, 32, 32) != 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 64, 64, 1, 1) != 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 1, 64, 1, 1) != k63 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 2) !=
            2 * k15) {
        std::cerr << "speculative accept workspace did not close over K+1 sampling columns\n";
        ++failures;
    }
    try {
        (void)ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 0, 15, 1, 1);
        std::cerr << "speculative accept workspace accepted an invalid draft interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += greedy_accept_case(1, 0);
    failures += greedy_accept_case(5, 2);
    failures += greedy_accept_case(5, 5);
    failures += greedy_accept_case(15, 7, 257);
    failures += greedy_accept_case(31, 31, 257);
    failures += greedy_accept_case(31, 30, 257);
    failures += greedy_penalty_case(64);
    failures += greedy_penalty_case(257);
    // Dense multiblock greedy non-finite guards (token_domain=257 > kSamplerTileItems=256):
    // no-penalty and penalized branches, each with a clean cross-check and a poisoned refusal.
    for (bool poison : {false, true}) {
        failures += greedy_multiblock_non_finite_case(0, 257, poison);
        failures += greedy_multiblock_non_finite_case(3, 257, poison);
        failures += greedy_penalty_non_finite_case(257, poison);
    }
    for (int domain : {64, 257, 248077}) {
        for (bool stochastic : {false, true}) {
            for (int rejection : {0, 2, 5}) {
                failures += masked_greedy_drafts_case(5, domain, stochastic, rejection);
            }
        }
    }
    for (int k : {1, 5, 15}) {
        for (int batch : {1, 8}) {
            failures += SparseAcceptSuite(k, batch).generated_general_case(true);
        }
    }
    failures += deterministic_sampling_case();
    failures += batched_sampling_workspace_stride_case();
    std::size_t sparse_peak = 0;
    for (int k = 1; k <= 31; ++k) {
        for (int batch = 1; batch <= 8; ++batch) {
            SparseAcceptSuite suite(k, batch);
            failures += suite.sparse_greedy_direct_case();
            failures += suite.generated_general_case();
            if (batch == 1) { failures += suite.generated_general_case(false, true); }
            sparse_peak = std::max(sparse_peak, suite.observed_workspace);
            if (batch == 1)
                for (int pattern = 1; pattern < 7; ++pattern)
                    failures += suite.sparse_greedy_direct_case(pattern);
        }
    }
    if (ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
            kSparseTokenDomain, {false}, 1, 31, 1, 8) != sparse_peak ||
        ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(kSparseTokenDomain, {true},
                                                                       1, 31, 1, 8) != 0) {
        std::cerr << "sparse interval workspace does not cover the observed domain\n";
        ++failures;
    }
    for (bool raw : {false, true}) {
        try {
            (void)ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                kSparseTokenDomain, {raw}, 1, 32, 1, 8);
            std::cerr << "sparse query admitted K=32 at B=8\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    for (int k : {1, 7, 15, 16, 31}) {
        SparseAcceptSuite suite(k, 8);
        failures += suite.sparse_general_mixed_case();
        failures += suite.sparse_greedy_direct_case(0, true);
        failures += SparseAcceptSuite(k, 1).repeated_history_case(false);
        failures += SparseAcceptSuite(k, 1).repeated_history_case(true);
    }
    // A greedy sparse round must refuse to license a token whose verify logit is non-finite. Both
    // greedy sparse routes are covered -- raw (the dedicated warp kernel) and the multiblock
    // group-finalize kernel -- and each batch mixes poisoned with clean rows, so a blanket refusal
    // fails just as loudly as no refusal at all.
    for (int k : {1, 7, 15}) {
        for (int batch : {1, 8}) {
            for (bool raw : {true, false}) {
                for (int poison = 0; poison < 2; ++poison) {
                    failures +=
                        SparseAcceptSuite(k, batch).sparse_non_finite_terminal_case(raw, poison);
                    // The same fixture with the NaN on an *accepted* column instead of the
                    // divergence column. A terminal-only guard passes this while licensing a
                    // draft the target never actually endorsed.
                    failures += SparseAcceptSuite(k, batch).sparse_non_finite_terminal_case(
                        raw, poison, /*penalized=*/false, /*poison_matched=*/true);
                }
            }
            // Penalized greedy takes the group-finalize routes instead, which select the terminal
            // from the penalized top-k. raw_greedy is false by construction: the envelope means
            // "all rows greedy *without* penalties".
            for (int poison = 0; poison < 2; ++poison) {
                failures += SparseAcceptSuite(k, batch).sparse_non_finite_terminal_case(
                    /*raw_greedy=*/false, poison, /*penalized=*/true);
            }
        }
    }

    if (failures != 0) {
        std::cerr << "speculative_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "speculative_round: PASS\n";
    return 0;
}
