#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<float> values(size_t n, uint32_t seed, float scale) {
    std::vector<float> result(n);
    for (float & x : result) {
        seed = seed * 1664525u + 1013904223u;
        x = (float(int32_t(seed >> 8)) / 8388608.0f - 1.0f) * scale;
    }
    return result;
}

static void set(ggml_tensor * t, uint32_t seed, float scale) {
    const auto data = values(ggml_nelements(t), seed, scale);
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> result(ggml_nelements(t));
    ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
    return result;
}

// Shared oracle from the Vulkan replay regression suite.
static std::vector<double> reference_fold(
        const std::vector<float> & initial, const std::vector<float> & key,
        const std::vector<float> & value, const std::vector<float> & gate,
        const std::vector<float> & beta, int keep) {
    std::vector<double> state(initial.begin(), initial.end());
    for (int t = 0; t < keep; ++t) {
        for (int h = 0; h < 48; ++h) {
            const double decay = std::exp(double(gate[t * 48 + h]));
            const float * k = key.data() + (t * 16 + h % 16) * 128;
            for (int c = 0; c < 128; ++c) {
                double * s = state.data() + (h * 128 + c) * 128;
                double dot = 0;
                for (int r = 0; r < 128; ++r) dot += decay * s[r] * k[r];
                const double delta = (value[(t * 48 + h) * 128 + c] - dot) * beta[t * 48 + h];
                for (int r = 0; r < 128; ++r) s[r] = decay * s[r] + k[r] * delta;
            }
        }
    }
    return state;
}

static std::vector<float> snapshot(ggml_backend_t backend, int tokens) {
    ggml_init_params params{2 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "context allocation failed");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    auto * b = ggml_dup_tensor(ctx, g);
    auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 48, 1);
    auto * out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 3);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "buffer allocation failed");
    ggml_backend_buffer_clear(buffer, 0);
    set(q, 123, 0.12f);
    set(k, 234, 0.12f);
    set(v, 345, 0.25f);
    set(g, 456, 0.02f);
    set(b, 567, 0.5f);
    set(s, 678, 0.1f);
    const auto initial = get(s);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "snapshot compute failed");
    const auto result = get(out);
    require(initial == get(s), "snapshot overwrote input state");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static void check_replay(ggml_backend_t backend, int tokens, int rounds, bool fold = true, int capacity = 0) {
    if (capacity == 0) capacity = tokens;
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "replay context allocation failed");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    auto * b = ggml_dup_tensor(ctx, g);
    auto * state_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128 * 128 * 48, 1);
    auto * s = ggml_reshape_4d(ctx, state_cache, 128, 128, 48, 1);
    auto * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * 10240, 1);
    auto * conv_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
    const int widths[] = {2048, 6144, 48, 48, 10240};
    ggml_tensor * inputs[] = {k, v, g, b, conv_input};
    ggml_tensor * records[5];
    for (int i = 0; i < 5; ++i) records[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, widths[i], capacity);
    auto * descriptor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(ggml_cuda_gdn_replay_layer));
    auto * reference = ggml_gated_delta_net(ctx, q, k, v, g, b, s, tokens);
    auto * recorded = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 0);
    auto * graph = ggml_new_graph(ctx);
    for (int i = 0; i < 5; ++i) {
        ggml_build_forward_expand(graph, ggml_cpy(ctx, inputs[i],
                    ggml_view_1d(ctx, records[i], ggml_nelements(inputs[i]), 0)));
    }
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, recorded);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "replay buffer allocation failed");
    set(q, 12, 0.1f);
    set(k, 23, 0.1f);
    set(v, 34, 0.2f);
    set(s, 45, 0.1f);
    set(conv, 56, 0.1f);
    set(conv_input, 67, 0.1f);
    auto gates = values(ggml_nelements(g), 78, .05f);
    auto betas = values(ggml_nelements(b), 89, .9f);
    for (auto & x : gates) x = -std::fabs(x);
    for (auto & x : betas) x = std::fabs(x);
    ggml_backend_tensor_set(g, gates.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, betas.data(), 0, ggml_nbytes(b));
    const ggml_cuda_gdn_replay_layer layer{static_cast<float *>(s->data), static_cast<float *>(conv->data),
        static_cast<float *>(records[0]->data), static_cast<float *>(records[1]->data), static_cast<float *>(records[2]->data),
        static_cast<float *>(records[3]->data), static_cast<float *>(records[4]->data)};
    ggml_backend_tensor_set(descriptor, &layer, 0, sizeof(layer));
    const auto original_key = get(k);
    const auto original_value = get(v);
    const auto columns = get(conv_input);
    require(ggml_backend_cuda_gdn_fold(nullptr, 0, 0, 0, nullptr), "zero Fold must not access descriptors");
    for (int round = 0; round < rounds; ++round) {
        ggml_backend_tensor_set(k, original_key.data(), 0, ggml_nbytes(k));
        const auto initial = get(s);
        const auto history = get(conv);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Record compute failed");
        require(initial == get(s), "Record changed committed state");
        require(history == get(conv), "Record changed committed convolution history");
        for (int i = 0; i < 5; ++i) {
            const auto input = get(inputs[i]);
            const auto record = get(records[i]);
            require(std::memcmp(input.data(), record.data(), input.size() * sizeof(float)) == 0,
                    "Replay capture differs from forward input");
        }
        const auto snapshots = get(reference);
        const auto output = get(recorded);
        require(std::memcmp(snapshots.data(), output.data(), ggml_nbytes(recorded)) == 0, "Record attention differs");
        if (!fold) continue;
        const int first = rounds == 1 ? 0 : round % (tokens + 1);
        const int last = rounds == 1 ? tokens : first;
        for (int keep = first; keep <= last; ++keep) {
            ggml_backend_tensor_set(s, initial.data(), 0, ggml_nbytes(s));
            ggml_backend_tensor_set(conv, history.data(), 0, ggml_nbytes(conv));
            for (int i = 0; i < 5; ++i) {
                auto data = get(inputs[i]);
                data.resize(ggml_nelements(records[i]), std::numeric_limits<float>::quiet_NaN());
                std::fill(data.begin() + keep * widths[i], data.end(), std::numeric_limits<float>::quiet_NaN());
                ggml_backend_tensor_set(records[i], data.data(), 0, ggml_nbytes(records[i]));
            }
            require(ggml_backend_cuda_gdn_fold(static_cast<const ggml_cuda_gdn_replay_layer *>(descriptor->data),
                        1, keep, capacity, nullptr), "Fold launch failed");
            const auto actual = get(s);
            const float * expected = keep == 0 ? initial.data()
                : snapshots.data() + output.size() + (tokens - keep) * initial.size();
            require(std::memcmp(expected, actual.data(), ggml_nbytes(s)) == 0, "Fold state differs from accepted snapshot");
            if (rounds == 1) {
                const auto oracle = reference_fold(initial, original_key, original_value, gates, betas, keep);
                for (size_t i = 0; i < actual.size(); ++i) {
                    require(std::isfinite(actual[i]) && std::fabs(actual[i] - oracle[i]) < 2e-6 + 2e-5 * std::fabs(oracle[i]),
                            "Fold differs from independent FP64 recurrence");
                }
            }
            const auto actual_conv = get(conv);
            for (int c = 0; c < 10240; ++c) {
                for (int i = 0; i < 3; ++i) {
                    const int source = keep + i;
                    const float expected_conv = source < 3 ? history[c * 3 + source] : columns[(source - 3) * 10240 + c];
                    require(actual_conv[c * 3 + i] == expected_conv, "Fold conv history differs");
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS %s: width=%d capacity=%d rounds=%d\n", fold ? "Record/Fold (all prefixes, rejected NaNs, nonzero state)" : "CPU Record", tokens, capacity, rounds);
}

static void check_multilayer(ggml_backend_t backend) {
    constexpr int count = 4, tokens = 6, keep = 2;
    struct layer_t {
        ggml_tensor * k;
        ggml_tensor * v;
        ggml_tensor * g;
        ggml_tensor * b;
        ggml_tensor * s;
        ggml_tensor * conv;
        ggml_tensor * input;
        ggml_tensor * reference;
    };
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "multilayer context allocation failed");
    auto * graph = ggml_new_graph(ctx);
    std::vector<layer_t> layers;
    for (int il = 0; il < count; ++il) {
        layer_t l{};
        l.k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
        l.v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
        l.g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
        l.b = ggml_dup_tensor(ctx, l.g);
        l.s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128 * 128 * 48, 1);
        l.conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * 10240, 1);
        l.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
        l.reference = ggml_gated_delta_net(ctx, l.k, l.k, l.v, l.g, l.b,
                ggml_reshape_4d(ctx, l.s, 128, 128, 48, 1), tokens);
        ggml_build_forward_expand(graph, l.reference);
        layers.push_back(l);
    }
    auto * descriptors = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, count * sizeof(ggml_cuda_gdn_replay_layer));
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "multilayer buffer allocation failed");
    std::vector<ggml_cuda_gdn_replay_layer> gpu_layers;
    std::vector<std::vector<float>> histories;
    uint32_t seed = 100;
    for (const auto & l : layers) {
        for (auto * t : {l.k, l.v, l.s, l.conv, l.input}) set(t, seed++, .1f);
        auto gates = values(48 * tokens, seed++, .05f);
        auto betas = values(48 * tokens, seed++, .9f);
        for (auto & x : gates) x = -std::fabs(x);
        for (auto & x : betas) x = std::fabs(x);
        ggml_backend_tensor_set(l.g, gates.data(), 0, ggml_nbytes(l.g));
        ggml_backend_tensor_set(l.b, betas.data(), 0, ggml_nbytes(l.b));
        gpu_layers.push_back({static_cast<float *>(l.s->data), static_cast<float *>(l.conv->data),
                static_cast<float *>(l.k->data), static_cast<float *>(l.v->data), static_cast<float *>(l.g->data),
                static_cast<float *>(l.b->data), static_cast<float *>(l.input->data)});
        histories.push_back(get(l.conv));
    }
    ggml_backend_tensor_set(descriptors, gpu_layers.data(), 0, ggml_nbytes(descriptors));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "multilayer reference failed");
    require(ggml_backend_cuda_gdn_fold(static_cast<const ggml_cuda_gdn_replay_layer *>(descriptors->data),
                count, keep, tokens, nullptr), "multilayer fold failed");
    for (int il = 0; il < count; ++il) {
        const auto & l = layers[il];
        const auto reference = get(l.reference), actual = get(l.s), conv = get(l.conv), input = get(l.input);
        const float * expected = reference.data() + 128 * 48 * tokens + (tokens - keep) * 128 * 128 * 48;
        require(std::memcmp(expected, actual.data(), ggml_nbytes(l.s)) == 0, "multilayer state differs");
        for (int c = 0; c < 10240; ++c) {
            for (int i = 0; i < 3; ++i) {
                const int source = keep + i;
                const float value = source < 3 ? histories[il][c * 3 + source] : input[(source - 3) * 10240 + c];
                require(conv[c * 3 + i] == value, "multilayer convolution differs");
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS native multilayer fold: layers=%d\n", count);
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
        auto * device = ggml_backend_dev_by_type(cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) return 77;
        auto backend = ggml_backend_dev_init(device, nullptr);
        require(backend != nullptr, "GPU backend initialization failed");
        if (argc == 1 || cpu) {
            for (int width : {1, 2, 3, 4, 5, 6}) check_replay(backend, width, 1, !cpu);
            for (int width : {1, 2, 3, 4, 5}) check_replay(backend, width, 1, !cpu, 6);
            if (!cpu) {
                check_multilayer(backend);
                check_replay(backend, 6, 1000);
            }
            ggml_backend_free(backend);
            return 0;
        }
        const bool write = argc == 3 && std::string(argv[1]) == "--write";
        const bool check = argc == 3 && std::string(argv[1]) == "--check";
        require(write || check, "usage: gdn-replay-test --write|--check snapshot.bin");
        std::fstream file(argv[2], std::ios::binary | (write ? std::ios::out | std::ios::trunc : std::ios::in));
        require(bool(file), "cannot open snapshot fixture");
        for (int tokens : {1, 2, 3, 17, 512}) {
            const auto result = snapshot(backend, tokens);
            const size_t bytes = result.size() * sizeof(float);
            if (write) {
                file.write(reinterpret_cast<const char *>(result.data()), bytes);
            } else {
                std::vector<float> expected(result.size());
                file.read(reinterpret_cast<char *>(expected.data()), bytes);
                require(bool(file), "truncated snapshot fixture");
                require(std::memcmp(expected.data(), result.data(), bytes) == 0, "FP32 snapshot differs from original kernel");
            }
            std::printf("PASS original snapshots: tokens=%d bytes=%zu\n", tokens, bytes);
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
