#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama-kvmem-gdn.h"
#if defined(KVMEM_TEST_CUDA)
#include "ggml-cuda.h"
#endif
#if defined(KVMEM_TEST_VULKAN)
#include "ggml-vulkan.h"
#endif

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

static void check_replay(ggml_backend_t backend, int tokens, int rounds, bool generic = false, bool fold = true, int capacity = 0) {
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
#if defined(KVMEM_TEST_CUDA)
    auto * descriptor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(ggml_cuda_gdn_replay_layer));
#endif
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
#if defined(KVMEM_TEST_CUDA)
    const ggml_cuda_gdn_replay_layer cuda_layer{static_cast<float *>(s->data), static_cast<float *>(conv->data),
        static_cast<float *>(records[0]->data), static_cast<float *>(records[1]->data), static_cast<float *>(records[2]->data),
        static_cast<float *>(records[3]->data), static_cast<float *>(records[4]->data)};
    ggml_backend_tensor_set(descriptor, &cuda_layer, 0, sizeof(cuda_layer));
#endif
#if defined(KVMEM_TEST_VULKAN)
    const ggml_vk_gdn_replay_layer vk_layer{state_cache, conv, records[0], records[1], records[2], records[3], records[4]};
#endif
    const llama_kvmem_gdn_layer ggml_layer{state_cache, conv, records[0], records[1], records[2], records[3], records[4]};
    size_t max_compute_bytes = 0;
    auto fold_prefix = [&](int keep) {
        if (generic) {
            size_t compute_bytes = 0;
            const bool ok = llama_kvmem_gdn_fold_ggml(backend, &ggml_layer, 1, keep, capacity, &compute_bytes);
            max_compute_bytes = std::max(max_compute_bytes, compute_bytes);
            return ok;
        }
#if defined(KVMEM_TEST_CUDA)
        if (ggml_backend_is_cuda(backend)) {
            return ggml_backend_cuda_gdn_fold(static_cast<const ggml_cuda_gdn_replay_layer *>(descriptor->data),
                    1, keep, capacity, nullptr);
        }
#endif
#if defined(KVMEM_TEST_VULKAN)
        if (ggml_backend_is_vk(backend)) {
            return ggml_backend_vk_gdn_fold(backend, &vk_layer, 1, keep, capacity);
        }
#endif
        return false;
    };
    const auto original_key = get(k);
    const auto original_value = get(v);
    const auto columns = get(conv_input);
    require(fold_prefix(0), "zero Fold must not access records");
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
            require(fold_prefix(keep), "Fold launch failed");
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
    if (generic) require(max_compute_bytes < 8 * 1024 * 1024, "GGML Fold scratch is unexpectedly large");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS %s: width=%d capacity=%d rounds=%d", fold ? (generic ? "GGML Record/Fold" : "native Record/Fold") : "Record", tokens, capacity, rounds);
    if (generic) std::printf(" scratch=%zu", max_compute_bytes);
    std::printf("\n");
}

static void check_generic_multilayer(ggml_backend_t backend, bool generic = true) {
    constexpr int n_layers = 4;
    constexpr int tokens = 6;
    constexpr int keep = 2;
    constexpr size_t state_elements = 128 * 128 * 48;
    constexpr size_t attention_elements = 128 * 48 * tokens;

    struct tensors {
        ggml_tensor * key;
        ggml_tensor * value;
        ggml_tensor * gate;
        ggml_tensor * beta;
        ggml_tensor * state;
        ggml_tensor * conv;
        ggml_tensor * conv_input;
    };

    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "multilayer context allocation failed");
    std::vector<tensors> tensors_by_layer;
    std::vector<llama_kvmem_gdn_layer> layers;
    for (int il = 0; il < n_layers; ++il) {
        auto * key = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
        auto * value = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
        auto * gate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
        auto * beta = ggml_dup_tensor(ctx, gate);
        auto * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128 * 128 * 48, 1);
        auto * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * 10240, 1);
        auto * conv_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
        tensors_by_layer.push_back({key, value, gate, beta, state, conv, conv_input});
        layers.push_back({state, conv, key, value, gate, beta, conv_input});
    }
    auto * graph = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> references;
    for (const auto & layer : tensors_by_layer) {
        auto * state = ggml_reshape_4d(ctx, layer.state, 128, 128, 48, 1);
        auto * reference = ggml_gated_delta_net(ctx, layer.key, layer.key, layer.value,
                layer.gate, layer.beta, state, tokens);
        references.push_back(reference);
        ggml_build_forward_expand(graph, reference);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "multilayer buffer allocation failed");

    auto gates = values(48 * tokens, 78, .05f);
    auto betas = values(48 * tokens, 89, .9f);
    for (auto & x : gates) x = -std::fabs(x);
    for (auto & x : betas) x = std::fabs(x);
    uint32_t seed = 23;
    for (const auto & layer : tensors_by_layer) {
        set(layer.key, seed++, .1f);
        set(layer.value, 34, .2f);
        set(layer.state, seed++, .1f);
        set(layer.conv, seed++, .1f);
        set(layer.conv_input, 67, .1f);
        ggml_backend_tensor_set(layer.gate, gates.data(), 0, ggml_nbytes(layer.gate));
        ggml_backend_tensor_set(layer.beta, betas.data(), 0, ggml_nbytes(layer.beta));
    }
    std::vector<std::vector<float>> histories;
    for (const auto & layer : tensors_by_layer) histories.push_back(get(layer.conv));
    const auto columns = get(tensors_by_layer.front().conv_input);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "multilayer reference compute failed");
    size_t compute_bytes = 0;
    if (generic) {
        require(llama_kvmem_gdn_fold_ggml(backend, layers.data(), layers.size(), keep, tokens, &compute_bytes),
                "multilayer GGML Fold failed");
    } else {
#if defined(KVMEM_TEST_VULKAN)
        std::vector<ggml_vk_gdn_replay_layer> vk_layers;
        for (const auto & layer : layers) {
            vk_layers.push_back({layer.state, layer.conv, layer.key, layer.value, layer.gate, layer.beta, layer.conv_input});
        }
        const auto before = get(layers.front().state);
        auto invalid = vk_layers;
        invalid.back().beta = nullptr;
        require(!ggml_backend_vk_gdn_fold(backend, invalid.data(), invalid.size(), keep, tokens),
                "Native Vulkan accepted an invalid last layer");
        require(before == get(layers.front().state), "Native Vulkan partially committed invalid layers");
        require(!ggml_backend_vk_gdn_fold(backend, vk_layers.data(), vk_layers.size(), tokens + 1, tokens),
                "Native Vulkan accepted a prefix beyond capacity");
        require(ggml_backend_vk_gdn_fold(backend, vk_layers.data(), vk_layers.size(), keep, tokens),
                "multilayer native Vulkan Fold failed");
#else
        require(false, "native multilayer test requires Vulkan");
#endif
    }
    require(compute_bytes < 8 * 1024 * 1024, "multilayer GGML Fold scratch scales with layer count");
    for (size_t il = 0; il < tensors_by_layer.size(); ++il) {
        const auto & layer = tensors_by_layer[il];
        const auto snapshots = get(references[il]);
        const auto & history = histories[il];
        const float * expected_state = snapshots.data() + attention_elements + (tokens - keep) * state_elements;
        const auto actual_state = get(layer.state);
        require(std::memcmp(expected_state, actual_state.data(), ggml_nbytes(layer.state)) == 0,
                "multilayer Fold state differs");
        const auto actual_conv = get(layer.conv);
        for (int c = 0; c < 10240; ++c) {
            for (int i = 0; i < 3; ++i) {
                const int source = keep + i;
                const float expected_conv = source < 3 ? history[c * 3 + source] : columns[(source - 3) * 10240 + c];
                require(actual_conv[c * 3 + i] == expected_conv, "multilayer Fold conv history differs");
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS %s multilayer Fold: layers=%d width=%d keep=%d scratch=%zu\n",
            generic ? "GGML" : "Vulkan", n_layers, tokens, keep, compute_bytes);
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
        auto * cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(cpu_device != nullptr, "CPU backend unavailable");
        auto cpu_backend = ggml_backend_dev_init(cpu_device, nullptr);
        require(cpu_backend != nullptr, "CPU backend initialization failed");
        if (argc == 1 || cpu) {
            for (int width : {1, 2, 3, 4, 5, 6}) check_replay(cpu_backend, width, 1, true);
            check_replay(cpu_backend, 6, 100, true);
            check_generic_multilayer(cpu_backend);
            if (cpu) {
                ggml_backend_free(cpu_backend);
                return 0;
            }
        }
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) {
            std::printf("SKIP native GPU fold: no GPU backend device\n");
            ggml_backend_free(cpu_backend);
            return 0;
        }
        auto backend = ggml_backend_dev_init(device, nullptr);
        require(backend != nullptr, "GPU backend initialization failed");
        if (argc == 1) {
            check_replay(backend, 6, 1, true);
            check_generic_multilayer(backend);
#if defined(KVMEM_TEST_VULKAN)
            if (ggml_backend_is_vk(backend)) check_generic_multilayer(backend, false);
#endif
            for (int width : {1, 2, 3, 4, 5, 6}) check_replay(backend, width, 1);
            for (int width : {1, 2, 3, 4, 5}) check_replay(backend, width, 1, false, true, 6);
            check_replay(backend, 6, 1000);
            ggml_backend_free(backend);
            ggml_backend_free(cpu_backend);
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
        ggml_backend_free(cpu_backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
