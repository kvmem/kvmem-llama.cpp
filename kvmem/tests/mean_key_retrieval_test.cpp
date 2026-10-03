#include "kvmem/mean_key_retrieval.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/kvmem_store.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }

int main() {
    try {
        for (uint32_t kv_heads : {2U, 4U}) {
            const uint32_t layers = 3, dim = 256, q_heads = kv_heads == 2 ? 16 : 24;
            const uint32_t kw = kv_heads * dim, qw = q_heads * dim, group = q_heads / kv_heads;
            kvmem::RawKvStore keys({layers, kw, 0, 64});
            std::vector<float> queries(layers * qw), sums(layers * 5 * kw);
            std::vector<uint32_t> counts{7, 0, 13};
            for (size_t i = 0; i < queries.size(); ++i) queries[i] = float(int(i * 13 % 89) - 44) / 16;
            for (uint32_t l = 0; l < layers; ++l) {
                for (uint32_t b = 0; b < 5; ++b) {
                    auto* row = sums.data() + (l * 5 + b) * kw;
                    for (uint32_t d = 0; d < kw; ++d) row[d] = float(int((l * 11 + b * 37 + d * 7) % 101) - 50) / 8;
                    keys.write_layer_mean_sum(b * 64, b == 4 ? 17 : 64, l, row);
                }
            }
            const auto actual = kvmem::mean_key_retrieval_scores(keys, 6,
                {queries.data(), counts.data(), layers, dim, q_heads, kv_heads});
            for (uint32_t b = 0; b < 5; ++b) {
                double expected = 0;
                for (uint32_t l : {0U, 2U}) {
                    double layer = 0;
                    for (uint32_t h = 0; h < kv_heads; ++h) {
                        double dot = 0, qn = 0, kn = 0;
                        // Independent FP64 reference in reverse feature order.
                        for (uint32_t i = dim; i > 0; --i) {
                            const auto d = i - 1;
                            double q = 0;
                            for (uint32_t g = 0; g < group; ++g) q += queries[l * qw + (h * group + g) * dim + d];
                            q /= double(counts[l]) * group;
                            const double k = sums[(l * 5 + b) * kw + h * dim + d] / double(b == 4 ? 17 : 64);
                            dot += q * k; qn += q * q; kn += k * k;
                        }
                        layer += dot / (std::sqrt(qn * kn) + 1e-6);
                    }
                    expected += layer / kv_heads / 2;
                }
                require(std::abs(actual[b] - expected) < 2e-6, "GQA cosine differs from independent FP64 oracle");
            }
            require(actual[5] == 0, "missing block is not zero");
            std::fill(counts.begin(), counts.end(), 0);
            const auto empty = kvmem::mean_key_retrieval_scores(keys, 6,
                {queries.data(), counts.data(), layers, dim, q_heads, kv_heads});
            require(std::all_of(empty.begin(), empty.end(), [](double v) { return v == 0; }), "empty query scored");
        }
        kvmem::KvMemStoreConfig config;
        config.block_tokens = 64; config.select_budget = 256; config.sink_blocks = 1; config.recent_blocks = 1;
        kvmem::KvMemStore selector(config);
        selector.register_append(1025);
        std::vector<double> scores(selector.block_count(), -0.5);
        scores[2] = 0.9; scores[5] = 0.8;
        selector.set_retrieval_scores(scores);
        const auto chosen = selector.pick_topk_blocks({16});
        require(chosen == std::vector<uint32_t>({0, 2, 5, 16}), "cold historical blocks or mandatory tail lost");
        std::cout << "MEAN_KEY_RETRIEVAL PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
