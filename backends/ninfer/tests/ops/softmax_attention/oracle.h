#pragma once

#include "ninfer/ops/attention_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ninfer::test {

// Independent logical Softmax Attention oracle. Callbacks expose represented public values and
// the entry-specific visible set; no production staging cast, tile, cache address, or reduction
// tree is reproduced here. Callbacks must be safe to call concurrently: large problems spread
// their independent (query, head) rows across host threads, and every row keeps the serial
// summation order, so the result does not depend on the thread count.
template <typename QueryValue, typename KeyValue, typename ValueValue, typename Visible,
          typename Store>
void naive_dense_softmax_attention(ops::AttentionHeadGeometry geometry, int query_tokens,
                                   int key_tokens, double scale, QueryValue query_value,
                                   KeyValue key_value, ValueValue value_value, Visible visible,
                                   Store store) {
    if (!ops::valid_attention_head_geometry(geometry) || query_tokens < 0 || key_tokens < 0) {
        throw std::invalid_argument("invalid naive Softmax Attention geometry");
    }
    const int group = geometry.query_heads / geometry.kv_heads;
    const int dims  = geometry.head_dim;
    const int rows  = query_tokens * geometry.query_heads;

    const auto attend_rows = [&](auto query_at, auto key_at, auto value_at) {
        const auto attend_row = [&](int row, std::vector<double>& scores) {
            const int query      = row / geometry.query_heads;
            const int query_head = row % geometry.query_heads;
            const int kv_head    = query_head / group;
            double maximum       = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < key_tokens; ++key) {
                if (!visible(query, key)) {
                    scores[static_cast<std::size_t>(key)] =
                        -std::numeric_limits<double>::infinity();
                    continue;
                }
                double dot = 0.0;
                for (int d = 0; d < dims; ++d) {
                    dot += query_at(d, query_head, query) * key_at(d, kv_head, key);
                }
                const double score                    = dot * scale;
                scores[static_cast<std::size_t>(key)] = score;
                maximum                               = std::max(maximum, score);
            }

            double denominator = 0.0;
            if (maximum != -std::numeric_limits<double>::infinity()) {
                for (int key = 0; key < key_tokens; ++key) {
                    double& score = scores[static_cast<std::size_t>(key)];
                    if (score == -std::numeric_limits<double>::infinity()) continue;
                    score = std::exp(score - maximum);
                    denominator += score;
                }
            }
            for (int d = 0; d < dims; ++d) {
                double numerator = 0.0;
                for (int key = 0; key < key_tokens; ++key) {
                    const double weight = scores[static_cast<std::size_t>(key)];
                    if (weight == -std::numeric_limits<double>::infinity()) continue;
                    numerator += weight * value_at(d, kv_head, key);
                }
                store(d, query_head, query, denominator > 0.0 ? numerator / denominator : 0.0);
            }
        };

        const double work =
            static_cast<double>(rows) * static_cast<double>(key_tokens) * static_cast<double>(dims);
        constexpr double kSerialWork   = 1 << 24;
        constexpr unsigned kMaxWorkers = 16;
        const unsigned workers =
            work < kSerialWork ? 1U
                               : std::min({std::max(1U, std::thread::hardware_concurrency()),
                                           kMaxWorkers, static_cast<unsigned>(std::max(rows, 1))});
        if (workers <= 1) {
            std::vector<double> scores(static_cast<std::size_t>(key_tokens));
            for (int row = 0; row < rows; ++row) attend_row(row, scores);
            return;
        }

        std::exception_ptr failure;
        std::mutex failure_mutex;
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (unsigned worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                try {
                    std::vector<double> scores(static_cast<std::size_t>(key_tokens));
                    for (int row = static_cast<int>(worker); row < rows;
                         row += static_cast<int>(workers)) {
                        attend_row(row, scores);
                    }
                } catch (...) {
                    const std::lock_guard lock(failure_mutex);
                    if (!failure) failure = std::current_exception();
                }
            });
        }
        for (std::thread& thread : threads) thread.join();
        if (failure) std::rethrow_exception(failure);
    };

    // The callbacks decode represented storage, and every query head of a group reads the same
    // keys and values, so the planes are read once per key some query sees, when they fit; a key
    // no query sees is never read. The values plane is stored by dimension so each output
    // element sums its keys contiguously, in the same order as the callback path.
    constexpr std::size_t kMaxPlaneValues = std::size_t{1} << 24;
    const std::size_t plane = static_cast<std::size_t>(geometry.kv_heads) *
                              static_cast<std::size_t>(key_tokens) * static_cast<std::size_t>(dims);
    if (plane == 0 || plane > kMaxPlaneValues) {
        attend_rows(query_value, key_value, value_value);
        return;
    }
    std::vector<char> seen(static_cast<std::size_t>(key_tokens), 0);
    for (int query = 0; query < query_tokens; ++query) {
        for (int key = 0; key < key_tokens; ++key) {
            if (seen[static_cast<std::size_t>(key)] == 0 && visible(query, key)) {
                seen[static_cast<std::size_t>(key)] = 1;
            }
        }
    }
    std::vector<double> queries(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dims));
    std::vector<double> keys(plane);
    std::vector<double> values(plane);
    for (int query = 0; query < query_tokens; ++query) {
        for (int head = 0; head < geometry.query_heads; ++head) {
            const std::size_t base =
                (static_cast<std::size_t>(query) * geometry.query_heads + head) * dims;
            for (int d = 0; d < dims; ++d) queries[base + d] = query_value(d, head, query);
        }
    }
    for (int head = 0; head < geometry.kv_heads; ++head) {
        for (int key = 0; key < key_tokens; ++key) {
            if (seen[static_cast<std::size_t>(key)] == 0) continue;
            for (int d = 0; d < dims; ++d) {
                keys[(static_cast<std::size_t>(head) * key_tokens + key) * dims + d] =
                    key_value(d, head, key);
                values[(static_cast<std::size_t>(head) * dims + d) * key_tokens + key] =
                    value_value(d, head, key);
            }
        }
    }
    attend_rows(
        [&](int d, int head, int query) {
            return queries[(static_cast<std::size_t>(query) * geometry.query_heads + head) * dims +
                           d];
        },
        [&](int d, int head, int key) {
            return keys[(static_cast<std::size_t>(head) * key_tokens + key) * dims + d];
        },
        [&](int d, int head, int key) {
            return values[(static_cast<std::size_t>(head) * dims + d) * key_tokens + key];
        });
}

} // namespace ninfer::test
