#include "models/qwen3_5/program/program_impl.h"
#include <cmath>
#include <cstdio>

namespace ninfer::models::qwen3_5::detail {
void ProgramImpl::prepare_memory_statistics(SequenceState& sequence, std::uint32_t query_begin,
                                            std::uint32_t query_end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics) {
        const auto& config = parameters.model.config().text;
        state.statistics = std::make_unique<kvmem::RawKvStore>(kvmem::RawKvStoreConfig{
            .n_layer = config.full_attention_layers,
            .n_embd_k = static_cast<std::uint32_t>(memory_statistics->key_sums.ne[0]),
            .block_tokens = 64});
        kvmem::KvMemStoreConfig selection;
        selection.block_tokens = 64;
        selection.select_budget = kvmem_options.selected_tokens;
        selection.sink_blocks = 1;
        selection.recent_blocks = 1;
        state.selector = std::make_unique<kvmem::KvMemStore>(selection);
        state.query_sum.assign(memory_statistics->query_sums.numel(), 0.0F);
        state.query_begin = query_begin;
        state.query_end = query_end;
    }
    if (state.query_begin != query_begin || state.query_end != query_end) {
        throw std::logic_error("KVMem query range changed inside one prefill");
    }
    memory_statistics_ranges = {0, static_cast<std::int32_t>(capacity),
        static_cast<std::int32_t>(query_begin), static_cast<std::int32_t>(query_end)};
    CUDA_CHECK(cudaMemcpyAsync(memory_statistics->ranges.data, memory_statistics_ranges.data(),
        sizeof(memory_statistics_ranges), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImpl::commit_memory_statistics(SequenceState& sequence, std::uint32_t begin,
                                           std::uint32_t end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics || begin != state.statistics_frontier || end <= begin ||
        end - begin > prefill_chunk) {
        throw std::logic_error("KVMem statistics require exactly one append at the evaluated frontier");
    }
    const auto& buffers = *memory_statistics;
    auto* host = static_cast<std::byte*>(memory_statistics_host->data());
    CUDA_CHECK(cudaMemcpyAsync(host, buffers.key_sums.data, buffers.key_sums.bytes(),
        cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(host + buffers.key_sums.bytes(), buffers.query_sums.data,
        buffers.query_sums.bytes(), cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
    const auto width = static_cast<std::uint32_t>(buffers.key_sums.ne[0]);
    const auto layer_stride = static_cast<std::size_t>(width) * buffers.key_sums.ne[1];
    const auto* keys = reinterpret_cast<const float*>(host);
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto at = begin; at < end;) {
            const auto count = std::min(end - at, 64 - at % 64);
            const auto bucket = at / 64 - begin / 64;
            const auto* sum = keys + layer * layer_stride + bucket * width;
            for (std::uint32_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(sum[channel])) { throw std::runtime_error("Nonfinite KVMem key sum"); }
            }
            state.statistics->write_layer_mean_sum(at, count, layer, sum);
            at += count;
        }
    }
    const auto query_first = std::max(begin, state.query_begin);
    const auto query_last = std::min(end, state.query_end);
    if (!state.query_replayed && query_last > query_first) {
        const auto* query = reinterpret_cast<const float*>(host + buffers.key_sums.bytes());
        for (std::size_t i = 0; i < state.query_sum.size(); ++i) {
            if (!std::isfinite(query[i])) { throw std::runtime_error("Nonfinite KVMem query sum"); }
            state.query_sum[i] += query[i];
        }
        state.query_tokens += query_last - query_first;
    }
    if (state.selector->total_tokens() != begin) {
        throw std::logic_error("KVMem block catalog lost its statistics frontier");
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
    if (kvmem_options.verify_transfers && (end == state.query_end || end % 64 == 0)) {
        std::fprintf(stderr, "KVMEM_STATISTICS frontier=%u query_tokens=%u query_begin=%u query_end=%u layers=%u mean_bytes=%zu\n",
            end, state.query_tokens, state.query_begin, state.query_end,
            state.statistics->config().n_layer, state.statistics->allocated_bytes());
    }
}
void ProgramImpl::read_memory_candidates() {
    if (!memory_candidate_statistics) return;
    const auto& keys = memory_candidate_statistics->key_sums;
    CUDA_CHECK(cudaMemcpyAsync(memory_candidate_host->data(), keys.data, keys.bytes(),
        cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
}

void ProgramImpl::commit_memory_candidates(SequenceState& sequence, std::uint32_t begin,
                                            std::uint32_t end, std::uint32_t first_column,
                                            std::uint32_t row_width) {
    if (!memory_candidate_statistics) { return; }
    auto& state = sequence.window;
    const auto& keys = memory_candidate_statistics->key_sums;
    if (!state.statistics || !state.selector || begin != state.statistics_frontier ||
        state.selector->total_tokens() != begin || end <= begin ||
        end - begin > row_width || first_column + row_width > static_cast<std::uint32_t>(keys.ne[1])) {
        throw std::logic_error("KVMem speculative statistics have an invalid committed prefix");
    }
    const auto width = static_cast<std::size_t>(keys.ne[0]);
    const auto stride = width * keys.ne[1];
    const auto* values = static_cast<const float*>(memory_candidate_host->data());
    // Each represented BF16 key was copied into its own FP32 column by
    // token_sums(block_tokens=1). Rejected and output-truncated columns never
    // enter the persistent mean-K store, including a partial logical page.
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto position = begin; position < end; ++position) {
            const auto* value = values + layer * stride + (first_column + position - begin) * width;
            for (std::size_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(value[channel])) {
                    throw std::runtime_error("Nonfinite committed KVMem candidate key");
                }
            }
            state.statistics->write_layer_mean_sum(position, 1, layer, value);
        }
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
}
} // namespace ninfer::models::qwen3_5::detail
