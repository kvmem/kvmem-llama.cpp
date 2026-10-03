#include "models/qwen3_5/state/decoder_state.h"

#include "ops/kv_cache/d256_profile.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(std::span<LayoutBuilder* const> builders,
                              std::span<const std::size_t> layer_rank, std::uint32_t layers,
                              std::uint32_t capacity, std::int32_t kv_heads, std::int32_t head_dim,
                              KvCacheStorage storage, std::int32_t table_rows,
                              std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    if (head_dim != ops::kD256KVCacheHeadDim) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }
    const ops::D256KVCacheProfile profile = ops::d256_kv_cache_profile(storage);
    const bool scaled                     = profile.quant_group != 0;

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    if (!layer_rank.empty() && layer_rank.size() != layers) {
        throw std::invalid_argument("Paged KV layer rank map does not cover every layer");
    }
    const auto rank_of = [&](std::uint32_t layer) {
        return layer_rank.empty() ? std::size_t{0} : layer_rank[layer];
    };

    KVPageGeometry geometry;
    std::vector<std::size_t> plane_rank;
    const std::size_t planes_per_layer = scaled ? 4ULL : 2ULL;
    geometry.planes.reserve(static_cast<std::size_t>(layers) * planes_per_layer);
    plane_rank.reserve(static_cast<std::size_t>(layers) * planes_per_layer);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {profile.key_code_dtype, profile.key_leading_extent, kv_heads, 256});
        geometry.planes.push_back(
            {profile.value_code_dtype, profile.value_leading_extent, kv_heads, 256});
        if (scaled) {
            geometry.planes.push_back(
                {profile.key_scale_dtype, profile.scale_leading_extent, kv_heads, 256});
            geometry.planes.push_back(
                {profile.value_scale_dtype, profile.value_scale_leading_extent, kv_heads, 256});
        }
        plane_rank.insert(plane_rank.end(), planes_per_layer, rank_of(layer));
    }

    // A block-table copy on every rank that runs attention layers of this cache, and always on
    // rank 0, which is where callers that address the tables directly look.
    std::vector<std::size_t> table_ranks{0};
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::size_t rank = rank_of(layer);
        if (std::find(table_ranks.begin(), table_ranks.end(), rank) == table_ranks.end()) {
            table_ranks.push_back(rank);
        }
    }
    std::sort(table_ranks.begin(), table_ranks.end());

    PagedKVCacheLayout out{
        .pages = plan_device_kv_page_pool(
            builders, plane_rank,
            DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                 .geometry         = std::move(geometry)}),
        .execution_tables = {},
        .layers         = layers,
        .max_context    = capacity,
        .kv_heads       = kv_heads,
        .head_dim       = head_dim,
        .storage        = storage,
    };
    for (const std::size_t rank : table_ranks) {
        if (rank >= builders.size() || builders[rank] == nullptr) {
            throw std::invalid_argument("Paged KV layer names a rank with no layout builder");
        }
        out.execution_tables.push_back(plan_kv_execution_tables(
            *builders[rank],
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows},
            rank));
    }
    return out;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    LayoutBuilder* const builders[] = {&builder};
    return plan_decoder_state(builders, spec);
}

DecoderStateLayout plan_decoder_state(std::span<LayoutBuilder* const> builders,
                                      const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builders, spec.text_layer_rank, spec.full_attention_layers,
                                spec.capacity, spec.kv_heads, spec.attention_head_dim,
                                spec.kv_storage, spec.kv_table_rows, spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builders.first(1), {}, spec.mtp_layers, spec.capacity,
                                   spec.kv_heads, spec.attention_head_dim, spec.kv_storage,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : PagedKVCache(std::span<const DeviceSpan>(&backing, 1), layout) {}

PagedKVCache::PagedKVCache(std::span<const DeviceSpan> backings, const PagedKVCacheLayout& layout)
    : pages_(backings, layout.pages), execution_tables_(backings, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), storage_(layout.storage) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, KVExecutionRowHandle row) noexcept
    : cache_(&cache), row_(row) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, &row_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, row.handle());
}

std::size_t PagedKVCache::layer_rank(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool scaled        = ops::d256_kv_cache_profile(storage_).quant_group != 0;
    const std::size_t stride = scaled ? 4ULL : 2ULL;
    return pages_.plane_rank(static_cast<std::size_t>(layer) * stride);
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer,
                                          const KVExecutionRowHandle* row) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool scaled        = ops::d256_kv_cache_profile(storage_).quant_group != 0;
    const std::size_t stride = scaled ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    // The layer reads the copy of the table on its own rank; a kernel cannot follow a pointer into
    // another device's memory.
    const Tensor block_table =
        row == nullptr ? Tensor() : execution_tables_.row(*row, layer_rank(layer));
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = scaled ? pages_.plane(base + 2) : Tensor(),
        .v_scale_pages = scaled ? pages_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .storage       = storage_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    const PagedKVLayerView direct = layer_view(layer, nullptr);
    return PagedKVBatchLayerView{
        .k_pages       = direct.k_pages,
        .v_pages       = direct.v_pages,
        .k_scale_pages = direct.k_scale_pages,
        .v_scale_pages = direct.v_scale_pages,
        .block_tables  = execution_tables_.matrix(layer_rank(layer)),
        .head_dim      = direct.head_dim,
        .num_kv_heads  = direct.num_kv_heads,
        .storage       = direct.storage,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : DecoderState(std::span<const DeviceSpan>(&backing, 1), layout) {}

DecoderState::DecoderState(std::span<const DeviceSpan> backings, const DecoderStateLayout& layout)
    : text_kv(backings, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backings, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::models::qwen3_5
