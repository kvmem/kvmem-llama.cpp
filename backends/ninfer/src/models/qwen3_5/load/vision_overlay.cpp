#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::loading {
namespace {

// Eviction ladder: a higher rank sits nearer the arena end and is borrowed first. Only the
// endpoint weights and the loaded speculative component are ranked, so the decoder layers are
// never touched. A ternary token table and output head are small enough that the drafter's
// adapter has to make up an encode window.
constexpr std::uint32_t kEvictRankDraft        = 300;
constexpr std::uint32_t kEvictRankMtp          = 400;
constexpr std::uint32_t kEvictRankProposalHead = 500;
constexpr std::uint32_t kEvictRankEmbedding    = 600;
constexpr std::uint32_t kEvictRankOutputHead   = 700;

constexpr std::size_t kStagingAlignment = 256;

std::size_t staging_align(std::size_t bytes) {
    return (bytes + kStagingAlignment - 1) / kStagingAlignment * kStagingAlignment;
}

// Pinned extent covering exactly the parents of one group. No foreign pinned parent may fall
// inside it, so uploading the extent stages the group and nothing else.
PinnedRange group_range(std::span<const WeightId> group, std::span<const BoundWeight> bound,
                        std::span<const std::byte> block) {
    const auto* const base = block.data();
    std::size_t begin      = std::numeric_limits<std::size_t>::max();
    std::size_t end        = 0;
    std::vector<const WeightParent*> members;
    for (const WeightId id : group) {
        for (const WeightRegion& part : bound[id.index].view.parts) {
            const auto* data = part.parent->data;
            if (data < base || data + part.parent->geometry.bytes > base + block.size()) {
                throw std::logic_error(bound[id.index].name +
                                       ": Vision overlay weight is outside the pinned block");
            }
            const auto offset = static_cast<std::size_t>(data - base);
            begin             = std::min(begin, offset);
            end = std::max(end, offset + static_cast<std::size_t>(part.parent->geometry.bytes));
            members.push_back(part.parent);
        }
    }
    if (members.empty() || end <= begin) {
        throw std::logic_error("Vision overlay group has no pinned weights");
    }
    for (const BoundWeight& weight : bound) {
        for (const WeightRegion& part : weight.view.parts) {
            const auto* data = part.parent->data;
            if (data < base || data >= base + block.size() ||
                std::find(members.begin(), members.end(), part.parent) != members.end()) {
                continue;
            }
            const auto offset = static_cast<std::size_t>(data - base);
            if (offset < end && offset + part.parent->geometry.bytes > begin) {
                throw std::logic_error("Vision overlay group interleaves with " + weight.name);
            }
        }
    }
    return {begin, end - begin};
}

} // namespace

void apply_vision_overlay_placement(Bindings& bindings, const ModelWeights& weights,
                                    std::pair<std::size_t, std::size_t> mtp_parameters,
                                    std::pair<std::size_t, std::size_t> draft_parameters) {
    for (std::size_t index = draft_parameters.first; index < draft_parameters.second; ++index) {
        bindings.evict(WeightId{index}, kEvictRankDraft);
    }
    for (std::size_t index = mtp_parameters.first; index < mtp_parameters.second; ++index) {
        bindings.evict(WeightId{index}, kEvictRankMtp);
    }
    if (weights.proposal) { bindings.evict(weights.proposal->head, kEvictRankProposalHead); }
    bindings.evict(weights.text.token_embedding, kEvictRankEmbedding);
    bindings.evict(weights.text.output_head, kEvictRankOutputHead);
}

VisionOverlayLayout vision_overlay_layout(const VisionWeights& weights,
                                          std::span<const BoundWeight> bound,
                                          std::span<const std::byte> pinned_block) {
    VisionOverlayLayout out;
    const std::array prelude{weights.patch_embedding, weights.patch_embedding_bias,
                             weights.position_embedding};
    out.prelude = group_range(prelude, bound, pinned_block);
    out.layers.reserve(weights.layers.size());
    for (const VisionBlockWeights& layer : weights.layers) {
        const std::array group{layer.norm1.weight, layer.norm1.bias,  layer.norm2.weight,
                               layer.norm2.bias,   layer.query,       layer.key,
                               layer.value,        layer.query_bias,  layer.key_bias,
                               layer.value_bias,   layer.output,      layer.output_bias,
                               layer.fc1,          layer.fc1_bias,    layer.fc2,
                               layer.fc2_bias};
        out.layers.push_back(group_range(group, bound, pinned_block));
        out.slot_bytes = std::max(out.slot_bytes, out.layers.back().bytes);
    }
    const std::array merger{weights.merger_norm.weight, weights.merger_norm.bias,
                            weights.merger_fc1,         weights.merger_fc1_bias,
                            weights.merger_fc2,         weights.merger_fc2_bias};
    out.merger        = group_range(merger, bound, pinned_block);
    out.staging_bytes = staging_align(out.prelude.bytes) + staging_align(out.merger.bytes) +
                        2 * staging_align(out.slot_bytes);
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
