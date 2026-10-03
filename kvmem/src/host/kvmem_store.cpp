#include "kvmem/kvmem_store.hpp"
#include "kvmem/memory_contract.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace kvmem {

namespace {

uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    if (value == 0) return 0;
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(value) + divisor - 1) / divisor);
}

uint32_t clamp_u32(uint32_t value, uint32_t lo, uint32_t hi) {
    return std::min(std::max(value, lo), hi);
}

uint32_t ceil_percent(uint32_t value, uint32_t percent) {
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(value) * percent + 99u) / 100u);
}

}  // namespace

void KvMemStore::set_runtime_select_budget(uint32_t tokens) {
    if (tokens == 0) {
        runtime_select_budget_ = 0;
        return;
    }
    if (tokens > cfg_.select_budget) {
        throw std::invalid_argument(
            "KVMem request semantic budget exceeds configured maximum");
    }
    if (tokens < cfg_.block_tokens || tokens % cfg_.block_tokens != 0) {
        throw std::invalid_argument(
            "KVMem request semantic budget must be a positive multiple of "
            "block_tokens");
    }
    const uint32_t blocks = tokens / cfg_.block_tokens;
    if (static_cast<uint64_t>(cfg_.sink_blocks) + cfg_.recent_blocks >
        blocks) {
        throw std::invalid_argument(
            "KVMem request semantic budget is smaller than the configured "
            "sink + recent allocation");
    }
    runtime_select_budget_ = tokens;
}

KvMemKeepAllocation resolve_kvmem_keep_allocation(
        uint32_t block_tokens,
        uint32_t select_budget,
        int64_t sink_blocks,
        int64_t recent_blocks,
        int64_t sink_tokens,
        int64_t recent_tokens) {
    if (block_tokens == 0) {
        throw std::runtime_error(
            "KVMem keep allocation requires block_tokens > 0");
    }
    if (select_budget < block_tokens) {
        throw std::runtime_error(
            "KVMem selection budget must contain at least one block");
    }
    if (sink_blocks >= 0 && sink_tokens >= 0) {
        throw std::runtime_error(
            "KVMem sink allocation cannot specify both blocks and tokens");
    }
    if (recent_blocks >= 0 && recent_tokens >= 0) {
        throw std::runtime_error(
            "KVMem recent allocation cannot specify both blocks and tokens");
    }

    KvMemKeepAllocation out;
    const uint32_t budget_blocks = select_budget / block_tokens;

    auto resolve_band = [&](int64_t explicit_blocks,
                            int64_t explicit_tokens,
                            uint32_t auto_tokens,
                            const char *name,
                            uint32_t &target_tokens,
                            uint32_t &blocks,
                            uint32_t &effective_tokens,
                            KvMemKeepSource &source) {
        if (explicit_blocks >= 0) {
            if (static_cast<uint64_t>(explicit_blocks) >
                std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("KVMem ") + name +
                    " block allocation is too large");
            }
            blocks = static_cast<uint32_t>(explicit_blocks);
            const uint64_t tokens =
                static_cast<uint64_t>(blocks) * block_tokens;
            if (tokens > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    std::string("KVMem ") + name +
                    " block allocation overflows token accounting");
            }
            target_tokens = static_cast<uint32_t>(tokens);
            effective_tokens = target_tokens;
            source = KvMemKeepSource::Blocks;
            return;
        }

        const uint64_t requested = explicit_tokens >= 0
            ? static_cast<uint64_t>(explicit_tokens)
            : static_cast<uint64_t>(auto_tokens);
        if (requested > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                std::string("KVMem ") + name +
                " token allocation is too large");
        }
        target_tokens = static_cast<uint32_t>(requested);
        blocks = ceil_div_u32(target_tokens, block_tokens);
        const uint64_t rounded_tokens =
            static_cast<uint64_t>(blocks) * block_tokens;
        if (rounded_tokens > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(
                std::string("KVMem ") + name +
                " rounded token allocation overflows accounting");
        }
        effective_tokens = static_cast<uint32_t>(rounded_tokens);
        source = explicit_tokens >= 0
            ? KvMemKeepSource::Tokens
            : KvMemKeepSource::Auto;
    };

    const uint32_t auto_sink_tokens =
        clamp_u32(ceil_percent(select_budget, 1), 1024, 2048);
    const uint32_t auto_recent_tokens =
        clamp_u32(ceil_percent(select_budget, 8), 4096, 16384);
    resolve_band(
        sink_blocks, sink_tokens, auto_sink_tokens, "sink",
        out.sink_target_tokens, out.sink_blocks,
        out.sink_effective_tokens, out.sink_source);
    resolve_band(
        recent_blocks, recent_tokens, auto_recent_tokens, "recent",
        out.recent_target_tokens, out.recent_blocks,
        out.recent_effective_tokens, out.recent_source);

    if (out.sink_blocks > budget_blocks ||
        out.recent_blocks > budget_blocks ||
        static_cast<uint64_t>(out.sink_blocks) + out.recent_blocks >
            budget_blocks) {
        throw std::runtime_error(
            "KVMem sink + recent allocation exceeds the selection budget");
    }
    return out;
}

int32_t KvMemStore::block_id_containing(uint32_t pos) const {
    if (blocks_.empty() || pos >= total_tokens_) {
        return -1;
    }
    uint32_t id = cfg_.block_tokens == 0 ? 0 : pos / cfg_.block_tokens;
    if (id >= blocks_.size()) {
        id = static_cast<uint32_t>(blocks_.size() - 1);
    }
    const KvMemBlock &guess = blocks_[id];
    if (pos >= guess.orig_pos_start && pos < guess.orig_pos_end()) {
        return static_cast<int32_t>(id);
    }
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        const KvMemBlock &b = blocks_[i];
        if (pos >= b.orig_pos_start && pos < b.orig_pos_end()) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

void KvMemStore::register_append(uint32_t n_new_tokens) {
    if (n_new_tokens == 0) return;
    const uint32_t bt = cfg_.block_tokens;

    // Extend the trailing partial block first, then spill into new blocks.
    uint32_t remaining = n_new_tokens;
    if (!blocks_.empty()) {
        KvMemBlock &last = blocks_.back();
        if (last.n_tokens < bt) {
            const uint32_t room = bt - last.n_tokens;
            const uint32_t take = std::min(room, remaining);
            last.n_tokens += take;
            remaining -= take;
            total_tokens_ += take;
        }
    }
    while (remaining > 0) {
        KvMemBlock b;
        b.block_id = static_cast<uint32_t>(blocks_.size());
        b.orig_pos_start = total_tokens_;
        b.n_tokens = std::min(bt, remaining);
        b.baked_pos = static_cast<int64_t>(b.orig_pos_start);  // baked at true pos after prefill
        blocks_.push_back(b);
        total_tokens_ += b.n_tokens;
        remaining -= b.n_tokens;
    }
}

std::vector<KvMemDroppedBlock> KvMemStore::truncate_to(uint32_t token_pos) {
    std::vector<KvMemDroppedBlock> dropped;
    if (token_pos >= total_tokens_) return dropped;
    // Pop fully-past blocks from the back, then shrink the trailing partial one.
    while (!blocks_.empty() && blocks_.back().orig_pos_start >= token_pos) {
        const KvMemBlock &b = blocks_.back();
        dropped.push_back(KvMemDroppedBlock{b.block_id, b.tier,
                                            b.gpu_slot, b.cpu_slot, b.nvme_slot});
        total_tokens_ -= b.n_tokens;
        blocks_.pop_back();
    }
    if (!blocks_.empty()) {
        KvMemBlock &last = blocks_.back();
        if (last.orig_pos_end() > token_pos) {
            const uint32_t keep = token_pos - last.orig_pos_start;
            total_tokens_ -= (last.n_tokens - keep);
            last.n_tokens = keep;
        }
    }
    return dropped;
}

void KvMemStore::accumulate_attn(const std::vector<double> &scores) {
    const uint32_t n = std::min<uint32_t>(static_cast<uint32_t>(scores.size()),
                                          block_count());
    for (uint32_t i = 0; i < n; ++i) {
        blocks_[i].profile_score += scores[i];
        blocks_[i].attn_score = blocks_[i].profile_score;
    }
}

void KvMemStore::set_attn_scores(const std::vector<double> &scores) {
    set_retrieval_scores(scores);
}

void KvMemStore::set_retrieval_scores(const std::vector<double> &scores) {
    const uint32_t n = block_count();
    for (uint32_t i = 0; i < n; ++i) {
        blocks_[i].retrieval_score = (i < scores.size()) ? scores[i] : 0.0;
        blocks_[i].attn_score = blocks_[i].retrieval_score;
    }
}

void KvMemStore::set_block_tier(uint32_t block_id, KvTier tier,
                                int32_t cpu_slot, int32_t nvme_slot) {
    if (block_id >= block_count()) return;
    KvMemBlock &b = blocks_[block_id];
    b.tier = tier;
    if (cfg_.optimize_stage_out) {
        if (tier == KvTier::GPU) {
            // GPU becomes active, but clean CPU/SSD cache copies remain valid.
            if (cpu_slot >= 0) b.cpu_slot = cpu_slot;
            if (nvme_slot >= 0) b.nvme_slot = nvme_slot;
        } else {
            b.gpu_slot = -1;
            b.cpu_slot = cpu_slot;
            if (nvme_slot >= 0) b.nvme_slot = nvme_slot;
        }
        return;
    }
    if (tier == KvTier::GPU) {
        b.cpu_slot = -1;
        b.nvme_slot = -1;
        b.ssd_clean = false;
    } else {
        b.gpu_slot = -1;
        b.cpu_slot = cpu_slot;
        b.nvme_slot = nvme_slot;
        b.ssd_clean = tier == KvTier::SSD && nvme_slot >= 0;
    }
}

void KvMemStore::set_block_gpu_slot(uint32_t block_id, int32_t gpu_slot) {
    if (block_id >= block_count()) return;
    blocks_[block_id].gpu_slot = gpu_slot;
}

void KvMemStore::set_block_cpu_copy(uint32_t block_id, int32_t cpu_slot) {
    if (block_id >= block_count()) return;
    blocks_[block_id].cpu_slot = cpu_slot;
}

void KvMemStore::set_block_ssd_backing(uint32_t block_id, int32_t nvme_slot,
                                       bool clean) {
    if (block_id >= block_count()) return;
    KvMemBlock &b = blocks_[block_id];
    b.nvme_slot = nvme_slot;
    b.ssd_clean = clean && nvme_slot >= 0;
}

void KvMemStore::set_block_io_in_flight(uint32_t block_id, bool in_flight) {
    if (block_id >= block_count()) return;
    blocks_[block_id].in_flight = in_flight;
}

void KvMemStore::set_block_baked_pos(uint32_t block_id, int64_t baked_pos) {
    if (block_id >= block_count()) return;
    blocks_[block_id].baked_pos = baked_pos;
}

void KvMemStore::record_block_rerope(uint32_t block_id, int64_t baked_pos) {
    if (block_id >= block_count()) return;
    KvMemBlock &block = blocks_[block_id];
    if (block.baked_pos != baked_pos) {
        ++block.remap_count;
        const uint64_t delta = static_cast<uint64_t>(
            block.baked_pos > baked_pos ? block.baked_pos - baked_pos
                                        : baked_pos - block.baked_pos);
        block.remap_abs_delta += delta;
    }
    block.baked_pos = baked_pos;
}

uint32_t KvMemStore::gpu_high_watermark_tokens(uint32_t pool_tokens) const {
    double r = cfg_.gpu_high_watermark;
    if (r <= 0.0 || r > 1.0) {
        r = 1.0;
    }
    const uint32_t n = static_cast<uint32_t>(pool_tokens * r);
    return std::max(n, cfg_.block_tokens);
}

uint32_t KvMemStore::gpu_low_watermark_tokens(uint32_t pool_tokens) const {
    double r = cfg_.gpu_low_watermark;
    if (r <= 0.0 || r > 1.0) {
        r = 1.0;
    }
    const uint32_t high = gpu_high_watermark_tokens(pool_tokens);
    uint32_t n = static_cast<uint32_t>(pool_tokens * r);
    n = std::max(n, cfg_.block_tokens);
    return std::min(n, high);
}

bool KvMemStore::prefill_needs_offload(uint32_t resident_tokens,
                                       uint32_t incoming_tokens,
                                       uint32_t pool_tokens) const {
    if (incoming_tokens == 0 || pool_tokens == 0) {
        return false;
    }
    const uint32_t next = resident_tokens + incoming_tokens;
    if (next > pool_tokens) {
        return true;
    }
    return next > gpu_high_watermark_tokens(pool_tokens);
}

std::vector<uint32_t> KvMemStore::pick_prefill_pressure_blocks(const std::vector<uint32_t> & mandatory) const {
    return constrain_media(pick_prefill_ungrouped(mandatory), mandatory, prefill_budget_blocks(), true);
}

std::vector<uint32_t> KvMemStore::pick_topk_blocks(const std::vector<uint32_t> & mandatory) const {
    return constrain_media(pick_topk_ungrouped(mandatory), mandatory, budget_blocks(), false);
}

std::vector<uint32_t> KvMemStore::constrain_media(std::vector<uint32_t> selected,
        const std::vector<uint32_t> & mandatory, uint32_t budget, bool recency) const {
    const uint32_t n = block_count();
    if (media_ranges_.empty() || n <= budget || budget == 0) return selected;
    std::vector<std::pair<uint32_t, uint32_t>> groups;
    for (const auto & range : media_ranges_) {
        if (range.first >= total_tokens_ || range.second <= range.first) continue;
        const uint32_t first = range.first / cfg_.block_tokens;
        const uint32_t end = std::min(n, (range.second + cfg_.block_tokens - 1) / cfg_.block_tokens);
        if (!groups.empty() && first < groups.back().second) groups.back().second = std::max(groups.back().second, end);
        else groups.emplace_back(first, end);
    }
    std::vector<uint32_t> lo(n), hi(n);
    for (uint32_t i = 0; i < n; ++i) { lo[i] = i; hi[i] = i + 1; }
    for (const auto & group : groups) for (uint32_t i = group.first; i < group.second; ++i) {
        lo[i] = group.first; hi[i] = group.second;
    }
    std::vector<bool> kept(n, false);
    uint32_t count = 0;
    auto keep = [&](uint32_t id, bool required) {
        if (id >= n) return;
        uint32_t need = 0;
        for (uint32_t i = lo[id]; i < hi[id]; ++i) need += !kept[i];
        if (count + need > budget) {
            if (required) throw std::runtime_error("mandatory image group and query exceed KV selection budget");
            return;
        }
        for (uint32_t i = lo[id]; i < hi[id]; ++i) kept[i] = true;
        count += need;
    };
    for (uint32_t i = 0; i < std::min(n, cfg_.sink_blocks); ++i) keep(i, true);
    // Keep the latest image as a whole, including its boundary blocks.
    if (!groups.empty()) keep(groups.back().first, true);
    // Retrieval suffixes are best effort when they exceed the budget. Preserve
    // the newest rows after reserving the sink and latest complete image; the
    // caller must skip query replay if any replay block was dropped.
    // During prefill these are incoming rows, which must all have slots.
    auto newest = mandatory;
    std::sort(newest.begin(), newest.end(), std::greater<uint32_t>());
    for (auto id : newest) keep(id, recency);
    auto better = [&](uint32_t a, uint32_t b) {
        if (!recency && blocks_[a].attn_score != blocks_[b].attn_score) return blocks_[a].attn_score > blocks_[b].attn_score;
        return a > b;
    };
    std::sort(selected.begin(), selected.end(), better);
    for (auto id : selected) keep(id, false);
    std::vector<uint32_t> rest;
    for (uint32_t i = 0; i < n; ++i) if (!kept[i]) rest.push_back(i);
    std::sort(rest.begin(), rest.end(), better);
    for (auto id : rest) keep(id, false);
    selected.clear();
    for (uint32_t i = 0; i < n; ++i) if (kept[i]) selected.push_back(i);
    return selected;
}

std::vector<uint32_t> KvMemStore::pick_prefill_pressure_blocks() const {
    return pick_prefill_pressure_blocks({});
}

std::vector<uint32_t> KvMemStore::pick_prefill_ungrouped(
        const std::vector<uint32_t> &mandatory_blocks) const {
    const uint32_t n = block_count();
    std::vector<uint32_t> selected;
    if (n == 0) return selected;

    const uint32_t budget = prefill_budget_blocks();
    if (budget == 0 || n <= budget) {
        selected.reserve(n);
        for (uint32_t i = 0; i < n; ++i) selected.push_back(i);
        return selected;
    }

    const uint32_t sink = std::min({cfg_.sink_blocks, budget, n});
    std::vector<uint8_t> kept(n, 0);
    uint32_t kept_count = 0;
    auto keep = [&](uint32_t id) {
        if (id < n && !kept[id]) {
            kept[id] = 1;
            ++kept_count;
        }
    };
    for (uint32_t id = 0; id < sink && kept_count < budget; ++id) keep(id);
    // Newest mandatory first so a huge query suffix cannot evict the tail
    // or throw. Drop older mandatory blocks that do not fit.
    uint32_t mand_kept = 0;
    uint32_t mand_unique = 0;
    {
        std::vector<uint8_t> seen(n, 0);
        for (uint32_t id : mandatory_blocks) {
            if (id < n && !seen[id]) {
                seen[id] = 1;
                ++mand_unique;
            }
        }
    }
    for (size_t i = mandatory_blocks.size(); i > 0 && kept_count < budget; --i) {
        keep(mandatory_blocks[i - 1]);
    }
    // Diagnostic membership includes mandatory blocks already kept as sink.
    // Selection and its priority order above are unchanged.
    {
        std::vector<uint8_t> counted(n, 0);
        for (uint32_t id : mandatory_blocks) {
            if (id < n && kept[id] && !counted[id]) { counted[id] = 1; ++mand_kept; }
        }
    }
    if (mand_kept < mand_unique) {
        std::fprintf(stderr,
                     "KVMEM_TRACE mandatory_trim policy=prefill kept=%u dropped=%u budget=%u\n",
                     mand_kept, mand_unique - mand_kept, budget);
    }
    for (uint32_t id = n; id > 0 && kept_count < budget; --id) {
        keep(id - 1);
    }
    selected.reserve(kept_count);
    for (uint32_t id = 0; id < n; ++id) {
        if (kept[id]) selected.push_back(id);
    }
    return selected;
}

std::vector<uint32_t> KvMemStore::pick_topk_blocks() const {
    return pick_topk_blocks({});
}

std::vector<uint32_t> KvMemStore::pick_topk_ungrouped(
        const std::vector<uint32_t> &mandatory_blocks) const {
    const uint32_t n = block_count();
    std::vector<uint32_t> selected;
    if (n == 0) return selected;

    const uint32_t budget = budget_blocks();
    if (budget == 0 || n <= budget) {
        // Everything fits: select all in order.
        selected.reserve(n);
        for (uint32_t i = 0; i < n; ++i) selected.push_back(i);
        return selected;
    }

    // Always-keep windows: first `sink_blocks` and last `recent_blocks`.
    const uint32_t sink = std::min(cfg_.sink_blocks, n);
    // Zero is literal: do not reserve any suffix blocks.  Earlier versions used
    // zero as an implicit "auto = budget/4", which made a 200K-token budget
    // silently pin 50K tokens and was both surprising and hard to control.
    const uint32_t recent = std::min(cfg_.recent_blocks, n);

    std::vector<bool> kept(n, false);
    uint32_t kept_count = 0;
    auto keep = [&](uint32_t id) {
        if (id < n && !kept[id]) { kept[id] = true; ++kept_count; }
    };
    for (uint32_t i = 0; i < sink && kept_count < budget; ++i) keep(i);
    uint32_t mand_kept = 0;
    uint32_t mand_unique = 0;
    {
        std::vector<uint8_t> seen(n, 0);
        for (uint32_t id : mandatory_blocks) {
            if (id < n && !seen[id]) {
                seen[id] = 1;
                ++mand_unique;
            }
        }
    }
    for (size_t i = mandatory_blocks.size(); i > 0 && kept_count < budget; --i) {
        keep(mandatory_blocks[i - 1]);
    }
    // Diagnostic membership includes mandatory blocks already kept as sink.
    // Selection and its priority order above are unchanged.
    {
        std::vector<uint8_t> counted(n, 0);
        for (uint32_t id : mandatory_blocks) {
            if (id < n && kept[id] && !counted[id]) { counted[id] = 1; ++mand_kept; }
        }
    }
    if (mand_kept < mand_unique) {
        std::fprintf(stderr,
                     "KVMEM_TRACE mandatory_trim policy=topk kept=%u dropped=%u budget=%u\n",
                     mand_kept, mand_unique - mand_kept, budget);
    }
    for (uint32_t i = 0; i < recent && kept_count < budget; ++i) {
        keep(n - 1 - i);
    }

    auto take_top = [&](uint32_t quota, auto score_fn) {
        if (kept_count >= budget || quota == 0) return;
        std::vector<uint32_t> candidates;
        candidates.reserve(n - kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (!kept[i]) candidates.push_back(i);
        }
        if (candidates.empty()) return;
        const uint32_t need = std::min<uint32_t>(
            std::min<uint32_t>(quota, budget - kept_count),
            static_cast<uint32_t>(candidates.size()));
        auto better = [&](uint32_t a, uint32_t b) {
            const double sa = score_fn(blocks_[a]);
            const double sb = score_fn(blocks_[b]);
            if (sa != sb) return sa > sb;
            return a > b;
        };
        if (need < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + need,
                             candidates.end(), better);
            candidates.resize(need);
        }
        for (uint32_t id : candidates) keep(id);
    };

    if (cfg_.select_policy == KvMemSelectPolicy::Quota && kept_count < budget) {
        uint32_t remaining = budget - kept_count;
        uint32_t retrieval_quota = cfg_.retrieval_blocks;
        uint32_t profile_quota = cfg_.profile_blocks;
        if (retrieval_quota == 0 && profile_quota == 0) {
            retrieval_quota = (remaining * 2) / 3;
            profile_quota = remaining - retrieval_quota;
        } else if (retrieval_quota == 0) {
            retrieval_quota = remaining > profile_quota ? remaining - profile_quota : 0;
        } else if (profile_quota == 0) {
            profile_quota = remaining > retrieval_quota ? remaining - retrieval_quota : 0;
        }

        take_top(retrieval_quota, [](const KvMemBlock &b) {
            return b.retrieval_score;
        });
        take_top(profile_quota, [](const KvMemBlock &b) {
            return b.profile_score;
        });
        // Fill any leftover quota with the configured method's combined score
        // so rounding, overlap, or zero explicit quotas still use the budget.
        if (kept_count < budget) {
            take_top(budget - kept_count, [](const KvMemBlock &b) {
                return b.attn_score;
            });
        }
        selected.reserve(kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (kept[i]) selected.push_back(i);
        }
        return selected;
    }

    // Fill the rest with the highest cumulative-attention middle blocks.
    if (kept_count < budget) {
        std::vector<uint32_t> candidates;
        candidates.reserve(n - kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (!kept[i]) candidates.push_back(i);
        }
        const uint32_t need = budget - kept_count;
        // Partial sort by attn_score desc; tie-break by recency (higher id) so
        // selection is deterministic.
        auto better = [&](uint32_t a, uint32_t b) {
            if (blocks_[a].attn_score != blocks_[b].attn_score) {
                return blocks_[a].attn_score > blocks_[b].attn_score;
            }
            return a > b;
        };
        if (need < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + need,
                             candidates.end(), better);
            candidates.resize(need);
        }
        for (uint32_t id : candidates) keep(id);
    }

    selected.reserve(kept_count);
    for (uint32_t i = 0; i < n; ++i) {
        if (kept[i]) selected.push_back(i);
    }
    return selected;
}

std::vector<uint32_t> KvMemStore::pick_semantic_groups(
        const std::vector<std::pair<uint32_t, uint32_t>> &groups,
        const std::vector<double> &group_scores,
        const std::vector<uint32_t> &mandatory_blocks) const {
    const uint32_t n = block_count();
    std::vector<uint32_t> selected;
    if (n == 0) return selected;
    if (groups.size() != group_scores.size()) {
        throw std::runtime_error(
            "KVMem semantic group spans/scores size mismatch");
    }

    const uint32_t budget = budget_blocks();
    if (budget == 0 || n <= budget) {
        selected.reserve(n);
        for (uint32_t id = 0; id < n; ++id) selected.push_back(id);
        return selected;
    }

    std::vector<uint8_t> kept(n, 0);
    uint32_t kept_count = 0;
    auto keep = [&](uint32_t id) {
        if (id < n && !kept[id]) {
            kept[id] = 1;
            ++kept_count;
        }
    };

    const uint32_t sink = std::min(cfg_.sink_blocks, n);
    const uint32_t recent = std::min(cfg_.recent_blocks, n);
    for (uint32_t id = 0; id < sink && kept_count < budget; ++id) keep(id);
    uint32_t mand_kept = 0;
    uint32_t mand_unique = 0;
    {
        std::vector<uint8_t> seen(n, 0);
        for (uint32_t id : mandatory_blocks) {
            if (id < n && !seen[id]) {
                seen[id] = 1;
                ++mand_unique;
            }
        }
    }
    for (size_t i = mandatory_blocks.size(); i > 0 && kept_count < budget; --i) {
        keep(mandatory_blocks[i - 1]);
    }
    // Diagnostic membership includes mandatory blocks already kept as sink.
    // Selection and its priority order above are unchanged.
    {
        std::vector<uint8_t> counted(n, 0);
        for (uint32_t id : mandatory_blocks) {
            if (id < n && kept[id] && !counted[id]) { counted[id] = 1; ++mand_kept; }
        }
    }
    if (mand_kept < mand_unique) {
        std::fprintf(stderr,
                     "KVMEM_TRACE mandatory_trim policy=semantic kept=%u dropped=%u budget=%u\n",
                     mand_kept, mand_unique - mand_kept, budget);
    }
    for (uint32_t i = 0; i < recent && kept_count < budget; ++i) {
        keep(n - 1 - i);
    }

    struct Candidate {
        uint32_t group = 0;
        uint32_t first = 0;
        uint32_t last = 0;
        double score = 0.0;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(groups.size());
    const uint32_t bt = std::max<uint32_t>(1, cfg_.block_tokens);
    const uint64_t covered_tokens =
        static_cast<uint64_t>(blocks_.back().orig_pos_start) +
        blocks_.back().n_tokens;
    for (uint32_t g = 0; g < groups.size(); ++g) {
        const uint32_t begin = groups[g].first;
        const uint32_t end = groups[g].second;
        if (end <= begin || begin >= covered_tokens) continue;
        const uint32_t capped_end = static_cast<uint32_t>(
            std::min<uint64_t>(end, covered_tokens));
        const uint32_t first = begin / bt;
        const uint32_t last = (capped_end - 1) / bt;
        if (first >= n) continue;
        candidates.push_back(Candidate{
            g, first, std::min(last, n - 1), group_scores[g]});
    }

    // GPU has already reduced fine-grained sub-block scores to one score per
    // logical group. Sorting O(groups log groups) is negligible for normal
    // conversation histories. Recency is the deterministic tie-break.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &a, const Candidate &b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.group > b.group;
              });

    uint32_t selected_groups = 0;
    uint32_t skipped_for_budget = 0;
    for (const Candidate &candidate : candidates) {
        uint32_t new_blocks = 0;
        for (uint32_t id = candidate.first; id <= candidate.last; ++id) {
            new_blocks += kept[id] ? 0u : 1u;
        }
        if (new_blocks > budget - kept_count) {
            ++skipped_for_budget;
            continue;
        }
        for (uint32_t id = candidate.first; id <= candidate.last; ++id) {
            keep(id);
        }
        ++selected_groups;
        if (kept_count == budget) break;
    }

    selected.reserve(kept_count);
    for (uint32_t id = 0; id < n; ++id) {
        if (kept[id]) selected.push_back(id);
    }
    if (std::getenv("KVMEM_TRACE")) {
        std::fprintf(
            stderr,
            "[bs-semantic-select] mode=%s reduce=%s alpha=%.3f "
            "groups=%zu candidates=%zu selected_groups=%u "
            "selected_blocks=%u budget_blocks=%u skipped_for_budget=%u "
            "unused_blocks=%u\n",
            cfg_.semantic_expansion == KvMemSemanticExpansion::Message
                ? "message" : "round",
            cfg_.group_score_reduce ==
                    KvMemGroupScoreReduce::LengthNormalizedMass
                ? "length-normalized-mass" : "max",
            cfg_.group_length_norm_alpha,
            groups.size(), candidates.size(), selected_groups, kept_count,
            budget, skipped_for_budget, budget - kept_count);
    }
    return selected;
}

KvMemPlan KvMemStore::set_selection(std::vector<uint32_t> selected_ids,
                                    bool force_raw_refresh) {
    // Sort + dedupe so window order is deterministic (ascending block_id =
    // original chronological order: sink first ... recent last).
    std::sort(selected_ids.begin(), selected_ids.end());
    selected_ids.erase(std::unique(selected_ids.begin(), selected_ids.end()),
                       selected_ids.end());
    // Drop out-of-range IDs defensively (external selector could be stale).
    selected_ids.erase(
        std::remove_if(selected_ids.begin(), selected_ids.end(),
                       [&](uint32_t id) { return id >= block_count(); }),
        selected_ids.end());

    KvMemPlan plan;

    std::vector<bool> now_selected(block_count(), false);
    for (uint32_t id : selected_ids) now_selected[id] = true;
    std::vector<bool> was_in_working_set(block_count(), false);
    for (const KvMemBlock &b : blocks_) {
        was_in_working_set[b.block_id] = b.in_working_set;
        if (b.in_working_set && now_selected[b.block_id]) {
            ++plan.selection_overlap_blocks;
        }
    }
    const bool force_full_reload = !cfg_.optimize_stage_in;

    // Stage-out: any GPU-resident block that is not selected. On the first
    // post-prefill selection, many cold blocks were never in the prior working
    // set, but they still occupy GPU KV pages and must be eligible for offload.
    // Their cache pages keep whatever bake they currently hold (baked_pos
    // unchanged); a future re-selection will de-rotate from there.
    for (auto &b : blocks_) {
        if (b.tier == KvTier::GPU &&
            (!now_selected[b.block_id] || force_full_reload)) {
            plan.stage_out.push_back(b.block_id);
            b.in_working_set = false;
        }
    }

    // Pack selected blocks contiguously into the window in ascending order.
    uint32_t window_pos = 0;
    for (uint32_t id : selected_ids) {
        KvMemBlock &b = blocks_[id];
        const bool naturally_reusable =
            was_in_working_set[id] && b.tier == KvTier::GPU;
        if (naturally_reusable) {
            if (b.baked_pos == static_cast<int64_t>(window_pos)) {
                ++plan.retained_position_stable;
            } else {
                ++plan.retained_position_moved;
            }
            if (!force_full_reload) ++plan.gpu_reused_blocks;
        }
        const bool cold = b.tier != KvTier::GPU || force_full_reload;
        if (!b.in_working_set) plan.stage_in.push_back(id);

        KvMemRemap rm;
        rm.block_id = id;
        rm.n_tokens = b.n_tokens;
        rm.from_base = static_cast<int32_t>(b.baked_pos);   // de-rotate source
        rm.to_base = static_cast<int32_t>(window_pos);      // new window slot
        rm.working_k_resident = !cold;
        const bool same_position =
            b.baked_pos == static_cast<int64_t>(window_pos);
        // A cold immutable block has no valid rotated working K on GPU: the
        // tier record intentionally stores V but uses the position-free raw-K
        // mirror as K authority. Even when its compact position is unchanged,
        // it must be rematerialized after stage-in. Position equality alone is
        // therefore insufficient to skip assembly.
        rm.skip = same_position &&
                  (!cfg_.immutable_source_k || rm.working_k_resident) &&
                  !force_raw_refresh;
        if (cfg_.immutable_source_k && (cold || !same_position)) {
            const uint64_t delta = static_cast<uint64_t>(
                b.baked_pos > static_cast<int64_t>(window_pos)
                    ? b.baked_pos - static_cast<int64_t>(window_pos)
                    : static_cast<int64_t>(window_pos) - b.baked_pos);
            const bool remap_limit =
                cfg_.immutable_refresh_remaps > 0 &&
                b.remap_count >= cfg_.immutable_refresh_remaps;
            const bool delta_limit =
                cfg_.immutable_refresh_abs_delta_tokens > 0 &&
                (b.remap_abs_delta >=
                     cfg_.immutable_refresh_abs_delta_tokens ||
                 delta >= cfg_.immutable_refresh_abs_delta_tokens -
                              std::min<uint64_t>(
                                  b.remap_abs_delta,
                                  cfg_.immutable_refresh_abs_delta_tokens));
            const bool baked_out_of_range =
                cfg_.immutable_max_baked_position > 0 &&
                (b.baked_pos < 0 ||
                 static_cast<uint64_t>(b.baked_pos) + b.n_tokens >
                     cfg_.immutable_max_baked_position);
            rm.raw_refresh =
                cold || remap_limit || delta_limit || baked_out_of_range;
        }
        if (cfg_.immutable_source_k && force_raw_refresh) {
            rm.raw_refresh = true;
        }
        if (rm.skip && rm.raw_refresh) {
            throw std::logic_error(
                "KVMem remap cannot both skip K assembly and refresh raw K");
        }
        plan.remaps.push_back(rm);

        b.in_working_set = true;
        if (!rm.skip) {
            const uint64_t delta = static_cast<uint64_t>(
                b.baked_pos > static_cast<int64_t>(window_pos)
                    ? b.baked_pos - static_cast<int64_t>(window_pos)
                    : static_cast<int64_t>(window_pos) - b.baked_pos);
            if (rm.raw_refresh) {
                b.remap_count = 0;
                b.remap_abs_delta = 0;
            } else {
                ++b.remap_count;
                b.remap_abs_delta += delta;
            }
        }
        b.baked_pos = static_cast<int64_t>(window_pos);
        window_pos += b.n_tokens;
    }
    plan.total_window_tokens = window_pos;
    return plan;
}

NativeResidencyPlan KvMemStore::plan_native_selection(std::vector<uint32_t> selected_ids) const {
    std::sort(selected_ids.begin(), selected_ids.end());
    for (auto id : selected_ids) {
        if (id >= block_count() || !blocks_[id].n_tokens)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "unknown/empty block in native selection");
    }
    std::vector<uint64_t> resident;
    for (const auto & b : blocks_) {
        if (b.in_flight || (b.gpu_slot >= 0 && b.tier != KvTier::GPU))
            throw MemoryError(MemoryErrorCode::Pending, "native residency is not stable");
        if (b.gpu_slot >= 0 && b.n_tokens) resident.push_back(b.block_id);
    }
    const auto delta = plan_residency(resident,
        std::vector<uint64_t>(selected_ids.begin(), selected_ids.end()), !cfg_.optimize_stage_in);
    NativeResidencyPlan plan;
    plan.selected = std::move(selected_ids);
    plan.stage_out.assign(delta.evict.begin(), delta.evict.end());
    plan.stage_in.assign(delta.restore.begin(), delta.restore.end());
    plan.gpu_reused_blocks = static_cast<uint32_t>(delta.retain.size());
    for (auto id : plan.selected) {
        plan.total_window_tokens += blocks_[id].n_tokens;
        plan.selection_overlap_blocks += blocks_[id].in_working_set;
    }
    return plan;
}

bool KvMemStore::commit_resident_selection(const std::vector<uint32_t> & selected_ids) {
    std::vector<uint32_t> resident;
    for (const auto & b : blocks_) {
        if (b.gpu_slot >= 0 && b.n_tokens) {
            if (b.tier != KvTier::GPU || b.in_flight) return false;
            resident.push_back(b.block_id);
        }
    }
    if (resident != selected_ids) return false;
    for (auto & b : blocks_) b.in_working_set = b.gpu_slot >= 0 && b.n_tokens > 0;
    return true;
}

void KvMemStore::clear_working_set() {
    for (auto &b : blocks_) {
        b.in_working_set = false;
    }
}

} // namespace kvmem
