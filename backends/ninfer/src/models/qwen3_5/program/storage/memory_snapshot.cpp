#include "models/qwen3_5/program/program_impl.h"
#include "kvmem/snapshot.hpp"
#include <bit>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#endif

namespace ninfer::models::qwen3_5::detail {
namespace {
namespace fs = std::filesystem;
constexpr std::uint64_t magic = 0x34454d454d4b494eULL; // NIKMEME4; schema is deliberately closed.
constexpr std::uint64_t header_bytes = 24;

fs::path directory(const ProgramImpl& program) {
    return program.kvmem_options.disk_path / "ninfer-kvmem-v4";
}
fs::path snapshot_path(const ProgramImpl& program, const PreparedSessionKey& key) {
    kvmem::SnapshotWriter digest([](const void*, std::size_t) {});
    digest.write(key.bytes.data(), key.size);
    std::ostringstream name;
    name << std::hex << std::setfill('0');
    for (const auto byte : program.parameters.model.info().artifact_id)
        name << std::setw(2) << std::to_integer<unsigned>(byte);
    name << '-' << std::setw(16) << digest.hash() << ".nkm";
    return directory(program) / name.str();
}
bool owned_file(const fs::directory_entry& entry, bool temporary = false) {
    if (!entry.is_regular_file() || entry.is_symlink()) return false;
    auto name = entry.path().filename().string();
    if (temporary) {
        if (!name.ends_with(".tmp")) return false;
        name.resize(name.size() - 4);
    }
    if (name.size() != 53 || name[32] != '-' || name.substr(49) != ".nkm") return false;
    for (std::size_t i = 0; i < 49; ++i)
        if (i != 32 && !((name[i] >= '0' && name[i] <= '9') || (name[i] >= 'a' && name[i] <= 'f'))) return false;
    return true;
}

// Exact compatibility bytes, not a hash of pointers or allocation handles.
std::vector<std::uint8_t> compatibility(const ProgramImpl& p) {
    std::vector<std::uint8_t> bytes;
    kvmem::SnapshotWriter out([&](const void* data, std::size_t count) {
        const auto* first = static_cast<const std::uint8_t*>(data);
        bytes.insert(bytes.end(), first, first + count);
    });
    out.scalar(std::uint32_t(4));
    out.write(p.parameters.model.info().artifact_id.data(), 16);
    out.scalar(std::uint32_t(std::endian::native == std::endian::little));
    out.scalar(p.capacity); out.scalar(p.prefill_chunk);
    out.scalar(p.kvmem_options.selected_tokens); out.scalar(p.kvmem_options.reserve_tokens);
    out.scalar(p.capture_identity_tag()); out.scalar(p.draft_window);
    out.scalar(p.ngram_draft_window); out.scalar(p.ngram_min_match);
    out.scalar(std::uint8_t(p.use_cuda_graph));
    out.scalar(p.device.props.major); out.scalar(p.device.props.minor);
    out.scalar(p.device.props.multiProcessorCount);
    out.scalar(p.rope_yarn.factor); out.scalar(p.rope_yarn.native_context);
    out.scalar(p.rope_yarn.interpolation_factor); out.scalar(p.rope_yarn.interpolation_threshold);
    const auto& load = p.parameters.model.options();
    for (bool value : {load.lm_head_q4, load.lm_head_q6, load.embedding_q4, load.embedding_q6,
                      load.mtp_experts_q4, load.mlp_a8_decode, load.prefill_a8,
                      load.prefill_cublas, load.prefill_cublas_projections, load.gdn_state_fp16})
        out.scalar(std::uint8_t(value));
    const auto& state = p.state_images->host_layout();
    out.scalar(std::uint64_t(state.image_bytes));
    out.scalar(std::uint32_t(state.spec.linear.conv_dtype));
    out.scalar(std::uint32_t(state.spec.linear.recurrent_dtype));
    const auto planes = [&](const auto& pool) {
        const auto layout = plan_host_kv_page_layout(pool.geometry());
        out.scalar(std::uint64_t(layout.page_stride));
        out.scalar(std::uint64_t(layout.planes.size()));
        for (const auto& plane : layout.planes) {
            out.scalar(std::uint64_t(plane.offset)); out.scalar(std::uint64_t(plane.page_payload_bytes));
        }
    };
    planes(p.text_kv_pages->physical_pool());
    if (p.backend_kv_pages) planes(p.backend_kv_pages->physical_pool());
    return bytes;
}
void publish(const fs::path& temporary, const fs::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("snapshot atomic replace failed");
#else
    fs::rename(temporary, destination);
#endif
}
} // namespace

// This codec deliberately supports text-only prefix metadata. Vision remains
// in the in-process identity path until its own persistent schema is qualified.
struct KvmemSnapshotCodec {
    static void write(kvmem::SnapshotWriter& out, const ResidentPrefixIdentity& identity) {
        if (!identity.vision_items_.empty()) throw std::logic_error("text snapshot contains media");
        out.vector(identity.token_types_);
        for (const auto& axis : identity.positions_) out.vector(axis);
        out.vector(identity.rewrite_execution_frontiers_);
    }
    static ResidentPrefixIdentity read(kvmem::SnapshotReader& in, std::uint32_t frontier) {
        ResidentPrefixIdentity identity;
        identity.token_types_ = in.vector<std::uint8_t>(frontier);
        if (identity.token_types_.size() != frontier) throw kvmem::SnapshotCorrupt("invalid text identity size");
        for (auto& axis : identity.positions_) {
            axis = in.vector<std::int32_t>(frontier);
            if (axis.size() != frontier) throw kvmem::SnapshotCorrupt("invalid position axis size");
        }
        identity.rewrite_execution_frontiers_ = in.vector<std::uint32_t>(frontier);
        return identity;
    }
};

void ProgramImpl::save_memory_snapshot(KvmemWindowState& window) noexcept {
    if (kvmem_options.disk_path.empty() || !window.session_key ||
        (!window.prefix_checkpoint && !window.endpoint_checkpoint && !window.replay_checkpoint)) return;
    fs::path temporary;
    try {
        const auto destination = snapshot_path(*this, *window.session_key);
        temporary = destination; temporary += ".tmp";
        fs::create_directories(directory(*this));
        const KvmemPrefixCheckpoint* latest = nullptr;
        for (const auto kind : {KvmemCheckpointKind::Base, KvmemCheckpointKind::Endpoint,
                                KvmemCheckpointKind::Rewrite})
            if (const auto* checkpoint = memory_checkpoint(window, kind);
                checkpoint && (!latest || checkpoint->prefix.size() > latest->prefix.size())) latest = checkpoint;
        if (!latest) return;
        const auto frontier = static_cast<std::uint32_t>(latest->prefix.size());
        window.statistics->truncate_to(frontier);
        window.statistics->restore_mean_checkpoint(frontier, latest->mean_tail);
        window.selector->truncate_to(frontier);
        window.statistics_frontier = frontier;
        const auto identity = compatibility(*this);
        const auto body = [&](kvmem::SnapshotWriter& out) {
            out.vector(identity);
            out.vector(std::vector<char>(window.session_key->view().begin(), window.session_key->view().end()));
            out.scalar(frontier);
            out.scalar(window.query_begin); out.scalar(window.query_end); out.scalar(window.query_tokens);
            out.vector(window.query_sum);
            for (const auto kind : {KvmemCheckpointKind::Base, KvmemCheckpointKind::Endpoint,
                                    KvmemCheckpointKind::Rewrite}) {
                const auto* checkpoint = memory_checkpoint(window, kind);
                out.scalar(std::uint8_t(checkpoint != nullptr));
                if (!checkpoint) continue;
                out.vector(checkpoint->prefix);
                KvmemSnapshotCodec::write(out, checkpoint->prefix_identity);
                out.vector(checkpoint->resident_pages); out.vector(checkpoint->mean_tail);
                out.write(checkpoint->state->data(), checkpoint->state->size());
            }
            window.statistics->snapshot_write(out);
            for (const auto& page : window.archive) {
                if (!page) throw std::logic_error("snapshot prefix has a missing page");
                out.write(page->data(), page->size());
            }
        };
        kvmem::SnapshotWriter size;
        body(size);
        const auto required = size.bytes() + header_bytes;
        if (required > kvmem_options.disk_bytes) throw std::runtime_error("snapshot exceeds disk quota");
        // Charge both the old record and its atomic replacement until publication.
        // Only files of this codec in this dedicated subdirectory are evictable.
        std::vector<fs::directory_entry> victims;
        std::uint64_t used = 0;
        for (const auto& entry : fs::directory_iterator(directory(*this))) {
            // A previous process may have stopped before atomic publication.
            // This cache is explicitly single-owner; incomplete records are not sources.
            if (owned_file(entry, true)) { fs::remove(entry.path()); continue; }
            if (!owned_file(entry)) continue;
            used += entry.file_size();
            if (entry.path() != destination) victims.push_back(entry);
        }
        std::sort(victims.begin(), victims.end(), [](const auto& a, const auto& b) {
            return a.last_write_time() < b.last_write_time();
        });
        for (const auto& victim : victims) {
            if (used <= kvmem_options.disk_bytes - required) break;
            const auto bytes = victim.file_size();
            if (fs::remove(victim.path())) used -= bytes;
        }
        if (used > kvmem_options.disk_bytes - required) throw std::runtime_error("snapshot atomic replacement exceeds disk quota");
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        const std::uint64_t header[]{magic, size.bytes(), 0};
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        kvmem::SnapshotWriter writer([&](const void* data, std::size_t bytes) {
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        });
        body(writer);
        const auto hash = writer.hash();
        file.seekp(16); file.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
        file.flush(); file.close();
        publish(temporary, destination);
        ++memory_disk_writes;
    } catch (const std::exception& error) {
        ++memory_disk_errors;
        std::error_code ignored;
        if (!temporary.empty()) fs::remove(temporary, ignored);
        std::fprintf(stderr, "KVMEM_DISK write skipped: %s\n", error.what());
    }
}

void ProgramImpl::load_memory_snapshot(const PreparedPromptData& prompt) noexcept {
    if (kvmem_options.disk_path.empty() || !prompt.context_cache.session_key ||
        !prompt.context_cache.update_session_index || !prompt.identity.reusable || prompt.has_media()) return;
    try {
        const auto path = snapshot_path(*this, *prompt.context_cache.session_key);
        if (!fs::exists(path)) return;
        const auto length = fs::file_size(path);
        if (length < header_bytes || length > kvmem_options.disk_bytes)
            throw kvmem::SnapshotCorrupt("snapshot envelope exceeds quota");
        std::ifstream file(path, std::ios::binary);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::uint64_t header[3];
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        if (header[0] != magic || header[1] != length - header_bytes)
            throw kvmem::SnapshotCorrupt("snapshot envelope mismatch");
        kvmem::SnapshotReader in([&](void* data, std::size_t bytes) {
            file.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
        }, header[1]);
        if (in.vector<std::uint8_t>(4096) != compatibility(*this))
            throw kvmem::SnapshotCorrupt("model, execution profile or native layout mismatch");
        const auto key = in.vector<char>(kPreparedSessionKeyCapacity);
        if (std::string_view(key.data(), key.size()) != prompt.context_cache.session_key->view())
            throw kvmem::SnapshotCorrupt("snapshot session identity mismatch");
        SequenceState imported;
        auto& window = imported.window;
        const auto frontier = in.scalar<std::uint32_t>();
        if (!frontier || frontier > capacity) throw kvmem::SnapshotCorrupt("invalid history frontier");
        const auto pages = kv_pages_for_tokens(frontier);
        const auto& config = parameters.model.config().text;
        const auto width = static_cast<std::uint32_t>(memory_statistics->key_sums.ne[0]);
        const auto state_bytes = state_images->host_layout().image_bytes;
        const auto page_bytes = plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry()).page_stride +
            (backend_kv_pages ? plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry()).page_stride : 0);
        if (pages > kvmem_options.host_bytes / page_bytes) throw kvmem::SnapshotCorrupt("snapshot exceeds Host quota");
        const auto required = pages * page_bytes;
        window.query_begin = in.scalar<std::uint32_t>();
        window.query_end = in.scalar<std::uint32_t>();
        window.query_tokens = in.scalar<std::uint32_t>();
        window.query_sum = in.vector<float>(memory_statistics->query_sums.numel());
        if (window.query_begin > window.query_end || window.query_end > capacity ||
            window.query_tokens > window.query_end - window.query_begin ||
            window.query_sum.size() != memory_statistics->query_sums.numel())
            throw kvmem::SnapshotCorrupt("invalid query attachment");
        const auto read_checkpoint = [&]() -> std::unique_ptr<KvmemPrefixCheckpoint> {
            const auto present = in.scalar<std::uint8_t>();
            if (present > 1) throw kvmem::SnapshotCorrupt("invalid checkpoint tag");
            if (!present) return {};
            auto checkpoint = std::make_unique<KvmemPrefixCheckpoint>();
            checkpoint->prefix = in.vector<TokenId>(frontier);
            const auto end = static_cast<std::uint32_t>(checkpoint->prefix.size());
            if (!end) throw kvmem::SnapshotCorrupt("empty checkpoint");
            checkpoint->prefix_identity = KvmemSnapshotCodec::read(in, end);
            checkpoint->resident_pages = in.vector<std::uint64_t>(kvmem_window_tokens / 64);
            const auto count = kv_pages_for_tokens(end);
            const auto& resident = checkpoint->resident_pages;
            if (resident.empty() || !std::is_sorted(resident.begin(), resident.end()) ||
                std::adjacent_find(resident.begin(), resident.end()) != resident.end() ||
                resident.back() >= count || (end % 64 && resident.back() != count - 1))
                throw kvmem::SnapshotCorrupt("snapshot compact view is invalid");
            const auto mean_values = end % 64
                ? (std::uint64_t(width) + 1) * config.full_attention_layers : 0;
            checkpoint->mean_tail = in.vector<float>(mean_values);
            if (checkpoint->mean_tail.size() != mean_values)
                throw kvmem::SnapshotCorrupt("invalid partial-page mean checkpoint");
            checkpoint->state = std::make_unique<PinnedHostBuffer>(state_bytes);
            in.read(checkpoint->state->data(), state_bytes);
            return checkpoint;
        };
        window.prefix_checkpoint = read_checkpoint();
        window.endpoint_checkpoint = read_checkpoint();
        window.replay_checkpoint = read_checkpoint();
        const KvmemPrefixCheckpoint* latest = nullptr;
        bool matching = false;
        for (const auto kind : {KvmemCheckpointKind::Base, KvmemCheckpointKind::Endpoint,
                                KvmemCheckpointKind::Rewrite}) {
            const auto* checkpoint = memory_checkpoint(window, kind);
            if (!checkpoint) continue;
            const auto end = static_cast<std::uint32_t>(checkpoint->prefix.size());
            if (!latest || end > latest->prefix.size()) latest = checkpoint;
            if (end < prompt.token_ids.size() &&
                std::equal(checkpoint->prefix.begin(), checkpoint->prefix.end(), prompt.token_ids.begin()) &&
                checkpoint->prefix_identity.matches(prompt, end) &&
                (prompt.token_ids.size() <= kvmem_window_tokens || !prompt.memory_query ||
                 end <= prompt.memory_query->begin || memory_same_query(window, prompt, end))) matching = true;
        }
        if (!latest || latest->prefix.size() != frontier)
            throw kvmem::SnapshotCorrupt("archive frontier has no complete checkpoint");
        // Free inactive records before importing payload; the hard H bound also
        // holds during import, including when later checksum verification fails.
        for (;;) {
            std::uint64_t used = 0;
            std::optional<std::size_t> oldest;
            for (std::size_t i = 0; i < memory_histories.size(); ++i) if (memory_histories[i]) {
                used += memory_histories[i]->host_bytes;
                if (!oldest || memory_histories[i]->publication_order < memory_histories[*oldest]->publication_order) oldest = i;
            }
            if (used <= kvmem_options.host_bytes - required) break;
            memory_histories[*oldest].reset(); ++memory_history_evictions;
        }
        window.statistics = std::make_unique<kvmem::RawKvStore>(kvmem::RawKvStoreConfig{
            .n_layer = config.full_attention_layers, .n_embd_k = width, .block_tokens = 64});
        window.statistics->snapshot_read(in, pages);
        window.statistics->restore_mean_checkpoint(frontier, latest->mean_tail);
        window.archive.resize(pages); window.host_versions.resize(pages);
        for (std::uint32_t i = 0; i < pages; ++i) {
            window.archive[i] = std::make_unique<KvmemHostRecord>(page_bytes);
            in.read(window.archive[i]->data(), page_bytes);
            window.host_versions[i] = std::min(64U, frontier - i * 64);
        }
        if (in.remaining() || in.hash() != header[2]) throw kvmem::SnapshotCorrupt("snapshot checksum mismatch");
        if (!matching) return;
        kvmem::KvMemStoreConfig selection;
        selection.block_tokens = 64; selection.select_budget = kvmem_options.selected_tokens;
        selection.sink_blocks = 1; selection.recent_blocks = 1;
        window.selector = std::make_unique<kvmem::KvMemStore>(selection);
        window.selector->register_append(frontier);
        window.statistics_frontier = frontier; window.host_bytes = required;
        window.session_key = prompt.context_cache.session_key;
        imported.text_kv_valid = frontier;
        if (backend_kv_pages) imported.mtp_kv_valid = frontier;
        bind_memory_snapshot(imported); // Fresh process-local version and session ID.
        std::size_t slot = 0, count = 0;
        for (std::size_t i = 0; i < memory_histories.size(); ++i) {
            if (memory_histories[i]) ++count; else slot = i;
        }
        if (count >= kvmem_options.retained_sessions) {
            slot = 0;
            for (std::size_t i = 0; i < memory_histories.size(); ++i)
                if (memory_histories[i] && (!memory_histories[slot] ||
                    memory_histories[i]->publication_order < memory_histories[slot]->publication_order)) slot = i;
            ++memory_history_evictions;
        }
        memory_histories[slot].emplace(std::move(window));
        std::error_code ignored;
        fs::last_write_time(path, fs::file_time_type::clock::now(), ignored);
        ++memory_disk_hits;
    } catch (const std::exception& error) {
        ++memory_disk_errors;
        std::fprintf(stderr, "KVMEM_DISK restore skipped; recomputing: %s\n", error.what());
    }
}

void ProgramImpl::erase_memory_snapshot(const PreparedSessionKey& key) noexcept {
    if (kvmem_options.disk_path.empty()) return;
    try { fs::remove(snapshot_path(*this, key)); }
    catch (const std::exception&) { ++memory_disk_errors; }
}
} // namespace ninfer::models::qwen3_5::detail
