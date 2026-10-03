// Host tier persistence of the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md
// §5.5). The file holds exactly what a restarted Engine can resume from: every Host-backed
// snapshot and the Host-resident block path it anchors on, with the slab bytes. Nothing in it is
// trusted unless the fingerprint (model artifact, KV and state formats, binary) and the Host
// geometry match; a damaged or foreign file loads nothing.

#include "models/qwen3_5/program/prefix/hybrid_cache.h"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::array<char, 8> kMagic  = {'N', 'I', 'N', 'F', 'H', 'P', 'C', '1'};
constexpr std::array<char, 8> kFooter = {'N', 'I', 'N', 'F', 'E', 'N', 'D', '1'};
constexpr std::uint32_t kVersion      = 1;
constexpr std::int32_t kRootIndex     = -1;

struct Geometry {
    std::uint64_t slab_bytes          = 0;
    std::uint64_t image_bytes         = 0;
    std::uint64_t block_payload_bytes = 0;
    std::uint64_t text_page_stride    = 0;
    std::uint64_t backend_page_stride = 0;
    std::uint64_t backend_offset      = 0;
    std::uint32_t image_slabs         = 0;

    [[nodiscard]] friend bool operator==(const Geometry&, const Geometry&) = default;
};

Geometry geometry_of(const HybridHostLayout& layout) {
    return Geometry{
        .slab_bytes          = layout.slab_bytes,
        .image_bytes         = layout.image_bytes,
        .block_payload_bytes = layout.block_payload_bytes,
        .text_page_stride    = layout.text.page_stride,
        .backend_page_stride = layout.backend ? layout.backend->page_stride : 0U,
        .backend_offset      = layout.backend_offset,
        .image_slabs         = layout.image_slabs,
    };
}

class Writer {
public:
    explicit Writer(const std::filesystem::path& path)
        : out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) { throw std::runtime_error("cannot create " + path.string()); }
    }

    template <class T>
    void value(const T& value) {
        bytes(&value, sizeof(T));
    }

    void bytes(const void* data, std::size_t count) {
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(count));
        if (!out_) { throw std::runtime_error("prefix cache file write failed"); }
        written_ += count;
    }

    void finish() {
        out_.flush();
        if (!out_) { throw std::runtime_error("prefix cache file flush failed"); }
        out_.close();
    }

    [[nodiscard]] std::uint64_t written() const noexcept { return written_; }

private:
    std::ofstream out_;
    std::uint64_t written_ = 0;
};

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : in_(path, std::ios::binary) {}

    [[nodiscard]] bool open() const noexcept { return static_cast<bool>(in_); }

    template <class T>
    [[nodiscard]] T value() {
        T out{};
        bytes(&out, sizeof(T));
        return out;
    }

    void bytes(void* data, std::size_t count) {
        in_.read(static_cast<char*>(data), static_cast<std::streamsize>(count));
        if (!in_) { throw std::runtime_error("prefix cache file is truncated"); }
        read_ += count;
    }

    void skip(std::size_t count) {
        in_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
        if (!in_) { throw std::runtime_error("prefix cache file is truncated"); }
        read_ += count;
    }

    [[nodiscard]] std::uint64_t read() const noexcept { return read_; }

private:
    std::ifstream in_;
    std::uint64_t read_ = 0;
};

double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

} // namespace

HybridPersistResult HybridPrefixCache::save(const std::filesystem::path& path,
                                            std::string_view fingerprint) const {
    HybridPersistResult out;
    const auto started = std::chrono::steady_clock::now();
    if (!host_tier()) {
        out.message = "the hybrid prefix cache has no Host tier to save";
        return out;
    }
    if (!pending_.empty() || restore_.open || !landing_.empty()) {
        throw std::logic_error("hybrid prefix cache save requires idle transfers");
    }
    std::vector<pc::NodeRef> nodes;
    std::vector<pc::SnapshotRef> snapshots;
    index_->collect_persistable(nodes, snapshots);

    std::filesystem::path temporary = path;
    temporary += ".tmp";
    try {
        Writer writer(temporary);
        writer.bytes(kMagic.data(), kMagic.size());
        writer.value(kVersion);
        writer.value(static_cast<std::uint32_t>(fingerprint.size()));
        writer.bytes(fingerprint.data(), fingerprint.size());
        const Geometry geometry = geometry_of(host_layout_);
        writer.value(geometry);
        writer.value(static_cast<std::uint32_t>(nodes.size()));
        writer.value(static_cast<std::uint32_t>(snapshots.size()));

        std::unordered_map<std::uint32_t, std::int32_t> file_index;
        file_index.reserve(nodes.size());
        for (const pc::NodeRef node : nodes) {
            const pc::BlockIdentity identity = index_->block_identity(node);
            const pc::NodeView view          = index_->node(node);
            writer.value(identity.parent.valid() ? file_index.at(identity.parent.index)
                                                 : kRootIndex);
            writer.value(identity.lookup_hash);
            writer.value(identity.extra);
            writer.bytes(identity.tokens.data(), identity.tokens.size_bytes());
            writer.bytes(slab(view.host_slab), host_layout_.slab_bytes);
            file_index.emplace(node.index, static_cast<std::int32_t>(file_index.size()));
        }
        for (const pc::SnapshotRef snapshot : snapshots) {
            const pc::SnapshotView view = index_->snapshot(snapshot);
            writer.value(view.anchor.valid() ? file_index.at(view.anchor.index) : kRootIndex);
            writer.value(view.frontier);
            writer.value(view.tail_len);
            writer.bytes(view.tail.data(), view.tail.size_bytes());
            writer.value(static_cast<std::uint8_t>(view.kind));
            writer.value(view.hits);
            for (const std::uint32_t slab_id : view.host_slabs) {
                writer.bytes(slab(slab_id), host_layout_.slab_bytes);
            }
        }
        writer.bytes(kFooter.data(), kFooter.size());
        writer.value(static_cast<std::uint32_t>(nodes.size()));
        writer.value(static_cast<std::uint32_t>(snapshots.size()));
        writer.finish();
        out.bytes = writer.written();
        std::filesystem::rename(temporary, path);
    } catch (const std::exception& error) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        out.message = error.what();
        return out;
    }
    out.ok        = true;
    out.blocks    = nodes.size();
    out.snapshots = snapshots.size();
    out.seconds   = seconds_since(started);
    return out;
}

HybridPersistResult HybridPrefixCache::load(const std::filesystem::path& path,
                                            std::string_view fingerprint) {
    HybridPersistResult out;
    const auto started = std::chrono::steady_clock::now();
    if (!host_tier()) {
        out.message = "the hybrid prefix cache has no Host tier to load into";
        return out;
    }
    if (index_->stats().nodes != 0 || index_->stats().snapshots != 0) {
        throw std::logic_error("hybrid prefix cache load requires an empty cache");
    }
    Reader reader(path);
    if (!reader.open()) {
        out.message =
            "no saved prefix cache at " + path.string() + " yet; it is written at shutdown";
        return out;
    }
    try {
        std::array<char, 8> magic{};
        reader.bytes(magic.data(), magic.size());
        if (magic != kMagic || reader.value<std::uint32_t>() != kVersion) {
            out.message = "not a prefix cache file of this format version";
            return out;
        }
        const auto fingerprint_bytes = reader.value<std::uint32_t>();
        if (fingerprint_bytes > (1U << 20U)) {
            out.message = "prefix cache file header is damaged";
            return out;
        }
        std::string saved(fingerprint_bytes, '\0');
        reader.bytes(saved.data(), saved.size());
        if (saved != fingerprint) {
            out.message = "saved prefix cache belongs to a different model, KV format or build";
            return out;
        }
        if (reader.value<Geometry>() != geometry_of(host_layout_)) {
            out.message = "saved prefix cache has a different Host geometry";
            return out;
        }
        const auto node_count     = reader.value<std::uint32_t>();
        const auto snapshot_count = reader.value<std::uint32_t>();

        // Restored index ids of the file's blocks; absent when a block (or its parent) did not
        // fit, so its descendants and the snapshots on it are skipped too.
        std::vector<std::optional<pc::NodeRef>> restored(node_count);
        std::array<TokenId, pc::kBlockTokens> tokens{};
        for (std::uint32_t index = 0; index < node_count; ++index) {
            const auto parent = reader.value<std::int32_t>();
            const auto hash   = reader.value<std::uint64_t>();
            const auto extra  = reader.value<std::uint64_t>();
            reader.bytes(tokens.data(), sizeof(tokens));
            if (parent >= static_cast<std::int32_t>(index) || parent < kRootIndex) {
                throw std::runtime_error("prefix cache file block order is damaged");
            }
            std::optional<pc::RestoredBlock> block;
            if (parent == kRootIndex || restored[static_cast<std::size_t>(parent)]) {
                const pc::NodeRef parent_ref = parent == kRootIndex
                                                   ? pc::NodeRef{}
                                                   : *restored[static_cast<std::size_t>(parent)];
                block = index_->restore_host_block(parent_ref, hash, tokens, extra);
            }
            if (!block) {
                reader.skip(host_layout_.slab_bytes);
                continue;
            }
            reader.bytes(slab(block->slab), host_layout_.slab_bytes);
            restored[index] = block->node;
            ++out.blocks;
        }
        for (std::uint32_t index = 0; index < snapshot_count; ++index) {
            const auto anchor   = reader.value<std::int32_t>();
            const auto frontier = reader.value<std::uint32_t>();
            const auto tail_len = reader.value<std::uint32_t>();
            if (tail_len >= pc::kBlockTokens || anchor >= static_cast<std::int32_t>(node_count) ||
                anchor < kRootIndex) {
                throw std::runtime_error("prefix cache file snapshot is damaged");
            }
            reader.bytes(tokens.data(), static_cast<std::size_t>(tail_len) * sizeof(TokenId));
            const auto kind           = static_cast<pc::SnapshotKind>(reader.value<std::uint8_t>());
            const auto hits           = reader.value<std::uint32_t>();
            const std::uint32_t slabs = host_layout_.image_slabs + (tail_len != 0 ? 1U : 0U);
            std::optional<pc::SnapshotRef> snapshot;
            if (anchor == kRootIndex || restored[static_cast<std::size_t>(anchor)]) {
                const pc::NodeRef anchor_ref = anchor == kRootIndex
                                                   ? pc::NodeRef{}
                                                   : *restored[static_cast<std::size_t>(anchor)];
                snapshot                     = index_->restore_host_snapshot(
                    anchor_ref, frontier, std::span<const TokenId>(tokens.data(), tail_len), kind,
                    hits);
            }
            if (!snapshot) {
                reader.skip(static_cast<std::size_t>(slabs) * host_layout_.slab_bytes);
                continue;
            }
            for (const std::uint32_t slab_id : index_->snapshot(*snapshot).host_slabs) {
                reader.bytes(slab(slab_id), host_layout_.slab_bytes);
            }
            ++out.snapshots;
        }
        std::array<char, 8> footer{};
        reader.bytes(footer.data(), footer.size());
        if (footer != kFooter || reader.value<std::uint32_t>() != node_count ||
            reader.value<std::uint32_t>() != snapshot_count) {
            throw std::runtime_error("prefix cache file footer is damaged");
        }
    } catch (const std::exception& error) {
        // Partially restored entries may hold unread slab bytes: drop them all.
        clear();
        out         = HybridPersistResult{};
        out.message = error.what();
        return out;
    }
    out.ok      = true;
    out.bytes   = reader.read();
    out.seconds = seconds_since(started);
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
