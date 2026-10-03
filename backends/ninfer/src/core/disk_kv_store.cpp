#include "core/disk_kv_store.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace ninfer {
namespace {

constexpr std::size_t kAlignment   = 4096;
constexpr std::size_t kTrailerSize = 128;

constexpr std::size_t round_up(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1) / alignment * alignment;
}

std::string index_path(const std::string& data_path) { return data_path + ".idx"; }

struct CrcTables {
    std::array<std::array<std::uint32_t, 256>, 8> t{};

    CrcTables() noexcept {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) { c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U; }
            t[0][i] = c;
        }
        for (std::size_t s = 1; s < t.size(); ++s) {
            for (std::uint32_t i = 0; i < 256; ++i) {
                t[s][i] = (t[s - 1][i] >> 8U) ^ t[0][t[s - 1][i] & 0xFFU];
            }
        }
    }
};

std::uint32_t load_le32(const std::byte* bytes) noexcept {
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

// Read before any cleanup call can overwrite the last OS error.
std::runtime_error os_error(const std::string& path, const char* operation) {
#ifdef _WIN32
    return std::runtime_error(path + ": " + operation + ": Win32 error " +
                              std::to_string(static_cast<unsigned long>(::GetLastError())));
#else
    return std::runtime_error(path + ": " + operation + ": " + std::strerror(errno));
#endif
}

} // namespace

std::uint32_t DiskKVStore::crc32(std::span<const std::byte> data) noexcept {
    static const CrcTables tables;
    const auto& t      = tables.t;
    const std::byte* p = data.data();
    std::size_t n      = data.size();
    std::uint32_t c    = 0xFFFFFFFFU;
    while (n >= 8) {
        c ^= load_le32(p);
        const std::uint32_t next = load_le32(p + 4);
        c = t[7][c & 0xFFU] ^ t[6][(c >> 8U) & 0xFFU] ^ t[5][(c >> 16U) & 0xFFU] ^ t[4][c >> 24U] ^
            t[3][next & 0xFFU] ^ t[2][(next >> 8U) & 0xFFU] ^ t[1][(next >> 16U) & 0xFFU] ^
            t[0][next >> 24U];
        p += 8;
        n -= 8;
    }
    for (; n > 0; --n, ++p) { c = t[0][(c ^ static_cast<std::uint32_t>(*p)) & 0xFFU] ^ (c >> 8U); }
    return c ^ 0xFFFFFFFFU;
}

std::size_t DiskKVStore::slot_pitch() const noexcept {
    return round_up(kSlotHeaderSize + options_.slot_size, kAlignment);
}

std::size_t DiskKVStore::payload_offset(std::uint32_t slot) const noexcept {
    return static_cast<std::size_t>(slot) * slot_pitch() + kSlotHeaderSize;
}

DiskKVStore::Header& DiskKVStore::header(std::uint32_t slot) const noexcept {
    return *reinterpret_cast<Header*>(base_ + static_cast<std::size_t>(slot) * slot_pitch());
}

bool DiskKVStore::matches(const Header& header, const DiskKVIdentity& id) noexcept {
    return header.magic == Header::kMagic && header.lo == id.lo && header.hi == id.hi &&
           header.tag == id.tag && header.frontier == id.frontier;
}

DiskKVStore::DiskKVStore(Options options) : options_(std::move(options)) {
    if (options_.path.empty()) { throw std::invalid_argument("disk KV store path is empty"); }
    if (options_.slot_size == 0) { throw std::invalid_argument("disk KV store slot size is zero"); }
    if (options_.max_slots == 0) {
        const std::size_t slots = options_.capacity_bytes / slot_pitch();
        if (slots == 0) { throw std::invalid_argument("disk KV store capacity holds no slot"); }
        options_.max_slots = static_cast<std::uint32_t>(
            std::min<std::size_t>(slots, std::numeric_limits<std::uint32_t>::max()));
    }
    max_slots_ = options_.max_slots;
    file_bytes_ =
        round_up(static_cast<std::size_t>(max_slots_) * slot_pitch(), kAlignment) + kTrailerSize;

    std::error_code error;
    const auto existing = std::filesystem::file_size(options_.path, error);
    const bool reuse    = !error && existing > 0;
    if (reuse && existing < file_bytes_) {
        throw std::runtime_error(options_.path +
                                 ": disk KV store is smaller than its configured geometry");
    }
    map_file(!reuse);
    readers_.assign(max_slots_, 0);
    lru_stamps_.assign(max_slots_, kNoStamp);
    if (!load_index()) { rebuild_from_scan(); }
}

DiskKVStore::~DiskKVStore() { unmap_file(); }

#ifdef _WIN32

void DiskKVStore::map_file(bool create) {
    const std::filesystem::path path(options_.path);
    const HANDLE file = ::CreateFileW(
        path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
        create ? OPEN_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { throw os_error(options_.path, "CreateFileW"); }
    file_handle_ = reinterpret_cast<std::intptr_t>(file);
    if (create) {
        LARGE_INTEGER size{};
        size.QuadPart = static_cast<LONGLONG>(file_bytes_);
        if (!::SetFilePointerEx(file, size, nullptr, FILE_BEGIN) || !::SetEndOfFile(file)) {
            const std::runtime_error error = os_error(options_.path, "SetEndOfFile");
            unmap_file();
            throw error;
        }
    }
    const HANDLE mapping = ::CreateFileMappingW(
        file, nullptr, PAGE_READWRITE, static_cast<DWORD>(std::uint64_t{file_bytes_} >> 32U),
        static_cast<DWORD>(file_bytes_ & 0xFFFFFFFFULL), nullptr);
    if (mapping == nullptr) {
        const std::runtime_error error = os_error(options_.path, "CreateFileMappingW");
        unmap_file();
        throw error;
    }
    mapping_handle_ = reinterpret_cast<std::intptr_t>(mapping);
    void* view      = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, file_bytes_);
    if (view == nullptr) {
        const std::runtime_error error = os_error(options_.path, "MapViewOfFile");
        unmap_file();
        throw error;
    }
    base_ = static_cast<std::byte*>(view);
}

void DiskKVStore::unmap_file() noexcept {
    if (base_ != nullptr) { ::UnmapViewOfFile(base_); }
    if (mapping_handle_ != -1) { ::CloseHandle(reinterpret_cast<HANDLE>(mapping_handle_)); }
    if (file_handle_ != -1) { ::CloseHandle(reinterpret_cast<HANDLE>(file_handle_)); }
    base_           = nullptr;
    mapping_handle_ = -1;
    file_handle_    = -1;
}

bool DiskKVStore::write_payload(std::uint32_t slot, std::span<const std::byte> bytes) noexcept {
    const HANDLE file = reinterpret_cast<HANDLE>(file_handle_);
    std::size_t done  = 0;
    while (done < bytes.size()) {
        const std::uint64_t offset = payload_offset(slot) + done;
        OVERLAPPED at{};
        at.Offset         = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
        at.OffsetHigh     = static_cast<DWORD>(offset >> 32U);
        const auto length = static_cast<DWORD>(
            std::min<std::size_t>(bytes.size() - done, std::size_t{1} << 30U));
        DWORD written = 0;
        if (!::WriteFile(file, bytes.data() + done, length, &written, &at) || written == 0) {
            return false;
        }
        done += written;
    }
    return true;
}

bool DiskKVStore::read_payload(std::uint32_t slot, std::span<std::byte> destination) noexcept {
    const HANDLE file = reinterpret_cast<HANDLE>(file_handle_);
    std::size_t done  = 0;
    while (done < destination.size()) {
        const std::uint64_t offset = payload_offset(slot) + done;
        OVERLAPPED at{};
        at.Offset         = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
        at.OffsetHigh     = static_cast<DWORD>(offset >> 32U);
        const auto length = static_cast<DWORD>(
            std::min<std::size_t>(destination.size() - done, std::size_t{1} << 30U));
        DWORD read = 0;
        if (!::ReadFile(file, destination.data() + done, length, &read, &at) || read == 0) {
            return false;
        }
        done += read;
    }
    return true;
}

#else

void DiskKVStore::map_file(bool create) {
    const int fd = ::open(options_.path.c_str(), create ? O_RDWR | O_CREAT : O_RDWR, 0644);
    if (fd < 0) { throw os_error(options_.path, "open"); }
    file_handle_ = fd;
    // A fresh file is sparse: disk blocks are allocated as slots are written.
    if (create && ::ftruncate(fd, static_cast<off_t>(file_bytes_)) != 0) {
        const std::runtime_error error = os_error(options_.path, "ftruncate");
        unmap_file();
        throw error;
    }
    void* mapped = ::mmap(nullptr, file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        const std::runtime_error error = os_error(options_.path, "mmap");
        unmap_file();
        throw error;
    }
    base_ = static_cast<std::byte*>(mapped);
}

void DiskKVStore::unmap_file() noexcept {
    if (base_ != nullptr) { ::munmap(base_, file_bytes_); }
    if (file_handle_ >= 0) { ::close(static_cast<int>(file_handle_)); }
    base_        = nullptr;
    file_handle_ = -1;
}

bool DiskKVStore::write_payload(std::uint32_t slot, std::span<const std::byte> bytes) noexcept {
    const int fd     = static_cast<int>(file_handle_);
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t written =
            ::pwrite(fd, bytes.data() + done, bytes.size() - done,
                     static_cast<off_t>(payload_offset(slot) + done));
        if (written < 0 && errno == EINTR) { continue; }
        if (written <= 0) { return false; }
        done += static_cast<std::size_t>(written);
    }
    return true;
}

bool DiskKVStore::read_payload(std::uint32_t slot, std::span<std::byte> destination) noexcept {
    const int fd     = static_cast<int>(file_handle_);
    std::size_t done = 0;
    while (done < destination.size()) {
        const ssize_t read =
            ::pread(fd, destination.data() + done, destination.size() - done,
                    static_cast<off_t>(payload_offset(slot) + done));
        if (read < 0 && errno == EINTR) { continue; }
        if (read <= 0) { return false; }
        done += static_cast<std::size_t>(read);
    }
    return true;
}

#endif

void DiskKVStore::lru_touch_locked(std::uint32_t slot, std::uint64_t stamp) {
    lru_erase_locked(slot);
    lru_.emplace(stamp, slot);
    lru_stamps_[slot] = stamp;
}

void DiskKVStore::lru_erase_locked(std::uint32_t slot) noexcept {
    if (lru_stamps_[slot] == kNoStamp) { return; }
    lru_.erase({lru_stamps_[slot], slot});
    lru_stamps_[slot] = kNoStamp;
}

bool DiskKVStore::load_index() {
    std::FILE* file = std::fopen(index_path(options_.path).c_str(), "rb");
    if (file == nullptr) { return false; }
    IndexHeader index_header{};
    std::vector<IndexRow> rows;
    bool valid = std::fread(&index_header, sizeof(index_header), 1, file) == 1 &&
                 index_header.magic == IndexHeader::kMagic && index_header.version == 1 &&
                 index_header.count <= max_slots_ &&
                 index_header.slot_size == static_cast<std::uint32_t>(slot_pitch()) &&
                 index_header.max_slots == max_slots_;
    if (valid) {
        rows.resize(index_header.count);
        valid = rows.empty() ||
                std::fread(rows.data(), sizeof(IndexRow), rows.size(), file) == rows.size();
    }
    std::fclose(file);
    if (!valid) { return false; }

    index_.clear();
    index_.reserve(rows.size());
    lru_.clear();
    lru_stamps_.assign(max_slots_, kNoStamp);
    std::vector<bool> live(max_slots_, false);
    for (const IndexRow& row : rows) {
        if (row.slot >= max_slots_ || live[row.slot]) { continue; }
        const DiskKVIdentity id{row.lo, row.hi, row.tag, row.frontier};
        const Header& slot_header = header(row.slot);
        // A row whose slot was reused since the index was written is dropped; payload integrity
        // is checked lazily by the CRC on read.
        if (!matches(slot_header, id)) {
            index_dirty_ = true;
            continue;
        }
        index_[id]     = row.slot;
        live[row.slot] = true;
        clock_         = std::max(clock_, slot_header.last_used);
        lru_touch_locked(row.slot, slot_header.last_used);
    }
    free_slots_.clear();
    for (std::uint32_t slot = max_slots_; slot-- > 0;) {
        if (!live[slot]) { free_slots_.push_back(slot); }
    }
    return true;
}

void DiskKVStore::rebuild_from_scan() {
    index_.clear();
    free_slots_.clear();
    lru_.clear();
    lru_stamps_.assign(max_slots_, kNoStamp);
    clock_ = 0;
    std::vector<bool> live(max_slots_, false);
    std::vector<std::byte> payload(options_.slot_size);
    for (std::uint32_t slot = 0; slot < max_slots_; ++slot) {
        const Header& slot_header = header(slot);
        if (slot_header.magic != Header::kMagic) { continue; }
        if (!read_payload(slot, payload) || crc32(payload) != slot_header.crc) { continue; }
        index_[{slot_header.lo, slot_header.hi, slot_header.tag, slot_header.frontier}] = slot;
        live[slot]                                                                      = true;
        clock_ = std::max(clock_, slot_header.last_used);
        lru_touch_locked(slot, slot_header.last_used);
    }
    for (std::uint32_t slot = max_slots_; slot-- > 0;) {
        if (!live[slot]) { free_slots_.push_back(slot); }
    }
    rebuilt_from_scan_ = true;
    index_dirty_       = true;
    persist_index_locked();
}

void DiskKVStore::persist_index_locked() {
    if (!index_dirty_) { return; }
    const std::string path      = index_path(options_.path);
    const std::string temporary = path + ".tmp";
    std::FILE* file             = std::fopen(temporary.c_str(), "wb");
    if (file == nullptr) { return; }
    IndexHeader index_header{.magic     = IndexHeader::kMagic,
                             .version   = 1,
                             .count     = static_cast<std::uint32_t>(index_.size()),
                             .slot_size = static_cast<std::uint32_t>(slot_pitch()),
                             .max_slots = max_slots_,
                             .clock     = clock_};
    bool ok = std::fwrite(&index_header, sizeof(index_header), 1, file) == 1;
    for (const auto& [id, slot] : index_) {
        if (!ok) { break; }
        const IndexRow row{id.lo, id.hi, id.tag, id.frontier, slot, 0};
        ok = std::fwrite(&row, sizeof(row), 1, file) == 1;
    }
    ok = std::fflush(file) == 0 && ok;
    ok = std::fclose(file) == 0 && ok;
    std::error_code error;
    if (ok) { std::filesystem::rename(temporary, path, error); }
    if (!ok || error) {
        std::filesystem::remove(temporary, error);
        return;
    }
    index_dirty_ = false;
}

void DiskKVStore::clear_slot(std::uint32_t slot) noexcept {
    // A slot never written is a hole in the sparse file and reads as zeros: writing its header
    // through the mapping would allocate the block, which faults on a full disk.
    Header& slot_header = header(slot);
    if (slot_header.magic != 0 || slot_header.last_used != 0) { slot_header = Header{}; }
}

std::optional<DiskKVIdentity> DiskKVStore::evict_one_lru_locked() {
    for (const auto& entry : lru_) {
        // A copy: erasing the entry below frees the node the loop reads from.
        const std::uint32_t slot = entry.second;
        if (readers_[slot] != 0) { continue; }
        const Header& victim_header = header(slot);
        const DiskKVIdentity id{victim_header.lo, victim_header.hi, victim_header.tag,
                                victim_header.frontier};
        lru_erase_locked(slot);
        index_.erase(id);
        clear_slot(slot);
        free_slots_.push_back(slot);
        index_dirty_ = true;
        return id;
    }
    return std::nullopt;
}

std::uint32_t DiskKVStore::live_slots() const {
    std::lock_guard lock(mutex_);
    return static_cast<std::uint32_t>(index_.size());
}

std::size_t DiskKVStore::used_bytes() const {
    std::lock_guard lock(mutex_);
    return index_.size() * slot_pitch();
}

bool DiskKVStore::upsert_page(const DiskKVIdentity& id, std::span<const std::byte> bytes,
                              std::vector<DiskKVIdentity>* evicted) {
    if (bytes.size() != options_.slot_size) { return false; }
    std::uint32_t slot  = 0;
    std::uint64_t stamp = 0;
    {
        std::lock_guard lock(mutex_);
        if (const auto found = index_.find(id); found != index_.end()) {
            header(found->second).last_used = ++clock_;
            lru_touch_locked(found->second, clock_);
            index_dirty_ = true;
            return true;
        }
        if (free_slots_.empty()) {
            const std::optional<DiskKVIdentity> victim = evict_one_lru_locked();
            if (!victim) { return false; }
            if (evicted != nullptr) { evicted->push_back(*victim); }
        }
        slot = free_slots_.back();
        free_slots_.pop_back();
        stamp = ++clock_;
        clear_slot(slot);
    }

    // The slot is neither free nor indexed, so no reader or evictor can reach it while the payload
    // is written outside the lock. The header is stamped last, under the lock, so the page becomes
    // reachable only complete.
    const std::uint32_t crc = crc32(bytes.first(options_.slot_size));
    const bool written      = write_payload(slot, bytes.first(options_.slot_size));

    std::lock_guard lock(mutex_);
    if (!written) {
        free_slots_.push_back(slot);
        return false;
    }
    if (index_.contains(id)) {
        free_slots_.push_back(slot);
        return true;
    }
    header(slot) = Header{.magic     = Header::kMagic,
                          .last_used = stamp,
                          .lo        = id.lo,
                          .hi        = id.hi,
                          .tag       = id.tag,
                          .frontier  = id.frontier,
                          .crc       = crc};
    index_[id] = slot;
    lru_touch_locked(slot, stamp);
    index_dirty_ = true;
    if (!options_.defer_index_updates) { persist_index_locked(); }
    return true;
}

bool DiskKVStore::read_page(const DiskKVIdentity& id, std::span<std::byte> destination) {
    if (destination.size() != options_.slot_size) { return false; }
    std::uint32_t slot = 0;
    std::uint32_t crc  = 0;
    {
        std::lock_guard lock(mutex_);
        const auto found = index_.find(id);
        if (found == index_.end() || !matches(header(found->second), id)) { return false; }
        slot = found->second;
        crc  = header(slot).crc;
        ++readers_[slot];
    }
    const bool intact = read_payload(slot, destination) &&
                        (!options_.verify_crc || crc32(destination) == crc);
    std::lock_guard lock(mutex_);
    --readers_[slot];
    if (intact) {
        header(slot).last_used = ++clock_;
        lru_touch_locked(slot, clock_);
        index_dirty_ = true;
    }
    return intact;
}

std::optional<DiskKVStore::ReadClaim> DiskKVStore::claim_read(const DiskKVIdentity& id) {
    std::lock_guard lock(mutex_);
    const auto found = index_.find(id);
    if (found == index_.end() || !matches(header(found->second), id)) { return std::nullopt; }
    const std::uint32_t slot = found->second;
    ++readers_[slot];
    return ReadClaim{.slot = slot, .offset = payload_offset(slot), .crc = header(slot).crc};
}

bool DiskKVStore::claim_intact(const ReadClaim& claim,
                               std::span<const std::byte> payload) const noexcept {
    return payload.size() == options_.slot_size &&
           (!options_.verify_crc || crc32(payload) == claim.crc);
}

void DiskKVStore::release_read(const ReadClaim& claim, bool intact) {
    std::lock_guard lock(mutex_);
    --readers_[claim.slot];
    if (intact) {
        header(claim.slot).last_used = ++clock_;
        lru_touch_locked(claim.slot, clock_);
        index_dirty_ = true;
    }
}

bool DiskKVStore::contains(const DiskKVIdentity& id) const {
    std::lock_guard lock(mutex_);
    return index_.contains(id);
}

bool DiskKVStore::touch(const DiskKVIdentity& id) {
    std::lock_guard lock(mutex_);
    const auto found = index_.find(id);
    if (found == index_.end()) { return false; }
    header(found->second).last_used = ++clock_;
    lru_touch_locked(found->second, clock_);
    index_dirty_ = true;
    return true;
}

bool DiskKVStore::evict(const DiskKVIdentity& id) {
    std::lock_guard lock(mutex_);
    const auto found = index_.find(id);
    if (found == index_.end() || readers_[found->second] != 0) { return false; }
    const std::uint32_t slot = found->second;
    index_.erase(found);
    lru_erase_locked(slot);
    clear_slot(slot);
    free_slots_.push_back(slot);
    index_dirty_ = true;
    persist_index_locked();
    return true;
}

std::vector<DiskKVIdentity> DiskKVStore::live_identities() const {
    std::lock_guard lock(mutex_);
    std::vector<DiskKVIdentity> out;
    out.reserve(index_.size());
    for (const auto& entry : index_) { out.push_back(entry.first); }
    return out;
}

void DiskKVStore::flush_index() {
    std::lock_guard lock(mutex_);
    persist_index_locked();
}

} // namespace ninfer
