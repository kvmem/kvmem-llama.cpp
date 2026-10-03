#pragma once

// Reads file ranges into host memory through Microsoft DirectStorage, on Windows builds with
// NINFER_DIRECTSTORAGE: every range of a batch goes to the NVMe queue at once, with little CPU
// work per read. Other builds have no reader, and callers keep their mapped-file reads.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace ninfer {

struct FileRangeRead {
    std::uint64_t offset = 0;
    std::span<std::byte> destination;
};

class DirectStorageReader {
public:
    [[nodiscard]] static constexpr bool available_in_build() noexcept {
#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)
        return true;
#else
        return false;
#endif
    }

    // Throws when this build or machine has no working DirectStorage.
    [[nodiscard]] static std::unique_ptr<DirectStorageReader> open();

    virtual ~DirectStorageReader() = default;

    // Reads every range of the file at `path` (UTF-8); false when any read failed.
    [[nodiscard]] virtual bool read(const std::string& path,
                                    std::span<const FileRangeRead> reads) = 0;
};

} // namespace ninfer
