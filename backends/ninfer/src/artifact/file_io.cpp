#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#ifdef _WIN32

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation, DWORD error) {
    throw ArtifactError(path.string() + ": " + operation + ": Win32 error " +
                        std::to_string(static_cast<unsigned long>(error)));
}

HANDLE handle(void* value) noexcept { return static_cast<HANDLE>(value); }

// POSIX open() does not stop another process or handle from replacing, renaming or deleting the
// file; share every access so a Reader behaves the same way on Windows.
constexpr DWORD kShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

// One positional read. Overlapped I/O carries the offset, so reads on the same handle never share
// a file pointer; a synchronous handle completes the request inside ReadFile.
DWORD positional_read(HANDLE file, bool overlapped, std::uint64_t offset, std::byte* destination,
                      DWORD count, const std::filesystem::path& path, const char* operation) {
    OVERLAPPED request{};
    request.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    request.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD read         = 0;
    if (!::ReadFile(file, destination, count, &read, &request)) {
        const DWORD error = ::GetLastError();
        if (error == ERROR_HANDLE_EOF) { return 0; }
        if (!overlapped || error != ERROR_IO_PENDING) { fail(path, operation, error); }
        if (!::GetOverlappedResult(file, &request, &read, TRUE)) {
            const DWORD final_error = ::GetLastError();
            if (final_error == ERROR_HANDLE_EOF) { return 0; }
            fail(path, operation, final_error);
        }
    }
    return read;
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

#ifdef _WIN32

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    const HANDLE file = ::CreateFileW(path_.c_str(), GENERIC_READ, kShareMode, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { fail(path_, "CreateFileW", ::GetLastError()); }
    file_ = file;
    BY_HANDLE_FILE_INFORMATION information{};
    if (!::GetFileInformationByHandle(file, &information)) {
        const DWORD error = ::GetLastError();
        ::CloseHandle(file);
        file_ = nullptr;
        fail(path_, "GetFileInformationByHandle", error);
    }
    if ((information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        ::CloseHandle(file);
        file_ = nullptr;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32U) |
             information.nFileSizeLow;
}

InputFile::~InputFile() {
    if (direct_file_ != nullptr) { ::CloseHandle(handle(direct_file_)); }
    if (file_ != nullptr) { ::CloseHandle(handle(file_)); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count =
            static_cast<DWORD>(std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024));
        const DWORD read = positional_read(handle(file_), false, offset, destination.data(), count,
                                           path_, "ReadFile");
        if (read == 0) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_file_ == nullptr) {
        // FILE_FLAG_NO_BUFFERING is the Windows O_DIRECT: sector-aligned offsets, sizes and
        // buffers, which the checks above already guarantee for the 4096-byte payload alignment.
        const HANDLE file = ::CreateFileW(
            path_.c_str(), GENERIC_READ, kShareMode, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED |
                FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr);
        if (file == INVALID_HANDLE_VALUE) { fail(path_, "CreateFileW direct", ::GetLastError()); }
        direct_file_ = file;
    }
    // ReadFile takes a DWORD count; stay a whole number of aligned blocks below 4 GiB.
    constexpr std::size_t kMaximumRequest = 1ULL << 30U;
    std::size_t total                     = 0;
    while (total < destination.size()) {
        const auto count =
            static_cast<DWORD>(std::min<std::size_t>(destination.size() - total, kMaximumRequest));
        const DWORD read = positional_read(handle(direct_file_), true, offset + total,
                                           destination.data() + total, count, path_,
                                           "direct ReadFile");
        total += read;
        if (read != count) { break; }
    }
    return total;
}

#else

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

#endif

} // namespace ninfer::artifact
