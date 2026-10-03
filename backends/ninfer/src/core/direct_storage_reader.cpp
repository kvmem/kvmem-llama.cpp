#include "core/direct_storage_reader.h"

#include <stdexcept>

#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

#    include <dstorage.h>

#    include <algorithm>
#    include <chrono>
#    include <limits>
#    include <mutex>
#    include <thread>
#    include <unordered_map>

namespace ninfer {
namespace {

template <typename Interface>
class ComReference {
public:
    ComReference()                               = default;
    ComReference(const ComReference&)            = delete;
    ComReference& operator=(const ComReference&) = delete;

    ComReference(ComReference&& other) noexcept : pointer_(other.pointer_) {
        other.pointer_ = nullptr;
    }

    ComReference& operator=(ComReference&& other) noexcept {
        if (this != &other) {
            reset();
            pointer_       = other.pointer_;
            other.pointer_ = nullptr;
        }
        return *this;
    }

    ~ComReference() { reset(); }

    void reset() noexcept {
        if (pointer_ != nullptr) { pointer_->Release(); }
        pointer_ = nullptr;
    }

    [[nodiscard]] Interface* get() const noexcept { return pointer_; }

    Interface* operator->() const noexcept { return pointer_; }

    void** out() noexcept {
        reset();
        return reinterpret_cast<void**>(&pointer_);
    }

private:
    Interface* pointer_ = nullptr;
};

std::wstring widen(const std::string& text) {
    if (text.empty()) { return {}; }
    const int length =
        ::MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    if (length <= 0) { throw std::invalid_argument("DirectStorage path is not valid UTF-8"); }
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(),
                          length);
    return wide;
}

class WindowsDirectStorageReader final : public DirectStorageReader {
public:
    WindowsDirectStorageReader() {
        // dstorage.dll is delay-loaded: a missing DLL is an error here rather than a crash at
        // the first DirectStorage call.
        if (::LoadLibraryW(L"dstorage.dll") == nullptr) {
            throw std::runtime_error("dstorage.dll was not found next to the executable");
        }
        if (FAILED(DStorageGetFactory(__uuidof(IDStorageFactory), factory_.out()))) {
            throw std::runtime_error("DirectStorage factory is unavailable");
        }
        DSTORAGE_QUEUE_DESC queue{};
        queue.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        queue.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
        queue.Priority   = DSTORAGE_PRIORITY_NORMAL;
        queue.Name       = "ninfer disk KV restore";
        queue.Device     = nullptr; // memory destinations only
        if (FAILED(factory_->CreateQueue(&queue, __uuidof(IDStorageQueue), queue_.out()))) {
            throw std::runtime_error("DirectStorage queue could not be created");
        }
        if (FAILED(factory_->CreateStatusArray(1, "ninfer disk KV restore",
                                               __uuidof(IDStorageStatusArray), status_.out()))) {
            throw std::runtime_error("DirectStorage status array could not be created");
        }
    }

    ~WindowsDirectStorageReader() override {
        if (queue_.get() != nullptr) { queue_->Close(); }
        for (auto& [path, file] : files_) {
            if (file.get() != nullptr) { file->Close(); }
        }
    }

    bool read(const std::string& path, std::span<const FileRangeRead> reads) override {
        std::lock_guard lock(mutex_);
        if (broken_) { return false; }
        IDStorageFile* file = open_file(path);
        if (file == nullptr) { return false; }
        // A batch never exceeds the queue: the status for a batch is enqueued behind its reads.
        const std::size_t batch_capacity = DSTORAGE_MAX_QUEUE_CAPACITY - 1U;
        for (std::size_t begin = 0; begin < reads.size(); begin += batch_capacity) {
            const std::size_t end = std::min(reads.size(), begin + batch_capacity);
            for (std::size_t index = begin; index < end; ++index) {
                const FileRangeRead& range = reads[index];
                if (range.destination.size() > std::numeric_limits<UINT32>::max()) { return false; }
                const auto size = static_cast<UINT32>(range.destination.size());
                DSTORAGE_REQUEST request{};
                request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                request.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
                request.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                request.Source.File.Source        = file;
                request.Source.File.Offset        = range.offset;
                request.Source.File.Size          = size;
                request.Destination.Memory.Buffer = range.destination.data();
                request.Destination.Memory.Size   = size;
                request.UncompressedSize          = size;
                request.CancellationTag           = batch_tag_;
                queue_->EnqueueRequest(&request);
            }
            queue_->EnqueueStatus(status_.get(), 0);
            queue_->Submit();
            const bool completed = wait_for_batch();
            ++batch_tag_;
            if (!completed) { return false; }
        }
        return true;
    }

private:
    IDStorageFile* open_file(const std::string& path) {
        if (const auto found = files_.find(path); found != files_.end()) {
            return found->second.get();
        }
        ComReference<IDStorageFile> file;
        if (FAILED(factory_->OpenFile(widen(path).c_str(), __uuidof(IDStorageFile), file.out()))) {
            return nullptr;
        }
        IDStorageFile* raw = file.get();
        files_.emplace(path, std::move(file));
        return raw;
    }

    bool wait_until_complete(std::chrono::steady_clock::duration budget) {
        const auto deadline = std::chrono::steady_clock::now() + budget;
        while (!status_->IsComplete(0)) {
            if (std::chrono::steady_clock::now() > deadline) { return false; }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        return true;
    }

    // A stuck batch must not hold a restore forever: it is cancelled, and the caller falls back
    // to mapped reads. A batch that does not even settle after cancellation could still write
    // into the caller's buffers later, so the reader takes no more work.
    bool wait_for_batch() {
        if (!wait_until_complete(std::chrono::seconds(10))) {
            queue_->CancelRequestsWithTag(~UINT64{0}, batch_tag_);
            if (!wait_until_complete(std::chrono::seconds(10))) { broken_ = true; }
            return false;
        }
        if (FAILED(status_->GetHResult(0))) {
            DSTORAGE_ERROR_RECORD record{};
            queue_->RetrieveErrorRecord(&record);
            return false;
        }
        return true;
    }

    std::mutex mutex_;
    UINT64 batch_tag_ = 1;
    bool broken_      = false;
    ComReference<IDStorageFactory> factory_;
    ComReference<IDStorageQueue> queue_;
    ComReference<IDStorageStatusArray> status_;
    std::unordered_map<std::string, ComReference<IDStorageFile>> files_;
};

} // namespace

std::unique_ptr<DirectStorageReader> DirectStorageReader::open() {
    return std::make_unique<WindowsDirectStorageReader>();
}

} // namespace ninfer

#else

namespace ninfer {

std::unique_ptr<DirectStorageReader> DirectStorageReader::open() {
    throw std::runtime_error("this build has no DirectStorage (Windows with NINFER_DIRECTSTORAGE)");
}

} // namespace ninfer

#endif
