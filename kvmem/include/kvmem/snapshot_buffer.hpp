#pragma once

#include "snapshot.hpp"
#include "host_kv_storage.hpp"
#include <algorithm>

namespace kvmem {
// A frozen allocation, not a copy of its contents. Bindings remain valid only
// while their owning detached store is frozen (no inference/vector relocation).
// Reclaim whole allocations: writing slices of a vector cannot release its RAM.
struct SnapshotBuffer {
    std::shared_ptr<void> binding;
    void * object = nullptr;
    uint64_t count = 0, size = 0;
    uint64_t (*capacity)(const void *) = nullptr;
    void (*write)(const void *, SnapshotWriter &) = nullptr;
    void (*read)(void *, uint64_t, SnapshotReader &) = nullptr;
    void (*clear)(void *) = nullptr;
    void * accounting = nullptr;
    void (*account)(void *, int64_t) = nullptr;

    uint64_t ram_bytes() const { return capacity(object); }
    void release() const {
        if (account) account(accounting, -int64_t(size));
        clear(object);
    }
    void restore(SnapshotReader & in) const {
        read(object, count, in);
        if (account) account(accounting, int64_t(size));
    }
    template<class T> static SnapshotBuffer bind(std::vector<T> & values) {
        static_assert(std::is_arithmetic<T>::value, "plain vector required");
        SnapshotBuffer b;
        b.object = &values; b.count = values.size(); b.size = uint64_t(values.size()) * sizeof(T);
        b.capacity = [](const void * p) { return uint64_t(static_cast<const std::vector<T> *>(p)->capacity()) * sizeof(T); };
        b.write = [](const void * p, SnapshotWriter & out) { out.vector(*static_cast<const std::vector<T> *>(p)); };
        b.read = [](void * p, uint64_t count, SnapshotReader & in) {
            // Exact length comes from our in-memory manifest, never the file.
            in.expect(count);
            auto & v = *static_cast<std::vector<T> *>(p);
            std::vector<T> next(static_cast<size_t>(count));
            in.read(next.data(), count * sizeof(T));
            v.swap(next);
        };
        b.clear = [](void * p) { std::vector<T>().swap(*static_cast<std::vector<T> *>(p)); };
        return b;
    }
    static SnapshotBuffer bind(HostKvRecord & record, HostKvStorage & storage) {
        struct Binding { HostKvRecord * record; HostKvStorage * storage; };
        auto binding = std::make_shared<Binding>(Binding{&record, &storage});
        SnapshotBuffer b;
        b.binding = binding; b.object = binding.get();
        b.count = b.size = record.size();
        b.capacity = [](const void * p) { return static_cast<const Binding *>(p)->record->resident_bytes(); };
        b.write = [](const void * p, SnapshotWriter & out) {
            const auto & r = *static_cast<const Binding *>(p)->record;
            out.scalar(uint64_t(r.size()));
            if (const auto lease = r.lease()) {
                if (lease.data()) out.write(lease.data(), lease.size());
                else {
                    uint8_t chunk[16384];
                    for (size_t offset = 0; offset < lease.size();) {
                        const auto bytes = std::min(sizeof(chunk), lease.size() - offset);
                        lease.read(offset, chunk, bytes); out.write(chunk, bytes); offset += bytes;
                    }
                }
            }
        };
        b.read = [](void * p, uint64_t count, SnapshotReader & in) {
            in.expect(count);
            if (count > SIZE_MAX || count > in.remaining()) throw SnapshotCorrupt("invalid native KV length");
            auto & binding = *static_cast<Binding *>(p);
            auto next = binding.storage->allocate(static_cast<size_t>(count));
            in.read(next.writable_data(), count);
            next.seal();
            *binding.record = std::move(next);
        };
        b.clear = [](void * p) { static_cast<Binding *>(p)->record->reset(); };
        return b;
    }
};
} // namespace kvmem
