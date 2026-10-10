#include "kvmem-session-transfer.h"
#include "kvmem/raw_kv_store.hpp"
#include <iostream>
#include <cstring>
#include <chrono>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<class F> static void rejects(F f) {
    bool failed = false; try { f(); } catch (const std::exception &) { failed = true; }
    check(failed, "expected transfer failure");
}
struct test_payload {
    std::vector<std::vector<uint8_t>> vectors;
    std::unique_ptr<kvmem_session_payload> payload;
    uint8_t seed;
    size_t accounted = 0;
    test_payload(int id, int blocks, size_t unit, uint64_t group = 1) : vectors(blocks), seed(uint8_t(id * 31)) {
        std::vector<kvmem::SnapshotBuffer> buffers;
        for (int i = 0; i < blocks; ++i) {
            auto & v = vectors[i]; v.resize(unit); accounted += unit;
            for (size_t j = 0; j < unit; ++j) v[j] = uint8_t(seed + i + j);
            auto b = kvmem::SnapshotBuffer::bind(v); b.accounting = &accounted;
            b.account = [](void * p, int64_t n) {
                auto & bytes = *static_cast<size_t *>(p);
                if (n < 0) { check(bytes >= size_t(-n), "accounting underflow"); bytes -= size_t(-n); }
                else bytes += size_t(n);
            };
            buffers.push_back(b);
        }
        payload = std::make_unique<kvmem_session_payload>(id, 17, buffers, group);
    }
    void verify(kvmem_session_files & files) {
        size_t bytes = 0;
        for (size_t i = 0; i < vectors.size(); ++i) {
            const auto & v = vectors[i]; bytes += v.size();
            for (size_t j = 0; j < v.size(); ++j) check(v[j] == uint8_t(seed+i+j), "cross-session or corrupt RAM");
        }
        check(bytes == accounted, "checkpoint accounting mismatch");
        for (uint32_t i = 0; i < payload->chunks.size(); ++i)
            check(payload->chunks[i].in_ram || files.ready(payload->id, i), "lost both chunk copies");
    }
};
struct fixture {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("kvmem-transfer-test-" + std::to_string(std::random_device{}()));
    std::unique_ptr<kvmem_session_files> files;
    test_payload a, b;
    uint64_t ram_cap, peak_ram = 0, peak_disk = 0;
    fixture(uint64_t unit, uint64_t ram_units = 15, uint64_t disk_units = 20, uint64_t group = 1)
        : a(1, 10, size_t(unit), group), b(2, 15, size_t(unit), group), ram_cap(ram_units * unit) {
        files = std::make_unique<kvmem_session_files>(root, disk_units * unit);
        for (uint32_t i = 0; i < b.payload->chunks.size(); ++i) b.payload->spill(*files, i);
        sample();
    }
    ~fixture() { files.reset(); std::error_code ec; std::filesystem::remove(root, ec); }
    uint64_t ram() const {
        uint64_t n = 0;
        for (const auto * p : {&a, &b}) for (const auto & v : p->vectors) n += v.capacity();
        return n;
    }
    uint64_t available() { check(ram() <= ram_cap, "RAM peak exceeded"); return ram_cap - ram(); }
    void sample() {
        peak_ram = std::max(peak_ram, ram()); peak_disk = std::max(peak_disk, files->bytes());
        check(ram() <= ram_cap, "RAM peak exceeded"); check(files->bytes() <= files->limit(), "disk peak exceeded");
    }
    kvmem_session_plan plan(bool reverse = false) {
        return kvmem_plan_session_transfer(*files, reverse ? a.payload.get() : b.payload.get(),
            {reverse ? b.payload.get() : a.payload.get()}, available(), files->bytes(), files->limit());
    }
    void execute(const kvmem_session_plan & p) {
        for (const auto & m : p.moves) {
            kvmem_session_plan single; single.moves.push_back(m);
            kvmem_execute_session_transfer(*files, single, [&] { return available(); }); sample();
        }
    }
    void verify() { a.verify(*files); b.verify(*files); sample(); }
};

static void routes() {
    constexpr uint64_t unit = 1024 * 1024;
    for (int route = 0; route < 3; ++route) {
        fixture f(unit, route == 1 ? 25 : 15, route == 0 ? 32 : 20);
        const auto p = f.plan();
        const auto expected = route == 0 ? kvmem_session_plan::path::disk_first :
            route == 1 ? kvmem_session_plan::path::ram_first : kvmem_session_plan::path::exchange;
        check(p.route == expected, "wrong transfer path");
        f.execute(p); f.verify();
        check(f.b.payload->complete() && f.a.payload->ram_bytes() == 0, "final residency");
        check(f.files->session_bytes(1) == f.a.payload->disk_bytes() && !f.files->contains(2), "final disk ownership");
        check(f.ram() == 15*unit, "active 15 must survive soft limit 12");
        if (route == 2) check(f.peak_ram == 15*unit && f.peak_disk <= 20*unit, "10/15/20 exchange peaks");
        f.execute(f.plan(true)); f.verify();
        check(f.a.payload->complete() && f.b.payload->ram_bytes() == 0, "reverse exchange");
    }
    // One indivisible allocation per session: no temporary room. Planning must
    // fail before writing/deleting anything, even though the final layout fits.
    fixture f(4096, 15, 20, UINT64_MAX);
    const auto before = f.files->bytes();
    rejects([&] { f.plan(); }); f.verify();
    check(f.a.payload->complete() && f.files->bytes() == before, "planning changed cache");
    // Real final quota shortage also produces no partial execution.
    rejects([&] { kvmem_plan_session_transfer(*f.files, nullptr, {f.a.payload.get()},
        1000000, f.files->bytes(), f.files->limit()); });
    check(f.a.payload->complete() && f.files->bytes() == before, "final shortage changed cache");
}

// Opt-in stability run at one tenth of the original GiB scenario.  Each
// allocation is a little over 102 MiB, so the planner must actually exchange
// chunks instead of relying on enough RAM or disk for both whole sessions.
static void scale_1to10() {
    constexpr uint64_t unit = (uint64_t(1) << 30) / 10;
    fixture f(unit, 15, 20);
    std::vector<double> forward_ms, reverse_ms;
    for (int cycle = 0; cycle < 2; ++cycle) {
        const auto forward = f.plan();
        check(forward.route == kvmem_session_plan::path::exchange, "1:10 forward did not exchange chunks");
        auto started = std::chrono::steady_clock::now();
        f.execute(forward);
        forward_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        f.verify();
        check(f.b.payload->complete() && f.a.payload->ram_bytes() == 0, "1:10 forward residency");
        check(f.files->session_bytes(1) == f.a.payload->disk_bytes() && !f.files->contains(2),
              "1:10 forward disk ownership");
        const auto reverse = f.plan(true);
        check(reverse.route == kvmem_session_plan::path::exchange, "1:10 reverse did not exchange chunks");
        started = std::chrono::steady_clock::now();
        f.execute(reverse);
        reverse_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        f.verify();
        check(f.a.payload->complete() && f.b.payload->ram_bytes() == 0, "1:10 reverse residency");
    }
    check(f.peak_ram <= 15*unit && f.peak_disk <= 20*unit, "1:10 transient capacity exceeded");
    std::cout << "1:10 exchange passed: A=" << 10*unit << " B=" << 15*unit
              << " disk_quota=" << 20*unit << " peak_ram=" << f.peak_ram
              << " peak_disk=" << f.peak_disk << " bytes\n";
    for (size_t i = 0; i < forward_ms.size(); ++i)
        std::cout << "cycle=" << i << " A_to_B_ms=" << forward_ms[i]
                  << " B_to_A_ms=" << reverse_ms[i] << "\n";
}

static void failure_matrix() {
    // Every write/read/publication/deletion boundary, including failures after
    // several chunks have moved. After each failure reconstruct the ORIGINAL
    // A session first, then switch to B again, verifying exact data both ways.
    for (const std::string phase : {"write", "rename", "read", "validated", "erase"}) {
        const int operations = phase == "write" || phase == "rename" ? 10 : 15;
        for (int fail_at = 1; fail_at <= operations; ++fail_at) {
            fixture f(4096);
            const auto p = f.plan();
            int seen = 0;
            f.files->fault = [&](const char * event, int, uint32_t) {
                f.sample();
                if (phase == event && ++seen == fail_at) throw std::runtime_error("injected transfer failure");
            };
            rejects([&] { f.execute(p); }); f.verify();
            f.files->fault = {};
            f.execute(f.plan(true)); f.verify(); check(f.a.payload->complete(), "could not resume outgoing A");
            f.execute(f.plan()); f.verify(); check(f.b.payload->complete(), "could not retry incoming B");
        }
    }
}

static void corruption_and_pressure() {
    fixture f(4096);
    // Valid file and checksum but wrong session/generation/chunk identity.
    const uint32_t index = 7;
    check(f.files->erase_chunk(2, index), "erase fixture chunk");
    const auto & c = f.b.payload->chunks[index];
    f.files->save_chunk(2, index, c.payload_bytes(), [&](kvmem::SnapshotWriter & out) {
        out.scalar(uint64_t(0x324b4e554843564bull)); out.scalar(int32_t(1));
        out.scalar(uint64_t(17)); out.scalar(index); out.scalar(uint64_t(1));
        out.scalar(uint64_t(4096)); std::vector<uint8_t> fake(4096, 1); out.write(fake.data(), fake.size());
    });
    rejects([&] { f.execute(f.plan()); }); f.verify();
    check(!f.b.payload->complete(), "corrupt B exposed as complete");
    // A remains recoverable even though B cannot be used. No need to discard A.
    const auto back = f.plan(true);
    f.execute(back); f.a.verify(*f.files); check(f.a.payload->complete(), "corrupt B destroyed A");

    fixture pressure(4096);
    const auto plan = pressure.plan();
    int reads = 0;
    rejects([&] { kvmem_execute_session_transfer(*pressure.files, plan, [&] {
        return ++reads > 2 ? uint64_t(0) : pressure.available();
    }); });
    pressure.verify(); pressure.execute(pressure.plan(true)); pressure.verify();
}

static void partial_io_and_cleanup() {
    for (const std::string phase : {"write_data", "read_data"}) {
        // Many allocations in one file. Fail after the first allocation was
        // processed to exercise rollback/accounting inside an unfinished chunk.
        fixture f(4096, 25, 32, UINT64_MAX);
        int seen = 0;
        f.files->fault = [&](const char * event, int, uint32_t) {
            if (phase == event && ++seen == 9) throw std::runtime_error("partial I/O");
        };
        rejects([&] { f.execute(f.plan()); }); f.verify();
        f.files->fault = {};
        f.execute(f.plan(true)); f.verify();
        f.execute(f.plan()); f.verify();
    }
    fixture f(4096, 25, 32, UINT64_MAX);
    const auto before = f.files->bytes();
    int writes = 0;
    f.files->fault = [&](const char * event, int, uint32_t) {
        if (std::string(event) == "erase" || (std::string(event) == "write_data" && ++writes == 9))
            throw std::runtime_error("partial write and cleanup failure");
    };
    rejects([&] { f.execute(f.plan()); }); f.verify();
    check(f.files->bytes() == before + f.a.payload->disk_bytes(), "failed temp cleanup lost quota charge");
    check(!f.files->ready(1, 0) && f.a.payload->complete(), "failed write lost RAM source");
    f.files->fault = {}; check(f.files->erase_chunk(1, 0), "retry temp cleanup");
    f.execute(f.plan()); f.verify();
    // Corrupt the checksum of cold A, after several other chunks have moved.
    {
        std::fstream file(f.files->path(1), std::ios::binary|std::ios::in|std::ios::out);
        file.seekg(-1, std::ios::end); char byte; file.read(&byte, 1); byte ^= 1;
        file.seekp(-1, std::ios::end); file.write(&byte, 1);
    }
    rejects([&] { f.execute(f.plan(true)); }); f.verify();
    check(!f.a.payload->complete(), "checksum failure exposed complete A");
}

static void raw_roundtrip() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 2; cfg.n_embd_k = cfg.n_embd_v = 64; cfg.block_tokens = 64;
    cfg.k_gpu_row_bytes = cfg.v_gpu_row_bytes = 68;
    kvmem::RawKvStore raw(cfg);
    std::vector<uint8_t> expected(71*68);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = uint8_t(i*17);
    for (uint32_t l = 0; l < 2; ++l) {
        raw.write_layer_k_gpu(0, 71, l, expected.data()); raw.write_layer_v_gpu(0, 71, l, expected.data());
    }
    std::vector<kvmem::SnapshotBuffer> buffers; raw.snapshot_buffers(buffers);
    kvmem_session_payload payload(7, 1, buffers, 4096);
    const auto root = std::filesystem::temp_directory_path() / ("kvmem-raw-transfer-" + std::to_string(std::random_device{}()));
    {
        kvmem_session_files files(root, payload.disk_bytes());
        const auto before = raw.allocated_bytes();
        for (uint32_t i = 0; i < payload.chunks.size(); ++i) payload.spill(files, i);
        check(raw.allocated_bytes() < before && payload.ram_bytes() == 0, "raw vectors not released");
        for (uint32_t i = 0; i < payload.chunks.size(); ++i) payload.restore(files, i);
        for (uint32_t l = 0; l < 2; ++l) for (uint32_t block = 0; block < 2; ++block) {
            const uint32_t n = block ? 7 : 64; std::vector<uint8_t> got(n*68);
            check(raw.copy_k_gpu(block, l, got.data(), n) && !std::memcmp(got.data(), expected.data()+block*64*68, got.size()), "raw K mismatch");
            check(raw.copy_v_gpu(block, l, got.data(), n) && !std::memcmp(got.data(), expected.data()+block*64*68, got.size()), "raw V mismatch");
        }
    }
    std::filesystem::remove(root / kvmem_session_cache_dir::root_lock_name());
    std::filesystem::remove(root);
}

static void shared_native_and_idle() {
    const auto root = std::filesystem::temp_directory_path() /
        ("kvmem-shared-idle-" + std::to_string(std::random_device{}()));
    {
        auto disk = std::make_shared<kvmem::SpillFile>(root, 4096);
        auto storage = std::make_shared<kvmem::HostKvStorage>(128, disk);
        kvmem::RawKvStoreConfig cfg;
        cfg.n_layer = 1; cfg.n_embd_k = cfg.n_embd_v = 4; cfg.block_tokens = 4;
        cfg.k_gpu_row_bytes = cfg.v_gpu_row_bytes = 8; cfg.payload_storage = storage;
        kvmem::RawKvStore raw(cfg);
        std::vector<uint8_t> native(96, 71), copy(32);
        std::vector<float> means(12 * 4, 2.0f);
        raw.write_layer_k_gpu(0, 12, 0, native.data());
        raw.write_layer_v_gpu(0, 12, 0, native.data());
        raw.write_layer_mean_k(0, 12, 0, means.data());
        std::vector<kvmem::SnapshotBuffer> buffers;
        raw.snapshot_buffers(buffers);
        check(buffers.size() == 3, "idle serializer duplicated shared native K/V");
        kvmem_session_payload payload(7, 1, buffers);
        kvmem_session_files files(root, 4096, {}, disk);
        const auto native_disk = raw.payload_bytes(true);
        files.fault = [](const char * point, int, uint32_t) {
            if (std::string(point) == "write_data") throw std::runtime_error("injected idle write failure");
        };
        rejects([&] { payload.spill(files, 0); });
        check(payload.complete() && files.bytes() == native_disk, "failed idle write leaked quota or lost source");
        files.fault = {};
        payload.spill(files, 0);
        check(files.bytes() == native_disk + payload.disk_bytes(), "native/idle D charges are separate");
        storage->trim(0);
        check(storage->charged_bytes() == 0 && raw.payload_bytes(true) == native.size() * 2,
              "idle native records did not remain independently tierable");
        const auto cold_bytes = files.bytes();
        files.fault = [](const char * point, int, uint32_t) {
            if (std::string(point) == "read_data") throw std::runtime_error("injected idle read failure");
        };
        rejects([&] { payload.restore(files, 0); });
        check(!payload.complete() && files.bytes() == cold_bytes,
              "failed idle read released the valid disk attachment");
        files.fault = {};
        payload.restore(files, 0);
        check(payload.complete() && files.bytes() == native.size() * 2,
              "restoring idle state copied or released native disk records");
        check(storage->charged_bytes() == 0, "idle restore eagerly promoted all native KV");
        for (uint32_t block = 0; block < 3; ++block) {
            check(raw.copy_k_gpu(block, 0, copy.data(), 4), "missing shared K");
            check(std::all_of(copy.begin(), copy.end(), [](uint8_t value) { return value == 71; }), "changed shared K");
            check(raw.copy_v_gpu(block, 0, copy.data(), 4), "missing shared V");
            check(std::all_of(copy.begin(), copy.end(), [](uint8_t value) { return value == 71; }), "changed shared V");
            float mean[4]; raw.mean_k(block, 0, mean);
            check(std::all_of(std::begin(mean), std::end(mean), [](float value) { return value == 2.0f; }),
                  "idle mean-K restore changed values");
        }
        // Restoring the attachment leaves a reusable hole between native
        // extents; exhaust the quota without assuming all free space is contiguous.
        std::vector<kvmem::SpillExtent> remaining;
        while (files.bytes() < 4096) remaining.push_back(disk->reserve(8));
        rejects([&] { payload.spill(files, 0); });
        check(payload.complete(), "shared D exhaustion released valid idle RAM");
    }
    std::filesystem::remove(root / kvmem_session_cache_dir::root_lock_name());
    check(std::filesystem::is_empty(root), "shared spill file leaked");
    std::filesystem::remove(root);
}

int main(int argc, char ** argv) {
    try {
    if (argc == 2 && std::string(argv[1]) == "--scale-1to10") {
        scale_1to10();
        return 0;
    }
    if (argc != 1) {
        std::cerr << "usage: kvmem-session-transfer-test [--scale-1to10]\n";
        return 2;
    }
    routes(); failure_matrix(); corruption_and_pressure(); partial_io_and_cleanup(); raw_roundtrip();
    shared_native_and_idle();
    std::cout << "session exchange paths, 10/15/20 peaks, fault recovery, identity and raw KV passed\n";
    } catch (const std::exception & error) {
        std::cerr << "session transfer failure: " << error.what() << '\n';
        return 1;
    }
}
