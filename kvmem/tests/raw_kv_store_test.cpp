#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"
#include "kvmem/nvme_kv_tier.hpp"
#include <filesystem>

#include <algorithm>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__,         \
                         __LINE__, #cond);                                     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Compare lazy normalization with the previous ordered FP32 sum + cached mean.
// Cancellation and non-power-of-two tails catch mean-to-sum reconstruction.
static void test_sum_only(uint32_t block_tokens) {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 3; // The other two layers model uncaptured recurrent layers.
    cfg.n_embd_k = 1024;
    cfg.n_embd_v = 1024;
    cfg.block_tokens = block_tokens;
    cfg.k_row_bytes = 1088; // Quantized product path has no raw-K fallback.
    kvmem::RawKvStore raw(cfg);
    const uint32_t count = 2 * block_tokens + 7;
    std::vector<float> input(count * cfg.n_embd_k);
    const float values[] = {1e7f, .125f, -1e7f, .3f, -7.1f, .001f, 9.7f};
    for (uint32_t t = 0; t < count; ++t) {
        for (uint32_t d = 0; d < cfg.n_embd_k; ++d) {
            input[t * cfg.n_embd_k + d] = values[(t + d) % 7];
        }
    }
    const auto verify = [&](uint32_t end) {
        const uint32_t blocks = (end + block_tokens - 1) / block_tokens;
        // Exactly one F32 vector per populated block/layer, with no lazy cache.
        CHECK(raw.bytes_k() == size_t(blocks) * cfg.n_embd_k * sizeof(float));
        for (uint32_t b = 0; b < blocks; ++b) {
            CHECK(raw.has_block(b));
            const uint32_t start = b * block_tokens;
            const uint32_t n = std::min(block_tokens, end - start);
            std::vector<float> expected(cfg.n_embd_k, 0.0f), got(cfg.n_embd_k);
            for (uint32_t t = start; t < start + n; ++t) {
                for (uint32_t d = 0; d < cfg.n_embd_k; ++d) {
                    expected[d] += input[t * cfg.n_embd_k + d];
                }
            }
            const float inv = 1.0f / static_cast<float>(n);
            for (auto & x : expected) x *= inv;
            for (int repeat = 0; repeat < 3; ++repeat) {
                raw.mean_k(b, 1, got.data());
                CHECK(std::memcmp(got.data(), expected.data(), got.size() * sizeof(float)) == 0);
            }
        }
        CHECK(raw.bytes_k() == size_t(blocks) * cfg.n_embd_k * sizeof(float));
    };
    raw.write_layer_mean_k(0, 5, 1, input.data());
    verify(5);
    const auto checkpoint = raw.mean_checkpoint(5);
    CHECK(checkpoint.size() == cfg.n_layer * (1 + cfg.n_embd_k));
    CHECK(checkpoint[0] == 0 && checkpoint[1 + cfg.n_embd_k] == 5);
    CHECK(raw.mean_checkpoint(block_tokens).empty());
    // Continue in uneven batches across block boundaries.
    for (uint32_t pos = 5; pos < count;) {
        const uint32_t n = std::min(11u, count - pos);
        raw.write_layer_mean_k(pos, n, 1, input.data() + pos * cfg.n_embd_k);
        pos += n;
        verify(pos);
    }
    raw.truncate_to(5);
    std::vector<float> missing(cfg.n_embd_k, 123.0f);
    raw.mean_k(0, 1, missing.data());
    for (float x : missing) CHECK(x == 0.0f);
    raw.restore_mean_checkpoint(5, checkpoint);
    verify(5);
    // Reads/rollback must not round-trip a mean back into an approximate sum.
    raw.write_layer_mean_k(5, count - 5, 1, input.data() + 5 * cfg.n_embd_k);
    verify(count);
    for (const auto & where : {std::pair<uint32_t, uint32_t>{0, 0}, {99, 1}, {0, 99}}) {
        std::fill(missing.begin(), missing.end(), 123.0f);
        raw.mean_k(where.first, where.second, missing.data());
        for (float x : missing) CHECK(x == 0.0f);
    }
    raw.truncate_to(0);
    CHECK(raw.bytes_k() == 0 && !raw.has_block(0));
}

// A host-store swap moves whole mirrors between owners, so two mirrors built
// from the same config must share nothing: the same block ids hold independent
// bytes, ownership travels with the object, and a clear on one leaves the other
// alone. The swap's lockstep between the trunk mirror and the MTP follower's
// rests on exactly that.
static void test_sibling_stores_are_independent() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 1;
    cfg.n_embd_k = 4;
    cfg.n_embd_v = 4;
    cfg.block_tokens = 4;
    cfg.k_gpu_row_bytes = 6;
    cfg.v_gpu_row_bytes = 6;
    auto a = std::make_unique<kvmem::RawKvStore>(cfg);
    auto b = std::make_unique<kvmem::RawKvStore>(cfg);

    std::vector<uint8_t> pa(24, 0xa1), pb(24, 0xb2), out(24, 0);
    a->write_layer_k_gpu(0, 4, 0, pa.data());
    a->write_layer_v_gpu(0, 4, 0, pa.data());
    b->write_layer_k_gpu(0, 4, 0, pb.data());
    a->wait_writes();
    b->wait_writes();
    CHECK(a->has_k_gpu(0, 0, 4) && b->has_k_gpu(0, 0, 4));
    CHECK(a->has_v_gpu(0, 0, 4) && !b->has_v_gpu(0, 0, 4));
    CHECK(a->copy_k_gpu(0, 0, out.data(), 4));
    CHECK(std::memcmp(out.data(), pa.data(), pa.size()) == 0);
    CHECK(b->copy_k_gpu(0, 0, out.data(), 4));
    CHECK(std::memcmp(out.data(), pb.data(), pb.size()) == 0);

    // Detach hands the whole object over; the bytes travel with it.
    std::unique_ptr<kvmem::RawKvStore> moved = std::move(a);
    CHECK(!a);
    CHECK(moved->copy_k_gpu(0, 0, out.data(), 4));
    CHECK(std::memcmp(out.data(), pa.data(), pa.size()) == 0);

    moved->clear();
    CHECK(!moved->has_block(0));
    CHECK(moved->bytes_k() == 0 && moved->bytes_v() == 0);
    CHECK(b->has_k_gpu(0, 0, 4));
    CHECK(b->copy_k_gpu(0, 0, out.data(), 4));
    CHECK(std::memcmp(out.data(), pb.data(), pb.size()) == 0);
}

// --kvmem-conversations-gb caps the sum of bytes_k() + bytes_v() per store, so
// that number has to be predictable: it is the packed GPU K/V held for the
// blocks the store still owns, plus the F32 mean sums, plus the NVMe tail.
static void test_store_bytes() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd_k = 4;
    cfg.n_embd_v = 4;
    cfg.block_tokens = 4;
    cfg.k_gpu_row_bytes = 6;
    cfg.v_gpu_row_bytes = 6;
    kvmem::RawKvStore raw(cfg);
    CHECK(raw.bytes_k() == 0 && raw.bytes_v() == 0);

    const size_t block = size_t(cfg.n_layer) * cfg.block_tokens *
            size_t(cfg.k_gpu_row_bytes + cfg.v_gpu_row_bytes);
    std::vector<uint8_t> packed(cfg.block_tokens * cfg.k_gpu_row_bytes, 0x5a);
    for (uint32_t il = 0; il < cfg.n_layer; ++il) {
        raw.write_layer_k_gpu(0, cfg.block_tokens, il, packed.data());
        raw.write_layer_v_gpu(0, cfg.block_tokens, il, packed.data());
    }
    raw.wait_writes();
    // One whole block of packed K and V for every layer, and nothing else:
    // write_layer_k_gpu does not capture a mean sum.
    CHECK(raw.bytes_k() + raw.bytes_v() == block);

    for (uint32_t il = 0; il < cfg.n_layer; ++il) {
        raw.write_layer_k_gpu(cfg.block_tokens, cfg.block_tokens, il, packed.data());
        raw.write_layer_v_gpu(cfg.block_tokens, cfg.block_tokens, il, packed.data());
    }
    raw.wait_writes();
    CHECK(raw.bytes_k() + raw.bytes_v() == 2 * block);

    // Dropping the second block returns the total to exactly the one-block
    // value, which is what makes an eviction's accounting exact.
    raw.truncate_to(cfg.block_tokens);
    CHECK(raw.bytes_k() + raw.bytes_v() == block);

    // A mean sum is F32 per embedding dimension per layer, on top of the
    // packed bytes, and is the reason the number is accounted store bytes
    // rather than pure packed KV.
    std::vector<float> mean(cfg.block_tokens * cfg.n_embd_k, 1.0f);
    raw.write_layer_mean_k(0, cfg.block_tokens, 0, mean.data());
    raw.wait_writes();
    CHECK(raw.bytes_k() + raw.bytes_v() == block + size_t(cfg.n_embd_k) * sizeof(float));

    raw.clear();
    CHECK(raw.bytes_k() == 0 && raw.bytes_v() == 0);
}

int main() {
    test_sum_only(32);
    test_sum_only(128);
    test_sibling_stores_are_independent();
    test_store_bytes();
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd_k = 4;
    cfg.n_embd_v = 4;
    cfg.block_tokens = 4;
    kvmem::RawKvStore raw(cfg);

    std::vector<float> k(8, 0.0f);
    std::vector<float> v(8, 1.0f);
    for (int i = 0; i < 8; ++i) {
        k[i] = static_cast<float>(i);
    }
    raw.write_layer_tokens(0, 2, 0, k.data(), nullptr);
    CHECK(raw.n_tokens(0) == 2);
    CHECK(raw.has_k(0, 0));
    CHECK(!raw.has_v(0, 0));
    // The K sum is captured at write so RAM scoring does not copy_k.
    std::vector<float> got(8, -1.0f);
    CHECK(raw.copy_k(0, 0, got.data()));
    CHECK(got[0] == 0.0f);
    CHECK(got[5] == 5.0f);
    CHECK(raw.bytes_v() == 0);
    CHECK(raw.bytes_k() > 0);

    raw.write_layer_tokens(0, 2, 0, nullptr, v.data());
    CHECK(raw.has_v(0, 0));
    std::vector<float> gv(8, 0.0f);
    CHECK(raw.copy_v(0, 0, gv.data()));
    CHECK(std::fabs(gv[0] - 1.0f) < 1e-3f);

    std::vector<float> mean(4, 0.0f);
    raw.mean_k(0, 0, mean.data());
    CHECK(std::fabs(mean[0] - 2.0f) < 1e-3f); // (0+4)/2
    CHECK(std::fabs(mean[1] - 3.0f) < 1e-3f); // (1+5)/2

    std::vector<float> k2(8, 0.0f);
    for (int i = 0; i < 8; ++i) {
        k2[i] = static_cast<float>(i + 8);
    }
    raw.write_layer_tokens(2, 2, 0, k2.data(), nullptr);
    CHECK(raw.n_tokens(0) == 4);
    raw.mean_k(0, 0, mean.data());
    CHECK(std::fabs(mean[0] - 6.0f) < 1e-3f); // (0+4+8+12)/4
    CHECK(std::fabs(mean[1] - 7.0f) < 1e-3f); // (1+5+9+13)/4

    kvmem::RopeConfig rc;
    rc.n_rot = 4;
    rc.n_embd_head = 4;
    rc.n_head_kv = 1;
    rc.freq_base = 10000.0f;
    std::vector<float> src(4, 0.0f);
    src[0] = 1.0f;
    src[2] = 1.0f;
    std::vector<float> dst(4, 0.0f);
    kvmem::rope_neox_apply(rc, src.data(), 1, 0, dst.data());
    // pos=0 → identity rotation
    CHECK(std::fabs(dst[0] - 1.0f) < 1e-5f);
    CHECK(std::fabs(dst[2] - 1.0f) < 1e-5f);

    kvmem::rope_neox_apply(rc, src.data(), 1, 1, dst.data());
    CHECK(std::fabs(dst[0] - src[0]) > 1e-6f || std::fabs(dst[2] - src[2]) > 1e-6f);

    // Hybrid: only attention layers are captured; layer 0 may stay empty.
    kvmem::RawKvStore raw_h(cfg);
    raw_h.write_layer_tokens(0, 2, 1, k.data(), nullptr);
    CHECK(raw_h.has_block(0));
    CHECK(raw_h.n_tokens(0) == 2);
    CHECK(!raw_h.has_k(0, 0));
    CHECK(raw_h.has_k(0, 1));
    std::vector<float> hgot(8, -1.0f);
    CHECK(raw_h.copy_k(0, 1, hgot.data()));
    CHECK(hgot[0] == 0.0f);

#if KVMEM_ENABLE_NVME
    kvmem::RawKvStoreConfig ncfg = cfg;
    ncfg.nvme_dir = (std::filesystem::temp_directory_path() / "kvmem_raw_k_test").string();
    ncfg.nvme_file = "raw.bin";
    ncfg.nvme_bytes = 4ull * 1024ull * 1024ull;
    kvmem::RawKvStore rawn(ncfg);
    CHECK(rawn.nvme_enabled());
    std::vector<float> kfull(16, 0.0f);
    for (int i = 0; i < 16; ++i) {
        kfull[i] = static_cast<float>(i);
    }
    rawn.write_layer_tokens(0, 4, 0, kfull.data(), nullptr);
    CHECK(rawn.has_k(0, 0));
    std::vector<float> nout(16, -1.0f);
    CHECK(rawn.copy_k(0, 0, nout.data()));
    CHECK(nout[0] == 0.0f);
    CHECK(nout[15] == 15.0f);
    std::vector<float> nmean(4, 0.0f);
    rawn.mean_k(0, 0, nmean.data());
    CHECK(std::fabs(nmean[0] - 6.0f) < 1e-2f); // (0+4+8+12)/4

#endif

    // F16 write: 0x3c00 is 1.0 in IEEE half.
    kvmem::RawKvStore raw16(cfg);
    std::vector<uint16_t> ones(8, 0x3c00);
    raw16.write_layer_tokens_f16(0, 2, 0, ones.data(), nullptr);
    std::vector<float> f16out(8, 0.0f);
    CHECK(raw16.copy_k(0, 0, f16out.data()));
    CHECK(std::fabs(f16out[0] - 1.0f) < 1e-3f);
    CHECK(std::fabs(f16out[7] - 1.0f) < 1e-3f);
    std::vector<float> m16(4, 0.0f);
    raw16.mean_k(0, 0, m16.data());
    CHECK(std::fabs(m16[0] - 1.0f) < 1e-3f);
    CHECK(std::fabs(m16[3] - 1.0f) < 1e-3f);

    kvmem::RawKvStoreConfig gcfg = cfg;
    gcfg.v_gpu_row_bytes = 6;
    kvmem::RawKvStore rawg(gcfg);
    std::vector<uint8_t> packed(12);
    for (int i = 0; i < 12; ++i) {
        packed[static_cast<size_t>(i)] = static_cast<uint8_t>(i + 1);
    }
    rawg.write_layer_v_gpu(0, 2, 0, packed.data());
    CHECK(rawg.has_v(0, 0));
    CHECK(rawg.has_v_gpu(0, 0));
    CHECK(!rawg.has_block(0));
    kvmem::RawKvStoreConfig kgcfg = cfg;
    kgcfg.k_gpu_row_bytes = 6;
    kvmem::RawKvStore rawkg(kgcfg);
    rawkg.write_layer_k_gpu(0, 2, 0, packed.data());
    CHECK(rawkg.has_k_gpu(0, 0));
    CHECK(rawkg.has_block(0));
    CHECK(!rawkg.has_k(0, 0));
    std::vector<uint8_t> kgout(12, 0);
    CHECK(rawkg.copy_k_gpu(0, 0, kgout.data(), 2));
    CHECK(kgout[0] == 1);
    CHECK(kgout[11] == 12);

    kvmem::RawKvStore rawm(cfg);
    rawm.write_layer_mean_k(0, 2, 0, k.data());
    CHECK(rawm.has_block(0));
    CHECK(!rawm.has_k(0, 0));
    CHECK(!rawm.has_k_gpu(0, 0));
    std::vector<float> mm(4, 0.0f);
    rawm.mean_k(0, 0, mm.data());
    CHECK(std::fabs(mm[0] - 2.0f) < 1e-3f);
    rawm.clear();
    CHECK(!rawm.has_block(0));
    rawm.write_layer_mean_k(0, 1, 0, k.data());
    std::vector<float> mm0(4, 0.0f);
    rawm.mean_k(0, 0, mm0.data());
    CHECK(std::fabs(mm0[0] - k[0]) < 1e-3f);
    rawm.write_layer_mean_k(4, 1, 0, k.data() + 4);
    CHECK(rawm.has_block(1));
    rawm.truncate_to(4);
    CHECK(rawm.has_block(0));
    CHECK(!rawm.has_block(1));
    kvmem::RawKvStore raws(cfg);
    raws.write_layer_mean_k(0, 1, 0, k.data());
    raws.write_layer_mean_k(1, 1, 0, k.data() + 4);
    std::vector<float> ms(4, 0.0f);
    raws.mean_k(0, 0, ms.data());
    CHECK(std::fabs(ms[0] - 2.0f) < 1e-3f);
    auto sumcfg = cfg;
    sumcfg.k_gpu_row_bytes = 6;
    kvmem::RawKvStore rawsum(sumcfg);
    std::vector<float> ksum(4, 0.0f);
    for (int d = 0; d < 4; ++d) {
        ksum[static_cast<size_t>(d)] = k[static_cast<size_t>(d)] + k[static_cast<size_t>(d + 4)];
    }
    rawsum.write_layer_mean_sum(0, 2, 0, ksum.data());
    std::vector<float> msum(4, 0.0f);
    rawsum.mean_k(0, 0, msum.data());
    CHECK(std::fabs(msum[0] - 2.0f) < 1e-3f);
    // Packed rows can run ahead of captured statistics: normalize by mean_tokens.
    const std::vector<uint8_t> sum_packed(24, 0);
    rawsum.write_layer_k_gpu(0, 4, 0, sum_packed.data());
    rawsum.mean_k(0, 0, msum.data());
    CHECK(msum[0] == 2.0f);
    rawsum.write_layer_mean_sum(2, 1, 0, k2.data());
    rawsum.mean_k(0, 0, msum.data());
    CHECK(msum[0] == (ksum[0] + k2[0]) * (1.0f / 3.0f));
    std::vector<uint8_t> gout(12, 0);
    CHECK(rawg.copy_v_gpu(0, 0, gout.data(), 2));
    CHECK(gout[0] == 1);
    CHECK(gout[11] == 12);
    std::vector<float> no_f32(8, 0.0f);
    CHECK(!rawg.copy_v(0, 0, no_f32.data()));

    kvmem::RawKvStoreConfig ngcfg = cfg;
    ngcfg.v_gpu_row_bytes = 6;
    ngcfg.nvme_dir = (std::filesystem::temp_directory_path() / "kvmem_raw_vgpu_test").string();
    ngcfg.nvme_file = "raw_vgpu.bin";
    ngcfg.nvme_bytes = 4ull * 1024ull * 1024ull;
    std::vector<uint8_t> packed4(24);
    for (int i = 0; i < 24; ++i) {
        packed4[static_cast<size_t>(i)] = static_cast<uint8_t>(i + 1);
    }
    std::vector<uint8_t> gout4(24, 0);
#if KVMEM_ENABLE_NVME
    kvmem::RawKvStore rawgn(ngcfg);
    CHECK(rawgn.nvme_enabled());
    rawgn.write_layer_v_gpu(0, 4, 0, packed4.data());
    CHECK(rawgn.has_v(0, 0));
    CHECK(rawgn.copy_v_gpu(0, 0, gout4.data(), 4));
    CHECK(gout4[0] == 1);
    CHECK(gout4[23] == 24);
    CHECK(!rawgn.copy_v(0, 0, no_f32.data()));

#endif

    kvmem::RawKvStoreConfig kcfg = cfg;
    kcfg.k_row_bytes = 6;
    kvmem::RawKvStore rawk(kcfg);
    rawk.write_layer_k_rows(0, 2, 0, packed.data(), k.data());
    CHECK(rawk.has_k(0, 0));
    std::vector<uint8_t> kout(12, 0);
    CHECK(rawk.copy_k_rows(0, 0, kout.data(), 2));
    CHECK(kout[0] == 1);
    CHECK(kout[11] == 12);
    CHECK(!rawk.copy_k(0, 0, no_f32.data()));
    std::vector<float> mk(4, 0.0f);
    rawk.mean_k(0, 0, mk.data());
    CHECK(std::fabs(mk[0] - 2.0f) < 1e-3f);

#if KVMEM_ENABLE_NVME
    kvmem::RawKvStoreConfig nkcfg = cfg;
    nkcfg.k_row_bytes = 6;
    nkcfg.nvme_dir = (std::filesystem::temp_directory_path() / "kvmem_raw_krow_test").string();
    nkcfg.nvme_file = "raw_krow.bin";
    nkcfg.nvme_bytes = 4ull * 1024ull * 1024ull;
    kvmem::RawKvStore rawkn(nkcfg);
    CHECK(rawkn.nvme_enabled());
    std::vector<float> krowf(16, 0.0f);
    for (int i = 0; i < 16; ++i) {
        krowf[static_cast<size_t>(i)] = static_cast<float>(i);
    }
    rawkn.write_layer_k_rows(0, 4, 0, packed4.data(), krowf.data());
    CHECK(rawkn.has_k(0, 0));
    std::vector<uint8_t> kout4(24, 0);
    CHECK(rawkn.copy_k_rows(0, 0, kout4.data(), 4));
    CHECK(kout4[0] == 1);
    CHECK(kout4[23] == 24);
    CHECK(!rawkn.copy_k(0, 0, no_f32.data()));
    std::vector<float> krowmean(4, 0.0f);
    rawkn.mean_k(0, 0, krowmean.data());
    CHECK(std::fabs(krowmean[0] - 6.0f) < 1e-3f);

#endif

    // Replacing a suffix preserves prefix bytes and rejects stale packed rows,
    // even if the mean-K capture has already extended the logical block.
    for (bool nvme : {false, true}) {
        if (nvme && !KVMEM_ENABLE_NVME) continue;
        auto tail_cfg = ngcfg;
        tail_cfg.k_gpu_row_bytes = 6;
        if (!nvme) tail_cfg.nvme_bytes = 0;
        kvmem::RawKvStore tail_store(tail_cfg);
        tail_store.write_layer_mean_k(0, 2, 0, k.data());
        const auto checkpoint = tail_store.mean_checkpoint(2);
        tail_store.write_layer_mean_k(2, 2, 0, k2.data());
        tail_store.write_layer_k_gpu(0, 4, 0, packed4.data());
        tail_store.write_layer_v_gpu(0, 4, 0, packed4.data());
        tail_store.truncate_to(2);
        tail_store.restore_mean_checkpoint(2, checkpoint);
        CHECK(tail_store.n_tokens(0) == 2);
        CHECK(!tail_store.copy_k_gpu(0, 0, gout4.data(), 4));
        CHECK(!tail_store.copy_v_gpu(0, 0, gout4.data(), 4));
        CHECK(tail_store.copy_k_gpu(0, 0, gout4.data(), 2));
        CHECK(std::memcmp(gout4.data(), packed4.data(), 12) == 0);
        std::vector<float> replacement(8, 20.0f);
        tail_store.write_layer_mean_k(2, 2, 0, replacement.data());
        CHECK(!tail_store.has_k_gpu(0, 0, 4));
        CHECK(!tail_store.has_v_gpu(0, 0, 4));
        tail_store.mean_k(0, 0, mean.data());
        CHECK(std::fabs(mean[0] - 11.0f) < 1e-3f);
        std::vector<uint8_t> new_bytes(12, 231);
        tail_store.write_layer_k_gpu(2, 2, 0, new_bytes.data());
        tail_store.write_layer_v_gpu(2, 2, 0, new_bytes.data());
        for (bool is_k : {false, true}) {
            CHECK(is_k ? tail_store.copy_k_gpu(0, 0, gout4.data(), 4)
                       : tail_store.copy_v_gpu(0, 0, gout4.data(), 4));
            CHECK(std::memcmp(gout4.data(), packed4.data(), 12) == 0);
            CHECK(std::memcmp(gout4.data() + 12, new_bytes.data(), 12) == 0);
        }
        tail_store.invalidate_packed_from(3);
        CHECK(!tail_store.has_k_gpu(0, 0, 4));
        CHECK(!tail_store.has_v_gpu(0, 0, 4));
        CHECK(tail_store.copy_k_gpu(0, 0, gout4.data(), 3));
        CHECK(tail_store.copy_v_gpu(0, 0, gout4.data(), 3));
    }
    return 0;
}
