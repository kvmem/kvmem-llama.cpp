// Numerical check for the host stage-in pipeline.
//
// The port runs the adapter's stage-in kernels as host code. This test drives
// the public kvmem_stagein_* entry points and compares their output against a
// reference implementation written from the mathematical definition, not from
// the kernel bodies:
//
//   Walsh-Hadamard  out[i] = (1/sqrt(n)) * sum_j x[j] * (-1)^popcount(i & j)
//   NeoX RoPE       rotate pairs (i, i + n_rot/2) by angle pos * theta[i]
//   q8_0 / q4_0     dequant and quant as defined by the block layout
//
// It is registered only on the host (Apple unified memory) backend.

#include "llama-kvmem-stagein.h"

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The adapter's transfer accounting lives in llama-memory-kvmem.cpp. This test
// links only the stage-in object file, so stub the counter out.
void kvmem_record_transfer(int, uint64_t) {}

namespace {

constexpr int QK8_0 = 32;
constexpr int QK4_0 = 32;

struct RefQ8 { uint16_t d; int8_t qs[QK8_0]; };            // 34 bytes
struct RefQ4 { uint16_t d; uint8_t qs[QK4_0 / 2]; };       // 18 bytes

// --- reference ------------------------------------------------------------

void ref_dequant_q8(const uint8_t * src, float * dst, int64_t n) {
    const int64_t nb = n / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        RefQ8 b;
        std::memcpy(&b, src + i * (int64_t) sizeof(RefQ8), sizeof(RefQ8));
        const float d = ggml_fp16_to_fp32(b.d);
        for (int j = 0; j < QK8_0; ++j) {
            dst[i * QK8_0 + j] = (float) b.qs[j] * d;
        }
    }
}

void ref_dequant_q4(const uint8_t * src, float * dst, int64_t n) {
    const int64_t nb = n / QK4_0;
    for (int64_t i = 0; i < nb; ++i) {
        RefQ4 b;
        std::memcpy(&b, src + i * (int64_t) sizeof(RefQ4), sizeof(RefQ4));
        const float d = ggml_fp16_to_fp32(b.d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            dst[i * QK4_0 + j]             = (float) ((b.qs[j] & 0x0F) - 8) * d;
            dst[i * QK4_0 + j + QK4_0 / 2] = (float) ((b.qs[j] >> 4)   - 8) * d;
        }
    }
}

void ref_rope_neox(float * x, int64_t n_tokens, int n_head, int n_embd_head,
                   int n_rot, int32_t pos0, const std::vector<float> & theta) {
    const int half = n_rot / 2;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < n_head; ++h) {
            float * row = x + (t * n_head + h) * n_embd_head;
            const float pos = (float) (pos0 + t);
            for (int i = 0; i < half; ++i) {
                const float ang = pos * theta[i];
                const float s   = sinf(ang);
                const float c   = cosf(ang);
                const float x0  = row[i];
                const float x1  = row[i + half];
                row[i]       = x0 * c - x1 * s;
                row[i + half] = x0 * s + x1 * c;
            }
        }
    }
}

void ref_fwht(float * x, int64_t n_rows, int64_t n_embd, int nrot) {
    const float scale = 1.0f / sqrtf((float) nrot);
    const int64_t nblocks = n_embd / nrot;
    std::vector<float> tmp((size_t) nrot);
    for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t b = 0; b < nblocks; ++b) {
            float * row = x + r * n_embd + b * nrot;
            for (int i = 0; i < nrot; ++i) {
                // out[i] = sum_j x[j] * (-1)^popcount(i & j)
                float acc = 0.0f;
                for (int j = 0; j < nrot; ++j) {
                    const int sign = (__builtin_popcount((unsigned) (i & j)) & 1) ? -1.0f : 1.0f;
                    acc += sign * row[j];
                }
                tmp[(size_t) i] = acc * scale;
            }
            for (int i = 0; i < nrot; ++i) {
                row[i] = tmp[(size_t) i];
            }
        }
    }
}

void ref_quant_q8(const float * x, int64_t n, std::vector<uint8_t> & out) {
    const int64_t nb = n / QK8_0;
    out.resize((size_t) nb * sizeof(RefQ8));
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = fmaxf(amax, fabsf(x[i * QK8_0 + j]));
        }
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        RefQ8 b;
        b.d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) {
            b.qs[j] = (int8_t) roundf(x[i * QK8_0 + j] * id);
        }
        std::memcpy(out.data() + i * (int64_t) sizeof(RefQ8), &b, sizeof(RefQ8));
    }
}

void ref_quant_q4(const float * x, int64_t n, std::vector<uint8_t> & out) {
    const int64_t nb = n / QK4_0;
    out.resize((size_t) nb * sizeof(RefQ4));
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        float vmax = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = x[i * QK4_0 + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                vmax = v;
            }
        }
        const float d  = vmax / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        RefQ4 b;
        b.d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            int xi0 = (int) (x[i * QK4_0 + j] * id + 8.5f);
            int xi1 = (int) (x[i * QK4_0 + j + QK4_0 / 2] * id + 8.5f);
            if (xi0 > 15) { xi0 = 15; }
            if (xi1 > 15) { xi1 = 15; }
            b.qs[j] = (uint8_t) ((xi0 & 15) | ((xi1 & 15) << 4));
        }
        std::memcpy(out.data() + i * (int64_t) sizeof(RefQ4), &b, sizeof(RefQ4));
    }
}

// --- harness --------------------------------------------------------------

int g_fail = 0;

void check(bool ok, const char * what) {
    if (ok) {
        printf("  ok   %s\n", what);
    } else {
        printf("  FAIL %s\n", what);
        ++g_fail;
    }
}

// Compares two quantized buffers, reporting the largest per-element drift.
// A +/-1 step on the integer codes is tolerable: the WHT is evaluated in a
// different summation order, so the last bit of the scale can land differently.
bool compare_blocks(const uint8_t * got, const uint8_t * want, size_t nbytes,
                    size_t block, const char * what) {
    size_t nblock = nbytes / block;
    int worst_d = 0;
    int worst_q = 0;
    size_t bad  = 0;
    for (size_t i = 0; i < nblock; ++i) {
        const uint16_t dg = 0;
        (void) dg;
        uint16_t d_got, d_want;
        std::memcpy(&d_got,  got  + i * block, sizeof(uint16_t));
        std::memcpy(&d_want, want + i * block, sizeof(uint16_t));
        const int dd = (int) d_got - (int) d_want;
        if (std::abs(dd) > std::abs(worst_d)) { worst_d = dd; }
        // fp16 bit patterns of the scale differ by more than 1 ulp?
        if (std::abs(dd) > 4) { ++bad; }
        for (size_t j = sizeof(uint16_t); j < block; ++j) {
            const int q = (int) got[i * block + j] - (int) want[i * block + j];
            if (std::abs(q) > std::abs(worst_q)) { worst_q = q; }
            if (std::abs(q) > 1) { ++bad; }
        }
    }
    printf("  %s: %zu blocks, worst scale delta %d, worst code delta %d, "
           "out-of-tolerance %zu\n", what, nblock, worst_d, worst_q, bad);
    return bad == 0;
}

// One end-to-end run: packed -> dequant -> RoPE -> WHT -> quantize.
bool run_case(ggml_type ty, int64_t nt, int n_head, int n_embd_head, int n_rot_rope,
              int nrot, int32_t pos0, unsigned seed) {
    const int64_t n_embd = (int64_t) n_head * n_embd_head;
    const size_t  block  = (ty == GGML_TYPE_Q8_0) ? sizeof(RefQ8) : sizeof(RefQ4);

    // Source floats, then packed with the reference quantizer.
    std::vector<float> src((size_t) (nt * n_embd));
    unsigned s = seed;
    for (size_t i = 0; i < src.size(); ++i) {
        s = s * 1103515245u + 12345u;
        src[i] = ((float) (s >> 16) / 32768.0f - 1.0f) * 3.5f;
    }

    std::vector<uint8_t> packed;
    if (ty == GGML_TYPE_Q8_0) {
        ref_quant_q8(src.data(), nt * n_embd, packed);
    } else {
        ref_quant_q4(src.data(), nt * n_embd, packed);
    }

    // Reference pipeline.
    std::vector<float> ref((size_t) (nt * n_embd));
    if (ty == GGML_TYPE_Q8_0) {
        ref_dequant_q8(packed.data(), ref.data(), nt * n_embd);
    } else {
        ref_dequant_q4(packed.data(), ref.data(), nt * n_embd);
    }
    const int n_theta = n_rot_rope / 2;
    std::vector<float> theta((size_t) n_theta);
    for (int i = 0; i < n_theta; ++i) {
        theta[(size_t) i] = 1.0f / powf(10000.0f, (float) i / (float) n_theta);
    }
    ref_rope_neox(ref.data(), nt, n_head, n_embd_head, n_rot_rope, pos0, theta);
    if (nrot > 0) {
        ref_fwht(ref.data(), nt, n_embd, nrot);
    }
    std::vector<uint8_t> want;
    if (ty == GGML_TYPE_Q8_0) {
        ref_quant_q8(ref.data(), nt * n_embd, want);
    } else {
        ref_quant_q4(ref.data(), nt * n_embd, want);
    }

    // Adapter pipeline, straight into a host destination.
    std::vector<uint8_t> got(want.size(), 0);
    int64_t copy_us = 0, rope_us = 0, had_us = 0, set_us = 0;

    const bool enq = kvmem_stagein_enqueue_k(
            ty, packed.data(), packed.size(), got.data(),
            nt, n_embd, nrot, n_head, n_embd_head, n_rot_rope, pos0,
            theta.data(), n_theta,
            &copy_us, &rope_us, &had_us, &set_us);
    if (!enq) {
        printf("  enqueue_k failed\n");
        return false;
    }
    if (!kvmem_stagein_flush(&copy_us, &rope_us, &had_us, &set_us)) {
        printf("  flush failed\n");
        return false;
    }
    kvmem_stagein_sync();

    return compare_blocks(got.data(), want.data(), want.size(), block, "stage-in");
}

}  // namespace

int main() {
    const size_t slab = (size_t) 32 * 1024 * 1024;

    printf("stage-in host backend: numerical check\n");

    if (!kvmem_stagein_gpu_ready(1 << 20, slab)) {
        printf("FAIL: kvmem_stagein_gpu_ready\n");
        return 1;
    }
    printf("  scratch ready (f32=%zu, packed=%zu)\n", (size_t) (1 << 20), slab);

    printf("q8_0, RoPE only (n_rot=32, no WHT)\n");
    check(run_case(GGML_TYPE_Q8_0, 8, 4, 64, 32, 0, 0, 1), "q8_0 rope");

    printf("q8_0, RoPE + WHT-64\n");
    check(run_case(GGML_TYPE_Q8_0, 8, 4, 64, 32, 64, 7, 2), "q8_0 rope+fwht64");

    printf("q8_0, WHT-128 (two heads wide)\n");
    check(run_case(GGML_TYPE_Q8_0, 6, 2, 128, 64, 128, 0, 3), "q8_0 fwht128");

    printf("q4_0, RoPE + WHT-64\n");
    check(run_case(GGML_TYPE_Q4_0, 8, 4, 64, 32, 64, 5, 4), "q4_0 rope+fwht64");

    kvmem_stagein_gpu_free();

    if (g_fail == 0) {
        printf("PASS: stage-in host pipeline matches the reference\n");
        return 0;
    }
    printf("FAIL: %d case(s) diverged\n", g_fail);
    return 1;
}
