// Public wiring for --mlp-a8-decode: which phase may take the integer route, and whether target
// planning reserves what that route allocates.
//
// The FP64 oracle in tests/ops/linear_swiglu/test_q4a8_int.cpp invokes the schedule directly, so it
// says nothing about either. Both were wrong once and neither failure is visible from a kernel
// test: the flag reached prefill, which quantised the tail chunk of a scoring run that documents
// itself as unaffected, and the planner passed A16Only, whose capacity across 16..32 columns is
// zero because that band routes to SmallTTiled -- so the shortfall was the whole allocation rather
// than a margin.

#include "guarded_main.h"
#include "models/qwen3_5/execution/ffn.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

using ninfer::QType;
using ninfer::models::qwen3_5::execution::DenseParameters;
using ninfer::models::qwen3_5::execution::ffn_workspace_bytes;
using ninfer::ops::LinearPolicy;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

// Planning reads only qtype, shape and policy, so the parameters need no device memory. The
// policies are what execution::Parameters assigns to the 27B groupwise-int MLP pair on sm_86,
// mirroring src/models/qwen3_5/execution/parameters.cpp's Prepare::dense().
DenseParameters parameters(bool prefill_a8, bool a8_decode) {
    DenseParameters out;
    out.gate_up.weight.qtype = QType::Q4_G64_FP16;
    out.gate_up.weight.n     = 34816;
    out.gate_up.weight.k     = 5120;
    out.down.weight.qtype    = QType::Q5_G64_FP16;
    out.down.weight.n        = 5120;
    out.down.weight.k        = 17408;
#if defined(NINFER_SM8X_COMPAT)
    if (prefill_a8) {
        out.gate_up.policy = LinearPolicy::AllowA8Int;
        out.down.policy    = LinearPolicy::AllowA8Int;
    }
#endif
    out.verify_gate_up_policy = out.gate_up.policy;
#if defined(NINFER_SM8X_COMPAT)
    // --mlp-a8-decode admits the decode route by this profile's own format/shape, independently
    // of --prefill-a8 -- the two flags must stay orthogonal (see parameters.cpp:dense()).
    if (a8_decode && out.gate_up.weight.qtype == QType::Q4_G64_FP16 &&
        out.gate_up.weight.n == 34816 && out.gate_up.weight.k == 5120) {
        out.verify_gate_up_policy = LinearPolicy::AllowA8IntDecode;
    }
#endif
    return out;
}

void run_planning() {
    const DenseParameters on  = parameters(true, true);
    const DenseParameters off = parameters(true, false);
    for (const std::int32_t width : {16, 24, 32}) {
        const std::size_t prefill = ffn_workspace_bytes(on, width, width, false, false);
        const std::size_t verify  = ffn_workspace_bytes(on, width, width, false, true);
        // The flag must not change what a prefill (or scoring) call reserves or runs.
        check(prefill == ffn_workspace_bytes(off, width, width, false, false),
              "prefill planning must ignore --mlp-a8-decode at T=" + std::to_string(width));
        check(verify >= prefill, "verify planning must not undercut prefill at T=" +
                                     std::to_string(width) + " (" + std::to_string(verify) + " < " +
                                     std::to_string(prefill) + ")");
#if defined(NINFER_SM8X_COMPAT)
        // The route stages one s8 code per (token, hidden) plus an FP16 scale per 64-k group, over
        // the padded tile width. Anything less than that and the arena can throw mid-round.
        const std::size_t codes = static_cast<std::size_t>(width) * 5120;
        check(verify >= prefill + codes,
              "verify planning must cover the integer scratch at T=" + std::to_string(width) +
                  " (needs at least " + std::to_string(codes) + " more than prefill's " +
                  std::to_string(prefill) + ", got " + std::to_string(verify) + ")");
#endif
        check(ffn_workspace_bytes(off, width, width, false, true) == prefill,
              "without the flag, verify planning must match prefill at T=" +
                  std::to_string(width));
    }

    // Outside the route's band the two phases have nothing to differ about.
    for (const std::int32_t width : {1, 8, 64}) {
        check(ffn_workspace_bytes(on, width, width, false, true) ==
                  ffn_workspace_bytes(on, width, width, false, false),
              "planning must match across phases at T=" + std::to_string(width) +
                  ", outside the route's 16..32 band");
    }
}

// --no-prefill-a8 and --mlp-a8-decode are documented as orthogonal: disabling prefill's full-tile
// route must not disable the separately requested decode route, and enabling decode must not
// re-enable prefill.
void run_combined_flags() {
    const DenseParameters decode_only = parameters(/*prefill_a8=*/false, /*a8_decode=*/true);
    const DenseParameters neither     = parameters(/*prefill_a8=*/false, /*a8_decode=*/false);
    const DenseParameters both        = parameters(/*prefill_a8=*/true, /*a8_decode=*/true);

    check(decode_only.gate_up.policy == LinearPolicy::A16Only,
          "--no-prefill-a8 must leave gate_up.policy at A16Only regardless of --mlp-a8-decode");
#if defined(NINFER_SM8X_COMPAT)
    check(decode_only.verify_gate_up_policy == LinearPolicy::AllowA8IntDecode,
          "--mlp-a8-decode must admit the decode route even when --no-prefill-a8 is set");
#endif

    for (const std::int32_t width : {16, 24, 32}) {
        const std::size_t prefill = ffn_workspace_bytes(decode_only, width, width, false, false);
        // --prefill-a8 is off, so a prefill/scoring call must reserve exactly the A16 amount,
        // regardless of --mlp-a8-decode.
        check(prefill == ffn_workspace_bytes(neither, width, width, false, false),
              "prefill planning must stay A16-only with --no-prefill-a8 --mlp-a8-decode at T=" +
                  std::to_string(width));
        // --mlp-a8-decode alone must still admit the same verify-phase decode route it gets
        // with --prefill-a8 also on -- the two flags must not interact.
        const std::size_t verify = ffn_workspace_bytes(decode_only, width, width, false, true);
        check(verify >= prefill, "verify planning must not undercut prefill under "
                                  "--no-prefill-a8 --mlp-a8-decode at T=" +
                                      std::to_string(width));
#if defined(NINFER_SM8X_COMPAT)
        check(verify > prefill,
              "verify planning must admit the decode route under --no-prefill-a8 "
              "--mlp-a8-decode at T=" +
                  std::to_string(width));
#endif
        check(verify == ffn_workspace_bytes(both, width, width, false, true),
              "decode-route verify planning must be identical with --prefill-a8 on or off at T=" +
                  std::to_string(width));
    }
}

int run() {
    run_planning();
    run_combined_flags();
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " --mlp-a8-decode phase gate and workspace planning\n";
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
