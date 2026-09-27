# ROCm contribution provenance

## PR #58 local integration

The RDNA2 quantized-KV Flash Attention VEC dispatch fix is by **Ciyan Ouyang (@zintown)**,
original commit `eda272b793819f3582aa6ac3a7c84b20c4d4ff94`, author email
`ciyan_ucas@outlook.com`, from https://github.com/kvmem/kvmem-llama.cpp/pull/58.
Its kernel-selection patch is retained unchanged; integration additionally wires it into
the Windows build entry. The original commit is retained locally at `review/pr58` and
as a format-patch backup. It is not yet part of this dirty working tree's commit ancestry;
final Git integration must retain that commit or carry explicit co-author attribution.
RDNA2 F16 draft KV is outside that fix. The author's working RDNA2 configuration uses
quantized draft KV. Current RX9060XT testing is RDNA4 regression coverage, not RDNA2 validation.

This integration preserves the original commit ancestry of both contributions.

| Contributor | Original contribution | Retained work / integration |
|---|---|---|
| dockylf | [PR #10](https://github.com/kvmem/kvmem-llama.cpp/pull/10), head `b183c3a` | Built the Windows HIP foundation: the backend-portable adapter abstraction in `llama-kvmem-gpu.h` that turned hard-coded CUDA runtime calls into a pluggable interface — the ground work that made any non-CUDA backend possible; the `-DGGML_HIP=ON` build backend; Windows host-library build fixes; Windows environment-variable semantics fix; the Windows ROCm/HIP build guide and helpers; server smoke checks. |
| FangJiangyi | [PR #33](https://github.com/kvmem/kvmem-llama.cpp/pull/33), head `a7460f3` | Brought up Linux ROCm end to end: native HIP compilation of the stage-in sources with a ROCm memory canary; enabled MTP GDN replay on a non-CUDA backend for the first time; multimodal benchmark tuning with a recommended-config/performance document containing the 7900 XTX measurements; cumulative-patch factory-header fix; CPU vision thread propagation. |
| Zeshen | Integrated branch | Cross-platform unification and HIP hardening: fixed the GDN fold kernel's hard-coded 32-lane partition so HIP uses the device wave width while CUDA stays at 32 lanes; isolated CUDA toolkit discovery to the CUDA backend; unified the cross-platform build entry with automatic GPU-architecture detection including WSL DXG; added build unit tests and strengthened the replay regression test with an FP64 reference and multi-layer fold coverage; ported the multimodal canary to Windows with WorkingSet and Private Bytes sampling; added dual-platform IQ3 launchers, relocatable Linux runtime libraries, EOL-safe source export and a VRAM telemetry fix; validated full 256K 33-round runs on both Windows and Linux with ROCm 7.2. |

Overlapping runtime aliases and build blocks have one implementation in the final tree.
The original commits remain accessible through merge ancestry even where lines were subsequently changed.
Line counts are not percentages of ownership.

The CUDA kernel launch remains 32 lanes. HIP selects the device wave width for GDN folding.
The standalone shim avoids requiring private ggml CUDA vendor headers from adapter code.
HIP staging is an object target: native HIP on Linux, AMD Clang CXX/HIP mode on Windows.
CUDA toolkit discovery is restricted to the CUDA backend.

For final integration, use a normal **merge commit** to retain this history.
A squash merge discards the individual commit ancestry; attribution would then need to be explicitly
carried into the final squash commit. GitHub associates author email addresses with accounts and
updates contributor graphs after commits reach the default branch; this process is not instantaneous.

Both original author identities are unchanged:

- dockylf: `dzdg_ddd@126.com`
- FangJiangyi: `2301111925@stu.pku.edu.cn`

AI assisted the integration and validation preparation. The submitter reviews the final changes before publication.
