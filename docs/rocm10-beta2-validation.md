# ROCm 10 beta2 local validation

## 2026-09-24 IQ3_XXS compiler workaround

Formal Windows server short validation completed: context32768,GPU KV8192,MTP2,loadnone,pure text. All4 requests passed (base,two tool appends,repeat), process exit0; both tool appends replayed7215 rows. This exercises the replay boundary without claiming a new full33/256K run. Build-plan unit tests11/11 passed. Evidence: `E:\Codex_work\diagnostics\win-iq3xxs-formal32k-20260924`.

On Windows RX9060XT, replacing only the IQ3_XXS object built by7.2 in the10 backend recovered the isolated matrix multiply from3.15ms to2.32ms. The10-only compiler option `--amdgpu-enable-rewrite-partial-reg-uses=false` then measured2.23-2.26ms and passed three CPU-reference checks. Both tests use unchanged model weights and math.

Same-model short benchmark, pp2048/tg64, three repetitions, q8_0 KV, load-mode none and Tensile:

| Build | Prefill t/s | Decode t/s |
| --- | ---: | ---: |
| ROCm10 before IQ3_XXS fix | 525.11 +/-2.90 | 16.32 +/-0.10 |
| ROCm10 diagnostic with fix | 550.04 +/-1.67 | 16.58 +/-0.36 |
| ROCm7.2 same source | 559.27 +/-8.82 | 15.12 +/-0.10 |

The local source now scopes this option to Windows,gfx1200,IQ3_XXS and HIP7.15+. `GGML_HIP_IQ3_XXS_REG_WORKAROUND=OFF` disables it for comparison. Other targets and Linux retain their existing settings. The formal14-target rebuild succeeded and all9 Windows regression tests passed (47.66s). The formal bench SHA256 is624D982A74EEC37E114DA4FD62BCA758E3D72E4515A0D78707860A3055EE672B; its pp2048 measured560.11+/-6.27t/s andtg64 16.68+/-0.35t/s, exit0. Evidence: `E:\Codex_work\diagnostics\win-iq3xxs-fullmodel-nopartial` and `E:\Codex_work\diagnostics\win-iq3xxs-nopartial\formal-ctest.log`. Separate Linux testing measured605.15+/-6.23pp before and604.80+/-5.80pp after the option, showing no useful gain, so it remains Windows-only. Linux7.2's512.03+/-154.42pp in that run was too variable to establish an SDK speed ranking.

## 2026-09-24 Windows native short verification (13:02)

The restored benchmark executable initially predated the Q2_K fix. After relinking, same-source Windows builds were tested sequentially on RX9060XT with IQ3, q8_0 KV, load-mode none, batch512, pp2048/tg128 and three repetitions, forcing Tensile for both SDKs:

| SDK | Prefill t/s | Decode t/s |
| --- | ---: | ---: |
| ROCm7.2 | 554.87 +/-2.18 | 15.52 +/-0.10 |
| ROCm10 | 523.66 +/-8.67 | 16.82 +/-0.12 |

In this short benchmark, ROCm10 decode is8.38% faster and prefill5.62% slower. These are not MTP agent metrics. All nine Windows regression cases have passing results: eight in the first CTest invocation, replay separately in37.43s with a120s bound after the initial30s timeout. ROCm10 default BLAS still exits with a ROCm error; the release launcher retains Tensile on RDNA4. Evidence: `E:\Codex_work\diagnostics\win-sdk-screen-20260924-a`, including executable hashes and exact arguments. No release was published.

## 2026-09-24 release-default MTP2 full run

The rebuilt ROCm 10 WSL server completed **33/33 HTTP200 requests** and 32 tool rounds with MTP2 replay, q8_0 KV, 28672-token GPU KV budget and the IQ3 model. The final prompt was **261546/262144**; the final prompt evaluation was **127.29 t/s**. Across all 33 server timing lines, 261578 newly evaluated prompt tokens took 1854.46 s (**141.05 effective prompt t/s**) and 3138 generated tokens took 162.51 s (**19.31 decode t/s**). The 33 `KVMEM_SPEC_STATS` lines recorded **2694 drafted, 1824 accepted, 67.71% accepted, 0 restores**. Runtime RSS peaked at **13811.77 MiB**; Windows GPU Adapter Memory sampled dedicated/shared maxima of **15024.46/1056.59 MiB**.

The final response was a literal `read_file` tool call for `batch_0010.py`, 159/512 completion tokens, `finish_reason=stop`. The canary exited 1 on its final-output assertion. The 33-request context/replay flow completed; the requested self-contained Python utility did not. This run overlapped a one-job, BelowNormal Windows ROCm 7.2 CPU compilation after approximately the first half of the requests, so its elapsed times are not an uncontended SDK speed comparison. Evidence: `/home/zetion/code/kvmem-rocm10-beta2/logs/rocm10-q2k-fixed-mtp2-budget28672-bounded33-20260924-113939/evidence` and `E:\Codex_work\rocm10-mtp2-budget28672-perf33-20260924-113938.gpu-memory.csv`.

The Windows 14-target ROCm10 server, CLI and benchmark compiled. Security software initially removed executables on launch; they were subsequently restored and the stale benchmark relinked. Native verification and the completed same-source7.2 control are recorded above. The model-free build planning tests passed11/11.

## 2026-09-24 lower GPU KV budget diagnostic

On the same RX 9060 XT, current-source WSL ROCm 10 MTP3 at a 28672-token GPU KV budget completed seven diagnostic tool rounds. The 57,539-token request ran at **127.05 prompt t/s** with 8170 query-replay rows; the earlier ROCm 10 run at a 36864-token budget ran at 21.21 prompt t/s. The same-current-source ROCm 7.2 reference at 36864 ran at 108.25 prompt t/s. The Windows GPU Adapter Memory counter measured dedicated/shared maxima of 14967/890 MiB in the reduced-budget run, versus 15352/1629 MiB in the original ROCm 10 run. The lower budget causes replay to begin one request earlier, so these are configuration comparisons, not an equal-configuration SDK speed claim. The IQ3 launchers now default to 28672 with an explicit override.

The full reduced-budget ROCm 10 run completed 33 HTTP200 requests and 32 tool rounds. Final prompt was 261546/262144; final prompt evaluation was 126.91 t/s. Runtime RSS peaked at 13821.55 MiB. Windows GPU Adapter Memory dedicated/shared maxima were 15027/1058 MiB. The model generated only **154/512** final tokens and emitted a literal tool call instead of the requested self-contained Python utility. The canary exited 1; this is a structural/context-length success but **not** a full benchmark acceptance. Actual final prompt plus generation was 261700/262144, not 262058. `scripts/rocm/verify_long_context.py` reports structural and final-length results separately. Full-run evidence: `/home/zetion/code/kvmem-rocm10-beta2/logs/rocm10-q2k-fixed-mtp3-budget28672-bounded33-20260924-100912/evidence` and `E:\Codex_work\rocm10-mtp3-budget28672-perf33-20260924-100912.gpu-memory.csv`.

The matched same-source/same-budget ROCm 7.2 MTP3 control completed 33/33 requests, final prompt 261546/262144 and 512/512 completion tokens. Total request time was 2042.42 s versus ROCm 10's 2071.96 s, a 1.45% difference; median ROCm10/7.2 tool-request time ratio was 1.019. ROCm 7.2 runtime RSS peaked at 13691.25 MiB, with Windows GPU dedicated/shared maxima 15082/1072 MiB. Its final answer began a Python utility but was truncated at `from __future__` by the 512-token limit, so **it also did not deliver a complete self-contained utility**. Evidence: `/home/zetion/code/kvmem-rocm10-beta2/logs/rocm72-samesource-mtp3-budget28672-bounded33-20260924-105301/evidence` and `E:\Codex_work\rocm72-samesource-mtp3-budget28672-perf33-20260924-105300.gpu-memory.csv`. The severe 57K-token ROCm 10 slowdown is absent at the reduced GPU KV budget; this result does not establish a general SDK speed advantage.

Across the 33 requests, the server evaluated 261578 new prompt tokens in 1878.83 s (**139.22 effective prefill t/s**) and generated 3057 tokens in 174.53 s (**17.52 decode t/s**). First-pass time summed to 1059.40 s (246.91 new-prompt tokens/s), replay to 775.69 s, and retrieval to 17.61 s; 27 query-replay rounds replayed 220347 rows. MTP accepted/drafted counts were not emitted by this binary, so no acceptance percentage is reported. A lightweight `KVMEM_PERF`-gated counter was added for future runs.

The earlier Windows ROCm 7.2 MTP3 runs under `D:\code\kvmem-rocm-unified\logs\task2-iq3-hip-mtp3-memory-none` and `...-memory-rerun` also ended with a literal tool call and `finish_reason=stop`. The earlier ROCm 7.2 Windows MTP2 runs `task2-iq3-hip` and `task2-iq3-hip-trace` generated 512 tokens but returned a batch/checksum table, **not** the requested self-contained Python utility. The prior RX 7900 XTX ROCm 7.14.1 PR #33 validation ended final generation at 232/512 tokens. A 512-token length pass alone is not semantic acceptance; the matched WSL 7.2 control shows the same incomplete-code outcome, while its different final text prevents assigning every output difference to one cause.

## Q2_K prefill regression and local fix (2026-09-24)

The ROCm 10/HIP 7.15 compiler spilled 491 VGPRs in the gfx1200 Q2_K MMQ J=64 kernel, adding 1496 bytes of per-thread scratch. With the same Q2_K source, ROCm 7.2 generated a spill-free kernel. A CPU-checked, exact-shape `MUL_MAT` measurement (m=12288, n=512, k=5120) took approximately 44 ms under the original ROCm 10 build and 5 ms when only the Q2_K translation unit was built with `-mllvm --unroll-threshold=50`. The build now applies this flag only to `mmq-instance-q2_k.cu` when HIP is at least 7.15; the option `GGML_HIP_Q2_K_UNROLL_WORKAROUND` can disable it.

The 14-target builds now scope that compiler option to gfx1200/gfx1201 within the Q2_K translation unit. The rebuilt WSL gfx1200 Q2_K J=64 kernel has 145 VGPRs, zero spills, and zero private scratch. Four nonzero CPU-reference matrix tests passed, including m=12288, n=512, k=5120. The same RX 9060 XT and IQ3 model measured **519.35 t/s pp8192** before architecture scoping and **465.72 t/s pp8192** after scoping, with `ROCBLAS_USE_HIPBLASLT=0`, batch/ubatch 512, q8_0 KV, and `--load-mode none`; the original ROCm 10 result was approximately 331–335 t/s. The scoped run measured 12.10 t/s tg32. Both Windows and WSL scoped 14-target builds completed. WSL CTest before scoping: 12 passed, 1 skipped because that test requires a model argument. Evidence: `/home/zetion/code/kvmem-rocm10-beta2/logs/wsl-q2k-tile-validation-20260924-085610` and `/home/zetion/code/kvmem-rocm10-beta2/logs/wsl-formal-q2k-bench-20260924-085631`.

At the original 36864 GPU KV budget, ROCm 10 slowed at the 57,539-token request: 21.67 t/s prompt evaluation versus 108.28 t/s in a same-day ROCm 7.2 WSL MTP3 control. A ROCm 10 MTP2 diagnostic also slowed in this region. Reducing the budget to 28672 restored the long-context request trend; the matched full-run measurements are above. No PR, push, or release was made for this fix.

Test machine: Windows 11 with RX 9060 XT (gfx1200, 16 GiB); AMD display driver `32.0.31021.6002`. WSL: Ubuntu 24.04 on the same machine. The model is `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12,120,016,960 bytes), with `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` (347,213,792 bytes).

| Check | Result |
| --- | --- |
| Windows earlier ROCm 10.0.0 four-target build, gfx1100/gfx1151/gfx1200/gfx1201 | Build completed; CTest 9/9 passed before the final 14-target rebuild. |
| Windows final 14-target common `llama-kvmem-server.exe` code objects | `llvm-readobj --offloading` lists all 14 requested gfx architectures in the final EXE; the Q2_K MMQ and KVMem stage-in object files list the same 14 targets. This checks compiled device images, not runtime behavior on unowned GPUs. |
| Windows final 14-target CTest | Blocked locally: security software removed two test EXEs on launch. The final build was completed, but a fresh 9/9 result is not available. |
| WSL ROCm 10.0.0 gfx120X, gfx1200 | Build completed; CTest 12 passed, 1 skipped (model argument required) |
| WSL ROCm 10.0.0 multiarch, 14 common GPU targets | Build completed [600/600]; CTest 12 passed, 1 skipped (model argument required) |
| WSL final 14-target code objects | `llvm-readobj --offloading` lists all 14 requested gfx targets in `libggml-hip.so.0.22.0` and the KVMem stage-in object. Runtime hardware testing remains limited to the local gfx1200. |
| Extracted local WSL 14-target package on RX 9060 XT | GPU enumeration succeeded; IQ3 pp128 190.76 t/s, tg16 12.62 t/s; exit 0 |
| Windows, `ROCBLAS_USE_HIPBLASLT=0`, `llama-bench -ngl 99 -p 128 -n 16` | pp128 247.64 t/s, tg16 13.85 t/s, exit 0 |
| WSL, `llama-bench -ngl 99 -p 128 -n 128` | pp128 218.78 t/s, tg128 12.94 t/s, exit 0 |
| Windows IQ3 image canary, MTP2 replay, q8_0 KV, `--load-mode none` | Image request HTTP 200; red square, blue circle, green triangle identified |
| Windows `start-iq3.ps1` automatic selection | Selected physical GPU 0/gfx1200, set rocBLAS backend, server `/health` returned HTTP 200 |
| Windows ROCm 7.2, matched `llama-bench` pp8192/tg32, `--load-mode none` | 542.59 / 15.59 t/s; with `ROCBLAS_USE_HIPBLASLT=0`: 514.97 / 15.57 t/s |
| Windows ROCm 10, matched `llama-bench` pp8192/tg32, `--load-mode none` | With `ROCBLAS_USE_HIPBLASLT=0`: 327.52 / 15.38 t/s; default backend aborted before a result |
| Windows standalone BF16 GEMM (M=48, N=512, K=5120), default backend | ROCm 7.2: 0.0489 ms/call; ROCm 10: 0.2868–0.3088 ms/call |
| WSL same BF16 GEMM, same GPU and default backend | ROCm 7.2: 0.0576–0.0578 ms/call; ROCm 10: 0.0671–0.0682 ms/call |
| WSL matched model/command and CPU build flags, ROCm 7.2 versus 10 | pp128 391.54 versus 208.93 t/s; pp8192 496.80 versus 335.20 t/s; all cases exited 0 |

The Windows RDNA4 launcher selects the rocBLAS GEMM backend automatically. Without `ROCBLAS_USE_HIPBLASLT=0`, the same Windows ROCm 10 build failed during `MUL_MAT`; a plain `llama-bench` also reproduced the failure. The WSL ROCm 10 build ran without this setting. Other GPU architectures were compiled but not hardware-tested here.

The matched short tests use the same RX 9060 XT, model, llama.cpp submodule commit, offload, KV type, load mode, prompt and generation lengths. Logs: `logs/rocm10-short-compare-20260924`. The ROCm 10 prefill slowdown persists with the game closed and without KVMem replay. The AMD display driver installed for these tests predates the version in AMD's ROCm 10 compatibility matrix. The gfx1200-only build completed and CTest passed 9/9. Its matched short result was pp8192 328.58 t/s and tg32 15.88 t/s (`logs/rocm10-short-diagnostic-20260924`), essentially identical to the multi-architecture result. Disabling HIP graphs gave pp8192 328.23 t/s and tg32 15.65 t/s. ROCm 10's default rocBLAS path reported `hipBLASLt execution failed` / `rocblas_status_internal_error` and aborted; forcing Tensile lets the benchmark finish. A separate representative rocBLAS GEMM probe (`E:\Codex_work\rocm_gemm_probe.cpp`, F16 17408×128×5120) measured 7.2 at 4.99–5.04 ms and 10.0 at 4.86 ms per GEMM. A wider GEMM probe produced no result; Windows became unresponsive and blue-screened with a 0x9F driver power-state failure. Windows Error Reporting named Intel `iaLPSS2_I2C_MTL_S.sys` in the crash bucket. Do not repeat the wide probe.

The BF16 512-batch result comes from the same standalone source, not KVMem. The Windows tests were repeated in 7.2/10/10/7.2 order (`logs/small-bf16-compare-20260924-031001`). Forcing Tensile made both Windows SDKs about 0.31 ms/call; explicitly preferring hipBLASLt left ROCm 10 around 0.29 ms/call. Under WSL, both SDKs remained close to 0.06 ms/call (`logs/wsl-small-bf16-compare-20260924-032418`). The Windows ROCm 10 single-arch and multiarch distributions contain byte-identical runtime DLLs, gfx1200 BLAS kernel pack, and hipBLASLt gfx1200 library files. The remaining high-priority comparison is the same test after updating the Windows AMD driver to the ROCm 10 compatibility-matrix version; the signed installer is downloaded but has not been installed.

The first same-model WSL comparison also found ROCm 10 slower at the whole-model level (`/home/zetion/code/kvmem-rocm10-beta2/logs/wsl-rocm-sdk-compare-20260924-034423`). Both runs used the same GPU, model, runtime flags, and forced rocBLAS backend, and all four cases exited 0. The 7.2 executable was built with `GGML_NATIVE` and `GGML_OPENMP` enabled; the ROCm 10 executable initially had both disabled. The matched-flags repeat below resolves that confound. The isolated BF16 result remains a clean same-source comparison of that one operation.

A second WSL run rebuilt the ROCm 10 single-gfx1200 diagnostic binary with both CPU flags enabled, matching 7.2 (`/home/zetion/code/kvmem-rocm10-beta2/logs/wsl-rocm-sdk-compare-20260924-035227`). Prefill remained slower: pp128 391.54 versus 208.93 t/s and pp8192 496.80 versus 335.20 t/s (7.2 versus 10). The GGML GPU source directories have no content differences after ignoring line endings; both llama.cpp submodules have the same revision. The overall KVMem patches differ because of newer upstream integration, so the whole-model benchmark is not a pure SDK comparison.

Running the matched pp8192 WSL case in reverse order gave ROCm 10 first at 334.65 t/s and ROCm 7.2 second at 528.36 t/s (`/home/zetion/code/kvmem-rocm10-beta2/logs/wsl-rocm-sdk-compare-20260924-040610`). The ROCm 10 result stayed near 335 t/s, ruling out the earlier 7.2-first ordering as the main explanation.

A diagnostic WSL ROCm 10 rebuild with `GGML_HIP_MMQ_MFMA=OFF` completed 198/198. Its pp8192 result remained 337.19 t/s versus 530.06 t/s for unchanged ROCm 7.2 (`/home/zetion/code/kvmem-rocm10-beta2/logs/wsl-rocm-sdk-compare-20260924-042125`). With each SDK's default BLAS selection, ROCm 10 was 342.81 t/s and 7.2 was 554.36 t/s (`...-042411`). Neither disabling MFMA nor allowing default hipBLASLt selection restores ROCm 10 prefill throughput in this WSL full-model test. The MFMA-off build is diagnostic only; the separate Linux common-target preview package retains the original build options.

The WSL IQ3_S code objects show register spills in ROCm 10's `J=32` MMQ variants and none in ROCm 7.2. That difference does not account for the measured batch-512 prefill regression: the RX 9060 XT reports 65,536 bytes of per-block shared memory, the IQ3_S `J=128` tile requires 57,856 bytes, and the dispatch loop therefore selects `J=128` for a 512-token MMQ batch. Both SDKs' `J=128` variants report zero spills. A batch-size sweep is prepared but has not run yet.

A diagnostic `GGML_CUDA_FORCE_CUBLAS` build bypassed MMQ for IQ3_S. Its pp128/tg16 result was 76.07/14.93 t/s, slower in prefill than the default-path 247.64/13.85 t/s short run. The pp8192/tg16 diagnostic did not complete after 137.5 seconds and was stopped. This is not a usable workaround. Original build artifacts were restored from backups after the test; no source edit was made. Evidence: `logs/rocm10-force-blas-20260924`.

The original 36864-budget ROCm 10 long-context run was stopped after a slow partial trend. The later 28672-budget run completed 33 requests; its final generated answer did not pass the semantic acceptance check.

The local Linux common-target preview archive is `artifacts/kvmem-rocm10-beta2-linux-common-local-preview.tar.gz` (642,966,987 bytes). Its archive checksum, internal `SHA256SUMS`, and full tar read passed. `BUILD-INFO.json` records 14 compiled GPU targets, ROCm 10.0.0, merged upstream source commit `04878610a713d116cb525aaa749e2bb5fe08d59c`, and `source_worktree_dirty=true`. Static full/light UI assets were taken from the existing build of the same pinned llama.cpp revision. Only gfx1200 was hardware-tested here.
