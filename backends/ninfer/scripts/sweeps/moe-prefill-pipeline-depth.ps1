# Where does the routed prefill gate/up Spread/Packed crossing actually sit on this card?
#
# TODO section 2c: sparse_moe_prefill_kernels.cu's kGateUpDeepJobsNum/Den (7/4) picks between a
# 2-stage "Spread" and a 6-stage "Packed" pipeline for the narrow routed gate/up kernel, based on
# jobs-per-expert crossing that ratio. The constant is upstream's, measured on an RTX 5090 whose L2
# is 16x this card's (96 MB against 6 MB), and the crossing depends on L2 size. Wrong constant means
# the wrong pipeline depth, not wrong output -- a performance risk, not a correctness one.
#
# **This sweeps the constant through the product, which is the method that source comment endorses
# over its own operator fixture.** The comment is explicit that the two disagree by about six times
# depending on L2 warmth, and that upstream picked 7/4 by sweeping the threshold through the server
# rather than by trusting the microbenchmark. So do the same thing here rather than porting the
# fixture: it needs no new code, and it measures the thing that matters.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\moe-prefill-pipeline-depth.ps1
#
# About 70 minutes: five ratios, each a rebuild of one .cu plus a link, then one 35B load and four
# prompt lengths. It edits the constant in place and restores it in a finally block, including on
# Ctrl+C -- check `git diff` afterwards anyway, because a constant left changed silently alters
# every later measurement.
#
# Read the prompt lengths separately. The comment says the effect is 0.10-0.17 points of server
# prefill "on prompts below the wide-plan bound" against +0.01-0.02 above it, and that null control
# is the point: a ratio that moves both is measuring something else.
#
# Two things about this box that can invalidate a run, both hit while writing this script:
#   * vmmemWSL can hold 27 GiB of the 64 GiB of host RAM, and the 35B artifact is 22.8 GB. Under
#     that pressure the machine pages and the numbers are noise. Guarded by host-memory.ps1, which
#     refuses below the artifact's size plus headroom -- 24.0 GiB for this 35B.
#   * the card drifts 3-5% between processes, and each ratio here is a separate process. Five
#     ratios spread over an hour is exactly the shape that drift corrupts, so the sweep repeats the
#     7/4 baseline as its LAST point as well as measuring it in sequence -- if the two 7/4 readings
#     disagree by more than the ratio spread, the run is void. See TODO section 3 on locking clocks.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
$bench    = '.\build-ninja\bench\ninfer_bench.exe'
$model    = "$modelDir\qwen3_6_35b_a3b.ninfer"
$source   = 'src\ops\sparse_moe\prefill\sparse_moe_prefill_kernels.cu'
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($p in @($bench, $model, $source)) { if (-not (Test-Path $p)) { throw "missing: $p" } }

# This was the only sweep here with a memory guard, and it is now the shared one -- every other
# script in this directory grew the same check. host-memory.ps1 sizes the requirement from the
# artifact instead of hardcoding 24 GiB, which comes out at exactly 24.0 GiB for this 35B and so
# reproduces the threshold this script used to carry inline.
Assert-NInferHostMemory -Artifacts @($model)

$original = Get-Content $source -Raw
# 7/4 appears twice: once in sequence, once repeated at the end as the drift control.
$ratios = @(
  @{ num = 3;  den = 2 }, @{ num = 25; den = 16 }, @{ num = 13; den = 8 },
  @{ num = 7;  den = 4 }, @{ num = 2;  den = 1 },  @{ num = 7;  den = 4 }
)

"ratio,num,den,pass,prompt,prefill_tok_s"
try {
  $pass = 0
  foreach ($r in $ratios) {
    $pass++
    $text = $original `
      -replace 'constexpr int kGateUpDeepJobsNum = \d+;', "constexpr int kGateUpDeepJobsNum = $($r.num);" `
      -replace 'constexpr int kGateUpDeepJobsDen = \d+;', "constexpr int kGateUpDeepJobsDen = $($r.den);"
    Set-Content -Path $source -Value $text -NoNewline
    $ratio = '{0:N4}' -f ($r.num / $r.den)

    cmd /c "call `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && cmake --build build-ninja --target ninfer_bench > `"$out\depth_build.log`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
      "$ratio,$($r.num),$($r.den),$pass,BUILD_FAILED,"
      Get-Content "$out\depth_build.log" -Tail 5 | ForEach-Object { "#   $_" }
      continue
    }

    $log = "$out\depth_$($r.num)_$($r.den)_$pass.log"
    & $bench --weights $model --kv-dtype int8 --max-ctx 16384 `
        -p 1024,2048,4096,8192 -r 3 --warmup 1 > $log 2>&1
    # Emit per row as it is parsed, so a long run shows progress instead of buffering to the end.
    Get-Content $log | Where-Object { $_ -match '^pp(\d+)\s' } | ForEach-Object {
      $f = ($_ -split '\s+') | Where-Object { $_ }
      "$ratio,$($r.num),$($r.den),$pass,$($f[0] -replace '^pp',''),$($f[3])"
    }
  }
}
finally {
  # Restore unconditionally. A constant left changed here silently alters every later measurement
  # in the repository, which is a far worse outcome than an incomplete sweep.
  Set-Content -Path $source -Value $original -NoNewline
  "# restored $source"
  git diff --stat -- $source | ForEach-Object { "#   $_" }
}
