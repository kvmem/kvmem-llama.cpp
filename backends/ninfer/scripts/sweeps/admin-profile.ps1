# Everything in this repository that needs an elevated shell, in one run.
#
# THIS MUST BE RUN AS ADMINISTRATOR. Open Windows Terminal or PowerShell with "Run as
# administrator" and run it from the repository root:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\admin-profile.ps1
#
# It needs admin for exactly two reasons, both verified on this box on 2026-09-09:
#
#   * ncu fails with ERR_NVGPUCTRPERM as a normal user. GPU performance counters are
#     administrator-only by default on Windows. That permission is satisfied by EITHER running
#     elevated OR setting HKLM\...\NvTweak\RmProfilingAdminOnly = 0 and rebooting. Running
#     elevated is strictly better: nothing persists, and no reboot.
#   * nvidia-smi -pl and -lgc are refused with "Insufficient Permissions" as a normal user.
#
# It changes two pieces of GPU state -- the power limit and the clock lock -- and restores both in
# a finally block, including on Ctrl+C. Nothing it does survives the script.
#
# Two things will make it fail on this box, both hit on 2026-09-09 and both guarded below.
#
#   * ncu's default kernel replay saves and restores the device memory a kernel could touch, and
#     with 21 GB of weights resident there is no room on the card, so it spills the backup to host
#     RAM. On top of the artifact's own host footprint that is ~42 GB, which does not fit beside
#     vmmemWSL. The failure is "==WARNING== Backing up device memory in system memory" followed by
#     "Unhandled C++ exception: bad allocation" and an empty report -- which reads like an ncu bug
#     and is not one. --replay-mode application re-runs the application once per pass instead and
#     needs no backup at all; it is slower (a full model load per pass) and it is the only mode
#     that works here on the 35B. See $ReplayMode below.
#   * host memory pressure. Guarded by host-memory.ps1, with the requirement raised because an ncu
#     run holds more than a plain bench run does.
#
# Results land under profiles\admin\ as text files. Nothing here needs interpreting live; the
# point is to capture what an unelevated session cannot.
[CmdletBinding()]
param(
  # Resolved below rather than here: $PSScriptRoot is empty while a param default is being bound,
  # so a repo-relative default written in this block silently resolves against the drive root.
  [string]$ModelDir = $env:NINFER_MODEL_DIR,
  [string]$Ncu      = $(if ($env:NINFER_NCU) { $env:NINFER_NCU } else { 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\ncu.bat' }),
  # 'application' is the default deliberately -- see the header. 'kernel' is ncu's own default and
  # is faster, but on this card it only survives on a model small enough that the save-and-restore
  # buffer fits in the ~3 GB of device memory the weights leave free.
  [ValidateSet('application', 'kernel')][string]$ReplayMode = 'application',
  # Which sections to run, so a forty-minute capture can be resumed rather than restarted. The
  # sections are independent: each opens its own process and writes its own file, and none reads
  # another's output. 1-4 load an artifact and are the slow ones; 5 profiles the schedule benches
  # and takes seconds.
  #
  # A string rather than an [int[]], because the invocation this script's own header documents is
  # `powershell -File`, and through that `-Sections 2,3,4,5` arrives as the single token "2345" and
  # fails ValidateRange with a message about the number two thousand three hundred and forty-five.
  # Splitting it here means one spelling works both through -File and when dot-sourced.
  [string]$Sections = '1,2,3,4,5,6',
  [switch]$SkipPower
)

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
if (-not $ModelDir) { $ModelDir = Get-NInferModelDir }
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Error 'Not elevated. Re-run this from an administrator shell -- everything here needs it.'
  exit 1
}

$out = 'profiles\admin'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$bench = '.\build-ninja\bench\ninfer_bench.exe'
$moe   = "$ModelDir\qwen3_6_35b_a3b.ninfer"
$dense = "$ModelDir\qwen3_8_27b.ninfer"
if (-not (Test-Path $Ncu)) { Write-Error "ncu not found at $Ncu; set NINFER_NCU"; exit 1 }

# Parse -Sections BEFORE the preflight, because what this run needs depends on what it was asked
# for. Sections 1 and 4 profile the 35B, 2 and 3 the 27B, 5 uses the standalone Op benches and no
# artifact at all, and 6 only moves the power limit. Demanding both artifacts and 35B-sized RAM up
# front made `-Sections 5` fail on a box that simply has no 35B -- which is this box, and which is
# exactly the machine the narrow-extent entries in TODO section 2c want profiled.
$selected = @($Sections -split '[,;\s]+' | Where-Object { $_ -ne '' } | ForEach-Object {
  $n = 0
  if (-not [int]::TryParse($_, [ref]$n) -or $n -lt 1 -or $n -gt 6) {
    Write-Error "-Sections takes numbers 1..6 separated by commas; got '$_'"
    exit 1
  }
  $n
})
$wanted = [System.Collections.Generic.HashSet[int]]::new([int[]]$selected)
"sections: $($selected -join ', ')   replay mode: $ReplayMode"

$needsMoe   = $wanted.Contains(1) -or $wanted.Contains(4)
$needsDense = $wanted.Contains(2) -or $wanted.Contains(3)
$needed     = @()
if ($needsMoe -or $needsDense) { $needed += $bench }
if ($needsMoe)                 { $needed += $moe }
if ($needsDense)               { $needed += $dense }
foreach ($p in $needed) {
  if (-not (Test-Path $p)) { Write-Error "missing: $p (required by the selected sections)"; exit 1 }
}

# An ncu run is heavier on host RAM than the plain bench the shared guard is calibrated for: even
# under application replay the tool holds its own buffers alongside the artifact. Ask for 4 GiB
# above the usual headroom rather than discovering the shortfall as a bad_alloc ten minutes in.
# Size it on the largest artifact this run will actually open, and skip the guard entirely when no
# artifact is opened at all.
$guarded = @($needed | Where-Object { $_ -ne $bench })
if ($guarded.Count -gt 0) {
  $largest = ($guarded | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Maximum).Maximum
  Assert-NInferHostMemory -Artifacts $guarded -RequiredGiB (($largest / 1GB) + 6.8)
} else {
  "host-memory guard: skipped, the selected sections open no artifact"
}

# Remember what to put back. power.limit is the enforced ceiling; default_limit is the factory one.
$originalLimit = (nvidia-smi --query-gpu=power.limit --format=csv,noheader | Select-Object -First 1)
$originalLimit = [double]($originalLimit -replace '[^0-9.]', '')
"original power limit: $originalLimit W"
$clocksLocked = $false
$powerChanged = $false

try {
  # ---------------------------------------------------------------- 1. the MoE expert gather
  #
  # TODO section 2c: sparse_moe_d3/d4 reach 40-45% of the card's read bandwidth where contiguous
  # weight kernels on the same decode step reach 78-82%, and the four sparse_moe stages are 38% of
  # decode busy time on the 35B. Three candidate causes -- address divergence from gathering eight
  # scattered experts of 256, L2 behaviour, or too little work per CTA to cover latency -- and
  # nsys cannot distinguish them. These sections can.
  #
  # Clocks are locked first so the counters describe one clock state rather than an average of the
  # card ramping. --graph-profiling defaults to node, which matters because decode replays a
  # captured CUDA graph and the default in older tools records a replay as one opaque entity.
  "locking clocks for the counter run"
  nvidia-smi -lgc 1500 2>&1 | Out-String | Write-Host
  if ($LASTEXITCODE -eq 0) { $clocksLocked = $true }

  if ($wanted.Contains(1)) {
  "== 1/6 ncu: MoE expert gather (35B, int8, decode) -> $out\moe_gather.txt"
  & $Ncu --target-processes all --graph-profiling node --replay-mode $ReplayMode `
      -k 'regex:sparse_moe' --launch-skip 64 --launch-count 12 `
      --section SpeedOfLight --section MemoryWorkloadAnalysis `
      --section MemoryWorkloadAnalysis_Tables --section Occupancy --section LaunchStats `
      --section SchedulerStats --section WarpStateStats `
      $bench --weights $moe --kv-dtype int8 --max-ctx 8192 -n 64 -r 1 --warmup 1 `
      > "$out\moe_gather.txt" 2>&1
  "  ncu exit $LASTEXITCODE"
  if (Select-String -Path "$out\moe_gather.txt" -Pattern 'ERR_NVGPUCTRPERM' -Quiet) {
    "  STILL PERMISSION-DENIED -- the shell is elevated but the driver refused; the registry"
    "  route (RmProfilingAdminOnly=0 plus reboot) is then the only option."
  }
  }

  # A contiguous kernel from the same step, as the reference the percentages above are against.
  # Without it the MoE numbers have no local baseline and have to be compared across runs.
  #
  # One kernel pattern, not an alternation. The first version passed -k 'regex:a|b' and PowerShell
  # parsed the | as a pipeline separator before ncu ever saw it, so the invocation became
  # "ncu ... regex:a" piped into a command named b. It died with "'b' is not recognized as an
  # internal or external command" and wrote an empty report -- which reads like an ncu problem and
  # is not one. Single pattern, via --kernel-name.
  if ($wanted.Contains(2)) {
  "== 2/6 ncu: contiguous reference kernels (27B, int8, decode) -> $out\contiguous_ref.txt"
  & $Ncu --target-processes all --graph-profiling node --replay-mode $ReplayMode `
      --kernel-name 'regex:q5_rowsplit_gemv' --launch-skip 64 --launch-count 8 `
      --section SpeedOfLight --section MemoryWorkloadAnalysis `
      --section MemoryWorkloadAnalysis_Tables --section Occupancy --section LaunchStats `
      $bench --weights $dense --kv-dtype int8 --max-ctx 8192 -n 64 -r 1 --warmup 1 `
      > "$out\contiguous_ref.txt" 2>&1
  "  ncu exit $LASTEXITCODE"
  }

  # ---------------------------------------------------------------- 3. prefill's MLP GEMMs
  #
  # TODO section 2c: q4a8_swiglu reaches 96-97 TOPS against a measured 314.8 TOPS INT8 ceiling, so
  # 31%, where a well-tuned large GEMM on Ampere usually reaches 60-80%. Two explanations have
  # already been ruled out without counters -- it is not a skinny-tile artifact (chunk 256..4,096
  # moves prefill 1.0%) and it is not dequantization (unpacking every 4-bit code costs 1-3% of the
  # call even at 8 int ops per code). It is also not memory: at this shape the kernel is
  # compute-bound over its memory floor by 4.3x.
  #
  # So the gap is inside the MMA pipeline, and these sections are what distinguishes the remaining
  # candidates: issue rate, shared-memory feeding, and occupancy. The MLP pair is 68% of prefill
  # FLOPs, which makes it the largest compute-side opportunity in the file.
  if ($wanted.Contains(3)) {
  "== 3/6 ncu: prefill MLP GEMMs (27B, int8, prefill) -> $out\prefill_mlp.txt"
  & $Ncu --target-processes all --graph-profiling node --replay-mode $ReplayMode `
      --kernel-name 'regex:q4a8_swiglu' --launch-skip 8 --launch-count 6 `
      --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy `
      --section LaunchStats --section SchedulerStats --section WarpStateStats `
      --section ComputeWorkloadAnalysis --section InstructionStats `
      $bench --weights $dense --kv-dtype int8 --max-ctx 8192 -p 4096 -r 1 --warmup 1 `
      > "$out\prefill_mlp.txt" 2>&1
  "  ncu exit $LASTEXITCODE"
  }

  # ---------------------------------------------------------------- 4. the MoE d4 over-fetch
  #
  # TODO section 2c, item 4 of the MoE entry: ncu's MemoryWorkloadAnalysis reported 12.12 MB of DRAM
  # traffic on d3 and 12.27 MB on d4 where the artifact inventory attributes 8.91 MB and 5.58 MB of
  # useful weight bytes -- 1.36x and 2.20x. That entry explicitly asks for dram__bytes_read.sum
  # before anything is optimised against it, because the attribution and the section reading are
  # separate measurements and a 2.2x over-fetch would be worth more than everything else in the
  # entry.
  #
  # Metrics rather than sections, deliberately: a named metric list collects in one or two passes
  # where the seven sections of step 1 need twenty-odd, and under application replay every pass is
  # another full 22.8 GB model load. This is the cheap half of step 1 and it is the half that
  # answers a yes/no question.
  if ($wanted.Contains(4)) {
  "== 4/6 ncu: MoE DRAM bytes, d3 and d4 (35B, int8, decode) -> $out\moe_dram_bytes.txt"
  & $Ncu --target-processes all --graph-profiling node --replay-mode $ReplayMode `
      --kernel-name 'regex:sparse_moe_d[34]' --launch-skip 64 --launch-count 8 `
      --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum `
      $bench --weights $moe --kv-dtype int8 --max-ctx 8192 -n 64 -r 1 --warmup 1 `
      > "$out\moe_dram_bytes.txt" 2>&1
  "  ncu exit $LASTEXITCODE"
  }

  # ---------------------------------------------------------------- 5. narrow-extent weight streaming
  #
  # The largest un-attributed gap in TODO, and the one two separate entries converge on. §2c's
  # eight-lane entry has q5_linear_add's mma_r64_c16 sitting at 24% of what its weight costs to
  # stream once, flat from T=4 to T=16; §2c's DFlash2-cliff entry has the GDN input projection's
  # grouped tile at 27% of its own weight-streaming floor at any width. Both entries independently
  # conclude the same thing: the remaining 3.7-4x is NOT tile geometry. Narrowing the tile was built
  # twice -- an R64C8 for q5_linear_add, which lost 6-32%, and the GDN C8/C16 tiles, which won 11%
  # and left the floor exactly where it was. So the mechanism is something else, and nothing in the
  # file has counters on it.
  #
  # This profiles the SCHEDULE BENCHES rather than the product, which is the whole reason it is
  # cheap: they allocate their own weights and never load an artifact, so there is no 22.8 GB model
  # load per replay pass and kernel replay would work here even if the card were full. Every
  # schedule of the Op runs in one process at one width, so the fast and slow tiles are profiled
  # under identical conditions in the same run -- which is the comparison that matters.
  #
  # WarpStateStats is the section to read first. "Weight-read-bound at a quarter of bandwidth" has a
  # small number of possible signatures and they are distinguishable: stall_long_scoreboard
  # dominating means the loads are issued but not enough of them are in flight; stall_mio_throttle
  # means the LSU itself is the limit; a poor sectors-per-request in the memory tables means each
  # fetch is delivering a fraction of a sector's worth of useful bytes. Those want different fixes.
  # Both are programs in the ninfer_benches bundle; the first argument selects one.
  $benches = '.\build-ninja\bench\ninfer_benches.exe'
  $narrow = @(
    @{ label = 'gdn_input_w8'; exe = $benches; out = 'narrow_gdn_input.txt'
       # Width 8 is the C8 serving cohort's extent and one past the DFlash2 k=6 cliff, so it is the
       # width both entries care about. All six schedules run, so the independent SIMT route is
       # captured beside the grouped MMA tiles as its own reference.
       args = @('ninfer_q4_q5_gdn_input_schedule_bench', '--tokens', '8', '--repeat', '9',
                '--warmup', '2') }
    @{ label = 'q5_linear_add_t8'; exe = $benches; out = 'narrow_q5_linear_add.txt'
       # T=8 is where split2_exact still beats mma_r64_c16 (74.8 us against 101.4) despite c16
       # being the flat one. Profiling both at the same width is how the 24%-of-floor figure
       # acquires a cause instead of a magnitude.
       args = @('ninfer_q5_linear_add_schedule_bench', '--tokens', '8', '--repeat', '9',
                '--warmup', '2') }
  )
  $step = 0
  foreach ($n in $(if ($wanted.Contains(5)) { $narrow } else { @() })) {
    $step++
    if (-not (Test-Path $n.exe)) { "== 5/6.$step SKIP $($n.label): missing $($n.exe)"; continue }
    "== 5/6.$step ncu: $($n.label) -> $out\$($n.out)"
    & $Ncu --target-processes all --replay-mode $ReplayMode `
        --section SpeedOfLight --section MemoryWorkloadAnalysis `
        --section MemoryWorkloadAnalysis_Tables --section Occupancy --section LaunchStats `
        --section SchedulerStats --section WarpStateStats --section InstructionStats `
        $n.exe @($n.args) `
        > "$out\$($n.out)" 2>&1
    "  ncu exit $LASTEXITCODE"
  }

  # ---------------------------------------------------------------- 6. what the power cap costs
  #
  # TODO section 3: the 315 W cap binds in essentially every busy sample, but memory clock never
  # leaves 9,501 MHz, so the bandwidth figures are already unthrottled and the prediction is that
  # the missing 35 W buys very little. This is the measurement that settles it. Clocks are released
  # first -- a locked clock would hide exactly the effect being looked for.
  if ((-not $SkipPower) -and $wanted.Contains(6)) {
    if ($clocksLocked) { nvidia-smi -rgc 2>&1 | Out-Null; $clocksLocked = $false }
    "== 6/6 power limit 350 W -> $out\power_350.txt"
    nvidia-smi -pl 350 2>&1 | Out-String | Write-Host
    if ($LASTEXITCODE -eq 0) {
      $powerChanged = $true
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\power-and-clocks.ps1 `
          > "$out\power_350.txt" 2>&1
      Get-Content "$out\power_350.txt" | Select-String '^==|^  ' | ForEach-Object { "  $_" }
    } else {
      "  -pl 350 refused even elevated; skipping"
    }
  }
}
finally {
  # Restore unconditionally. A left-behind clock lock or power limit silently changes every
  # measurement taken afterwards, which is worse than not having run this at all.
  if ($clocksLocked) { "restoring clocks"; nvidia-smi -rgc 2>&1 | Out-Null }
  if ($powerChanged) {
    "restoring power limit to $originalLimit W"
    nvidia-smi -pl $originalLimit 2>&1 | Out-Null
  }
  nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv | ForEach-Object { "  $_" }
}

""
"== done. Wrote:"
Get-ChildItem $out -File | ForEach-Object { "   {0,10:N0} bytes  {1}" -f $_.Length, $_.FullName }
""
"The ncu files are the ones that matter; they are long, and are meant to be read by whoever asked"
"for them rather than skimmed here. Reading order, by how much is riding on each:"
"  narrow_*.txt        the largest un-attributed gap in TODO -- read WarpStateStats first"
"  moe_dram_bytes.txt  is d4's 2.2x over-fetch real? one number decides it"
"  prefill_mlp.txt     issue rate vs shared-memory feeding vs occupancy, on 68% of prefill FLOPs"
"  moe_gather.txt      the full section sweep the percentages in TODO come from"
