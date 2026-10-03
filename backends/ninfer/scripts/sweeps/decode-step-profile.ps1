# Where does a decode step's time actually go?
#
# TODO.md section 2c says decode reaches only part of the card's memory bandwidth and that the
# next step is "a profile of one decode step to find where the stall is -- occupancy, L2
# behaviour, or a launch gap between the per-layer kernels". This is that profile.
#
# It answers the cheapest question first, which is the launch-gap one: add up the time the GPU
# spends inside kernels during the captured window and compare it with the wall time of the
# window. Whatever is missing is the GPU idle between kernels, and no amount of kernel tuning
# recovers it. Then it ranks kernels by total time so the next question has somewhere to go.
#
# **--cuda-graph-trace node is not optional here.** Decode replays a captured CUDA graph, and
# nsys's default graph tracing records the replay as one opaque entity rather than its nodes. The
# per-kernel trace then contains only the handful of kernels launched outside the graph, and the
# arithmetic below reports 99.3% of the window as GPU idle -- which is an artifact of the
# instrument, not a launch gap. Ask for node-level tracing and the graph's kernels appear.
#
# ninfer_bench's --profile-measured brackets exactly one measured repetition with
# cudaProfilerStart/Stop, and nsys --capture-range=cudaProfilerApi records only that, so the
# report is one clean decode run rather than model loading and warmup.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\decode-step-profile.ps1
#
# A few minutes per configuration. Reports land under $NINFER_SWEEP_OUT as .nsys-rep plus the
# exported CSVs, which are the artifact -- the console summary below is a convenience.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null
# Host memory pressure produces plausible numbers rather than an error, so it is guarded
# rather than trusted -- see host-memory.ps1 for what goes wrong and why.
Assert-NInferHostMemory -Artifacts @(
    "$modelDir\qwen3_8_27b.ninfer",
    "$modelDir\qwen3_6_35b_a3b.ninfer")

$nsys = if ($env:NINFER_NSYS) { $env:NINFER_NSYS } else {
  'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe' }
if (-not (Test-Path $nsys)) { throw "nsys not found at $nsys; set NINFER_NSYS" }

# Two workloads per model, because the first attempt at this conflated them and the answer is
# different for each. `-pg 4096,128` prefills 4,096 tokens and then decodes 128, and the prefill
# dominates the capture completely -- the 27B's top kernel came back as 256 launches of
# q4a8_swiglu, which is 4 prefill chunks x 64 layers, and causal_attention_*prompt*_i8 sat in the
# top eight. None of that is decode. `-n 128` decodes only, on a shallow cache, which is the
# workload the roofline question is actually about: decode streams the resident weights whatever
# the cache depth, so a shallow cache is fine for asking where the time goes.
$configs = @(
  @{ key='27b-dense-decode';  path="$modelDir\qwen3_8_27b.ninfer";     args=@('-n','128') }
  @{ key='35b-moe-decode';    path="$modelDir\qwen3_6_35b_a3b.ninfer"; args=@('-n','128') }
  @{ key='27b-dense-prefill'; path="$modelDir\qwen3_8_27b.ninfer";     args=@('-pg','4096,128') }
  @{ key='35b-moe-prefill';   path="$modelDir\qwen3_6_35b_a3b.ninfer"; args=@('-pg','4096,128') }
)

foreach ($c in $configs) {
  if (-not (Test-Path $c.path)) { "$($c.key): MISSING $($c.path)"; continue }
  $stem = "$out\prof_$($c.key)"
  # Everything nsys can write for this stem, not just the two inputs to the next `nsys profile`
  # call. Without clearing the exported CSVs too, a later `nsys stats` failure (nonzero exit, no
  # new CSV written) leaves the previous run's files in place, `Test-Path $trace` finds them, and
  # this iteration reports someone else's numbers as its own.
  Remove-Item "$stem.nsys-rep","$stem.sqlite",`
      "$stem`_cuda_gpu_trace.csv","$stem`_cuda_gpu_kern_sum.csv" -ErrorAction SilentlyContinue

  & $nsys profile --force-overwrite true --output $stem `
      --trace cuda --cuda-graph-trace node --capture-range cudaProfilerApi --capture-range-end stop `
      .\build-ninja\bench\ninfer_bench.exe `
      --weights $c.path --kv-dtype int8 --max-ctx 8192 `
      @($c.args) -r 1 --warmup 1 --profile-measured `
      > "$stem.log" 2>&1
  if (-not (Test-Path "$stem.nsys-rep")) {
    "$($c.key): profile FAILED"
    Get-Content "$stem.log" -Tail 5 | ForEach-Object { "    $_" }
    continue
  }

  & $nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv `
      --output "$stem" "$stem.nsys-rep" > "$stem.stats.log" 2>&1
  if ($LASTEXITCODE -ne 0) {
    "$($c.key): nsys stats FAILED (exit $LASTEXITCODE)"
    Get-Content "$stem.stats.log" -Tail 5 | ForEach-Object { "    $_" }
    continue
  }

  $trace = "$stem`_cuda_gpu_trace.csv"
  $kern  = "$stem`_cuda_gpu_kern_sum.csv"
  if (-not (Test-Path $trace)) { "$($c.key): no trace CSV"; continue }

  # Busy time is the union of kernel intervals; wall time is first start to last end. The
  # difference is the GPU sitting idle waiting for the host to launch the next kernel, which on a
  # per-layer decode with dozens of small launches is the failure mode worth ruling out first.
  $rows = Import-Csv $trace | Where-Object { $_.'Start (ns)' -and $_.'Duration (ns)' }
  if (-not $rows) { "$($c.key): trace CSV had no kernel rows"; continue }
  $starts = $rows | ForEach-Object { [double]$_.'Start (ns)' }
  $ends   = $rows | ForEach-Object { [double]$_.'Start (ns)' + [double]$_.'Duration (ns)' }
  $wall   = ($ends | Measure-Object -Maximum).Maximum - ($starts | Measure-Object -Minimum).Minimum

  # Union of kernel intervals, not a sum of durations: decode launches on multiple streams, and
  # concurrent kernels overlap in time. Summing durations double-counts that overlap, which
  # understates idle time (or drives it negative) exactly when concurrency is doing its job.
  # Merge-sweep instead: sort by start, extend the current run while the next interval overlaps
  # or touches it, and bank the run's width once a gap opens.
  $intervals = for ($i = 0; $i -lt $rows.Count; $i++) {
    [pscustomobject]@{ Start = $starts[$i]; End = $ends[$i] }
  }
  $busy = 0.0
  $runStart = $null
  $runEnd   = $null
  foreach ($iv in ($intervals | Sort-Object Start)) {
    if ($null -eq $runStart) {
      $runStart = $iv.Start
      $runEnd   = $iv.End
    } elseif ($iv.Start -le $runEnd) {
      if ($iv.End -gt $runEnd) { $runEnd = $iv.End }
    } else {
      $busy    += $runEnd - $runStart
      $runStart = $iv.Start
      $runEnd   = $iv.End
    }
  }
  if ($null -ne $runStart) { $busy += $runEnd - $runStart }

  ""
  "== $($c.key), int8, one measured repetition =="
  "  kernel launches        : {0:N0}" -f $rows.Count
  "  GPU busy               : {0:N3} ms" -f ($busy / 1e6)
  "  window wall            : {0:N3} ms" -f ($wall / 1e6)
  "  idle between kernels   : {0:N3} ms  ({1:N1}% of the window)" -f `
      (($wall - $busy) / 1e6), (100 * ($wall - $busy) / $wall)
  if (Test-Path $kern) {
    "  top kernels by total time:"
    Import-Csv $kern | Select-Object -First 8 | ForEach-Object {
      "    {0,8:N2} ms  {1,6} x  {2}" -f ([double]$_.'Total Time (ns)' / 1e6), $_.Instances, $_.Name
    }
  }
}
"== done =="
