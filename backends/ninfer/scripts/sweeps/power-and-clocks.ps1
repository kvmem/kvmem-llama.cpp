# Does the 315 W cap bound the numbers in TODO.md?
#
# TODO section 3 asked what the power cap costs. This samples nvidia-smi through one decode and one
# prefill and reports what actually throttles. The answer as of 2026-09-09 on this box: the cap
# binds continuously (power pins within 1 W of the limit whenever the card is busy) but the memory
# clock never leaves 9,501 MHz in either phase, so every bandwidth figure in that file is measured
# at unthrottled memory. SM clock is what pays -- about 11% below rated boost on decode, 3% on
# prefill.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\power-and-clocks.ps1
#
# Read the two phases separately. Decode and prefill sit at different clocks for the same power,
# and averaging them hides exactly the thing worth seeing. Rows with utilization <= 50% are dropped
# because they are model loading and process teardown, not the workload.
#
# The residual question -- what the missing 35 W would buy -- needs `nvidia-smi -pl 350`, which
# requires an elevated shell. Without one this script tells you what the cap is doing, not what
# lifting it would do. Re-run it after `-pl 350` to answer that.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
$bench    = '.\build-ninja\bench\ninfer_bench.exe'
$model    = "$modelDir\qwen3_8_27b.ninfer"
New-Item -ItemType Directory -Force -Path $out | Out-Null
# Host memory pressure produces plausible numbers rather than an error, so it is guarded
# rather than trusted -- see host-memory.ps1 for what goes wrong and why.
Assert-NInferHostMemory -Artifacts @("$modelDir\qwen3_8_27b.ninfer")
if (-not (Test-Path $model)) { throw "model not found: $model" }

nvidia-smi --query-gpu=power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.memory `
    --format=csv | ForEach-Object { "# $_" }

$fields = 'power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,' +
          'clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_thermal_slowdown,' +
          'clocks_throttle_reasons.sw_thermal_slowdown'

$phases = @(
  @{ key = 'decode';  args = @('-n','1024','-r','2') }
  @{ key = 'prefill'; args = @('-p','8192','-r','3') }
)

foreach ($phase in $phases) {
  $csv = "$out\power_$($phase.key).csv"
  # Start the sampler first: the interesting samples are the ones under load, and nvidia-smi's own
  # first reading lags by up to a second.
  $smi = Start-Process -FilePath 'nvidia-smi' -PassThru -NoNewWindow -RedirectStandardOutput $csv `
      -ArgumentList '--query-gpu',$fields,'--format=csv,noheader','-lms','500'
  try {
    & $bench --weights $model --kv-dtype int8 --max-ctx 8192 @($phase.args) --warmup 1 2>&1 |
        Where-Object { $_ -match '^(pp|tg|test)' }
  } finally {
    try { $smi.Kill(); $smi.WaitForExit(5000) } catch {}
  }

  $rows = @(Import-Csv $csv -Header 'pw','sm','mem','t','util','pcap','hwt','swt' |
      Where-Object { $_.util -and [int]($_.util -replace '[^0-9]','') -gt 50 })
  if (-not $rows) { "$($phase.key): no busy samples"; continue }
  $num = { param($v) [double](($v -replace '[^0-9.]','')) }
  $pw  = $rows | ForEach-Object { & $num $_.pw }
  $sm  = $rows | ForEach-Object { & $num $_.sm }
  $mem = $rows | ForEach-Object { & $num $_.mem }
  $tp  = $rows | ForEach-Object { & $num $_.t }
  $cap = ($rows | Where-Object { $_.pcap -match 'Active' -and $_.pcap -notmatch 'Not Active' }).Count
  $thm = ($rows | Where-Object { ($_.hwt -match 'Active' -and $_.hwt -notmatch 'Not Active') -or
                                 ($_.swt -match 'Active' -and $_.swt -notmatch 'Not Active') }).Count
  # A real median, not Measure-Object -Average: the ramp-up samples are outliers at both ends
  # (36 W idle at one end, a 1,905 MHz boost spike before power settles at the other) and a mean
  # reported as a median is how a 1,905 MHz reading ends up quoted as a steady clock.
  $median = { param($v) $sorted = @($v | Sort-Object); $sorted[[int]($sorted.Count / 2)] }
  $stat = { param($v) "{0:N0} median ({1:N0}-{2:N0})" -f `
      (& $median $v), (($v | Measure-Object -Minimum).Minimum),
      (($v | Measure-Object -Maximum).Maximum) }
  ""
  "== $($phase.key), 27B int8, $($rows.Count) busy samples =="
  "  board power       : $(& $stat $pw) W"
  "  sw_power_cap      : $cap of $($rows.Count) samples active"
  "  thermal throttle  : $thm of $($rows.Count) samples active"
  "  SM clock          : $(& $stat $sm) MHz"
  "  memory clock      : $(& $stat $mem) MHz"
  "  temperature       : $(& $stat $tp) C"
}
""
"== done; per-sample CSVs under $out =="
