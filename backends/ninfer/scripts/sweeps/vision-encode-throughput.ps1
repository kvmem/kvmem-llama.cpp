# What does Vision cost, in time?
#
# TODO section 2c: "Vision has essentially one performance number in the entire repository" -- one
# DFlash2 acceptance figure on the committed image fixture, and nothing about encode throughput,
# how it scales with resolution, or what overlay residency costs in time rather than in bytes.
# This measures those three.
#
# It works because apps/ninfer reports the Vision stage separately from text prefill:
#
#   generate    vision                          134 ms
#   generate    text prefill                    912 ms
#
# so the encode cost is directly readable rather than having to be subtracted out of a total. The
# `prompt tokens` line gives the token count the image turned into, and tokens/ms of vision stage
# is the throughput figure this entry wanted.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\vision-encode-throughput.ps1
#
# About 20 minutes. Prints CSV to stdout and leaves per-run logs under $NINFER_SWEEP_OUT.
#
# Two things to know before reading the output.
#
# **Resolutions are not free-form.** The preprocessor snaps an image to a multiple of the merged
# patch size, so 1000x1000 and 1024x1024 can produce the same token count. The sweep uses exact
# multiples of 112 (four merged 28px patches) so each step is a clean change in token count rather
# than a rounding artifact, and it prints the token count beside every row so a plateau is visible
# rather than mysterious.
#
# **The first run of any configuration pays one-off costs** -- weight load dominates wall time and
# the Vision tower's first use may allocate. Every point here runs the model fresh, so compare the
# `vision_ms` column across rows and ignore `total`; and the sweep takes two measured repetitions
# per point and reports both, because a single number here cannot be distinguished from a hiccup.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
$cli      = '.\build-ninja\apps\ninfer.exe'
$model    = "$modelDir\qwen3_8_27b.ninfer"
$source   = 'bench\fixtures\ttft\media\load_00.png'
New-Item -ItemType Directory -Force -Path "$out\vision" | Out-Null
# Host memory pressure produces plausible numbers rather than an error, so it is guarded
# rather than trusted -- see host-memory.ps1 for what goes wrong and why.
Assert-NInferHostMemory -Artifacts @("$modelDir\qwen3_8_27b.ninfer")
foreach ($p in @($cli, $model, $source)) {
  if (-not (Test-Path $p)) { throw "missing: $p" }
}

# Resize the committed fixture rather than generating synthetic noise: a real photograph exercises
# the encoder's arithmetic the way production does, and flat or random images can both be
# unrepresentative (one compresses trivially, the other defeats every cache).
$py = @'
import sys
from PIL import Image
src, dst, side = sys.argv[1], sys.argv[2], int(sys.argv[3])
Image.open(src).convert("RGB").resize((side, side), Image.LANCZOS).save(dst)
'@
$pyPath = Join-Path $out 'vision\resize.py'
Set-Content -Path $pyPath -Value $py -Encoding UTF8

$sides = @(224, 448, 672, 896, 1120, 1344, 1568)
foreach ($side in $sides) {
  $dst = "$out\vision\img_$side.png"
  if (-not (Test-Path $dst)) { python $pyPath $source $dst $side }
}

function Invoke-Point {
  param([string]$Label, [string]$MessagesPath, [string[]]$Extra, [int]$Rep)
  $log = "$out\vision\$Label`_$Rep.log"
  & $cli $model --messages $MessagesPath --max-new 8 --max-context 8192 `
      --kv-dtype int8 --no-thinking @Extra > "$log.txt" 2> $log
  if ($LASTEXITCODE -ne 0) { return $null }
  $t = Get-Content $log -Raw
  $ms = { param($n)
    if ($t -match "generate\s+$n\s+([\d.]+)\s*(ms|s)\b") {
      if ($Matches[2] -eq 's') { [double]$Matches[1] * 1000 } else { [double]$Matches[1] }
    } else { $null } }
  [pscustomobject]@{
    vision_ms  = & $ms 'vision'
    prefill_ms = & $ms 'text prefill'
    tokens     = if ($t -match 'prompt tokens\s+(\d+)') { [int]$Matches[1] } else { $null }
    runtime    = if ($t -match 'runtime reservation\s+([\d.]+)\s*(\w+)') { "$($Matches[1]) $($Matches[2])" } else { '' }
  }
}

# One image, sweeping resolution, at both residencies. Residency is the "what does overlay cost in
# time" question: it keeps the Vision tower in host memory and borrows device memory per image, so
# the tower has to cross PCIe on every request and the cost should land in the vision stage.
"config,side,rep,image_tokens,vision_ms,text_prefill_ms,tokens_per_ms,runtime_reservation"
foreach ($residency in @('resident', 'overlay')) {
  foreach ($side in $sides) {
    $msgs = "$out\vision\msg_$side.json"
    $abs  = (Resolve-Path "$out\vision\img_$side.png").Path
    $payload = @{ role = 'user'; content = @(
        @{ type = 'image'; image = $abs },
        @{ type = 'text';  text  = 'Describe this image in one sentence.' }) }
    ConvertTo-Json @($payload) -Depth 6 | Set-Content -Path $msgs -Encoding UTF8
    for ($rep = 1; $rep -le 2; $rep++) {
      $r = Invoke-Point -Label "r$residency`_$side" -MessagesPath $msgs `
          -Extra @('--vision', '--vision-residency', $residency) -Rep $rep
      if ($null -eq $r) { "$residency,$side,$rep,,FAILED,,,"; continue }
      $tpm = if ($r.vision_ms -and $r.vision_ms -gt 0) { '{0:N2}' -f ($r.tokens / $r.vision_ms) } else { '' }
      "$residency,$side,$rep,$($r.tokens),$($r.vision_ms),$($r.prefill_ms),$tpm,$($r.runtime)"
    }
  }
}

# Several images in one request, at the resolution the fixtures ship at. Encode cost should be
# linear in images if the tower is the bottleneck and sublinear if per-request overhead dominates;
# either answer is worth having, and neither is recorded anywhere.
foreach ($count in @(1, 2, 4)) {
  $msgs = "$out\vision\msg_multi_$count.json"
  $parts = @()
  for ($i = 0; $i -lt $count; $i++) {
    $abs = (Resolve-Path ("bench\fixtures\ttft\media\load_{0:d2}.png" -f $i)).Path
    $parts += @{ type = 'image'; image = $abs }
  }
  $parts += @{ type = 'text'; text = 'Describe these images in one sentence.' }
  ConvertTo-Json @(@{ role = 'user'; content = $parts }) -Depth 6 |
      Set-Content -Path $msgs -Encoding UTF8
  for ($rep = 1; $rep -le 2; $rep++) {
    $r = Invoke-Point -Label "multi$count" -MessagesPath $msgs `
        -Extra @('--vision', '--vision-residency', 'overlay') -Rep $rep
    if ($null -eq $r) { "multi-$count,1024,$rep,,FAILED,,,"; continue }
    $tpm = if ($r.vision_ms -and $r.vision_ms -gt 0) { '{0:N2}' -f ($r.tokens / $r.vision_ms) } else { '' }
    "multi-$count,1024,$rep,$($r.tokens),$($r.vision_ms),$($r.prefill_ms),$tpm,$($r.runtime)"
  }
}
