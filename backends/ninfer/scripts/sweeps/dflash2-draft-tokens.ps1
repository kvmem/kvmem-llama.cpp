# Is DFlash2 on text a loss at every draft count, or only at seven?
#
# TODO section 2c records DFlash2 costing 20-22% against no speculation on the 27B, measured with
# --draft-tokens 7 throughout because docs/cli.md calls seven "the checkpoint recommendation". The
# arithmetic says why: 20% acceptance over seven drafts yields ~2.4 tokens per round while each
# round verifies eight columns instead of one. Fewer drafts means a cheaper round for a
# proportionally smaller gain, so there may be a crossover below seven -- and nobody had looked.
#
# The engine accepts 1..15 for DFlash and DFlash2 (src/product/speculative_options.h), so this
# sweeps past seven as well: if the curve is still rising at seven the recommendation is wrong in
# the other direction.
#
# Measured at depth. A plain tg128 run seeds ~128 tokens of cache and tells you nothing about the
# regime anyone serves in, so every point here prefills 4,096 tokens first and times 128 decode
# steps on top of that cache (-pg 4096,128, prefill and decode timed separately).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\dflash2-draft-tokens.ps1
#
# About 45 minutes. Read the CSVs it leaves under $NINFER_SWEEP_OUT, not just this summary.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null
# Host memory pressure produces plausible numbers rather than an error, so it is guarded
# rather than trusted -- see host-memory.ps1 for what goes wrong and why.
Assert-NInferHostMemory -Artifacts @("$modelDir\qwen3_8_27b_dflash2.ninfer")

# DFlash2 exists only in this artifact. It is a different file from the qwen3_8_27b.ninfer the rest
# of the 27B numbers come from, which is why the no-speculation baseline below is measured on this
# same file rather than borrowed from another sweep: a cross-artifact comparison would be measuring
# the bundle as well as the backend.
$weights = "$modelDir\qwen3_8_27b_dflash2.ninfer"
if (-not (Test-Path $weights)) { throw "Missing DFlash2 artifact: $weights" }

$depth = 4096
$configs = @( @{ label='none'; args=@() } )
foreach ($n in 1,2,3,4,5,6,7,8,10,12) {
  $configs += @{ label="dflash2-$n";      args=@('--spec','dflash2','--draft-tokens',[string]$n) }
  $configs += @{ label="dflash2-$n+head"; args=@('--spec','dflash2','--draft-tokens',[string]$n,'--lm-head-draft') }
}
# The backend to beat: MTP3 with the draft head is the best speculative option measured on this
# model, and the reason DFlash2 looks bad is that it is 36% behind it on the same file.
$configs += @{ label='mtp3+head'; args=@('--spec','mtp','--draft-tokens','3','--lm-head-draft') }

"config,draft_tokens,decode_tok_s,stddev,acceptance,spec_rounds,prefill_tok_s"
foreach ($c in $configs) {
  $tag = "d2dt_$($c.label -replace '\+','p')"
  $csv = "$out\$tag.csv"
  $argv = @('--weights',$weights,'--kv-dtype','int8','--max-ctx',[string]($depth + 512),
            '-pg',"$depth,128",'-r','3','--warmup','1','-o','csv','--output-file',$csv) + $c.args
  & .\build-ninja\bench\ninfer_bench.exe @argv > "$out\$tag.log" 2>&1
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) {
    "$($c.label),,FAILED,,,,"
    Get-Content "$out\$tag.log" -Tail 3 | ForEach-Object { "    $_" }
    continue
  }
  $r = Import-Csv $csv | Where-Object { $_.kind -eq 'pp+tg' } | Select-Object -First 1
  if (-not $r) { "$($c.label),,NO-ROW,,,,"; continue }
  "$($c.label),$($r.draft_tokens),$($r.decode_output_tok_s_mean),$($r.decode_output_tok_s_stddev)," +
    "$($r.spec_acceptance_rate),$($r.spec_rounds),$($r.prefill_tok_s_mean)"
}
"== done =="
