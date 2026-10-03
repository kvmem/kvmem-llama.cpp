$ErrorActionPreference = 'Continue'
# Run from the repository root regardless of where this is invoked from, rather than a hardcoded
# path: these are committed, and the next person's checkout will not be at C:\ninfer-fork.
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

# Both overridable so this is not welded to one machine. The output directory is created here
# because it does not exist in a clean checkout -- profiles/ is gitignored.
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


# The second channel by which KV dtype moves decode speed. README already records that rk8v4 drops
# MTP acceptance 71.27% -> 65.86% on the 27B: a coarser value plane changes which tokens the draft
# proposes, and acceptance turns on exact token agreement rather than mean error. That effect is
# larger than the memory-traffic one, so a speed table built only from unspeculated runs would
# recommend the wrong dtype to anyone running MTP. Captures acceptance alongside tok/s so the two
# channels can be told apart rather than conflated.
$models = @(
  @{ key='27b-dense'; path="$modelDir\qwen3_8_27b.ninfer" },
  @{ key='35b-moe';   path="$modelDir\qwen3_6_35b_a3b.ninfer" }
)
$dtypes = @('bf16','int8','fp8','rk8v4','rk4v4','k8v4','nvfp4')

"model,kv,depth,decode_tok_s,stddev,spec_acceptance,kv_payload_bytes"
foreach ($m in $models) {
  foreach ($d in $dtypes) {
    $csv = "$out\spec_$($m.key)_$d.csv"
    $log = "$out\spec_$($m.key)_$d.log"
    & .\build-ninja\bench\ninfer_bench.exe `
        --weights $m.path --kv-dtype $d --max-ctx 40960 `
        --spec mtp --draft-tokens 3 --lm-head-draft `
        -pg '4096,128;32768,128' -r 3 --warmup 1 `
        -o csv --output-file $csv > $log 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) {
      "$($m.key),$d,ALL,FAILED,,,"
      Get-Content $log -Tail 3 | ForEach-Object { "    $_" }
      continue
    }
    foreach ($r in Import-Csv $csv) {
      # kind_string() emits "pp+tg" for a PrefillDecode test, not "prefill_decode".
      if ($r.kind -ne 'pp+tg') { continue }
      "$($m.key),$d,$($r.n_prompt),$($r.decode_output_tok_s_mean),$($r.decode_output_tok_s_stddev),$($r.spec_acceptance_rate),$($r.kv_payload_bytes)"
    }
  }
}
