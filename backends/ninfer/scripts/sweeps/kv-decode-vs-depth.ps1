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


# Decode cost of a KV dtype scales with cache DEPTH: attention re-reads the whole cache every
# step, so a tg128 run seeded with one token (~128 deep) cannot show the effect at all. Each
# -pg P,128 case prefills P tokens and then times 128 decode steps on top of that cache, which
# is the number a user at P tokens of context actually feels. Depths share one process per
# dtype so the model loads once. --max-ctx is pinned across every run so capacity is a control
# rather than a variable.
$models = @(
  @{ key='27b-dense'; path="$modelDir\qwen3_8_27b.ninfer" },
  @{ key='35b-moe';   path="$modelDir\qwen3_6_35b_a3b.ninfer" }
)
# All seven by default. NINFER_SWEEP_DTYPES narrows it -- three hours is a lot to spend when a
# change touched three of the six kernels and the other three are only a control. Comma-separated,
# e.g. NINFER_SWEEP_DTYPES=int8,fp8,k8v4,nvfp4.
$dtypes = if ($env:NINFER_SWEEP_DTYPES) {
  $env:NINFER_SWEEP_DTYPES -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else {
  @('bf16','int8','fp8','rk8v4','rk4v4','k8v4','nvfp4')
}

"model,kv,depth,decode_tok_s,stddev,kv_payload_bytes"
foreach ($m in $models) {
  foreach ($d in $dtypes) {
    $csv = "$out\$($m.key)_$d.csv"
    $log = "$out\$($m.key)_$d.log"
    & .\build-ninja\bench\ninfer_bench.exe `
        --weights $m.path --kv-dtype $d --max-ctx 40960 `
        -pg '4096,128;16384,128;32768,128' -r 3 --warmup 1 `
        -o csv --output-file $csv > $log 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) {
      "$($m.key),$d,ALL,FAILED,,"
      Get-Content $log -Tail 3 | ForEach-Object { "    $_" }
      continue
    }
    foreach ($r in Import-Csv $csv) {
      if ($r.kind -ne 'pp+tg') { continue }
      "$($m.key),$d,$($r.n_prompt),$($r.decode_output_tok_s_mean),$($r.decode_output_tok_s_stddev),$($r.kv_payload_bytes)"
    }
  }
}
