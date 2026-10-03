$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))
$d        = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
. "$PSScriptRoot\model-dir.ps1"
$modelDir = Get-NInferModelDir
$python   = if ($env:NINFER_PYTHON) { $env:NINFER_PYTHON } else { 'python' }

# Reads the per-run CSVs that kv-decode-vs-depth.ps1 leaves in $NINFER_SWEEP_OUT -- run that first.
# No GPU needed.
#
# Note those CSVs are ninfer_bench's own full output, written by --output-file, and carry every
# column. They are NOT the six-column summary that sweep prints to stdout, which drops most of the
# fields. The files are the artifact.
#
# TWO denominators, because only one of them is a ceiling anything reaches.
#
# 936.1 GB/s is the 3090's advertised figure: 384-bit GDDR6X at 19.5 Gbps. No kernel reaches it.
# Measured here with tools/hbm_bandwidth_probe.cu, 4 GiB working set (683x L2), best of five:
#
#     cudaMemsetAsync (write)   863.3 GB/s   92.2% of advertised
#     kernel uint4 read         854.2 GB/s   91.3%
#     kernel uint4 write        824.1 GB/s   88.0%
#     cudaMemcpyAsync D2D       813.2 GB/s   86.9%
#
# Decode streams weights and KV *in*, so the read rate is its ceiling: 854.2 GB/s. Reporting
# against the advertised number understates the decode path by about six points.
#
# And the NUMERATOR is not weights_capacity_bytes. That counts weights the decode path never
# streams -- the vision tower, MTP and DFlash when unselected, the draft head with speculation
# off, and all but one row of the token embedding -- and for an MoE it is not even the right
# shape, since one token touches 8 experts of 256 per layer. Using it returns 418% of peak on the
# A3B, which is not a result. tools/decode_byte_accounting.py counts what is actually read, from
# the artifact's own object directory, and this script asks it rather than hardcoding the answer.
$ADVERTISED_GBs = 936.1
$ACHIEVABLE_GBs = 854.2

$artifacts = @{ '27b-dense' = 'qwen3_8_27b.ninfer'; '35b-moe' = 'qwen3_6_35b_a3b.ninfer' }
$perToken  = @{}
foreach ($model in $artifacts.Keys) {
  $path = Join-Path $modelDir $artifacts[$model]
  if (-not (Test-Path $path)) { continue }
  $json = & $python tools\decode_byte_accounting.py $path --json 2>$null
  if ($LASTEXITCODE -eq 0 -and $json) { $perToken[$model] = [double](($json | ConvertFrom-Json).per_token) }
}
if ($perToken.Count -eq 0) { throw "decode_byte_accounting.py produced nothing; is python on PATH?" }

"{0,-10} {1,-7} {2,-8} {3,-9} {4,-11} {5,-11} {6,-13} {7}" -f 'model','kv','depth','tok/s','read GB','KV GB','% advertised','% achievable'
foreach ($model in @('27b-dense','35b-moe')) {
  if (-not $perToken.ContainsKey($model)) { continue }
  $wGB = $perToken[$model] / 1e9
  foreach ($kv in @('int8','rk8v4','fp8','k8v4','nvfp4','bf16')) {
    $f = "$d\${model}_$kv.csv"
    if (-not (Test-Path $f)) { continue }
    foreach ($r in (Import-Csv $f | Where-Object { $_.kind -eq 'pp+tg' })) {
      $tok    = [double]$r.decode_output_tok_s_mean
      $perTok = [double]$r.kv_payload_bytes / 40960          # bytes of KV per token
      $kvGB   = ($perTok * [double]$r.n_prompt) / 1e9        # KV actually attended at this depth
      $bw     = ($wGB + $kvGB) * $tok
      "{0,-10} {1,-7} {2,-8} {3,-9} {4,-11} {5,-11} {6,-13} {7}" -f $model, $kv, $r.n_prompt,
        ("{0:N2}" -f $tok), ("{0:N3}" -f $wGB), ("{0:N3}" -f $kvGB),
        ("{0:N1}%" -f (100*$bw/$ADVERTISED_GBs)),
        ("{0:N1}% ({1:N0} GB/s)" -f (100*$bw/$ACHIEVABLE_GBs), $bw)
    }
  }
}

"`n=== decode falloff, 4,096 -> 32,768 tokens of cache ==="
"{0,-10} {1,-7} {2,-10} {3,-10} {4}" -f 'model','kv','4096','32768','falloff'
foreach ($model in @('27b-dense','35b-moe')) {
  foreach ($kv in @('int8','rk8v4','fp8','k8v4','nvfp4','bf16')) {
    $f = "$d\${model}_$kv.csv"
    if (-not (Test-Path $f)) { continue }
    $rows = Import-Csv $f | Where-Object { $_.kind -eq 'pp+tg' }
    $a = $rows | Where-Object { [int]$_.n_prompt -eq 4096 }  | Select-Object -First 1
    $b = $rows | Where-Object { [int]$_.n_prompt -eq 32768 } | Select-Object -First 1
    if (-not $a -or -not $b) { continue }
    $x = [double]$a.decode_output_tok_s_mean; $y = [double]$b.decode_output_tok_s_mean
    "{0,-10} {1,-7} {2,-10} {3,-10} {4}" -f $model, $kv, ("{0:N2}" -f $x), ("{0:N2}" -f $y),
      ("{0:N1}%" -f (100*($y-$x)/$x))
  }
}

"`n=== prefill, for contrast (compute-bound rather than bandwidth-bound) ==="
"{0,-10} {1,-7} {2,-8} {3}" -f 'model','kv','prompt','prefill tok/s'
foreach ($model in @('27b-dense','35b-moe')) {
  $f = "$d\${model}_int8.csv"
  if (-not (Test-Path $f)) { continue }
  foreach ($r in (Import-Csv $f | Where-Object { $_.kind -eq 'pp+tg' })) {
    "{0,-10} {1,-7} {2,-8} {3}" -f $model, 'int8', $r.n_prompt, ("{0:N1}" -f [double]$r.prefill_tok_s_mean)
  }
}
