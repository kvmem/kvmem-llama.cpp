# What does each speculative backend actually cost in memory?
#
# docs/config-calculator.html modelled speculation as a weights delta and nothing else, which made
# every speculative configuration it reported roughly 170 MiB optimistic (TODO section 2b). The
# missing pieces are the CUDA graph allowance, the non-KV sequence block and the KV cache itself,
# all three of which move when a draft window is added.
#
# Everything needed is reported by the engine at load, so this asks for one token and reads the
# summary rather than measuring anything. Two contexts per configuration separate the fixed part of
# the sequence block from its per-token part; a third is a check that the two-point solve is right.
#
# Roughly ten minutes for the whole matrix, dominated by artifact load time.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\speculative-memory-terms.ps1
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
    "$modelDir\qwen3_8_27b_dflash2.ninfer",
    "$modelDir\qwen3_6_35b_a3b.ninfer")

# DFlash2 lives in its own artifact. The calculator's 27B row is measured against
# qwen3_8_27b.ninfer, so its DFlash2 entries are a different file from its other entries -- which
# is worth stating rather than smoothing over, and is why `artifact` is a column below.
$dense   = "$modelDir\qwen3_8_27b.ninfer"
$dense2  = "$modelDir\qwen3_8_27b_dflash2.ninfer"
$moe     = "$modelDir\qwen3_6_35b_a3b.ninfer"

$configs = @(
  @{ model='27b'; key='none';            weights=$dense;  spec=@() }
  @{ model='27b'; key='mtp3';            weights=$dense;  spec=@('--spec','mtp','--draft-tokens','3') }
  @{ model='27b'; key='mtp3+head';       weights=$dense;  spec=@('--spec','mtp','--draft-tokens','3','--lm-head-draft') }
  @{ model='27b'; key='dflash2-7';       weights=$dense2; spec=@('--spec','dflash2','--draft-tokens','7') }
  @{ model='27b'; key='dflash2-7+head';  weights=$dense2; spec=@('--spec','dflash2','--draft-tokens','7','--lm-head-draft') }
  @{ model='35b'; key='none';            weights=$moe;    spec=@() }
  @{ model='35b'; key='mtp3';            weights=$moe;    spec=@('--spec','mtp','--draft-tokens','3') }
  @{ model='35b'; key='mtp3+head';       weights=$moe;    spec=@('--spec','mtp','--draft-tokens','3','--lm-head-draft') }
  @{ model='35b'; key='dflash-3';        weights=$moe;    spec=@('--spec','dflash','--draft-tokens','3') }
  @{ model='35b'; key='dflash-3+head';   weights=$moe;    spec=@('--spec','dflash','--draft-tokens','3','--lm-head-draft') }
)

# int8 for the solve; one nvfp4 point per configuration checks that the speculative KV multiplier
# is a property of the draft window rather than of the storage format.
$points = @(
  @{ kv='int8';  ctx=8192  }
  @{ kv='int8';  ctx=16384 }
  @{ kv='int8';  ctx=32768 }
  @{ kv='nvfp4'; ctx=16384 }
)

"model,spec,artifact,kv,max_ctx,weights_bytes,sequence_bytes,kv_payload_bytes,workspace_bytes,graph_bytes"
foreach ($c in $configs) {
  if (-not (Test-Path $c.weights)) { "$($c.model),$($c.key),MISSING,,,,,,,"; continue }
  foreach ($p in $points) {
    $tag = "spm_$($c.model)_$($c.key -replace '\+','p')_$($p.kv)_$($p.ctx)"
    $csv = "$out\$tag.csv"
    $argv = @('--weights',$c.weights,'--kv-dtype',$p.kv,'--max-ctx',[string]$p.ctx,
              '-n','1','-r','1','--warmup','0','-o','csv','--output-file',$csv) + $c.spec
    & .\build-ninja\bench\ninfer_bench.exe @argv > "$out\$tag.log" 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $csv)) {
      "$($c.model),$($c.key),$(Split-Path -Leaf $c.weights),$($p.kv),$($p.ctx),FAILED,,,,"
      Get-Content "$out\$tag.log" -Tail 3 | ForEach-Object { "    $_" }
      continue
    }
    $r = Import-Csv $csv | Select-Object -First 1
    "$($c.model),$($c.key),$(Split-Path -Leaf $c.weights),$($p.kv),$($p.ctx)," +
      "$($r.weights_capacity_bytes),$($r.sequence_capacity_bytes),$($r.kv_payload_bytes)," +
      "$($r.workspace_capacity_bytes),$($r.cuda_graph_allowance_bytes)"
  }
}
"== done =="
