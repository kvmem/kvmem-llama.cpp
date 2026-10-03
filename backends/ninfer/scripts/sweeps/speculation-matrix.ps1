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
    "$modelDir\qwen3_8_27b_dflash2.ninfer",
    "$modelDir\qwen3_6_35b_a3b.ninfer")


$models = @(
  @{ key='27b-dense';   path="$modelDir\qwen3_8_27b.ninfer" },
  @{ key='27b-dflash2'; path="$modelDir\qwen3_8_27b_dflash2.ninfer" },
  @{ key='35b-moe';     path="$modelDir\qwen3_6_35b_a3b.ninfer" }
)

# spec label -> extra args. dflash2 only exists on the dflash2 artifact; dflash v1 only on the 35B.
$configs = @(
  @{ label='none';            args=@();                                                  models=@('27b-dense','27b-dflash2','35b-moe') },
  @{ label='mtp3';            args=@('--spec','mtp','--draft-tokens','3');               models=@('27b-dense','27b-dflash2','35b-moe') },
  @{ label='mtp3+head';       args=@('--spec','mtp','--draft-tokens','3','--lm-head-draft'); models=@('27b-dense','27b-dflash2','35b-moe') },
  @{ label='dflash2-7';       args=@('--spec','dflash2','--draft-tokens','7');           models=@('27b-dflash2') },
  @{ label='dflash2-7+head';  args=@('--spec','dflash2','--draft-tokens','7','--lm-head-draft'); models=@('27b-dflash2') },
  @{ label='dflash-3';        args=@('--spec','dflash','--draft-tokens','3');            models=@('35b-moe') },
  @{ label='dflash-3+head';   args=@('--spec','dflash','--draft-tokens','3','--lm-head-draft'); models=@('35b-moe') }
)

"model,spec,kv,decode_tok_s,stddev,weights_gib,sequence_mib,workspace_mib,kv_payload_mib,spec_acc,spec_round"
foreach ($m in $models) {
  foreach ($c in $configs) {
    if ($c.models -notcontains $m.key) { continue }
    $log = "$out\$($m.key)_$($c.label).log"
    $a = @('--weights', $m.path, '--kv-dtype','int8','--max-ctx','8192','-n','128','-r','3','--warmup','1') + $c.args
    & .\build-ninja\bench\ninfer_bench.exe @a > $log 2>&1
    if ($LASTEXITCODE -ne 0) { "$($m.key),$($c.label),int8,FAILED,,,,,,,"; continue }

    $txt = Get-Content $log -Raw
    $dec = if ($txt -match 'tg128\s+\S+\s+\S+\s+\S+\s+([\d.]+)\s*±\s*([\d.]+)') { $Matches[1],$Matches[2] } else { '','' }
    $wt  = if ($txt -match 'weights ([\d.]+) GiB')      { $Matches[1] } else { '' }
    $seq = if ($txt -match 'sequence ([\d.]+) MiB')     { $Matches[1] } else { '' }
    $wsp = if ($txt -match 'workspace ([\d.]+) MiB')    { $Matches[1] } else { '' }
    $kvp = if ($txt -match 'KV payload ([\d.]+) MiB')   { $Matches[1] } else { '' }
    $acc = if ($txt -match 'tg128.*?(\d+\.\d+)%')       { $Matches[1] } else { '' }
    $rnd = if ($txt -match '(\d+\.\d+)\s+[\d.]+ MiB\s*$') { $Matches[1] } else { '' }
    "$($m.key),$($c.label),int8,$($dec[0]),$($dec[1]),$wt,$seq,$wsp,$kvp,$acc,$rnd"
  }
}
