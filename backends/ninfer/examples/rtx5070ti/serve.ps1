<#
Single-request RTX 5070 Ti preset. Runs in the foreground; Ctrl+C stops the server.
Keep required CUDA/FFmpeg/curl runtime DLLs beside the executable or on PATH.
Engine performance was measured with thinking disabled; xhigh is a usage default.
#>
param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$Model,
    [ValidateSet('low','medium','xhigh')][string]$ReasoningEffort='xhigh',
    [ValidateRange(1,65535)][int]$Port=18081,
    [switch]$DryRun
)
$ErrorActionPreference='Stop'
$exePath=(Resolve-Path -LiteralPath $Executable).Path
$modelPath=(Resolve-Path -LiteralPath $Model).Path
$profile=Join-Path $PSScriptRoot 'device-profile.json'
$serverArgs=@(
    $modelPath,'--host','127.0.0.1','--port',[string]$Port,
    '--model-id','swift-1.5-qwen3.8-27b-xxs',
    '--max-context','147456','--kv-capacity','147456','--max-concurrency','1',
    '--prefill-chunk','640','--kv-dtype','rk8v4',
    '--cuda-graph-allowance-mib','72','--spec','mtp','--draft-tokens','2',
    '--ngram-draft-tokens','31','--gdn-state-fp16',
    '--device-profile','auto','--device-profile-path',$profile,
    '--use-alt-prefix-caching','--host-cache-mib','4096','--device-snapshot-slots','1',
    '--temperature','1','--top-p','0.95','--top-k','20','--min-p','0',
    '--presence-penalty','0','--frequency-penalty','0','--seed','42',
    '--default-reasoning-effort',$ReasoningEffort,'--preserve-thinking'
)
if ($DryRun) {
    [ordered]@{executable=$exePath;arguments=$serverArgs;environment=@{
        NINFER_PREFILL_ALIGN='0';NINFER_PROMPT_FAST=$null;CUDA_LAUNCH_BLOCKING=$null
    }} | ConvertTo-Json -Depth 4
    return
}
$savedEnvironment=@{}
foreach ($name in @('NINFER_PREFILL_ALIGN','NINFER_PROMPT_FAST','CUDA_LAUNCH_BLOCKING')) {
    $savedEnvironment[$name]=[Environment]::GetEnvironmentVariable($name,'Process')
}
try {
    $env:NINFER_PREFILL_ALIGN='0'
    Remove-Item Env:NINFER_PROMPT_FAST -ErrorAction SilentlyContinue
    Remove-Item Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue
    & $exePath @serverArgs
    if ($LASTEXITCODE -ne 0) { throw "ninfer-serve exited with code $LASTEXITCODE" }
} finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name,$savedEnvironment[$name],'Process')
    }
}
