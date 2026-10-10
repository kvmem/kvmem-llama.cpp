#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('llamacpp', 'ninfer')][string]$Backend,
    [string]$Model,
    [string]$Worker,
    [string]$Gpu = $env:CUDA_VISIBLE_DEVICES,
    [uint32[]]$StageLayers = @(),
    [ValidateRange(1, 65535)][int]$Port = 18200,
    [string]$ListenHost = '127.0.0.1',
    [ValidateRange(256, 2147483647)][int]$Context = 262144,
    [ValidateRange(128, 2147483647)][int]$Budget = 4096,
    [ValidateRange(64, 2147483647)][int]$Reserve = 1024,
    [ValidateRange(1, 1048576)][int]$HostMiB = 16384,
    [ValidateRange(128, 2147483647)][int]$Prefill = 128,
    [ValidateRange(1, 2147483647)][int]$MaxTokens = 1024,
    [string]$KvType,
    [ValidateRange(0, 15)][int]$MtpDrafts = 0,
    [switch]$AdaptiveMtp,
    [switch]$FastPrefill,
    [ValidateRange(0, 63)][int]$NgramDrafts = 0,
    [ValidateRange(4, 64)][int]$NgramMinMatch = 12,
    [ValidateRange(1, 8)][int]$Concurrency = 1,
    [ValidateRange(1, 16)][int]$RetainedSessions = 4,
    [switch]$Vision,
    [ValidateSet('resident', 'cpu')][string]$VisionResidency = 'resident',
    [ValidateRange(1, 16384)][int]$VisionTokens = 1024,
    [ValidateSet('off', 'auto')][string]$DeviceProfile = 'off',
    [ValidateRange(0, 2147483647)][int]$ThinkingBudget = 0,
    [ValidateRange(0, 2147483647)][int]$SinkTokens = 0,
    [ValidateRange(64, 2147483647)][int]$RecentTokens = 64,
    [string]$DiskPath,
    [ValidateRange(1, 1048576)][int]$DiskMiB = 4096,
    [string[]]$WorkerArgs = @(),
    [switch]$Describe,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$capabilities = @{
    backend = $Backend; model_format = if ($Backend -eq 'ninfer') { '.ninfer' } else { '.gguf' }
    text_chat = $true; streaming = $true; history_reuse = $true
    speculative_decoding = $true; multimodal = $true
    concurrent_requests = $true; disk_restore = $true
    active_kv_tiering = $true; ssd_persistence = 'process'
    cross_backend_kv_transfer = $false
}
if ($Backend -eq 'ninfer') {
    $capabilities.disk_restore = $false
    $capabilities.supported_combinations = @{
        text = @{ kv = @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); max_concurrency = 8; mtp_drafts = @(0..15) }
        images = @{ kv = @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); max_concurrency = 8; residency = @('resident', 'cpu') }
        rk8v4 = @{ max_concurrency = 8; text_only = $false }
        cold_disk = @{ enabled = $false; reason = 'replaced_by_process_local_tiering' }
        active_disk = @{ enabled = $true; shared_active_idle_quota = $true; max_concurrency = 8 }
        ngram = @{ max_concurrency = 8; text_only = $false; image_kv = @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); requires_mtp = $true; max_drafts = 63; concurrent_max_drafts = 15; wide_max_concurrency = 1; concurrent_width_policy = 'clamp_to_15' }
        lookup_ngram = @{ enabled = $false; reason = 'legacy_path_disabled'; replacement = '--ngram-draft-tokens --ngram-min-match' }
        adaptive_mtp = @{ max_concurrency = 8; text_only = $false; ngram = $true; ngram_max_concurrency = 8; ngram_concurrent_max_drafts = 15; cold_disk = $false }
        fast_prefill = @{ kv = @('int8', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'); text_only = $false; max_concurrency = 8; residency = @('resident', 'cpu'); cold_disk = $false }
        multi_gpu = @{ mode = 'layer'; max_devices = 8; vision_residency = @('resident', 'cpu'); mtp = $true; host_history = 'shared' }
        video = $false
    }
    if (@($WorkerArgs | Where-Object { $_ -match '^--lookup-ngram(=|$)' }).Count) {
        throw 'KVMem disables legacy --lookup-ngram; use -NgramDrafts and -NgramMinMatch'
    }
}
if ($Describe) { $capabilities | ConvertTo-Json -Depth 4; return }
if (@($WorkerArgs | Where-Object { $_ -match '^--kvmem-disk-(path|mib)(=|$)' }).Count) {
    throw 'Use -DiskPath and -DiskMiB to configure shared active/idle SSD storage'
}
if (($PSBoundParameters.ContainsKey('DiskPath') -or $PSBoundParameters.ContainsKey('DiskMiB')) -and
    [string]::IsNullOrWhiteSpace($DiskPath)) {
    throw 'SSD storage requires a nonempty -DiskPath'
}
if (!$Worker) {
    $file = if ($Backend -eq 'ninfer') { 'ninfer-kvmem-server.exe' } else { 'llama-kvmem-server.exe' }
    $Worker = Join-Path $root "workers/$Backend/$file"
}
foreach ($entry in @(@('Worker', $Worker), @('Model', $Model))) {
    if (!$entry[1] -or !(Test-Path -LiteralPath $entry[1] -PathType Leaf)) {
        throw "Missing $($entry[0]) file: $($entry[1])"
    }
}
$Worker = (Resolve-Path -LiteralPath $Worker).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
if ([IO.Path]::GetExtension($Model) -ne $capabilities.model_format) {
    throw "$Backend requires a $($capabilities.model_format) model"
}
if (!$Gpu -or $Gpu.Trim() -eq '-1') { throw 'Select a GPU UUID with -Gpu or CUDA_VISIBLE_DEVICES' }
if ($Backend -eq 'ninfer') {
    $gpuIds = @($Gpu.Split(',') | ForEach-Object { $_.Trim() })
    if ($gpuIds.Count -gt 8 -or @($gpuIds | Where-Object { !$_ -or $_ -eq '-1' }).Count -or
        @($gpuIds | Select-Object -Unique).Count -ne $gpuIds.Count) {
        throw 'ninfer requires 1..8 distinct GPU UUIDs in -Gpu, in stage order'
    }
    $Gpu = $gpuIds -join ','
    if ($StageLayers.Count -and ($gpuIds.Count -lt 2 -or $StageLayers.Count -ne $gpuIds.Count -or
        @($StageLayers | Where-Object { $_ -eq 0 }).Count)) {
        throw 'StageLayers requires one positive layer count per GPU and at least two GPUs'
    }
    if (@($WorkerArgs | Where-Object { $_ -match '^--(device|devices|stage-layers)(=|$)' }).Count) {
        throw 'Use -Gpu and -StageLayers to configure ninfer devices'
    }
    if (@($WorkerArgs | Where-Object { $_ -match '^--default-thinking-budget(=|$)' }).Count) {
        throw 'Use -ThinkingBudget to configure the ninfer thinking budget'
    }
    if (@($WorkerArgs | Where-Object { $_ -match '^--kvmem-(sink|recent)-tokens(=|$)' }).Count) {
        throw 'Use -SinkTokens and -RecentTokens to configure the ninfer keep windows'
    }
}
if (!$KvType) { $KvType = if ($Backend -eq 'ninfer') { 'int8' } else { 'q8_0' } }
if ($Backend -eq 'ninfer') {
    $requestedNgramDrafts = $NgramDrafts
    if ($Concurrency -gt 1 -and $NgramDrafts -gt 15) { $NgramDrafts = 15 }
    if ($Budget % 64 -or $Reserve % 64 -or $Prefill % 128 -or $Prefill -gt $Reserve -or
        ([long]$Budget + $Reserve) -gt $Context -or $KvType -notin @('bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8')) {
        throw 'ninfer requires B/R aligned to 64, prefill aligned to 128 and <= R, B+R <= Context, bf16/int8/fp8/nvfp4/k8v4/rk8v4/rk4v4/rk4v4-e8/rk2v4-e8'
    }
    if ($AdaptiveMtp -and !$MtpDrafts) {
        throw 'Adaptive MTP requires MTP; image, ngram and disk limits still apply'
    }
    if ($FastPrefill -and ($KvType -notin @('int8', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8'))) {
        throw 'Fast prefill requires INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8'
    }
    if ($VisionResidency -ne 'resident' -and !$Vision) { throw 'CPU Vision requires -Vision' }
    $sinkBlocks = if ($SinkTokens -lt 64) { 1 } else { [int][Math]::Floor($SinkTokens / 64) }
    $recentBlocks = [int][Math]::Floor($RecentTokens / 64)
    if (($sinkBlocks + $recentBlocks) * 64 -gt $Budget) {
        throw 'ninfer sink and recent tokens exceed -Budget after rounding down to 64-token pages'
    }
    if ($NgramDrafts -and (!$MtpDrafts -or
        ([Math]::Max($MtpDrafts, $NgramDrafts) + $MtpDrafts) -gt $Reserve)) {
        throw 'ngram requires MTP and reserve >= max(MTP drafts, effective ngram drafts) + MTP drafts'
    }
    if ($requestedNgramDrafts -ne $NgramDrafts) {
        Write-Warning "NgramDrafts reduced from $requestedNgramDrafts to $NgramDrafts for Concurrency $Concurrency"
    }
    $nativeArgs = @($Model, '--host', $ListenHost, '--port', "$Port", '--max-context', "$Context",
        '--max-concurrency', "$Concurrency", '--prefill-chunk', "$Prefill", '--default-max-tokens', "$MaxTokens",
        '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve", '--kvmem-host-mib', "$HostMiB",
        '--kvmem-sink-tokens', "$SinkTokens", '--kvmem-recent-tokens', "$RecentTokens",
        '--kvmem-sessions', "$RetainedSessions", '--kv-dtype', $KvType, '--device-profile', $DeviceProfile)
    if ($gpuIds.Count -gt 1) { $nativeArgs += @('--devices', ((0..($gpuIds.Count - 1)) -join ',')) }
    else { $nativeArgs += @('--device', '0') }
    if ($StageLayers.Count) { $nativeArgs += @('--stage-layers', ($StageLayers -join ',')) }
    if ($MtpDrafts) { $nativeArgs += @('--spec', 'mtp', '--draft-tokens', "$MtpDrafts",
        '--ngram-draft-tokens', "$NgramDrafts", '--ngram-min-match', "$NgramMinMatch") }
    if ($AdaptiveMtp) { $nativeArgs += '--adaptive-mtp' }
    if ($FastPrefill) { $nativeArgs += '--fast-prefill-kernel' }
    if ($Vision) { $nativeArgs += @('--vision', '--vision-residency', $VisionResidency, '--vision-max-merged', "$VisionTokens") }
    if ($ThinkingBudget -gt 0) { $nativeArgs += @('--default-thinking-budget', "$ThinkingBudget") }
} else {
    foreach ($name in @('StageLayers', 'MtpDrafts', 'AdaptiveMtp', 'FastPrefill', 'NgramDrafts', 'NgramMinMatch', 'Concurrency', 'RetainedSessions', 'Vision', 'VisionResidency', 'VisionTokens', 'DeviceProfile', 'ThinkingBudget', 'SinkTokens', 'RecentTokens')) {
        if ($PSBoundParameters.ContainsKey($name)) { throw "-$name configures ninfer; use llama native options in -WorkerArgs" }
    }
    $nativeArgs = @('-m', $Model, '--host', $ListenHost, '--port', "$Port", '-c', "$Context",
        '-n', "$MaxTokens", '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve",
        '--kv-dtype', $KvType, '-b', "$Prefill", '-ub', "$Prefill")
    if ($PSBoundParameters.ContainsKey('HostMiB') -or $DiskPath) { $nativeArgs += @('--kvmem-host-mib', "$HostMiB") }
}
if ($DiskPath) { $nativeArgs += @('--kvmem-disk-path', $DiskPath, '--kvmem-disk-mib', "$DiskMiB") }
$nativeArgs += $WorkerArgs
if ($DryRun) {
    @{ backend = $Backend; argv = @($Worker) + $nativeArgs; capabilities = $capabilities
       environment = @{ CUDA_VISIBLE_DEVICES = $Gpu; CUDA_DEVICE_ORDER = 'PCI_BUS_ID' } } |
        ConvertTo-Json -Depth 6
    return
}
$info = New-KVMemProcessInfo $Worker $nativeArgs
$info.EnvironmentVariables['CUDA_VISIBLE_DEVICES'] = $Gpu
$info.EnvironmentVariables['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
$child = New-Object System.Diagnostics.Process
$child.StartInfo = $info
$started = $false
try {
    Write-Host "Starting $Backend at http://$ListenHost`:$Port/ (Ctrl+C to stop)"
    $started = $child.Start()
    while (!$child.WaitForExit(250)) {}
    if ($child.ExitCode -ne 0) { throw "$Backend exited with code $($child.ExitCode)" }
} finally {
    if ($started -and !$child.HasExited) { $child.Kill(); $child.WaitForExit() }
    $child.Dispose()
}
