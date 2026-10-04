#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('llamacpp', 'ninfer')][string]$Backend,
    [string]$Model,
    [string]$Worker,
    [string]$Gpu = $env:CUDA_VISIBLE_DEVICES,
    [ValidateRange(1, 65535)][int]$Port = 18200,
    [string]$ListenHost = '127.0.0.1',
    [ValidateRange(256, 2147483647)][int]$Context = 262144,
    [ValidateRange(128, 2147483647)][int]$Budget = 4096,
    [ValidateRange(64, 2147483647)][int]$Reserve = 1024,
    [ValidateRange(1, 1048576)][int]$HostMiB = 16384,
    [ValidateRange(128, 2147483647)][int]$Prefill = 128,
    [ValidateRange(1, 2147483647)][int]$MaxTokens = 1024,
    [string]$KvType,
    [ValidateSet(0, 1, 2, 3, 4)][int]$MtpDrafts = 0,
    [ValidateRange(0, 63)][int]$NgramDrafts = 0,
    [ValidateRange(4, 64)][int]$NgramMinMatch = 12,
    [ValidateRange(1, 4)][int]$Concurrency = 1,
    [ValidateRange(1, 16)][int]$RetainedSessions = 4,
    [switch]$Vision,
    [ValidateRange(1, 16384)][int]$VisionTokens = 256,
    [ValidateSet('off', 'auto')][string]$DeviceProfile = 'off',
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
    cross_backend_kv_transfer = $false
}
if ($Backend -eq 'ninfer') {
    $capabilities.supported_combinations = @{
        text = @{ kv = @('bf16', 'int8', 'nvfp4'); max_concurrency = 4; mtp_drafts = @(0, 1, 2, 3, 4) }
        images = @{ kv = @('bf16', 'int8', 'nvfp4'); max_concurrency = 1; residency = 'resident' }
        rk8v4 = @{ max_concurrency = 1; text_only = $true }
        cold_disk = @{ max_concurrency = 1; text_only = $true; device_profile = 'off' }
        ngram = @{ max_concurrency = 1; text_only = $true; requires_mtp = $true; max_drafts = 63 }
        adaptive_mtp = $false; video = $false
    }
}
if ($Describe) { $capabilities | ConvertTo-Json -Depth 4; return }
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
if (!$KvType) { $KvType = if ($Backend -eq 'ninfer') { 'int8' } else { 'q8_0' } }
if ($Backend -eq 'ninfer') {
    if ($Budget % 64 -or $Reserve % 64 -or $Prefill % 128 -or $Prefill -gt $Reserve -or
        ([long]$Budget + $Reserve) -gt $Context -or $KvType -notin @('int8', 'bf16', 'nvfp4', 'rk8v4')) {
        throw 'ninfer requires B/R aligned to 64, prefill aligned to 128 and <= R, B+R <= Context, int8/bf16/nvfp4/rk8v4'
    }
    if (($Vision -and $Concurrency -ne 1) -or
        ($KvType -eq 'rk8v4' -and ($Vision -or $Concurrency -ne 1)) -or
        ($DiskPath -and ($Vision -or $Concurrency -ne 1 -or $DeviceProfile -ne 'off'))) {
        throw 'Images require one lane; RK8V4 and cold disk require one text lane; cold disk also requires DeviceProfile off'
    }
    if ($NgramDrafts -and (!$MtpDrafts -or $Vision -or $Concurrency -ne 1 -or
        ([Math]::Max($MtpDrafts, $NgramDrafts) + $MtpDrafts) -gt $Reserve)) {
        throw 'ngram requires MTP, one text lane, and reserve >= max(MTP drafts, ngram drafts) + MTP drafts'
    }
    $nativeArgs = @($Model, '--host', $ListenHost, '--port', "$Port", '--max-context', "$Context",
        '--max-concurrency', "$Concurrency", '--prefill-chunk', "$Prefill", '--default-max-tokens', "$MaxTokens",
        '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve", '--kvmem-host-mib', "$HostMiB",
        '--kvmem-sessions', "$RetainedSessions", '--kv-dtype', $KvType, '--device', '0', '--device-profile', $DeviceProfile)
    if ($MtpDrafts) { $nativeArgs += @('--spec', 'mtp', '--draft-tokens', "$MtpDrafts",
        '--ngram-draft-tokens', "$NgramDrafts", '--ngram-min-match', "$NgramMinMatch") }
    if ($Vision) { $nativeArgs += @('--vision', '--vision-residency', 'resident', '--vision-max-merged', "$VisionTokens") }
    if ($DiskPath) { $nativeArgs += @('--kvmem-disk-path', $DiskPath, '--kvmem-disk-mib', "$DiskMiB", '--derive-session-keys') }
} else {
    foreach ($name in @('MtpDrafts', 'NgramDrafts', 'NgramMinMatch', 'Concurrency', 'RetainedSessions', 'Vision', 'VisionTokens', 'DeviceProfile', 'DiskPath', 'DiskMiB')) {
        if ($PSBoundParameters.ContainsKey($name)) { throw "-$name configures ninfer; use llama native options in -WorkerArgs" }
    }
    if ($PSBoundParameters.ContainsKey('HostMiB')) {
        throw '-HostMiB configures ninfer native payload storage; use llama native options in -WorkerArgs'
    }
    $nativeArgs = @('-m', $Model, '--host', $ListenHost, '--port', "$Port", '-c', "$Context",
        '-n', "$MaxTokens", '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve",
        '--kv-dtype', $KvType, '-b', "$Prefill", '-ub', "$Prefill")
}
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
