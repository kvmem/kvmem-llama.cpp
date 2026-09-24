#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Model = (Join-Path $env:LOCALAPPDATA 'KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'),
    [string]$BuildDir,
    [string]$UiDir,
    [string]$Gpu = '0',
    [ValidateRange(1, 65535)][int]$Port = 18202,
    [ValidateRange(128, 262144)][int]$Context = 131072,
    [ValidateRange(128, 262144)][int]$Budget = 24576,
    [ValidateRange(128, 262144)][int]$Reserve = 10240,
    [ValidateRange(-1, 262144)][int]$ReasoningBudget = 4096,
    [ValidateRange(1, 4096)][int]$Batch = 128,
    [ValidateSet('q8_0', 'q5_0', 'q4_0')][string]$KvType = 'q8_0',
    [ValidateSet('f16', 'q8_0', 'q5_0', 'q4_0')][string]$DraftKvType = 'f16',
    [switch]$Mtp,
    [switch]$NoMtp,
    [ValidateRange(1, 2)][int]$DraftTokens = 1,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
if ($NoMtp) {
    if ($PSBoundParameters.ContainsKey('Mtp') -and $Mtp) { throw 'Choose either -Mtp or -NoMtp' }
    $Mtp = $false
}
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$BuildDir) {
    if (Test-Path -LiteralPath (Join-Path $root 'bin/llama-kvmem-server.exe')) { $BuildDir = $root }
    else { $BuildDir = Join-Path $root 'build-win-bonsai-release' }
}
$binary = Join-Path $BuildDir 'bin/llama-kvmem-server.exe'
if ($Mtp -and !$PSBoundParameters.ContainsKey('Model')) {
    $Model = Join-Path $env:LOCALAPPDATA 'KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q8_0.gguf'
}
foreach ($path in @($binary, $Model)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
if ($Budget % 128 -ne 0 -or $Reserve % 128 -ne 0) { throw 'Budget and Reserve must be multiples of 128' }
if ($Budget + $Reserve -gt $Context) { throw 'Budget + Reserve must not exceed Context' }
$binary = (Resolve-Path -LiteralPath $binary).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$specType = if ($Mtp) { 'draft-mtp' } else { 'none' }
$serverArgs = @('-m', $Model, '-ngl', '99', '--host', '127.0.0.1', '--port', "$Port",
    '-c', "$Context", '-b', "$Batch", '--ubatch-size', "$Batch", '-n', "$Reserve",
    '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve", '--kvmem-block-tokens', '128',
    '--kv-dtype', $KvType, '--spec-type', $specType, '--kvmem-mtp-state', 'snapshots',
    '--kvmem-query-policy', 'user', '--kvmem-query-replay', 'auto', '--flash-attn', 'on',
    '--enable-thinking', '--reasoning-budget', "$ReasoningBudget")
if ($Mtp) { $serverArgs += @('--spec-draft-n-max', "$DraftTokens", '--spec-kv-dtype', $DraftKvType) }
if ($UiDir) { $serverArgs += @('--ui-dir', (Resolve-Path -LiteralPath $UiDir).Path) }
if ($DryRun) {
    @{ argv = @($binary) + $serverArgs; environment = @{ CUDA_VISIBLE_DEVICES = $Gpu; CUDA_DEVICE_ORDER = 'PCI_BUS_ID' } } |
        ConvertTo-Json -Depth 5
    return
}
$probe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
$probe.Server.ExclusiveAddressUse = $true
try { $probe.Start() } finally { $probe.Stop() }
$info = New-KVMemProcessInfo $binary $serverArgs
$info.CreateNoWindow = $true
$info.RedirectStandardOutput = $true
$info.RedirectStandardError = $true
$info.StandardOutputEncoding = [Text.Encoding]::UTF8
$info.StandardErrorEncoding = [Text.Encoding]::UTF8
$info.EnvironmentVariables['CUDA_VISIBLE_DEVICES'] = $Gpu
$info.EnvironmentVariables['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
$child = New-Object System.Diagnostics.Process
$child.StartInfo = $info
$started = $false
try {
    Write-Host "Starting Bonsai (speculation: $specType) at http://127.0.0.1:$Port/"
    $started = $child.Start()
    # Drain both pipes concurrently; stderr contains normal server timing logs.
    $streams = @(
        @{ Reader = $child.StandardOutput; Pending = $child.StandardOutput.ReadLineAsync(); Ended = $false },
        @{ Reader = $child.StandardError; Pending = $child.StandardError.ReadLineAsync(); Ended = $false }
    )
    while (!$child.HasExited -or !$streams[0].Ended -or !$streams[1].Ended) {
        $progress = $false
        foreach ($stream in $streams) {
            if (!$stream.Ended -and $stream.Pending.IsCompleted) {
                $line = $stream.Pending.GetAwaiter().GetResult()
                if ($null -eq $line) { $stream.Ended = $true }
                else {
                    Write-Host $line
                    $stream.Pending = $stream.Reader.ReadLineAsync()
                }
                $progress = $true
            }
        }
        if (!$progress) { Start-Sleep -Milliseconds 10 }
    }
    $child.WaitForExit()
    $code = $child.ExitCode
} finally {
    if ($started -and !$child.HasExited) { $child.Kill(); $child.WaitForExit() }
    $child.Dispose()
}
exit $code
