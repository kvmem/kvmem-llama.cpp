#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Mmproj,
    [ValidatePattern('^\d+(,\d+)*$')][string]$HipDevices,
    [ValidateSet('none','layer','row')][string]$SplitMode = 'layer',
    [ValidatePattern('^\d+(\.\d+)?(,\d+(\.\d+)?)*$')][string]$TensorSplit,
    [ValidateSet('f16','q8_0','q4_0')][string]$DraftKv = 'f16',
    [ValidateRange(8192,262144)][int]$GpuKvBudget = 28672,
    [string]$BuildDir,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$BuildDir) {
    $BuildDir = if (Test-Path -LiteralPath "$root/bin/llama-kvmem-server.exe") { $root } else { "$root/build-hip-win" }
}
$binary = Join-Path $BuildDir 'bin/llama-kvmem-server.exe'
$Port = 18200
$UiDir = Join-Path $BuildDir 'share/kvmem/ui'
foreach ($path in @($binary, $Model, $Mmproj)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
$binary = (Resolve-Path -LiteralPath $binary).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$Mmproj = (Resolve-Path -LiteralPath $Mmproj).Path
if (!(Test-Path -LiteralPath "$UiDir/index.html")) {
    throw 'Full UI is missing. Extract the complete runtime package, or build it with scripts/build-webui.py --full-ui.'
}
$serverArgs = @(
    '-m', $Model, '--mmproj', $Mmproj, '--no-mmproj-offload',
    '--device', 'ROCm0', '-ngl', '99', '--load-mode', 'none',
    '--host', '127.0.0.1', '--port', "$Port", '--webui', '--ui-dir', $UiDir,
    '-c', '262144', '-b', '512', '-n', '16384',
    '--kvmem', '--kvmem-budget', "$GpuKvBudget", '--kvmem-gen-reserve', '16384',
    '--kvmem-block-tokens', '128', '--kvmem-query-policy', 'user', '--kv-dtype', 'q8_0',
    '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2', '--spec-kv-dtype', $DraftKv,
    '--kvmem-mtp-state', 'replay', '--image-max-tokens', '512',
    '--enable-thinking', '--reasoning-budget', '4096'
)
$selectedPhysical = if ($HipDevices) { $HipDevices } else { $env:HIP_VISIBLE_DEVICES }
if ($selectedPhysical) {
    if ($selectedPhysical -notmatch '^\d+(,\d+)*$') { throw 'HIP_VISIBLE_DEVICES must be comma-separated physical indices.' }
    $indices = @($selectedPhysical -split ',')
    if (@($indices | Select-Object -Unique).Count -ne $indices.Count) { throw 'Duplicate GPU index.' }
    $serverArgs[$serverArgs.IndexOf('--device') + 1] = ((0..($indices.Count - 1) | ForEach-Object { "ROCm$_" }) -join ',')
}
$serverArgs += @('--split-mode', $SplitMode)
if ($TensorSplit) { $serverArgs += @('--tensor-split', $TensorSplit) }
if ($DryRun) { @{ HIP_VISIBLE_DEVICES=$selectedPhysical; argv=(@($binary) + $serverArgs) } | ConvertTo-Json -Depth 4; return }
$oldPath = $env:PATH
$oldHipDevices = $env:HIP_VISIBLE_DEVICES
$oldRocblasBackend = $env:ROCBLAS_USE_HIPBLASLT
try {
    # Packaged DLLs take precedence; a source build can use an installed SDK.
    $sdk = if ($env:ROCM_PATH) { $env:ROCM_PATH } else { $env:HIP_PATH }
    $env:PATH = (Split-Path $binary) + ';' + $(if ($sdk) { "$sdk/bin;" }) + $oldPath
    if ($HipDevices) {
        $env:HIP_VISIBLE_DEVICES = $HipDevices
    } elseif (!$oldHipDevices) {
        $targets = @()
        $buildInfo = Join-Path $BuildDir 'BUILD-INFO.json'
        $cache = Join-Path $BuildDir 'CMakeCache.txt'
        if (Test-Path -LiteralPath $buildInfo) {
            $info = Get-Content -LiteralPath $buildInfo -Raw | ConvertFrom-Json
            $targets = @($info.build_options.GPU_TARGETS -split ';' | Where-Object { $_ })
        } elseif (Test-Path -LiteralPath $cache) {
            $entry = Get-Content -LiteralPath $cache | Where-Object { $_ -match '^(GPU_TARGETS|CMAKE_HIP_ARCHITECTURES):[^=]*=' } | Select-Object -First 1
            if ($entry) { $targets = @(($entry -split '=', 2)[1] -split ';' | Where-Object { $_ }) }
        }
        $probe = [Diagnostics.ProcessStartInfo]::new()
        $probe.FileName = $binary
        $probe.Arguments = '--list-devices'
        $probe.UseShellExecute = $false
        $probe.RedirectStandardOutput = $true
        $probe.RedirectStandardError = $true
        $process = [Diagnostics.Process]::Start($probe)
        $devicesText = $process.StandardOutput.ReadToEnd() + $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw "Could not list HIP devices. Select a physical device with -HipDevices. $devicesText" }
        $devices = @([regex]::Matches($devicesText, '(?m)\bDevice\s+(\d+):[^\r\n]*?\b(gfx[0-9a-f]+)\b[^\r\n]*?VRAM:\s*(\d+)\s*MiB') | ForEach-Object {
            [pscustomobject]@{ Index = [int]$_.Groups[1].Value; Target = $_.Groups[2].Value; Vram = [int]$_.Groups[3].Value }
        })
        if ($targets.Count) { $devices = @($devices | Where-Object { $targets -contains $_.Target }) }
        if (!$devices.Count) { throw "No detected HIP device matches this build. Select one with -HipDevices or build for its gfx target. $devicesText" }
        $env:HIP_VISIBLE_DEVICES = [string]($devices | Sort-Object Vram -Descending | Select-Object -First 1).Index
    }
    if (!$oldRocblasBackend) {
        $selectedProbe = [Diagnostics.ProcessStartInfo]::new()
        $selectedProbe.FileName = $binary
        $selectedProbe.Arguments = '--list-devices'
        $selectedProbe.UseShellExecute = $false
        $selectedProbe.RedirectStandardOutput = $true
        $selectedProbe.RedirectStandardError = $true
        $selectedProcess = [Diagnostics.Process]::Start($selectedProbe)
        $selectedDevice = $selectedProcess.StandardOutput.ReadToEnd() + $selectedProcess.StandardError.ReadToEnd()
        $selectedProcess.WaitForExit()
        if ($selectedProcess.ExitCode -ne 0) { throw "Could not inspect selected HIP device. $selectedDevice" }
        if ($selectedDevice -match '\bgfx120[01]\b') {
            $env:ROCBLAS_USE_HIPBLASLT = '0'
            Write-Host 'RDNA4: using rocBLAS GEMM backend.'
        }
    }
    Write-Host "IQ3 / HIP / physical GPUs $env:HIP_VISIBLE_DEVICES : http://127.0.0.1:$Port/ (Ctrl+C to stop)"
    & $binary @serverArgs
    $code = $LASTEXITCODE
} finally {
    $env:PATH = $oldPath
    $env:HIP_VISIBLE_DEVICES = $oldHipDevices
    $env:ROCBLAS_USE_HIPBLASLT = $oldRocblasBackend
}
exit $code
