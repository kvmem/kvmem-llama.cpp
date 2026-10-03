#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('core', 'llamacpp', 'ninfer', 'all')][string]$Backend = 'core',
    [string]$BuildRoot,
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$CudaArchitectures = '120a',
    [ValidateRange(1, 64)][int]$Jobs = 2,
    [string[]]$CMakeArgs = @(),
    [switch]$ConfigureOnly,
    [switch]$BuildOnly,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($ConfigureOnly -and $BuildOnly) { throw 'Choose ConfigureOnly or BuildOnly, not both.' }
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$BuildRoot) { $BuildRoot = Join-Path $source 'build-backends' }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
$backends = if ($Backend -eq 'all') { @('llamacpp', 'ninfer') } else { @($Backend) }
$plans = foreach ($name in $backends) {
    $build = Join-Path $BuildRoot $name
    $tree = if ($name -eq 'ninfer') { Join-Path $source 'backends/ninfer' } else { $source }
    $configure = @('-S', $tree, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release')
    $target = @()
    if ($name -eq 'core') {
        $configure += @('-DKVMEM_BUILD_LLAMA=OFF', '-DKVMEM_BUILD_SERVER_TESTS=OFF')
    } elseif ($name -eq 'llamacpp') {
        $configure += @('-DKVMEM_BUILD_LLAMA=ON', '-DGGML_CUDA=ON',
            '-DGGML_CUDA_FA_QUANTS=all', '-DBUILD_SHARED_LIBS=OFF',
            "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures")
        $target = @('--target', 'llama-kvmem-server')
    } else {
        $configure += @("-DNINFER_KVMEM_SOURCE_DIR=$source/kvmem", '-DNINFER_BUILD_APPS=ON',
            '-DNINFER_BUILD_BENCHMARKS=OFF', "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures")
        if ($CudaArchitectures -eq '120a') { $configure += '-DNINFER_SM120_NATIVE=ON' }
        $target = @('--target', 'ninfer-serve')
    }
    if ($name -ne 'core' -and $CudaPath) {
        $configure += @("-DCMAKE_CUDA_COMPILER=$CudaPath/bin/nvcc.exe", "-DCUDAToolkit_ROOT=$CudaPath")
    }
    $configure += $CMakeArgs
    @{ backend = $name; configure = $configure
       build = @('--build', $build, '--parallel', "$Jobs") + $target }
}
if ($DryRun) { $plans | ConvertTo-Json -Depth 5; return }
if (!$env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -version '[17.0,18.0)' -latest -products '*' `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vs) { throw 'Install Visual Studio 2022 C++ Build Tools or use its x64 developer shell.' }
    & (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
}
if ($CudaPath) { $env:PATH = "$CudaPath/bin;$CudaPath/bin/x64;$env:PATH" }
foreach ($plan in $plans) {
    if (!$BuildOnly) {
        $configureArgs = $plan.configure
        & cmake @configureArgs
        if ($LASTEXITCODE -ne 0) { throw "Configure failed: $($plan.backend)" }
    }
    if (!$ConfigureOnly) {
        $buildArgs = $plan.build
        & cmake @buildArgs
        if ($LASTEXITCODE -ne 0) { throw "Build failed: $($plan.backend)" }
    }
}
