#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$SourceDir,
    [string]$BuildDir,
    [string]$CudaPath = $env:CUDA_PATH,
    [switch]$ExperimentalCuda129,
    [string]$CudaArchitectures = '75-real;80-real;86-real;89-real;90-real;120a-real',
    [ValidateRange(1, 64)][int]$Jobs = 4,
    [switch]$HostOnly,
    [switch]$BuildOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (!$SourceDir) { $SourceDir = Join-Path $PSScriptRoot '../..' }
$SourceDir = (Resolve-Path -LiteralPath $SourceDir).Path
if (!$BuildDir) { $BuildDir = Join-Path $SourceDir 'build-win' }
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
Set-Location -LiteralPath $SourceDir
[Environment]::CurrentDirectory = $SourceDir

# Initialize the x64 MSVC environment, including the bundled CMake and Ninja.
if (!$env:VSCMD_VER) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vs) { throw 'Install Visual Studio C++ Build Tools first' }
    & (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
}
if ($env:VSCMD_ARG_TGT_ARCH -ne 'x64') { throw 'An x64 Visual Studio developer environment is required' }
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
foreach ($tool in 'cmake', 'ninja', 'cl') { $null = Get-Command $tool -ErrorAction Stop }

if (!$HostOnly) {
    # Use the maintained patch, never the developer's unrecorded submodule edits.
    $llama = Join-Path $SourceDir 'llama.cpp'
    $patch = Join-Path $SourceDir 'patches/llama-kvmem-current.patch'
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & git -C $llama apply --reverse --check $patch 2>$null
        $applied = $LASTEXITCODE -eq 0
    } finally { $ErrorActionPreference = $savedPreference }
    if (!$applied) {
        Invoke-Checked git @('-C', $llama, 'apply', '--check', $patch)
        Invoke-Checked git @('-C', $llama, 'apply', $patch)
    }
}
$options = @('-S', $SourceDir, '-B', $BuildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=cl',
    '-DBUILD_SHARED_LIBS=OFF', '-DKVMEM_ENABLE_NVME=OFF')
if (!$HostOnly) {
    $options += @('-DCMAKE_C_COMPILER=cl', '-DGGML_BACKEND_DL=OFF',
    '-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON',
    '-DGGML_F16C=ON', '-DGGML_BMI2=ON', '-DGGML_AVX512=OFF',
    '-DGGML_CUDA_FA_ALL_QUANTS=ON')
}
if ($HostOnly) {
    $options += '-DKVMEM_BUILD_LLAMA=OFF'
} else {
    if (!$CudaPath -or !(Test-Path -LiteralPath (Join-Path $CudaPath 'bin/nvcc.exe'))) {
        throw 'Install CUDA Toolkit and set CUDA_PATH, or pass -CudaPath'
    }
    $nvccVersion = & (Join-Path $CudaPath 'bin/nvcc.exe') --version
    if ($LASTEXITCODE -ne 0 -or ($nvccVersion -join ' ') -notmatch 'V(\d+\.\d+\.\d+)') {
        throw 'Cannot determine nvcc version'
    }
    if ($ExperimentalCuda129) {
        if ($Matches[1] -ne '12.9.86') { throw 'Experimental CUDA 12.9 build requires nvcc 12.9.86' }
    } elseif ([version]$Matches[1] -lt [version]'13.2.86') {
        throw 'CUDA Toolkit 13.2 Update 2 (nvcc 13.2.86) or newer is required; rebuild in a new directory.'
    }
    $env:PATH = "$CudaPath\bin;$CudaPath\bin\x64;$env:PATH"
    $options += @('-DKVMEM_BUILD_LLAMA=ON', '-DGGML_CUDA=ON',
        "-DCMAKE_CUDA_COMPILER=$CudaPath/bin/nvcc.exe", "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
        "-DCUDAToolkit_ROOT=$CudaPath")
}
Invoke-Checked cmake $options
$targets = @('kvmem_store_test', 'pinned_kv_tier_test', 'nvme_disabled_test', 'kvmem_runtime_test', 'raw_kv_store_test')
if (!$HostOnly) {
    $targets += @('llama-kvmem-server', 'llama-kvmem-cli', 'llama-quantize',
        'kvmem-chat-id-test', 'kvmem-reasoning-budget-test', 'kvmem-chat-template-test', 'kvmem-server-options-test',
        'kvmem-server-progress-test', 'kvmem-output-limit-test', 'kvmem-responses-test')
}
Invoke-Checked cmake (@('--build', $BuildDir, '--parallel', "$Jobs", '--target') + $targets)
if (!$BuildOnly) {
    Invoke-Checked ctest @('--test-dir', $BuildDir, '--output-on-failure', '-R',
        '^(kvmem_store_test|pinned_kv_tier_test|nvme_disabled_test|kvmem_runtime_test|raw_kv_store_test|kvmem-chat-id-test|kvmem-reasoning-budget-test|kvmem-chat-template-test|kvmem-server-options-test|kvmem-server-progress-test|kvmem-output-limit-test|kvmem-responses-test)$')
    Write-Host "Built and tested: $BuildDir"
} else { Write-Host "Built only; runtime tests NOT run: $BuildDir" }
