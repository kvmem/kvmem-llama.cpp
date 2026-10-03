#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$SourceDir,
    [string]$BuildDir,
    [string]$Python = 'python',
    [string]$CudaPath = $env:CUDA_PATH,
    [switch]$ExperimentalCuda129,
    [switch]$Vulkan,
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
# Prefer English diagnostics; installations without English resources use a local language.
$env:VSLANG = '1033'
$null = New-Item -ItemType Directory -Path $BuildDir -Force
$probeSource = Join-Path $BuildDir 'kvmem-msvc-probe.c'
$probeHeader = Join-Path $BuildDir 'kvmem-msvc-probe.h'
$probeObject = Join-Path $BuildDir 'kvmem-msvc-probe.obj'
[IO.File]::WriteAllText($probeSource, '#include "kvmem-msvc-probe.h"')
[IO.File]::WriteAllText($probeHeader, '')
$probeLines = & cl /nologo /showIncludes /c $probeSource "/Fo$probeObject" 2>&1
if ($LASTEXITCODE -ne 0) { throw "MSVC include-prefix probe failed: $probeLines" }
$includePrefix = $null
foreach ($line in $probeLines) {
    if ("$line" -match '^(.+?)[A-Za-z]:[\\/].*kvmem-msvc-probe\.h$') {
        $includePrefix = $Matches[1].TrimEnd() + ' '
        break
    }
}
if (!$includePrefix) { throw 'Cannot detect the MSVC include dependency prefix' }
$resetCppDependencies = $false
$ninjaRules = Join-Path $BuildDir 'CMakeFiles/rules.ninja'
if (Test-Path -LiteralPath $ninjaRules) {
    $previousPrefix = [regex]::Match([IO.File]::ReadAllText($ninjaRules), '(?m)^msvc_deps_prefix = (.*)$')
    if ($previousPrefix.Success) {
        $resetCppDependencies = $previousPrefix.Groups[1].Value.TrimEnd() -ne $includePrefix.TrimEnd()
    }
}
$rulesOverride = Join-Path $BuildDir 'kvmem-msvc-rules.cmake'
$escapedPrefix = $includePrefix.Replace('\', '\\').Replace('"', '\"')
[IO.File]::WriteAllText($rulesOverride, "set(CMAKE_CL_SHOWINCLUDES_PREFIX `"$escapedPrefix`")`n", [Text.UTF8Encoding]::new($false))


if (!$HostOnly) {
    Invoke-Checked $Python @((Join-Path $SourceDir 'scripts/prepare-backends.py'), '--backend', 'llamacpp')
}

$options = @('-S', $SourceDir, '-B', $BuildDir, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=cl',
    "-DCMAKE_USER_MAKE_RULES_OVERRIDE=$rulesOverride",
    '-DBUILD_SHARED_LIBS=OFF', '-DKVMEM_ENABLE_NVME=OFF')
if (!$HostOnly) {
    $options += @('-DCMAKE_C_COMPILER=cl', '-DGGML_BACKEND_DL=OFF',
    '-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON',
    '-DGGML_F16C=ON', '-DGGML_BMI2=ON', '-DGGML_AVX512=OFF',
    '-DGGML_CUDA_FA_QUANTS=all')
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
    if ($Vulkan) {
        $vulkanSdk = $env:VULKAN_SDK
        if (!$vulkanSdk) { $vulkanSdk = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine') }
        if (!$vulkanSdk -or !(Test-Path -LiteralPath (Join-Path $vulkanSdk 'Bin/glslc.exe')) -or
            !(Test-Path -LiteralPath (Join-Path $vulkanSdk 'Include/spirv/unified1/spirv.hpp'))) {
            throw 'Install the complete Vulkan SDK and set VULKAN_SDK before using -Vulkan'
        }
        $env:VULKAN_SDK = $vulkanSdk
        $env:PATH = "$(Join-Path $vulkanSdk 'Bin');$env:PATH"
        $options += '-DGGML_VULKAN=ON'
    }
}
Invoke-Checked cmake $options
if ($resetCppDependencies) {
    Write-Host 'MSVC dependency prefix changed; rebuilding C/C++ objects to record valid include dependencies.'
    foreach ($object in Get-ChildItem -LiteralPath $BuildDir -Recurse -File -Filter '*.obj') {
        if ($object.Name -like '*.cu.obj') { continue }
        if (!$object.FullName.StartsWith($BuildDir.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Object path is outside the build directory'
        }
        Remove-Item -LiteralPath $object.FullName
    }
}
$targets = @('backend_rebind_test', 'kvmem-lane-pool-test', 'kvmem_store_test', 'pinned_kv_tier_test',
    'kvmem-conversation-store-test', 'kvmem-session-snapshot-test',
    'kvmem-session-transfer-test', 'kvmem-session-cache-lifecycle-worker')
if (!$HostOnly) {
    $targets += @('llama-kvmem-server', 'llama-kvmem-cli', 'llama-quantize',
        'kvmem-chat-id-test', 'kvmem-reasoning-budget-test', 'kvmem-chat-template-test', 'kvmem-server-options-test',
        'kvmem-server-progress-test', 'kvmem-output-limit-test', 'kvmem-responses-test', 'kvmem-gdn-output-fusion-test')
}
Invoke-Checked cmake (@('--build', $BuildDir, '--parallel', "$Jobs", '--target') + $targets)
if (!$BuildOnly) {
    Invoke-Checked ctest @('--test-dir', $BuildDir, '--output-on-failure', '-R',
        '^(backend_rebind_test|kvmem-lane-pool-test|kvmem_store_test|pinned_kv_tier_test|kvmem-conversation-store-test|kvmem-session-snapshot-test|kvmem-session-transfer-test|kvmem-session-cache-lifecycle-test|kvmem-chat-id-test|kvmem-reasoning-budget-test|kvmem-chat-template-test|kvmem-server-options-test|kvmem-server-progress-test|kvmem-output-limit-test|kvmem-responses-test|kvmem-gdn-output-fusion-test)$')
    Write-Host "Built and tested: $BuildDir"
} else { Write-Host "Built only; runtime tests NOT run: $BuildDir" }
