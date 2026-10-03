#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SourceDir,
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$SourceManifest,
    [Parameter(Mandatory)][string]$SourceArchive,
    [Parameter(Mandatory)][string]$OutputDir,
    [string]$CudaPath = $env:CUDA_PATH,
    [switch]$ExperimentalCuda129,
    [string]$CudaLicense,
    [string]$UiDir,
    [string]$ValidationReport,
    [ValidateSet('Runtime', 'Quantizer')][string]$Component = 'Runtime'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$null = Get-Command dumpbin -ErrorAction Stop # Run in x64 Developer PowerShell.
$SourceDir = (Resolve-Path -LiteralPath $SourceDir).Path
$BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
$SourceArchive = (Resolve-Path -LiteralPath $SourceArchive).Path
$SourceManifest = (Resolve-Path -LiteralPath $SourceManifest).Path
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $OutputDir) { throw 'OutputDir must be a new directory' }
if (Test-Path -LiteralPath "$OutputDir.zip") { throw 'Output ZIP already exists' }
$manifest = Get-Content -LiteralPath $SourceManifest -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    $path = [IO.Path]::GetFullPath((Join-Path $SourceDir $entry.Name))
    if (!$path.StartsWith($SourceDir.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid source manifest path' }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $entry.Value) {
        throw "Source differs from manifest: $($entry.Name)"
    }
}
$cache = Get-Content -LiteralPath (Join-Path $BuildDir 'CMakeCache.txt') -Raw
if ($cache -notmatch '(?m)^KVMEM_ENABLE_NVME:BOOL=OFF\r?$') { throw 'Expected an NVMe-disabled build' }
if ($cache -notmatch '(?m)^CMAKE_BUILD_TYPE:STRING=Release\r?$') { throw 'Expected a Release build' }
$cuda = $cache -match '(?m)^GGML_CUDA:BOOL=ON\r?$'
if ($cuda) {
    if (!$CudaPath) { throw 'CudaPath is required for CUDA builds' }
    if (!$CudaLicense) { $CudaLicense = Join-Path $CudaPath 'EULA.txt' }
    if (!(Test-Path -LiteralPath $CudaLicense -PathType Leaf)) { throw 'Supply the CUDA Toolkit license with -CudaLicense' }
}
if ($UiDir -and !(Test-Path -LiteralPath (Join-Path $UiDir 'index.html'))) { throw 'UiDir has no index.html' }
if ($ValidationReport) {
    $ValidationReport = (Resolve-Path -LiteralPath $ValidationReport).Path
    $null = Get-Content -LiteralPath $ValidationReport -Raw | ConvertFrom-Json
}
if (!$UiDir -and (Test-Path -LiteralPath (Join-Path $BuildDir 'share/kvmem/ui/index.html'))) {
    $UiDir = Join-Path $BuildDir 'share/kvmem/ui'
}
foreach ($dir in 'bin', 'licenses', 'provenance', 'scripts/windows') {
    $null = New-Item -ItemType Directory -Path (Join-Path $OutputDir $dir) -Force
}
$queue = [Collections.Generic.Queue[string]]::new()
$executables = @('llama-kvmem-server.exe', 'llama-kvmem-cli.exe')
if ($Component -eq 'Quantizer') { $executables = @('llama-quantize.exe'); $UiDir = $null }
foreach ($name in $executables) {
    $dest = Join-Path $OutputDir "bin/$name"
    Copy-Item -LiteralPath (Join-Path $BuildDir "bin/$name") -Destination $dest
    $queue.Enqueue($dest)
}
$search = @((Join-Path $BuildDir 'bin'))
if ($cuda) { $search += @((Join-Path $CudaPath 'bin'), (Join-Path $CudaPath 'bin/x64')) }
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$runtimeDlls = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
while ($queue.Count) {
    $file = $queue.Dequeue()
    $dependencies = & dumpbin /NOLOGO /DEPENDENTS $file
    if ($LASTEXITCODE -ne 0) { throw "dumpbin failed: $file" }
    foreach ($line in $dependencies) {
        if ($line -notmatch '^\s+([\w.\-]+\.dll)\s*$') { continue }
        $name = $Matches[1]
        if (!$seen.Add($name)) { continue }
        # The driver and Microsoft runtimes are supplied by their installers.
        if ($name -match '^(msvcp|vcruntime|vcomp|concrt)\d') { $null = $runtimeDlls.Add($name); continue }
        if ($name -ieq 'nvcuda.dll' -or $name -match '^(api-ms-|ext-ms-)') { continue }
        $found = $null
        foreach ($dir in $search) {
            $candidate = Join-Path $dir $name
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $found = $candidate; break }
        }
        if ($found) {
            $dest = Join-Path $OutputDir "bin/$name"
            Copy-Item -LiteralPath $found -Destination $dest
            $queue.Enqueue($dest)
        } elseif (!(Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32/$name"))) {
            throw "Unresolved DLL dependency: $name (from $file)"
        }
    }
}
if ($Component -eq 'Runtime') {
foreach ($name in 'native-process.ps1', 'start-server.ps1', 'start-iq3.ps1', 'start-iq4.ps1') {
    Copy-Item -LiteralPath (Join-Path $SourceDir "scripts/windows/$name") -Destination (Join-Path $OutputDir 'scripts/windows')
}
}
$readme = 'README.md'
if ($Component -eq 'Quantizer') { $readme = 'README-quantizer.md' }
Copy-Item -LiteralPath (Join-Path $SourceDir ('scripts/windows/' + $readme)) -Destination (Join-Path $OutputDir 'README.md')
Copy-Item -LiteralPath (Join-Path $SourceDir 'backends/llamacpp/LICENSE') -Destination (Join-Path $OutputDir 'licenses/llama.cpp-MIT.txt')
Copy-Item -LiteralPath (Join-Path $SourceDir 'README.md') -Destination (Join-Path $OutputDir 'licenses/KVMem-README.md')
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $SourceDir 'backends/llamacpp/vendor') -Recurse -File) {
    if ($file.Name -notmatch '^(LICENSE|COPYING|NOTICE)') { continue }
    $relative = $file.FullName.Substring($SourceDir.Length + 1)
    $dest = Join-Path $OutputDir "licenses/$relative"
    $null = New-Item -ItemType Directory -Path (Split-Path $dest) -Force
    Copy-Item -LiteralPath $file.FullName -Destination $dest
}
if ($cuda) { Copy-Item -LiteralPath $CudaLicense -Destination (Join-Path $OutputDir 'licenses/NVIDIA-CUDA.txt') }
if ($UiDir) {
    $null = New-Item -ItemType Directory -Path (Join-Path $OutputDir 'share/kvmem') -Force
    Copy-Item -LiteralPath $UiDir -Destination (Join-Path $OutputDir 'share/kvmem/ui') -Recurse
}
Copy-Item -LiteralPath $SourceManifest -Destination (Join-Path $OutputDir 'provenance/source-manifest.json')
if ($ValidationReport) { Copy-Item -LiteralPath $ValidationReport -Destination (Join-Path $OutputDir 'VALIDATION.json') }
$options = @{}
foreach ($line in $cache.Split([char]10)) {
    if ($line -match '^((?:CMAKE_(?:BUILD_TYPE|CUDA_ARCHITECTURES|CXX_COMPILER|CUDA_COMPILER)|BUILD_SHARED_LIBS|KVMEM_ENABLE_NVME|GGML_\w+)):[^=]+=(.*)') {
        $key = $Matches[1]
        $value = $Matches[2].TrimEnd([char]13)
        foreach ($pair in @(@($SourceDir, '<SOURCE_DIR>'), @($BuildDir, '<BUILD_DIR>'), @($CudaPath, '<CUDA_PATH>'))) {
            if ($pair[0]) { $value = $value.Replace($pair[0], $pair[1]).Replace($pair[0].Replace('\', '/'), $pair[1]) }
        }
        $options[$key] = $value
    }
}
$info = @{ component = $Component; status = 'experimental-windows-build'; version = (Get-Content (Join-Path $SourceDir 'VERSION') -Raw).Trim();
    source_archive_sha256 = (Get-FileHash -LiteralPath $SourceArchive -Algorithm SHA256).Hash.ToLowerInvariant();
    source_manifest_sha256 = (Get-FileHash -LiteralPath $SourceManifest -Algorithm SHA256).Hash.ToLowerInvariant();
    build_options = $options; nvme_supported = $false; models_included = $false;
    msvc_toolset = $env:VCToolsVersion; os_version = [Environment]::OSVersion.VersionString;
    cpu_requirement = 'x86_64 with AVX2, FMA, F16C and BMI2';
    external_msvc_runtime_dlls = @($runtimeDlls); gpu_runtime_validation = 'not certified by packaging' }
foreach ($key in 'base_commit', 'llama_commit') {
    if ($manifest.PSObject.Properties[$key]) { $info[$key] = $manifest.$key }
}
if ($cuda) {
    $version = & (Join-Path $CudaPath 'bin/nvcc.exe') --version
    if ($LASTEXITCODE -ne 0) { throw 'Could not record CUDA compiler version' }
    $versionText = $version -join "`n"
    if ($versionText -notmatch 'V(\d+\.\d+\.\d+)') { throw 'Unknown nvcc version' }
    $nvccVersion = $Matches[1]
    if ($ExperimentalCuda129) {
        if ($nvccVersion -ne '12.9.86') { throw 'Experimental CUDA 12.9 package requires nvcc 12.9.86' }
    } elseif ([version]$nvccVersion -lt [version]'13.2.86') { throw 'nvcc 13.2.86 or newer is required' }
    $info.experimental_cuda129 = [bool]$ExperimentalCuda129
    $compilerRecords = @(Get-ChildItem -LiteralPath (Join-Path $BuildDir 'CMakeFiles') -Filter CMakeCUDACompiler.cmake -Recurse -File)
    if (!$compilerRecords.Count) { throw 'Missing recorded CUDA compiler version' }
    foreach ($record in $compilerRecords) {
        $compilerText = Get-Content -LiteralPath $record.FullName -Raw
        if ($compilerText -notmatch 'set\(CMAKE_CUDA_COMPILER_VERSION "([^" ]+)"\)') { throw 'Invalid compiler record' }
        if ($Matches[1] -ne $nvccVersion) { throw 'Toolkit version differs from the configured build compiler' }
    }
    $info.cuda_compiler = $versionText
}
if ($ValidationReport) { $info.validation_report_sha256 = (Get-FileHash -LiteralPath $ValidationReport -Algorithm SHA256).Hash.ToLowerInvariant() }
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $OutputDir 'BUILD-INFO.json'), ($info | ConvertTo-Json -Depth 5), $utf8)
$lines = foreach ($file in Get-ChildItem -LiteralPath $OutputDir -Recurse -File | Sort-Object FullName) {
    $relative = $file.FullName.Substring($OutputDir.Length + 1).Replace('\', '/')
    (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $relative
}
[IO.File]::WriteAllLines((Join-Path $OutputDir 'SHA256SUMS'), [string[]]$lines, $utf8)
Compress-Archive -LiteralPath $OutputDir -DestinationPath "$OutputDir.zip"
[IO.File]::WriteAllText("$OutputDir.zip.sha256", (Get-FileHash -LiteralPath "$OutputDir.zip" -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + [IO.Path]::GetFileName("$OutputDir.zip") + "`n", $utf8)
Write-Host "Package: $OutputDir.zip; distribute the matching source archive separately."
