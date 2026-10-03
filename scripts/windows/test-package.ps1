#Requires -Version 5.1
# Structural packaging test using an existing PE; no model or generated EXE runs.
param([Parameter(Mandatory)][string]$TestExe)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$temp = Join-Path ([IO.Path]::GetTempPath()) ('kvmem-package-' + [guid]::NewGuid().ToString('N'))
function Check([bool]$Ok, [string]$Message) { if (!$Ok) { throw $Message } }
try {
    $source = Join-Path $temp 'source'
    $build = Join-Path $temp 'build'
    $package = Join-Path $temp 'package'
    foreach ($dir in 'source/scripts/windows', 'source/backends/llamacpp/vendor', 'build/bin', 'ui') {
        $null = New-Item -ItemType Directory -Path (Join-Path $temp $dir) -Force
    }
    foreach ($name in 'native-process.ps1', 'start-server.ps1', 'start-iq3.ps1', 'start-iq4.ps1', 'README.md', 'README-quantizer.md') {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $source 'scripts/windows')
    }
    [IO.File]::WriteAllText((Join-Path $source 'VERSION'), 'test-fixture')
    [IO.File]::WriteAllText((Join-Path $source 'README.md'), 'Packaging test fixture')
    [IO.File]::WriteAllText((Join-Path $source 'backends/llamacpp/LICENSE'), 'Packaging test fixture')
    [IO.File]::WriteAllText((Join-Path $temp 'ui/index.html'), '<title>Fixture UI</title>')
    foreach ($name in 'llama-kvmem-server.exe', 'llama-kvmem-cli.exe', 'llama-quantize.exe') {
        Copy-Item -LiteralPath $TestExe -Destination (Join-Path $build "bin/$name")
    }
    [IO.File]::WriteAllText((Join-Path $build 'CMakeCache.txt'), "KVMEM_ENABLE_NVME:BOOL=OFF`nCMAKE_BUILD_TYPE:STRING=Release`nGGML_CUDA:BOOL=OFF`n")
    $files = @{}
    foreach ($file in Get-ChildItem -LiteralPath $source -Recurse -File) {
        $files[$file.FullName.Substring($source.Length + 1).Replace('\', '/')] = (Get-FileHash $file.FullName).Hash
    }
    $manifest = Join-Path $temp 'manifest.json'
    [IO.File]::WriteAllText($manifest, (@{files = $files} | ConvertTo-Json -Depth 4))
    $archive = Join-Path $temp 'source.zip'
    Compress-Archive -LiteralPath $source -DestinationPath $archive
    $options = @{SourceDir=$source; BuildDir=$build; SourceManifest=$manifest; SourceArchive=$archive; OutputDir=$package; UiDir=(Join-Path $temp 'ui')}
    & (Join-Path $PSScriptRoot 'package.ps1') @options
    Check (Test-Path -LiteralPath "$package.zip") 'missing ZIP'
    Check (!(Test-Path -LiteralPath (Join-Path $package 'bin/llama-quantize.exe'))) 'runtime includes quantizer'
    Check (Test-Path -LiteralPath (Join-Path $package 'share/kvmem/ui/index.html')) 'missing UI'
    foreach ($line in Get-Content -LiteralPath (Join-Path $package 'SHA256SUMS')) {
        $hash, $name = $line -split '  ', 2
        Check ((Get-FileHash -LiteralPath (Join-Path $package $name)).Hash -ieq $hash) "checksum mismatch: $name"
    }
    $info = Get-Content -LiteralPath (Join-Path $package 'BUILD-INFO.json') -Raw | ConvertFrom-Json
    Check (!$info.nvme_supported) 'incorrect NVMe capability'
    Check ($info.source_archive_sha256 -ieq (Get-FileHash $archive).Hash) 'source hash'
    $options.OutputDir = Join-Path $temp 'quantizer'
    $options.Component = 'Quantizer'
    & (Join-Path $PSScriptRoot 'package.ps1') @options
    Check (Test-Path -LiteralPath (Join-Path $options.OutputDir 'bin/llama-quantize.exe')) 'missing quantizer'
    Check (!(Test-Path -LiteralPath (Join-Path $options.OutputDir 'bin/llama-kvmem-server.exe'))) 'quantizer includes server'
    Check (!(Test-Path -LiteralPath (Join-Path $options.OutputDir 'share/kvmem/ui/index.html'))) 'quantizer includes UI'
    [IO.File]::AppendAllText((Join-Path $source 'VERSION'), 'changed')
    $options.OutputDir = Join-Path $temp 'bad-package'
    $failed = $false
    try { & (Join-Path $PSScriptRoot 'package.ps1') @options } catch { $failed = $true }
    Check $failed 'source drift must fail'
    Write-Output 'Windows package: PE dependency scan, ZIP/UI/checksums and source-drift rejection passed'
} finally {
    $resolved = [IO.Path]::GetFullPath($temp)
    $base = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($base, [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($resolved) -notlike 'kvmem-package-*') { throw 'Unsafe cleanup path' }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
