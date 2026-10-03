# Build the Windows release archive for the version in the repository's VERSION file.
#
#   $env:NINFER_BUILD_ROOT = 'C:\ninfer\build-ninja'; .\scripts\package-release.ps1
#   .\scripts\build.ps1 -Package         configure + build, then this
#
# There is one packager, not one per release. Everything version-specific is derived from VERSION,
# whose content is the full release tag (for example 0.10.0-rtx3090): the text before the first '-'
# names RELEASE_NOTES_<version>.md and the checksum file, and the whole tag names the archive. To cut
# a release, bump VERSION and write its release notes; there is nothing here to copy and edit.
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot

$VersionFile = Join-Path $RepoRoot 'VERSION'
if (-not (Test-Path -LiteralPath $VersionFile)) { throw "Missing $VersionFile" }
$ReleaseTag = (Get-Content -LiteralPath $VersionFile -Raw).Trim()
if (-not $ReleaseTag) { throw "$VersionFile is empty" }
$ReleaseVersion = $ReleaseTag.Split('-')[0]
$ReleaseNotes = "RELEASE_NOTES_$ReleaseVersion.md"
# Checked before anything is deleted or built into dist, so a forgotten release-notes file costs
# nothing rather than a finished archive that has to be thrown away.
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $ReleaseNotes))) {
    throw "Missing $(Join-Path $RepoRoot $ReleaseNotes): write the release notes for $ReleaseTag before packaging it"
}

$BuildRoot = if ($env:NINFER_BUILD_ROOT) { $env:NINFER_BUILD_ROOT } else { Join-Path $RepoRoot 'build-ninja' }
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductName = "ninfer-rtx3090-windows-x64-$ReleaseTag"
$ProductRoot = Join-Path $DistRoot $ProductName
$ArchivePath = Join-Path $DistRoot "$ProductName.zip"
$ChecksumPath = Join-Path $DistRoot "SHA256SUMS-v$ReleaseVersion-windows.txt"

# Ninja is a single-config generator, so release binaries land directly under
# apps\ / bench\ rather than an apps\Release\ subdirectory.
$Products = @(
    @{ Source = 'apps\ninfer.exe'; Destination = 'ninfer.exe' },
    @{ Source = 'apps\ninfer-serve.exe'; Destination = 'ninfer-serve.exe' },
    @{ Source = 'bench\ninfer_bench.exe'; Destination = 'ninfer_bench.exe' }
)

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
$resolvedDist = (Resolve-Path -LiteralPath $DistRoot).Path
$resolvedProductParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $ProductRoot))
if ($resolvedProductParent -ne $resolvedDist -or (Split-Path -Leaf $ProductRoot) -ne $ProductName) {
    throw "Refusing to package outside the expected dist directory: $ProductRoot"
}
if (Test-Path -LiteralPath $ProductRoot) { Remove-Item -LiteralPath $ProductRoot -Recurse -Force }
if (Test-Path -LiteralPath $ArchivePath) { Remove-Item -LiteralPath $ArchivePath -Force }
# The checksum goes too, and before anything can fail. It is written last, so leaving a previous
# run's copy in place means a mid-run failure can leave `dist` holding a new archive beside a
# checksum for the old one -- and publishing that pair is worse than publishing neither.
if (Test-Path -LiteralPath $ChecksumPath) { Remove-Item -LiteralPath $ChecksumPath -Force }
New-Item -ItemType Directory -Path $ProductRoot | Out-Null

foreach ($product in $Products) {
    $source = Join-Path $BuildRoot $product.Source
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing release product: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $ProductRoot $product.Destination)
}
Get-ChildItem -LiteralPath (Join-Path $BuildRoot 'apps') -Filter '*.dll' | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $ProductRoot
}
# The executables import cublas64_12.dll (the cuBLAS prefill route), which imports cublasLt64_12.dll.
# Neither is in the vcpkg apps\ directory: they come from the CUDA Toolkit, which a typical Windows
# machine does not have, so without them every executable here fails to start with "cublas64_12.dll
# was not found" -- before --prefill-cublas is even asked for. NVIDIA lists both as redistributable in
# the CUDA Toolkit EULA. Take them from the toolkit that built the binaries.
$CudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { $null }
foreach ($name in 'cublas64_12.dll', 'cublasLt64_12.dll') {
    $found = if ($CudaBin) { Join-Path $CudaBin $name } else { $null }
    if (-not $found -or -not (Test-Path -LiteralPath $found)) {
        throw "Missing ${name}: set CUDA_PATH to the CUDA 12.x Toolkit that built the release (looked in $CudaBin)"
    }
    Copy-Item -LiteralPath $found -Destination $ProductRoot
}
# NVIDIA's grant to redistribute those two DLLs (Attachment A of the EULA: cublas.dll and cublasLt.dll,
# including files whose names carry a version such as cublas64_12.dll) is conditional: the
# distribution must be consistent with the EULA and protect NVIDIA's rights. Its text therefore ships
# beside them, and the archive README says which files it covers.
$Eula = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'EULA.txt' } else { $null }
if (-not $Eula -or -not (Test-Path -LiteralPath $Eula)) {
    throw "Missing the CUDA Toolkit EULA (looked for $Eula): it must ship with the cuBLAS DLLs"
}
Copy-Item -LiteralPath $Eula -Destination (Join-Path $ProductRoot 'NVIDIA-CUDA-EULA.txt')
Copy-Item -LiteralPath (Join-Path $RepoRoot 'VERSION') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $ProductRoot
# The archive README must describe the archive. docs\rtx-3090-windows.md is written for a checkout
# -- it points at scripts\download-model.bat, and the packager copies that script to the archive
# root -- so a user following it from inside the archive got a missing-file error.
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs\release-archive-windows.md') -Destination (Join-Path $ProductRoot 'README.md')
Copy-Item -LiteralPath (Join-Path $RepoRoot $ReleaseNotes) -Destination $ProductRoot
# Explicit rather than matched by pattern, so a new script is shipped only once someone has decided
# it belongs in the archive.
foreach ($script in @('run.bat', 'download-model.bat')) {
    $source = Join-Path $RepoRoot "scripts\$script"
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing release script: $source" }
    Copy-Item -LiteralPath $source -Destination $ProductRoot
}

# A missing DLL only shows up on a machine that lacks it, and this one has the CUDA Toolkit and the
# vcpkg tree on its PATH -- which is how the executables nearly shipped needing cublas64_12.dll, so
# that none of them started on a PC without the Toolkit. Start each one from the folder about to be
# archived with nothing but Windows on PATH: a DLL the archive does not carry stops it (0xC0000135)
# and stops the release here.
$CleanPath = "$env:SystemRoot\System32;$env:SystemRoot"
foreach ($exe in ($Products | ForEach-Object { $_.Destination })) {
    $null = cmd /c "set PATH=$CleanPath&& `"$(Join-Path $ProductRoot $exe)`" --help 2>&1"
    if ($LASTEXITCODE -ne 0) {
        throw "$exe did not start with only Windows on PATH (exit $LASTEXITCODE): a DLL it needs is missing from the archive or cannot be loaded"
    }
}

$innerHashes = Get-ChildItem -LiteralPath $ProductRoot -File | Sort-Object Name | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($_.Name)"
}
# LF, not CRLF. `Set-Content` writes CRLF on Windows, and `sha256sum -c` then looks for a file whose
# name ends in a carriage return: "avcodec-62.dll: FAILED open or read" for every line. The Linux
# archive's list is written by sha256sum itself and has always been LF, so this also makes the two
# archives' checksum files interchangeable. Get-FileHash comparisons are unaffected either way.
[IO.File]::WriteAllText((Join-Path $ProductRoot 'SHA256SUMS.txt'), (($innerHashes -join "`n") + "`n"), [Text.ASCIIEncoding]::new())
Compress-Archive -LiteralPath $ProductRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
$archiveHash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
[IO.File]::WriteAllText($ChecksumPath,
    "$($archiveHash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $ArchivePath)`n",
    [Text.ASCIIEncoding]::new())

Get-Item -LiteralPath $ArchivePath, $ChecksumPath |
    Select-Object Name, @{ Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }
