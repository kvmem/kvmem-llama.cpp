#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$LlamaWorker,
    [Parameter(Mandatory)][string]$NinferWorker,
    [Parameter(Mandatory)][string]$NinferSource,
    [Parameter(Mandatory)][string]$OutputDir,
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$VcpkgInstalled,
    [string]$ValidationReport
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
$null = Get-Command dumpbin -ErrorAction Stop
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$NinferSource = (Resolve-Path -LiteralPath $NinferSource).Path
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $OutputDir) { throw 'OutputDir must be new' }
if (!$CudaPath -or !(Test-Path -LiteralPath (Join-Path $CudaPath 'EULA.txt'))) { throw 'CUDA Toolkit and EULA required' }
foreach ($worker in $LlamaWorker, $NinferWorker) {
    if (!(Test-Path -LiteralPath $worker -PathType Leaf)) { throw "Missing worker: $worker" }
}
foreach ($dir in 'workers/llamacpp', 'workers/ninfer', 'scripts/windows', 'licenses', 'provenance') {
    $null = New-Item -ItemType Directory -Path (Join-Path $OutputDir $dir) -Force
}
$workerInfo = @{}
foreach ($entry in @(@('llamacpp', $LlamaWorker, 'llama-kvmem-server.exe'),
                    @('ninfer', $NinferWorker, 'ninfer-kvmem-server.exe'))) {
    $backend = $entry[0]; $inputFile = (Resolve-Path -LiteralPath $entry[1]).Path
    $destination = Join-Path $OutputDir "workers/$backend"
    $executable = Join-Path $destination $entry[2]
    Copy-Item -LiteralPath $inputFile -Destination $executable
    $search = @((Split-Path $inputFile), (Join-Path $CudaPath 'bin'), (Join-Path $CudaPath 'bin/x64'))
    if ($VcpkgInstalled) { $search += Join-Path $VcpkgInstalled 'x64-windows/bin' }
    $search += $env:PATH.Split(';')
    $queue = [Collections.Generic.Queue[string]]::new(); $queue.Enqueue($executable)
    $seen = @{}
    # Backends may load local plugins at runtime, outside the PE import table.
    foreach ($dll in Get-ChildItem -LiteralPath (Split-Path $inputFile) -Filter '*.dll' -File) {
        $copy = Join-Path $destination $dll.Name
        Copy-Item -LiteralPath $dll.FullName -Destination $copy
        $seen[$dll.Name] = $true; $queue.Enqueue($copy)
    }
    while ($queue.Count) {
        $file = $queue.Dequeue()
        $dependencies = & dumpbin /nologo /dependents $file
        if ($LASTEXITCODE -ne 0) { throw "dumpbin failed: $file" }
        foreach ($line in $dependencies) {
            if ($line -notmatch '^\s+([\w.+-]+\.dll)\s*$') { continue }
            $name = $Matches[1]
            if ($seen.ContainsKey($name)) { continue }; $seen[$name] = $true
            if ($name -match '^(api-ms-|ext-ms-)' -or
                (Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32/$name"))) { continue }
            $found = $null
            foreach ($dir in $search) {
                if (!$dir) { continue }
                $candidate = Join-Path $dir $name
                if (Test-Path -LiteralPath $candidate -PathType Leaf) { $found = $candidate; break }
            }
            if (!$found) { throw "Unresolved dependency $name from $file" }
            $copy = Join-Path $destination $name
            Copy-Item -LiteralPath $found -Destination $copy; $queue.Enqueue($copy)
        }
    }
    $info = New-KVMemProcessInfo $executable @('--help')
    $info.EnvironmentVariables['PATH'] = "$env:SystemRoot/System32;$env:SystemRoot"
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($info)
    $stdout = $process.StandardOutput.ReadToEndAsync(); $stderr = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    [IO.File]::WriteAllText((Join-Path $OutputDir "provenance/$backend-help.txt"), $stdout.Result + $stderr.Result)
    if ($process.ExitCode -ne 0) { throw "$backend failed with a clean runtime PATH" }
    $process.Dispose()
    $workerInfo[$backend] = @{ sha256=(Get-FileHash -LiteralPath $executable).Hash.ToLowerInvariant();
        path="workers/$backend/$($entry[2])"; clean_path_help_passed=$true }
}
foreach ($script in 'start-backend.ps1', 'start-ninfer.ps1', 'native-process.ps1') {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $script) -Destination (Join-Path $OutputDir 'scripts/windows')
}
Copy-Item -LiteralPath (Join-Path $source 'docs') -Destination (Join-Path $OutputDir 'docs') -Recurse
$packageReadme = @'
# KVMem Windows workers

This package contains llama.cpp and ninfer workers, their runtime dependencies and
the common launcher. Models are supplied separately. Install the NVIDIA driver
and Microsoft Visual C++ 2015-2022 x64 Runtime before starting a worker.

See [launch and backend options](docs/multi-backend.md) and
[active RAM/SSD storage](docs/active-kv-tiering.md). Use `-Describe` to inspect
launcher capabilities, and `-DryRun` to inspect the exact worker arguments.

```powershell
./scripts/windows/start-backend.ps1 -Backend ninfer -Model 'D:/models/model.ninfer' -DryRun
```

`-HostMiB`, `-DiskPath` and `-DiskMiB` configure shared native KV RAM and process-local
SSD storage for either backend. SSD contents live only for the worker lifetime;
they cannot restore a session after restart or transfer it between backends.

`BUILD-INFO.json` records the source versions and dependency-loading checks.
`provenance/validation.json`, when present, records the actual runtime acceptance
scope. Dependency packaging alone does not qualify every model or configuration.
'@
[IO.File]::WriteAllText((Join-Path $OutputDir 'README.md'), $packageReadme, [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $CudaPath 'EULA.txt') -Destination (Join-Path $OutputDir 'licenses/CUDA-EULA.txt')
Copy-Item -LiteralPath (Join-Path $source 'backends/llamacpp/LICENSE') -Destination (Join-Path $OutputDir 'licenses/llama-MIT.txt')
Copy-Item -LiteralPath (Join-Path $source 'README.md') -Destination (Join-Path $OutputDir 'licenses/KVMem-README.md')
Copy-Item -LiteralPath (Join-Path $NinferSource 'LICENSE') -Destination (Join-Path $OutputDir 'licenses/ninfer-LICENSE')
foreach ($tree in @(@('ninfer-third-party', (Join-Path $NinferSource 'third_party')),
                   @('llama-vendor', (Join-Path $source 'backends/llamacpp/vendor')))) {
    foreach ($file in Get-ChildItem -LiteralPath $tree[1] -Recurse -File) {
        if ($file.Name -notmatch '^(LICENSE|COPYING|NOTICE)') { continue }
        $relative = $file.FullName.Substring($tree[1].Length).TrimStart('\')
        $dest = Join-Path $OutputDir "licenses/$($tree[0])/$relative"
        $null = New-Item -ItemType Directory -Path (Split-Path $dest) -Force
        Copy-Item -LiteralPath $file.FullName -Destination $dest
    }
}
if ($VcpkgInstalled) {
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $VcpkgInstalled 'x64-windows/share') -Filter copyright -Recurse -File) {
        $dest = Join-Path $OutputDir ('licenses/vcpkg/' + $file.Directory.Name + '.txt')
        $null = New-Item -ItemType Directory -Path (Split-Path $dest) -Force
        Copy-Item -LiteralPath $file.FullName -Destination $dest
    }
}
if ($ValidationReport) { Copy-Item -LiteralPath $ValidationReport -Destination (Join-Path $OutputDir 'provenance/validation.json') }
$manifest = @{ workers=$workerInfo; models_included=$false; kv_cross_backend_transfer=$false;
    backend_sources=(Get-Content -LiteralPath (Join-Path $source 'backends/versions.json') -Raw | ConvertFrom-Json);
    runtime_validation='See attached validation report; packaging certifies dependency loading only' }
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText((Join-Path $OutputDir 'BUILD-INFO.json'), ($manifest | ConvertTo-Json -Depth 6), $utf8)
$hashes = foreach ($file in Get-ChildItem -LiteralPath $OutputDir -Recurse -File | Sort-Object FullName) {
    (Get-FileHash -LiteralPath $file.FullName).Hash.ToLowerInvariant() + '  ' +
        $file.FullName.Substring($OutputDir.Length + 1).Replace('\', '/')
}
[IO.File]::WriteAllText((Join-Path $OutputDir 'SHA256SUMS'), ($hashes -join "`n") + "`n", $utf8)
Write-Output "Prepared unpacked package: $OutputDir"
