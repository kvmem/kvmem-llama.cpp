#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$launcher = Join-Path $PSScriptRoot 'start-backend.ps1'
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testDir = Join-Path $tempRoot ('kvmem-backend-test-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $testDir
try {
    foreach ($name in 'worker.exe', 'model with spaces.ninfer', 'model.gguf') {
        [IO.File]::WriteAllText((Join-Path $testDir $name), 'dry-run fixture')
    }
    $common = @{ Worker = (Join-Path $testDir 'worker.exe'); Gpu = 'GPU-test'; DryRun = $true }
    $native = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -WorkerArgs @('--chat-template', 'template with spaces.jinja') | ConvertFrom-Json
    if ($native.backend -ne 'ninfer' -or $native.argv[1] -ne (Join-Path $testDir 'model with spaces.ninfer') -or
        $native.argv[-1] -ne 'template with spaces.jinja' -or
        $native.environment.CUDA_VISIBLE_DEVICES -ne 'GPU-test' -or !$native.capabilities.multimodal) {
        throw 'ninfer dry-run lost model, argv, device or capability boundaries'
    }
    $mtp = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -Concurrency 2 -RetainedSessions 8 | ConvertFrom-Json
    if ($mtp.argv -notcontains '--ngram-draft-tokens' -or $mtp.argv -notcontains '--kvmem-sessions' -or
        $mtp.capabilities.supported_combinations.text.max_concurrency -ne 4) { throw 'MTP/concurrency mapping failed' }
    foreach ($drafts in @(1, 4, 5, 15)) {
        $fixed = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
            -MtpDrafts $drafts | ConvertFrom-Json
        if ($fixed.argv[[array]::IndexOf($fixed.argv, '--draft-tokens') + 1] -ne "$drafts" -or
            $fixed.capabilities.supported_combinations.text.mtp_drafts -notcontains $drafts) {
            throw 'Fixed MTP draft count mapping failed'
        }
    }
    $adaptive = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 15 -AdaptiveMtp | ConvertFrom-Json
    if ($adaptive.argv -notcontains '--adaptive-mtp') { throw 'Adaptive MTP mapping failed' }
    $fast = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -KvType rk8v4 -Concurrency 4 -FastPrefill | ConvertFrom-Json
    if ($fast.argv -notcontains '--fast-prefill-kernel') { throw 'Fast RK8V4 concurrency mapping failed' }
    $cpu = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -KvType rk8v4 -Vision -VisionResidency cpu | ConvertFrom-Json
    if ($cpu.argv[[array]::IndexOf($cpu.argv, '--vision-residency') + 1] -ne 'cpu') { throw 'CPU Vision mapping failed' }
    $cold = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -KvType rk8v4 -DiskPath 'cache with spaces' -DiskMiB 1024 | ConvertFrom-Json
    if ($cold.argv -notcontains 'cache with spaces' -or $cold.argv -notcontains 'rk8v4') { throw 'Cold cache/format mapping failed' }
    $copy = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -NgramDrafts 31 -KvType rk8v4 | ConvertFrom-Json
    if ($copy.argv[[array]::IndexOf($copy.argv, '--ngram-draft-tokens') + 1] -ne '31') {
        throw 'ngram mapping failed'
    }
    $llama = & $launcher @common -Backend llamacpp -Model (Join-Path $testDir 'model.gguf') `
        -WorkerArgs @('--spec-type', 'draft-mtp') | ConvertFrom-Json
    if ($llama.argv -notcontains 'draft-mtp' -or !$llama.capabilities.speculative_decoding -or
        $llama.argv -contains '--kvmem-host-mib') { throw 'llama native options changed' }
    foreach ($format in 'nvfp4', 'k8v4') {
        foreach ($drafts in 0..4) {
            foreach ($lanes in 1..4) {
                $nvfp4 = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                    -KvType $format -MtpDrafts $drafts -Concurrency $lanes | ConvertFrom-Json
                if ($nvfp4.argv[[array]::IndexOf($nvfp4.argv, '--kv-dtype') + 1] -ne $format -or
                    $nvfp4.capabilities.supported_combinations.text.kv -notcontains $format -or
                    ($drafts -and $nvfp4.argv[[array]::IndexOf($nvfp4.argv, '--ngram-draft-tokens') + 1] -ne '0')) {
                    throw 'NVFP4/K8V4 text/MTP mapping or disabled ngram settings failed'
                }
            }
            $image = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -KvType $format -MtpDrafts $drafts -Vision | ConvertFrom-Json
            if ($image.argv -notcontains '--vision' -or $image.capabilities.supported_combinations.images.kv -notcontains $format) {
                throw 'NVFP4/K8V4 resident-image mapping failed'
            }
            $snapshot = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -KvType $format -MtpDrafts $drafts -DiskPath 'nvfp4 cache with spaces' | ConvertFrom-Json
            if ($snapshot.argv -notcontains 'nvfp4 cache with spaces' -or $snapshot.argv -notcontains '--derive-session-keys') {
                throw 'NVFP4/K8V4 cold snapshot mapping failed'
            }
        }
    }
    $bad = @(
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=16},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=-1},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model.gguf')},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Budget=385},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Reserve=64},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Vision=$true; Concurrency=2},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); AdaptiveMtp=$true; MtpDrafts=15; Concurrency=2},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); AdaptiveMtp=$true; MtpDrafts=15; DiskPath='cache'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='k8v4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; Vision=$true},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; DiskPath='cache'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvfp4'; Vision=$true; Concurrency=2},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvfp4'; NgramDrafts=31},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='fp8'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvpf4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); DiskPath='cache'; DeviceProfile='auto'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); NgramDrafts=31},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=3; NgramDrafts=15; Concurrency=2},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=3; NgramDrafts=15; Vision=$true},
        @{Backend='llamacpp'; Model=(Join-Path $testDir 'model.gguf'); HostMiB=1024}
    )
    foreach ($argsCase in $bad) {
        $rejected = $false
        try { $null = & $launcher @common @argsCase } catch { $rejected = $true }
        if (!$rejected) { throw 'Invalid backend configuration was accepted' }
    }
    foreach ($backend in 'ninfer', 'llamacpp') {
        $description = & $launcher -Backend $backend -Describe | ConvertFrom-Json
        if ($description.backend -ne $backend -or !$description.text_chat) { throw 'Capability query failed' }
    }
    Write-Output 'Backend launcher PASS (argv, paths, capability selection, invalid combinations)'
} finally {
    $resolved = [IO.Path]::GetFullPath($testDir)
    if (!$resolved.StartsWith($tempRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notlike 'kvmem-backend-test-*') { throw 'Unexpected fixture directory' }
    Get-ChildItem -LiteralPath $resolved -File | ForEach-Object { Remove-Item -LiteralPath $_.FullName }
    Remove-Item -LiteralPath $resolved
}
