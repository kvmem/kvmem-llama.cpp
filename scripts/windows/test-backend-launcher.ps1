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
    $common = @{ Worker = (Join-Path $testDir 'worker.exe'); Gpu = 'GPU-test'; DryRun = $true; WarningAction = 'SilentlyContinue' }
    $native = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -WorkerArgs @('--chat-template', 'template with spaces.jinja') | ConvertFrom-Json
    if ($native.backend -ne 'ninfer' -or $native.argv[1] -ne (Join-Path $testDir 'model with spaces.ninfer') -or
        $native.argv[-1] -ne 'template with spaces.jinja' -or
        $native.environment.CUDA_VISIBLE_DEVICES -ne 'GPU-test' -or !$native.capabilities.multimodal) {
        throw 'ninfer dry-run lost model, argv, device or capability boundaries'
    }
    if ($native.capabilities.disk_restore -or $native.capabilities.supported_combinations.cold_disk.enabled -or
        $native.capabilities.supported_combinations.adaptive_mtp.cold_disk -or
        $native.argv -contains '--kvmem-disk-path' -or $native.argv -contains '--kvmem-disk-mib') {
        throw 'ninfer advertised or enabled disabled cold snapshots'
    }
    if ($native.capabilities.supported_combinations.lookup_ngram.enabled -or
        $native.capabilities.supported_combinations.lookup_ngram.reason -ne 'legacy_path_disabled') {
        throw 'ninfer advertised disabled legacy lookup drafting'
    }
    foreach ($lookupArgs in @(
        @('--lookup-ngram', '4'), @('--lookup-ngram=4'),
        @('--lookup-ngram', '0'), @('--lookup-ngram=0'), @('--lookup-ngram')
    )) {
        foreach ($drafts in @(0, 3)) {
            $disabled = $false
            try {
                $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                    -MtpDrafts $drafts -NgramDrafts $(if ($drafts) { 15 } else { 0 }) -WorkerArgs $lookupArgs
            } catch {
                $disabled = $_.Exception.Message -match 'disables legacy --lookup-ngram.*-NgramDrafts'
            }
            if (!$disabled) { throw 'Launcher allowed disabled legacy lookup drafting' }
        }
    }
    $dualArgs = $common.Clone()
    $dualArgs.Gpu = ' GPU-second , GPU-first '
    $dual = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -StageLayers 30,34 -Vision -VisionResidency cpu -MtpDrafts 3 | ConvertFrom-Json
    if ($dual.environment.CUDA_VISIBLE_DEVICES -ne 'GPU-second,GPU-first' -or
        $dual.argv -contains '--device' -or
        $dual.argv[[array]::IndexOf($dual.argv, '--devices') + 1] -ne '0,1' -or
        $dual.argv[[array]::IndexOf($dual.argv, '--stage-layers') + 1] -ne '30,34' -or
        $dual.capabilities.supported_combinations.multi_gpu.mode -ne 'layer') {
        throw 'Multi-GPU launcher lost UUID order, stage split or capability'
    }
    foreach ($badGpu in @('GPU-test,GPU-test', 'GPU-test,', ',GPU-test', '-1,GPU-test', '0,1,2,3,4,5,6,7,8')) {
        $argsBad = $common.Clone(); $argsBad.Gpu = $badGpu
        $rejected = $false
        try { $null = & $launcher @argsBad -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') }
        catch { $rejected = $_.Exception.Message -match 'distinct GPU UUIDs' }
        if (!$rejected) { throw 'Invalid multi-GPU list accepted' }
    }
    foreach ($badSplit in @(@(64), @(0,64), @(20,20,24))) {
        $rejected = $false
        try { $null = & $launcher @dualArgs -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') -StageLayers $badSplit }
        catch { $rejected = $_.Exception.Message -match 'one positive layer count per GPU' }
        if (!$rejected) { throw 'Invalid stage layer counts accepted' }
    }
    foreach ($diskCase in @(
        @{DiskPath='cache with spaces'; DiskMiB=1024}, @{DiskPath=''}, @{DiskMiB=4096},
        @{WorkerArgs=@('--kvmem-disk-path', 'cache')},
        @{WorkerArgs=@('--kvmem-disk-mib', '0')},
        @{WorkerArgs=@('--kvmem-disk-path=cache')}
    )) {
        $disabled = $false
        try { $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') @diskCase }
        catch { $disabled = $_.Exception.Message -match 'cold snapshots are temporarily disabled' }
        if (!$disabled) { throw 'Launcher allowed disabled ninfer cold snapshots' }
    }
    $mtp = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -Concurrency 2 -RetainedSessions 8 | ConvertFrom-Json
    if ($mtp.argv -notcontains '--ngram-draft-tokens' -or $mtp.argv -notcontains '--kvmem-sessions' -or
        $mtp.capabilities.supported_combinations.text.max_concurrency -ne 8) { throw 'MTP/concurrency mapping failed' }
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
        -KvType rk8v4 -Concurrency 4 -FastPrefill -MtpDrafts 15 -AdaptiveMtp | ConvertFrom-Json
    if ($fast.argv -notcontains '--fast-prefill-kernel') { throw 'Fast RK8V4 concurrency mapping failed' }
    foreach ($format in 'int8', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8') {
        foreach ($lanes in 1..8) {
            foreach ($residency in 'resident', 'cpu') {
                $fastImage = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                    -KvType $format -Concurrency $lanes -FastPrefill -Vision -VisionResidency $residency `
                    -MtpDrafts 3 -AdaptiveMtp -NgramDrafts 31 | ConvertFrom-Json
                if ($fastImage.argv -notcontains '--fast-prefill-kernel' -or
                    $fastImage.argv -notcontains '--vision' -or
                    $fastImage.argv[[array]::IndexOf($fastImage.argv, '--kv-dtype') + 1] -ne $format -or
                    $fastImage.argv[[array]::IndexOf($fastImage.argv, '--ngram-draft-tokens') + 1] -ne "$(if($lanes -gt 1){15}else{31})" -or
                    $fastImage.capabilities.supported_combinations.fast_prefill.text_only -or
                    $fastImage.capabilities.supported_combinations.fast_prefill.max_concurrency -ne 8 -or
                    $fastImage.capabilities.supported_combinations.fast_prefill.kv -notcontains $format) {
                    throw 'Fast image/concurrency/ngram mapping or capability failed'
                }
            }
        }
    }
    $cpu = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -KvType rk8v4 -Vision -VisionResidency cpu -MtpDrafts 15 -AdaptiveMtp | ConvertFrom-Json
    if ($cpu.argv[[array]::IndexOf($cpu.argv, '--vision-residency') + 1] -ne 'cpu') { throw 'CPU Vision mapping failed' }
    foreach ($residency in 'resident', 'cpu') {
        foreach ($lanes in 1..8) {
            $imageLanes = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -Vision -VisionResidency $residency -Concurrency $lanes -MtpDrafts 3 -AdaptiveMtp | ConvertFrom-Json
            if ($imageLanes.argv[[array]::IndexOf($imageLanes.argv, '--max-concurrency') + 1] -ne "$lanes" -or
                $imageLanes.argv[[array]::IndexOf($imageLanes.argv, '--vision-max-merged') + 1] -ne '1024' -or
                $imageLanes.argv[[array]::IndexOf($imageLanes.argv, '--ngram-draft-tokens') + 1] -ne '0' -or
                $imageLanes.capabilities.supported_combinations.images.max_concurrency -ne 8) {
                throw 'Concurrent Vision mapping or default preprocessing limit failed'
            }
        }
    }
    $copy = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -NgramDrafts 31 -KvType rk8v4 | ConvertFrom-Json
    if ($copy.argv[[array]::IndexOf($copy.argv, '--ngram-draft-tokens') + 1] -ne '31') {
        throw 'ngram mapping failed'
    }
    if ($copy.argv -contains '--default-thinking-budget') { throw 'thinking budget was enabled by default' }
    $budget = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -ThinkingBudget 8192 | ConvertFrom-Json
    if ($budget.argv[[array]::IndexOf($budget.argv, '--default-thinking-budget') + 1] -ne '8192') {
        throw 'thinking budget mapping failed'
    }
    $budgetRejected = $false
    try {
        $null = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
            -WorkerArgs @('--default-thinking-budget', '8192')
    } catch {
        $budgetRejected = $_.Exception.Message -match 'Use -ThinkingBudget'
    }
    if (!$budgetRejected) { throw 'native thinking-budget argument was accepted' }
    $llama = & $launcher @common -Backend llamacpp -Model (Join-Path $testDir 'model.gguf') `
        -WorkerArgs @('--spec-type', 'draft-mtp') | ConvertFrom-Json
    if ($llama.argv -notcontains 'draft-mtp' -or !$llama.capabilities.speculative_decoding -or
        !$llama.capabilities.disk_restore -or $llama.argv -contains '--kvmem-host-mib') { throw 'llama native options changed' }
    foreach ($residency in 'resident', 'cpu') {
        foreach ($adaptive in $false, $true) {
            $imageCopy = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                -MtpDrafts 4 -AdaptiveMtp:$adaptive -NgramDrafts 31 -Vision -VisionResidency $residency | ConvertFrom-Json
            if ($imageCopy.argv -notcontains '--vision' -or
                $imageCopy.capabilities.supported_combinations.ngram.text_only -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'k8v4' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'bf16' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'rk8v4' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'int8' -or
                $imageCopy.capabilities.supported_combinations.ngram.image_kv -notcontains 'nvfp4' -or
                $imageCopy.argv[[array]::IndexOf($imageCopy.argv, '--vision-residency') + 1] -ne $residency -or
                $imageCopy.argv[[array]::IndexOf($imageCopy.argv, '--ngram-draft-tokens') + 1] -ne '31' -or
                (($imageCopy.argv -contains '--adaptive-mtp') -ne $adaptive)) { throw 'Image/ngram mapping failed' }
        }
    }
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
        }
    }
    foreach ($lanes in 2..8) {
        foreach ($kv in 'bf16', 'int8', 'fp8', 'nvfp4', 'k8v4', 'rk8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8') {
            foreach ($width in 0, 1, 15, 16, 31, 63) {
                foreach ($adaptive in $false, $true) {
                    $copyLanes = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                        -Concurrency $lanes -KvType $kv -MtpDrafts 3 -AdaptiveMtp:$adaptive -NgramDrafts $width | ConvertFrom-Json
                    if ($copyLanes.argv[[array]::IndexOf($copyLanes.argv, '--ngram-draft-tokens') + 1] -ne "$([Math]::Min($width, 15))" -or
                        $copyLanes.capabilities.supported_combinations.ngram.max_concurrency -ne 8 -or
                        $copyLanes.capabilities.supported_combinations.ngram.concurrent_max_drafts -ne 15 -or
                        $copyLanes.capabilities.supported_combinations.ngram.concurrent_width_policy -ne 'clamp_to_15') {
                        throw 'Concurrent ngram mapping or capability failed'
                    }
                    foreach ($residency in 'resident', 'cpu') {
                        $image = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
                            -Concurrency $lanes -KvType $kv -MtpDrafts 3 -AdaptiveMtp:$adaptive -NgramDrafts $width `
                            -Vision -VisionResidency $residency | ConvertFrom-Json
                        if ($image.argv[[array]::IndexOf($image.argv, '--kv-dtype') + 1] -ne $kv -or
                            $image.argv[[array]::IndexOf($image.argv, '--ngram-draft-tokens') + 1] -ne "$([Math]::Min($width, 15))" -or
                            $image.capabilities.supported_combinations.ngram.image_kv -notcontains $kv) {
                            throw 'Concurrent image/ngram KV mapping failed'
                        }
                    }
                }
            }
        }
    }
    $clamped = & $launcher @common -Backend ninfer -Model (Join-Path $testDir 'model with spaces.ninfer') `
        -MtpDrafts 3 -NgramDrafts 31 -Concurrency 8 -WarningVariable normalizationWarnings | ConvertFrom-Json
    if ($clamped.argv[[array]::IndexOf($clamped.argv, '--ngram-draft-tokens') + 1] -ne '15' -or
        $normalizationWarnings.Count -ne 1 -or
        "$($normalizationWarnings[0])" -notmatch '31 to 15.*Concurrency 8') {
        throw 'Concurrent ngram clamp must report the requested and effective widths once'
    }
    $bad = @(
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=16},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=-1},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model.gguf')},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Budget=385},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Reserve=64},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); Vision=$true; Concurrency=9},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); AdaptiveMtp=$true; MtpDrafts=0},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='k8v4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='bf16'; Vision=$true},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='fp8'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; KvType='nvfp4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); FastPrefill=$true; DiskPath='cache'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvfp4'; Vision=$true; Concurrency=9},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvfp4'; NgramDrafts=31},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='unknown-kv'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); KvType='nvpf4'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); DiskPath='cache'; DeviceProfile='auto'},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); NgramDrafts=31},
        @{Backend='ninfer'; Model=(Join-Path $testDir 'model with spaces.ninfer'); MtpDrafts=3; NgramDrafts=64; Concurrency=2},
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
