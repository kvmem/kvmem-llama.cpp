$ErrorActionPreference = 'Continue'
# Run from the repository root regardless of where this is invoked from, rather than a hardcoded
# path: these are committed, and the next person's checkout will not be at C:\ninfer-fork.
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

# Both overridable so this is not welded to one machine. The output directory is created here
# because it does not exist in a clean checkout -- profiles/ is gitignored.
. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null
# Host memory pressure produces plausible numbers rather than an error, so it is guarded
# rather than trusted -- see host-memory.ps1 for what goes wrong and why.
Assert-NInferHostMemory -Artifacts @("$modelDir\qwen3_8_27b.ninfer")


# Same corpus, mode and window as the three already recorded in README.md, so the new rows are
# directly comparable rather than a separate experiment: ninfer-ppl-1m-v1, --quick, 4096/2048.
# The three that were measured through the attention race #49 fixed. Override to add a control:
# NINFER_SWEEP_DTYPES=int8,fp8,nvfp4,k8v4 checks that this harness still reproduces README's int8
# figure, which is the only way to know a changed fp8 number means the kernel changed rather than
# the corpus, window or build having drifted underneath it.
$dtypes = if ($env:NINFER_SWEEP_DTYPES) {
  $env:NINFER_SWEEP_DTYPES -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else {
  @('fp8','nvfp4','k8v4')
}

foreach ($d in $dtypes) {
    $log = "$out\$d.log"
    # Clear any previous run's output first. ninfer-perplexity writes report.json under a
    # timestamped subdirectory of --output, so a rerun that fails leaves the earlier report in
    # place and the check below would pick it up and print it as a fresh result -- a stale number
    # reported as a new measurement, which is the one failure mode a sweep must not have.
    Remove-Item -Recurse -Force "$out\$d" -ErrorAction SilentlyContinue
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & .\build-ninja\apps\ninfer-perplexity.exe `
        "$modelDir\qwen3_8_27b.ninfer" `
        --corpus 'eval\corpora\perplexity-1m\manifest.json' --quick `
        --context 4096 --stride 2048 --kv-dtype $d `
        --output "$out\$d" --log-level warning > $log 2>&1
    $code = $LASTEXITCODE
    $report = Get-ChildItem -Recurse -Filter report.json "$out\$d" -ErrorAction SilentlyContinue | Select-Object -First 1
    # Exit code as well as the report's existence: a run that produced a report and then failed
    # should be reported as a failure, not quietly read.
    if ($code -eq 0 -and $report) {
        $j = Get-Content $report.FullName -Raw | ConvertFrom-Json
        "{0,-7} ppl={1}  tokens={2}  {3}s" -f $d, $j.overall.perplexity, $j.overall.scored_tokens, [math]::Round($sw.Elapsed.TotalSeconds,0)
    } else {
        "{0,-7} FAILED (exit {1})" -f $d, $code
        Get-Content $log -Tail 3 | ForEach-Object { "        $_" }
    }
}
