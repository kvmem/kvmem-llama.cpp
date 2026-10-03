# One step of the perplexity-drift bisect: build whatever is checked out, score int8, print it.
#
# TODO section 3: `int8` scored 4.343263 when the figures were published in 838c8b5d (2026-08-29)
# and 4.342425 at master. Same corpus, same window, same 261,167 scored tokens, and the artifact
# has not been reconverted since before the old measurement -- so something between the two changed
# arithmetic in some kernel.
#
# THERE ARE TWO TRANSITIONS IN THAT RANGE, NOT ONE. 66378f06 scores 4.342864374753507, a third
# value: 838c8b5d -> 66378f06 is -0.0092% and 66378f06 -> HEAD is -0.0101%. Bisect the two halves
# separately; a single bisect will converge on one of them and name a commit that cannot reproduce
# the whole drift.
#
# START THE BISECT `--first-parent`. The naive walk descends into the upstream catch-up branch,
# where CMake refuses with "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '86'" (e.g.
# 00f02055). Those commits are unbuildable on this card -- not flaky, unbuildable -- and
# --first-parent both avoids them and cuts the range from 477 commits to 96.
#
# COPY THIS FILE OUT OF THE REPO BEFORE YOU START, WITH ITS TWO HELPERS. `git bisect` checks out
# commits that predate them, which deletes them mid-run:
#
#   copy scripts\sweeps\ppl-bisect-step.ps1 scripts\sweeps\model-dir.ps1 ^
#        scripts\sweeps\host-memory.ps1 C:\bisect
#   powershell -NoProfile -File C:\bisect\ppl-bisect-step.ps1 -Repo C:\ninfer-fork\ninfer-3090
#
# Copying this file alone works only while the checked-out commit still has the helpers, and the
# earlier half of the range does not -- host-memory.ps1 was added 2026-09-10. Pass -Repo and the
# script Set-Locations there itself.
#
# Run from (or pointed at) a detached HEAD under `git bisect`, then classify:
#
#   4.343263...  -> old behaviour  -> git bisect good
#   4.342425...  -> new behaviour  -> git bisect bad
#   build failed -> git bisect skip
#
# Deliberately prints the full precision. The two values differ in the fifth decimal, and a
# four-figure print would make them look identical and the bisect would converge on nothing.
[CmdletBinding()]
param(
  [string]$Repo   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  # Left empty on purpose: resolved below through Get-NInferModelDir, the same contract every other
  # sweep here uses. A param() default cannot do it -- $PSScriptRoot is empty while a default is
  # being bound, so a repo-relative path written here resolves against the drive root and silently
  # yields C:\models. See the comment at the top of model-dir.ps1.
  [string]$Model  = '',
  [string]$Out    = 'profiles\ppl_bisect',
  [string]$VcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
  [string]$Cuda   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe'
)
$ErrorActionPreference = 'Continue'

# -Repo is used on both sides of a Set-Location -- the helper search below runs before the
# directory change, Get-NInferModelDir -Root after it -- so a relative value would mean two
# different directories. `-Repo repo` from a parent directory would find the helpers under
# .\repo\scripts\sweeps and then probe repo\repo\models for the artifact, reporting a perfectly
# good checkout as missing. Canonicalise once, here, against the directory the caller invoked from.
$resolvedRepo = Resolve-Path -LiteralPath $Repo -ErrorAction SilentlyContinue
if (-not $resolvedRepo) { Write-Error "-Repo does not exist: $Repo (resolved from $PWD)"; exit 1 }
$Repo = $resolvedRepo.Path

# Resolve the two shared helpers BEFORE Set-Location, and from either place they can legitimately
# live, because this file is meant to be run from a copy outside the repo (see the header):
#
#   * beside the copy -- $PSScriptRoot, which is the copy's own directory;
#   * in the checkout -- $Repo\scripts\sweeps, which works only when the checked-out commit
#     actually has them. It often will not: host-memory.ps1 was added 2026-09-10 and every commit
#     the earlier half of this bisect visits predates it. That is why depending on the repo copy
#     alone is wrong here, and why the copy's own directory is tried first.
#
# If neither has them, fail with the command to fix it. A bare `. "$PSScriptRoot\model-dir.ps1"`
# on a lone copy dies with "The term ... is not recognized", which says nothing about what to do.
$helperDirs = @($PSScriptRoot, (Join-Path $Repo 'scripts\sweeps'))
foreach ($helper in @('model-dir.ps1', 'host-memory.ps1')) {
  $found = $helperDirs | Where-Object { $_ } |
           ForEach-Object { Join-Path $_ $helper } |
           Where-Object { Test-Path -LiteralPath $_ } |
           Select-Object -First 1
  if (-not $found) {
    Write-Error @"
missing helper: $helper

Looked in:
$($helperDirs | Where-Object { $_ } | ForEach-Object { "  $_" } | Out-String)
Copy the three files together, then run the copy:

  copy scripts\sweeps\ppl-bisect-step.ps1 scripts\sweeps\model-dir.ps1 scripts\sweeps\host-memory.ps1 <somewhere outside the repo>
"@
    exit 1
  }
  . $found
}

Set-Location $Repo

# -Root, not a bare call: model-dir.ps1 derives its candidates from its own $PSScriptRoot, which
# for the copied invocation in the header is C:\bisect rather than the checkout -- so the default
# artifact would be looked for beside the copy and the step would fail on a repository that has
# the file. Set-Location above does not help; $PSScriptRoot is not a working directory.
if (-not $Model) { $Model = Join-Path (Get-NInferModelDir -Root $Repo) 'qwen3_8_27b.ninfer' }
if (-not (Test-Path -LiteralPath $Model)) { Write-Error "missing artifact: $Model"; exit 1 }

# The output directory must exist before the redirection below, not after. `> "$Out.log"` is
# evaluated by the shell, so on a fresh checkout with no profiles\ directory the run dies on the
# redirect having already built -- twelve minutes for nothing.
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

# A step scores an 18.2 GB artifact, and a scoring run that pages does not give a wrong number, it
# gives a plausible one -- see host-memory.ps1. Guard before the build, so a shortfall costs
# seconds rather than a build plus eight minutes of scoring.
Assert-NInferHostMemory -Artifacts @($Model)

$sha = (git rev-parse --short HEAD).Trim()
"== bisect step at $sha  ($(git log -1 --format='%ad %s' --date=short))"

# Build ONLY ninfer-perplexity. The default target also builds the test binaries, and those break
# independently of the thing being measured -- at de5fc15a, tests/ops/softmax_attention/
# causal_cache.cpp fails under MSVC with "error C3493: 'order' cannot be implicitly captured".
# Building everything would have skipped a perfectly measurable commit. It is also much faster.
#
# Redirect the build whole, then grep -- never truncate a build pipeline. And always delete
# build.log first: a killed build leaves the PREVIOUS run's BUILD_EXIT=0 behind and the step reads
# a stale success.
Remove-Item build.log -ErrorAction SilentlyContinue
$bat = Join-Path $env:TEMP "ninfer_bisect_build_$PID.bat"
@(
  '@echo off'
  "call `"$VcVars`" >nul 2>&1"
  "set CUDACXX=$Cuda"
  'cmake --build build-ninja --target ninfer-perplexity > build.log 2>&1'
  'echo BUILD_EXIT=%ERRORLEVEL% >> build.log'
) | Set-Content -LiteralPath $bat -Encoding ASCII
try { & cmd.exe /c $bat | Out-Null } finally { Remove-Item $bat -ErrorAction SilentlyContinue }

$m = (Select-String -Path build.log -Pattern 'BUILD_EXIT=(\d+)' -ErrorAction SilentlyContinue)
if (-not $m -or $m.Matches.Groups[1].Value -ne '0') {
  "   BUILD FAILED at $sha -- classify with: git bisect skip"
  Select-String -Path build.log -Pattern 'error|FAILED' | Select-Object -First 3 | ForEach-Object { "     $($_.Line)" }
  exit 2
}

Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
nvidia-smi -lgc 1500 | Out-Null
try {
  # No --log-level: it postdates part of the bisect range and older builds reject it outright.
  & .\build-ninja\apps\ninfer-perplexity.exe $Model `
      --corpus 'eval\corpora\perplexity-1m\manifest.json' --quick `
      --context 4096 --stride 2048 --kv-dtype int8 `
      --output $Out > "$Out.log" 2>&1
  $code = $LASTEXITCODE
} finally { nvidia-smi -rgc | Out-Null }

$report = Get-ChildItem -Recurse -Filter report.json $Out -ErrorAction SilentlyContinue | Select-Object -First 1
if ($code -ne 0 -or -not $report) {
  "   RUN FAILED at $sha (exit $code) -- classify with: git bisect skip"
  Get-Content "$Out.log" -Tail 3 -ErrorAction SilentlyContinue | ForEach-Object { "     $_" }
  exit 3
}
$j = Get-Content $report.FullName -Raw | ConvertFrom-Json
"   $sha  ppl = $($j.overall.perplexity)  tokens = $($j.overall.scored_tokens)"
"   838c8b5d = 4.343263 | 66378f06 = 4.342864 | HEAD = 4.342425"
