# Where the .ninfer artifacts are, for the sweeps in this directory. Dot-source it and call
# Get-NInferModelDir; every sweep here does.
#
# A checkout can hold artifacts in two places and both are legitimate:
#
#   * the repository's own models\ -- what .gitignore has always ignored, and where a large local
#     collection tends to end up because it sits beside build-ninja\ rather than inside scripts\;
#   * scripts\models\ -- where scripts\download-model.{bat,sh} put them by default, and what
#     docs/rtx-3090-linux.md documents as the download location.
#
# A sweep that hardcodes either one fails for whoever followed the other instruction, so probe:
# the first candidate that actually contains an artifact wins, repository root first. Testing for
# *.ninfer rather than for the directory matters -- an empty models\ directory left behind by a
# cleared download would otherwise shadow a populated one.
#
# NINFER_MODEL_DIR is taken verbatim and never probed. Falling back past a directory the caller
# named turns their typo into a missing-artifact error about a path they never mentioned.
#
# Do not move this logic into a param() default in a calling script: $PSScriptRoot is empty while a
# param default is being bound, so a repo-relative path written there resolves against the drive
# root and silently yields C:\models. Measured, not assumed.
#
# -Root exists for the one caller that cannot use the default: ppl-bisect-step.ps1 is copied out of
# the repository with this file beside it, so the $PSScriptRoot below is the copy's directory and
# the candidates land next to the copy rather than in the checkout. That caller knows the
# repository root (it takes -Repo) and passes it here; everyone else omits it and gets the
# derived-from-this-file behaviour unchanged.
function Get-NInferModelDir {
    param([string]$Root = '')

    if ($env:NINFER_MODEL_DIR) { return $env:NINFER_MODEL_DIR }
    # $PSScriptRoot inside a function is the directory of the file that *defined* it, which is this
    # one, regardless of which sweep dot-sourced it or what the current location is -- hence -Root
    # for a caller running from a copy outside the repository.
    $repoRoot = if ($Root) { [IO.Path]::GetFullPath($Root) }
                else       { [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..')) }
    $candidates = @(
        [IO.Path]::GetFullPath((Join-Path $repoRoot 'models')),
        [IO.Path]::GetFullPath((Join-Path $repoRoot 'scripts\models'))
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -Path (Join-Path $candidate '*.ninfer')) { return $candidate }
    }
    # Neither holds an artifact. Return the repository's own models\ so the caller's own
    # "model not found" throw names the conventional location instead of an empty string.
    return $candidates[0]
}
