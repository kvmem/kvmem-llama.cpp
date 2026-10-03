# The host-RAM guard every sweep in this directory needs. Dot-source it and call
# Assert-NInferHostMemory with the artifacts the sweep is about to load.
#
# Why this exists, and why it is not optional. vmmemWSL can hold 27 GiB of this box's 64 GiB while
# WSL is running, and the 35B artifact is 22.8 GB. Loading it under that pressure does not fail --
# the machine pages, the run completes, and every timing it prints is noise that looks exactly like
# a result. moe-prefill-pipeline-depth.ps1 was abandoned once for precisely this before it grew the
# check, and it was the only script here that had one. `wsl --shutdown` reclaims the memory.
#
# The requirement is sized from the artifact rather than fixed, because a 27B-only sweep does not
# need what a 35B one does and refusing it at 20 GiB free would be a false alarm. The headroom
# term is the one that was actually validated: the hand-written guard demanded 24 GiB for the
# 22,783,246,080-byte (21.22 GiB) 35B artifact, i.e. 2.8 GiB above the file, which is what the
# process wants for pinned host-KV, workspaces and the loader's own staging. Keeping that constant
# reproduces the original threshold exactly on the 35B and scales it everywhere else.
#
# Set NINFER_SKIP_MEMORY_GUARD=1 to proceed anyway. That is for someone who knows their box is not
# this one, not for getting past a genuine shortfall -- see the note above about what a paging run
# produces.

$script:NInferHostMemoryHeadroomGiB = 2.8

function Assert-NInferHostMemory {
    [CmdletBinding()]
    param(
        # Paths to the .ninfer artifacts the sweep will load. The largest one sets the requirement:
        # sweeps here load one model at a time and release it before the next.
        [Parameter(Mandatory = $true)][string[]]$Artifacts,
        # Override the computed requirement outright, for a caller that knows better.
        [double]$RequiredGiB = 0
    )

    $os      = Get-CimInstance Win32_OperatingSystem
    # FreePhysicalMemory is in KiB, so /1MB converts KiB -> GiB.
    $freeGiB = $os.FreePhysicalMemory / 1MB

    if ($RequiredGiB -le 0) {
        $largestBytes = 0
        foreach ($a in $Artifacts) {
            if (Test-Path -LiteralPath $a) {
                $len = (Get-Item -LiteralPath $a).Length
                if ($len -gt $largestBytes) { $largestBytes = $len }
            }
        }
        if ($largestBytes -eq 0) {
            # Nothing measurable -- the caller's own "missing artifact" throw is the better error,
            # so do not invent a threshold here.
            "# host RAM free: {0:N1} GiB (no artifact measured; guard skipped)" -f $freeGiB
            return
        }
        $RequiredGiB = ($largestBytes / 1GB) + $script:NInferHostMemoryHeadroomGiB
    }

    "# host RAM free: {0:N1} GiB, need {1:N1} GiB" -f $freeGiB, $RequiredGiB

    if ($freeGiB -lt $RequiredGiB) {
        if ($env:NINFER_SKIP_MEMORY_GUARD -eq '1') {
            "# NINFER_SKIP_MEMORY_GUARD=1 -- proceeding under memory pressure; treat every number below as void"
            return
        }
        # -f binds tighter than +, so the format must be applied to the whole string rather than to
        # the last piece of a concatenation.
        throw ("only {0:N1} GiB of host RAM free but this run needs {1:N1} GiB; it will page and the timings will be noise rather than an error. Check vmmemWSL (wsl --shutdown reclaims up to 27 GiB) and retry, or set NINFER_SKIP_MEMORY_GUARD=1 to override." -f $freeGiB, $RequiredGiB)
    }
}
