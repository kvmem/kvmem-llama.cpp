# Does any DFlash2 draft count make it positive on *text*?
#
# TODO.md section 2c records DFlash2 costing ~20% against no speculation on the 27B, always measured
# with --draft-tokens 7, and asks whether a lower count crosses over. The obvious way to answer that
# -- sweep draft counts through ninfer_bench -- does not work, and finding out why is the point of
# this script existing separately.
#
# `bench/fixtures/bench_corpus.ids` holds 65,536 tokens drawn from **682 distinct ids**, with 98.4%
# of its bigrams repeated, because it is a curated bank tiled to length. Its own manifest says
# "repetition fills length only and does not bias throughput", which is true for plain decode and
# false for anything that drafts: a draft model predicts that text perfectly. Swept through
# ninfer_bench, DFlash2 reports **100% acceptance at every draft count from 1 to 12**, and decode
# rises monotonically to 159 tok/s because each round emits k+1 free tokens. None of that is a
# statement about text.
#
# So measure the serving path on the model's own generated continuation instead. The model writes
# natural prose in response to a real prompt, which is the distribution DFlash2 meets in production,
# and acceptance then reflects genuine drafting difficulty rather than a tiled fixture.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\dflash2-draft-tokens-realtext.ps1
#
# About 25 minutes. Greedy at every configuration, which is necessary but not sufficient for the
# comparison to be of speed on identical output. The hashes this sweep records are what settled
# that: DFlash2 and MTP do NOT match each other, DFlash2's own output varies with the draft count
# (eight distinct outputs across k = 1..12), and every speculative configuration diverges from the
# width-1 greedy path within the first hundred tokens -- squarely inside this sweep's --max-new 256
# window. See docs/performance.md, which carries the same finding and the retraction of the earlier
# "byte-identical to each other" claim. So this asserts nothing about identity; it hashes each run's
# generated text (content_sha256 below) and leaves the comparison to whoever reads the CSV.
#
# It also reports acceptance. TODO.md section 3 wants DFlash2's acceptance and tokens-per-round on
# realistic text and records that the committed corpus cannot supply them -- it is 65,536 tokens
# over 682 distinct ids and reports exactly 100% acceptance at every draft count, which is a
# statement about the fixture. This sweep already generates real prose through the serving path, and
# apps/cli/main.cpp already prints `<backend> acceptance rate` and `<backend> acceptance length`
# to stderr, so the numbers only ever needed parsing out. Note `acceptance length` is
# `1 + accepted/rounds`, i.e. tokens emitted per round including the always-free verified one --
# the same quantity TODO's cliff table calls tok/round.
$ErrorActionPreference = 'Continue'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

. "$PSScriptRoot\model-dir.ps1"
. "$PSScriptRoot\host-memory.ps1"
$modelDir = Get-NInferModelDir
$out      = if ($env:NINFER_SWEEP_OUT) { $env:NINFER_SWEEP_OUT } else { 'profiles\sweeps' }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$weights = "$modelDir\qwen3_8_27b_dflash2.ninfer"
if (-not (Test-Path $weights)) { throw "Missing DFlash2 artifact: $weights" }
Assert-NInferHostMemory -Artifacts @($weights)

# A prompt that provokes a long, substantive, non-repetitive answer. Deliberately not a list or a
# table: those are exactly the shapes a draft head finds easy, and the question here is what
# ordinary prose costs.
$prompt = 'Explain how a paged key-value cache lets a transformer serve many concurrent requests ' +
          'without reserving each one its maximum context up front. Cover fragmentation, the ' +
          'block table indirection, and what happens when a request outgrows its allocation.'

$configs = @( @{ label='none'; args=@() } )
foreach ($n in 1,2,3,4,5,6,7,8,10,12) {
  $configs += @{ label="dflash2-$n";      args=@('--spec','dflash2','--draft-tokens',[string]$n) }
  $configs += @{ label="dflash2-$n+head"; args=@('--spec','dflash2','--draft-tokens',[string]$n,'--lm-head-draft') }
}
$configs += @{ label='mtp3';      args=@('--spec','mtp','--draft-tokens','3') }
$configs += @{ label='mtp3+head'; args=@('--spec','mtp','--draft-tokens','3','--lm-head-draft') }

"config,rep,decode_tok_s,generated_tokens,rounds,drafted,accepted,acceptance_rate_pct,tok_per_round,content_sha256"
foreach ($c in $configs) {
  for ($rep = 1; $rep -le 3; $rep++) {
    $stem = "$out\rt_$($c.label -replace '\+','p')_$rep"
    $log  = "$stem.err.log"
    $text = "$stem.txt"
    $argv = @($weights,'--prompt',$prompt,'--max-new','256','--max-context','8192',
              '--kv-dtype','int8','--greedy','--no-thinking') + $c.args
    # apps/cli/main.cpp writes generated text to stdout and every stage/summary line (including
    # decode_tok_s and generated_tokens below) to stderr -- captured separately so $text is exactly
    # the model's output and can be hashed, rather than the merged stream this used to read.
    & .\build-ninja\apps\ninfer.exe @argv > $text 2> $log
    if ($LASTEXITCODE -ne 0) { "$($c.label),$rep,FAILED,,,,,,,"; Get-Content $log -Tail 2 | ForEach-Object { "    $_" }; continue }
    $txt  = Get-Content $log -Raw
    $dec  = if ($txt -match 'decode speed\s+([\d.]+) tok/s') { $Matches[1] } else { '' }
    $gen  = if ($txt -match 'generated tokens\s+(\d+)')      { $Matches[1] } else { '' }
    # The backend prefixes every speculative metric with its own name ("dflash2 rounds", "mtp
    # acceptance rate"), so match the label rather than a fixed backend -- one regex serves both
    # arms and the unspeculated control simply leaves the columns empty.
    $rounds = if ($txt -match '\S+ rounds\s+(\d+)')                          { $Matches[1] } else { '' }
    $draft  = if ($txt -match '\S+ drafted tokens\s+(\d+)')                  { $Matches[1] } else { '' }
    $acc    = if ($txt -match '\S+ accepted tokens\s+(\d+)')                 { $Matches[1] } else { '' }
    # format_pretty_percent emits "12.3%" or the literal "n/a" when nothing was drafted; leave the
    # column empty in the n/a case rather than writing a word into a numeric column.
    $rate   = if ($txt -match '\S+ acceptance rate\s+([\d.]+)%')             { $Matches[1] } else { '' }
    $tpr    = if ($txt -match '\S+ acceptance length\s+([\d.]+) tok/round') { $Matches[1] } else { '' }
    # Hash the generated text so a reader can tell whether two configurations produced identical
    # output.
    #
    # TODO.md recorded this column as coming out empty with "26 error blocks" and blamed a missing
    # Get-FileHash. That diagnosis is wrong: Windows PowerShell 5.1.26100 on this box has the
    # cmdlet and hashes fine. What actually happens is that Get-FileHash on a path that does not
    # exist raises a **non-terminating** error from a Resolve-Path inside its own implementation
    # and returns nothing at all -- so `.Hash` on the nothing yields an empty string and the error
    # prints once per iteration, which is precisely the reported symptom. A bare try/catch does not
    # help either, because a non-terminating error never reaches catch.
    #
    # So: -ErrorAction Stop to make it catchable, and the reason lands in the column instead of an
    # error block. Whatever leaves $text missing is then visible in the CSV rather than inferred
    # from console noise.
    $hash = if (-not (Test-Path -LiteralPath $text)) {
      'ERR:no-stdout-file'
    } else {
      try { (Get-FileHash -LiteralPath $text -Algorithm SHA256 -ErrorAction Stop).Hash }
      catch { "ERR:$($_.Exception.GetType().Name)" }
    }
    "$($c.label),$rep,$dec,$gen,$rounds,$draft,$acc,$rate,$tpr,$hash"
  }
}
"== done =="
