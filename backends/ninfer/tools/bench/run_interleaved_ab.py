#!/usr/bin/env python3
"""Compare two ninfer builds by interleaving them, because this card drifts between processes.

Why this exists rather than "run arm A ten times, then arm B ten times". The RTX 3090 in this box
moves 3-5% between processes as it heats and the power cap clamps, which is larger than most of the
effects anyone measures here. Two comparisons in TODO's history came out with opposite-signed drift
(+2.9% then -3.8%) on code paths that had not changed at all. Sequential arms cannot separate a
real change from that ramp; interleaved arms can, because both meet the same conditions within each
repetition.

Four properties, all of which were learned the hard way and three of which are easy to get wrong:

  1. Both executables must live in the SAME directory as the build that produced them -- in
     practice build-ninja/apps/. An executable copied elsewhere fails DLL resolution, exits 127
     and writes an empty log, which reads exactly like a model-loading failure and has cost hours.
     Checked in preflight, and diagnosed explicitly if it happens anyway.
  2. The arms alternate INSIDE each repetition rather than one arm then the other. This tool also
     swaps which arm leads on alternate repetitions, so a within-repetition warming trend lands on
     each arm equally often instead of always favouring the second one.
  3. The result is the PAIRED median -- the median of per-repetition ratios -- plus how many pairs
     came out positive. Not the ratio of the two medians. On this hardware those two statistics
     disagree by more than the effects being measured, so the distinction is not pedantry.
  4. A control configuration, whose code path is identical in both arms, can be measured alongside
     and divided out. In the GDN-tile comparison the control is what made the result readable at
     all: draft counts 4 and 5 stay on an unchanged route, so any ratio they show is pure drift and
     the real arms can be normalised against it.

Usage -- {exe} is substituted with each arm's executable in turn:

    python tools/bench/run_interleaved_ab.py \\
        --arm-a build-ninja/apps/ninfer.exe \\
        --arm-b build-ninja/apps/ninfer-newtile.exe \\
        --reps 6 \\
        --config 'k6:{exe} models/qwen3_8_27b_dflash2.ninfer --prompt "..." --max-new 256 \\
                  --greedy --spec dflash2 --draft-tokens 6' \\
        --control 'k4:{exe} models/qwen3_8_27b_dflash2.ninfer --prompt "..." --max-new 256 \\
                  --greedy --spec dflash2 --draft-tokens 4'

Each --config and --control is 'label:command'. The label may not contain a colon; everything after
the first colon is the command, split with shlex in POSIX mode so quoting behaves the same on both
platforms.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# The CLI prints its summary block to stderr, one "summary <label> <value>" line per metric, so the
# default metric is read out of the combined stream. ninfer_bench prints a CSV instead; point
# --metric-regex at whatever the tool being compared emits.
DEFAULT_METRIC_REGEX = r"decode speed\s+([\d.]+) tok/s"


@dataclass
class Sample:
    label: str
    arm: str
    rep: int
    value: float | None
    seconds: float
    exit_code: int
    detail: str = ""


@dataclass
class Config:
    label: str
    argv_template: list[str]
    is_control: bool = False
    samples: list[Sample] = field(default_factory=list)


def split_command(command: str) -> list[str]:
    r"""Split a command string into argv, without eating Windows path separators.

    shlex's POSIX mode treats backslash as an escape character, so an absolute Windows path is
    silently mutilated rather than rejected: C:\ninfer-fork\build-ninja\apps\x.exe comes back as
    "C:ninfer-forkbuild-ninjaappsx.exe", which then fails as a missing file for a reason the
    message does not mention. Non-POSIX mode keeps separators intact but leaves quotes attached to
    the token, so strip one matched pair -- that is what the shell would have consumed.
    """
    if os.name != "nt":
        return shlex.split(command, posix=True)
    tokens = shlex.split(command, posix=False)
    stripped: list[str] = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        stripped.append(token)
    return stripped


def parse_config(spec: str, *, is_control: bool) -> Config:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"expected 'label:command', got {spec!r}")
    label, _, command = spec.partition(":")
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"empty label in {spec!r}")
    argv = split_command(command)
    if not argv:
        raise argparse.ArgumentTypeError(f"empty command in {spec!r}")
    if not any("{exe}" in token for token in argv):
        raise argparse.ArgumentTypeError(
            f"command for {label!r} contains no {{exe}} placeholder, so both arms would run the "
            "same binary and the comparison would be of nothing"
        )
    return Config(label=label, argv_template=argv, is_control=is_control)


def preflight(arm_a: Path, arm_b: Path) -> None:
    """Refuse the two mistakes that produce a confident wrong answer rather than an error."""
    for path in (arm_a, arm_b):
        if not path.is_file():
            sys.exit(f"missing executable: {path}")
    if arm_a.resolve() == arm_b.resolve():
        sys.exit(f"both arms resolve to the same file ({arm_a.resolve()}); nothing to compare")
    if arm_a.resolve().parent != arm_b.resolve().parent:
        # Not fatal on every platform, but on Windows it is the exit-127 trap in the docstring and
        # there is no reason to allow it: copying the second build in beside the first is one line.
        sys.exit(
            f"the two arms are in different directories:\n"
            f"  A: {arm_a.resolve().parent}\n"
            f"  B: {arm_b.resolve().parent}\n"
            "Place both inside the build tree that produced them (build-ninja/apps/) under "
            "different names. An executable run from elsewhere fails DLL resolution with exit 127 "
            "and an empty log, which looks like a model failure and is not one."
        )


def run_once(config: Config, arm: str, exe: Path, rep: int, metric: re.Pattern[str],
             timeout: float, echo: bool) -> Sample:
    argv = [token.replace("{exe}", str(exe)) for token in config.argv_template]
    started = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              errors="replace")
    except subprocess.TimeoutExpired:
        return Sample(config.label, arm, rep, None, time.monotonic() - started, -1, "timeout")
    elapsed = time.monotonic() - started
    blob = (proc.stdout or "") + (proc.stderr or "")

    if proc.returncode != 0:
        detail = f"exit {proc.returncode}"
        if proc.returncode == 127 or not blob.strip():
            # The specific failure this harness is documented to avoid. Say so rather than letting
            # the reader conclude the model is broken.
            detail += " with no output -- almost certainly DLL resolution, not the model"
        else:
            detail += ": " + blob.strip().splitlines()[-1][:160]
        return Sample(config.label, arm, rep, None, elapsed, proc.returncode, detail)

    found = metric.search(blob)
    if not found:
        return Sample(config.label, arm, rep, None, elapsed, 0,
                      "metric regex did not match the output")
    value = float(found.group(1))
    if echo:
        print(f"    {config.label:<16} {arm}  rep {rep}  {value:>10.3f}  ({elapsed:.1f}s)",
              flush=True)
    return Sample(config.label, arm, rep, value, elapsed, 0)


def paired_ratios(config: Config) -> list[tuple[int, float]]:
    """One ratio per repetition, B relative to A. Repetitions missing either arm are dropped."""
    by_rep: dict[int, dict[str, float]] = {}
    for sample in config.samples:
        if sample.value is not None:
            by_rep.setdefault(sample.rep, {})[sample.arm] = sample.value
    ratios = []
    for rep in sorted(by_rep):
        pair = by_rep[rep]
        if "A" in pair and "B" in pair and pair["A"] > 0:
            ratios.append((rep, pair["B"] / pair["A"]))
    return ratios


def report(configs: list[Config], control: Config | None) -> None:
    control_ratios = dict(paired_ratios(control)) if control else {}
    control_median = statistics.median(control_ratios.values()) if control_ratios else None

    print()
    print("== per-configuration paired result (B relative to A)")
    print(f"{'config':<18}{'pairs':>6}{'median':>10}{'min':>10}{'max':>10}{'positive':>10}"
          f"{'normalised':>12}")
    for config in configs:
        ratios = paired_ratios(config)
        if not ratios:
            print(f"{config.label:<18}{'0':>6}   no complete pairs")
            continue
        values = [r for _, r in ratios]
        median = statistics.median(values)
        positive = sum(1 for v in values if v > 1.0)
        # Normalising per repetition, then taking the median, keeps the pairing intact -- dividing
        # the two medians would throw away exactly the information the control was measured for.
        if control_ratios and not config.is_control:
            per_rep = [r / control_ratios[rep] for rep, r in ratios if rep in control_ratios]
            normalised = f"{(statistics.median(per_rep) - 1.0) * 100:+.2f}%" if per_rep else "-"
        else:
            normalised = "-"
        print(f"{config.label:<18}{len(values):>6}{(median - 1) * 100:>+9.2f}%"
              f"{(min(values) - 1) * 100:>+9.2f}%{(max(values) - 1) * 100:>+9.2f}%"
              f"{positive:>7}/{len(values):<2}{normalised:>12}")

    if control_median is not None:
        print()
        print(f"control {control.label!r} moved {(control_median - 1) * 100:+.2f}% between arms. "
              "Its code path is identical in both, so that is drift:")
        if abs(control_median - 1.0) > 0.02:
            print("  ** over 2% -- read the normalised column, not the median one, and consider "
                  "more repetitions.")
        else:
            print("  under 2%, so the raw medians are already readable.")

    failures = [s for c in configs for s in c.samples if s.value is None]
    if failures:
        print()
        print(f"== {len(failures)} run(s) produced no metric")
        for sample in failures[:12]:
            print(f"  {sample.label:<18}arm {sample.arm} rep {sample.rep}: {sample.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-a", type=Path, required=True, help="baseline executable")
    parser.add_argument("--arm-b", type=Path, required=True, help="candidate executable")
    parser.add_argument("--reps", type=int, default=6,
                        help="repetitions; each one runs both arms of every configuration")
    parser.add_argument("--config", action="append", default=[], metavar="LABEL:COMMAND",
                        help="a configuration to compare; repeatable")
    parser.add_argument("--control", metavar="LABEL:COMMAND",
                        help="a configuration whose code path is IDENTICAL in both arms, used to "
                             "measure and divide out drift")
    parser.add_argument("--metric-regex", default=DEFAULT_METRIC_REGEX,
                        help="regex with one capturing group holding the number to compare "
                             f"(default: {DEFAULT_METRIC_REGEX!r})")
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-run timeout in seconds")
    parser.add_argument("--quiet", action="store_true", help="do not echo each sample")
    args = parser.parse_args(argv)

    if not args.config:
        parser.error("at least one --config is required")
    if args.reps < 2:
        parser.error("--reps must be at least 2; a single pair has no spread to report")

    preflight(args.arm_a, args.arm_b)
    metric = re.compile(args.metric_regex)

    # parse_config raises ArgumentTypeError, which argparse only renders nicely when it is used as
    # a `type=`. These are parsed after the fact so a bad spec would otherwise land as a traceback.
    try:
        configs = [parse_config(spec, is_control=False) for spec in args.config]
        control = parse_config(args.control, is_control=True) if args.control else None
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    measured = configs + ([control] if control else [])

    print(f"A = {args.arm_a}")
    print(f"B = {args.arm_b}")
    print(f"{len(measured)} configuration(s) x {args.reps} repetitions x 2 arms = "
          f"{len(measured) * args.reps * 2} runs")

    for rep in range(1, args.reps + 1):
        # Swap the leading arm every other repetition. Whichever arm runs second in a repetition
        # meets a slightly warmer card, and alternating spreads that equally rather than letting it
        # accumulate on one side.
        order = ("A", "B") if rep % 2 else ("B", "A")
        print(f"  rep {rep}/{args.reps} (order {order[0]}{order[1]})", flush=True)
        for config in measured:
            for arm in order:
                exe = args.arm_a if arm == "A" else args.arm_b
                config.samples.append(
                    run_once(config, arm, exe, rep, metric, args.timeout, not args.quiet))

    report(configs, control)
    # A comparison in which some run produced no number is not a comparison; make that an exit
    # code so a script driving this notices.
    return 1 if any(s.value is None for c in measured for s in c.samples) else 0


if __name__ == "__main__":
    sys.exit(main())
