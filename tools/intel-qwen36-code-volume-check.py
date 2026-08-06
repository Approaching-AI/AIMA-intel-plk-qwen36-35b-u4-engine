#!/usr/bin/env python3
"""Code-volume ratchet gate — guardrail #1 from meta-engine-factory §0.3.

"Port structure, do not flatten it." A from-scratch LLM forward pass has a tiny
unique structure: N layers are the *same* parameterized layer repeated. The
implementation should be ~O(1) in layer count. When an agent slides into
hand-writing one file per kernel/layer/boundary variant, the harness balloons,
and "code exploding / build slowing" is itself the trip-wire that says *you went
off the rails* — stop, generalize, refactor.

The engine CORE here is clean (one layer.cpp + one loop.cpp). The sprawl is in
the EXPLORATION layer: on 2026-06-30 the GPU bring-up added ~79 probe scripts in
one day, one per layer index (gpu-resident-layer7..23) with z-correction split
across 3 files instead of a flag. validate_repo.py only WARNed, and only watched
`*-compare.py` — the GPU `*-probe.py` files slipped straight through. This gate
closes both holes:
  - it counts `*-probe.py` (the actual GPU sprawl channel), not just `*-compare.py`;
  - it is a no-growth RATCHET that BLOCKS (exit 1) on growth past a frozen
    high-water, so a new per-layer probe fails the build, while the existing 172
    files stay committable (collapse + deletion are target-parity-gated; see the
    parameterized-compare-runner SOP). Lower the ceilings as you collapse; never
    raise them.

Pure stdlib, local. Exit non-zero when a ceiling is exceeded.

Usage:
  python3 tools/intel-qwen36-code-volume-check.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
TESTS = REPO / "engine" / "tests"

# Frozen high-water ceilings (counts on 2026-06-30, the day the gate was added).
# The disease is "keep adding per-variant files"; these block GROWTH past the
# freeze without forcing deletion now. RATCHET DOWN as the parameterized GPU/CPU
# runners land on target; NEVER raise.
CEILINGS = {
    "tools/*-probe.py": 64,
    "tools/*-compare.py": 35,
    "engine/tests/*_compare.cpp": 34,
}

# The clean skeleton this should stay shaped like (§0.3 size target).
SKELETON = [REPO / "engine" / "src" / "layer.cpp", REPO / "engine" / "src" / "loop.cpp"]
LINE_CAP = 1500  # advisory: single source files over this should split into per-op TUs

# The parameterized decode-smoke runner is the RIGHT structure (one runner + flags,
# not one file per variant) — but the flags then sprawled the same way: ~97
# add_argument, most dead-end --cpu-shadow-* / --*-affine / --*-extrapolate producers
# for CLOSED routes that already live in rejected-routes.json. Ratchet the flag count
# too; ratchet DOWN as dead flags are deleted, never up.
RUNNER = REPO / "tools" / "intel-qwen36-r2-gpu-decode-smoke.py"
RUNNER_FLAG_CEILING = 97
# Harness-MODE flags are loop infrastructure the methodology itself prescribes
# (ch.2 §2.2 collapse ritual / remote cache / resident session evidence), not
# experiment variants — they are excluded from the variant-flag ratchet BY NAME
# and printed for transparency.
# Keep this list short; deleting dead VARIANT flags is still owed (STATUS
# "collapse historical GPU probes/flags"), and the variant ceiling never rises.
RUNNER_HARNESS_FLAGS = (
    "--explore",
    "--label",
    "--no-remote-cache",
    "--resident-session-repeats",
    "--stream-resident-sse-events",
)


def count(glob_spec: str) -> int:
    base, pat = glob_spec.rsplit("/", 1)
    d = REPO / base
    return len(list(d.glob(pat))) if d.is_dir() else 0


def flag_count(p: Path) -> tuple[int, int]:
    """(variant flag count, exempt harness-mode flags present)."""
    if not p.exists():
        return 0, 0
    text = p.read_text(encoding="utf-8", errors="replace")
    total = len(re.findall(r"add_argument\(", text))
    harness = sum(1 for f in RUNNER_HARNESS_FLAGS if f'"{f}"' in text)
    return total - harness, harness


def line_count(p: Path) -> int:
    return len(p.read_text(encoding="utf-8", errors="replace").splitlines()) if p.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-ceilings", action="store_true",
                    help="print current counts as ceilings (to ratchet down after a collapse)")
    args = ap.parse_args()

    counts = {spec: count(spec) for spec in CEILINGS}
    if args.set_ceilings:
        for spec, n in counts.items():
            print(f'    "{spec}": {n},')
        print(f'    RUNNER_FLAG_CEILING = {flag_count(RUNNER)[0]}')
        return 0

    skel = sum(line_count(p) for p in SKELETON)
    print(f"skeleton engine/src/layer.cpp+loop.cpp: {skel} lines (the §0.3 shape target)")

    breaches = []
    for spec, ceil in CEILINGS.items():
        n = counts[spec]
        flag = "OK" if n <= ceil else "GREW"
        print(f"  {spec}: {n} (ceiling {ceil}) {flag}")
        if n > ceil:
            breaches.append(f"{spec} GREW to {n} > no-growth ceiling {ceil}")

    if RUNNER.exists():
        nflags, nharness = flag_count(RUNNER)
        flag = "OK" if nflags <= RUNNER_FLAG_CEILING else "GREW"
        print(f"  {RUNNER.name} variant flags: {nflags} (ceiling {RUNNER_FLAG_CEILING}; "
              f"+{nharness} harness-mode exempt: {', '.join(RUNNER_HARNESS_FLAGS)}) {flag}")
        if nflags > RUNNER_FLAG_CEILING:
            breaches.append(
                f"{RUNNER.name} add_argument GREW to {nflags} > no-growth ceiling "
                f"{RUNNER_FLAG_CEILING} (a new --flag per closed variant is the same "
                f"sprawl as a new file; fold or delete a dead flag first)"
            )

    # Advisory line-cap report (warn only; splitting big files is target-gated).
    big = []
    for rel in ("engine/src", "engine/tests", "engine/include", "tools"):
        d = REPO / rel
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix in (".cpp", ".hpp", ".h", ".py") and line_count(p) > LINE_CAP:
                big.append(f"{p.relative_to(REPO)} {line_count(p)} lines")
    if big:
        print(f"  (advisory: {len(big)} file(s) over {LINE_CAP} lines — split into per-op TUs: "
              f"{', '.join(big)})")

    if breaches:
        print()
        print("** GUARDRAIL #1 TRIPPED — code volume GREW (no-growth ratchet, §0.3) **")
        for b in breaches:
            print(f"  - {b}")
        print("  Do NOT add a per-layer/per-boundary file. Fold it into the parameterized")
        print("  runner behind a flag (--layer / --z-source), or delete a dead variant first.")
        print("  A new probe per layer index is the off-the-rails signal itself.")
        return 1

    print("code volume not growing (within no-growth ceilings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
