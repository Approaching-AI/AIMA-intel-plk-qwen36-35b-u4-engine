#!/usr/bin/env python3
"""Roll up engine compare boundary results into a single ladder.json.

Reads the boundary registry (`engine/boundaries.json`) and, for each boundary,
the latest `output/<artifact_prefix>-*/correctness.json`. Emits
`output/ladder.json` with a per-boundary pass/gate summary plus an overall
rollup.

This is the single generated source of truth for "which compares pass and how
many". It replaces the per-compare numbers that used to be hand-copied into the
two contracts, `goals/`, `doc/README.md`, and `validate_repo.py`.

Usage:
    python3 tools/iq36-ladder.py            # write output/ladder.json
    python3 tools/iq36-ladder.py --check    # also exit nonzero if a boundary is missing/failed
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "engine" / "boundaries.json"
DEFAULT_OUT = ROOT / "output" / "ladder.json"


def latest_artifact(prefix: str) -> Path | None:
    dirs = sorted(glob.glob(str(ROOT / "output" / f"{prefix}-*")))
    return Path(dirs[-1]) if dirs else None


def boundary_row(b: dict) -> dict:
    art = latest_artifact(b["artifact_prefix"])
    row = {
        "id": b["id"],
        "target": b["target"],
        "artifact_prefix": b["artifact_prefix"],
    }
    if art is None:
        row.update({"artifact": None, "passed": None, "status": "no_artifact"})
        return row
    correctness = {}
    cf = art / "correctness.json"
    if cf.exists():
        correctness = json.loads(cf.read_text(encoding="utf-8"))
    passed = correctness.get("required_checks_passed")
    row.update({
        "artifact": art.relative_to(ROOT).as_posix(),
        "passed": passed,
        "gate": correctness.get("gate"),
        "r1_native_correctness_gate_closed": correctness.get(
            "r1_native_correctness_gate_closed"
        ),
        "status": "pass" if passed is True else ("fail" if passed is False else "unknown"),
    })
    return row


def build_ladder() -> dict:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    rows = [boundary_row(b) for b in reg["boundaries"]]
    passed = [r for r in rows if r.get("passed") is True]
    return {
        "schema_version": "intel-qwen36-ladder-v1",
        "workstream": reg["workstream"],
        "selector_gate": "r1_native_gguf_correctness_first_token_loop",
        "boundary_count": len(rows),
        "passed_count": len(passed),
        "all_present": all(r["status"] != "no_artifact" for r in rows),
        "all_passed": len(passed) == len(rows) and len(rows) > 0,
        "note": (
            "Component compares only. R1 native token gate closure is recorded by "
            "the latest r1-native-candidate-jsonl artifact and STATUS.md; a "
            "component pass is not a speed claim."
        ),
        "boundaries": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if any boundary is missing an artifact or failed",
    )
    args = ap.parse_args()

    ladder = build_ladder()
    out = Path(args.out)
    out.write_text(json.dumps(ladder, indent=2) + "\n", encoding="utf-8")
    print(
        f"ladder: {ladder['passed_count']}/{ladder['boundary_count']} boundaries passing "
        f"-> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}"
    )
    for r in ladder["boundaries"]:
        if r["status"] != "pass":
            print(f"  {r['status']:>12}  {r['id']}")
    if args.check and not ladder["all_passed"]:
        raise SystemExit("ladder check failed: not all boundaries passing")


if __name__ == "__main__":
    main()
