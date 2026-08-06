#!/usr/bin/env python3
"""Evaluate the first engine resident GPU hot-loop explore row.

This gate consumes the artifact-free explore-log row from the default-off
ResidentGpuHotDecodeLoop path and decides whether it justifies promotion,
repeat/confirm, or a route correction. It does not claim speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-engine-resident-hot-loop-explore-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_SEQ95 = (
    ROOT / "output/resident-decode-loop-overhead-root-gate-20260707Tseq95Z/metrics.json"
)
DEFAULT_SEQ96 = (
    ROOT / "output/engine-resident-gpu-hot-loop-source-gate-20260707Tseq96Z/metrics.json"
)
DEFAULT_EXPLORE = ROOT / "output/explore-log.jsonl"
DEFAULT_LABEL = "engine-resident-gpu-hot-loop-seq97"
DEFAULT_OUT_DIR = ROOT / "output/engine-resident-gpu-hot-loop-explore-gate-20260707Tseq97Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _find_explore(path: Path, label: str) -> dict[str, Any]:
  found: dict[str, Any] | None = None
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    row = json.loads(line)
    if isinstance(row, dict) and row.get("label") == label:
      found = row
  if found is None:
    raise SystemExit(f"explore label not found: {label}")
  return found


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  seq95 = _load_json(args.seq95)
  seq96 = _load_json(args.seq96)
  row = _find_explore(args.explore_log, args.label)
  goal_anchor = frontier.get("goal_anchor")
  goal_anchor = goal_anchor if isinstance(goal_anchor, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  seq95_speed = seq95.get("speed_profile")
  seq95_speed = seq95_speed if isinstance(seq95_speed, dict) else {}

  current_best_tps = _num(goal_anchor.get("current_best_tps"))
  floor_tps = _num(goal_anchor.get("same_host_vulkan_floor_tps"))
  explore_tps = _num(row.get("tps"))
  rel_vs_best = (
      (explore_tps - current_best_tps) / current_best_tps
      if current_best_tps > 0.0 else 0.0
  )
  noise_rel = _num(noise.get("rel"))
  token_core_unprofiled_ns = _num(row.get("token_core_unprofiled_ns"))
  seq95_unprofiled_ns = _num(seq95_speed.get("unprofiled_wall_ns"))
  unprofiled_delta_ns = seq95_unprofiled_ns - token_core_unprofiled_ns
  tokens = int(_num(row.get("decode_tokens")))
  unprofiled_delta_ms_per_token = (
      (unprofiled_delta_ns / 1_000_000.0) / tokens if tokens > 0 else 0.0
  )
  floor_gap_ms_per_token = _num(seq95_speed.get("floor_gap_ms_per_token"))
  required_total_ms = floor_gap_ms_per_token * tokens
  achieved_fraction = (
      (unprofiled_delta_ns / 1_000_000.0) / required_total_ms
      if required_total_ms > 0.0 else 0.0
  )

  checks = [
      {
          "name": "seq96_source_gate_passed",
          "pass": seq96.get("required_checks_passed") is True
          and seq96.get("selected_next_route")
          == "engine_resident_gpu_hot_loop_explore",
      },
      {
          "name": "explore_row_present_and_top1_preserved",
          "pass": row.get("label") == args.label
          and row.get("top1_matches_native") is True
          and tokens == 8,
      },
      {
          "name": "explore_did_not_reach_frontier_or_floor",
          "pass": explore_tps < current_best_tps and explore_tps < floor_tps,
          "detail": {
              "explore_tps": explore_tps,
              "current_best_tps": current_best_tps,
              "floor_tps": floor_tps,
              "relative_vs_best": rel_vs_best,
              "noise_rel": noise_rel,
          },
      },
      {
          "name": "regression_is_outside_noise_band",
          "pass": rel_vs_best < -noise_rel,
      },
      {
          "name": "unprofiled_bucket_not_moved_enough",
          "pass": achieved_fraction < 0.05
          and unprofiled_delta_ms_per_token < floor_gap_ms_per_token,
          "detail": {
              "seq95_unprofiled_ns": seq95_unprofiled_ns,
              "explore_token_core_unprofiled_ns": token_core_unprofiled_ns,
              "unprofiled_delta_ms_per_token": unprofiled_delta_ms_per_token,
              "floor_gap_ms_per_token": floor_gap_ms_per_token,
              "achieved_fraction_of_required_total": achieved_fraction,
          },
      },
  ]
  required = all(check["pass"] for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_engine_hot_loop_api_shell_as_speed_cut"
          if required else "engine_hot_loop_explore_gate_failed"
      ),
      "selected_next_route": (
          "run_gpu_hybrid_decode_token_unprofiled_source_profile_gate"
          if required else "manual_review_engine_hot_loop_explore"
      ),
      "next_action": (
          "Do not promote or repeat the API-shell hot-loop path. Add a source "
          "profile gate inside RunGpuHybridDecodeToken/token assembly to locate "
          "the remaining unprofiled bucket; only then choose a source cut. "
          "Keep per-boundary micro-probes closed."
          if required else "Fix failed explore gate checks before changing route."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "seq95": _rel(args.seq95),
          "seq96": _rel(args.seq96),
          "explore_log": _rel(args.explore_log),
          "label": args.label,
      },
      "explore": {
          "ts": row.get("ts"),
          "label": row.get("label"),
          "source_sha": row.get("source_sha"),
          "tps": explore_tps,
          "top1_matches_native": row.get("top1_matches_native"),
          "required_checks_passed": row.get("required_checks_passed"),
          "failed_checks": row.get("failed_checks"),
          "decode_ns": row.get("decode_ns"),
          "kernel_sum_min_us": row.get("kernel_sum_min_us"),
          "gpu_loop_bookkeeping_ns": row.get("gpu_loop_bookkeeping_ns"),
          "token_core_unprofiled_ns": row.get("token_core_unprofiled_ns"),
      },
      "comparison": {
          "current_best_tps": current_best_tps,
          "floor_tps": floor_tps,
          "relative_vs_best": rel_vs_best,
          "noise_rel": noise_rel,
          "outside_noise_regression": rel_vs_best < -noise_rel,
          "seq95_unprofiled_ns": seq95_unprofiled_ns,
          "unprofiled_delta_ns": unprofiled_delta_ns,
          "unprofiled_delta_ms_per_token": unprofiled_delta_ms_per_token,
          "required_total_ms": required_total_ms,
          "achieved_fraction_of_required_total": achieved_fraction,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": "tools/intel-qwen36-engine-resident-hot-loop-explore-gate.py",
      "inputs": payload["inputs"],
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in payload["checks"] if not row["pass"]]
  explore = payload["explore"]
  comparison = payload["comparison"]
  lines = [
      "# Engine Resident GPU Hot-Loop Explore Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- explore tps: `{explore['tps']:.8f}`",
      f"- current best tps: `{comparison['current_best_tps']:.8f}`",
      f"- relative vs best: `{comparison['relative_vs_best']:.3%}`",
      f"- unprofiled movement: `{comparison['unprofiled_delta_ms_per_token']:.6f}` ms/token",
      f"- achieved required unprofiled cut: `{comparison['achieved_fraction_of_required_total']:.3%}`",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is explore gate evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--seq95", type=Path, default=DEFAULT_SEQ95)
  parser.add_argument("--seq96", type=Path, default=DEFAULT_SEQ96)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE)
  parser.add_argument("--label", default=DEFAULT_LABEL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
