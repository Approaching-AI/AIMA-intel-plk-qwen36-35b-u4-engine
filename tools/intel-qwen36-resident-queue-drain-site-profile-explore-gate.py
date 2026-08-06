#!/usr/bin/env python3
"""Gate the artifact-free resident queue drain-site profile explore row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-queue-drain-site-profile-explore-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_SEQ157 = (
    ROOT
    / "output/resident-queue-drain-site-profile-target-compile-gate-20260708Tseq157Z"
    / "metrics.json"
)
DEFAULT_LABEL = "resident-queue-drain-site-profile-seq158"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-queue-drain-site-profile-explore-gate-20260708Tseq158Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    item = json.loads(line)
    if isinstance(item, dict):
      rows.append(item)
  return rows


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _nested(obj: Any, *keys: str) -> Any:
  cur = obj
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


def _label_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
  found: dict[str, Any] | None = None
  for row in rows:
    if row.get("label") == label:
      found = row
  if found is None:
    raise SystemExit(f"missing explore-log row label {label!r}")
  return found


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "noise_rel": _num(noise.get("rel")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
  }


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  value = _nested(row, "profile_smoke", "resident_queue_drain_site_profile_ns")
  return value if isinstance(value, dict) else {}


def _per_token(ns: float, tokens: float) -> float:
  return (ns / 1_000_000.0) / tokens if tokens > 0.0 else 0.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  seq157 = _load_json(args.seq157)
  row = _label_row(_load_jsonl(args.explore_log), args.label)
  frontier_state = _frontier_state(frontier)
  profile = _profile(row)
  tokens = _num(row.get("decode_tokens")) or 1.0
  buckets = {
      "selected_down_wait_drain_site": _num(
          profile.get("selected_down_wait_drain_site")),
      "ffn_tail_drain_site": _num(profile.get("ffn_tail_drain_site")),
      "attention_front_drain_site": _num(
          profile.get("attention_front_drain_site")),
  }
  per_token = {key: _per_token(value, tokens) for key, value in buckets.items()}
  largest = max(per_token, key=per_token.get) if per_token else ""
  required_checks_passed = False
  checks = [
      {
          "name": "seq157_target_compile_selected_explore_gate",
          "pass": (
              seq157.get("required_checks_passed") is True
              and seq157.get("selected_next_route")
              == "resident_queue_drain_site_profile_explore_gate"
          ),
          "detail": {
              "seq157_disposition": seq157.get("disposition"),
              "seq157_selected_next_route": seq157.get("selected_next_route"),
          },
      },
      {
          "name": "explore_row_is_artifact_free_profile_row",
          "pass": (
              row.get("schema") == "iq36-explore-log-v0"
              and row.get("label") == args.label
              and row.get("resident_queue_drain_site_profile") is True
              and _nested(row, "profile_smoke",
                          "resident_queue_drain_site_profile_enabled") is True
              and row.get("speedup_claims_allowed") is False
          ),
          "detail": {
              "label": row.get("label"),
              "source_sha": row.get("source_sha"),
              "tps": row.get("tps"),
              "required_checks_passed": row.get("required_checks_passed"),
              "failed_checks": row.get("failed_checks"),
          },
      },
      {
          "name": "profile_row_preserves_greedy_top1",
          "pass": row.get("top1_matches_native") is True,
      },
      {
          "name": "drain_site_profile_has_floor_sized_bucket",
          "pass": (
              per_token.get(largest, 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
              and _num(profile.get("profiled")) > 0.0
          ),
          "detail": {
              "largest_bucket": largest,
              "largest_ms_per_token": per_token.get(largest, 0.0),
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
              "buckets_ms_per_token": per_token,
              "profile_ns": profile,
          },
      },
      {
          "name": "selected_down_wait_remains_largest_drain_site",
          "pass": largest == "selected_down_wait_drain_site",
          "detail": {"largest_bucket": largest, "buckets_ms_per_token": per_token},
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
          ),
          "detail": frontier_state,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "explore_log": _rel(args.explore_log),
          "seq157_target_compile_gate": _rel(args.seq157),
          "label": args.label,
      },
      "frontier": frontier_state,
      "explore_row": {
          "ts": row.get("ts"),
          "label": row.get("label"),
          "source_sha": row.get("source_sha"),
          "config_sha": row.get("config_sha"),
          "tps": row.get("tps"),
          "decode_tokens": row.get("decode_tokens"),
          "top1_matches_native": row.get("top1_matches_native"),
          "required_checks_passed": row.get("required_checks_passed"),
          "failed_checks": row.get("failed_checks"),
          "speedup_claims_allowed": row.get("speedup_claims_allowed"),
      },
      "drain_site_profile": {
          "profile_ns": profile,
          "buckets_ms_per_token": per_token,
          "largest_bucket": largest,
          "largest_ms_per_token": per_token.get(largest, 0.0),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_resident_queue_drain_site_profile_explore"
          if required_checks_passed
          else "reject_resident_queue_drain_site_profile_explore"
      ),
      "selected_next_route": (
          "resident_selected_down_wait_drain_route_gate"
          if required_checks_passed
          else "resident_queue_drain_site_profile_fix_gate"
      ),
      "next_route_reason": (
          "The artifact-free profile row preserves top-1 and shows the largest "
          "floor-sized drain site is selected-down wait. Next unit is route "
          "control/design for selected-down wait removal, not a speed claim."
          if required_checks_passed else
          "Fix the drain-site profile row before route selection."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": metrics["speedup_claims_allowed"],
      "inputs": metrics["inputs"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  drain = metrics["drain_site_profile"]
  summary = [
      "# Resident Queue Drain-Site Profile Explore Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- largest_bucket: `{drain['largest_bucket']}`",
      f"- largest_ms_per_token: `{drain['largest_ms_per_token']:.3f}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is artifact-free profile evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--seq157", type=Path, default=DEFAULT_SEQ157)
  parser.add_argument("--label", default=DEFAULT_LABEL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "largest_bucket": metrics["drain_site_profile"]["largest_bucket"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
