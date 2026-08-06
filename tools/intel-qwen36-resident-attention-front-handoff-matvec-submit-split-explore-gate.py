#!/usr/bin/env python3
"""Gate the artifact-free attention-front handoff matvec submit-split profile row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-resident-attention-front-handoff-matvec-submit-split-explore-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_SEQ177 = (
    ROOT
    / "output/resident-attention-front-handoff-matvec-submit-split-target-compile-gate-20260708Tseq177Z"
    / "metrics.json"
)
DEFAULT_LABEL = "resident-attention-front-handoff-matvec-submit-split-seq178"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-attention-front-handoff-matvec-submit-split-explore-gate-20260708Tseq178Z"
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


def _dict(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


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


def _per_token(ns: float, tokens: float) -> float:
  return (ns / 1_000_000.0) / tokens if tokens > 0.0 else 0.0


def _per_token_map(values: dict[str, Any], tokens: float) -> dict[str, float]:
  return {key: _per_token(_num(value), tokens) for key, value in values.items()}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  seq177 = _load_json(args.seq177)
  row = _label_row(_load_jsonl(args.explore_log), args.label)
  frontier_state = _frontier_state(frontier)
  tokens = _num(row.get("decode_tokens")) or 1.0
  profile = _dict(_nested(
      row, "profile_smoke",
      "attention_front_handoff_matvec_submit_split_profile_ns"))
  buckets_ms = _per_token_map(profile, tokens)
  wall = _dict(_nested(row, "profile_smoke", "wall_profile_ns"))
  wall_ms = _per_token_map(wall, tokens)

  key_buckets = {
      key: buckets_ms.get(key, 0.0)
      for key in [
          "kernel_setup",
          "kernel_wait",
          "kernel_enqueue",
          "kernel_finish",
          "event_profile",
          "queue_drain_cleanup",
      ]
  }
  largest = max(key_buckets, key=key_buckets.get) if key_buckets else ""
  floor_sized = {
      key: value
      for key, value in key_buckets.items()
      if value >= frontier_state["floor_gap_ms_per_token"]
  }

  checks = [
      {
          "name": "seq177_target_compile_selected_explore_gate",
          "pass": (
              seq177.get("required_checks_passed") is True
              and seq177.get("selected_next_route")
              == "resident_attention_front_handoff_matvec_submit_split_explore_gate"
          ),
          "detail": {
              "seq177_disposition": seq177.get("disposition"),
              "seq177_selected_next_route": seq177.get("selected_next_route"),
          },
      },
      {
          "name": "explore_row_is_artifact_free_noqueue_handoff_matvec_submit_split_row",
          "pass": (
              row.get("schema") == "iq36-explore-log-v0"
              and row.get("label") == args.label
              and row.get("opencl_no_queue_profiling") is True
              and row.get("attention_front_handoff_matvec_submit_split_profile") is True
              and _nested(
                  row, "profile_smoke",
                  "attention_front_handoff_matvec_submit_split_profile_enabled") is True
              and row.get("speedup_claims_allowed") is False
          ),
          "detail": {
              "label": row.get("label"),
              "source_sha": row.get("source_sha"),
              "config_sha": row.get("config_sha"),
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
          "name": "handoff_matvec_wait_finish_are_floor_sized_buckets",
          "pass": (
              largest in {"kernel_wait", "kernel_finish"}
              and key_buckets.get("kernel_wait", 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
              and key_buckets.get("kernel_finish", 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
          ),
          "detail": {
              "largest_bucket": largest,
              "kernel_wait_ms_per_token": key_buckets.get("kernel_wait", 0.0),
              "kernel_finish_ms_per_token": key_buckets.get(
                  "kernel_finish", 0.0),
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
              "floor_sized": floor_sized,
          },
      },
      {
          "name": "wall_profile_still_has_floor_sized_remaining_buckets",
          "pass": (
              wall_ms.get("attention_front", 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
              and wall_ms.get("linear_preconv", 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
              and wall_ms.get("selected_ffn", 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
              and wall_ms.get("lm_head_gpu", 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
              and wall_ms.get("ffn_tail", 0.0)
              >= frontier_state["floor_gap_ms_per_token"]
          ),
          "detail": {
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
              "wall_ms_per_token": wall_ms,
          },
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
          "seq177_target_compile_gate": _rel(args.seq177),
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
      "attention_front_handoff_matvec_submit_split_profile": {
          "profile_ns": profile,
          "buckets_ms_per_token": key_buckets,
          "floor_sized_buckets": floor_sized,
          "largest_bucket": largest,
          "largest_ms_per_token": key_buckets.get(largest, 0.0),
      },
      "wall_profile": {
          "profile_ns": wall,
          "buckets_ms_per_token": wall_ms,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_resident_attention_front_handoff_matvec_submit_split_profile_explore"
          if required_checks_passed else
          "reject_resident_attention_front_handoff_matvec_submit_split_profile_explore"
      ),
      "selected_next_route": (
          "post_attention_front_handoff_matvec_submit_split_route_gate"
          if required_checks_passed else
          "resident_attention_front_handoff_matvec_submit_split_profile_fix_gate"
      ),
      "next_route_reason": (
          "The handoff matvec-submit-split profile row preserves top-1 and shows the "
          "resident attention-front handoff matvec bucket is dominated by "
          "kernel wait/finish time; enqueue, setup, event profile, and "
          "queue-drain cleanup are not floor-sized. The next unit is route "
          "control, not a "
          "speed claim or another handoff matvec-submit-split row."
          if required_checks_passed else
          "Fix the attention-front handoff matvec submit-split profile row before route "
          "selection."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": metrics["speedup_claims_allowed"],
      "inputs": metrics["inputs"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  profile = metrics["attention_front_handoff_matvec_submit_split_profile"]
  summary = [
      "# Resident Attention-Front Handoff Matvec Submit-Split Profile Explore Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- largest_bucket: `{profile['largest_bucket']}`",
      f"- largest_ms_per_token: `{profile['largest_ms_per_token']:.3f}`",
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
  parser.add_argument("--seq177", type=Path, default=DEFAULT_SEQ177)
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
      "largest_bucket": (
          metrics["attention_front_handoff_matvec_submit_split_profile"]["largest_bucket"]),
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
