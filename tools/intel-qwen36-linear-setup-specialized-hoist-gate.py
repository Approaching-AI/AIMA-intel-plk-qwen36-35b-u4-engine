#!/usr/bin/env python3
"""Gate the specialized fixed per-layer setup hoist route.

This is route-control evidence only. It consumes artifact-free explore rows
from the fixed setup hoist cut and decides whether that source cut is a speed
candidate. It does not claim speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-linear-setup-specialized-hoist-gate-v0"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_EXPLORE = ROOT / "output/explore-log.jsonl"
DEFAULT_BASELINE_PROFILE_LABEL = "token-core-dispatch-gap-source-profile-seq99"
DEFAULT_DEFAULT_LABEL = "compile-time-default-noenv-seq105"
DEFAULT_HOIST_LABEL = "compile-time-linear-setup-hoist-seq106"
DEFAULT_HOIST_PROFILE_LABEL = "compile-time-linear-setup-hoist-source-profile-seq107"
DEFAULT_OUT_DIR = ROOT / "output/linear-setup-specialized-hoist-gate-20260707Tseq105-107Z"


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


def _profile(row: dict[str, Any], key: str) -> dict[str, Any]:
  value = row.get(key)
  return value if isinstance(value, dict) else {}


def _ratio(after: float, before: float) -> float:
  return after / before if before > 0.0 else 0.0


def _bucket_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> dict[str, Any]:
  before_value = _num(before.get(key))
  after_value = _num(after.get(key))
  return {
      "before_ns": before_value,
      "after_ns": after_value,
      "delta_ns": after_value - before_value,
      "after_over_before": _ratio(after_value, before_value),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  goal_anchor = frontier.get("goal_anchor")
  goal_anchor = goal_anchor if isinstance(goal_anchor, dict) else {}
  goal_budget = frontier.get("goal_budget")
  goal_budget = goal_budget if isinstance(goal_budget, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}

  current_best_tps = _num(goal_anchor.get("current_best_tps"))
  floor_tps = _num(goal_anchor.get("same_host_vulkan_floor_tps"))
  noise_rel = _num(noise.get("rel"))
  per_token = goal_budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  wall_ms = _num(per_token.get("wall"))
  floor_budget_ms = _num((goal_budget.get("verdict") or {}).get("floor_budget_ms_per_token"))
  floor_gap_ms = max(0.0, wall_ms - floor_budget_ms) if wall_ms > 0.0 else 0.0

  baseline_profile = _find_explore(args.explore_log, args.baseline_profile_label)
  default_row = _find_explore(args.explore_log, args.default_label)
  hoist = _find_explore(args.explore_log, args.hoist_label)
  hoist_profile = _find_explore(args.explore_log, args.hoist_profile_label)

  baseline_dispatch = _profile(baseline_profile, "dispatch_gap_source_profile_ns")
  baseline_token_core = _profile(baseline_profile, "token_core_wall_profile_ns")
  hoist_dispatch = _profile(hoist_profile, "dispatch_gap_source_profile_ns")
  hoist_token_core = _profile(hoist_profile, "token_core_wall_profile_ns")
  bucket_keys = [
      "linear_setup",
      "ffn_tail_setup",
      "full_attention_setup",
      "full_attention_core_prep",
      "profiled",
  ]
  reductions = {
      key: _bucket_delta(baseline_dispatch, hoist_dispatch, key)
      for key in bucket_keys
  }
  speed_rel_vs_best = (
      (_num(hoist.get("tps")) - current_best_tps) / current_best_tps
      if current_best_tps > 0.0 else 0.0
  )
  default_rel_vs_best = (
      (_num(default_row.get("tps")) - current_best_tps) / current_best_tps
      if current_best_tps > 0.0 else 0.0
  )
  hoist_profile_rel_vs_best = (
      (_num(hoist_profile.get("tps")) - current_best_tps) / current_best_tps
      if current_best_tps > 0.0 else 0.0
  )
  dispatch_profiled_after_ms_per_token = (
      _num(hoist_dispatch.get("profiled")) / _num(hoist_profile.get("decode_tokens")) / 1e6
      if _num(hoist_profile.get("decode_tokens")) > 0.0 else 0.0
  )
  layer_dispatch_gap_after_ms_per_token = (
      _num(hoist_token_core.get("layer_dispatch_gap")) /
      _num(hoist_profile.get("decode_tokens")) / 1e6
      if _num(hoist_profile.get("decode_tokens")) > 0.0 else 0.0
  )

  checks = [
      {
          "name": "compile_time_default_control_preserves_top1",
          "pass": (
              default_row.get("label") == args.default_label
              and default_row.get("top1_matches_native") is True
              and default_row.get("linear_setup_specialized_hoist") is False
              and default_row.get("token_core_source_profile") is False
          ),
          "detail": {
              "tps": default_row.get("tps"),
              "relative_vs_best": default_rel_vs_best,
              "source_sha": default_row.get("source_sha"),
          },
      },
      {
          "name": "specialized_hoist_speed_row_preserves_top1_but_misses_frontier",
          "pass": (
              hoist.get("label") == args.hoist_label
              and hoist.get("top1_matches_native") is True
              and hoist.get("linear_setup_specialized_hoist") is True
              and _num(hoist.get("tps")) < current_best_tps
              and _num(hoist.get("tps")) < floor_tps
          ),
          "detail": {
              "hoist_tps": hoist.get("tps"),
              "current_best_tps": current_best_tps,
              "floor_tps": floor_tps,
              "relative_vs_best": speed_rel_vs_best,
              "noise_rel": noise_rel,
              "source_sha": hoist.get("source_sha"),
          },
      },
      {
          "name": "hoist_profile_preserves_top1_and_profiles_enabled_path",
          "pass": (
              hoist_profile.get("label") == args.hoist_profile_label
              and hoist_profile.get("top1_matches_native") is True
              and hoist_profile.get("token_core_source_profile") is True
              and hoist_profile.get("linear_setup_specialized_hoist") is True
              and _num(hoist_profile.get("token_core_unprofiled_ns")) <= 50_000
          ),
          "detail": {
              "profile_tps": hoist_profile.get("tps"),
              "relative_vs_best": hoist_profile_rel_vs_best,
              "token_core_unprofiled_ns": hoist_profile.get("token_core_unprofiled_ns"),
              "source_sha": hoist_profile.get("source_sha"),
          },
      },
      {
          "name": "fixed_setup_buckets_reduced_by_specialized_hoist",
          "pass": (
              reductions["linear_setup"]["after_over_before"] < 0.10
              and reductions["ffn_tail_setup"]["after_over_before"] < 0.05
              and reductions["full_attention_setup"]["after_over_before"] < 0.10
              and reductions["profiled"]["after_over_before"] < 0.20
          ),
          "detail": reductions,
      },
      {
          "name": "remaining_fixed_setup_is_below_floor_gap",
          "pass": (
              dispatch_profiled_after_ms_per_token < floor_gap_ms
              and layer_dispatch_gap_after_ms_per_token < floor_gap_ms
          ),
          "detail": {
              "floor_gap_ms_per_token": floor_gap_ms,
              "dispatch_profiled_after_ms_per_token": dispatch_profiled_after_ms_per_token,
              "layer_dispatch_gap_after_ms_per_token": layer_dispatch_gap_after_ms_per_token,
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
          "reject_specialized_setup_hoist_as_speed_cut"
          if required else "linear_setup_specialized_hoist_gate_failed"
      ),
      "selected_next_route": (
          "post_setup_hoist_route_selection_gate"
          if required else "manual_review_linear_setup_specialized_hoist"
      ),
      "next_action": (
          "Do not promote the fixed setup hoist and do not continue sweeping "
          "setup-cache variants. The hoist proves the source bucket can be "
          "removed, but token-emitting speed remains below the frontier and "
          "the remaining fixed setup/profiled dispatch work is smaller than "
          "the floor gap. Route selection must move back to the larger "
          "goal-budget stage gaps or a kernel-side component proof."
          if required else "Fix failed gate checks before changing route."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "explore_log": _rel(args.explore_log),
          "baseline_profile_label": args.baseline_profile_label,
          "default_label": args.default_label,
          "hoist_label": args.hoist_label,
          "hoist_profile_label": args.hoist_profile_label,
      },
      "frontier": {
          "current_best_tps": current_best_tps,
          "floor_tps": floor_tps,
          "noise_rel": noise_rel,
          "wall_ms_per_token": wall_ms,
          "floor_gap_ms_per_token": floor_gap_ms,
      },
      "baseline_source_profile": {
          "ts": baseline_profile.get("ts"),
          "label": baseline_profile.get("label"),
          "source_sha": baseline_profile.get("source_sha"),
          "tps": baseline_profile.get("tps"),
          "dispatch_gap_source_profile_ns": baseline_dispatch,
          "token_core_wall_profile_ns": baseline_token_core,
      },
      "compile_time_default": {
          "ts": default_row.get("ts"),
          "label": default_row.get("label"),
          "source_sha": default_row.get("source_sha"),
          "tps": default_row.get("tps"),
          "top1_matches_native": default_row.get("top1_matches_native"),
          "relative_vs_best": default_rel_vs_best,
      },
      "specialized_hoist": {
          "ts": hoist.get("ts"),
          "label": hoist.get("label"),
          "source_sha": hoist.get("source_sha"),
          "tps": hoist.get("tps"),
          "top1_matches_native": hoist.get("top1_matches_native"),
          "relative_vs_best": speed_rel_vs_best,
          "decode_ns": hoist.get("decode_ns"),
          "kernel_sum_min_us": hoist.get("kernel_sum_min_us"),
          "token_core_unprofiled_ns": hoist.get("token_core_unprofiled_ns"),
      },
      "specialized_hoist_source_profile": {
          "ts": hoist_profile.get("ts"),
          "label": hoist_profile.get("label"),
          "source_sha": hoist_profile.get("source_sha"),
          "tps": hoist_profile.get("tps"),
          "top1_matches_native": hoist_profile.get("top1_matches_native"),
          "relative_vs_best": hoist_profile_rel_vs_best,
          "token_core_unprofiled_ns": hoist_profile.get("token_core_unprofiled_ns"),
          "dispatch_gap_source_profile_ns": hoist_dispatch,
          "token_core_wall_profile_ns": hoist_token_core,
          "dispatch_profiled_after_ms_per_token": dispatch_profiled_after_ms_per_token,
          "layer_dispatch_gap_after_ms_per_token": layer_dispatch_gap_after_ms_per_token,
      },
      "bucket_reductions": reductions,
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": "tools/intel-qwen36-linear-setup-specialized-hoist-gate.py",
      "inputs": payload["inputs"],
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in payload["checks"] if not row["pass"]]
  hoist = payload["specialized_hoist"]
  profile = payload["specialized_hoist_source_profile"]
  reductions = payload["bucket_reductions"]
  lines = [
      "# Linear Setup Specialized Hoist Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- hoist speed row: `{_num(hoist['tps']):.8f}` tok/s",
      f"- frontier/floor: `{payload['frontier']['current_best_tps']:.8f}` / `{payload['frontier']['floor_tps']:.2f}` tok/s",
      f"- hoist source-profile row: `{_num(profile['tps']):.8f}` tok/s",
      f"- token-core unprofiled after profile: `{profile['token_core_unprofiled_ns']}` ns",
      f"- dispatch profiled after hoist: `{profile['dispatch_gap_source_profile_ns'].get('profiled')}` ns",
      f"- linear setup: `{reductions['linear_setup']['before_ns']}` -> `{reductions['linear_setup']['after_ns']}` ns",
      f"- FFN-tail setup: `{reductions['ffn_tail_setup']['before_ns']}` -> `{reductions['ffn_tail_setup']['after_ns']}` ns",
      f"- full-attention setup: `{reductions['full_attention_setup']['before_ns']}` -> `{reductions['full_attention_setup']['after_ns']}` ns",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE)
  parser.add_argument("--baseline-profile-label", default=DEFAULT_BASELINE_PROFILE_LABEL)
  parser.add_argument("--default-label", default=DEFAULT_DEFAULT_LABEL)
  parser.add_argument("--hoist-label", default=DEFAULT_HOIST_LABEL)
  parser.add_argument("--hoist-profile-label", default=DEFAULT_HOIST_PROFILE_LABEL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps({
      "required_checks_passed": payload["required_checks_passed"],
      "disposition": payload["disposition"],
      "selected_next_route": payload["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
