#!/usr/bin/env python3
"""Record the soft-reflection route decision from existing evidence.

This gate is deliberately local: it reads the current frontier plus the already
accepted/rejected route gates and writes one machine artifact that answers the
ch.3 §3.5 question before another probe is launched. It is route-selection
evidence only, not runtime speed evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc" / "active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-soft-reflection-route-gate-v0"
DEFAULT_OUT_DIR = ROOT / "output/soft-reflection-route-gate-20260706Tseq78Z"


def load_json(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return payload


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def candidate_by_seq(routes: dict[str, Any], seq: int) -> dict[str, Any]:
  for row in routes.get("candidate_history", []):
    if isinstance(row, dict) and row.get("seq") == seq:
      return row
  raise SystemExit(f"routes-ledger.json: missing candidate seq {seq}")


def rejected_by_route(rejected: dict[str, Any], route: str) -> dict[str, Any] | None:
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and row.get("route") == route:
      return row
  return None


def num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def check(label: str, passed: bool, detail: dict[str, Any] | None = None) -> dict[str, Any]:
  row: dict[str, Any] = {"label": label, "pass": bool(passed)}
  if detail is not None:
    row["detail"] = detail
  return row


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = load_json(args.frontier)
  routes = load_json(args.routes)
  rejected = load_json(args.rejected)
  dpas = load_json(args.dpas_gate)
  down_tail = load_json(args.down_tail_gate)
  shared_q8 = load_json(args.shared_q8_gate)

  seq52 = candidate_by_seq(routes, 52)
  seq66 = candidate_by_seq(routes, 66)
  seq70 = candidate_by_seq(routes, 70)
  seq77 = candidate_by_seq(routes, 77)

  no_progress = frontier.get("no_progress", {})
  goal_anchor = frontier.get("goal_anchor", {})
  goal_budget = frontier.get("goal_budget", {})
  budget_verdict = goal_budget.get("verdict", {}) if isinstance(goal_budget, dict) else {}
  last_improvement = no_progress.get("last_significant_improvement", {})
  best_ts = last_improvement.get("ts")

  dpas_derived = dpas.get("derived", {})
  dpas_budget = dpas.get("budget_source", {})
  dpas_selected_targets = dpas_derived.get("selected_budget_share_targets", {})
  dpas_required_fit_whole = num(
      dpas_derived.get("selected_swiglu_required_speedup_to_fit_whole_layer_budget")
  )
  dpas_max_existing_ratio = num(
      dpas_derived.get("max_existing_ratio_bound")
  )
  dpas_target_layer_us = num(dpas_budget.get("target_layer_budget_us"))
  dpas_selected_us = num(dpas_budget.get("selected_swiglu_fusion_us"))
  dpas_half_share_us = num(
      (dpas_selected_targets.get("0.500") or {}).get("selected_budget_us")
  )
  dpas_half_share_speedup = num(
      (dpas_selected_targets.get("0.500") or {}).get("required_speedup")
  )

  down_derived = down_tail.get("derived", {})
  shared_verdict = shared_q8.get("verdict", {})

  rejected_direct = rejected_by_route(
      rejected, "gpu_direct_q6_down_tail_atomic_from_q6_workitems"
  )
  rejected_shared_q8 = rejected_by_route(
      rejected, "gpu_linear_preconv_shared_q8_preconv_bundle_decode"
  )

  checks = [
      check(
          "frontier_soft_reflection_breached",
          bool(no_progress.get("soft_reflection_breached"))
          and not bool(no_progress.get("hard_stall_breached")),
          {
              "runs_since_significant_improvement": no_progress.get(
                  "runs_since_significant_improvement"
              ),
              "soft_threshold": no_progress.get("soft_reflection_threshold"),
              "hard_threshold": no_progress.get("hard_stall_threshold"),
          },
      ),
      check(
          "current_best_below_floor_but_close",
          num(goal_anchor.get("current_best_tps"))
          < num(goal_anchor.get("same_host_vulkan_floor_tps"))
          and num(budget_verdict.get("overhead_only_ceiling_tok_s"))
          > num(goal_anchor.get("same_host_vulkan_floor_tps")),
          {
              "current_best_tps": goal_anchor.get("current_best_tps"),
              "floor_tps": goal_anchor.get("same_host_vulkan_floor_tps"),
              "floor_gap_ms_per_token": num(goal_budget.get("per_token_ms", {}).get("wall"))
              - num(budget_verdict.get("floor_budget_ms_per_token")),
              "overhead_only_ceiling_tok_s": budget_verdict.get(
                  "overhead_only_ceiling_tok_s"
              ),
          },
      ),
      check(
          "shared_q8_speed_route_closed",
          bool(shared_verdict.get("shared_q8_profile_closes_speed_route"))
          and seq77.get("disposition") == "rejected_shared_q8_preconv_speed_route_regresses_8tok"
          and rejected_shared_q8 is not None,
          {
              "seq77_delta_pct_vs_current_source_baseline": seq77.get(
                  "delta_pct_vs_current_source_baseline"
              ),
              "seq77_candidate_tps": seq77.get("shared_q8_candidate_tps"),
          },
      ),
      check(
          "current_dpas_occupancy_same_shape_closed",
          bool(dpas_derived.get("occupancy_only_gateup_tiling_closed"))
          and dpas_required_fit_whole > dpas_max_existing_ratio,
          {
              "seq52_disposition": seq52.get("disposition"),
              "selected_swiglu_us": dpas_selected_us,
              "target_layer_budget_us": dpas_target_layer_us,
              "required_speedup_to_fit_whole_layer": dpas_required_fit_whole,
              "max_existing_ratio_bound": dpas_max_existing_ratio,
          },
      ),
      check(
          "naive_down_tail_serial_shape_closed",
          bool(down_derived.get("hidden_row_serial_q6_down_tail_fusion_closed"))
          and seq66.get("disposition") == "route_shape_gate_rejects_naive_serial_fusion",
          {
              "parallelism_collapse_factor": seq66.get(
                  "q6_down_parallelism_collapse_factor_for_hidden_row_serial_fusion"
              ),
              "minimum_shape": down_derived.get("minimum_admissible_fusion_shape"),
          },
      ),
      check(
          "direct_atomic_down_tail_closed",
          seq70.get("disposition") == "rejected_direct_q6_down_tail_regression"
          and rejected_direct is not None,
          {
              "seq70_tps": seq70.get("measured_tps"),
              "seq70_selected_ffn_wall_ns": seq70.get("selected_ffn_wall_ns"),
              "seq70_ffn_tail_wall_ns": seq70.get("ffn_tail_wall_ns"),
          },
      ),
  ]

  selected_next_route = "dpas_storage_work_distribution_design_gate"
  next_contract = {
      "route": selected_next_route,
      "why_this_branch": (
          "The down-to-tail branch has closed local carrier, hidden-row serial, "
          "post-down atomic, and direct atomic shapes. DPAS still has exactness "
          "evidence, but only if the next proof changes storage, tiling, or work "
          "distribution beyond the current Q4 occupancy/full-tensor bounds."
      ),
      "must_not_repeat": [
          "shared-Q8 preconv without a specific qkv_conv regression fix",
          "DPAS occupancy-only/local-size/dispatch fill on the current lane",
          "hidden-row serial down-to-tail fusion",
          "post-down or direct global-atomic down-to-tail reduction",
      ],
      "minimum_target": {
          "selected_fused_gateup_swiglu_current_us": dpas_selected_us,
          "whole_layer_budget_us_at_8k_prefill": dpas_target_layer_us,
          "required_speedup_to_fit_whole_layer": dpas_required_fit_whole,
          "existing_ratio_bound": dpas_max_existing_ratio,
          "verdict": (
              "A DPAS follow-up is admissible only if it can plausibly exceed "
              "the existing ratio bound and move selected fused gate/up-to-SwiGLU "
              "toward <= whole-layer budget."
          ),
      },
      "preferred_target": {
          "selected_half_layer_share_budget_us": dpas_half_share_us,
          "required_speedup_for_half_layer_share": dpas_half_share_speedup,
      },
      "fallback_if_not_met": (
          "Return to a materially different non-atomic down-to-tail component "
          "proof that preserves rows_per_expert*9 contributor parallelism and "
          "does not use global float atomics."
      ),
  }

  all_checks_passed = all(row["pass"] for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "speedup_claims_allowed": False,
      "inputs": {
          "frontier": display_path(args.frontier),
          "routes": display_path(args.routes),
          "rejected": display_path(args.rejected),
          "dpas_gate": display_path(args.dpas_gate),
          "down_tail_gate": display_path(args.down_tail_gate),
          "shared_q8_gate": display_path(args.shared_q8_gate),
      },
      "frontier": {
          "best_ts": best_ts,
          "current_best_tps": goal_anchor.get("current_best_tps"),
          "floor_tps": goal_anchor.get("same_host_vulkan_floor_tps"),
          "last_significant_artifact": last_improvement.get("artifact"),
          "runs_since_significant_improvement": no_progress.get(
              "runs_since_significant_improvement"
          ),
          "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
          "hard_stall_breached": no_progress.get("hard_stall_breached"),
          "glide_projected_runs_to_floor": (
              (no_progress.get("glide_slope") or {}).get("projected_runs_to_floor")
          ),
          "review_recorded_before_this_gate": no_progress.get(
              "review_recorded_for_current_best"
          ),
      },
      "checks": checks,
      "verdict": {
          "soft_reflection_review_ready": all_checks_passed,
          "selected_next_route": selected_next_route,
          "record_goal_stall_review_for_best_ts": best_ts,
          "reason": (
              "The current decode route is close enough that random microcuts are "
              "the risk, not a need to abandon the GPU route. Existing evidence "
              "closes shared-Q8 preconv, current-lane DPAS occupancy fill, and "
              "serial/atomic down-to-tail shapes. The next bounded move is a "
              "DPAS storage/work-distribution design gate with an explicit "
              ">2.377x selected-lane target, otherwise fall back to non-atomic "
              "down-to-tail."
          ),
      },
      "next_proof_contract": next_contract,
  }


def write_outputs(out_dir: Path, metrics: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "metrics": "metrics.json",
      "speedup_claims_allowed": False,
      "selected_next_route": metrics["verdict"]["selected_next_route"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  verdict = metrics["verdict"]
  contract = metrics["next_proof_contract"]["minimum_target"]
  summary = [
      "# Soft-Reflection Route Gate",
      "",
      f"- selected_next_route: `{verdict['selected_next_route']}`",
      f"- soft_reflection_review_ready: `{str(verdict['soft_reflection_review_ready']).lower()}`",
      f"- record_goal_stall_review_for_best_ts: `{verdict['record_goal_stall_review_for_best_ts']}`",
      f"- required DPAS selected-lane speedup: `{contract['required_speedup_to_fit_whole_layer']:.6f}x`",
      f"- existing ratio bound: `{contract['existing_ratio_bound']:.6f}x`",
      "",
      verdict["reason"],
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--frontier", type=Path, default=ACTIVE / "frontier.json")
  ap.add_argument("--routes", type=Path, default=ACTIVE / "routes-ledger.json")
  ap.add_argument("--rejected", type=Path, default=ACTIVE / "rejected-routes.json")
  ap.add_argument(
      "--dpas-gate",
      type=Path,
      default=ROOT / "output/dpas-gateup-tiling-budget-20260706Tseq52Z/metrics.json",
  )
  ap.add_argument(
      "--down-tail-gate",
      type=Path,
      default=ROOT / "output/down-tail-fusion-budget-20260706Tseq66Z/metrics.json",
  )
  ap.add_argument(
      "--shared-q8-gate",
      type=Path,
      default=ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z/metrics.json",
  )
  ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = ap.parse_args()

  metrics = compute(args)
  write_outputs(args.out_dir, metrics)
  print(display_path(args.out_dir))
  return 0 if metrics["verdict"]["soft_reflection_review_ready"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
