#!/usr/bin/env python3
"""Profile-root gate for the resident/full-GPU decode loop after seq154.

This gate consumes existing profile rows and closure ledgers only. It identifies
the next floor-sized bucket that needs source-profile attribution before any
new token-emitting explore row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-full-gpu-decode-loop-profile-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_SEQ53 = ROOT / "output/q6-defer-drain-budget-20260706Tseq53Z/metrics.json"
DEFAULT_SEQ134 = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-full-core-parity-speed-gate-20260707Tseq134Z"
    / "metrics.json"
)
DEFAULT_SEQ154 = (
    ROOT
    / "output/post-full-attention-core-history-implementation-route-gate-20260708Tseq154Z"
    / "metrics.json"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = ROOT / "output/resident-full-gpu-decode-loop-profile-gate-20260708Tseq155Z"

BASELINE_LABEL = "selected-shared-q4q6-down-cold-q6-experts-profile"
LAYER_OUTPUT_LABEL = (
    "resident-hidden-state-carrier-layer-output-loop-full-core-parity-speed-seq134"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    payload = json.loads(line)
    if isinstance(payload, dict):
      rows.append(payload)
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
  for row in reversed(rows):
    if row.get("label") == label:
      return row
  raise SystemExit(f"missing explore-log row label {label!r}")


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
  glide = no_progress.get("glide_slope")
  glide = glide if isinstance(glide, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "gpu_kernel_busy_floor_ms_per_token": _num(
          per_token.get("gpu_kernel_busy_floor")),
      "non_kernel_overhead_ms_per_token": _num(
          per_token.get("non_kernel_overhead")),
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
      "can_reach_floor_without_kernel_work": bool(
          verdict.get("can_reach_floor_without_kernel_work")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
      "glide_projected_runs_to_floor": glide.get("projected_runs_to_floor"),
  }


def _stage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  rows = _nested(frontier, "goal_budget", "stage_kernel_gap_estimates_ms_per_token")
  out: dict[str, float] = {}
  if not isinstance(rows, list):
    return out
  for row in rows:
    if isinstance(row, dict) and isinstance(row.get("stage"), str):
      out[row["stage"]] = _num(row.get("gap_ms_per_token"))
  return out


def _substage_gaps(frontier: dict[str, Any]) -> dict[str, float]:
  rows = _nested(frontier, "goal_budget", "substage_gap_estimates_ms_per_token")
  out: dict[str, float] = {}
  if not isinstance(rows, list):
    return out
  for row in rows:
    if (
        isinstance(row, dict)
        and isinstance(row.get("stage"), str)
        and isinstance(row.get("substage"), str)
    ):
      out[f"{row['stage']}.{row['substage']}"] = _num(
          row.get("gap_ms_per_token"))
  return out


def _largest(mapping: dict[str, float]) -> tuple[str, float]:
  if not mapping:
    return "", 0.0
  key = max(mapping, key=lambda item: mapping[item])
  return key, mapping[key]


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  payload = row.get("profile_smoke")
  return payload if isinstance(payload, dict) else {}


def _tokens(row: dict[str, Any], profile: dict[str, Any]) -> float:
  return (
      _num(row.get("decode_tokens"))
      or _num(profile.get("decode_continuation_output_tokens"))
      or 1.0
  )


def _ms_per_token(row: dict[str, Any], *keys: str) -> float:
  profile = _profile(row)
  ns = _num(_nested(profile, *keys))
  return (ns / 1_000_000.0) / _tokens(row, profile)


def _profile_summary(row: dict[str, Any]) -> dict[str, Any]:
  profile = _profile(row)
  return {
      "label": row.get("label"),
      "ts": row.get("ts"),
      "top1_matches_native": row.get("top1_matches_native"),
      "tps": _num(row.get("tps")),
      "decode_tokens": int(_tokens(row, profile)),
      "opencl_no_queue_profiling": row.get("opencl_no_queue_profiling"),
      "selected_ffn_ms_per_token": _ms_per_token(
          row, "wall_profile_ns", "selected_ffn"),
      "ffn_tail_ms_per_token": _ms_per_token(
          row, "wall_profile_ns", "ffn_tail"),
      "attention_front_ms_per_token": _ms_per_token(
          row, "wall_profile_ns", "attention_front"),
      "selected_down_wait_ms_per_token": _ms_per_token(
          row, "selected_ffn_wall_profile_ns", "down_kernel_wait"),
      "selected_down_ms_per_token": _ms_per_token(
          row, "selected_ffn_wall_profile_ns", "down"),
      "selected_gate_up_ms_per_token": _ms_per_token(
          row, "selected_ffn_wall_profile_ns", "gate_up"),
      "selected_raw_setup_ms_per_token": _ms_per_token(
          row, "selected_ffn_wall_profile_ns", "raw_setup"),
      "token_core_unprofiled_ms_per_token": (
          _num(row.get("token_core_unprofiled_ns")) / 1_000_000.0
      ) / _tokens(row, profile),
  }


def _source_marker_state(text: str) -> dict[str, Any]:
  existing = [
      "selected_ffn_down_kernel_wait_wall_ns",
      "ffn_tail_wall_ns",
      "attention_front_wall_ns",
      "selected_ffn_wall_profile_ns",
      "gpu_loop_bookkeeping_wall_profile_ns",
  ]
  proposed_absent = [
      "resident_queue_drain_site_profile",
      "selected_down_wait_drain_site_wall_ns",
      "ffn_tail_drain_site_wall_ns",
      "attention_front_drain_site_wall_ns",
      "IQ36_RESIDENT_QUEUE_DRAIN_SITE_PROFILE",
  ]
  missing_existing = [marker for marker in existing if marker not in text]
  present_proposed = [marker for marker in proposed_absent if marker in text]
  return {
      "existing_profile_markers": existing,
      "missing_existing_profile_markers": missing_existing,
      "proposed_drain_site_markers": proposed_absent,
      "present_proposed_drain_site_markers": present_proposed,
      "existing_profile_markers_passed": not missing_existing,
      "drain_site_split_absent": not present_proposed,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq53 = _load_json(args.seq53)
  seq134 = _load_json(args.seq134)
  seq154 = _load_json(args.seq154)
  rows = _load_jsonl(args.explore_log)
  decode_source = args.decode_source.read_text(encoding="utf-8")

  frontier_state = _frontier_state(frontier)
  stage_gaps = _stage_gaps(frontier)
  substage_gaps = _substage_gaps(frontier)
  largest_stage, largest_stage_gap = _largest(stage_gaps)
  largest_substage, largest_substage_gap = _largest(substage_gaps)
  baseline_row = _label_row(rows, BASELINE_LABEL)
  carrier_row = _label_row(rows, LAYER_OUTPUT_LABEL)
  baseline = _profile_summary(baseline_row)
  carrier = _profile_summary(carrier_row)
  source_markers = _source_marker_state(decode_source)
  rejected_names = _rejected_names(rejected)
  floor_gap = frontier_state["floor_gap_ms_per_token"]
  seq53_derived = seq53.get("derived")
  seq53_derived = seq53_derived if isinstance(seq53_derived, dict) else {}
  seq53_verdict = seq53.get("verdict")
  seq53_verdict = seq53_verdict if isinstance(seq53_verdict, dict) else {}
  seq134_speed = seq134.get("speed")
  seq134_speed = seq134_speed if isinstance(seq134_speed, dict) else {}

  selected_wait_reduction = (
      baseline["selected_down_wait_ms_per_token"]
      - carrier["selected_down_wait_ms_per_token"]
  )
  selected_ffn_reduction = (
      baseline["selected_ffn_ms_per_token"] - carrier["selected_ffn_ms_per_token"]
  )
  ffn_tail_growth = (
      carrier["ffn_tail_ms_per_token"] - baseline["ffn_tail_ms_per_token"]
  )
  attention_front_growth = (
      carrier["attention_front_ms_per_token"]
      - baseline["attention_front_ms_per_token"]
  )
  drain_relocation = {
      "baseline_label": BASELINE_LABEL,
      "carrier_label": LAYER_OUTPUT_LABEL,
      "selected_down_wait_reduction_ms_per_token": selected_wait_reduction,
      "selected_ffn_reduction_ms_per_token": selected_ffn_reduction,
      "ffn_tail_growth_ms_per_token": ffn_tail_growth,
      "attention_front_growth_ms_per_token": attention_front_growth,
      "ffn_tail_plus_attention_growth_ms_per_token": (
          ffn_tail_growth + attention_front_growth),
      "net_selected_tail_attention_delta_ms_per_token": (
          -selected_ffn_reduction + ffn_tail_growth + attention_front_growth),
  }

  required_closed = {
      "current_resident_hidden_state_carrier_per_boundary_handle_board",
      "current_resident_hidden_state_carrier_layer_output_loop_full_core_parity_speed_shape",
      "current_resident_hidden_state_carrier_full_attention_qkv_handle_speed_shape",
      "current_resident_hidden_state_carrier_full_attention_core_history_handle_speed_shape",
      "gpu_q6_defer_finish_without_tail_drain_elimination",
      "gpu_q6_nonatomic_down_tail_decode_fusion",
      "gpu_selected_shared_q6_down_tail_rowgroup_local_reduce_component",
      "current_selected_ffn_kernel_layout_component_board",
      "current_moe_routed_down_fusion_board",
      "engine_resident_gpu_hot_loop_api_shell",
      "linear_setup_specialized_hoist",
  }
  missing_closed = sorted(required_closed - rejected_names)

  checks = [
      {
          "name": "seq154_selected_this_profile_gate",
          "pass": (
              seq154.get("required_checks_passed") is True
              and seq154.get("selected_next_route")
              == "resident_full_gpu_decode_loop_profile_gate"
              and _has_switch(
                  routes,
                  "select_resident_full_gpu_decode_loop_profile_gate",
                  154,
              )
          ),
          "detail": {"seq154": _rel(args.seq154)},
      },
      {
          "name": "frontier_hard_stall_requires_profile_not_probe",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
              and frontier_state["can_reach_floor_without_kernel_work"] is True
              and frontier_state["non_kernel_overhead_ms_per_token"] > floor_gap
          ),
          "detail": frontier_state,
      },
      {
          "name": "paired_profile_row_identifies_floor_sized_wait_bucket",
          "pass": (
              baseline["top1_matches_native"] is True
              and baseline["decode_tokens"] == 8
              and largest_stage == "selected_ffn"
              and largest_stage_gap > floor_gap
              and largest_substage == "selected_ffn.down_kernel_wait"
              and largest_substage_gap > floor_gap
              and baseline["selected_down_wait_ms_per_token"] > floor_gap
          ),
          "detail": {
              "baseline": baseline,
              "largest_stage": largest_stage,
              "largest_stage_gap_ms_per_token": largest_stage_gap,
              "largest_substage": largest_substage,
              "largest_substage_gap_ms_per_token": largest_substage_gap,
          },
      },
      {
          "name": "layer_output_loop_profile_confirms_drain_relocation",
          "pass": (
              carrier["top1_matches_native"] is True
              and carrier["decode_tokens"] == 8
              and selected_wait_reduction > floor_gap
              and ffn_tail_growth > floor_gap
              and (ffn_tail_growth + attention_front_growth) > floor_gap
              and _num(seq134_speed.get("relative_delta"))
              <= frontier_state["noise_rel"]
          ),
          "detail": {
              "carrier": carrier,
              "seq134_speed": seq134_speed,
              "drain_relocation": drain_relocation,
          },
      },
      {
          "name": "seq53_q6_defer_confirms_tail_drain_shift",
          "pass": (
              seq53_verdict.get("tail_drain_shift_confirmed") is True
              and _num(seq53_derived.get("selected_down_wait_saved_ms_per_token"))
              > floor_gap
              and _num(seq53_derived.get("ffn_tail_growth_ms_per_token"))
              > floor_gap
              and seq53_derived.get("tail_drain_elimination_clears_floor") is True
              and seq53_derived.get("promotion_outside_noise") is False
          ),
          "detail": {
              "seq53_derived": seq53_derived,
              "seq53_verdict": seq53_verdict,
          },
      },
      {
          "name": "closed_route_board_blocks_repeating_known_shapes",
          "pass": not missing_closed,
          "detail": {
              "missing_closed_routes": missing_closed,
              "required_closed_routes": sorted(required_closed),
          },
      },
      {
          "name": "source_has_coarse_wall_profiles_but_lacks_drain_site_split",
          "pass": (
              source_markers["existing_profile_markers_passed"]
              and source_markers["drain_site_split_absent"]
          ),
          "detail": source_markers,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)
  selected_next_route = (
      "resident_queue_drain_site_profile_source_gate"
      if required_checks_passed
      else "resident_full_gpu_decode_loop_profile_review"
  )
  disposition = (
      "select_resident_queue_drain_site_profile_source_gate"
      if required_checks_passed
      else "resident_full_gpu_decode_loop_profile_gate_incomplete"
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "explore_log": _rel(args.explore_log),
          "seq53_q6_defer_drain": _rel(args.seq53),
          "seq134_layer_output_speed_gate": _rel(args.seq134),
          "seq154_route_gate": _rel(args.seq154),
          "decode_source": _rel(args.decode_source),
      },
      "frontier": frontier_state,
      "stage_gap_ms_per_token": stage_gaps,
      "substage_gap_ms_per_token": substage_gaps,
      "baseline_profile": baseline,
      "carrier_profile": carrier,
      "drain_relocation": drain_relocation,
      "profile_findings": {
          "floor_sized_bucket": "selected_ffn.down_kernel_wait",
          "root_signal": (
              "selected-down wait can be collapsed, but current source shifts "
              "the drain into FFN-tail and attention-front rather than reducing "
              "full-loop wall"
          ),
          "minimum_next_proof": (
              "Add default-off source-profile attribution that splits selected "
              "down wait, FFN-tail, and attention-front drain/wait sites before "
              "any new speed explore row."
          ),
      },
      "source_markers": source_markers,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "source_profile_gate_allowed": required_checks_passed,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_action": (
          "Build resident_queue_drain_site_profile_source_gate. It must add "
          "default-off attribution for selected down wait, FFN-tail drain, and "
          "attention-front drain sites in the resident/full-GPU loop, without "
          "launching a speed row. The first token-emitting run after that source "
          "gate is a single artifact-free profile explore."
          if required_checks_passed
          else "Fix failed profile-gate checks before launching another token row."
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
      "tool": "tools/intel-qwen36-resident-full-gpu-decode-loop-profile-gate.py",
      "inputs": metrics["inputs"],
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  frontier = metrics["frontier"]
  drain = metrics["drain_relocation"]
  lines = [
      "# Resident Full-GPU Decode-Loop Profile Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- floor_gap_ms_per_token: `{frontier['floor_gap_ms_per_token']:.3f}`",
      f"- selected_down_wait_reduction_ms_per_token: `{drain['selected_down_wait_reduction_ms_per_token']:.3f}`",
      f"- ffn_tail_plus_attention_growth_ms_per_token: `{drain['ffn_tail_plus_attention_growth_ms_per_token']:.3f}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["profile_findings"]["minimum_next_proof"],
      "",
      "This is profile/root-cause route evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--seq53", type=Path, default=DEFAULT_SEQ53)
  parser.add_argument("--seq134", type=Path, default=DEFAULT_SEQ134)
  parser.add_argument("--seq154", type=Path, default=DEFAULT_SEQ154)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
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
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
