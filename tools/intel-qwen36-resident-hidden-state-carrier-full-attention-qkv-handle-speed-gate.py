#!/usr/bin/env python3
"""Gate the full-attention QK/V handle speed explore row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-resident-hidden-state-carrier-full-attention-qkv-handle-"
    "speed-gate-v2"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ139 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-decode-gate-20260707Tseq139Z/metrics.json"
)
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_LABEL = "resident-hidden-state-carrier-full-attention-qkv-handle-speed-seq140"
DEFAULT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z/result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-speed-gate-20260707Tseq140Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _smoke(payload: dict[str, Any]) -> dict[str, Any]:
  value = payload.get("smoke")
  return value if isinstance(value, dict) else payload


def _profile(row: dict[str, Any]) -> dict[str, Any]:
  value = row.get("profile_smoke")
  if isinstance(value, dict):
    return value
  return _smoke(row)


def _cache(row: dict[str, Any]) -> dict[str, Any]:
  value = row.get("cache")
  return value if isinstance(value, dict) else {}


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


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
  }


def _find_explore_row(path: Path, label: str) -> dict[str, Any]:
  selected: dict[str, Any] | None = None
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    row = json.loads(line)
    if isinstance(row, dict) and row.get("label") == label:
      selected = row
  if selected is None:
    raise SystemExit(f"{path}: no explore row with label {label!r}")
  return selected


def _tps(row: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = row.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  smoke = _smoke(row)
  value = smoke.get("gpu_hybrid_decode_tok_s")
  if isinstance(value, (int, float)):
    return float(value)
  tokens = _num(row.get("decode_tokens") or smoke.get("decode_continuation_output_tokens"))
  ns = _num(row.get("decode_ns") or smoke.get("gpu_hybrid_decode_ns"))
  return tokens * 1_000_000_000.0 / ns if tokens > 0.0 and ns > 0.0 else 0.0


def _tokens(row: dict[str, Any]) -> float:
  smoke = _smoke(row)
  return _num(
      row.get("decode_tokens")
      or smoke.get("decode_continuation_output_tokens")
      or smoke.get("decode_tokens_per_session")
  )


def _decode_ns(row: dict[str, Any]) -> float:
  smoke = _smoke(row)
  return _num(row.get("decode_ns") or smoke.get("gpu_hybrid_decode_ns"))


def _stage_total_ns(row: dict[str, Any], stage: str) -> float:
  prof = _profile(row)
  wall = prof.get("wall_profile_ns")
  if not isinstance(wall, dict):
    return 0.0
  return _num(wall.get(stage))


def _stage_ms_per_token(row: dict[str, Any], stage: str) -> float:
  tokens = _tokens(row)
  total = _stage_total_ns(row, stage)
  return total / tokens / 1e6 if tokens > 0.0 else 0.0


def _stage_deltas(best: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, float]]:
  stages = [
      "full_qk",
      "full_v",
      "selected_ffn",
      "ffn_tail",
      "attention_front",
      "full_core",
      "layer_input_rmsnorm",
      "linear_preconv",
      "linear_delta",
      "lm_head_gpu",
      "router_gpu",
  ]
  out: dict[str, dict[str, float]] = {}
  for stage in stages:
    best_ms = _stage_ms_per_token(best, stage)
    candidate_ms = _stage_ms_per_token(candidate, stage)
    out[stage] = {
        "best_ms_per_token": round(best_ms, 6),
        "candidate_ms_per_token": round(candidate_ms, 6),
        "delta_ms_per_token": round(candidate_ms - best_ms, 6),
    }
  return out


def _stack_flags(row: dict[str, Any]) -> dict[str, Any]:
  return {
      "defer_ffn_down_finish_bundle": row.get(
          "defer_ffn_down_finish_bundle"),
      "resident_hidden_state_carrier_full_attention_qkv_handle": row.get(
          "resident_hidden_state_carrier_full_attention_qkv_handle"),
      "resident_hidden_state_carrier_layer_output_handle_loop": row.get(
          "resident_hidden_state_carrier_layer_output_handle_loop"),
      "resident_hidden_state_carrier_selected_shared_tail": row.get(
          "resident_hidden_state_carrier_selected_shared_tail"),
      "selected_shared_q4_gateup_combined": row.get(
          "selected_shared_q4_gateup_combined"),
      "selected_shared_q4_down_combined": row.get(
          "selected_shared_q4_down_combined"),
      "selected_shared_q6_down_combined": row.get(
          "selected_shared_q6_down_combined"),
      "reuse_selected_q8_for_shared_ffn": row.get(
          "reuse_selected_q8_for_shared_ffn"),
      "ffn_tail_resident_input": row.get("ffn_tail_resident_input"),
      "linear_final_device_q8_handoff": row.get(
          "linear_final_device_q8_handoff"),
      "attention_front_resident_residual_input": row.get(
          "attention_front_resident_residual_input"),
      "opencl_no_queue_profiling": row.get("opencl_no_queue_profiling"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq139 = _load_json(args.seq139)
  best_payload = _load_json(args.best)
  explore = _find_explore_row(args.explore_log, args.label)
  frontier_state = _frontier_state(frontier)
  flags = _stack_flags(explore)
  best_flags = _stack_flags(best_payload)
  explore_cache = _cache(explore)
  best_cache = _cache(best_payload)

  best_tps = _tps(best_payload)
  explore_tps = _tps(explore)
  noise_rel = frontier_state["noise_rel"]
  rel_delta = (explore_tps - best_tps) / best_tps if best_tps > 0.0 else 0.0
  tokens = _tokens(explore)
  wall_delta_ms_per_token = (
      (_decode_ns(explore) - _decode_ns(best_payload)) / tokens / 1e6
      if tokens > 0.0 else 0.0
  )
  deltas = _stage_deltas(best_payload, explore)
  qkv_wall_growth = (
      deltas["full_qk"]["delta_ms_per_token"]
      + deltas["full_v"]["delta_ms_per_token"]
  )

  checks = [
      {
          "name": "seq139_selected_speed_explore_gate",
          "pass": (
              seq139.get("required_checks_passed") is True
              and seq139.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_qkv_handle_speed_explore_gate"
              and _has_switch(
                  routes,
                  "accept_full_attention_qkv_handle_decode_switch_to_speed_explore_gate",
                  139,
              )
          ),
      },
      {
          "name": "explore_row_is_artifact_free_speed_probe",
          "pass": (
              explore.get("schema") == "iq36-explore-log-v0"
              and explore.get("label") == args.label
              and explore.get("kind") == "r2-gpu-decode-smoke"
              and explore.get("required_checks_passed") is False
              and explore.get("top1_matches_native") is True
              and explore_cache.get("binary_hit") is True
              and explore_cache.get("tokens_hit") is True
              and best_cache.get("binary_hit") is True
              and best_cache.get("tokens_hit") is True
              and "gpu_decode_required_checks_passed" in (
                  explore.get("failed_checks") or [])
          ),
          "detail": {
              "ts": explore.get("ts"),
              "failed_checks": explore.get("failed_checks"),
              "top1_matches_native": explore.get("top1_matches_native"),
              "candidate_cache": explore_cache,
              "frontier_cache": best_cache,
          },
      },
      {
          "name": "explore_row_uses_qkv_handle_stack",
          "pass": (
              flags["defer_ffn_down_finish_bundle"] is True
              and flags["defer_ffn_down_finish_bundle"]
              == best_flags["defer_ffn_down_finish_bundle"]
              and flags["resident_hidden_state_carrier_full_attention_qkv_handle"] is True
              and flags["resident_hidden_state_carrier_layer_output_handle_loop"] is True
              and flags["resident_hidden_state_carrier_selected_shared_tail"] is True
              and flags["selected_shared_q4_gateup_combined"] is True
              and flags["selected_shared_q4_down_combined"] is True
              and flags["selected_shared_q6_down_combined"] is True
              and flags["reuse_selected_q8_for_shared_ffn"] is True
              and flags["ffn_tail_resident_input"] is False
              and flags["linear_final_device_q8_handoff"] is False
              and flags["attention_front_resident_residual_input"] is False
              and flags["opencl_no_queue_profiling"] is True
          ),
          "detail": {
              "candidate_flags": flags,
              "frontier_flags": best_flags,
          },
      },
      {
          "name": "speed_delta_is_significant_regression",
          "pass": (
              explore_tps < best_tps
              and rel_delta <= -max(noise_rel, 0.0)
              and frontier_state["current_best_tps"] < frontier_state["floor_tps"]
          ),
          "detail": {
              "best_tps": best_tps,
              "explore_tps": explore_tps,
              "relative_delta": rel_delta,
              "noise_rel": noise_rel,
              "wall_delta_ms_per_token": wall_delta_ms_per_token,
          },
      },
      {
          "name": "qkv_handle_wall_grew_not_speed_claim",
          "pass": qkv_wall_growth > 0.0 and wall_delta_ms_per_token > 0.0,
          "detail": {
              "qk_plus_v_growth_ms_per_token": qkv_wall_growth,
              "wall_delta_ms_per_token": wall_delta_ms_per_token,
          },
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "seq139_decode_gate": _rel(args.seq139),
          "explore_log": _rel(args.explore_log),
          "explore_label": args.label,
          "best_result": _rel(args.best),
      },
      "frontier": frontier_state,
      "speed": {
          "best_tps": best_tps,
          "explore_tps": explore_tps,
          "relative_delta": rel_delta,
          "relative_delta_pct": rel_delta * 100.0,
          "noise_rel": noise_rel,
          "wall_delta_ms_per_token": wall_delta_ms_per_token,
          "speedup_claims_allowed": False,
      },
      "stage_deltas_ms_per_token": deltas,
      "route_signal": {
          "qk_plus_v_growth_ms_per_token": qkv_wall_growth,
          "selected_ffn_delta_ms_per_token": (
              deltas["selected_ffn"]["delta_ms_per_token"]),
          "verdict": (
              "The QK/V handle path is correctness-valid but regresses the "
              "speed row materially versus the current frontier. It does not "
              "earn a confirm or promotion row."
          ),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_full_attention_qkv_handle_as_speed_cut"
          if required_checks_passed
          else "full_attention_qkv_handle_speed_gate_incomplete"
      ),
      "selected_next_route": (
          "post_full_attention_qkv_handle_route_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_full_attention_qkv_handle_speed_review"
      ),
      "next_route_reason": (
          "Close the current full-attention QK/V handle speed shape. The next "
          "unit is route-control from existing evidence, not another QK/V "
          "handle speed row."
          if required_checks_passed
          else "The speed-gate evidence is incomplete; do not launch another "
               "QK/V handle row until the mismatch is resolved."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  speed = metrics["speed"]
  signal = metrics["route_signal"]
  lines = [
      "# Resident Hidden-State Carrier Full-Attention QK/V Handle Speed Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      (
          f"Explore speed was `{speed['explore_tps']:.8f}` tok/s versus "
          f"`{speed['best_tps']:.8f}` tok/s, relative delta "
          f"`{speed['relative_delta_pct']:.3f}%` with a "
          f"`{speed['noise_rel'] * 100:.2f}%` noise band."
      ),
      "",
      signal["verdict"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq139", type=Path, default=DEFAULT_SEQ139)
  parser.add_argument("--explore-log", type=Path, default=DEFAULT_EXPLORE_LOG)
  parser.add_argument("--label", default=DEFAULT_LABEL)
  parser.add_argument("--best", type=Path, default=DEFAULT_BEST)
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
