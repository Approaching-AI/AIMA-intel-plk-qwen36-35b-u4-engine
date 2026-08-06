#!/usr/bin/env python3
"""Gate the full-attention core/history implementation speed explore row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-resident-hidden-state-carrier-full-attention-core-history-"
    "implementation-speed-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ152 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-decode-gate-20260707Tseq152Z"
    / "metrics.json"
)
DEFAULT_EXPLORE_LOG = ROOT / "output/explore-log.jsonl"
DEFAULT_LABEL = "resident-hidden-state-carrier-full-attention-core-history-implementation-speed-seq153"
DEFAULT_BEST = (
    ROOT
    / "output/r2-gpu-selected-shared-q4q6-down-cold-q6-experts-noqueue-speed-20260705T143006Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-speed-gate-20260707Tseq153Z"
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


def _stage_ms_per_token(row: dict[str, Any], stage: str) -> float:
  prof = _profile(row)
  wall = prof.get("wall_profile_ns")
  if not isinstance(wall, dict):
    return 0.0
  tokens = _tokens(row)
  return _num(wall.get(stage)) / tokens / 1e6 if tokens > 0.0 else 0.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq152 = _load_json(args.seq152)
  best_payload = _load_json(args.best)
  explore = _find_explore_row(args.explore_log, args.label)
  frontier_state = _frontier_state(frontier)
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
  full_qk_growth = _stage_ms_per_token(explore, "full_qk") - _stage_ms_per_token(
      best_payload, "full_qk")
  full_v_growth = _stage_ms_per_token(explore, "full_v") - _stage_ms_per_token(
      best_payload, "full_v")
  rope_growth = _stage_ms_per_token(explore, "full_rope_cpu") - _stage_ms_per_token(
      best_payload, "full_rope_cpu")

  checks = [
      {
          "name": "seq152_selected_speed_explore_gate",
          "pass": (
              seq152.get("required_checks_passed") is True
              and seq152.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_core_history_implementation_speed_explore_gate"
              and _has_switch(
                  routes,
                  "accept_full_attention_core_history_implementation_decode_switch_to_speed_explore_gate",
                  152,
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
          "name": "explore_row_uses_core_history_handle_stack",
          "pass": (
              explore.get("defer_ffn_down_finish_bundle") is True
              and explore.get(
                  "resident_hidden_state_carrier_full_attention_core_history_handle")
              is True
              and explore.get(
                  "resident_hidden_state_carrier_full_attention_qkv_handle")
              is True
              and explore.get(
                  "resident_hidden_state_carrier_layer_output_handle_loop")
              is True
              and explore.get(
                  "resident_hidden_state_carrier_selected_shared_tail")
              is True
              and explore.get("selected_shared_q4_gateup_combined") is True
              and explore.get("selected_shared_q4_down_combined") is True
              and explore.get("selected_shared_q6_down_combined") is True
              and explore.get("reuse_selected_q8_for_shared_ffn") is True
              and explore.get("ffn_tail_resident_input") is False
              and explore.get("linear_final_device_q8_handoff") is False
              and explore.get("attention_front_resident_residual_input") is False
              and explore.get("opencl_no_queue_profiling") is True
          ),
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
          "name": "full_attention_boundary_growth_explains_regression",
          "pass": (
              full_qk_growth + full_v_growth + rope_growth > 0.0
              and wall_delta_ms_per_token > 0.0
          ),
          "detail": {
              "full_qk_growth_ms_per_token": full_qk_growth,
              "full_v_growth_ms_per_token": full_v_growth,
              "full_rope_cpu_growth_ms_per_token": rope_growth,
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
          "seq152_decode_gate": _rel(args.seq152),
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
      "route_signal": {
          "full_qk_growth_ms_per_token": full_qk_growth,
          "full_v_growth_ms_per_token": full_v_growth,
          "full_rope_cpu_growth_ms_per_token": rope_growth,
          "verdict": (
              "The core/history handle implementation is correctness-valid "
              "but materially slower than the current frontier. It does not "
              "earn a confirm or promotion row."
          ),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_full_attention_core_history_implementation_as_speed_cut"
          if required_checks_passed
          else "full_attention_core_history_implementation_speed_gate_incomplete"
      ),
      "selected_next_route": (
          "post_full_attention_core_history_implementation_route_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_full_attention_core_history_implementation_speed_review"
      ),
      "next_route_reason": (
          "Close this core/history implementation as a speed cut. The next "
          "unit is route-control from existing evidence, not another "
          "core/history handle speed row."
          if required_checks_passed
          else "The speed-gate evidence is incomplete; do not launch another "
               "core/history handle row until the mismatch is resolved."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  speed = metrics["speed"]
  lines = [
      "# Resident Hidden-State Carrier Full-Attention Core/History Implementation Speed Gate",
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
      metrics["route_signal"]["verdict"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq152", type=Path, default=DEFAULT_SEQ152)
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
