#!/usr/bin/env python3
"""Gate the carrier layer-output handle-loop decode correctness row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-resident-hidden-state-carrier-layer-output-handle-loop-"
    "decode-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ128 = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-target-compile-gate-20260707Tseq128Z/metrics.json"
)
DEFAULT_DISTRIBUTION = (
    ROOT
    / "output/r2-gpu-resident-hidden-state-carrier-layer-output-loop-distribution-20260707Tseq129Z/result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-decode-gate-20260707Tseq129Z"
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
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _tps(row: dict[str, Any]) -> float:
  for key in ("tps", "gpu_hybrid_decode_tok_s"):
    value = row.get(key)
    if isinstance(value, (int, float)):
      return float(value)
  tokens = _num(row.get("decode_tokens") or row.get("decode_continuation_output_tokens"))
  ns = _num(row.get("decode_ns") or row.get("gpu_hybrid_decode_ns"))
  return tokens * 1_000_000_000.0 / ns if tokens > 0.0 and ns > 0.0 else 0.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq128 = _load_json(args.seq128)
  distribution_payload = _load_json(args.distribution)
  distribution = _smoke(distribution_payload)
  dist = distribution.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  frontier_state = _frontier_state(frontier)

  top1_ok = (
      distribution.get("top1_matches_native") is True
      and _num(dist.get("top1_rate")) >= 0.99
  )
  ruler_failed = (
      distribution_payload.get("required_checks_passed") is False
      and dist.get("required_checks_passed") is False
      and (
          _num(dist.get("max_kld")) >= 0.005
          or _num(dist.get("top1_rate")) < 0.99
      )
  )
  checks = [
      {
          "name": "seq128_selected_decode_gate",
          "pass": (
              seq128.get("required_checks_passed") is True
              and seq128.get("selected_next_route")
              == "resident_hidden_state_carrier_layer_output_handle_loop_decode_gate"
              and _has_switch(
                  routes,
                  "accept_layer_output_handle_loop_compile_switch_to_decode_gate",
                  128,
              )
          ),
      },
      {
          "name": "distribution_uses_layer_output_loop_stack",
          "pass": (
              distribution.get("resident_hidden_state_carrier_enabled") is True
              and distribution.get(
                  "resident_hidden_state_carrier_preconv_bundle_enabled") is True
              and distribution.get(
                  "resident_hidden_state_carrier_selected_shared_tail_enabled") is True
              and distribution.get(
                  "resident_hidden_state_carrier_layer_output_handle_loop_enabled")
              is True
              and distribution.get("ffn_tail_resident_input_enabled") is False
              and _num(distribution.get("attention_front_handoff_layers")) > 0
          ),
          "detail": {
              "carrier": distribution.get("resident_hidden_state_carrier_enabled"),
              "preconv_bundle": distribution.get(
                  "resident_hidden_state_carrier_preconv_bundle_enabled"),
              "selected_tail": distribution.get(
                  "resident_hidden_state_carrier_selected_shared_tail_enabled"),
              "layer_output_loop": distribution.get(
                  "resident_hidden_state_carrier_layer_output_handle_loop_enabled"),
              "ffn_tail_resident_input": distribution.get(
                  "ffn_tail_resident_input_enabled"),
              "attention_front_handoff_effective": distribution.get(
                  "resident_attention_front_handoff_enabled"),
              "attention_front_handoff_layers": distribution.get(
                  "attention_front_handoff_layers"),
          },
      },
      {
          "name": "teacher_forced_top1_preserved",
          "pass": top1_ok,
          "detail": {
              "top1_matches_native": distribution.get("top1_matches_native"),
              "top1_match_count": distribution.get("top1_match_count"),
              "top1_rate": dist.get("top1_rate"),
          },
      },
      {
          "name": "distribution_ruler_rejects_current_loop",
          "pass": ruler_failed,
          "detail": {
              "payload_required_checks_passed": distribution_payload.get(
                  "required_checks_passed"),
              "distribution_required_checks_passed": dist.get(
                  "required_checks_passed"),
              "max_kld": dist.get("max_kld"),
              "top1_rate": dist.get("top1_rate"),
              "min_logits_cosine": dist.get("min_logits_cosine"),
              "logits_cosine_pass": dist.get("logits_cosine_pass"),
          },
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": frontier_state["current_best_tps"] < frontier_state["floor_tps"],
          "detail": frontier_state,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "seq128_target_compile_gate": _rel(args.seq128),
          "distribution": _rel(args.distribution),
      },
      "frontier": frontier_state,
      "distribution": {
          "tps": _tps(distribution),
          "payload_required_checks_passed": distribution_payload.get(
              "required_checks_passed"),
          "top1_matches_native": distribution.get("top1_matches_native"),
          "top1_match_count": distribution.get("top1_match_count"),
          "max_kld": dist.get("max_kld"),
          "top1_rate": dist.get("top1_rate"),
          "min_logits_cosine": dist.get("min_logits_cosine"),
          "logits_cosine_pass": dist.get("logits_cosine_pass"),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_resident_hidden_state_carrier_layer_output_handle_loop_decode_correctness"
          if required_checks_passed
          else "resident_hidden_state_carrier_layer_output_handle_loop_decode_gate_failed"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_layer_output_handle_loop_correctness_root_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_layer_output_handle_loop_decode_review"
      ),
      "next_route_reason": (
          "The loop preserves teacher-forced top-1, but the distribution ruler "
          "rejects the current source shape (max KLD exceeds 0.005). Do not run "
          "a speed row; the next unit is a correctness/root-cause gate for the "
          "layer-output handle loop."
          if required_checks_passed
          else "The decode gate checks are inconclusive; review the distribution "
               "artifact before any speed row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Resident Hidden-State Carrier Layer-Output Handle Loop Decode Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      metrics["next_route_reason"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq128", type=Path, default=DEFAULT_SEQ128)
  parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
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
