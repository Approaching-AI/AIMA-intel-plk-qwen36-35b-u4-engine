#!/usr/bin/env python3
"""Root-cause the layer-output handle-loop correctness failure without speed."""

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
    "correctness-root-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ129 = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-decode-gate-20260707Tseq129Z/metrics.json"
)
DEFAULT_LOOP_DISTRIBUTION = (
    ROOT
    / "output/r2-gpu-resident-hidden-state-carrier-layer-output-loop-distribution-20260707Tseq129Z/result.json"
)
DEFAULT_PASSING_BOUNDARY = (
    ROOT
    / "output/r2-gpu-resident-hidden-state-carrier-full-boundary-distribution-20260707Tseq123Z/result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-correctness-root-gate-20260707Tseq130Z"
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


def _dist(payload: dict[str, Any]) -> dict[str, Any]:
  smoke = _smoke(payload)
  value = smoke.get("distribution_ladder")
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
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq129 = _load_json(args.seq129)
  loop_payload = _load_json(args.loop_distribution)
  passing_payload = _load_json(args.passing_boundary)
  source = args.decode_source.read_text(encoding="utf-8")

  loop = _smoke(loop_payload)
  passing = _smoke(passing_payload)
  loop_dist = _dist(loop_payload)
  passing_dist = _dist(passing_payload)
  frontier_state = _frontier_state(frontier)

  source_checks = {
      "full_core_path_receives_resident_residual_handle": (
          "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm"
          in source and "attention_residual_input_handle" in source),
      "full_core_tail_uses_carrier_readback_policy": (
          "DecodeCarrierLayerOutputReadbackRequired(layer)" in source),
  }
  source_detail = dict(source_checks)
  source_detail["full_core_handoff_rejection_text_present_elsewhere"] = (
      "incompatible with --resident-full-core-attention-front-handoff" in source)
  checks = [
      {
          "name": "seq129_selected_correctness_root_gate",
          "pass": (
              seq129.get("required_checks_passed") is True
              and seq129.get("selected_next_route")
              == "resident_hidden_state_carrier_layer_output_handle_loop_correctness_root_gate"
              and _has_switch(
                  routes,
                  "reject_layer_output_handle_loop_decode_switch_to_correctness_root_gate",
                  129,
              )
          ),
      },
      {
          "name": "loop_distribution_failed_only_ruler_not_top1",
          "pass": (
              loop_payload.get("required_checks_passed") is False
              and loop.get("top1_matches_native") is True
              and _num(loop_dist.get("top1_rate")) >= 0.99
              and _num(loop_dist.get("max_kld")) >= 0.005
          ),
          "detail": {
              "max_kld": loop_dist.get("max_kld"),
              "top1_rate": loop_dist.get("top1_rate"),
              "min_logits_cosine": loop_dist.get("min_logits_cosine"),
          },
      },
      {
          "name": "passing_carrier_boundary_kept_full_core_handoff",
          "pass": (
              passing_payload.get("required_checks_passed") is True
              and passing.get("top1_matches_native") is True
              and passing_dist.get("required_checks_passed") is True
              and passing.get("resident_full_core_attention_front_handoff_enabled")
              is True
              and _num(passing.get("full_core_attention_front_handoff_layers")) > 0
          ),
          "detail": {
              "max_kld": passing_dist.get("max_kld"),
              "full_core_handoff_layers": passing.get(
                  "full_core_attention_front_handoff_layers"),
          },
      },
      {
          "name": "failing_loop_disabled_full_core_handoff",
          "pass": (
              loop.get("resident_hidden_state_carrier_layer_output_handle_loop_enabled")
              is True
              and loop.get("resident_full_core_attention_front_handoff_enabled")
              is False
              and _num(loop.get("full_core_attention_front_handoff_layers")) == 0.0
          ),
          "detail": {
              "loop_enabled": loop.get(
                  "resident_hidden_state_carrier_layer_output_handle_loop_enabled"),
              "full_core_handoff_enabled": loop.get(
                  "resident_full_core_attention_front_handoff_enabled"),
              "full_core_handoff_layers": loop.get(
                  "full_core_attention_front_handoff_layers"),
              "attention_front_handoff_layers": loop.get(
                  "attention_front_handoff_layers"),
          },
      },
      {
          "name": "source_has_full_core_parity_handle_path",
          "pass": all(source_checks.values()),
          "detail": source_detail,
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
          "decode_source": _rel(args.decode_source),
          "seq129_decode_gate": _rel(args.seq129),
          "loop_distribution": _rel(args.loop_distribution),
          "passing_boundary": _rel(args.passing_boundary),
      },
      "frontier": frontier_state,
      "comparison": {
          "loop_max_kld": loop_dist.get("max_kld"),
          "loop_top1_rate": loop_dist.get("top1_rate"),
          "loop_full_core_handoff_layers": loop.get(
              "full_core_attention_front_handoff_layers"),
          "passing_max_kld": passing_dist.get("max_kld"),
          "passing_top1_rate": passing_dist.get("top1_rate"),
          "passing_full_core_handoff_layers": passing.get(
              "full_core_attention_front_handoff_layers"),
      },
      "source_checks": source_checks,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "bind_layer_output_loop_correctness_root_to_full_core_handoff_parity"
          if required_checks_passed
          else "layer_output_loop_correctness_root_evidence_incomplete"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_layer_output_handle_loop_full_core_parity_source_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_layer_output_handle_loop_correctness_review"
      ),
      "next_route_reason": (
          "The failing loop row changed two things at once: it enabled the "
          "layer-output handle loop and disabled the accepted full-core "
          "attention-front handoff. Source now has a resident residual handle "
          "path and carrier readback policy in the full-core handoff branch. "
          "Gate that parity path before any speed row."
          if required_checks_passed
          else "The root-cause evidence is incomplete; inspect the loop and "
               "passing boundary artifacts before adding another decode row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Resident Hidden-State Carrier Layer-Output Handle Loop Correctness Root Gate",
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
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq129", type=Path, default=DEFAULT_SEQ129)
  parser.add_argument(
      "--loop-distribution", type=Path, default=DEFAULT_LOOP_DISTRIBUTION)
  parser.add_argument(
      "--passing-boundary", type=Path, default=DEFAULT_PASSING_BOUNDARY)
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
