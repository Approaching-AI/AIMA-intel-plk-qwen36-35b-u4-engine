#!/usr/bin/env python3
"""Bind the seq84 attention/linear regression to closed root causes.

This gate does not run the target and does not create speed evidence. It
prevents another blind attention/linear decode probe by comparing seq84 against
the earlier simple linear-final and shared-Q8 preconv closures, then checking
the source still contains the corresponding device-Q8/readback and shared-Q8
qkv/conv chains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-attn-linear-regression-root-gate-v0"
DEFAULT_SEQ84 = ROOT / "output/attn-linear-event-lifetime-decode-gate-20260707Tseq84Z/metrics.json"
DEFAULT_SEQ77 = ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z/metrics.json"
DEFAULT_SEQ51 = ROOT / "output/attn-linear-handoff-budget-20260706Tseq51Z/metrics.json"
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OUT_DIR = ROOT / "output/attn-linear-regression-root-gate-20260707Tseq85Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _closed_route_names(rejected: dict[str, Any], *needles: str) -> list[str]:
  rows = rejected.get("rejected")
  if not isinstance(rows, list):
    return []
  lower_needles = tuple(item.lower() for item in needles)
  out: list[str] = []
  for row in rows:
    if not isinstance(row, dict):
      continue
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("route", "class", "reason", "runtime_cleanup")
    ).lower()
    if any(needle in haystack for needle in lower_needles):
      route = row.get("route")
      if isinstance(route, str):
        out.append(route)
  return out


def _has_markers(text: str, markers: list[str]) -> dict[str, Any]:
  missing = [marker for marker in markers if marker not in text]
  return {
      "pass": not missing,
      "missing": missing,
      "marker_count": len(markers),
  }


def _frontier_summary(frontier: dict[str, Any]) -> dict[str, Any]:
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(
          (no_progress.get("last_significant_improvement") or {}).get("tps")
          if isinstance(no_progress.get("last_significant_improvement"), dict)
          else None
      ),
      "floor_tps": _num(verdict.get("floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"
      ),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  seq84 = _load_json(args.seq84)
  seq77 = _load_json(args.seq77)
  seq51 = _load_json(args.seq51)
  frontier = _load_json(args.frontier)
  rejected = _load_json(args.rejected)
  source = "\n".join([
      _read(args.decode_source),
      _read(args.engine_source),
      _read(args.engine_header),
  ])

  seq84_derived = seq84.get("derived") if isinstance(seq84, dict) else {}
  seq84_derived = seq84_derived if isinstance(seq84_derived, dict) else {}
  seq77_derived = seq77.get("derived") if isinstance(seq77, dict) else {}
  seq77_derived = seq77_derived if isinstance(seq77_derived, dict) else {}
  seq51_derived = seq51.get("derived") if isinstance(seq51, dict) else {}
  seq51_derived = seq51_derived if isinstance(seq51_derived, dict) else {}
  frontier_state = _frontier_summary(frontier)

  attention_delta = _num(seq84_derived.get("attention_front_delta_ms_per_token"))
  simple_attention_delta = _num(seq51_derived.get("attention_front_growth_ms_per_token"))
  qkv_delta = _num(seq84_derived.get("linear_preconv_qkv_conv_delta_ms_per_token"))
  shared_q8_qkv_delta = _num(seq77_derived.get("linear_preconv_qkv_conv_ms_per_token"))
  final_read_saved = -_num(seq84_derived.get("linear_delta_final_read_delta_ms_per_token"))
  preconv_input_q8_saved = _num(seq84["rows"]["baseline"]["linear_preconv_input_q8_ms_per_token"])
  preconv_alpha_beta_saved = _num(seq84["rows"]["baseline"]["linear_preconv_alpha_beta_ms_per_token"])
  measured_savings = final_read_saved + preconv_input_q8_saved + preconv_alpha_beta_saved
  measured_regressions = max(0.0, attention_delta) + max(0.0, qkv_delta)

  attention_markers = _has_markers(
      source,
      [
          "RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm",
          "RunQ8QuantizeWithBsumsKernel(",
          "clFinish(device Q8 input-handle residual/norm)",
          "clEnqueueReadBuffer(device Q8 input-handle residual)",
          "clEnqueueReadBuffer(device Q8 input-handle normalized)",
      ],
  )
  qkv_markers = _has_markers(
      source,
      [
          "RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder",
          "RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder",
          "RunQ8QuantizeWithBsumsKernel(",
          "clFinish(shared-Q8 Q6 preconv conv)",
          "clFinish(shared-Q8 Q4 preconv state copy)",
          "run.timing.qkv_min_us =",
          "shared.timing.q8_quantize_min_us + shared.timing.qkv_matvec.min_us",
      ],
  )

  closed_attention = _closed_route_names(
      rejected,
      "linear_final_device_q8_handoff",
      "simple linear-final",
      "final_output handoff",
  )
  closed_shared_q8 = _closed_route_names(
      rejected,
      "shared-q8 preconv",
      "shared_q8_preconv",
      "linear_preconv_shared_q8",
  )
  closed_combined = _closed_route_names(
      rejected,
      "attention_linear_event_lifetime_combined_alias",
      "combined event-lifetime",
  )

  checks = [
      {
          "name": "seq84_decode_gate_valid_negative",
          "pass": seq84.get("required_checks_passed") is True
          and seq84.get("disposition")
          == "rejected_combined_event_lifetime_attention_front_growth",
      },
      {
          "name": "attention_delta_matches_simple_device_q8_handoff_class",
          "pass": attention_delta > frontier_state["floor_gap_ms_per_token"]
          and abs(attention_delta - simple_attention_delta) <= 0.25,
          "detail": {
              "seq84_attention_delta_ms_per_token": attention_delta,
              "seq51_attention_delta_ms_per_token": simple_attention_delta,
          },
      },
      {
          "name": "qkv_delta_matches_closed_shared_q8_preconv_class",
          "pass": qkv_delta > frontier_state["floor_gap_ms_per_token"]
          and abs(qkv_delta - shared_q8_qkv_delta) <= 0.10,
          "detail": {
              "seq84_qkv_delta_ms_per_token": qkv_delta,
              "seq77_qkv_delta_ms_per_token": shared_q8_qkv_delta,
          },
      },
      {
          "name": "measured_regressions_exceed_measured_savings",
          "pass": measured_regressions > measured_savings,
          "detail": {
              "measured_savings_ms_per_token": measured_savings,
              "measured_regressions_ms_per_token": measured_regressions,
          },
      },
      {
          "name": "attention_source_still_has_device_q8_readback_chain",
          "pass": attention_markers["pass"],
          "detail": attention_markers,
      },
      {
          "name": "qkv_source_still_has_shared_q8_serial_chain",
          "pass": qkv_markers["pass"],
          "detail": qkv_markers,
      },
      {
          "name": "closed_route_board_contains_required_classes",
          "pass": bool(closed_attention) and bool(closed_shared_q8) and bool(closed_combined),
          "detail": {
              "closed_attention_routes": closed_attention,
              "closed_shared_q8_routes": closed_shared_q8,
              "closed_combined_routes": closed_combined,
          },
      },
  ]
  required = all(check["pass"] for check in checks)
  disposition = (
      "attention_linear_regression_roots_bound_to_closed_classes"
      if required
      else "attention_linear_regression_root_evidence_incomplete"
  )
  next_action = (
      "Do not launch another attention/linear decode candidate until a source "
      "or component gate proves one of the two root chains changed: the "
      "attention-front device-Q8 path must avoid the residual/norm readback "
      "chain, or shared-Q8 preconv must remove the qkv/conv serialization. "
      "Otherwise record a route switch to a different dominant bucket."
      if required
      else "Fix the missing root-cause evidence before selecting another target probe."
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "next_action": next_action,
      "inputs": {
          "seq84": _rel(args.seq84),
          "seq77": _rel(args.seq77),
          "seq51": _rel(args.seq51),
          "frontier": _rel(args.frontier),
          "rejected": _rel(args.rejected),
          "decode_source": _rel(args.decode_source),
          "engine_source": _rel(args.engine_source),
          "engine_header": _rel(args.engine_header),
      },
      "frontier": frontier_state,
      "derived": {
          "attention_front_delta_ms_per_token": attention_delta,
          "simple_device_q8_attention_delta_ms_per_token": simple_attention_delta,
          "attention_delta_difference_ms_per_token": attention_delta - simple_attention_delta,
          "linear_preconv_qkv_conv_delta_ms_per_token": qkv_delta,
          "shared_q8_preconv_qkv_conv_delta_ms_per_token": shared_q8_qkv_delta,
          "qkv_delta_difference_ms_per_token": qkv_delta - shared_q8_qkv_delta,
          "linear_delta_final_read_saved_ms_per_token": final_read_saved,
          "linear_preconv_input_q8_saved_ms_per_token": preconv_input_q8_saved,
          "linear_preconv_alpha_beta_saved_ms_per_token": preconv_alpha_beta_saved,
          "measured_savings_ms_per_token": measured_savings,
          "measured_regressions_ms_per_token": measured_regressions,
          "net_regression_after_local_savings_ms_per_token": measured_regressions - measured_savings,
      },
      "closed_routes": {
          "attention_device_q8": closed_attention,
          "linear_preconv_shared_q8": closed_shared_q8,
          "combined_event_lifetime": closed_combined,
      },
      "checks": checks,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  failed = [check["name"] for check in payload["checks"] if check["pass"] is not True]
  derived = payload["derived"]
  lines = [
      "# Attention/Linear Regression Root Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- attention-front delta: `{derived['attention_front_delta_ms_per_token']:.3f}` ms/token",
      f"- simple device-Q8 class delta: `{derived['simple_device_q8_attention_delta_ms_per_token']:.3f}` ms/token",
      f"- qkv/conv delta: `{derived['linear_preconv_qkv_conv_delta_ms_per_token']:.3f}` ms/token",
      f"- shared-Q8 preconv class delta: `{derived['shared_q8_preconv_qkv_conv_delta_ms_per_token']:.3f}` ms/token",
      f"- local savings: `{derived['measured_savings_ms_per_token']:.3f}` ms/token",
      f"- measured regressions: `{derived['measured_regressions_ms_per_token']:.3f}` ms/token",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is route-control evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seq84", type=Path, default=DEFAULT_SEQ84)
  parser.add_argument("--seq77", type=Path, default=DEFAULT_SEQ77)
  parser.add_argument("--seq51", type=Path, default=DEFAULT_SEQ51)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_HEADER)
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
