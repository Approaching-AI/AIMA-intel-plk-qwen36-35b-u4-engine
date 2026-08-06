#!/usr/bin/env python3
"""Gate the router all-linear qkv-delta source route.

This is route-control/source evidence only. It consumes the seq263/264/288/291
passing diagnostics, the seq289/290 product-approximation failures, and the
current decode source before any further token row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-router-qkv-delta-source-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ263 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-linear-qkvcol-delta-blockq16-top1024-20260708Tseq263Z"
)
DEFAULT_SEQ264 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-linear-qkvcol-delta-blockq16-top512-20260708Tseq264Z"
)
DEFAULT_SEQ288 = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-all-linear-qkvcol-delta-blockq16-top1024-20260708Tseq288Z"
)
DEFAULT_SEQ289 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-live-round-all-linear-q10-20260708Tseq289Z"
)
DEFAULT_SEQ290 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-all-linear-qkvcol-selected-affine-top1024-20260708Tseq290Z"
)
DEFAULT_SEQ291 = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-all-linear-qkvcol-delta-blockq16-top512-20260708Tseq291Z"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/router-qkv-delta-source-gate-20260708Tseq292Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
HIDDEN_SIZE = 8192
DECODE_TOKENS = 8
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _dict(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _nested(obj: Any, *keys: str) -> Any:
  cur = obj
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


def _summary_metric(summary: str, label: str) -> float:
  match = re.search(rf"^- {re.escape(label)}: `([^`]+)`", summary, re.M)
  if match is None:
    return 0.0
  try:
    return float(match.group(1))
  except ValueError:
    return 0.0


def _summary_bool(summary: str, label: str) -> bool:
  match = re.search(rf"^- {re.escape(label)}: `([^`]+)`", summary, re.M)
  return bool(match and match.group(1).lower() == "true")


def _artifact_summary(path: Path) -> dict[str, Any]:
  summary_path = path / "SUMMARY.md"
  result_path = path / "result.json"
  smoke_path = path / "smoke.json"
  summary = _read(summary_path)
  result = _load_json(result_path) if result_path.exists() else {}
  smoke = _load_json(smoke_path) if smoke_path.exists() else {}
  smoke_inner = _dict(_dict(result).get("smoke"))
  ladder = _dict(smoke_inner.get("distribution_ladder"))
  return {
      "artifact": _rel(path),
      "summary_sha256": _sha256(summary_path),
      "result_sha256": _sha256(result_path) if result_path.exists() else None,
      "smoke_sha256": _sha256(smoke_path) if smoke_path.exists() else None,
      "required_checks_passed": _summary_bool(summary, "required checks passed"),
      "distribution_ladder_passed": _summary_bool(
          summary, "distribution ladder passed"),
      "max_kld": _summary_metric(summary, "distribution max KLD"),
      "top1_rate": _summary_metric(summary, "distribution top-1 rate"),
      "min_logit_cosine": _summary_metric(
          summary, "distribution min logit cosine"),
      "tps": _summary_metric(summary, "GPU hybrid decode tok/s"),
      "applied_values": int(_num(smoke_inner.get(
          "cpu_shadow_layer_input_delta_values"))),
      "fired_layers": int(_num(smoke_inner.get(
          "cpu_shadow_layer_input_delta_layers"))),
      "live_round_layers": int(_num(smoke_inner.get(
          "live_layer_input_round_layers"))),
      "ladder_top1_count": int(_num(ladder.get("top1_match_count"))),
      "smoke_created_at": smoke.get("created_at"),
  }


def _pass(row: dict[str, Any]) -> bool:
  return (
      row["required_checks_passed"] is True
      and row["distribution_ladder_passed"] is True
      and row["max_kld"] < KLD_THRESHOLD
      and row["top1_rate"] >= TOP1_THRESHOLD
  )


def _fail_distribution(row: dict[str, Any]) -> bool:
  return row["max_kld"] >= KLD_THRESHOLD


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  for row in routes.get("candidate_history", []):
    if (
        isinstance(row, dict)
        and row.get("seq") == seq
        and row.get("disposition") == disposition
    ):
      return True
  return False


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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  out: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      out.add(row["route"])
  return out


def _present(text: str, label: str, pattern: str) -> dict[str, Any]:
  match = re.search(pattern, text, re.S)
  line = None if match is None else text.count("\n", 0, match.start()) + 1
  return {"label": label, "present": match is not None, "line": line}


def _absent(text: str, label: str, pattern: str) -> dict[str, Any]:
  match = re.search(pattern, text, re.S)
  line = None if match is None else text.count("\n", 0, match.start()) + 1
  return {"label": label, "absent": match is None, "line": line}


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = _dict(frontier.get("goal_anchor"))
  no_progress = _dict(frontier.get("no_progress"))
  noise = _dict(no_progress.get("noise"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "noise_rel": _num(noise.get("rel")),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  decode_source = _read(args.decode_source)
  rejected_names = _rejected_names(rejected)

  artifacts = {
      "seq263_math_top1024": _artifact_summary(args.seq263),
      "seq264_math_top512": _artifact_summary(args.seq264),
      "seq288_code_top1024": _artifact_summary(args.seq288),
      "seq289_live_round_q10": _artifact_summary(args.seq289),
      "seq290_selected_affine_top1024": _artifact_summary(args.seq290),
      "seq291_code_top512": _artifact_summary(args.seq291),
  }
  top512_values = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * 512
  top1024_values = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * 1024
  correction_shape = {
      "layers": ALL_LINEAR_LAYERS,
      "layer_count": len(ALL_LINEAR_LAYERS),
      "hidden_size": HIDDEN_SIZE,
      "decode_tokens": DECODE_TOKENS,
      "top512_values_expected": top512_values,
      "top512_fraction_per_layer": 512 / HIDDEN_SIZE,
      "top1024_values_expected": top1024_values,
      "top1024_fraction_per_layer": 1024 / HIDDEN_SIZE,
  }

  source_present = [
      _present(
          decode_source,
          "linear_qkv_col_abs_selector_exists",
          r'if \(text == "linear_qkv_col_abs"\) return 4;'),
      _present(
          decode_source,
          "qkv_column_abs_weight_decoder_exists",
          r"DecodeLinearQkvColumnAbsWeights\("),
      _present(
          decode_source,
          "topk_shadow_delta_apply_exists",
          r"DecodeApplyTopKShadowDelta\("),
      _present(
          decode_source,
          "shadow_delta_requires_cpu_shadow_state",
          r"--cpu-shadow-layer-input-delta-layers/.*require --cpu-shadow-state-each-token"),
      _present(
          decode_source,
          "live_layer_rounding_diagnostic_exists",
          r"g_decode_live_layer_input_round_layers"),
  ]
  source_absent = [
      _absent(
          decode_source,
          "product_qkv_delta_source_flag_absent",
          r"IQ36_ROUTER_QKV_DELTA|router_qkv_delta_source|all_linear_qkv_delta_source"),
  ]

  checks = [
      {
          "name": "seq264_math_top512_passes",
          "pass": _pass(artifacts["seq264_math_top512"]),
          "detail": artifacts["seq264_math_top512"],
      },
      {
          "name": "seq291_code_top512_passes",
          "pass": _pass(artifacts["seq291_code_top512"])
          and artifacts["seq291_code_top512"]["applied_values"] == top512_values,
          "detail": artifacts["seq291_code_top512"],
      },
      {
          "name": "top1024_margin_passes_math_and_code",
          "pass": (
              _pass(artifacts["seq263_math_top1024"])
              and _pass(artifacts["seq288_code_top1024"])
              and artifacts["seq288_code_top1024"]["applied_values"]
              == top1024_values
          ),
          "detail": {
              "math": artifacts["seq263_math_top1024"],
              "code": artifacts["seq288_code_top1024"],
          },
      },
      {
          "name": "product_approximations_failed",
          "pass": (
              _fail_distribution(artifacts["seq289_live_round_q10"])
              and _fail_distribution(artifacts["seq290_selected_affine_top1024"])
          ),
          "detail": {
              "live_round": artifacts["seq289_live_round_q10"],
              "selected_affine": artifacts["seq290_selected_affine_top1024"],
          },
      },
      {
          "name": "routes_record_current_diagnostic_and_approximation_rows",
          "pass": (
              _has_candidate(
                  routes, 291,
                  "set_top512_as_math_code_lower_passing_qkv_delta_bound")
              and _has_candidate(
                  routes, 290,
                  "reject_rounding_and_static_affine_as_all_linear_qkv_delta_repairs")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_product_speed_gate",
                  288)
          ),
      },
      {
          "name": "closed_approximation_classes_are_recorded",
          "pass": {
              "router_math_static_or_lagged_qkv_delta_predictors",
              "router_math_live_round_or_selected_affine_qkv_delta_approximation",
              "router_math_split_full_attention_projection_arithmetic_residual_fix",
          }.issubset(rejected_names),
      },
      {
          "name": "source_has_only_shadow_bound_delta_diagnostic",
          "pass": all(row["present"] for row in source_present)
          and all(row["absent"] for row in source_absent),
          "detail": {
              "present": source_present,
              "absent": source_absent,
          },
      },
  ]

  required_checks_passed = all(check.get("pass") is True for check in checks)
  selected_next_route = "router_prompt_all_linear_qkv_delta_component_source_gate"
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required_checks_passed,
      "disposition": (
          "select_all_linear_qkv_delta_component_source_gate"
          if required_checks_passed else "blocked_before_source_gate"),
      "selected_next_route": selected_next_route,
      "frontier": _frontier_state(frontier),
      "correction_shape": correction_shape,
      "artifacts": artifacts,
      "source": {
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "present_checks": source_present,
          "absent_checks": source_absent,
      },
      "checks": checks,
      "next_gate_contract": {
          "route": selected_next_route,
          "must_do": [
              "add default-off source/component wiring before any token row",
              "compute or avoid the current-token all-linear qkv/residual drift",
              "target at least top512-equivalent correction strength across the 30 linear layers",
              "run router_math_reason_001 and router_code_reason_002 distribution before promotion expansion",
          ],
          "must_not_do": [
              "use CPU shadow state, host sync/readback, or oracle values as product evidence",
              "rerun live rounding, static selected-affine, seed, lagged, layer36-only, or group-entry-only repairs",
              "expand to long-context or cold/prefix promotion rows before router distribution passes",
          ],
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq263", type=Path, default=DEFAULT_SEQ263)
  parser.add_argument("--seq264", type=Path, default=DEFAULT_SEQ264)
  parser.add_argument("--seq288", type=Path, default=DEFAULT_SEQ288)
  parser.add_argument("--seq289", type=Path, default=DEFAULT_SEQ289)
  parser.add_argument("--seq290", type=Path, default=DEFAULT_SEQ290)
  parser.add_argument("--seq291", type=Path, default=DEFAULT_SEQ291)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()

  metrics = compute(args)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  metrics_path = args.out_dir / "metrics.json"
  metrics_path.write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  summary = [
      "# Router QKV Delta Source Gate",
      "",
      f"- required checks passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected next route: `{metrics['selected_next_route']}`",
      f"- top512 values: `{metrics['correction_shape']['top512_values_expected']}`",
      f"- top512 fraction per layer: `{metrics['correction_shape']['top512_fraction_per_layer']}`",
      "",
      "## Checks",
  ]
  for check in metrics["checks"]:
    summary.append(f"- {check['name']}: `{str(check.get('pass')).lower()}`")
  (args.out_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n",
                                           encoding="utf-8")
  print(f"wrote {_rel(metrics_path)}")
  print(f"required_checks_passed={metrics['required_checks_passed']}")
  if not metrics["required_checks_passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
