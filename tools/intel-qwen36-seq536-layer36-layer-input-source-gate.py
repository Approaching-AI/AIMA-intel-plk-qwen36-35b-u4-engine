#!/usr/bin/env python3
"""Classify the layer-36 layer-input source gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq536-layer36-layer-input-source-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ535 = (
    ROOT
    / "output/seq535-layer36-ffn-input-source-gate-20260709Tseq535Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq535-layer36-ffn-input-source-math-20260709Tseq535Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq535-layer36-ffn-input-source-code-20260709Tseq535Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/seq536-layer36-layer-input-source-gate-20260709Tseq536Z"
)

CURRENT_ROUTE = "router_prompt_distribution_layer36_layer_input_source_gate"
NEXT_ROUTE = "router_prompt_distribution_layer35_ffn_math_source_gate"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SOURCE_LAYER = 35
CONSUMER_LAYER = 36
COSINE_THRESHOLD = 0.9999
MAX_GPU_CPU_MATH_ABS = 2.0e-5
BOUNDARY_MATCH_EPS = 1.0e-6


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _load_smoke(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  payload = _load_json(path)
  if isinstance(payload, dict) and isinstance(payload.get("smoke"), dict):
    return payload, payload["smoke"]
  if isinstance(payload, dict):
    return payload, payload
  raise TypeError(f"{path} does not contain a JSON object")


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _failed_steps(smoke: dict[str, Any]) -> list[dict[str, Any]]:
  steps = _dist(smoke).get("steps")
  steps = steps if isinstance(steps, list) else []
  return [
      step for step in steps
      if isinstance(step, dict)
      and isinstance(step.get("token_index"), int)
      and (_num(step.get("kld")) > KLD_THRESHOLD
           or step.get("top1_matches") is False)
  ]


def _table_step(smoke: dict[str, Any], table: str, token_index: int) -> dict[str, Any]:
  rows = smoke.get(table)
  rows = rows if isinstance(rows, list) else []
  for row in rows:
    if isinstance(row, dict) and row.get("token_index") == token_index:
      return row
  return {}


def _layer(step: dict[str, Any], layer: int) -> dict[str, Any]:
  rows = step.get("layers")
  rows = rows if isinstance(rows, list) else []
  for row in rows:
    if isinstance(row, dict) and row.get("layer") == layer:
      return row
  return {}


def _row(path: Path, label: str) -> dict[str, Any]:
  payload, smoke = _load_smoke(path)
  dist = _dist(smoke)
  entries: list[dict[str, Any]] = []
  for step in _failed_steps(smoke):
    token_index = int(step["token_index"])
    boundary = _table_step(smoke, "layer_boundary_diff_by_step", token_index)
    residual = _table_step(smoke, "residual_source_diff_by_step", token_index)
    components = _table_step(
        smoke, "ffn_component_source_diff_by_step", token_index)
    source_boundary = _layer(boundary, SOURCE_LAYER)
    consumer_boundary = _layer(boundary, CONSUMER_LAYER)
    source_residual = _layer(residual, SOURCE_LAYER)
    source_components = _layer(components, SOURCE_LAYER)
    boundary_link_delta = max(
        abs(_num(source_boundary.get("output_cosine"), 1.0) -
            _num(consumer_boundary.get("input_cosine"), 1.0)),
        abs(_num(source_boundary.get("output_max_abs_diff")) -
            _num(consumer_boundary.get("input_max_abs_diff"))),
        abs(_num(source_boundary.get("output_rmse")) -
            _num(consumer_boundary.get("input_rmse"))),
    )
    entries.append({
        "token_index": token_index,
        "kld": step.get("kld"),
        "top1_matches": step.get("top1_matches"),
        "layer35_output_cosine": source_boundary.get("output_cosine"),
        "layer35_output_max_abs_diff":
            source_boundary.get("output_max_abs_diff"),
        "layer36_input_cosine": consumer_boundary.get("input_cosine"),
        "layer36_input_max_abs_diff":
            consumer_boundary.get("input_max_abs_diff"),
        "boundary_link_delta": boundary_link_delta,
        "layer35_layer_input_cosine":
            source_residual.get("layer_input_cosine"),
        "layer35_attention_output_cosine":
            source_residual.get("attention_output_cosine"),
        "layer35_ffn_input_cosine": source_residual.get("ffn_input_cosine"),
        "layer35_ffn_delta_cosine": source_residual.get("ffn_delta_cosine"),
        "layer35_cpu_ffn_from_gpu_input_cosine":
            source_residual.get("cpu_ffn_from_gpu_input_cosine"),
        "layer35_gpu_output_vs_cpu_ffn_cosine":
            source_residual.get("gpu_output_vs_cpu_ffn_cosine"),
        "layer35_gpu_output_vs_cpu_ffn_max_abs_diff":
            source_residual.get("gpu_output_vs_cpu_ffn_max_abs_diff"),
        "layer35_gpu_ffn_delta_vs_cpu_max_abs_diff":
            source_residual.get("gpu_ffn_delta_vs_cpu_max_abs_diff"),
        "layer35_gpu_ffn_norm_vs_cpu_max_abs_diff":
            source_residual.get("gpu_ffn_norm_vs_cpu_max_abs_diff"),
        "router_ids_match": source_components.get("router_ids_match"),
        "router_weight_max_abs_diff":
            source_components.get("router_weight_max_abs_diff"),
        "selected_gate_up_cosine": source_components.get(
            "selected_gate_up_cosine"),
        "selected_swiglu_cosine": source_components.get(
            "selected_swiglu_cosine"),
        "selected_down_cosine": source_components.get("selected_down_cosine"),
        "shared_down_cosine": source_components.get("shared_down_cosine"),
    })

  def min_metric(name: str, default: float = 1.0) -> float:
    return min((_num(row.get(name), default) for row in entries), default=default)

  def max_metric(name: str) -> float:
    return max((_num(row.get(name)) for row in entries), default=0.0)

  row = {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "position_count": dist.get("position_count"),
      "full_attention_state_diff_enabled": smoke.get(
          "full_attention_state_diff_enabled"),
      "diagnostic_layer_range": payload.get("diagnostic_layer_range"),
      "failed_step_count": len(entries),
      "failed_token_indices": [row["token_index"] for row in entries],
      "failed_steps": entries,
      "min_layer35_output_cosine": min_metric("layer35_output_cosine"),
      "max_layer35_output_abs_diff": max_metric("layer35_output_max_abs_diff"),
      "min_layer36_input_cosine": min_metric("layer36_input_cosine"),
      "max_layer36_input_abs_diff": max_metric("layer36_input_max_abs_diff"),
      "max_boundary_link_delta": max_metric("boundary_link_delta"),
      "min_layer35_ffn_input_cosine": min_metric("layer35_ffn_input_cosine"),
      "min_layer35_gpu_output_vs_cpu_ffn_cosine":
          min_metric("layer35_gpu_output_vs_cpu_ffn_cosine"),
      "max_layer35_gpu_output_vs_cpu_ffn_abs_diff":
          max_metric("layer35_gpu_output_vs_cpu_ffn_max_abs_diff"),
      "max_layer35_gpu_ffn_delta_vs_cpu_abs_diff":
          max_metric("layer35_gpu_ffn_delta_vs_cpu_max_abs_diff"),
      "max_layer35_gpu_ffn_norm_vs_cpu_abs_diff":
          max_metric("layer35_gpu_ffn_norm_vs_cpu_max_abs_diff"),
      "router_id_mismatch_count":
          sum(1 for entry in entries if entry.get("router_ids_match") is False),
  }
  row["target_row_ran"] = (
      row["target_returncode"] == 2
      and row["distribution_required_checks_passed"] is False
      and row["kld_pass"] is False
      and row["top1_pass"] is True
      and _num(row["top1_rate"]) >= TOP1_THRESHOLD
      and _num(row["max_kld"]) > KLD_THRESHOLD
      and row["full_attention_state_diff_enabled"] is True
      and row["diagnostic_layer_range"] == "35:37")
  return row


def _max_metric(rows: list[dict[str, Any]], name: str) -> float:
  return max((_num(row.get(name)) for row in rows), default=0.0)


def _min_metric(rows: list[dict[str, Any]], name: str) -> float:
  return min((_num(row.get(name), 1.0) for row in rows), default=1.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq535 = _load_json(args.seq535)
  math_row = _row(args.math, "seq536_layer36_layer_input_source_math")
  code_row = _row(args.code, "seq536_layer36_layer_input_source_code")
  rows = [math_row, code_row]
  max_boundary_link_delta = _max_metric(rows, "max_boundary_link_delta")
  min_gpu_cpu_cosine = _min_metric(
      rows, "min_layer35_gpu_output_vs_cpu_ffn_cosine")
  max_gpu_cpu_abs = _max_metric(
      rows, "max_layer35_gpu_output_vs_cpu_ffn_abs_diff")
  ffn_math_mismatch = (
      min_gpu_cpu_cosine < COSINE_THRESHOLD
      or max_gpu_cpu_abs > MAX_GPU_CPU_MATH_ABS)

  checks = [
      {
          "name": "seq535_selected_layer36_layer_input_source_route",
          "pass": (
              seq535.get("required_checks_passed") is True
              and seq535.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 535,
                  "reject_layer36_attention_output_dominant_source_select_layer36_layer_input_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_layer36_layer_input_source_gate",
                  535)),
      },
      {
          "name": "layer36_layer_input_source_rows_target_ran",
          "pass": math_row.get("target_row_ran") is True
          and code_row.get("target_row_ran") is True,
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "layer36_input_equals_layer35_output_on_failed_steps",
          "pass": max_boundary_link_delta <= BOUNDARY_MATCH_EPS,
          "detail": {
              "source_layer": SOURCE_LAYER,
              "consumer_layer": CONSUMER_LAYER,
              "max_boundary_link_delta": max_boundary_link_delta,
              "threshold": BOUNDARY_MATCH_EPS,
          },
      },
      {
          "name": "layer35_live_ffn_math_mismatch_detected",
          "pass": ffn_math_mismatch,
          "detail": {
              "min_gpu_output_vs_cpu_ffn_cosine": min_gpu_cpu_cosine,
              "max_gpu_output_vs_cpu_ffn_abs_diff": max_gpu_cpu_abs,
              "cosine_threshold": COSINE_THRESHOLD,
              "abs_threshold": MAX_GPU_CPU_MATH_ABS,
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq535": _rel(args.seq535),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "layer36_layer_input_source_gate_allowed": required,
      "layer35_ffn_math_source_gate_allowed": required,
      "rows": {"math": math_row, "code": code_row},
      "combined": {
          "failed_step_count":
              math_row["failed_step_count"] + code_row["failed_step_count"],
          "max_boundary_link_delta": max_boundary_link_delta,
          "min_gpu_output_vs_cpu_ffn_cosine": min_gpu_cpu_cosine,
          "max_gpu_output_vs_cpu_ffn_abs_diff": max_gpu_cpu_abs,
          "min_layer35_ffn_input_cosine":
              _min_metric(rows, "min_layer35_ffn_input_cosine"),
          "router_id_mismatch_count":
              math_row["router_id_mismatch_count"] +
              code_row["router_id_mismatch_count"],
      },
      "disposition": (
          "select_layer35_ffn_math_source_due_live_ffn_math_mismatch"
          if required else
          "block_layer36_layer_input_source_inconsistent_evidence"),
      "rejected_route": None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Layer36 input equals layer35 output on failed steps, but layer35 "
          "GPU FFN output on live GPU FFN input does not match CPU within the "
          "prior math-closure tolerance. The next unit is layer35 FFN math "
          "source attribution before FFN-input attribution, product "
          "correction, speed promotion, or long-context rows."
          if required else
          "Layer36 layer-input source evidence is inconsistent; do not switch "
          "routes or run speed/promotion/long-context rows."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  rows = metrics["rows"]
  lines = [
      "# Seq536 Layer36 Layer-Input Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- max_boundary_link_delta: `{metrics['combined']['max_boundary_link_delta']}`",
      f"- max_gpu_output_vs_cpu_ffn_abs_diff: `{metrics['combined']['max_gpu_output_vs_cpu_ffn_abs_diff']}`",
      "",
      "## Evidence",
      "",
      f"- math KLD/top1/failed steps: `{rows['math']['max_kld']}` / `{rows['math']['top1_rate']}` / `{rows['math']['failed_step_count']}`",
      f"- code KLD/top1/failed steps: `{rows['code']['max_kld']}` / `{rows['code']['top1_rate']}` / `{rows['code']['failed_step_count']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is correctness-route evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq535", type=Path, default=DEFAULT_SEQ535)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
