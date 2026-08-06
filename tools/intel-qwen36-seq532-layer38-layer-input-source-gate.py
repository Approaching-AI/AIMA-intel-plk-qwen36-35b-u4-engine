#!/usr/bin/env python3
"""Classify the layer-38 layer-input source gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq532-layer38-layer-input-source-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_MATH = (
    ROOT
    / "output/seq532-layer38-layer-input-source-math-20260709Tseq532Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq532-layer38-layer-input-source-code-20260709Tseq532Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/seq532-layer38-layer-input-source-gate-20260709Tseq532Z"
)

CURRENT_ROUTE = "router_prompt_distribution_layer38_layer_input_source_gate"
NEXT_ROUTE = "router_prompt_distribution_layer37_ffn_input_source_gate"
REJECTED_ROUTE = "router_prompt_distribution_layer37_ffn_math_source"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SOURCE_LAYER = 37
CONSUMER_LAYER = 38
COSINE_THRESHOLD = 0.9999
CPU_MATCH_ABS_THRESHOLD = 2.0e-6
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


def _load_smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  if isinstance(payload, dict) and isinstance(payload.get("smoke"), dict):
    return payload["smoke"]
  if isinstance(payload, dict):
    return payload
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
  smoke = _load_smoke(path)
  dist = _dist(smoke)
  failed = _failed_steps(smoke)
  entries: list[dict[str, Any]] = []
  for step in failed:
    token_index = int(step["token_index"])
    boundary = _table_step(smoke, "layer_boundary_diff_by_step", token_index)
    residual = _table_step(smoke, "residual_source_diff_by_step", token_index)
    components = _table_step(
        smoke, "ffn_component_source_diff_by_step", token_index)
    layer37_boundary = _layer(boundary, SOURCE_LAYER)
    layer38_boundary = _layer(boundary, CONSUMER_LAYER)
    layer37_residual = _layer(residual, SOURCE_LAYER)
    layer37_components = _layer(components, SOURCE_LAYER)
    boundary_link_delta = max(
        abs(_num(layer37_boundary.get("output_cosine"), 1.0) -
            _num(layer38_boundary.get("input_cosine"), 1.0)),
        abs(_num(layer37_boundary.get("output_max_abs_diff")) -
            _num(layer38_boundary.get("input_max_abs_diff"))),
        abs(_num(layer37_boundary.get("output_rmse")) -
            _num(layer38_boundary.get("input_rmse"))),
    )
    cpu_from_gpu_match_delta = max(
        abs(_num(layer37_boundary.get("output_cosine"), 1.0) -
            _num(layer37_residual.get("cpu_ffn_from_gpu_input_cosine"), 1.0)),
        abs(_num(layer37_boundary.get("output_max_abs_diff")) -
            _num(layer37_residual.get("cpu_ffn_from_gpu_input_max_abs_diff"))),
        abs(_num(layer37_boundary.get("output_rmse")) -
            _num(layer37_residual.get("cpu_ffn_from_gpu_input_rmse"))),
    )
    entries.append({
        "token_index": token_index,
        "kld": step.get("kld"),
        "top1_matches": step.get("top1_matches"),
        "layer37_output_cosine": layer37_boundary.get("output_cosine"),
        "layer37_output_max_abs_diff":
            layer37_boundary.get("output_max_abs_diff"),
        "layer38_input_cosine": layer38_boundary.get("input_cosine"),
        "layer38_input_max_abs_diff":
            layer38_boundary.get("input_max_abs_diff"),
        "boundary_link_delta": boundary_link_delta,
        "layer37_layer_input_cosine":
            layer37_residual.get("layer_input_cosine"),
        "layer37_attention_output_cosine":
            layer37_residual.get("attention_output_cosine"),
        "layer37_ffn_input_cosine": layer37_residual.get("ffn_input_cosine"),
        "layer37_ffn_delta_cosine": layer37_residual.get("ffn_delta_cosine"),
        "layer37_cpu_ffn_from_gpu_input_cosine":
            layer37_residual.get("cpu_ffn_from_gpu_input_cosine"),
        "layer37_cpu_ffn_from_gpu_input_max_abs_diff":
            layer37_residual.get("cpu_ffn_from_gpu_input_max_abs_diff"),
        "layer37_gpu_output_vs_cpu_ffn_cosine":
            layer37_residual.get("gpu_output_vs_cpu_ffn_cosine"),
        "layer37_gpu_output_vs_cpu_ffn_max_abs_diff":
            layer37_residual.get("gpu_output_vs_cpu_ffn_max_abs_diff"),
        "layer37_gpu_ffn_delta_vs_cpu_max_abs_diff":
            layer37_residual.get("gpu_ffn_delta_vs_cpu_max_abs_diff"),
        "layer37_gpu_ffn_norm_vs_cpu_max_abs_diff":
            layer37_residual.get("gpu_ffn_norm_vs_cpu_max_abs_diff"),
        "cpu_from_gpu_matches_layer37_output_delta":
            cpu_from_gpu_match_delta,
        "router_ids_match": layer37_components.get("router_ids_match"),
        "router_weight_max_abs_diff":
            layer37_components.get("router_weight_max_abs_diff"),
        "selected_down_cosine": layer37_components.get("selected_down_cosine"),
        "shared_down_cosine": layer37_components.get("shared_down_cosine"),
    })

  def min_metric(name: str, default: float = 1.0) -> float:
    return min((_num(row.get(name), default) for row in entries), default=default)

  def max_metric(name: str) -> float:
    return max((_num(row.get(name)) for row in entries), default=0.0)

  position_count = _num(dist.get("position_count"))
  top1_rate = _num(dist.get("top1_rate"))
  row = {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_match_count": dist.get("top1_match_count"),
      "top1_rate": dist.get("top1_rate"),
      "position_count": dist.get("position_count"),
      "diagnostic_layer_start": smoke.get("diagnostic_layer_start"),
      "diagnostic_layer_end": smoke.get("diagnostic_layer_end"),
      "diagnostic_token_limit": smoke.get("diagnostic_token_limit"),
      "failed_step_count": len(entries),
      "failed_token_indices": [row["token_index"] for row in entries],
      "failed_steps": entries,
      "min_layer37_output_cosine":
          min_metric("layer37_output_cosine"),
      "max_layer37_output_abs_diff":
          max_metric("layer37_output_max_abs_diff"),
      "min_layer38_input_cosine": min_metric("layer38_input_cosine"),
      "max_layer38_input_abs_diff": max_metric("layer38_input_max_abs_diff"),
      "max_boundary_link_delta": max_metric("boundary_link_delta"),
      "min_layer37_ffn_input_cosine":
          min_metric("layer37_ffn_input_cosine"),
      "min_layer37_ffn_delta_cosine":
          min_metric("layer37_ffn_delta_cosine"),
      "max_cpu_from_gpu_matches_layer37_output_delta":
          max_metric("cpu_from_gpu_matches_layer37_output_delta"),
      "min_layer37_gpu_output_vs_cpu_ffn_cosine":
          min_metric("layer37_gpu_output_vs_cpu_ffn_cosine"),
      "max_layer37_gpu_output_vs_cpu_ffn_abs_diff":
          max_metric("layer37_gpu_output_vs_cpu_ffn_max_abs_diff"),
      "max_layer37_gpu_ffn_delta_vs_cpu_abs_diff":
          max_metric("layer37_gpu_ffn_delta_vs_cpu_max_abs_diff"),
      "max_layer37_gpu_ffn_norm_vs_cpu_abs_diff":
          max_metric("layer37_gpu_ffn_norm_vs_cpu_max_abs_diff"),
      "router_id_mismatch_count":
          sum(1 for entry in entries if entry.get("router_ids_match") is False),
  }
  row["target_row_ran"] = (
      row["required_checks_passed"] is False
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and position_count >= 8
      and top1_rate >= TOP1_THRESHOLD)
  row["diagnostic_window_ok"] = (
      row.get("diagnostic_layer_start") == SOURCE_LAYER
      and row.get("diagnostic_layer_end") == CONSUMER_LAYER + 1
      and _num(row.get("diagnostic_token_limit")) >= 8)
  row["boundary_link_ok"] = (
      bool(entries)
      and row["max_boundary_link_delta"] <= BOUNDARY_MATCH_EPS)
  row["gpu_ffn_math_matches_cpu"] = (
      bool(entries)
      and row["min_layer37_gpu_output_vs_cpu_ffn_cosine"] >= COSINE_THRESHOLD
      and row["max_layer37_gpu_output_vs_cpu_ffn_abs_diff"]
      <= CPU_MATCH_ABS_THRESHOLD
      and row["max_layer37_gpu_ffn_delta_vs_cpu_abs_diff"]
      <= CPU_MATCH_ABS_THRESHOLD)
  row["cpu_from_gpu_input_explains_layer37_output"] = (
      bool(entries)
      and row["max_cpu_from_gpu_matches_layer37_output_delta"]
      <= BOUNDARY_MATCH_EPS)
  row["ffn_input_drift_present"] = (
      bool(entries)
      and row["min_layer37_ffn_input_cosine"] < COSINE_THRESHOLD)
  return row


def _write_summary(metrics: dict[str, Any], out_dir: Path) -> None:
  lines = [
      "# Seq532 layer38 layer-input source gate",
      "",
      f"- required_checks_passed: {metrics['required_checks_passed']}",
      f"- disposition: {metrics['disposition']}",
      f"- selected_next_route: {metrics['selected_next_route']}",
      f"- rejected_route: {metrics['rejected_route']}",
      f"- math failed steps: {metrics['math']['failed_token_indices']}",
      f"- code failed steps: {metrics['code']['failed_token_indices']}",
      (
          "- layer38 input equals layer37 output: "
          f"max metric delta {metrics['combined']['max_boundary_link_delta']}"
      ),
      (
          "- layer37 GPU FFN math vs CPU live-input max abs: "
          f"{metrics['combined']['max_layer37_gpu_output_vs_cpu_ffn_abs_diff']}"
      ),
      (
          "- layer37 FFN-input drift min cosine: "
          f"{metrics['combined']['min_layer37_ffn_input_cosine']}"
      ),
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()

  routes = _load_json(args.routes)
  math_row = _row(args.math, "seq532_layer38_layer_input_source_math")
  code_row = _row(args.code, "seq532_layer38_layer_input_source_code")
  combined = {
      "failed_step_count":
          math_row["failed_step_count"] + code_row["failed_step_count"],
      "max_boundary_link_delta":
          max(math_row["max_boundary_link_delta"],
              code_row["max_boundary_link_delta"]),
      "min_layer37_output_cosine":
          min(math_row["min_layer37_output_cosine"],
              code_row["min_layer37_output_cosine"]),
      "max_layer37_output_abs_diff":
          max(math_row["max_layer37_output_abs_diff"],
              code_row["max_layer37_output_abs_diff"]),
      "min_layer37_ffn_input_cosine":
          min(math_row["min_layer37_ffn_input_cosine"],
              code_row["min_layer37_ffn_input_cosine"]),
      "max_layer37_gpu_output_vs_cpu_ffn_abs_diff":
          max(math_row["max_layer37_gpu_output_vs_cpu_ffn_abs_diff"],
              code_row["max_layer37_gpu_output_vs_cpu_ffn_abs_diff"]),
      "max_layer37_gpu_ffn_delta_vs_cpu_abs_diff":
          max(math_row["max_layer37_gpu_ffn_delta_vs_cpu_abs_diff"],
              code_row["max_layer37_gpu_ffn_delta_vs_cpu_abs_diff"]),
      "router_id_mismatch_count":
          math_row["router_id_mismatch_count"]
          + code_row["router_id_mismatch_count"],
  }

  checks = [
      {
          "name": "seq531_selected_layer38_layer_input_source_route",
          "pass": _has_candidate(
              routes, 531,
              "reject_layer38_ffn_input_coupled_attention_math_source_select_layer38_layer_input_source")
          and _has_switch(
              routes,
              "select_router_prompt_distribution_layer38_layer_input_source_gate",
              531),
      },
      {
          "name": "layer38_layer_input_source_rows_target_ran",
          "pass": math_row["target_row_ran"] and code_row["target_row_ran"],
      },
      {
          "name": "diagnostic_window_covers_layer37_and_layer38",
          "pass": (math_row["diagnostic_window_ok"]
                   and code_row["diagnostic_window_ok"]),
      },
      {
          "name": "layer38_input_equals_layer37_output",
          "pass": math_row["boundary_link_ok"] and code_row["boundary_link_ok"],
      },
      {
          "name": "layer37_gpu_ffn_math_matches_cpu_live_input",
          "pass": (math_row["gpu_ffn_math_matches_cpu"]
                   and code_row["gpu_ffn_math_matches_cpu"]),
      },
      {
          "name": "layer37_cpu_from_gpu_input_explains_output_drift",
          "pass": (math_row["cpu_from_gpu_input_explains_layer37_output"]
                   and code_row["cpu_from_gpu_input_explains_layer37_output"]),
      },
      {
          "name": "layer37_ffn_input_drift_present",
          "pass": (math_row["ffn_input_drift_present"]
                   and code_row["ffn_input_drift_present"]),
      },
  ]
  required = all(check["pass"] for check in checks)
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "correctness_promotion_allowed": False,
      "current_route": CURRENT_ROUTE,
      "selected_next_route": NEXT_ROUTE,
      "rejected_route": REJECTED_ROUTE,
      "disposition": (
          "reject_layer37_ffn_math_source_select_layer37_ffn_input_source"
          if required else
          "block_layer38_layer_input_source_inconsistent_evidence"),
      "checks": checks,
      "math": math_row,
      "code": code_row,
      "combined": combined,
      "inputs": {
          "routes": _rel(args.routes),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
  }
  args.out_dir.mkdir(parents=True, exist_ok=True)
  (args.out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  _write_summary(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
