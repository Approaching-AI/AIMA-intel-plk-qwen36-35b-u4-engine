#!/usr/bin/env python3
"""Classify the layer-37 layer-input source gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq534-layer37-layer-input-source-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ533 = (
    ROOT
    / "output/seq533-layer37-ffn-input-source-gate-20260709Tseq533Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq533-layer37-ffn-input-source-math-20260709Tseq533Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq533-layer37-ffn-input-source-code-20260709Tseq533Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/seq534-layer37-layer-input-source-gate-20260709Tseq534Z"
)

CURRENT_ROUTE = "router_prompt_distribution_layer37_layer_input_source_gate"
NEXT_ROUTE = "router_prompt_distribution_layer36_ffn_input_source_gate"
REJECTED_ROUTE = "router_prompt_distribution_layer36_ffn_math_source"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SOURCE_LAYER = 36
CONSUMER_LAYER = 37
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
    cpu_from_gpu_match_delta = max(
        abs(_num(source_boundary.get("output_cosine"), 1.0) -
            _num(source_residual.get("cpu_ffn_from_gpu_input_cosine"), 1.0)),
        abs(_num(source_boundary.get("output_max_abs_diff")) -
            _num(source_residual.get("cpu_ffn_from_gpu_input_max_abs_diff"))),
        abs(_num(source_boundary.get("output_rmse")) -
            _num(source_residual.get("cpu_ffn_from_gpu_input_rmse"))),
    )
    entries.append({
        "token_index": token_index,
        "kld": step.get("kld"),
        "top1_matches": step.get("top1_matches"),
        "layer36_output_cosine": source_boundary.get("output_cosine"),
        "layer36_output_max_abs_diff":
            source_boundary.get("output_max_abs_diff"),
        "layer37_input_cosine": consumer_boundary.get("input_cosine"),
        "layer37_input_max_abs_diff":
            consumer_boundary.get("input_max_abs_diff"),
        "boundary_link_delta": boundary_link_delta,
        "layer36_layer_input_cosine":
            source_residual.get("layer_input_cosine"),
        "layer36_attention_output_cosine":
            source_residual.get("attention_output_cosine"),
        "layer36_ffn_input_cosine": source_residual.get("ffn_input_cosine"),
        "layer36_ffn_delta_cosine": source_residual.get("ffn_delta_cosine"),
        "layer36_cpu_ffn_from_gpu_input_cosine":
            source_residual.get("cpu_ffn_from_gpu_input_cosine"),
        "layer36_gpu_output_vs_cpu_ffn_cosine":
            source_residual.get("gpu_output_vs_cpu_ffn_cosine"),
        "layer36_gpu_output_vs_cpu_ffn_max_abs_diff":
            source_residual.get("gpu_output_vs_cpu_ffn_max_abs_diff"),
        "layer36_gpu_ffn_delta_vs_cpu_max_abs_diff":
            source_residual.get("gpu_ffn_delta_vs_cpu_max_abs_diff"),
        "layer36_gpu_ffn_norm_vs_cpu_max_abs_diff":
            source_residual.get("gpu_ffn_norm_vs_cpu_max_abs_diff"),
        "cpu_from_gpu_matches_layer36_output_delta":
            cpu_from_gpu_match_delta,
        "router_ids_match": source_components.get("router_ids_match"),
        "router_weight_max_abs_diff":
            source_components.get("router_weight_max_abs_diff"),
        "selected_down_cosine": source_components.get("selected_down_cosine"),
        "shared_down_cosine": source_components.get("shared_down_cosine"),
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
      "min_layer36_output_cosine": min_metric("layer36_output_cosine"),
      "max_layer36_output_abs_diff": max_metric("layer36_output_max_abs_diff"),
      "min_layer37_input_cosine": min_metric("layer37_input_cosine"),
      "max_layer37_input_abs_diff": max_metric("layer37_input_max_abs_diff"),
      "max_boundary_link_delta": max_metric("boundary_link_delta"),
      "min_layer36_ffn_input_cosine": min_metric("layer36_ffn_input_cosine"),
      "max_cpu_from_gpu_matches_layer36_output_delta":
          max_metric("cpu_from_gpu_matches_layer36_output_delta"),
      "min_layer36_gpu_output_vs_cpu_ffn_cosine":
          min_metric("layer36_gpu_output_vs_cpu_ffn_cosine"),
      "max_layer36_gpu_output_vs_cpu_ffn_abs_diff":
          max_metric("layer36_gpu_output_vs_cpu_ffn_max_abs_diff"),
      "max_layer36_gpu_ffn_delta_vs_cpu_abs_diff":
          max_metric("layer36_gpu_ffn_delta_vs_cpu_max_abs_diff"),
      "max_layer36_gpu_ffn_norm_vs_cpu_abs_diff":
          max_metric("layer36_gpu_ffn_norm_vs_cpu_max_abs_diff"),
      "router_id_mismatch_count":
          sum(1 for entry in entries if entry.get("router_ids_match") is False),
  }
  row["target_row_ran"] = (
      row["target_returncode"] == 2
      and row["distribution_required_checks_passed"] is False
      and row["kld_pass"] is False
      and row["top1_pass"] is True
      and position_count >= 8
      and top1_rate >= TOP1_THRESHOLD
      and row["full_attention_state_diff_enabled"] is True
      and row["diagnostic_layer_range"] == "36:38")
  row["boundary_link_ok"] = (
      bool(entries)
      and row["max_boundary_link_delta"] <= BOUNDARY_MATCH_EPS)
  row["gpu_ffn_math_matches_cpu"] = (
      bool(entries)
      and row["min_layer36_gpu_output_vs_cpu_ffn_cosine"] >= COSINE_THRESHOLD
      and row["max_layer36_gpu_output_vs_cpu_ffn_abs_diff"]
      <= MAX_GPU_CPU_MATH_ABS
      and row["max_layer36_gpu_ffn_delta_vs_cpu_abs_diff"]
      <= MAX_GPU_CPU_MATH_ABS
      and row["max_layer36_gpu_ffn_norm_vs_cpu_abs_diff"]
      <= MAX_GPU_CPU_MATH_ABS)
  row["cpu_from_gpu_input_explains_layer36_output"] = (
      bool(entries)
      and row["max_cpu_from_gpu_matches_layer36_output_delta"]
      <= MAX_GPU_CPU_MATH_ABS)
  row["ffn_input_drift_present"] = (
      bool(entries)
      and row["min_layer36_ffn_input_cosine"] < COSINE_THRESHOLD)
  return row


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq533 = _load_json(args.seq533)
  math_row = _row(args.math, "seq534_layer37_layer_input_source_math")
  code_row = _row(args.code, "seq534_layer37_layer_input_source_code")
  checks = [
      {
          "name": "seq533_selected_layer37_layer_input_source_route",
          "pass": (
              seq533.get("required_checks_passed") is True
              and seq533.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 533,
                  "reject_layer37_attention_output_dominant_source_select_layer37_layer_input_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_layer37_layer_input_source_gate",
                  533)),
      },
      {
          "name": "layer37_layer_input_source_rows_target_ran",
          "pass": math_row["target_row_ran"] and code_row["target_row_ran"],
      },
      {
          "name": "layer37_input_equals_layer36_output",
          "pass": math_row["boundary_link_ok"] and code_row["boundary_link_ok"],
      },
      {
          "name": "layer36_gpu_ffn_math_matches_cpu_live_input",
          "pass": (math_row["gpu_ffn_math_matches_cpu"]
                   and code_row["gpu_ffn_math_matches_cpu"]),
      },
      {
          "name": "layer36_cpu_from_gpu_input_explains_output_drift",
          "pass": (math_row["cpu_from_gpu_input_explains_layer36_output"]
                   and code_row["cpu_from_gpu_input_explains_layer36_output"]),
      },
      {
          "name": "layer36_ffn_input_drift_present",
          "pass": (math_row["ffn_input_drift_present"]
                   and code_row["ffn_input_drift_present"]),
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  combined = {
      "failed_step_count":
          math_row["failed_step_count"] + code_row["failed_step_count"],
      "max_boundary_link_delta":
          max(math_row["max_boundary_link_delta"],
              code_row["max_boundary_link_delta"]),
      "max_layer36_gpu_output_vs_cpu_ffn_abs_diff":
          max(math_row["max_layer36_gpu_output_vs_cpu_ffn_abs_diff"],
              code_row["max_layer36_gpu_output_vs_cpu_ffn_abs_diff"]),
      "max_layer36_gpu_ffn_delta_vs_cpu_abs_diff":
          max(math_row["max_layer36_gpu_ffn_delta_vs_cpu_abs_diff"],
              code_row["max_layer36_gpu_ffn_delta_vs_cpu_abs_diff"]),
      "min_layer36_ffn_input_cosine":
          min(math_row["min_layer36_ffn_input_cosine"],
              code_row["min_layer36_ffn_input_cosine"]),
      "router_id_mismatch_count":
          math_row["router_id_mismatch_count"]
          + code_row["router_id_mismatch_count"],
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "correctness_promotion_allowed": False,
      "current_route": CURRENT_ROUTE,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "rejected_route": REJECTED_ROUTE if required else None,
      "disposition": (
          "reject_layer36_ffn_math_source_select_layer36_ffn_input_source"
          if required else
          "block_layer37_layer_input_source_inconsistent_evidence"),
      "checks": checks,
      "math": math_row,
      "code": code_row,
      "combined": combined,
      "inputs": {
          "routes": _rel(args.routes),
          "seq533": _rel(args.seq533),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  lines = [
      "# Seq534 Layer37 Layer-Input Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- max_boundary_link_delta: `{metrics['combined']['max_boundary_link_delta']}`",
      f"- max_layer36_gpu_output_vs_cpu_ffn_abs_diff: `{metrics['combined']['max_layer36_gpu_output_vs_cpu_ffn_abs_diff']}`",
      f"- min_layer36_ffn_input_cosine: `{metrics['combined']['min_layer36_ffn_input_cosine']}`",
      "",
      "This is correctness-route evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq533", type=Path, default=DEFAULT_SEQ533)
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
