#!/usr/bin/env python3
"""Classify a router-distribution FFN-input source attribution row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-router-ffn-input-source-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_PREDECESSOR = (
    ROOT
    / "output/seq537-layer35-ffn-math-source-gate-20260709Tseq537Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq538-layer35-ffn-input-source-math-20260710Tseq538Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq538-layer35-ffn-input-source-code-20260710Tseq538Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/seq538-layer35-ffn-input-source-gate-20260710Tseq538Z"
)

KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
BRANCH_SHARE_THRESHOLD = 0.60
MAX_CLOSURE_ERROR = 1.0e-6
MAX_NATIVE_CPU_TRACE_ERROR = 1.0e-5

REQUIRED_SENSITIVITY_FIELDS = {
    "index",
    "final_residual_component",
    "layer_input_component_at_native_attention",
    "attention_output_component_at_gpu_layer",
    "attention_output_component_at_native_layer",
    "layer_input_component_at_gpu_attention",
    "native_attention_path_closure_error",
    "gpu_attention_path_closure_error",
    "native_cpu_trace_error",
    "gpu_cpu_trace_error",
}


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


def _has_switch(routes: dict[str, Any], decision: str, seq: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _projection(step: dict[str, Any]) -> dict[str, Any]:
  projections = step.get("head_pair_projections")
  if not isinstance(projections, list) or not projections:
    return {}
  projection = projections[0]
  return projection if isinstance(projection, dict) else {}


def _layer_sensitivity(projection: dict[str, Any], layer: int) -> dict[str, Any]:
  rows = projection.get("ffn_input_sensitivity_by_layer")
  if not isinstance(rows, list):
    return {}
  for row in rows:
    if isinstance(row, dict) and row.get("layer") == layer:
      return row
  return {}


def _component_sums(dims: list[dict[str, Any]]) -> dict[str, float]:
  def sum_abs(name: str) -> float:
    return sum(abs(_num(row.get(name))) for row in dims)

  layer_native = sum_abs("layer_input_component_at_native_attention")
  attn_gpu = sum_abs("attention_output_component_at_gpu_layer")
  attn_native = sum_abs("attention_output_component_at_native_layer")
  layer_gpu = sum_abs("layer_input_component_at_gpu_attention")
  return {
      "final_residual_abs_sum": sum_abs("final_residual_component"),
      "layer_input_native_attention_abs_sum": layer_native,
      "attention_output_gpu_layer_abs_sum": attn_gpu,
      "attention_output_native_layer_abs_sum": attn_native,
      "layer_input_gpu_attention_abs_sum": layer_gpu,
      "layer_input_share_native_attention_path": (
          layer_native / (layer_native + attn_gpu)
          if layer_native + attn_gpu > 0.0 else 0.0),
      "layer_input_share_gpu_attention_path": (
          layer_gpu / (layer_gpu + attn_native)
          if layer_gpu + attn_native > 0.0 else 0.0),
  }


def _failed_step(step: dict[str, Any], layer: int) -> dict[str, Any]:
  projection = _projection(step)
  sensitivity = _layer_sensitivity(projection, layer)
  dims = sensitivity.get("dims")
  dims = [row for row in dims if isinstance(row, dict)] if (
      isinstance(dims, list)) else []
  top_dims = projection.get("top_pair_projection_dims")
  top_dims = top_dims if isinstance(top_dims, list) else []
  closures = [
      abs(_num(row.get("native_attention_path_closure_error")))
      for row in dims
  ] + [
      abs(_num(row.get("gpu_attention_path_closure_error")))
      for row in dims
  ]
  native_trace_errors = [
      abs(_num(row.get("native_cpu_trace_error"))) for row in dims
  ]
  gpu_trace_errors = [
      abs(_num(row.get("gpu_cpu_trace_error"))) for row in dims
  ]
  return {
      "token_index": step.get("token_index"),
      "token_position": step.get("token_position"),
      "kld": step.get("kld"),
      "top1_matches": step.get("top1_matches"),
      "positive_token_id": projection.get("positive_token_id"),
      "negative_token_id": projection.get("negative_token_id"),
      "sensitivity_layer": sensitivity.get("layer"),
      "sensitivity_dim_count": len(dims),
      "top_pair_projection_dim_count": len(top_dims),
      "sensitivity_fields_present": (
          bool(dims)
          and all(REQUIRED_SENSITIVITY_FIELDS.issubset(row) for row in dims)),
      "max_path_closure_error": max(closures) if closures else None,
      "max_native_cpu_trace_error": (
          max(native_trace_errors) if native_trace_errors else None),
      "max_gpu_cpu_trace_error": (
          max(gpu_trace_errors) if gpu_trace_errors else None),
      "component_sums": _component_sums(dims),
      "top_sensitivity_dims": dims[:8],
  }


def _row(path: Path, label: str, layer: int) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else payload
  dist = _dist(smoke)
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  failed_steps = [
      _failed_step(step, layer) for step in steps
      if isinstance(step, dict)
      and (_num(step.get("kld")) > KLD_THRESHOLD
           or step.get("top1_matches") is False)
  ]
  boundary_steps = smoke.get("layer_boundary_diff_by_step")
  boundary_steps = boundary_steps if isinstance(boundary_steps, list) else []
  boundary_by_token = {
      row.get("token_index"): row
      for row in boundary_steps
      if isinstance(row, dict) and isinstance(row.get("token_index"), int)
  }
  for step in failed_steps:
    boundary = boundary_by_token.get(step.get("token_index"), {})
    layers = boundary.get("layers")
    layers = layers if isinstance(layers, list) else []
    step["boundary_layer_count"] = len(layers)
    step["first_input_cosine_below_9999"] = boundary.get(
        "first_input_cosine_below_9999")
    step["first_output_cosine_below_9999"] = boundary.get(
        "first_output_cosine_below_9999")
    first_output = step["first_output_cosine_below_9999"]
    step["first_output_layer"] = next((
        row for row in layers
        if isinstance(row, dict) and row.get("layer") == first_output
    ), None)
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "distribution_required_checks_passed": dist.get(
          "required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "full_attention_state_diff_enabled": smoke.get(
          "full_attention_state_diff_enabled"),
      "diagnostic_layer_range": payload.get("diagnostic_layer_range"),
      "step_count": len(steps),
      "failed_step_count": len(failed_steps),
      "failed_steps": failed_steps,
  }


def _row_ran(row: dict[str, Any], diagnostic_range: str) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and row.get("full_attention_state_diff_enabled") is True
      and row.get("diagnostic_layer_range") == diagnostic_range)


def _sensitivity_available(row: dict[str, Any], layer: int,
                           max_gpu_trace_error: float) -> bool:
  failed = row.get("failed_steps")
  if not isinstance(failed, list) or not failed:
    return False
  for step in failed:
    if step.get("sensitivity_layer") != layer:
      return False
    if step.get("sensitivity_fields_present") is not True:
      return False
    if step.get("sensitivity_dim_count") != step.get(
        "top_pair_projection_dim_count"):
      return False
    if _num(step.get("max_path_closure_error")) > MAX_CLOSURE_ERROR:
      return False
    if _num(step.get("max_native_cpu_trace_error")) > MAX_NATIVE_CPU_TRACE_ERROR:
      return False
    if _num(step.get("max_gpu_cpu_trace_error")) > max_gpu_trace_error:
      return False
  return True


def _branch_state(step: dict[str, Any]) -> str:
  sums = step.get("component_sums")
  sums = sums if isinstance(sums, dict) else {}
  native_share = _num(sums.get("layer_input_share_native_attention_path"))
  gpu_share = _num(sums.get("layer_input_share_gpu_attention_path"))
  if native_share >= BRANCH_SHARE_THRESHOLD and gpu_share >= BRANCH_SHARE_THRESHOLD:
    return "layer_input_dominant"
  if native_share <= 1.0 - BRANCH_SHARE_THRESHOLD and gpu_share <= (
      1.0 - BRANCH_SHARE_THRESHOLD):
    return "attention_output_dominant"
  return "mixed"


def _branch_states(*rows: dict[str, Any]) -> list[str]:
  return [
      _branch_state(step)
      for row in rows
      for step in row.get("failed_steps", [])
  ]


def _first_material_output_layers(*rows: dict[str, Any]) -> list[int]:
  return sorted({
      int(step["first_output_cosine_below_9999"])
      for row in rows
      for step in row.get("failed_steps", [])
      if isinstance(step.get("first_output_cosine_below_9999"), int)
      and step["first_output_cosine_below_9999"] >= 0
  })


def _global_boundary_available(*rows: dict[str, Any]) -> bool:
  failed = [
      step
      for row in rows
      for step in row.get("failed_steps", [])
  ]
  return bool(failed) and all(
      step.get("boundary_layer_count") == 40
      and isinstance(step.get("first_input_cosine_below_9999"), int)
      and isinstance(step.get("first_output_cosine_below_9999"), int)
      and step.get("first_output_layer") is not None
      for step in failed)


def _max_failed_metric(name: str, *rows: dict[str, Any]) -> float:
  return max((
      _num(step.get(name))
      for row in rows
      for step in row.get("failed_steps", [])
  ), default=0.0)


def _min_layer_input_share(*rows: dict[str, Any]) -> float:
  shares = [
      _num(step.get("component_sums", {}).get(name))
      for row in rows
      for step in row.get("failed_steps", [])
      for name in (
          "layer_input_share_native_attention_path",
          "layer_input_share_gpu_attention_path",
      )
  ]
  return min(shares) if shares else 0.0


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  predecessor = _load_json(args.predecessor)
  layer = args.source_layer
  current_route = f"router_prompt_distribution_layer{layer}_ffn_input_source_gate"
  math_row = _row(args.math, f"seq{args.sequence}_layer{layer}_math", layer)
  code_row = _row(args.code, f"seq{args.sequence}_layer{layer}_code", layer)
  prior_math_bound = _num(
      predecessor.get("combined", {}).get(
          "max_live_gpu_output_vs_cpu_ffn_abs_diff"))
  max_gpu_trace_error = max(
      MAX_NATIVE_CPU_TRACE_ERROR, prior_math_bound * 1.05 + 1.0e-6)
  checks = [
      {
          "name": "predecessor_selected_current_route",
          "pass": (
              predecessor.get("required_checks_passed") is True
              and predecessor.get("selected_next_route") == current_route
              and _has_candidate(routes, args.predecessor_seq,
                                 args.predecessor_disposition)
              and _has_switch(routes, f"select_{current_route}",
                              args.predecessor_seq)),
      },
      {
          "name": "ffn_input_source_rows_target_ran",
          "pass": (
              _row_ran(math_row, args.diagnostic_layer_range)
              and _row_ran(code_row, args.diagnostic_layer_range)),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "ffn_input_sensitivity_dimensions_close",
          "pass": (
              _sensitivity_available(math_row, layer, max_gpu_trace_error)
              and _sensitivity_available(code_row, layer, max_gpu_trace_error)),
          "detail": {
              "source_layer": layer,
              "max_path_closure_error": MAX_CLOSURE_ERROR,
              "max_native_cpu_trace_error": MAX_NATIVE_CPU_TRACE_ERROR,
              "max_gpu_cpu_trace_error": max_gpu_trace_error,
              "prior_live_ffn_math_error_bound": prior_math_bound,
          },
      },
      {
          "name": "global_layer_boundary_first_divergence_available",
          "pass": _global_boundary_available(math_row, code_row),
          "detail": {
              "first_material_output_layers": _first_material_output_layers(
                  math_row, code_row),
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  states = _branch_states(math_row, code_row)
  first_material_layers = _first_material_output_layers(math_row, code_row)
  if states and all(state == "layer_input_dominant" for state in states):
    classification = "layer_input_dominant"
    disposition = (
        f"reject_layer{layer}_attention_output_dominant_source_"
        f"select_layer{layer}_layer_input_source")
    rejected_route = (
        f"router_prompt_distribution_layer{layer}_attention_output_dominant_source")
    selected_next_route = (
        f"router_prompt_distribution_layer{layer}_layer_input_source_gate")
  elif states and all(
      state == "attention_output_dominant" for state in states):
    classification = "attention_output_dominant"
    disposition = (
        f"reject_layer{layer}_layer_input_dominant_source_"
        f"select_layer{layer}_attention_output_source")
    rejected_route = (
        f"router_prompt_distribution_layer{layer}_layer_input_dominant_source")
    selected_next_route = (
        f"router_prompt_distribution_layer{layer}_attention_output_source_gate")
  else:
    classification = "coupled_or_mixed"
    first_layer = min(first_material_layers, default=layer)
    last_layer = max(first_material_layers, default=layer)
    disposition = (
        f"reject_layer{layer}_ffn_input_single_branch_dominance_"
        f"select_layers{first_layer}_{last_layer}_first_material_divergence")
    rejected_route = (
        f"router_prompt_distribution_layer{layer}_ffn_input_single_branch_dominance")
    selected_next_route = (
        f"router_prompt_distribution_layers{first_layer}_{last_layer}_"
        "first_material_divergence_source_gate")
  if not required:
    disposition = f"block_layer{layer}_ffn_input_source_inconsistent_evidence"
    rejected_route = None
    selected_next_route = current_route

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "source_layer": layer,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "branch_classification": classification if required else None,
      "branch_states": states,
      "first_material_output_layers": first_material_layers,
      "rows": {"math": math_row, "code": code_row},
      "combined": {
          "failed_step_count": (
              math_row["failed_step_count"] + code_row["failed_step_count"]),
          "min_layer_input_share": _min_layer_input_share(math_row, code_row),
          "max_path_closure_error": _max_failed_metric(
              "max_path_closure_error", math_row, code_row),
          "max_native_cpu_trace_error": _max_failed_metric(
              "max_native_cpu_trace_error", math_row, code_row),
          "max_gpu_cpu_trace_error": _max_failed_metric(
              "max_gpu_cpu_trace_error", math_row, code_row),
      },
      "disposition": disposition,
      "rejected_route": rejected_route,
      "selected_next_route": selected_next_route,
      "next_route_reason": (
          f"Layer-{layer} FFN-input counterfactuals close within the prior "
          f"FFN-math error bound and classify the failed router steps as "
          f"{classification}. The same rows place the first material layer "
          f"output divergence at {first_material_layers}; continue with "
          f"{selected_next_route} instead of descending one layer at a time."
          if required else
          f"Layer-{layer} FFN-input source evidence is inconsistent; keep "
          f"{current_route} open and do not run promotion rows."),
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
      f"# Seq{metrics['sequence']} Layer{metrics['source_layer']} FFN-Input Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- branch_classification: `{metrics['branch_classification']}`",
      f"- first_material_output_layers: `{metrics['first_material_output_layers']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- failed_checks: `{failed}`",
      f"- min_layer_input_share: `{metrics['combined']['min_layer_input_share']}`",
      f"- max_gpu_cpu_trace_error: `{metrics['combined']['max_gpu_cpu_trace_error']}`",
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
  parser.add_argument("--predecessor", type=Path, default=DEFAULT_PREDECESSOR)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--sequence", type=int, default=538)
  parser.add_argument("--source-layer", type=int, default=35)
  parser.add_argument("--predecessor-seq", type=int, default=537)
  parser.add_argument(
      "--predecessor-disposition",
      default="reject_layer35_ffn_math_source_select_layer35_ffn_input_source")
  parser.add_argument("--diagnostic-layer-range", default="35:37")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
