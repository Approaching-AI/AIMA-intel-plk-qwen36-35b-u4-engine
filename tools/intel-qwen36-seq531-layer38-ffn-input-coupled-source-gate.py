#!/usr/bin/env python3
"""Classify layer-38 coupled FFN-input source rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq531-layer38-ffn-input-coupled-source-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ530 = (
    ROOT
    / "output/seq530-layer38-ffn-input-source-gate-20260709Tseq530Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq531-layer38-ffn-input-coupled-source-math-20260709Tseq531Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq531-layer38-ffn-input-coupled-source-code-20260709Tseq531Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq531-layer38-ffn-input-coupled-source-gate-20260709Tseq531Z"
)

CURRENT_ROUTE = "router_prompt_distribution_layer38_ffn_input_coupled_source_gate"
NEXT_ROUTE = "router_prompt_distribution_layer38_layer_input_source_gate"
REJECTED_ROUTE = (
    "router_prompt_distribution_layer38_ffn_input_coupled_attention_math_source"
)
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SOURCE_LAYER = 38
COSINE_THRESHOLD = 0.9999
MAX_GPU_CPU_MATH_ABS = 1.0e-4
MATCH_EPS = 1.0e-6

PRECONV_CPU_METRICS = [
    "gpu_attn_norm_vs_cpu",
    "gpu_qkv_vs_cpu",
    "gpu_gate_vs_cpu",
    "gpu_beta_vs_cpu",
    "gpu_z_vs_cpu",
]
PRECONV_DRIFT_METRICS = [
    "attn_norm_from_gpu_input",
    "qkv_from_gpu_attn_norm",
    "gate_from_gpu_attn_norm",
    "beta_from_gpu_attn_norm",
    "z_from_gpu_attn_norm",
]
PROJECTION_MATCH_PAIRS = [
    ("gpu_final_input_vs_native", "gpu_delta_gpu_z_input_vs_native"),
    ("gpu_final_projection_vs_native", "gpu_delta_gpu_z_projection_vs_native"),
]


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


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _projection(step: dict[str, Any]) -> dict[str, Any]:
  projections = step.get("head_pair_projections")
  if not isinstance(projections, list) or not projections:
    return {}
  projection = projections[0]
  return projection if isinstance(projection, dict) else {}


def _failed_indices(smoke: dict[str, Any]) -> set[int]:
  dist = _dist(smoke)
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  return {
      int(step["token_index"]) for step in steps
      if isinstance(step, dict)
      and isinstance(step.get("token_index"), int)
      and _num(step.get("kld")) > KLD_THRESHOLD
  }


def _layer_rows(smoke: dict[str, Any],
                table_name: str,
                failed_indices: set[int]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  table = smoke.get(table_name)
  table = table if isinstance(table, list) else []
  for step in table:
    if not isinstance(step, dict) or step.get("token_index") not in failed_indices:
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if isinstance(row, dict) and row.get("layer") == SOURCE_LAYER:
        item = dict(row)
        item["token_index"] = step.get("token_index")
        out.append(item)
  return out


def _metric_summary(rows: list[dict[str, Any]],
                    metrics: list[str]) -> dict[str, Any]:
  summary: dict[str, Any] = {"observation_count": len(rows)}
  for name in metrics:
    summary[f"min_{name}_cosine"] = 1.0
    summary[f"max_{name}_abs_diff"] = 0.0
    summary[f"{name}_available_count"] = 0
  for row in rows:
    for name in metrics:
      if row.get(f"{name}_available") is False:
        continue
      cosine = row.get(f"{name}_cosine")
      max_abs = row.get(f"{name}_max_abs_diff")
      if isinstance(cosine, (int, float)):
        summary[f"{name}_available_count"] += 1
        summary[f"min_{name}_cosine"] = min(
            summary[f"min_{name}_cosine"], float(cosine))
      if isinstance(max_abs, (int, float)):
        summary[f"max_{name}_abs_diff"] = max(
            summary[f"max_{name}_abs_diff"], float(max_abs))
  return summary


def _boundary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  return {
      "observation_count": len(rows),
      "min_input_cosine": min(
          (_num(row.get("input_cosine"), 1.0) for row in rows), default=1.0),
      "max_input_abs_diff": max(
          (_num(row.get("input_max_abs_diff")) for row in rows), default=0.0),
      "min_output_cosine": min(
          (_num(row.get("output_cosine"), 1.0) for row in rows), default=1.0),
      "max_output_abs_diff": max(
          (_num(row.get("output_max_abs_diff")) for row in rows), default=0.0),
  }


def _projection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  metrics = [name for pair in PROJECTION_MATCH_PAIRS for name in pair]
  summary = _metric_summary(rows, metrics)
  for left, right in PROJECTION_MATCH_PAIRS:
    summary[f"{left}_matches_{right}"] = (
        abs(_num(summary.get(f"min_{left}_cosine"), 1.0) -
            _num(summary.get(f"min_{right}_cosine"), 1.0)) <= MATCH_EPS
        and abs(_num(summary.get(f"max_{left}_abs_diff")) -
                _num(summary.get(f"max_{right}_abs_diff"))) <= MATCH_EPS)
  return summary


def _failed_projection_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  failed = []
  dist = _dist(smoke)
  steps = dist.get("steps")
  steps = steps if isinstance(steps, list) else []
  for step in steps:
    if not isinstance(step, dict) or _num(step.get("kld")) <= KLD_THRESHOLD:
      continue
    projection = _projection(step)
    dims = projection.get("layer38_ffn_input_sensitivity_dims")
    dims = [row for row in dims if isinstance(row, dict)] if (
        isinstance(dims, list)) else []
    failed.append({
        "token_index": step.get("token_index"),
        "kld": step.get("kld"),
        "sensitivity_layer": projection.get(
            "layer38_ffn_input_sensitivity_layer"),
        "sensitivity_dim_count": len(dims),
    })
  return {
      "failed_step_count": len(failed),
      "failed_steps": failed,
      "sensitivity_available": (
          bool(failed)
          and all(row.get("sensitivity_layer") == SOURCE_LAYER
                  and row.get("sensitivity_dim_count") > 0
                  for row in failed)),
  }


def _row(path: Path, label: str) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else payload
  dist = _dist(smoke)
  failed_indices = _failed_indices(smoke)
  preconv_rows = _layer_rows(
      smoke, "linear_preconv_source_diff_by_step", failed_indices)
  attention_rows = _layer_rows(
      smoke, "linear_attention_diff_by_step", failed_indices)
  projection_rows = _layer_rows(
      smoke, "linear_projection_input_sensitivity_diff_by_step",
      failed_indices)
  boundary_rows = _layer_rows(
      smoke, "layer_boundary_diff_by_step", failed_indices)
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "full_attention_state_diff_enabled": smoke.get(
          "full_attention_state_diff_enabled"),
      "diagnostic_layer_range": payload.get("diagnostic_layer_range"),
      "diagnostic_token_limit": smoke.get("diagnostic_token_limit"),
      "failed_indices": sorted(failed_indices),
      "failed_projection": _failed_projection_summary(smoke),
      "preconv": _metric_summary(
          preconv_rows, PRECONV_CPU_METRICS + PRECONV_DRIFT_METRICS),
      "attention": _metric_summary(
          attention_rows, ["delta_output", "final_output"]),
      "projection_input": _projection_summary(projection_rows),
      "boundary": _boundary_summary(boundary_rows),
  }


def _row_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and row.get("full_attention_state_diff_enabled") is True
      and row.get("diagnostic_layer_range") == "38:40"
      and row.get("diagnostic_token_limit") == 8)


def _same_observation_counts(row: dict[str, Any]) -> bool:
  failed_count = len(row.get("failed_indices", []))
  return (
      failed_count > 0
      and row.get("failed_projection", {}).get("sensitivity_available") is True
      and row.get("preconv", {}).get("observation_count") == failed_count
      and row.get("attention", {}).get("observation_count") == failed_count
      and row.get("projection_input", {}).get("observation_count") == failed_count
      and row.get("boundary", {}).get("observation_count") == failed_count)


def _preconv_math_ok(row: dict[str, Any]) -> bool:
  preconv = row.get("preconv")
  preconv = preconv if isinstance(preconv, dict) else {}
  failed_count = len(row.get("failed_indices", []))
  for name in PRECONV_CPU_METRICS:
    if preconv.get(f"{name}_available_count") != failed_count:
      return False
    if _num(preconv.get(f"min_{name}_cosine"), 1.0) < COSINE_THRESHOLD:
      return False
    if _num(preconv.get(f"max_{name}_abs_diff")) > MAX_GPU_CPU_MATH_ABS:
      return False
  return True


def _live_input_source(row: dict[str, Any]) -> bool:
  preconv = row.get("preconv")
  preconv = preconv if isinstance(preconv, dict) else {}
  boundary = row.get("boundary")
  boundary = boundary if isinstance(boundary, dict) else {}
  projection = row.get("projection_input")
  projection = projection if isinstance(projection, dict) else {}
  failed_count = len(row.get("failed_indices", []))
  drift_metrics_ok = True
  for name in PRECONV_DRIFT_METRICS:
    drift_metrics_ok = (
        drift_metrics_ok
        and preconv.get(f"{name}_available_count") == failed_count
        and _num(preconv.get(f"min_{name}_cosine"), 1.0) < COSINE_THRESHOLD
        and _num(preconv.get(f"max_{name}_abs_diff")) > 0.0)
  projection_matches = all(
      projection.get(f"{left}_matches_{right}") is True
      for left, right in PROJECTION_MATCH_PAIRS)
  return (
      _num(boundary.get("min_input_cosine"), 1.0) < COSINE_THRESHOLD
      and _num(boundary.get("max_input_abs_diff")) > 0.0
      and drift_metrics_ok
      and projection_matches)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq530 = _load_json(args.seq530)
  math_row = _row(args.math, "seq531_layer38_ffn_input_coupled_source_math")
  code_row = _row(args.code, "seq531_layer38_ffn_input_coupled_source_code")

  checks = [
      {
          "name": "seq530_selected_layer38_coupled_source_route",
          "pass": (
              seq530.get("required_checks_passed") is True
              and seq530.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 530,
                  "reject_layer38_ffn_input_single_branch_dominance_select_coupled_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_layer38_ffn_input_coupled_source_gate",
                  530)),
      },
      {
          "name": "layer38_coupled_source_rows_target_ran",
          "pass": _row_ran(math_row) and _row_ran(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "failed_step_diagnostics_cover_layer38",
          "pass": _same_observation_counts(math_row)
          and _same_observation_counts(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "layer38_preconv_math_matches_cpu_on_live_input",
          "pass": _preconv_math_ok(math_row) and _preconv_math_ok(code_row),
          "detail": {
              "cosine_threshold": COSINE_THRESHOLD,
              "max_gpu_cpu_math_abs": MAX_GPU_CPU_MATH_ABS,
              "math_preconv": math_row.get("preconv"),
              "code_preconv": code_row.get("preconv"),
          },
      },
      {
          "name": "layer38_coupled_source_inherits_live_layer_input",
          "pass": _live_input_source(math_row) and _live_input_source(code_row),
          "detail": {
              "cosine_threshold": COSINE_THRESHOLD,
              "match_eps": MATCH_EPS,
              "math": {
                  "boundary": math_row.get("boundary"),
                  "preconv": math_row.get("preconv"),
                  "projection_input": math_row.get("projection_input"),
              },
              "code": {
                  "boundary": code_row.get("boundary"),
                  "preconv": code_row.get("preconv"),
                  "projection_input": code_row.get("projection_input"),
              },
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq530": _rel(args.seq530),
          "math": _rel(args.math),
          "code": _rel(args.code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "layer38_attention_math_source_allowed": False,
      "layer38_layer_input_source_gate_allowed": required,
      "rows": {"math": math_row, "code": code_row},
      "disposition": (
          "reject_layer38_ffn_input_coupled_attention_math_source_select_layer38_layer_input_source"
          if required else
          "block_layer38_ffn_input_coupled_source_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The direct layer38 FFN-input branch is the live layer input, and "
          "the attention branch inherits that same live-input drift through "
          "attention norm/qkv/gate/beta/z while GPU preconv math matches CPU "
          "on the live input. The coupled source is therefore layer38 input, "
          "i.e. layer37 output, not attention math/projection. Attribute "
          "layer38 input next before product correction, speed promotion, or "
          "long-context rows."
          if required else
          "Layer38 coupled source evidence is inconsistent; do not switch "
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
      "# Seq531 Layer38 FFN-Input Coupled Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Evidence",
      "",
      f"- math KLD/top1/failed steps: `{rows['math']['max_kld']}` / `{rows['math']['top1_rate']}` / `{len(rows['math']['failed_indices'])}`",
      f"- code KLD/top1/failed steps: `{rows['code']['max_kld']}` / `{rows['code']['top1_rate']}` / `{len(rows['code']['failed_indices'])}`",
      f"- math layer38 input/preconv qkv cosines: `{rows['math']['boundary']['min_input_cosine']}` / `{rows['math']['preconv']['min_qkv_from_gpu_attn_norm_cosine']}`",
      f"- code layer38 input/preconv qkv cosines: `{rows['code']['boundary']['min_input_cosine']}` / `{rows['code']['preconv']['min_qkv_from_gpu_attn_norm_cosine']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is correctness-route evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq530", type=Path, default=DEFAULT_SEQ530)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
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
