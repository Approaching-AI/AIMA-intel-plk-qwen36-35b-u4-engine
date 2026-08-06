#!/usr/bin/env python3
"""Classify seq495 linear-z sensitivity to the z source."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SEQ495_GATE = (
    ROOT
    / "tools/intel-qwen36-seq495-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq496-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-linear-z-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ495 = (
    ROOT
    / "output/seq495-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-gap-gate-20260709Tseq495Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq496-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-linear-z-gap-gate-20260709Tseq496Z"
)

COSINE_THRESHOLD = 0.9999
EXPECTED_EVENTS = 8
SOURCE_MATCH_EPS = 1.0e-7
MATERIAL_ABS_EPS = 1.0e-12


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ495 = _load_module(SEQ495_GATE, "iq36_seq495_gate")
CURRENT_ROUTE = SEQ495.LINEAR_Z_ROUTE
Z_SOURCE_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_gap_gate", "_linear_z_source_gap_gate")
Z_MATH_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_gap_gate", "_linear_z_math_gap_gate")
LINEAR_INPUT_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_gap_gate", "_linear_input_gap_gate")
DIAG_PREFIX = SEQ495.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ495.DISPOSITION_PREFIX
EXPECTED_CASES = SEQ495.SEQ494.SEQ493.EXPECTED_CASES


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", [])
  )


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
  )


def _load_smoke(row: dict[str, Any]) -> dict[str, Any]:
  out_dir = row.get("out_dir")
  if not isinstance(out_dir, str):
    return {}
  smoke_path = ROOT / out_dir / "smoke.json"
  if not smoke_path.exists():
    return {}
  smoke = _load_json(smoke_path)
  return smoke if isinstance(smoke, dict) else {}


def _summary(
    steps: list[Any],
    source_layers: list[int],
    names: list[str],
    *,
    boundary: bool = False,
) -> dict[str, Any]:
  selected = set(source_layers)
  out: dict[str, Any] = {"observation_count": 0}
  for name in names:
    out[f"min_{name}_cosine"] = 1.0
    out[f"max_{name}_abs_diff"] = 0.0
  for step in steps:
    if not isinstance(step, dict):
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for layer in layers:
      if not isinstance(layer, dict) or layer.get("layer") not in selected:
        continue
      observed = False
      for name in names:
        if not boundary and layer.get(f"{name}_available") is False:
          continue
        cosine = layer.get(f"{name}_cosine")
        max_abs = layer.get(f"{name}_max_abs_diff")
        if isinstance(cosine, (int, float)):
          observed = True
          out[f"min_{name}_cosine"] = min(
              out[f"min_{name}_cosine"], float(cosine))
        if isinstance(max_abs, (int, float)):
          out[f"max_{name}_abs_diff"] = max(
              out[f"max_{name}_abs_diff"], float(max_abs))
      if observed:
        out["observation_count"] += 1
  return out


def _q8_summary(steps: list[Any], source_layers: list[int],
                prefix: str) -> dict[str, Any]:
  selected = set(source_layers)
  out = {
      "observation_count": 0,
      "max_qs_mismatch_count": 0,
      "max_bsums_mismatch_count": 0,
      "max_d_abs_diff": 0.0,
      "max_d_rmse": 0.0,
  }
  for step in steps:
    if not isinstance(step, dict):
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for layer in layers:
      if not isinstance(layer, dict) or layer.get("layer") not in selected:
        continue
      if layer.get(f"{prefix}_q8_available") is not True:
        continue
      out["observation_count"] += 1
      out["max_qs_mismatch_count"] = max(
          out["max_qs_mismatch_count"],
          int(_num(layer.get(f"{prefix}_q8_qs_mismatch_count"))))
      out["max_bsums_mismatch_count"] = max(
          out["max_bsums_mismatch_count"],
          int(_num(layer.get(f"{prefix}_q8_bsums_mismatch_count"))))
      out["max_d_abs_diff"] = max(
          out["max_d_abs_diff"], _num(layer.get(f"{prefix}_q8_d_max_abs_diff")))
      out["max_d_rmse"] = max(
          out["max_d_rmse"], _num(layer.get(f"{prefix}_q8_d_rmse")))
  return out


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  smoke = _load_smoke(row)
  source_layers = row.get("source_layers")
  source_layers = source_layers if isinstance(source_layers, list) else [0]
  source_layers = [
      int(layer) for layer in source_layers if isinstance(layer, int)]
  projection_steps = smoke.get("linear_projection_input_sensitivity_diff_by_step")
  projection_steps = projection_steps if isinstance(projection_steps, list) else []
  boundary_steps = smoke.get("layer_boundary_diff_by_step")
  boundary_steps = boundary_steps if isinstance(boundary_steps, list) else []
  linear_steps = smoke.get("linear_attention_diff_by_step")
  linear_steps = linear_steps if isinstance(linear_steps, list) else []
  preconv_steps = smoke.get("linear_preconv_source_diff_by_step")
  preconv_steps = preconv_steps if isinstance(preconv_steps, list) else []
  final_steps = smoke.get("linear_final_mix_diff_by_step")
  final_steps = final_steps if isinstance(final_steps, list) else []
  return {
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "source_layers": source_layers,
      "projection": _summary(projection_steps, source_layers, [
          "gpu_final_projection_vs_native",
          "native_delta_gpu_z_projection_vs_native",
          "gpu_delta_native_z_projection_vs_native",
      ]),
      "projection_q8_final": _q8_summary(
          projection_steps, source_layers, "gpu_final"),
      "projection_q8_z": _q8_summary(
          projection_steps, source_layers, "native_delta_gpu_z"),
      "boundary": _summary(boundary_steps, source_layers, [
          "input", "output"], boundary=True),
      "linear_attention": _summary(linear_steps, source_layers, [
          "attn_norm", "z", "delta_output", "final_output"]),
      "preconv": _summary(preconv_steps, source_layers, [
          "attn_norm_from_gpu_input",
          "z_from_gpu_attn_norm",
          "gpu_z_vs_cpu",
          "qkv_from_gpu_attn_norm",
          "gpu_qkv_vs_cpu",
      ]),
      "final_mix": _summary(final_steps, source_layers, [
          "gpu_kernel_final",
          "native_delta_native_z_cpu",
          "native_delta_gpu_z_cpu",
          "gpu_delta_native_z_cpu",
      ]),
  }


def _min_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return min((_num(row.get(group, {}).get(key), 1.0) for row in rows),
             default=1.0)


def _max_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return max((_num(row.get(group, {}).get(key)) for row in rows), default=0.0)


def _all_observed(rows: list[dict[str, Any]], group: str) -> bool:
  return all(
      row.get(group, {}).get("observation_count") == EXPECTED_EVENTS
      for row in rows)


def _material(max_abs: float) -> bool:
  return max_abs > MATERIAL_ABS_EPS


def _same_float(left: float, right: float) -> bool:
  return abs(left - right) <= SOURCE_MATCH_EPS


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq495 = _load_json(args.seq495)
  case_rows = seq495.get("runs")
  case_rows = case_rows if isinstance(case_rows, list) else []
  rows = [_case_summary(row) for row in case_rows if isinstance(row, dict)]
  seq495_classification = str(seq495.get("diagnostic_classification"))
  z_source_classification = seq495_classification.replace(
      "_linear_z_gap", "_linear_z_source_gap")
  z_math_classification = seq495_classification.replace(
      "_linear_z_gap", "_linear_z_math_gap")
  linear_input_classification = seq495_classification.replace(
      "_linear_z_gap", "_linear_input_gap")

  preconditions_pass = (
      seq495.get("required_checks_passed") is True
      and seq495.get("selected_next_route") == CURRENT_ROUTE
      and seq495_classification.endswith("_linear_z_gap")
      and _has_candidate(routes, 495, str(seq495.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 495)
  )
  diagnostics_loaded = (
      len(rows) == len(EXPECTED_CASES)
      and _all_observed(rows, "projection")
      and _all_observed(rows, "boundary")
      and _all_observed(rows, "linear_attention")
      and _all_observed(rows, "preconv")
      and _all_observed(rows, "final_mix")
      and all(row.get("projection_q8_final", {}).get("observation_count")
              == EXPECTED_EVENTS for row in rows)
      and all(row.get("projection_q8_z", {}).get("observation_count")
              == EXPECTED_EVENTS for row in rows)
  )

  final_projection_min = _min_case(
      rows, "projection", "min_gpu_final_projection_vs_native_cosine")
  z_projection_min = _min_case(
      rows, "projection", "min_native_delta_gpu_z_projection_vs_native_cosine")
  delta_projection_min = _min_case(
      rows, "projection", "min_gpu_delta_native_z_projection_vs_native_cosine")
  final_projection_abs = _max_case(
      rows, "projection", "max_gpu_final_projection_vs_native_abs_diff")
  z_projection_abs = _max_case(
      rows, "projection", "max_native_delta_gpu_z_projection_vs_native_abs_diff")

  final_qs = max(
      (int(_num(row.get("projection_q8_final", {})
                .get("max_qs_mismatch_count"))) for row in rows),
      default=0)
  z_qs = max(
      (int(_num(row.get("projection_q8_z", {})
                .get("max_qs_mismatch_count"))) for row in rows),
      default=0)

  seq495_final_projection = _num(
      seq495.get("min_cosines", {}).get("gpu_final_projection_vs_native"), 1.0)
  seq495_z_projection = _num(
      seq495.get("min_cosines", {})
      .get("native_delta_gpu_z_projection_vs_native"), 1.0)
  linear_z_sensitivity_reproduced = (
      seq495.get("linear_z_gap") is True
      and diagnostics_loaded
      and _same_float(final_projection_min, seq495_final_projection)
      and _same_float(z_projection_min, seq495_z_projection)
      and _same_float(final_projection_min, z_projection_min)
      and final_qs > 0
      and final_qs == z_qs)

  boundary_input_min = _min_case(rows, "boundary", "min_input_cosine")
  boundary_input_abs = _max_case(rows, "boundary", "max_input_abs_diff")
  boundary_output_min = _min_case(rows, "boundary", "min_output_cosine")
  boundary_output_abs = _max_case(rows, "boundary", "max_output_abs_diff")
  attn_norm_from_input_min = _min_case(
      rows, "preconv", "min_attn_norm_from_gpu_input_cosine")
  attn_norm_from_input_abs = _max_case(
      rows, "preconv", "max_attn_norm_from_gpu_input_abs_diff")
  z_from_norm_min = _min_case(
      rows, "preconv", "min_z_from_gpu_attn_norm_cosine")
  z_from_norm_abs = _max_case(
      rows, "preconv", "max_z_from_gpu_attn_norm_abs_diff")
  gpu_z_vs_cpu_min = _min_case(
      rows, "preconv", "min_gpu_z_vs_cpu_cosine")
  gpu_z_vs_cpu_abs = _max_case(
      rows, "preconv", "max_gpu_z_vs_cpu_abs_diff")
  final_z_min = _min_case(
      rows, "final_mix", "min_native_delta_gpu_z_cpu_cosine")
  final_z_abs = _max_case(
      rows, "final_mix", "max_native_delta_gpu_z_cpu_abs_diff")
  native_recompute_min = _min_case(
      rows, "final_mix", "min_native_delta_native_z_cpu_cosine")
  native_recompute_abs = _max_case(
      rows, "final_mix", "max_native_delta_native_z_cpu_abs_diff")

  live_input_exact = (
      diagnostics_loaded
      and boundary_input_min >= COSINE_THRESHOLD
      and boundary_input_abs == 0.0
      and attn_norm_from_input_min >= COSINE_THRESHOLD
      and attn_norm_from_input_abs == 0.0)
  z_vector_material = (
      diagnostics_loaded
      and z_from_norm_min >= COSINE_THRESHOLD
      and _material(z_from_norm_abs)
      and final_z_min >= COSINE_THRESHOLD
      and _material(final_z_abs))
  z_math_ok = (
      diagnostics_loaded
      and gpu_z_vs_cpu_min >= COSINE_THRESHOLD)
  final_mix_replay_ok = (
      diagnostics_loaded
      and native_recompute_min >= COSINE_THRESHOLD
      and native_recompute_abs == 0.0
      and final_z_min >= COSINE_THRESHOLD)
  linear_input_material = (
      diagnostics_loaded
      and (_material(boundary_input_abs) or _material(attn_norm_from_input_abs)))

  z_source_gap = (
      linear_z_sensitivity_reproduced
      and live_input_exact
      and z_vector_material
      and z_math_ok
      and final_mix_replay_ok)
  z_math_gap = linear_z_sensitivity_reproduced and not z_math_ok
  linear_input_gap = (
      linear_z_sensitivity_reproduced
      and linear_input_material
      and z_math_ok)

  diagnostic_classification = (
      z_source_classification
      if z_source_gap else
      z_math_classification
      if z_math_gap else
      linear_input_classification
      if linear_input_gap else
      f"{seq495_classification}_unclassified"
  )
  selected_next = (
      Z_SOURCE_ROUTE
      if z_source_gap else
      Z_MATH_ROUTE
      if z_math_gap else
      LINEAR_INPUT_ROUTE
      if linear_input_gap else
      CURRENT_ROUTE)

  checks = [
      {"name": "seq495_linear_z_gap_gate", "pass": preconditions_pass},
      {"name": "linear_z_source_rows_loaded",
       "pass": diagnostics_loaded,
       "detail": rows},
      {"name": "linear_z_q8_sensitivity_reproduced",
       "pass": linear_z_sensitivity_reproduced,
       "detail": {
           "seq495_final_projection_cosine": seq495_final_projection,
           "final_projection_cosine": final_projection_min,
           "seq495_z_projection_cosine": seq495_z_projection,
           "z_projection_cosine": z_projection_min,
           "delta_projection_cosine": delta_projection_min,
           "final_q8_qs": final_qs,
           "z_q8_qs": z_qs,
       }},
      {"name": "live_input_exact_but_z_material",
       "pass": live_input_exact and z_vector_material,
       "detail": {
           "min_boundary_input_cosine": boundary_input_min,
           "max_boundary_input_abs_diff": boundary_input_abs,
           "min_boundary_output_cosine": boundary_output_min,
           "max_boundary_output_abs_diff": boundary_output_abs,
           "min_attn_norm_from_gpu_input_cosine": attn_norm_from_input_min,
           "max_attn_norm_from_gpu_input_abs_diff": (
               attn_norm_from_input_abs),
           "min_z_from_gpu_attn_norm_cosine": z_from_norm_min,
           "max_z_from_gpu_attn_norm_abs_diff": z_from_norm_abs,
           "min_native_delta_gpu_z_cpu_cosine": final_z_min,
           "max_native_delta_gpu_z_cpu_abs_diff": final_z_abs,
       }},
      {"name": "gpu_z_math_matches_cpu",
       "pass": z_math_ok,
       "detail": {
           "min_gpu_z_vs_cpu_cosine": gpu_z_vs_cpu_min,
           "max_gpu_z_vs_cpu_abs_diff": gpu_z_vs_cpu_abs,
       }},
      {"name": "final_mix_z_replay_is_clean",
       "pass": final_mix_replay_ok,
       "detail": {
           "min_native_delta_native_z_cpu_cosine": native_recompute_min,
           "max_native_delta_native_z_cpu_abs_diff": native_recompute_abs,
           "min_native_delta_gpu_z_cpu_cosine": final_z_min,
           "max_native_delta_gpu_z_cpu_abs_diff": final_z_abs,
       }},
      {"name": "linear_z_source_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "z_source_gap": z_source_gap,
           "z_math_gap": z_math_gap,
           "linear_input_gap": linear_input_gap,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq495": _rel(args.seq495),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "z_source_gap": z_source_gap,
      "z_math_gap": z_math_gap,
      "linear_input_gap": linear_input_gap,
      "linear_z_sensitivity_reproduced": linear_z_sensitivity_reproduced,
      "live_input_exact": live_input_exact,
      "z_vector_material": z_vector_material,
      "z_math_ok": z_math_ok,
      "final_mix_replay_ok": final_mix_replay_ok,
      "min_cosines": {
          "final_projection": final_projection_min,
          "z_projection": z_projection_min,
          "delta_projection": delta_projection_min,
          "boundary_input": boundary_input_min,
          "boundary_output": boundary_output_min,
          "attn_norm_from_gpu_input": attn_norm_from_input_min,
          "z_from_gpu_attn_norm": z_from_norm_min,
          "gpu_z_vs_cpu": gpu_z_vs_cpu_min,
          "native_delta_gpu_z_cpu": final_z_min,
          "native_delta_native_z_cpu": native_recompute_min,
      },
      "max_abs_diffs": {
          "final_projection": final_projection_abs,
          "z_projection": z_projection_abs,
          "boundary_input": boundary_input_abs,
          "boundary_output": boundary_output_abs,
          "attn_norm_from_gpu_input": attn_norm_from_input_abs,
          "z_from_gpu_attn_norm": z_from_norm_abs,
          "gpu_z_vs_cpu": gpu_z_vs_cpu_abs,
          "native_delta_gpu_z_cpu": final_z_abs,
          "native_delta_native_z_cpu": native_recompute_abs,
      },
      "q8_mismatches": {
          "final_qs": final_qs,
          "z_qs": z_qs,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{seq495_classification}_source_gap_classification"
          if required else
          f"block_{seq495_classification}_source_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The linear-z Q8 sensitivity is carried by a material z vector while "
          "live input and attention-norm input are exact, GPU z math matches "
          "CPU, and native final-mix replay is clean. Root z source next."
          if required and selected_next == Z_SOURCE_ROUTE else
          "GPU z math no longer matches CPU. Root linear z math next."
          if required and selected_next == Z_MATH_ROUTE else
          "The linear-z sensitivity follows material live input. Root live "
          "linear input next."
          if required and selected_next == LINEAR_INPUT_ROUTE else
          "Linear-z source evidence is incomplete; keep this gate open."
      ),
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
  c = metrics["min_cosines"]
  a = metrics["max_abs_diffs"]
  q = metrics["q8_mismatches"]
  lines = [
      "# Seq496 Selected Gate-Up Projection Q8 Input-Sensitivity Linear-Z Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- projection final/z/delta cosines: `{c['final_projection']}` / `{c['z_projection']}` / `{c['delta_projection']}`",
      f"- projection final/z max abs: `{a['final_projection']}` / `{a['z_projection']}`",
      f"- boundary-input/attn-norm-input/z/gpu-z/native-z cosines: `{c['boundary_input']}` / `{c['attn_norm_from_gpu_input']}` / `{c['z_from_gpu_attn_norm']}` / `{c['gpu_z_vs_cpu']}` / `{c['native_delta_gpu_z_cpu']}`",
      f"- boundary-input/attn-norm-input/z/gpu-z/native-z max abs: `{a['boundary_input']}` / `{a['attn_norm_from_gpu_input']}` / `{a['z_from_gpu_attn_norm']}` / `{a['gpu_z_vs_cpu']}` / `{a['native_delta_gpu_z_cpu']}`",
      f"- q8 final/z qs: `{q['final_qs']}` / `{q['z_qs']}`",
      f"- z_source_gap: `{str(metrics['z_source_gap']).lower()}`",
      f"- z_math_gap: `{str(metrics['z_math_gap']).lower()}`",
      f"- linear_input_gap: `{str(metrics['linear_input_gap']).lower()}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is distribution/correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq495", type=Path, default=DEFAULT_SEQ495)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "diagnostic_classification": metrics["diagnostic_classification"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
