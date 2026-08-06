#!/usr/bin/env python3
"""Classify seq494 projection Q8 input sensitivity across final-mix inputs."""

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
SEQ494_GATE = (
    ROOT
    / "tools/intel-qwen36-seq494-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-source-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq495-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ494 = (
    ROOT
    / "output/seq494-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-source-gap-gate-20260709Tseq494Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq495-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-gap-gate-20260709Tseq495Z"
)

COSINE_THRESHOLD = 0.9999
EXPECTED_EVENTS = 8
SOURCE_MATCH_EPS = 1.0e-7

INPUT_METRICS = [
    "gpu_final_input_vs_native",
    "gpu_delta_native_z_input_vs_native",
    "native_delta_gpu_z_input_vs_native",
    "gpu_delta_gpu_z_input_vs_native",
]
PROJECTION_METRICS = [
    "gpu_final_projection_vs_native",
    "gpu_delta_native_z_projection_vs_native",
    "native_delta_gpu_z_projection_vs_native",
    "gpu_delta_gpu_z_projection_vs_native",
]
Q8_PREFIXES = [
    "gpu_final",
    "gpu_delta_native_z",
    "native_delta_gpu_z",
    "gpu_delta_gpu_z",
]


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ494 = _load_module(SEQ494_GATE, "iq36_seq494_gate")
CURRENT_ROUTE = SEQ494.Q8_INPUT_ROUTE
LINEAR_DELTA_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_gap_gate")
LINEAR_Z_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_z_gap_gate")
LINEAR_DELTA_Z_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap_gate")
FINAL_KERNEL_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_final_kernel_gap_gate")
DIAG_PREFIX = SEQ494.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ494.DISPOSITION_PREFIX


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


def _metric_summary(steps: list[Any], source_layers: list[int]) -> dict[str, Any]:
  selected = set(source_layers)
  metrics = INPUT_METRICS + PROJECTION_METRICS
  out: dict[str, Any] = {"observation_count": 0}
  for name in metrics:
    out[f"min_{name}_cosine"] = 1.0
    out[f"max_{name}_abs_diff"] = 0.0
    out[f"first_{name}_gap"] = None
  for prefix in Q8_PREFIXES:
    out[f"{prefix}_q8_observation_count"] = 0
    out[f"{prefix}_max_q8_qs_mismatch_count"] = 0
    out[f"{prefix}_max_q8_bsums_mismatch_count"] = 0
    out[f"{prefix}_max_q8_d_abs_diff"] = 0.0
    out[f"{prefix}_max_q8_d_rmse"] = 0.0
  for step in steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer") not in selected:
        continue
      if row.get("gpu_final_projection_vs_native_available") is True:
        out["observation_count"] += 1
      for name in metrics:
        if row.get(f"{name}_available") is not True:
          continue
        cosine = _num(row.get(f"{name}_cosine"), 1.0)
        max_abs = _num(row.get(f"{name}_max_abs_diff"))
        out[f"min_{name}_cosine"] = min(out[f"min_{name}_cosine"], cosine)
        out[f"max_{name}_abs_diff"] = max(out[f"max_{name}_abs_diff"], max_abs)
        if out[f"first_{name}_gap"] is None and cosine < COSINE_THRESHOLD:
          out[f"first_{name}_gap"] = {
              "token_index": token_index,
              "layer": row.get("layer"),
              f"{name}_cosine": cosine,
              f"{name}_max_abs_diff": max_abs,
          }
      for prefix in Q8_PREFIXES:
        if row.get(f"{prefix}_q8_available") is not True:
          continue
        out[f"{prefix}_q8_observation_count"] += 1
        out[f"{prefix}_max_q8_qs_mismatch_count"] = max(
            out[f"{prefix}_max_q8_qs_mismatch_count"],
            int(_num(row.get(f"{prefix}_q8_qs_mismatch_count"))))
        out[f"{prefix}_max_q8_bsums_mismatch_count"] = max(
            out[f"{prefix}_max_q8_bsums_mismatch_count"],
            int(_num(row.get(f"{prefix}_q8_bsums_mismatch_count"))))
        out[f"{prefix}_max_q8_d_abs_diff"] = max(
            out[f"{prefix}_max_q8_d_abs_diff"],
            _num(row.get(f"{prefix}_q8_d_max_abs_diff")))
        out[f"{prefix}_max_q8_d_rmse"] = max(
            out[f"{prefix}_max_q8_d_rmse"],
            _num(row.get(f"{prefix}_q8_d_rmse")))
  return out


def _load_smoke(row: dict[str, Any]) -> dict[str, Any]:
  out_dir = row.get("out_dir")
  if not isinstance(out_dir, str):
    return {}
  smoke_path = ROOT / out_dir / "smoke.json"
  if not smoke_path.exists():
    return {}
  smoke = _load_json(smoke_path)
  return smoke if isinstance(smoke, dict) else {}


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
  smoke = _load_smoke(row)
  source_layers = row.get("source_layers")
  source_layers = source_layers if isinstance(source_layers, list) else [1]
  source_layers = [int(layer) for layer in source_layers if isinstance(layer, int)]
  steps = smoke.get("linear_projection_input_sensitivity_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  return {
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "source_layers": source_layers,
      "projection_input_sensitivity": _metric_summary(steps, source_layers),
  }


def _min_case(rows: list[dict[str, Any]], key: str) -> float:
  return min(
      (_num(row.get("projection_input_sensitivity", {}).get(key), 1.0)
       for row in rows),
      default=1.0)


def _max_case(rows: list[dict[str, Any]], key: str) -> float:
  return max(
      (_num(row.get("projection_input_sensitivity", {}).get(key))
       for row in rows),
      default=0.0)


def _q8_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
  return {
      "max_qs_mismatch_count": int(_max_case(
          rows, f"{prefix}_max_q8_qs_mismatch_count")),
      "max_bsums_mismatch_count": int(_max_case(
          rows, f"{prefix}_max_q8_bsums_mismatch_count")),
      "max_d_abs_diff": _max_case(rows, f"{prefix}_max_q8_d_abs_diff"),
      "max_d_rmse": _max_case(rows, f"{prefix}_max_q8_d_rmse"),
      "mismatch_observed": (
          _max_case(rows, f"{prefix}_max_q8_qs_mismatch_count") > 0
          or _max_case(rows, f"{prefix}_max_q8_bsums_mismatch_count") > 0
          or _max_case(rows, f"{prefix}_max_q8_d_abs_diff") > 0.0),
  }


def _perturbed(cosine: float, max_abs: float) -> bool:
  return cosine >= COSINE_THRESHOLD and cosine < 1.0 and max_abs > 0.0


def _same_float(left: float, right: float) -> bool:
  return abs(left - right) <= SOURCE_MATCH_EPS


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq494 = _load_json(args.seq494)
  case_rows = seq494.get("runs")
  case_rows = case_rows if isinstance(case_rows, list) else []
  rows = [_case_summary(row) for row in case_rows if isinstance(row, dict)]
  seq494_classification = str(seq494.get("diagnostic_classification"))
  linear_delta_classification = seq494_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap",
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_gap")
  linear_z_classification = seq494_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap",
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_z_gap")
  linear_delta_z_classification = seq494_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap",
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_delta_z_gap")
  final_kernel_classification = seq494_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap",
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_linear_final_kernel_gap")

  preconditions_pass = (
      seq494.get("required_checks_passed") is True
      and seq494.get("selected_next_route") == CURRENT_ROUTE
      and seq494_classification.endswith(
          "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap")
      and _has_candidate(routes, 494, str(seq494.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 494)
  )
  diagnostics_emitted = (
      len(rows) == len(SEQ494.SEQ493.EXPECTED_CASES)
      and all(
          row.get("projection_input_sensitivity", {})
          .get("observation_count") == EXPECTED_EVENTS
          and all(
              row.get("projection_input_sensitivity", {})
              .get(f"{prefix}_q8_observation_count") == EXPECTED_EVENTS
              for prefix in Q8_PREFIXES)
          for row in rows)
  )

  min_cosines = {
      name: _min_case(rows, f"min_{name}_cosine")
      for name in INPUT_METRICS + PROJECTION_METRICS
  }
  max_abs_diffs = {
      name: _max_case(rows, f"max_{name}_abs_diff")
      for name in INPUT_METRICS + PROJECTION_METRICS
  }
  q8_mismatches = {
      prefix: _q8_summary(rows, prefix) for prefix in Q8_PREFIXES
  }

  seq494_attention_output_min = _num(
      seq494.get("min_cosines", {}).get("front_attention_output"), 1.0)
  seq494_attention_output_abs = _num(
      seq494.get("max_abs_diffs", {}).get("front_attention_output"))
  seq494_projection_input_min = _num(
      seq494.get("min_cosines", {}).get("front_projection_input"), 1.0)
  seq494_projection_input_abs = _num(
      seq494.get("max_abs_diffs", {}).get("front_projection_input"))
  seq494_q8 = seq494.get("q8_mismatch", {})

  final_projection_replays_attention = (
      _same_float(
          min_cosines["gpu_final_projection_vs_native"],
          seq494_attention_output_min)
      and _same_float(
          max_abs_diffs["gpu_final_projection_vs_native"],
          seq494_attention_output_abs))
  final_input_replays_projection_input = (
      _same_float(
          min_cosines["gpu_final_input_vs_native"],
          seq494_projection_input_min)
      and _same_float(
          max_abs_diffs["gpu_final_input_vs_native"],
          seq494_projection_input_abs))
  final_q8_replays_seq494 = (
      q8_mismatches["gpu_final"]["max_qs_mismatch_count"]
      == int(_num(seq494_q8.get("max_q8_qs_mismatch_count")))
      and q8_mismatches["gpu_final"]["max_bsums_mismatch_count"]
      == int(_num(seq494_q8.get("max_q8_bsums_mismatch_count")))
      and _same_float(q8_mismatches["gpu_final"]["max_d_abs_diff"],
                      _num(seq494_q8.get("max_q8_d_abs_diff"))))

  final_projection_perturbed = _perturbed(
      min_cosines["gpu_final_projection_vs_native"],
      max_abs_diffs["gpu_final_projection_vs_native"])
  final_q8_sensitivity = (
      diagnostics_emitted
      and final_projection_replays_attention
      and final_input_replays_projection_input
      and final_q8_replays_seq494
      and final_projection_perturbed
      and q8_mismatches["gpu_final"]["mismatch_observed"])

  def variant_perturbed(name: str, prefix: str) -> bool:
    return (
        _perturbed(min_cosines[name], max_abs_diffs[name])
        and q8_mismatches[prefix]["mismatch_observed"])

  delta_variant_sensitivity = variant_perturbed(
      "gpu_delta_native_z_projection_vs_native", "gpu_delta_native_z")
  z_variant_sensitivity = variant_perturbed(
      "native_delta_gpu_z_projection_vs_native", "native_delta_gpu_z")
  combined_variant_sensitivity = variant_perturbed(
      "gpu_delta_gpu_z_projection_vs_native", "gpu_delta_gpu_z")

  def reproduces_final(metric: str, prefix: str) -> bool:
    return (
        _same_float(min_cosines[metric],
                    min_cosines["gpu_final_projection_vs_native"])
        and _same_float(max_abs_diffs[metric],
                        max_abs_diffs["gpu_final_projection_vs_native"])
        and q8_mismatches[prefix]["max_qs_mismatch_count"]
        == q8_mismatches["gpu_final"]["max_qs_mismatch_count"]
        and q8_mismatches[prefix]["max_bsums_mismatch_count"]
        == q8_mismatches["gpu_final"]["max_bsums_mismatch_count"])

  delta_reproduces_final = reproduces_final(
      "gpu_delta_native_z_projection_vs_native", "gpu_delta_native_z")
  z_reproduces_final = reproduces_final(
      "native_delta_gpu_z_projection_vs_native", "native_delta_gpu_z")
  combined_reproduces_final = reproduces_final(
      "gpu_delta_gpu_z_projection_vs_native", "gpu_delta_gpu_z")
  individual_variants_less_sensitive = (
      min_cosines["gpu_delta_native_z_projection_vs_native"]
      > min_cosines["gpu_final_projection_vs_native"]
      and min_cosines["native_delta_gpu_z_projection_vs_native"]
      > min_cosines["gpu_final_projection_vs_native"]
      and q8_mismatches["gpu_delta_native_z"]["max_qs_mismatch_count"]
      < q8_mismatches["gpu_final"]["max_qs_mismatch_count"]
      and q8_mismatches["native_delta_gpu_z"]["max_qs_mismatch_count"]
      < q8_mismatches["gpu_final"]["max_qs_mismatch_count"])

  linear_delta_gap = (
      final_q8_sensitivity
      and delta_variant_sensitivity
      and delta_reproduces_final
      and not z_reproduces_final)
  linear_z_gap = (
      final_q8_sensitivity
      and z_variant_sensitivity
      and z_reproduces_final
      and not delta_reproduces_final)
  linear_delta_z_gap = (
      final_q8_sensitivity
      and combined_variant_sensitivity
      and combined_reproduces_final
      and delta_variant_sensitivity
      and z_variant_sensitivity
      and individual_variants_less_sensitive)
  linear_final_kernel_gap = (
      final_q8_sensitivity
      and not delta_reproduces_final
      and not z_reproduces_final
      and not combined_reproduces_final)

  diagnostic_classification = (
      linear_delta_classification
      if linear_delta_gap else
      linear_z_classification
      if linear_z_gap else
      linear_delta_z_classification
      if linear_delta_z_gap else
      final_kernel_classification
      if linear_final_kernel_gap else
      f"{seq494_classification}_unclassified"
  )
  selected_next = (
      LINEAR_DELTA_ROUTE
      if linear_delta_gap else
      LINEAR_Z_ROUTE
      if linear_z_gap else
      LINEAR_DELTA_Z_ROUTE
      if linear_delta_z_gap else
      FINAL_KERNEL_ROUTE
      if linear_final_kernel_gap else
      CURRENT_ROUTE)

  checks = [
      {"name": "seq494_projection_q8_input_sensitivity_gate",
       "pass": preconditions_pass},
      {"name": "projection_input_sensitivity_rows_loaded",
       "pass": diagnostics_emitted,
       "detail": rows},
      {"name": "gpu_final_projection_replays_seq494_attention_output",
       "pass": (
           final_projection_replays_attention
           and final_input_replays_projection_input
           and final_q8_replays_seq494),
       "detail": {
           "seq494_front_attention_output_cosine": seq494_attention_output_min,
           "gpu_final_projection_vs_native_cosine": (
               min_cosines["gpu_final_projection_vs_native"]),
           "seq494_front_projection_input_cosine": seq494_projection_input_min,
           "gpu_final_input_vs_native_cosine": (
               min_cosines["gpu_final_input_vs_native"]),
           "seq494_q8_mismatch": seq494_q8,
           "gpu_final_q8_mismatch": q8_mismatches["gpu_final"],
       }},
      {"name": "delta_and_z_variants_are_perturbed",
       "pass": delta_variant_sensitivity and z_variant_sensitivity,
       "detail": {
           "delta_projection_cosine": (
               min_cosines["gpu_delta_native_z_projection_vs_native"]),
           "z_projection_cosine": (
               min_cosines["native_delta_gpu_z_projection_vs_native"]),
           "delta_q8": q8_mismatches["gpu_delta_native_z"],
           "z_q8": q8_mismatches["native_delta_gpu_z"],
       }},
      {"name": "final_mix_variant_classifies_source",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "delta_reproduces_final": delta_reproduces_final,
           "z_reproduces_final": z_reproduces_final,
           "combined_reproduces_final": combined_reproduces_final,
           "combined_projection_cosine": (
               min_cosines["gpu_delta_gpu_z_projection_vs_native"]),
           "final_projection_cosine": (
               min_cosines["gpu_final_projection_vs_native"]),
           "combined_q8": q8_mismatches["gpu_delta_gpu_z"],
           "individual_variants_less_sensitive": (
               individual_variants_less_sensitive),
       }},
      {"name": "final_mix_source_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "linear_delta_gap": linear_delta_gap,
           "linear_z_gap": linear_z_gap,
           "linear_delta_z_gap": linear_delta_z_gap,
           "linear_final_kernel_gap": linear_final_kernel_gap,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq494": _rel(args.seq494),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "gpu_final_q8_sensitivity": final_q8_sensitivity,
      "linear_delta_gap": linear_delta_gap,
      "linear_z_gap": linear_z_gap,
      "linear_delta_z_gap": linear_delta_z_gap,
      "linear_final_kernel_gap": linear_final_kernel_gap,
      "min_cosines": min_cosines,
      "max_abs_diffs": max_abs_diffs,
      "q8_mismatches": q8_mismatches,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{seq494_classification}_final_mix_gap_classification"
          if required else
          f"block_{seq494_classification}_final_mix_gap_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The combined GPU-delta/GPU-z final-mix replay reproduces the final "
          "projection Q8 sensitivity, while the one-sided delta and z variants "
          "are perturbed but less sensitive. Root the coupled linear delta/z "
          "path next."
          if required and selected_next == LINEAR_DELTA_Z_ROUTE else
          "The GPU-delta/native-z variant reproduces the final Q8 sensitivity. "
          "Root the linear delta path next."
          if required and selected_next == LINEAR_DELTA_ROUTE else
          "The native-delta/GPU-z variant reproduces the final Q8 sensitivity. "
          "Root the linear z path next."
          if required and selected_next == LINEAR_Z_ROUTE else
          "The final input reproduces the Q8 sensitivity but final-mix variants "
          "do not. Root final-kernel sensitivity next."
          if required and selected_next == FINAL_KERNEL_ROUTE else
          "Final-mix projection input-sensitivity evidence is incomplete; keep "
          "this gate open."
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
  q8 = metrics["q8_mismatches"]
  lines = [
      "# Seq495 Selected Gate-Up FFN-Norm Attention-Output Projection Q8 Input-Sensitivity Final-Mix Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min projection cosines final/delta/z/both: `{c['gpu_final_projection_vs_native']}` / `{c['gpu_delta_native_z_projection_vs_native']}` / `{c['native_delta_gpu_z_projection_vs_native']}` / `{c['gpu_delta_gpu_z_projection_vs_native']}`",
      f"- min input cosines final/delta/z/both: `{c['gpu_final_input_vs_native']}` / `{c['gpu_delta_native_z_input_vs_native']}` / `{c['native_delta_gpu_z_input_vs_native']}` / `{c['gpu_delta_gpu_z_input_vs_native']}`",
      f"- q8 mismatches final qs/bsums/d: `{q8['gpu_final']['max_qs_mismatch_count']}` / `{q8['gpu_final']['max_bsums_mismatch_count']}` / `{q8['gpu_final']['max_d_abs_diff']}`",
      f"- q8 mismatches delta/z/both qs: `{q8['gpu_delta_native_z']['max_qs_mismatch_count']}` / `{q8['native_delta_gpu_z']['max_qs_mismatch_count']}` / `{q8['gpu_delta_gpu_z']['max_qs_mismatch_count']}`",
      f"- linear_delta_gap: `{str(metrics['linear_delta_gap']).lower()}`",
      f"- linear_z_gap: `{str(metrics['linear_z_gap']).lower()}`",
      f"- linear_delta_z_gap: `{str(metrics['linear_delta_z_gap']).lower()}`",
      f"- linear_final_kernel_gap: `{str(metrics['linear_final_kernel_gap']).lower()}`",
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
  parser.add_argument("--seq494", type=Path, default=DEFAULT_SEQ494)
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
