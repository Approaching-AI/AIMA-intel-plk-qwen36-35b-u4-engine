#!/usr/bin/env python3
"""Classify seq493 attention-output source to projection Q8 input sensitivity."""

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
SEQ493_GATE = (
    ROOT
    / "tools/intel-qwen36-seq493-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq494-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-source-gap-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ493 = (
    ROOT
    / "output/seq493-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-gap-gate-20260709Tseq493Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq494-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-source-gap-gate-20260709Tseq494Z"
)

COSINE_THRESHOLD = 0.9999
EXPECTED_EVENTS = 8
SOURCE_MATCH_EPS = 1.0e-8


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ493 = _load_module(SEQ493_GATE, "iq36_seq493_gate")
CURRENT_ROUTE = SEQ493.ATTENTION_OUTPUT_ROUTE
Q8_INPUT_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_source_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap_gate")
PROJECTION_INPUT_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_source_gap_gate",
    "_ffn_norm_input_sensitivity_attention_projection_input_sensitivity_gap_gate")
GPU_KERNEL_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_source_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_gpu_kernel_gap_gate")
NATIVE_RECOMPUTE_ROUTE = CURRENT_ROUTE.replace(
    "_ffn_norm_input_sensitivity_attention_output_source_gap_gate",
    "_ffn_norm_input_sensitivity_attention_output_projection_native_recompute_gap_gate")
DIAG_PREFIX = SEQ493.DIAG_PREFIX
DISPOSITION_PREFIX = SEQ493.DISPOSITION_PREFIX


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


def _metric_summary(steps: list[Any], source_layers: list[int],
                    metric_names: list[str], required_metric: str) -> dict[str, Any]:
  selected = set(source_layers)
  out: dict[str, Any] = {"observation_count": 0}
  for name in metric_names:
    out[f"min_{name}_cosine"] = 1.0
    out[f"max_{name}_abs_diff"] = 0.0
    out[f"first_{name}_gap"] = None
  for step in steps:
    if not isinstance(step, dict):
      continue
    token_index = step.get("token_index")
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer") not in selected:
        continue
      if row.get(f"{required_metric}_available") is not True:
        continue
      out["observation_count"] += 1
      for name in metric_names:
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
  return out


def _projection_source_summary(smoke: dict[str, Any],
                               source_layers: list[int]) -> dict[str, Any]:
  steps = smoke.get("linear_projection_source_diff_by_step")
  steps = steps if isinstance(steps, list) else []
  out = _metric_summary(steps, source_layers, [
      "native_recompute_vs_native",
      "cpu_projection_from_gpu_input_vs_native",
      "gpu_output_vs_cpu_projection_from_gpu_input",
      "native_recompute_vs_cpu_projection_from_gpu_input",
  ], "native_recompute_vs_native")
  out.update({
      "q8_observation_count": 0,
      "max_q8_qs_mismatch_count": 0,
      "max_q8_bsums_mismatch_count": 0,
      "max_q8_d_abs_diff": 0.0,
  })
  selected = set(source_layers)
  for step in steps:
    if not isinstance(step, dict):
      continue
    layers = step.get("layers")
    layers = layers if isinstance(layers, list) else []
    for row in layers:
      if not isinstance(row, dict) or row.get("layer") not in selected:
        continue
      if row.get("q8_bridge_available") is not True:
        continue
      out["q8_observation_count"] += 1
      out["max_q8_qs_mismatch_count"] = max(
          out["max_q8_qs_mismatch_count"],
          int(_num(row.get("q8_qs_mismatch_count"))))
      out["max_q8_bsums_mismatch_count"] = max(
          out["max_q8_bsums_mismatch_count"],
          int(_num(row.get("q8_bsums_mismatch_count"))))
      out["max_q8_d_abs_diff"] = max(
          out["max_q8_d_abs_diff"], _num(row.get("q8_d_max_abs_diff")))
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
  front_steps = smoke.get("attention_front_diff_by_step")
  front_steps = front_steps if isinstance(front_steps, list) else []
  front = _metric_summary(front_steps, source_layers, [
      "projection_input",
      "attention_output",
  ], "attention_output")
  projection = _projection_source_summary(smoke, source_layers)
  return {
      "case_id": row.get("case_id"),
      "out_dir": row.get("out_dir"),
      "source_layers": source_layers,
      "attention_front": front,
      "projection_source": projection,
  }


def _min_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return min(
      (_num(row.get(group, {}).get(key), 1.0) for row in rows),
      default=1.0)


def _max_case(rows: list[dict[str, Any]], group: str, key: str) -> float:
  return max(
      (_num(row.get(group, {}).get(key)) for row in rows),
      default=0.0)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq493 = _load_json(args.seq493)
  case_rows = seq493.get("runs")
  case_rows = case_rows if isinstance(case_rows, list) else []
  rows = [_case_summary(row) for row in case_rows if isinstance(row, dict)]
  seq493_classification = str(seq493.get("diagnostic_classification"))
  q8_classification = seq493_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_source_gap",
      "_ffn_norm_input_sensitivity_attention_output_projection_q8_input_sensitivity_gap")
  projection_input_classification = seq493_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_source_gap",
      "_ffn_norm_input_sensitivity_attention_projection_input_sensitivity_gap")
  gpu_kernel_classification = seq493_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_source_gap",
      "_ffn_norm_input_sensitivity_attention_output_projection_gpu_kernel_gap")
  native_recompute_classification = seq493_classification.replace(
      "_ffn_norm_input_sensitivity_attention_output_source_gap",
      "_ffn_norm_input_sensitivity_attention_output_projection_native_recompute_gap")

  preconditions_pass = (
      seq493.get("required_checks_passed") is True
      and seq493.get("selected_next_route") == CURRENT_ROUTE
      and seq493_classification.endswith(
          "_ffn_norm_input_sensitivity_attention_output_source_gap")
      and _has_candidate(routes, 493, str(seq493.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 493)
  )
  diagnostics_emitted = (
      len(rows) == len(SEQ493.EXPECTED_CASES)
      and all(
          row.get("attention_front", {}).get("observation_count") == EXPECTED_EVENTS
          and row.get("projection_source", {}).get("observation_count") == EXPECTED_EVENTS
          and row.get("projection_source", {}).get("q8_observation_count") == EXPECTED_EVENTS
          for row in rows)
  )

  front_projection_input_min = _min_case(
      rows, "attention_front", "min_projection_input_cosine")
  front_projection_input_abs = _max_case(
      rows, "attention_front", "max_projection_input_abs_diff")
  front_attention_output_min = _min_case(
      rows, "attention_front", "min_attention_output_cosine")
  front_attention_output_abs = _max_case(
      rows, "attention_front", "max_attention_output_abs_diff")
  native_recompute_min = _min_case(
      rows, "projection_source", "min_native_recompute_vs_native_cosine")
  native_recompute_abs = _max_case(
      rows, "projection_source", "max_native_recompute_vs_native_abs_diff")
  cpu_from_gpu_input_min = _min_case(
      rows, "projection_source",
      "min_cpu_projection_from_gpu_input_vs_native_cosine")
  cpu_from_gpu_input_abs = _max_case(
      rows, "projection_source",
      "max_cpu_projection_from_gpu_input_vs_native_abs_diff")
  gpu_vs_cpu_projection_min = _min_case(
      rows, "projection_source",
      "min_gpu_output_vs_cpu_projection_from_gpu_input_cosine")
  gpu_vs_cpu_projection_abs = _max_case(
      rows, "projection_source",
      "max_gpu_output_vs_cpu_projection_from_gpu_input_abs_diff")
  native_vs_cpu_projection_min = _min_case(
      rows, "projection_source",
      "min_native_recompute_vs_cpu_projection_from_gpu_input_cosine")
  native_vs_cpu_projection_abs = _max_case(
      rows, "projection_source",
      "max_native_recompute_vs_cpu_projection_from_gpu_input_abs_diff")
  max_q8_qs_mismatch = max(
      (int(_num(row.get("projection_source", {})
                .get("max_q8_qs_mismatch_count"))) for row in rows),
      default=0)
  max_q8_bsums_mismatch = max(
      (int(_num(row.get("projection_source", {})
                .get("max_q8_bsums_mismatch_count"))) for row in rows),
      default=0)
  max_q8_d_abs_diff = _max_case(
      rows, "projection_source", "max_q8_d_abs_diff")

  seq493_attention_output_min = _num(
      seq493.get("min_cosines", {}).get("attention_output"), 1.0)
  seq493_attention_output_abs = _num(
      seq493.get("max_abs_diffs", {}).get("attention_output"))
  attention_source_replayed = (
      abs(front_attention_output_min - seq493_attention_output_min)
      <= SOURCE_MATCH_EPS
      and abs(front_attention_output_abs - seq493_attention_output_abs)
      <= SOURCE_MATCH_EPS)
  attention_output_perturbed = (
      diagnostics_emitted
      and front_attention_output_min >= COSINE_THRESHOLD
      and front_attention_output_min < 1.0
      and front_attention_output_abs > 0.0)
  projection_input_observed = (
      diagnostics_emitted
      and front_projection_input_min >= COSINE_THRESHOLD
      and front_projection_input_abs > 0.0)
  native_recompute_ok = (
      diagnostics_emitted
      and native_recompute_min >= COSINE_THRESHOLD
      and native_recompute_abs == 0.0)
  cpu_from_gpu_input_replays_attention = (
      diagnostics_emitted
      and abs(cpu_from_gpu_input_min - front_attention_output_min)
      <= SOURCE_MATCH_EPS
      and abs(cpu_from_gpu_input_abs - front_attention_output_abs)
      <= SOURCE_MATCH_EPS)
  native_vs_cpu_projection_replays_attention = (
      diagnostics_emitted
      and abs(native_vs_cpu_projection_min - front_attention_output_min)
      <= SOURCE_MATCH_EPS
      and abs(native_vs_cpu_projection_abs - front_attention_output_abs)
      <= SOURCE_MATCH_EPS)
  gpu_output_matches_cpu_projection = (
      diagnostics_emitted
      and gpu_vs_cpu_projection_min >= COSINE_THRESHOLD)
  q8_mismatch_observed = (
      diagnostics_emitted
      and (max_q8_qs_mismatch > 0
           or max_q8_bsums_mismatch > 0
           or max_q8_d_abs_diff > 0.0))

  q8_input_sensitivity_gap = (
      seq493.get("attention_output_source_gap") is True
      and diagnostics_emitted
      and attention_source_replayed
      and attention_output_perturbed
      and projection_input_observed
      and native_recompute_ok
      and cpu_from_gpu_input_replays_attention
      and native_vs_cpu_projection_replays_attention
      and gpu_output_matches_cpu_projection
      and q8_mismatch_observed)
  projection_input_source_gap = (
      seq493.get("attention_output_source_gap") is True
      and diagnostics_emitted
      and attention_source_replayed
      and projection_input_observed
      and not q8_mismatch_observed)
  gpu_kernel_gap = (
      seq493.get("attention_output_source_gap") is True
      and diagnostics_emitted
      and attention_source_replayed
      and not gpu_output_matches_cpu_projection)
  native_recompute_gap = (
      seq493.get("attention_output_source_gap") is True
      and diagnostics_emitted
      and not native_recompute_ok)

  diagnostic_classification = (
      q8_classification
      if q8_input_sensitivity_gap else
      projection_input_classification
      if projection_input_source_gap else
      gpu_kernel_classification
      if gpu_kernel_gap else
      native_recompute_classification
      if native_recompute_gap else
      f"{DIAG_PREFIX}_linear_input_source_linear_input_source_layer_input_upstream_layer_output_ffn_delta_ffn_norm_input_attention_output_source_gap_unclassified"
  )
  selected_next = (
      Q8_INPUT_ROUTE
      if q8_input_sensitivity_gap else
      PROJECTION_INPUT_ROUTE
      if projection_input_source_gap else
      GPU_KERNEL_ROUTE
      if gpu_kernel_gap else
      NATIVE_RECOMPUTE_ROUTE
      if native_recompute_gap else
      CURRENT_ROUTE)

  checks = [
      {"name": "seq493_attention_output_source_gate",
       "pass": preconditions_pass},
      {"name": "projection_source_rows_loaded",
       "pass": diagnostics_emitted,
       "detail": rows},
      {"name": "attention_output_source_replayed",
       "pass": attention_source_replayed,
       "detail": {
           "seq493_min_attention_output_cosine": seq493_attention_output_min,
           "front_min_attention_output_cosine": front_attention_output_min,
           "seq493_max_attention_output_abs_diff": seq493_attention_output_abs,
           "front_max_attention_output_abs_diff": front_attention_output_abs,
       }},
      {"name": "projection_input_clean_observed",
       "pass": projection_input_observed,
       "detail": {
           "min_front_projection_input_cosine": front_projection_input_min,
           "max_front_projection_input_abs_diff": front_projection_input_abs,
           "min_front_attention_output_cosine": front_attention_output_min,
           "max_front_attention_output_abs_diff": front_attention_output_abs,
       }},
      {"name": "native_projection_recompute_exact",
       "pass": native_recompute_ok,
       "detail": {
           "min_native_recompute_vs_native_cosine": native_recompute_min,
           "max_native_recompute_vs_native_abs_diff": native_recompute_abs,
       }},
      {"name": "projection_from_gpu_input_replays_attention_output",
       "pass": (
           cpu_from_gpu_input_replays_attention
           and native_vs_cpu_projection_replays_attention),
       "detail": {
           "min_cpu_projection_from_gpu_input_vs_native_cosine": (
               cpu_from_gpu_input_min),
           "max_cpu_projection_from_gpu_input_vs_native_abs_diff": (
               cpu_from_gpu_input_abs),
           "min_native_recompute_vs_cpu_projection_from_gpu_input_cosine": (
               native_vs_cpu_projection_min),
           "max_native_recompute_vs_cpu_projection_from_gpu_input_abs_diff": (
               native_vs_cpu_projection_abs),
       }},
      {"name": "gpu_projection_matches_cpu_projection",
       "pass": gpu_output_matches_cpu_projection,
       "detail": {
           "min_gpu_output_vs_cpu_projection_from_gpu_input_cosine": (
               gpu_vs_cpu_projection_min),
           "max_gpu_output_vs_cpu_projection_from_gpu_input_abs_diff": (
               gpu_vs_cpu_projection_abs),
       }},
      {"name": "q8_input_sensitivity_classified",
       "pass": selected_next != CURRENT_ROUTE,
       "detail": {
           "q8_input_sensitivity_gap": q8_input_sensitivity_gap,
           "projection_input_source_gap": projection_input_source_gap,
           "gpu_kernel_gap": gpu_kernel_gap,
           "native_recompute_gap": native_recompute_gap,
           "max_q8_qs_mismatch_count": max_q8_qs_mismatch,
           "max_q8_bsums_mismatch_count": max_q8_bsums_mismatch,
           "max_q8_d_abs_diff": max_q8_d_abs_diff,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq493": _rel(args.seq493),
      },
      "runs": rows,
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": diagnostic_classification,
      "q8_input_sensitivity_gap": q8_input_sensitivity_gap,
      "projection_input_source_gap": projection_input_source_gap,
      "gpu_kernel_gap": gpu_kernel_gap,
      "native_recompute_gap": native_recompute_gap,
      "min_cosines": {
          "front_projection_input": front_projection_input_min,
          "front_attention_output": front_attention_output_min,
          "native_recompute_vs_native": native_recompute_min,
          "cpu_projection_from_gpu_input_vs_native": cpu_from_gpu_input_min,
          "gpu_output_vs_cpu_projection_from_gpu_input": (
              gpu_vs_cpu_projection_min),
          "native_recompute_vs_cpu_projection_from_gpu_input": (
              native_vs_cpu_projection_min),
      },
      "max_abs_diffs": {
          "front_projection_input": front_projection_input_abs,
          "front_attention_output": front_attention_output_abs,
          "native_recompute_vs_native": native_recompute_abs,
          "cpu_projection_from_gpu_input_vs_native": cpu_from_gpu_input_abs,
          "gpu_output_vs_cpu_projection_from_gpu_input": (
              gpu_vs_cpu_projection_abs),
          "native_recompute_vs_cpu_projection_from_gpu_input": (
              native_vs_cpu_projection_abs),
      },
      "q8_mismatch": {
          "max_q8_qs_mismatch_count": max_q8_qs_mismatch,
          "max_q8_bsums_mismatch_count": max_q8_bsums_mismatch,
          "max_q8_d_abs_diff": max_q8_d_abs_diff,
      },
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{seq493_classification}_classification"
          if required else
          f"block_{seq493_classification}_classification"
      ),
      "selected_next_route": selected_next if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Attention-output sensitivity is reproduced by CPU projection from "
          "the live GPU Q8 input; native recompute is exact, GPU projection "
          "matches CPU projection, and Q8 qs/bsums/d mismatches are observed. "
          "Root projection Q8 input sensitivity next."
          if required and selected_next == Q8_INPUT_ROUTE else
          "Attention-output sensitivity follows the projection input without "
          "Q8 bridge mismatch. Root projection input next."
          if required and selected_next == PROJECTION_INPUT_ROUTE else
          "GPU projection no longer matches CPU projection on the live input. "
          "Root projection kernel math next."
          if required and selected_next == GPU_KERNEL_ROUTE else
          "Native projection recompute is not exact. Root native recompute next."
          if required and selected_next == NATIVE_RECOMPUTE_ROUTE else
          "Attention-output projection-source evidence is incomplete; keep "
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
  a = metrics["max_abs_diffs"]
  q = metrics["q8_mismatch"]
  lines = [
      "# Seq494 Selected Gate-Up FFN-Norm Input-Sensitivity Attention-Output Source Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min projection-input/attention-output/native-recompute/cpu-from-gpu/gpu-vs-cpu cosines: `{c['front_projection_input']}` / `{c['front_attention_output']}` / `{c['native_recompute_vs_native']}` / `{c['cpu_projection_from_gpu_input_vs_native']}` / `{c['gpu_output_vs_cpu_projection_from_gpu_input']}`",
      f"- max projection-input/attention-output/native-recompute/cpu-from-gpu/gpu-vs-cpu abs: `{a['front_projection_input']}` / `{a['front_attention_output']}` / `{a['native_recompute_vs_native']}` / `{a['cpu_projection_from_gpu_input_vs_native']}` / `{a['gpu_output_vs_cpu_projection_from_gpu_input']}`",
      f"- q8 qs/bsums/d mismatches: `{q['max_q8_qs_mismatch_count']}` / `{q['max_q8_bsums_mismatch_count']}` / `{q['max_q8_d_abs_diff']}`",
      f"- q8_input_sensitivity_gap: `{str(metrics['q8_input_sensitivity_gap']).lower()}`",
      f"- projection_input_source_gap: `{str(metrics['projection_input_source_gap']).lower()}`",
      f"- gpu_kernel_gap: `{str(metrics['gpu_kernel_gap']).lower()}`",
      f"- native_recompute_gap: `{str(metrics['native_recompute_gap']).lower()}`",
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
  parser.add_argument("--seq493", type=Path, default=DEFAULT_SEQ493)
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
