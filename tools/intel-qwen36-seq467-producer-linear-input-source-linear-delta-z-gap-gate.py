#!/usr/bin/env python3
"""Split seq466 producer linear input-source source linear delta/z drift."""

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
BASE_GATE = (
    ROOT
    / "tools/intel-qwen36-router-full-attention-deeper-nested-producer-"
    "linear-input-source-linear-delta-z-gap-gate.py"
)

SCHEMA_VERSION = (
    "intel-qwen36-seq467-producer-linear-input-source-linear-delta-z-gap-gate-v0"
)
_PATTERN = (
    "producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source"
)
SOURCE_GAP_ROUTE = (
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_"
    + ((_PATTERN + "_") * 6)
    + _PATTERN
    + "_gap_gate"
)
ATTENTION_ROUTE = SOURCE_GAP_ROUTE.replace(
    "_linear_input_source_gap_gate",
    "_linear_input_source_producer_linear_input_source_attention_output_gap_gate")
CURRENT_ROUTE = ATTENTION_ROUTE.replace(
    "_producer_linear_input_source_attention_output_gap_gate",
    "_producer_linear_input_source_linear_delta_z_gap_gate")
NEXT_ROUTE = CURRENT_ROUTE.replace(
    "_producer_linear_input_source_linear_delta_z_gap_gate",
    "_producer_linear_input_source_linear_input_gap_gate")
MATH_ROUTE = CURRENT_ROUTE.replace(
    "_producer_linear_input_source_linear_delta_z_gap_gate",
    "_producer_linear_input_source_linear_preconv_math_gap_gate")
_SOURCE_GAP_SOURCE = SOURCE_GAP_ROUTE.replace(
    "router_prompt_full_attention_layer_input_preceding_linear_input_source_",
    "source_layer_input_preceding_linear_input_source_",
    1,
)
DISPOSITION_PREFIX = _SOURCE_GAP_SOURCE.removesuffix(
    "_producer_linear_input_source_linear_input_source_layer_input_preceding_"
    "linear_input_source_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_gap_gate")
DIAG_PREFIX = (
    f"{DISPOSITION_PREFIX}_producer_linear_input_source_linear_input_source_"
    "layer_input_preceding_linear_input_source_producer"
)
DISPOSITION_PREFIX = DIAG_PREFIX.removesuffix("_producer")

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ466 = (
    ROOT
    / "output/seq466-producer-linear-input-source-attention-output-gap-gate-20260709Tseq466Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq467-producer-linear-input-source-linear-delta-z-gap-gate-20260709Tseq467Z"
)
COSINE_THRESHOLD = 0.9999


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_seq467_producer_linear_input_source_source_delta_z_base",
      BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base source delta/z gate: {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_base()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _patch_base() -> None:
  BASE.SCHEMA_VERSION = SCHEMA_VERSION
  BASE.CURRENT_ROUTE = CURRENT_ROUTE
  BASE.NEXT_ROUTE = NEXT_ROUTE
  BASE.MATH_ROUTE = MATH_ROUTE
  BASE.DISPOSITION_PREFIX = DISPOSITION_PREFIX
  BASE.DIAG_PREFIX = DIAG_PREFIX

  original_has_candidate = BASE._has_candidate
  original_has_switch = BASE._has_switch

  def has_candidate(routes: dict[str, Any], seq: int,
                    disposition: str) -> bool:
    if seq == 406:
      return original_has_candidate(routes, 466, disposition)
    return original_has_candidate(routes, seq, disposition)

  def has_switch(routes: dict[str, Any], decision: str,
                 seq_covered: int) -> bool:
    if seq_covered == 406:
      return original_has_switch(routes, decision, 466)
    return original_has_switch(routes, decision, seq_covered)

  BASE._has_candidate = has_candidate
  BASE._has_switch = has_switch


def _apply_final_mix_delta_z_classification(
    metrics: dict[str, Any], seq466: dict[str, Any]) -> None:
  min_delta_native_z = _num(
      seq466.get("min_gpu_delta_native_z_cpu_cosine"), 1.0)
  min_native_delta_z = _num(
      seq466.get("min_native_delta_gpu_z_cpu_cosine"), 1.0)
  min_native_recompute = _num(
      seq466.get("min_native_delta_native_z_cpu_cosine"), 1.0)
  final_mix_delta_z = (
      min_delta_native_z < COSINE_THRESHOLD
      and min_native_delta_z < COSINE_THRESHOLD
      and min_native_recompute >= COSINE_THRESHOLD)
  direct_delta_z = bool(metrics.get("delta_z_gap_reproduced"))

  for row in metrics.get("checks", []):
    if row.get("name") == "deeper_nested_source_linear_delta_and_z_gaps_reproduced":
      row["name"] = "deeper_nested_source_linear_final_mix_delta_z_gap_reproduced"
      row["pass"] = final_mix_delta_z
      row["detail"] = {
          "min_delta_output_cosine": metrics["min_delta_output_cosine"],
          "min_z_cosine": metrics["min_z_cosine"],
          "min_gpu_delta_native_z_cpu_cosine": min_delta_native_z,
          "min_native_delta_gpu_z_cpu_cosine": min_native_delta_z,
          "min_native_delta_native_z_cpu_cosine": min_native_recompute,
      }

  required = all(bool(row.get("pass")) for row in metrics.get("checks", []))
  metrics["direct_delta_z_gap_reproduced"] = direct_delta_z
  metrics["final_mix_delta_z_gap_reproduced"] = final_mix_delta_z
  metrics["delta_z_gap_reproduced"] = final_mix_delta_z
  metrics["min_gpu_delta_native_z_cpu_cosine"] = min_delta_native_z
  metrics["min_native_delta_gpu_z_cpu_cosine"] = min_native_delta_z
  metrics["min_native_delta_native_z_cpu_cosine"] = min_native_recompute
  metrics["required_checks_passed"] = required
  if required:
    metrics["diagnostic_classification"] = (
        f"{DIAG_PREFIX}_linear_input_source_linear_layer_input_attn_norm_drift"
    )
    metrics["disposition"] = (
        f"accept_{DISPOSITION_PREFIX}_producer_linear_input_source_linear_delta_z_gap_classification"
    )
    metrics["selected_next_route"] = NEXT_ROUTE
    metrics["next_route_reason"] = (
        "The final-mix split reproduces the source linear gap from both GPU "
        "delta with native z and native delta with GPU z, while native "
        "delta/native z recompute is exact. Live input / attention-norm drift "
        "is present and preconv math matches CPU, so root source linear "
        "input next."
    )


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  lines = [
      "# Seq467 Source Linear Delta/Z Gap Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- min final-output cosine: `{metrics['min_final_output_cosine']}`",
      f"- min direct delta/z cosine: `{metrics['min_delta_output_cosine']}` / `{metrics['min_z_cosine']}`",
      f"- min final-mix delta/z cosine: `{metrics['min_gpu_delta_native_z_cpu_cosine']}` / `{metrics['min_native_delta_gpu_z_cpu_cosine']}`",
      f"- min attn-norm cosine: `{metrics['min_attn_norm_cosine']}`",
      f"- min GPU preconv math cosines: `{metrics['min_gpu_attn_norm_vs_cpu_cosine']}` / `{metrics['min_gpu_qkv_vs_cpu_cosine']}` / `{metrics['min_gpu_z_vs_cpu_cosine']}`",
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
  parser.add_argument("--seq466", type=Path, default=DEFAULT_SEQ466)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  _patch_base()
  args = parse_args()
  base_args = argparse.Namespace(
      routes=args.routes,
      seq406=args.seq466,
      out_dir=args.out_dir,
  )
  metrics = BASE.compute(base_args)
  _apply_final_mix_delta_z_classification(metrics, _load_json(args.seq466))
  metrics["inputs"]["seq466"] = metrics["inputs"].pop("seq406")
  for row in metrics.get("checks", []):
    if row.get("name") == "seq406_selected_deeper_nested_source_linear_delta_z_gap_gate":
      row["name"] = "seq466_selected_deeper_nested_source_linear_delta_z_gap_gate"
    if row.get("name") == "seq406_run_summaries_available":
      row["name"] = "seq466_run_summaries_available"
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
