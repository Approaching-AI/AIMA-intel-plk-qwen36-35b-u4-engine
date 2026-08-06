#!/usr/bin/env python3
"""Gate the layer-input product-source runtime-handle retention fix."""

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
    / "tools/intel-qwen36-router-full-attention-layer-input-product-"
    "implementation-source-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-layer-input-product-"
    "implementation-source-fix-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ338 = (
    ROOT
    / "output/router-full-attention-layer-input-product-implementation-"
    "probe-gate-20260708Tseq338Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-full-attention-layer-input-product-implementation-"
    "fix-generate-only-20260708Tseq339Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-layer-input-product-implementation-"
    "source-fix-gate-20260708Tseq339Z"
)

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
DECODE_TOKENS = 8
HIDDEN_SIZE = 2048
ROOT_VALUES = len(PRODUCER_LAYERS) * DECODE_TOKENS * HIDDEN_SIZE


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_layer_input_source_gate", BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base gate: {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_base()


def _counter_failure(seq338: dict[str, Any]) -> dict[str, Any]:
  failed = {
      row.get("name"): row
      for row in seq338.get("checks", [])
      if isinstance(row, dict) and row.get("pass") is not True
  }
  counter = failed.get("product_layer_input_source_counters_ready")
  detail = counter.get("detail") if isinstance(counter, dict) else {}
  detail = detail if isinstance(detail, dict) else {}
  return {
      "seq338_rejected": seq338.get("required_checks_passed") is False,
      "disposition": (
          seq338.get("disposition")
          == "reject_full_attention_layer_input_product_implementation_probe"),
      "selected_fix_route": (
          seq338.get("selected_next_route")
          == "router_prompt_full_attention_layer_input_product_implementation_probe_fix_gate"),
      "failed_counter_check": counter is not None,
      "observed_layers_zero": detail.get("observed_layers") == 0,
      "observed_values_zero": detail.get("observed_values") == 0,
      "observed_misses_nine": detail.get("observed_misses") == 9,
      "observed_ready_false": detail.get("observed_ready") is False,
  }


def _keep_prev_markers(text: str) -> dict[str, Any]:
  section = BASE._section(
      text, "DecodeKeepPrevLayerOutputHandle", "DecodeCarrierLayerOutputHandleLoopActive")
  return {
      "extern_flag_present": (
          "extern bool g_decode_full_attention_layer_input_product_source" in text),
      "keep_prev_function_present": "DecodeKeepPrevLayerOutputHandle" in text,
      "keep_prev_retains_layer_input_source": (
          "g_decode_full_attention_layer_input_product_source" in section),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = BASE._load_json(args.routes)
  seq338 = BASE._load_json(args.seq338)
  source_text = BASE._read(args.decode_source)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = BASE._load_json(result_path)
  generated_cpp = BASE._read(cpp_path)
  source = BASE._source_markers(source_text, include_python=True)
  generated = BASE._source_markers(generated_cpp, include_python=False)
  manifest_checks = BASE._manifest_checks(result, args.generate_dir)
  source_keep_prev = _keep_prev_markers(source_text)
  generated_keep_prev = _keep_prev_markers(generated_cpp)
  compile_run = BASE._compile_source(args, args.out_dir)
  seq338_checks = _counter_failure(seq338)

  checks = [
      {
          "name": "seq338_selected_probe_fix_gate",
          "pass": (
              all(seq338_checks.values())
              and BASE._has_candidate(
                  routes, 338,
                  "reject_full_attention_layer_input_product_implementation_probe")
              and BASE._has_switch(
                  routes,
                  "select_router_prompt_full_attention_layer_input_product_implementation_probe_fix_gate",
                  338)
          ),
          "detail": seq338_checks,
      },
      {
          "name": "source_retains_prev_layer_output_for_layer_input_source",
          "pass": all(source_keep_prev.values()),
          "detail": source_keep_prev,
      },
      {
          "name": "generated_cpp_retains_prev_layer_output_for_layer_input_source",
          "pass": all(generated_keep_prev.values()),
          "detail": generated_keep_prev,
      },
      {
          "name": "source_still_has_product_owned_layer_input_contract",
          "pass": source["present"] and source["absent"],
          "detail": source,
      },
      {
          "name": "generated_cpp_still_has_product_owned_layer_input_contract",
          "pass": generated["present"] and generated["absent"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_product_source_not_token_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "generated_source_compiles_locally",
          "pass": compile_run.get("returncode") == 0,
          "detail": compile_run,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": BASE._rel(args.routes),
          "decode_source": BASE._rel(args.decode_source),
          "decode_source_sha256": BASE._sha256(args.decode_source),
          "seq338_probe_gate": BASE._rel(args.seq338),
          "generate_only_result": BASE._rel(result_path),
          "generated_cpp": BASE._rel(cpp_path),
          "generated_cpp_sha256": BASE._sha256(cpp_path),
      },
      "seq338_counter_failure": seq338_checks,
      "source_keep_prev": source_keep_prev,
      "generated_keep_prev": generated_keep_prev,
      "source": source,
      "generated": generated,
      "manifest_checks": manifest_checks,
      "compile": compile_run,
      "layer_input_source": {
          "root": "full_attention_layer_input",
          "producer_layers": PRODUCER_LAYERS,
          "producer_layer_count": len(PRODUCER_LAYERS),
          "decode_tokens": DECODE_TOKENS,
          "hidden_size": HIDDEN_SIZE,
          "root_values": ROOT_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "target_compile_allowed": required,
      "layer_input_product_source_fix_present": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_layer_input_product_implementation_source_fix"
          if required else
          "reject_full_attention_layer_input_product_implementation_source_fix"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_layer_input_product_implementation_fix_target_compile_gate"
          if required else
          "router_prompt_full_attention_layer_input_product_implementation_probe_fix_gate"
      ),
      "next_route_reason": (
          "The layer-input product source now retains the previous layer output "
          "handle whenever the product-owned layer-input source is enabled. "
          "Target compile is required before rerunning the counter probe."
          if required else
          "The probe-fix source did not prove handle retention and product "
          "source shape together. Keep fixing source/generate-only evidence."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not bool(row.get("pass"))]
  lines = [
      "# Router Full-Attention Layer-Input Product Source Fix Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- layer-input root values: `{metrics['layer_input_source']['root_values']}`",
      f"- source keep-prev: `{str(all(metrics['source_keep_prev'].values())).lower()}`",
      f"- generated keep-prev: `{str(all(metrics['generated_keep_prev'].values())).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. It does not launch a token row or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq338", type=Path, default=DEFAULT_SEQ338)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--cxx", default="c++")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "disposition": metrics["disposition"],
      "layer_input_product_source_fix_present": metrics[
          "layer_input_product_source_fix_present"],
      "out_dir": BASE._rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
