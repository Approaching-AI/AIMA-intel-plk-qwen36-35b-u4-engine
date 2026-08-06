#!/usr/bin/env python3
"""Gate source-only full-attention layer-input product-source scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-layer-input-product-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ332 = (
    ROOT
    / "output/router-full-attention-residual-value-gap-diagnostic-gate-20260708Tseq332Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-full-attention-layer-input-product-generate-only-20260708Tseq333Z"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-layer-input-product-source-gate-20260708Tseq333Z"
)

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
DECODE_TOKENS = 8
HIDDEN_SIZE = 2048
SOURCE_VALUES = len(PRODUCER_LAYERS) * DECODE_TOKENS * HIDDEN_SIZE


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _line_of(text: str, pattern: str, *, regex: bool = True) -> int | None:
  if regex:
    match = re.search(pattern, text, flags=re.S | re.M)
    if match is None:
      return None
    return text.count("\n", 0, match.start()) + 1
  index = text.find(pattern)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def _present(text: str, label: str, pattern: str, *,
             regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "present": line is not None, "line": line}


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


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


def _section(text: str, start: str, end: str) -> str:
  start_index = text.find(start)
  if start_index < 0:
    return ""
  end_index = text.find(end, start_index)
  if end_index < 0:
    return text[start_index:]
  return text[start_index:end_index]


def _source_markers(text: str, *, include_python: bool) -> dict[str, Any]:
  struct_section = _section(
      text,
      "struct DecodeFullAttentionLayerInputProductSource",
      "bool DecodeFullAttentionLayerInputProductSourceReady",
  )
  ready_section = _section(
      text,
      "bool DecodeFullAttentionLayerInputProductSourceReady",
      "std::vector<std::uint64_t>",
  )
  present = [
      _present(text, "env_gate",
               "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE", regex=False),
      _present(text, "cxx_arg",
               "full_attention_layer_input_product_source", regex=False),
      _present(text, "cxx_global",
               "bool g_decode_full_attention_layer_input_product_source = false;",
               regex=False),
      _present(text, "contract_struct",
               "DecodeFullAttentionLayerInputProductSource", regex=False),
      _present(text, "contract_function",
               "DecodeFullAttentionLayerInputProductSourceContract",
               regex=False),
      _present(text, "ready_function",
               "DecodeFullAttentionLayerInputProductSourceReady", regex=False),
      _present(text, "capture_function",
               "DecodeCaptureFullAttentionLayerInputProductSource",
               regex=False),
      _present(text, "reset_function",
               "DecodeResetFullAttentionLayerInputProductSourceHandlesForToken",
               regex=False),
      _present(struct_section, "source_only_guard_field",
               "bool source_only_guard = true;", regex=False),
      _present(struct_section, "product_owned_default_false",
               "bool product_owned_source = false;", regex=False),
      _present(struct_section, "cpu_shadow_free_default_false",
               "bool cpu_shadow_free = false;", regex=False),
      _present(struct_section, "host_sync_free_default_false",
               "bool host_sync_free = false;", regex=False),
      _present(struct_section, "resident_layer_input_handle_field",
               "bool resident_layer_input_handle_source = false;",
               regex=False),
      _present(ready_section, "ready_requires_not_product_owned",
               "!source.product_owned_source", regex=False),
      _present(ready_section, "ready_requires_source_only_guard",
               "source.source_only_guard", regex=False),
      _present(text, "resident_handle_guard",
               "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE requires resident attention-front source handles",
               regex=False),
      _present(text, "cpu_shadow_guard",
               "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE is incompatible with CPU-shadow values",
               regex=False),
      _present(text, "source_only_runtime_guard",
               "IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE is source-gate only",
               regex=False),
      _present(text, "stdout_enabled",
               "full_attention_layer_input_product_source_enabled",
               regex=False),
      _present(text, "stdout_ready",
               "full_attention_layer_input_product_source_ready", regex=False),
  ]
  if include_python:
    present.extend([
        _present(text, "python_env_parse",
                 'os.environ.get("IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE")',
                 regex=False),
        _present(text, "run_env_propagates_gate",
                 r"env_parts[\s\S]*?IQ36_FULL_ATTENTION_LAYER_INPUT_PRODUCT_SOURCE"),
        _present(text, "manifest_records_source",
                 '"full_attention_layer_input_product_source"', regex=False),
        _present(text, "manifest_records_values",
                 '"full_attention_layer_input_product_values"', regex=False),
    ])
  return {
      "present": _all_present(present),
      "present_checks": present,
  }


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "full_attention_layer_input_product_source_enabled": (
          result.get("full_attention_layer_input_product_source") is True),
      "full_attention_layer_input_product_layers": (
          result.get("full_attention_layer_input_product_layers")
          == PRODUCER_LAYERS),
      "full_attention_layer_input_product_decode_tokens": (
          result.get("full_attention_layer_input_product_decode_tokens")
          == DECODE_TOKENS),
      "full_attention_layer_input_product_hidden_size": (
          result.get("full_attention_layer_input_product_hidden_size")
          == HIDDEN_SIZE),
      "full_attention_layer_input_product_values": (
          result.get("full_attention_layer_input_product_values")
          == SOURCE_VALUES),
      "residual_product_source_disabled": (
          result.get("full_attention_residual_product_source") is False),
      "residual_product_consumer_source_disabled": (
          result.get("full_attention_residual_product_consumer_source")
          is False),
      "qkv_delta_product_consumer_source_disabled": (
          result.get("router_qkv_delta_product_consumer_source") is False),
      "frontier_stack_present": (
          result.get("shared_q4_runner") is True
          and result.get("resident_q4_weights") is True
          and result.get("resident_selected_q4_experts") is True
          and result.get("resident_selected_q6_experts") is True
          and result.get("resident_selected_q6_sorted_cache") is True
          and result.get("resident_selected_q6_rowstripe") is True
          and result.get("resident_selected_cache_topk") == 16
          and result.get("resident_shared_q6_down") is True
          and result.get("resident_full_attention_v_q6") is True
          and result.get("resident_linear_q6_qkv") is True
          and result.get("resident_q4_cpu_order_z") is True
          and result.get("resident_linear_conv_weights") is True
          and result.get("resident_linear_state") is True
          and result.get("resident_postconv_delta_handoff") is True
          and result.get("resident_norm_weights") is True
          and result.get("resident_gate_up_swiglu_handoff") is True
          and result.get("resident_attention_front_handoff") is True
          and result.get("resident_full_core_attention_front_handoff") is True
          and result.get("gpu_router") is True
          and result.get("gpu_lm_head_q6") is True),
      "speedup_claims_forbidden": (
          result.get("speedup_claims_allowed") is False),
      "no_smoke_json": not (generate_dir / "smoke.json").exists(),
  }


def _compile_source(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  compile_dir = out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  generated_cpp = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  cmd = [
      args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
      _rel(generated_cpp), "-o", _rel(compile_dir / "r2_gpu_decode_smoke.o"),
  ]
  result = subprocess.run(
      cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      check=False)
  return {
      "cmd": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq332 = _load_json(args.seq332)
  decode_source = _read(args.decode_source)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = _read(cpp_path)
  source = _source_markers(decode_source, include_python=True)
  generated = _source_markers(generated_cpp, include_python=False)
  manifest_checks = _manifest_checks(result, args.generate_dir)
  compile_run = _compile_source(args, args.out_dir)

  checks = [
      {
          "name": "seq332_selected_full_attention_layer_input_product_source_gate",
          "pass": (
              seq332.get("required_checks_passed") is True
              and seq332.get("selected_next_route")
              == "router_prompt_full_attention_layer_input_product_source_gate"
              and _has_candidate(
                  routes, 332,
                  "accept_full_attention_residual_value_gap_split_diagnostic")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_layer_input_product_source_gate",
                  332)
          ),
      },
      {
          "name": "decode_source_has_source_only_layer_input_scaffold",
          "pass": source["present"],
          "detail": source,
      },
      {
          "name": "generated_cpp_has_source_only_layer_input_scaffold",
          "pass": generated["present"],
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_records_layer_input_source_shape",
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
          "routes": _rel(args.routes),
          "seq332_residual_value_gap_diagnostic": _rel(args.seq332),
          "decode_source": _rel(args.decode_source),
          "generate_dir": _rel(args.generate_dir),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
      },
      "decode_source_sha256": _sha256(args.decode_source),
      "generated_cpp_sha256": _sha256(cpp_path),
      "result_sha256": _sha256(result_path),
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
          "values": SOURCE_VALUES,
          "source_only_guarded": True,
      },
      "checks": checks,
      "required_checks_passed": required,
      "full_attention_layer_input_product_source_scaffold_present": required,
      "target_compile_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_layer_input_product_source_scaffold"
          if required else
          "block_before_full_attention_layer_input_product_target_compile"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_layer_input_product_target_compile_gate"
          if required else
          "router_prompt_full_attention_layer_input_product_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off source-only wiring now records the full-attention "
          "layer-input root shape without CPU-shadow compatibility. Target "
          "compile is the next admissible gate before any token probe, router "
          "distribution row, speed promotion, or long-context expansion."
          if required else
          "The layer-input product-source scaffold is incomplete; fix source "
          "and generate-only evidence before target compile or token rows."
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
  lines = [
      "# Router Full-Attention Layer-Input Product Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- scaffold_present: `{str(metrics['full_attention_layer_input_product_source_scaffold_present']).lower()}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- layer-input source values: `{metrics['layer_input_source']['values']}`",
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
  parser.add_argument("--seq332", type=Path, default=DEFAULT_SEQ332)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--cxx", default="c++")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "target_compile_allowed": metrics["target_compile_allowed"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
