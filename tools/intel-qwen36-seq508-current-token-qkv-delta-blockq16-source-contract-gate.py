#!/usr/bin/env python3
"""Gate current-token qkv-delta block-q16 source contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-seq508-current-token-qkv-delta-blockq16-source-"
    "contract-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ507 = (
    ROOT
    / "output/seq507-current-token-qkv-delta-design-gate-20260709Tseq507Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z"
    / "token-input"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/seq508-current-token-qkv-delta-blockq16-generate-only-20260709Tseq508Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq508-current-token-qkv-delta-blockq16-source-contract-gate-20260709Tseq508Z"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
DECODE_TOKENS = 8
TOPK = 512
EXPECTED_VALUES = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * TOPK
ROWBLOCK16_26MASK = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,"
    "24,25,26,28,29,30,33,34,36,37,38"
)
SELECTED_ROUTE = (
    "router_prompt_all_linear_current_token_qkv_delta_blockq16_source_contract_gate"
)


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


def _absent(text: str, label: str, pattern: str, *,
            regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "absent": line is None, "line": line}


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


def _all_absent(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("absent") is True for row in rows)


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


def _section(text: str, start: str, end: str) -> str:
  start_index = text.find(start)
  if start_index < 0:
    return ""
  end_index = text.find(end, start_index)
  if end_index < 0:
    return text[start_index:]
  return text[start_index:end_index]


def _run_generate_only(args: argparse.Namespace) -> dict[str, Any]:
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_NO_QUEUE_PROFILING": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16": "1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS": (
          ROWBLOCK16_26MASK),
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED": "1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED": "1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE": "1",
      "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE": "1",
  })
  cmd = [
      sys.executable, _rel(args.decode_source),
      "--token-input-dir", str(args.token_input_dir.resolve()),
      "--case-id", args.case_id,
      "--decode-tokens", str(DECODE_TOKENS),
      "--lm-head-threads", "16",
      "--shared-q4-runner",
      "--resident-q4-weights",
      "--resident-selected-q4-experts",
      "--resident-selected-q6-experts",
      "--resident-selected-q6-sorted-cache",
      "--resident-selected-q6-rowstripe",
      "--resident-selected-cache-topk", "16",
      "--resident-shared-q6-down",
      "--resident-full-attention-v-q6",
      "--resident-linear-q6-qkv",
      "--resident-q4-cpu-order-z",
      "--resident-linear-conv-weights",
      "--resident-linear-state",
      "--resident-postconv-delta-handoff",
      "--resident-norm-weights",
      "--resident-gate-up-swiglu-handoff",
      "--resident-attention-front-handoff",
      "--resident-full-core-attention-front-handoff",
      "--gpu-router",
      "--gpu-lm-head-q6",
      "--opencl-double-swiglu",
      "--generate-only",
      "--out-dir", str(args.generate_dir.resolve()),
  ]
  result = subprocess.run(
      cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, check=False)
  return {
      "cmd": cmd,
      "env": {key: env[key] for key in sorted(env) if key.startswith("IQ36_")},
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
  }


def _decode_markers(text: str, *, include_python: bool) -> dict[str, Any]:
  contract = _section(
      text,
      "DecodeRouterQkvDeltaBlockQ16SourceContract()",
      "bool DecodeRouterQkvDeltaBlockQ16SourceReady",
  )
  ready = _section(
      text,
      "bool DecodeRouterQkvDeltaBlockQ16SourceReady",
      "bool g_decode_full_attention_residual_product_source",
  )
  present = [
      _present(text, "env_gate",
               "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE", regex=False),
      _present(text, "cxx_arg",
               "router_qkv_delta_blockq16_source", regex=False),
      _present(text, "cxx_global",
               "bool g_decode_router_qkv_delta_blockq16_source = false;",
               regex=False),
      _present(text, "contract_struct",
               "DecodeRouterQkvDeltaBlockQ16Source", regex=False),
      _present(text, "contract_function",
               "DecodeRouterQkvDeltaBlockQ16SourceContract", regex=False),
      _present(text, "ready_function",
               "DecodeRouterQkvDeltaBlockQ16SourceReady", regex=False),
      _present(contract, "product_owned_true",
               "source.product_owned_source = true", regex=False),
      _present(contract, "cpu_shadow_free_true",
               "source.cpu_shadow_free = true", regex=False),
      _present(contract, "host_sync_free_true",
               "source.host_sync_free = true", regex=False),
      _present(contract, "all_linear_true",
               "source.all_linear_consumer_source = true", regex=False),
      _present(contract, "entry_group_true",
               "source.covers_entry_group_consumers = true", regex=False),
      _present(contract, "current_token_source_true",
               "source.live_current_token_layer_input_source = true",
               regex=False),
      _present(contract, "qkv_weighted_selector_true",
               "source.live_qkv_weighted_selector_source = true", regex=False),
      _present(contract, "qkv_column_topk_true",
               "source.qkv_column_topk_selector_source = true", regex=False),
      _present(contract, "blockq16_overlay_true",
               "source.block_q16_delta_overlay_kernel_source = true",
               regex=False),
      _present(contract, "sparse_additive_true",
               "source.sparse_delta_additive_overlay_source = true",
               regex=False),
      _present(ready, "ready_requires_all_layers",
               "source.consumer_layers == kDecodeRouterQkvDeltaComponentLayers",
               regex=False),
      _present(ready, "ready_requires_blockq16",
               "source.block_q16_delta_overlay_kernel_source", regex=False),
      _present(text, "requires_shared_runner",
               "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE requires --shared-q4-runner",
               regex=False),
      _present(text, "cpu_shadow_guard",
               "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE is incompatible with CPU-shadow values",
               regex=False),
      _present(text, "source_only_guard",
               "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE is source-gate only",
               regex=False),
      _present(text, "stdout_ready",
               "router_qkv_delta_blockq16_source_ready", regex=False),
      _present(text, "stdout_values",
               "router_qkv_delta_blockq16_values", regex=False),
  ]
  if include_python:
    present.extend([
        _present(text, "python_env_parse",
                 'os.environ.get("IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE")',
                 regex=False),
        _present(text, "run_env_propagates_gate",
                 r"env_parts[\s\S]*?IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE"),
        _present(text, "summary_records_gate",
                 "router qkv-delta block-q16 source", regex=False),
        _present(text, "manifest_records_gate",
                 '"router_qkv_delta_blockq16_source"', regex=False),
    ])
  absent = [
      _absent(text, "no_producer_source_requirement",
              "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE requires IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE",
              regex=False),
      _absent(text, "no_product_consumer_requirement",
              "IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE requires IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE",
              regex=False),
  ]
  return {
      "present": _all_present(present),
      "absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _kernel_markers(opencl: str) -> dict[str, Any]:
  block = _section(
      opencl,
      "__kernel void qkv_delta_blockq16_overlay_f32",
      "__kernel void ffn_moe_weighted_aggregate_f32",
  )
  present = [
      _present(opencl, "blockq16_kernel",
               "__kernel void qkv_delta_blockq16_overlay_f32", regex=False),
      _present(block, "base_input",
               "__global const float* base", regex=False),
      _present(block, "selected_indices",
               "__global const int* selected_indices", regex=False),
      _present(block, "selected_q_delta",
               "__global const short* selected_q_delta", regex=False),
      _present(block, "block_scales",
               "__global const float* block_scales", regex=False),
      _present(block, "block64_scale_index",
               "const uint block = ((uint)index) >> 6", regex=False),
      _present(block, "additive_overlay",
               "output[index] = base[index] +", regex=False),
  ]
  absent = [
      _absent(block, "not_replacement_overlay",
              "output[index] = source[index]", regex=False),
  ]
  return {
      "present": _all_present(present),
      "absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "blockq16_source_enabled": (
          result.get("router_qkv_delta_blockq16_source") is True),
      "blockq16_source_ready": (
          result.get("router_qkv_delta_blockq16_source_ready") is True),
      "blockq16_topk": (
          result.get("router_qkv_delta_blockq16_topk") == TOPK),
      "blockq16_layers": (
          result.get("router_qkv_delta_blockq16_layers") == ALL_LINEAR_LAYERS),
      "blockq16_values": (
          result.get("router_qkv_delta_blockq16_values") == EXPECTED_VALUES),
      "blockq16_selector": (
          result.get("router_qkv_delta_blockq16_selector")
          == "linear_qkv_col_abs"),
      "blockq16_value_mode": (
          result.get("router_qkv_delta_blockq16_value_mode")
          == "shadow_delta_block_q16"),
      "producer_source_disabled": (
          result.get("router_qkv_delta_layer_input_producer_source") is False),
      "device_sparse_overlay_disabled": (
          result.get("router_qkv_delta_device_sparse_overlay_source") is False),
      "product_consumer_source_disabled": (
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
  commands = [
      [
          args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
          _rel(generated_cpp), "-o", _rel(compile_dir / "r2_gpu_decode_smoke.o"),
      ],
      [
          args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
          _rel(args.engine_source), "-o", _rel(compile_dir / "gpu_q4x8_matvec.o"),
      ],
  ]
  runs = []
  for index, command in enumerate(commands):
    proc = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    stdout_path = compile_dir / f"compile{index}.stdout.txt"
    stderr_path = compile_dir / f"compile{index}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    runs.append({
        "command": command,
        "returncode": proc.returncode,
        "stdout": _rel(stdout_path),
        "stderr": _rel(stderr_path),
    })
  return {
      "passed": all(row["returncode"] == 0 for row in runs),
      "runs": runs,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq507 = _load_json(args.seq507)
  generate_run = _run_generate_only(args)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path) if result_path.exists() else {}
  generated_cpp = _read(cpp_path) if cpp_path.exists() else ""
  decode_source = _read(args.decode_source)
  opencl_source = _read(args.opencl_source)
  source = _decode_markers(decode_source, include_python=True)
  generated = _decode_markers(generated_cpp, include_python=False)
  kernel = _kernel_markers(opencl_source)
  manifest_checks = _manifest_checks(result, args.generate_dir)
  compile_result = _compile_source(args, args.out_dir) if cpp_path.exists() else {
      "passed": False,
      "runs": [{"returncode": 1, "stderr": "generated cpp missing"}],
  }
  checks = [
      {
          "name": "seq507_selected_blockq16_source_contract",
          "pass": (
              seq507.get("required_checks_passed") is True
              and seq507.get("selected_next_route") == SELECTED_ROUTE
              and (seq507.get("correction_shape") or {}).get("required_values")
                  == EXPECTED_VALUES
              and (seq507.get("correction_shape") or {}).get("value_mode")
                  == "shadow_delta_block_q16"
              and _has_candidate(
                  routes, 507,
                  "accept_current_token_blockq16_qkv_delta_design_contract")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_current_token_qkv_delta_blockq16_source_contract_gate",
                  507)
          ),
      },
      {
          "name": "generate_only_completed",
          "pass": generate_run.get("returncode") == 0,
          "detail": generate_run,
      },
      {
          "name": "decode_source_has_blockq16_contract",
          "pass": source["present"] and source["absent"],
          "detail": source,
      },
      {
          "name": "generated_cpp_has_blockq16_contract",
          "pass": generated["present"] and generated["absent"],
          "detail": generated,
      },
      {
          "name": "opencl_has_blockq16_additive_overlay_kernel_shape",
          "pass": kernel["present"] and kernel["absent"],
          "detail": kernel,
      },
      {
          "name": "generate_only_manifest_records_blockq16_shape",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "generated_source_compiles_locally",
          "pass": compile_result.get("passed") is True,
          "detail": compile_result,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq507": _rel(args.seq507),
          "token_input_dir": _rel(args.token_input_dir),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "generate_dir": _rel(args.generate_dir),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": (
              _sha256(cpp_path) if cpp_path.exists() else None),
      },
      "generate_run": generate_run,
      "source": source,
      "generated": generated,
      "kernel": kernel,
      "manifest_checks": manifest_checks,
      "compile": compile_result,
      "blockq16_contract": {
          "consumer_layers": ALL_LINEAR_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "topk": TOPK,
          "expected_values": EXPECTED_VALUES,
          "selector": "linear_qkv_col_abs",
          "value_mode": "shadow_delta_block_q16",
          "overlay": "sparse_additive_blockq16_delta",
      },
      "checks": checks,
      "required_checks_passed": required,
      "blockq16_source_contract_present": required,
      "target_compile_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_current_token_qkv_delta_blockq16_source_contract"
          if required else
          "reject_current_token_qkv_delta_blockq16_source_contract"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_target_compile_gate"
          if required else
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_source_fix_gate"
      ),
      "next_route_reason": (
          "The source contract is default-off, CPU-shadow-free, covers all 30 "
          "linear layers including 0/1/2, records qkv-column top512 block-q16 "
          "shape, adds an additive sparse block-q16 overlay kernel shape, and "
          "compiles locally. Target compile is required before token probes or "
          "router distribution."
          if required else
          "The block-q16 source contract is incomplete. Fix source, generated "
          "manifest, kernel shape, or local compile before target compile or "
          "token rows."
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
      "# Seq508 Current-Token QKV-Delta Block-Q16 Source Contract Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- blockq16_source_contract_present: `{str(metrics['blockq16_source_contract_present']).lower()}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- expected values: `{metrics['blockq16_contract']['expected_values']}`",
      f"- selector/value mode: `{metrics['blockq16_contract']['selector']}` / `{metrics['blockq16_contract']['value_mode']}`",
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
  parser.add_argument("--seq507", type=Path, default=DEFAULT_SEQ507)
  parser.add_argument("--token-input-dir", type=Path,
                      default=DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--case-id", default="router_math_reason_001")
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL_SOURCE)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--cxx", default="c++")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "blockq16_source_contract_present": metrics[
          "blockq16_source_contract_present"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
      "target_compile_allowed": metrics["target_compile_allowed"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
