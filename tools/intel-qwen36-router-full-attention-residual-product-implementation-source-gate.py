#!/usr/bin/env python3
"""Gate product-owned full-attention residual source wiring."""

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
    "intel-qwen36-router-full-attention-residual-product-"
    "implementation-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ320 = (
    ROOT
    / "output/router-full-attention-residual-product-probe-gate-20260708Tseq320Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-full-attention-residual-product-implementation-generate-only-20260708Tseq321Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-full-attention-residual-product-implementation-source-gate-20260708Tseq321Z"
)

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
DECODE_TOKENS = 8
HIDDEN_SIZE = 2048
ROOT_VALUES = len(PRODUCER_LAYERS) * DECODE_TOKENS * HIDDEN_SIZE


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


def _count(text: str, pattern: str) -> int:
  return len(re.findall(pattern, text, flags=re.S | re.M))


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
  contract_section = _section(
      text,
      "DecodeFullAttentionResidualProductSourceContract()",
      "bool DecodeFullAttentionResidualProductSourceReady",
  )
  ready_section = _section(
      text,
      "bool DecodeFullAttentionResidualProductSourceReady",
      "std::vector<std::uint64_t>",
  )
  capture_call_count = _count(
      text,
      r"DecodeCaptureFullAttentionResidualProductSource\(\s*layer,\s*ffn_residual_handle_for_tail,\s*stats\)",
  )
  present = [
      _present(text, "env_gate",
               "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE", regex=False),
      _present(text, "cxx_arg",
               "full_attention_residual_product_source", regex=False),
      _present(text, "cxx_global",
               "bool g_decode_full_attention_residual_product_source = false;",
               regex=False),
      _present(text, "source_handle_vector",
               "g_decode_full_attention_residual_product_source_handles",
               regex=False),
      _present(text, "contract_struct",
               "DecodeFullAttentionResidualProductSource", regex=False),
      _present(contract_section, "product_owned_true",
               "source.product_owned_source = true", regex=False),
      _present(contract_section, "cpu_shadow_free_true",
               "source.cpu_shadow_free = true", regex=False),
      _present(contract_section, "host_sync_free_true",
               "source.host_sync_free = true", regex=False),
      _present(contract_section, "resident_handle_source_true",
               "source.resident_attention_residual_handle_source = true",
               regex=False),
      _present(contract_section, "source_only_guard_false",
               "source.source_only_guard = false", regex=False),
      _present(ready_section, "ready_requires_product_owned",
               "source.product_owned_source", regex=False),
      _present(ready_section, "ready_requires_no_source_guard",
               "!source.source_only_guard", regex=False),
      _present(text, "capture_function",
               "DecodeCaptureFullAttentionResidualProductSource", regex=False),
      _present(text, "capture_uses_resident_handle",
               "resident_residual_handle", regex=False),
      _present(text, "reset_function",
               "DecodeResetFullAttentionResidualProductSourceHandlesForToken",
               regex=False),
      _present(text, "stdout_layers",
               "full_attention_residual_product_source_layers", regex=False),
      _present(text, "stdout_values",
               "full_attention_residual_product_source_values", regex=False),
      _present(text, "stdout_misses",
               "full_attention_residual_product_source_misses", regex=False),
      _present(text, "stdout_ready",
               "full_attention_residual_product_source_ready", regex=False),
      _present(text, "cpu_shadow_guard",
               "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE is incompatible with CPU-shadow values",
               regex=False),
  ]
  if include_python:
    present.extend([
        _present(text, "python_env_parse",
                 'os.environ.get("IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE")',
                 regex=False),
        _present(text, "run_env_propagates_gate",
                 r"env_parts[\s\S]*?IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE"),
        _present(text, "manifest_records_source",
                 '"full_attention_residual_product_source"', regex=False),
        _present(text, "summary_records_layers",
                 "full-attention residual product source layers", regex=False),
    ])
  absent = [
      _absent(text, "no_residual_source_only_guard",
              "IQ36_FULL_ATTENTION_RESIDUAL_PRODUCT_SOURCE is source-gate only",
              regex=False),
      _absent(text, "no_disabled_require",
              r"Require\(!args\.full_attention_residual_product_source"),
  ]
  return {
      "present": _all_present(present) and capture_call_count >= 3,
      "absent": _all_absent(absent),
      "capture_call_count": capture_call_count,
      "captures_linear_full_core_and_full_core_handoff": capture_call_count >= 3,
      "present_checks": present,
      "absent_checks": absent,
  }


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "full_attention_residual_product_source_enabled": (
          result.get("full_attention_residual_product_source") is True),
      "full_attention_residual_product_layers": (
          result.get("full_attention_residual_product_layers")
          == PRODUCER_LAYERS),
      "full_attention_residual_product_decode_tokens": (
          result.get("full_attention_residual_product_decode_tokens")
          == DECODE_TOKENS),
      "full_attention_residual_product_hidden_size": (
          result.get("full_attention_residual_product_hidden_size")
          == HIDDEN_SIZE),
      "full_attention_residual_product_values": (
          result.get("full_attention_residual_product_values")
          == ROOT_VALUES),
      "qkv_delta_producer_source_disabled": (
          result.get("router_qkv_delta_layer_input_producer_source") is False),
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
  seq320 = _load_json(args.seq320)
  source_text = _read(args.decode_source)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = _read(cpp_path)
  source = _source_markers(source_text, include_python=True)
  generated = _source_markers(generated_cpp, include_python=False)
  manifest_checks = _manifest_checks(result, args.generate_dir)
  compile_run = _compile_source(args, args.out_dir)
  checks = [
      {
          "name": "seq320_selected_implementation_source_gate",
          "pass": (
              seq320.get("required_checks_passed") is True
              and seq320.get("selected_next_route")
              == "router_prompt_full_attention_residual_product_implementation_source_gate"
              and seq320.get("residual_product_source_present") is False
              and _has_candidate(
                  routes, 320,
                  "reject_source_only_residual_as_product_probe_select_implementation_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_residual_product_implementation_source_gate",
                  320)
          ),
      },
      {
          "name": "source_has_product_owned_residual_contract",
          "pass": source["present"] and source["absent"],
          "detail": source,
      },
      {
          "name": "generated_cpp_has_product_owned_residual_contract",
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
          "routes": _rel(args.routes),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "seq320_probe_gate": _rel(args.seq320),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "source": source,
      "generated": generated,
      "manifest_checks": manifest_checks,
      "compile": compile_run,
      "residual_source": {
          "root": "full_attention_ffn_residual_input",
          "producer_layers": PRODUCER_LAYERS,
          "producer_layer_count": len(PRODUCER_LAYERS),
          "decode_tokens": DECODE_TOKENS,
          "hidden_size": HIDDEN_SIZE,
          "root_values": ROOT_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "target_compile_allowed": required,
      "residual_product_source_present": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_residual_product_implementation_source"
          if required else
          "reject_full_attention_residual_product_implementation_source"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_product_implementation_target_compile_gate"
          if required else
          "router_prompt_full_attention_residual_product_implementation_source_fix_gate"
      ),
      "next_route_reason": (
          "Product-owned residual source wiring now captures resident "
          "full-attention FFN residual handles and records ready/value/miss "
          "counters without CPU-shadow values. Target compile is required "
          "before any token probe, router distribution row, speed promotion, "
          "or long-context expansion."
          if required else
          "The product-owned residual source wiring is incomplete. Fix the "
          "source/generate-only contract before target compile or token rows."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if not bool(row.get("pass"))
  ]
  lines = [
      "# Router Full-Attention Residual Product Implementation Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- residual_product_source_present: `{str(metrics['residual_product_source_present']).lower()}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- residual root values: `{metrics['residual_source']['root_values']}`",
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
  parser.add_argument("--seq320", type=Path, default=DEFAULT_SEQ320)
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
      "out_dir": _rel(args.out_dir),
      "residual_product_source_present": metrics[
          "residual_product_source_present"],
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
