#!/usr/bin/env python3
"""Gate device sparse-overlay source for router qkv-delta.

This is source evidence. It verifies that the engine has an owned resident F32
selected-value overlay primitive and that decode-smoke exposes a default-off,
source-only route for the all-linear qkv-delta consumer to use it later.
"""

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
    "intel-qwen36-router-qkv-delta-device-sparse-overlay-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ306 = (
    ROOT
    / "output/router-qkv-delta-product-source-from-producer-source-gate-20260708Tseq306Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-qkv-delta-device-sparse-overlay-generate-only-20260708Tseq307Z"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-device-sparse-overlay-source-gate-20260708Tseq307Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
TOPK = 512
DECODE_TOKENS = 8
TOP512_VALUES = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * TOPK
PRODUCER_VALUES = len(PRODUCER_LAYERS) * DECODE_TOKENS * 2048


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


def _engine_markers(header: str, source: str, opencl: str) -> dict[str, Any]:
  combined = "\n".join([header, source, opencl])
  present = [
      _present(header, "overlay_run_struct",
               "GpuRouterQkvDeltaSelectedValueOverlayRun", regex=False),
      _present(header, "overlay_api",
               "RunRouterQkvDeltaSelectedValueOverlay", regex=False),
      _present(source, "overlay_impl",
               "RunRouterQkvDeltaSelectedValueOverlay", regex=False),
      _present(source, "overlay_kernel_handle",
               "kernel_qkv_delta_sparse_overlay_", regex=False),
      _present(source, "overlay_kernel_create",
               'CreateNamedKernel("qkv_delta_sparse_overlay_f32")',
               regex=False),
      _present(source, "overlay_index_scratch",
               "qkv_delta_sparse_overlay_scratch_indices_", regex=False),
      _present(source, "overlay_copies_base_buffer",
               "clEnqueueCopyBuffer(queue_, base.buffer, out_buffer",
               regex=False),
      _present(source, "overlay_registers_resident_output",
               "ResidentF32Buffer{out_buffer, base.values, base.bytes}",
               regex=False),
      _present(opencl, "overlay_opencl_kernel",
               "__kernel void qkv_delta_sparse_overlay_f32", regex=False),
      _present(opencl, "overlay_opencl_selected_indices",
               "selected_indices", regex=False),
      _present(opencl, "overlay_opencl_copies_source_value",
               "output[index] = source[index]", regex=False),
  ]
  absent = [
      _absent(combined, "no_cpu_shadow_overlay_name",
              "CPU-shadow sparse overlay", regex=False),
  ]
  return {
      "device_sparse_overlay_source_present": _all_present(present),
      "no_cpu_shadow_overlay_marker": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _decode_markers(text: str) -> dict[str, Any]:
  present = [
      _present(text, "overlay_env",
               "IQ36_ROUTER_QKV_DELTA_DEVICE_SPARSE_OVERLAY_SOURCE",
               regex=False),
      _present(text, "overlay_contract",
               "DecodeRouterQkvDeltaDeviceSparseOverlaySourceContract",
               regex=False),
      _present(text, "overlay_ready_function",
               "DecodeRouterQkvDeltaDeviceSparseOverlaySourceReady",
               regex=False),
      _present(text, "overlay_product_owned_true",
               "source.product_owned_source = true", regex=False),
      _present(text, "overlay_cpu_shadow_free_true",
               "source.cpu_shadow_free = true", regex=False),
      _present(text, "overlay_host_sync_free_true",
               "source.host_sync_free = true", regex=False),
      _present(text, "overlay_selected_kernel_true",
               "source.selected_value_overlay_kernel_source = true",
               regex=False),
      _present(text, "overlay_helper",
               "DecodeRouterQkvDeltaSelectedValueOverlaySourceHandle",
               regex=False),
      _present(text, "overlay_runner_call",
               "RunRouterQkvDeltaSelectedValueOverlay", regex=False),
      _present(text, "overlay_source_only_cpp_guard",
               "IQ36_ROUTER_QKV_DELTA_DEVICE_SPARSE_OVERLAY_SOURCE is source-gate only; use generate-only",
               regex=False),
      _present(text, "overlay_source_only_python_guard",
               "IQ36_ROUTER_QKV_DELTA_DEVICE_SPARSE_OVERLAY_SOURCE is source-gate only; ",
               regex=False),
      _present(text, "overlay_stdout_ready",
               "router_qkv_delta_device_sparse_overlay_source_ready",
               regex=False),
      _present(text, "overlay_manifest",
               "router_qkv_delta_device_sparse_overlay_source", regex=False),
  ]
  absent = [
      _absent(text, "no_component_product_contract_helper",
              "DecodeRouterQkvDeltaComponentProductSourceContract",
              regex=False),
  ]
  return {
      "overlay_source_markers_present": _all_present(present),
      "component_not_promoted": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "producer_source_enabled": (
          result.get("router_qkv_delta_layer_input_producer_source") is True),
      "overlay_source_enabled": (
          result.get("router_qkv_delta_device_sparse_overlay_source") is True),
      "overlay_topk": (
          result.get("router_qkv_delta_device_sparse_overlay_topk") == TOPK),
      "overlay_consumer_layers": (
          result.get("router_qkv_delta_device_sparse_overlay_consumer_layers")
          == ALL_LINEAR_LAYERS),
      "overlay_producer_layers": (
          result.get("router_qkv_delta_device_sparse_overlay_producer_layers")
          == PRODUCER_LAYERS),
      "producer_root_values": (
          result.get("router_qkv_delta_layer_input_producer_root_values")
          == PRODUCER_VALUES),
      "speedup_claims_forbidden": (
          result.get("speedup_claims_allowed") is False),
      "no_smoke_json": not (generate_dir / "smoke.json").exists(),
  }


def _compile_source(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  compile_dir = out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  generated_cpp = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  runs = []
  commands = [
      [
          args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
          _rel(generated_cpp), "-o", _rel(compile_dir / "r2_gpu_decode_smoke.o"),
      ],
      [
          args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
          "engine/src/gpu_q4x8_matvec.cpp", "-o",
          _rel(compile_dir / "gpu_q4x8_matvec.o"),
      ],
  ]
  for index, command in enumerate(commands):
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=args.compile_timeout_s,
    )
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


def compute(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq306 = _load_json(args.seq306)
  decode_source = _read(args.decode_source)
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generated_cpp = _read(generated_cpp_path)
  generate_result = _load_json(args.generate_dir / "result.json")
  engine_header = _read(args.engine_header)
  engine_source = _read(args.engine_source)
  opencl_source = _read(args.opencl_source)
  source = _decode_markers(decode_source)
  generated = _decode_markers(generated_cpp)
  engine = _engine_markers(engine_header, engine_source, opencl_source)
  manifest_checks = _manifest_checks(generate_result, args.generate_dir)
  compile_result = _compile_source(args, out_dir)
  checks = [
      {
          "name": "seq306_selected_device_sparse_overlay_source_gate",
          "pass": (
              seq306.get("required_checks_passed") is True
              and seq306.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_device_sparse_overlay_source_gate"
              and seq306.get("qkv_delta_product_consumer_present") is False
              and _has_candidate(
                  routes, 306,
                  "reject_missing_qkv_delta_product_consumer_select_device_sparse_overlay_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_device_sparse_overlay_source_gate",
                  306)
          ),
      },
      {
          "name": "engine_has_resident_selected_value_sparse_overlay",
          "pass": (
              engine["device_sparse_overlay_source_present"] and
              engine["no_cpu_shadow_overlay_marker"]),
          "detail": engine,
      },
      {
          "name": "decode_source_has_default_off_overlay_source",
          "pass": (
              source["overlay_source_markers_present"] and
              source["component_not_promoted"]),
          "detail": source,
      },
      {
          "name": "generated_cpp_has_default_off_overlay_source",
          "pass": (
              generated["overlay_source_markers_present"] and
              generated["component_not_promoted"]),
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_overlay_source_not_token_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "local_source_compile_passed",
          "pass": compile_result["passed"],
          "detail": compile_result,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq306": _rel(args.seq306),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "generated_cpp": _rel(generated_cpp_path),
          "generated_cpp_sha256": _sha256(generated_cpp_path),
          "generate_only_result": _rel(args.generate_dir / "result.json"),
          "engine_header": _rel(args.engine_header),
          "engine_header_sha256": _sha256(args.engine_header),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "engine": engine,
      "source": source,
      "generated": generated,
      "generate_only_manifest": manifest_checks,
      "local_compile": compile_result,
      "consumer_requirement": {
          "all_linear_layers": ALL_LINEAR_LAYERS,
          "topk": TOPK,
          "decode_tokens": DECODE_TOKENS,
          "top512_values": TOP512_VALUES,
      },
      "producer_requirement": {
          "producer_layers": PRODUCER_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "producer_values": PRODUCER_VALUES,
      },
      "disposition": "accept_device_sparse_overlay_source",
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_device_sparse_overlay_target_compile_gate"
      ),
      "target_compile_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "next_route_reason": (
          "The resident F32 selected-value sparse overlay primitive and "
          "default-off decode source now exist and compile locally. The next "
          "unit is target compile/cache evidence for this generated source; "
          "decode probes, router distribution, and speed promotion remain "
          "blocked until target compile and later consumer/probe gates pass."
      ),
  }


def write_summary(metrics: dict[str, Any], out_dir: Path) -> None:
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  lines = [
      "# Router QKV Delta Device Sparse Overlay Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- target_compile_allowed: `{str(metrics['target_compile_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- top512 consumer values: `{TOP512_VALUES}`",
      f"- producer values: `{PRODUCER_VALUES}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source/generate-only evidence. It does not launch a token row or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq306", type=Path, default=DEFAULT_SEQ306)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_ENGINE_HEADER)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL_SOURCE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--cxx", default="c++")
  parser.add_argument("--compile-timeout-s", type=int, default=120)
  args = parser.parse_args()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  metrics = compute(args, out_dir)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  write_summary(metrics, out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
