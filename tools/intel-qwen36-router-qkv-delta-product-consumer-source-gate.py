#!/usr/bin/env python3
"""Gate all-linear qkv-delta product-consumer source wiring."""

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
    "intel-qwen36-router-qkv-delta-product-consumer-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ309 = (
    ROOT
    / "output/router-qkv-delta-device-sparse-overlay-probe-gate-20260708Tseq309Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-qkv-delta-product-consumer-generate-only-20260708Tseq310Z"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-product-consumer-source-gate-20260708Tseq310Z"
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


def _source_markers(text: str) -> dict[str, Any]:
  present = [
      _present(text, "consumer_env",
               "IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE", regex=False),
      _present(text, "consumer_contract",
               "DecodeRouterQkvDeltaProductConsumerSourceContract",
               regex=False),
      _present(text, "consumer_ready",
               "DecodeRouterQkvDeltaProductConsumerSourceReady", regex=False),
      _present(text, "product_owned_true",
               "source.product_owned_source = true", regex=False),
      _present(text, "cpu_shadow_free_true",
               "source.cpu_shadow_free = true", regex=False),
      _present(text, "host_sync_free_true",
               "source.host_sync_free = true", regex=False),
      _present(text, "producer_handle_input_true",
               "source.resident_producer_handle_input = true", regex=False),
      _present(text, "overlay_kernel_source_true",
               "source.selected_value_overlay_kernel_source = true",
               regex=False),
      _present(text, "all_linear_consumer_true",
               "source.all_linear_consumer_source = true", regex=False),
      _present(text, "live_qkv_selector_true",
               "source.live_qkv_weighted_selector_source = true", regex=False),
      _present(text, "consumer_helper",
               "DecodeRouterQkvDeltaProductConsumerSourceHandle",
               regex=False),
      _present(text, "selected_indices_helper",
               "DecodeRouterQkvDeltaProductSelectedIndices", regex=False),
      _present(text, "producer_mapping_helper",
               "DecodeRouterQkvDeltaProductProducerLayerForConsumer",
               regex=False),
      _present(text, "overlay_helper_used",
               "DecodeRouterQkvDeltaSelectedValueOverlaySourceHandle",
               regex=False),
      _present(text, "producer_handle_vector_used",
               "g_decode_router_qkv_delta_full_attention_residual_source_handles",
               regex=False),
      _present(text, "consumer_stdout_ready",
               "router_qkv_delta_product_consumer_source_ready", regex=False),
      _present(text, "consumer_cpu_shadow_guard",
               "IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE is incompatible with CPU-shadow values",
               regex=False),
  ]
  absent = [
      _absent(text, "no_consumer_source_only_guard",
              "IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE is source-gate only",
              regex=False),
  ]
  return {
      "consumer_source_markers_present": _all_present(present),
      "consumer_not_source_only_guarded": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def _manifest_checks(result: dict[str, Any], generate_dir: Path) -> dict[str, bool]:
  return {
      "generate_only": result.get("generate_only") is True,
      "producer_source_enabled": (
          result.get("router_qkv_delta_layer_input_producer_source") is True),
      "overlay_source_disabled": (
          result.get("router_qkv_delta_device_sparse_overlay_source") is False),
      "product_consumer_source_enabled": (
          result.get("router_qkv_delta_product_consumer_source") is True),
      "product_consumer_topk": (
          result.get("router_qkv_delta_product_consumer_topk") == TOPK),
      "product_consumer_layers": (
          result.get("router_qkv_delta_product_consumer_layers")
          == ALL_LINEAR_LAYERS),
      "product_consumer_values": (
          result.get("router_qkv_delta_product_consumer_values")
          == TOP512_VALUES),
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
  seq309 = _load_json(args.seq309)
  source_text = _read(args.decode_source)
  generated_cpp = _read(args.generate_dir / "r2_gpu_decode_smoke.cpp")
  result = _load_json(args.generate_dir / "result.json")
  source = _source_markers(source_text)
  generated = _source_markers(generated_cpp)
  manifest_checks = _manifest_checks(result, args.generate_dir)
  compile_run = _compile_source(args, args.out_dir)

  checks = [
      {
          "name": "seq309_selected_product_consumer_source_gate",
          "pass": (
              seq309.get("required_checks_passed") is True
              and seq309.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_product_consumer_source_gate"
              and seq309.get("product_consumer_source_allowed") is True
              and _has_candidate(
                  routes, 309,
                  "accept_source_only_overlay_probe_select_product_consumer_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_product_consumer_source_gate",
                  309)
          ),
      },
      {
          "name": "decode_source_wires_product_consumer",
          "pass": (
              source["consumer_source_markers_present"]
              and source["consumer_not_source_only_guarded"]),
          "detail": source,
      },
      {
          "name": "generated_source_wires_product_consumer",
          "pass": (
              generated["consumer_source_markers_present"]
              and generated["consumer_not_source_only_guarded"]),
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_records_product_consumer_shape",
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
          "seq309_overlay_probe": _rel(args.seq309),
          "generate_dir": _rel(args.generate_dir),
          "decode_source": _rel(args.decode_source),
      },
      "source_sha256": _sha256(args.decode_source),
      "generated_cpp_sha256": _sha256(args.generate_dir / "r2_gpu_decode_smoke.cpp"),
      "result_sha256": _sha256(args.generate_dir / "result.json"),
      "consumer_requirement": {
          "all_linear_layers": ALL_LINEAR_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "topk": TOPK,
          "top512_values": TOP512_VALUES,
      },
      "producer_requirement": {
          "producer_layers": PRODUCER_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "producer_values": PRODUCER_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "qkv_delta_product_consumer_present": required,
      "target_compile_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_qkv_delta_product_consumer_source"
          if required else
          "block_before_qkv_delta_product_consumer_target_compile"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_product_consumer_target_compile_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_product_consumer_source_fix_gate"
      ),
      "next_route_reason": (
          "The all-linear qkv-delta product consumer source now consumes "
          "resident producer handles through the selected-value sparse-overlay "
          "primitive and records the top512 shape in generate-only form. Target "
          "compile is the next admissible gate before any token probe, router "
          "distribution row, or speed promotion."
          if required else
          "The product consumer source is incomplete; fix source/generate-only "
          "evidence before target compile or token rows."
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
      "# Router QKV Delta Product Consumer Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- qkv_delta_product_consumer_present: `{str(metrics['qkv_delta_product_consumer_present']).lower()}`",
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


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq309", type=Path, default=DEFAULT_SEQ309)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
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
