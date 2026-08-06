#!/usr/bin/env python3
"""Target-compile the router qkv-delta product-consumer source."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = (
    "intel-qwen36-router-qkv-delta-product-consumer-target-compile-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ310 = (
    ROOT
    / "output/router-qkv-delta-product-consumer-source-gate-20260708Tseq310Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-qkv-delta-product-consumer-generate-only-20260708Tseq310Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-product-consumer-target-compile-gate-20260708Tseq311Z"
)
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
TOP512_VALUES = len(ALL_LINEAR_LAYERS) * 8 * 512
PRODUCER_VALUES = len(PRODUCER_LAYERS) * 8 * 2048


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _load_decode_smoke_module() -> Any:
  path = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
  spec = importlib.util.spec_from_file_location("iq36_decode_smoke", path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load decode smoke module: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


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


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _last_command(result: dict[str, Any], key: str) -> dict[str, Any]:
  value = result.get(key)
  return value if isinstance(value, dict) else {}


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
          result.get("router_qkv_delta_product_consumer_topk") == 512),
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


def _generated_checks(generated_cpp: str) -> dict[str, bool]:
  capture_call_count = len(re.findall(
      r"DecodeCaptureRouterQkvDeltaFullAttentionResidualSource\(\s*layer,",
      generated_cpp,
      flags=re.S | re.M,
  ))
  return {
      "consumer_env_gate_present": (
          "IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE" in generated_cpp),
      "consumer_contract_present": (
          "DecodeRouterQkvDeltaProductConsumerSourceContract" in generated_cpp),
      "consumer_helper_present": (
          "DecodeRouterQkvDeltaProductConsumerSourceHandle" in generated_cpp),
      "consumer_selected_indices_present": (
          "DecodeRouterQkvDeltaProductSelectedIndices" in generated_cpp),
      "overlay_helper_present": (
          "DecodeRouterQkvDeltaSelectedValueOverlaySourceHandle"
          in generated_cpp),
      "overlay_kernel_source_embedded": (
          "qkv_delta_sparse_overlay_f32" in generated_cpp),
      "consumer_stdout_ready_present": (
          "router_qkv_delta_product_consumer_source_ready" in generated_cpp),
      "producer_capture_still_present": capture_call_count >= 3,
      "consumer_not_source_only_guarded": (
          "IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE is source-gate only"
          not in generated_cpp),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq310 = _load_json(args.seq310)
  result_path = args.generate_dir / "result.json"
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generate_result = _load_json(result_path)
  generated_cpp = generated_cpp_path.read_text(encoding="utf-8")
  smoke = _load_decode_smoke_module()
  cache_root = f"{args.remote_root}/cache"

  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      cache_root,
      smoke.SOURCE_FILES,
      ROOT,
      generated_cpp_path,
      "tests/r2_gpu_decode_smoke.cpp",
      lambda remote_dir: smoke.build_command(remote_dir, args.env_script),
      "build/r2-gpu-decode-smoke",
      args.timeout_s,
  )
  build = _last_command(compile_result, "build")
  publish = _last_command(compile_result, "publish")
  manifest_checks = _manifest_checks(generate_result, args.generate_dir)
  generated_checks = _generated_checks(generated_cpp)

  checks = [
      {
          "name": "seq310_selected_product_consumer_target_compile_gate",
          "pass": (
              seq310.get("required_checks_passed") is True
              and seq310.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_product_consumer_target_compile_gate"
              and seq310.get("target_compile_allowed") is True
              and _has_candidate(
                  routes, 310,
                  "accept_qkv_delta_product_consumer_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_product_consumer_target_compile_gate",
                  310)
          ),
      },
      {
          "name": "generate_only_manifest_is_product_consumer_source",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "generated_cpp_has_product_consumer_source",
          "pass": all(generated_checks.values()),
          "detail": generated_checks,
      },
      {
          "name": "target_binary_compile_or_cache_hit_passed",
          "pass": compile_result.get("ok") is True,
          "detail": {
              "cache_hit": compile_result.get("hit"),
              "key": compile_result.get("key"),
              "binary": compile_result.get("binary"),
              "build_returncode": build.get("returncode"),
              "publish_returncode": publish.get("returncode"),
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "seq310_source_gate": _rel(args.seq310),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(generated_cpp_path),
          "host": args.host,
          "env_script": args.env_script,
          "remote_root": args.remote_root,
      },
      "frontier": _frontier_state(frontier),
      "compile": compile_result,
      "compile_summary": {
          "ok": compile_result.get("ok"),
          "cache_hit": compile_result.get("hit"),
          "key": compile_result.get("key"),
          "binary": compile_result.get("binary"),
          "build_returncode": build.get("returncode"),
          "publish_returncode": publish.get("returncode"),
      },
      "generated": generated_checks,
      "checks": checks,
      "required_checks_passed": required,
      "product_consumer_probe_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "consumer_requirement": {
          "all_linear_layers": ALL_LINEAR_LAYERS,
          "decode_tokens": 8,
          "topk": 512,
          "top512_values": TOP512_VALUES,
      },
      "producer_requirement": {
          "producer_layers": PRODUCER_LAYERS,
          "decode_tokens": 8,
          "producer_values": PRODUCER_VALUES,
      },
      "disposition": (
          "accept_qkv_delta_product_consumer_target_compile"
          if required else
          "reject_qkv_delta_product_consumer_target_compile"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_product_consumer_probe_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_product_consumer_compile_fix_gate"
      ),
      "next_route_reason": (
          "The generated product-consumer source compiles on the target without "
          "launching a token row. The next admissible unit is a guarded product "
          "consumer probe; decode rows and router distribution rows remain "
          "blocked until the probe and later decode gates pass."
          if required else
          "The generated product-consumer source did not compile on target. Fix "
          "compile errors before any probe or decode row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": metrics["schema_version"],
      "workstream": metrics["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": metrics["speedup_claims_allowed"],
      "inputs": metrics["inputs"],
      "compile_summary": metrics["compile_summary"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  summary = metrics["compile_summary"]
  lines = [
      "# Router QKV Delta Product Consumer Target Compile Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- compile_ok: `{str(summary['ok']).lower()}`",
      f"- cache_hit: `{str(summary['cache_hit']).lower()}`",
      f"- binary key: `{summary['key']}`",
      f"- binary: `{summary['binary']}`",
      f"- product_consumer_probe_allowed: `{str(metrics['product_consumer_probe_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- top512 consumer values: `{TOP512_VALUES}`",
      f"- producer values: `{PRODUCER_VALUES}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is target-compile evidence only. It does not launch a token row or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq310", type=Path, default=DEFAULT_SEQ310)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "compile_ok": metrics["compile_summary"]["ok"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
