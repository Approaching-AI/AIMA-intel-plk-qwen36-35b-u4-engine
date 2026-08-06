#!/usr/bin/env python3
"""Target-compile the product-owned qkv-delta layer-input producer source."""

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
    "intel-qwen36-router-qkv-delta-layer-input-producer-implementation-"
    "target-compile-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ302 = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-implementation-source-gate-20260708Tseq302Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-implementation-generate-only-20260708Tseq302Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-implementation-target-compile-gate-20260708Tseq303Z"
)
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
ROOT_VALUES = 147456


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
      "component_source_disabled": (
          result.get("router_qkv_delta_component_source") is False),
      "producer_source": (
          result.get("router_qkv_delta_layer_input_producer_source") is True),
      "producer_layers": (
          result.get("router_qkv_delta_layer_input_producer_layers")
          == PRODUCER_LAYERS),
      "producer_root_values": (
          result.get("router_qkv_delta_layer_input_producer_root_values")
          == ROOT_VALUES),
      "speedup_claims_forbidden": (
          result.get("speedup_claims_allowed") is False),
      "no_smoke_json": not (generate_dir / "smoke.json").exists(),
  }


def _generated_checks(generated_cpp: str) -> dict[str, bool]:
  capture_call_count = len(re.findall(
      r"DecodeCaptureRouterQkvDeltaFullAttentionResidualSource\(\s*layer,\s*ffn_residual_handle_for_tail,\s*stats\)",
      generated_cpp,
      flags=re.S | re.M,
  ))
  return {
      "env_gate_present": (
          "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE" in generated_cpp),
      "product_contract_present": (
          "DecodeRouterQkvDeltaFullAttentionResidualProductSourceContract"
          in generated_cpp),
      "product_source_true": "source.product_owned_source = true" in generated_cpp,
      "cpu_shadow_free_true": "source.cpu_shadow_free = true" in generated_cpp,
      "host_sync_free_true": "source.host_sync_free = true" in generated_cpp,
      "resident_handle_source_true": (
          "source.resident_attention_residual_handle_source = true"
          in generated_cpp),
      "source_ready_function_present": (
          "DecodeRouterQkvDeltaFullAttentionResidualSourceReady"
          in generated_cpp),
      "capture_function_present": (
          "DecodeCaptureRouterQkvDeltaFullAttentionResidualSource"
          in generated_cpp),
      "captures_linear_full_core_and_full_core_handoff": capture_call_count >= 3,
      "stdout_ready_present": (
          "router_qkv_delta_layer_input_producer_source_ready"
          in generated_cpp),
      "source_only_guard_absent": (
          "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE is source-gate only"
          not in generated_cpp),
      "disabled_require_absent": (
          "Require(!args.router_qkv_delta_layer_input_producer_source"
          not in generated_cpp),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq302 = _load_json(args.seq302)
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
          "name": "seq302_selected_target_compile_gate",
          "pass": (
              seq302.get("required_checks_passed") is True
              and seq302.get("selected_next_route")
              == "router_prompt_full_attention_residual_layer_input_producer_implementation_target_compile_gate"
              and seq302.get("producer_product_source_present") is True
              and _has_candidate(
                  routes, 302,
                  "accept_full_attention_residual_layer_input_producer_implementation_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_residual_layer_input_producer_implementation_target_compile_gate",
                  302)
          ),
      },
      {
          "name": "generate_only_manifest_is_product_source_not_decode_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "generated_cpp_has_product_source_without_guard",
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
          "seq302_source_gate": _rel(args.seq302),
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
      "producer_probe_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_residual_layer_input_producer_implementation_target_compile"
          if required else
          "reject_full_attention_residual_layer_input_producer_implementation_target_compile"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_layer_input_producer_implementation_probe_gate"
          if required else
          "router_prompt_full_attention_residual_layer_input_producer_implementation_compile_fix_gate"
      ),
      "next_route_reason": (
          "The generated product-owned full-attention residual layer-input "
          "producer source compiles on the target without launching a token "
          "row. The next admissible unit is a runtime probe of the producer "
          "counters; decode and router distribution rows remain blocked."
          if required else
          "The generated product-owned producer source did not compile on "
          "target. Fix compile errors before any probe or decode row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [
      row["name"] for row in metrics["checks"]
      if not bool(row.get("pass"))
  ]
  summary = metrics["compile_summary"]
  lines = [
      "# Router QKV Delta Layer-Input Producer Implementation Target Compile Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- producer_probe_allowed: `{str(metrics['producer_probe_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- target compile ok: `{str(summary['ok']).lower()}`",
      f"- cache hit: `{str(summary['cache_hit']).lower()}`",
      f"- binary key: `{summary['key']}`",
      f"- binary: `{summary['binary']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is compile-only evidence. It does not launch a token row or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq302", type=Path, default=DEFAULT_SEQ302)
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
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "producer_probe_allowed": metrics["producer_probe_allowed"],
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
