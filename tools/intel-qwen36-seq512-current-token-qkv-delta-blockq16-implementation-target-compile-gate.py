#!/usr/bin/env python3
"""Target-compile the current-token qkv-delta block-q16 implementation."""

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
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SOURCE_GATE = (
    ROOT
    / "tools/intel-qwen36-seq511-current-token-qkv-delta-blockq16-implementation-source-gate.py"
)
SMOKE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
SCHEMA_VERSION = (
    "intel-qwen36-seq512-current-token-qkv-delta-blockq16-"
    "implementation-target-compile-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ511 = (
    ROOT
    / "output/seq511-current-token-qkv-delta-blockq16-implementation-source-gate-20260709Tseq511Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/seq511-current-token-qkv-delta-blockq16-implementation-generate-only-20260709Tseq511Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq512-current-token-qkv-delta-blockq16-implementation-target-compile-gate-20260709Tseq512Z"
)
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


SOURCE = _load_module(SOURCE_GATE, "iq36_blockq16_impl_source_gate")
SMOKE = _load_module(SMOKE_SOURCE, "iq36_decode_smoke")


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


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
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


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


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq511 = _load_json(args.seq511)
  result_path = args.generate_dir / "result.json"
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generate_result = _load_json(result_path)
  generated_cpp = _read(generated_cpp_path)
  engine_header = _read(ENGINE_HEADER)
  engine_source = _read(ENGINE_SOURCE)
  opencl_source = _read(OPENCL_SOURCE)
  cache_root = f"{args.remote_root}/cache"

  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      cache_root,
      SMOKE.SOURCE_FILES,
      ROOT,
      generated_cpp_path,
      "tests/r2_gpu_decode_smoke.cpp",
      lambda remote_dir: SMOKE.build_command(remote_dir, args.env_script),
      "build/r2-gpu-decode-smoke",
      args.timeout_s,
  )
  build = _last_command(compile_result, "build")
  publish = _last_command(compile_result, "publish")
  manifest_checks = SOURCE._manifest_checks(generate_result, args.generate_dir)
  generated_checks = SOURCE._decode_markers(generated_cpp, include_python=False)
  engine_checks = SOURCE._engine_markers(
      engine_header, engine_source, opencl_source)

  checks = [
      {
          "name": "seq511_selected_implementation_target_compile_gate",
          "pass": (
              seq511.get("required_checks_passed") is True
              and seq511.get("selected_next_route")
              == "router_prompt_all_linear_current_token_qkv_delta_blockq16_implementation_target_compile_gate"
              and seq511.get("target_compile_allowed") is True
              and _has_candidate(
                  routes, 511,
                  "accept_current_token_qkv_delta_blockq16_implementation_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_current_token_qkv_delta_blockq16_implementation_target_compile_gate",
                  511)
          ),
      },
      {
          "name": "generate_only_manifest_is_blockq16_implementation",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "generated_cpp_has_blockq16_implementation",
          "pass": generated_checks["present"] and generated_checks["absent"],
          "detail": generated_checks,
      },
      {
          "name": "engine_has_blockq16_overlay_runner",
          "pass": engine_checks["present"] and engine_checks["absent"],
          "detail": engine_checks,
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
          "seq511_source_gate": _rel(args.seq511),
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
      "engine": engine_checks,
      "checks": checks,
      "required_checks_passed": required,
      "blockq16_probe_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "blockq16_implementation": {
          "all_linear_layers": SOURCE.ALL_LINEAR_LAYERS,
          "decode_tokens": SOURCE.DECODE_TOKENS,
          "topk": SOURCE.TOPK,
          "expected_values": SOURCE.EXPECTED_VALUES,
          "selector": "linear_qkv_col_abs",
          "value_mode": "shadow_delta_block_q16",
          "overlay": "resident_f32_additive_sparse_blockq16_delta",
      },
      "disposition": (
          "accept_current_token_qkv_delta_blockq16_implementation_target_compile"
          if required else
          "reject_current_token_qkv_delta_blockq16_implementation_target_compile"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_implementation_probe_gate"
          if required else
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_implementation_compile_fix_gate"
      ),
      "next_route_reason": (
          "The generated product-wired block-q16 source compiles on target "
          "without launching a token row. The next admissible unit is a "
          "one-token runtime probe of block-q16 counters; decode rows and "
          "router distribution rows remain blocked."
          if required else
          "The product-wired block-q16 source did not compile on target. Fix "
          "compile errors before any probe or decode row."
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
      "# Seq512 Current-Token QKV-Delta Block-Q16 Implementation Target Compile Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- blockq16_probe_allowed: `{str(metrics['blockq16_probe_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- binary: `{metrics['compile_summary'].get('binary')}`",
      f"- compile key: `{metrics['compile_summary'].get('key')}`",
      f"- expected values: `{metrics['blockq16_implementation']['expected_values']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is target-compile evidence. It does not launch a token row or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq511", type=Path, default=DEFAULT_SEQ511)
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
      "blockq16_probe_allowed": metrics["blockq16_probe_allowed"],
      "compile_ok": metrics["compile_summary"]["ok"],
      "compile_key": metrics["compile_summary"]["key"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
