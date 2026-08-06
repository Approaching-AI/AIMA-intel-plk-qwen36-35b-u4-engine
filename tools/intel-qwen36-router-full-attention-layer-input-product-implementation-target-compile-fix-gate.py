#!/usr/bin/env python3
"""Target-compile the fixed full-attention layer-input source."""

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
    "implementation-target-compile-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-router-full-attention-layer-input-product-"
    "implementation-target-compile-fix-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ339 = (
    ROOT
    / "output/router-full-attention-layer-input-product-implementation-"
    "source-fix-gate-20260708Tseq339Z"
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
    "target-compile-fix-gate-20260708Tseq340Z"
)
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
DECODE_TOKENS = 8
HIDDEN_SIZE = 2048
ROOT_VALUES = len(PRODUCER_LAYERS) * DECODE_TOKENS * HIDDEN_SIZE


def _load_base() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_layer_input_target_gate", BASE_GATE)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load base gate: {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


BASE = _load_base()


def _last_command(result: dict[str, Any], key: str) -> dict[str, Any]:
  value = result.get(key)
  return value if isinstance(value, dict) else {}


def _keep_prev_markers(text: str) -> dict[str, bool]:
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
  frontier = BASE._load_json(args.frontier)
  routes = BASE._load_json(args.routes)
  seq339 = BASE._load_json(args.seq339)
  result_path = args.generate_dir / "result.json"
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generate_result = BASE._load_json(result_path)
  generated_cpp = generated_cpp_path.read_text(encoding="utf-8")
  smoke = BASE._load_decode_smoke_module()
  cache_root = f"{args.remote_root}/cache"

  compile_result = BASE.iq36_local.ensure_cached_binary(
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
  manifest_checks = BASE._manifest_checks(generate_result, args.generate_dir)
  generated_checks = BASE._generated_checks(generated_cpp)
  keep_prev = _keep_prev_markers(generated_cpp)

  checks = [
      {
          "name": "seq339_selected_fix_target_compile_gate",
          "pass": (
              seq339.get("required_checks_passed") is True
              and seq339.get("selected_next_route")
              == "router_prompt_full_attention_layer_input_product_implementation_fix_target_compile_gate"
              and seq339.get("target_compile_allowed") is True
              and BASE._has_candidate(
                  routes, 339,
                  "accept_full_attention_layer_input_product_implementation_source_fix")
              and BASE._has_switch(
                  routes,
                  "select_router_prompt_full_attention_layer_input_product_implementation_fix_target_compile_gate",
                  339)
          ),
      },
      {
          "name": "generate_only_manifest_is_fixed_product_source_not_decode_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "generated_cpp_has_fixed_product_source_without_guard",
          "pass": all(generated_checks.values()) and all(keep_prev.values()),
          "detail": {
              "product_source": generated_checks,
              "keep_prev": keep_prev,
          },
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
          "frontier": BASE._rel(args.frontier),
          "routes": BASE._rel(args.routes),
          "seq339_source_fix_gate": BASE._rel(args.seq339),
          "generate_only_result": BASE._rel(result_path),
          "generated_cpp": BASE._rel(generated_cpp_path),
          "host": args.host,
          "env_script": args.env_script,
          "remote_root": args.remote_root,
      },
      "frontier": BASE._frontier_state(frontier),
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
      "keep_prev": keep_prev,
      "checks": checks,
      "required_checks_passed": required,
      "layer_input_source_probe_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_layer_input_product_implementation_fix_target_compile"
          if required else
          "reject_full_attention_layer_input_product_implementation_fix_target_compile"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_layer_input_product_implementation_fix_probe_gate"
          if required else
          "router_prompt_full_attention_layer_input_product_implementation_compile_fix_gate"
      ),
      "next_route_reason": (
          "The fixed product-owned full-attention layer-input source compiles "
          "on the target without launching a token row. Rerun the counter "
          "probe with this binary before decode or router distribution."
          if required else
          "The fixed product-owned layer-input source did not compile on "
          "target. Fix compile errors before any probe or decode row."
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
      "tool": BASE._rel(Path(__file__)),
      "selected_next_route": metrics["selected_next_route"],
      "speedup_claims_allowed": metrics["speedup_claims_allowed"],
      "inputs": metrics["inputs"],
      "compile_summary": metrics["compile_summary"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [row["name"] for row in metrics["checks"] if not bool(row.get("pass"))]
  summary = metrics["compile_summary"]
  lines = [
      "# Router Full-Attention Layer-Input Product Fix Target Compile Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- layer_input_source_probe_allowed: `{str(metrics['layer_input_source_probe_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- target compile ok: `{str(summary['ok']).lower()}`",
      f"- cache hit: `{str(summary['cache_hit']).lower()}`",
      f"- binary key: `{summary['key']}`",
      f"- binary: `{summary['binary']}`",
      f"- layer-input source values: `{ROOT_VALUES}`",
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
  parser.add_argument("--seq339", type=Path, default=DEFAULT_SEQ339)
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
      "compile_ok": metrics["compile_summary"]["ok"],
      "disposition": metrics["disposition"],
      "layer_input_source_probe_allowed": metrics[
          "layer_input_source_probe_allowed"],
      "out_dir": BASE._rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
