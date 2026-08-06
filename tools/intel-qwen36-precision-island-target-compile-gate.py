#!/usr/bin/env python3
"""Compile the layer0/1 GPU final-norm precision island on the target."""

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


SOURCE_GATE = ROOT / "tools/intel-qwen36-precision-island-source-gate.py"
SMOKE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
SCHEMA_VERSION = "intel-qwen36-precision-island-target-compile-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_target_compile_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_one_token_probe_gate"
)


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


SOURCE = _load_module(SOURCE_GATE, "iq36_precision_island_source_gate")
SMOKE = _load_module(SMOKE_SOURCE, "iq36_precision_island_decode_smoke")


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _last_command(result: dict[str, Any], key: str) -> dict[str, Any]:
  value = result.get(key)
  return value if isinstance(value, dict) else {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  source_gate = _load(args.source_gate)
  manifest_path = args.generate_dir / "result.json"
  generated_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  manifest = _load(manifest_path)
  generated = generated_path.read_text(encoding="utf-8")
  generated_markers = SOURCE._source_markers(  # noqa: SLF001
      generated, include_wrapper=False)
  cache_root = f"{args.remote_root}/cache"
  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      cache_root,
      SMOKE.SOURCE_FILES,
      ROOT,
      generated_path,
      "tests/r2_gpu_decode_smoke.cpp",
      lambda remote_dir: SMOKE.build_command(remote_dir, args.env_script),
      "build/r2-gpu-decode-smoke",
      args.timeout_s,
  )
  build = _last_command(compile_result, "build")
  publish = _last_command(compile_result, "publish")
  source_selects = (
      source_gate.get("required_checks_passed") is True
      and source_gate.get("target_compile_allowed") is True
      and source_gate.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 543, CURRENT_ROUTE)
      and _has_switch(
          routes, 543,
          "select_router_prompt_distribution_layer0_1_gpu_final_norm_precision_island_target_compile_gate")
  )
  manifest_passes = (
      manifest.get("generate_only") is True
      and manifest.get("layer0_1_precision_island_source") is True
      and manifest.get("linear_final_cpu_shape_layers") == SOURCE.EXPECTED_LAYERS
      and manifest.get("speedup_claims_allowed") is False
  )
  checks = [
      {"name": "seq543_selected_target_compile_gate", "pass": source_selects},
      {"name": "generate_only_manifest_is_precision_island_source",
       "pass": manifest_passes},
      {
          "name": "generated_cpp_preserves_parameterized_selector",
          "pass": all(row["pass"] for row in generated_markers),
          "detail": generated_markers,
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
  required = all(bool(check["pass"]) for check in checks)
  compile_summary = {
      "ok": compile_result.get("ok"),
      "cache_hit": compile_result.get("hit"),
      "key": compile_result.get("key"),
      "binary": compile_result.get("binary"),
      "build_returncode": build.get("returncode"),
      "publish_returncode": publish.get("returncode"),
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "source_gate": _rel(args.source_gate),
          "generate_only_result": _rel(manifest_path),
          "generated_cpp": _rel(generated_path),
          "host": args.host,
          "env_script": args.env_script,
          "remote_root": args.remote_root,
      },
      "compile": compile_result,
      "compile_summary": compile_summary,
      "checks": checks,
      "required_checks_passed": required,
      "one_token_probe_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_layer0_1_gpu_final_norm_precision_island_target_compile"
          if required else
          "reject_layer0_1_gpu_final_norm_precision_island_target_compile"
      ),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The parameterized layer0/1 GPU final-norm selector compiles on Arc B390 "
          "without launching a token. Run one teacher-forced diagnostic token with "
          "the active layer set recorded before any multi-token distribution row."
          if required else
          "Fix the target compile before any token or distribution row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "compile_summary": metrics["compile_summary"],
          "selected_next_route": metrics["selected_next_route"],
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  summary = metrics["compile_summary"]
  lines = [
      f"# Seq{metrics['sequence']} Precision-Island Target Compile Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- compile_ok: `{str(summary['ok']).lower()}`",
      f"- cache_hit: `{str(summary['cache_hit']).lower()}`",
      f"- binary key: `{summary['key']}`",
      f"- one_token_probe_allowed: `{str(metrics['one_token_probe_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is target-compile evidence only. It is not token or speed evidence.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=544)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--source-gate", type=Path,
      default=ROOT / "output/seq543-layer0-1-gpu-final-norm-precision-island-source-gate-20260710Tseq543Z/metrics.json")
  parser.add_argument(
      "--generate-dir", type=Path,
      default=ROOT / "output/seq543-layer0-1-gpu-final-norm-precision-island-source-20260710Tseq543Z")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq544-layer0-1-gpu-final-norm-precision-island-target-compile-gate-20260710Tseq544Z")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default="local")
  parser.add_argument(
      "--env-script",
      default="/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-gpu")
  parser.add_argument("--timeout-s", type=int, default=7200)
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "compile_summary": metrics["compile_summary"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
