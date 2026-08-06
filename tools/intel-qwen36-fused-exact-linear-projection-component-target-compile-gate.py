#!/usr/bin/env python3
"""Compile the fused exact projection component binary on Arc B390."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


PROBE_SOURCE = ROOT / "tools/intel-qwen36-gpu-q4x8-output-projection-probe.py"
SCHEMA_VERSION = (
    "intel-qwen36-fused-exact-linear-projection-component-target-compile-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_component_target_compile_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_component_probe_gate"
)


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


PROBE = _load_module(PROBE_SOURCE, "iq36_fused_exact_projection_probe")


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


def _command(result: dict[str, Any], key: str) -> dict[str, Any]:
  value = result.get(key)
  return value if isinstance(value, dict) else {}


def _build_command(remote_dir: str, env_script: str) -> str:
  return " && ".join([
      f"source {shlex.quote(env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4_cpu_order_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_output_projection_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-output-projection-probe')}"
      ),
  ])


def _probe_markers(source: str) -> list[dict[str, Any]]:
  markers = {
      "probe_calls_locked_component_api": (
          "runner.RunRowblock16CpuOrderFinalize(" in source),
      "probe_records_candidate_timing": (
          "rowblock16_cpuorder_finalize_gpu_kernel_min_us" in source
          and "rowblock16_cpuorder_finalize_gpu_kernel_mean_us" in source),
      "probe_records_candidate_exactness": (
          "linear_attn_out_rowblock16_cpuorder_finalize" in source
          and "rowblock16_cpuorder_finalize_bit_exact_vs_cpu" in source),
      "probe_requires_bit_exact_component_before_success": (
          "rowblock16_cpuorder_finalize_gpu_vs_cpu.max_abs_diff == 0.0"
          in source
          and "rowblock16_cpuorder_finalize_gpu_vs_cpu.rmse == 0.0"
          in source),
  }
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  source_gate = _load(args.source_gate)
  probe_source = args.probe_source.read_text(encoding="utf-8")
  opencl_source = args.opencl_source.read_text(encoding="utf-8")
  generated_path = args.out_dir / "gpu_q4x8_output_projection_probe.cpp"
  generated_path.parent.mkdir(parents=True, exist_ok=True)
  generated_path.write_text(
      PROBE.PROBE_CPP.replace(
          "@@OPENCL_SOURCE_LITERAL@@",
          PROBE.cpp_raw_string_literal(opencl_source)),
      encoding="utf-8")
  generated_source = generated_path.read_text(encoding="utf-8")
  probe_markers = _probe_markers(probe_source)
  generated_markers = _probe_markers(generated_source)
  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      f"{args.remote_root}/cache",
      PROBE.SOURCE_FILES,
      ROOT,
      generated_path,
      "tests/gpu_q4x8_output_projection_probe.cpp",
      lambda remote_dir: _build_command(remote_dir, args.env_script),
      "build/iq36-gpu-q4x8-output-projection-probe",
      args.timeout_s,
  )
  build = _command(compile_result, "build")
  publish = _command(compile_result, "publish")
  compile_summary = {
      "ok": compile_result.get("ok"),
      "cache_hit": compile_result.get("hit"),
      "key": compile_result.get("key"),
      "binary": compile_result.get("binary"),
      "build_returncode": build.get("returncode"),
      "publish_returncode": publish.get("returncode"),
  }
  source_selects = (
      source_gate.get("required_checks_passed") is True
      and source_gate.get("component_source_passed") is True
      and source_gate.get("component_target_compile_allowed") is True
      and source_gate.get("component_probe_allowed") is False
      and source_gate.get("token_row_allowed") is False
      and source_gate.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 587, CURRENT_ROUTE)
      and _has_switch(
          routes, 587,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "component_target_compile_gate"))
  generated_contract = (
      PROBE.SCHEMA_VERSION
      == "intel-qwen36-gpu-q4x8-output-projection-probe-v3"
      and all(row["pass"] for row in probe_markers)
      and all(row["pass"] for row in generated_markers)
      and "__kernel void q4k_x8_matvec_rowblock16_cpuorder_finalize("
      in generated_source)
  no_execution_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("probe.json", "probe-result.json", "run.json", "tokens.jsonl"))
  checks = [
      {"name": "seq587_selected_target_compile_only_gate",
       "pass": source_selects},
      {"name": "generated_probe_contains_locked_kernel_call_and_fields",
       "pass": generated_contract,
       "detail": {
           "probe_source": probe_markers,
           "generated_cpp": generated_markers,
       }},
      {"name": "target_binary_compile_or_fresh_cache_publish_passed",
       "pass": compile_result.get("ok") is True,
       "detail": compile_summary},
      {"name": "target_compile_gate_did_not_execute_binary",
       "pass": no_execution_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "source_gate": _rel(args.source_gate),
          "probe_source": _rel(args.probe_source),
          "opencl_source": _rel(args.opencl_source),
          "generated_cpp": _rel(generated_path),
          "host": args.host,
          "env_script": args.env_script,
          "remote_root": args.remote_root,
      },
      "compile": compile_result,
      "compile_summary": compile_summary,
      "checks": checks,
      "required_checks_passed": required,
      "component_probe_allowed": required,
      "component_repeat_and_confirm_required": True,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_fused_exact_projection_component_target_compile"
          if required else "reject_fused_exact_projection_component_target_compile"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The captured-payload component binary target-compiles on Arc B390 "
          "without execution. Run exactly one paired representative-layer "
          "component repeat and confirm from this binary; require bit-exact "
          "GPU-vs-CPU output and candidate time <=198.195858994 us in both "
          "rows before any decode source or token."
          if required else
          "Repair the target binary, generated candidate markers, or route "
          "provenance before component execution."),
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
          "component_probe_allowed": metrics["component_probe_allowed"],
          "component_repeat_and_confirm_required": True,
          "decode_integration_allowed": False,
          "token_row_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  summary = metrics["compile_summary"]
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Projection Target Compile",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- compile_ok: `{str(summary['ok']).lower()}`",
      f"- cache_hit: `{str(summary['cache_hit']).lower()}`",
      f"- binary key: `{summary['key']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is target-compile evidence only. The binary was not executed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=588)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--source-gate", type=Path,
      default=ROOT / "output/seq587-fused-exact-linear-projection-component-source-gate-20260710Tseq587Z/metrics.json")
  parser.add_argument("--probe-source", type=Path, default=PROBE_SOURCE)
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq588-fused-exact-linear-projection-component-target-compile-gate-20260710Tseq588Z")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=PROBE.DEFAULT_HOST)
  parser.add_argument("--env-script", default=PROBE.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=PROBE.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=7200)
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "compile_summary": metrics["compile_summary"],
      "component_probe_allowed": metrics["component_probe_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
