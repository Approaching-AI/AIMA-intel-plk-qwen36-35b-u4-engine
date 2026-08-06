#!/usr/bin/env python3
"""Target-compile the captured exact-preprojection component probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


PROBE_SOURCE = (
    ROOT / "tools/intel-qwen36-all-linear-preprojection-parity-component-harness.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-component-target-compile-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_target_compile_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_probe_gate"
)
REPAIR_CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_repair_target_compile_gate"
)
REPAIR_SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_final_probe_gate"
)


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


PROBE = _load_module(PROBE_SOURCE, "iq36_exact_preprojection_component_probe")


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


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


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
          f"{shlex.quote(remote_dir + '/tests/exact_preprojection_component_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-exact-preprojection-component-probe')}"
      ),
  ])


def _run_local(command: list[str], out_dir: Path,
               stem: str) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  stdout_path = out_dir / f"{stem}.stdout.txt"
  stderr_path = out_dir / f"{stem}.stderr.txt"
  stdout_path.write_text(completed.stdout, encoding="utf-8")
  stderr_path.write_text(completed.stderr, encoding="utf-8")
  return {
      "passed": completed.returncode == 0,
      "command": command,
      "returncode": completed.returncode,
      "stdout": _rel(stdout_path),
      "stderr": _rel(stderr_path),
  }


def _probe_markers(source: str) -> list[dict[str, Any]]:
  markers = {
      "current_and_exact_q6_conv_apis_share_q8_and_weight_handles": all(
          marker in source for marker in [
              "RunResidentRawQ6KThenResidentConvState(",
              "RunResidentRawQ6KThenResidentConvStateCpuOrder(",
              "q6_handle, q8, conv_weights_handle",
          ]),
      "conv_states_are_cloned_from_one_seed": all(
          marker in source for marker in [
              "current_conv_state_handle =",
              "exact_conv_state_handle =",
              "CloneResidentF32Buffer(conv_state_seed_handle)",
          ]),
      "current_and_exact_postconv_apis_share_inputs": all(
          marker in source for marker in [
              "RunPostConvPrepThenLinearAttentionDeltaResidentState(",
              "RunPostConvPrepThenLinearAttentionDeltaResidentStateCpuOrder(",
              "conv_output_seed_handle, current_recurrent_state_handle",
              "conv_output_seed_handle, exact_recurrent_state_handle",
              "gate, beta, z, norm_weight",
          ]),
      "recurrent_states_are_cloned_from_one_seed": all(
          marker in source for marker in [
              "current_recurrent_state_handle =",
              "exact_recurrent_state_handle =",
              "CloneResidentF32Buffer(recurrent_state_seed_handle)",
          ]),
      "projection_paths_share_one_quantized_input": all(
          marker in source for marker in [
              "runner.RunRowblock16(",
              "runner.RunRowblock16CpuOrderFinalize(",
              "projection_packed, projection_q8.qs, projection_q8.bsums",
          ]),
      "probe_requires_all_exact_component_boundaries_bit_exact": all(
          marker in source for marker in [
              "BitExact(exact_qkv_vs_cpu)",
              "BitExact(exact_conv_output_vs_cpu)",
              "BitExact(exact_conv_state_vs_cpu)",
              "BitExact(exact_attention_vs_cpu)",
              "BitExact(exact_final_vs_cpu)",
              "BitExact(exact_recurrent_state_vs_cpu)",
              "BitExact(exact_projection_vs_cpu)",
          ]),
      "stateful_timing_uses_fresh_clones_and_single_step_calls": (
          "for (int i = 0; i < args.samples; ++i)" in source
          and "kConvKernelSize, 1, true" in source
          and "norm_epsilon, 1, true" in source),
      "probe_records_whole_changed_shell_distribution": all(
          marker in source for marker in [
              "current_changed_shell_min_us",
              "exact_changed_shell_min_us",
              "candidate_added_min_us",
              "candidate_added_mean_us",
          ]),
  }
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def _remote_binary_identity(host: str, binary: str,
                            timeout_s: int) -> dict[str, Any]:
  result = iq36_local.run_target(
      host, f"sha256sum {shlex.quote(binary)}", timeout_s)
  stdout = str(result.get("stdout", "")).strip()
  digest = stdout.split(maxsplit=1)[0] if stdout else ""
  return {
      "returncode": result.get("returncode"),
      "sha256": digest if len(digest) == 64 else None,
      "command": result.get("cmd"),
      "stderr": result.get("stderr"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  args.out_dir.mkdir(parents=True, exist_ok=True)
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  routes = _load(args.routes)
  source_gate = _load(args.source_gate)
  probe_source = args.probe_source.read_text(encoding="utf-8")
  opencl_source = args.opencl_source.read_text(encoding="utf-8")
  generated_path = args.out_dir / "exact_preprojection_component_probe.cpp"
  generated_path.write_text(
      PROBE.generate_cpp(opencl_source), encoding="utf-8")
  generated_source = generated_path.read_text(encoding="utf-8")
  probe_markers = _probe_markers(probe_source)
  generated_markers = _probe_markers(generated_source)
  local_compile = _run_local([
      args.cxx, "-std=c++17", "-Iengine/include", "-fsyntax-only",
      _rel(generated_path),
  ], compile_dir, "generated-probe-syntax")

  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      f"{args.remote_root}/cache",
      PROBE.SOURCE_FILES,
      ROOT,
      generated_path,
      "tests/exact_preprojection_component_probe.cpp",
      lambda remote_dir: _build_command(remote_dir, args.env_script),
      "build/iq36-exact-preprojection-component-probe",
      args.timeout_s,
  )
  build = _command(compile_result, "build")
  publish = _command(compile_result, "publish")
  identity = (
      _remote_binary_identity(
          args.host, str(compile_result.get("binary")), args.timeout_s)
      if compile_result.get("ok") is True else
      {"returncode": None, "sha256": None})
  compile_summary = {
      "ok": compile_result.get("ok"),
      "cache_hit": compile_result.get("hit"),
      "key": compile_result.get("key"),
      "binary": compile_result.get("binary"),
      "binary_sha256": identity.get("sha256"),
      "build_returncode": build.get("returncode"),
      "publish_returncode": publish.get("returncode"),
  }

  current_route = REPAIR_CURRENT_ROUTE if args.repair else CURRENT_ROUTE
  selected_next_route = (
      REPAIR_SELECTED_NEXT_ROUTE if args.repair else SELECTED_NEXT_ROUTE)
  source_selects = (
      source_gate.get("required_checks_passed") is True
      and source_gate.get(
          "source_repair_passed" if args.repair else "component_source_passed")
      is True
      and source_gate.get(
          "repair_target_compile_allowed"
          if args.repair else "component_target_compile_allowed") is True
      and source_gate.get("component_probe_allowed") is False
      and source_gate.get("token_row_allowed") is False
      and source_gate.get("selected_next_route") == current_route
      and _has_candidate(routes, 602 if args.repair else 598, current_route)
      and _has_switch(
          routes, 602 if args.repair else 598,
          "select_router_prompt_distribution_all_linear_preprojection_"
          + ("parity_component_repair_target_compile_gate"
             if args.repair else "parity_component_target_compile_gate")))
  generated_contract = (
      PROBE.SCHEMA_VERSION
      == "intel-qwen36-all-linear-preprojection-parity-component-probe-v0"
      and all(row["pass"] for row in probe_markers)
      and all(row["pass"] for row in generated_markers)
      and "__kernel void q6k_linear_qkv_cpuorder_nofma(" in generated_source
      and "__kernel void linear_attn_delta_recurrent_final_cpuorder_nofma_f32("
      in generated_source)
  no_execution_evidence = not any(
      (args.out_dir / name).exists()
      for name in (
          "probe.json", "probe-result.json", "run.json", "tokens.jsonl",
          "repeat.json", "confirm.json"))
  identity_ok = (
      identity.get("returncode") == 0
      and isinstance(identity.get("sha256"), str)
      and len(identity["sha256"]) == 64)
  checks = [
      {"name": (
           "seq602_selected_repair_target_compile_only_gate"
           if args.repair else "seq598_selected_target_compile_only_gate"),
       "pass": source_selects},
      {"name": "generated_probe_binds_identical_inputs_and_fresh_state_clones",
       "pass": generated_contract,
       "detail": {
           "probe_source": probe_markers,
           "generated_cpp": generated_markers,
       }},
      {"name": "generated_probe_compiles_locally",
       "pass": local_compile["passed"], "detail": local_compile},
      {"name": "target_binary_compile_or_exact_cache_publish_passed",
       "pass": compile_result.get("ok") is True,
       "detail": compile_summary},
      {"name": "target_binary_identity_recorded",
       "pass": identity_ok, "detail": identity},
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
          "probe_source_sha256": _sha256(args.probe_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "generated_cpp": _rel(generated_path),
          "generated_cpp_sha256": _sha256(generated_path),
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
          ("accept_repaired_cpuorder_preprojection_bundle_v1_target_compile"
           if args.repair else
           "accept_cpuorder_preprojection_bundle_v1_target_compile")
          if required else
          "repair_exact_preprojection_component_target_compile"),
      "selected_next_route": (
          selected_next_route if required else current_route),
      "next_route_reason": (
          "The captured-layer component binary target-compiles on Arc B390 "
          "without execution and has a source-keyed binary identity. Run "
          + ("the one final paired repeat and confirm from this repaired binary. "
             if args.repair else
             "exactly one paired repeat and confirm from this binary. ")
          + "Require "
          "bit-exact QKV, convolution output/state, attention/final output, "
          "recurrent state, and projection, plus added changed-shell wall no "
          "greater than 6.841858993929781 us in both rows."
          if required else
          "Repair probe binding, local syntax, target compilation, or binary "
          "identity before component execution."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
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
      f"# Seq{metrics['sequence']} Exact Preprojection Target Compile",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- compile_ok: `{str(summary['ok']).lower()}`",
      f"- cache_hit: `{str(summary['cache_hit']).lower()}`",
      f"- binary key: `{summary['key']}`",
      f"- binary SHA256: `{summary['binary_sha256']}`",
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
  parser.add_argument("--sequence", type=int, default=599)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--source-gate", type=Path,
      default=ROOT / (
          "output/seq598-all-linear-preprojection-parity-component-source-"
          "gate-20260710Tseq598Z/metrics.json"))
  parser.add_argument("--probe-source", type=Path, default=PROBE_SOURCE)
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq599-all-linear-preprojection-parity-component-target-"
          "compile-gate-20260710Tseq599Z"))
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=PROBE.DEFAULT_HOST)
  parser.add_argument("--env-script", default=PROBE.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=PROBE.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--cxx", default="clang++")
  parser.add_argument("--repair", action="store_true")
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
