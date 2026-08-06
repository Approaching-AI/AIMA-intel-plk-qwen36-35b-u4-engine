#!/usr/bin/env python3
"""Target-compile the captured Level Zero component harness."""

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


HARNESS_SOURCE = (
    ROOT / "tools/intel-qwen36-gpu-level-zero-postconv-recurrent-component-harness.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-gpu-level-zero-postconv-recurrent-component-target-compile-v0")
CURRENT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_component_target_compile_gate")
SELECTED_NEXT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_component_probe_gate")


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


HARNESS = _load_module(HARNESS_SOURCE, "iq36_level_zero_component_harness")


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


def _command(result: dict[str, Any], key: str) -> dict[str, Any]:
  value = result.get(key)
  return value if isinstance(value, dict) else {}


def _build_command(remote_dir: str, env_script: str) -> str:
  return " && ".join([
      f"source {shlex.quote(env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4_cpu_order_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_level_zero_postconv_recurrent.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/level_zero_component_probe.cpp')} "
          "-lze_loader -ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-level-zero-component-probe')}")
  ])


def _remote_identity(host: str, binary: str,
                     timeout_s: int) -> dict[str, Any]:
  result = iq36_local.run_target(
      host,
      " && ".join([
          f"sha256sum {shlex.quote(binary)}",
          f"ldd {shlex.quote(binary)}",
      ]),
      timeout_s)
  lines = str(result.get("stdout", "")).splitlines()
  digest = lines[0].split(maxsplit=1)[0] if lines else ""
  return {
      "returncode": result.get("returncode"),
      "sha256": digest if len(digest) == 64 else None,
      "dependencies": lines[1:] if len(lines) > 1 else [],
      "command": result.get("cmd"),
      "stderr": result.get("stderr"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  source_gate = _load(args.source_gate)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  module_sha = source_gate.get("native_module", {}).get("sha256")
  generated_path = args.out_dir / "level_zero_component_probe.cpp"
  generated = HARNESS.generate_cpp(
      args.opencl_source.read_text(encoding="utf-8"))
  generated += (
      f"\n// seq{args.source_sequence} native module SHA256: "
      + str(module_sha) + "\n")
  generated_path.write_text(generated, encoding="utf-8")
  local_compile = _run_local([
      args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
      "-Iengine/include", "-fsyntax-only", _rel(generated_path),
  ], compile_dir, "generated-harness-syntax")
  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      f"{args.remote_root}/cache",
      HARNESS.SOURCE_FILES,
      ROOT,
      generated_path,
      "tests/level_zero_component_probe.cpp",
      lambda remote_dir: _build_command(remote_dir, args.env_script),
      "build/iq36-level-zero-component-probe",
      args.timeout_s,
  )
  build = _command(compile_result, "build")
  publish = _command(compile_result, "publish")
  identity = (
      _remote_identity(
          args.host, str(compile_result.get("binary")), args.timeout_s)
      if compile_result.get("ok") is True else
      {"returncode": None, "sha256": None, "dependencies": []})
  dependencies = "\n".join(identity.get("dependencies", []))
  dependency_ok = (
      "libze_loader.so.1" in dependencies
      and "libOpenCL" not in dependencies
      and "llama" not in dependencies.lower()
      and "openvino" not in dependencies.lower())
  identity_ok = (
      identity.get("returncode") == 0
      and isinstance(identity.get("sha256"), str)
      and len(identity["sha256"]) == 64)
  generated_contract = all(marker in generated for marker in [
      "GpuLevelZeroPostconvRecurrentRunner level_zero(",
      "RunCurrentWallSamples(",
      "level_zero.Run(",
      "level_zero_input, args.samples)",
      "BitExact(q_compare)",
      "BitExact(k_compare)",
      "BitExact(v_compare)",
      "BitExact(attention_compare)",
      "BitExact(state_compare)",
      "BitExact(final_compare)",
      "candidate_added_min_us",
      "kAddedWallMaxUs = 6.841858993929781",
      str(module_sha),
  ])
  source_selects = (
      source_gate.get("required_checks_passed") is True
      and source_gate.get("component_source_passed") is True
      and source_gate.get("target_compile_allowed") is True
      and source_gate.get("component_probe_allowed") is False
      and source_gate.get("selected_next_route") == args.current_route
      and source_gate.get("disposition") == args.source_disposition
      and _has_candidate(routes, args.source_sequence, args.current_route)
      and _has_switch(
          routes, args.source_sequence, args.source_decision))
  no_execution_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl", "repeat.json"))
  checks = [
      {"name": f"seq{args.source_sequence}_selected_level_zero_target_compile_only",
       "pass": source_selects},
      {"name": "generated_harness_locks_module_exactness_and_budget_contract",
       "pass": generated_contract},
      {"name": "generated_harness_compiles_locally",
       "pass": local_compile["passed"], "detail": local_compile},
      {"name": "target_binary_compile_or_exact_cache_publish_passed",
       "pass": compile_result.get("ok") is True,
       "detail": {
           "ok": compile_result.get("ok"),
           "cache_hit": compile_result.get("hit"),
           "key": compile_result.get("key"),
           "binary": compile_result.get("binary"),
           "build_returncode": build.get("returncode"),
           "publish_returncode": publish.get("returncode"),
       }},
      {"name": "binary_identity_and_level_zero_only_dependency_recorded",
       "pass": identity_ok and dependency_ok, "detail": identity},
      {"name": "target_compile_gate_did_not_execute_binary_or_access_model",
       "pass": no_execution_evidence and "/home/intel/models" not in _build_command(
           "REMOTE", args.env_script)},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": args.schema_version,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "tool": args.tool_path,
      "inputs": {
          "routes": _rel(args.routes),
          "source_gate": _rel(args.source_gate),
          "harness_source": _rel(args.harness_source),
          "harness_source_sha256": _sha256(args.harness_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "generated_cpp": _rel(generated_path),
          "generated_cpp_sha256": _sha256(generated_path),
          "native_module_sha256": module_sha,
          "host": args.host,
          "env_script": args.env_script,
          "remote_root": args.remote_root,
      },
      "compile": compile_result,
      "binary_identity": identity,
      "checks": checks,
      "required_checks_passed": required,
      "target_compile_passed": required,
      "component_probe_allowed": required,
      "component_repeat_and_confirm_required": True,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          args.accept_disposition
          if required else
          args.repair_disposition),
      "selected_next_route": (
          args.selected_next_route if required else args.current_route),
      "next_route_reason": (
          "Run exactly one paired repeat/confirm from this source-keyed binary "
          f"and exact seq{args.source_sequence} native module. Require six bit-exact boundaries "
          "and added paired host wall <=6.841858993929781 us in both rows."
          if required else
          "Repair generated harness, target compile, binary identity, or "
          "dependency closure before any kernel execution."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": metrics["tool"],
          "inputs": metrics["inputs"],
          "binary_identity": metrics["binary_identity"],
          "target_compile_passed": metrics["target_compile_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "component_repeat_and_confirm_required": True,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Level Zero Component Target Compile",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- cache key: `{metrics['compile'].get('key')}`",
      f"- binary SHA256: `{metrics['binary_identity'].get('sha256')}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "The target binary was compiled and identified only. It was not executed and did not access the model.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=616)
  parser.add_argument("--source-sequence", type=int, default=615)
  parser.add_argument("--schema-version", default=SCHEMA_VERSION)
  parser.add_argument("--current-route", default=CURRENT_ROUTE)
  parser.add_argument("--selected-next-route", default=SELECTED_NEXT_ROUTE)
  parser.add_argument(
      "--source-decision",
      default="select_gpu_level_zero_postconv_recurrent_component_target_compile_gate")
  parser.add_argument(
      "--source-disposition",
      default="accept_level_zero_ocloc_fused_postconv_recurrent_v1_source")
  parser.add_argument(
      "--accept-disposition",
      default="accept_level_zero_ocloc_fused_postconv_recurrent_v1_target_compile")
  parser.add_argument(
      "--repair-disposition",
      default="repair_level_zero_ocloc_fused_postconv_recurrent_v1_target_compile")
  parser.add_argument(
      "--tool-path",
      default=_rel(Path(__file__)))
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--source-gate", type=Path,
      default=ROOT / (
          "output/seq615-gpu-level-zero-postconv-recurrent-component-source-"
          "gate-20260710Tseq615Z/metrics.json"))
  parser.add_argument("--harness-source", type=Path,
                      default=HARNESS_SOURCE)
  parser.add_argument("--opencl-source", type=Path,
                      default=HARNESS.OPENCL_SOURCE)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=HARNESS.DEFAULT_HOST)
  parser.add_argument("--env-script", default=HARNESS.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=HARNESS.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--cxx", default="clang++")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq616-gpu-level-zero-postconv-recurrent-component-target-"
          "compile-gate-20260710Tseq616Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "target_compile_passed": metrics["target_compile_passed"],
      "component_probe_allowed": metrics["component_probe_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
