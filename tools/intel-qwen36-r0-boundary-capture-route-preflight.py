#!/usr/bin/env python3
"""Preflight the route for capturing per-boundary oracle tensors."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-boundary-capture-route-preflight-v0"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SIZE_BYTES = "21166755168"
LLAMA_INSTALL_DIR = "/home/intel/llama-cpp/llama-b9518"
LLAMA_SERVER = f"{LLAMA_INSTALL_DIR}/llama-server"
LLAMA_LIB = f"{LLAMA_INSTALL_DIR}/libllama.so.0.0.9518"
OPENVINO_MODEL = "/home/intel/Qwen3.6-35B-A3B-ov"
INTEL_ENV = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
PRIOR_ORACLE_ROOT = (
    "/home/intel/intel-box-run/native-llama-generation-oracle-cpu-20260615T133419Z"
)
PRIOR_ORACLE_TOOL = f"{PRIOR_ORACLE_ROOT}/tools/native/llama_generation_oracle.py"
REQUIRED_INPUT_RECORDS = 524
REQUIRED_OUTPUT_RECORDS = 524
REQUIRED_BOUNDARY_TYPES = 17


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-boundary-capture-route-preflight-<UTC>.",
  )
  return parser.parse_args()


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def run(cmd: list[str], *, timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "command": cmd,
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr + f"\nlocal timeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_script: str, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_script, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        value = json.loads(line)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected object")
      rows.append(value)
  return rows


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


def parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key:
      values[key.strip()] = value.strip()
  return values


def bool_value(values: dict[str, str], key: str) -> bool:
  return values.get(key) == "true"


def int_value(values: dict[str, str], key: str) -> int | None:
  try:
    return int(values.get(key, ""))
  except ValueError:
    return None


def model_path(contract: dict[str, Any]) -> str | None:
  model = contract.get("model", {})
  if not isinstance(model, dict):
    return None
  value = model.get("path") or model.get("gguf_model_path")
  return value if isinstance(value, str) else None


def line_count(text: str) -> int:
  return len([line for line in text.splitlines() if line.strip()])


def first_matching_line(text: str, pattern: str) -> str | None:
  compiled = re.compile(pattern)
  for line in text.splitlines():
    line = line.strip()
    if compiled.search(line):
      return line
  return None


def parse_llama_version(text: str) -> dict[str, Any]:
  version_line = first_matching_line(text, r"^version:")
  if not version_line:
    return {
        "build_number": None,
        "commit_short": None,
        "raw_version_line": None,
      }
  match = re.search(r"version:\s+(\d+)\s+\(([0-9a-fA-F]+)\)", version_line)
  return {
      "build_number": int(match.group(1)) if match else None,
      "commit_short": match.group(2) if match else None,
      "raw_version_line": version_line,
  }


def capture_queue_summary(capture_queue_dir: Path) -> dict[str, Any]:
  input_rows = load_jsonl(capture_queue_dir / "boundary-input-tasks.jsonl")
  output_rows = load_jsonl(capture_queue_dir / "boundary-output-tasks.jsonl")
  boundary_types = {
      row["boundary_type"]
      for row in input_rows
      if isinstance(row.get("boundary_type"), str)
  }
  source_cases = sorted({
      row["source_prompt_case_id"]
      for row in input_rows
      if isinstance(row.get("source_prompt_case_id"), str)
  })
  source_positions = sorted({
      row["source_token_position"]
      for row in input_rows
      if isinstance(row.get("source_token_position"), int)
  })
  return {
      "boundary_input_task_count": len(input_rows),
      "boundary_output_task_count": len(output_rows),
      "boundary_type_count": len(boundary_types),
      "capture_queue_path": rel(capture_queue_dir),
      "source_prompt_case_ids": source_cases,
      "source_token_positions": source_positions,
  }


def build_summary(payload: dict[str, Any]) -> str:
  route = payload["route_decision"]
  requirements = payload["boundary_capture_requirements"]
  target = payload["target_footholds"]
  lines = [
      "# R0 Boundary Capture Route Preflight",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- required boundary input records: {requirements['boundary_input_task_count']}",
      f"- required boundary output records: {requirements['boundary_output_task_count']}",
      f"- boundary types: {requirements['boundary_type_count']}",
      f"- llama install is binary-only: `{str(target['llama_install_binary_only']).lower()}`",
      f"- llama source tree present on target: `{str(target['llama_source_tree_present']).lower()}`",
      f"- Intel env build tools present: `{str(target['intel_env_build_tools_present']).lower()}`",
      f"- llama build: `{target['llama_runtime_version']['raw_version_line']}`",
      f"- OpenVINO model present: `{str(target['openvino_model_present']).lower()}`",
      f"- selected next route: `{route['selected_next_route']}`",
      f"- current environment can capture full boundary bundle now: `{str(route['current_environment_can_capture_boundary_bundle_now']).lower()}`",
      f"- R0 oracle gate closed: `{str(route['r0_oracle_gate_closed']).lower()}`",
      "",
      "This is a route preflight only. It does not capture tensors and does",
      "not create an oracle bundle.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-boundary-capture-route-preflight-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  runtime_preflight_path = latest("r0-oracle-runtime-preflight-*", "preflight.json")
  capture_queue_path = latest("r0-oracle-capture-queue-*", "capture-queue.json")
  if runtime_preflight_path is None:
    raise SystemExit("no oracle runtime preflight artifact found under output/")
  if capture_queue_path is None:
    raise SystemExit("no oracle capture queue artifact found under output/")

  oracle_contract = load_json(oracle_contract_path)
  model_contract = load_json(model_contract_path)
  runtime_preflight = load_json(runtime_preflight_path)
  queue = capture_queue_summary(capture_queue_path.parent)

  inventory_script = "\n".join([
      "set -u",
      f"printf 'model_present='; test -f {shlex.quote(MODEL_PATH)} && echo true || echo false",
      f"printf 'model_size_bytes='; stat -c %s {shlex.quote(MODEL_PATH)} 2>/dev/null || echo missing",
      f"printf 'llama_install_dir_present='; test -d {shlex.quote(LLAMA_INSTALL_DIR)} && echo true || echo false",
      f"printf 'llama_server_present='; test -x {shlex.quote(LLAMA_SERVER)} && echo true || echo false",
      f"printf 'llama_lib_present='; test -f {shlex.quote(LLAMA_LIB)} && echo true || echo false",
      f"printf 'intel_env_present='; test -f {shlex.quote(INTEL_ENV)} && echo true || echo false",
      f"printf 'llama_install_cpp_source_count='; find {shlex.quote(LLAMA_INSTALL_DIR)} -maxdepth 3 -type f \\( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' -o -name '*.h' -o -name '*.hpp' \\) 2>/dev/null | wc -l",
      "printf 'llama_source_cmakelists_count='; find /home/intel -maxdepth 5 -type f -name CMakeLists.txt 2>/dev/null | grep -E '/(llama|ggml|llama.cpp)' | wc -l",
      "printf 'llama_source_candidate_dir_count='; find /home/intel -maxdepth 5 -type d \\( -iname '*llama.cpp*' -o -iname '*ggml*' \\) 2>/dev/null | wc -l",
      "printf 'cmake_present='; command -v cmake >/dev/null 2>&1 && echo true || echo false",
      "printf 'ninja_present='; command -v ninja >/dev/null 2>&1 && echo true || echo false",
      "printf 'gxx_present='; command -v g++ >/dev/null 2>&1 && echo true || echo false",
      "printf 'git_present='; command -v git >/dev/null 2>&1 && echo true || echo false",
      f"printf 'openvino_model_present='; test -d {shlex.quote(OPENVINO_MODEL)} && echo true || echo false",
      f"printf 'openvino_language_model_xml_present='; test -f {shlex.quote(OPENVINO_MODEL + '/openvino_language_model.xml')} && echo true || echo false",
      f"printf 'prior_oracle_tool_present='; test -f {shlex.quote(PRIOR_ORACLE_TOOL)} && echo true || echo false",
      "printf 'home_available_gib='; df -BG /home/intel 2>/dev/null | awk 'NR==2 {gsub(/G/, \"\", $4); print $4}'",
      "printf 'hostname='; hostname",
  ])
  intel_env_toolchain_script = "\n".join([
      "set -u",
      f"if test -f {shlex.quote(INTEL_ENV)}; then . {shlex.quote(INTEL_ENV)}; fi",
      "printf 'intel_env_cmake_present='; command -v cmake >/dev/null 2>&1 && echo true || echo false",
      "printf 'intel_env_cmake_path='; command -v cmake 2>/dev/null || echo missing",
      "printf 'intel_env_cmake_version='; cmake --version 2>/dev/null | head -1 || echo missing",
      "printf 'intel_env_ninja_present='; command -v ninja >/dev/null 2>&1 && echo true || echo false",
      "printf 'intel_env_ninja_path='; command -v ninja 2>/dev/null || echo missing",
      "printf 'intel_env_ninja_version='; ninja --version 2>/dev/null || echo missing",
      "printf 'intel_env_gxx_present='; command -v g++ >/dev/null 2>&1 && echo true || echo false",
      "printf 'intel_env_gxx_path='; command -v g++ 2>/dev/null || echo missing",
      "printf 'intel_env_gxx_version='; g++ --version 2>/dev/null | head -1 || echo missing",
  ])
  commands = {
      "inventory": (inventory_script, 45),
      "intel_env_toolchain": (intel_env_toolchain_script, 45),
      "llama_server_version": (f"{shlex.quote(LLAMA_SERVER)} --version 2>&1 || true", 30),
      "llama_install_listing": (
          f"test -d {shlex.quote(LLAMA_INSTALL_DIR)} && ls -la {shlex.quote(LLAMA_INSTALL_DIR)} | sed -n '1,120p' || true",
          30,
      ),
      "llama_source_candidates": (
          "{ find /home/intel -maxdepth 5 -type f -name CMakeLists.txt 2>/dev/null | "
          "grep -E '/(llama|ggml|llama\\.cpp)' || true; "
          "find /home/intel -maxdepth 5 -type d "
          "\\( -iname '*llama.cpp*' -o -iname '*ggml*' \\) 2>/dev/null || true; } "
          "| sed -n '1,200p'",
          45,
      ),
      "llama_symbol_probe": (
          f"test -f {shlex.quote(LLAMA_LIB)} && "
          f"(nm -D {shlex.quote(LLAMA_LIB)} 2>/dev/null | c++filt | "
          "grep -E 'llama_(decode|tokenize|batch|model|context)|ggml_' | sed -n '1,160p') || true",
          45,
      ),
      "openvino_model_probe": (
          f"find {shlex.quote(OPENVINO_MODEL)} -maxdepth 2 -type f 2>/dev/null | sed -n '1,120p'; "
          f"find {shlex.quote(OPENVINO_MODEL)} -maxdepth 2 -type f -name '*.xml' -print -quit | "
          "xargs -r grep -E 'Parameter|Result|MatMul|RMS|TopK|Gather|past|present' -m 40 2>/dev/null || true",
          45,
      ),
      "locked_model_process_check": (
          "pgrep -af '[l]lama-server.*qwen3.6-35b-a3b-q4_k_m.gguf' || true",
          20,
      ),
  }
  raw_status: dict[str, dict[str, Any]] = {}
  for name, (script, timeout_s) in commands.items():
    result = run_target(args.host, script, timeout_s=timeout_s)
    raw_status[name] = {
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
    }
    (raw_dir / f"{name}.stdout").write_text(result["stdout"], encoding="utf-8")
    (raw_dir / f"{name}.stderr").write_text(result["stderr"], encoding="utf-8")

  inventory_values = parse_key_values((raw_dir / "inventory.stdout").read_text(encoding="utf-8"))
  toolchain_values = parse_key_values(
      (raw_dir / "intel_env_toolchain.stdout").read_text(encoding="utf-8")
  )
  llama_version = parse_llama_version(
      (raw_dir / "llama_server_version.stdout").read_text(encoding="utf-8")
  )
  llama_install_cpp_source_count = int_value(inventory_values, "llama_install_cpp_source_count") or 0
  llama_source_cmakelists_count = int_value(inventory_values, "llama_source_cmakelists_count") or 0
  llama_source_candidate_dir_count = int_value(inventory_values, "llama_source_candidate_dir_count") or 0
  symbol_count = line_count((raw_dir / "llama_symbol_probe.stdout").read_text(encoding="utf-8"))
  source_candidate_count = line_count((raw_dir / "llama_source_candidates.stdout").read_text(encoding="utf-8"))
  locked_model_process_count = line_count(
      (raw_dir / "locked_model_process_check.stdout").read_text(encoding="utf-8")
  )
  source_tree_present = llama_source_cmakelists_count > 0
  install_binary_only = (
      bool_value(inventory_values, "llama_install_dir_present")
      and bool_value(inventory_values, "llama_server_present")
      and llama_install_cpp_source_count == 0
  )
  system_build_tools_present = (
      bool_value(inventory_values, "cmake_present")
      and bool_value(inventory_values, "gxx_present")
  )
  intel_env_build_tools_present = (
      bool_value(toolchain_values, "intel_env_cmake_present")
      and bool_value(toolchain_values, "intel_env_gxx_present")
      and bool_value(toolchain_values, "intel_env_ninja_present")
  )
  build_tools_present = system_build_tools_present or intel_env_build_tools_present
  runtime_routes = runtime_preflight.get("oracle_runtime_routes", {})
  stock_boundary_route_status = (
      runtime_routes.get("per_boundary_tensors", {}).get("route_status")
  )
  locked_model_contract_matches = (
      model_path(oracle_contract) == MODEL_PATH
      and model_path(model_contract) == MODEL_PATH
  )
  can_capture_now = source_tree_present and build_tools_present
  selected_route = (
      "instrument_existing_llama_cpp_source_tree"
      if can_capture_now
      else "stage_exact_llama_cpp_source_commit_and_build_with_intel_env"
      if intel_env_build_tools_present
      else "stage_or_reconstruct_instrumented_llama_cpp_source_build"
  )
  route_status = (
      "candidate_llama_cpp_graph_instrumentation_route"
      if can_capture_now
      else "missing_instrumentable_llama_cpp_source_tree_on_target"
  )
  target_footholds = {
      "build_tools_present": build_tools_present,
      "cmake_present": bool_value(inventory_values, "cmake_present"),
      "git_present": bool_value(inventory_values, "git_present"),
      "gxx_present": bool_value(inventory_values, "gxx_present"),
      "home_available_gib": int_value(inventory_values, "home_available_gib"),
      "hostname": inventory_values.get("hostname"),
      "llama_dynamic_symbol_probe_count": symbol_count,
      "llama_install_binary_only": install_binary_only,
      "llama_install_cpp_source_count": llama_install_cpp_source_count,
      "llama_install_dir": LLAMA_INSTALL_DIR,
      "llama_install_dir_present": bool_value(inventory_values, "llama_install_dir_present"),
      "llama_lib_present": bool_value(inventory_values, "llama_lib_present"),
      "llama_runtime_version": llama_version,
      "llama_server_present": bool_value(inventory_values, "llama_server_present"),
      "llama_source_candidate_path_count": source_candidate_count,
      "llama_source_cmakelists_count": llama_source_cmakelists_count,
      "llama_source_candidate_dir_count": llama_source_candidate_dir_count,
      "llama_source_tree_present": source_tree_present,
      "locked_model_process_count": locked_model_process_count,
      "locked_model_process_present": locked_model_process_count > 0,
      "model_present": bool_value(inventory_values, "model_present"),
      "model_size_bytes": inventory_values.get("model_size_bytes"),
      "model_size_matches_contract": inventory_values.get("model_size_bytes") == MODEL_SIZE_BYTES,
      "ninja_present": bool_value(inventory_values, "ninja_present"),
      "intel_env_build_tools_present": intel_env_build_tools_present,
      "intel_env_cmake_path": toolchain_values.get("intel_env_cmake_path"),
      "intel_env_cmake_present": bool_value(toolchain_values, "intel_env_cmake_present"),
      "intel_env_cmake_version": toolchain_values.get("intel_env_cmake_version"),
      "intel_env_gxx_path": toolchain_values.get("intel_env_gxx_path"),
      "intel_env_gxx_present": bool_value(toolchain_values, "intel_env_gxx_present"),
      "intel_env_gxx_version": toolchain_values.get("intel_env_gxx_version"),
      "intel_env_ninja_path": toolchain_values.get("intel_env_ninja_path"),
      "intel_env_ninja_present": bool_value(toolchain_values, "intel_env_ninja_present"),
      "intel_env_ninja_version": toolchain_values.get("intel_env_ninja_version"),
      "intel_env_present": bool_value(inventory_values, "intel_env_present"),
      "openvino_language_model_xml_present": bool_value(
          inventory_values,
          "openvino_language_model_xml_present",
      ),
      "openvino_model_present": bool_value(inventory_values, "openvino_model_present"),
      "prior_oracle_tool_present": bool_value(inventory_values, "prior_oracle_tool_present"),
  }
  route_decision = {
      "current_environment_can_capture_boundary_bundle_now": can_capture_now,
      "locked_gguf_model_required_for_boundary_bundle": True,
      "openvino_route_role": (
          "denominator_and_sanity_reference_only_not_locked_gguf_boundary_source"
      ),
      "r0_oracle_gate_closed": False,
      "route_status": route_status,
      "selected_next_route": selected_route,
      "stock_boundary_route_status": stock_boundary_route_status,
      "why_not_stock_route": [
          "llama-server and OpenVINO command paths do not expose the 17 required architecture boundary tensors as bundle JSONL",
          "installed llama.cpp runtime is binary-only on target, so graph-level tensor hooks cannot be added in place",
          "dynamic library symbol probe did not expose a stable per-boundary hook surface",
      ],
      "next_required_steps": [
          "stage official llama.cpp source for commit 7c158fbb4, matching the target llama-b9518 runtime version line",
          "add a gated GGUF/llama.cpp graph instrumentation mode for the queued short_math_001 prefill-last-token boundary source",
          "build the instrumented runtime in user space via the existing Intel env toolchain",
          "dump boundary-references/inputs.jsonl and boundary-references/outputs.jsonl with tensor payload paths, shape metadata, dtype metadata, source_prompt_case_id, and source_token_position",
          "run tools/intel-qwen36-r0-oracle-bundle-validate.py against the assembled bundle before resident harness load",
      ],
  }
  payload = {
      "boundary_capture_requirements": queue,
      "created_at": created_at,
      "evidence": {
          "capture_queue": rel(capture_queue_path),
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
          "raw_dir": rel(raw_dir),
          "runtime_preflight": rel(runtime_preflight_path),
      },
      "host": args.host,
      "raw_command_status": raw_status,
      "route_decision": route_decision,
      "schema_version": SCHEMA_VERSION,
      "target_footholds": target_footholds,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "capture_queue_boundary_requirements_match_contract",
          "pass": queue["boundary_input_task_count"] == REQUIRED_INPUT_RECORDS
          and queue["boundary_output_task_count"] == REQUIRED_OUTPUT_RECORDS
          and queue["boundary_type_count"] == REQUIRED_BOUNDARY_TYPES,
      },
      {
          "name": "runtime_preflight_records_missing_stock_boundary_route",
          "pass": stock_boundary_route_status == "missing_stock_boundary_tensor_capture_route",
          "stock_boundary_route_status": stock_boundary_route_status,
      },
      {
          "name": "locked_model_contract_matches_gguf_runtime",
          "pass": locked_model_contract_matches
          and target_footholds["model_present"]
          and target_footholds["model_size_matches_contract"],
      },
      {
          "name": "target_has_no_locked_model_llama_server_process",
          "pass": target_footholds["locked_model_process_present"] is False,
          "locked_model_process_count": locked_model_process_count,
      },
      {
          "name": "preflight_does_not_close_oracle_gate",
          "pass": route_decision["r0_oracle_gate_closed"] is False
          and route_decision["current_environment_can_capture_boundary_bundle_now"] is False,
      },
      {
          "name": "selected_route_requires_instrumented_llama_cpp_source_build",
          "pass": route_decision["selected_next_route"]
          in {
              "stage_or_reconstruct_instrumented_llama_cpp_source_build",
              "stage_exact_llama_cpp_source_commit_and_build_with_intel_env",
          },
          "route_status": route_status,
      },
      {
          "name": "target_llama_runtime_version_identified",
          "pass": llama_version["build_number"] == 9518
          and llama_version["commit_short"] == "7c158fbb4",
          "llama_runtime_version": llama_version,
      },
      {
          "name": "intel_env_toolchain_available_for_user_space_build",
          "pass": intel_env_build_tools_present,
          "cmake": toolchain_values.get("intel_env_cmake_version"),
          "gxx": toolchain_values.get("intel_env_gxx_version"),
          "ninja": toolchain_values.get("intel_env_ninja_version"),
      },
  ]
  correctness = {
      "checks": checks,
      "gate": "r0_boundary_capture_route_preflight",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-boundary-capture-route-preflight.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "preflight.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("boundary_input_task_count", queue["boundary_input_task_count"]),
        ("boundary_output_task_count", queue["boundary_output_task_count"]),
        ("llama_source_tree_present", source_tree_present),
        ("llama_install_binary_only", install_binary_only),
        ("intel_env_build_tools_present", intel_env_build_tools_present),
        ("current_environment_can_capture_boundary_bundle_now", can_capture_now),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_boundary_capture_route_preflight",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"boundary capture route preflight output: {out_dir}")
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
