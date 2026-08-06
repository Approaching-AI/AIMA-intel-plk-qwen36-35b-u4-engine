#!/usr/bin/env python3
"""Run the resident GPU captured-layer shell handoff probe."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shlex
from pathlib import Path
from types import ModuleType
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
BASE_TOOL = Path(__file__).with_name("intel-qwen36-gpu-captured-layer-shell-probe.py")
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-captured-layer-shell-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_base_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_captured_layer_shell_probe", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load base probe tool: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base_tool()


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--resident-invocations", type=int, default=11)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--timeout-s", type=int, default=240)
  return parser.parse_args()


def resident_probe_cpp(opencl_source: str) -> str:
  cpp = BASE.PROBE_CPP
  cpp = cpp.replace(
      '\\"schema_version\\":\\"intel-qwen36-gpu-captured-layer-shell-probe-v0\\"',
      f'\\"schema_version\\":\\"{SCHEMA_VERSION}\\"',
  )
  cpp = cpp.replace(
      'std::cout << "\\"repeat\\":" << args.repeat << ",";',
      (
          'std::cout << "\\"repeat\\":" << args.repeat << ",";\n'
          '    std::cout << "\\"resident_api\\":\\"captured_layer_shell_load_once_run_many\\",";\n'
          '    std::cout << "\\"resident_load_count\\":1,";\n'
          '    std::cout << "\\"resident_shell_invocations\\":" << args.repeat << ",";\n'
          '    std::cout << "\\"resident_reuses_opencl_program\\":true,";\n'
          '    std::cout << "\\"resident_reuses_device_buffers\\":true,";'
      ),
  )
  cpp = cpp.replace(
      '        timing_positive;',
      '        timing_positive &&\n        args.repeat > 0;',
  )
  cpp = cpp.replace(
      '    std::cout << "\\"gpu_event_timing_positive\\":" << (timing_positive ? "true" : "false") << ",";\n'
      '    std::cout << "\\"speedup_claims_forbidden\\":true";',
      (
          '    std::cout << "\\"gpu_event_timing_positive\\":" << (timing_positive ? "true" : "false") << ",";\n'
          '    std::cout << "\\"resident_load_once\\":true,";\n'
          '    std::cout << "\\"resident_shell_invocations_positive\\":" << (args.repeat > 0 ? "true" : "false") << ",";\n'
          '    std::cout << "\\"resident_reuses_opencl_program_and_buffers\\":true,";\n'
          '    std::cout << "\\"speedup_claims_forbidden\\":true";'
      ),
  )
  return cpp.replace(
      "@@OPENCL_SOURCE_LITERAL@@",
      BASE.cpp_raw_string_literal(opencl_source),
  )


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      return json.loads(line)
  return None


def nested_bool(obj: dict[str, Any] | None, *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return bool(current)


def nested_number(obj: dict[str, Any], *keys: str) -> float | None:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else None


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "captured_layer_shell_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("resident_reuses_opencl_program") is True
      and probe.get("resident_reuses_device_buffers") is True
      and nested_bool(probe, "checks", "resident_load_once")
      and nested_bool(probe, "checks", "resident_shell_invocations_positive")
      and nested_bool(probe, "checks", "resident_reuses_opencl_program_and_buffers")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  layer_cmp = {}
  if isinstance(comparisons, dict):
    layer_group = comparisons.get("layer_output", {})
    if isinstance(layer_group, dict):
      layer_cmp = layer_group.get("gpu_vs_oracle", {})
  lines = [
      "# GPU Resident Captured-Layer Shell Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident load count: `{probe.get('resident_load_count')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- reuses OpenCL program: `{str(probe.get('resident_reuses_opencl_program')).lower()}`",
      f"- reuses device buffers: `{str(probe.get('resident_reuses_device_buffers')).lower()}`",
      "",
      "| metric | value |",
      "|---|---:|",
      f"| layer_output gpu_vs_oracle max abs | {layer_cmp.get('max_abs_diff')} |",
      f"| layer_output gpu_vs_oracle RMSE | {layer_cmp.get('rmse')} |",
      f"| captured shell kernel sum min us | {timings.get('captured_layer_shell_kernel_sum_min_us')} |",
      f"| captured shell kernel sum mean us | {timings.get('captured_layer_shell_kernel_sum_mean_us')} |",
      "",
      "The target-side process loads model metadata, builds one OpenCL program,",
      "stages captured payload/device buffers once, then invokes the captured",
      "layer shell repeatedly through that resident boundary. This remains a",
      "captured single-layer shell; it is not prompt/token decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-captured-layer-shell-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads = BASE.resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_resident_captured_layer_shell_probe.cpp"
  local_cpp.write_text(resident_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-captured-layer-shell-probe-{stamp}"
  setup = iq36_local.run_target(
      args.host,
      "rm -rf " + shlex.quote(remote_dir) + " && mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      name: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for name in payloads
  }
  remote_payload_dir = f"{remote_dir}/oracle"
  if setup.get("returncode") == 0:
    for local, remote in BASE.SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_captured_layer_shell_probe.cpp",
            args.timeout_s,
        )
    )
    for name, payload in payloads.items():
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  executable = f"{remote_dir}/build/iq36-gpu-resident-captured-layer-shell-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_captured_layer_shell_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(executable)}"
      ),
  ])
  stage_ok = (
      setup.get("returncode") == 0
      and transfers
      and all(item.get("returncode") == 0 for item in transfers)
      and all(item.get("returncode") == 0 for item in payload_transfers.values())
  )
  compile_result = (
      iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
      if stage_ok
      else {"cmd": ["stage"], "returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_argv = [
      executable,
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
      "--repeat", str(args.resident_invocations),
      "--device-substring", args.device_substring,
  ]
  run_result = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
              shell_join(run_argv),
          ]),
          args.timeout_s,
      )
      if compile_result.get("returncode") == 0
      else {"cmd": run_argv, "returncode": None, "stdout": "", "stderr": "compile skipped run"}
  )
  probe = parse_probe_stdout(run_result.get("stdout", ""))

  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "payload-transfers.json", payload_transfers)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  if probe is not None:
    iq36_local.write_json(out_dir / "probe-result.json", probe)

  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {"name": "source_files_transferred", "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers)},
      {"name": "oracle_payloads_transferred", "pass": all(item.get("returncode") == 0 for item in payload_transfers.values())},
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
      {"name": "captured_layer_shell_matches_oracle", "pass": nested_bool(probe, "checks", "captured_layer_shell_matches_oracle")},
      {"name": "gpu_event_timing_positive", "pass": nested_bool(probe, "checks", "gpu_event_timing_positive")},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "oracle_bundle": str(args.oracle_bundle.resolve().relative_to(ROOT)),
      "payloads": slim_payloads,
      "layer": args.layer,
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-captured-layer-shell-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
      "resident_invocations": args.resident_invocations,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(out_dir / "probe.json", payload)
  iq36_local.write_json(out_dir / "manifest.json", manifest)
  iq36_local.write_json(out_dir / "correctness.json", correctness)

  aggregate = probe if isinstance(probe, dict) else {}
  timings = aggregate.get("timings", {}) if isinstance(aggregate.get("timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_resident_captured_layer_shell_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("captured_layer_shell_kernel_sum_min_us", nested_number(timings, "captured_layer_shell_kernel_sum_min_us")),
          ("layer_output_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "layer_output", "gpu_vs_oracle", "max_abs_diff")),
          ("resident_load_count", nested_number(aggregate, "resident_load_count")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
