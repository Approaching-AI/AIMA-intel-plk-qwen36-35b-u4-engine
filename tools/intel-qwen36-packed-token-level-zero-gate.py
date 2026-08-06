#!/usr/bin/env python3
"""Gate one-submit Level Zero execution for the packed token schedule.

This gate measures a unique-byte streaming proxy for every logical command in
the seq732 schedule. It proves the target command-list mechanism and host
boundary budget; accepted Q4/Q6 component evidence remains the model-math
carrier. The result is not a product-decode or speedup claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-packed-token-level-zero-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-packed-token-level-zero-probe"
MODULE_SOURCE = ROOT / "engine/gpu/opencl/packed_token_stream_probe.cl"
SCHEDULE_GATE = (
    ROOT / "output/packed-token-schedule-gate-20260712Tseq735-state-cleanZ/result.json")
STATE_BUDGET = (
    ROOT / "output/packed-token-state-budget-gate-20260712Tseq734cleanZ/result.json")
ROUTE_GATE = (
    ROOT / "output/product-decode-route-gate-20260712Tseq731cleanZ/result.json")
CONSENSUS_GATE = (
    ROOT / "output/native-consensus-gate-20260712Tseq730cleanZ/result.json")
SOURCE_PATHS = [
    MODULE_SOURCE,
    ROOT / "engine/tools/packed_token_level_zero_probe.cpp",
    ROOT / "engine/include/intel_qwen36/packed_token_schedule.hpp",
    ROOT / "engine/src/packed_token_schedule.cpp",
    ROOT / "engine/CMakeLists.txt",
    ROOT / "engine/boundaries.json",
]
STRICT_BYTES = 2_128_395_904
COMMAND_COUNT = 252
KERNEL_RATE_GB_S = 106.524608569878
KERNEL_CAP_US = 19_980.321285140562
HOST_CAP_US = 100.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--samples", type=int, default=7)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/packed-token-level-zero-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(command: list[str], timeout_s: int = 180) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
  except subprocess.TimeoutExpired as exc:
    return {
        "command": command, "returncode": 124,
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") +
            "\ntimeout",
    }
  return {
      "command": command, "returncode": proc.returncode,
      "stdout": proc.stdout, "stderr": proc.stderr,
  }


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def parse_stdout(row: dict[str, Any]) -> dict[str, Any]:
  try:
    value = json.loads(str(row.get("stdout", "")).strip())
  except json.JSONDecodeError:
    return {}
  return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  row = run(["git", *args], timeout_s=30)
  return row["stdout"].strip() if row["returncode"] == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty), "dirty_paths": dirty.splitlines(),
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(paths: list[Path]) -> tuple[str, dict[str, str]]:
  aggregate = hashlib.sha256()
  files: dict[str, str] = {}
  for path in paths:
    relative = str(path.relative_to(ROOT))
    payload = path.read_bytes()
    files[relative] = hashlib.sha256(payload).hexdigest()
    aggregate.update(relative.encode("utf-8"))
    aggregate.update(b"\0")
    aggregate.update(payload)
  return aggregate.hexdigest(), files


def row_passes(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is True and
      row.get("command_count") == COMMAND_COUNT and
      row.get("kernel_count") == COMMAND_COUNT and
      row.get("barrier_count") == COMMAND_COUNT - 1 and
      row.get("command_list_record_count") == 1 and
      row.get("strict_stream_bytes_per_token") == STRICT_BYTES and
      row.get("payload_allocation_bytes") == STRICT_BYTES and
      row.get("checksums_change_with_token_control") is True and
      row.get("maps_native_only") is True and
      finite(row.get("device_time_min_us")) and
      float(row["device_time_min_us"]) <= KERNEL_CAP_US and
      finite(row.get("effective_stream_gb_s")) and
      float(row["effective_stream_gb_s"]) >= KERNEL_RATE_GB_S and
      finite(row.get("submit_min_us")) and
      float(row["submit_min_us"]) <= HOST_CAP_US and
      finite(row.get("host_residual_min_us")) and
      float(row["host_residual_min_us"]) <= HOST_CAP_US)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  generated = out / "generated"
  generated.mkdir()
  required = [args.model, args.cmake, SCHEDULE_GATE, STATE_BUDGET, ROUTE_GATE,
              CONSENSUS_GATE, *SOURCE_PATHS]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  schedule = load_json(SCHEDULE_GATE)
  state_budget = load_json(STATE_BUDGET)
  route = load_json(ROUTE_GATE)
  consensus = load_json(CONSENSUS_GATE)
  clinfo = run(["clinfo"], timeout_s=30)
  cl_extensions = clinfo.get("stdout", "").lower()
  module_compile = run([
      "ocloc", "compile", "-file", str(MODULE_SOURCE), "-device", "0xb080",
      "-options", "-cl-std=CL2.0", "-output", "iq36_packed_token_stream",
      "-out_dir", str(generated), "-output_no_suffix", "--format", "zebin",
      "-q",
  ])
  module = generated / "iq36_packed_token_stream.bin"
  module_validate = run(["ocloc", "validate", "-file", str(module)]) \
      if module.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "module missing",
      }
  configure = run([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B",
      str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
  ])
  build = run([
      str(args.cmake), "--build", str(args.build_dir), "--target", TARGET,
      "-j", "8",
  ]) if configure["returncode"] == 0 else {
      "command": [], "returncode": 125, "stdout": "",
      "stderr": "configure failed",
  }
  binary = args.build_dir / TARGET

  def probe(label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = run([
        str(binary), str(args.model), str(module), str(args.warmup),
        str(args.samples),
    ], timeout_s=120)
    write_json(out / f"{label}-run.json", raw)
    parsed = parse_stdout(raw)
    write_json(out / f"{label}.json", parsed)
    return raw, parsed

  repeat_raw, repeat = probe("repeat") if (
      module_validate["returncode"] == 0 and build["returncode"] == 0) else (
          {"returncode": 125, "stdout": "", "stderr": "build missing"}, {})
  confirm_raw, confirm = probe("confirm") if repeat else (
      {"returncode": 125, "stdout": "", "stderr": "repeat missing"}, {})
  link_map = run(["ldd", str(binary)], timeout_s=30) \
      if binary.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "binary missing",
      }
  lower_link_map = (link_map.get("stdout", "") +
                    link_map.get("stderr", "")).lower()
  aggregate_sha, source_files = source_digest(SOURCE_PATHS)
  device_min_values = [
      float(row["device_time_min_us"])
      for row in (repeat, confirm)
      if finite(row.get("device_time_min_us"))
  ]
  device_spread_fraction = (
      (max(device_min_values) - min(device_min_values)) /
      min(device_min_values)
      if len(device_min_values) == 2 and min(device_min_values) > 0
      else math.inf)

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("state_complete_schedule_admits_backend_implementation",
            schedule.get("required_checks_passed") is True and
            schedule.get("disposition") ==
                "admit_packed_token_backend_implementation"),
      check("seq734_state_budget_is_bound",
            state_budget.get("required_checks_passed") is True and
            state_budget.get("state_census", {}).get(
                "strict_stream_bytes_per_token") == STRICT_BYTES and
            math.isclose(
                float(state_budget.get("admission", {}).get(
                    "kernel_schedule_ms_max", math.inf)),
                KERNEL_CAP_US / 1000.0, abs_tol=1e-9)),
      check("opencl_command_buffer_extension_absent",
            clinfo["returncode"] == 0 and
            "cl_khr_command_buffer" not in cl_extensions),
      check("level_zero_native_module_compiles_and_validates",
            module_compile["returncode"] == 0 and
            module_validate["returncode"] == 0 and module.is_file()),
      check("probe_builds", configure["returncode"] == 0 and
            build["returncode"] == 0),
      check("repeat_exact_schedule_proxy_passes",
            repeat_raw["returncode"] == 0 and row_passes(repeat),
            device_us=repeat.get("device_time_min_us"),
            effective_gb_s=repeat.get("effective_stream_gb_s"),
            host_residual_us=repeat.get("host_residual_min_us")),
      check("confirm_exact_schedule_proxy_passes",
            confirm_raw["returncode"] == 0 and row_passes(confirm),
            device_us=confirm.get("device_time_min_us"),
            effective_gb_s=confirm.get("effective_stream_gb_s"),
            host_residual_us=confirm.get("host_residual_min_us")),
      check("repeat_confirm_device_window_is_stable",
            finite(device_spread_fraction) and device_spread_fraction <= 0.05,
            spread_fraction=device_spread_fraction),
      check("level_zero_only_native_dependency",
            link_map["returncode"] == 0 and
            "libze_loader" in lower_link_map and
            "openvino" not in lower_link_map and
            "libdnnl" not in lower_link_map),
      check("real_carrier_and_consensus_anchors_remain_closed",
            route.get("required_checks_passed") is True and
            route.get("selected_route", {}).get("id") ==
                "resident_packed_full_token_schedule_v5" and
            consensus.get("required_checks_passed") is True and
            len(consensus.get("rows", [])) == 3),
  ]
  passed = all(row["pass"] for row in checks)
  created_at = iso_now()
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "git": state,
      "source": {"aggregate_sha256": aggregate_sha, "files": source_files},
      "module": {
          "path": str(module.relative_to(ROOT)),
          "sha256": sha256(module) if module.is_file() else None,
      },
      "inputs": {
          "schedule_gate": str(SCHEDULE_GATE.relative_to(ROOT)),
          "state_budget": str(STATE_BUDGET.relative_to(ROOT)),
          "route_gate": str(ROUTE_GATE.relative_to(ROOT)),
          "consensus_gate": str(CONSENSUS_GATE.relative_to(ROOT)),
          "model": str(args.model),
      },
      "admission": {
          "strict_stream_bytes_per_token": STRICT_BYTES,
          "logical_command_count": COMMAND_COUNT,
          "device_time_us_max": KERNEL_CAP_US,
          "kernel_window_gb_s_min": KERNEL_RATE_GB_S,
          "host_residual_us_max": HOST_CAP_US,
      },
      "repeat": repeat,
      "confirm": confirm,
      "repeat_confirm_device_spread_fraction": device_spread_fraction,
      "checks": checks,
      "required_checks_passed": passed,
      "disposition": (
          "admit_level_zero_real_stage_backend_port"
          if passed else "reject_level_zero_packed_schedule_mechanism"),
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed,
      "product_promotion_ready": False, "speedup_claims_allowed": False,
  })
  write_json(out / "build.json", {
      "clinfo": clinfo, "module_compile": module_compile,
      "module_validate": module_validate, "configure": configure,
      "build": build, "link_map": link_map,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "artifact": str(out),
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "git": state, "source_sha256": aggregate_sha,
      "required_checks_passed": passed, "speedup_claims_allowed": False,
  })
  with (out / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in [
        ("repeat_device_min_us", repeat.get("device_time_min_us")),
        ("confirm_device_min_us", confirm.get("device_time_min_us")),
        ("repeat_effective_stream_gb_s", repeat.get("effective_stream_gb_s")),
        ("confirm_effective_stream_gb_s", confirm.get("effective_stream_gb_s")),
        ("repeat_host_residual_min_us", repeat.get("host_residual_min_us")),
        ("confirm_host_residual_min_us", confirm.get("host_residual_min_us")),
    ]:
      fh.write(json.dumps({
          "metric": metric, "phase": "backend_mechanism", "value": value,
      }, sort_keys=True) + "\n")
  repeat_us = float(repeat.get("device_time_min_us", math.nan))
  confirm_us = float(confirm.get("device_time_min_us", math.nan))
  (out / "summary.md").write_text("\n".join([
      "# Packed token Level Zero backend mechanism gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- logical kernels / barriers: `"
      f"{repeat.get('kernel_count')} / {repeat.get('barrier_count')}`",
      f"- unique payload bytes: `{repeat.get('payload_allocation_bytes')}`",
      f"- repeat / confirm device time: `"
      f"{repeat_us / 1000.0:.3f} / {confirm_us / 1000.0:.3f} ms`",
      f"- repeat / confirm effective proxy rate: `"
      f"{float(repeat.get('effective_stream_gb_s', math.nan)):.3f} / "
      f"{float(confirm.get('effective_stream_gb_s', math.nan)):.3f} GB/s`",
      f"- repeat / confirm host residual: `"
      f"{float(repeat.get('host_residual_min_us', math.nan)):.3f} / "
      f"{float(confirm.get('host_residual_min_us', math.nan)):.3f} us`",
      "- command list records / token submits: `1 / 1 per measured token`", "",
      "The proxy streams a disjoint buffer range for every seq732 command and "
      "includes 251 dependency barriers. It admits porting real accepted stages "
      "to the Level Zero backend; it does not execute model math or claim product "
      "speed.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out), "pass": passed,
      "repeat_device_ms": repeat_us / 1000.0,
      "confirm_device_ms": confirm_us / 1000.0,
      "repeat_effective_gb_s": repeat.get("effective_stream_gb_s"),
      "confirm_effective_gb_s": confirm.get("effective_stream_gb_s"),
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
