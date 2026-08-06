#!/usr/bin/env python3
"""Prove that a model-specific NPU blob runs without OpenVINO/oneDNN linkage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-npu-level-zero-native-blob-legality-v1"
HEADER = ROOT / "engine/include/intel_qwen36/npu_level_zero_graph_ext.hpp"
SOURCE = ROOT / "engine/tools/npu_level_zero_graph_blob_preflight.cpp"
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_INCLUDE = Path("/home/intel/intel-box-env/conda/include")
DEFAULT_LIB = Path("/home/intel/intel-box-env/conda/lib")
DEFAULT_OPENVINO_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
UPSTREAM_REVISION = "c7cb5d218ca14f6a81b3ef0bb89e718e9fcdba8e"


IR_GENERATOR = r'''
import json
import sys

import numpy as np
from openvino import Model, Type, get_version, serialize
from openvino import opset15 as ov

xml_path, bin_path = sys.argv[1:3]
source = ov.parameter([1, 16], Type.f32, name="input")
two = ov.constant(np.array(2.0, dtype=np.float32))
one = ov.constant(np.array(1.0, dtype=np.float32))
result = ov.add(ov.multiply(source, two), one, name="output")
result.output(0).get_tensor().set_names({"output"})
model = Model([result], [source], "iq36_npu_native_blob_legality")
serialize(model, xml_path, bin_path)
print(json.dumps({
    "bin": bin_path,
    "openvino_version": get_version(),
    "output_values": 16,
    "xml": xml_path,
}, sort_keys=True))
'''


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--level-zero-include", type=Path,
                      default=DEFAULT_INCLUDE)
  parser.add_argument("--level-zero-lib", type=Path, default=DEFAULT_LIB)
  parser.add_argument("--openvino-python", type=Path,
                      default=DEFAULT_OPENVINO_PYTHON)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/npu-level-zero-blob-legality-{stamp}"
  return args


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  started = time.perf_counter()
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": command,
        "returncode": result.returncode,
        "stderr": result.stderr,
        "stdout": result.stdout,
        "timed_out": False,
        "wall_s": time.perf_counter() - started,
    }
  except subprocess.TimeoutExpired as exc:
    return {
        "command": command,
        "returncode": 124,
        "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
        "timed_out": True,
        "wall_s": time.perf_counter() - started,
    }


def write_run(raw_dir: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw_dir / f"{name}.command.json", {
      "command": result["command"],
      "returncode": result["returncode"],
      "timed_out": result["timed_out"],
      "wall_s": result["wall_s"],
  })
  (raw_dir / f"{name}.stdout").write_text(
      result["stdout"], encoding="utf-8")
  (raw_dir / f"{name}.stderr").write_text(
      result["stderr"], encoding="utf-8")


def parse_last_json(text: str) -> dict[str, Any]:
  for line in reversed(text.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  required = [
      HEADER,
      SOURCE,
      args.cxx,
      args.level_zero_include / "level_zero/ze_api.h",
      args.level_zero_lib / "libze_loader.so",
      args.openvino_python,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  xml_path = raw_dir / "native-blob-legality.xml"
  bin_path = raw_dir / "native-blob-legality.bin"
  blob_path = raw_dir / "native-blob-legality.blob"
  inputs_path = raw_dir / "native-blob-legality.inputs"
  reference_path = raw_dir / "native-blob-legality.reference"
  binary_path = raw_dir / "npu-level-zero-graph-blob-preflight"

  generate = run([
      str(args.openvino_python), "-c", IR_GENERATOR,
      str(xml_path), str(bin_path),
  ], args.timeout_s)
  write_run(raw_dir, "generate-ir", generate)

  build = run([
      str(args.cxx), "-std=gnu++20", "-O2", "-Wall", "-Wextra",
      "-Wpedantic", "-Werror", f"-I{ROOT / 'engine/include'}",
      f"-I{args.level_zero_include}", str(SOURCE),
      f"-L{args.level_zero_lib}",
      f"-Wl,-rpath,{args.level_zero_lib}", "-lze_loader",
      "-o", str(binary_path),
  ], args.timeout_s)
  write_run(raw_dir, "build", build)

  ldd = (
      run(["ldd", str(binary_path)], args.timeout_s)
      if build["returncode"] == 0 else
      {"command": ["ldd", str(binary_path)], "returncode": 125,
       "stderr": "build failed", "stdout": "", "timed_out": False,
       "wall_s": 0.0})
  write_run(raw_dir, "ldd", ldd)

  compile_run = (
      run([
          str(binary_path), "--mode", "compile", "--xml", str(xml_path),
          "--blob", str(blob_path), "--inputs", str(inputs_path),
          "--reference", str(reference_path),
      ], args.timeout_s)
      if build["returncode"] == 0 and generate["returncode"] == 0 else
      {"command": [str(binary_path), "--mode", "compile"],
       "returncode": 125, "stderr": "prerequisite failed", "stdout": "",
       "timed_out": False, "wall_s": 0.0})
  write_run(raw_dir, "compile-graph", compile_run)
  compile_probe = parse_last_json(compile_run["stdout"])

  native_run = (
      run([
          str(binary_path), "--mode", "run", "--blob", str(blob_path),
          "--inputs", str(inputs_path), "--reference", str(reference_path),
      ], args.timeout_s)
      if compile_run["returncode"] == 0 else
      {"command": [str(binary_path), "--mode", "run"],
       "returncode": 125, "stderr": "compile graph failed", "stdout": "",
       "timed_out": False, "wall_s": 0.0})
  write_run(raw_dir, "run-native-blob", native_run)
  native_probe = parse_last_json(native_run["stdout"])

  dependency_text = ldd["stdout"].lower()
  header_text = HEADER.read_text(encoding="utf-8")
  artifact_hashes = {
      path.name: sha256_file(path)
      for path in (xml_path, bin_path, blob_path, inputs_path, reference_path)
      if path.is_file()
  }
  evidence_checks = [
      check("clean_committed_source", git_state()["dirty"] is False,
            git=git_state()),
      check("pinned_official_graph_abi_header",
            UPSTREAM_REVISION in header_text and "SPDX-License-Identifier: MIT" in
            header_text, upstream_revision=UPSTREAM_REVISION),
      check("tiny_static_ir_generated",
            generate["returncode"] == 0 and xml_path.is_file() and
            bin_path.is_file()),
      check("native_harness_builds_with_werror", build["returncode"] == 0),
      check("executable_links_level_zero",
            ldd["returncode"] == 0 and "libze_loader" in dependency_text),
      check("executable_does_not_link_openvino_or_onednn",
            "openvino" not in dependency_text and "dnnl" not in dependency_text,
            ldd=ldd["stdout"].splitlines()),
      check("ir_compiles_through_level_zero_graph_abi",
            compile_run["returncode"] == 0 and
            compile_probe.get("mode") == "compile" and
            compile_probe.get("device") == "Intel(R) AI Boost" and
            int(compile_probe.get("native_blob_bytes", 0)) > 0,
            probe=compile_probe),
      check("compiler_process_maps_no_openvino",
            compile_probe.get("openvino_mapped") is False,
            probe=compile_probe),
      check("native_blob_runs_in_fresh_process",
            native_run["returncode"] == 0 and
            native_probe.get("mode") == "run" and
            native_probe.get("device") == "Intel(R) AI Boost",
            probe=native_probe),
      check("native_blob_output_is_byte_exact",
            native_probe.get("compared_bytes") == 64 and
            native_probe.get("mismatch_bytes") == 0,
            compared_bytes=native_probe.get("compared_bytes"),
            mismatch_bytes=native_probe.get("mismatch_bytes")),
      check("native_process_maps_no_openvino",
            native_probe.get("openvino_mapped") is False,
            probe=native_probe),
      check("system_npu_driver_compiler_mapping_recorded",
            isinstance(native_probe.get("npu_driver_compiler_mapped"), bool),
            mapped=native_probe.get("npu_driver_compiler_mapped"),
            note=("The Level Zero NPU driver maps its system compiler library "
                  "even for a native blob; the application does not link it.")),
      check("speedup_claims_forbidden", True),
  ]
  required_passed = all(row["pass"] for row in evidence_checks)
  disposition = (
      "admit_level_zero_native_blob_runtime_for_exact_component"
      if required_passed else
      "reject_gpu_npu_route_on_runtime_legality")
  state = git_state()
  result = {
      "artifact_hashes": artifact_hashes,
      "checks": evidence_checks,
      "compile_probe": compile_probe,
      "created_at": created_at,
      "disposition": disposition,
      "git": state,
      "native_probe": native_probe,
      "required_checks_passed": required_passed,
      "runtime_boundary": {
          "application_link_dependencies": ldd["stdout"].splitlines(),
          "offline_ir_generator": "OpenVINO Python; outside native runtime",
          "runtime_api": "Level Zero loader plus ZE_extension_graph v1 ABI",
          "system_driver_compiler_mapped_for_native_blob":
              native_probe.get("npu_driver_compiler_mapped"),
      },
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "abi_header": str(HEADER.relative_to(ROOT)),
          "abi_header_sha256": sha256_file(HEADER),
          "abi_upstream_revision": UPSTREAM_REVISION,
          "harness": str(SOURCE.relative_to(ROOT)),
          "harness_sha256": sha256_file(SOURCE),
      },
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": evidence_checks,
      "compared_bytes": native_probe.get("compared_bytes"),
      "mismatch_bytes": native_probe.get("mismatch_bytes"),
      "required_checks_passed": required_passed,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "native NPU graph-blob runtime legality gate only",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": state,
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "native_blob_bytes",
       "value": compile_probe.get("native_blob_bytes")},
      {"metric": "native_compared_bytes",
       "value": native_probe.get("compared_bytes")},
      {"metric": "native_mismatch_bytes",
       "value": native_probe.get("mismatch_bytes")},
      {"metric": "required_checks_passed", "value": required_passed},
  ]
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in metrics:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  summary = [
      "# Level Zero NPU native-blob runtime legality gate",
      "",
      f"- device: `{native_probe.get('device')}`",
      f"- graph extension / compiler: `{compile_probe.get('graph_extension')}` / "
      f"`{compile_probe.get('compiler_major')}."
      f"{compile_probe.get('compiler_minor')}`",
      f"- native blob bytes: `{compile_probe.get('native_blob_bytes')}`",
      f"- native output compared / mismatched bytes: "
      f"`{native_probe.get('compared_bytes')} / "
      f"{native_probe.get('mismatch_bytes')}`",
      f"- OpenVINO mapped in native process: "
      f"`{native_probe.get('openvino_mapped')}`",
      f"- system NPU driver compiler mapped internally: "
      f"`{native_probe.get('npu_driver_compiler_mapped')}`",
      f"- required checks passed: `{str(required_passed).lower()}`",
      f"- disposition: `{disposition}`",
      "",
      "The application binary links only Level Zero plus the C/C++ runtime; it",
      "does not link or map OpenVINO/oneDNN.  The installed NPU driver maps its",
      "own compiler library even for native blobs, which is recorded as a system",
      "driver behavior.  This gate authorizes the exact component boundary only",
      "and is not a product correctness or speed claim.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_passed,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
