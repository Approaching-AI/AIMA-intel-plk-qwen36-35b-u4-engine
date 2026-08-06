#!/usr/bin/env python3
"""Gate the fixed NPU third of ADR 0013's exact Q6_K M=1 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-npu-exact-q6-representation-gate-v0"
DEFAULT_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
DEFAULT_VECTOR = (
    ROOT / "output/lm-head-q4-surrogate-gate-20260711Tseq627cleanZ/"
    "raw/final-norm-vectors.f32")
DEFAULT_VECTOR_METADATA = DEFAULT_VECTOR.parent / "vector-metadata.json"
DEFAULT_OPENVINO_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
DEFAULT_CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
DEFAULT_INCLUDE = Path("/home/intel/intel-box-env/conda/include")
DEFAULT_LIB = Path("/home/intel/intel-box-env/conda/lib")
WORKER = ROOT / "engine/tools/npu_exact_q6_representation_worker.py"
NATIVE_HARNESS = ROOT / "engine/tools/npu_level_zero_graph_blob_preflight.cpp"
ABI_HEADER = ROOT / "engine/include/intel_qwen36/npu_level_zero_graph_ext.hpp"
TENSOR_NAME = "output.weight"
EXPECTED_COLUMNS = 2_048
EXPECTED_TOTAL_ROWS = 248_320
FIXED_NPU_ROWS = 82_752
FIXED_GPU_ROWS = EXPECTED_TOTAL_ROWS - FIXED_NPU_ROWS
Q6_BLOCK_BYTES = 210
QK_K = 256
Q6_KILL_GB_S = 96.0
COMPONENT_COSINE_MIN = 0.999
COMPONENT_RELATIVE_L2_MAX = 0.002
REPRESENTATION_RELATIVE_L2_MAX = 1.0e-5


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  parser.add_argument("--tensor-index", type=Path, default=DEFAULT_INDEX)
  parser.add_argument("--vector", type=Path, default=DEFAULT_VECTOR)
  parser.add_argument(
      "--vector-metadata", type=Path, default=DEFAULT_VECTOR_METADATA)
  parser.add_argument("--vector-index", type=int, default=0)
  parser.add_argument("--rows", type=int, default=FIXED_NPU_ROWS)
  parser.add_argument(
      "--openvino-python", type=Path, default=DEFAULT_OPENVINO_PYTHON)
  parser.add_argument("--cxx", type=Path, default=DEFAULT_CXX)
  parser.add_argument("--level-zero-include", type=Path, default=DEFAULT_INCLUDE)
  parser.add_argument("--level-zero-lib", type=Path, default=DEFAULT_LIB)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=9)
  parser.add_argument("--timeout-s", type=int, default=1_200)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.rows <= 0 or args.rows > EXPECTED_TOTAL_ROWS:
    parser.error("rows must be in [1, 248320]")
  if args.vector_index < 0 or args.warmup < 0 or args.repeat <= 0:
    parser.error("invalid vector index, warmup, or repeat")
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/npu-exact-q6-representation-{stamp}"
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


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
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "timed_out": True,
        "wall_s": time.perf_counter() - started,
    }


def skipped(command: list[str], reason: str) -> dict[str, Any]:
  return {
      "command": command,
      "returncode": 125,
      "stderr": reason,
      "stdout": "",
      "timed_out": False,
      "wall_s": 0.0,
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


def load_tensor(path: Path, name: str) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      value = json.loads(line)
      if isinstance(value, dict) and value.get("name") == name:
        return value
  raise SystemExit(f"tensor {name!r} not found in {path}")


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def number(value: Any) -> float | None:
  if isinstance(value, (int, float)) and math.isfinite(float(value)):
    return float(value)
  return None


def component_correct(comparison: dict[str, Any], rows: int) -> bool:
  cosine = number(comparison.get("cosine"))
  relative_l2 = number(comparison.get("relative_l2"))
  return bool(
      comparison.get("finite") is True and
      comparison.get("compared") == rows and
      cosine is not None and cosine >= COMPONENT_COSINE_MIN and
      relative_l2 is not None and relative_l2 <= COMPONENT_RELATIVE_L2_MAX)


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  required = [
      args.model,
      args.tensor_index,
      args.vector,
      args.vector_metadata,
      args.openvino_python,
      args.cxx,
      args.level_zero_include / "level_zero/ze_api.h",
      args.level_zero_lib / "libze_loader.so",
      WORKER,
      NATIVE_HARNESS,
      ABI_HEADER,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  tensor = load_tensor(args.tensor_index, TENSOR_NAME)
  dims = tensor.get("dims")
  if not isinstance(dims, list) or len(dims) != 2:
    raise SystemExit(f"unexpected {TENSOR_NAME} dimensions: {dims!r}")
  columns, total_rows = (int(dims[0]), int(dims[1]))
  row_bytes = columns // QK_K * Q6_BLOCK_BYTES
  total_payload_bytes = total_rows * row_bytes
  npu_payload_bytes = args.rows * row_bytes
  vector_metadata = json.loads(args.vector_metadata.read_text(encoding="utf-8"))
  vector_record = (
      vector_metadata[args.vector_index]
      if isinstance(vector_metadata, list) and
      args.vector_index < len(vector_metadata) else None)

  xml_path = raw_dir / "exact-q6-low4-high2.xml"
  bin_path = raw_dir / "exact-q6-low4-high2.bin"
  blob_path = raw_dir / "exact-q6-low4-high2.blob"
  inputs_path = raw_dir / "exact-q6-low4-high2.inputs"
  reference_path = raw_dir / "exact-q6-low4-high2.reference"
  binary_path = raw_dir / "npu-level-zero-graph-blob-preflight"
  created_at = iso_now()

  worker_command = [
      str(args.openvino_python), str(WORKER),
      "--model", str(args.model),
      "--tensor-name", TENSOR_NAME,
      "--tensor-offset", str(tensor.get("absolute_offset")),
      "--tensor-rows", str(total_rows),
      "--columns", str(columns),
      "--rows", str(args.rows),
      "--vector", str(args.vector),
      "--vector-index", str(args.vector_index),
      "--xml", str(xml_path),
      "--bin", str(bin_path),
      "--warmup", str(args.warmup),
      "--repeat", str(args.repeat),
  ]
  worker_run = run(worker_command, args.timeout_s)
  write_run(raw_dir, "openvino-npu-worker", worker_run)
  worker_probe = parse_last_json(worker_run["stdout"])

  build_command = [
      str(args.cxx), "-std=gnu++20", "-O2", "-Wall", "-Wextra",
      "-Wpedantic", "-Werror", f"-I{ROOT / 'engine/include'}",
      f"-I{args.level_zero_include}", str(NATIVE_HARNESS),
      f"-L{args.level_zero_lib}",
      f"-Wl,-rpath,{args.level_zero_lib}", "-lze_loader",
      "-o", str(binary_path),
  ]
  build_run = run(build_command, args.timeout_s)
  write_run(raw_dir, "build-native-harness", build_run)
  ldd_run = (
      run(["ldd", str(binary_path)], args.timeout_s)
      if build_run["returncode"] == 0 else
      skipped(["ldd", str(binary_path)], "native harness build failed"))
  write_run(raw_dir, "ldd-native-harness", ldd_run)

  compile_command = [
      str(binary_path), "--mode", "compile", "--xml", str(xml_path),
      "--blob", str(blob_path), "--inputs", str(inputs_path),
      "--reference", str(reference_path), "--warmup", str(args.warmup),
      "--repeat", str(args.repeat),
  ]
  compile_run = (
      run(compile_command, args.timeout_s)
      if build_run["returncode"] == 0 and worker_run["returncode"] == 0 and
      xml_path.is_file() and bin_path.is_file() else
      skipped(compile_command, "worker or native harness prerequisite failed"))
  write_run(raw_dir, "compile-level-zero-graph", compile_run)
  compile_probe = parse_last_json(compile_run["stdout"])

  native_command = [
      str(binary_path), "--mode", "run", "--blob", str(blob_path),
      "--inputs", str(inputs_path), "--reference", str(reference_path),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
  ]
  native_run = (
      run(native_command, args.timeout_s)
      if compile_run["returncode"] == 0 else
      skipped(native_command, "Level Zero graph compilation failed"))
  write_run(raw_dir, "run-native-blob", native_run)
  native_probe = parse_last_json(native_run["stdout"])

  representation = worker_probe.get("representation", {})
  comparisons = worker_probe.get("comparison", {})
  representation_comparison = comparisons.get(
      "representation_vs_q6_q8_oracle", {})
  npu_comparison = comparisons.get("npu_vs_q6_q8_oracle", {})
  native_min_us = number(native_probe.get("execution_min_us"))
  native_median_us = number(native_probe.get("execution_median_us"))
  npu_source_min_gb_s = (
      npu_payload_bytes / (native_min_us * 1_000.0)
      if native_min_us and native_min_us > 0 else None)
  npu_source_median_gb_s = (
      npu_payload_bytes / (native_median_us * 1_000.0)
      if native_median_us and native_median_us > 0 else None)
  zero_gpu_optimistic_ceiling_gb_s = (
      total_payload_bytes / (native_min_us * 1_000.0)
      if native_min_us and native_min_us > 0 else None)
  required_npu_source_gb_s = Q6_KILL_GB_S * args.rows / total_rows
  blob_bytes = int(compile_probe.get("native_blob_bytes", 0) or 0)
  blob_expansion = blob_bytes / npu_payload_bytes if npu_payload_bytes else None
  dependency_text = ldd_run["stdout"].lower()
  xml_text = xml_path.read_text(encoding="utf-8") if xml_path.is_file() else ""
  representation_relative_l2 = number(
      representation_comparison.get("relative_l2"))
  state = git_state()

  checks = [
      check("clean_committed_source", state["dirty"] is False, git=state),
      check("locked_real_q6_tensor",
            tensor.get("ggml_type_name") == "Q6_K" and
            tensor.get("name") == TENSOR_NAME and
            columns == EXPECTED_COLUMNS and total_rows == EXPECTED_TOTAL_ROWS and
            int(tensor.get("nbytes", -1)) == total_payload_bytes,
            tensor=tensor),
      check("fixed_2_to_1_partition",
            args.rows == FIXED_NPU_ROWS and
            FIXED_GPU_ROWS + args.rows == total_rows,
            gpu_rows=FIXED_GPU_ROWS, npu_rows=args.rows,
            gpu_to_npu_ratio=FIXED_GPU_ROWS / args.rows),
      check("real_q8_input_recorded",
            isinstance(vector_record, dict) and
            vector_record.get("source_required_checks_passed") is True,
            vector_index=args.vector_index, vector_record=vector_record),
      check("worker_completed", worker_run["returncode"] == 0 and
            "fatal_error" not in worker_probe,
            fatal_error=worker_probe.get("fatal_error")),
      check("all_real_npu_rows_consumed",
            worker_probe.get("source", {}).get("payload_bytes") ==
            npu_payload_bytes and
            worker_probe.get("source", {}).get("rows") == args.rows,
            source=worker_probe.get("source")),
      check("exact_low4_high2_source_ir",
            representation.get("low4_min") == 0 and
            representation.get("low4_max") == 15 and
            representation.get("high2_min") == 0 and
            representation.get("high2_max") == 3 and
            'element_type="u4"' in xml_text and
            'element_type="u2"' in xml_text,
            representation=representation),
      check("representation_matches_q6_q8_oracle",
            representation_comparison.get("finite") is True and
            representation_comparison.get("compared") == args.rows and
            representation_relative_l2 is not None and
            representation_relative_l2 <= REPRESENTATION_RELATIVE_L2_MAX,
            comparison=representation_comparison,
            relative_l2_max=REPRESENTATION_RELATIVE_L2_MAX),
      check("npu_all_value_component_correctness",
            component_correct(npu_comparison, args.rows),
            comparison=npu_comparison,
            cosine_min=COMPONENT_COSINE_MIN,
            relative_l2_max=COMPONENT_RELATIVE_L2_MAX),
      check("native_harness_builds_with_werror", build_run["returncode"] == 0),
      check("native_harness_links_level_zero_only",
            ldd_run["returncode"] == 0 and
            "libze_loader" in dependency_text and
            "openvino" not in dependency_text and "dnnl" not in dependency_text,
            ldd=ldd_run["stdout"].splitlines()),
      check("exact_ir_compiles_to_native_blob",
            compile_run["returncode"] == 0 and blob_bytes > 0 and
            compile_probe.get("device") == "Intel(R) AI Boost",
            probe=compile_probe),
      check("native_blob_runs_in_fresh_process",
            native_run["returncode"] == 0 and
            native_probe.get("device") == "Intel(R) AI Boost" and
            native_probe.get("compared_bytes") == args.rows * 4 and
            native_probe.get("mismatch_bytes") == 0,
            probe=native_probe),
      check("native_process_maps_no_openvino",
            native_probe.get("openvino_mapped") is False,
            probe=native_probe),
      check("fixed_partition_optimistic_ceiling_reaches_96_gb_s",
            zero_gpu_optimistic_ceiling_gb_s is not None and
            zero_gpu_optimistic_ceiling_gb_s >= Q6_KILL_GB_S,
            calculation=(
                "total Q6 bytes / best native NPU time; GPU is charged zero time"),
            npu_source_min_gb_s=npu_source_min_gb_s,
            required_npu_source_gb_s=required_npu_source_gb_s,
            required_pair_gb_s=Q6_KILL_GB_S,
            zero_gpu_optimistic_ceiling_gb_s=zero_gpu_optimistic_ceiling_gb_s),
      check("speedup_claims_forbidden", True),
  ]
  required_passed = all(row["pass"] for row in checks)
  runtime_ok = all(row["pass"] for row in checks if row["name"] in {
      "native_harness_builds_with_werror",
      "native_harness_links_level_zero_only",
      "exact_ir_compiles_to_native_blob",
      "native_blob_runs_in_fresh_process",
      "native_process_maps_no_openvino",
  })
  correctness_ok = all(row["pass"] for row in checks if row["name"] in {
      "representation_matches_q6_q8_oracle",
      "npu_all_value_component_correctness",
  })
  bandwidth_ok = next(
      row["pass"] for row in checks
      if row["name"] == "fixed_partition_optimistic_ceiling_reaches_96_gb_s")
  if not runtime_ok:
    disposition = "reject_fixed_gpu_npu_route_on_exact_npu_runtime_legality"
  elif not correctness_ok:
    disposition = "reject_fixed_gpu_npu_route_on_exact_q6_correctness"
  elif not bandwidth_ok:
    disposition = "reject_fixed_gpu_npu_route_below_q6_kill_number"
  elif required_passed:
    disposition = "admit_fixed_gpu_npu_route_to_variable_m_prefill_gate"
  else:
    disposition = "reject_noncanonical_exact_q6_gate_run"

  artifact_hashes = {
      path.name: sha256_file(path)
      for path in (xml_path, bin_path, blob_path, inputs_path, reference_path)
      if path.is_file()
  }
  result = {
      "artifact_hashes": artifact_hashes,
      "checks": checks,
      "created_at": created_at,
      "disposition": disposition,
      "git": state,
      "native_compile_probe": compile_probe,
      "native_run_probe": native_probe,
      "performance": {
          "fixed_gpu_rows": FIXED_GPU_ROWS,
          "fixed_npu_rows": args.rows,
          "native_blob_bytes": blob_bytes,
          "native_blob_to_raw_q6_expansion": blob_expansion,
          "npu_raw_q6_bytes": npu_payload_bytes,
          "npu_source_min_gb_s": npu_source_min_gb_s,
          "npu_source_median_gb_s": npu_source_median_gb_s,
          "pair_kill_number_gb_s": Q6_KILL_GB_S,
          "required_npu_source_gb_s": required_npu_source_gb_s,
          "total_raw_q6_bytes": total_payload_bytes,
          "zero_gpu_optimistic_ceiling_gb_s":
              zero_gpu_optimistic_ceiling_gb_s,
      },
      "required_checks_passed": required_passed,
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "abi_header": str(ABI_HEADER.relative_to(ROOT)),
          "abi_header_sha256": sha256_file(ABI_HEADER),
          "native_harness": str(NATIVE_HARNESS.relative_to(ROOT)),
          "native_harness_sha256": sha256_file(NATIVE_HARNESS),
          "tensor_index": str(args.tensor_index.relative_to(ROOT)),
          "vector": str(args.vector.relative_to(ROOT)),
          "vector_metadata": str(args.vector_metadata.relative_to(ROOT)),
          "worker": str(WORKER.relative_to(ROOT)),
          "worker_sha256": sha256_file(WORKER),
      },
      "speedup_claims_allowed": False,
      "worker": worker_probe,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "npu_vs_q6_q8_oracle": npu_comparison,
      "representation_vs_q6_q8_oracle": representation_comparison,
      "required_checks_passed": required_passed,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "fixed M=1 exact Q6_K representation and kill-number gate",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": state,
      "model_path": str(args.model),
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "npu_component_relative_l2",
       "value": npu_comparison.get("relative_l2")},
      {"metric": "npu_component_cosine", "value": npu_comparison.get("cosine")},
      {"metric": "native_blob_bytes", "value": blob_bytes},
      {"metric": "native_blob_to_raw_q6_expansion", "value": blob_expansion},
      {"metric": "native_npu_source_min_gb_s", "value": npu_source_min_gb_s},
      {"metric": "native_npu_source_median_gb_s",
       "value": npu_source_median_gb_s},
      {"metric": "zero_gpu_optimistic_ceiling_gb_s",
       "value": zero_gpu_optimistic_ceiling_gb_s},
      {"metric": "required_checks_passed", "value": required_passed},
  ]
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in metrics:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  summary = [
      "# Fixed NPU exact-Q6 representation gate",
      "",
      f"- tensor / fixed split: `{TENSOR_NAME}` / "
      f"GPU `{FIXED_GPU_ROWS}` rows + NPU `{args.rows}` rows",
      f"- real NPU Q6 bytes: `{npu_payload_bytes}`",
      f"- NPU vs Q6_K/Q8_K oracle: cosine "
      f"`{npu_comparison.get('cosine')}`, relative L2 "
      f"`{npu_comparison.get('relative_l2')}` over "
      f"`{npu_comparison.get('compared')}` outputs",
      f"- native blob bytes / raw-Q6 expansion: `{blob_bytes}` / "
      f"`{blob_expansion}`",
      f"- native NPU raw-Q6-equivalent min / median: "
      f"`{npu_source_min_gb_s}` / `{npu_source_median_gb_s}` GB/s",
      f"- zero-GPU optimistic pair ceiling / required: "
      f"`{zero_gpu_optimistic_ceiling_gb_s}` / `{Q6_KILL_GB_S}` GB/s",
      f"- required checks passed: `{str(required_passed).lower()}`",
      f"- disposition: `{disposition}`",
      "",
      "The source graph preserves every real Q6_K low4/high2 code and group-16",
      "scale before compilation.  The performance verdict uses the best native",
      "Level Zero execution and charges the GPU zero time, so a failed ceiling",
      "cannot be repaired by GPU concurrency or host-runtime overhead removal.",
      "This is a component route decision, not a product speedup claim.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_passed,
      "zero_gpu_optimistic_ceiling_gb_s": zero_gpu_optimistic_ceiling_gb_s,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
