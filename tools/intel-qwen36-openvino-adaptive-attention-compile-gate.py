#!/usr/bin/env python3
"""Compile-audit the four-stage OpenVINO adaptive-attention graph owner.

This gate applies the repository patch to a disposable copy of the pinned
current custom primitive, compiles that translation unit with the existing
OpenVINO L0 build command, links an isolated plugin, and compiles minimal
top-512 and top-256 static graphs at the locked 64k/output-512 geometry.  It
captures the live L0 native compiler-cache binaries and their exact OpenCL
build options without executing an inference request, then audits the ABI and
compiler resources.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-adaptive-attention-compile-gate-v0"
ROUTE = "openvino_attention_adaptive_exact_block32_i8_kv_graph_compile"
NEXT_ROUTE = "openvino_attention_adaptive_layer3_boundary"

OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/openvino-90214e-l0-gpu")
OV_OUTPUT = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
NINJA = Path("/home/intel/intel-box-env/conda/bin/ninja")
AR = Path("/home/intel/intel-box-env/conda/bin/ar")
OCLOC = Path("/usr/bin/ocloc")

CUSTOM_CPP_REL = Path(
    "src/plugins/intel_gpu/src/graph/impls/ocl/custom_primitive.cpp")
CUSTOM_CPP = OV_SOURCE / CUSTOM_CPP_REL
CUSTOM_OBJECT_TARGET = (
    "src/plugins/intel_gpu/src/graph/impls/ocl/CMakeFiles/"
    "openvino_intel_gpu_ocl_obj.dir/custom_primitive.cpp.o")
LM_HEAD_CPP_REL = Path(
    "src/plugins/intel_gpu/src/graph/impls/ocl/iq36_lm_head_i8q4.cpp")
LM_HEAD_CPP = OV_SOURCE / LM_HEAD_CPP_REL
LM_HEAD_OBJECT_TARGET = (
    "src/plugins/intel_gpu/src/graph/impls/ocl/CMakeFiles/"
    "openvino_intel_gpu_ocl_obj.dir/iq36_lm_head_i8q4.cpp.o")
PLUGIN_TARGET = OV_OUTPUT / "libopenvino_intel_gpu_plugin.so"
GRAPH_ARCHIVE = OV_OUTPUT / "libopenvino_intel_gpu_graph.a"
PATCH = ROOT / (
    "engine/openvino/iq36-custom-adaptive-attention-multikernel.patch")
LM_HEAD_PATCH = ROOT / (
    "engine/openvino/iq36-lm-head-i8q4-adaptive-correction.patch")
CUSTOM_XML = ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
SOURCE_GATE = ROOT / (
    "output/openvino-adaptive-attention-source-abi-gate-"
    "20260721Tseq1726-block2d-all512-clean/gate.json")

PINNED_COMMIT = "90214e5be05"
MAX_CHUNKS = 129
FIXED_COLD_CAPACITY = 65536
EXACT_HISTORY_CAPACITY = 66560
HOT_WINDOW = 16384
CHUNK_TOKENS = 512
KV_HEADS = 2
Q_HEADS = 16
HEAD_DIM = 256
SCALE_BYTES = 16
KEY_TILE_TOKENS = 16
HOT_KEY_WORDS_PER_BLOCK = 2048
EXPECTED_ENTRIES = (
    "iq36_adaptive_attention_partial",
    "iq36_adaptive_attention_select_reduce_union",
    "iq36_adaptive_attention_correct_normalize",
    "iq36_adaptive_attention_ordered_update",
)
EXPECTED_WORKGROUPS = {
    EXPECTED_ENTRIES[0]: [128, 1, 1],
    EXPECTED_ENTRIES[1]: [256, 1, 1],
    EXPECTED_ENTRIES[2]: [128, 1, 1],
    EXPECTED_ENTRIES[3]: [128, 1, 1],
}
EXPECTED_SIMD = {
    EXPECTED_ENTRIES[0]: 16,
    EXPECTED_ENTRIES[1]: 32,
    EXPECTED_ENTRIES[2]: 16,
    EXPECTED_ENTRIES[3]: 16,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--worker-config", type=Path)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.worker_config is None and args.output is None:
    parser.error("--output is required outside worker mode")
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as stream:
    for row in rows:
      stream.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def run_command(
    command: list[str], timeout_s: int, *, cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
  started = time.perf_counter_ns()
  try:
    run = subprocess.run(
        command, cwd=cwd, env=environment, check=False,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout_s)
    return {
        "command": command,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "returncode": run.returncode,
        "stdout": run.stdout,
        "stderr": run.stderr,
    }
  except subprocess.TimeoutExpired as exc:
    return {
        "command": command,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "returncode": 124,
        "stdout": str(exc.stdout or ""),
        "stderr": str(exc.stderr or exc),
    }


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_output(cwd: Path, *arguments: str) -> str:
  run = subprocess.run(
      ["git", *arguments], cwd=cwd, check=False, capture_output=True,
      text=True, encoding="utf-8", errors="replace")
  if run.returncode != 0:
    raise RuntimeError(
        f"git {' '.join(arguments)} failed: {run.stderr.strip()}")
  return run.stdout


def git_state(output: Path) -> dict[str, Any]:
  rows = git_output(ROOT, "status", "--porcelain").splitlines()
  try:
    output_relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    output_relative = ""
  rows = [
      row for row in rows
      if not output_relative or output_relative not in row]
  return {
      "commit": git_output(ROOT, "rev-parse", "HEAD").strip(),
      "dirty": bool(rows),
      "dirty_paths": rows,
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def ninja_command(target: str, timeout_s: int) -> tuple[str, dict[str, Any]]:
  result = run_command([
      str(NINJA), "-C", str(OV_BUILD), "-t", "commands", target,
  ], timeout_s)
  lines = [line for line in result["stdout"].splitlines() if line.strip()]
  if result["returncode"] != 0 or not lines:
    raise RuntimeError(f"cannot recover ninja command for {target}")
  return lines[-1], result


def build_isolated_plugin(
    raw: Path, timeout_s: int, stop_bytes: int,
    memory: list[dict[str, Any]],
) -> dict[str, Any]:
  build = raw / "build"
  build.mkdir()
  patched_cpp = build / "custom_primitive.cpp"
  object_path = build / "custom_primitive.cpp.o"
  patched_lm_head_cpp = build / "iq36_lm_head_i8q4.cpp"
  lm_head_object_path = build / "iq36_lm_head_i8q4.cpp.o"
  archive_path = build / "libopenvino_intel_gpu_graph-adaptive.a"
  plugin_path = build / "libopenvino_intel_gpu_plugin-adaptive.so"

  apply_check = run_command([
      "git", "-C", str(OV_SOURCE), "apply", "--check", str(PATCH),
  ], timeout_s)
  patch_run = run_command([
      "patch", "-o", str(patched_cpp), str(CUSTOM_CPP), str(PATCH),
  ], timeout_s)
  compile_text, ninja_compile = ninja_command(
      CUSTOM_OBJECT_TARGET, timeout_s)
  compile_command = shlex.split(compile_text)
  source_text = str(CUSTOM_CPP)
  if source_text not in compile_command:
    raise RuntimeError("custom primitive source missing from compile command")
  compile_command[compile_command.index(source_text)] = str(patched_cpp)
  for flag, replacement in (("-o", object_path), ("-MF", build / "custom_primitive.cpp.o.d")):
    if flag not in compile_command:
      raise RuntimeError(f"{flag} missing from compile command")
    compile_command[compile_command.index(flag) + 1] = str(replacement)
  compile_command.append(f"-I{CUSTOM_CPP.parent}")
  sample_memory("before-cxx-compile", stop_bytes, memory)
  compile_run = run_command(
      compile_command, timeout_s, cwd=OV_BUILD)
  sample_memory("after-cxx-compile", stop_bytes, memory)

  lm_head_apply_check = run_command([
      "git", "-C", str(OV_SOURCE), "apply", "--check", str(LM_HEAD_PATCH),
  ], timeout_s)
  lm_head_patch_run = run_command([
      "patch", "-o", str(patched_lm_head_cpp), str(LM_HEAD_CPP),
      str(LM_HEAD_PATCH),
  ], timeout_s)
  lm_head_compile_text, lm_head_ninja_compile = ninja_command(
      LM_HEAD_OBJECT_TARGET, timeout_s)
  lm_head_compile_command = shlex.split(lm_head_compile_text)
  lm_head_source_text = str(LM_HEAD_CPP)
  if lm_head_source_text not in lm_head_compile_command:
    raise RuntimeError("LM-head source missing from compile command")
  lm_head_compile_command[
      lm_head_compile_command.index(lm_head_source_text)] = str(
          patched_lm_head_cpp)
  for flag, replacement in (
      ("-o", lm_head_object_path),
      ("-MF", build / "iq36_lm_head_i8q4.cpp.o.d")):
    if flag not in lm_head_compile_command:
      raise RuntimeError(f"{flag} missing from LM-head compile command")
    lm_head_compile_command[
        lm_head_compile_command.index(flag) + 1] = str(replacement)
  lm_head_compile_command.append(f"-I{LM_HEAD_CPP.parent}")
  sample_memory("before-lm-head-cxx-compile", stop_bytes, memory)
  lm_head_compile_run = run_command(
      lm_head_compile_command, timeout_s, cwd=OV_BUILD)
  sample_memory("after-lm-head-cxx-compile", stop_bytes, memory)

  archive_run: dict[str, Any] = {
      "command": [], "returncode": 1, "stdout": "", "stderr": "skipped"}
  link_run: dict[str, Any] = {
      "command": [], "returncode": 1, "stdout": "", "stderr": "skipped"}
  ninja_link: dict[str, Any] = {}
  link_text = ""
  if (compile_run["returncode"] == 0 and object_path.is_file() and
      lm_head_apply_check["returncode"] == 0 and
      lm_head_patch_run["returncode"] == 0 and
      lm_head_compile_run["returncode"] == 0 and
      lm_head_object_path.is_file()):
    shutil.copyfile(GRAPH_ARCHIVE, archive_path)
    archive_run = run_command([
        str(AR), "rcs", str(archive_path), str(object_path),
        str(lm_head_object_path),
    ], timeout_s)
  if archive_run["returncode"] == 0:
    link_text, ninja_link = ninja_command(str(PLUGIN_TARGET), timeout_s)
    if str(GRAPH_ARCHIVE) not in link_text or str(PLUGIN_TARGET) not in link_text:
      raise RuntimeError("graph archive or plugin output missing from link command")
    isolated_link = link_text.replace(
        str(GRAPH_ARCHIVE), str(archive_path)).replace(
            f"-o {PLUGIN_TARGET}", f"-o {plugin_path}")
    sample_memory("before-plugin-link", stop_bytes, memory)
    link_run = run_command(
        ["bash", "-c", f"set -o pipefail; {isolated_link}"],
        timeout_s, cwd=OV_BUILD)
    sample_memory("after-plugin-link", stop_bytes, memory)

  payload = {
      "apply_check": apply_check,
      "archive": {
          **archive_run,
          "path": relative(archive_path),
          "sha256": sha256(archive_path) if archive_path.is_file() else None,
      },
      "compile": {
          **compile_run,
          "ninja_query": ninja_compile,
          "object": relative(object_path),
          "object_sha256": sha256(object_path) if object_path.is_file() else None,
          "patched_source": relative(patched_cpp),
          "patched_source_sha256": (
              sha256(patched_cpp) if patched_cpp.is_file() else None),
      },
      "lm_head": {
          "apply_check": lm_head_apply_check,
          "compile": {
              **lm_head_compile_run,
              "ninja_query": lm_head_ninja_compile,
              "object": relative(lm_head_object_path),
              "object_sha256": (
                  sha256(lm_head_object_path)
                  if lm_head_object_path.is_file() else None),
              "patched_source": relative(patched_lm_head_cpp),
              "patched_source_sha256": (
                  sha256(patched_lm_head_cpp)
                  if patched_lm_head_cpp.is_file() else None),
          },
          "patch": lm_head_patch_run,
      },
      "link": {
          **link_run,
          "ninja_query": ninja_link,
          "plugin": relative(plugin_path),
          "plugin_sha256": sha256(plugin_path) if plugin_path.is_file() else None,
      },
      "pass": (
          apply_check["returncode"] == 0 and
          patch_run["returncode"] == 0 and
          compile_run["returncode"] == 0 and object_path.is_file() and
          lm_head_apply_check["returncode"] == 0 and
          lm_head_patch_run["returncode"] == 0 and
          lm_head_compile_run["returncode"] == 0 and
          lm_head_object_path.is_file() and
          archive_run["returncode"] == 0 and
          link_run["returncode"] == 0 and plugin_path.is_file()),
      "patch": patch_run,
  }
  write_json(raw / "plugin-build.json", payload)
  return payload


def load_graph_module() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_adaptive_compile_graph", GRAPH_MODULE)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load graph module: {GRAPH_MODULE}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def worker_main(config_path: Path) -> int:
  config = load_json(config_path)
  raw = Path(config["raw"])
  topk = int(config["topk"])
  packed_kv_variant = config.get("packed_kv_variant")
  plugin = Path(config["plugin"])
  result_path = raw / "worker-result.json"
  started = time.perf_counter_ns()
  try:
    import openvino as ov

    graph = load_graph_module()
    registry = raw / "candidate-plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
    core.set_property("GPU", {"CONFIG_FILE": str(CUSTOM_XML.resolve())})

    physical_hot_capacity = 1 + EXACT_HISTORY_CAPACITY
    hot_key_blocks = (
        physical_hot_capacity + KEY_TILE_TOKENS - 1) // KEY_TILE_TOKENS
    hot_key_storage_blocks = 3 * hot_key_blocks + 1
    specifications = (
        ("query", ov.Type.f16, [1, Q_HEADS, 1, HEAD_DIM]),
        ("hot_key_bits", ov.Type.i32,
         [1, KV_HEADS, hot_key_storage_blocks, HOT_KEY_WORDS_PER_BLOCK]),
        ("hot_value", ov.Type.f16,
         [1, KV_HEADS, physical_hot_capacity, HEAD_DIM]),
        ("current_key", ov.Type.f16, [1, KV_HEADS, 1, HEAD_DIM]),
        ("current_value", ov.Type.f16, [1, KV_HEADS, 1, HEAD_DIM]),
        ("cold_key", ov.Type.i8,
         [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, HEAD_DIM]),
        ("cold_value", ov.Type.i8,
         [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, HEAD_DIM]),
        ("cold_key_scale", ov.Type.i8,
         [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, SCALE_BYTES]),
        ("cold_value_scale", ov.Type.i8,
         [1, KV_HEADS, FIXED_COLD_CAPACITY + 1, SCALE_BYTES]),
        ("mask", ov.Type.f32, [1, 1, 1, 1]),
        ("eviction_shape", ov.Type.i8, [1, KV_HEADS, 1, HEAD_DIM]),
        ("eviction_count", ov.Type.i32, [1, 1, 1, 1]),
        ("decode_length", ov.Type.i32, [1, 1, 1, MAX_CHUNKS]),
    )
    parameters = [
        ov.opset13.parameter(shape, element_type, name=name)
        for name, element_type, shape in specifications]
    # The product graph is dynamic and therefore uses the GPU plugin's new
    # shape-inference path, which is also the path that supports multi-output
    # custom operations.  Keep every adaptive carrier statically specialized,
    # but retain one disconnected dynamic shape carrier so this compile-only
    # model exercises the same lowering path.
    dynamic_carrier = ov.opset13.parameter(
        ov.PartialShape([1, 1, -1, 1]), ov.Type.f32,
        name="dynamic_shape_infer_carrier")
    parameters.append(dynamic_carrier)
    operation_class = graph.adaptive_attention_custom_class(
        ov, topk, packed_kv_variant=packed_kv_variant)
    operation = operation_class([
        value.output(0) for value in parameters[:len(specifications)]])
    operation.set_friendly_name(f"iq36_adaptive_compile_top{topk}")
    results = [
        ov.opset13.result(operation.output(index))
        for index in range(operation.get_output_size())]
    results.append(ov.opset13.result(dynamic_carrier.output(0)))
    model = ov.Model(results, parameters, f"iq36_adaptive_top{topk}_compile")
    model.validate_nodes_and_infer_types()
    compile_started = time.perf_counter_ns()
    compiled = core.compile_model(model, "GPU", {
        "PERFORMANCE_HINT": "LATENCY",
        "ACTIVATIONS_SCALE_FACTOR": 0.0,
    })
    compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
    payload = {
        "compile_ms": compile_ms,
        "compiled_inputs": [
            {"name": value.get_any_name(), "shape": (
                 list(value.shape) if value.partial_shape.is_static else
                 str(value.partial_shape)),
             "type": str(value.element_type)}
            for value in compiled.inputs],
        "compiled_outputs": [
            {"shape": (
                 list(value.shape) if value.partial_shape.is_static else
                 str(value.partial_shape)),
             "type": str(value.element_type)}
            for value in compiled.outputs],
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "max_chunks": MAX_CHUNKS,
        "plugin": str(plugin.resolve()),
        "plugin_sha256": sha256(plugin),
        "packed_kv_variant": packed_kv_variant,
        "required_checks_passed": True,
        "topk": topk,
        "worker_executed_inference": False,
    }
    write_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0
  except Exception as exc:  # compile diagnostics belong in the artifact
    payload = {
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "error": f"{type(exc).__name__}: {exc}",
        "required_checks_passed": False,
        "topk": topk,
        "worker_executed_inference": False,
    }
    write_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 2


def run_compile_worker(
    topk: int, plugin: Path, raw: Path, timeout_s: int, stop_bytes: int,
    memory: list[dict[str, Any]],
) -> dict[str, Any]:
  worker = raw / f"top{topk}"
  worker.mkdir()
  cache = worker / "compiler-cache"
  cache.mkdir()
  config = worker / "worker-config.json"
  write_json(config, {
      "plugin": str(plugin.resolve()),
      "raw": str(worker.resolve()),
      "topk": topk,
  })
  environment = os.environ.copy()
  environment.update({
      "NEO_CACHE_DIR": str(cache.resolve()),
  })
  sample_memory(f"before-top{topk}-compile", stop_bytes, memory)
  run = run_command([
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(config.resolve()),
      "--timeout-s", str(timeout_s),
  ], timeout_s, environment=environment)
  sample_memory(f"after-top{topk}-compile", stop_bytes, memory)
  payload = {
      **run,
      "environment": {
          "NEO_CACHE_DIR": environment["NEO_CACHE_DIR"]},
      "result": (
          load_json(worker / "worker-result.json")
          if (worker / "worker-result.json").is_file() else {}),
      "topk": topk,
  }
  write_json(worker / "worker-command.json", payload)
  return payload


def kernel_blocks(text: str) -> dict[str, str]:
  main = text.split("kernels_misc_info:", maxsplit=1)[0]
  matches = list(re.finditer(r"^  - name:\s+(\S+)\s*$", main, re.MULTILINE))
  blocks = {}
  for index, match in enumerate(matches):
    end = matches[index + 1].start() if index + 1 < len(matches) else len(main)
    blocks[match.group(1)] = main[match.end():end]
  return blocks


def integer_field(block: str, name: str, default: int = 0) -> int:
  match = re.search(
      rf"^\s+{re.escape(name)}:\s+(\d+)\s*$", block, re.MULTILINE)
  return int(match.group(1)) if match else default


def list_field(block: str, name: str) -> list[int]:
  match = re.search(
      rf"^\s+{re.escape(name)}:\s+\[\s*([^]]+)\]\s*$",
      block, re.MULTILINE)
  return [int(value.strip()) for value in match.group(1).split(",")] \
      if match else []


def scratch_bytes(block: str) -> int:
  return sum(int(value) for value in re.findall(
      r"^\s+- type:\s+scratch\s*$"
      r".*?^\s+size:\s+(\d+)\s*$",
      block, re.MULTILINE | re.DOTALL))


def stage_from_options(options: str) -> int | None:
  match = re.search(r"(?:^|\s)-DIQ36_ADAPTIVE_STAGE=(\d+)(?:\s|$)", options)
  return int(match.group(1)) if match else None


def disassemble_programs(
    topk: int, worker: Path, timeout_s: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  resources = []
  disassembly_runs = []
  cache_files = sorted(
      (worker / "compiler-cache").glob("*/*/*.l0_cache"))
  for ordinal, binary in enumerate(cache_files):
    destination = worker / "disassembly" / binary.stem
    destination.mkdir(parents=True)
    run = run_command([
        str(OCLOC), "disasm", "-file", str(binary),
        "-dump", str(destination),
    ], timeout_s)
    disassembly_runs.append({
        **run, "binary": relative(binary), "ordinal": ordinal,
        "topk": topk,
    })
    ze_info = destination / ".ze_info"
    if run["returncode"] != 0 or not ze_info.is_file():
      continue
    blocks = kernel_blocks(ze_info.read_text(encoding="utf-8"))
    user_blocks = {
        name: block for name, block in blocks.items()
        if name in EXPECTED_ENTRIES}
    options_path = destination / ".misc.buildOptions"
    options = (
        options_path.read_text(encoding="utf-8", errors="replace")
        if options_path.is_file() else "")
    for name, block in user_blocks.items():
      reported_spill = max(
          integer_field(block, "spill_mem_size"),
          integer_field(block, "spill_size"))
      resource = {
          "barrier_count": integer_field(block, "barrier_count"),
          "binary": relative(binary),
          "binary_bytes": binary.stat().st_size if binary.is_file() else 0,
          "binary_sha256": sha256(binary) if binary.is_file() else None,
          "eu_thread_count": integer_field(block, "eu_thread_count"),
          "grf_count": integer_field(block, "grf_count"),
          "kernel": name,
          "max_chunks": MAX_CHUNKS,
          "options": options,
          "ordinal": ordinal,
          "pointer_argument_count": len(re.findall(
              r"arg_type:\s+arg_bypointer", block)),
          "private_size": integer_field(block, "private_size"),
          "required_work_group_size": list_field(
              block, "required_work_group_size"),
          "simd_size": integer_field(block, "simd_size"),
          "slm_size": integer_field(block, "slm_size"),
          "source": relative(
              ROOT / "engine/openvino/custom/"
              "iq36_adaptive_attention_decode.cl"),
          "source_sha256": sha256(
              ROOT / "engine/openvino/custom/"
              "iq36_adaptive_attention_decode.cl"),
          "spill_mem_size": max(reported_spill, scratch_bytes(block)),
          "stage": stage_from_options(options),
          "topk": topk,
      }
      resources.append(resource)
  return resources, disassembly_runs


def summary_markdown(payload: dict[str, Any]) -> str:
  resources = payload["resources"]
  lines = [
      "# Adaptive-attention compile gate",
      "",
      f"- Verdict: `{payload['verdict']}`",
      f"- Required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- Commit: `{payload['git']['commit']}`",
      f"- Isolated plugin SHA256: `{payload['plugin_sha256']}`",
      "- Inference requests executed: `0`",
      "",
      "| top-k | stage | SIMD | GRF | EU threads | SLM B | barriers | private B | spill B | ABI pointers |",
      "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in sorted(resources, key=lambda value: (
      int(value["topk"]), int(value.get("stage") or 0))):
    lines.append(
        f"| {row['topk']} | `{row['kernel']}` | {row['simd_size']} | "
        f"{row['grf_count']} | {row['eu_thread_count']} | {row['slm_size']} | "
        f"{row['barrier_count']} | {row['private_size']} | "
        f"{row['spill_mem_size']} | "
        f"{row['pointer_argument_count']} |")
  lines.extend([
      "",
      "This gate compiles and disassembles only. It does not execute a layer,",
      "model, decode loop, or token-producing worker.",
      "",
  ])
  return "\n".join(lines)


def orchestrator_main(args: argparse.Namespace) -> int:
  assert args.output is not None
  out = args.output.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  git = git_state(out)
  stop_bytes = int(args.memory_stop_gib * 1024 ** 3)
  memory = []
  sample_memory("start", stop_bytes, memory)
  source_gate = load_json(SOURCE_GATE)
  pinned_head = git_output(OV_SOURCE, "rev-parse", "HEAD").strip()

  plugin_build = build_isolated_plugin(
      raw, args.timeout_s, stop_bytes, memory)
  plugin_path = Path(str(plugin_build["link"]["plugin"]))
  if not plugin_path.is_absolute():
    plugin_path = ROOT / plugin_path
  workers = []
  if plugin_build["pass"]:
    for topk in (512, 256):
      workers.append(run_compile_worker(
          topk, plugin_path, raw, args.timeout_s, stop_bytes, memory))

  resources = []
  disassembly = []
  codegen_summary = {}
  for topk in (512, 256):
    worker_dir = raw / f"top{topk}"
    if not worker_dir.is_dir():
      codegen_summary[str(topk)] = {
          "cache_files": [], "resource_rows": []}
      continue
    top_resources, top_disassembly = disassemble_programs(
        topk, worker_dir, args.timeout_s)
    resources.extend(top_resources)
    disassembly.extend(top_disassembly)
    codegen_summary[str(topk)] = {
        "cache_files": [
            {"bytes": path.stat().st_size, "path": relative(path),
             "sha256": sha256(path)}
            for path in sorted(
                (worker_dir / "compiler-cache").glob("*/*/*.l0_cache"))],
        "resource_rows": top_resources,
    }
  sample_memory("finish", stop_bytes, memory)

  worker_pass = (
      len(workers) == 2 and
      all(row["returncode"] == 0 and
          row.get("result", {}).get("required_checks_passed") is True and
          row.get("result", {}).get("worker_executed_inference") is False
          for row in workers))
  stage_compile_pass = all(
      len([row for row in resources if row["topk"] == topk]) == 4 and
      {row["kernel"] for row in resources if row["topk"] == topk}
          == set(EXPECTED_ENTRIES) and
      {row["stage"] for row in resources if row["topk"] == topk}
          == {1, 2, 3, 4} and
      all(f"-DIQ36_ADAPTIVE_TOPK={topk}U" in row["options"]
          for row in resources if row["topk"] == topk)
      for topk in (512, 256))
  resource_pass = (
      len(resources) == 8 and
      all(row["kernel"] in EXPECTED_ENTRIES and
          row["stage"] == EXPECTED_ENTRIES.index(row["kernel"]) + 1 and
          row["pointer_argument_count"] == 19 and
          row["required_work_group_size"] == EXPECTED_WORKGROUPS[row["kernel"]] and
          row["simd_size"] == EXPECTED_SIMD[row["kernel"]] and
          0 < row["grf_count"] <= 256 and
          row["slm_size"] <= 65536 and
          row["private_size"] == 0 and
          row["spill_mem_size"] == 0
          for row in resources) and
      {row["topk"] for row in resources} == {256, 512} and
      all({row["kernel"] for row in resources if row["topk"] == topk}
          == set(EXPECTED_ENTRIES) for topk in (256, 512)))
  source_gate_admitted = (
      source_gate.get("required_checks_passed") is True and
      source_gate.get("compile_admitted") is True and
      source_gate.get("one_layer_worker_admitted") is False)
  memory_pass = bool(memory) and all(
      int(row["available_bytes"]) >= stop_bytes for row in memory)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], value=git),
      check("seq1726_admits_compile_only", source_gate_admitted),
      check("pinned_openvino_source_is_exact",
            pinned_head.startswith(PINNED_COMMIT), head=pinned_head),
      check("patch_applies_to_pinned_current_source",
            plugin_build["apply_check"]["returncode"] == 0),
      check("patched_custom_primitive_compiles_with_werror",
            plugin_build["compile"]["returncode"] == 0),
      check("threshold11_lm_head_patch_applies_and_compiles_with_werror",
            plugin_build["lm_head"]["apply_check"]["returncode"] == 0 and
            plugin_build["lm_head"]["patch"]["returncode"] == 0 and
            plugin_build["lm_head"]["compile"]["returncode"] == 0),
      check("isolated_candidate_plugin_links", plugin_build["pass"],
            plugin=plugin_build["link"]["plugin"],
            sha256=plugin_build["link"]["plugin_sha256"]),
      check("top512_and_top256_compile_without_inference", worker_pass),
      check("each_topk_builds_exactly_four_ordered_stage_programs",
            stage_compile_pass),
      check("all_eight_native_stage_binaries_disassemble", len(resources) == 8),
      check("all_stage_abis_and_workgroups_are_exact", resource_pass),
      check("all_stage_kernels_are_spill_free", bool(resources) and all(
          row["spill_mem_size"] == 0 for row in resources)),
      check("all_stage_kernels_avoid_private_scratch", bool(resources) and all(
          row["private_size"] == 0 for row in resources)),
      check("all_stage_resources_fit_ptl", bool(resources) and all(
          row["simd_size"] == EXPECTED_SIMD[row["kernel"]] and
          0 < row["grf_count"] <= 256 and
          row["slm_size"] <= 65536 for row in resources)),
      check("memory_guard_never_tripped", memory_pass,
            stop_bytes=stop_bytes, samples=memory),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_adaptive_attention_layer3_boundary"
      if required_checks_passed else
      "repair_adaptive_attention_compile_or_resources")
  payload = {
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "layer3_boundary_worker_admitted": required_checks_passed,
      "long_worker_admitted": False,
      "memory": memory,
      "model_worker_admitted": False,
      "next_route": NEXT_ROUTE if required_checks_passed else ROUTE,
      "plugin_sha256": plugin_build["link"]["plugin_sha256"],
      "product_worker_admitted": False,
      "required_checks_passed": required_checks_passed,
      "resources": resources,
      "route": ROUTE,
      "schema_version": SCHEMA,
      "source_gate": relative(SOURCE_GATE),
      "codegen": codegen_summary,
      "verdict": verdict,
      "worker_count": len(workers),
      "worker_inference_count": 0,
      "workers": workers,
      "workstream": WS,
  }
  write_json(raw / "disassembly.json", disassembly)
  write_json(raw / "resource-metrics.json", resources)
  write_json(raw / "codegen-summary.json", codegen_summary)
  write_json(out / "gate.json", payload)
  write_jsonl(out / "metrics.jsonl", resources)
  (out / "summary.md").write_text(
      summary_markdown(payload), encoding="utf-8")
  print(json.dumps({
      "layer3_boundary_worker_admitted": required_checks_passed,
      "output": relative(out),
      "required_checks_passed": required_checks_passed,
      "resource_rows": len(resources),
      "verdict": verdict,
  }, sort_keys=True), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  parsed = parse_args()
  if parsed.worker_config is not None:
    raise SystemExit(worker_main(parsed.worker_config.resolve()))
  raise SystemExit(orchestrator_main(parsed))
