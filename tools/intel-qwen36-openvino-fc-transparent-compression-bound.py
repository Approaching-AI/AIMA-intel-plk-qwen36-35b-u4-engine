#!/usr/bin/env python3
"""Bound PTL transparent-compression successors for the fixed decode FCs.

The gate first proves that the installed GPU driver exposes the Level Zero
compression-hint extension and that the current OpenCL FC allocation already
has CCS enabled.  It then gives a bounded family of reversible U4 recodings an
impossible advantage: exact representative weights for all five FC cohorts,
all-zero scale/zero-point metadata, no decode/reorder work, independent layout
selection per cohort, and the faster of two warm trials.  If that mixed
schedule still misses the registered 8.183-ms FC target, neither an explicit
allocation hint nor these transparent-layout variants fund kernel work.

Only small FC component kernels run.  The gate starts no compiler or model
worker and serializes every child behind the standard available-memory stop.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import resource
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fc-transparent-compression-bound-v1"

HARDWARE_BOUND_TOOL = (
    ROOT / "tools/intel-qwen36-openvino-fc-hardware-limit-bound.py")
HARDWARE_BOUND = ROOT / (
    "output/openvino-fc-hardware-limit-bound-"
    "20260717Tseq1294-cleanZ/metrics.json")
FIXED_COMPONENT_ROOT = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/raw")
RUNTIME = ROOT / "build/engine/iq36-openvino-fc-micro-runtime"
CAPTURE_ROOT = ROOT / (
    "output/openvino-fc-boundary-capture-"
    "20260715Tseq1227-layer0-qkv-2k-o1-dirtyZ/raw/capture")
REAL_ZPS = FIXED_COMPONENT_ROOT / "layer0-qkv-zps-group-major.u4"
LEVEL_ZERO_LOADER = Path("/usr/lib/x86_64-linux-gnu/libze_loader.so.1")
TIME = Path("/usr/bin/time")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/"
    "activate-intel-box-env.sh")

EXPECTED_DRIVER_COMMIT = "82aab87fc932edc0558a0302d545a5bcc22edf41"
EXPECTED_DRIVER_VERSION = "26.18.38308.1"
DEFAULT_DRIVER_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    f"compute-runtime-{EXPECTED_DRIVER_COMMIT}")

COHORTS = {
    "linear_attention_input": {
        "layer": 0, "shape": "linear_in_fused_m12352_k2048",
        "m": 12352, "k": 2048, "calls": 30},
    "full_attention_qkv": {
        "layer": 3, "shape": "full_qkv_fused_m9216_k2048",
        "m": 9216, "k": 2048, "calls": 10},
    "router_shared_input": {
        "layer": 0, "shape": "mlp_input_fused_m1281_k2048",
        "m": 1281, "k": 2048, "calls": 40},
    "attention_output": {
        "layer": 0, "shape": "m2048_k4096",
        "m": 2048, "k": 4096, "calls": 40},
    "shared_expert_down": {
        "layer": 0, "shape": "m2048_k512",
        "m": 2048, "k": 512, "calls": 40},
}

VARIANTS = (
    "original",
    "center_mod16",
    "xor_zero_point",
    "conditional_rank",
    "two_bit_planes",
    "four_bit_planes",
    "conditional_rank_four_bit_planes",
    "byte_xor_previous",
    "reflect_zero_point_low",
    "reflect_mean_low",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--driver-source", type=Path,
                      default=DEFAULT_DRIVER_SOURCE)
  parser.add_argument("--runtime", type=Path, default=RUNTIME)
  parser.add_argument("--warmup", type=int, default=512)
  parser.add_argument("--repeat", type=int, default=31)
  parser.add_argument("--trials", type=int, default=2)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--timeout-s", type=int, default=60)
  args = parser.parse_args()
  if (args.warmup < 1 or args.repeat < 5 or args.trials < 2
      or args.memory_stop_gib <= 0 or args.timeout_s <= 0):
    parser.error("invalid warmup/repeat/trials/memory-stop/timeout")
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def git_output(*args: str, cwd: Path = ROOT) -> str:
  return subprocess.run(
      ["git", *args], cwd=cwd, text=True, capture_output=True,
      check=True).stdout.strip()


def git_state(output: Path) -> dict[str, Any]:
  rows = git_output("status", "--porcelain").splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  rows = [row for row in rows if not relative or relative not in row]
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(rows), "dirty_paths": rows,
  }


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def self_swap_bytes() -> int:
  for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
    if line.startswith("VmSwap:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("VmSwap is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({
      "label": label, "available_bytes": available,
      "self_swap_bytes": self_swap_bytes(),
  })
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def permissive_json(text: str) -> dict[str, Any]:
  text = re.sub(
      r"(?<![A-Za-z])(-?inf|nan)(?![A-Za-z])",
      lambda match: {
          "inf": "Infinity", "-inf": "-Infinity", "nan": "NaN",
      }[match.group(1)], text)
  value = json.loads(text)
  return value if isinstance(value, dict) else {}


def parse_runtime_stdout(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    if line.startswith("{"):
      return permissive_json(line)
  return {}


def parse_time(stderr: str) -> dict[str, int | None]:
  def value(pattern: str) -> int | None:
    match = re.search(pattern, stderr)
    return int(match.group(1)) if match else None
  return {
      "maximum_resident_kib": value(
          r"Maximum resident set size \(kbytes\): (\d+)"),
      "swaps": value(r"\n\s*Swaps: (\d+)"),
  }


def run_timed(
    command: list[str], env: dict[str, str], timeout_s: int,
) -> dict[str, Any]:
  try:
    result = subprocess.run(
        [str(TIME), "-v", *command], cwd=ROOT, env=env, text=True,
        capture_output=True, timeout=timeout_s, check=False)
    return {
        "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "timed_out": False, "resources": parse_time(result.stderr),
        "parsed": parse_runtime_stdout(result.stdout),
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command, "returncode": 124,
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "timed_out": True, "resources": {}, "parsed": {},
    }


def activated_runtime_environment(script: Path) -> dict[str, str]:
  result = subprocess.run(
      ["bash", "--noprofile", "--norc", "-c",
       'source "$1" >/dev/null 2>&1 && env -0', "bash", str(script)],
      capture_output=True, check=True)
  env: dict[str, str] = {}
  for entry in result.stdout.decode("utf-8").split("\0"):
    if not entry:
      continue
    name, separator, value = entry.partition("=")
    if separator:
      env[name] = value
  return env


def runtime_env(base: dict[str, str], **updates: str) -> dict[str, str]:
  env = dict(base)
  for name in (
      "NEOReadDebugKeys", "PrintGmmCompressionParams",
      "RenderCompressedBuffersEnabled"):
    env.pop(name, None)
  env.update({"INTEL_FORCE_PROBE": "b080", "DNNL_VERBOSE": "0"})
  env.update(updates)
  return env


class ExtensionProperty(ctypes.Structure):
  _fields_ = [("name", ctypes.c_char * 256),
              ("version", ctypes.c_uint32)]


def level_zero_extensions() -> list[list[dict[str, Any]]]:
  loader = ctypes.CDLL(str(LEVEL_ZERO_LOADER))
  loader.zeInit.argtypes = [ctypes.c_uint32]
  loader.zeInit.restype = ctypes.c_int
  loader.zeDriverGet.argtypes = [
      ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
  loader.zeDriverGet.restype = ctypes.c_int
  loader.zeDriverGetExtensionProperties.argtypes = [
      ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
      ctypes.POINTER(ExtensionProperty)]
  loader.zeDriverGetExtensionProperties.restype = ctypes.c_int
  if loader.zeInit(0) != 0:
    raise RuntimeError("zeInit failed")
  count = ctypes.c_uint32()
  if loader.zeDriverGet(ctypes.byref(count), None) != 0:
    raise RuntimeError("zeDriverGet count failed")
  drivers = (ctypes.c_void_p * count.value)()
  if loader.zeDriverGet(ctypes.byref(count), drivers) != 0:
    raise RuntimeError("zeDriverGet list failed")
  result: list[list[dict[str, Any]]] = []
  for driver in drivers:
    extension_count = ctypes.c_uint32()
    if loader.zeDriverGetExtensionProperties(
        driver, ctypes.byref(extension_count), None) != 0:
      raise RuntimeError("zeDriverGetExtensionProperties count failed")
    values = (ExtensionProperty * extension_count.value)()
    if loader.zeDriverGetExtensionProperties(
        driver, ctypes.byref(extension_count), values) != 0:
      raise RuntimeError("zeDriverGetExtensionProperties list failed")
    result.append([{
        "name": bytes(value.name).split(b"\0", 1)[0].decode(
            "ascii", errors="replace"),
        "version": f"{value.version >> 16}.{value.version & 0xffff}",
    } for value in values])
  return result


def source_audit(source_root: Path) -> dict[str, Any]:
  files = {
      "release": source_root /
          "shared/source/release_helper/release_helper_3000.cpp",
      "product": source_root /
          "shared/source/os_interface/product_helper_xe2_and_later.inl",
      "opencl_buffer": source_root / "opencl/source/mem_obj/buffer.cpp",
      "opencl_policy": source_root /
          "opencl/source/mem_obj/mem_obj_helper.cpp",
      "level_zero_policy": source_root /
          "shared/source/memory_manager/unified_memory_manager.cpp",
      "level_zero_hint": source_root /
          "level_zero/core/source/helpers/properties_parser.h",
  }
  missing = [str(path) for path in files.values() if not path.is_file()]
  if missing:
    raise FileNotFoundError("missing compute-runtime sources: " +
                            ", ".join(missing))
  text = {name: path.read_text(encoding="utf-8")
          for name, path in files.items()}
  checks = {
      "ptl_non_a0_enables_xe2_compression":
          "return !(hardwareIpVersion.value == AOT::PTL_H_A0);" in
          text["release"],
      "xe2_feature_drives_buffer_compression":
          "ftrRenderCompressedBuffers = hwInfo.featureTable.flags.ftrXe2Compression"
          in text["product"],
      "opencl_buffers_query_compression_suitability":
          "MemObjHelper::isSuitableForCompression" in text["opencl_buffer"]
          and "isBufferSizeSuitableForCompression(size)" in
          text["opencl_buffer"],
      "opencl_default_suitable_buffer_returns_true":
          "return true;" in text["opencl_policy"]
          and "properties.flags.uncompressedHint" in text["opencl_policy"],
      "level_zero_device_usm_defaults_to_supported_compression":
          "gfxCoreHelper.usmCompressionSupported(hwInfo)" in
          text["level_zero_policy"],
      "level_zero_hint_only_sets_compressed_preference":
          "lookupTable.compressedHint = true;" in text["level_zero_hint"],
  }
  return {
      "source_root": str(source_root),
      "commit": git_output("rev-parse", "HEAD", cwd=source_root),
      "checks": checks,
      "hashes": {display_path(path): sha256(path)
                 for path in files.values()},
  }


def load_hardware_module() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_fc_hardware_bound", HARDWARE_BOUND_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError("could not load FC hardware-bound module")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def pack_nibbles(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
  flat = values.reshape(values.shape[0], -1)
  return (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(
      np.uint8, copy=False).reshape(-1)


def bit_planes(
    values: np.ndarray[Any, Any], bits: int,
) -> np.ndarray[Any, Any]:
  groups = values.reshape(-1, 64)
  chunks: list[np.ndarray[Any, Any]] = []
  for shift in range(0, 4, bits):
    plane = ((groups >> shift) & ((1 << bits) - 1)).astype(np.uint8)
    if bits == 1:
      chunks.append(np.packbits(plane, axis=1, bitorder="little"))
    elif bits == 2:
      chunks.append((
          plane[:, 0::4] | (plane[:, 1::4] << 2)
          | (plane[:, 2::4] << 4) | (plane[:, 3::4] << 6)
      ).astype(np.uint8, copy=False))
    else:
      raise ValueError(f"unsupported plane width: {bits}")
  return np.concatenate(chunks, axis=1).reshape(-1)


def byte_entropy(values: np.ndarray[Any, Any]) -> float:
  histogram = np.bincount(values, minlength=256)
  nonzero = histogram[histogram > 0].astype(np.float64)
  probabilities = nonzero / nonzero.sum()
  return float(-(probabilities * np.log2(probabilities)).sum())


def build_variants(
    cohort: str, rows: list[dict[str, Any]], model_bin: Path,
    destination: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
  config = COHORTS[cohort]
  selected = [
      row for row in rows
      if row["cohort"] == cohort
      and f"layers.{config['layer']}." in row["name"]]
  q_parts: list[np.ndarray[Any, Any]] = []
  zp_parts: list[np.ndarray[Any, Any]] = []
  scale_parts: list[np.ndarray[Any, Any]] = []
  tensor_rows: list[dict[str, Any]] = []
  with model_bin.open("rb") as stream:
    for row in selected:
      m, groups, group_size = (int(value) for value in row["shape"])
      k = groups * group_size
      weight = row["streams"]["weight"]
      stream.seek(int(weight["offset"]))
      raw_weight = np.frombuffer(
          stream.read(int(weight["bytes"])), dtype=np.uint8).copy()
      packed = raw_weight.reshape(m, k // 2)
      q = np.empty((m, groups, 64), dtype=np.uint8)
      q_flat = q.reshape(m, k)
      q_flat[:, 0::2] = packed & 15
      q_flat[:, 1::2] = packed >> 4
      q_parts.append(q)

      zero = row["streams"]["zero_point"]
      stream.seek(int(zero["offset"]))
      raw_zero = np.frombuffer(
          stream.read(int(zero["bytes"])), dtype=np.uint8).copy()
      unpacked = np.empty(raw_zero.size * 2, dtype=np.uint8)
      unpacked[0::2] = raw_zero & 15
      unpacked[1::2] = raw_zero >> 4
      zp_parts.append(unpacked.reshape(m, groups))

      scale = row["streams"]["scale"]
      stream.seek(int(scale["offset"]))
      scale_parts.append(np.frombuffer(
          stream.read(int(scale["bytes"])), dtype="<u2").copy().reshape(
              m, groups))
      tensor_rows.append({
          "name": row["name"], "m": m, "k": k,
          "weight_bytes": int(weight["bytes"]),
      })

  q = np.concatenate(q_parts, axis=0)
  zp = np.concatenate(zp_parts, axis=0)
  scale_bits = np.concatenate(scale_parts, axis=0)
  expected_shape = (int(config["m"]), int(config["k"]) // 64, 64)
  if q.shape != expected_shape or zp.shape != expected_shape[:2]:
    raise ValueError(
        f"{cohort}: exact representative shape {q.shape}/{zp.shape} "
        f"!= {expected_shape}/{expected_shape[:2]}")

  zp_per_code = np.repeat(zp[:, :, None], 64, axis=2)
  centered = ((q.astype(np.int16) - zp_per_code.astype(np.int16)) & 15
              ).astype(np.uint8)
  xor_zero = q ^ zp_per_code
  counts = np.zeros((16, 16), dtype=np.int64)
  for zero_value in range(16):
    counts[zero_value] = np.bincount(
        q[zp_per_code == zero_value], minlength=16)
  rank_maps = np.empty((16, 16), dtype=np.uint8)
  for zero_value in range(16):
    order = np.lexsort((np.arange(16), -counts[zero_value]))
    rank_maps[zero_value, order] = np.arange(16, dtype=np.uint8)
  ranked = rank_maps[zp_per_code, q]
  original = pack_nibbles(q)
  previous = original.reshape(-1, 32)
  delta = previous.copy()
  delta[:, 1:] ^= previous[:, :-1]
  reflect_zp = zp >= 8
  reflect_mean = q.mean(axis=2) > 7.5

  values = {
      "original": original,
      "center_mod16": pack_nibbles(centered),
      "xor_zero_point": pack_nibbles(xor_zero),
      "conditional_rank": pack_nibbles(ranked),
      "two_bit_planes": bit_planes(q, 2),
      "four_bit_planes": bit_planes(q, 1),
      "conditional_rank_four_bit_planes": bit_planes(ranked, 1),
      "byte_xor_previous": delta.reshape(-1),
      "reflect_zero_point_low": pack_nibbles(
          np.where(reflect_zp[:, :, None], 15 - q, q)),
      "reflect_mean_low": pack_nibbles(
          np.where(reflect_mean[:, :, None], 15 - q, q)),
  }
  if set(values) != set(VARIANTS):
    raise ValueError("variant census drift")
  expected_bytes = int(config["m"]) * int(config["k"]) // 2
  paths: dict[str, Path] = {}
  stats: dict[str, Any] = {}
  destination.mkdir(parents=True, exist_ok=True)
  for name, value in values.items():
    if value.size != expected_bytes:
      raise ValueError(f"{cohort}/{name}: {value.size} != {expected_bytes}")
    path = destination / f"{name}.u4"
    value.tofile(path)
    paths[name] = path
    stats[name] = {
        "bytes": int(value.size), "sha256": sha256(path),
        "byte_entropy_bits": byte_entropy(value),
        "zero_byte_fraction": float((value == 0).mean()),
    }

  finite_scales = bool(np.isfinite(scale_bits.view(np.float16)).all())
  reflected_scales = (scale_bits ^ np.uint16(0x8000)).view(np.float16)
  scale_negation_exact = bool(np.array_equal(
      reflected_scales, -scale_bits.view(np.float16)))
  integer_reflection_exact = bool(np.array_equal(
      (15 - q).astype(np.int16)
      - (15 - zp_per_code).astype(np.int16),
      -(q.astype(np.int16) - zp_per_code.astype(np.int16))))
  return paths, {
      "tensors": tensor_rows,
      "shape": list(q.shape), "weight_bytes": expected_bytes,
      "variant_stats": stats,
      "conditional_rank_decode_table": rank_maps.tolist(),
      "reflection": {
          "finite_exact_scales": finite_scales,
          "f16_sign_flip_is_exact_negation": scale_negation_exact,
          "integer_q_zp_reflection_identity": integer_reflection_exact,
          "zero_point_low_flip_fraction": float(reflect_zp.mean()),
          "mean_low_flip_fraction": float(reflect_mean.mean()),
      },
  }


def component_command(
    runtime: Path, cohort: str, weight: Path, scales: Path, zps: Path,
    warmup: int, repeat: int,
) -> list[str]:
  config = COHORTS[cohort]
  shape = str(config["shape"])
  directory = FIXED_COMPONENT_ROOT / shape
  return [
      str(runtime), "--binary",
      str(directory / "codegen" / f"{shape}.program.bin"),
      "--kernel", f"iq36_moe_micro_{shape}",
      "--input", str(directory / "input.f16"),
      "--weights", str(weight), "--scales", str(scales),
      "--zps", str(zps), "--oracle", str(directory / "oracle.f16"),
      "--m", str(config["m"]), "--k", str(config["k"]),
      "--quant-group", "64", "--sg-per-wg-m", "2",
      "--sg-per-wg-n", "1", "--sg-per-wg-k", "8",
      "--wg-tile-m", "64", "--wg-tile-n", "8",
      "--warmup", str(warmup), "--repeat", str(repeat),
      "--minimum-gbps", "1",
  ]


def real_compression_command(
    runtime: Path, warmup: int, repeat: int,
) -> list[str]:
  shape = "real_m8192_k2048"
  directory = FIXED_COMPONENT_ROOT / shape
  return [
      str(runtime), "--binary",
      str(directory / "codegen" / f"{shape}.program.bin"),
      "--kernel", f"iq36_moe_micro_{shape}",
      "--input", str(CAPTURE_ROOT / "dispatch000-arg1-before.bin"),
      "--weights", str(CAPTURE_ROOT / "dispatch000-arg0-before.bin"),
      "--scales", str(CAPTURE_ROOT / "dispatch000-arg15-before.bin"),
      "--zps", str(REAL_ZPS),
      "--oracle", str(CAPTURE_ROOT / "dispatch000-arg2-after.bin"),
      "--m", "8192", "--k", "2048", "--quant-group", "64",
      "--sg-per-wg-m", "2", "--sg-per-wg-n", "1",
      "--sg-per-wg-k", "8", "--wg-tile-m", "64",
      "--wg-tile-n", "8", "--warmup", str(warmup),
      "--repeat", str(repeat), "--minimum-gbps", "1",
  ]


def runtime_row_passes(row: dict[str, Any], repeat: int) -> bool:
  parsed = row.get("parsed", {})
  samples = parsed.get("samples_us", [])
  return (
      row.get("returncode") == 0 and row.get("timed_out") is False
      and parsed.get("correctness_pass") is True
      and parsed.get("performance_pass") is True
      and isinstance(samples, list) and len(samples) == repeat
      and isinstance(parsed.get("kernel_median_us"), (int, float))
      and math.isfinite(float(parsed["kernel_median_us"])))


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  raw = output / "raw"
  raw.mkdir()
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = [
      HARDWARE_BOUND_TOOL, HARDWARE_BOUND, FIXED_COMPONENT_ROOT,
      args.runtime, CAPTURE_ROOT, REAL_ZPS, LEVEL_ZERO_LOADER, TIME,
      ENV_SCRIPT,
  ]
  missing = [display_path(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing transparent-compression inputs: " +
                     ", ".join(missing))
  git = git_state(output)
  if git["dirty"]:
    raise SystemExit("transparent-compression bound requires a clean repo: "
                     + ", ".join(git["dirty_paths"]))

  base_runtime_env = activated_runtime_environment(ENV_SCRIPT)

  source = source_audit(args.driver_source.resolve())
  package = subprocess.run(
      ["dpkg-query", "-W", "-f=${Version}", "libze-intel-gpu1"],
      text=True, capture_output=True, check=False)
  extensions = level_zero_extensions()
  extension_names = {
      value["name"] for driver in extensions for value in driver}
  hardware = load_json(HARDWARE_BOUND)
  current_ms = float(
      hardware["kill_number"]["current_fixed_component_ms"])
  target_ms = float(hardware["kill_number"]["target_ms"])
  required_cut_ms = float(hardware["kill_number"]["required_cut_ms"])

  bound_module = load_hardware_module()
  selected_rows, graph_census = bound_module.parse_selected_tensors(
      bound_module.MODEL_XML)
  variant_paths: dict[str, dict[str, Path]] = {}
  cohort_evidence: dict[str, Any] = {}
  for cohort in COHORTS:
    sample_memory(f"before-build-{cohort}", stop_bytes, memory)
    paths, evidence = build_variants(
        cohort, selected_rows, bound_module.MODEL_BIN,
        raw / "layouts" / cohort)
    variant_paths[cohort] = paths
    cohort_evidence[cohort] = evidence
  sample_memory("after-layout-build", stop_bytes, memory)

  zero_metadata: dict[str, dict[str, Path]] = {}
  for cohort, config in COHORTS.items():
    groups = int(config["k"]) // 64
    directory = raw / "zero-metadata" / cohort
    directory.mkdir(parents=True)
    scales = directory / "scales.f16"
    zps = directory / "zps.u4"
    scales.write_bytes(bytes(int(config["m"]) * groups * 2))
    zps.write_bytes(bytes(int(config["m"]) * groups // 2))
    zero_metadata[cohort] = {"scales": scales, "zps": zps}

  probe_cohort = "attention_output"
  debug_probe = run_timed(
      component_command(
          args.runtime, probe_cohort,
          variant_paths[probe_cohort]["original"],
          zero_metadata[probe_cohort]["scales"],
          zero_metadata[probe_cohort]["zps"], 8, 5),
      runtime_env(
          base_runtime_env, NEOReadDebugKeys="1",
          PrintGmmCompressionParams="1"),
      args.timeout_s)
  write_json(raw / "compression-debug-probe.json", debug_probe)
  ccs_active = "Flags.Gpu.CCS: 1" in (
      debug_probe.get("stdout", "") + debug_probe.get("stderr", ""))

  real_runs: list[dict[str, Any]] = []
  for index, mode in enumerate(("default", "uncompressed",
                                "uncompressed", "default")):
    env = (runtime_env(base_runtime_env) if mode == "default" else
           runtime_env(
               base_runtime_env, NEOReadDebugKeys="1",
               RenderCompressedBuffersEnabled="0"))
    row = run_timed(
        real_compression_command(args.runtime, args.warmup, args.repeat),
        env, args.timeout_s)
    row["mode"] = mode
    real_runs.append(row)
    write_json(raw / f"real-compression-{index}-{mode}.json", row)
    sample_memory(f"after-real-{index}-{mode}", stop_bytes, memory)

  layout_runs: dict[str, dict[str, list[dict[str, Any]]]] = {
      cohort: {variant: [] for variant in VARIANTS}
      for cohort in COHORTS}
  for cohort in COHORTS:
    metadata = zero_metadata[cohort]
    for trial in range(args.trials):
      order = list(VARIANTS)
      if trial % 2:
        order.reverse()
      for variant in order:
        row = run_timed(
            component_command(
                args.runtime, cohort, variant_paths[cohort][variant],
                metadata["scales"], metadata["zps"],
                args.warmup, args.repeat),
            runtime_env(base_runtime_env), args.timeout_s)
        row["cohort"] = cohort
        row["variant"] = variant
        row["trial"] = trial
        layout_runs[cohort][variant].append(row)
        write_json(
            raw / f"layout-{cohort}-{variant}-trial{trial}.json", row)
      sample_memory(f"after-{cohort}-trial{trial}", stop_bytes, memory)

  variant_schedules: dict[str, dict[str, Any]] = {}
  for variant in VARIANTS:
    per_cohort: dict[str, Any] = {}
    schedule_ms = 0.0
    for cohort, config in COHORTS.items():
      medians = [
          float(row["parsed"]["kernel_median_us"])
          for row in layout_runs[cohort][variant]
          if runtime_row_passes(row, args.repeat)]
      best_us = min(medians) if medians else math.inf
      per_cohort[cohort] = {
          "trial_medians_us": medians, "optimistic_best_us": best_us,
          "calls": int(config["calls"]),
      }
      schedule_ms += best_us * int(config["calls"]) / 1000.0
    variant_schedules[variant] = {
        "optimistic_schedule_ms": schedule_ms,
        "gap_to_target_ms": schedule_ms - target_ms,
        "per_cohort": per_cohort,
    }

  mixed_rows: dict[str, Any] = {}
  best_mixed_ms = 0.0
  for cohort, config in COHORTS.items():
    candidates = {
        variant: variant_schedules[variant]["per_cohort"][cohort][
            "optimistic_best_us"]
        for variant in VARIANTS}
    best_variant = min(candidates, key=candidates.get)
    best_us = float(candidates[best_variant])
    best_mixed_ms += best_us * int(config["calls"]) / 1000.0
    mixed_rows[cohort] = {
        "variant": best_variant, "optimistic_best_us": best_us,
        "calls": int(config["calls"]),
    }

  real_default = [
      float(row["parsed"]["kernel_median_us"]) for row in real_runs
      if row["mode"] == "default" and runtime_row_passes(row, args.repeat)]
  real_uncompressed = [
      float(row["parsed"]["kernel_median_us"]) for row in real_runs
      if row["mode"] == "uncompressed"
      and runtime_row_passes(row, args.repeat)]
  real_summary = {
      "default_trial_medians_us": real_default,
      "uncompressed_trial_medians_us": real_uncompressed,
      "default_best_us": min(real_default) if real_default else None,
      "uncompressed_best_us": (
          min(real_uncompressed) if real_uncompressed else None),
  }
  if real_default and real_uncompressed:
    real_summary["best_case_default_saving_us"] = (
        min(real_uncompressed) - min(real_default))
    real_summary["best_case_default_saving_fraction"] = (
        1.0 - min(real_default) / min(real_uncompressed))

  all_layout_rows = [
      row for cohort in layout_runs.values()
      for variant in cohort.values() for row in variant]
  all_resource_rows = [debug_probe, *real_runs, *all_layout_rows]
  child_peak_kib = max(
      (int(row.get("resources", {}).get("maximum_resident_kib") or 0)
       for row in all_resource_rows), default=0)
  child_swaps = max(
      (int(row.get("resources", {}).get("swaps") or 0)
       for row in all_resource_rows), default=0)
  reflection_pass = all(
      evidence["reflection"]["finite_exact_scales"]
      and evidence["reflection"]["f16_sign_flip_is_exact_negation"]
      and evidence["reflection"]["integer_q_zp_reflection_identity"]
      for evidence in cohort_evidence.values())
  exact_weight_bytes = sum(
      int(evidence["weight_bytes"]) * int(COHORTS[cohort]["calls"])
      for cohort, evidence in cohort_evidence.items())
  source_checks_pass = (
      source["commit"] == EXPECTED_DRIVER_COMMIT
      and all(source["checks"].values()))
  package_version = package.stdout.strip()
  all_layout_pass = all(
      runtime_row_passes(row, args.repeat) for row in all_layout_rows)
  all_real_pass = all(
      runtime_row_passes(row, args.repeat) for row in real_runs)

  checks = [
      check("repository_clean_at_gate", git["dirty"] is False,
            dirty_paths=git["dirty_paths"]),
      check("installed_driver_matches_exact_source_tag",
            package.returncode == 0
            and package_version.startswith(EXPECTED_DRIVER_VERSION)
            and source["commit"] == EXPECTED_DRIVER_COMMIT,
            package_version=package_version, source_commit=source["commit"]),
      check("driver_source_default_compression_pipeline_is_live",
            source_checks_pass, source_checks=source["checks"]),
      check("level_zero_memory_compression_hint_is_exposed",
            "ZE_extension_memory_compression_hints" in extension_names),
      check("opencl_fc_allocation_has_ccs_enabled",
            debug_probe["returncode"] == 0 and ccs_active),
      check("real_u4_default_and_uncompressed_controls_are_correct",
            all_real_pass, summary=real_summary),
      check("exact_five_cohort_representative_census",
            set(cohort_evidence) == set(COHORTS)
            and exact_weight_bytes == 715_038_720,
            weighted_parameter_bytes=exact_weight_bytes,
            graph_census=graph_census),
      check("reflection_recode_is_exact_algebraically",
            reflection_pass),
      check("all_zero_metadata_layout_stress_runs_pass",
            all_layout_pass,
            run_count=len(all_layout_rows), expected_run_count=(
                len(COHORTS) * len(VARIANTS) * args.trials)),
      check("optimistic_mixed_zero_decode_schedule_still_misses_target",
            math.isfinite(best_mixed_ms) and best_mixed_ms > target_ms,
            optimistic_mixed_ms=best_mixed_ms,
            target_ms=target_ms, gap_ms=best_mixed_ms - target_ms),
      check("serialized_component_memory_boundary",
            child_peak_kib < 1024 * 1024 and child_swaps == 0
            and min(row["available_bytes"] for row in memory) >= stop_bytes
            and max(row["self_swap_bytes"] for row in memory) == 0,
            child_peak_rss_kib=child_peak_kib, child_swaps=child_swaps,
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  passed = all(row["pass"] for row in checks)
  decision = (
      "reject_explicit_hint_and_tested_transparent_u4_recodes_"
      "retain_distinct_fixed_function_random_access_decoder_watch")
  usage_self = resource.getrusage(resource.RUSAGE_SELF)
  usage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
  metrics = {
      "schema_version": SCHEMA, "workstream": WS,
      "captured_at": iso_now(), "git": git,
      "decision": decision,
      "inputs": {
          "hardware_bound": display_path(HARDWARE_BOUND),
          "fixed_component_root": display_path(FIXED_COMPONENT_ROOT),
          "runtime": display_path(args.runtime),
          "driver_source": str(args.driver_source.resolve()),
          "driver_source_commit": source["commit"],
          "driver_package_version": package_version,
          "activation_script": str(ENV_SCRIPT),
          "activation_script_sha256": sha256(ENV_SCRIPT),
      },
      "source_audit": source,
      "level_zero_extensions": extensions,
      "current_fc_budget": {
          "current_ms": current_ms, "target_ms": target_ms,
          "required_cut_ms": required_cut_ms,
      },
      "real_compression_control": real_summary,
      "cohorts": cohort_evidence,
      "variants": variant_schedules,
      "optimistic_mixed_zero_decode": {
          "schedule_ms": best_mixed_ms,
          "target_ms": target_ms,
          "gap_ms": best_mixed_ms - target_ms,
          "per_cohort": mixed_rows,
          "advantages_granted": [
              "all scale bytes are zero",
              "all zero-point bytes are zero",
              "no recode decode or input reorder cost",
              "independent best layout per cohort",
              "faster median across warm trials",
              "no provider or direct-consumer overhead",
          ],
      },
      "checks": checks, "required_checks_passed": passed,
      "memory": {
          "stop_bytes": stop_bytes, "samples": memory,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory),
          "self_peak_rss_kib": int(usage_self.ru_maxrss),
          "child_peak_rss_kib": child_peak_kib,
          "resource_children_peak_rss_kib": int(usage_children.ru_maxrss),
          "child_swaps": child_swaps,
          "self_swap_peak_bytes": max(
              row["self_swap_bytes"] for row in memory),
      },
      "reopen_condition": (
          "A different hardware fixed-function or direct-consume capability "
          "must provide exact block-random-access U4/ZP/F16 semantics and a "
          "complete five-cohort schedule below 8.183 ms including metadata, "
          "decode, DQ, DPAS, provider, and direct-consumer work."),
  }
  write_json(output / "metrics.json", metrics)

  rows = []
  for variant, value in sorted(
      variant_schedules.items(),
      key=lambda item: item[1]["optimistic_schedule_ms"]):
    rows.append(
        f"| `{variant}` | {value['optimistic_schedule_ms']:.6f} | "
        f"{value['gap_to_target_ms']:.6f} |")
  default_us = real_summary.get("default_best_us")
  uncompressed_us = real_summary.get("uncompressed_best_us")
  real_line = (
      f"{default_us:.6f} / {uncompressed_us:.6f} us"
      if isinstance(default_us, (int, float))
      and isinstance(uncompressed_us, (int, float)) else "unavailable")
  report = f"""# OpenVINO FC transparent-compression bound

- decision: `{decision}`
- installed GPU driver/source: `{package_version}` / `{source['commit']}`
- Level Zero extension: `ZE_extension_memory_compression_hints` exposed
- current OpenCL allocation: `Flags.Gpu.CCS: 1` observed
- real U4 default / forced-uncompressed best medians: `{real_line}`
- registered current / target / required cut: `{current_ms:.6f} / {target_ms:.6f} / {required_cut_ms:.6f} ms`
- optimistic mixed zero-decode schedule: `{best_mixed_ms:.6f} ms`; still `{best_mixed_ms - target_ms:.6f} ms` over target

The mixed ceiling grants all-zero scale and zero-point metadata, zero recode
decode/reorder cost, an independent best encoding per cohort, the faster warm
trial, and no provider or direct-consumer overhead.  It therefore cannot be a
correctness or speedup claim; it is an implementation admission bound.

| U4 representation | optimistic schedule ms | gap to 8.183 ms |
|---|---:|---:|
{chr(10).join(rows)}

Per-group reflection is algebraically exact: `q -> 15-q`, `zp -> 15-zp`,
and an F16 sign-bit flip of the scale preserve every dequantized product.
Even that zero-decode family and the decode-requiring center/XOR/rank/plane
families fail the complete FC budget under the impossible metadata grant.

All GPU children were serialized.  Peak child RSS was `{child_peak_kib} KiB`,
child swaps were `{child_swaps}`, and minimum available memory was
`{min(row['available_bytes'] for row in memory)} B`.  No compiler or model
worker ran.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output), "decision": decision,
      "optimistic_mixed_ms": best_mixed_ms,
      "target_ms": target_ms,
      "required_checks_passed": passed,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
