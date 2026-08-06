#!/usr/bin/env python3
"""Gate independent fixed-FC streams against the retained fused carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fixed-fc-multi-output-component-gate-v0"
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
ONEDNN_SOURCE = OV_SOURCE / "src/plugins/intel_gpu/thirdparty/onednn_gpu"
ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
BUILD_DIR = ROOT / "build/engine"
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
FUSED_HOST = ROOT / "engine/gpu/opencl/openvino_fc_micro_host.cl"
MULTI_HOST = ROOT / "engine/gpu/opencl/openvino_fc_multi_output_host.cl"
SEQ1233 = ROOT / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ/metrics.json")
SEQ1353 = ROOT / (
    "output/openvino-fixed-fc-graph-integration-contract-audit-"
    "20260718Tseq1353-cleanZ/metrics.json")
CAPTURE = ROOT / (
    "output/openvino-fc-boundary-capture-20260715Tseq1227-"
    "layer0-qkv-2k-o1-dirtyZ/raw/capture")
PREPACKED_REAL_ZP = ROOT / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ/raw/"
    "layer0-qkv-zps-group-major.u4")

GROUP_SIZE = 64
EXPECTED_SHIM_SHA256 = (
    "a31a6e7ab718cd5f6df1b3b89ab496fac8315c3cf79ccae135e8fbb6d9b53e87")
EXPECTED_MICRO_SHA256 = (
    "a9e536491c082225df6a2f76f8cc7d4ffc06ced4118af39e936c0e8098027824")
COHORTS = (
    {"name": "real_m8192_k2048", "widths": [8192], "k": 2048,
     "count": 0, "source": "locked_layer0_qkv_capture"},
    {"name": "linear_input", "widths": [8192, 32, 32, 4096], "k": 2048,
     "count": 30, "source": "deterministic_nonzero_synthetic"},
    {"name": "full_qkv", "widths": [8192, 512, 512], "k": 2048,
     "count": 10, "source": "deterministic_nonzero_synthetic"},
    {"name": "router_shared", "widths": [1, 512, 512, 256], "k": 2048,
     "count": 40, "source": "deterministic_nonzero_synthetic"},
    {"name": "attention_output", "widths": [2048], "k": 4096,
     "count": 40, "source": "deterministic_nonzero_synthetic"},
    {"name": "shared_expert_down", "widths": [2048], "k": 512,
     "count": 40, "source": "deterministic_nonzero_synthetic"},
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--warmup", type=int, default=512)
  parser.add_argument("--repeat", type=int, default=31)
  parser.add_argument("--blocks", type=int, default=8)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  args = parser.parse_args()
  if (args.warmup < 1 or args.repeat < 5 or args.blocks < 8 or
      args.timeout_s <= 0 or args.memory_preflight_gib <= 0.0 or
      args.memory_stop_gib <= 0.0 or
      args.memory_preflight_gib < args.memory_stop_gib):
    parser.error("timing, timeout, and memory arguments are invalid")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-fixed-fc-multi-output-{stamp}"
  return args


def run(command: list[str], timeout_s: int,
        cwd: Path = ROOT) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True,
        timeout=timeout_s, check=False, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout or "", "stderr": error.stderr or "",
            "timed_out": True}


def run_intel(command: list[str], timeout_s: int) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 && export DNNL_VERBOSE=0 && " +
      shlex.join(command))
  return run(["bash", "-lc", shell], timeout_s)


def write_run(raw: Path, label: str, row: dict[str, Any]) -> None:
  (raw / f"{label}.command.json").write_text(
      json.dumps(row.get("command", []), indent=2) + "\n", encoding="utf-8")
  (raw / f"{label}.stdout").write_text(
      str(row.get("stdout", "")), encoding="utf-8")
  (raw / f"{label}.stderr").write_text(
      str(row.get("stderr", "")), encoding="utf-8")


def parse_json_line(row: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(row.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def git_output(*parts: str, cwd: Path = ROOT) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=cwd, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def memory_snapshot(label: str) -> dict[str, Any]:
  values: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, rest = line.split(":", 1)
    if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
      values[key] = int(rest.split()[0]) * 1024
  return {"label": label, "available_bytes": values["MemAvailable"],
          "swap_total_bytes": values.get("SwapTotal", 0),
          "swap_free_bytes": values.get("SwapFree", 0)}


def guard_memory(label: str, minimum_bytes: int,
                 samples: list[dict[str, Any]]) -> None:
  sample = memory_snapshot(label)
  samples.append(sample)
  if sample["available_bytes"] < minimum_bytes:
    raise RuntimeError(
        f"memory guard at {label}: {sample['available_bytes']} < {minimum_bytes}")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def pack_u4(values: np.ndarray) -> bytes:
  flat = np.asarray(values, dtype=np.uint8).reshape(-1)
  if flat.size % 2 != 0:
    raise RuntimeError("U4 logical stream must have even element count")
  packed = ((flat[0::2] & 0x0F) |
            ((flat[1::2] & 0x0F) << 4)).astype(np.uint8)
  return packed.tobytes()


def create_synthetic_streams(directory: Path, widths: list[int],
                             k: int, make_reference: bool = False
                             ) -> dict[str, Any]:
  rng = np.random.default_rng((sum(widths) << 16) ^ k ^ 0x4D554C54)
  input_path = directory / "input.f16"
  rng.uniform(-0.25, 0.25, k).astype(np.float16).tofile(input_path)
  groups = k // GROUP_SIZE
  weight_payloads: list[bytes] = []
  scale_arrays: list[np.ndarray] = []
  zp_arrays: list[np.ndarray] = []
  logical_weights: list[np.ndarray] = []
  weights: list[Path] = []
  scales: list[Path] = []
  zps: list[Path] = []
  for index, width in enumerate(widths):
    weight_bytes = rng.integers(
        0, 256, width * k // 2, dtype=np.uint8)
    weight = weight_bytes.tobytes()
    scale = rng.uniform(0.0005, 0.003, (groups, width)).astype(np.float16)
    zp = rng.integers(0, 16, (groups, width), dtype=np.uint8)
    weight_path = directory / f"weight{index}.u4"
    scale_path = directory / f"scale{index}.f16"
    zp_path = directory / f"zp{index}.u4"
    weight_path.write_bytes(weight)
    scale.tofile(scale_path)
    zp_path.write_bytes(pack_u4(zp))
    weight_payloads.append(weight)
    scale_arrays.append(scale)
    zp_arrays.append(zp)
    if make_reference:
      unpacked = np.empty((width, k), dtype=np.uint8)
      unpacked[:, 0::2] = weight_bytes.reshape(width, -1) & 0x0F
      unpacked[:, 1::2] = weight_bytes.reshape(width, -1) >> 4
      logical_weights.append(unpacked)
    weights.append(weight_path)
    scales.append(scale_path)
    zps.append(zp_path)
  fused_weights = directory / "fused-weights.u4"
  fused_scales = directory / "fused-scales.f16"
  fused_zps = directory / "fused-zps.u4"
  fused_weights.write_bytes(b"".join(weight_payloads))
  np.concatenate(scale_arrays, axis=1).tofile(fused_scales)
  fused_zps.write_bytes(pack_u4(np.concatenate(zp_arrays, axis=1)))
  reference: Path | None = None
  if make_reference:
    input_values = np.fromfile(input_path, dtype=np.float16).astype(np.float32)
    input_groups = input_values.reshape(groups, GROUP_SIZE)
    outputs: list[np.ndarray] = []
    for weight, scale, zp in zip(logical_weights, scale_arrays, zp_arrays):
      dequant = (
          weight.reshape(weight.shape[0], groups, GROUP_SIZE).astype(np.float32)
          - zp.T[:, :, None].astype(np.float32))
      dequant *= scale.T[:, :, None].astype(np.float32)
      outputs.append(np.sum(
          dequant * input_groups[None, :, :], axis=(1, 2),
          dtype=np.float32).astype(np.float16))
    reference = directory / "cpu-reference.f16"
    np.concatenate(outputs).tofile(reference)
  return {"input": input_path, "weights": weights, "scales": scales,
          "zps": zps, "fused_weights": fused_weights,
          "fused_scales": fused_scales, "fused_zps": fused_zps,
          "reference": reference}


def real_streams() -> dict[str, Any]:
  weight = CAPTURE / "dispatch000-arg0-before.bin"
  input_path = CAPTURE / "dispatch000-arg1-before.bin"
  scale = CAPTURE / "dispatch000-arg15-before.bin"
  reference = CAPTURE / "dispatch000-arg2-after.bin"
  return {"input": input_path, "weights": [weight], "scales": [scale],
          "zps": [PREPACKED_REAL_ZP], "fused_weights": weight,
          "fused_scales": scale, "fused_zps": PREPACKED_REAL_ZP,
          "reference": reference}


def codegen_build_command(binary: Path) -> list[str]:
  includes = [
      ONEDNN_SOURCE / "src/gpu/intel/gemm/jit",
      ONEDNN_SOURCE / "src/gpu/intel/gemm/jit/dnnl_gpu_intel_gemm_jit",
      ONEDNN_BUILD / "include", ONEDNN_SOURCE / "include",
      ONEDNN_SOURCE / "third_party/opencl", ONEDNN_SOURCE / "third_party",
      ONEDNN_SOURCE / "src", ONEDNN_SOURCE / "src/gpu/intel/jit/config",
      ONEDNN_SOURCE / "third_party/ngen",
      ONEDNN_SOURCE / "src/gpu/intel/gemm/jit/include",
  ]
  return [
      str(CXX), "-std=c++17", "-O3", "-DNDEBUG", "-fopenmp",
      "-fno-operator-names", "-DCL_TARGET_OPENCL_VERSION=120",
      "-DDNNL_X64=1", "-DGEMMSTONE_BUILD_12HP", "-DGEMMSTONE_BUILD_12LP",
      "-DGEMMSTONE_BUILD_12P7", "-DGEMMSTONE_BUILD_12P8",
      "-DGEMMSTONE_BUILD_XE2", "-DGEMMSTONE_BUILD_XE3",
      "-DGEMMSTONE_BUILD_XE3P", "-DGEMMSTONE_CONFIG", "-DNGEN_CONFIG",
      *[f"-I{path}" for path in includes], str(CODEGEN),
      str(ONEDNN_BUILD / "src/libdnnl.a"), "-lOpenCL", "-ldl", "-lpthread",
      "-o", str(binary),
  ]


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_samples: list[dict[str, Any]] = []
  guard_memory("preflight", preflight_bytes, memory_samples)

  required = [ENV_SCRIPT, CMAKE, CXX, CODEGEN, FUSED_HOST, MULTI_HOST,
              SEQ1233, SEQ1353, PREPACKED_REAL_ZP,
              ONEDNN_BUILD / "src/libdnnl.a",
              ROOT / "engine/boundaries.json"]
  required += [CAPTURE / name for name in (
      "dispatch000-arg0-before.bin", "dispatch000-arg1-before.bin",
      "dispatch000-arg15-before.bin", "dispatch000-arg2-after.bin")]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  onednn_commit = git_output("rev-parse", "HEAD", cwd=ONEDNN_SOURCE)
  onednn_dirty = git_output("status", "--porcelain", cwd=ONEDNN_SOURCE)
  seq1233 = load_json(SEQ1233)
  seq1353 = load_json(SEQ1353)
  arithmetic = seq1353["arithmetic"]
  anchor_schedule_ms = float(arithmetic["seq1233_fixed_schedule_ms"])
  allowed_overhead_ms = float(arithmetic["source_screen_margin_ms"])
  schedule_cap_ms = anchor_schedule_ms + allowed_overhead_ms

  configure = run([
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(args.build_dir)],
      args.timeout_s)
  write_run(raw, "configure", configure)
  engine_build = run([
      str(CMAKE), "--build", str(args.build_dir), "--target",
      "iq36-openvino-fc-multi-output-runtime", "-j1"], args.timeout_s)
  write_run(raw, "engine-build", engine_build)
  runtime = args.build_dir / "iq36-openvino-fc-multi-output-runtime"
  linkage = (run(["ldd", str(runtime)], args.timeout_s)
             if engine_build["returncode"] == 0 and runtime.exists() else
             {"command": ["ldd", str(runtime)], "returncode": 125,
              "stdout": "", "stderr": "runtime build failed",
              "timed_out": False})
  write_run(raw, "runtime-linkage", linkage)
  codegen_binary = raw / "openvino-fc-multi-codegen"
  codegen_build = run(codegen_build_command(codegen_binary), args.timeout_s)
  write_run(raw, "codegen-build", codegen_build)

  cohort_rows: list[dict[str, Any]] = []
  for cohort in COHORTS:
    name = str(cohort["name"])
    widths = [int(value) for value in cohort["widths"]]
    m = sum(widths)
    k = int(cohort["k"])
    shape_dir = raw / name
    baseline_dir = shape_dir / "baseline-codegen"
    candidate_dir = shape_dir / "candidate-codegen"
    shape_dir.mkdir(parents=True)
    streams = (real_streams() if cohort["source"] == "locked_layer0_qkv_capture"
               else create_synthetic_streams(
                   shape_dir, widths, k, make_reference=name == "router_shared"))

    common_codegen = [
        str(codegen_binary), "--decode-fc", "--shape-name", name,
        "--m", str(m), "--k", str(k)]
    baseline_command = [
        *common_codegen, "--dump-dir", str(baseline_dir),
        "--host-source", str(FUSED_HOST)]
    candidate_command = [
        *common_codegen, "--projection-widths", ",".join(map(str, widths)),
        "--dump-dir", str(candidate_dir), "--host-source", str(MULTI_HOST)]
    guard_memory(f"before-{name}-baseline-codegen", stop_bytes, memory_samples)
    baseline_run = (run_intel(baseline_command, args.timeout_s)
                    if codegen_build["returncode"] == 0 else
                    {"command": baseline_command, "returncode": 125,
                     "stdout": "", "stderr": "codegen build failed",
                     "timed_out": False})
    write_run(raw, f"{name}-baseline-codegen", baseline_run)
    guard_memory(f"after-{name}-baseline-codegen", stop_bytes, memory_samples)
    candidate_run = (run_intel(candidate_command, args.timeout_s)
                     if baseline_run["returncode"] == 0 else
                     {"command": candidate_command, "returncode": 125,
                      "stdout": "", "stderr": "baseline codegen failed",
                      "timed_out": False})
    write_run(raw, f"{name}-candidate-codegen", candidate_run)
    guard_memory(f"after-{name}-candidate-codegen", stop_bytes, memory_samples)
    baseline_codegen = parse_json_line(baseline_run)
    candidate_codegen = parse_json_line(candidate_run)
    baseline_package = (baseline_codegen.get("packages") or [{}])[0]
    candidate_package = (candidate_codegen.get("packages") or [{}])[0]
    settings = candidate_package.get("settings", {})

    runtime_command = [
        str(runtime), "--baseline-binary",
        str(baseline_dir / f"{name}.program.bin"), "--candidate-binary",
        str(candidate_dir / f"{name}.program.bin"), "--kernel",
        f"iq36_moe_micro_{name}", "--input", str(streams["input"]),
        "--baseline-weights", str(streams["fused_weights"]),
        "--baseline-scales", str(streams["fused_scales"]),
        "--baseline-zps", str(streams["fused_zps"]), "--weights",
        ",".join(str(path) for path in streams["weights"]), "--scales",
        ",".join(str(path) for path in streams["scales"]), "--zps",
        ",".join(str(path) for path in streams["zps"]), "--widths",
        ",".join(map(str, widths)), "--k", str(k), "--quant-group",
        str(GROUP_SIZE), "--sg-per-wg-m", str(settings.get("sg_per_wg_m", 0)),
        "--sg-per-wg-n", str(settings.get("sg_per_wg_n", 0)),
        "--sg-per-wg-k", str(settings.get("sg_per_wg_k", 0)),
        "--wg-tile-m", str(settings.get("wg_tile_m", 0)),
        "--wg-tile-n", str(settings.get("wg_tile_n", 0)),
        "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        "--blocks", str(args.blocks), "--actual-prefix",
        str(shape_dir / "actual")]
    if streams["reference"] is not None:
      runtime_command += ["--reference", str(streams["reference"])]
    if name == "router_shared":
      runtime_command += ["--allow-baseline-difference"]
    guard_memory(f"before-{name}-runtime", stop_bytes, memory_samples)
    runtime_run = (run_intel(runtime_command, args.timeout_s)
                   if candidate_run["returncode"] == 0 and
                   engine_build["returncode"] == 0 else
                   {"command": runtime_command, "returncode": 125,
                    "stdout": "", "stderr": "build/codegen failed",
                    "timed_out": False})
    write_run(raw, f"{name}-runtime", runtime_run)
    guard_memory(f"after-{name}-runtime", stop_bytes, memory_samples)
    result = parse_json_line(runtime_run)

    baseline_shim = baseline_dir / f"{name}.shim.cl"
    candidate_shim = candidate_dir / f"{name}.shim.cl"
    baseline_micro = baseline_dir / f"{name}.micro.bin"
    candidate_micro = candidate_dir / f"{name}.micro.bin"
    hashes = {
        "baseline_shim": sha256(baseline_shim) if baseline_shim.exists() else None,
        "candidate_shim": sha256(candidate_shim) if candidate_shim.exists() else None,
        "baseline_micro": sha256(baseline_micro) if baseline_micro.exists() else None,
        "candidate_micro": sha256(candidate_micro) if candidate_micro.exists() else None,
    }
    cohort_rows.append({**cohort, "m": m,
                        "baseline_codegen_returncode": baseline_run["returncode"],
                        "candidate_codegen_returncode": candidate_run["returncode"],
                        "runtime_returncode": runtime_run["returncode"],
                        "baseline_package": baseline_package,
                        "candidate_package": candidate_package,
                        "hashes": hashes, "runtime": result})

  schedule_samples_ms: list[float] = []
  overhead_samples_ms: list[float] = []
  sample_shape_ok = all(
      len(row["runtime"].get("blocks", [])) == args.blocks
      for row in cohort_rows)
  if sample_shape_ok:
    for block in range(args.blocks):
      overhead_ms = sum(
          int(row["count"]) *
          float(row["runtime"]["blocks"][block]["delta_us"]) / 1000.0
          for row in cohort_rows)
      overhead_samples_ms.append(overhead_ms)
      schedule_samples_ms.append(anchor_schedule_ms + overhead_ms)
  inference = (latency_cap_inference(
      schedule_samples_ms, cap=schedule_cap_ms, min_samples=8)
      if sample_shape_ok else {})
  ucb_ms = (float(inference["upper_confidence_bound_ms"])
            if "upper_confidence_bound_ms" in inference else None)
  overhead_ucb_ms = (ucb_ms - anchor_schedule_ms
                     if ucb_ms is not None else None)
  fixed_fc_saving_lcb_ms = (
      float(arithmetic["seq1233_stock_fc_ms"]) - ucb_ms
      if ucb_ms is not None else None)
  combined_saving_lcb_ms = (
      fixed_fc_saving_lcb_ms + float(arithmetic["seq1327_qk_saving_ms"])
      if fixed_fc_saving_lcb_ms is not None else None)
  kill_number_ms = float(arithmetic["kill_number_ms"])
  retained_margin_ms = (combined_saving_lcb_ms - kill_number_ms
                        if combined_saving_lcb_ms is not None else None)

  linkage_text = str(linkage.get("stdout", "")).lower()
  all_hashes = [value for row in cohort_rows
                for value in row["hashes"].values()]
  expected_globals = {
      "real_m8192_k2048": ([4096, 1, 8], [4096, 1, 8]),
      "linear_input": ([6176, 1, 8], [6208, 1, 8]),
      "full_qkv": ([4608, 1, 8], [4608, 1, 8]),
      "router_shared": ([672, 1, 8], [672, 1, 8]),
      "attention_output": ([1024, 1, 8], [1024, 1, 8]),
      "shared_expert_down": ([1024, 1, 8], [1024, 1, 8]),
  }
  universal_hashes = all(
      value == (EXPECTED_SHIM_SHA256 if "shim" in key else EXPECTED_MICRO_SHA256)
      for row in cohort_rows for key, value in row["hashes"].items())
  all_equivalent = all(
      row["runtime"].get("correctness_pass") is True and
      (row["name"] == "router_shared" or
       row["runtime"].get("baseline_candidate_compare", {}).get(
           "exact_rate") == 1.0)
      for row in cohort_rows)
  router_reference = next(
      row for row in cohort_rows if row["name"] == "router_shared"
  )["runtime"].get("candidate_reference_compare", {})
  real_reference = cohort_rows[0]["runtime"].get(
      "candidate_reference_compare", {})
  exact_workgroups = all(
      row["runtime"].get("baseline_global") == expected_globals[row["name"]][0]
      and row["runtime"].get("candidate_global") == expected_globals[row["name"]][1]
      and row["runtime"].get("local") == [32, 1, 8]
      for row in cohort_rows)
  no_spills = all(
      row["runtime"].get("baseline_spill_memory_bytes") == 0 and
      row["runtime"].get("candidate_spill_memory_bytes") == 0
      for row in cohort_rows)
  checks = [
      check("repository_clean_at_gate", not dirty, dirty=dirty),
      check("pinned_onednn_source_clean", not onednn_dirty,
            commit=onednn_commit, dirty=onednn_dirty),
      check("seq1353_source_admission_is_conclusive",
            seq1353.get("required_checks_passed") is True and
            seq1353.get("verdict") ==
            "admit_fixed_fc_multi_output_standalone_component_source_cut"),
      check("seq1233_anchor_is_complete",
            seq1233.get("required_checks_passed") is True and
            int(seq1233["aggregate"]["dominant_bytes"]) == 770_901_120),
      check("configure_and_serial_build_passed",
            configure["returncode"] == 0 and engine_build["returncode"] == 0),
      check("codegen_build_passed", codegen_build["returncode"] == 0),
      check("runtime_has_no_openvino_or_onednn_dynamic_dependency",
            linkage["returncode"] == 0 and "openvino" not in linkage_text and
            "dnnl" not in linkage_text and "onednn" not in linkage_text),
      check("all_baseline_candidate_programs_generated",
            all(row["baseline_codegen_returncode"] == 0 and
                row["candidate_codegen_returncode"] == 0
                for row in cohort_rows)),
      check("universal_microkernel_package_preserved_byte_exact",
            universal_hashes,
            hashes=sorted({str(value) for value in all_hashes})),
      check("independent_projection_workgroup_census_exact",
            exact_workgroups),
      check("all_even_m_outputs_bit_exact_and_router_runtime_passes",
            all_equivalent,
            exception=("the odd-M fused synthetic carrier aliases U4 zero-point "
                       "rows; router outputs use an independent CPU oracle")),
      check("scalar_router_and_independent_router_outputs_are_numeric",
            bool(router_reference.get("finite")) and
            float(router_reference.get("cosine", 0.0)) >= 0.999 and
            float(router_reference.get("relative_l2", 1.0)) <= 0.002,
            compare=router_reference),
      check("locked_real_qkv_reference_remains_tight",
            bool(real_reference.get("finite")) and
            float(real_reference.get("cosine", 0.0)) >= 0.999 and
            float(real_reference.get("relative_l2", 1.0)) <= 0.002,
            compare=real_reference),
      check("candidate_has_no_register_spill", no_spills),
      check("complete_schedule_inference_has_eight_paired_blocks",
            sample_shape_ok and inference.get("sample_count_pass") is True,
            samples_ms=schedule_samples_ms),
      check("memory_preflight_and_stop_guards_never_tripped",
            memory_samples[0]["available_bytes"] >= preflight_bytes and
            all(row["available_bytes"] >= stop_bytes
                for row in memory_samples),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory_samples)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  component_admission_passed = (
      required_checks_passed and inference.get("rate_pass") is True)
  verdict = ("admit_isolated_fixed_fc_graph_integration_source_cut"
             if component_admission_passed else
             "close_fixed_fc_multi_output_component_on_complete_schedule_ucb"
             if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA, "created_at": created_at, "workstream": WORKSTREAM,
      "git": {"commit": commit, "dirty": bool(dirty), "status": dirty},
      "source": {"seq1233": str(SEQ1233.relative_to(ROOT)),
                 "seq1353": str(SEQ1353.relative_to(ROOT)),
                 "onednn_commit": onednn_commit},
      "cohorts": cohort_rows,
      "schedule": {
          "anchor_seq1233_ms": anchor_schedule_ms,
          "allowed_pointer_overhead_ms": allowed_overhead_ms,
          "maximum_fixed_fc_schedule_ms": schedule_cap_ms,
          "overhead_samples_ms": overhead_samples_ms,
          "candidate_schedule_samples_ms": schedule_samples_ms,
          "overhead_point_ms": (statistics.median(overhead_samples_ms)
                                if overhead_samples_ms else None),
          "overhead_ucb_ms": overhead_ucb_ms,
          "fixed_fc_saving_lcb_ms": fixed_fc_saving_lcb_ms,
          "combined_qk_plus_fixed_fc_saving_lcb_ms": combined_saving_lcb_ms,
          "kill_number_ms": kill_number_ms,
          "retained_margin_ms": retained_margin_ms,
          "inference": inference,
      },
      "runtime": {"warmup": args.warmup, "repeat": args.repeat,
                  "paired_blocks": args.blocks,
                  "memory_preflight_bytes": preflight_bytes,
                  "memory_stop_bytes": stop_bytes},
      "memory_samples": memory_samples, "checks": checks,
      "component_admission_passed": component_admission_passed,
      "verdict": verdict, "required_checks_passed": required_checks_passed,
  }
  (out / "metrics.json").write_text(
      json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  manifest = {
      "schema": f"{SCHEMA}-manifest-v0", "created_at": created_at,
      "artifact": str(out.relative_to(ROOT)), "git": metrics["git"],
      "host": {"target_alias": "intel-ptl-local", "kernel": platform.release()},
      "inputs": metrics["source"], "runtime": metrics["runtime"],
      "verdict": verdict, "required_checks_passed": required_checks_passed,
  }
  (out / "manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  rows = "\n".join(
      f"| `{row['name']}` | `{row['widths']}` | {row['count']} | "
      f"{row['runtime'].get('baseline_kernel_median_us', 0):.3f} | "
      f"{row['runtime'].get('candidate_kernel_median_us', 0):.3f} |"
      for row in cohort_rows)
  summary = f"""# OpenVINO independent fixed-FC stream component gate

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`.

| cohort | independent widths | calls/token | fused us | independent us |
|---|---|---:|---:|---:|
{rows}

All independent outputs are compared bit-for-bit with the retained fused
carrier, while the locked real layer-0 QKV output is also checked against its
captured stock oracle. The complete 160-dispatch schedule uses eight paired
ABBA/BAAB blocks. Its median pointer charge is
`{statistics.median(overhead_samples_ms) if overhead_samples_ms else None} ms`
and one-sided 95% upper charge is `{overhead_ucb_ms} ms`; the admitted margin
is `{allowed_overhead_ms} ms`. The QK plus fixed-FC lower-bound saving retains
`{retained_margin_ms} ms` above the `{kill_number_ms} ms` kill-number.
"""
  (out / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({"artifact": str(out.relative_to(ROOT)),
                    "verdict": verdict, "overhead_ucb_ms": overhead_ucb_ms,
                    "retained_margin_ms": retained_margin_ms,
                    "required_checks_passed": required_checks_passed},
                   separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
