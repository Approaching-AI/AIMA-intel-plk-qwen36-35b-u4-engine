#!/usr/bin/env python3
"""Run the sole pinned F16/U4 active-expert complete-FFN source gate."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-complete-ffn-microkernel-source-gate-v0"
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
ONEDNN_SOURCE = OV_SOURCE / "src/plugins/intel_gpu/thirdparty/onednn_gpu"
ONEDNN_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
BUILD_DIR = ROOT / "build/engine"
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
HOST_SOURCE = ROOT / "engine/gpu/opencl/openvino_moe_micro_host.cl"
SUPPORT_SOURCE = ROOT / "engine/gpu/opencl/openvino_moe_micro_support.cl"
PREPACK_TOOL = ROOT / "tools/intel-qwen36-openvino-moe-micro-prepack.py"
CAPTURE = ROOT / (
    "output/complete-ffn-boundary-gate-20260712Tseq763cleanZ/raw/"
    "capture/payloads")
CAP_US = 6250.0
NOISE_FRACTION = 0.005


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--repeat", type=int, default=21)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--openvino-source", type=Path, default=OV_SOURCE)
  parser.add_argument("--onednn-source", type=Path, default=ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path, default=ONEDNN_BUILD)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  args = parser.parse_args()
  if args.warmup < 1 or args.repeat < 5 or args.timeout_s <= 0:
    parser.error("warmup/repeat/timeout arguments are invalid")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/complete-ffn-microkernel-source-gate-{stamp}"
  return args


def run(command: list[str], timeout_s: int, cwd: Path = ROOT,
        env: dict[str, str] | None = None) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
        timeout=timeout_s, check=False, encoding="utf-8", errors="replace")
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout or "", "stderr": error.stderr or "",
            "timed_out": True}


def run_intel(command: list[str], args: argparse.Namespace,
              cwd: Path = ROOT) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 && export DNNL_VERBOSE=0 && " +
      shlex.join(command))
  return run(["bash", "-lc", shell], args.timeout_s, cwd=cwd)


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


def git_output(*parts: str, cwd: Path = ROOT) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=cwd, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def number(text: str, key: str) -> int | None:
  match = re.search(rf"\b{re.escape(key)}:\s*(\d+)", text)
  return int(match.group(1)) if match else None


def relative(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  codegen_dir = raw / "codegen"
  prepack_dir = raw / "prepacked"
  capture_files = {
      "input": CAPTURE / "attn_post_norm-27__tok1023__ord0.bin",
      "topk": CAPTURE / "ffn_moe_topk-27__tok1023__ord1.bin",
      "router": CAPTURE / "ffn_moe_weights_norm-27__tok1023__ord2.bin",
      "oracle": CAPTURE / "ffn_out-27__tok1023__ord10.bin",
  }
  required = [
      ENV_SCRIPT, CMAKE, CXX, OV_PYTHON, args.openvino_source,
      args.onednn_source, args.onednn_build / "src/libdnnl.a",
      CODEGEN, HOST_SOURCE, SUPPORT_SOURCE, PREPACK_TOOL,
      ROOT / "engine/boundaries.json", *capture_files.values(),
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing source-gate inputs: " + ", ".join(missing))

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  ov_commit = git_output("rev-parse", "HEAD", cwd=args.openvino_source)
  ov_dirty = git_output("status", "--porcelain", cwd=args.openvino_source)
  onednn_commit = git_output("rev-parse", "HEAD", cwd=args.onednn_source)
  onednn_dirty = git_output("status", "--porcelain", cwd=args.onednn_source)

  onednn_build = run_intel([
      str(CMAKE), "--build", str(args.onednn_build), "--target", "dnnl",
      "-j16"], args)
  write_run(raw, "onednn-build", onednn_build)
  configure = run([
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(args.build_dir)],
      args.timeout_s)
  write_run(raw, "configure", configure)
  engine_build = run([
      str(CMAKE), "--build", str(args.build_dir), "--target",
      "iq36-openvino-moe-micro-runtime", "-j16"], args.timeout_s)
  write_run(raw, "engine-build", engine_build)
  runtime = args.build_dir / "iq36-openvino-moe-micro-runtime"
  linkage = (
      run(["ldd", str(runtime)], args.timeout_s)
      if engine_build["returncode"] == 0 and runtime.exists() else
      {"command": ["ldd", str(runtime)], "returncode": 125,
       "stdout": "", "stderr": "runtime build failed", "timed_out": False})
  write_run(raw, "runtime-linkage", linkage)
  linkage_text = str(linkage.get("stdout", "")).lower()
  native_linkage = (
      linkage["returncode"] == 0 and "openvino" not in linkage_text and
      "dnnl" not in linkage_text and "onednn" not in linkage_text)

  source = args.onednn_source
  build = args.onednn_build
  includes = [
      source / "src/gpu/intel/gemm/jit",
      source / "src/gpu/intel/gemm/jit/dnnl_gpu_intel_gemm_jit",
      build / "include", source / "include", source / "third_party/opencl",
      source / "third_party", source / "src",
      source / "src/gpu/intel/jit/config", source / "third_party/ngen",
      source / "src/gpu/intel/gemm/jit/include",
  ]
  codegen_binary = raw / "openvino-moe-micro-codegen"
  codegen_build_command = [
      str(CXX), "-std=c++17", "-O3", "-DNDEBUG", "-fopenmp",
      "-fno-operator-names", "-DCL_TARGET_OPENCL_VERSION=120",
      "-DDNNL_X64=1", "-DGEMMSTONE_BUILD_12HP",
      "-DGEMMSTONE_BUILD_12LP", "-DGEMMSTONE_BUILD_12P7",
      "-DGEMMSTONE_BUILD_12P8", "-DGEMMSTONE_BUILD_XE2",
      "-DGEMMSTONE_BUILD_XE3", "-DGEMMSTONE_BUILD_XE3P",
      "-DGEMMSTONE_CONFIG", "-DNGEN_CONFIG",
      *[f"-I{path}" for path in includes], str(CODEGEN),
      str(build / "src/libdnnl.a"), "-lOpenCL", "-ldl", "-lpthread",
      "-o", str(codegen_binary),
  ]
  codegen_build = (
      run(codegen_build_command, args.timeout_s)
      if onednn_build["returncode"] == 0 else
      {"command": codegen_build_command, "returncode": 125,
       "stdout": "", "stderr": "oneDNN build failed", "timed_out": False})
  write_run(raw, "codegen-build", codegen_build)
  codegen_command = [
      str(codegen_binary), "--dump-dir", str(codegen_dir),
      "--host-source", str(HOST_SOURCE),
  ]
  codegen_run = (
      run_intel(codegen_command, args)
      if codegen_build["returncode"] == 0 else
      {"command": codegen_command, "returncode": 125,
       "stdout": "", "stderr": "codegen build failed", "timed_out": False})
  write_run(raw, "codegen", codegen_run)
  codegen = parse_json_line(codegen_run)

  disassembly: dict[str, dict[str, Any]] = {}
  for kind in ("gate", "up", "down"):
    binary = codegen_dir / f"{kind}.program.bin"
    dump = codegen_dir / f"{kind}.disasm"
    command = [
        "ocloc", "disasm", "-file", str(binary), "-dump", str(dump),
        "-device", "0xb080"]
    row = (
        run_intel(command, args) if binary.exists() else
        {"command": command, "returncode": 125, "stdout": "",
         "stderr": "program binary missing", "timed_out": False})
    write_run(raw, f"{kind}-disasm", row)
    ze_path = dump / ".ze_info"
    ze_info = (ze_path.read_text(encoding="utf-8", errors="replace")
               if ze_path.exists() else "")
    asm_paths = sorted(set(dump.glob("*.asm")) | set(dump.glob(".*.asm")))
    assembly = "\n".join(path.read_text(
        encoding="utf-8", errors="replace") for path in asm_paths)
    disassembly[kind] = {
        "returncode": row["returncode"],
        "grf_count": number(ze_info, "grf_count"),
        "eu_thread_count": number(ze_info, "eu_thread_count"),
        "barrier_count": number(ze_info, "barrier_count"),
        "simd_size": number(ze_info, "simd_size"),
        "has_dpas": "has_dpas:        true" in ze_info,
        "scratch_or_spill": bool(re.search(r"scratch|spill", ze_info + assembly,
                                            re.IGNORECASE)),
    }

  prepack_command = [
      str(OV_PYTHON), str(PREPACK_TOOL), "--out-dir", str(prepack_dir)]
  prepack_run = run(prepack_command, args.timeout_s)
  write_run(raw, "prepack", prepack_run)
  prepack = parse_json_line(prepack_run)

  runtime_command = [
      str(runtime), "--prep-dir", str(prepack_dir),
      "--gate-binary", str(codegen_dir / "gate.program.bin"),
      "--up-binary", str(codegen_dir / "up.program.bin"),
      "--down-binary", str(codegen_dir / "down.program.bin"),
      "--support-source", str(SUPPORT_SOURCE),
      "--input", str(capture_files["input"]),
      "--topk", str(capture_files["topk"]), "--topk-stride", "1024",
      "--router-weights", str(capture_files["router"]),
      "--oracle", str(capture_files["oracle"]),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--cap-us", str(CAP_US),
  ]
  runtime_rows: list[dict[str, Any]] = []
  prerequisites = (
      engine_build["returncode"] == 0 and runtime.exists() and
      codegen_run["returncode"] == 0 and prepack_run["returncode"] == 0)
  for label in ("repeat", "confirm"):
    row = (
        run_intel(runtime_command, args) if prerequisites else
        {"command": runtime_command, "returncode": 125, "stdout": "",
         "stderr": "runtime prerequisite failed", "timed_out": False})
    write_run(raw, label, row)
    runtime_rows.append({
        "label": label, "returncode": row["returncode"],
        "probe": parse_json_line(row),
    })

  packages = codegen.get("packages", [])
  provider_exact = (
      codegen_run["returncode"] == 0 and len(packages) == 3 and
      all(row.get("quant_group_size") == 32 and
          row.get("grf_min") == 256 and row.get("barrier_count") == 1 and
          row.get("systolic") is True and row.get("program_bytes", 0) > 0 and
          row.get("settings") == {
              "sg_per_wg_m": 8, "sg_per_wg_n": 4, "sg_per_wg_k": 1,
              "wg_tile_m": 256, "wg_tile_n": 192, "slm_size": 0}
          for row in packages))
  resources_pass = all(
      row["returncode"] == 0 and row["grf_count"] == 256 and
      row["eu_thread_count"] == 4 and row["barrier_count"] == 1 and
      row["simd_size"] == 16 and row["has_dpas"] and
      not row["scratch_or_spill"] for row in disassembly.values())
  prepack_exact = (
      prepack_run["returncode"] == 0 and prepack.get("experts") == 257 and
      prepack.get("shared_expert_id") == 256 and
      prepack.get("quant_group_size") == 32 and
      prepack.get("activation_type") == "f16" and
      prepack.get("weight_type") == "u4" and
      prepack.get("scale_type") == "f16" and
      len(prepack.get("files", {})) == 13)
  probes = [row["probe"] for row in runtime_rows]
  evaluated = all(
      bool(probe) and row["returncode"] in (0, 2)
      for row, probe in zip(runtime_rows, probes))
  medians = [float(probe.get("device_span_median_us", math.inf))
             for probe in probes]
  matrix_medians = [sum(float(probe.get("stage_median_us", {}).get(name, 0.0))
                            for name in ("up", "gate", "down"))
                    for probe in probes]
  pair_spread = (
      abs(medians[0] - medians[1]) / min(medians)
      if len(medians) == 2 and all(math.isfinite(value) and value > 0
                                  for value in medians) else math.inf)
  correctness_rows = [bool(probe.get("correctness_pass")) for probe in probes]
  performance_rows = [bool(probe.get("performance_pass")) for probe in probes]
  native_rows = [
      probe.get("maps_native_only") is True and
      probe.get("timed_host_upload_bytes") == 0 and
      probe.get("timed_host_readback_bytes") == 0 and
      probe.get("complete_active_experts") == 223
      for probe in probes]

  checks = [
      check("repository_clean_at_gate", dirty == "",
            dirty_paths=dirty.splitlines()),
      check("pinned_openvino_source_exact",
            ov_commit == OV_COMMIT and ov_dirty == "", observed=ov_commit,
            expected=OV_COMMIT, dirty_paths=ov_dirty.splitlines()),
      check("pinned_openvino_onednn_source_exact",
            onednn_commit == ONEDNN_COMMIT and onednn_dirty == "",
            observed=onednn_commit, expected=ONEDNN_COMMIT,
            dirty_paths=onednn_dirty.splitlines()),
      check("pinned_onednn_static_codegen_library_builds",
            onednn_build["returncode"] == 0),
      check("native_runtime_builds",
            configure["returncode"] == 0 and engine_build["returncode"] == 0),
      check("native_runtime_has_no_openvino_or_onednn_linkage",
            native_linkage, linkage=linkage.get("stdout", "")),
      check("source_exact_group32_f16_u4_provider_packages_build",
            provider_exact, packages=packages),
      check("three_fused_programs_are_simd16_dpas_256grf_no_spill",
            resources_pass, resources=disassembly),
      check("gguf_q4k_routed_plus_shared_prepack_exact", prepack_exact,
            manifest=prepack),
      check("repeat_and_confirm_complete", evaluated,
            returncodes=[row["returncode"] for row in runtime_rows]),
      check("repeat_and_confirm_native_only_zero_timed_transfer",
            evaluated and all(native_rows), rows=native_rows),
      check("repeat_and_confirm_final_ffn_out_correct",
            evaluated and all(correctness_rows), rows=correctness_rows,
            cosine_min=0.999, relative_l2_max=0.002),
      check("repeat_and_confirm_clear_6250us_complete_ffn_cap",
            evaluated and all(performance_rows), rows=performance_rows,
            medians_us=medians, cap_us=CAP_US,
            matrix_only_stage_medians_us=matrix_medians),
      check("repeat_confirm_spread_inside_noise_band",
            pair_spread <= NOISE_FRACTION,
            spread_fraction=pair_spread, noise_fraction=NOISE_FRACTION),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  evaluation_completed = (
      provider_exact and resources_pass and prepack_exact and native_linkage and
      evaluated and all(native_rows))
  disposition = (
      "accept_f16_u4_active_expert_complete_ffn_source"
      if required_checks_passed else
      ("reject_f16_u4_active_expert_complete_ffn_source"
       if evaluation_completed else
       "incomplete_f16_u4_active_expert_complete_ffn_source_gate"))
  selected_next = (
      "native_prefill_product_integration"
      if required_checks_passed else
      "native_prefill_product_route_reflection_gate")
  reason = (
      "The pinned source is correct, stable, and below the complete-FFN cap."
      if required_checks_passed else
      "The sole pinned F16/U4 source is compiler-safe and native-only, but "
      "fails its complete-FFN terminal contract. Close it without a datatype, "
      "tile, subgroup, workgroup, or expert-bucket sweep and return to route "
      "reflection.")
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "commit": commit,
      "evaluation_completed": evaluation_completed,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition, "selected_next_route": selected_next,
      "next_route_reason": reason, "checks": checks,
      "provider": codegen, "compiler_resources": disassembly,
      "prepack": prepack, "rows": runtime_rows,
      "complete_ffn_cap_us": CAP_US,
      "device_span_medians_us": medians,
      "matrix_only_stage_medians_us": matrix_medians,
      "paired_spread_fraction": pair_spread,
      "inputs": {
          "capture": relative(CAPTURE),
          "openvino_source": str(args.openvino_source.resolve()),
          "onednn_source": str(args.onednn_source.resolve()),
          "host_source": relative(HOST_SOURCE),
          "support_source": relative(SUPPORT_SOURCE),
      },
      "stop_condition": (
          "No datatype, tile, subgroup, workgroup, expert-bucket, or synthetic-"
          "assignment sweep is admitted after any source-gate failure."),
  }
  (out / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in checks if not row["pass"]]
  (out / "summary.md").write_text("\n".join([
      "# Complete FFN F16/U4 microkernel source gate", "",
      f"- required_checks_passed: `{str(required_checks_passed).lower()}`",
      f"- evaluation_completed: `{str(evaluation_completed).lower()}`",
      f"- disposition: `{disposition}`",
      f"- device medians: `{medians} us`",
      f"- matrix-only stage medians: `{matrix_medians} us`",
      f"- paired spread: `{pair_spread:.6%}`",
      f"- failed checks: `{failed}`", "", reason, ""]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required_checks_passed,
      "evaluation_completed": evaluation_completed,
      "disposition": disposition, "device_span_medians_us": medians,
      "matrix_only_stage_medians_us": matrix_medians,
      "paired_spread_fraction": pair_spread,
      "selected_next_route": selected_next,
      "out_dir": relative(out)}, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
