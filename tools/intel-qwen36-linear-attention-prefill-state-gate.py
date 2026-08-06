#!/usr/bin/env python3
"""Gate one fixed 1024-token native linear-attention state kernel."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-linear-attention-prefill-state-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
KERNEL = ROOT / "engine/gpu/opencl/prefill_attention.cl"
CAPTURE = (
    ROOT / "output/linear-attention-prefill-boundary-"
    "20260712Tseq750cleanZ/raw/capture-output")
OPENVINO_PROFILE = (
    ROOT / "output/openvino-hidden-prefill-profile-"
    "20260712Tseq751cleanZ/profile.json")
OPENVINO_ZE_INFO = (
    ROOT / "output/openvino-hidden-prefill-profile-"
    "20260712Tseq751cleanZ/raw/openvino-gdn-disassembly/.ze_info")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
TARGET = "iq36-linear-attention-prefill-state-probe"

# Pre-registered target arithmetic.  The same-host OpenVINO PERF_COUNT row
# reports 72.088 ms for the thirty 1024-token GatedDeltaNet stages.  The hard
# component ruler is its mean divided by the product's 1.10 ratio and rounded
# down.  The much tighter 407 us current-schedule residual is retained only as
# a direction diagnostic: seq669's fast mixed-MoE representation is not
# accuracy-accepted, so it cannot define the sole hard component cap.
PREFILL_TARGET_TOK_S = 2510.0
TILE_TOKENS = 1024
MIXED_MOE_CONFIRM_US = 384_692.255
NON_MOE_TILE_US = TILE_TOKENS / PREFILL_TARGET_TOK_S * 1.0e6 - MIXED_MOE_CONFIRM_US
NON_MOE_PER_LAYER_US = NON_MOE_TILE_US / 40.0
STATE_CORE_SHARE = 0.70
CURRENT_SCHEDULE_RESIDUAL_CAP_US = 407.0
OPENVINO_GDN_TOTAL_US = 72_088.0
OPENVINO_GDN_LAYER_COUNT = 30
# PERF_COUNT is a one-shot component attribution run rather than the product
# timing lane.  Require a fresh clean-tree row to reproduce the calibration
# within two percent while keeping the registered 72.088 ms ruler unchanged.
OPENVINO_PROFILE_REPEAT_TOLERANCE = 0.02
PRODUCT_RATIO = 1.10
OPENVINO_RELATIVE_CAP_US = (
    OPENVINO_GDN_TOTAL_US / OPENVINO_GDN_LAYER_COUNT / PRODUCT_RATIO)
STATE_CORE_CAP_US = 2_184.0
NOISE_FRACTION = 0.005
COSINE_MINIMUM = 0.999
RELATIVE_L2_MAXIMUM = 0.002


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--kernel", type=Path, default=KERNEL)
  parser.add_argument("--capture", type=Path, default=CAPTURE)
  parser.add_argument("--openvino-profile", type=Path,
                      default=OPENVINO_PROFILE)
  parser.add_argument("--openvino-ze-info", type=Path,
                      default=OPENVINO_ZE_INFO)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--storage", choices=("f32", "f16", "chunk64"),
                      default="f32")
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--repeat", type=int, default=21)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.jobs, args.warmup, args.repeat, args.timeout_s) <= 0:
    parser.error("jobs, warmup, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / (
        f"output/linear-attention-prefill-{args.storage}-state-{stamp}")
  return args


def rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout if isinstance(error.stdout, str) else "",
            "stderr": error.stderr if isinstance(error.stderr, str) else "",
            "timed_out": True}


def run_env(command: list[str], args: argparse.Namespace) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0 && "
      f"{shlex.join(command)}")
  return run(["bash", "-lc", shell], args.timeout_s)


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw / f"{name}.command.json", {
      "command": result["command"], "returncode": result["returncode"],
      "timed_out": result["timed_out"]})
  (raw / f"{name}.stdout").write_text(
      str(result["stdout"]), encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      str(result["stderr"]), encoding="utf-8")


def parse_probe(result: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(result.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def capture_inventory(capture: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  summary_path = capture / "capture-summary.json"
  index_path = capture / "tensor-dumps.jsonl"
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  inventory: dict[str, Any] = {}
  for line in index_path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    inventory[str(row["tensor_name"])] = {
        "ne": row["ne"], "nbytes": row["nbytes"],
        "payload_path": row["payload_path"]}
  return summary, inventory


def comparison_pass(value: dict[str, Any], expected_count: int) -> bool:
  return (
      value.get("count") == expected_count and
      value.get("finite_pairs") == expected_count and
      value.get("finite") is True and value.get("passes") is True and
      float(value.get("cosine", -math.inf)) >= COSINE_MINIMUM and
      float(value.get("relative_l2", math.inf)) <= RELATIVE_L2_MAXIMUM)


def ze_execution_block(text: str, name_prefix: str) -> str:
  lines = text.splitlines()
  for begin, line in enumerate(lines):
    stripped = line.strip()
    if not stripped.startswith("- name:"):
      continue
    name = stripped.split("name:", 1)[1].strip()
    if not name.startswith(name_prefix):
      continue
    end = begin + 1
    while end < len(lines) and not lines[end].strip().startswith("- name:"):
      end += 1
    return "\n".join(lines[begin:end])
  return ""


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required_paths = [args.model, args.kernel, args.capture,
                    args.capture / "capture-summary.json",
                    args.capture / "tensor-dumps.jsonl", args.env_script,
                    args.openvino_profile, args.openvino_ze_info,
                    args.cmake, ROOT / "engine/boundaries.json"]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  dirty = git_output("status", "--porcelain")
  commit = git_output("rev-parse", "HEAD")
  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  summary, inventory = capture_inventory(args.capture)
  openvino_profile = json.loads(
      args.openvino_profile.read_text(encoding="utf-8"))
  openvino_runs = openvino_profile.get("runs", [])
  openvino_run = openvino_runs[0] if len(openvino_runs) == 1 else {}
  openvino_gdn_rows = [
      row for row in openvino_run.get("top_nodes", [])
      if row.get("node_type") == "GatedDeltaNet"]
  openvino_gdn_total_us = 1000.0 * float(
      openvino_run.get("category_ms", {}).get(
          "linear_attention_gated_delta", math.inf))
  openvino_ze_info = args.openvino_ze_info.read_text(
      encoding="utf-8", errors="replace")
  openvino_gdn_execution = ze_execution_block(
      openvino_ze_info, "gated_delta_net_ref")
  expected_shapes = {
      "q_conv_predelta-0": ([128, 16, 1024, 1], 8_388_608),
      "k_conv_predelta-0": ([128, 16, 1024, 1], 8_388_608),
      # V is a view into the 8192-wide convolved Q/K/V row.  The capture keeps
      # the real 8192-float token stride; the probe compacts that view once,
      # before the timed resident loop.
      "v_conv_predelta-0": ([128, 32, 1024, 1], 33_538_048),
      "gate-0": ([32, 1024, 1, 1], 131_072),
      "beta_sigmoid-0": ([1, 32, 1024, 1], 131_072),
      "state_predelta-0": ([128, 128, 32, 1], 2_097_152),
      "new_state-0": ([128, 128, 32, 1], 2_097_152),
      "attn_output-0": ([128, 32, 1024, 1], 16_777_216),
      "z-0": ([4096, 1024, 1, 1], 16_777_216),
      "final_output-0": ([4096, 1024, 1, 1], 16_777_216),
  }
  capture_shape_ok = all(
      name in inventory and inventory[name]["ne"] == shape and
      inventory[name]["nbytes"] == nbytes
      for name, (shape, nbytes) in expected_shapes.items())

  build_dir = raw / "build"
  configure = run_env([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B", str(build_dir),
      "-DCMAKE_BUILD_TYPE=Release"], args)
  write_run(raw, "configure", configure)
  build = run_env([
      str(args.cmake), "--build", str(build_dir), f"-j{args.jobs}",
      "--target", TARGET], args) if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False}
  write_run(raw, "build", build)
  probe_binary = build_dir / TARGET
  program_binary = raw / "prefill-attention.bin"
  probe_command = [
      str(probe_binary), "--model", str(args.model), "--kernel",
      str(args.kernel), "--capture", str(args.capture), "--layer", "0",
      "--storage", args.storage,
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--cap-us", str(STATE_CORE_CAP_US), "--dump-binary",
      str(program_binary)]
  rows = []
  for label in ("repeat", "confirm"):
    result = run_env(probe_command, args) if build["returncode"] == 0 else {
        "command": probe_command, "returncode": 125, "stdout": "",
        "stderr": "build failed", "timed_out": False}
    write_run(raw, label, result)
    rows.append({"label": label, "returncode": result["returncode"],
                 "probe": parse_probe(result)})

  disassembly_dir = raw / "disassembly"
  disassembly_dir.mkdir(exist_ok=True)
  disassembly = run_env([
      "ocloc", "disasm", "-file", str(program_binary), "-dump",
      str(disassembly_dir)], args) if program_binary.is_file() else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "program binary missing", "timed_out": False}
  write_run(raw, "ocloc-disasm", disassembly)
  ze_info_path = disassembly_dir / ".ze_info"
  native_ze_info = (
      ze_info_path.read_text(encoding="utf-8", errors="replace")
      if ze_info_path.is_file() else "")
  native_execution = ze_execution_block(
      native_ze_info,
      f"iq36_linear_prefill_recurrent_{args.storage}")

  state_core_medians = [
      float(row["probe"].get("state_core_median_us", math.inf))
      for row in rows]
  spread_fraction = (
      abs(state_core_medians[0] - state_core_medians[1]) /
      min(state_core_medians)
      if len(state_core_medians) == 2 and
      all(math.isfinite(value) and value > 0 for value in state_core_medians)
      else math.inf)
  row_correctness_passes = []
  row_timing_passes = []
  for row in rows:
    probe = row["probe"]
    row_correctness_passes.append(
        row["returncode"] in (0, 2) and
        comparison_pass(probe.get("attention_comparison", {}), 4_194_304) and
        comparison_pass(probe.get("state_comparison", {}), 524_288) and
        comparison_pass(probe.get("final_comparison", {}), 4_194_304) and
        probe.get("timed_host_upload_bytes") == 0 and
        probe.get("timed_host_read_bytes") == 0 and
        probe.get("forbidden_runtime_mapped") is False and
        probe.get("storage") == args.storage and
        probe.get("recurrent_workgroups") ==
        (512 if args.storage in ("f16", "chunk64") else 256) and
        (args.storage != "chunk64" or
         (probe.get("chunk_size") == 64 and
          probe.get("chunk_count") == 16 and
          probe.get("chunk_scan_workgroups") == 256)))
    row_timing_passes.append(
        float(probe.get("state_core_median_us", math.inf)) <=
        STATE_CORE_CAP_US)

  kernel_source = args.kernel.read_text(encoding="utf-8")
  residual_cap_arithmetic = NON_MOE_PER_LAYER_US * STATE_CORE_SHARE
  openvino_profile_passes = (
      openvino_run.get("seq_len") == TILE_TOKENS and
      len(openvino_gdn_rows) == OPENVINO_GDN_LAYER_COUNT and
      abs(openvino_gdn_total_us - OPENVINO_GDN_TOTAL_US) /
      OPENVINO_GDN_TOTAL_US <= OPENVINO_PROFILE_REPEAT_TOLERANCE and
      all(row.get("exec_type") == "ocl::gated_delta_net::ref___f16"
          for row in openvino_gdn_rows))
  openvino_resource_shape_passes = (
      "grf_count:       128" in openvino_gdn_execution and
      "simd_size:       16" in openvino_gdn_execution and
      "eu_thread_count: 8" in openvino_gdn_execution and
      openvino_ze_info.count("type_name:       'half*;8'") >= 8)
  if args.storage == "chunk64":
    chunk_kernel_names = (
        "iq36_linear_prefill_chunk64_prepare_f32",
        "iq36_linear_prefill_chunk64_scan_f32",
        "iq36_linear_prefill_chunk64_output_f32")
    chunk_execution = [
        ze_execution_block(native_ze_info, name)
        for name in chunk_kernel_names]
    kernel_shape_passes = (
        "#define IQ36_LINEAR_CHUNK_SIZE 64U" in kernel_source and
        "#define IQ36_LINEAR_CHUNK_COUNT 16U" in kernel_source and
        all(name in kernel_source for name in chunk_kernel_names))
    compiler_shape_passes = (
        disassembly["returncode"] == 0 and
        all(block and "scratch" not in block.lower() and
            "spill" not in block.lower() for block in chunk_execution))
  elif args.storage == "f16":
    kernel_shape_passes = (
        "half state_shard[8][8]" in kernel_source and
        "iq36_linear_prefill_recurrent_f16" in kernel_source and
        "iq36_linear_prefill_norm_gate_f16" in kernel_source and
        "intel_reqd_sub_group_size(16)" in kernel_source)
    compiler_shape_passes = (
        disassembly["returncode"] == 0 and
        "grf_count:       128" in native_execution and
        "simd_size:       16" in native_execution and
        "eu_thread_count: 8" in native_execution and
        "scratch" not in native_execution.lower() and
        "spill" not in native_execution.lower())
  else:
    kernel_shape_passes = (
        "float state_shard[4][16]" in kernel_source and
        "token < IQ36_LINEAR_TOKEN_COUNT" in kernel_source and
        "iq36_linear_prefill_recurrent_f32" in kernel_source and
        "iq36_linear_prefill_norm_gate_f32" in kernel_source and
        "intel_reqd_sub_group_size(32)" in kernel_source)
    compiler_shape_passes = (
        disassembly["returncode"] == 0 and
        "grf_count:       256" in native_execution and
        "simd_size:       32" in native_execution and
        "eu_thread_count: 4" in native_execution and
        "scratch" not in native_execution.lower() and
        "spill" not in native_execution.lower())
  checks = [
      check("repository_clean_at_gate", dirty == "", dirty_paths=dirty.splitlines()),
      check("locked_model_and_layer0_1024_capture",
            args.model.resolve() == MODEL.resolve() and
            summary.get("token_count") == 1024 and
            summary.get("batch_all") is True and
            summary.get("linear_component_layer") == 0 and capture_shape_ok,
            summary=summary, expected_shapes=expected_shapes),
      check("same_host_openvino_gdn_profile_and_resource_shape_locked",
            openvino_profile_passes and openvino_resource_shape_passes,
            profile=rel(args.openvino_profile),
            ze_info=rel(args.openvino_ze_info),
            gdn_total_us=openvino_gdn_total_us,
            calibration_total_us=OPENVINO_GDN_TOTAL_US,
            repeat_tolerance=OPENVINO_PROFILE_REPEAT_TOLERANCE,
            gdn_rows=len(openvino_gdn_rows)),
      check("target_derived_state_core_cap_preregistered",
            STATE_CORE_CAP_US <= OPENVINO_RELATIVE_CAP_US and
            STATE_CORE_CAP_US == 2184.0 and
            CURRENT_SCHEDULE_RESIDUAL_CAP_US <= residual_cap_arithmetic,
            tile_cap_us=TILE_TOKENS / PREFILL_TARGET_TOK_S * 1.0e6,
            mixed_moe_confirm_us=MIXED_MOE_CONFIRM_US,
            non_moe_tile_us=NON_MOE_TILE_US,
            non_moe_per_layer_us=NON_MOE_PER_LAYER_US,
            state_core_share=STATE_CORE_SHARE,
            current_schedule_residual_arithmetic_us=residual_cap_arithmetic,
            current_schedule_residual_cap_us=CURRENT_SCHEDULE_RESIDUAL_CAP_US,
            openvino_gdn_total_us=OPENVINO_GDN_TOTAL_US,
            openvino_gdn_mean_us=(OPENVINO_GDN_TOTAL_US /
                                  OPENVINO_GDN_LAYER_COUNT),
            product_ratio=PRODUCT_RATIO,
            openvino_relative_cap_us=OPENVINO_RELATIVE_CAP_US,
            registered_hard_cap_us=STATE_CORE_CAP_US),
      check("single_fixed_register_resident_kernel_shape",
            kernel_shape_passes, storage=args.storage),
      check("compiler_confirms_registered_resource_shape_without_scratch",
            compiler_shape_passes,
            storage=args.storage, ze_info=rel(ze_info_path)),
      check("repeat_and_confirm_pass_real_boundary_correctness",
            all(row_correctness_passes),
            row_correctness_passes=row_correctness_passes),
      check("repeat_and_confirm_clear_state_core_cap",
            all(row_timing_passes), row_timing_passes=row_timing_passes,
            medians_us=state_core_medians, cap_us=STATE_CORE_CAP_US),
      check("repeat_confirm_spread_inside_noise_band",
            spread_fraction <= NOISE_FRACTION,
            spread_fraction=spread_fraction,
            noise_fraction=NOISE_FRACTION),
  ]
  required = all(bool(row["pass"]) for row in checks)
  checks_without_timing = [
      row for row in checks
      if row["name"] not in {
          "repeat_and_confirm_clear_state_core_cap",
          "repeat_confirm_spread_inside_noise_band"}]
  route_gate_completed = all(bool(row["pass"]) for row in checks_without_timing)
  evaluation_completed = (
      dirty == "" and capture_shape_ok and openvino_profile_passes and
      openvino_resource_shape_passes and kernel_shape_passes and
      build["returncode"] == 0 and disassembly["returncode"] == 0 and
      all(row["returncode"] in (0, 2) and bool(row["probe"])
          for row in rows))
  best_state_core_us = min(state_core_medians, default=math.inf)
  miss_ratio = best_state_core_us / STATE_CORE_CAP_US
  terminal_timing_miss = all(
      value >= 1.5 * STATE_CORE_CAP_US for value in state_core_medians)
  select_fp16 = (
      args.storage == "f32" and not required and evaluation_completed and
      compiler_shape_passes and all(row_correctness_passes))
  select_chunked_scan = (
      args.storage == "f16" and not required and evaluation_completed)
  reject_chunk64 = (
      args.storage == "chunk64" and not required and evaluation_completed)
  disposition = (
      f"accept_register_resident_linear_prefill_{args.storage}_state_core"
      if required else
      ("reject_f32_register_state_select_fp16_resource_parity"
       if select_fp16 else
       ("reject_fp16_resource_parity_select_chunked_scan_gdn"
        if select_chunked_scan else
        ("reject_chunk64_wy_gdn_return_route_reflection"
         if reject_chunk64 else
         f"reject_register_resident_linear_prefill_{args.storage}_state_core"))))
  selected_next_route = (
      "native_linear_prefill_norm_preconv_projection_and_conv_tile"
      if required else
      ("native_linear_prefill_fp16_resource_parity_design_gate"
       if select_fp16 else
       ("native_linear_prefill_chunked_scan_gdn_design_gate"
        if select_chunked_scan else "native_prefill_route_reflection_gate")))
  if required and args.storage == "chunk64":
    next_reason = (
        "The fixed chunk-64 WY pipeline clears the same-host OpenVINO-relative "
        "2184 us core ruler twice while real attention, state, and final "
        "normalized output pass. Attach native Q4 preprojection/convolution "
        "and charge the separate normalization into the complete linear tile.")
  elif required:
    next_reason = (
        "The fixed register-resident recurrence clears the same-host "
        "OpenVINO-relative 2184 us state-core ruler twice while recurrence, "
        "state, and final normalized output pass real 1024-token boundaries. "
        "The separate normalization remains outside this timing cap. Fuse or "
        "reduce that cost, attach native Q4 preprojection and convolution "
        "without adding a host bridge, then measure the whole linear tile.")
  elif select_fp16:
    next_reason = (
        "The sequential token recurrence is numerically valid and the compiler "
        "reports 256 GRFs/SIMD32/four EU threads with no spill/scratch, but it "
        f"did not stably clear the 2184 us ruler twice: medians are "
        f"{state_core_medians} with {spread_fraction:.3%} paired spread. "
        "The measured "
        "OpenVINO kernel uses half inputs/state, 128 GRFs, SIMD16, and eight EU "
        "threads. Admit exactly one FP16 resource-parity design with the same "
        "real F32 boundary thresholds; if it fails, switch algorithms to "
        "chunked/scan GDN rather than sweeping workgroup variants.")
  elif select_chunked_scan:
    next_reason = (
        "The one admitted FP16 resource-parity design completed on the real "
        "1024-token boundaries but did not satisfy the registered numeric, "
        "compiler-resource, and timing checks. Close FP16 workgroup mapping; "
        "change algorithm to a chunked/scan GDN design rather than sweeping "
        "more workgroup variants.")
  elif reject_chunk64:
    next_reason = (
        "The one pre-registered chunk-64 WY pipeline completed its real "
        "boundary, compiler, and timing evaluation but failed at least one "
        "acceptance axis. Close chunk-size and precision variants and return "
        "to route reflection with the measured stage split.")
  else:
    next_reason = (
        "The pre-registered mapping did not complete a stable correctness/"
        "compiler diagnostic. Inspect the failed axis and return to route "
        "reflection; do not sweep workgroup shapes.")
  metrics = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "commit": commit,
      "storage": args.storage,
      "inputs": {"model": str(args.model), "kernel": rel(args.kernel),
                 "capture": rel(args.capture),
                 "openvino_profile": rel(args.openvino_profile),
                 "openvino_ze_info": rel(args.openvino_ze_info)},
      "budget": {
          "claim_boundary": "same-host component mechanism gate only",
          "prefill_target_tok_s": PREFILL_TARGET_TOK_S,
          "tile_tokens": TILE_TOKENS,
          "tile_cap_us": TILE_TOKENS / PREFILL_TARGET_TOK_S * 1.0e6,
          "mixed_moe_confirm_us": MIXED_MOE_CONFIRM_US,
          "non_moe_tile_us": NON_MOE_TILE_US,
          "non_moe_per_layer_us": NON_MOE_PER_LAYER_US,
          "state_core_share": STATE_CORE_SHARE,
          "current_schedule_residual_arithmetic_us": residual_cap_arithmetic,
          "current_schedule_residual_cap_us":
              CURRENT_SCHEDULE_RESIDUAL_CAP_US,
          "openvino_gdn_total_us": OPENVINO_GDN_TOTAL_US,
          "openvino_gdn_layer_count": OPENVINO_GDN_LAYER_COUNT,
          "openvino_gdn_mean_us": (OPENVINO_GDN_TOTAL_US /
                                    OPENVINO_GDN_LAYER_COUNT),
          "product_ratio": PRODUCT_RATIO,
          "openvino_relative_cap_us": OPENVINO_RELATIVE_CAP_US,
          "registered_hard_cap_us": STATE_CORE_CAP_US,
          "caveat": (
              "PERF_COUNT adds overhead and this gate excludes projections, "
              "convolution, full attention, MoE, and product integration; a "
              "pass cannot establish whole-layer or product prefill speed.")},
      "rows": rows, "checks": checks,
      "required_checks_passed": required,
      "route_gate_completed": route_gate_completed,
      "evaluation_completed": evaluation_completed,
      "best_state_core_us": best_state_core_us,
      "timing_miss_ratio": miss_ratio,
      "terminal_timing_miss": terminal_timing_miss,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_route_reason": next_reason,
  }
  write_json(out / "result.json", metrics)
  write_json(out / "correctness.json", {
      "required_checks_passed": required, "checks": checks,
      "rows": [{"label": row["label"],
                "attention": row["probe"].get("attention_comparison"),
                "state": row["probe"].get("state_comparison"),
                "final": row["probe"].get("final_comparison")}
               for row in rows]})
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "created_at": created_at,
      "commit": commit, "git_dirty": bool(dirty),
      "commands": [row["probe"] for row in rows],
      "required_checks_passed": required})
  failed = [row["name"] for row in checks if not row["pass"]]
  (out / "summary.md").write_text("\n".join([
      f"# Linear-attention prefill {args.storage} state gate", "",
      f"- required_checks_passed: `{str(required).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- registered cap: `{STATE_CORE_CAP_US:.3f} us`",
      f"- repeat/confirm state-core medians: `{state_core_medians}`",
      f"- spread: `{spread_fraction:.6%}`",
      f"- failed checks: `{failed}`", "", metrics["next_route_reason"], "",
      "This is a recurrence/final-normalization mechanism gate, not a whole-",
      "layer, token, or product speed claim.", ""]), encoding="utf-8")
  print(json.dumps({
      "required_checks_passed": required,
      "disposition": metrics["disposition"],
      "storage": args.storage,
      "state_core_medians_us": state_core_medians,
      "spread_fraction": spread_fraction,
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": rel(out)}, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
