#!/usr/bin/env python3
"""Run ADR 0011's static in-core Q4_K affine codegen feasibility gate."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import shlex
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-in-core-affine-codegen-feasibility-gate-v1"
ONEDNN_COMMIT = "01b479323f794da1a7a41a6fc084c7e11ccc2c3b"
KERNEL = ROOT / "engine/gpu/opencl/in_core_q4k_affine_codegen_preflight.cl"
DEFAULT_SEQ649 = (
    ROOT / "output/onednn-grouped-u4-moe-preflight-gate-20260711Tseq649cleanZ")
DEFAULT_SEQ650 = (
    ROOT / "output/onednn-grouped-q4k-moe-component-gate-20260711Tseq650cleanZ")
DEFAULT_SWIGLU = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-20260711Tseq646cleanZ/"
    "raw/capture/payloads/ffn_moe_swiglu-27__tok1023__ord3.bin")
SWIGLU_SHA256 = "187dd69ae740f39951330fbadb48407f791f9ed5145bfbac53f73c076917b648"
DEFAULT_ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
DEFAULT_ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    f"oneDNN-{ONEDNN_COMMIT}")
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seq649", type=Path, default=DEFAULT_SEQ649)
  parser.add_argument("--seq650", type=Path, default=DEFAULT_SEQ650)
  parser.add_argument("--swiglu", type=Path, default=DEFAULT_SWIGLU)
  parser.add_argument("--env-script", type=Path, default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--onednn-source", type=Path,
                      default=DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/in-core-affine-codegen-feasibility-gate-{stamp}"
  return args


def run(command: list[str], timeout_s: int, cwd: Path) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": command,
        "returncode": process.returncode,
        "stderr": process.stderr,
        "stdout": process.stdout,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "timed_out": True,
    }


def shell_run(command: list[str], env_script: Path, timeout_s: int,
              cwd: Path) -> dict[str, Any]:
  shell = f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
  shell += shlex.join(command)
  return run(["bash", "-lc", shell], timeout_s, cwd)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected a JSON object")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(root: Path, *parts: str) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=root, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output(ROOT, "status", "--porcelain")
  return {
      "commit": git_output(ROOT, "rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def matching_lines(path: Path, needles: list[str]) -> list[dict[str, Any]]:
  lines = path.read_text(encoding="utf-8").splitlines()
  matches: list[dict[str, Any]] = []
  for needle in needles:
    found = [
        {"line": index, "text": line.strip()}
        for index, line in enumerate(lines, start=1) if needle in line
    ]
    matches.append({"needle": needle, "matches": found})
  return matches


def f16_roundtrip(path: Path) -> dict[str, Any]:
  values = array.array("f")
  with path.open("rb") as handle:
    values.fromfile(handle, path.stat().st_size // 4)
  max_abs = 0.0
  mean_abs_sum = 0.0
  squared_sum = 0.0
  mismatch_count = 0
  for value in values:
    rounded = struct.unpack("<e", struct.pack("<e", value))[0]
    difference = abs(rounded - value)
    max_abs = max(max_abs, difference)
    mean_abs_sum += difference
    squared_sum += difference * difference
    mismatch_count += difference > 5e-3
  count = len(values)
  return {
      "compared_value_count": count,
      "finite": all(math.isfinite(value) for value in values),
      "max_abs_diff": max_abs,
      "mean_abs_diff": mean_abs_sum / count,
      "mismatch_count": mismatch_count,
      "rmse": math.sqrt(squared_sum / count),
  }


def build_model(seq649: dict[str, Any], seq650: dict[str, Any],
                target: dict[str, Any]) -> dict[str, Any]:
  active_experts = int(seq650["probe"]["active_experts"])
  assignments = int(seq650["probe"]["assignment_count"])
  hidden = 2048
  intermediate = 512
  q4_codes = int(seq650["probe"]["active_q4_code_count"])
  q4_code_bytes = q4_codes // 2
  group64_entries = (
      active_experts * (hidden // 64) * (2 * intermediate) +
      active_experts * (intermediate // 64) * hidden)
  group32_entries = group64_entries * 2
  synthetic_active_payload = (
      q4_code_bytes + group64_entries * 2 + group64_entries // 2)
  exact_active_payload = (
      q4_code_bytes + group32_entries * 4 + group32_entries * 4)
  payload_delta = exact_active_payload - synthetic_active_payload
  bandwidth_gb_s = float(seq650["budget"]["planning_gb_s"])
  bytes_per_us = bandwidth_gb_s * 1000.0
  paired_input_save = assignments * hidden * 2
  paired_output_save = assignments * intermediate * 2
  down_f32_expansion = assignments * hidden * 2
  contribution_bytes = assignments * hidden * 4
  final_output_bytes = 1024 * hidden * 4
  inverse_map_bytes = assignments * 4
  scatter_bytes = contribution_bytes + final_output_bytes + inverse_map_bytes
  base_us = float(seq649["probe"]["minimum_us"])
  gather_us = float(seq650["probe"]["stage_us"]["gather"])
  traffic_without_gather_us = (
      base_us + payload_delta / bytes_per_us -
      paired_input_save / bytes_per_us -
      paired_output_save / bytes_per_us +
      down_f32_expansion / bytes_per_us + scatter_bytes / bytes_per_us)
  compute_units = int(target["runtime"]["opencl_compute_units"])
  max_clock_mhz = int(target["runtime"]["opencl_max_clock_mhz"])
  subgroup_width = 16
  generous_simd_fma_per_cu_cycle = 2
  residual_fma_count = (
      assignments * intermediate * (hidden // 32) * 2 +
      assignments * hidden * (intermediate // 32))
  scalar_fma_per_us = (
      compute_units * subgroup_width * generous_simd_fma_per_cu_cycle *
      max_clock_mhz)
  arithmetic_floor_us = residual_fma_count / scalar_fma_per_us
  with_gather_us = traffic_without_gather_us + gather_us + arithmetic_floor_us
  direct_gather_ideal_us = traffic_without_gather_us + arithmetic_floor_us
  cap_us = float(seq650["budget"]["kernel_cap_us"])
  return {
      "assumptions": {
          "bandwidth_gb_s": bandwidth_gb_s,
          "compute_units": compute_units,
          "max_clock_mhz": max_clock_mhz,
          "residual_fma_throughput": (
              "two SIMD16 FMA vectors per OpenCL CU-cycle; intentionally "
              "more generous than one vector per cycle"),
          "swiglu_exp_cost_us": 0.0,
          "synchronization_cost_us": 0.0,
          "note": (
              "Optimistic admission projection. It credits paired gate/up "
              "input/output traffic savings, omits exp and synchronization, "
              "and still charges the source-required grouped gather plus "
              "deterministic F32 contribution/scatter plane."),
      },
      "bytes": {
          "active_q4_codes": q4_codes,
          "active_q4_code_bytes": q4_code_bytes,
          "group64_coefficient_entries": group64_entries,
          "group32_coefficient_entries": group32_entries,
          "seq649_synthetic_active_payload": synthetic_active_payload,
          "exact_scale_min_active_payload": exact_active_payload,
          "exact_payload_delta": payload_delta,
          "paired_gateup_input_read_saved": paired_input_save,
          "paired_gateup_output_saved": paired_output_save,
          "down_f16_to_f32_expansion": down_f32_expansion,
          "deterministic_contribution_plane": contribution_bytes,
          "final_output": final_output_bytes,
          "inverse_map": inverse_map_bytes,
          "scatter_total": scatter_bytes,
      },
      "operations": {
          "floating_q4k_residual_fma_count": residual_fma_count,
          "swiglu_exp_count": assignments * intermediate,
          "router_weight_multiply_count": assignments * hidden,
      },
      "timing_us": {
          "cap": cap_us,
          "seq649_three_core_minimum": base_us,
          "seq650_measured_grouped_gather": gather_us,
          "traffic_projection_without_gather_or_arithmetic":
              traffic_without_gather_us,
          "generous_residual_arithmetic_floor": arithmetic_floor_us,
          "direct_gather_ideal_floor": direct_gather_ideal_us,
          "source_realizable_relaxed_floor": with_gather_us,
          "source_realizable_floor_over_cap": with_gather_us - cap_us,
          "source_realizable_cap_fraction": with_gather_us / cap_us,
      },
  }


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  ocloc_dir = raw_dir / "ocloc"
  disasm_dir = raw_dir / "disasm"
  ocloc_dir.mkdir(parents=True, exist_ok=False)
  disasm_dir.mkdir()
  grouped_cl = args.onednn_source / "src/gpu/intel/matmul/grouped_micro_gemm.cl"
  grouped_cpp = args.onednn_source / "src/gpu/intel/matmul/grouped_micro_gemm.cpp"
  postops_cpp = args.onednn_source / "src/gpu/intel/matmul/grouped_post_ops_gen.cpp"
  required = [
      args.seq649 / "result.json", args.seq650 / "result.json", args.swiglu,
      args.env_script, args.onednn_source, grouped_cl, grouped_cpp, postops_cpp,
      KERNEL, TARGET_CONTRACT,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))
  if sha256_file(args.swiglu) != SWIGLU_SHA256:
    raise SystemExit("locked SwiGLU oracle hash mismatch")

  created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  seq649 = load_json(args.seq649 / "result.json")
  seq650 = load_json(args.seq650 / "result.json")
  target = load_json(TARGET_CONTRACT)
  model = build_model(seq649, seq650, target)
  f16_oracle = f16_roundtrip(args.swiglu)

  compile_result = shell_run([
      "ocloc", "-file", str(KERNEL), "-device", "0xb080",
      "-options", "-cl-std=CL2.0",
  ], args.env_script, args.timeout_s, ocloc_dir)
  write_json(raw_dir / "compile.json", compile_result)
  native_bins = sorted(ocloc_dir.glob("*.bin"))
  disasm_result = (
      run([
          "ocloc", "disasm", "-file", str(native_bins[0]), "-dump",
          str(disasm_dir), "-device", "0xb080",
      ], args.timeout_s, ROOT)
      if compile_result["returncode"] == 0 and native_bins else
      {"command": [], "returncode": 1, "stdout": "", "stderr": "compile failed",
       "timed_out": False})
  write_json(raw_dir / "disasm.json", disasm_result)
  assembly = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob("*.asm"))
  ze_info = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in disasm_dir.rglob(".ze_info"))
  compile_text = str(compile_result["stdout"]) + str(compile_result["stderr"])

  grouped_cl_text = grouped_cl.read_text(encoding="utf-8")
  grouped_cpp_text = grouped_cpp.read_text(encoding="utf-8")
  postops_text = postops_cpp.read_text(encoding="utf-8")
  kernel_text = KERNEL.read_text(encoding="utf-8")
  source_commit = git_output(args.onednn_source, "rev-parse", "HEAD")
  source_dirty = bool(git_output(args.onednn_source, "status", "--porcelain"))
  source_inspection = {
      "files": {
          str(grouped_cl): {
              "sha256": sha256_file(grouped_cl),
              "anchors": matching_lines(grouped_cl, [
                  "src += src_offset * ldsrc", "dst += src_offset * lddst",
                  "ugemm_grouped_c_type c_tile = ugemm_grouped(",
                  "apply_post_ops_chain(&c_tile", "store_results(&c_tile",
              ]),
          },
          str(grouped_cpp): {
              "sha256": sha256_file(grouped_cpp),
              "anchors": matching_lines(grouped_cpp, [
                  "problem.Tc_ext = problem.Ts = problem.Tc = Type::f32",
                  "gws[2] *= pd()->is_gemv_ ? m_all : pd()->ngroups_",
              ]),
          },
          str(postops_cpp): {
              "sha256": sha256_file(postops_cpp),
              "anchors": matching_lines(postops_cpp, [
                  "po.len() <= 3", "e.is_eltwise() || e.is_binary()",
                  "e.eltwise.alg == alg_kind::eltwise_swish",
                  "e.binary.alg == alg_kind::binary_mul",
              ]),
          },
          str(KERNEL.relative_to(ROOT)): {
              "sha256": sha256_file(KERNEL),
              "anchors": matching_lines(KERNEL, [
                  "return convert_float8(dot) * scale - minimum * input_sum",
                  "const float8 swiglu =", "convert_half8_rte(swiglu)",
                  "weighted_partial_output[lane] = restored * router_weights[lane]",
              ]),
          },
      },
      "facts": {
          "accumulator_is_f32":
              "problem.Tc_ext = problem.Ts = problem.Tc = Type::f32" in grouped_cpp_text,
          "post_ops_execute_before_store":
              grouped_cl_text.index("apply_post_ops_chain(&c_tile") <
              grouped_cl_text.index("store_results(&c_tile"),
          "one_microkernel_call_per_grouped_kernel":
              grouped_cl_text.count("ugemm_grouped_c_type c_tile = ugemm_grouped(") == 1,
          "source_and_destination_are_compact_grouped":
              "src += src_offset * ldsrc" in grouped_cl_text and
              "dst += src_offset * lddst" in grouped_cl_text,
          "third_dispatch_axis_is_expert_group":
              "gws[2] *= pd()->is_gemv_ ? m_all : pd()->ngroups_" in grouped_cpp_text,
          "public_postops_lack_swiglu_pair_and_affine_min":
              "e.is_eltwise() || e.is_binary()" in postops_text and
              "eltwise_swish" in postops_text and "binary_mul" in postops_text,
          "preflight_affine_precedes_f16_store":
              kernel_text.index("const float8 gate = restore_q4k_affine") <
              kernel_text.index("const float8 swiglu =") <
              kernel_text.index("convert_half8_rte(swiglu)"),
          "preflight_down_materializes_weighted_partial":
              "weighted_partial_output[lane]" in kernel_text,
          "single_kernel_global_deterministic_cross_expert_reduction": False,
          "reason": (
              "The grouped wrapper dispatches and stores compact rows by expert. "
              "A token's eight expert contributions are owned by distinct "
              "workgroups; OpenCL provides no grid barrier or deterministic "
              "cross-workgroup reduction inside this kernel. Atomic addition is "
              "not deterministic. Reassigning ownership by token reverts to the "
              "closed M1/handwritten topology; eight rank-serial launches reread "
              "expert weights eight times."),
      },
  }
  write_json(raw_dir / "source-inspection.json", source_inspection)
  write_json(raw_dir / "model.json", model)

  codegen_checks = [
      check("ocloc_compile_passed", compile_result["returncode"] == 0),
      check("exact_ptl_device_selected", "ptl-h-a0" in compile_text),
      check("ocloc_disassembly_passed", disasm_result["returncode"] == 0),
      check("three_m8_u4_dpas_instructions_present",
            assembly.lower().count("dpas.8x8") >= 3,
            observed=assembly.lower().count("dpas.8x8")),
      check("u4_source_precision_present", ":u4" in assembly.lower()),
      check("f32_affine_mad_present", "mad (16|m0)" in assembly.lower()),
      check("swiglu_exp_present", "math.exp" in assembly.lower()),
      check("both_kernels_report_dpas", ze_info.count("has_dpas:        true") >= 2),
      check("pinned_onednn_source_commit", source_commit == ONEDNN_COMMIT,
            observed=source_commit, required=ONEDNN_COMMIT),
      check("pinned_onednn_source_clean", not source_dirty),
  ]
  source_checks = [
      check("f32_accumulator_available",
            source_inspection["facts"]["accumulator_is_f32"]),
      check("epilogue_executes_before_destination_store",
            source_inspection["facts"]["post_ops_execute_before_store"]),
      check("preflight_affine_precedes_f16_store",
            source_inspection["facts"]["preflight_affine_precedes_f16_store"]),
      check("post_affine_f16_oracle_roundtrip_passes",
            f16_oracle["mismatch_count"] == 0 and
            f16_oracle["max_abs_diff"] <= 5e-3),
      check("direct_token_mapping_without_grouped_gather_present", False,
            reason="the selected microkernel requires compact src_offset rows"),
      check("no_materialized_contribution_plane", False,
            reason="expert-owned down workgroups require weighted partial output"),
      check("deterministic_cross_expert_reduction_in_single_kernel", False,
            reason=source_inspection["facts"]["reason"]),
  ]
  timing = model["timing_us"]
  performance_checks = [
      check("source_realizable_relaxed_floor_below_cap",
            timing["source_realizable_relaxed_floor"] <= timing["cap"],
            observed_us=timing["source_realizable_relaxed_floor"],
            required_us=timing["cap"]),
      check("complete_upper_bound_demonstrated", False,
            reason=(
                "the optimistic floor already exceeds the cap while omitting "
                "SwiGLU exp and synchronization; no complete upper bound exists")),
  ]
  evidence_checks_passed = all(row["pass"] for row in codegen_checks)
  architecture_checks_passed = all(row["pass"] for row in source_checks)
  performance_checks_passed = all(row["pass"] for row in performance_checks)
  required_checks_passed = (
      evidence_checks_passed and architecture_checks_passed and
      performance_checks_passed)
  disposition = (
      "admit_one_real_in_core_affine_grouped_kernel"
      if required_checks_passed else
      "reject_in_core_affine_grouped_codegen_on_aggregation_and_floor")
  result = {
      "architecture_checks_passed": architecture_checks_passed,
      "checks": codegen_checks + source_checks + performance_checks,
      "codegen_checks_passed": evidence_checks_passed,
      "created_at": created_at,
      "disposition": disposition,
      "f16_post_affine_oracle_roundtrip": f16_oracle,
      "git": git_state(),
      "model": model,
      "onednn_source": {
          "commit": source_commit,
          "path": str(args.onednn_source),
      },
      "performance_checks_passed": performance_checks_passed,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "kernel": str(KERNEL.relative_to(ROOT)),
          "kernel_sha256": sha256_file(KERNEL),
          "seq649": str(args.seq649.relative_to(ROOT)),
          "seq650": str(args.seq650.relative_to(ROOT)),
          "swiglu_oracle": str(args.swiglu.relative_to(ROOT)),
          "swiglu_sha256": SWIGLU_SHA256,
          "target_contract": str(TARGET_CONTRACT.relative_to(ROOT)),
      },
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", result)
  write_json(out_dir / "correctness.json", {
      "applicable": False,
      "checks": codegen_checks + source_checks,
      "post_affine_f16_oracle_roundtrip": f16_oracle,
      "reason": "static codegen feasibility; no real component was authorized",
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "static source/ISA and traffic feasibility gate",
  })
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "seq649_three_core_minimum_us",
       "value": timing["seq649_three_core_minimum"]},
      {"metric": "source_realizable_relaxed_floor_us",
       "value": timing["source_realizable_relaxed_floor"]},
      {"metric": "complete_cap_us", "value": timing["cap"]},
      {"metric": "floor_over_cap_us",
       "value": timing["source_realizable_floor_over_cap"]},
      {"metric": "post_affine_f16_mismatch_count",
       "value": f16_oracle["mismatch_count"]},
      {"metric": "required_checks_passed", "value": required_checks_passed},
  ]
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in metrics:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  failed = [row["name"] for row in result["checks"] if not row["pass"]]
  (out_dir / "summary.md").write_text("\n".join([
      "# In-core exact-Q4_K affine codegen feasibility gate",
      "",
      "- target: `0xb080` / `ptl-h-a0`",
      f"- DPAS instructions observed: `{assembly.lower().count('dpas.8x8')}`",
      f"- post-affine F16 oracle mismatches: `{f16_oracle['mismatch_count']}`",
      f"- direct token-map grouped source: `false`",
      f"- deterministic no-plane cross-expert reduction: `false`",
      f"- optimistic source-realizable relaxed floor / cap: "
      f"`{timing['source_realizable_relaxed_floor']:.3f} / {timing['cap']:.3f} us`",
      f"- failed checks: `{failed}`",
      f"- required checks passed: `{str(required_checks_passed).lower()}`",
      f"- disposition: `{disposition}`",
      "",
      "The local epilogue arithmetic codegens correctly, but the selected",
      "expert-owned grouped topology cannot perform a deterministic eight-expert",
      "token reduction without a partial plane or a different work owner. The",
      "relaxed partial-plane model already misses the cap while omitting exp and",
      "synchronization. This closes the architecture before a real kernel.",
      "",
  ]), encoding="utf-8")
  print(json.dumps({
      "disposition": disposition,
      "floor_us": timing["source_realizable_relaxed_floor"],
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_checks_passed,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
