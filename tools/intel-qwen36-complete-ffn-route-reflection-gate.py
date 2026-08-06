#!/usr/bin/env python3
"""Close the routed-only M8 route and select one complete-FFN source gate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-complete-ffn-route-reflection-gate-v0"
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OV_MODEL = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
TENSOR_INDEX = ROOT / (
    "output/r1-native-gguf-load-map-20260705T071855Z/tensor-index.jsonl")
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
SEQ759 = ROOT / "output/fused-expert-ffn-design-gate-20260712Tseq759cleanZ/result.json"
SEQ761 = ROOT / "output/fused-expert-ffn-m8-source-gate-20260712Tseq761cleanZ/result.json"
PROFILE = ROOT / "output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/profile.json"
PROBE = ROOT / "tools/intel-qwen36-openvino-moe-route-measure.py"
FFN_CAP_US = 6250.0
TOKENS = 1024
ROUTED_ASSIGNMENTS = 8192


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--openvino-source", type=Path, default=OV_SOURCE)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--openvino-model", type=Path, default=OV_MODEL)
  parser.add_argument("--llama-source", type=Path, default=LLAMA_SOURCE)
  parser.add_argument("--repeat", type=int, default=6)
  parser.add_argument("--timeout-s", type=int, default=600)
  args = parser.parse_args()
  if args.repeat < 4 or args.timeout_s <= 0:
    parser.error("repeat must be >=4 and timeout-s must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/complete-ffn-route-reflection-gate-{stamp}"
  return args


def read_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise RuntimeError(f"expected object: {path}")
  return value


def git_output(*parts: str, cwd: Path = ROOT) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=cwd, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def run_probe(mode: str, args: argparse.Namespace) -> dict[str, Any]:
  command = [
      str(args.openvino_python), str(PROBE), "--mode", mode,
      "--model", str(args.openvino_model), "--seq-len", str(TOKENS),
      "--repeat", str(args.repeat),
  ]
  try:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True,
        timeout=args.timeout_s, check=False, encoding="utf-8",
        errors="replace")
    return {
        "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command, "returncode": 124,
        "stdout": error.stdout or "", "stderr": error.stderr or "",
        "timed_out": True,
    }


def write_run(raw: Path, label: str, row: dict[str, Any]) -> None:
  (raw / f"{label}.command.json").write_text(
      json.dumps(row["command"], indent=2) + "\n", encoding="utf-8")
  (raw / f"{label}.stdout").write_text(
      str(row["stdout"]), encoding="utf-8")
  (raw / f"{label}.stderr").write_text(
      str(row["stderr"]), encoding="utf-8")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def tensor_rows() -> dict[str, dict[str, Any]]:
  wanted = {
      "blk.27.ffn_down_exps.weight",
      "blk.27.ffn_down_shexp.weight",
      "blk.27.ffn_gate_inp_shexp.weight",
      "blk.27.ffn_gate_shexp.weight",
      "blk.27.ffn_gate_up_exps.weight",
      "blk.27.ffn_up_shexp.weight",
  }
  found: dict[str, dict[str, Any]] = {}
  for line in TENSOR_INDEX.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row.get("name") in wanted:
      found[str(row["name"])] = row
  return found


def source_contains_all(text: str, needles: list[str]) -> bool:
  return all(needle in text for needle in needles)


def finite_ratio(numerator: float, denominator: float) -> float | None:
  if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
    return None
  return numerator / denominator


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)

  required = [
      args.openvino_source, args.openvino_python, args.openvino_model,
      args.llama_source, TENSOR_INDEX, MODEL_CONTRACT, SEQ759, SEQ761,
      PROFILE, PROBE,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing inputs: " + ", ".join(missing))

  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  ov_commit = git_output("rev-parse", "HEAD", cwd=args.openvino_source)
  ov_dirty = git_output("status", "--porcelain", cwd=args.openvino_source)

  ov_impl_path = args.openvino_source / (
      "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/"
      "moe_3gemm_swiglu_opt.cpp")
  ov_gen_path = args.openvino_source / (
      "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/"
      "moe_3gemm_gen_micro.cpp")
  ov_kernel_path = args.openvino_source / (
      "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_gemm.cl")
  llama_path = args.llama_source / "src/models/qwen3next.cpp"
  for path in (ov_impl_path, ov_gen_path, ov_kernel_path, llama_path):
    if not path.exists():
      raise SystemExit(f"missing source input: {path}")

  ov_impl = ov_impl_path.read_text(encoding="utf-8")
  ov_gen = ov_gen_path.read_text(encoding="utf-8")
  ov_kernel = ov_kernel_path.read_text(encoding="utf-8")
  llama = llama_path.read_text(encoding="utf-8")
  contract = read_json(MODEL_CONTRACT)
  seq759 = read_json(SEQ759)
  seq761 = read_json(SEQ761)
  profile = read_json(PROFILE)
  tensors = tensor_rows()

  expected_dims = {
      "blk.27.ffn_gate_up_exps.weight": [2048, 1024, 256],
      "blk.27.ffn_down_exps.weight": [512, 2048, 256],
      "blk.27.ffn_gate_shexp.weight": [2048, 512],
      "blk.27.ffn_up_shexp.weight": [2048, 512],
      "blk.27.ffn_down_shexp.weight": [512, 2048],
      "blk.27.ffn_gate_inp_shexp.weight": [2048],
  }
  shapes_match = (
      set(tensors) == set(expected_dims) and
      all(tensors[name].get("dims") == dims
          for name, dims in expected_dims.items()))

  hidden = 2048
  inter = 512
  routed_true_macs = ROUTED_ASSIGNMENTS * 3 * hidden * inter
  shared_true_macs = TOKENS * 3 * hidden * inter
  complete_true_macs = routed_true_macs + shared_true_macs
  routed_padded_macs = int(seq759["design"]["matrix_macs"])
  fixed_nonmatrix_us = float(seq759["design"]["fixed_nonmatrix_us"])
  complete_m8_macs = routed_padded_macs + shared_true_macs
  matrix_budget_us = FFN_CAP_US - fixed_nonmatrix_us
  corrected_m8_rate_tmac_s = complete_m8_macs / matrix_budget_us / 1e6
  projected_at_old_floor_us = complete_m8_macs / 5.4e6 + fixed_nonmatrix_us

  probe_runs: dict[str, dict[str, Any]] = {}
  probe_rows: dict[str, dict[str, Any]] = {}
  for mode in ("default_grouped", "micro", "onednn_loop"):
    run = run_probe(mode, args)
    write_run(raw, mode, run)
    probe_runs[mode] = run
    try:
      row = json.loads(run["stdout"])
    except (json.JSONDecodeError, TypeError):
      row = {}
    probe_rows[mode] = row if isinstance(row, dict) else {}

  medians = {
      mode: float(row.get("warm_wall_median_ms", math.inf))
      for mode, row in probe_rows.items()
  }
  micro_vs_grouped = finite_ratio(
      medians["default_grouped"], medians["micro"])
  micro_vs_loop = finite_ratio(
      medians["onednn_loop"], medians["micro"])
  versions = {str(row.get("openvino_version", ""))
              for row in probe_rows.values()}
  probes_complete = all(
      run["returncode"] == 0 and bool(probe_rows[mode])
      for mode, run in probe_runs.items())
  profile_version = str(profile.get("openvino_version", ""))
  source_version_exact = (
      len(versions) == 1 and profile_version in versions and
      "90214e5be05" in next(iter(versions), ""))

  checks = [
      check("repository_clean_at_gate", dirty == "", dirty_paths=dirty.splitlines()),
      check("exact_openvino_source_commit", ov_commit == OV_COMMIT and ov_dirty == "",
            expected=OV_COMMIT, observed=ov_commit,
            source_dirty_paths=ov_dirty.splitlines()),
      check("installed_openvino_version_matches_source", source_version_exact,
            versions=sorted(versions)),
      check("model_contract_names_shared_expert_and_moe_residual",
            "shared_expert" in contract.get("boundary_types", []) and
            "moe_residual" in contract.get("boundary_types", [])),
      check("llama_complete_ffn_boundary_includes_shared_gate_and_final_add",
            source_contains_all(llama, [
                'cb(moe_out, "ffn_moe_out", il);',
                'cb(ffn_shexp, "ffn_shexp", il);',
                'cb(shared_gate, "shared_expert_gate", il);',
                'ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);',
                'cur = ggml_add(ctx0, moe_out, ffn_shexp);',
                'cb(cur, "ffn_out", il);',
            ])),
      check("layer27_shared_and_routed_shapes_exact", shapes_match,
            observed={name: row.get("dims") for name, row in tensors.items()}),
      check("seq761_terminal_source_failure_present",
            seq761.get("disposition") ==
            "reject_fixed_m8_expert_major_source_below_matrix_rate" and
            seq761.get("required_checks_passed") is False),
      check("old_5p4_floor_fails_restored_complete_boundary",
            projected_at_old_floor_us > FFN_CAP_US and
            corrected_m8_rate_tmac_s > 5.4,
            projected_us=projected_at_old_floor_us,
            cap_us=FFN_CAP_US,
            corrected_rate_tmac_s=corrected_m8_rate_tmac_s),
      check("openvino_default_grouped_and_micro_routes_source_exact",
            source_contains_all(ov_impl, [
                "use_grouped_gemm_prefill = true;",
                "grouped_gemm supersedes micro_gemm",
                'std::getenv("MOE_USE_MICRO_GEMM_PREFILL")',
                "exec_prefill_micro_gemm",
                "exec_prefill_grouped_gemm",
            ])),
      check("openvino_micro_f16_u4_active_expert_abi_source_exact",
            source_contains_all(ov_gen, [
                "problem_moe.Ta_ext = convert_type",
                "problem_moe.Tb = problem_moe.Tb_ext = micro::Type::f16;",
                "problem_moe.Tc = micro::Type::f32;",
                "sizes.n = static_cast<int32_t>(n);",
                "micro::select_gemm_microkernel",
                "MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS",
                "MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT",
                "MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT",
                "generateShim",
            ]) and source_contains_all(ov_kernel, [
                "experts_ids", "input_offset_per_expert", "n_array",
                "WEIGHT_SCALE_DT", "USE_SLM",
            ])),
      check("three_openvino_route_differentials_complete", probes_complete,
            returncodes={mode: run["returncode"]
                         for mode, run in probe_runs.items()}),
      check("micro_route_is_target_measured_best_of_three",
            probes_complete and medians["micro"] < medians["default_grouped"]
            and medians["micro"] < medians["onednn_loop"],
            medians_ms=medians, micro_vs_grouped=micro_vs_grouped,
            micro_vs_onednn_loop=micro_vs_loop),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  selected = (
      "native_prefill_f16_u4_active_expert_microkernel_complete_ffn_source_gate"
      if required_checks_passed else
      "native_prefill_product_route_reflection_required")
  disposition = (
      "close_routed_m8_restore_complete_ffn_select_f16_u4_microkernel_source"
      if required_checks_passed else
      "complete_ffn_route_reflection_incomplete")

  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "commit": commit,
      "evaluation_completed": probes_complete,
      "required_checks_passed": required_checks_passed,
      "disposition": disposition,
      "selected_next_route": selected,
      "checks": checks,
      "boundary_correction": {
          "prior_oracle_end": "ffn_moe_out",
          "required_oracle_end": "ffn_out",
          "routed_true_macs": routed_true_macs,
          "shared_true_macs": shared_true_macs,
          "complete_true_macs": complete_true_macs,
          "routed_m8_padded_macs": routed_padded_macs,
          "complete_m8_matrix_macs": complete_m8_macs,
          "shared_fraction_of_true_matrix_work":
              shared_true_macs / complete_true_macs,
          "shared_work_uplift_over_routed":
              shared_true_macs / routed_true_macs,
          "fixed_nonmatrix_us": fixed_nonmatrix_us,
          "matrix_budget_us": matrix_budget_us,
          "old_rate_floor_tmac_s": 5.4,
          "corrected_m8_rate_floor_tmac_s": corrected_m8_rate_tmac_s,
          "projected_complete_m8_us_at_old_floor": projected_at_old_floor_us,
          "registered_cap_us": FFN_CAP_US,
      },
      "source_route": {
          "openvino_source_commit": ov_commit,
          "installed_versions": sorted(versions),
          "default_route": "onednn_grouped_gemm",
          "selected_architecture": "active_expert_f16_by_u4_microkernel",
          "abi": [
              "f16_gathered_input", "u4_or_i4_weight", "f32_output",
              "active_expert_ids_i32", "expert_start_offsets_i32",
              "expert_token_lengths_i32", "runtime_m_i32", "runtime_k_i32",
              "f16_per_group_scales", "optional_quantized_zero_points",
              "optional_local_slm",
          ],
          "codegen_boundary": (
              "OpenVINO and oneDNN are build-time source/generator dependencies; "
              "only emitted OpenCL source or device binary may enter the native runtime"),
          "inference_note": (
              "Standalone offline extraction is source-feasible because the exact "
              "generator returns KernelData source, shim, workgroups, scalar ABI, and "
              "SLM size; the next source gate must prove the build and native-only maps."),
      },
      "route_differential": {
          "seq_len": TOKENS,
          "first_row_excluded": True,
          "warm_wall_medians_ms": medians,
          "micro_speedup_vs_default_grouped": micro_vs_grouped,
          "micro_speedup_vs_onednn_loop": micro_vs_loop,
          "not_product_speed_evidence": True,
          "reason": (
              "Synthetic hidden-body inputs and PERF_COUNT isolate architecture direction; "
              "they do not replace the raw-prompt OpenVINO denominator."),
          "prior_profile_wall_ms": profile["runs"][0]["wall_ms"],
          "prior_profiled_sum_ms": profile["runs"][0]["profiled_sum_ms"],
      },
      "next_gate_contract": {
          "single_fixed_route": selected,
          "capture": [
              "attn_post_norm", "ffn_moe_topk", "ffn_moe_weights_norm",
              "ffn_moe_out", "ffn_shexp", "shared_expert_gate",
              "shared_expert_gate_sigmoid", "ffn_shexp_gated", "ffn_out",
          ],
          "performance_cap_us": FFN_CAP_US,
          "correctness": {
              "cosine_min": 0.999, "relative_l2_max": 0.002,
              "finite": True, "oracle_end": "ffn_out",
          },
          "runtime": {
              "native_only_maps": True, "timed_host_upload_bytes": 0,
              "timed_host_readback_bytes": 0,
          },
          "stop_condition": (
              "A codegen/build, compiler-resource, complete-ffn correctness, cap, or "
              "0.5% repeat/confirm failure closes this source without datatype, tile, "
              "subgroup, workgroup, expert-bucket, or synthetic-assignment sweeps."),
      },
      "inputs": {
          "seq759": str(SEQ759.relative_to(ROOT)),
          "seq761": str(SEQ761.relative_to(ROOT)),
          "profile": str(PROFILE.relative_to(ROOT)),
          "model_contract": str(MODEL_CONTRACT.relative_to(ROOT)),
          "tensor_index": str(TENSOR_INDEX.relative_to(ROOT)),
          "openvino_source": str(args.openvino_source.resolve()),
          "llama_source": str(args.llama_source.resolve()),
      },
  }
  (out / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out / "summary.md").write_text(
      "\n".join([
          "# Complete FFN route reflection gate", "",
          f"- disposition: `{disposition}`",
          f"- required checks passed: `{str(required_checks_passed).lower()}`",
          f"- routed-only seq761 matrix: `{seq761['matrix_minima_us'][0]:.3f} / {seq761['matrix_minima_us'][1]:.3f} us`",
          f"- restored shared-expert matrix work: `{shared_true_macs / 1e9:.3f}B MACs`",
          f"- corrected fixed-M8 rate floor: `{corrected_m8_rate_tmac_s:.3f} TMAC/s`",
          f"- projected fixed-M8 complete boundary at 5.4 TMAC/s: `{projected_at_old_floor_us:.3f} us`",
          f"- warmed route medians (grouped / micro / loop): `{medians['default_grouped']:.3f} / {medians['micro']:.3f} / {medians['onednn_loop']:.3f} ms`",
          f"- selected next route: `{selected}`", "",
          "The OpenVINO route differential is architecture evidence only; it is not a product speed or correctness claim.", "",
      ]), encoding="utf-8")
  print(json.dumps(result, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
