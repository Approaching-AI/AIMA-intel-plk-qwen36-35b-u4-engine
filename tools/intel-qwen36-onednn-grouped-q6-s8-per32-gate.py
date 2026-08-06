#!/usr/bin/env python3
"""Gate the fixed oneDNN grouped Q6-to-S8-per-K32 requant carrier."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-onednn-grouped-q6-s8-per32-gate-v1"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
BASE_GATE_PATH = ROOT / "tools/intel-qwen36-onednn-q4k-bucket-component-gate.py"
SOURCE = ROOT / "engine/tools/onednn_grouped_q6_exact_preflight.cpp"
GGUF_SOURCE = ROOT / "engine/src/gguf_loader.cpp"
ONEDNN_PATCH = ROOT / "engine/gpu/opencl/onednn-grouped-s8-u4-fused.patch"
CAPTURE = (
    ROOT / "output/all-layer-mixed-component-20260711Tseq671cleanZ/"
    "raw/capture/payloads")
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")
LAYER = 39
TENSOR = "blk.39.ffn_down_exps.weight"
CAP_US = 4316.404
COSINE_MIN = 0.999
RELATIVE_L2_MAX = 0.002
TERNARY_DENSITY_MAX = 0.423513
PATCHED_ONEDNN_PATHS = [
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
    "src/gpu/intel/ocl/engine.cpp",
]


def load_base() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_base_gate", BASE_GATE_PATH)
  if spec is None or spec.loader is None:
    raise SystemExit(f"could not import {BASE_GATE_PATH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--capture", type=Path, default=CAPTURE)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--onednn-source", type=Path,
                      default=BASE.DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path, default=ONEDNN_BUILD)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--skip-onednn-build", action="store_true")
  parser.add_argument("--with-s4-residual", action="store_true")
  parser.add_argument("--with-ternary-residual", action="store_true")
  parser.add_argument("--affine-u8", action="store_true")
  parser.add_argument("--external-affine-zp", action="store_true")
  parser.add_argument("--activation-lsq", action="store_true")
  parser.add_argument("--full-gram-lsq", action="store_true")
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.warmup, args.repeat, args.jobs, args.timeout_s) <= 0:
    parser.error("warmup, repeat, jobs, and timeout must be positive")
  if sum((args.with_s4_residual, args.with_ternary_residual,
          args.affine_u8, args.external_affine_zp,
          args.activation_lsq, args.full_gram_lsq)) > 1:
    parser.error("select at most one alternate representation")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = ("onednn-grouped-q6-full-gram-lsq-gate"
            if args.full_gram_lsq else
            "onednn-grouped-q6-activation-lsq-gate"
            if args.activation_lsq else
            "onednn-grouped-q6-external-affine-zp-gate"
            if args.external_affine_zp else
            "onednn-grouped-q6-affine-u8-per32-gate"
            if args.affine_u8 else
            "onednn-grouped-q6-s8-ternary-residual-census"
            if args.with_ternary_residual else
            "onednn-grouped-q6-s8-s4-residual-gate"
            if args.with_s4_residual else
            "onednn-grouped-q6-s8-per32-gate")
    args.out_dir = ROOT / f"output/{stem}-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def failed_run(command: list[str], reason: str) -> dict[str, Any]:
  return {"command": command, "returncode": 125, "stderr": reason,
          "stdout": "", "timed_out": False}


def git_bytes(root: Path, *parts: str) -> bytes:
  result = subprocess.run(
      ["git", *parts], cwd=root, check=False, capture_output=True)
  return result.stdout if result.returncode == 0 else b""


def run_env(command: list[str], args: argparse.Namespace) -> dict[str, Any]:
  exports = {
      "INTEL_FORCE_PROBE": "b080",
      "DNNL_VERBOSE": "0",
      "DNNL_PRIMITIVE_CACHE_CAPACITY": "0",
      "IQ36_GENERATE_S8_GROUPED": "1",
  }
  export_text = " ".join(
      f"{key}={shlex.quote(value)}" for key, value in exports.items())
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      f"export {export_text} && {shlex.join(command)}")
  return BASE.run(["bash", "-lc", shell], args.timeout_s)


def parse_probe(result: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(result.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def payload(capture: Path, stem: str) -> Path:
  matches = sorted(capture.glob(f"{stem}__tok1023__ord*.bin"))
  if len(matches) != 1:
    raise SystemExit(f"expected one {stem} payload, found {matches}")
  return matches[0]


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  state = BASE.git_state()
  payloads = {
      "swiglu": payload(args.capture, f"ffn_moe_swiglu-{LAYER}"),
      "topk": payload(args.capture, f"ffn_moe_topk-{LAYER}"),
      "oracle": payload(args.capture, f"ffn_moe_down-{LAYER}"),
  }
  required = [
      args.model, args.capture, args.env_script, args.cxx,
      args.onednn_source, args.onednn_build, SOURCE, GGUF_SOURCE,
      ONEDNN_PATCH, *payloads.values(),
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  source_commit = BASE.git_output(args.onednn_source, "rev-parse", "HEAD")
  source_diff = git_bytes(
      args.onednn_source, "diff", "--unified=0", "--",
      *PATCHED_ONEDNN_PATHS)
  source_status = git_bytes(
      args.onednn_source, "status", "--porcelain").decode("utf-8")
  expected_status = sorted(f" M {path}" for path in PATCHED_ONEDNN_PATHS)
  patch_exact = (
      source_diff == ONEDNN_PATCH.read_bytes() and
      sorted(source_status.splitlines()) == expected_status)

  onednn_build_command = [
      "cmake", "--build", str(args.onednn_build), "--target", "dnnl",
      "-j", str(args.jobs),
  ]
  onednn_build = (
      {"command": onednn_build_command, "returncode": 0,
       "stderr": "skipped by request", "stdout": "", "timed_out": False}
      if args.skip_onednn_build else run_env(onednn_build_command, args))
  BASE.write_run_logs(raw, "onednn-build", onednn_build)

  binary = raw / "iq36-onednn-grouped-q6-s8-per32"
  compile_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG", "-fopenmp",
      "-DCL_TARGET_OPENCL_VERSION=300", "-I", str(ROOT / "engine/include"),
      "-I", str(args.onednn_build / "include"),
      "-I", str(args.onednn_source / "include"), str(SOURCE),
      str(GGUF_SOURCE), "-L", str(args.onednn_build / "src"),
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL", "-o",
      str(binary),
  ]
  compile_result = (
      run_env(compile_command, args) if onednn_build["returncode"] == 0 else
      failed_run(compile_command, "oneDNN build failed"))
  BASE.write_run_logs(raw, "compile", compile_result)

  representation = (
      "requant-s8-per32-full-gram-lsq"
      if args.full_gram_lsq else
      "requant-s8-per32-activation-lsq"
      if args.activation_lsq else
      "requant-s8-affine-per32-external-zp"
      if args.external_affine_zp else
      "requant-u8-affine-per32" if args.affine_u8 else
      "requant-s8-per32-ternary-residual"
      if args.with_ternary_residual else
      "requant-s8-per32-s4-residual"
      if args.with_s4_residual else "requant-s8-per32")
  command = [
      str(binary), "--model", str(args.model), "--tensor", TENSOR,
      "--swiglu", str(payloads["swiglu"]),
      "--topk", str(payloads["topk"]), "--topk-stride", "1024",
      "--oracle", str(payloads["oracle"]), "--encoding",
      "u8-affine" if args.affine_u8 else "s8",
      "--representation", representation,
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--cap-us", str(CAP_US),
  ]
  runs: list[dict[str, Any]] = []
  probes: list[dict[str, Any]] = []
  for label in ("primary", "confirm"):
    result = (
        run_env(command, args) if compile_result["returncode"] == 0 else
        failed_run(command, "compile failed"))
    BASE.write_run_logs(raw, label, result)
    runs.append(result)
    probes.append(parse_probe(result))

  affine_codegen_crash = (
      args.affine_u8 and
      all(row["returncode"] in (-11, 139) for row in runs))

  evidence_checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("pinned_onednn_source_commit",
            source_commit == BASE.ONEDNN_COMMIT,
            observed=source_commit, required=BASE.ONEDNN_COMMIT),
      check("onednn_source_diff_exactly_matches_repo_patch", patch_exact,
            patch_sha256=BASE.sha256_file(ONEDNN_PATCH)),
      check("locked_model_and_worst_layer39_capture",
            args.model.resolve() == MODEL.resolve() and
            args.capture.resolve() == CAPTURE.resolve()),
      check("preflight_compiles", compile_result["returncode"] == 0),
  ]
  if args.affine_u8:
    evidence_checks.append(
        check("affine_u8_per32_codegen_crashes_deterministically",
              affine_codegen_crash,
              returncodes=[row["returncode"] for row in runs]))
  correctness_checks: list[dict[str, Any]] = []
  performance_checks: list[dict[str, Any]] = []
  for label, run_result, probe in ([] if affine_codegen_crash else zip(
      ("primary", "confirm"), runs, probes)):
    comparison = probe.get("comparison", {})
    host_probe = probe.get("host_repacked_probe", {})
    requant = probe.get("weight_requantization", {})
    residual = probe.get("residual_correction", {})
    correctness_checks += [
        check(f"{label}_all_values_compared",
              comparison.get("compared_value_count") == 16_777_216 and
              comparison.get("finite") is True),
        check(f"{label}_generated_core_matches_host_repacked_sample",
              float(host_probe.get(
                  "gpu_vs_repacked_max_abs", float("inf"))) <= 1e-5,
              host_repacked_probe=host_probe),
        check(f"{label}_fixed_representation",
              probe.get("encoding") ==
                  ("u8-affine" if args.affine_u8 else "s8") and
              probe.get("representation") == representation and
              requant.get("group_values") == 32 and
              (not args.external_affine_zp or
               probe.get("external_affine_correction") is True) and
              (not (args.activation_lsq or args.full_gram_lsq) or
               (probe.get("activation_lsq") or {}).get(
                   "gram_row_count") == 8192) and
              (not args.full_gram_lsq or
               (probe.get("activation_lsq") or {}).get(
                   "gram_width") == 512) and
              (not (args.with_s4_residual or
                    args.with_ternary_residual) or
               (probe.get("residual_implementation") ==
                    "grouped_gemm:micro" and
                residual.get("group_values") == 32 and
                residual.get("code_max") ==
                    (1 if args.with_ternary_residual else 7))),
              weight_requantization=requant,
              residual_correction=residual),
        check(f"{label}_component_accuracy_contract",
              run_result["returncode"] in (0, 2) and
              probe.get("correctness_pass") is True and
              float(comparison.get("cosine", float("-inf"))) >= COSINE_MIN and
              float(comparison.get("relative_l2", float("inf"))) <=
                  RELATIVE_L2_MAX,
              comparison=comparison),
    ]
    performance_checks.append(
        check(f"{label}_raw_core_below_fixed_cap",
              probe.get("performance_pass") is True and
              float(probe.get("minimum_us", float("inf"))) <= CAP_US,
              minimum_us=probe.get("minimum_us"), cap_us=CAP_US,
              raw_core_only=probe.get("raw_core_only")))

  feasibility_checks: list[dict[str, Any]] = []
  if args.with_ternary_residual:
    for label, probe in zip(("primary", "confirm"), probes):
      residual = probe.get("residual_correction", {})
      feasibility_checks.append(
          check(f"{label}_ternary_density_below_zero_overhead_ceiling",
                float(residual.get("nonzero_density", float("inf"))) <
                    TERNARY_DENSITY_MAX,
                nonzero_count=residual.get("nonzero_count"),
                nonzero_density=residual.get("nonzero_density"),
                required_max=TERNARY_DENSITY_MAX))

  evidence_passed = all(row["pass"] for row in evidence_checks)
  correctness_passed = all(row["pass"] for row in correctness_checks)
  performance_passed = all(row["pass"] for row in performance_checks)
  feasibility_passed = all(row["pass"] for row in feasibility_checks)
  accepted = (not affine_codegen_crash and evidence_passed and
              correctness_passed and
              (feasibility_passed if args.with_ternary_residual else
               performance_passed))
  required_passed = (
      evidence_passed and affine_codegen_crash
      if args.affine_u8 else accepted)
  created_at = iso_now()
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "git": state,
      "model": str(args.model),
      "tensor": TENSOR,
      "capture": str(args.capture),
      "payload_sha256": {
          name: BASE.sha256_file(path) for name, path in payloads.items()},
      "component_contract": {
          "cosine_min": COSINE_MIN,
          "relative_l2_max": RELATIVE_L2_MAX,
          "finite_outputs_required": True,
      },
      "fixed_layer39_cap_us": CAP_US,
      "representation": {
          "name": ("affine U8 with F32 scale and U8 zero point per K32"
                   if args.affine_u8 else
                   "full-Gram activation-LSQ S8 per K32"
                   if args.full_gram_lsq else
                   "shared-Gram activation-LSQ S8 per K32"
                   if args.activation_lsq else
                   "recentered S8 plus external affine zero point per K32"
                   if args.external_affine_zp else
                   "symmetric S8 main plus optimal ternary residual per K32"
                   if args.with_ternary_residual else
                   "symmetric S8 main plus signed-S4 residual per K32"
                   if args.with_s4_residual else
                   "symmetric S8 requantization with F32 scale per K32"),
          "weight_group_values": 32,
          "source_group_values": 256,
          "s4_residual": args.with_s4_residual,
          "ternary_residual": args.with_ternary_residual,
          "affine_u8": args.affine_u8,
          "external_affine_zero_point": args.external_affine_zp,
          "activation_lsq": args.activation_lsq,
          "full_gram_lsq": args.full_gram_lsq,
      },
      "source_patch": {
          "path": str(ONEDNN_PATCH.relative_to(ROOT)),
          "sha256": BASE.sha256_file(ONEDNN_PATCH),
          "onednn_commit": source_commit,
      },
      "probes": {"primary": probes[0], "confirm": probes[1]},
      "checks": (evidence_checks + correctness_checks + performance_checks +
                 feasibility_checks),
      "evidence_checks_passed": evidence_passed,
      "correctness_checks_passed": correctness_passed,
      "performance_checks_passed": performance_passed,
      "feasibility_checks_passed": feasibility_passed,
      "required_checks_passed": required_passed,
      "component_accepted": accepted and not args.with_ternary_residual,
      "sparse_implementation_admitted": (
          accepted if args.with_ternary_residual else False),
      "disposition": (
          "accept_full_gram_lsq_s8_per32_preflight"
          if accepted and args.full_gram_lsq else
          "accept_activation_lsq_s8_per32_preflight"
          if accepted and args.activation_lsq else
          "accept_external_affine_zero_point_preflight"
          if accepted and args.external_affine_zp else
          "reject_affine_u8_per32_codegen_use_external_zero_point"
          if affine_codegen_crash else
          "admit_sparse_ternary_residual_implementation"
          if accepted and args.with_ternary_residual else
          "accept_s8_s4_residual_raw_codegen_preflight"
          if accepted and args.with_s4_residual else
          "accept_s8_per32_raw_codegen_preflight" if accepted else
          "reject_s8_s4_residual_timing"
          if args.with_s4_residual and performance_passed is False else
          "reject_ternary_residual_feasibility"
          if args.with_ternary_residual else
          "reject_affine_u8_per32_representation"
          if args.affine_u8 else
          "reject_external_affine_zero_point_composition"
          if args.external_affine_zp else
          "reject_activation_lsq_s8_per32_quantizer"
          if args.activation_lsq else
          "reject_full_gram_lsq_s8_per32_quantizer"
          if args.full_gram_lsq else
          "reject_s8_per32_requant_representation"),
      "speedup_claims_allowed": False,
  }
  BASE.write_json(out / "result.json", result)
  BASE.write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "checks": correctness_checks,
      "required_checks_passed": correctness_passed,
      "component_accepted": accepted and not args.with_ternary_residual,
      "speedup_claims_allowed": False,
  })
  BASE.write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "artifact": str(out),
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "git": state,
      "required_checks_passed": required_passed,
      "component_accepted": accepted and not args.with_ternary_residual,
      "sparse_implementation_admitted": (
          accepted if args.with_ternary_residual else False),
      "speedup_claims_allowed": False,
  })
  minimums = [float(probe.get("minimum_us", float("inf")))
              for probe in probes]
  relative_l2 = [
      float((probe.get("comparison") or {}).get("relative_l2", float("inf")))
      for probe in probes]
  summary = [
      ("# oneDNN grouped Q6 S8-main plus S4-residual gate"
       if args.with_s4_residual else
       "# Q6 full-Gram activation-LSQ S8-per-K32 gate"
       if args.full_gram_lsq else
       "# Q6 activation-LSQ S8-per-K32 gate"
       if args.activation_lsq else
       "# Q6 recentered-S8 external affine-zero-point gate"
       if args.external_affine_zp else
       "# oneDNN grouped Q6 affine-U8-per-K32 gate"
       if args.affine_u8 else
       "# Q6 optimal-ternary residual feasibility census"
       if args.with_ternary_residual else
       "# oneDNN grouped Q6-to-S8-per-K32 gate"),
      "",
      f"- Gate evidence complete: `{str(required_passed).lower()}`",
      f"- Raw-core primary/confirm: `{minimums[0]:.3f} / "
      f"{minimums[1]:.3f} us` versus `{CAP_US:.3f} us`",
      f"- Primary/confirm relative L2: `{relative_l2[0]:.9f} / "
      f"{relative_l2[1]:.9f}` versus `{RELATIVE_L2_MAX:.3f}`",
      "- This is a raw component preflight, not a product speed claim.",
      "",
  ]
  if args.with_ternary_residual:
    densities = [
        float((probe.get("residual_correction") or {}).get(
            "nonzero_density", float("inf"))) for probe in probes]
    summary.insert(
        -2, f"- Ternary nonzero density: `{densities[0]:.9f} / "
        f"{densities[1]:.9f}` versus `<{TERNARY_DENSITY_MAX:.6f}`")
  if affine_codegen_crash:
    summary = [
        "# oneDNN grouped Q6 affine-U8-per-K32 gate",
        "",
        f"- Rejection evidence complete: `{str(required_passed).lower()}`",
        f"- Primitive/JIT return codes: "
        f"`{[row['returncode'] for row in runs]}`",
        "- No generated component exists; use an external compact "
        "zero-point correction or change representation.",
        "",
    ]
  (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out),
      "required_checks_passed": required_passed,
      "s8_minimum_us": minimums,
      "s8_relative_l2": relative_l2,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
