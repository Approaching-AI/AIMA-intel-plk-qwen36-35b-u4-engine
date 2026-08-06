#!/usr/bin/env python3
"""Record the terminal oneDNN exact-Q6/per-16 codegen capability gate."""

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
SCHEMA = "intel-qwen36-onednn-grouped-q6-exact-preflight-gate-v0"
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
COMPONENT_COSINE_MIN = 0.999
COMPONENT_RELATIVE_L2_MAX = 0.002
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
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.warmup, args.repeat, args.jobs, args.timeout_s) <= 0:
    parser.error("warmup, repeat, jobs, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / (
        f"output/onednn-grouped-q6-exact-preflight-gate-{stamp}")
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


def probe_command(binary: Path, args: argparse.Namespace,
                  payloads: dict[str, Path], encoding: str,
                  warmup: int, repeat: int) -> list[str]:
  return [
      str(binary), "--model", str(args.model), "--tensor", TENSOR,
      "--swiglu", str(payloads["swiglu"]),
      "--topk", str(payloads["topk"]), "--topk-stride", "1024",
      "--oracle", str(payloads["oracle"]), "--encoding", encoding,
      "--warmup", str(warmup), "--repeat", str(repeat),
      "--cap-us", str(CAP_US),
  ]


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
      if args.skip_onednn_build else
      run_env(onednn_build_command, args))
  BASE.write_run_logs(raw, "onednn-build", onednn_build)

  binary = raw / "iq36-onednn-grouped-q6-exact-preflight"
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

  u8_command = probe_command(binary, args, payloads, "u8-zp32", 1, 1)
  s8_command = probe_command(
      binary, args, payloads, "s8", args.warmup, args.repeat)
  u8_runs: list[dict[str, Any]] = []
  s8_runs: list[dict[str, Any]] = []
  s8_probes: list[dict[str, Any]] = []
  for label in ("primary", "confirm"):
    u8_result = (
        run_env(u8_command, args) if compile_result["returncode"] == 0 else
        failed_run(u8_command, "compile failed"))
    BASE.write_run_logs(raw, f"u8-zp32-{label}", u8_result)
    u8_runs.append(u8_result)
    s8_result = (
        run_env(s8_command, args) if compile_result["returncode"] == 0 else
        failed_run(s8_command, "compile failed"))
    BASE.write_run_logs(raw, f"s8-{label}", s8_result)
    s8_runs.append(s8_result)
    s8_probes.append(parse_probe(s8_result))

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
      check("u8_zp32_codegen_crashes_deterministically",
            all(row["returncode"] in (-11, 139) for row in u8_runs),
            returncodes=[row["returncode"] for row in u8_runs]),
  ]
  correctness_checks: list[dict[str, Any]] = []
  performance_checks: list[dict[str, Any]] = []
  for label, run_result, probe in zip(
      ("primary", "confirm"), s8_runs, s8_probes):
    comparison = probe.get("comparison", {})
    scale_probe = probe.get("scale_granularity_probe", {})
    correctness_checks += [
        check(f"{label}_s8_runs_full_all_value_comparison",
              run_result["returncode"] == 2 and
              comparison.get("compared_value_count") == 16_777_216 and
              comparison.get("finite") is True),
        check(f"{label}_exact_per16_cpu_sample_matches_oracle",
              float(scale_probe.get(
                  "exact_per16_vs_oracle_max_abs", float("inf"))) <= 1e-5,
              scale_granularity_probe=scale_probe),
        check(f"{label}_jit_uses_first_scale_for_each_k32_pair",
              float(scale_probe.get(
                  "gpu_vs_effective_per32_max_abs", float("inf"))) <= 1e-5 and
              float(scale_probe.get(
                  "gpu_vs_exact_per16_max_abs", 0.0)) > 1e-3,
              scale_granularity_probe=scale_probe),
        check(f"{label}_s8_per16_component_contract_rejected",
              probe.get("correctness_pass") is False and
              float(comparison.get("cosine", 1.0)) < COMPONENT_COSINE_MIN and
              float(comparison.get("relative_l2", 0.0)) >
                  COMPONENT_RELATIVE_L2_MAX,
              comparison=comparison),
    ]
    performance_checks.append(
        check(f"{label}_raw_codegen_core_is_below_fixed_cap",
              probe.get("performance_pass") is True and
              float(probe.get("minimum_us", float("inf"))) <= CAP_US,
              minimum_us=probe.get("minimum_us"), cap_us=CAP_US,
              raw_core_only=probe.get("raw_core_only")))

  evidence_passed = all(row["pass"] for row in evidence_checks)
  diagnosis_passed = all(row["pass"] for row in correctness_checks)
  raw_timing_passed = all(row["pass"] for row in performance_checks)
  required_passed = evidence_passed and diagnosis_passed and raw_timing_passed
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
          "cosine_min": COMPONENT_COSINE_MIN,
          "relative_l2_max": COMPONENT_RELATIVE_L2_MAX,
          "finite_outputs_required": True,
      },
      "fixed_layer39_cap_us": CAP_US,
      "source_patch": {
          "path": str(ONEDNN_PATCH.relative_to(ROOT)),
          "sha256": BASE.sha256_file(ONEDNN_PATCH),
          "onednn_commit": source_commit,
      },
      "u8_zp32_returncodes": [row["returncode"] for row in u8_runs],
      "s8_probes": {"primary": s8_probes[0], "confirm": s8_probes[1]},
      "checks": evidence_checks + correctness_checks + performance_checks,
      "evidence_checks_passed": evidence_passed,
      "terminal_diagnosis_passed": diagnosis_passed,
      "raw_timing_checks_passed": raw_timing_passed,
      "required_checks_passed": required_passed,
      "component_accepted": False,
      "disposition": (
          "reject_onednn_exact_per16_codegen_change_representation_family"
          if required_passed else "incomplete_exact_per16_codegen_rejection"),
      "speedup_claims_allowed": False,
  }
  BASE.write_json(out / "result.json", result)
  BASE.write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "checks": correctness_checks,
      "required_checks_passed": diagnosis_passed,
      "component_accepted": False,
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
      "component_accepted": False,
      "speedup_claims_allowed": False,
  })
  minimums = [float(probe.get("minimum_us", float("inf")))
              for probe in s8_probes]
  relative_l2 = [
      float((probe.get("comparison") or {}).get("relative_l2", float("inf")))
      for probe in s8_probes]
  summary = [
      "# oneDNN grouped exact-Q6/per-16 preflight gate",
      "",
      f"- Rejection evidence complete: `{str(required_passed).lower()}`",
      f"- U8 plus per-16 zero-point return codes: "
      f"`{[row['returncode'] for row in u8_runs]}`",
      f"- S8 raw-core primary/confirm: `{minimums[0]:.3f} / "
      f"{minimums[1]:.3f} us` versus `{CAP_US:.3f} us`",
      f"- S8 primary/confirm relative L2: `{relative_l2[0]:.9f} / "
      f"{relative_l2[1]:.9f}`",
      "- The S8 dot product is correct, but the generated kernel applies one "
      "weight scale to both 16-value halves of each K32 DPAS group.",
      "- This is a component-route rejection, not a product speed claim.",
      "",
  ]
  (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out),
      "required_checks_passed": required_passed,
      "component_accepted": False,
      "u8_zp32_returncodes": [row["returncode"] for row in u8_runs],
      "s8_minimum_us": minimums,
      "s8_relative_l2": relative_l2,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
