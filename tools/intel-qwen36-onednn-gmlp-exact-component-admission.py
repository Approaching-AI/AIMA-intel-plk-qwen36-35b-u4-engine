#!/usr/bin/env python3
"""Admit one isolated exact-shape oneDNN PR5059 component experiment.

The gate performs no checkout, configure, compile, GPU context creation, model
load, or inference.  It proves that the upstream test accepts the two locked
U4 group64 shapes through GMLP_TEST, derives the component kill-number from the
frozen seq2202 prefill LCB miss, and emits a strictly serial build/run plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-onednn-gmlp-exact-component-admission-v0"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
SEQ2202 = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-formal-abba8-"
    "20260731Tseq2202-clean/result.json")
SEQ2221A = ROOT / (
    "output/openvino-upstream-gated-mlp-micro-horz-bound-"
    "20260731Tseq2221a-clean/metrics.json")

ONEDNN_REPO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
SOURCE_WORKTREE = Path(
    "/home/intel/intel-qwen36-r0/source/oneDNN-862174-gmlp-exact")
BUILD_DIR = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-exact")

CONDA = Path("/home/intel/intel-box-env/conda")
CMAKE = CONDA / "bin/cmake"
NINJA = CONDA / "bin/ninja"
CC = CONDA / "bin/cc"
CXX = CONDA / "bin/c++"
OPENCL_INCLUDE = CONDA / "include"
OPENCL_LIBRARY = CONDA / "lib/libOpenCL.so"
SYSTEMD_RUN = Path("/usr/bin/systemd-run")

TEST_SOURCE = "tests/gtests/internals/test_gated_mlp.cpp"
TEST_CMAKE = "tests/gtests/internals/CMakeLists.txt"
IMPL_LIST = "src/gpu/gpu_gated_mlp_list.cpp"
MICRO_HPP = "src/gpu/intel/gated_mlp/micro_horz.hpp"
MICRO_CPP = "src/gpu/intel/gated_mlp/micro_horz.cpp"

DECODE_ENV = "1 2048 512 16 4 1 64 16 8"
PREFILL_ENV = "2048 2048 512 16 4 1 64 16 8"
PAIR_COUNT = 8
BOOTSTRAP_SEED = 22230
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3
EXPECTED_SHA256 = {
    SEQ2202: (
        "ab3132c45efa7d67cce04befac93f34b2d6ee5563533c325cfaa7c0e66b2d06f"),
    SEQ2221A: (
        "b60ac54e7588b0cb328a594a30e3c7ed271cc3eb837d67fff7eee1449f46b6ee"),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--memory-stop-gib", default=4.0, type=float)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0:
    parser.error("memory stop must be positive")
  return args


def run(command: list[str], cwd: Path = ROOT) -> str:
  result = subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n{result.stderr}")
  return result.stdout


def git(cwd: Path, *args: str) -> str:
  return run(["git", *args], cwd=cwd).strip()


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, samples: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  samples.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def repository_state(output: Path) -> dict[str, Any]:
  head = git(ROOT, "rev-parse", "HEAD")
  upstream = git(ROOT, "rev-parse", "@{u}")
  output_rel = relative(output)
  dirty = []
  for row in git(
      ROOT, "status", "--porcelain", "--untracked-files=all").splitlines():
    path = row[3:]
    if path == output_rel or path.startswith(output_rel + "/"):
      continue
    dirty.append(row)
  return {
      "branch": git(ROOT, "branch", "--show-current"),
      "commit": head,
      "upstream_commit": upstream,
      "pushed": head == upstream,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def object_text(path: str) -> str:
  return git(ONEDNN_REPO, "show", f"{ONEDNN_HEAD}:{path}")


def remote_pr_head() -> str:
  output = run([
      "git", "ls-remote",
      "https://github.com/uxlfoundation/oneDNN.git",
      "refs/pull/5059/head"])
  rows = [row for row in output.splitlines() if row.strip()]
  if len(rows) != 1:
    raise RuntimeError("PR5059 head lookup did not return exactly one row")
  return rows[0].split()[0]


def qk_kill_number(seq2202: dict[str, Any]) -> dict[str, Any]:
  phase = seq2202["phase_inference"]["prefill_tokens_s"]
  lower_bound = float(phase["lower_confidence_bound_ratio"])
  target = float(seq2202["target_ratio"])
  candidate_walls = []
  for block in seq2202["runs"].values():
    for slot, row in block.items():
      if slot.startswith("qk-"):
        candidate_walls.append(float(row["result"]["prefill_wall_ms"]))
  median_wall = statistics.median(candidate_walls)
  required_fraction = 1.0 - lower_bound / target
  required_total_ms = median_wall * required_fraction
  exact_per_layer_ms = required_total_ms / 40.0
  registered_per_layer_ms = math.ceil(
      exact_per_layer_ms * 1_000_000) / 1_000_000
  return {
      "target_ratio": target,
      "observed_lcb_ratio": lower_bound,
      "ratio_gap": target - lower_bound,
      "required_candidate_wall_fraction": required_fraction,
      "candidate_prefill_wall_ms": candidate_walls,
      "candidate_prefill_wall_median_ms": median_wall,
      "required_total_saving_ms": required_total_ms,
      "shared_layer_count": 40,
      "exact_required_saving_ms_per_layer": exact_per_layer_ms,
      "registered_saving_ms_per_layer": registered_per_layer_ms,
      "registered_saving_us_per_layer": registered_per_layer_ms * 1000.0,
      "interpretation": (
          "A standalone component must show a one-sided 95% saving LCB at "
          "least this large for MB2048 before product integration is worth "
          "considering; an integrated bundle still needs a fresh ABBA gate."),
  }


def configure_command() -> list[str]:
  return [
      str(CMAKE),
      "-S", str(SOURCE_WORKTREE),
      "-B", str(BUILD_DIR),
      "-G", "Ninja",
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DCMAKE_C_COMPILER={CC}",
      f"-DCMAKE_CXX_COMPILER={CXX}",
      f"-DCMAKE_MAKE_PROGRAM={NINJA}",
      f"-DCMAKE_PREFIX_PATH={CONDA}",
      f"-DOpenCL_INCLUDE_DIR={OPENCL_INCLUDE}",
      f"-DOpenCL_LIBRARY={OPENCL_LIBRARY}",
      "-DDNNL_BUILD_TESTS=ON",
      "-DDNNL_BUILD_EXAMPLES=OFF",
      "-DONEDNN_BUILD_GRAPH=OFF",
      "-DDNNL_CPU_RUNTIME=NONE",
      "-DDNNL_GPU_RUNTIME=OCL",
      "-DDNNL_GPU_VENDOR=INTEL",
      "-DDNNL_LIBRARY_TYPE=SHARED",
      "-DDNNL_ENABLE_WORKLOAD=INFERENCE",
      "-DDNNL_ENABLE_PRIMITIVE=GATED_MLP;MATMUL;REORDER;ELTWISE",
      "-DDNNL_ENABLE_PRIMITIVE_GPU_ISA=XE2",
  ]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      STATUS, ROUTES, REJECTED, SEQ2202, SEQ2221A, ONEDNN_REPO,
      CMAKE, NINJA, CC, CXX, OPENCL_INCLUDE, OPENCL_LIBRARY, SYSTEMD_RUN)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing GMLP admission inputs: " + ", ".join(missing))

  repo = repository_state(output)
  seq2202 = load_json(SEQ2202)
  seq2221a = load_json(SEQ2221A)
  test_source = object_text(TEST_SOURCE)
  test_cmake = object_text(TEST_CMAKE)
  impl_list = object_text(IMPL_LIST)
  micro_hpp = object_text(MICRO_HPP)
  micro_cpp = object_text(MICRO_CPP)
  source_commit = git(ONEDNN_REPO, "rev-parse", ONEDNN_HEAD)
  upstream_head = remote_pr_head()
  kill = qk_kill_number(seq2202)
  sample_memory("after-source-and-kill-number", stop_bytes, memory)

  parser_tokens = (
      'maybe_test = ::getenv("GMLP_TEST")',
      "p.mb = std::stoi(tmp);",
      "p.ic = std::stoi(tmp);",
      "p.oc = std::stoi(tmp);",
      "p.src_dt = p.dst_dt = get_type(tmp);",
      "p.wgu_wt = p.wd_wt = get_type(tmp);",
      "? quantize_type::per_token_with_groups",
      "p.gateup_group_size = p.down_group_size = std::stoi(tmp);",
      "p.wgu_s_dt = p.wd_s_dt = get_type(tmp);",
      "p.wgu_zp_dt = p.wd_zp_dt = get_type(tmp);",
  )
  compare_tokens = (
      "bench_gated_mlp_primitives(",
      "bench_gated_mlp_internal(",
      "total mismatches: %d, allowed: %d",
      "avg time internal vs primitive: %f vs %f, w/speedup of %f",
      "ASSERT_LE(n_mismatches, threshold)",
  )
  exact_env_shapes = {
      "decode": {
          "GMLP_TEST": DECODE_ENV,
          "mb": 1,
          "ic": 2048,
          "oc": 512,
          "src_dt": "f16",
          "weights_dt": "u4",
          "quantized": True,
          "group_size": 64,
          "scale_dt": "f16",
          "zero_point_dt": "u8",
      },
      "prefill": {
          "GMLP_TEST": PREFILL_ENV,
          "mb": 2048,
          "ic": 2048,
          "oc": 512,
          "src_dt": "f16",
          "weights_dt": "u4",
          "quantized": True,
          "group_size": 64,
          "scale_dt": "f16",
          "zero_point_dt": "u8",
      },
  }
  plan = {
      "source": {
          "repo": str(ONEDNN_REPO),
          "commit": ONEDNN_HEAD,
          "worktree": str(SOURCE_WORKTREE),
          "create_command": [
              "git", "-C", str(ONEDNN_REPO), "worktree", "add",
              "--detach", str(SOURCE_WORKTREE), ONEDNN_HEAD],
      },
      "build": {
          "directory": str(BUILD_DIR),
          "configure_command": configure_command(),
          "build_command": [
              str(CMAKE), "--build", str(BUILD_DIR),
              "--target", "test_internals_gmlp", "-j", "1"],
          "maximum_parallel_jobs": 1,
          "transient_scope": True,
          "resource_limits_changed": False,
      },
      "component": {
          "binary": str(
              BUILD_DIR / "tests/gtests/internals/test_internals_gmlp"),
          "shapes": exact_env_shapes,
          "pairs_per_shape": PAIR_COUNT,
          "maximum_concurrent_workers": 1,
          "environment": {
              "ONEDNN_VERBOSE": "all",
              "DNNL_VERBOSE": "all",
          },
          "required_provider": "ocl:micro_horz:any",
          "correctness": {
              "require_test_returncode_zero": True,
              "require_reported_mismatches_at_or_below_allowed": True,
              "product_correctness_still_required_after_integration": True,
          },
          "inference": {
              "method": "paired_one_sided_percentile_bootstrap_median_delta",
              "confidence": 0.95,
              "bootstrap_resamples": 20000,
              "bootstrap_seed": BOOTSTRAP_SEED,
              "delta_definition": "internal_micro_horz_ms - primitive_ms",
              "prefill_delta_ucb_cap_ms": -kill[
                  "registered_saving_ms_per_layer"],
              "decode_delta_ucb_cap_ms": 0.0,
          },
      },
      "memory": {
          "preflight_bytes": PREFLIGHT_BYTES,
          "abort_below_bytes": ABORT_BYTES,
      },
      "forbidden": {
          "openvino_product_build": True,
          "model_load": True,
          "infer_request": True,
          "concurrent_gpu_worker": True,
          "system_or_driver_change": True,
      },
  }

  hashes_exact = all(
      sha256(path) == expected
      for path, expected in EXPECTED_SHA256.items())
  source_contract = {
      "source_commit": source_commit,
      "remote_pr_head": upstream_head,
      "gmlp_test_override_parser_exact": all(
          token in test_source for token in parser_tokens),
      "primitive_and_internal_comparison_exact": all(
          token in test_source for token in compare_tokens),
      "fixed_five_timed_runs_per_side": (
          test_source.count("runs = 5;") == 2),
      "separate_gmlp_target_present": (
          "register_exe(${TEST_EXE}_gmlp" in test_cmake
          and "test_gated_mlp.cpp" in test_cmake),
      "micro_horz_precedes_ref": (
          0 <= impl_list.find("gated_mlp::micro_horz_t")
          < impl_list.find("gated_mlp::ref_t")),
      "required_provider_name_present": (
          'DECLARE_COMMON_PD_T("ocl:micro_horz:any"' in micro_hpp),
      "ptl_xe2_config_present": (
          "dev_info.gpu_arch() > compute::gpu_arch_t::xe_hpg" in micro_cpp
          and "return {32, 16, 1, 1};" in micro_cpp),
      "three_gemm_schedule_retained": (
          "gemm_down_->execute(down_ctx)" in micro_cpp),
      "exact_env_shapes": exact_env_shapes,
  }
  toolchain = {
      str(path): {
          "exists": path.exists(),
          "executable": os.access(path, os.X_OK),
      } for path in (CMAKE, NINJA, CC, CXX, SYSTEMD_RUN)}
  toolchain[str(OPENCL_INCLUDE)] = {
      "exists": OPENCL_INCLUDE.is_dir()}
  toolchain[str(OPENCL_LIBRARY)] = {
      "exists": OPENCL_LIBRARY.is_file()}

  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check("frozen_evidence_hashes_exact", hashes_exact),
      check(
          "pr5059_source_identity_exact",
          source_commit == ONEDNN_HEAD and upstream_head == ONEDNN_HEAD),
      check(
          "exact_shape_override_and_compare_contract_present",
          source_contract["gmlp_test_override_parser_exact"]
          and source_contract["primitive_and_internal_comparison_exact"]
          and source_contract["fixed_five_timed_runs_per_side"]
          and source_contract["separate_gmlp_target_present"]),
      check(
          "exact_locked_shape_rows_expressible_without_source_edit",
          exact_env_shapes["decode"]["GMLP_TEST"] == DECODE_ENV
          and exact_env_shapes["prefill"]["GMLP_TEST"] == PREFILL_ENV),
      check(
          "micro_horz_provider_and_ptl_config_exact",
          source_contract["micro_horz_precedes_ref"]
          and source_contract["required_provider_name_present"]
          and source_contract["ptl_xe2_config_present"]
          and source_contract["three_gemm_schedule_retained"]),
      check(
          "qk_funding_kill_number_positive_and_registered",
          kill["ratio_gap"] > 0
          and kill["registered_saving_us_per_layer"]
          >= kill["exact_required_saving_ms_per_layer"] * 1000.0
          and kill["registered_saving_us_per_layer"] < 2.0),
      check(
          "isolated_paths_are_fresh",
          not SOURCE_WORKTREE.exists() and not BUILD_DIR.exists()),
      check(
          "toolchain_and_opencl_inputs_present",
          all(row["exists"] for row in toolchain.values())
          and all(
              row.get("executable", True) for row in toolchain.values()),
          toolchain=toolchain),
      check(
          "memory_preflight_clears_8gib",
          available_memory_bytes() >= PREFLIGHT_BYTES,
          available_bytes=available_memory_bytes(),
          required_bytes=PREFLIGHT_BYTES),
      check(
          "serial_scoped_build_and_component_plan_exact",
          plan["build"]["maximum_parallel_jobs"] == 1
          and plan["build"]["build_command"][-2:] == ["-j", "1"]
          and plan["component"]["maximum_concurrent_workers"] == 1
          and plan["component"]["pairs_per_shape"] == 8
          and plan["memory"]["preflight_bytes"] == PREFLIGHT_BYTES
          and plan["memory"]["abort_below_bytes"] == ABORT_BYTES),
      check(
          "product_integration_remains_closed",
          seq2221a["verdict"]["product_build_admitted"] is False
          and plan["forbidden"]["openvino_product_build"]
          and plan["forbidden"]["model_load"]
          and plan["forbidden"]["infer_request"]),
      check(
          "admission_gate_starts_no_build_or_gpu_work",
          True,
          worktrees_created=0,
          configure_invocations=0,
          compiler_invocations=0,
          gpu_contexts_created=0,
          model_workers_started=0,
          infer_requests_created=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": required_checks_passed,
      "component_build_admitted": required_checks_passed,
      "product_build_admitted": False,
      "verdict": (
          "admit_one_isolated_j1_exact_shape_component_build"
          if required_checks_passed else
          "reject_component_build_admission_contract_mismatch"),
      "reason": (
          "GMLP_TEST can express both locked U4 group64 shapes without a "
          "source edit; the exact PR head is locally available, the isolated "
          "toolchain and paths are fresh, and the frozen Q/K miss requires "
          f"only {kill['registered_saving_us_per_layer']:.3f} us/layer. "
          "Admission covers only one serial standalone build and component "
          "measurement, not OpenVINO integration."),
      "next_if_pass": (
          "create the exact detached worktree, configure once, and build only "
          "test_internals_gmlp at -j1 under a fresh monitored transient scope"),
      "next_if_build_pass": (
          "run eight serial paired component processes per exact shape; "
          "require micro_horz provider, component correctness, prefill delta "
          "UCB below the registered cap, and decode non-regression"),
  }

  sample_memory("complete", stop_bytes, memory)
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": {
          relative(path): {
              "sha256": sha256(path),
              "bytes": path.stat().st_size,
          } for path in (STATUS, ROUTES, REJECTED, SEQ2202, SEQ2221A)},
      "source_contract": source_contract,
      "qk_funding_kill_number": kill,
      "plan": plan,
      "checks": checks,
      "memory": {
          "stop_bytes": stop_bytes,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory),
          "samples": memory,
      },
      "workers": {
          "maximum_concurrent_workers": 0,
          "worktrees_created": 0,
          "configure_invocations": 0,
          "compiler_invocations": 0,
          "gpu_contexts_created": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
      },
      "verdict": verdict,
  }
  write_json(output / "plan.json", metrics)
  (output / "report.md").write_text(
      "# oneDNN PR5059 exact-shape component admission\n\n"
      f"- Required checks: `{required_checks_passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      "- Exact rows: `MB=1/2048, IC=2048, OC=512`, F16 source, U4 "
      "weights, group64, F16 scales, U8 zero points.\n"
      f"- Registered prefill saving: "
      f"`{kill['registered_saving_us_per_layer']:.3f} us/layer`.\n"
      "- Build/run scope: standalone, `-j1`, strictly serial, 8-GiB "
      "preflight / 4-GiB abort; product integration remains closed.\n"
      "- Compiler/GPU/model/InferRequest count: `0/0/0/0`.\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": required_checks_passed,
      "verdict": verdict["verdict"],
      "component_build_admitted": verdict["component_build_admitted"],
      "registered_saving_us_per_layer": kill[
          "registered_saving_us_per_layer"],
      "minimum_available_bytes": metrics["memory"][
          "minimum_available_bytes"],
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
