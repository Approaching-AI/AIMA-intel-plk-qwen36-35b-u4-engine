#!/usr/bin/env python3
"""Gate the sole exact-attention dual-cohort standalone GPU component."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import shlex
import statistics
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
BASE_GATE = ROOT / (
    "tools/intel-qwen36-openvino-exact-score-staging-component-gate.py")
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
RUNNER = ROOT / "engine/tools/exact_attention_vrt160_component.cpp"
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
SHIMS = ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl"
BOUNDARIES = ROOT / "engine/boundaries.json"
CODEGEN_GATE = ROOT / (
    "output/openvino-exact-attention-dual-cohort-codegen-"
    "20260723Tseq2134-clean/result.json")
PINNED_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu")
PINNED_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-20db-micro-static")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-exact-attention-vrt160-component"
PINNED_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
CONTROL_GRFS = 128
MAX_CANDIDATE_GRFS = 128
CONTROL_LOCAL_ITEMS = 256
CANDIDATE_LOCAL_ITEMS = 512
MIN_CANDIDATE_LOCAL_MEMORY = 59392
MAX_CANDIDATE_LOCAL_MEMORY = 64 * 1024
DELTA_CAP_MS = -0.1175998
MIN_SAMPLES = 20


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_preflight_gib < 8.0:
    parser.error("--memory-preflight-gib must be at least 8")
  if args.memory_stop_gib < 4.0:
    parser.error("--memory-stop-gib must be at least 4")
  if args.memory_preflight_gib <= args.memory_stop_gib:
    parser.error("--memory-preflight-gib must exceed --memory-stop-gib")
  return args


def load_base() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_staging_gate", BASE_GATE)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def fixed_result(
    result: dict[str, Any], expected_candidate_grfs: int,
) -> bool:
  return bool(
      result.get("schema_version") ==
          "intel-qwen36-exact-attention-dual-cohort-component-v1"
      and result.get("algorithm") ==
          "generated_m256_n16_dual_cohort_pipeline"
      and result.get("context_tokens") == 131072
      and result.get("head_dim") == 256
      and result.get("query_heads") == 16
      and result.get("kv_heads") == 2
      and result.get("gqa_group") == 8
      and result.get("useful_workgroups") == 2
      and result.get("output_compared_values") == 4096
      and result.get("control_register_count") == CONTROL_GRFS
      and result.get("candidate_register_count") == expected_candidate_grfs
      and 0 < expected_candidate_grfs <= MAX_CANDIDATE_GRFS
      and result.get("control_spill_memory_bytes") == 0
      and result.get("candidate_spill_memory_bytes") == 0
      and result.get("control_local_workgroup_items") ==
          CONTROL_LOCAL_ITEMS
      and result.get("candidate_local_workgroup_items") ==
          CANDIDATE_LOCAL_ITEMS
      and MIN_CANDIDATE_LOCAL_MEMORY
          <= int(result.get("candidate_local_memory_bytes", -1))
          <= MAX_CANDIDATE_LOCAL_MEMORY
      and int(result.get("candidate_maximum_workgroup_items", -1))
          >= CANDIDATE_LOCAL_ITEMS
      and result.get("candidate_preferred_workgroup_multiple") == 16
      and result.get("sample_count") == MIN_SAMPLES
      and result.get("schedule") ==
          "interleaved_control_candidate_candidate_control")


def distribution_pass(result: dict[str, Any]) -> bool:
  rows = result.get("paired_samples", [])
  if not isinstance(rows, list) or len(rows) != MIN_SAMPLES:
    return False
  for index, row in enumerate(rows):
    if not isinstance(row, dict) or row.get("sample") != index:
      return False
    if row.get("order") != (
        "control_candidate" if index % 2 == 0 else "candidate_control"):
      return False
    try:
      control = float(row["control_ms"])
      candidate = float(row["candidate_ms"])
      delta = float(row["differential_ms"])
    except (KeyError, TypeError, ValueError):
      return False
    if not all(math.isfinite(value) and value > 0.0
               for value in (control, candidate)):
      return False
    if not math.isfinite(delta):
      return False
    if not math.isclose(
        delta, candidate - control, rel_tol=0.0, abs_tol=2.0e-6):
      return False
  return True


def main() -> int:
  args = parse_args()
  base = load_base()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  base.sample_memory("start-preflight", preflight_bytes, memory)

  required_paths = [
      SOURCE, RUNNER, CODEGEN, SHIMS, BOUNDARIES, CODEGEN_GATE,
      PINNED_SOURCE, PINNED_BUILD / "src/libdnnl.a",
      base.CXX, CMAKE, ENV_SCRIPT]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit(
        "missing dual-cohort component inputs: " + ", ".join(missing))

  git = base.git_state(out_dir)
  provider_head = base.run(
      ["git", "-C", str(PINNED_SOURCE), "rev-parse", "HEAD"],
      30).stdout.strip()
  provider_status = base.run([
      "git", "-C", str(PINNED_SOURCE), "status", "--short", "--",
      "src/gpu/intel/gemm/jit", "src/gpu/intel/jit/config",
      "third_party/ngen",
  ], 30).stdout.strip()
  codegen_gate = load_json(CODEGEN_GATE)
  codegen_candidate = codegen_gate.get("candidate", {}).get("result", {})
  source_text = SOURCE.read_text(encoding="utf-8")
  dual_begin = source_text.index(
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT")
  dual_end = source_text.index("\n#endif", dual_begin)
  dual_source = source_text[dual_begin:dual_end]
  runner_text = RUNNER.read_text(encoding="utf-8")
  codegen_text = CODEGEN.read_text(encoding="utf-8")
  boundaries = load_json(BOUNDARIES)
  registered = [
      row for row in boundaries.get("infra_targets", [])
      if row.get("target") == TARGET
      and row.get("source") ==
          "tools/exact_attention_vrt160_component.cpp"]
  codegen_bound = bool(
      codegen_gate.get("schema_version") ==
          "intel-qwen36-openvino-exact-attention-dual-cohort-"
          "codegen-gate-v1"
      and codegen_gate.get("verdict") ==
          "admit_one_exact_attention_dual_cohort_component"
      and codegen_gate.get("required_checks_passed") is True
      and codegen_gate.get("component_admitted") is True
      and codegen_gate.get("kernel_worker_launched") is False
      and codegen_gate.get("model_worker_admitted") is False
      and codegen_gate.get("git", {}).get("dirty") is False
      and codegen_gate.get("inputs", {}).get(display(SOURCE))
          == sha256(SOURCE)
      and 0 < int(codegen_candidate.get("kernel_register_count", -1))
          <= MAX_CANDIDATE_GRFS
      and codegen_candidate.get("kernel_spill_memory_bytes") == 0
      and MIN_CANDIDATE_LOCAL_MEMORY
          <= int(codegen_candidate.get(
              "kernel_local_memory_bytes", -1))
          <= MAX_CANDIDATE_LOCAL_MEMORY
      and int(codegen_candidate.get(
          "kernel_maximum_workgroup_size", -1))
          >= CANDIDATE_LOCAL_ITEMS)
  source_checks = {
      "exact_dual_cohort_source_is_fixed":
          "__kernel void iq36_exact_score_dual_cohort(" in source_text
          and "__attribute__((reqd_work_group_size(16, 32, 1)))"
              in source_text
          and "#define IQ36_DUAL_PRODUCER_SUBGROUPS 16" in source_text
          and "#define IQ36_DUAL_CONSUMER_SUBGROUPS 16" in source_text,
      "deterministic_max_named_pipeline_and_no_v_prefetch":
          "float reduced_running_max = -INFINITY;" in dual_source
          and "subgroup_row < IQ36_DUAL_CONSUMER_SUBGROUPS"
              in dual_source
          and "tile_atomic_max_full(" not in dual_source
          and "__local NamedBarrier_t* pipeline_barrier" in dual_source
          and "named_barrier_init(IQ36_DUAL_TOTAL_SUBGROUPS)"
              in dual_source
          and "cooperative_prefetch_2d_rem(\n"
              "          value_base + (ulong)key_begin * IQ36_D"
              not in dual_source,
      "runner_is_fixed_twenty_pair_bitwise_gate":
          "constexpr int kSamples = 20;" in runner_text
          and "output_mismatch_count" in runner_text
          and "interleaved_control_candidate_candidate_control"
              in runner_text
          and "iq36_exact_score_dual_cohort" in runner_text
          and "kDualCohortLocalY = 32" in runner_text
          and "--dual-cohort" in runner_text,
      "codegen_locks_single_dual_candidate":
          "const bool fixed_dual_cohort" in codegen_text
          and "--exact-attention-dual-cohort" in codegen_text
          and 'g_existing_kernel_name == "iq36_exact_score_dual_cohort"'
              in codegen_text
          and 'g_host_define == "IQ36_COMPONENT_PROGRAM=4"'
              in codegen_text,
      "shared_boundary_target_registered_once": len(registered) == 1,
  }
  base.sample_memory("after-source-audit", stop_bytes, memory)

  codegen_binary = raw_dir / "openvino-micro-codegen"
  build_command = base.codegen_build_command(codegen_binary)
  codegen_build = base.run(build_command, args.timeout_s)
  write_json(raw_dir / "codegen-build.json", {
      "command": build_command,
      "returncode": codegen_build.returncode,
      "stdout": codegen_build.stdout,
      "stderr": codegen_build.stderr,
  })
  base.sample_memory("after-codegen-build", stop_bytes, memory)

  program_specs = {
      "capture": {
          "define": "IQ36_COMPONENT_PROGRAM=1",
          "kernel": "iq36_exact_score_serial_capture",
          "dual": False,
      },
      "control128": {
          "define": "IQ36_COMPONENT_PROGRAM=2",
          "kernel": "iq36_exact_score_fused",
          "dual": False,
      },
      "candidate_dual": {
          "define": "IQ36_COMPONENT_PROGRAM=4",
          "kernel": "iq36_exact_score_dual_cohort",
          "dual": True,
      },
  }
  programs: dict[str, Path] = {}
  codegen_runs: dict[str, dict[str, Any]] = {}
  codegen_results: dict[str, dict[str, Any]] = {}
  for label, spec in program_specs.items():
    program_dir = raw_dir / "programs" / label
    program_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(codegen_binary),
        "--fuse-existing-shim", str(SHIMS),
        "--host-source", str(SOURCE),
        "--kernel-name", str(spec["kernel"]),
        "--host-define", str(spec["define"]),
        "--register-file-size", str(CONTROL_GRFS),
        "--provider-commit", PINNED_COMMIT,
        "--dump-dir", str(program_dir),
    ]
    if spec["dual"]:
      command.insert(1, "--exact-attention-dual-cohort")
    shell = (
        f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && " +
        " ".join(shlex.quote(part) for part in command))
    completed = (
        base.run(["bash", "-lc", shell], args.timeout_s)
        if codegen_build.returncode == 0 else
        subprocess.CompletedProcess(command, 1, "", "codegen build failed"))
    codegen_runs[label] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    codegen_results[label] = base.parse_last_json(completed.stdout)
    programs[label] = program_dir / "existing_shim.program.bin"
    base.sample_memory(f"after-{label}-codegen", stop_bytes, memory)
  write_json(raw_dir / "program-codegen.json", codegen_runs)

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"]
  configure = base.run(configure_command, 300)
  target_build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TARGET, "-j1"]
  target_build = base.run(target_build_command, args.timeout_s)
  write_json(raw_dir / "component-build.json", {
      "configure": {
          "command": configure_command,
          "returncode": configure.returncode,
          "stdout": configure.stdout,
          "stderr": configure.stderr,
      },
      "build": {
          "command": target_build_command,
          "returncode": target_build.returncode,
          "stdout": target_build.stdout,
          "stderr": target_build.stderr,
      },
  })
  base.sample_memory("after-component-build", stop_bytes, memory)
  executable = BUILD_DIR / TARGET
  target_ok = bool(
      configure.returncode == 0 and target_build.returncode == 0
      and executable.is_file())

  capture_result = codegen_results["capture"]
  control_result = codegen_results["control128"]
  candidate_result = codegen_results["candidate_dual"]
  expected_candidate_grfs = int(
      candidate_result.get("kernel_register_count", -1))
  capture_ok = bool(
      codegen_runs["capture"]["returncode"] == 0
      and programs["capture"].is_file()
      and capture_result.get("register_file_size") == CONTROL_GRFS
      and capture_result.get("exact_attention_dual_cohort") is False)
  control_ok = bool(
      codegen_runs["control128"]["returncode"] == 0
      and programs["control128"].is_file()
      and control_result.get("register_file_size") == CONTROL_GRFS
      and control_result.get("kernel_register_count") == CONTROL_GRFS
      and control_result.get("kernel_spill_memory_bytes") == 0
      and control_result.get("exact_attention_dual_cohort") is False)
  candidate_ok = bool(
      codegen_runs["candidate_dual"]["returncode"] == 0
      and programs["candidate_dual"].is_file()
      and candidate_result.get("register_file_size") == CONTROL_GRFS
      and 0 < expected_candidate_grfs <= MAX_CANDIDATE_GRFS
      and candidate_result.get("kernel_spill_memory_bytes") == 0
      and MIN_CANDIDATE_LOCAL_MEMORY
          <= int(candidate_result.get("kernel_local_memory_bytes", -1))
          <= MAX_CANDIDATE_LOCAL_MEMORY
      and int(candidate_result.get(
          "kernel_maximum_workgroup_size", -1))
          >= CANDIDATE_LOCAL_ITEMS
      and candidate_result.get("exact_attention_dual_cohort") is True
      and candidate_result.get("exact_attention_vrt160") is False)
  codegen_ok = bool(
      codegen_build.returncode == 0
      and provider_head == PINNED_COMMIT
      and provider_status == ""
      and capture_ok and control_ok and candidate_ok)

  time_path = raw_dir / "component.time.txt"
  worker_command = [
      str(executable),
      str(programs["capture"]),
      str(programs["control128"]),
      str(programs["candidate_dual"]),
      "--dual-cohort",
  ]
  shell = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"/usr/bin/time -v -o {shlex.quote(str(time_path))} " +
      " ".join(shlex.quote(part) for part in worker_command))
  base.sample_memory("before-component-worker", preflight_bytes, memory)
  worker = (
      base.run(["bash", "-lc", shell], args.timeout_s)
      if target_ok and codegen_ok and codegen_bound
      and all(source_checks.values()) else
      subprocess.CompletedProcess(
          worker_command, 1, "", "pre-component gate failed"))
  base.sample_memory("after-component-worker", stop_bytes, memory)
  (raw_dir / "component.stdout").write_text(
      worker.stdout, encoding="utf-8")
  (raw_dir / "component.stderr").write_text(
      worker.stderr, encoding="utf-8")
  write_json(raw_dir / "worker-command.json", {
      "command": worker_command,
      "returncode": worker.returncode,
  })
  result = base.parse_last_json(worker.stdout)
  resources = base.parse_time(time_path)
  deltas = [
      float(row.get("differential_ms", math.nan))
      for row in result.get("paired_samples", [])
      if isinstance(row, dict)]
  inference = base.signed_delta_cap_inference(deltas, DELTA_CAP_MS)
  distribution_ok = distribution_pass(result)
  numeric_pass = bool(
      result.get("numeric_pass") is True
      and result.get("output_mismatch_count") == 0)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("clean_codegen_gate_admits_exactly_one_component", codegen_bound),
      check("fixed_source_contract", all(source_checks.values()),
            source_checks=source_checks),
      check("pinned_codegen_and_three_programs_pass", codegen_ok,
            provider_head=provider_head,
            provider_status=provider_status,
            program_results=codegen_results),
      check("component_build_is_serial_j1", target_ok,
            build_command=target_build_command),
      check("standalone_worker_executes", worker.returncode == 0,
            returncode=worker.returncode),
      check("fixed_128k_m256_n16_dual_cohort_shape_and_resources",
            fixed_result(result, expected_candidate_grfs)),
      check("twenty_pair_interleaved_distribution", distribution_ok),
      check("candidate_output_is_bitwise_control_exact", numeric_pass),
      check("one_sided_95pct_delta_ucb_clears_layer_cap",
            inference.get("rate_pass") is True,
            inference=inference),
      check("worker_rss_and_swap_are_bounded",
            int(resources.get("maximum_resident_kib", 1 << 62)) <
                4 * 1024 * 1024
            and int(resources.get("swaps", -1)) == 0,
            resources=resources),
      check("memory_guards_never_tripped",
            all(row["pass"] for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "promote_exact_attention_dual_cohort_component"
      if required else "reject_exact_attention_dual_cohort_component")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in (
          SOURCE, RUNNER, CODEGEN, SHIMS, BOUNDARIES, CODEGEN_GATE)]
  payload = {
      "schema_version":
          "intel-qwen36-openvino-exact-attention-dual-cohort-"
          "component-gate-v1",
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_promoted": required,
      "component_rejected": not required,
      "graph_integration_admitted": required,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_worker_admitted": False,
      "product_claim_allowed": False,
      "gpu_component_worker_launched": worker.returncode in {0, 2},
      "model_worker_launched": False,
      "checks": checks,
      "result": result,
      "performance_inference": inference,
      "delta_cap_ms_per_layer": DELTA_CAP_MS,
      "control_median_ms": (
          statistics.median(
              float(row["control_ms"])
              for row in result.get("paired_samples", []))
          if distribution_ok else None),
      "candidate_median_ms": (
          statistics.median(
              float(row["candidate_ms"])
              for row in result.get("paired_samples", []))
          if distribution_ok else None),
      "worker_resources": resources,
      "worker_command": worker_command,
      "codegen_results": codegen_results,
      "provider_head": provider_head,
      "provider_status": provider_status,
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "correctness.json", {
      "numeric_pass": numeric_pass,
      "output_compared_values": result.get("output_compared_values"),
      "output_mismatch_count": result.get("output_mismatch_count"),
  })
  write_json(out_dir / "performance.json", {
      "paired_samples": result.get("paired_samples", []),
      "inference": inference,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version":
          "intel-qwen36-exact-attention-dual-cohort-component-"
          "manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": [
          "result.json", "correctness.json", "performance.json",
          "summary.md", "raw/codegen-build.json",
          "raw/program-codegen.json", "raw/component-build.json",
          "raw/component.stdout", "raw/component.stderr",
          "raw/component.time.txt", "raw/worker-command.json",
      ],
  })
  (out_dir / "summary.md").write_text(
      "\n".join([
          "# Exact-attention dual-cohort component gate",
          "",
          f"Verdict: **{verdict}**. Required checks: "
          f"`{str(required).lower()}`.",
          "",
          f"- output mismatches: "
          f"`{result.get('output_mismatch_count')}` / "
          f"`{result.get('output_compared_values')}`",
          f"- control/candidate median: "
          f"`{payload['control_median_ms']} / "
          f"{payload['candidate_median_ms']} ms/layer`",
          f"- candidate-minus-control median / one-sided 95% UCB / cap: "
          f"`{inference.get('point_estimate_ms')} / "
          f"{inference.get('upper_confidence_bound_ms')} / "
          f"{inference.get('cap_ms')} ms/layer`",
          f"- control/candidate actual GRFs: "
          f"`{result.get('control_register_count')} / "
          f"{result.get('candidate_register_count')}`",
          f"- candidate SLM / workgroup: "
          f"`{result.get('candidate_local_memory_bytes')} B / "
          f"{result.get('candidate_local_workgroup_items')} items`",
          f"- peak RSS / swaps: "
          f"`{resources.get('maximum_resident_kib')} KiB / "
          f"{resources.get('swaps')}`",
          "",
          "No graph, plugin, or model worker ran. Integration is admitted only",
          "when bitwise equality and the product-derived negative delta cap pass.",
          "",
      ]), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "numeric_pass": numeric_pass,
      "delta_median_ms": inference.get("point_estimate_ms"),
      "delta_ucb_ms": inference.get("upper_confidence_bound_ms"),
      "delta_cap_ms": DELTA_CAP_MS,
      "graph_integration_admitted": required,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
