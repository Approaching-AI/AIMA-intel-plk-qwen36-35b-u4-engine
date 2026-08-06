#!/usr/bin/env python3
"""Audit the sole triple-cohort compile without compiling it again.

Seq2145 produced a valid program and resource record but its gate counted KQ
and VS markers across inactive preprocessor branches.  This audit scopes the
same fused source to the triple-cohort branch, verifies that the current host
source is byte-identical to the compiled input, and reuses the one existing
program.  It launches no compiler, OpenCL kernel, plugin, or model worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "triple-cohort-codegen-audit-v1")
HOST_SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
CODEGEN_GATE = ROOT / (
    "tools/intel-qwen36-openvino-exact-attention-"
    "triple-cohort-codegen-gate.py")
COMPONENT_BOUND = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/result.json")
ORIGINAL_DIR = ROOT / (
    "output/openvino-exact-attention-triple-cohort-codegen-"
    "20260724Tseq2145-clean")
ORIGINAL_RESULT = ORIGINAL_DIR / "result.json"
ORIGINAL_RUN = ORIGINAL_DIR / "raw/triple-cohort-codegen.json"
FUSED_SOURCE = ORIGINAL_DIR / (
    "raw/triple-cohort/existing_shim.fused.cl")
PROGRAM = ORIGINAL_DIR / (
    "raw/triple-cohort/existing_shim.program.bin")

EXPECTED_COMPILE_COMMIT = "1d5cc2a8c17523fbc04d129458aac17621dc2974"
EXPECTED_PROVIDER_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
EXPECTED_KERNEL = "iq36_exact_score_triple_cohort"
EXPECTED_DEFINE = "IQ36_COMPONENT_PROGRAM=8"
EXPECTED_REGISTER_COUNT = 96
EXPECTED_SPILL_BYTES = 0
EXPECTED_LOCAL_MEMORY_BYTES = 61_472
EXPECTED_MAX_WORKGROUP_ITEMS = 1024


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib < 4.0:
    parser.error("--memory-stop-gib must be at least 4")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


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


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  dirty = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [row for row in dirty if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def triple_kernel_source(source: str) -> str:
  start = source.find(
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_TRIPLE_COHORT")
  if start < 0:
    return ""
  end = source.find(
      "#if IQ36_COMPONENT_PROGRAM == IQ36_COMPONENT_DUAL_COHORT", start)
  return source[start:end] if end >= 0 else ""


def source_sha_from_original(
    original: dict[str, Any], path: str,
) -> str:
  for row in original.get("sources", []):
    if isinstance(row, dict) and row.get("path") == path:
      return str(row.get("sha256", ""))
  return ""


def summary(payload: dict[str, Any]) -> str:
  result = payload["candidate_result"]
  return "\n".join([
      "# Exact-attention triple-cohort codegen audit",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- register count / spill: "
      f"`{result.get('kernel_register_count')} / "
      f"{result.get('kernel_spill_memory_bytes')} B`",
      f"- actual SLM / two-WG use / device budget: "
      f"`{result.get('kernel_local_memory_bytes')} / "
      f"{2 * int(result.get('kernel_local_memory_bytes', 0))} / "
      "`131072 B`",
      f"- maximum / required workgroup items: "
      f"`{result.get('kernel_maximum_workgroup_size')} / 768`",
      f"- reused program bytes / SHA256: "
      f"`{payload['program']['bytes']} / "
      f"{payload['program']['sha256']}`",
      "",
      "This audit launches no compiler or kernel.  It only repairs the",
      "inactive-branch marker census on the sole seq2145 compile.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_start = available_memory_bytes()
  if memory_start < stop_bytes:
    raise RuntimeError(
        f"memory stop at start: {memory_start} < {stop_bytes}")

  required_paths = (
      HOST_SOURCE, CODEGEN_GATE, COMPONENT_BOUND, ORIGINAL_RESULT,
      ORIGINAL_RUN, FUSED_SOURCE, PROGRAM)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing triple-cohort audit inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  bound = load_json(COMPONENT_BOUND)
  original = load_json(ORIGINAL_RESULT)
  original_run = load_json(ORIGINAL_RUN)
  host_source = HOST_SOURCE.read_text(encoding="utf-8")
  fused_source = FUSED_SOURCE.read_text(encoding="utf-8")
  current_kernel = triple_kernel_source(host_source)
  fused_kernel = triple_kernel_source(fused_source)
  gate_source = CODEGEN_GATE.read_text(encoding="utf-8")
  candidate = original.get("candidate_result", {})
  original_failed = [
      row.get("name") for row in original.get("checks", [])
      if isinstance(row, dict) and row.get("pass") is False]
  original_other_checks_pass = all(
      row.get("pass") is True
      for row in original.get("checks", [])
      if isinstance(row, dict)
      and row.get("name") != "fused_source_contains_one_exact_candidate")

  compiled_host_sha = source_sha_from_original(
      original, "engine/gpu/opencl/exact_score_staging_component.cl")
  current_host_sha = sha256(HOST_SOURCE)
  program = {
      "path": display(PROGRAM),
      "bytes": PROGRAM.stat().st_size,
      "sha256": sha256(PROGRAM),
  }
  fixed_scoped_counts = {
      "kernel":
          fused_kernel.count(
              "__kernel void iq36_exact_score_triple_cohort("),
      "first_kq":
          fused_kernel.count(
              "iq36_component_score_tile first_score = ugemm_kq("),
      "next_kq":
          fused_kernel.count(
              "iq36_component_score_tile next_score = ugemm_kq("),
      "vs":
          fused_kernel.count(
              "iq36_component_accumulator_tile chunk_accumulator = "
              "ugemm_vs("),
  }
  whole_file_counts = {
      "kernel":
          fused_source.count(
              "__kernel void iq36_exact_score_triple_cohort("),
      "first_kq":
          fused_source.count(
              "iq36_component_score_tile first_score = ugemm_kq("),
      "next_kq":
          fused_source.count(
              "iq36_component_score_tile next_score = ugemm_kq("),
      "vs":
          fused_source.count(
              "iq36_component_accumulator_tile chunk_accumulator = "
              "ugemm_vs("),
  }
  raw_stdout = str(original_run.get("stdout", ""))
  raw_candidate = {}
  for line in reversed(raw_stdout.splitlines()):
    try:
      parsed = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(parsed, dict):
      raw_candidate = parsed
      break

  resource_pass = bool(
      candidate.get("kernel_register_count") ==
          EXPECTED_REGISTER_COUNT
      and candidate.get("kernel_spill_memory_bytes") ==
          EXPECTED_SPILL_BYTES
      and candidate.get("kernel_local_memory_bytes") ==
          EXPECTED_LOCAL_MEMORY_BYTES
      and candidate.get("kernel_maximum_workgroup_size") ==
          EXPECTED_MAX_WORKGROUP_ITEMS
      and candidate.get("kernel_preferred_workgroup_multiple") == 16
      and 2 * EXPECTED_LOCAL_MEMORY_BYTES <= 131_072)
  checks = [
      check("repository_clean_at_audit", not git["dirty"], git=git),
      check("seq2144_admitted_exactly_one_compile",
            bound.get("required_checks_passed") is True
            and bound.get("compiler_resource_probe_admitted") is True
            and bound.get("component_implementation_admitted") is False),
      check("seq2145_is_the_clean_sole_compile",
            original.get("schema_version") ==
                "intel-qwen36-openvino-exact-attention-"
                "triple-cohort-codegen-gate-v1"
            and original.get("git", {}).get("commit") ==
                EXPECTED_COMPILE_COMMIT
            and original.get("git", {}).get("dirty") is False
            and original.get("compiler_workers_launched") is True
            and original.get("kernel_worker_launched") is False
            and original.get("model_worker_launched") is False),
      check("only_failure_was_unscoped_inactive_branch_census",
            original_failed ==
                ["fused_source_contains_one_exact_candidate"]
            and original_other_checks_pass,
            original_failed_checks=original_failed,
            whole_file_counts=whole_file_counts),
      check("compiled_host_source_is_still_byte_identical",
            bool(compiled_host_sha)
            and compiled_host_sha == current_host_sha
            and current_kernel == fused_kernel,
            compiled_host_sha256=compiled_host_sha,
            current_host_sha256=current_host_sha),
      check("gate_now_scopes_the_fused_candidate_branch",
            "fused_kernel_source = triple_kernel_source(fused_source)"
                in gate_source
            and gate_source.count(
                "fused_kernel_source.count(") == 4),
      check("scoped_generated_carrier_census_is_exact",
            fixed_scoped_counts == {
                "kernel": 1, "first_kq": 1, "next_kq": 1, "vs": 1},
            scoped_counts=fixed_scoped_counts),
      check("raw_compile_record_and_result_are_identical",
            original_run.get("returncode") == 0
            and raw_candidate == candidate
            and candidate.get("kernel_name") == EXPECTED_KERNEL
            and candidate.get("host_define") == EXPECTED_DEFINE
            and candidate.get("openvino_onednn_commit") ==
                EXPECTED_PROVIDER_COMMIT
            and candidate.get("register_file_size") == 128
            and candidate.get("program_bytes") == program["bytes"]),
      check("actual_resources_pass_the_registered_contract",
            resource_pass, candidate_result=candidate),
      check("reused_program_is_nonempty",
            program["bytes"] > 0, program=program),
      check("no_compiler_kernel_plugin_or_model_worker_launched", True),
  ]
  memory_end = available_memory_bytes()
  checks.append(check(
      "memory_guard_never_tripped",
      min(memory_start, memory_end) >= stop_bytes,
      minimum_available_bytes=min(memory_start, memory_end),
      memory_stop_bytes=stop_bytes))
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_attention_triple_cohort_component"
      if required else
      "close_exact_attention_triple_cohort_after_audit")
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "original_artifact": display(ORIGINAL_DIR),
      "original_verdict": original.get("verdict"),
      "audit_scope":
          "repair inactive preprocessor branch census only",
      "component_admitted": required,
      "kernel_enqueue_admitted": required,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "compiler_workers_launched": False,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
      "checks": checks,
      "candidate_result": candidate,
      "scoped_counts": fixed_scoped_counts,
      "whole_file_counts": whole_file_counts,
      "program": program,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": [
          {"label": "start", "available_bytes": memory_start},
          {"label": "end", "available_bytes": memory_end},
      ],
      "sources": [
          {"path": display(path), "sha256": sha256(path)}
          for path in required_paths
      ],
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "resources.json", {
      "candidate_result": candidate,
      "program": program,
      "compiler_reused": True,
      "compiler_workers_launched": False,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": payload["sources"],
      "files": ["result.json", "resources.json", "summary.md"],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "register_count": candidate.get("kernel_register_count"),
      "spill_memory_bytes": candidate.get("kernel_spill_memory_bytes"),
      "local_memory_bytes": candidate.get("kernel_local_memory_bytes"),
      "maximum_workgroup_items":
          candidate.get("kernel_maximum_workgroup_size"),
      "component_admitted": required,
      "compiler_workers_launched": False,
      "kernel_worker_launched": False,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
