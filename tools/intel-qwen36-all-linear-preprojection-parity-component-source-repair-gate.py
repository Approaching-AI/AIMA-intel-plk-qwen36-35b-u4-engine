#!/usr/bin/env python3
"""Source-gate the one authorized Q6 lane-accumulation repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-all-linear-preprojection-parity-component-source-repair-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_source_repair_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_all_linear_preprojection_parity_component_repair_target_compile_gate"
)
RAW_DELIMITER = "IQ36PREPROJ"


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _run(command: list[str], out_dir: Path, stem: str) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, check=False)
  stdout_path = out_dir / f"{stem}.stdout.txt"
  stderr_path = out_dir / f"{stem}.stderr.txt"
  stdout_path.write_text(completed.stdout, encoding="utf-8")
  stderr_path.write_text(completed.stderr, encoding="utf-8")
  return {
      "passed": completed.returncode == 0,
      "command": command,
      "returncode": completed.returncode,
      "stdout": _rel(stdout_path),
      "stderr": _rel(stderr_path),
  }


def _extract_embedded_opencl(generated_cpp: str) -> str:
  begin_marker = f'R"{RAW_DELIMITER}('
  end_marker = f'){RAW_DELIMITER}"'
  begin = generated_cpp.find(begin_marker)
  if begin < 0:
    raise ValueError("baseline generated C++ raw OpenCL start not found")
  begin += len(begin_marker)
  end = generated_cpp.find(end_marker, begin)
  if end < 0:
    raise ValueError("baseline generated C++ raw OpenCL end not found")
  return generated_cpp[begin:end]


def _kernel_parts(opencl: str) -> tuple[str, str, str]:
  begin = opencl.find("__kernel void q6k_linear_qkv_cpuorder_nofma(")
  end = opencl.find(
      "__kernel void linear_attn_conv_cpuorder_nofma_f32(", begin)
  if begin < 0 or end <= begin:
    raise ValueError("exact Q6 kernel boundaries not found")
  return opencl[:begin], opencl[begin:end], opencl[end:]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  source_baseline = _load(args.source_baseline)
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  current_opencl = args.opencl_source.read_text(encoding="utf-8")
  baseline_generated = args.baseline_generated_cpp.read_text(encoding="utf-8")
  baseline_opencl = _extract_embedded_opencl(baseline_generated)
  baseline_prefix, baseline_q6, baseline_suffix = _kernel_parts(baseline_opencl)
  current_prefix, current_q6, current_suffix = _kernel_parts(current_opencl)
  cpu = args.cpu_source.read_text(encoding="utf-8")

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("repairable_contract_violation") is True
      and predecessor.get("source_repair_allowed") is True
      and predecessor.get("target_compile_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 601, CURRENT_ROUTE)
      and _has_switch(
          routes, 601,
          "select_router_prompt_distribution_all_linear_preprojection_"
          "parity_component_source_repair_gate"))
  repair_is_only_opencl_q6_segment = (
      current_prefix == baseline_prefix
      and current_suffix == baseline_suffix
      and current_q6 != baseline_q6)
  old_contract = all(marker in baseline_q6 for marker in [
      "float sum = 0.0f;",
      "sum = sum + block_lane;",
  ]) and "float sums[8]" not in baseline_q6
  new_contract = all(marker in current_q6 for marker in [
      "float sums[8];",
      "sums[lane] = 0.0f;",
      "sums[lane] = sums[lane] + block_lane;",
      "float sum = 0.0f;",
      "sum = sum + sums[lane];",
  ])
  cpu_contract = all(marker in cpu for marker in [
      "float sums[8] = {};",
      "accumulate_q6_k_q8_k_block_direct(",
      "for (const float lane_sum : sums)",
      "sum += lane_sum;",
  ])
  exact_begin = current_opencl.rfind("#pragma OPENCL FP_CONTRACT OFF", 0,
                                     current_opencl.find(current_q6))
  exact_end = current_opencl.find(
      "#pragma OPENCL FP_CONTRACT ON", current_opencl.find(current_q6))
  scoped = (
      exact_begin >= 0 and exact_end > exact_begin
      and exact_begin < current_opencl.find(current_q6) < exact_end)
  non_opencl_hashes_unchanged = all([
      source_baseline.get("inputs", {}).get("header_source_sha256")
      == _sha256(args.header_source),
      source_baseline.get("inputs", {}).get("runner_source_sha256")
      == _sha256(args.runner_source),
      source_baseline.get("inputs", {}).get("decode_source_sha256")
      == _sha256(args.decode_source),
  ])
  opencl_compile = _run([
      args.clang, "-x", "cl", "-target", "spir64", "-cl-std=CL1.2",
      "-fsyntax-only", _rel(args.opencl_source),
  ], compile_dir, "opencl-syntax")
  runner_compile = _run([
      args.cxx, "-std=c++20", "-Iengine/include", "-O0", "-c",
      _rel(args.runner_source), "-o", _rel(compile_dir / "gpu_q4x8_matvec.o"),
  ], compile_dir, "runner-compile")
  code_volume = _run([
      "python3", "tools/intel-qwen36-code-volume-check.py",
  ], compile_dir, "code-volume")
  no_runtime_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("probe.json", "run.json", "tokens.jsonl"))
  checks = [
      {"name": "seq601_authorized_one_source_repair_only",
       "pass": predecessor_selects},
      {"name": "repair_changes_only_the_exact_q6_opencl_segment",
       "pass": repair_is_only_opencl_q6_segment},
      {"name": "scalar_block_lane_contract_becomes_cpu_lane_reduction",
       "pass": old_contract and new_contract and cpu_contract,
       "detail": {
           "baseline_scalar_contract": old_contract,
           "current_lane_contract": new_contract,
           "cpu_lane_contract": cpu_contract,
       }},
      {"name": "exact_q6_remains_inside_scoped_fp_contract_off",
       "pass": scoped},
      {"name": "header_runner_and_decode_are_unchanged_from_seq598",
       "pass": non_opencl_hashes_unchanged},
      {"name": "complete_opencl_program_passes_spir12_syntax",
       "pass": opencl_compile["passed"], "detail": opencl_compile},
      {"name": "runner_still_compiles_locally",
       "pass": runner_compile["passed"], "detail": runner_compile},
      {"name": "code_volume_ceiling_is_preserved",
       "pass": code_volume["passed"], "detail": code_volume},
      {"name": "source_repair_gate_created_no_runtime_evidence",
       "pass": no_runtime_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "source_baseline": _rel(args.source_baseline),
          "baseline_generated_cpp": _rel(args.baseline_generated_cpp),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "header_source_sha256": _sha256(args.header_source),
          "runner_source_sha256": _sha256(args.runner_source),
          "decode_source_sha256": _sha256(args.decode_source),
      },
      "checks": checks,
      "required_checks_passed": required,
      "source_repair_passed": required,
      "repair_target_compile_allowed": required,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_exact_q6_lane_accumulation_source_repair"
          if required else "repair_exact_q6_lane_accumulation_source"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The one authorized source repair changes only the exact Q6 kernel, "
          "matches CPU lane accumulation structurally, and passes complete "
          "local compilation plus 97/97. Target-compile a fresh component "
          "binary next without execution."
          if required else
          "Repair the exact Q6 lane order, source isolation, local compile, or "
          "flag ceiling before any target command."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "source_repair_passed": metrics["source_repair_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "component_probe_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Exact Q6 Source Repair",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is source evidence only. No target command or component ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=602)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq601-all-linear-preprojection-parity-component-route-"
          "close-gate-20260710Tseq601Z/metrics.json"))
  parser.add_argument(
      "--source-baseline", type=Path,
      default=ROOT / (
          "output/seq598-all-linear-preprojection-parity-component-source-"
          "gate-20260710Tseq598Z/metrics.json"))
  parser.add_argument(
      "--baseline-generated-cpp", type=Path,
      default=ROOT / (
          "output/seq599-all-linear-preprojection-parity-component-target-"
          "compile-gate-20260710Tseq599Z/exact_preprojection_component_probe.cpp"))
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument("--cpu-source", type=Path,
                      default=ROOT / "engine/src/gguf_loader.cpp")
  parser.add_argument("--header-source", type=Path,
                      default=ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp")
  parser.add_argument("--runner-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4x8_matvec.cpp")
  parser.add_argument("--decode-source", type=Path,
                      default=ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq602-all-linear-preprojection-parity-component-source-"
          "repair-gate-20260710Tseq602Z"))
  parser.add_argument("--clang", default="clang")
  parser.add_argument("--cxx", default="clang++")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "repair_target_compile_allowed": metrics[
          "repair_target_compile_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
