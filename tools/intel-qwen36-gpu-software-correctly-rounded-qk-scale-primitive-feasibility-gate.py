#!/usr/bin/env python3
"""Prove one bounded software correctly-rounded FP32 reciprocal route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-gpu-software-correctly-rounded-qk-scale-feasibility-v0")
CURRENT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_feasibility_gate")
SELECTED_NEXT_ROUTE = (
    "gpu_software_correctly_rounded_qk_scale_primitive_design_gate")

VERIFIER_CPP = r'''#include <bit>
#include <cstdint>
#include <iostream>

std::uint32_t iq36_cr_recip_normal_f32_bits(std::uint32_t input_bits) {
  const std::uint32_t exponent = (input_bits >> 23U) & 0xffU;
  const std::uint32_t fraction = input_bits & 0x7fffffU;
  if (exponent == 0U || exponent == 0xffU || (input_bits >> 31U) != 0U) {
    return 0xffffffffU;
  }
  const std::uint64_t significand = 0x800000ULL | fraction;
  const int input_unbiased = static_cast<int>(exponent) - 127;
  std::uint64_t quotient;
  int output_unbiased;
  if (significand == 0x800000ULL) {
    quotient = 0x800000ULL;
    output_unbiased = -input_unbiased;
  } else {
    constexpr std::uint64_t numerator = 1ULL << 47U;
    quotient = numerator / significand;
    const std::uint64_t remainder = numerator % significand;
    const std::uint64_t twice_remainder = remainder << 1U;
    if (twice_remainder > significand ||
        (twice_remainder == significand && (quotient & 1ULL) != 0ULL)) {
      ++quotient;
    }
    output_unbiased = -input_unbiased - 1;
    if (quotient == 0x1000000ULL) {
      quotient >>= 1U;
      ++output_unbiased;
    }
  }
  const int output_exponent = output_unbiased + 127;
  if (output_exponent <= 0 || output_exponent >= 255) {
    return 0xfffffffeU;
  }
  return (static_cast<std::uint32_t>(output_exponent) << 23U) |
         (static_cast<std::uint32_t>(quotient) & 0x7fffffU);
}

int main() {
  std::uint64_t mismatches = 0;
  std::uint32_t first_fraction = 0;
  std::uint32_t first_expected = 0;
  std::uint32_t first_actual = 0;
  for (std::uint32_t fraction = 0; fraction <= 0x7fffffU; ++fraction) {
    const std::uint32_t input_bits = (127U << 23U) | fraction;
    volatile float denominator = std::bit_cast<float>(input_bits);
    volatile float numerator = 1.0f;
    const float expected = numerator / denominator;
    const std::uint32_t expected_bits = std::bit_cast<std::uint32_t>(expected);
    const std::uint32_t actual_bits = iq36_cr_recip_normal_f32_bits(input_bits);
    if (actual_bits != expected_bits) {
      if (mismatches == 0) {
        first_fraction = fraction;
        first_expected = expected_bits;
        first_actual = actual_bits;
      }
      ++mismatches;
    }
  }
  std::cout << "{\"mantissas_checked\":8388608,"
            << "\"mismatch_count\":" << mismatches << ","
            << "\"first_fraction\":" << first_fraction << ","
            << "\"first_expected_bits\":" << first_expected << ","
            << "\"first_actual_bits\":" << first_actual << "}\n";
  return mismatches == 0 ? 0 : 2;
}
'''


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


def _run(command: list[str]) -> dict[str, Any]:
  completed = subprocess.run(
      command, cwd=ROOT, text=True, capture_output=True, check=False)
  return {
      "command": command,
      "returncode": completed.returncode,
      "stdout": completed.stdout,
      "stderr": completed.stderr,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  component = _load(args.component)
  cpu_source = args.cpu_source.read_text(encoding="utf-8")
  module_source = args.module_source.read_text(encoding="utf-8")

  args.out_dir.mkdir(parents=True, exist_ok=True)
  compile_dir = args.out_dir / "compile"
  compile_dir.mkdir(parents=True, exist_ok=True)
  verifier_source = compile_dir / "cr_recip_exhaustive.cpp"
  verifier_binary = compile_dir / "cr_recip_exhaustive"
  verifier_source.write_text(VERIFIER_CPP, encoding="utf-8")
  compile_result = _run([
      "c++", "-std=c++20", "-O2", "-fno-fast-math", "-ffp-contract=off",
      str(verifier_source), "-o", str(verifier_binary),
  ])
  run_result = (
      _run([str(verifier_binary)])
      if compile_result["returncode"] == 0 else {
          "command": [str(verifier_binary)],
          "returncode": None,
          "stdout": "",
          "stderr": "compile failed",
      })
  try:
    verifier = json.loads(str(run_result.get("stdout", "")).strip())
  except json.JSONDecodeError:
    verifier = {}

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("level_zero_component_closed") is True
      and predecessor.get("primitive_feasibility_gate_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and _has_candidate(routes, 618, CURRENT_ROUTE)
      and _has_switch(
          routes, 618,
          "select_gpu_software_correctly_rounded_qk_scale_primitive_"
          "feasibility_gate"))
  rows = component.get("rows", [])
  rows = rows if isinstance(rows, list) else []
  component_localizes = (
      len(rows) == 2
      and all(row.get("budget_passed") is True for row in rows)
      and all(
          row.get("comparisons", {}).get(
              "v_conv_predelta_vs_cpu", {}).get("mismatch_count") == 0
          for row in rows)
      and all(
          row.get("comparisons", {}).get(
              "q_conv_predelta_vs_cpu", {}).get("mismatch_count") == 90
          and row.get("comparisons", {}).get(
              "k_conv_predelta_vs_cpu", {}).get("mismatch_count") == 336
          for row in rows))
  source_formulas_match = (
      "sum += static_cast<double>(value) * static_cast<double>(value);"
      in cpu_source
      and "1.0f / std::max(std::sqrt(static_cast<float>(sum)), norm_epsilon)"
      in cpu_source
      and "sum = sum + (double)head_value * (double)head_value;"
      in module_source
      and "1.0f / fmax(sqrt(sum_f32), norm_epsilon)" in module_source)
  exhaustive_passes = (
      compile_result.get("returncode") == 0
      and run_result.get("returncode") == 0
      and verifier.get("mantissas_checked") == 8388608
      and verifier.get("mismatch_count") == 0)

  headroom = predecessor.get("primitive_feasibility", {}).get(
      "primitive_added_us_per_layer_max")
  headroom_passes = isinstance(headroom, (int, float)) and headroom > 0.0
  no_target_evidence = not any(
      (args.out_dir / name).exists()
      for name in ("raw", "run.json", "probe.json", "tokens.jsonl"))

  numeric_basis = {
      "specification": (
          "https://registry.khronos.org/OpenCL/specs/unified/html/"
          "OpenCL_C.html#opencl-numerical-compliance"),
      "binary32_add_mul_and_conversion": "correctly_rounded",
      "binary32_sqrt": "correctly_rounded",
      "binary32_reciprocal": "up_to_2.5_ulp_without_optional_flag",
      "selected_primitive": (
          "positive-normal binary32 reciprocal by exact uint64 quotient, "
          "remainder, and ties-to-even rounding"),
      "domain": (
          "positive normal denominator with normal reciprocal; Q/K denominator "
          "is max(correctly-rounded sqrt(sum_f32), positive norm_epsilon)"),
      "proof_scope": (
          "all 2^23 binary32 significands at exponent 0; exponent scaling does "
          "not change quotient/remainder rounding inside the normal domain"),
  }
  checks = [
      {"name": "seq618_selected_primitive_feasibility_only",
       "pass": route_selects},
      {"name": "seq617_localizes_qk_after_exact_v_with_positive_timing",
       "pass": component_localizes},
      {"name": "cpu_and_level_zero_differ_at_non_correctly_rounded_reciprocal_surface",
       "pass": source_formulas_match},
      {"name": "integer_reciprocal_matches_host_rne_for_all_binary32_significands",
       "pass": exhaustive_passes,
       "detail": verifier},
      {"name": "primitive_has_positive_incremental_component_headroom",
       "pass": headroom_passes,
       "detail": {"added_us_per_layer_max": headroom}},
      {"name": "feasibility_gate_created_no_target_or_token_evidence",
       "pass": no_target_evidence},
  ]
  required = all(bool(row["pass"]) for row in checks)
  design = {
      "primitive_id": "iq36_cr_recip_normal_f32_u64_v1",
      "input": "positive normal binary32 denominator",
      "output": "round-to-nearest-even binary32 reciprocal",
      "operations": [
          "extract exponent and 24-bit significand",
          "compute exact 48-by-24-bit quotient and remainder in uint64",
          "round quotient ties-to-even",
          "renormalize carry and reassemble binary32 bits",
      ],
      "floating_divide_or_rsqrt_allowed": False,
      "compiler_correct_divide_sqrt_flag_allowed": False,
      "third_party_runtime_dependency": False,
      "incremental_added_us_per_layer_max": headroom,
      "next_gate": (
          "lock one Level Zero v2 component contract around this primitive; "
          "do not add source or run the target in the design gate"),
      "stop_condition": (
          "one source implementation, one fresh target compile, and one paired "
          "component repeat/confirm; any Q/K mismatch or timing failure closes "
          "the correctly-rounded primitive route"),
  }
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "component": _rel(args.component),
          "cpu_source": _rel(args.cpu_source),
          "module_source": _rel(args.module_source),
      },
      "numeric_basis": numeric_basis,
      "verifier": {
          "source": _rel(verifier_source),
          "binary": _rel(verifier_binary),
          "compile": compile_result,
          "run": run_result,
          "result": verifier,
      },
      "design": design,
      "checks": checks,
      "required_checks_passed": required,
      "primitive_feasible": required,
      "design_gate_allowed": required,
      "source_allowed": False,
      "target_command_allowed": False,
      "component_probe_allowed": False,
      "decode_integration_allowed": False,
      "token_row_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_iq36_cr_recip_normal_f32_u64_v1_feasibility"
          if required else "reject_or_repair_correctly_rounded_qk_feasibility"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE),
      "next_route_reason": (
          "The exact uint64 quotient/remainder primitive covers every binary32 "
          "significand and fits the seq617 timing envelope arithmetically. Lock "
          "one v2 component design next; source and target execution remain "
          "blocked."
          if required else
          "Do not add source until route selection, exhaustive reciprocal proof, "
          "source attribution, and timing headroom all pass."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "numeric_basis": metrics["numeric_basis"],
          "design": metrics["design"],
          "primitive_feasible": metrics["primitive_feasible"],
          "selected_next_route": metrics["selected_next_route"],
          "source_allowed": False,
          "target_command_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  result = metrics["verifier"]["result"]
  lines = [
      f"# Seq{metrics['sequence']} Correctly-Rounded Q/K Scale Feasibility",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- primitive_feasible: `{str(metrics['primitive_feasible']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- mantissas_checked: `{result.get('mantissas_checked')}`",
      f"- reciprocal_mismatches: `{result.get('mismatch_count')}`",
      f"- incremental_added_us_per_layer_max: `"
      f"{metrics['design']['incremental_added_us_per_layer_max']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This gate compiled and ran a local integer verifier only. No target command ran.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=619)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / (
          "output/seq618-gpu-level-zero-postconv-recurrent-component-route-"
          "close-gate-20260710Tseq618Z/metrics.json"))
  parser.add_argument(
      "--component", type=Path,
      default=ROOT / (
          "output/seq617-gpu-level-zero-postconv-recurrent-component-probe-"
          "gate-20260710Tseq617Z/metrics.json"))
  parser.add_argument("--cpu-source", type=Path,
                      default=ROOT / "engine/src/gguf_loader.cpp")
  parser.add_argument("--module-source", type=Path,
                      default=ROOT / "engine/gpu/level_zero/iq36_postconv_recurrent.cl")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq619-gpu-software-correctly-rounded-qk-scale-primitive-"
          "feasibility-gate-20260710Tseq619Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "primitive_feasible": metrics["primitive_feasible"],
      "design_gate_allowed": metrics["design_gate_allowed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
