#!/usr/bin/env python3
"""Reflect on closed OpenVINO routes and select at most one bounded successor.

This gate consumes existing evidence only.  It diagnoses whether the latest
failure is a metric, scope, or algorithm-route problem; audits terminal reopen
conditions; and may admit one compile-only codegen gate.  It launches no
compiler, model, OpenVINO worker, or GPU command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-route-exhaustion-reflection-v0"
CURRENT_ROUTE = "openvino_route_exhaustion_reflection"
SELECTED_ROUTE = "openvino_hot_cold_partial_storage_specialization_codegen_gate"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
ACCEPTED = ACTIVE / "accepted-cuts.json"
REJECTED = ACTIVE / "rejected-routes.json"
ACCEPTANCE = ROOT / "benchmarks" / WS / "acceptance-matrix.json"
BASE_COMPONENT = ROOT / (
    "output/openvino-direct-i8-hybrid-k2-v4-attention-component-"
    "20260715Tseq1269-cleanZ/result.json")
SPLIT_BOUND = ROOT / (
    "output/openvino-split-state-owner-hot16k-k2-v4-bound-"
    "20260715Tseq1274-cleanZ/metrics.json")
SPLIT_COMPONENT = ROOT / (
    "output/openvino-split-state-owner-hot16k-k2-v4-attention-component-"
    "20260715Tseq1275-cleanZ/result.json")
PARTIAL_SOURCE = ROOT / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
PARTIAL_ZE = ROOT / (
    "output/openvino-direct-i8-group4-dispatch-audit-"
    "20260715Tseq1266-diagnostic/standalone-disasm/.ze_info")
PARTIAL_ASM = ROOT / (
    "output/openvino-direct-i8-group4-dispatch-audit-"
    "20260715Tseq1266-diagnostic/standalone-disasm/"
    ".text.iq36_direct_i8_hotcold_partial.asm")

CAP_MS = 0.5618915
REQUIRED_CLOSED_ROUTES = {
    "openvino_fixed_shape_decode_u4_f16_microkernel_v28n",
    "openvino_provider_aware_fc_linear_state_bundle_v28p",
    "openvino_fc_full_attention_projection_consumer_boundary_v28q",
    "openvino_decode_elementwise_residual_bundle_v28r",
    "openvino_dense_f16_attention_algorithm_v28s",
    "openvino_whole_layer_materialization_superkernel_v28t",
    "openvino_priority_prefill_transpose_gdn_adjacent_fusion_v28u",
    "openvino_all_ten_fixed_direct_i8_group32_product_v28v",
    "openvino_long_prefill_dq_fc_moe_complete_bound_v28w",
    "openvino_context_attention_current_microkernel_v28x",
    "openvino_group4_direct_i8_single_owner_integration_v28y",
    "openvino_hybrid_k2_v4_direct_i8_single_owner_integration_v28z",
    "openvino_split_state_owner_hot16k_k2_v4_component_v29a",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  status = [row for row in status if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def candidate_selects(
    routes: dict[str, Any], sequence: int, selected: str,
) -> bool:
  return any(
      row.get("seq") == sequence and row.get("selected_next_route") == selected
      for row in routes.get("candidate_history", [])
      if isinstance(row, dict))


def switch_selects(
    routes: dict[str, Any], sequence: int, decision: str,
) -> bool:
  return any(
      row.get("seq_covered") == sequence
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
      if isinstance(row, dict))


def kernel_block(text: str, name: str) -> str:
  match = re.search(
      rf"  - name:\s+{re.escape(name)}\n(.*?)(?=\n  - name:|\n"
      r"kernels_misc_info:|\Z)", text, re.DOTALL)
  return match.group(1) if match else ""


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required_paths = (
      STATUS, ROUTES, ACCEPTED, REJECTED, ACCEPTANCE, BASE_COMPONENT,
      SPLIT_BOUND, SPLIT_COMPONENT, PARTIAL_SOURCE, PARTIAL_ZE, PARTIAL_ASM)
  missing = [
      display_path(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing reflection inputs: " + ", ".join(missing))

  git = git_state(output)
  routes = load_json(ROUTES)
  accepted = load_json(ACCEPTED)
  rejected = load_json(REJECTED)
  acceptance = load_json(ACCEPTANCE)
  base = load_json(BASE_COMPONENT)
  split_bound = load_json(SPLIT_BOUND)
  split = load_json(SPLIT_COMPONENT)
  status_text = STATUS.read_text(encoding="utf-8")
  source_text = PARTIAL_SOURCE.read_text(encoding="utf-8")
  ze_text = PARTIAL_ZE.read_text(encoding="utf-8")
  asm_lines = len(PARTIAL_ASM.read_text(
      encoding="utf-8", errors="replace").splitlines())
  sample_memory("after-evidence-load", stop_bytes, memory)

  closed = {
      str(row["route"]): row for row in rejected.get("rejected", [])
      if isinstance(row, dict) and isinstance(row.get("route"), str)}
  missing_closed = sorted(REQUIRED_CLOSED_ROUTES - set(closed))
  empty_reopen = sorted(
      route for route in REQUIRED_CLOSED_ROUTES
      if not closed.get(route, {}).get("reopen_condition"))

  split_result = split["result"]
  samples = split_result["samples"]
  partial_inference = latency_cap_inference(
      [float(row["partial_ms"]) for row in samples],
      cap=CAP_MS, min_samples=20)
  reduce_inference = latency_cap_inference(
      [float(row["reduce_ms"]) for row in samples],
      cap=CAP_MS, min_samples=20)
  update_inference = latency_cap_inference(
      [float(row["update_ms"]) for row in samples],
      cap=CAP_MS, min_samples=20)
  total_inference = split["performance_inference"]
  base_inference = base["performance_inference"]
  total_cut_ms = float(total_inference["upper_confidence_bound_ms"]) - CAP_MS
  partial_cut_after_free_tail_ms = (
      float(partial_inference["upper_confidence_bound_ms"]) - CAP_MS)
  ideal_scaled_ucb = float(split_bound["byte_scaled_timing"]["scaled_ucb_ms"])
  non_byte_excess_ms = (
      float(total_inference["upper_confidence_bound_ms"]) - ideal_scaled_ucb)
  required_non_byte_recovery_fraction = total_cut_ms / non_byte_excess_ms
  partial_budget_ms = (
      CAP_MS - float(reduce_inference["upper_confidence_bound_ms"])
      - float(update_inference["upper_confidence_bound_ms"]))
  partial_required_speedup = (
      float(partial_inference["upper_confidence_bound_ms"]) /
      partial_budget_ms)

  partial_block = kernel_block(
      ze_text, "iq36_direct_i8_hotcold_partial")
  source_shape = {
      "branch_is_uniform_per_workgroup": (
          "const bool cold_chunk = chunk_begin < IQ36_COLD_TOKENS;" in
          source_text),
      "cold_and_hot_paths_share_one_entrypoint": (
          "__kernel void iq36_direct_i8_hotcold_partial(" in source_text
          and "if (cold_chunk)" in source_text
          and "} else {" in source_text),
      "chunk_and_workspace_shape_is_fixed": (
          "#define IQ36_CHUNK_TOKENS 512U" in source_text
          and "#define IQ36_CHUNK_COUNT " in source_text
          and "partial_output" in source_text),
      "current_partial_is_simd16_128grf_no_spill": (
          "simd_size:       16" in partial_block
          and "grf_count:       128" in partial_block
          and "slm_size:        18496" in partial_block
          and "spill" not in partial_block.lower()
          and asm_lines == 2992),
  }

  token_contract = acceptance.get("accuracy", {}).get("tokens", {})
  promotion = acceptance.get("performance_promotion", {})
  active = routes.get("active_route", {})
  accepted_ids = {
      row.get("id") for row in accepted.get("accepted", [])
      if isinstance(row, dict)}
  no_runtime_evidence = not any(
      (output / name).exists()
      for name in ("run.json", "probe.json", "tokens.jsonl", "worker.time"))
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1275_selected_exact_route_exhaustion_reflection",
            active.get("id") == CURRENT_ROUTE
            and candidate_selects(routes, 1275, CURRENT_ROUTE)
            and switch_selects(
                routes, 1275,
                "close_split_state_owner_component_enter_route_exhaustion_reflection")
            and "route-exhaustion" in status_text),
      check("product_contract_cannot_be_relaxed_by_reflection",
            token_contract.get(
                "deterministic_greedy_exact_match_required") is True
            and token_contract.get("first_divergence_blocks_promotion") is True
            and promotion.get("route_rejection_completes_project_goal") is False
            and acceptance.get("accuracy", {}).get(
                "teacher_forced_distribution", {}).get(
                    "kl_divergence_max") == 0.005),
      check("required_closed_routes_and_reopen_conditions_are_present",
            not missing_closed and not empty_reopen,
            missing_routes=missing_closed, empty_reopen_conditions=empty_reopen),
      check("accepted_carrier_and_fine_codec_evidence_remain_distinct",
            {
                "openvino_level_zero_linear_state_alias",
                "openvino_group4_direct_i8_component_and_one_layer_semantics",
                "openvino_hybrid_k2_v4_direct_i8_component_and_one_layer_semantics",
            }.issubset(accepted_ids)),
      check("seq1275_is_latency_only_failure_not_metric_or_scope_failure",
            split.get("required_checks_passed") is False
            and split_result.get("numeric_pass") is True
            and split_result.get("output_relative_l2") == 0.000143412849339
            and split_result.get("execution_order") ==
                "partial_then_reduce_then_update"
            and split.get("checks", [])[7].get("name") ==
                "one_sided_95pct_ucb_clears_complete_cap"
            and split.get("checks", [])[7].get("pass") is False
            and all(
                row.get("pass") is True
                for index, row in enumerate(split.get("checks", []))
                if index != 7)),
      check("partial_kernel_is_the_complete_measured_bottleneck",
            float(total_inference["upper_confidence_bound_ms"]) == 0.587707
            and float(partial_inference["upper_confidence_bound_ms"]) ==
                0.576354
            and partial_cut_after_free_tail_ms > 0.0
            and partial_required_speedup < 1.05,
            total_inference=total_inference,
            partial_inference=partial_inference,
            reduce_inference=reduce_inference,
            update_inference=update_inference,
            partial_budget_ms=partial_budget_ms,
            required_partial_speedup=partial_required_speedup),
      check("measured_non_byte_excess_can_fund_one_codegen_gate",
            base_inference.get("upper_confidence_bound_ms") == 0.540728
            and ideal_scaled_ucb < CAP_MS
            and non_byte_excess_ms > total_cut_ms
            and required_non_byte_recovery_fraction < 0.92,
            ideal_scaled_ucb_ms=ideal_scaled_ucb,
            observed_ucb_ms=total_inference.get("upper_confidence_bound_ms"),
            non_byte_excess_ms=non_byte_excess_ms,
            required_cut_ms=total_cut_ms,
            required_non_byte_recovery_fraction=
                required_non_byte_recovery_fraction),
      check("source_and_stored_codegen_expose_storage_class_specialization",
            all(source_shape.values()), source_shape=source_shape),
      check("reflection_created_no_runtime_or_compile_evidence",
            no_runtime_evidence),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  selected_route = SELECTED_ROUTE if required_checks_passed else CURRENT_ROUTE

  alternatives = [
      {
          "rank": 1,
          "route": SELECTED_ROUTE,
          "status": "selected_compile_only",
          "reason": (
              "Keep hot16k/cold16k K2/V4, chunk512, workgroups, workspace, "
              "reduce, and update fixed. Compile distinct cold-only and "
              "hot-only partial entrypoints so the uniform storage-class "
              "branch and inactive path do not share GRF/code size."),
      },
      {
          "rank": 2,
          "route": "openvino_integer_dpas_attention_arithmetic_bound",
          "status": "parked_source_only",
          "reason": (
              "PTL exposes integer DPAS, but query/weight quantization adds a "
              "new correctness boundary. It needs an offline real-boundary "
              "error and complete traffic/compute bound before source."),
      },
      {
          "rank": 3,
          "route": "openvino_locked_target_infeasibility_record",
          "status": "terminal_not_goal_completion",
          "reason": (
              "Use only after every materially different complete bound "
              "fails. The acceptance contract explicitly says route rejection "
              "does not complete the project goal."),
      },
  ]
  codegen_contract = {
      "source_change": (
          "split iq36_direct_i8_hotcold_partial into cold-only and hot-only "
          "entrypoints; do not change codec, window, token tile, chunk, "
          "workgroup, subgroup, workspace, reduce, update, or sample policy"),
      "compiler_only": True,
      "gpu_kernel_execution_allowed": False,
      "required_kernel_count": 2,
      "required_simd": 16,
      "maximum_grf_per_specialized_kernel": 96,
      "maximum_slm_bytes": 18496,
      "spills_allowed": False,
      "each_specialized_asm_lines_must_be_below": asm_lines,
      "passing_action": (
          "admit one 20-sample component only after both specialized kernels "
          "pass codegen and exact source-isolation checks"),
      "failing_action": (
          "close specialization without GPU execution and source-bound the "
          "parked integer-DPAS arithmetic route"),
  }
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "required_checks_passed": required_checks_passed,
      "reflection_passed": required_checks_passed,
      "diagnosis": "algorithm_route" if required_checks_passed else "audit_gap",
      "metric_change_allowed": False,
      "scope_change_allowed": False,
      "source_edit_allowed": required_checks_passed,
      "compiler_only_gate_allowed": required_checks_passed,
      "gpu_worker_allowed": False,
      "model_worker_allowed": False,
      "long_worker_allowed": False,
      "product_worker_allowed": False,
      "speedup_claim_allowed": False,
      "selected_next_route": selected_route,
      "alternatives": alternatives,
      "latency_accounting": {
          "component_cap_ms": CAP_MS,
          "total": total_inference,
          "partial": partial_inference,
          "reduce": reduce_inference,
          "update": update_inference,
          "total_cut_ms": total_cut_ms,
          "partial_cut_after_free_tail_ms": partial_cut_after_free_tail_ms,
          "partial_budget_ms": partial_budget_ms,
          "partial_required_speedup": partial_required_speedup,
          "ideal_byte_scaled_ucb_ms": ideal_scaled_ucb,
          "non_byte_excess_ms": non_byte_excess_ms,
          "required_non_byte_recovery_fraction":
              required_non_byte_recovery_fraction,
      },
      "source_shape": source_shape,
      "codegen_contract": codegen_contract,
      "checks": checks,
      "memory_samples": memory,
      "inputs": {display_path(path): sha256(path) for path in required_paths},
  }
  (output / "metrics.json").write_text(
      json.dumps(result, indent=2) + "\n", encoding="utf-8")
  summary = f"""# OpenVINO route-exhaustion reflection

Required checks: **{str(required_checks_passed).lower()}**. Diagnosis:
`{result['diagnosis']}`. Selected next route: `{selected_route}`.

Seq1275 fails latency only. Total UCB is
`{total_inference['upper_confidence_bound_ms']} ms`; partial alone is
`{partial_inference['upper_confidence_bound_ms']} ms`, so deleting reduce and
update still misses the `{CAP_MS} ms` cap by
`{partial_cut_after_free_tail_ms:.9f} ms`. Keeping the measured tail requires
only a `{(partial_required_speedup - 1.0) * 100.0:.3f}%` partial cut.

The source uses one SIMD16/128-GRF/18,496-B-SLM, 2992-line entrypoint for a
workgroup-uniform cold/hot branch. The only selected successor keeps every
semantic and scheduling parameter fixed and compiles cold-only and hot-only
entrypoints. It advances only if both reach <=96 GRFs, remain spill-free, and
shrink below the combined assembly size. No compiler or GPU command ran here.

The integer-DPAS arithmetic route is parked behind a separate offline numeric
and roofline bound. Contract relaxation is forbidden, and target infeasibility
is not goal completion.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "required_checks_passed": required_checks_passed,
      "diagnosis": result["diagnosis"],
      "selected_next_route": selected_route,
      "compiler_launched": False,
      "gpu_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
