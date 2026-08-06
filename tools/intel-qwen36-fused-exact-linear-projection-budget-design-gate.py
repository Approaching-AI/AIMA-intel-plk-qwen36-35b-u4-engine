#!/usr/bin/env python3
"""Lock one same-context exact Q4 projection design under the floor budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-fused-exact-linear-projection-budget-design-v0"
DESIGN_SCHEMA_VERSION = (
    "intel-qwen36-rowblock16-cpuorder-finalize-kernel-design-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_budget_design_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_fused_exact_linear_projection_component_source_gate"
)


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


def _source_markers(opencl: str, runner: str, cpu_order: str,
                    reference: str) -> list[dict[str, Any]]:
  markers = {
      "rowblock16_uses_one_16_lane_workgroup_per_output_row": (
          "__kernel void q4k_x8_matvec_rowblock16_reduce(" in opencl
          and "const uint row = (uint)get_group_id(0);" in opencl
          and "const uint block_index = lid;" in opencl
          and "constexpr std::size_t kRowblock16LocalSize = 16;" in runner
          and "static_cast<std::size_t>(rows) * local" in runner
      ),
      "packed_q4x8_preserves_each_rows_raw_q4_bytes_and_metadata": (
          "void AppendQ4Kx8Block(" in runner
          and "const int src_id = i % 8;" in runner
          and "const int src_offset = (i / 8) * 8;" in runner
          and "dst_qs + i * 8" in runner
          and "+ 16 + src_offset, 8);" in runner
          and "std::memcpy(dst_d + i * 2" in runner
          and "std::memcpy(dst_dmin + i * 2" in runner
          and "auto* dst_scales = dst + 32;" in runner
      ),
      "cpu_order_oracle_accumulates_eight_integer_lanes_by_block": (
          "#pragma OPENCL FP_CONTRACT OFF" in cpu_order
          and "int lane_sums[8];" in cpu_order
          and "for (uint block_index = 0; block_index < blocks_per_row;"
          in cpu_order
          and "sums[lane] += d * (float)lane_sums[lane];" in cpu_order
          and "min_sum -= dmin * (float)grouped_min_sum;" in cpu_order
      ),
      "cpu_reference_has_the_same_direct_integer_and_float_order": (
          "void accumulate_q4_k_q8_k_block_direct(" in reference
          and "std::array<std::int32_t, 8> lane_sums{};" in reference
          and "lane_sums[lane_index] +=" in reference
          and "min_sum -= dmin * static_cast<float>(grouped_min_sum);"
          in reference
      ),
      "current_rowblock16_tree_reduction_is_the_only_order_mismatch": (
          "__local float partial[16];" in opencl
          and "partial[lid] = sumf - sum_minf;" in opencl
          and "for (uint stride = 8U; stride > 0U; stride >>= 1U)" in opencl
      ),
  }
  return [{"name": name, "pass": passed}
          for name, passed in markers.items()]


def compute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  component = _load(args.component)
  prior_budget = _load(args.prior_budget)
  opencl = args.opencl_source.read_text(encoding="utf-8")
  runner = args.runner_source.read_text(encoding="utf-8")
  cpu_order = args.cpu_order_source.read_text(encoding="utf-8")
  reference = args.reference_source.read_text(encoding="utf-8")

  predecessor_design = predecessor.get("design", {})
  predecessor_kill = predecessor_design.get("kill_number", {})
  component_row = component.get("component", {})
  prior_kill = prior_budget.get("kill_number", {})
  baseline_us = float(component_row.get("rowblock16_min_us", math.inf))
  separate_us = float(component_row.get("cpu_order_min_us", math.inf))
  added_ceiling_us = float(
      predecessor_kill.get("maximum_fused_added_us_per_layer", -math.inf))
  absolute_ceiling_us = baseline_us + added_ceiling_us
  current_added_us = separate_us - baseline_us
  required_total_kernel_ratio = separate_us / absolute_ceiling_us

  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("component_design_allowed") is True
      and predecessor.get("component_probe_allowed") is False
      and predecessor.get("token_row_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and predecessor_design.get("route") == CURRENT_ROUTE
      and _has_candidate(routes, 585, CURRENT_ROUTE)
      and _has_switch(
          routes, 585,
          "select_router_prompt_distribution_fused_exact_linear_projection_"
          "budget_design_gate"))
  kill_number_valid = (
      component.get("required_checks_passed") is True
      and component_row.get("cpu_order_gpu_vs_cpu_max_abs_diff") == 0
      and prior_budget.get("required_checks_passed") is True
      and math.isclose(baseline_us, 191.354, rel_tol=0.0, abs_tol=1e-9)
      and math.isclose(separate_us, 209.166, rel_tol=0.0, abs_tol=1e-9)
      and math.isclose(current_added_us, 17.812,
                       rel_tol=0.0, abs_tol=1e-9)
      and math.isclose(
          added_ceiling_us,
          float(prior_kill.get("floor_headroom_us", 0.0)) / 30.0,
          rel_tol=0.0, abs_tol=1e-12)
      and math.isclose(absolute_ceiling_us, 198.1958589939298,
                       rel_tol=0.0, abs_tol=1e-9))

  source_markers = _source_markers(opencl, runner, cpu_order, reference)
  design = {
      "schema_version": DESIGN_SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "name": "rowblock16_cpuorder_finalize",
      "kernel_name": "q4k_x8_matvec_rowblock16_cpuorder_finalize",
      "scope": "Q4_K BPR16 linear-attention output projection",
      "algorithm": {
          "workgroup": (
              "One 16-work-item group per output row; local ID equals the "
              "input block index."
          ),
          "parallel_integer_stage": (
              "Each work-item reconstructs its row's packed Q4 bytes and "
              "scale/min metadata, then computes the exact eight int32 lane "
              "sums and one grouped-min int32 value for one block."
          ),
          "local_state": {
              "lane_sums_shape": [16, 8],
              "grouped_min_shape": [16],
              "element_type": "int32",
              "bytes": 16 * (8 + 1) * 4,
          },
          "ordered_float_finalize": (
              "After one local-memory barrier, local ID 0 iterates blocks "
              "0..15, updates eight float sums and min_sum in the CPU oracle "
              "order with FP_CONTRACT OFF, then adds min_sum followed by "
              "lane sums 0..7 and writes the existing output buffer."
          ),
      },
      "exactness_argument": {
          "integer_products_per_block_current_and_candidate": 256,
          "integer_grouping_is_exact": True,
          "packed_byte_mapping_is_lossless": True,
          "float_block_order_matches_cpu_oracle": True,
          "final_lane_order_matches_cpu_oracle": True,
          "fp_contraction_disabled": True,
          "required_component_max_abs_diff": 0,
      },
      "dispatch_contract": {
          "same_gpu_runner_context": True,
          "same_packed_q4_weight_buffer": True,
          "same_q8_input_buffers": True,
          "same_output_buffer": True,
          "global_work_items": "rows * 16",
          "local_work_items": 16,
          "replacement_dispatch_count": 1,
          "additional_dispatch_count": 0,
          "host_readback_or_bridge": False,
          "runtime_decode_selector_added_at_source_gate": False,
      },
      "budget": {
          "rowblock16_baseline_us": baseline_us,
          "current_separate_cpuorder_us": separate_us,
          "current_added_us_per_layer": current_added_us,
          "maximum_added_us_per_layer": added_ceiling_us,
          "maximum_candidate_absolute_us": absolute_ceiling_us,
          "required_cpuorder_total_kernel_speedup_ratio": (
              required_total_kernel_ratio),
          "linear_layer_count": 30,
          "floor_headroom_us_per_token": predecessor_kill.get(
              "floor_headroom_us_per_token"),
      },
      "bounded_performance_argument": {
          "same_parallel_integer_product_count_as_rowblock16": True,
          "current_tree_barriers": 5,
          "candidate_barriers": 1,
          "candidate_local_memory_bytes": 576,
          "only_new_serial_region": (
              "Sixteen ordered updates for each of eight independent lane "
              "sums plus min_sum and the final eight-lane fold."
          ),
          "verdict": (
              "Plausible under the 3.452% added-wall ceiling, but not a speed "
              "result; one same-context representative-layer component "
              "repeat plus confirm must measure it."
          ),
      },
      "component_acceptance": {
          "source_and_local_compile_gate_required_first": True,
          "target_compile_required_before_probe": True,
          "same_payload_baseline_candidate_cpu_reference_required": True,
          "candidate_gpu_vs_cpu_max_abs_diff": 0,
          "candidate_us_max": absolute_ceiling_us,
          "candidate_added_us_max": added_ceiling_us,
          "repeat_and_confirm_required": True,
          "token_row_allowed": False,
          "subset_sweep_allowed": False,
      },
      "claim_policy": {
          "correctness_claim_allowed": False,
          "speedup_claim_allowed": False,
          "promotion_allowed": False,
      },
  }
  design_is_single_and_bounded = (
      design["name"] == "rowblock16_cpuorder_finalize"
      and design["algorithm"]["local_state"]["bytes"] == 576
      and design["dispatch_contract"]["additional_dispatch_count"] == 0
      and design["dispatch_contract"]["host_readback_or_bridge"] is False
      and design["component_acceptance"]["candidate_gpu_vs_cpu_max_abs_diff"]
      == 0
      and design["component_acceptance"]["token_row_allowed"] is False
      and design["component_acceptance"]["subset_sweep_allowed"] is False)
  performance_probe_is_justified = (
      design["exactness_argument"][
          "integer_products_per_block_current_and_candidate"] == 256
      and design["bounded_performance_argument"]["current_tree_barriers"] == 5
      and design["bounded_performance_argument"]["candidate_barriers"] == 1
      and required_total_kernel_ratio < 1.06)
  checks = [
      {"name": "seq585_selected_design_only_gate",
       "pass": predecessor_selects},
      {"name": "seq555_exact_component_and_seq563_kill_number_match",
       "pass": kill_number_valid,
       "detail": design["budget"]},
      {"name": "live_sources_support_lossless_packed_cpu_order_finalize",
       "pass": all(row["pass"] for row in source_markers),
       "detail": source_markers},
      {"name": "exactly_one_no_bridge_no_extra_dispatch_design_is_locked",
       "pass": design_is_single_and_bounded},
      {"name": "component_measurement_is_justified_but_not_prejudged",
       "pass": performance_probe_is_justified,
       "detail": design["bounded_performance_argument"]},
  ]
  required = all(bool(row["pass"]) for row in checks)
  metrics = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "component": _rel(args.component),
          "prior_budget": _rel(args.prior_budget),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "runner_source": _rel(args.runner_source),
          "runner_source_sha256": _sha256(args.runner_source),
          "cpu_order_source": _rel(args.cpu_order_source),
          "cpu_order_source_sha256": _sha256(args.cpu_order_source),
          "reference_source": _rel(args.reference_source),
          "reference_source_sha256": _sha256(args.reference_source),
      },
      "design": design,
      "checks": checks,
      "required_checks_passed": required,
      "design_passed": required,
      "component_source_allowed": required,
      "component_target_compile_allowed": False,
      "component_probe_allowed": False,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_rowblock16_cpuorder_finalize_design_select_component_source"
          if required else "reject_fused_exact_projection_design"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_distribution_correctness_route_reflection_gate"),
      "next_route_reason": (
          "The packed rowblock16 layout can preserve the CPU oracle's eight "
          "integer lane sums and serialize only its floating-point finalize "
          "inside the existing workgroup. Add this one kernel and same-runner "
          "component API in source-only form, then local-compile it. No target "
          "probe or token is authorized by this design result."
          if required else
          "The source mapping, exact-order argument, or floor budget is not "
          "closed; do not implement or probe a fused projection variant."),
  }
  return metrics, design


def write_outputs(metrics: dict[str, Any], design: dict[str, Any],
                  out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "design.json").write_text(
      json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": _rel(Path(__file__)),
          "inputs": metrics["inputs"],
          "design": _rel(out_dir / "design.json"),
          "selected_next_route": metrics["selected_next_route"],
          "component_source_allowed": metrics["component_source_allowed"],
          "component_probe_allowed": False,
          "token_row_allowed": False,
          "speedup_claims_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  budget = design["budget"]
  lines = [
      f"# Seq{metrics['sequence']} Fused Exact Projection Budget Design",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- design: `{design['name']}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- baseline / candidate ceiling: `{budget['rowblock16_baseline_us']}` / "
      f"`{budget['maximum_candidate_absolute_us']}` us",
      f"- maximum added wall: `{budget['maximum_added_us_per_layer']}` us/layer",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is a source design result, not component correctness or speed evidence.",
      "No target command, token, validation case, or test case was used.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=586)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq585-learned-correction-route-close-gate-20260710Tseq585Z/metrics.json")
  parser.add_argument(
      "--component", type=Path,
      default=ROOT / "output/seq555-layer0-linear-output-projection-cpuorder-component-target-gate-20260710Tseq555Z/metrics.json")
  parser.add_argument(
      "--prior-budget", type=Path,
      default=ROOT / "output/seq563-all-linear-norm-to-projection-parity-feasibility-gate-20260710Tseq563Z/metrics.json")
  parser.add_argument("--opencl-source", type=Path,
                      default=ROOT / "engine/gpu/opencl/q4x8_matvec.cl")
  parser.add_argument("--runner-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4x8_matvec.cpp")
  parser.add_argument("--cpu-order-source", type=Path,
                      default=ROOT / "engine/src/gpu_q4_cpu_order_matvec.cpp")
  parser.add_argument("--reference-source", type=Path,
                      default=ROOT / "engine/src/gguf_loader.cpp")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq586-fused-exact-linear-projection-budget-design-gate-20260710Tseq586Z")
  args = parser.parse_args()
  metrics, design = compute(args)
  write_outputs(metrics, design, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "design": design["name"],
      "candidate_absolute_us_max": design["budget"][
          "maximum_candidate_absolute_us"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
