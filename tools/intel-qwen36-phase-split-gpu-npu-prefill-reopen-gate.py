#!/usr/bin/env python3
"""Recheck one prefill-only GPU+NPU route after GPU-only decode acceptance.

This gate uses existing evidence only.  It does not reopen the failed NPU
decode carrier or authorize a partition/compiler/precision sweep.
"""

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
SCHEMA = "intel-qwen36-phase-split-gpu-npu-prefill-reopen-v0"
ACCEPTANCE = ROOT / "benchmarks" / WORKSTREAM / "acceptance-matrix.json"
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
SEQ653 = ROOT / "output/product-architecture-feasibility-reconciliation-20260711Tseq653v2cleanZ/result.json"
SEQ655 = ROOT / "output/npu-exact-q6-representation-20260711Tseq655cleanZ/result.json"
SEQ743 = ROOT / "output/packed-token-level-zero-real-backend-gate-20260712Tseq743-distribution-cleanZ/result.json"
SEQ744 = ROOT / "output/packed-token-level-zero-real-backend-gate-20260712Tseq744-distribution-confirm-cleanZ/result.json"
SEQ759 = ROOT / "output/fused-expert-ffn-design-gate-20260712Tseq759cleanZ/result.json"
SEQ762 = ROOT / "output/complete-ffn-route-reflection-gate-20260712Tseq762cleanZ/result.json"
SEQ764 = ROOT / "output/complete-ffn-microkernel-source-gate-20260712Tseq764cleanZ/result.json"
SEQ765 = ROOT / "output/native-prefill-product-route-reconciliation-20260712Tseq765cleanZ/result.json"
FFN_CAP_US = 6250.0
NOISE_FRACTION = 0.005


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--acceptance", type=Path, default=ACCEPTANCE)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--seq653", type=Path, default=SEQ653)
  parser.add_argument("--seq655", type=Path, default=SEQ655)
  parser.add_argument("--seq743", type=Path, default=SEQ743)
  parser.add_argument("--seq744", type=Path, default=SEQ744)
  parser.add_argument("--seq759", type=Path, default=SEQ759)
  parser.add_argument("--seq762", type=Path, default=SEQ762)
  parser.add_argument("--seq764", type=Path, default=SEQ764)
  parser.add_argument("--seq765", type=Path, default=SEQ765)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/phase-split-gpu-npu-prefill-reopen-{stamp}"
  return args


def load(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise RuntimeError(f"expected object: {path}")
  return value


def rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def git_output(*parts: str) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=ROOT, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def probe_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
  return [
      row["probe"] for row in result.get("rows", [])
      if isinstance(row, dict) and isinstance(row.get("probe"), dict)]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  acceptance = load(args.acceptance)
  model = load(args.model_contract)
  seq653 = load(args.seq653)
  seq655 = load(args.seq655)
  seq743 = load(args.seq743)
  seq744 = load(args.seq744)
  seq759 = load(args.seq759)
  seq762 = load(args.seq762)
  seq764 = load(args.seq764)
  seq765 = load(args.seq765)

  aggregate_tops = float(seq653["bounds"]["prefill"]["aggregate_m64_tops"])
  routed_padded_macs = int(seq759["design"]["matrix_macs"])
  hidden = int(model["model"]["hidden_size"])
  intermediate = int(model["model"]["moe_intermediate_size"])
  tile_tokens = int(seq765["product_tile"]["tokens"])
  shared_macs = tile_tokens * 3 * hidden * intermediate
  complete_macs = routed_padded_macs + shared_macs
  complete_ops = 2 * complete_macs
  projected_matrix_us = complete_ops / (aggregate_tops * 1e6)

  fixed_nonmatrix_us = float(seq759["design"]["fixed_nonmatrix_us"])
  scalar_gate_us = min(
      float(row["stage_median_us"]["shared_scalar_gate"])
      for row in probe_rows(seq764))
  projected_complete_us = projected_matrix_us + fixed_nonmatrix_us + scalar_gate_us
  margin_us = FFN_CAP_US - projected_complete_us
  matrix_window_us = FFN_CAP_US - fixed_nonmatrix_us - scalar_gate_us
  required_aggregate_tops = complete_ops / (matrix_window_us * 1e6)
  proxy_rate_margin = aggregate_tops / required_aggregate_tops - 1.0
  two_noise_bands_us = 2.0 * NOISE_FRACTION * FFN_CAP_US

  concurrent = [
      float(row["wall_us"])
      for row in seq653["worker"]["components"]["m64"]["concurrent"]]
  proxy_spread = (max(concurrent) - min(concurrent)) / min(concurrent)

  decode_pass = all(
      result.get("required_checks_passed") is True
      and result.get("disposition")
      == "accept_short_decode_performance_and_distribution_slice"
      for result in (seq743, seq744))
  npu_native_legal = any(
      row.get("name") == "native_process_maps_no_openvino"
      and row.get("pass") is True
      for row in seq655.get("checks", []))
  npu_exact_numeric = any(
      row.get("name") == "npu_all_value_component_correctness"
      and row.get("pass") is True
      for row in seq655.get("checks", []))
  complete_boundary = any(
      row.get("name") == "model_contract_names_shared_expert_and_moe_residual"
      and row.get("pass") is True
      for row in seq762.get("checks", []))
  old_route_exhaustion = (
      seq765.get("reconciliation_completed") is True
      and seq765.get("disposition")
      == "no_admissible_native_prefill_route_under_locked_contract")

  dirty = git_output("status", "--porcelain")
  checks = [
      check("repository_clean_at_gate", dirty == "", dirty_paths=dirty.splitlines()),
      check("product_contract_remains_1p10",
            float(acceptance["r0_target_policy"]["minimum_openvino_speedup_ratio"])
            == 1.1),
      check("later_gpu_only_decode_repeat_confirm_pass", decode_pass),
      check("npu_native_blob_boundary_is_legal_without_openvino_runtime",
            npu_native_legal),
      check("npu_exact_low_bit_component_numeric_precedent_passes",
            npu_exact_numeric),
      check("restored_complete_ffn_boundary_includes_shared_expert",
            complete_boundary),
      check("old_route_exhaustion_record_is_present", old_route_exhaustion),
      check("phase_split_projection_clears_complete_ffn_cap",
            projected_complete_us <= FFN_CAP_US,
            projected_us=projected_complete_us, cap_us=FFN_CAP_US),
      check("projection_margin_exceeds_two_registered_noise_bands",
            margin_us >= two_noise_bands_us,
            margin_us=margin_us, required_us=two_noise_bands_us),
      check("proxy_is_directional_not_promotable_due_to_spread",
            proxy_spread > NOISE_FRACTION,
            spread_fraction=proxy_spread,
            noise_fraction=NOISE_FRACTION),
  ]
  required = all(bool(row["pass"]) for row in checks)

  return {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "commit": git_output("rev-parse", "HEAD"),
      "inputs": {
          "acceptance": rel(args.acceptance),
          "model_contract": rel(args.model_contract),
          "seq653": rel(args.seq653),
          "seq655": rel(args.seq655),
          "seq743": rel(args.seq743),
          "seq744": rel(args.seq744),
          "seq759": rel(args.seq759),
          "seq762": rel(args.seq762),
          "seq764": rel(args.seq764),
          "seq765": rel(args.seq765),
      },
      "new_rationale": {
          "decode": (
              "Seq743/744 now supply a passing native GPU-only decode phase, "
              "so the failed NPU M=1 decode carrier is no longer on the "
              "prefill route's critical path."),
          "prefill": (
              "Seq759 supplies a 626.566 us fused router/scatter shell and "
              "seq762 restores shared-expert work; neither fact existed in "
              "the ADR-0014 stop decision."),
          "route_is_materially_distinct": True,
          "reopens_npu_decode": False,
      },
      "projection": {
          "aggregate_proxy_tops": aggregate_tops,
          "routed_padded_macs": routed_padded_macs,
          "shared_true_macs": shared_macs,
          "complete_macs": complete_macs,
          "complete_ops": complete_ops,
          "projected_matrix_us": projected_matrix_us,
          "fixed_nonmatrix_us": fixed_nonmatrix_us,
          "shared_scalar_gate_us": scalar_gate_us,
          "projected_complete_us": projected_complete_us,
          "cap_us": FFN_CAP_US,
          "margin_us": margin_us,
          "two_noise_bands_us": two_noise_bands_us,
          "required_aggregate_tops": required_aggregate_tops,
          "proxy_rate_margin_fraction": proxy_rate_margin,
          "proxy_paired_spread_fraction": proxy_spread,
          "cross_device_sync_budget_in_projection_us": 0.0,
          "interpretation": (
              "The bound is narrow and intentionally optimistic. Exact Q4_K "
              "affine work, cross-device synchronization, shared final add, "
              "and every queue drain must be present in the measured source "
              "gate; none may be waived because it is absent here."),
      },
      "source_gate_contract": {
          "route": "gpu_npu_prefill_only_exact_q4_complete_ffn_component_v1",
          "fixed_partition": "2:1 GPU:NPU disjoint routed rows; shared expert on GPU",
          "layer": 27,
          "tokens": tile_tokens,
          "oracle_end": "ffn_out",
          "runtime": "native Level Zero GPU plus NPU graph ABI; no OpenVINO/oneDNN mapping",
          "correctness": {"cosine_min": 0.999, "relative_l2_max": 0.002},
          "performance": {
              "complete_repeat_confirm_us_max": FFN_CAP_US,
              "paired_spread_fraction_max": NOISE_FRACTION,
              "timed_host_upload_bytes": 0,
              "timed_host_readback_bytes": 0,
          },
          "must_charge": [
              "exact Q4_K group-32 scale and affine-min semantics",
              "routed and shared gate/up, SwiGLU, down, weighting, and final add",
              "GPU/NPU shared allocation, fences, synchronization, and queue drains",
              "router, gather, compact assignment, and deterministic scatter",
          ],
          "stop_condition": (
              "Any legality, build, all-value correctness, complete-cap, or "
              "noise failure closes the phase-split source. No partition, "
              "compiler flag, graph shape, precision, bucket, or sync sweep."),
      },
      "checks": checks,
      "required_checks_passed": required,
      "route_reopened": required,
      "source_implementation_allowed": required,
      "speedup_claims_allowed": False,
      "product_goal_complete": False,
      "owner_contract_change_required": False,
      "disposition": (
          "reopen_one_prefill_only_gpu_npu_exact_complete_ffn_source_gate"
          if required else "retain_owner_contract_decision"),
      "selected_next_route": (
          "gpu_npu_prefill_only_exact_q4_complete_ffn_component_gate"
          if required else
          "owner_contract_decision_after_measured_1p10_prefill_exhaustion"),
      "next_route_reason": (
          "The later GPU-only decode pass removes the failed NPU decode lane, "
          "and the corrected complete-FFN projection clears the cap by more "
          "than two noise bands. Run one exact source gate; its narrow margin "
          "forbids all tuning after any failure."
          if required else
          "The phase-split projection does not justify superseding route exhaustion."),
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  p = result["projection"]
  failed = [row["name"] for row in result["checks"] if not row["pass"]]
  lines = [
      "# Phase-split GPU+NPU Prefill Reopen Gate",
      "",
      f"- required_checks_passed: `{str(result['required_checks_passed']).lower()}`",
      f"- route_reopened: `{str(result['route_reopened']).lower()}`",
      f"- disposition: `{result['disposition']}`",
      f"- selected_next_route: `{result['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      (f"Projected complete FFN: `{p['projected_complete_us']:.3f} us` "
       f"versus `{p['cap_us']:.3f} us`; margin `{p['margin_us']:.3f} us`."),
      (f"Required aggregate rate: `{p['required_aggregate_tops']:.3f} TOPS`; "
       f"proxy `{p['aggregate_proxy_tops']:.3f} TOPS`."),
      "",
      result["next_route_reason"],
      "",
      "This gate used existing evidence only; no target command ran.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  required = [
      args.acceptance, args.model_contract, args.seq653, args.seq655,
      args.seq743, args.seq744, args.seq759, args.seq762, args.seq764,
      args.seq765,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required inputs: " + ", ".join(missing))
  result = compute(args)
  write_outputs(result, args.out_dir.resolve())
  print(json.dumps({
      "required_checks_passed": result["required_checks_passed"],
      "route_reopened": result["route_reopened"],
      "disposition": result["disposition"],
      "selected_next_route": result["selected_next_route"],
      "out_dir": rel(args.out_dir),
  }, sort_keys=True))
  return 0 if result["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
