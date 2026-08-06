#!/usr/bin/env python3
"""Reconcile measured native-prefill routes against the locked product cap.

This is ADR 0047's evidence-only gate.  It runs no target kernel and does not
turn route exhaustion into project completion or a performance claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc" / "active" / WORKSTREAM
SCHEMA = "intel-qwen36-native-prefill-product-route-reconciliation-v0"
ACCEPTANCE = ROOT / "benchmarks" / WORKSTREAM / "acceptance-matrix.json"
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
SEQ753 = ROOT / "output/linear-attention-prefill-state-20260712Tseq753cleanZ/result.json"
SEQ758 = ROOT / "output/linear-prefill-nonstate-feasibility-20260712Tseq758cleanZ/result.json"
SEQ764 = ROOT / "output/complete-ffn-microkernel-source-gate-20260712Tseq764cleanZ/result.json"
OV_PROFILE = ROOT / "output/openvino-hidden-prefill-profile-20260712Tseq751cleanZ/profile.json"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--acceptance", type=Path, default=ACCEPTANCE)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--routes", type=Path, default=ROUTES)
  parser.add_argument("--rejected", type=Path, default=REJECTED)
  parser.add_argument("--seq753", type=Path, default=SEQ753)
  parser.add_argument("--seq758", type=Path, default=SEQ758)
  parser.add_argument("--seq764", type=Path, default=SEQ764)
  parser.add_argument("--openvino-profile", type=Path, default=OV_PROFILE)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/native-prefill-product-route-reconciliation-{stamp}"
  return args


def load(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise RuntimeError(f"expected JSON object: {path}")
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


def row_probes(result: dict[str, Any]) -> list[dict[str, Any]]:
  return [
      row["probe"] for row in result.get("rows", [])
      if isinstance(row, dict) and isinstance(row.get("probe"), dict)]


def has_switch(routes: dict[str, Any]) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == 764
      and row.get("decision")
      == "close_f16_u4_complete_ffn_select_product_prefill_route_reconciliation"
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def has_rejection(rejected: dict[str, Any]) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("route")
      == "native_prefill_f16_u4_active_expert_microkernel_complete_ffn_source_v1"
      for row in rejected.get("rejected", []))


def adr_inventory() -> list[dict[str, Any]]:
  rows = [
      ("cpu_single_device", "doc/adr/0013-select-gpu-npu-parameterized-exact-component-gate.md",
       "CPU decode is 4.2 tok/s and cannot satisfy the product."),
      ("npu_and_gpu_npu_hybrid", "doc/adr/0014-close-fixed-gpu-npu-request-owner-contract-decision.md",
       "The fixed native NPU share caps the pair at 39.376 GB/s; ADR 0015 preserves the closure."),
      ("gpu_grouped_and_handwritten_exact_q4", "doc/adr/0012-close-grouped-prefill-select-product-feasibility-reconciliation.md",
       "Separated, grouped, handwritten, in-core, and compressed-partial routes are closed."),
      ("gpu_linear_state_and_projection", "doc/adr/0044-close-xmx-chunked-gdn-select-whole-linear-stage.md",
       "Register, chunked scalar/XMX, and whole-linear variants reached terminal gates."),
      ("gpu_complete_ffn", "doc/adr/0047-close-f16-u4-complete-ffn-select-product-route-reconciliation.md",
       "M8 and the pinned F16/U4 provider source both fail the complete FFN contract."),
      ("openvino_or_onednn_final_runtime", "AGENTS.md",
       "OpenVINO and oneDNN are references/offline codegen only, not final runtime dependencies."),
  ]
  return [{
      "family": family,
      "decision": path,
      "decision_present": (ROOT / path).exists(),
      "status": "closed_under_locked_contract",
      "reason": reason,
  } for family, path, reason in rows]


def compute(args: argparse.Namespace) -> dict[str, Any]:
  acceptance = load(args.acceptance)
  model = load(args.model_contract)
  routes = load(args.routes)
  rejected = load(args.rejected)
  seq753 = load(args.seq753)
  seq758 = load(args.seq758)
  seq764 = load(args.seq764)
  profile = load(args.openvino_profile)

  target_tps = float(acceptance["bootstrap_targets"]["prefill_tokens_s"]["8192"])
  tile_tokens = 1024
  tile_cap_ms = tile_tokens * 1000.0 / target_tps
  layers = int(model["model"]["layers"])
  linear_layers = int(model["model"]["linear_attention_layers"])
  full_attention_layers = int(model["model"]["full_attention_layers"])

  state_rows = row_probes(seq753)
  projection_rows = row_probes(seq758)
  state_min_us = min(float(row["state_core_median_us"]) for row in state_rows)
  projection_min_us = min(
      float(row["complete_projection_median_us"]) for row in projection_rows)
  ffn_matrix_min_us = min(float(value) for value in seq764["matrix_only_stage_medians_us"])

  ffn_matrix_all_ms = ffn_matrix_min_us * layers / 1000.0
  linear_state_all_ms = state_min_us * linear_layers / 1000.0
  linear_projection_all_ms = projection_min_us * linear_layers / 1000.0
  replay_core_ms = ffn_matrix_all_ms + linear_state_all_ms + linear_projection_all_ms
  replay_gap_ms = replay_core_ms - tile_cap_ms
  replay_required_speedup = replay_core_ms / tile_cap_ms

  profile_run = profile["runs"][0]
  ov_profiled_sum_ms = float(profile_run["profiled_sum_ms"])
  ov_gap_ms = ov_profiled_sum_ms - tile_cap_ms
  ov_required_speedup = ov_profiled_sum_ms / tile_cap_ms
  inventory = adr_inventory()

  state_correct = all(
      row.get("attention_comparison", {}).get("passes") is True
      and row.get("state_comparison", {}).get("passes") is True
      and row.get("final_comparison", {}).get("passes") is True
      for row in state_rows)
  seq764_terminal = (
      seq764.get("evaluation_completed") is True
      and seq764.get("required_checks_passed") is False
      and seq764.get("disposition")
      == "reject_f16_u4_active_expert_complete_ffn_source"
      and max(float(value) for value in seq764["matrix_only_stage_medians_us"])
      > float(seq764["complete_ffn_cap_us"]))
  projection_is_optimistic = all(
      row.get("all_projection_correctness_passed") is False
      for row in projection_rows)
  ffn_is_optimistic = all(
      row.get("correctness_pass") is False for row in row_probes(seq764))

  dirty = git_output("status", "--porcelain")
  checks = [
      check("repository_clean_at_gate", dirty == "", dirty_paths=dirty.splitlines()),
      check("locked_8k_prefill_target_is_2510_tps", target_tps == 2510.0,
            observed=target_tps),
      check("locked_model_shape_is_40_layers_30_linear_10_full",
            (layers, linear_layers, full_attention_layers) == (40, 30, 10),
            observed=[layers, linear_layers, full_attention_layers]),
      check("seq764_terminal_complete_ffn_failure_is_recorded", seq764_terminal),
      check("seq753_real_state_boundaries_are_correct", state_correct),
      check("optimistic_replay_uses_faster_inaccurate_ffn_and_projection_rows",
            projection_is_optimistic and ffn_is_optimistic),
      check("optimistic_current_source_core_already_exceeds_product_tile_cap",
            replay_core_ms > tile_cap_ms, core_ms=replay_core_ms,
            cap_ms=tile_cap_ms),
      check("openvino_profile_is_hidden_body_without_lm_head",
            profile.get("lm_head_removed") is not None
            and profile_run.get("seq_len") == tile_tokens
            and ov_profiled_sum_ms > tile_cap_ms),
      check("seq764_route_switch_and_rejection_are_canonical",
            has_switch(routes) and has_rejection(rejected)),
      check("all_independent_route_decisions_are_present",
            all(row["decision_present"] for row in inventory)),
  ]
  required = all(bool(row["pass"]) for row in checks)
  no_admissible_route = required and replay_core_ms > tile_cap_ms

  return {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "commit": git_output("rev-parse", "HEAD"),
      "inputs": {
          "acceptance": rel(args.acceptance),
          "model_contract": rel(args.model_contract),
          "routes": rel(args.routes),
          "rejected": rel(args.rejected),
          "seq753": rel(args.seq753),
          "seq758": rel(args.seq758),
          "seq764": rel(args.seq764),
          "openvino_profile": rel(args.openvino_profile),
      },
      "product_tile": {
          "bucket": 8192,
          "tokens": tile_tokens,
          "prefill_target_tokens_s": target_tps,
          "cap_ms": tile_cap_ms,
      },
      "optimistic_current_source_replay": {
          "ffn_matrix_only_us_per_layer": ffn_matrix_min_us,
          "ffn_matrix_only_all_layers_ms": ffn_matrix_all_ms,
          "linear_state_us_per_linear_layer": state_min_us,
          "linear_state_all_layers_ms": linear_state_all_ms,
          "linear_projection_us_per_linear_layer": projection_min_us,
          "linear_projection_all_layers_ms": linear_projection_all_ms,
          "core_sum_ms": replay_core_ms,
          "gap_over_cap_ms": replay_gap_ms,
          "required_speedup": replay_required_speedup,
          "omitted_chargeable_work": [
              "ten full-attention layers",
              "linear convolution and controls",
              "normalization and residuals",
              "FFN gather, activation, weighting, scalar gate, and scatter",
              "embedding and final output stages",
          ],
          "why_optimistic": (
              "The FFN and projection timing rows already fail component "
              "accuracy; using their faster times understates any admissible "
              "exact replay, and omitted work is charged as zero."),
      },
      "openvino_hidden_profile": {
          "profiled_sum_ms": ov_profiled_sum_ms,
          "gap_over_cap_ms": ov_gap_ms,
          "required_speedup": ov_required_speedup,
          "wall_ms_with_perf_count": float(profile_run["wall_ms"]),
          "lm_head_removed": profile.get("lm_head_removed"),
          "role": (
              "directional compiler/kernel attribution only; converted U4 "
              "weights and synthetic hidden input are not locked-GGUF "
              "component correctness evidence"),
      },
      "independent_route_inventory": inventory,
      "checks": checks,
      "required_checks_passed": required,
      "reconciliation_completed": required,
      "product_prefill_route_feasible_under_locked_contract": not no_admissible_route,
      "source_implementation_allowed": False,
      "speedup_claims_allowed": False,
      "project_goal_complete": False,
      "owner_contract_decision_required": no_admissible_route,
      "disposition": (
          "no_admissible_native_prefill_route_under_locked_contract"
          if no_admissible_route else "repair_product_route_reconciliation"),
      "selected_next_route": (
          "owner_contract_decision_after_measured_1p10_prefill_exhaustion"
          if no_admissible_route else
          "native_prefill_product_route_reconciliation_v1"),
      "decision_dimensions": [
          "target hardware or additional accelerator",
          "locked model or precision representation",
          "component correctness threshold",
          "batch size",
          "final OpenVINO/oneDNN runtime dependency",
          "minimum OpenVINO speedup ratio",
      ],
      "next_route_reason": (
          "The optimistic replay of the fastest measured source kernels is "
          f"already {replay_gap_ms:.3f} ms over the complete tile cap before "
          "required work, and every independent CPU/GPU/NPU/runtime family "
          "has a canonical terminal decision. A new kernel would relitigate a "
          "closed family; an owner must change one named contract dimension "
          "or supply independently verified new hardware/compiler capability."
          if no_admissible_route else
          "Repair missing or inconsistent evidence before selecting direction."),
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  replay = result["optimistic_current_source_replay"]
  profile = result["openvino_hidden_profile"]
  failed = [row["name"] for row in result["checks"] if not row["pass"]]
  lines = [
      "# Native Prefill Product Route Reconciliation",
      "",
      f"- required_checks_passed: `{str(result['required_checks_passed']).lower()}`",
      f"- reconciliation_completed: `{str(result['reconciliation_completed']).lower()}`",
      f"- disposition: `{result['disposition']}`",
      f"- selected_next_route: `{result['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      "",
      (f"Optimistic current-source core: `{replay['core_sum_ms']:.3f} ms` "
       f"versus `{result['product_tile']['cap_ms']:.3f} ms`; gap "
       f"`{replay['gap_over_cap_ms']:.3f} ms`."),
      (f"OpenVINO hidden-body profiled kernel sum: "
       f"`{profile['profiled_sum_ms']:.3f} ms`; this is directional only."),
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
      args.acceptance, args.model_contract, args.routes, args.rejected,
      args.seq753, args.seq758, args.seq764, args.openvino_profile,
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required inputs: " + ", ".join(missing))
  result = compute(args)
  write_outputs(result, args.out_dir.resolve())
  print(json.dumps({
      "required_checks_passed": result["required_checks_passed"],
      "disposition": result["disposition"],
      "selected_next_route": result["selected_next_route"],
      "out_dir": rel(args.out_dir),
  }, sort_keys=True))
  return 0 if result["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
