#!/usr/bin/env python3
"""Authorize the resident hidden-state carrier source contract after seq116.

This is route-control/source-design evidence only. It checks that the current
local handoff, MoE/down-tail, and shared-Q8 preconv boards are closed, that the
required resident primitives exist, and that the live decode source still lacks
the full hidden-state carrier contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-hidden-state-carrier-contract-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_SEQ54 = ROOT / "output/resident-hidden-carrier-gate-20260706Tseq54Z/metrics.json"
DEFAULT_SEQ59 = ROOT / "output/device-q8-q4-no-readback-contract-gate-20260706Tseq59Z/metrics.json"
DEFAULT_SEQ60 = ROOT / "output/device-q8-q6-no-readback-contract-gate-20260706Tseq60Z/metrics.json"
DEFAULT_SEQ61 = ROOT / "output/ffn-tail-resident-input-contract-gate-20260706Tseq61Z/metrics.json"
DEFAULT_SEQ62 = ROOT / "output/q6-selected-shared-defer-contract-gate-20260706Tseq62Z/metrics.json"
DEFAULT_SEQ71 = ROOT / "output/linear-preconv-carrier-gap-gate-20260706Tseq71Z/metrics.json"
DEFAULT_SEQ72 = ROOT / "output/linear-preconv-carrier-primitive-gate-20260706Tseq72Z/metrics.json"
DEFAULT_SEQ73 = ROOT / "output/linear-preconv-carrier-bundle-gate-20260706Tseq73Z/metrics.json"
DEFAULT_SEQ77 = ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z/metrics.json"
DEFAULT_SEQ90 = ROOT / "output/linear-preconv-qkv-conv-root-probe-20260707Tseq90Z/metrics.json"
DEFAULT_SEQ116 = ROOT / "output/moe-routed-down-fusion-route-gate-20260707Tseq116Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/resident-hidden-state-carrier-contract-gate-20260707Tseq117Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_text(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _nested(obj: Any, *keys: str) -> Any:
  cur = obj
  for key in keys:
    if not isinstance(cur, dict):
      return None
    cur = cur.get(key)
  return cur


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _accepted_ids(accepted: dict[str, Any]) -> set[str]:
  ids: set[str] = set()
  for row in accepted.get("accepted", []):
    if isinstance(row, dict) and isinstance(row.get("id"), str):
      ids.add(row["id"])
  return ids


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _parked_route(routes: dict[str, Any], route_id: str) -> dict[str, Any] | None:
  for row in routes.get("parked_routes", []):
    if isinstance(row, dict) and row.get("id") == route_id:
      return row
  return None


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _line_of(text: str, pattern: str, *, regex: bool = False) -> int | None:
  if regex:
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
      return None
    return text.count("\n", 0, match.start()) + 1
  index = text.find(pattern)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def _present_check(text: str, label: str, pattern: str, *,
                   regex: bool = False) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "present": line is not None, "line": line}


def _absent_check(text: str, label: str, pattern: str, *,
                  regex: bool = False) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "absent": line is None, "line": line}


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


def _all_absent(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("absent") is True for row in rows)


def _source_checks(decode_text: str, header_text: str) -> dict[str, Any]:
  host_boundary_checks = [
      _present_check(
          decode_text,
          "linear_layer_api_requires_host_residual_vector",
          r"std::vector<float>\s+RunGpuHybridLinearLayerLive\([^\)]*"
          r"const std::vector<float>&\s+residual",
          regex=True,
      ),
      _present_check(
          decode_text,
          "full_attention_layer_api_requires_host_residual_vector",
          r"std::vector<float>\s+RunGpuHybridFullAttentionLayerLive\([^\)]*"
          r"const std::vector<float>&\s+residual",
          regex=True,
      ),
      _present_check(
          decode_text,
          "attention_front_handle_path_still_requires_host_residual",
          r"AttentionFrontRun\s+RunGpuAttentionFrontFromInputHandle\([^\)]*"
          r"const std::vector<float>&\s+residual_input",
          regex=True,
      ),
      _present_check(
          decode_text,
          "ffn_tail_api_requires_host_ffn_norm_and_attention_residual",
          r"std::vector<float>\s+RunGpuHybridFfnTail\([^\)]*"
          r"const std::vector<float>&\s+ffn_input,[^\)]*"
          r"const std::vector<float>&\s+attention_residual",
          regex=True,
      ),
      _present_check(
          decode_text,
          "tail_output_handle_only_feeds_next_layer_rmsnorm_scope",
          "g_decode_prev_layer_output_handle = resident_tail.layer_output_handle",
      ),
      _present_check(
          decode_text,
          "linear_preconv_front_accepts_resident_attn_norm_handle_but_still_has_host_path",
          r"RunGpuPreConvFront\([^\)]*layer_input_gpu\.attn_norm[^\)]*"
          r"layer_input_gpu\.attn_norm_handle",
          regex=True,
      ),
  ]
  primitive_api_checks = [
      _present_check(
          header_text,
          "q4_resident_input_device_q8_projection_api",
          "RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8(",
      ),
      _present_check(
          header_text,
          "q6_resident_input_device_q8_projection_api",
          "RunF32InputHandleDeviceQ8ThenResidentRawQ6K(",
      ),
      _present_check(
          header_text,
          "q6_resident_input_preconv_qkv_conv_api",
          "RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState(",
      ),
      _present_check(
          header_text,
          "q4_resident_input_preconv_qkv_conv_api",
          "RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState(",
      ),
      _present_check(
          header_text,
          "q6_shared_device_q8_preconv_bundle_api",
          "RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder(",
      ),
      _present_check(
          header_text,
          "q4_shared_device_q8_preconv_bundle_api",
          "RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder(",
      ),
      _present_check(
          header_text,
          "ffn_tail_resident_inputs_api",
          "RunFfnTailFromDownHandlesResidentInputs(",
      ),
      _present_check(
          header_text,
          "ffn_tail_atomic_resident_inputs_api",
          "RunFfnTailAtomicFromDownHandlesResidentInputs(",
      ),
      _present_check(
          header_text,
          "rmsnorm_resident_input_api",
          "RunRmsNormHiddenResidentInputResidentWeight(",
      ),
  ]
  carrier_absence_checks = [
      _absent_check(
          decode_text + "\n" + header_text,
          "no_resident_hidden_state_carrier_struct_yet",
          "ResidentHiddenStateCarrier",
      ),
      _absent_check(
          decode_text + "\n" + header_text,
          "no_resident_hidden_state_carrier_symbol_yet",
          "resident_hidden_state_carrier",
      ),
  ]
  return {
      "host_boundary_checks": host_boundary_checks,
      "primitive_api_checks": primitive_api_checks,
      "carrier_absence_checks": carrier_absence_checks,
      "host_boundaries_present": _all_present(host_boundary_checks),
      "primitive_apis_present": _all_present(primitive_api_checks),
      "full_carrier_contract_absent": _all_absent(carrier_absence_checks),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  accepted = _load_json(args.accepted)
  rejected = _load_json(args.rejected)
  seq54 = _load_json(args.seq54)
  seq59 = _load_json(args.seq59)
  seq60 = _load_json(args.seq60)
  seq61 = _load_json(args.seq61)
  seq62 = _load_json(args.seq62)
  seq71 = _load_json(args.seq71)
  seq72 = _load_json(args.seq72)
  seq73 = _load_json(args.seq73)
  seq77 = _load_json(args.seq77)
  seq90 = _load_json(args.seq90)
  seq116 = _load_json(args.seq116)
  decode_text = _load_text(args.decode_source)
  header_text = _load_text(args.gpu_header)

  frontier_state = _frontier_state(frontier)
  accepted_ids = _accepted_ids(accepted)
  rejected_names = _rejected_names(rejected)
  resident_route = _parked_route(routes, "resident_decode_loop_streaming")
  source = _source_checks(decode_text, header_text)

  required_foundation_ids = {
      "r2_gpu_decode_resident_decode_loop_api_extraction",
      "r2_gpu_decode_resident_session_buffer_ownership",
      "r2_gpu_decode_resident_model_state_bank",
      "r2_gpu_decode_resident_device_state_handle_bank",
      "r2_gpu_decode_resident_norm_small_tensor_store",
      "r2_gpu_decode_resident_f32_matvec_weight_store",
      "r2_gpu_decode_resident_linear_q6_qkv_weight_store",
      "r2_gpu_decode_resident_packed_q4_weight_store",
      "r2_gpu_decode_resident_linear_conv_weight_store",
      "r2_gpu_decode_resident_raw_q6_weight_store",
      "r2_gpu_decode_resident_selected_q6_weight_store",
      "r2_gpu_decode_resident_q4_cpu_order_weight_store",
      "r2_gpu_decode_resident_lm_head_weight_store",
  }
  required_closed_routes = {
      "gpu_local_tail_handoffs_without_hidden_state_carrier",
      "gpu_ffn_tail_plus_attention_residual_carrier_noqueue",
      "gpu_full_carrier_without_resident_input_preconv_handoff",
      "gpu_linear_preconv_qkv_only_resident_input_wiring",
      "gpu_linear_preconv_shared_q8_preconv_bundle_decode",
      "gpu_linear_preconv_shared_q8_qkv_conv_root_component",
      "current_moe_routed_down_fusion_board",
  }
  missing_foundations = sorted(required_foundation_ids - accepted_ids)
  missing_closed_routes = sorted(required_closed_routes - rejected_names)

  seq54_drain = seq54.get("seq53_drain_accounting")
  seq54_drain = seq54_drain if isinstance(seq54_drain, dict) else {}
  seq77_derived = seq77.get("derived")
  seq77_derived = seq77_derived if isinstance(seq77_derived, dict) else {}
  seq90_derived = seq90.get("derived")
  seq90_derived = seq90_derived if isinstance(seq90_derived, dict) else {}

  checks = [
      {
          "name": "seq116_selected_carrier_contract_gate",
          "pass": (
              seq116.get("required_checks_passed") is True
              and seq116.get("selected_next_route")
                  == "resident_hidden_state_carrier_contract_gate"
              and _has_switch(
                  routes,
                  "close_current_moe_routed_down_board_switch_to_resident_hidden_state_carrier_contract",
                  116,
              )
          ),
          "detail": {
              "seq116_disposition": seq116.get("disposition"),
              "seq116_selected_next_route": seq116.get("selected_next_route"),
          },
      },
      {
          "name": "frontier_below_floor_and_tail_drain_elimination_can_clear_floor",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and seq54_drain.get("tail_drain_elimination_clears_floor") is True
              and _num(seq54_drain.get("projected_tps_without_tail_growth"))
                  > frontier_state["floor_tps"]
          ),
          "detail": {
              "current_best_tps": frontier_state["current_best_tps"],
              "floor_tps": frontier_state["floor_tps"],
              "floor_gap_ms_per_token": frontier_state["floor_gap_ms_per_token"],
              "projected_tps_without_tail_growth": seq54_drain.get(
                  "projected_tps_without_tail_growth"),
              "selected_down_wait_saved_ms_per_token": seq54_drain.get(
                  "selected_down_wait_saved_ms_per_token"),
              "ffn_tail_growth_ms_per_token": seq54_drain.get(
                  "ffn_tail_growth_ms_per_token"),
          },
      },
      {
          "name": "resident_loop_foundations_are_accepted",
          "pass": not missing_foundations and resident_route is not None,
          "detail": {
              "missing": missing_foundations,
              "resident_route_rank": (
                  None if resident_route is None else resident_route.get("rank")),
          },
      },
      {
          "name": "local_handoff_and_partial_carrier_boards_are_closed",
          "pass": (
              not missing_closed_routes
              and _nested(
                  seq54,
                  "derived",
                  "local_handoff_closed_without_hidden_state_carrier",
              ) is True
              and _nested(
                  seq54,
                  "derived",
                  "resident_hidden_state_carrier_or_down_tail_fusion_required",
              ) is True
          ),
          "detail": {"missing": missing_closed_routes},
      },
      {
          "name": "resident_primitives_ready_but_not_decode_speed_paths",
          "pass": (
              _nested(seq59, "derived", "primitive_ready_for_carrier_wiring") is True
              and _nested(seq60, "derived", "primitive_ready_for_carrier_wiring") is True
              and _nested(seq61, "derived", "primitive_ready_for_carrier_wiring") is True
              and _nested(seq62, "derived", "primitive_ready_for_carrier_wiring") is True
              and _nested(seq59, "verdict", "decode_speed_path_enabled") is False
              and _nested(seq60, "verdict", "decode_speed_path_enabled") is False
              and _nested(seq61, "verdict", "decode_speed_path_enabled") is False
              and _nested(seq62, "verdict", "decode_speed_path_enabled") is False
          ),
          "detail": {
              "q4_device_q8_ready": _nested(
                  seq59, "derived", "primitive_ready_for_carrier_wiring"),
              "q6_device_q8_ready": _nested(
                  seq60, "derived", "primitive_ready_for_carrier_wiring"),
              "ffn_tail_ready": _nested(
                  seq61, "derived", "primitive_ready_for_carrier_wiring"),
              "q6_defer_ready": _nested(
                  seq62, "derived", "primitive_ready_for_carrier_wiring"),
          },
      },
      {
          "name": "preconv_carrier_requires_bundle_not_qkv_only",
          "pass": (
              _nested(seq71, "derived", "linear_preconv_gap_blocks_existing_carrier")
                  is True
              and _nested(seq72, "derived", "primitive_ready_for_preconv_wiring")
                  is True
              and _nested(seq73, "derived", "qkv_only_preconv_wiring_promotable")
                  is False
              and _nested(seq73, "derived", "shared_device_q8_preconv_bundle_required")
                  is True
          ),
          "detail": {
              "seq71_full_carrier_wirable_with_existing_primitives": _nested(
                  seq71, "verdict", "full_carrier_wirable_with_existing_primitives"),
              "seq72_primitive_ready": _nested(
                  seq72, "derived", "primitive_ready_for_preconv_wiring"),
              "seq73_qkv_only_promotable": _nested(
                  seq73, "derived", "qkv_only_preconv_wiring_promotable"),
              "seq73_shared_device_q8_bundle_required": _nested(
                  seq73, "derived", "shared_device_q8_preconv_bundle_required"),
          },
      },
      {
          "name": "existing_shared_q8_preconv_speed_route_is_closed",
          "pass": (
              _nested(seq77, "verdict", "shared_q8_profile_closes_speed_route")
                  is True
              and _num(seq77_derived.get("tps_delta_pct_vs_current_source_baseline"))
                  < -0.5
              and _nested(seq90, "derived", "required_checks_passed") is True
              and seq90_derived.get("component_delta_floor_covering") is False
              and seq90_derived.get("component_qkv_conv_non_growth_or_bounded")
                  is False
          ),
          "detail": {
              "seq77_tps_delta_pct_vs_current_source_baseline": seq77_derived.get(
                  "tps_delta_pct_vs_current_source_baseline"),
              "seq77_linear_preconv_qkv_conv_delta_ms_per_token": seq77_derived.get(
                  "linear_preconv_qkv_conv_ms_per_token"),
              "seq90_component_delta_floor_covering": seq90_derived.get(
                  "component_delta_floor_covering"),
              "seq90_component_estimated_delta_ms_per_token": seq90_derived.get(
                  "component_estimated_delta_ms_per_token"),
          },
      },
      {
          "name": "live_source_still_has_host_vector_boundaries",
          "pass": source["host_boundaries_present"],
          "detail": source["host_boundary_checks"],
      },
      {
          "name": "carrier_primitives_are_visible_in_current_header",
          "pass": source["primitive_apis_present"],
          "detail": source["primitive_api_checks"],
      },
      {
          "name": "full_hidden_state_carrier_contract_not_present_yet",
          "pass": source["full_carrier_contract_absent"],
          "detail": source["carrier_absence_checks"],
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  contract_requirements = [
      "Add one O(1) resident hidden-state carrier contract that owns layer-input, attention-norm, attention-residual, FFN-norm/router-input, selected/shared down, and layer-output resident handles across the layer loop.",
      "Keep host vectors as oracle/debug shadows until a paired distribution gate passes; do not treat component numeric agreement as token correctness.",
      "Linear preconv must consume the resident attention-norm handle through a bundled shared-device-Q8 path; qkv-only wiring and the seq77 shared-Q8 decode row are not admissible speed evidence.",
      "Selected/shared FFN and FFN tail must consume resident FFN-norm, down-output, and attention-residual handles before any tail-drain removal speed row is admissible.",
      "Do not launch a token-emitting decode row from this gate; the next unit is a default-off source scaffold/contract gate with compile/source evidence first.",
  ]

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "accepted": _rel(args.accepted),
          "rejected": _rel(args.rejected),
          "decode_source": {
              "path": _rel(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "gpu_header": {
              "path": _rel(args.gpu_header),
              "sha256": _sha256(args.gpu_header),
          },
          "seq54_resident_hidden_carrier_gate": _rel(args.seq54),
          "seq59_device_q8_q4_contract": _rel(args.seq59),
          "seq60_device_q8_q6_contract": _rel(args.seq60),
          "seq61_ffn_tail_resident_input_contract": _rel(args.seq61),
          "seq62_q6_selected_shared_defer_contract": _rel(args.seq62),
          "seq71_linear_preconv_carrier_gap": _rel(args.seq71),
          "seq72_linear_preconv_carrier_primitive": _rel(args.seq72),
          "seq73_linear_preconv_carrier_bundle": _rel(args.seq73),
          "seq77_shared_q8_preconv_profile": _rel(args.seq77),
          "seq90_qkv_conv_root_probe": _rel(args.seq90),
          "seq116_moe_route_gate": _rel(args.seq116),
      },
      "frontier": frontier_state,
      "carrier_contract_summary": {
          "resident_route_id": (
              None if resident_route is None else resident_route.get("id")),
          "resident_route_rank": (
              None if resident_route is None else resident_route.get("rank")),
          "tail_drain_elimination_clears_floor": seq54_drain.get(
              "tail_drain_elimination_clears_floor"),
          "projected_tps_without_tail_growth": seq54_drain.get(
              "projected_tps_without_tail_growth"),
          "selected_down_wait_saved_ms_per_token": seq54_drain.get(
              "selected_down_wait_saved_ms_per_token"),
          "ffn_tail_growth_ms_per_token": seq54_drain.get(
              "ffn_tail_growth_ms_per_token"),
          "host_boundaries_present": source["host_boundaries_present"],
          "primitive_apis_present": source["primitive_apis_present"],
          "full_carrier_contract_absent": source["full_carrier_contract_absent"],
          "shared_q8_preconv_tps_delta_pct_vs_baseline": seq77_derived.get(
              "tps_delta_pct_vs_current_source_baseline"),
          "qkv_conv_root_delta_ms_per_token": seq90_derived.get(
              "component_estimated_delta_ms_per_token"),
      },
      "contract_requirements": contract_requirements,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "source_cut_allowed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": "authorize_resident_hidden_state_carrier_source_contract",
      "selected_next_route": "resident_hidden_state_carrier_source_scaffold_gate",
      "next_route_reason": (
          "Seq116 closed the current MoE/down-tail board, while seq54 shows "
          "tail-drain elimination can clear the floor only if hidden state stops "
          "round-tripping through host vectors. The needed Q4/Q6 device-Q8, "
          "preconv, FFN-tail, and Q6 defer primitives exist, but local handoffs, "
          "qkv-only preconv wiring, and the existing shared-Q8 preconv speed path "
          "are closed. The next admissible unit is a default-off source scaffold "
          "for the resident hidden-state carrier contract."
      ),
      "next_action": (
          "Add a source-only resident hidden-state carrier scaffold that threads "
          "resident handles through layer input RMSNorm, attention residual, FFN "
          "norm/router input, selected/shared FFN, FFN tail, and next-layer "
          "RMSNorm boundaries. Keep it default-off and compile/source-gated before "
          "any token-emitting decode row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Resident Hidden-State Carrier Contract Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- source_cut_allowed: `{str(metrics['source_cut_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- component_probe_allowed: `{str(metrics['component_probe_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      metrics["next_route_reason"],
      "",
      "## Contract Requirements",
      "",
  ]
  summary.extend(f"- {item}" for item in metrics["contract_requirements"])
  summary.extend([
      "",
      "## Next",
      "",
      metrics["next_action"],
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--gpu-header", type=Path, default=DEFAULT_GPU_HEADER)
  parser.add_argument("--seq54", type=Path, default=DEFAULT_SEQ54)
  parser.add_argument("--seq59", type=Path, default=DEFAULT_SEQ59)
  parser.add_argument("--seq60", type=Path, default=DEFAULT_SEQ60)
  parser.add_argument("--seq61", type=Path, default=DEFAULT_SEQ61)
  parser.add_argument("--seq62", type=Path, default=DEFAULT_SEQ62)
  parser.add_argument("--seq71", type=Path, default=DEFAULT_SEQ71)
  parser.add_argument("--seq72", type=Path, default=DEFAULT_SEQ72)
  parser.add_argument("--seq73", type=Path, default=DEFAULT_SEQ73)
  parser.add_argument("--seq77", type=Path, default=DEFAULT_SEQ77)
  parser.add_argument("--seq90", type=Path, default=DEFAULT_SEQ90)
  parser.add_argument("--seq116", type=Path, default=DEFAULT_SEQ116)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
