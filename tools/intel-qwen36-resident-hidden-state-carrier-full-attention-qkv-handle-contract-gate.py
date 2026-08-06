#!/usr/bin/env python3
"""Gate the full-attention QK/V resident input-handle contract.

This is source/design route-control evidence only. It consumes the seq135 route
gate and audits the current generated carrier loop shape before any source edit,
target compile, or token row.
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
SCHEMA_VERSION = (
    "intel-qwen36-resident-hidden-state-carrier-full-attention-qkv-handle-"
    "contract-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_ACCEPTED = ACTIVE / "accepted-cuts.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ135 = ROOT / "output/post-layer-output-handle-loop-route-gate-20260707Tseq135Z/metrics.json"
DEFAULT_GENERATED_CPP = (
    ROOT
    / "output/resident-hidden-state-carrier-layer-output-handle-loop-full-core-parity-generate-only-20260707Tseq131Z/r2_gpu_decode_smoke.cpp"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-qkv-handle-contract-gate-20260707Tseq136Z"
)


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


def _line_of(text: str, pattern: str, *, regex: bool = True) -> int | None:
  if regex:
    match = re.search(pattern, text, flags=re.S)
    if match is None:
      return None
    return text.count("\n", 0, match.start()) + 1
  index = text.find(pattern)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def _present(text: str, label: str, pattern: str, *,
             regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "present": line is not None, "line": line}


def _absent(text: str, label: str, pattern: str, *,
            regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "absent": line is None, "line": line}


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
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
  }


def _stage_gap(frontier: dict[str, Any], stage: str) -> float:
  rows = _nested(frontier, "goal_budget", "stage_kernel_gap_estimates_ms_per_token")
  if not isinstance(rows, list):
    return 0.0
  for row in rows:
    if isinstance(row, dict) and row.get("stage") == stage:
      return _num(row.get("gap_ms_per_token"))
  return 0.0


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


def _source_shape(generated_cpp: str, engine_text: str, all_text: str) -> dict[str, Any]:
  present = [
      _present(
          generated_cpp,
          "layer_input_rmsnorm_exposes_attention_norm_handle",
          r"struct LayerInputRmsNormRun\s*\{[^}]*attn_norm_handle",
      ),
      _present(
          generated_cpp,
          "carrier_captures_attention_norm_handle",
          r"CaptureAttentionNorm\(\s*rms_gpu\.attn_norm_handle\s*\)",
      ),
      _present(
          generated_cpp,
          "full_attention_qk_front_uses_host_attention_norm_vector",
          r"FullAttentionQkRun RunGpuFullAttentionQkFront\([^)]*"
          r"const std::vector<float>& attn_norm",
      ),
      _present(
          generated_cpp,
          "full_attention_qk_call_passes_host_attention_norm",
          r"RunGpuFullAttentionQkFront\(\s*model_path,\s*\*t\.q_tensor,"
          r"\s*\*t\.k_tensor,\s*rms_gpu\.attn_norm",
      ),
      _present(
          generated_cpp,
          "full_attention_qk_quantizes_host_attention_norm",
          r"RunGpuFullAttentionQkFront\([^}]*"
          r"QuantizeQ8KInputPlanes\(attn_norm\)",
      ),
      _present(
          generated_cpp,
          "full_attention_v_any_uses_host_attention_norm_vector",
          r"FullAttentionVQ6Run RunGpuFullAttentionVAny\([^)]*"
          r"const std::vector<float>& attn_norm",
      ),
      _present(
          generated_cpp,
          "full_attention_v_call_passes_host_attention_norm",
          r"RunGpuFullAttentionVAny\(\s*model_path,\s*\*t\.v_tensor,"
          r"\s*rms_gpu\.attn_norm",
      ),
      _present(
          generated_cpp,
          "resident_v_q6_helper_still_quantizes_host_attention_norm",
          r"RunGpuFullAttentionVQ6Resident[\s\S]*?"
          r"QuantizeQ8KInputPlanes\(attn_norm\)",
      ),
      _present(
          engine_text,
          "q4_projection_input_handle_api_exists",
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8\(",
      ),
      _present(
          engine_text,
          "q6_projection_input_handle_api_exists",
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6K\(",
      ),
      _present(
          engine_text,
          "rmsnorm_outputs_resident_handle",
          r"struct GpuRmsNormRun\s*\{[^}]*output_handle",
      ),
  ]
  absent = [
      _absent(
          all_text,
          "no_full_attention_qkv_handle_env_gate_yet",
          r"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_QKV_HANDLE",
      ),
      _absent(
          generated_cpp,
          "no_qk_front_input_handle_helper_yet",
          r"RunGpuFullAttentionQkFrontFromInputHandle",
      ),
      _absent(
          generated_cpp,
          "no_v_any_input_handle_helper_yet",
          r"RunGpuFullAttentionVAnyFromInputHandle",
      ),
      _absent(
          generated_cpp,
          "no_v_q6_input_handle_helper_yet",
          r"RunGpuFullAttentionVQ6ResidentInputHandle",
      ),
      _absent(
          generated_cpp,
          "qk_call_does_not_pass_attention_norm_handle_yet",
          r"RunGpuFullAttentionQkFront\([^;]*attn_norm_handle",
      ),
      _absent(
          generated_cpp,
          "v_call_does_not_pass_attention_norm_handle_yet",
          r"RunGpuFullAttentionVAny\([^;]*attn_norm_handle",
      ),
  ]
  return {
      "present_checks": present,
      "absent_checks": absent,
      "current_host_boundary_present": all(row["present"] for row in present[:8]),
      "required_primitives_present": all(row["present"] for row in present[8:]),
      "new_qkv_handle_wiring_absent": all(row["absent"] for row in absent),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier_json = _load_json(args.frontier)
  frontier = _frontier_state(frontier_json)
  routes = _load_json(args.routes)
  accepted = _load_json(args.accepted)
  rejected = _load_json(args.rejected)
  seq135 = _load_json(args.seq135)
  generated_cpp = _load_text(args.generated_cpp)
  decode_source = _load_text(args.decode_source)
  engine_text = "\n".join([
      _load_text(args.engine_source),
      _load_text(args.engine_header),
  ])
  all_text = "\n".join([generated_cpp, decode_source, engine_text])

  source_shape = _source_shape(generated_cpp, engine_text, all_text)
  accepted_ids = _accepted_ids(accepted)
  rejected_names = _rejected_names(rejected)
  full_core_gap = _stage_gap(frontier_json, "full_core")
  attention_front_gap = _stage_gap(frontier_json, "attention_front")
  floor_gap = frontier["floor_gap_ms_per_token"]

  required_accepted = {
      "r2_gpu_decode_resident_packed_q4_weight_store",
      "r2_gpu_decode_resident_raw_q6_weight_store",
      "r2_gpu_decode_resident_linear_q6_qkv_weight_store",
      "full_core_resident_norm_handoff",
      "attention_front_handoff_linear",
  }
  required_closed = {
      "current_full_core_attention_front_kernel_algorithm_board",
      "current_resident_hidden_state_carrier_layer_output_loop_full_core_parity_speed_shape",
      "gpu_lm_head_rmsnorm_q6_handoff",
      "gpu_lm_head_device_topk",
  }
  missing_accepted = sorted(required_accepted - accepted_ids)
  missing_closed = sorted(required_closed - rejected_names)

  contract = {
      "decode_probe_allowed": False,
      "token_row_allowed": False,
      "source_cut_allowed": True,
      "target_compile_required_before_decode": True,
      "initial_enablement_scope": (
          "full-attention Q/K/V projection input handles only: consume the "
          "resident attention-norm handle from layer-input RMSNorm, keep Q/K/V "
          "projection outputs host-visible for q/k RMSNorm, rope, K/V history, "
          "diagnostics, and the existing full-core paths"
      ),
      "source_gate_must_add": [
          "a default-off full-attention QK/V resident-input gate",
          "Q/K Q4 projection helpers that consume an attention-norm input handle",
          "V Q4/Q6 projection helpers that consume an attention-norm input handle",
          "generated manifest evidence with the carrier layer-output loop and full-core parity still on",
          "guards that preserve host Q/K/V outputs until a later core/history handle contract exists",
      ],
      "source_gate_must_not_do": [
          "launch a token row",
          "change full-attention core/gate math",
          "skip Q/K/V output readback or K/V history updates",
          "skip final LM-head readback",
          "claim speed before correctness, target compile, confirm, and paired distribution",
      ],
  }

  checks = [
      {
          "name": "seq135_selected_this_contract_gate",
          "pass": (
              seq135.get("required_checks_passed") is True
              and seq135.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_qkv_handle_contract_gate"
              and _has_switch(
                  routes,
                  "select_full_attention_qkv_input_handle_contract_gate",
                  135,
              )
          ),
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier["current_best_tps"] < frontier["floor_tps"]
              and frontier["wall_ms_per_token"]
              > frontier["floor_budget_ms_per_token"]
              > 0.0
              and frontier["review_recorded_for_current_best"] is True
          ),
          "detail": frontier,
      },
      {
          "name": "accepted_resident_projection_primitives_ready",
          "pass": (
              not missing_accepted
              and source_shape["required_primitives_present"]
          ),
          "detail": {
              "missing_accepted": missing_accepted,
              "primitive_checks": source_shape["present_checks"][8:],
          },
      },
      {
          "name": "current_source_keeps_full_attention_qkv_host_boundary",
          "pass": source_shape["current_host_boundary_present"],
          "detail": source_shape["present_checks"][:8],
      },
      {
          "name": "full_attention_qkv_handle_wiring_absent_before_source_gate",
          "pass": source_shape["new_qkv_handle_wiring_absent"],
          "detail": source_shape["absent_checks"],
      },
      {
          "name": "full_attention_boundary_is_floor_sized_but_contract_only",
          "pass": (
              max(full_core_gap, attention_front_gap) > floor_gap
              and not missing_closed
          ),
          "detail": {
              "full_core_gap_ms_per_token": full_core_gap,
              "attention_front_gap_ms_per_token": attention_front_gap,
              "floor_gap_ms_per_token": floor_gap,
              "missing_closed_routes": missing_closed,
          },
      },
      {
          "name": "contract_forbids_token_row_until_source_and_compile_gates",
          "pass": (
              contract["source_cut_allowed"] is True
              and contract["decode_probe_allowed"] is False
              and contract["target_compile_required_before_decode"] is True
          ),
          "detail": contract,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "accepted": _rel(args.accepted),
          "rejected": _rel(args.rejected),
          "seq135": _rel(args.seq135),
          "generated_cpp": {
              "path": _rel(args.generated_cpp),
              "sha256": _sha256(args.generated_cpp),
          },
          "decode_source": {
              "path": _rel(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "engine_source": {
              "path": _rel(args.engine_source),
              "sha256": _sha256(args.engine_source),
          },
          "engine_header": {
              "path": _rel(args.engine_header),
              "sha256": _sha256(args.engine_header),
          },
      },
      "frontier": frontier,
      "source_shape": source_shape,
      "contract": contract,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_qkv_handle_contract_select_source_gate"
          if required_checks_passed
          else "full_attention_qkv_handle_contract_incomplete"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_full_attention_qkv_handle_source_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_full_attention_qkv_handle_contract_review"
      ),
      "next_action": (
          "Add default-off source wiring for full-attention QK/V resident "
          "attention-norm input handles, generate source-only evidence, and do "
          "not launch a token row until a target compile gate passes."
          if required_checks_passed
          else "Resolve the contract evidence before any source edit or token row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Full-Attention QK/V Handle Contract Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      "## Summary",
      "",
      (
          "The current carrier loop still passes host attention_norm into "
          "full-attention QK/V projection helpers. Existing input-handle "
          "projection primitives are present, so the next step is a no-token "
          "source gate."
      ),
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq135", type=Path, default=DEFAULT_SEQ135)
  parser.add_argument("--generated-cpp", type=Path, default=DEFAULT_GENERATED_CPP)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_ENGINE_HEADER)
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
