#!/usr/bin/env python3
"""Gate the full-attention core/history resident-boundary contract.

This is no-token route-control evidence. It consumes the post-QK/V route gate
and audits the current source before any core/history source edit, target
compile, or token row.
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
    "intel-qwen36-resident-hidden-state-carrier-full-attention-core-history-"
    "contract-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ143 = ROOT / "output/post-full-attention-qkv-handle-route-gate-20260707Tseq143Z/metrics.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-contract-gate-20260707Tseq144Z"
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


def _line_of(text: str, pattern: str) -> int | None:
  match = re.search(pattern, text, flags=re.S)
  if match is None:
    return None
  return text.count("\n", 0, match.start()) + 1


def _present(text: str, label: str, pattern: str) -> dict[str, Any]:
  line = _line_of(text, pattern)
  return {"label": label, "present": line is not None, "line": line}


def _absent(text: str, label: str, pattern: str) -> dict[str, Any]:
  line = _line_of(text, pattern)
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
      "overhead_only_ceiling_tok_s": _num(
          verdict.get("overhead_only_ceiling_tok_s")),
  }


def _stage_gap(frontier: dict[str, Any], stage: str) -> float:
  rows = _nested(frontier, "goal_budget", "stage_kernel_gap_estimates_ms_per_token")
  if not isinstance(rows, list):
    return 0.0
  for row in rows:
    if isinstance(row, dict) and row.get("stage") == stage:
      return _num(row.get("gap_ms_per_token"))
  return 0.0


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


def _source_shape(source: str, engine_text: str) -> dict[str, Any]:
  present = [
      _present(
          source,
          "qk_outputs_still_host_split_for_q_norm",
          r"const auto q_split = SplitFullAttentionQ\(qk_gpu\.q_full\)",
      ),
      _present(
          source,
          "q_norm_still_cpu_from_host_q_raw",
          r"ApplyRepeatedRmsNormFull\(\s*q_split\.q_raw",
      ),
      _present(
          source,
          "k_norm_still_cpu_from_host_k_raw",
          r"ApplyRepeatedRmsNormFull\(\s*qk_gpu\.k_raw",
      ),
      _present(
          source,
          "rope_still_cpu_from_host_qk_normed",
          r"run_qwen36_full_attention_rope\(\s*q_normed,\s*k_normed",
      ),
      _present(
          source,
          "k_history_still_host_flatten",
          r"FlattenFullAttentionHistory\(state->full_k\[layer\],\s*rope\.k_rope\)",
      ),
      _present(
          source,
          "v_history_still_host_flatten",
          r"FlattenFullAttentionHistory\(state->full_v\[layer\],\s*v_gpu\.v\)",
      ),
      _present(
          source,
          "k_history_update_still_host_vector",
          r"state->full_k\[layer\]\.push_back\(k_rope_for_state\)",
      ),
      _present(
          source,
          "v_history_update_still_host_vector",
          r"state->full_v\[layer\]\.push_back\(v_for_state\)",
      ),
      _present(
          source,
          "full_core_handoff_still_takes_host_vectors",
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\(\s*"
          r"rope\.q_rope,\s*k_history_flat,\s*v_history_flat,\s*qk_gpu\.q_full",
      ),
      _present(
          engine_text,
          "full_core_handoff_api_is_host_vector_api",
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\("
          r"[\s\S]*const std::vector<float>& q_rope",
      ),
  ]
  absent = [
      _absent(
          source,
          "no_core_history_env_gate_yet",
          r"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_FULL_ATTENTION_CORE_HISTORY_HANDLE",
      ),
      _absent(
          source,
          "no_device_qk_norm_rope_helper_yet",
          r"RunGpuFullAttentionQkNormRope.*Handle",
      ),
      _absent(
          source,
          "no_resident_kv_history_handle_bank_yet",
          r"(ResidentFullAttentionHistory|FullAttentionHistoryHandle|"
          r"full_attention_kv_history_handle)",
      ),
      _absent(
          engine_text,
          "no_full_core_resident_history_input_handle_api_yet",
          r"RunFullAttentionCoreGateThen.*History.*Handle",
      ),
  ]
  return {
      "present_checks": present,
      "absent_checks": absent,
      "host_core_history_boundary_present": all(row["present"] for row in present),
      "new_core_history_wiring_absent": all(row["absent"] for row in absent),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier_json = _load_json(args.frontier)
  frontier = _frontier_state(frontier_json)
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq143 = _load_json(args.seq143)
  source = _load_text(args.decode_source)
  engine_text = "\n".join([
      _load_text(args.engine_source),
      _load_text(args.engine_header),
  ])
  source_shape = _source_shape(source, engine_text)
  rejected_names = _rejected_names(rejected)
  floor_gap = frontier["floor_gap_ms_per_token"]
  full_core_gap = _stage_gap(frontier_json, "full_core")
  attention_front_gap = _stage_gap(frontier_json, "attention_front")

  required_closed = {
      "current_full_core_attention_front_kernel_algorithm_board",
      "gpu_full_attention_state_resident_history",
      "gpu_full_attention_flat_history_cache",
      "current_resident_hidden_state_carrier_full_attention_qkv_handle_speed_shape",
  }
  missing_closed = sorted(required_closed - rejected_names)
  contract = {
      "decode_probe_allowed": False,
      "token_row_allowed": False,
      "source_cut_allowed": True,
      "target_compile_required_before_decode": True,
      "initial_enablement_scope": (
          "Define a resident full-attention core/history boundary only. The "
          "contract may add default-off scaffolding for device Q/K normalization, "
          "RoPE, resident K/V history append/flatten, and full-core host-vector "
          "input replacement, but it must not change full-core math or launch a "
          "token row."
      ),
      "source_gate_must_add": [
          "a default-off env gate for the full-attention core/history resident boundary",
          "manifest/source evidence that Q/K/V input-handle speed shape remains closed",
          "a resident K/V history ownership plan that is not the closed K/V-history-only route",
          "device-side Q/K normalization plus RoPE ownership or an explicit component-proof precondition",
          "full-core input-handle API shape for q_rope, K/V history, and q_full",
          "fallback guards preserving existing host-vector full-core handoff and diagnostics",
      ],
      "source_gate_must_not_do": [
          "launch a token row",
          "reopen standalone resident K/V history or flat-history cache",
          "change full-core softmax/gate math",
          "skip paired distribution correctness before any speed row",
          "claim speed without a new best, confirm, and paired distribution evidence",
      ],
  }

  checks = [
      {
          "name": "seq143_selected_this_contract_gate",
          "pass": (
              seq143.get("required_checks_passed") is True
              and seq143.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_core_history_contract_gate"
              and _has_switch(
                  routes,
                  "select_full_attention_core_history_contract_gate",
                  143,
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
          "name": "current_source_keeps_core_history_host_boundary",
          "pass": source_shape["host_core_history_boundary_present"],
          "detail": source_shape["present_checks"],
      },
      {
          "name": "core_history_wiring_absent_before_source_gate",
          "pass": source_shape["new_core_history_wiring_absent"],
          "detail": source_shape["absent_checks"],
      },
      {
          "name": "core_history_boundary_is_floor_sized_but_not_algorithm_board",
          "pass": (
              full_core_gap > floor_gap
              and attention_front_gap > floor_gap
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
  selected_next_route = (
      "resident_hidden_state_carrier_full_attention_core_history_source_gate"
      if required_checks_passed
      else "resident_hidden_state_carrier_full_attention_core_history_contract_review"
  )
  disposition = (
      "accept_full_attention_core_history_contract_select_source_gate"
      if required_checks_passed
      else "full_attention_core_history_contract_gate_incomplete"
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq143_route_gate": _rel(args.seq143),
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
      "component_probe_allowed": False,
      "source_contract_gate_allowed": False,
      "source_gate_allowed": required_checks_passed,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next_route,
      "next_action": (
          "Add source/generate-only evidence for the default-off full-attention "
          "core/history resident boundary. Do not launch a token row until a "
          "target compile gate passes."
          if required_checks_passed
          else "Fix failed contract checks before any source edit."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      "# Resident Hidden-State Carrier Full-Attention Core/History Contract Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      metrics["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq143", type=Path, default=DEFAULT_SEQ143)
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
