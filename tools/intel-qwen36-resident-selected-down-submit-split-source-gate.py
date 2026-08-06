#!/usr/bin/env python3
"""Audit selected-down submit/finish split source wiring.

This is source/generate-only evidence. It verifies the default-off split of the
selected-down kernel wait wall into enqueue and finish components before target
compile or any token row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-resident-selected-down-submit-split-source-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ159 = (
    ROOT
    / "output/resident-selected-down-wait-drain-route-gate-20260708Tseq159Z"
    / "metrics.json"
)
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-selected-down-submit-split-generate-only-20260708Tseq160Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-selected-down-submit-split-source-gate-20260708Tseq160Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


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
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
  }


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  for row in routes.get("candidate_history", []):
    if (
        isinstance(row, dict)
        and row.get("seq") == seq
        and row.get("disposition") == disposition
    ):
      return True
  return False


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


def _marker_state(text: str, required: list[str],
                  forbidden: list[str] | None = None) -> dict[str, Any]:
  forbidden = forbidden or []
  missing = [marker for marker in required if marker not in text]
  present_forbidden = [marker for marker in forbidden if marker in text]
  return {
      "required": required,
      "missing": missing,
      "forbidden": forbidden,
      "present_forbidden": present_forbidden,
      "pass": not missing and not present_forbidden,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq159 = _load_json(args.seq159)
  result_path = args.generate_dir / "result.json"
  cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  result = _load_json(result_path)
  generated_cpp = _read(cpp_path)
  engine = _read(args.engine_source)
  header = _read(args.engine_header)
  decode = _read(args.decode_source)
  frontier_state = _frontier_state(frontier)

  engine_checks = _marker_state(
      engine,
      [
          "IQ36_SELECTED_DOWN_SUBMIT_SPLIT_PROFILE",
          "bool SelectedDownSubmitSplitProfile()",
          "kernel_enqueue_wall_ns",
          "kernel_finish_wall_ns",
          "const bool split_profile = SelectedDownSubmitSplitProfile();",
          "timing.kernel_enqueue_wall_ns = kernel_enqueue_wall_ns;",
          "timing.kernel_finish_wall_ns = kernel_finish_wall_ns;",
      ],
  )
  header_checks = _marker_state(
      header,
      [
          "std::uint64_t kernel_enqueue_wall_ns = 0;",
          "std::uint64_t kernel_finish_wall_ns = 0;",
      ],
  )
  decode_checks = _marker_state(
      decode,
      [
          "selected_down_submit_split_profile",
          "IQ36_SELECTED_DOWN_SUBMIT_SPLIT_PROFILE",
          "selected_ffn_down_kernel_enqueue_wall_ns",
          "selected_ffn_down_kernel_finish_wall_ns",
          "\\\"down_kernel_enqueue\\\"",
          "\\\"down_kernel_finish\\\"",
          "down_kernel_enqueue_wall_ns",
          "down_kernel_finish_wall_ns",
      ],
      ["--selected-down-submit-split-profile"],
  )
  generated_checks = _marker_state(
      generated_cpp,
      [
          "down_kernel_enqueue_wall_ns",
          "down_kernel_finish_wall_ns",
          "selected_ffn_down_kernel_enqueue_wall_ns",
          "selected_ffn_down_kernel_finish_wall_ns",
          "\\\"down_kernel_enqueue\\\"",
          "\\\"down_kernel_finish\\\"",
          "IQ36_SELECTED_DOWN_SUBMIT_SPLIT_PROFILE",
      ],
  )
  manifest_checks = {
      "generate_only": result.get("generate_only") is True,
      "selected_down_submit_split_profile": (
          result.get("selected_down_submit_split_profile") is True),
      "decode_tokens_eight": result.get("decode_tokens") == 8,
      "accepted_frontier_stack_present": (
          result.get("shared_q4_runner") is True
          and result.get("resident_q4_weights") is True
          and result.get("resident_selected_q4_experts") is True
          and result.get("resident_selected_q6_experts") is True
          and result.get("resident_selected_q6_rowstripe") is True
          and result.get("resident_shared_q6_down") is True
          and result.get("resident_full_attention_v_q6") is True
          and result.get("resident_linear_q6_qkv") is True
          and result.get("resident_linear_state") is True
          and result.get("resident_norm_weights") is True
          and result.get("resident_full_core_attention_front_handoff") is True
          and result.get("gpu_router") is True
          and result.get("gpu_lm_head_q6") is True
      ),
      "no_smoke_json": not (args.generate_dir / "smoke.json").exists(),
  }

  checks = [
      {
          "name": "seq159_selected_source_gate",
          "pass": (
              seq159.get("required_checks_passed") is True
              and seq159.get("selected_next_route")
              == "resident_selected_down_submit_split_source_gate"
              and _has_candidate(
                  routes, 159,
                  "select_resident_selected_down_submit_split_source_gate")
              and _has_switch(
                  routes,
                  "select_resident_selected_down_submit_split_source_gate",
                  159,
              )
          ),
          "detail": {
              "seq159_disposition": seq159.get("disposition"),
              "seq159_selected_next_route": seq159.get("selected_next_route"),
          },
      },
      {
          "name": "engine_split_profile_default_off",
          "pass": engine_checks["pass"],
          "detail": engine_checks,
      },
      {
          "name": "header_timing_fields_present",
          "pass": header_checks["pass"],
          "detail": header_checks,
      },
      {
          "name": "decode_source_records_and_propagates_split_profile",
          "pass": decode_checks["pass"],
          "detail": decode_checks,
      },
      {
          "name": "generated_cpp_has_split_fields",
          "pass": generated_checks["pass"],
          "detail": generated_checks,
      },
      {
          "name": "generate_only_manifest_is_not_token_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": (
              frontier_state["current_best_tps"] < frontier_state["floor_tps"]
              and frontier_state["hard_stall_breached"] is True
              and frontier_state["review_recorded_for_current_best"] is True
          ),
          "detail": frontier_state,
      },
  ]
  required = all(check["pass"] for check in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "disposition": (
          "accept_resident_selected_down_submit_split_source_wiring"
          if required else
          "reject_resident_selected_down_submit_split_source_wiring"
      ),
      "selected_next_route": (
          "resident_selected_down_submit_split_target_compile_gate"
          if required else
          "resident_selected_down_submit_split_source_fix_gate"
      ),
      "next_route_reason": (
          "Default-off selected-down submit/finish split source wiring is "
          "present and generated without a token row. Target compile is required "
          "before a noqueue profile explore can use it."
          if required else
          "The selected-down submit split source contract is incomplete; fix it "
          "before target compile or decode."
      ),
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "seq159_route_gate": _rel(args.seq159),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "engine_header": _rel(args.engine_header),
          "engine_header_sha256": _sha256(args.engine_header),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "generate_only_result": _rel(result_path),
          "generated_cpp": _rel(cpp_path),
          "generated_cpp_sha256": _sha256(cpp_path),
      },
      "frontier": frontier_state,
      "source": {
          "engine": engine_checks,
          "header": header_checks,
          "decode": decode_checks,
          "generated_cpp": generated_checks,
      },
      "generate_manifest_checks": manifest_checks,
      "checks": checks,
  }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": _rel(Path(__file__)),
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": payload["speedup_claims_allowed"],
      "inputs": payload["inputs"],
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [check["name"] for check in payload["checks"] if not check["pass"]]
  lines = [
      "# Resident Selected-Down Submit Split Source Gate",
      "",
      f"- required_checks_passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected_next_route: `{payload['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(payload['decode_probe_allowed']).lower()}`",
      f"- target_compile_required_before_decode: `{str(payload['target_compile_required_before_decode']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      payload["next_route_reason"],
      "",
      "This is source/generate-only evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq159", type=Path, default=DEFAULT_SEQ159)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_HEADER)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(payload, args.out_dir)
  print(json.dumps({
      "required_checks_passed": payload["required_checks_passed"],
      "disposition": payload["disposition"],
      "selected_next_route": payload["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
