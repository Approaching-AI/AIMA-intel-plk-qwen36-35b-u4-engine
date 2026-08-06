#!/usr/bin/env python3
"""Audit the carrier selected/shared FFN tail source path.

This is source/generate-only evidence. It verifies that the carrier path uses
resident FFN norm/residual handles for selected/shared FFN tail inputs without
turning on the closed standalone FFN-tail env route or launching decode.
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
    "intel-qwen36-resident-hidden-state-carrier-selected-shared-tail-source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ120 = ROOT / "output/resident-hidden-state-carrier-preconv-bundle-target-compile-gate-20260707Tseq120Z/metrics.json"
DEFAULT_GENERATE_DIR = ROOT / "output/resident-hidden-state-carrier-selected-shared-tail-generate-only-20260707Tseq121Z"
DEFAULT_OUT_DIR = ROOT / "output/resident-hidden-state-carrier-selected-shared-tail-source-gate-20260707Tseq121Z"


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


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "soft_reflection_breached": no_progress.get("soft_reflection_breached"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
  }


def _source_checks(text: str, *, require_python_manifest: bool) -> dict[str, Any]:
  carrier_tail_checks = [
      _present_check(
          text,
          "selected_shared_tail_global_exists",
          "g_decode_resident_hidden_state_carrier_selected_shared_tail",
      ),
      _present_check(
          text,
          "selected_shared_tail_env_gate_exists",
          'std::getenv("IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL")',
      ),
      _present_check(
          text,
          "selected_shared_tail_run_command_env_forwarded",
          '"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL"',
      ),
      _present_check(
          text,
          "selected_shared_tail_smoke_json_field_exists",
          "resident_hidden_state_carrier_selected_shared_tail_enabled",
      ),
      _present_check(
          text,
          "carrier_ffn_norm_handle_is_selected_tail_input",
          "g_decode_resident_hidden_state_carrier.ffn_norm_handle",
      ),
      _present_check(
          text,
          "carrier_attention_residual_handle_is_tail_residual",
          "g_decode_resident_hidden_state_carrier.attention_residual_handle",
      ),
      _present_check(
          text,
          "carrier_selected_shared_tail_guard_exists",
          "use_carrier_selected_shared_tail",
      ),
      _present_check(
          text,
          "carrier_resident_tail_guard_exists",
          "use_carrier_resident_tail",
      ),
      _present_check(
          text,
          "carrier_tail_enables_resident_input_tail_without_closed_env",
          "g_decode_ffn_tail_resident_input ||",
      ),
      _present_check(
          text,
          "resident_input_tail_primitive_present",
          "RunFfnTailFromDownHandlesResidentInputs",
      ),
      _present_check(
          text,
          "prev_layer_output_handle_retained_for_carrier",
          "g_decode_resident_hidden_state_carrier_selected_shared_tail;",
      ),
  ]
  default_off_checks = [
      _absent_check(
          text,
          "no_cli_variant_flag_added",
          "--resident-hidden-state-carrier-selected-shared-tail",
      ),
      _present_check(
          text,
          "closed_tail_env_still_separate",
          "IQ36_FFN_TAIL_RESIDENT_INPUT",
      ),
  ]
  if require_python_manifest:
    carrier_tail_checks.append(
        _present_check(
            text,
            "python_manifest_records_selected_shared_tail",
            '"resident_hidden_state_carrier_selected_shared_tail"',
        )
    )
  prerequisite_checks = []
  if require_python_manifest:
    prerequisite_checks = [
        _present_check(
            text,
            "selected_shared_tail_requires_hidden_state_carrier",
            'not os.environ.get(\n      "IQ36_RESIDENT_HIDDEN_STATE_CARRIER"',
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_preconv_bundle",
            "not args.resident_hidden_state_carrier_preconv_bundle",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_shared_q4_runner",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL requires --shared-q4-runner",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_gpu_router",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL requires --gpu-router",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_resident_q4_weights",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL requires --resident-q4-weights",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_selected_q4_experts",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL requires --resident-selected-q4-experts",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_selected_q6_experts",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL requires --resident-selected-q6-experts",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_selected_q6_rowstripe",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL requires --resident-selected-q6-rowstripe",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_shared_q6_down",
            "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_SELECTED_SHARED_TAIL requires --resident-shared-q6-down",
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_q4_gateup_combined",
            'not os.environ.get("IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED")',
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_q4_down_combined",
            'not os.environ.get("IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED")',
        ),
        _present_check(
            text,
            "selected_shared_tail_requires_q6_down_combined",
            'not os.environ.get("IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED")',
        ),
    ]
  return {
      "carrier_tail_checks": carrier_tail_checks,
      "default_off_checks": default_off_checks,
      "prerequisite_checks": prerequisite_checks,
      "carrier_tail_source_present": _all_present(carrier_tail_checks),
      "default_off_contract_present": (
          _all_absent([row for row in default_off_checks if "absent" in row])
          and _all_present([row for row in default_off_checks if "present" in row])
      ),
      "prerequisites_present": (
          True if not prerequisite_checks else _all_present(prerequisite_checks)
      ),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq120 = _load_json(args.seq120)
  decode_text = _load_text(args.decode_source)
  generate_result_path = args.generate_dir / "result.json"
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generate_result = _load_json(generate_result_path)
  generated_text = _load_text(generated_cpp_path)
  smoke_path = args.generate_dir / "smoke.json"

  frontier_state = _frontier_state(frontier)
  source = _source_checks(decode_text, require_python_manifest=True)
  generated = _source_checks(generated_text, require_python_manifest=False)

  generate_manifest_checks = {
      "generate_only": generate_result.get("generate_only") is True,
      "resident_hidden_state_carrier": (
          generate_result.get("resident_hidden_state_carrier") is True),
      "resident_hidden_state_carrier_preconv_bundle": (
          generate_result.get("resident_hidden_state_carrier_preconv_bundle")
          is True),
      "resident_hidden_state_carrier_selected_shared_tail": (
          generate_result.get(
              "resident_hidden_state_carrier_selected_shared_tail") is True),
      "linear_preconv_shared_q8_false": (
          generate_result.get("linear_preconv_shared_q8") is False),
      "ffn_tail_resident_input_false": (
          generate_result.get("ffn_tail_resident_input") is False),
      "gpu_router": generate_result.get("gpu_router") is True,
      "resident_selected_q4_experts": (
          generate_result.get("resident_selected_q4_experts") is True),
      "resident_selected_q6_experts": (
          generate_result.get("resident_selected_q6_experts") is True),
      "resident_selected_q6_rowstripe": (
          generate_result.get("resident_selected_q6_rowstripe") is True),
      "resident_shared_q6_down": (
          generate_result.get("resident_shared_q6_down") is True),
      "selected_shared_q4_gateup_combined": (
          generate_result.get("selected_shared_q4_gateup_combined") is True),
      "selected_shared_q4_down_combined": (
          generate_result.get("selected_shared_q4_down_combined") is True),
      "selected_shared_q6_down_combined": (
          generate_result.get("selected_shared_q6_down_combined") is True),
      "no_smoke_json": not smoke_path.exists(),
  }

  checks = [
      {
          "name": "seq120_selected_selected_shared_tail_source_gate",
          "pass": (
              seq120.get("required_checks_passed") is True
              and seq120.get("selected_next_route")
                  == "resident_hidden_state_carrier_selected_shared_tail_source_gate"
              and _has_switch(
                  routes,
                  "accept_preconv_bundle_compile_switch_to_selected_shared_tail_source_gate",
                  120,
              )
          ),
          "detail": {
              "seq120_disposition": seq120.get("disposition"),
              "seq120_selected_next_route": seq120.get("selected_next_route"),
          },
      },
      {
          "name": "source_selected_shared_tail_contract_present",
          "pass": (
              source["carrier_tail_source_present"]
              and source["default_off_contract_present"]
              and source["prerequisites_present"]
          ),
          "detail": source,
      },
      {
          "name": "generated_cpp_selected_shared_tail_contract_present",
          "pass": (
              generated["carrier_tail_source_present"]
              and generated["default_off_contract_present"]
          ),
          "detail": generated,
      },
      {
          "name": "generate_only_manifest_is_carrier_tail_not_decode_row",
          "pass": all(generate_manifest_checks.values()),
          "detail": generate_manifest_checks,
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": frontier_state["current_best_tps"] < frontier_state["floor_tps"],
          "detail": frontier_state,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "seq120_compile_gate": _rel(args.seq120),
          "generate_only_result": _rel(generate_result_path),
          "generated_cpp": _rel(generated_cpp_path),
          "generated_cpp_sha256": _sha256(generated_cpp_path),
      },
      "frontier": frontier_state,
      "source": source,
      "generated": generated,
      "generate_manifest_checks": generate_manifest_checks,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "source_cut_allowed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_resident_hidden_state_carrier_selected_shared_tail_source_wiring"
          if required_checks_passed
          else "reject_resident_hidden_state_carrier_selected_shared_tail_source_wiring"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_selected_shared_tail_target_compile_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_selected_shared_tail_source_fix_gate"
      ),
      "next_route_reason": (
          "The carrier selected/shared tail source path is env-gated separately "
          "from the closed standalone FFN-tail route, uses the carrier FFN norm "
          "and residual handles, and generate-only evidence keeps "
          "ffn_tail_resident_input=false with no decode row. The next admissible "
          "evidence is target compile only."
          if required_checks_passed
          else "The selected/shared tail carrier source contract is incomplete. "
               "Fix source/generate-only evidence before any compile or decode row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Resident Hidden-State Carrier Selected/Shared Tail Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- target_compile_required_before_decode: `{str(metrics['target_compile_required_before_decode']).lower()}`",
      "",
      "## Summary",
      "",
      metrics["next_route_reason"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq120", type=Path, default=DEFAULT_SEQ120)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
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
