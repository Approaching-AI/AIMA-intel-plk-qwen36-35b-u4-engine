#!/usr/bin/env python3
"""Audit the resident hidden-state carrier preconv bundle source path.

This is source/generate-only evidence. It verifies that the preconv bundle is
separate from the closed shared-Q8 speed flag, consumes the carrier's resident
attention-norm handle, and still does not launch a token-emitting decode row.
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
    "intel-qwen36-resident-hidden-state-carrier-preconv-bundle-source-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ73 = ROOT / "output/linear-preconv-carrier-bundle-gate-20260706Tseq73Z/metrics.json"
DEFAULT_SEQ77 = ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z/metrics.json"
DEFAULT_SEQ90 = ROOT / "output/linear-preconv-qkv-conv-root-probe-20260707Tseq90Z/metrics.json"
DEFAULT_SEQ118 = ROOT / "output/resident-hidden-state-carrier-source-scaffold-gate-20260707Tseq118Z/metrics.json"
DEFAULT_GENERATE_DIR = ROOT / "output/resident-hidden-state-carrier-preconv-bundle-generate-only-20260707Tseq119Z"
DEFAULT_OUT_DIR = ROOT / "output/resident-hidden-state-carrier-preconv-bundle-source-gate-20260707Tseq119Z"


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
  preconv_bundle_checks = [
      _present_check(
          text,
          "carrier_preconv_bundle_global_exists",
          "bool g_decode_resident_hidden_state_carrier_preconv_bundle = false",
      ),
      _present_check(
          text,
          "carrier_preconv_bundle_env_gate_exists",
          'std::getenv("IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE")',
      ),
      _present_check(
          text,
          "carrier_preconv_bundle_run_command_env_forwarded",
          '"IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE"',
      ),
      _present_check(
          text,
          "carrier_preconv_bundle_uses_carrier_attention_norm_handle",
          "g_decode_resident_hidden_state_carrier.attention_norm_handle",
      ),
      _present_check(
          text,
          "carrier_preconv_bundle_or_path_into_preconv_front",
          "g_decode_linear_preconv_shared_q8 || use_carrier_preconv_bundle",
      ),
      _present_check(
          text,
          "shared_device_q8_q4_preconv_bundle_call_present",
          "RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder",
      ),
      _present_check(
          text,
          "shared_device_q8_q6_preconv_bundle_call_present",
          "RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder",
      ),
  ]
  default_off_checks = [
      _absent_check(text, "no_cli_variant_flag_added",
                    "--resident-hidden-state-carrier-preconv-bundle"),
      _present_check(
          text,
          "smoke_json_records_preconv_bundle",
          "resident_hidden_state_carrier_preconv_bundle_enabled",
      ),
  ]
  if require_python_manifest:
    default_off_checks.append(
        _present_check(
            text,
            "python_manifest_records_preconv_bundle",
            '"resident_hidden_state_carrier_preconv_bundle"',
        )
    )
  prerequisite_checks = [
      _present_check(
          text,
          "preconv_bundle_requires_hidden_state_carrier",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE requires",
      ),
      _present_check(
          text,
          "preconv_bundle_requires_shared_q4_runner",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE requires --shared-q4-runner",
      ),
      _present_check(
          text,
          "preconv_bundle_requires_resident_q6_qkv",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE requires --resident-linear-q6-qkv",
      ),
      _present_check(
          text,
          "preconv_bundle_requires_resident_conv_weights",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE requires --resident-linear-conv-weights",
      ),
      _present_check(
          text,
          "preconv_bundle_requires_resident_linear_state",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE requires --resident-linear-state",
      ),
      _present_check(
          text,
          "preconv_bundle_requires_resident_norm_weights",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE requires --resident-norm-weights",
      ),
      _present_check(
          text,
          "preconv_bundle_requires_resident_q4_cpu_order_z",
          "IQ36_RESIDENT_HIDDEN_STATE_CARRIER_PRECONV_BUNDLE requires --resident-q4-cpu-order-z",
      ),
  ]
  return {
      "preconv_bundle_checks": preconv_bundle_checks,
      "default_off_checks": default_off_checks,
      "prerequisite_checks": prerequisite_checks,
      "preconv_bundle_source_present": _all_present(preconv_bundle_checks),
      "default_off_contract_present": (
          _all_absent([row for row in default_off_checks if "absent" in row])
          and _all_present([row for row in default_off_checks if "present" in row])
      ),
      "prerequisites_present": _all_present(prerequisite_checks),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq73 = _load_json(args.seq73)
  seq77 = _load_json(args.seq77)
  seq90 = _load_json(args.seq90)
  seq118 = _load_json(args.seq118)
  decode_text = _load_text(args.decode_source)
  generate_result_path = args.generate_dir / "result.json"
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generate_result = _load_json(generate_result_path)
  generated_text = _load_text(generated_cpp_path)
  smoke_path = args.generate_dir / "smoke.json"

  frontier_state = _frontier_state(frontier)
  source = _source_checks(decode_text, require_python_manifest=True)
  generated = _source_checks(generated_text, require_python_manifest=False)
  seq77_derived = (
      seq77.get("derived") if isinstance(seq77.get("derived"), dict) else {})
  seq90_derived = (
      seq90.get("derived") if isinstance(seq90.get("derived"), dict) else {})

  checks = [
      {
          "name": "seq118_selected_preconv_bundle_source_gate",
          "pass": (
              seq118.get("required_checks_passed") is True
              and seq118.get("selected_next_route")
                  == "resident_hidden_state_carrier_preconv_bundle_source_gate"
              and _has_switch(
                  routes,
                  "accept_carrier_scaffold_switch_to_preconv_bundle_source_gate",
                  118,
              )
          ),
          "detail": {
              "seq118_disposition": seq118.get("disposition"),
              "seq118_selected_next_route": seq118.get("selected_next_route"),
          },
      },
      {
          "name": "seq73_requires_bundled_preconv_not_qkv_only",
          "pass": (
              _nested(seq73, "derived", "qkv_only_preconv_wiring_promotable")
                  is False
              and _nested(seq73, "derived", "shared_device_q8_preconv_bundle_required")
                  is True
          ),
          "detail": {
              "qkv_only_preconv_wiring_promotable": _nested(
                  seq73, "derived", "qkv_only_preconv_wiring_promotable"),
              "shared_device_q8_preconv_bundle_required": _nested(
                  seq73, "derived", "shared_device_q8_preconv_bundle_required"),
          },
      },
      {
          "name": "closed_shared_q8_speed_route_not_reopened",
          "pass": (
              _nested(seq77, "verdict", "shared_q8_profile_closes_speed_route")
                  is True
              and _num(seq77_derived.get("tps_delta_pct_vs_current_source_baseline"))
                  < -0.5
              and _nested(seq90, "derived", "required_checks_passed") is True
              and seq90_derived.get("component_delta_floor_covering") is False
          ),
          "detail": {
              "seq77_tps_delta_pct_vs_current_source_baseline": seq77_derived.get(
                  "tps_delta_pct_vs_current_source_baseline"),
              "seq90_component_delta_floor_covering": seq90_derived.get(
                  "component_delta_floor_covering"),
          },
      },
      {
          "name": "source_preconv_bundle_wiring_present",
          "pass": (
              source["preconv_bundle_source_present"]
              and source["default_off_contract_present"]
              and source["prerequisites_present"]
          ),
          "detail": {
              "preconv_bundle_checks": source["preconv_bundle_checks"],
              "default_off_checks": source["default_off_checks"],
              "prerequisite_checks": source["prerequisite_checks"],
          },
      },
      {
          "name": "generate_only_manifest_uses_carrier_bundle_not_closed_shared_q8_flag",
          "pass": (
              generate_result.get("generate_only") is True
              and generate_result.get("resident_hidden_state_carrier") is True
              and generate_result.get("resident_hidden_state_carrier_preconv_bundle")
                  is True
              and generate_result.get("linear_preconv_shared_q8") is False
              and not smoke_path.exists()
          ),
          "detail": {
              "generate_only": generate_result.get("generate_only"),
              "resident_hidden_state_carrier": generate_result.get(
                  "resident_hidden_state_carrier"),
              "resident_hidden_state_carrier_preconv_bundle": generate_result.get(
                  "resident_hidden_state_carrier_preconv_bundle"),
              "linear_preconv_shared_q8": generate_result.get(
                  "linear_preconv_shared_q8"),
              "smoke_json_exists": smoke_path.exists(),
          },
      },
      {
          "name": "generated_cpp_contains_preconv_bundle_wiring",
          "pass": (
              generated_cpp_path.exists()
              and generated["preconv_bundle_source_present"]
              and generated["default_off_contract_present"]
          ),
          "detail": {
              "generated_cpp": _rel(generated_cpp_path),
              "preconv_bundle_checks": generated["preconv_bundle_checks"],
              "default_off_checks": generated["default_off_checks"],
          },
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
          "decode_source": {
              "path": _rel(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "seq73_bundle_gate": _rel(args.seq73),
          "seq77_shared_q8_profile": _rel(args.seq77),
          "seq90_qkv_conv_root_probe": _rel(args.seq90),
          "seq118_scaffold_gate": _rel(args.seq118),
          "generate_only_result": {
              "path": _rel(generate_result_path),
              "sha256": _sha256(generate_result_path),
          },
          "generated_cpp": {
              "path": _rel(generated_cpp_path),
              "sha256": _sha256(generated_cpp_path),
          },
      },
      "frontier": frontier_state,
      "preconv_bundle_summary": {
          "source_sha": _sha256(args.decode_source),
          "generated_source_sha": generate_result.get("source_sha"),
          "manifest_linear_preconv_shared_q8": generate_result.get(
              "linear_preconv_shared_q8"),
          "manifest_carrier_bundle": generate_result.get(
              "resident_hidden_state_carrier_preconv_bundle"),
          "preconv_bundle_source_present": source[
              "preconv_bundle_source_present"],
          "default_off_contract_present": source[
              "default_off_contract_present"],
          "prerequisites_present": source["prerequisites_present"],
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "source_cut_allowed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": "accept_resident_hidden_state_carrier_preconv_bundle_source_wiring",
      "selected_next_route": "resident_hidden_state_carrier_preconv_bundle_target_compile_gate",
      "next_route_reason": (
          "The carrier preconv bundle source path is now env-gated separately "
          "from the closed --linear-preconv-shared-q8 speed route, consumes the "
          "carrier attention-norm handle, and calls the bundled shared-device-Q8 "
          "Q4/Q6 qkv+conv plus alpha/beta/z primitives. The next admissible "
          "evidence is target compile only, still with no token-emitting decode row."
      ),
      "next_action": (
          "Run a target compile gate for the generated carrier-preconv-bundle "
          "source. If it compiles, continue to a source/compile gate for carrying "
          "the resulting resident handles into selected/shared FFN and FFN tail; "
          "do not launch a decode row before the full carrier boundary is wired."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Resident Hidden-State Carrier Preconv Bundle Source Gate",
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
      "## Next",
      "",
      metrics["next_action"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq73", type=Path, default=DEFAULT_SEQ73)
  parser.add_argument("--seq77", type=Path, default=DEFAULT_SEQ77)
  parser.add_argument("--seq90", type=Path, default=DEFAULT_SEQ90)
  parser.add_argument("--seq118", type=Path, default=DEFAULT_SEQ118)
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
