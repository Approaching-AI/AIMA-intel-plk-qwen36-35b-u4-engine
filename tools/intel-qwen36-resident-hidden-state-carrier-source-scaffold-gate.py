#!/usr/bin/env python3
"""Audit the default-off resident hidden-state carrier source scaffold.

This is source/generate-only evidence. It verifies that the scaffold adds a
carrier contract object and capture points without enabling a token-emitting
decode row or claiming a speed path.
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
SCHEMA_VERSION = "intel-qwen36-resident-hidden-state-carrier-source-scaffold-gate-v0"

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ117 = ROOT / "output/resident-hidden-state-carrier-contract-gate-20260707Tseq117Z/metrics.json"
DEFAULT_GENERATE_DIR = ROOT / "output/resident-hidden-state-carrier-source-scaffold-generate-only-20260707Tseq118Z"
DEFAULT_OUT_DIR = ROOT / "output/resident-hidden-state-carrier-source-scaffold-gate-20260707Tseq118Z"


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


def _count(text: str, pattern: str, *, regex: bool = False) -> int:
  if regex:
    return len(re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL))
  return text.count(pattern)


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


def _source_contract_checks(text: str) -> dict[str, Any]:
  scaffold_checks = [
      _present_check(text, "carrier_struct_exists", "struct ResidentHiddenStateCarrier"),
      _present_check(text, "carrier_global_bool_exists",
                     "bool g_decode_resident_hidden_state_carrier_enabled = false"),
      _present_check(text, "carrier_global_object_exists",
                     "ResidentHiddenStateCarrier g_decode_resident_hidden_state_carrier"),
      _present_check(text, "carrier_begin_layer_method_exists", "void BeginLayer("),
      _present_check(text, "carrier_capture_attention_norm_method_exists",
                     "void CaptureAttentionNorm("),
      _present_check(text, "carrier_capture_attention_front_method_exists",
                     "void CaptureAttentionFront("),
      _present_check(text, "carrier_capture_ffn_inputs_method_exists",
                     "void CaptureFfnInputs("),
      _present_check(text, "carrier_capture_ffn_down_method_exists",
                     "void CaptureFfnDown("),
      _present_check(text, "carrier_capture_layer_output_method_exists",
                     "void CaptureLayerOutput("),
  ]
  default_off_checks = [
      _present_check(text, "carrier_env_gate_exists",
                     'std::getenv("IQ36_RESIDENT_HIDDEN_STATE_CARRIER")'),
      _present_check(text, "carrier_env_forwarded_to_remote_run_command",
                     '"IQ36_RESIDENT_HIDDEN_STATE_CARRIER"'),
      _present_check(text, "carrier_json_field_exists",
                     "resident_hidden_state_carrier_enabled"),
      _absent_check(text, "no_cli_variant_flag_added",
                    "--resident-hidden-state-carrier"),
  ]
  capture_counts = {
      "begin_layer_calls": _count(
          text, "g_decode_resident_hidden_state_carrier.BeginLayer("),
      "attention_norm_capture_calls": _count(
          text, "g_decode_resident_hidden_state_carrier.CaptureAttentionNorm("),
      "attention_front_capture_calls": _count(
          text, "g_decode_resident_hidden_state_carrier.CaptureAttentionFront("),
      "ffn_input_capture_calls": _count(
          text, "g_decode_resident_hidden_state_carrier.CaptureFfnInputs("),
      "ffn_down_capture_calls": _count(
          text, "g_decode_resident_hidden_state_carrier.CaptureFfnDown("),
      "layer_output_capture_calls": _count(
          text, "g_decode_resident_hidden_state_carrier.CaptureLayerOutput("),
  }
  host_shadow_checks = [
      _present_check(
          text,
          "linear_layer_still_keeps_host_residual_shadow",
          r"std::vector<float>\s+RunGpuHybridLinearLayerLive\([^\)]*"
          r"const std::vector<float>&\s+residual",
          regex=True,
      ),
      _present_check(
          text,
          "full_attention_layer_still_keeps_host_residual_shadow",
          r"std::vector<float>\s+RunGpuHybridFullAttentionLayerLive\([^\)]*"
          r"const std::vector<float>&\s+residual",
          regex=True,
      ),
      _present_check(
          text,
          "ffn_tail_still_keeps_host_shadow_inputs",
          r"std::vector<float>\s+RunGpuHybridFfnTail\([^\)]*"
          r"const std::vector<float>&\s+ffn_input,[^\)]*"
          r"const std::vector<float>&\s+attention_residual",
          regex=True,
      ),
  ]
  capture_contract_ready = (
      capture_counts["begin_layer_calls"] >= 2
      and capture_counts["attention_norm_capture_calls"] >= 2
      and capture_counts["attention_front_capture_calls"] >= 2
      and capture_counts["ffn_input_capture_calls"] >= 1
      and capture_counts["ffn_down_capture_calls"] >= 2
      and capture_counts["layer_output_capture_calls"] >= 3
  )
  return {
      "scaffold_checks": scaffold_checks,
      "default_off_checks": default_off_checks,
      "host_shadow_checks": host_shadow_checks,
      "capture_counts": capture_counts,
      "scaffold_present": _all_present(scaffold_checks),
      "default_off_contract_present": (
          _all_present([row for row in default_off_checks if "present" in row])
          and _all_absent([row for row in default_off_checks if "absent" in row])
      ),
      "capture_contract_ready": capture_contract_ready,
      "host_shadow_contract_preserved": _all_present(host_shadow_checks),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq117 = _load_json(args.seq117)
  decode_text = _load_text(args.decode_source)
  generate_result_path = args.generate_dir / "result.json"
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generate_result = _load_json(generate_result_path)
  generated_text = _load_text(generated_cpp_path)

  frontier_state = _frontier_state(frontier)
  source_contract = _source_contract_checks(decode_text)
  generated_contract = _source_contract_checks(generated_text)
  smoke_path = args.generate_dir / "smoke.json"

  checks = [
      {
          "name": "seq117_authorized_source_scaffold",
          "pass": (
              seq117.get("required_checks_passed") is True
              and seq117.get("selected_next_route")
                  == "resident_hidden_state_carrier_source_scaffold_gate"
              and _has_switch(
                  routes,
                  "authorize_resident_hidden_state_carrier_source_scaffold",
                  117,
              )
          ),
          "detail": {
              "seq117_disposition": seq117.get("disposition"),
              "seq117_selected_next_route": seq117.get("selected_next_route"),
          },
      },
      {
          "name": "source_scaffold_contract_present",
          "pass": (
              source_contract["scaffold_present"]
              and source_contract["capture_contract_ready"]
          ),
          "detail": {
              "scaffold_checks": source_contract["scaffold_checks"],
              "capture_counts": source_contract["capture_counts"],
          },
      },
      {
          "name": "scaffold_is_default_off_env_gate_not_cli_variant",
          "pass": source_contract["default_off_contract_present"],
          "detail": source_contract["default_off_checks"],
      },
      {
          "name": "host_shadow_contract_preserved",
          "pass": source_contract["host_shadow_contract_preserved"],
          "detail": source_contract["host_shadow_checks"],
      },
      {
          "name": "generate_only_artifact_contains_scaffold",
          "pass": (
              generate_result.get("generate_only") is True
              and generated_cpp_path.exists()
              and not smoke_path.exists()
              and generated_contract["scaffold_present"]
              and generated_contract["capture_contract_ready"]
              and generated_contract["default_off_contract_present"]
          ),
          "detail": {
              "generate_only": generate_result.get("generate_only"),
              "generated_cpp": generate_result.get("generated_cpp"),
              "smoke_json_exists": smoke_path.exists(),
              "generated_capture_counts": generated_contract["capture_counts"],
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
          "seq117_contract_gate": _rel(args.seq117),
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
      "source_scaffold_summary": {
          "source_sha": _sha256(args.decode_source),
          "generated_source_sha": generate_result.get("source_sha"),
          "scaffold_present": source_contract["scaffold_present"],
          "capture_contract_ready": source_contract["capture_contract_ready"],
          "default_off_contract_present": source_contract[
              "default_off_contract_present"],
          "host_shadow_contract_preserved": source_contract[
              "host_shadow_contract_preserved"],
          "capture_counts": source_contract["capture_counts"],
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "source_cut_allowed": required_checks_passed,
      "decode_probe_allowed": False,
      "component_probe_allowed": False,
      "target_compile_required_before_decode": True,
      "speedup_claims_allowed": False,
      "disposition": "accept_default_off_resident_hidden_state_carrier_scaffold",
      "selected_next_route": "resident_hidden_state_carrier_preconv_bundle_source_gate",
      "next_route_reason": (
          "The default-off carrier object and capture points now exist in the "
          "generated decode source without adding a CLI variant or launching a "
          "decode row. The scaffold preserves host shadows and only records the "
          "resident handles already produced by the layer loop. The next source "
          "gate must turn the scaffold into a real bundled preconv carrier path "
          "without using the closed qkv-only or seq77 shared-Q8 speed routes."
      ),
      "next_action": (
          "Build resident_hidden_state_carrier_preconv_bundle_source_gate: add "
          "a default-off bundled preconv carrier path that consumes the resident "
          "attention-norm handle once, feeds qkv+conv and alpha/beta/z consumers, "
          "and remains source/compile-gated before any token-emitting decode row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  summary = [
      "# Resident Hidden-State Carrier Source Scaffold Gate",
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
  parser.add_argument("--seq117", type=Path, default=DEFAULT_SEQ117)
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
