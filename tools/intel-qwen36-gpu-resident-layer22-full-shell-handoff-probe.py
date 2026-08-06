#!/usr/bin/env python3
"""Promote the resident GPU layer-22 full-shell/l_out handoff gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import iq36_local


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
STATE_INPUT_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer22-state-input-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer22-full-shell-handoff-probe-v0"
STATE_INPUT_API = "layer5_to_layer22_state_input_load_once_run_many"
FULL_SHELL_API = "layer5_to_layer22_full_shell_lout_load_once_run_many"


def load_state_input_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer22_state_input_probe", STATE_INPUT_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer22 state/input tool: {STATE_INPUT_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L22 = load_state_input_tool()
CORE = L22.CORE
PRECONV = L22.PRECONV


def selected_out_dir(argv: list[str]) -> Path | None:
  for index, item in enumerate(argv):
    if item == "--out-dir":
      if index + 1 >= len(argv):
        raise SystemExit("--out-dir requires a value")
      return Path(argv[index + 1])
    if item.startswith("--out-dir="):
      return Path(item.split("=", 1)[1])
  return None


def with_out_dir(argv: list[str], out_dir: Path) -> list[str]:
  result = list(argv)
  for index, item in enumerate(result):
    if item == "--out-dir":
      if index + 1 >= len(result):
        raise SystemExit("--out-dir requires a value")
      result[index + 1] = str(out_dir)
      return result
    if item.startswith("--out-dir="):
      result[index] = f"--out-dir={out_dir}"
      return result
  result.extend(["--out-dir", str(out_dir)])
  return result


def comparison_stats(probe: dict[str, Any] | None, name: str, lane: str) -> dict[str, Any]:
  if not isinstance(probe, dict):
    return {}
  comparisons = probe.get("comparisons", {})
  if not isinstance(comparisons, dict):
    return {}
  group = comparisons.get(name, {})
  if not isinstance(group, dict):
    return {}
  stats = group.get(lane, {})
  return stats if isinstance(stats, dict) else {}


def magnitude_comparison_passed(probe: dict[str, Any] | None, name: str, lane: str) -> bool:
  stats = comparison_stats(probe, name, lane)
  thresholds = CORE.STRICT_COMPARISON_THRESHOLDS
  return (
      stats.get("same_size") is True
      and stats.get("finite") is True
      and stats.get("mismatch_count") == 0
      and stats.get("max_abs_diff", 1.0) <= thresholds["max_abs_diff"]
      and stats.get("rmse", 1.0) <= thresholds["rmse"]
  )


def install_full_shell_overrides() -> None:
  L22.SCHEMA_VERSION = SCHEMA_VERSION
  original_cpp = L22.layer22_state_input_probe_cpp

  def full_shell_cpp(opencl_source: str) -> str:
    cpp = original_cpp(opencl_source)
    return L22.replace_once(cpp, STATE_INPUT_API, FULL_SHELL_API)

  def resident_fields_ok(
      probe: dict[str, Any] | None,
      expected_invocations: int,
      device_substring: str,
  ) -> bool:
    return (
        isinstance(probe, dict)
        and probe.get("resident_api") == FULL_SHELL_API
        and probe.get("resident_load_count") == 1
        and probe.get("resident_shell_invocations") == expected_invocations
        and probe.get("layer19_lout_boundary") == "live_gpu_l_out_19"
        and device_substring in str(probe.get("layer19_lout_device_name", ""))
        and probe.get("layer20_residual_input_boundary") == "live_gpu_l_out_19"
        and device_substring in str(probe.get("layer20_tail_device_name", ""))
        and probe.get("layer21_residual_input_boundary") == "live_gpu_l_out_20"
        and device_substring in str(probe.get("layer21_tail_device_name", ""))
        and probe.get("layer22_residual_input_boundary") == "live_gpu_l_out_21"
        and probe.get("layer22_qkv_tensor_type") in {"Q4_K", "Q6_K"}
        and probe.get("layer22_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
        and probe.get("layer22_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
        and probe.get("layer22_conv_state_boundary") == "captured_conv_state"
        and device_substring in str(probe.get("layer22_layer_input_device_name", ""))
        and device_substring in str(probe.get("layer22_preconv_device_name", ""))
        and device_substring in str(probe.get("layer22_tail_device_name", ""))
        and PRECONV.nested_bool(probe, "checks", "resident_load_once")
        and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
    )

  L22.layer22_state_input_probe_cpp = full_shell_cpp
  L22.resident_fields_ok = resident_fields_ok


def full_shell_checks(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
  return [
      {"name": "resident_api_full_shell_lout", "pass": isinstance(probe, dict) and probe.get("resident_api") == FULL_SHELL_API},
      {"name": "layer22_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer22_state_input_handoff_matches")},
      {"name": "layer22_full_shell_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l22_layer_output", "gpu_vs_oracle")},
      {"name": "l22_final_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l22_final_output", "gpu_vs_oracle")},
      {"name": "l22_linear_attn_out_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l22_linear_attn_out", "gpu_vs_oracle")},
      {"name": "l22_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l22_attn_output", "gpu_vs_oracle")},
      {"name": "l22_attn_residual_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l22_attn_residual", "gpu_vs_oracle")},
      {"name": "l22_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l22_attn_post_norm", "gpu_vs_oracle")},
      {"name": "l22_selected_gate_up_matches_oracle", "pass": CORE.comparison_passed(probe, "l22_selected_gate_up", "gpu_vs_oracle")},
      {"name": "l22_selected_down_magnitude_matches_oracle", "pass": magnitude_comparison_passed(probe, "l22_selected_down", "gpu_vs_oracle")},
      {"name": "l22_shared_down_magnitude_matches_oracle", "pass": magnitude_comparison_passed(probe, "l22_shared_down", "gpu_vs_oracle")},
      {"name": "l22_ffn_moe_out_magnitude_matches_oracle", "pass": magnitude_comparison_passed(probe, "l22_ffn_moe_out", "gpu_vs_oracle")},
      {"name": "l22_ffn_shexp_gated_magnitude_matches_oracle", "pass": magnitude_comparison_passed(probe, "l22_ffn_shexp_gated", "gpu_vs_oracle")},
      {"name": "l22_ffn_out_magnitude_matches_oracle", "pass": magnitude_comparison_passed(probe, "l22_ffn_out", "gpu_vs_oracle")},
      {"name": "l22_layer_output_lout_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l22_layer_output", "gpu_vs_oracle")},
      {"name": "l22_conv_state_after_matches_native", "pass": CORE.comparison_passed(probe, "l22_conv_state_after", "gpu_vs_cpu")},
      {"name": "l22_recurrent_state_matches_native", "pass": CORE.comparison_passed(probe, "l22_recurrent_state", "gpu_vs_cpu")},
      {"name": "layer22_lout_boundary_live_gpu", "pass": isinstance(probe, dict) and probe.get("layer22_residual_input_boundary") == "live_gpu_l_out_21" and "B390" in str(probe.get("layer22_tail_device_name", ""))},
  ]


def read_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    data = json.load(fh)
  if not isinstance(data, dict):
    raise SystemExit(f"expected JSON object at {path}")
  return data


def stats_pass(stats: dict[str, Any], thresholds: dict[str, float]) -> bool:
  return (
      stats.get("same_size") is True
      and stats.get("finite") is True
      and stats.get("mismatch_count") == 0
      and stats.get("max_abs_diff", 1.0) <= thresholds["max_abs_diff"]
      and stats.get("rmse", 1.0) <= thresholds["rmse"]
      and stats.get("cosine", 0.0) >= thresholds["min_cosine"]
  )


def strict_diagnostic_failures(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
  if not isinstance(probe, dict):
    return []
  comparisons = probe.get("comparisons", {})
  if not isinstance(comparisons, dict):
    return []
  failures: list[dict[str, Any]] = []
  for name, group in comparisons.items():
    if not str(name).startswith("l22_") or not isinstance(group, dict):
      continue
    for lane, stats in group.items():
      if not isinstance(stats, dict):
        continue
      required = {"same_size", "finite", "mismatch_count", "max_abs_diff", "rmse", "cosine"}
      if not required.issubset(stats):
        continue
      if not stats_pass(stats, CORE.STRICT_COMPARISON_THRESHOLDS):
        failures.append({
            "name": name,
            "lane": lane,
            "max_abs_diff": stats.get("max_abs_diff"),
            "rmse": stats.get("rmse"),
            "cosine": stats.get("cosine"),
            "mismatch_count": stats.get("mismatch_count"),
        })
  return failures


def comparison_lane(probe: dict[str, Any], name: str) -> dict[str, Any]:
  comparisons = probe.get("comparisons", {})
  if not isinstance(comparisons, dict):
    return {}
  group = comparisons.get(name, {})
  if not isinstance(group, dict):
    return {}
  lane = group.get("gpu_vs_oracle", {})
  return lane if isinstance(lane, dict) else {}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  diagnostics = payload.get("layer22_internal_strict_diagnostic_failures", [])
  lines = [
      "# GPU Resident Layer-22 Full-Shell/l_out Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api') if isinstance(probe, dict) else None}`",
      f"- raw full-shell diagnostic flag: `{PRECONV.nested_bool(probe, 'checks', 'layer17', 'comparisons_passed')}`",
      f"- internal strict diagnostic failures: `{len(diagnostics) if isinstance(diagnostics, list) else 0}`",
      f"- layer 22 residual input boundary: `{probe.get('layer22_residual_input_boundary') if isinstance(probe, dict) else None}`",
      f"- layer 22 qkv tensor type: `{probe.get('layer22_qkv_tensor_type') if isinstance(probe, dict) else None}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l22_final_output",
      "l22_linear_attn_out",
      "l22_attn_output",
      "l22_attn_residual",
      "l22_attn_post_norm",
      "l22_selected_gate_up",
      "l22_selected_down",
      "l22_shared_down",
      "l22_ffn_moe_out",
      "l22_ffn_out",
      "l22_layer_output",
  ):
    lane = comparison_lane(probe, name) if isinstance(probe, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "The raw `checks.layer17.comparisons_passed` flag is recorded as an",
      "internal diagnostic. Required closure here is GPU-vs-oracle full",
      "shell/l_out plus state/recurrent GPU-vs-native sanity. Q6 internal",
      "FFN lanes use the live-FFN magnitude policy, with strict misses kept",
      "in `promotion.json`.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def promote_artifact(out_dir: Path, inner_rc: int) -> int:
  probe_path = out_dir / "probe.json"
  manifest_path = out_dir / "manifest.json"
  correctness_path = out_dir / "correctness.json"
  if not probe_path.exists():
    return inner_rc if inner_rc != 0 else 1

  payload = read_json(probe_path)
  manifest = read_json(manifest_path)
  correctness = read_json(correctness_path)
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else None
  base_checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
  checks = [item for item in base_checks if isinstance(item, dict)]
  checks.extend(full_shell_checks(probe))
  required_checks_passed = all(item.get("pass") is True for item in checks)
  failed_checks = [str(item.get("name")) for item in checks if item.get("pass") is not True]
  diagnostic_failures = strict_diagnostic_failures(probe)

  comparison_thresholds = payload.get("comparison_thresholds", {})
  if isinstance(comparison_thresholds, dict):
    comparison_thresholds["layer22_full_shell_oracle_policy"] = CORE.FULL_ATTN_COMPARISON_THRESHOLDS
    comparison_thresholds["layer22_ffn_strict_component"] = CORE.STRICT_COMPARISON_THRESHOLDS

  for doc in (payload, manifest, correctness):
    doc["schema_version"] = SCHEMA_VERSION
    doc["required_checks_passed"] = required_checks_passed
    doc["speedup_claims_allowed"] = False
  payload["promotion_gate"] = "layer22_full_shell_lout"
  payload["promoted_from_tool"] = "tools/intel-qwen36-gpu-resident-layer22-state-input-handoff-probe.py"
  payload["checks"] = checks
  payload["failed_checks"] = failed_checks
  payload["comparison_thresholds"] = comparison_thresholds
  payload["layer22_internal_strict_diagnostic_failures"] = diagnostic_failures
  manifest["tool"] = "tools/intel-qwen36-gpu-resident-layer22-full-shell-handoff-probe.py"
  manifest["artifact"] = str(out_dir)
  manifest["promotion_gate"] = "layer22_full_shell_lout"
  manifest["failed_checks"] = failed_checks
  manifest["layer22_internal_strict_diagnostic_failure_count"] = len(diagnostic_failures)
  correctness["checks"] = checks
  correctness["failed_checks"] = failed_checks
  correctness["comparison_thresholds"] = comparison_thresholds

  promotion = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "artifact": str(out_dir),
      "promotion_gate": "layer22_full_shell_lout",
      "raw_layer22_full_shell_matches_oracle": PRECONV.nested_bool(probe, "checks", "layer17", "comparisons_passed"),
      "layer22_internal_strict_diagnostic_failures": diagnostic_failures,
      "checks": checks,
      "failed_checks": failed_checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(probe_path, payload)
  iq36_local.write_json(manifest_path, manifest)
  iq36_local.write_json(correctness_path, correctness)
  iq36_local.write_json(out_dir / "promotion.json", promotion)

  aggregate = probe if isinstance(probe, dict) else {}
  timings = aggregate.get("timings", {}) if isinstance(aggregate.get("timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_resident_layer22_full_shell_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", payload.get("resident_invocations")),
          ("raw_layer22_full_shell_matches_oracle", PRECONV.nested_bool(probe, "checks", "layer17", "comparisons_passed")),
          ("layer22_internal_strict_diagnostic_failure_count", len(diagnostic_failures)),
          ("resident_layer22_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer22_state_input_kernel_sum_min_us")),
          ("l22_final_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("l22_selected_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l22_shared_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_shared_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l22_ffn_out_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_ffn_out", "gpu_vs_oracle", "max_abs_diff")),
          ("l22_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l22_layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  return 0 if required_checks_passed else 1


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--out-dir", type=Path, default=None)
  parsed, _ = parser.parse_known_args(argv)
  args = list(sys.argv[1:] if argv is None else argv)
  out_dir = parsed.out_dir or selected_out_dir(args)
  if out_dir is None:
    out_dir = ROOT / f"output/gpu-resident-layer22-full-shell-handoff-probe-{L22.utc_stamp()}"
  forwarded_args = with_out_dir(args, out_dir)
  install_full_shell_overrides()
  old_argv = sys.argv
  try:
    sys.argv = [str(STATE_INPUT_TOOL), *forwarded_args]
    inner_rc = L22.main()
  finally:
    sys.argv = old_argv
  rc = promote_artifact(out_dir, inner_rc)
  print(out_dir)
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
