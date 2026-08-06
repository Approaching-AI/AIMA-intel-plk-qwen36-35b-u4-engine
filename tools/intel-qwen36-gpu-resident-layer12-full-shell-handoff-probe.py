#!/usr/bin/env python3
"""Promote the resident GPU layer-12 full-shell/l_out handoff gate."""

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
    "intel-qwen36-gpu-resident-layer12-state-input-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer12-full-shell-handoff-probe-v0"
STATE_INPUT_API = "layer5_to_layer12_state_input_load_once_run_many"
FULL_SHELL_API = "layer5_to_layer12_full_shell_lout_load_once_run_many"


def load_state_input_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer12_state_input_probe", STATE_INPUT_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer12 state/input tool: {STATE_INPUT_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L12 = load_state_input_tool()
CORE = L12.CORE
PRECONV = L12.PRECONV


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


def install_full_shell_overrides() -> None:
  L12.SCHEMA_VERSION = SCHEMA_VERSION
  original_cpp = L12.layer12_state_input_probe_cpp

  def full_shell_cpp(opencl_source: str) -> str:
    cpp = original_cpp(opencl_source)
    return L12.replace_once(cpp, STATE_INPUT_API, FULL_SHELL_API)

  def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
    return (
        isinstance(probe, dict)
        and probe.get("resident_api") == FULL_SHELL_API
        and probe.get("resident_load_count") == 1
        and probe.get("resident_shell_invocations") == expected_invocations
        and probe.get("layer11_lout_boundary") == "live_gpu_l_out_11"
        and probe.get("layer12_residual_input_boundary") == "live_gpu_l_out_11"
        and probe.get("layer12_qkv_tensor_type") in {"Q4_K", "Q6_K"}
        and probe.get("layer12_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
        and probe.get("layer12_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
        and probe.get("layer12_conv_state_boundary") == "captured_conv_state"
        and "B390" in str(probe.get("layer12_tail_device_name", ""))
        and PRECONV.nested_bool(probe, "checks", "resident_load_once")
        and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
    )

  L12.layer12_state_input_probe_cpp = full_shell_cpp
  L12.resident_fields_ok = resident_fields_ok


def full_shell_checks(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
  return [
      {"name": "resident_api_full_shell_lout", "pass": isinstance(probe, dict) and probe.get("resident_api") == FULL_SHELL_API},
      {"name": "layer12_full_shell_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer12_full_shell_matches_oracle")},
      {"name": "l12_final_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_final_output", "gpu_vs_oracle")},
      {"name": "l12_linear_attn_out_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_linear_attn_out", "gpu_vs_oracle")},
      {"name": "l12_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_attn_output", "gpu_vs_oracle")},
      {"name": "l12_attn_residual_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_attn_residual", "gpu_vs_oracle")},
      {"name": "l12_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_attn_post_norm", "gpu_vs_oracle")},
      {"name": "l12_selected_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l12_selected_down", "gpu_vs_oracle")},
      {"name": "l12_shared_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l12_shared_down", "gpu_vs_oracle")},
      {"name": "l12_ffn_moe_out_matches_oracle", "pass": CORE.comparison_passed(probe, "l12_ffn_moe_out", "gpu_vs_oracle")},
      {"name": "l12_ffn_shexp_gated_matches_oracle", "pass": CORE.comparison_passed(probe, "l12_ffn_shexp_gated", "gpu_vs_oracle")},
      {"name": "l12_ffn_out_matches_oracle", "pass": CORE.comparison_passed(probe, "l12_ffn_out", "gpu_vs_oracle")},
      {"name": "l12_layer_output_lout_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_layer_output", "gpu_vs_oracle")},
      {"name": "layer12_lout_boundary_live_gpu", "pass": isinstance(probe, dict) and probe.get("layer12_residual_input_boundary") == "live_gpu_l_out_11" and "B390" in str(probe.get("layer12_tail_device_name", ""))},
  ]


def read_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as fh:
    data = json.load(fh)
  if not isinstance(data, dict):
    raise SystemExit(f"expected JSON object at {path}")
  return data


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
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-12 Full-Shell/l_out Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api') if isinstance(probe, dict) else None}`",
      f"- layer 12 residual input boundary: `{probe.get('layer12_residual_input_boundary') if isinstance(probe, dict) else None}`",
      f"- layer 12 qkv tensor type: `{probe.get('layer12_qkv_tensor_type') if isinstance(probe, dict) else None}`",
      f"- layer 12 selected/shared down: `{probe.get('layer12_selected_down_tensor_type') if isinstance(probe, dict) else None}` / `{probe.get('layer12_shared_down_tensor_type') if isinstance(probe, dict) else None}`",
      f"- layer 12 conv state boundary: `{probe.get('layer12_conv_state_boundary') if isinstance(probe, dict) else None}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l12_residual_input",
      "l12_attn_norm",
      "l12_linear_attn_qkv_mixed",
      "l12_conv_output_raw",
      "l12_final_output",
      "l12_linear_attn_out",
      "l12_attn_output",
      "l12_attn_residual",
      "l12_attn_post_norm",
      "l12_selected_down",
      "l12_shared_down",
      "l12_ffn_out",
      "l12_layer_output",
  ):
    lane = comparison_lane(probe, name) if isinstance(probe, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer12_state_input | {timings.get('resident_layer12_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| through_layer12_state_input | {timings.get('resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process reruns the closed layer-12 state/input path and",
      "promotes the already-computed full shell through FFN and `l_out-12` into",
      "required checks. It remains single-token correctness evidence, not a",
      "decode throughput or speedup claim.",
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

  comparison_thresholds = payload.get("comparison_thresholds", {})
  if isinstance(comparison_thresholds, dict):
    comparison_thresholds["layer12_full_shell_oracle_policy"] = CORE.FULL_ATTN_COMPARISON_THRESHOLDS
    comparison_thresholds["layer12_ffn_strict_component"] = CORE.STRICT_COMPARISON_THRESHOLDS

  for doc in (payload, manifest, correctness):
    doc["schema_version"] = SCHEMA_VERSION
    doc["required_checks_passed"] = required_checks_passed
    doc["speedup_claims_allowed"] = False
  payload["promotion_gate"] = "layer12_full_shell_lout"
  payload["promoted_from_tool"] = "tools/intel-qwen36-gpu-resident-layer12-state-input-handoff-probe.py"
  payload["checks"] = checks
  payload["failed_checks"] = failed_checks
  payload["comparison_thresholds"] = comparison_thresholds
  manifest["tool"] = "tools/intel-qwen36-gpu-resident-layer12-full-shell-handoff-probe.py"
  manifest["artifact"] = str(out_dir)
  manifest["promotion_gate"] = "layer12_full_shell_lout"
  manifest["failed_checks"] = failed_checks
  correctness["checks"] = checks
  correctness["failed_checks"] = failed_checks
  correctness["comparison_thresholds"] = comparison_thresholds

  iq36_local.write_json(probe_path, payload)
  iq36_local.write_json(manifest_path, manifest)
  iq36_local.write_json(correctness_path, correctness)
  iq36_local.write_json(
      out_dir / "promotion.json",
      {
          "schema_version": SCHEMA_VERSION,
          "workstream": WORKSTREAM,
          "artifact": str(out_dir),
          "promotion_gate": "layer12_full_shell_lout",
          "checks": checks,
          "failed_checks": failed_checks,
          "required_checks_passed": required_checks_passed,
          "speedup_claims_allowed": False,
      },
  )

  aggregate = probe if isinstance(probe, dict) else {}
  timings = aggregate.get("timings", {}) if isinstance(aggregate.get("timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_resident_layer12_full_shell_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", payload.get("resident_invocations")),
          ("resident_layer12_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer12_state_input_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us")),
          ("l12_final_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("l12_attn_residual_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_attn_residual", "gpu_vs_oracle", "max_abs_diff")),
          ("l12_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("l12_selected_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l12_shared_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_shared_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l12_ffn_out_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_ffn_out", "gpu_vs_oracle", "max_abs_diff")),
          ("l12_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_layer_output", "gpu_vs_oracle", "max_abs_diff")),
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
    out_dir = ROOT / f"output/gpu-resident-layer12-full-shell-handoff-probe-{L12.utc_stamp()}"
  forwarded_args = with_out_dir(args, out_dir)
  install_full_shell_overrides()
  old_argv = sys.argv
  try:
    sys.argv = [str(STATE_INPUT_TOOL), *forwarded_args]
    inner_rc = L12.main()
  finally:
    sys.argv = old_argv
  rc = promote_artifact(out_dir, inner_rc)
  print(out_dir)
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
