#!/usr/bin/env python3
"""Run layer-23 live full-attention with layer-22 native z correction.

This wraps the layer-23 oracle-input diagnostic and changes only the layer-22
delta/final input `z`: `RunGpuPostConvDelta` receives CPU/native `z` for layer
22 while layer 23 still consumes live GPU `l_out-22`. This tests whether a
real z-projection correction path can close the non-bypassed layer23 live gate.

This is diagnostic evidence only, not a backend implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
ORACLE_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer23-oracle-input-full-attn-diagnostic.py"
)
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer23-l22-native-z-correction-diagnostic-v0"
NATIVE_Z_API = "layer5_to_layer23_l22_native_z_correction_diagnostic"


def load_oracle_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer23_oracle_diag", ORACLE_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer23 oracle diagnostic tool: {ORACLE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


ORACLE = load_oracle_tool()
BASE = ORACLE.BASE
ORACLE_SCHEMA_VERSION = ORACLE.SCHEMA_VERSION
ORACLE_DIAGNOSTIC_CPP = ORACLE.diagnostic_cpp
ORACLE_WRITE_SUMMARY = ORACLE.write_summary


def replace_once(text: str, old: str, new: str) -> str:
  return BASE.replace_once(text, old, new)


def diagnostic_cpp(opencl_source: str) -> str:
  cpp = ORACLE_DIAGNOSTIC_CPP(opencl_source)
  cpp = replace_once(cpp, ORACLE_SCHEMA_VERSION, SCHEMA_VERSION)
  cpp = replace_once(
      cpp,
      '''  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      preconv_gpu.z,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
      '''  const std::vector<float>& z_for_delta =
      (t.layer == 22)
          ? static_cast<const std::vector<float>&>(native_preconv.z)
          : static_cast<const std::vector<float>&>(preconv_gpu.z);
  const auto delta_gpu = RunGpuPostConvDelta(
      preconv_gpu.q_conv_predelta,
      preconv_gpu.k_conv_predelta,
      preconv_gpu.v_conv_predelta,
      preconv_gpu.gate,
      preconv_gpu.beta_sigmoid,
      oracle.state,
      z_for_delta,
      ssm_norm_weight,
      rms_norm_epsilon,
      args.device_substring,
      args.repeat);
''',
  )
  cpp = replace_once(
      cpp,
      '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
''',
      '''  AppendCompare(result.comparisons, "z",
                native_preconv.z, preconv_gpu.z, oracle.z);
  if (t.layer == 22) {
    AppendCompare(result.comparisons, "z_native_correction_input",
                  native_preconv.z, z_for_delta, oracle.z);
  }
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
      '''    std::cout << "\\"layer23_arc_device_selected\\":"
              << (layer18_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer22_native_z_correction_diagnostic\\":true,";
    std::cout << "\\"layer23_oracle_input_strict_matches_oracle\\":"
''',
  )
  return cpp


def parse_args() -> Any:
  args = ORACLE.BASE_PARSE_ARGS()
  if args.out_dir is None:
    stamp = BASE.utc_stamp()
    args.out_dir = (
        BASE.ROOT
        / f"output/gpu-resident-layer23-l22-native-z-correction-diagnostic-{stamp}"
    )
  return args


def _lane(probe: dict[str, Any], name: str, lane: str) -> dict[str, Any]:
  comparisons = probe.get("comparisons", {})
  if not isinstance(comparisons, dict):
    return {}
  group = comparisons.get(name, {})
  if not isinstance(group, dict):
    return {}
  stats = group.get(lane, {})
  return stats if isinstance(stats, dict) else {}


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  ORACLE_WRITE_SUMMARY(path, payload)
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  lines = [
      "",
      "## Layer-22 Native Z Correction Diagnostic",
      "",
      f"- correction flag: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer22_native_z_correction_diagnostic')}`",
      f"- non-bypassed layer23 full-attn passed: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer23_full_attn_core_output_matches_oracle')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name, lane_name in (
      ("l22_z", "gpu_vs_oracle"),
      ("l22_z_native_correction_input", "gpu_vs_oracle"),
      ("l22_final_output", "gpu_vs_oracle"),
      ("l22_linear_attn_out", "gpu_vs_oracle"),
      ("l22_attn_post_norm", "gpu_vs_oracle"),
      ("l22_layer_output", "gpu_vs_oracle"),
      ("l23_residual_input", "gpu_vs_oracle"),
      ("l23_attn_norm", "gpu_vs_oracle"),
      ("l23_attn_output", "gpu_vs_oracle"),
      ("l23_attn_post_norm", "gpu_vs_oracle"),
  ):
    stats = _lane(probe, name, lane_name) if isinstance(probe, dict) else {}
    lines.append(
        f"| {name} | {lane_name} | {stats.get('max_abs_diff')} | {stats.get('rmse')} |"
    )
  lines += [
      "",
      "Layer 23 consumes live GPU `l_out-22` in this run. Only layer22 z input",
      "to delta/final is corrected to CPU/native z; `final_output` is not",
      "oracle-bypassed.",
      "",
  ]
  path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines),
                  encoding="utf-8")


def main() -> int:
  BASE.SCHEMA_VERSION = SCHEMA_VERSION
  BASE.L23_API = NATIVE_Z_API
  BASE.layer23_full_attn_probe_cpp = diagnostic_cpp
  BASE.parse_args = parse_args
  BASE.write_summary = write_summary
  return BASE.main()


if __name__ == "__main__":
  raise SystemExit(main())
