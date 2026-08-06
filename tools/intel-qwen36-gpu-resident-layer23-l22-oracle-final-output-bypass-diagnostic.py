#!/usr/bin/env python3
"""Run a layer-23 diagnostic with layer-22 oracle final_output bypass.

This wraps the layer-23 oracle-input diagnostic and changes only the layer-22
linear-attention output-projection input: the raw GPU delta `final_output`
comparison is still emitted, but the actual layer-22 dataflow into
`ssm_out.weight` uses captured/oracle `final_output`.

This is a diagnostic, not a backend promotion route.
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
SCHEMA_VERSION = (
    "intel-qwen36-gpu-resident-layer23-l22-oracle-final-output-bypass-diagnostic-v0"
)
BYPASS_API = "layer5_to_layer23_l22_oracle_final_output_bypass_diagnostic"


def load_oracle_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer23_oracle_diag", ORACLE_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer23 oracle diagnostic tool: {ORACLE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


ORACLE = load_oracle_tool()
BASE = ORACLE.BASE
BASE_SCHEMA_VERSION = BASE.SCHEMA_VERSION
ORACLE_SCHEMA_VERSION = ORACLE.SCHEMA_VERSION
ORACLE_DIAGNOSTIC_CPP = ORACLE.diagnostic_cpp
ORACLE_WRITE_SUMMARY = ORACLE.write_summary


def replace_once(text: str, old: str, new: str) -> str:
  return BASE.replace_once(text, old, new)


def diagnostic_cpp(opencl_source: str) -> str:
  old_base_schema = BASE.SCHEMA_VERSION
  BASE.SCHEMA_VERSION = BASE_SCHEMA_VERSION
  try:
    cpp = ORACLE_DIAGNOSTIC_CPP(opencl_source)
  finally:
    BASE.SCHEMA_VERSION = old_base_schema

  cpp = replace_once(cpp, ORACLE_SCHEMA_VERSION, SCHEMA_VERSION)
  cpp = replace_once(
      cpp,
      '''  const auto attention_gpu = RunGpuAttentionFront(
      args.model_path, *t.output_tensor, delta_gpu.final_output, gpu_residual_input,
      ffn_norm_weight, rms_norm_epsilon, args.device_substring, args.repeat);
''',
      '''  const std::vector<float>& gpu_final_output_for_attention =
      (t.layer == 22)
          ? static_cast<const std::vector<float>&>(oracle.final_output)
          : static_cast<const std::vector<float>&>(delta_gpu.final_output);
  const auto attention_gpu = RunGpuAttentionFront(
      args.model_path, *t.output_tensor, gpu_final_output_for_attention,
      gpu_residual_input, ffn_norm_weight, rms_norm_epsilon,
      args.device_substring, args.repeat);
''',
  )
  cpp = replace_once(
      cpp,
      '''  if (t.layer == 22) {
    AppendCompare(result.comparisons, "final_output_same_gpu_preconv",
                  gpu_input_native_final_output, delta_gpu.final_output,
                  oracle.final_output);
  }
''',
      '''  if (t.layer == 22) {
    AppendCompare(result.comparisons, "final_output_same_gpu_preconv",
                  gpu_input_native_final_output, delta_gpu.final_output,
                  oracle.final_output);
    AppendCompare(result.comparisons, "final_output_oracle_bypass_input",
                  native_postconv.final_output, gpu_final_output_for_attention,
                  oracle.final_output);
  }
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer23_residual_input_boundary\\":\\"live_gpu_l_out_22\\",";
    std::cout << "\\"layer23_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer23_residual_input_boundary\\":\\"live_gpu_l_out_22\\",";
    std::cout << "\\"layer22_final_output_bypass\\":\\"oracle_final_output_diagnostic\\",";
    std::cout << "\\"layer23_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer23_residual_input_from_layer22_live_gpu_lout\\":true,";
    std::cout << "\\"layer23_payload_counts_ok\\":"
''',
      '''    std::cout << "\\"layer23_residual_input_from_layer22_live_gpu_lout\\":true,";
    std::cout << "\\"layer22_final_output_oracle_bypass_diagnostic\\":true,";
    std::cout << "\\"layer23_payload_counts_ok\\":"
''',
  )
  return cpp


def parse_args() -> Any:
  args = ORACLE.BASE_PARSE_ARGS()
  if args.out_dir is None:
    stamp = BASE.utc_stamp()
    args.out_dir = (
        BASE.ROOT
        / f"output/gpu-resident-layer23-l22-oracle-final-output-bypass-diagnostic-{stamp}"
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
      "## Layer-22 Oracle Final-Output Bypass Diagnostic",
      "",
      f"- bypass source: `{probe.get('layer22_final_output_bypass')}`",
      f"- bypass check flag: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer22_final_output_oracle_bypass_diagnostic')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name, lane_name in (
      ("l22_final_output", "gpu_vs_oracle"),
      ("l22_final_output_same_gpu_preconv", "cpu_vs_oracle"),
      ("l22_final_output_same_gpu_preconv", "gpu_vs_oracle"),
      ("l22_final_output_oracle_bypass_input", "gpu_vs_oracle"),
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
      "The raw layer-22 GPU `final_output` lane is still reported. Only the",
      "diagnostic dataflow into layer-22 `ssm_out.weight` is bypassed with the",
      "captured oracle tensor to test downstream sensitivity.",
      "",
  ]
  path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines),
                  encoding="utf-8")


def main() -> int:
  BASE.SCHEMA_VERSION = SCHEMA_VERSION
  BASE.L23_API = BYPASS_API
  BASE.layer23_full_attn_probe_cpp = diagnostic_cpp
  BASE.parse_args = parse_args
  BASE.write_summary = write_summary
  return BASE.main()


if __name__ == "__main__":
  raise SystemExit(main())
