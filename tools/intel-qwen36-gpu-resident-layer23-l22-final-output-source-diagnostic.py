#!/usr/bin/env python3
"""Run layer-22 final_output source-isolation diagnostics.

This builds on the layer-23 layer-22-oracle-final-output bypass diagnostic.
The bypass keeps downstream layer23 closed while this tool adds extra
layer-22 `RunGpuPostConvDelta` variants to isolate which pre-`ssm_out.weight`
inputs cause the tiny `final_output` drift.

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
BYPASS_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer23-l22-oracle-final-output-bypass-diagnostic.py"
)
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer23-l22-final-output-source-diagnostic-v0"
SOURCE_API = "layer5_to_layer23_l22_final_output_source_diagnostic"


def load_bypass_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer23_l22_bypass_diag", BYPASS_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer22 bypass diagnostic tool: {BYPASS_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BYPASS = load_bypass_tool()
BASE = BYPASS.BASE
BYPASS_SCHEMA_VERSION = BYPASS.SCHEMA_VERSION
BYPASS_DIAGNOSTIC_CPP = BYPASS.diagnostic_cpp
BYPASS_WRITE_SUMMARY = BYPASS.write_summary


def replace_once(text: str, old: str, new: str) -> str:
  return BASE.replace_once(text, old, new)


def diagnostic_cpp(opencl_source: str) -> str:
  cpp = BYPASS_DIAGNOSTIC_CPP(opencl_source)
  cpp = replace_once(cpp, BYPASS_SCHEMA_VERSION, SCHEMA_VERSION)
  cpp = replace_once(
      cpp,
      '''  const std::vector<float>& gpu_final_output_for_attention =
      (t.layer == 22)
          ? static_cast<const std::vector<float>&>(oracle.final_output)
          : static_cast<const std::vector<float>&>(delta_gpu.final_output);
''',
      '''  std::vector<float> l22_final_output_all_oracle_inputs;
  std::vector<float> l22_final_output_gpu_qkv_oracle_modulators;
  std::vector<float> l22_final_output_oracle_qkv_gpu_modulators;
  std::vector<float> l22_final_output_oracle_qkv_gpu_gate;
  std::vector<float> l22_final_output_oracle_qkv_gpu_beta;
  std::vector<float> l22_final_output_oracle_qkv_gpu_z;
  if (t.layer == 22) {
    const auto l22_delta_all_oracle_inputs = RunGpuPostConvDelta(
        oracle.q, oracle.k, oracle.v, oracle.gate, oracle.beta_sigmoid,
        oracle.state, oracle.z, ssm_norm_weight, rms_norm_epsilon,
        args.device_substring, args.repeat);
    l22_final_output_all_oracle_inputs =
        l22_delta_all_oracle_inputs.final_output;

    const auto l22_delta_gpu_qkv_oracle_modulators = RunGpuPostConvDelta(
        preconv_gpu.q_conv_predelta, preconv_gpu.k_conv_predelta,
        preconv_gpu.v_conv_predelta, oracle.gate, oracle.beta_sigmoid,
        oracle.state, oracle.z, ssm_norm_weight, rms_norm_epsilon,
        args.device_substring, args.repeat);
    l22_final_output_gpu_qkv_oracle_modulators =
        l22_delta_gpu_qkv_oracle_modulators.final_output;

    const auto l22_delta_oracle_qkv_gpu_modulators = RunGpuPostConvDelta(
        oracle.q, oracle.k, oracle.v, preconv_gpu.gate,
        preconv_gpu.beta_sigmoid, oracle.state, preconv_gpu.z,
        ssm_norm_weight, rms_norm_epsilon, args.device_substring,
        args.repeat);
    l22_final_output_oracle_qkv_gpu_modulators =
        l22_delta_oracle_qkv_gpu_modulators.final_output;

    const auto l22_delta_oracle_qkv_gpu_gate = RunGpuPostConvDelta(
        oracle.q, oracle.k, oracle.v, preconv_gpu.gate,
        oracle.beta_sigmoid, oracle.state, oracle.z, ssm_norm_weight,
        rms_norm_epsilon, args.device_substring, args.repeat);
    l22_final_output_oracle_qkv_gpu_gate =
        l22_delta_oracle_qkv_gpu_gate.final_output;

    const auto l22_delta_oracle_qkv_gpu_beta = RunGpuPostConvDelta(
        oracle.q, oracle.k, oracle.v, oracle.gate,
        preconv_gpu.beta_sigmoid, oracle.state, oracle.z, ssm_norm_weight,
        rms_norm_epsilon, args.device_substring, args.repeat);
    l22_final_output_oracle_qkv_gpu_beta =
        l22_delta_oracle_qkv_gpu_beta.final_output;

    const auto l22_delta_oracle_qkv_gpu_z = RunGpuPostConvDelta(
        oracle.q, oracle.k, oracle.v, oracle.gate, oracle.beta_sigmoid,
        oracle.state, preconv_gpu.z, ssm_norm_weight, rms_norm_epsilon,
        args.device_substring, args.repeat);
    l22_final_output_oracle_qkv_gpu_z =
        l22_delta_oracle_qkv_gpu_z.final_output;
  }

  const std::vector<float>& gpu_final_output_for_attention =
      (t.layer == 22)
          ? static_cast<const std::vector<float>&>(oracle.final_output)
          : static_cast<const std::vector<float>&>(delta_gpu.final_output);
''',
  )
  cpp = replace_once(
      cpp,
      '''    AppendCompare(result.comparisons, "final_output_oracle_bypass_input",
                  native_postconv.final_output, gpu_final_output_for_attention,
                  oracle.final_output);
  }
''',
      '''    AppendCompare(result.comparisons, "final_output_oracle_bypass_input",
                  native_postconv.final_output, gpu_final_output_for_attention,
                  oracle.final_output);
    AppendCompare(result.comparisons, "final_output_source_all_oracle_inputs",
                  oracle.final_output, l22_final_output_all_oracle_inputs,
                  oracle.final_output);
    AppendCompare(result.comparisons, "final_output_source_gpu_qkv_oracle_modulators",
                  oracle.final_output,
                  l22_final_output_gpu_qkv_oracle_modulators,
                  oracle.final_output);
    AppendCompare(result.comparisons, "final_output_source_oracle_qkv_gpu_modulators",
                  oracle.final_output,
                  l22_final_output_oracle_qkv_gpu_modulators,
                  oracle.final_output);
    AppendCompare(result.comparisons, "final_output_source_oracle_qkv_gpu_gate",
                  oracle.final_output,
                  l22_final_output_oracle_qkv_gpu_gate,
                  oracle.final_output);
    AppendCompare(result.comparisons, "final_output_source_oracle_qkv_gpu_beta",
                  oracle.final_output,
                  l22_final_output_oracle_qkv_gpu_beta,
                  oracle.final_output);
    AppendCompare(result.comparisons, "final_output_source_oracle_qkv_gpu_z",
                  oracle.final_output,
                  l22_final_output_oracle_qkv_gpu_z,
                  oracle.final_output);
  }
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer22_final_output_oracle_bypass_diagnostic\\":true,";
    std::cout << "\\"layer23_payload_counts_ok\\":"
''',
      '''    std::cout << "\\"layer22_final_output_oracle_bypass_diagnostic\\":true,";
    std::cout << "\\"layer22_final_output_source_diagnostic\\":true,";
    std::cout << "\\"layer23_payload_counts_ok\\":"
''',
  )
  return cpp


def parse_args() -> Any:
  args = BYPASS.ORACLE.BASE_PARSE_ARGS()
  if args.out_dir is None:
    stamp = BASE.utc_stamp()
    args.out_dir = (
        BASE.ROOT
        / f"output/gpu-resident-layer23-l22-final-output-source-diagnostic-{stamp}"
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
  BYPASS_WRITE_SUMMARY(path, payload)
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  lines = [
      "",
      "## Layer-22 Final-Output Source Diagnostic",
      "",
      f"- source diagnostic flag: `{BASE.PRECONV.nested_bool(probe, 'checks', 'layer22_final_output_source_diagnostic')}`",
      "",
      "| variant | max abs | RMSE |",
      "|---|---:|---:|",
  ]
  for name in (
      "l22_final_output",
      "l22_final_output_source_all_oracle_inputs",
      "l22_final_output_source_gpu_qkv_oracle_modulators",
      "l22_final_output_source_oracle_qkv_gpu_modulators",
      "l22_final_output_source_oracle_qkv_gpu_gate",
      "l22_final_output_source_oracle_qkv_gpu_beta",
      "l22_final_output_source_oracle_qkv_gpu_z",
  ):
    stats = _lane(probe, name, "gpu_vs_oracle") if isinstance(probe, dict) else {}
    lines.append(
        f"| {name} | {stats.get('max_abs_diff')} | {stats.get('rmse')} |"
    )
  lines += [
      "",
      "All variants compare the GPU delta/final-norm result against captured",
      "layer22 oracle `final_output`. The live downstream path is still the",
      "diagnostic oracle-final-output bypass, so these numbers locate source",
      "drift and do not implement a production correction.",
      "",
  ]
  path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines),
                  encoding="utf-8")


def main() -> int:
  BASE.SCHEMA_VERSION = SCHEMA_VERSION
  BASE.L23_API = SOURCE_API
  BASE.layer23_full_attn_probe_cpp = diagnostic_cpp
  BASE.parse_args = parse_args
  BASE.write_summary = write_summary
  return BASE.main()


if __name__ == "__main__":
  raise SystemExit(main())
