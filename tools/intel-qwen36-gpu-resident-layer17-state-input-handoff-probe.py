#!/usr/bin/env python3
"""Run the resident GPU layer-17 state/input handoff probe."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import shlex
from pathlib import Path
from types import ModuleType
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
L16_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer16-state-input-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer17-state-input-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_l16_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer16_state_input_probe", L16_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer16 state/input tool: {L16_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L16 = load_l16_tool()
L15 = L16.L15
CORE = L16.CORE
L7_INPUT = L16.L7_INPUT
L8 = L16.L8
L11 = L16.L11
L11_FFN = L16.L11_FFN
L11_FULL = L16.L11_FULL
TWO = L16.TWO
PRECONV = L16.PRECONV


def replace_once(text: str, old: str, new: str) -> str:
  return L16.replace_once(text, old, new)


def layer17_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L16.layer16_state_input_probe_cpp(opencl_source)
  for old, new in {
      L16.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer16_state_input_load_once_run_many":
          "layer5_to_layer17_state_input_load_once_run_many",
      "layer16 state/input handoff probe expects --layer 5":
          "layer17 state/input handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer10 = args.layer + 10;
    const int layer11 = args.layer + 11;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16,
            "layer17 state/input handoff probe expects --layer 5");
''',
      '''    const int layer10 = args.layer + 10;
    const int layer11 = args.layer + 11;
    const int layer12 = args.layer + 12;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17,
            "layer17 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer10_tensors = ResolveFullAttentionTensorBundle(index, layer10);
    const auto layer11_tensors = ResolveLayerTensorBundle(index, layer11);
''',
      '''    const auto layer10_tensors = ResolveFullAttentionTensorBundle(index, layer10);
    const auto layer11_tensors = ResolveLayerTensorBundle(index, layer11);
    const auto layer12_tensors = ResolveLayerTensorBundle(index, layer12);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer10_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l15_");
    const auto layer11_oracle = LoadLayerOraclePayloads(args.payload_dir, "l16");
''',
      '''    const auto layer10_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l15_");
    const auto layer11_oracle = LoadLayerOraclePayloads(args.payload_dir, "l16");
    const auto layer12_oracle = LoadLayerOraclePayloads(args.payload_dir, "l17");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer11_run = RunResidentLinearLayerShell(
        args, index, layer11_tensors, layer11_oracle,
        layer10_tail_gpu.layer_output, layer10_tail_gpu.layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer11_run = RunResidentLinearLayerShell(
        args, index, layer11_tensors, layer11_oracle,
        layer10_tail_gpu.layer_output, layer10_tail_gpu.layer_output, rms_norm_epsilon);
    const auto layer12_run = RunResidentLinearLayerShell(
        args, index, layer12_tensors, layer12_oracle,
        layer11_run.gpu_layer_output, layer11_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer11_run.arc_selected &&
        layer11_qkv_q4_q6_boundary &&
        layer11_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''        layer11_run.arc_selected &&
        layer11_qkv_q4_q6_boundary &&
        layer11_down_q4_q6_boundary;
    const auto find_l12_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer12_run.comparisons.begin(),
              layer12_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer12_run.comparisons.end(),
                  "layer17 comparison missing: " + name);
          return *found;
        };
    const auto& layer12_residual_input = find_l12_group("residual_input");
    const auto& layer12_attn_norm = find_l12_group("attn_norm");
    const auto& layer12_qkv = find_l12_group("linear_attn_qkv_mixed");
    const auto& layer12_conv_output_raw = find_l12_group("conv_output_raw");
    const bool layer12_state_input_gpu_cpu_ok =
        ComparePassed(layer12_residual_input.gpu_vs_cpu) &&
        ComparePassed(layer12_attn_norm.gpu_vs_cpu) &&
        ComparePassed(layer12_qkv.gpu_vs_cpu) &&
        ComparePassed(layer12_conv_output_raw.gpu_vs_cpu) &&
        ComparePassed(layer12_run.conv_state_after_gpu_vs_cpu);
    const bool layer12_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer12_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer12_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer12_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer12_conv_output_raw.gpu_vs_oracle);
    const bool layer12_shapes_ok = ShapesPassed(layer12_run.shape_checks);
    const bool layer12_qkv_q4_q6_boundary =
        (layer12_tensors.qkv_tensor->type == 12 || layer12_tensors.qkv_tensor->type == 14);
    const bool layer12_down_q4_q6_boundary =
        (layer12_tensors.selected_down_tensor->type == 12 ||
         layer12_tensors.selected_down_tensor->type == 14) &&
        (layer12_tensors.shared_down_tensor->type == 12 ||
         layer12_tensors.shared_down_tensor->type == 14);
    const bool layer12_state_input_timing_positive =
        layer12_run.timing.layer_input_rmsnorm_min_us > 0.0 &&
        layer12_run.timing.preconv_to_postconv_kernel_sum_min_us > 0.0;
    const bool layer12_ok =
        layer12_shapes_ok &&
        layer12_run.payload_counts_ok &&
        layer12_state_input_gpu_cpu_ok &&
        layer12_state_input_oracle_policy_ok &&
        layer12_state_input_timing_positive &&
        layer12_run.arc_selected &&
        layer12_qkv_q4_q6_boundary &&
        layer12_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer10_ok &&
        layer11_ok &&
        layer2_timing_positive &&
''',
      '''        layer10_ok &&
        layer11_ok &&
        layer12_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer11_state_input_sum_min =
        layer11_run.timing.layer_input_rmsnorm_min_us +
        layer11_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer11_state_input_sum_min =
        layer11_run.timing.layer_input_rmsnorm_min_us +
        layer11_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer12_state_input_sum_min =
        layer12_run.timing.layer_input_rmsnorm_min_us +
        layer12_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer16_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer11_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer16_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer16_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer11_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer16_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer17_residual_input_boundary\\":\\"live_gpu_l_out_16\\",";
    std::cout << "\\"layer17_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer12_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer17_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer12_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer17_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer12_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer17_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer16_tail_device_name\\":\\"" << JsonEscape(layer11_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer16_tail_device_name\\":\\"" << JsonEscape(layer11_run.tail_device_name) << "\\",";
    std::cout << "\\"layer17_layer_input_device_name\\":\\"" << JsonEscape(layer12_run.layer_input_device_name) << "\\",";
    std::cout << "\\"layer17_preconv_device_name\\":\\"" << JsonEscape(layer12_run.preconv_device_name) << "\\",";
    std::cout << "\\"layer17_tail_device_name\\":\\"" << JsonEscape(layer12_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer10_shared_gpu.program_build_ms +
                  layer10_tail_gpu.program_build_ms +
                  layer11_run.program_build_ms)
''',
      '''                  layer10_shared_gpu.program_build_ms +
                  layer10_tail_gpu.program_build_ms +
                  layer11_run.program_build_ms +
                  layer12_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer10_shared_gpu.build_log +
                            layer10_tail_gpu.build_log +
                            layer11_run.build_log)
''',
      '''                            layer10_shared_gpu.build_log +
                            layer10_tail_gpu.build_log +
                            layer11_run.build_log +
                            layer12_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_to_layer16_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_to_layer16_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer17_state_input_kernel_sum_min_us\\":"
              << layer12_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_to_layer17_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l16", layer11_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l16", layer11_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l17", layer12_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l16_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer11_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l16_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer11_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l17_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer12_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l17_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer12_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer11\\":";
    WriteLayerChecks(layer11_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer11\\":";
    WriteLayerChecks(layer11_run);
    std::cout << ",\\"layer12\\":";
    WriteLayerChecks(layer12_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer16_state_input_handoff_matches\\":"
              << (layer11_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer16_state_input_handoff_matches\\":"
              << (layer11_ok ? "true" : "false") << ",";
    std::cout << "\\"layer17_residual_input_from_layer16_live_gpu_lout\\":true,";
    std::cout << "\\"layer17_payload_counts_ok\\":"
              << (layer12_run.payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer17_shapes_ok\\":"
              << (layer12_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer17_qkv_q4_q6_boundary\\":"
              << (layer12_qkv_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer17_down_q4_q6_boundary\\":"
              << (layer12_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer17_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer17_gpu_cpu_matches_native\\":"
              << (layer12_state_input_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer17_state_input_oracle_policy_matches\\":"
              << (layer12_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer17_state_input_handoff_matches\\":"
              << (layer12_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer10_timing_positive &&
                  layer11_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer10_timing_positive &&
                  layer11_state_input_timing_positive &&
                  layer12_state_input_timing_positive ? "true" : "false") << ",";
''',
  )
  return cpp


def iso_now() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
  return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--all-history-json", type=Path, default=DEFAULT_ALL_HISTORY)
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--resident-invocations", type=int, default=5)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--conv-history-probe", type=Path, default=None)
  parser.add_argument("--next-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer8-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer9-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer10-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer12-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer13-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer14-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer16-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer17-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "layer5_to_layer17_state_input_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer16_residual_input_boundary") == "live_gpu_l_out_15"
      and probe.get("layer17_residual_input_boundary") == "live_gpu_l_out_16"
      and probe.get("layer17_qkv_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer17_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer17_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer17_conv_state_boundary") == "captured_conv_state"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def layer16_predecessor_checks(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
  checks = L16.layer15_predecessor_checks(probe)
  checks.extend([
      {"name": "layer16_residual_input_from_layer15_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer16_residual_input_from_layer15_live_gpu_lout")},
      {"name": "layer16_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer16_payload_counts_ok")},
      {"name": "layer16_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer16_shapes_ok")},
      {"name": "layer16_qkv_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer16_qkv_q4_q6_boundary")},
      {"name": "layer16_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer16_down_q4_q6_boundary")},
      {"name": "layer16_conv_state_input_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer16_conv_state_input_boundary")},
      {"name": "layer16_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer16_gpu_cpu_matches_native")},
      {"name": "layer16_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer16_state_input_oracle_policy_matches")},
      {"name": "layer16_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer16_state_input_handoff_matches")},
      {"name": "l16_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l16_residual_input", "gpu_vs_oracle")},
      {"name": "l16_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l16_attn_norm", "gpu_vs_oracle")},
      {"name": "l16_qkv_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l16_linear_attn_qkv_mixed", "gpu_vs_oracle")},
      {"name": "l16_conv_output_raw_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l16_conv_output_raw", "gpu_vs_oracle")},
      {"name": "l16_final_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l16_final_output", "gpu_vs_oracle")},
      {"name": "l16_layer_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l16_layer_output", "gpu_vs_oracle")},
      {"name": "l16_conv_state_after_matches_native", "pass": CORE.comparison_passed(probe, "l16_conv_state_after", "gpu_vs_cpu")},
      {"name": "l16_recurrent_state_matches_native", "pass": CORE.comparison_passed(probe, "l16_recurrent_state", "gpu_vs_cpu")},
      {"name": "layer16_internal_comparisons_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer11", "comparisons_passed")},
  ])
  return checks


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-17 State/Input Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 17 residual input boundary: `{probe.get('layer17_residual_input_boundary')}`",
      f"- layer 17 qkv tensor type: `{probe.get('layer17_qkv_tensor_type')}`",
      f"- layer 17 selected/shared down: `{probe.get('layer17_selected_down_tensor_type')}` / `{probe.get('layer17_shared_down_tensor_type')}`",
      f"- layer 17 conv state boundary: `{probe.get('layer17_conv_state_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l17_residual_input",
      "l17_attn_norm",
      "l17_linear_attn_qkv_mixed",
      "l17_conv_output_raw",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(
        f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |"
    )
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer17_state_input | {timings.get('resident_layer17_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| through_layer17_state_input | {timings.get('resident_layer5_6_7_8_9_10_11_12_13_14_15_16_to_layer17_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 16, then feeds live GPU `l_out-16` into layer 17 RMSNorm, Q4/Q6",
      "QKV, and F32 conv with captured layer-17 conv state. This is captured",
      "single-token state/input evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-17 state/input handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  layer3 = args.layer + 3
  layer4 = args.layer + 4
  layer5 = args.layer + 5
  layer6 = args.layer + 6
  layer7 = args.layer + 7
  layer8 = args.layer + 8
  layer9 = args.layer + 9
  layer10 = args.layer + 10
  layer11 = args.layer + 11
  layer12 = args.layer + 12
  conv0_path = (args.conv_history_probe.resolve() if args.conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer0).resolve())
  conv1_path = (args.next_conv_history_probe.resolve() if args.next_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer1).resolve())
  conv2_path = (args.layer8_conv_history_probe.resolve() if args.layer8_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer3).resolve())
  conv3_path = (args.layer9_conv_history_probe.resolve() if args.layer9_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer4).resolve())
  conv4_path = (args.layer10_conv_history_probe.resolve() if args.layer10_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer5).resolve())
  conv5_path = (args.layer12_conv_history_probe.resolve() if args.layer12_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer7).resolve())
  conv6_path = (args.layer13_conv_history_probe.resolve() if args.layer13_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer8).resolve())
  conv7_path = (args.layer14_conv_history_probe.resolve() if args.layer14_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer9).resolve())
  conv8_path = (args.layer16_conv_history_probe.resolve() if args.layer16_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer11).resolve())
  conv9_path = (args.layer17_conv_history_probe.resolve() if args.layer17_conv_history_probe is not None
                else TWO.latest_conv_history_probe_for_layer(layer12).resolve())
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer17-state-input-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  payloads2, conv2 = TWO.prefixed_payloads(layer3, conv2_path, "l3")
  payloads3, conv3 = TWO.prefixed_payloads(layer4, conv3_path, "l4")
  payloads4, conv4 = TWO.prefixed_payloads(layer5, conv4_path, "l5")
  payloads5, conv5 = TWO.prefixed_payloads(layer7, conv5_path, "l12")
  payloads6, conv6 = TWO.prefixed_payloads(layer8, conv6_path, "l13")
  payloads7, conv7 = TWO.prefixed_payloads(layer9, conv7_path, "l14")
  payloads8, conv8 = TWO.prefixed_payloads(layer11, conv8_path, "l16")
  payloads9, conv9 = TWO.prefixed_payloads(layer12, conv9_path, "l17")
  layer7_payloads, layer7_history = L7_INPUT.resolve_full_attention_payloads(
      all_history_json, layer2
  )
  CORE.add_layer7_tail_payloads(layer7_payloads)
  L8.add_layer7_ffn_payloads(layer7_payloads)
  layer11_payloads, layer11_history = L11.resolve_prefixed_full_attention_payloads(
      all_history_json, layer6, "l6_"
  )
  L11_FFN.add_layer11_ffn_payloads(layer11_payloads)
  layer15_payloads, layer15_history = L11_FULL.resolve_prefixed_full_attention_payloads(
      all_history_json, layer10, "l15_"
  )
  L15.add_layer15_ffn_payloads(layer15_payloads)
  payloads = {
      **payloads0,
      **payloads1,
      **layer7_payloads,
      **payloads2,
      **payloads3,
      **payloads4,
      **layer11_payloads,
      **payloads5,
      **payloads6,
      **payloads7,
      **layer15_payloads,
      **payloads8,
      **payloads9,
  }
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (
          opencl_source
          + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL
          + CORE.FULL_ATTN_CORE_EXTRA_OPENCL
      ).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer17_state_input_handoff_probe.cpp"
  local_cpp.write_text(layer17_state_input_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer17-state-input-handoff-probe-{stamp}"
  setup = iq36_local.run_target(
      args.host,
      "rm -rf " + shlex.quote(remote_dir) + " && mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      name: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for name in payloads
  }
  remote_payload_dir = f"{remote_dir}/oracle"
  if setup.get("returncode") == 0:
    for local, remote in PRECONV.POSTCONV.ATTENTION.SHARED.SELECTED.SOURCE_FILES:
      transfers.append(
          iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/gpu_resident_layer17_state_input_handoff_probe.cpp",
            args.timeout_s,
        )
    )
    for name, payload in payloads.items():
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer17-state-input-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer17_state_input_handoff_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(executable)}"
      ),
  ])
  stage_ok = (
      setup.get("returncode") == 0
      and transfers
      and all(item.get("returncode") == 0 for item in transfers)
      and all(item.get("returncode") == 0 for item in payload_transfers.values())
  )
  compile_result = (
      iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
      if stage_ok
      else {"cmd": ["stage"], "returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_argv = [
      executable,
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
      "--repeat", str(args.resident_invocations),
      "--device-substring", args.device_substring,
  ]
  run_result = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
              PRECONV.shell_join(run_argv),
          ]),
          args.timeout_s,
      )
      if compile_result.get("returncode") == 0
      else {"cmd": run_argv, "returncode": None, "stdout": "", "stderr": "compile skipped run"}
  )
  probe = PRECONV.parse_probe_stdout(run_result.get("stdout", ""))

  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "payload-transfers.json", payload_transfers)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  if probe is not None:
    iq36_local.write_json(out_dir / "probe-result.json", probe)

  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {"name": "source_files_transferred", "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers)},
      {"name": "oracle_payloads_transferred", "pass": all(item.get("returncode") == 0 for item in payload_transfers.values())},
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")) and "B390" in str(probe.get("layer17_preconv_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
  ]
  checks.extend(layer16_predecessor_checks(probe))
  checks.extend([
      {"name": "layer17_residual_input_from_layer16_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer17_residual_input_from_layer16_live_gpu_lout")},
      {"name": "layer17_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer17_payload_counts_ok")},
      {"name": "layer17_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer17_shapes_ok")},
      {"name": "layer17_qkv_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer17_qkv_q4_q6_boundary")},
      {"name": "layer17_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer17_down_q4_q6_boundary")},
      {"name": "layer17_conv_state_input_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer17_conv_state_input_boundary")},
      {"name": "layer17_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer17_gpu_cpu_matches_native")},
      {"name": "layer17_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer17_state_input_oracle_policy_matches")},
      {"name": "layer17_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer17_state_input_handoff_matches")},
      {"name": "l17_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l17_residual_input", "gpu_vs_oracle")},
      {"name": "l17_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l17_attn_norm", "gpu_vs_oracle")},
      {"name": "l17_qkv_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l17_linear_attn_qkv_mixed", "gpu_vs_oracle")},
      {"name": "l17_conv_output_raw_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l17_conv_output_raw", "gpu_vs_oracle")},
      {"name": "l17_conv_state_after_matches_native", "pass": CORE.comparison_passed(probe, "l17_conv_state_after", "gpu_vs_cpu")},
      {"name": "l17_recurrent_state_matches_native", "pass": CORE.comparison_passed(probe, "l17_recurrent_state", "gpu_vs_cpu")},
      {"name": "gpu_event_timing_positive", "pass": PRECONV.nested_bool(probe, "checks", "gpu_event_timing_positive")},
      {"name": "speedup_claims_forbidden", "pass": True},
  ])
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  comparison_thresholds = {
      "strict_component": CORE.STRICT_COMPARISON_THRESHOLDS,
      "full_attn_component": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer15_full_shell_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer16_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer17_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "oracle_bundle": str(args.oracle_bundle.resolve().relative_to(ROOT)),
      "conv_history_probes": {
          "layer0": str(conv0_path.relative_to(ROOT)),
          "layer1": str(conv1_path.relative_to(ROOT)),
          "layer3": str(conv2_path.relative_to(ROOT)),
          "layer4": str(conv3_path.relative_to(ROOT)),
          "layer5": str(conv4_path.relative_to(ROOT)),
          "layer7": str(conv5_path.relative_to(ROOT)),
          "layer8": str(conv6_path.relative_to(ROOT)),
          "layer9": str(conv7_path.relative_to(ROOT)),
          "layer11": str(conv8_path.relative_to(ROOT)),
          "layer12": str(conv9_path.relative_to(ROOT)),
          "layer13": str(conv6_path.relative_to(ROOT)),
          "layer14": str(conv7_path.relative_to(ROOT)),
          "layer16": str(conv8_path.relative_to(ROOT)),
          "layer17": str(conv9_path.relative_to(ROOT)),
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
          "layer3": conv2.get("capture_artifact"),
          "layer4": conv3.get("capture_artifact"),
          "layer5": conv4.get("capture_artifact"),
          "layer7": conv5.get("capture_artifact"),
          "layer8": conv6.get("capture_artifact"),
          "layer9": conv7.get("capture_artifact"),
          "layer11": conv8.get("capture_artifact"),
          "layer12": conv9.get("capture_artifact"),
          "layer13": conv6.get("capture_artifact"),
          "layer14": conv7.get("capture_artifact"),
          "layer16": conv8.get("capture_artifact"),
          "layer17": conv9.get("capture_artifact"),
      },
      "all_history": {
          "layer7": layer7_history,
          "layer11": layer11_history,
          "layer15": layer15_history,
      },
      "ffn_payload_root": str(L15.LAYER15_FFN_PAYLOAD_ROOT.relative_to(ROOT)),
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11, layer12],
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "embedded_opencl_source_sha256": embedded_opencl_hash,
      "comparison_thresholds": comparison_thresholds,
      "probe_extra_opencl": [
          "rms_norm_hidden_f32",
          "full_attn_core_f32",
          "full_attn_gate_f32",
          "q6k_selected_down_matvec_row",
      ],
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-layer17-state-input-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": payload["layers"],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
      "all_history": payload["all_history"],
      "comparison_thresholds": comparison_thresholds,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "comparison_thresholds": comparison_thresholds,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  iq36_local.write_json(out_dir / "probe.json", payload)
  iq36_local.write_json(out_dir / "manifest.json", manifest)
  iq36_local.write_json(out_dir / "correctness.json", correctness)

  aggregate = probe if isinstance(probe, dict) else {}
  timings = aggregate.get("timings", {}) if isinstance(aggregate.get("timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_resident_layer17_state_input_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer17_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer17_state_input_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_to_layer17_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_to_layer17_state_input_kernel_sum_min_us")),
          ("layer17_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l17_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer17_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l17_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer17_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l17_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer17_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l17_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
