#!/usr/bin/env python3
"""Run the resident GPU layer-20 state/input handoff probe."""

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
L19_FFN_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer19-ffn-lout-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer20-state-input-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
def load_l19_ffn_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer19_ffn_lout_probe", L19_FFN_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer19 FFN/l_out tool: {L19_FFN_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L19 = load_l19_ffn_tool()
L18 = L19.L18
L17 = L19.L17
L16 = L19.L16
L15 = L19.L15
CORE = L19.CORE
L7_INPUT = L19.L7_INPUT
L8 = L19.L8
L11 = L19.L11
L11_FFN = L19.L11_FFN
L11_FULL = L19.L11_FULL
TWO = L19.TWO
PRECONV = L19.PRECONV


def replace_once(text: str, old: str, new: str) -> str:
  return L19.replace_once(text, old, new)


def layer20_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L19.layer19_ffn_lout_probe_cpp(opencl_source)
  for old, new in {
      L19.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer19_ffn_lout_load_once_run_many":
          "layer5_to_layer20_state_input_load_once_run_many",
      "layer19 full-attention handoff probe expects --layer 5":
          "layer20 state/input handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer13 = args.layer + 13;
    const int layer14 = args.layer + 14;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18 && layer14 == 19,
            "layer20 state/input handoff probe expects --layer 5");
''',
      '''    const int layer13 = args.layer + 13;
    const int layer14 = args.layer + 14;
    const int layer15 = args.layer + 15;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15 && layer11 == 16 && layer12 == 17 && layer13 == 18 && layer14 == 19 && layer15 == 20,
            "layer20 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer13_tensors = ResolveLayerTensorBundle(index, layer13);
    const auto layer14_tensors = ResolveFullAttentionTensorBundle(index, layer14);
''',
      '''    const auto layer13_tensors = ResolveLayerTensorBundle(index, layer13);
    const auto layer14_tensors = ResolveFullAttentionTensorBundle(index, layer14);
    const auto layer15_tensors = ResolveLayerTensorBundle(index, layer15);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer13_oracle = LoadLayerOraclePayloads(args.payload_dir, "l18");
    const auto layer14_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l19_");
''',
      '''    const auto layer13_oracle = LoadLayerOraclePayloads(args.payload_dir, "l18");
    const auto layer14_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l19_");
    const auto layer15_oracle = LoadLayerOraclePayloads(args.payload_dir, "l20");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer14_tail_gpu = RunGpuShell(
        layer14_shared_input_gate_weights, layer14_ffn_input,
        layer14_selected_gpu.down, layer14_oracle_weights_norm,
        layer14_shared_gpu.down, layer14_attention_gpu.attn_residual,
        args.device_substring, args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer14_tail_gpu = RunGpuShell(
        layer14_shared_input_gate_weights, layer14_ffn_input,
        layer14_selected_gpu.down, layer14_oracle_weights_norm,
        layer14_shared_gpu.down, layer14_attention_gpu.attn_residual,
        args.device_substring, args.repeat);

    const auto layer15_run = RunResidentLinearLayerShell(
        args, index, layer15_tensors, layer15_oracle,
        layer14_tail_gpu.layer_output, layer14_tail_gpu.layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer14_v_gpu_boundary &&
        layer14_ffn_tensor_shapes_ok &&
        layer14_ffn_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''        layer14_v_gpu_boundary &&
        layer14_ffn_tensor_shapes_ok &&
        layer14_ffn_down_q4_q6_boundary;
    const auto find_l15_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer15_run.comparisons.begin(),
              layer15_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer15_run.comparisons.end(),
                  "layer20 comparison missing: " + name);
          return *found;
        };
    const auto& layer15_residual_input = find_l15_group("residual_input");
    const auto& layer15_attn_norm = find_l15_group("attn_norm");
    const auto& layer15_qkv = find_l15_group("linear_attn_qkv_mixed");
    const auto& layer15_conv_output_raw = find_l15_group("conv_output_raw");
    const bool layer15_state_input_gpu_cpu_ok =
        ComparePassed(layer15_residual_input.gpu_vs_cpu) &&
        ComparePassed(layer15_attn_norm.gpu_vs_cpu) &&
        ComparePassed(layer15_qkv.gpu_vs_cpu) &&
        ComparePassed(layer15_conv_output_raw.gpu_vs_cpu) &&
        ComparePassed(layer15_run.conv_state_after_gpu_vs_cpu);
    const bool layer15_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer15_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer15_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer15_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer15_conv_output_raw.gpu_vs_oracle);
    const bool layer15_shapes_ok = ShapesPassed(layer15_run.shape_checks);
    const bool layer15_qkv_q4_q6_boundary =
        (layer15_tensors.qkv_tensor->type == 12 || layer15_tensors.qkv_tensor->type == 14);
    const bool layer15_down_q4_q6_boundary =
        (layer15_tensors.selected_down_tensor->type == 12 ||
         layer15_tensors.selected_down_tensor->type == 14) &&
        (layer15_tensors.shared_down_tensor->type == 12 ||
         layer15_tensors.shared_down_tensor->type == 14);
    const bool layer15_state_input_timing_positive =
        layer15_run.timing.layer_input_rmsnorm_min_us > 0.0 &&
        layer15_run.timing.preconv_to_postconv_kernel_sum_min_us > 0.0;
    const bool layer15_ok =
        layer15_shapes_ok &&
        layer15_run.payload_counts_ok &&
        layer15_state_input_gpu_cpu_ok &&
        layer15_state_input_oracle_policy_ok &&
        layer15_state_input_timing_positive &&
        layer15_run.arc_selected &&
        layer15_qkv_q4_q6_boundary &&
        layer15_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer13_ok &&
        layer14_ok &&
        layer2_timing_positive &&
''',
      '''        layer13_ok &&
        layer14_ok &&
        layer15_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer14_ffn_sum_min =
        layer14_selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        layer14_shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        layer14_tail_gpu.timing.shell_sum_min_us;
''',
      '''    const double layer14_ffn_sum_min =
        layer14_selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        layer14_shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        layer14_tail_gpu.timing.shell_sum_min_us;
    const double layer15_state_input_sum_min =
        layer15_run.timing.layer_input_rmsnorm_min_us +
        layer15_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "," << layer14 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "," << layer11 << "," << layer12 << "," << layer13 << "," << layer14 << "," << layer15 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer19_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer14_shared_down_tensor->type)) << "\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer19_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer14_shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer20_residual_input_boundary\\":\\"live_gpu_l_out_19\\",";
    std::cout << "\\"layer20_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer15_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer20_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer15_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer20_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer15_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer20_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer19_lout_device_name\\":\\"" << JsonEscape(layer14_tail_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer19_lout_device_name\\":\\"" << JsonEscape(layer14_tail_gpu.device_name) << "\\",";
    std::cout << "\\"layer20_layer_input_device_name\\":\\"" << JsonEscape(layer15_run.layer_input_device_name) << "\\",";
    std::cout << "\\"layer20_preconv_device_name\\":\\"" << JsonEscape(layer15_run.preconv_device_name) << "\\",";
    std::cout << "\\"layer20_tail_device_name\\":\\"" << JsonEscape(layer15_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer14_selected_gpu.program_build_ms +
                  layer14_shared_gpu.program_build_ms +
                  layer14_tail_gpu.program_build_ms)
''',
      '''                  layer14_selected_gpu.program_build_ms +
                  layer14_shared_gpu.program_build_ms +
                  layer14_tail_gpu.program_build_ms +
                  layer15_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer14_selected_gpu.build_log +
                            layer14_shared_gpu.build_log +
                            layer14_tail_gpu.build_log)
''',
      '''                            layer14_selected_gpu.build_log +
                            layer14_shared_gpu.build_log +
                            layer14_tail_gpu.build_log +
                            layer15_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_lout_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_lout_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min) << ",";
    std::cout << "\\"resident_layer20_state_input_kernel_sum_min_us\\":"
              << layer15_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_to_layer20_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min +
                  layer10_ffn_sum_min + layer11_state_input_sum_min +
                  layer12_state_input_sum_min + layer13_state_input_sum_min +
                  layer14_attention_sum_min + layer14_ffn_sum_min +
                  layer15_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteNamedCompareGroups(layer14_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer14_ffn_live_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WriteNamedCompareGroups(layer14_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer14_ffn_live_groups);
    WritePrefixedCompareGroups("l20", layer15_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l18_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer13_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l18_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer13_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l20_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer15_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l20_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer15_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer14_full_attention_shapes\\":";
    WriteFullAttentionShapeChecks(layer14_full_shapes);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer14_full_attention_shapes\\":";
    WriteFullAttentionShapeChecks(layer14_full_shapes);
    std::cout << ",\\"layer15\\":";
    WriteLayerChecks(layer15_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer19_arc_device_selected\\":"
              << (layer14_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer19_arc_device_selected\\":"
              << (layer14_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer20_residual_input_from_layer19_live_gpu_lout\\":true,";
    std::cout << "\\"layer20_payload_counts_ok\\":"
              << (layer15_run.payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer20_shapes_ok\\":"
              << (layer15_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer20_qkv_q4_q6_boundary\\":"
              << (layer15_qkv_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer20_down_q4_q6_boundary\\":"
              << (layer15_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer20_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer20_gpu_cpu_matches_native\\":"
              << (layer15_state_input_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer20_state_input_oracle_policy_matches\\":"
              << (layer15_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer20_state_input_handoff_matches\\":"
              << (layer15_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer12_state_input_timing_positive &&
                  layer13_state_input_timing_positive &&
                  layer14_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer12_state_input_timing_positive &&
                  layer13_state_input_timing_positive &&
                  layer14_timing_positive &&
                  layer15_state_input_timing_positive ? "true" : "false") << ",";
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
  parser.add_argument("--layer18-conv-history-probe", type=Path, default=None)
  parser.add_argument("--layer20-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(
    probe: dict[str, Any] | None,
    expected_invocations: int,
    device_substring: str,
) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "layer5_to_layer20_state_input_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer19_lout_boundary") == "live_gpu_l_out_19"
      and device_substring in str(probe.get("layer19_lout_device_name", ""))
      and probe.get("layer20_residual_input_boundary") == "live_gpu_l_out_19"
      and probe.get("layer20_qkv_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer20_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer20_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer20_conv_state_boundary") == "captured_conv_state"
      and device_substring in str(probe.get("layer20_layer_input_device_name", ""))
      and device_substring in str(probe.get("layer20_preconv_device_name", ""))
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def layer18_predecessor_checks(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
  checks = L18.layer17_predecessor_checks(probe)
  checks.extend([
      {"name": "layer18_residual_input_from_layer17_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer18_residual_input_from_layer17_live_gpu_lout")},
      {"name": "layer18_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer18_payload_counts_ok")},
      {"name": "layer18_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer18_shapes_ok")},
      {"name": "layer18_qkv_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer18_qkv_q4_q6_boundary")},
      {"name": "layer18_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer18_down_q4_q6_boundary")},
      {"name": "layer18_conv_state_input_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer18_conv_state_input_boundary")},
      {"name": "layer18_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer18_gpu_cpu_matches_native")},
      {"name": "layer18_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer18_state_input_oracle_policy_matches")},
      {"name": "layer18_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer18_state_input_handoff_matches")},
      {"name": "l18_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_residual_input", "gpu_vs_oracle")},
      {"name": "l18_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_attn_norm", "gpu_vs_oracle")},
      {"name": "l18_qkv_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_linear_attn_qkv_mixed", "gpu_vs_oracle")},
      {"name": "l18_conv_output_raw_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_conv_output_raw", "gpu_vs_oracle")},
      {"name": "l18_final_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_final_output", "gpu_vs_oracle")},
      {"name": "l18_layer_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l18_layer_output", "gpu_vs_oracle")},
      {"name": "l18_conv_state_after_matches_native", "pass": CORE.comparison_passed(probe, "l18_conv_state_after", "gpu_vs_cpu")},
      {"name": "l18_recurrent_state_matches_native", "pass": CORE.comparison_passed(probe, "l18_recurrent_state", "gpu_vs_cpu")},
      {"name": "layer18_internal_comparisons_passed", "pass": PRECONV.nested_bool(probe, "checks", "layer13", "comparisons_passed")},
  ])
  return checks


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-20 State/Input Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 20 residual input boundary: `{probe.get('layer20_residual_input_boundary')}`",
      f"- layer 20 qkv tensor type: `{probe.get('layer20_qkv_tensor_type')}`",
      f"- layer 20 selected/shared down: `{probe.get('layer20_selected_down_tensor_type')}` / `{probe.get('layer20_shared_down_tensor_type')}`",
      f"- layer 20 conv state boundary: `{probe.get('layer20_conv_state_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l20_residual_input",
      "l20_attn_norm",
      "l20_linear_attn_qkv_mixed",
      "l20_conv_output_raw",
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
      f"| layer20_state_input | {timings.get('resident_layer20_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| through_layer20_state_input | {timings.get('resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_to_layer20_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 19, then feeds live GPU `l_out-19` into layer 20 RMSNorm, Q4/Q6",
      "QKV, and F32 conv with captured layer-20 conv state. This is captured",
      "single-token state/input evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-20 state/input handoff")

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
  layer13 = args.layer + 13
  layer14 = args.layer + 14
  layer15 = args.layer + 15
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
  conv10_path = (args.layer18_conv_history_probe.resolve() if args.layer18_conv_history_probe is not None
                 else TWO.latest_conv_history_probe_for_layer(layer13).resolve())
  conv11_path = (args.layer20_conv_history_probe.resolve() if args.layer20_conv_history_probe is not None
                 else TWO.latest_conv_history_probe_for_layer(layer15).resolve())
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer20-state-input-handoff-probe-{stamp}"
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
  payloads10, conv10 = TWO.prefixed_payloads(layer13, conv10_path, "l18")
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
  layer19_payloads, layer19_history = L11_FULL.resolve_prefixed_full_attention_payloads(
      all_history_json, layer14, "l19_"
  )
  L19.add_layer19_ffn_payloads(layer19_payloads)
  payloads11, conv11 = TWO.prefixed_payloads(layer15, conv11_path, "l20")
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
      **payloads10,
      **layer19_payloads,
      **payloads11,
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
  local_cpp = out_dir / "gpu_resident_layer20_state_input_handoff_probe.cpp"
  local_cpp.write_text(layer20_state_input_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer20-state-input-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer20_state_input_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer20-state-input-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer20_state_input_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and args.device_substring in str(probe.get("device_name", "")) and args.device_substring in str(probe.get("layer20_preconv_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations, args.device_substring)},
  ]
  checks.extend(layer18_predecessor_checks(probe))
  checks.extend([
      {"name": "layer19_residual_input_from_layer18_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer19_residual_input_from_layer18_live_gpu_lout")},
      {"name": "layer19_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer19_payload_counts_ok")},
      {"name": "layer19_full_attn_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer19_full_attn_input_matches_oracle")},
      {"name": "layer19_full_attn_component_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer19_full_attn_component_policy_matches_oracle")},
      {"name": "layer19_full_attn_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer19_full_attn_core_output_matches_oracle")},
      {"name": "layer19_v_gpu_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer19_v_gpu_boundary")},
      {"name": "layer19_ffn_tensor_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer19_ffn_tensor_shapes_ok")},
      {"name": "layer19_ffn_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer19_ffn_down_q4_q6_boundary")},
      {"name": "layer19_live_ffn_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer19_live_ffn_gpu_cpu_matches_native")},
      {"name": "layer19_live_ffn_oracle_magnitude_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer19_live_ffn_oracle_magnitude_policy_matches")},
      {"name": "layer19_live_layer_output_full_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer19_live_layer_output_full_policy_matches_oracle")},
      {"name": "layer19_ffn_lout_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer19_ffn_lout_handoff_matches")},
      {"name": "layer19_lout_boundary_live_gpu", "pass": PRECONV.nested_bool(probe, "checks", "layer19_lout_boundary_live_gpu")},
      {"name": "layer19_ffn_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer19_ffn_q6_boundary")},
      {"name": "layer19_arc_device_selected", "pass": PRECONV.nested_bool(probe, "checks", "layer19_arc_device_selected")},
      {"name": "l19_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_residual_input", "gpu_vs_oracle")},
      {"name": "l19_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_norm", "gpu_vs_oracle")},
      {"name": "l19_q_full_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_q_full", "gpu_vs_oracle")},
      {"name": "l19_q_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_q_rope", "gpu_vs_oracle")},
      {"name": "l19_k_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_k_rope", "gpu_vs_oracle")},
      {"name": "l19_v_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_v", "gpu_vs_oracle")},
      {"name": "l19_attn_pregate_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_pregate", "gpu_vs_oracle")},
      {"name": "l19_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_output", "gpu_vs_oracle")},
      {"name": "l19_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_attn_post_norm", "gpu_vs_oracle")},
      {"name": "l19_selected_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l19_selected_down", "gpu_vs_oracle")},
      {"name": "l19_shared_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l19_shared_down", "gpu_vs_oracle")},
      {"name": "l19_ffn_out_matches_oracle", "pass": CORE.comparison_passed(probe, "l19_ffn_out", "gpu_vs_oracle")},
      {"name": "l19_layer_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l19_layer_output", "gpu_vs_oracle")},
      {"name": "layer20_residual_input_from_layer19_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer20_residual_input_from_layer19_live_gpu_lout")},
      {"name": "layer20_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer20_payload_counts_ok")},
      {"name": "layer20_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer20_shapes_ok")},
      {"name": "layer20_qkv_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer20_qkv_q4_q6_boundary")},
      {"name": "layer20_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer20_down_q4_q6_boundary")},
      {"name": "layer20_conv_state_input_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer20_conv_state_input_boundary")},
      {"name": "layer20_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer20_gpu_cpu_matches_native")},
      {"name": "layer20_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer20_state_input_oracle_policy_matches")},
      {"name": "layer20_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer20_state_input_handoff_matches")},
      {"name": "l20_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l20_residual_input", "gpu_vs_oracle")},
      {"name": "l20_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l20_attn_norm", "gpu_vs_oracle")},
      {"name": "l20_qkv_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l20_linear_attn_qkv_mixed", "gpu_vs_oracle")},
      {"name": "l20_conv_output_raw_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l20_conv_output_raw", "gpu_vs_oracle")},
      {"name": "l20_conv_state_after_matches_native", "pass": CORE.comparison_passed(probe, "l20_conv_state_after", "gpu_vs_cpu")},
      {"name": "l20_recurrent_state_matches_native", "pass": CORE.comparison_passed(probe, "l20_recurrent_state", "gpu_vs_cpu")},
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
      "layer18_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer19_full_attention_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer20_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "live_ffn_oracle_magnitude": {
          "max_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["max_abs_diff"],
          "rmse": CORE.STRICT_COMPARISON_THRESHOLDS["rmse"],
          "mismatch_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["mismatch_abs_diff"],
      },
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
          "layer18": str(conv10_path.relative_to(ROOT)),
          "layer20": str(conv11_path.relative_to(ROOT)),
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
          "layer18": conv10.get("capture_artifact"),
          "layer20": conv11.get("capture_artifact"),
      },
      "all_history": {
          "layer7": layer7_history,
          "layer11": layer11_history,
          "layer15": layer15_history,
          "layer19": layer19_history,
      },
      "ffn_payload_root": str(L19.LAYER19_FFN_PAYLOAD_ROOT.relative_to(ROOT)),
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11, layer12, layer13, layer14, layer15],
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
      "tool": "tools/intel-qwen36-gpu-resident-layer20-state-input-handoff-probe.py",
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
      "gpu_resident_layer20_state_input_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer19_full_attn_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attn_input_kernel_sum_min_us")),
          ("resident_layer19_full_attn_core_output_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attn_core_output_kernel_sum_min_us")),
          ("resident_layer19_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_full_attention_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_full_attention_kernel_sum_min_us")),
          ("resident_layer19_ffn_lout_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer19_ffn_lout_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_lout_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_to_layer19_lout_kernel_sum_min_us")),
          ("resident_layer20_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer20_state_input_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_to_layer20_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_to_layer20_state_input_kernel_sum_min_us")),
          ("layer19_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_q_full_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_q_full", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer19_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_selected_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_shared_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_shared_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_ffn_out_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_ffn_out", "gpu_vs_oracle", "max_abs_diff")),
          ("l19_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l19_layer_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer20_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l20_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
