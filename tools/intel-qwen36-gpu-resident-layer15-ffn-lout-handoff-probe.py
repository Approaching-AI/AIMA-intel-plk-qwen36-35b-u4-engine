#!/usr/bin/env python3
"""Run the resident GPU layer-15 FFN/l_out handoff probe."""

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
L14_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer14-state-input-handoff-probe.py"
)
L11_FULL_ATTN_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer11-full-attn-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer15-ffn-lout-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
LAYER15_FFN_PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"

LAYER15_FFN_PAYLOAD_SPECS = {
    "l15_ffn_topk": ("l15_ffn_moe_topk.bin", "ffn_moe_topk-15__tok15__ord*.bin", 32),
    "l15_ffn_weights_norm": (
        "l15_ffn_moe_weights_norm.bin",
        "ffn_moe_weights_norm-15__tok15__ord*.bin",
        32,
    ),
    "l15_ffn_gate_up": (
        "l15_ffn_moe_gate_up.bin",
        "ffn_moe_gate_up-15__tok15__ord*.bin",
        32768,
    ),
    "l15_ffn_swiglu": (
        "l15_ffn_moe_swiglu.bin",
        "ffn_moe_swiglu-15__tok15__ord*.bin",
        16384,
    ),
    "l15_ffn_down": ("l15_ffn_moe_down.bin", "ffn_moe_down-15__tok15__ord*.bin", 65536),
    "l15_ffn_weighted": (
        "l15_ffn_moe_weighted.bin",
        "ffn_moe_weighted-15__tok15__ord*.bin",
        65536,
    ),
    "l15_ffn_moe_out": (
        "l15_ffn_moe_out.bin",
        "ffn_moe_out-15__tok15__ord*.bin",
        8192,
    ),
    "l15_ffn_shexp": ("l15_ffn_shexp.bin", "ffn_shexp-15__tok15__ord*.bin", 8192),
    "l15_shared_gate": (
        "l15_shared_expert_gate.bin",
        "shared_expert_gate-15__tok15__ord*.bin",
        4,
    ),
    "l15_shared_gate_sigmoid": (
        "l15_shared_expert_gate_sigmoid.bin",
        "shared_expert_gate_sigmoid-15__tok15__ord*.bin",
        4,
    ),
    "l15_ffn_shexp_gated": (
        "l15_ffn_shexp_gated.bin",
        "ffn_shexp_gated-15__tok15__ord*.bin",
        8192,
    ),
    "l15_ffn_out": ("l15_ffn_out.bin", "ffn_out-15__tok15__ord*.bin", 8192),
    "l15_l_out": ("l15_l_out.bin", "l_out-15__tok15__ord*.bin", 8192),
}


def load_l14_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer14_state_input_probe", L14_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer14 state/input tool: {L14_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def load_l11_full_attn_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer11_full_attn_probe", L11_FULL_ATTN_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer11 full-attn tool: {L11_FULL_ATTN_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L14 = load_l14_tool()
L11_FULL = load_l11_full_attn_tool()
L11_FFN = L14.L11_FFN
L11 = L14.L11
CORE = L14.CORE
L7_INPUT = L14.L7_INPUT
L8 = L14.L8
TWO = L14.TWO
PRECONV = L14.PRECONV


def replace_once(text: str, old: str, new: str) -> str:
  return L14.replace_once(text, old, new)


def find_layer15_ffn_payload(pattern: str, expected_bytes: int) -> Path:
  matches = sorted(LAYER15_FFN_PAYLOAD_ROOT.glob(pattern))
  if len(matches) != 1:
    raise SystemExit(f"expected one layer15 FFN payload for {pattern}, found {len(matches)}")
  path = matches[0].resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}: {path.stat().st_size}")
  return path


def payload_record(path: Path, stage_name: str, expected_bytes: int) -> dict[str, Any]:
  path = path.resolve()
  if path.stat().st_size != expected_bytes:
    raise SystemExit(f"payload size mismatch for {path}: {path.stat().st_size}")
  return {
      "local_path": path,
      "path": str(path.relative_to(ROOT)),
      "sha256": iq36_local.sha256_file(path),
      "size_bytes": expected_bytes,
      "stage_name": stage_name,
  }


def add_layer15_ffn_payloads(payloads: dict[str, dict[str, Any]]) -> None:
  for name, (stage_name, pattern, expected_bytes) in LAYER15_FFN_PAYLOAD_SPECS.items():
    payloads[name] = payload_record(
        find_layer15_ffn_payload(pattern, expected_bytes),
        stage_name,
        expected_bytes,
    )


def layer13_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L12.layer12_state_input_probe_cpp(opencl_source)
  for old, new in {
      L12.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer12_state_input_load_once_run_many":
          "layer5_to_layer13_state_input_load_once_run_many",
      "layer12 state/input handoff probe expects --layer 5":
          "layer13 state/input handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer6 = args.layer + 6;
    const int layer7 = args.layer + 7;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12,
            "layer13 state/input handoff probe expects --layer 5");
''',
      '''    const int layer6 = args.layer + 6;
    const int layer7 = args.layer + 7;
    const int layer8 = args.layer + 8;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13,
            "layer13 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer7_tensors = ResolveLayerTensorBundle(index, layer7);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer7_tensors = ResolveLayerTensorBundle(index, layer7);
    const auto layer8_tensors = ResolveLayerTensorBundle(index, layer8);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer7_oracle = LoadLayerOraclePayloads(args.payload_dir, "l12");
    const auto oracle_attn_residual =
''',
      '''    const auto layer7_oracle = LoadLayerOraclePayloads(args.payload_dir, "l12");
    const auto layer8_oracle = LoadLayerOraclePayloads(args.payload_dir, "l13");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer7_run = RunResidentLinearLayerShell(
        args, index, layer7_tensors, layer7_oracle,
        layer6_tail_gpu.layer_output, layer6_tail_gpu.layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer7_run = RunResidentLinearLayerShell(
        args, index, layer7_tensors, layer7_oracle,
        layer6_tail_gpu.layer_output, layer6_tail_gpu.layer_output, rms_norm_epsilon);
    const auto layer8_run = RunResidentLinearLayerShell(
        args, index, layer8_tensors, layer8_oracle,
        layer7_run.gpu_layer_output, layer7_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer7_ok =
        layer7_shapes_ok &&
        layer7_run.payload_counts_ok &&
        layer7_state_input_gpu_cpu_ok &&
        layer7_state_input_oracle_policy_ok &&
        layer7_state_input_timing_positive &&
        layer7_run.arc_selected &&
        layer7_qkv_q4_q6_boundary &&
        layer7_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer7_ok =
        layer7_shapes_ok &&
        layer7_run.payload_counts_ok &&
        layer7_state_input_gpu_cpu_ok &&
        layer7_state_input_oracle_policy_ok &&
        layer7_state_input_timing_positive &&
        layer7_run.arc_selected &&
        layer7_qkv_q4_q6_boundary &&
        layer7_down_q4_q6_boundary;
    const auto find_l8_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer8_run.comparisons.begin(),
              layer8_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer8_run.comparisons.end(),
                  "layer13 comparison missing: " + name);
          return *found;
        };
    const auto& layer8_residual_input = find_l8_group("residual_input");
    const auto& layer8_attn_norm = find_l8_group("attn_norm");
    const auto& layer8_qkv = find_l8_group("linear_attn_qkv_mixed");
    const auto& layer8_conv_output_raw = find_l8_group("conv_output_raw");
    const bool layer8_state_input_gpu_cpu_ok =
        ComparePassed(layer8_residual_input.gpu_vs_cpu) &&
        ComparePassed(layer8_attn_norm.gpu_vs_cpu) &&
        ComparePassed(layer8_qkv.gpu_vs_cpu) &&
        ComparePassed(layer8_conv_output_raw.gpu_vs_cpu) &&
        ComparePassed(layer8_run.conv_state_after_gpu_vs_cpu);
    const bool layer8_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer8_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer8_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer8_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer8_conv_output_raw.gpu_vs_oracle);
    const bool layer8_shapes_ok = ShapesPassed(layer8_run.shape_checks);
    const bool layer8_qkv_q4_q6_boundary =
        (layer8_tensors.qkv_tensor->type == 12 || layer8_tensors.qkv_tensor->type == 14);
    const bool layer8_down_q4_q6_boundary =
        (layer8_tensors.selected_down_tensor->type == 12 ||
         layer8_tensors.selected_down_tensor->type == 14) &&
        (layer8_tensors.shared_down_tensor->type == 12 ||
         layer8_tensors.shared_down_tensor->type == 14);
    const bool layer8_state_input_timing_positive =
        layer8_run.timing.layer_input_rmsnorm_min_us > 0.0 &&
        layer8_run.timing.preconv_to_postconv_kernel_sum_min_us > 0.0;
    const bool layer8_ok =
        layer8_shapes_ok &&
        layer8_run.payload_counts_ok &&
        layer8_state_input_gpu_cpu_ok &&
        layer8_state_input_oracle_policy_ok &&
        layer8_state_input_timing_positive &&
        layer8_run.arc_selected &&
        layer8_qkv_q4_q6_boundary &&
        layer8_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer5_ok &&
        layer6_ok &&
        layer7_ok &&
        layer2_timing_positive &&
''',
      '''        layer5_ok &&
        layer6_ok &&
        layer7_ok &&
        layer8_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer7_state_input_sum_min =
        layer7_run.timing.layer_input_rmsnorm_min_us +
        layer7_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer7_state_input_sum_min =
        layer7_run.timing.layer_input_rmsnorm_min_us +
        layer7_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer8_state_input_sum_min =
        layer8_run.timing.layer_input_rmsnorm_min_us +
        layer8_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer12_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer12_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer13_residual_input_boundary\\":\\"live_gpu_l_out_12\\",";
    std::cout << "\\"layer13_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer8_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer13_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer8_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer13_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer8_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer13_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer12_tail_device_name\\":\\"" << JsonEscape(layer7_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer12_tail_device_name\\":\\"" << JsonEscape(layer7_run.tail_device_name) << "\\",";
    std::cout << "\\"layer13_layer_input_device_name\\":\\"" << JsonEscape(layer8_run.layer_input_device_name) << "\\",";
    std::cout << "\\"layer13_preconv_device_name\\":\\"" << JsonEscape(layer8_run.preconv_device_name) << "\\",";
    std::cout << "\\"layer13_tail_device_name\\":\\"" << JsonEscape(layer8_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer6_shared_gpu.program_build_ms +
                  layer6_tail_gpu.program_build_ms +
                  layer7_run.program_build_ms)
''',
      '''                  layer6_shared_gpu.program_build_ms +
                  layer6_tail_gpu.program_build_ms +
                  layer7_run.program_build_ms +
                  layer8_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer6_shared_gpu.build_log +
                            layer6_tail_gpu.build_log +
                            layer7_run.build_log)
''',
      '''                            layer6_shared_gpu.build_log +
                            layer6_tail_gpu.build_log +
                            layer7_run.build_log +
                            layer8_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer13_state_input_kernel_sum_min_us\\":"
              << layer8_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_to_layer13_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l12", layer7_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l12", layer7_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l13", layer8_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l12_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer7_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l12_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer7_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l13_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer8_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l13_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer8_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer7\\":";
    WriteLayerChecks(layer7_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer7\\":";
    WriteLayerChecks(layer7_run);
    std::cout << ",\\"layer8\\":";
    WriteLayerChecks(layer8_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer12_full_shell_matches_oracle\\":"
              << (layer7_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer12_full_shell_matches_oracle\\":"
              << (layer7_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"layer13_residual_input_from_layer12_live_gpu_lout\\":true,";
    std::cout << "\\"layer13_payload_counts_ok\\":"
              << (layer8_run.payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer13_shapes_ok\\":"
              << (layer8_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer13_qkv_q4_q6_boundary\\":"
              << (layer8_qkv_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer13_down_q4_q6_boundary\\":"
              << (layer8_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer13_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer13_gpu_cpu_matches_native\\":"
              << (layer8_state_input_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer13_state_input_oracle_policy_matches\\":"
              << (layer8_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer13_state_input_handoff_matches\\":"
              << (layer8_ok ? "true" : "false") << ",";
    std::cout << "\\"layer13_full_shell_matches_oracle\\":"
              << (layer8_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer4_run.timing_positive && layer5_run.timing_positive &&
                  layer6_timing_positive && layer7_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer4_run.timing_positive && layer5_run.timing_positive &&
                  layer6_timing_positive && layer7_state_input_timing_positive &&
                  layer8_state_input_timing_positive ? "true" : "false") << ",";
''',
  )
  return cpp


def layer14_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L13.layer13_state_input_probe_cpp(opencl_source)
  for old, new in {
      L13.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer13_state_input_load_once_run_many":
          "layer5_to_layer14_state_input_load_once_run_many",
      "layer13 state/input handoff probe expects --layer 5":
          "layer14 state/input handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer7 = args.layer + 7;
    const int layer8 = args.layer + 8;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13,
            "layer14 state/input handoff probe expects --layer 5");
''',
      '''    const int layer7 = args.layer + 7;
    const int layer8 = args.layer + 8;
    const int layer9 = args.layer + 9;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14,
            "layer14 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer8_tensors = ResolveLayerTensorBundle(index, layer8);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer8_tensors = ResolveLayerTensorBundle(index, layer8);
    const auto layer9_tensors = ResolveLayerTensorBundle(index, layer9);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer8_oracle = LoadLayerOraclePayloads(args.payload_dir, "l13");
    const auto oracle_attn_residual =
''',
      '''    const auto layer8_oracle = LoadLayerOraclePayloads(args.payload_dir, "l13");
    const auto layer9_oracle = LoadLayerOraclePayloads(args.payload_dir, "l14");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer8_run = RunResidentLinearLayerShell(
        args, index, layer8_tensors, layer8_oracle,
        layer7_run.gpu_layer_output, layer7_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer8_run = RunResidentLinearLayerShell(
        args, index, layer8_tensors, layer8_oracle,
        layer7_run.gpu_layer_output, layer7_run.gpu_layer_output, rms_norm_epsilon);
    const auto layer9_run = RunResidentLinearLayerShell(
        args, index, layer9_tensors, layer9_oracle,
        layer8_run.gpu_layer_output, layer8_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer8_ok =
        layer8_shapes_ok &&
        layer8_run.payload_counts_ok &&
        layer8_state_input_gpu_cpu_ok &&
        layer8_state_input_oracle_policy_ok &&
        layer8_state_input_timing_positive &&
        layer8_run.arc_selected &&
        layer8_qkv_q4_q6_boundary &&
        layer8_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer8_ok =
        layer8_shapes_ok &&
        layer8_run.payload_counts_ok &&
        layer8_state_input_gpu_cpu_ok &&
        layer8_state_input_oracle_policy_ok &&
        layer8_state_input_timing_positive &&
        layer8_run.arc_selected &&
        layer8_qkv_q4_q6_boundary &&
        layer8_down_q4_q6_boundary;
    const auto find_l9_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer9_run.comparisons.begin(),
              layer9_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer9_run.comparisons.end(),
                  "layer14 comparison missing: " + name);
          return *found;
        };
    const auto& layer9_residual_input = find_l9_group("residual_input");
    const auto& layer9_attn_norm = find_l9_group("attn_norm");
    const auto& layer9_qkv = find_l9_group("linear_attn_qkv_mixed");
    const auto& layer9_conv_output_raw = find_l9_group("conv_output_raw");
    const bool layer9_state_input_gpu_cpu_ok =
        ComparePassed(layer9_residual_input.gpu_vs_cpu) &&
        ComparePassed(layer9_attn_norm.gpu_vs_cpu) &&
        ComparePassed(layer9_qkv.gpu_vs_cpu) &&
        ComparePassed(layer9_conv_output_raw.gpu_vs_cpu) &&
        ComparePassed(layer9_run.conv_state_after_gpu_vs_cpu);
    const bool layer9_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer9_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer9_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer9_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer9_conv_output_raw.gpu_vs_oracle);
    const bool layer9_shapes_ok = ShapesPassed(layer9_run.shape_checks);
    const bool layer9_qkv_q4_q6_boundary =
        (layer9_tensors.qkv_tensor->type == 12 || layer9_tensors.qkv_tensor->type == 14);
    const bool layer9_down_q4_q6_boundary =
        (layer9_tensors.selected_down_tensor->type == 12 ||
         layer9_tensors.selected_down_tensor->type == 14) &&
        (layer9_tensors.shared_down_tensor->type == 12 ||
         layer9_tensors.shared_down_tensor->type == 14);
    const bool layer9_state_input_timing_positive =
        layer9_run.timing.layer_input_rmsnorm_min_us > 0.0 &&
        layer9_run.timing.preconv_to_postconv_kernel_sum_min_us > 0.0;
    const bool layer9_ok =
        layer9_shapes_ok &&
        layer9_run.payload_counts_ok &&
        layer9_state_input_gpu_cpu_ok &&
        layer9_state_input_oracle_policy_ok &&
        layer9_state_input_timing_positive &&
        layer9_run.arc_selected &&
        layer9_qkv_q4_q6_boundary &&
        layer9_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer6_ok &&
        layer7_ok &&
        layer8_ok &&
        layer2_timing_positive &&
''',
      '''        layer6_ok &&
        layer7_ok &&
        layer8_ok &&
        layer9_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer8_state_input_sum_min =
        layer8_run.timing.layer_input_rmsnorm_min_us +
        layer8_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer8_state_input_sum_min =
        layer8_run.timing.layer_input_rmsnorm_min_us +
        layer8_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer9_state_input_sum_min =
        layer9_run.timing.layer_input_rmsnorm_min_us +
        layer9_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer13_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer13_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer14_residual_input_boundary\\":\\"live_gpu_l_out_13\\",";
    std::cout << "\\"layer14_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer9_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer14_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer9_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer14_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer9_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer14_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer13_tail_device_name\\":\\"" << JsonEscape(layer8_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer13_tail_device_name\\":\\"" << JsonEscape(layer8_run.tail_device_name) << "\\",";
    std::cout << "\\"layer14_layer_input_device_name\\":\\"" << JsonEscape(layer9_run.layer_input_device_name) << "\\",";
    std::cout << "\\"layer14_preconv_device_name\\":\\"" << JsonEscape(layer9_run.preconv_device_name) << "\\",";
    std::cout << "\\"layer14_tail_device_name\\":\\"" << JsonEscape(layer9_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer6_tail_gpu.program_build_ms +
                  layer7_run.program_build_ms +
                  layer8_run.program_build_ms)
''',
      '''                  layer6_tail_gpu.program_build_ms +
                  layer7_run.program_build_ms +
                  layer8_run.program_build_ms +
                  layer9_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer6_tail_gpu.build_log +
                            layer7_run.build_log +
                            layer8_run.build_log)
''',
      '''                            layer6_tail_gpu.build_log +
                            layer7_run.build_log +
                            layer8_run.build_log +
                            layer9_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_to_layer13_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_to_layer13_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer14_state_input_kernel_sum_min_us\\":"
              << layer9_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_to_layer14_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l13", layer8_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l13", layer8_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l14", layer9_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l13_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer8_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l13_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer8_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l14_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer9_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l14_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer9_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer8\\":";
    WriteLayerChecks(layer8_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer8\\":";
    WriteLayerChecks(layer8_run);
    std::cout << ",\\"layer9\\":";
    WriteLayerChecks(layer9_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer13_full_shell_matches_oracle\\":"
              << (layer8_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer13_full_shell_matches_oracle\\":"
              << (layer8_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"layer14_residual_input_from_layer13_live_gpu_lout\\":true,";
    std::cout << "\\"layer14_payload_counts_ok\\":"
              << (layer9_run.payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer14_shapes_ok\\":"
              << (layer9_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer14_qkv_q4_q6_boundary\\":"
              << (layer9_qkv_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer14_down_q4_q6_boundary\\":"
              << (layer9_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer14_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer14_gpu_cpu_matches_native\\":"
              << (layer9_state_input_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer14_state_input_oracle_policy_matches\\":"
              << (layer9_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer14_state_input_handoff_matches\\":"
              << (layer9_ok ? "true" : "false") << ",";
    std::cout << "\\"layer14_full_shell_matches_oracle\\":"
              << (layer9_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer6_timing_positive && layer7_state_input_timing_positive &&
                  layer8_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer6_timing_positive && layer7_state_input_timing_positive &&
                  layer8_state_input_timing_positive &&
                  layer9_state_input_timing_positive ? "true" : "false") << ",";
''',
  )
  return cpp


def layer15_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L14.layer14_state_input_probe_cpp(opencl_source)
  for old, new in {
      L14.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer14_state_input_load_once_run_many":
          "layer5_to_layer15_state_input_load_once_run_many",
      "layer14 state/input handoff probe expects --layer 5":
          "layer15 state/input handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer8 = args.layer + 8;
    const int layer9 = args.layer + 9;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14,
            "layer15 state/input handoff probe expects --layer 5");
''',
      '''    const int layer8 = args.layer + 8;
    const int layer9 = args.layer + 9;
    const int layer10 = args.layer + 10;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15,
            "layer15 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer9_tensors = ResolveLayerTensorBundle(index, layer9);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer9_tensors = ResolveLayerTensorBundle(index, layer9);
    const auto layer10_tensors = ResolveLayerTensorBundle(index, layer10);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer9_oracle = LoadLayerOraclePayloads(args.payload_dir, "l14");
    const auto oracle_attn_residual =
''',
      '''    const auto layer9_oracle = LoadLayerOraclePayloads(args.payload_dir, "l14");
    const auto layer10_oracle = LoadLayerOraclePayloads(args.payload_dir, "l15");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer9_run = RunResidentLinearLayerShell(
        args, index, layer9_tensors, layer9_oracle,
        layer8_run.gpu_layer_output, layer8_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer9_run = RunResidentLinearLayerShell(
        args, index, layer9_tensors, layer9_oracle,
        layer8_run.gpu_layer_output, layer8_run.gpu_layer_output, rms_norm_epsilon);
    const auto layer10_run = RunResidentLinearLayerShell(
        args, index, layer10_tensors, layer10_oracle,
        layer9_run.gpu_layer_output, layer9_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer9_ok =
        layer9_shapes_ok &&
        layer9_run.payload_counts_ok &&
        layer9_state_input_gpu_cpu_ok &&
        layer9_state_input_oracle_policy_ok &&
        layer9_state_input_timing_positive &&
        layer9_run.arc_selected &&
        layer9_qkv_q4_q6_boundary &&
        layer9_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer9_ok =
        layer9_shapes_ok &&
        layer9_run.payload_counts_ok &&
        layer9_state_input_gpu_cpu_ok &&
        layer9_state_input_oracle_policy_ok &&
        layer9_state_input_timing_positive &&
        layer9_run.arc_selected &&
        layer9_qkv_q4_q6_boundary &&
        layer9_down_q4_q6_boundary;
    const auto find_l10_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer10_run.comparisons.begin(),
              layer10_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer10_run.comparisons.end(),
                  "layer15 comparison missing: " + name);
          return *found;
        };
    const auto& layer10_residual_input = find_l10_group("residual_input");
    const auto& layer10_attn_norm = find_l10_group("attn_norm");
    const auto& layer10_qkv = find_l10_group("linear_attn_qkv_mixed");
    const auto& layer10_conv_output_raw = find_l10_group("conv_output_raw");
    const bool layer10_state_input_gpu_cpu_ok =
        ComparePassed(layer10_residual_input.gpu_vs_cpu) &&
        ComparePassed(layer10_attn_norm.gpu_vs_cpu) &&
        ComparePassed(layer10_qkv.gpu_vs_cpu) &&
        ComparePassed(layer10_conv_output_raw.gpu_vs_cpu) &&
        ComparePassed(layer10_run.conv_state_after_gpu_vs_cpu);
    const bool layer10_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer10_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer10_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer10_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer10_conv_output_raw.gpu_vs_oracle);
    const bool layer10_shapes_ok = ShapesPassed(layer10_run.shape_checks);
    const bool layer10_qkv_q4_q6_boundary =
        (layer10_tensors.qkv_tensor->type == 12 || layer10_tensors.qkv_tensor->type == 14);
    const bool layer10_down_q4_q6_boundary =
        (layer10_tensors.selected_down_tensor->type == 12 ||
         layer10_tensors.selected_down_tensor->type == 14) &&
        (layer10_tensors.shared_down_tensor->type == 12 ||
         layer10_tensors.shared_down_tensor->type == 14);
    const bool layer10_state_input_timing_positive =
        layer10_run.timing.layer_input_rmsnorm_min_us > 0.0 &&
        layer10_run.timing.preconv_to_postconv_kernel_sum_min_us > 0.0;
    const bool layer10_ok =
        layer10_shapes_ok &&
        layer10_run.payload_counts_ok &&
        layer10_state_input_gpu_cpu_ok &&
        layer10_state_input_oracle_policy_ok &&
        layer10_state_input_timing_positive &&
        layer10_run.arc_selected &&
        layer10_qkv_q4_q6_boundary &&
        layer10_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer7_ok &&
        layer8_ok &&
        layer9_ok &&
        layer2_timing_positive &&
''',
      '''        layer7_ok &&
        layer8_ok &&
        layer9_ok &&
        layer10_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer9_state_input_sum_min =
        layer9_run.timing.layer_input_rmsnorm_min_us +
        layer9_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer9_state_input_sum_min =
        layer9_run.timing.layer_input_rmsnorm_min_us +
        layer9_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer10_state_input_sum_min =
        layer10_run.timing.layer_input_rmsnorm_min_us +
        layer10_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer14_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer14_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer15_residual_input_boundary\\":\\"live_gpu_l_out_14\\",";
    std::cout << "\\"layer15_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer10_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer15_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer10_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer15_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer10_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer15_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer14_tail_device_name\\":\\"" << JsonEscape(layer9_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer14_tail_device_name\\":\\"" << JsonEscape(layer9_run.tail_device_name) << "\\",";
    std::cout << "\\"layer15_layer_input_device_name\\":\\"" << JsonEscape(layer10_run.layer_input_device_name) << "\\",";
    std::cout << "\\"layer15_preconv_device_name\\":\\"" << JsonEscape(layer10_run.preconv_device_name) << "\\",";
    std::cout << "\\"layer15_tail_device_name\\":\\"" << JsonEscape(layer10_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer7_run.program_build_ms +
                  layer8_run.program_build_ms +
                  layer9_run.program_build_ms)
''',
      '''                  layer7_run.program_build_ms +
                  layer8_run.program_build_ms +
                  layer9_run.program_build_ms +
                  layer10_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer7_run.build_log +
                            layer8_run.build_log +
                            layer9_run.build_log)
''',
      '''                            layer7_run.build_log +
                            layer8_run.build_log +
                            layer9_run.build_log +
                            layer10_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_to_layer14_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_to_layer14_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer15_state_input_kernel_sum_min_us\\":"
              << layer10_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l14", layer9_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l14", layer9_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l15", layer10_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l14_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer9_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l14_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer9_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l15_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer10_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l15_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer10_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer9\\":";
    WriteLayerChecks(layer9_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer9\\":";
    WriteLayerChecks(layer9_run);
    std::cout << ",\\"layer10\\":";
    WriteLayerChecks(layer10_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer14_full_shell_matches_oracle\\":"
              << (layer9_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer14_full_shell_matches_oracle\\":"
              << (layer9_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"layer15_residual_input_from_layer14_live_gpu_lout\\":true,";
    std::cout << "\\"layer15_payload_counts_ok\\":"
              << (layer10_run.payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_shapes_ok\\":"
              << (layer10_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_qkv_q4_q6_boundary\\":"
              << (layer10_qkv_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_down_q4_q6_boundary\\":"
              << (layer10_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer15_gpu_cpu_matches_native\\":"
              << (layer10_state_input_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_state_input_oracle_policy_matches\\":"
              << (layer10_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_state_input_handoff_matches\\":"
              << (layer10_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_full_shell_matches_oracle\\":"
              << (layer10_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer8_state_input_timing_positive &&
                  layer9_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer8_state_input_timing_positive &&
                  layer9_state_input_timing_positive &&
                  layer10_state_input_timing_positive ? "true" : "false") << ",";
''',
  )
  return cpp


def layer15_full_attn_probe_cpp(opencl_source: str) -> str:
  cpp = L14.layer14_state_input_probe_cpp(opencl_source)
  for old, new in {
      L14.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer14_state_input_load_once_run_many":
          "layer5_to_layer15_full_attention_load_once_run_many",
      "layer14 state/input handoff probe expects --layer 5":
          "layer15 full-attention handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer8 = args.layer + 8;
    const int layer9 = args.layer + 9;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14,
            "layer15 full-attention handoff probe expects --layer 5");
''',
      '''    const int layer8 = args.layer + 8;
    const int layer9 = args.layer + 9;
    const int layer10 = args.layer + 10;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12 && layer8 == 13 && layer9 == 14 && layer10 == 15,
            "layer15 full-attention handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer9_tensors = ResolveLayerTensorBundle(index, layer9);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer9_tensors = ResolveLayerTensorBundle(index, layer9);
    const auto layer10_tensors = ResolveFullAttentionTensorBundle(index, layer10);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer9_oracle = LoadLayerOraclePayloads(args.payload_dir, "l14");
    const auto oracle_attn_residual =
''',
      '''    const auto layer9_oracle = LoadLayerOraclePayloads(args.payload_dir, "l14");
    const auto layer10_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l15_");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto oracle_layer_output =
        iq36::add_vectors(oracle_attn_residual, oracle_ffn_out);
''',
      '''    const auto oracle_layer_output =
        iq36::add_vectors(oracle_attn_residual, oracle_ffn_out);
    const auto layer10_oracle_attn_residual =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_full_attn_residual.bin"));
    const auto layer10_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_full_attn_post_norm.bin"));
    const bool layer10_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer10_oracle) &&
        layer10_oracle_attn_residual.size() == kHiddenSize &&
        layer10_oracle_attn_post_norm.size() == kHiddenSize;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto full_shapes = CheckFullAttentionShapes(layer2_tensors);
    const bool full_shapes_ok = FullAttentionShapesPassed(full_shapes);
''',
      '''    const auto full_shapes = CheckFullAttentionShapes(layer2_tensors);
    const bool full_shapes_ok = FullAttentionShapesPassed(full_shapes);
    const auto layer10_full_shapes = CheckFullAttentionShapes(layer10_tensors);
    const bool layer10_full_shapes_ok = FullAttentionShapesPassed(layer10_full_shapes);
''',
  )
  cpp = replace_once(
      cpp,
      '''    const std::string selected_gate_up_tensor_name =
''',
      '''    const auto layer10_attn_norm_weight =
        ReadF32TensorPayload(model, *layer10_tensors.attn_norm_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const auto layer10_q_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer10_tensors.q_norm_tensor_name, 0);
    const auto layer10_k_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                layer10_tensors.k_norm_tensor_name, 0);
    const auto layer10_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer10, "post_attention_norm.weight"), 0);
    const std::string selected_gate_up_tensor_name =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer9_run = RunResidentLinearLayerShell(
        args, index, layer9_tensors, layer9_oracle,
        layer8_run.gpu_layer_output, layer8_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer9_run = RunResidentLinearLayerShell(
        args, index, layer9_tensors, layer9_oracle,
        layer8_run.gpu_layer_output, layer8_run.gpu_layer_output, rms_norm_epsilon);

    const auto layer10_native_qkv = iq36::run_qwen36_full_attention_qkv_projection(
        args.model_path,
        index,
        layer10,
        layer9_run.gpu_layer_output,
        rms_norm_epsilon);
    const auto layer10_rms_gpu = RunGpuLayerInputRmsNorm(
        layer9_run.gpu_layer_output,
        layer10_attn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);
    const auto layer10_qk_gpu = RunGpuFullAttentionQkFront(
        args.model_path,
        *layer10_tensors.q_tensor,
        *layer10_tensors.k_tensor,
        layer10_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer10_v_gpu = RunGpuFullAttentionVAny(
        args.model_path,
        *layer10_tensors.v_tensor,
        layer10_rms_gpu.attn_norm,
        args.device_substring,
        args.repeat);
    const auto layer10_gpu_q_split = SplitFullAttentionQ(layer10_qk_gpu.q_full);
    const auto layer10_gpu_q_normed = ApplyRepeatedRmsNormFull(
        layer10_gpu_q_split.q_raw, layer10_q_norm_weight, rms_norm_epsilon);
    const auto layer10_gpu_k_normed = ApplyRepeatedRmsNormFull(
        layer10_qk_gpu.k_raw, layer10_k_norm_weight, rms_norm_epsilon);
    const auto layer10_gpu_rope = iq36::run_qwen36_full_attention_rope(
        layer10_gpu_q_normed,
        layer10_gpu_k_normed,
        kFullSourceTokenPosition,
        head_dim,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        kRopeFreqScale,
        kRopeExtFactor,
        kRopeAttnFactor,
        kRopeBetaFast,
        kRopeBetaSlow);
    const auto layer10_native_rope = iq36::run_qwen36_full_attention_rope(
        layer10_native_qkv.q_normed,
        layer10_native_qkv.k_normed,
        kFullSourceTokenPosition,
        head_dim,
        rope_dimension_count,
        rope_sections,
        rope_context_length,
        rope_freq_base,
        kRopeFreqScale,
        kRopeExtFactor,
        kRopeAttnFactor,
        kRopeBetaFast,
        kRopeBetaSlow);
    auto layer10_native_k_history = layer10_oracle.k_history;
    auto layer10_native_v_history = layer10_oracle.v_history;
    layer10_native_k_history.push_back(layer10_oracle.k_rope);
    layer10_native_v_history.push_back(layer10_oracle.v);
    const auto layer10_native_core = iq36::run_qwen36_full_attention_core(
        layer10_oracle.q_rope,
        layer10_native_k_history,
        layer10_native_v_history,
        head_dim,
        q_head_count,
        kv_head_count,
        kAttentionScale);
    const auto layer10_native_gate = iq36::run_qwen36_full_attention_gate(
        layer10_oracle.q_full, layer10_native_core.attn_pregate, head_dim);
    const auto layer10_native_attn_output = iq36::matvec_tensor(
        args.model_path,
        index,
        LayerTensorName(layer10, "attn_output.weight"),
        layer10_native_gate.attn_gated);
    const auto layer10_native_attn_residual =
        iq36::add_vectors(layer9_run.gpu_layer_output, layer10_native_attn_output);
    const auto layer10_native_attn_post_norm =
        iq36::apply_rms_norm(layer10_native_attn_residual,
                             layer10_ffn_norm_weight,
                             rms_norm_epsilon);

    auto layer10_gpu_k_history = layer10_oracle.k_history;
    auto layer10_gpu_v_history = layer10_oracle.v_history;
    layer10_gpu_k_history.push_back(layer10_oracle.k_rope);
    layer10_gpu_v_history.push_back(layer10_v_gpu.v);
    std::vector<float> layer10_gpu_k_history_flat;
    std::vector<float> layer10_gpu_v_history_flat;
    for (const auto& item : layer10_gpu_k_history) {
      layer10_gpu_k_history_flat.insert(layer10_gpu_k_history_flat.end(), item.begin(), item.end());
    }
    for (const auto& item : layer10_gpu_v_history) {
      layer10_gpu_v_history_flat.insert(layer10_gpu_v_history_flat.end(), item.begin(), item.end());
    }
    const auto layer10_core_gate_gpu = RunGpuFullAttentionCoreGate(
        layer10_oracle.q_rope,
        layer10_gpu_k_history_flat,
        layer10_gpu_v_history_flat,
        layer10_qk_gpu.q_full,
        args.device_substring,
        args.repeat);
    const auto layer10_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer10, "attn_output.weight")),
        layer10_core_gate_gpu.attn_gated,
        layer9_run.gpu_layer_output,
        layer10_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer9_ok =
        layer9_shapes_ok &&
        layer9_run.payload_counts_ok &&
        layer9_state_input_gpu_cpu_ok &&
        layer9_state_input_oracle_policy_ok &&
        layer9_state_input_timing_positive &&
        layer9_run.arc_selected &&
        layer9_qkv_q4_q6_boundary &&
        layer9_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer9_ok =
        layer9_shapes_ok &&
        layer9_run.payload_counts_ok &&
        layer9_state_input_gpu_cpu_ok &&
        layer9_state_input_oracle_policy_ok &&
        layer9_state_input_timing_positive &&
        layer9_run.arc_selected &&
        layer9_qkv_q4_q6_boundary &&
        layer9_down_q4_q6_boundary;
    std::vector<NamedCompareGroup> layer10_strict_groups;
    AppendCpuGpuOracleCompare(layer10_strict_groups, "l15_residual_input",
                              layer9_run.gpu_layer_output,
                              layer9_run.gpu_layer_output,
                              layer10_oracle.residual_input);
    AppendCpuGpuOracleCompare(layer10_strict_groups, "l15_attn_norm",
                              layer10_native_qkv.attention_norm,
                              layer10_rms_gpu.attn_norm,
                              layer10_oracle.attn_norm);
    AppendCpuGpuOracleCompare(layer10_strict_groups, "l15_q_full",
                              layer10_native_qkv.q_full,
                              layer10_qk_gpu.q_full,
                              layer10_oracle.q_full);
    AppendCpuGpuOracleCompare(layer10_strict_groups, "l15_q_rope",
                              layer10_native_rope.q_rope,
                              layer10_gpu_rope.q_rope,
                              layer10_oracle.q_rope);
    AppendCpuGpuOracleCompare(layer10_strict_groups, "l15_k_rope",
                              layer10_native_rope.k_rope,
                              layer10_gpu_rope.k_rope,
                              layer10_oracle.k_rope);
    AppendCpuGpuOracleCompare(layer10_strict_groups, "l15_v",
                              layer10_native_qkv.v,
                              layer10_v_gpu.v,
                              layer10_oracle.v);
    std::vector<NamedCompareGroup> layer10_full_attention_groups;
    AppendFullAttentionComponentCompare(layer10_full_attention_groups, "l15_attn_pregate",
                                        layer10_native_core.attn_pregate,
                                        layer10_core_gate_gpu.attn_pregate,
                                        layer10_oracle.attn_pregate);
    AppendFullAttentionComponentCompare(layer10_full_attention_groups, "l15_attn_gated",
                                        layer10_native_gate.attn_gated,
                                        layer10_core_gate_gpu.attn_gated,
                                        layer10_oracle.attn_gated);
    AppendFullAttentionComponentCompare(layer10_full_attention_groups, "l15_attn_output",
                                        layer10_native_attn_output,
                                        layer10_attention_gpu.linear_attn_out,
                                        layer10_oracle.attn_output);
    AppendFullAttentionComponentCompare(layer10_full_attention_groups, "l15_attn_residual",
                                        layer10_native_attn_residual,
                                        layer10_attention_gpu.attn_residual,
                                        layer10_oracle_attn_residual);
    AppendFullAttentionComponentCompare(layer10_full_attention_groups, "l15_attn_post_norm",
                                        layer10_native_attn_post_norm,
                                        layer10_attention_gpu.attn_post_norm,
                                        layer10_oracle_attn_post_norm);
    const auto layer10_k_raw_gpu_vs_cpu = iq36::compare_vectors(
        layer10_qk_gpu.k_raw, layer10_native_qkv.k_raw, kMismatchThreshold);
    const bool layer10_strict_input_ok =
        CompareGroupsPassed(layer10_strict_groups) &&
        ComparePassed(layer10_k_raw_gpu_vs_cpu);
    bool layer10_full_component_ok = true;
    for (const auto& group : layer10_full_attention_groups) {
      layer10_full_component_ok =
          layer10_full_component_ok &&
          ComparePassedFullAttentionComponent(group.gpu_vs_oracle);
    }
    const bool layer10_comparisons_ok =
        layer10_strict_input_ok && layer10_full_component_ok;
    const bool layer10_timing_positive =
        layer10_rms_gpu.timing.rmsnorm_min_us > 0.0 &&
        layer10_qk_gpu.timing.q_projection_min_us > 0.0 &&
        layer10_qk_gpu.timing.k_projection_min_us > 0.0 &&
        layer10_v_gpu.timing.v_projection_min_us > 0.0 &&
        layer10_core_gate_gpu.timing.core_min_us > 0.0 &&
        layer10_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer10_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer10_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer10_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
    const bool layer10_arc_selected =
        layer10_rms_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_attention_gpu.device_name.find(args.device_substring) != std::string::npos;
    const bool layer10_v_gpu_boundary =
        (layer10_tensors.v_tensor->type == 12 || layer10_tensors.v_tensor->type == 14) &&
        layer10_v_gpu.v.size() == kFullKvValues;
    const bool layer10_ffn_q6_boundary =
        iq36::find_tensor(index, LayerTensorName(layer10, "ffn_down_exps.weight"))->type == 14 &&
        iq36::find_tensor(index, LayerTensorName(layer10, "ffn_down_shexp.weight"))->type == 14;
    const bool layer10_ok =
        layer10_full_shapes_ok &&
        layer10_payload_counts_ok &&
        metadata_ok &&
        layer10_comparisons_ok &&
        layer10_timing_positive &&
        layer10_arc_selected &&
        layer10_v_gpu_boundary;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer7_ok &&
        layer8_ok &&
        layer9_ok &&
        layer2_timing_positive &&
''',
      '''        layer7_ok &&
        layer8_ok &&
        layer9_ok &&
        layer10_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer9_state_input_sum_min =
        layer9_run.timing.layer_input_rmsnorm_min_us +
        layer9_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer9_state_input_sum_min =
        layer9_run.timing.layer_input_rmsnorm_min_us +
        layer9_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer10_input_sum_min =
        layer10_rms_gpu.timing.rmsnorm_min_us +
        layer10_qk_gpu.timing.qk_projection_kernel_sum_min_us +
        layer10_v_gpu.timing.v_projection_min_us;
    const double layer10_core_output_sum_min =
        layer10_core_gate_gpu.timing.core_gate_kernel_sum_min_us +
        layer10_attention_gpu.timing.attention_front_kernel_sum_min_us;
    const double layer10_attention_sum_min =
        layer10_input_sum_min + layer10_core_output_sum_min;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "," << layer8 << "," << layer9 << "," << layer10 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer14_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer14_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer15_residual_input_boundary\\":\\"live_gpu_l_out_14\\",";
    std::cout << "\\"layer15_v_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer10_tensors.v_tensor->type)) << "\\",";
    std::cout << "\\"layer15_v_projection_boundary\\":\\""
              << (layer10_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer15_ffn_boundary\\":\\"q6_down_pending\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer14_tail_device_name\\":\\"" << JsonEscape(layer9_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer14_tail_device_name\\":\\"" << JsonEscape(layer9_run.tail_device_name) << "\\",";
    std::cout << "\\"layer15_core_gate_device_name\\":\\"" << JsonEscape(layer10_core_gate_gpu.device_name) << "\\",";
    std::cout << "\\"layer15_output_projection_device_name\\":\\"" << JsonEscape(layer10_attention_gpu.device_name) << "\\",";
    std::cout << "\\"layer15_v_projection_device_name\\":\\"" << JsonEscape(layer10_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer7_run.program_build_ms +
                  layer8_run.program_build_ms +
                  layer9_run.program_build_ms)
''',
      '''                  layer7_run.program_build_ms +
                  layer8_run.program_build_ms +
                  layer9_run.program_build_ms +
                  layer10_rms_gpu.program_build_ms + layer10_qk_gpu.program_build_ms +
                  layer10_v_gpu.program_build_ms +
                  layer10_core_gate_gpu.program_build_ms +
                  layer10_attention_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer7_run.build_log +
                            layer8_run.build_log +
                            layer9_run.build_log)
''',
      '''                            layer7_run.build_log +
                            layer8_run.build_log +
                            layer9_run.build_log +
                            layer10_rms_gpu.build_log + layer10_qk_gpu.build_log +
                            layer10_v_gpu.build_log +
                            layer10_core_gate_gpu.build_log +
                            layer10_attention_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer6_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer6_attention_gpu.timing);
''',
      '''    std::cout << ",\\"layer6_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer6_attention_gpu.timing);
    std::cout << ",\\"layer10_full_attn_input\\":";
    WriteFullAttentionTiming(layer10_rms_gpu, layer10_qk_gpu);
    std::cout << ",\\"layer10_v_projection\\":";
    WriteFullAttentionVQ6Timing(layer10_v_gpu.timing);
    std::cout << ",\\"layer10_core_gate\\":";
    WriteFullCoreGateTiming(layer10_core_gate_gpu.timing);
    std::cout << ",\\"layer10_output_front\\":";
    WriteFullAttentionOutputFrontTiming(layer10_attention_gpu.timing);
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_to_layer14_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_to_layer14_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min) << ",";
    std::cout << "\\"resident_layer15_full_attn_input_kernel_sum_min_us\\":"
              << layer10_input_sum_min << ",";
    std::cout << "\\"resident_layer15_full_attn_core_output_kernel_sum_min_us\\":"
              << layer10_core_output_sum_min << ",";
    std::cout << "\\"resident_layer15_full_attention_kernel_sum_min_us\\":"
              << layer10_attention_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_full_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l14", layer9_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l14", layer9_run.comparisons, &first_compare);
    std::cout << ",";
    WriteNamedCompareGroups(layer10_strict_groups);
    std::cout << ",\\"l15_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer10_k_raw_gpu_vs_cpu);
    std::cout << "},";
    WriteNamedCompareGroups(layer10_full_attention_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer9\\":";
    WriteLayerChecks(layer9_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer9\\":";
    WriteLayerChecks(layer9_run);
    std::cout << ",\\"layer10_full_attention_shapes\\":";
    WriteFullAttentionShapeChecks(layer10_full_shapes);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer14_full_shell_matches_oracle\\":"
              << (layer9_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer14_full_shell_matches_oracle\\":"
              << (layer9_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"layer15_residual_input_from_layer14_live_gpu_lout\\":true,";
    std::cout << "\\"layer15_payload_counts_ok\\":"
              << (layer10_payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_full_attn_input_matches_oracle\\":"
              << (layer10_strict_input_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_full_attn_component_policy_matches_oracle\\":"
              << (layer10_full_component_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_full_attn_core_output_matches_oracle\\":"
              << (layer10_comparisons_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_v_gpu_boundary\\":"
              << (layer10_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_ffn_q6_boundary\\":"
              << (layer10_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_arc_device_selected\\":"
              << (layer10_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer8_state_input_timing_positive &&
                  layer9_state_input_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer8_state_input_timing_positive &&
                  layer9_state_input_timing_positive &&
                  layer10_timing_positive ? "true" : "false") << ",";
''',
  )
  return cpp


def layer15_ffn_lout_probe_cpp(opencl_source: str) -> str:
  cpp = layer15_full_attn_probe_cpp(opencl_source)
  for old, new in {
      "layer5_to_layer15_full_attention_load_once_run_many":
          "layer5_to_layer15_ffn_lout_load_once_run_many",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const auto layer10_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_full_attn_post_norm.bin"));
''',
      '''    const auto layer10_oracle_attn_post_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_full_attn_post_norm.bin"));
    const auto layer10_oracle_expert_ids =
        ReadI32VectorFile(JoinPath(args.payload_dir, "l15_ffn_moe_topk.bin"));
    const auto layer10_oracle_weights_norm =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_moe_weights_norm.bin"));
    const auto layer10_oracle_selected_gate_up =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_moe_gate_up.bin"));
    const auto layer10_oracle_selected_swiglu =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_moe_swiglu.bin"));
    const auto layer10_oracle_selected_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_moe_down.bin"));
    const auto layer10_oracle_weighted =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_moe_weighted.bin"));
    const auto layer10_oracle_moe_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_moe_out.bin"));
    const auto layer10_oracle_shared_down =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_shexp.bin"));
    const auto layer10_oracle_shared_gate =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_shared_expert_gate.bin"));
    const auto layer10_oracle_shared_sigmoid =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_shared_expert_gate_sigmoid.bin"));
    const auto layer10_oracle_shared_gated =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_shexp_gated.bin"));
    const auto layer10_oracle_ffn_out =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_ffn_out.bin"));
    const auto layer10_oracle_layer_output =
        iq36::read_f32_vector_file(JoinPath(args.payload_dir, "l15_l_out.bin"));
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer10_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer10_oracle) &&
        layer10_oracle_attn_residual.size() == kHiddenSize &&
        layer10_oracle_attn_post_norm.size() == kHiddenSize;
''',
      '''    const bool layer10_payload_counts_ok =
        FullAttentionPayloadCountsOk(layer10_oracle) &&
        layer10_oracle_attn_residual.size() == kHiddenSize &&
        layer10_oracle_attn_post_norm.size() == kHiddenSize &&
        layer10_oracle_expert_ids.size() == kExpertUsedCount &&
        layer10_oracle_weights_norm.size() == kExpertUsedCount &&
        layer10_oracle_selected_gate_up.size() == kGateUpValueCount &&
        layer10_oracle_selected_swiglu.size() == kSwiGluValueCount &&
        layer10_oracle_selected_down.size() == kWeightedValueCount &&
        layer10_oracle_weighted.size() == kWeightedValueCount &&
        layer10_oracle_moe_out.size() == kHiddenSize &&
        layer10_oracle_shared_down.size() == kHiddenSize &&
        layer10_oracle_shared_gate.size() == 1 &&
        layer10_oracle_shared_sigmoid.size() == 1 &&
        layer10_oracle_shared_gated.size() == kHiddenSize &&
        layer10_oracle_ffn_out.size() == kHiddenSize &&
        layer10_oracle_layer_output.size() == kHiddenSize;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer10_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer10, "post_attention_norm.weight"), 0);
    const std::string selected_gate_up_tensor_name =
''',
      '''    const auto layer10_ffn_norm_weight =
        iq36::decode_tensor_row(args.model_path, index,
                                LayerTensorName(layer10, "post_attention_norm.weight"), 0);
    const std::string layer10_selected_gate_up_tensor_name =
        LayerTensorName(layer10, "ffn_gate_up_exps.weight");
    const std::string layer10_selected_down_tensor_name =
        LayerTensorName(layer10, "ffn_down_exps.weight");
    const std::string layer10_shared_gate_tensor_name =
        LayerTensorName(layer10, "ffn_gate_shexp.weight");
    const std::string layer10_shared_up_tensor_name =
        LayerTensorName(layer10, "ffn_up_shexp.weight");
    const std::string layer10_shared_down_tensor_name =
        LayerTensorName(layer10, "ffn_down_shexp.weight");
    const std::string layer10_shared_input_gate_tensor_name =
        LayerTensorName(layer10, "ffn_gate_inp_shexp.weight");
    const auto* layer10_selected_gate_up_tensor =
        iq36::find_tensor(index, layer10_selected_gate_up_tensor_name);
    const auto* layer10_selected_down_tensor =
        iq36::find_tensor(index, layer10_selected_down_tensor_name);
    const auto* layer10_shared_gate_tensor =
        iq36::find_tensor(index, layer10_shared_gate_tensor_name);
    const auto* layer10_shared_up_tensor =
        iq36::find_tensor(index, layer10_shared_up_tensor_name);
    const auto* layer10_shared_down_tensor =
        iq36::find_tensor(index, layer10_shared_down_tensor_name);
    const auto* layer10_shared_input_gate_tensor =
        iq36::find_tensor(index, layer10_shared_input_gate_tensor_name);
    Require(layer10_selected_gate_up_tensor != nullptr, "layer15 selected gate-up tensor missing");
    Require(layer10_selected_down_tensor != nullptr, "layer15 selected down tensor missing");
    Require(layer10_shared_gate_tensor != nullptr, "layer15 shared gate tensor missing");
    Require(layer10_shared_up_tensor != nullptr, "layer15 shared up tensor missing");
    Require(layer10_shared_down_tensor != nullptr, "layer15 shared down tensor missing");
    Require(layer10_shared_input_gate_tensor != nullptr, "layer15 shared input gate tensor missing");
    const auto layer10_shared_input_gate_weights =
        ReadF32TensorPayload(model, *layer10_shared_input_gate_tensor,
                             static_cast<std::size_t>(kHiddenSize));
    const bool layer10_ffn_tensor_shapes_ok =
        layer10_selected_gate_up_tensor->type == 12 &&
        (layer10_selected_down_tensor->type == 12 || layer10_selected_down_tensor->type == 14) &&
        layer10_shared_gate_tensor->type == 12 &&
        layer10_shared_up_tensor->type == 12 &&
        (layer10_shared_down_tensor->type == 12 || layer10_shared_down_tensor->type == 14) &&
        layer10_shared_input_gate_tensor->type == 0 &&
        layer10_selected_gate_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount} &&
        layer10_selected_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize, kExpertCount} &&
        layer10_shared_gate_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        layer10_shared_up_tensor->dims ==
            std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
        layer10_shared_down_tensor->dims ==
            std::vector<std::uint64_t>{kIntermediateSize, kHiddenSize} &&
        layer10_shared_input_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize};
    const std::string selected_gate_up_tensor_name =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer10_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer10, "attn_output.weight")),
        layer10_core_gate_gpu.attn_gated,
        layer9_run.gpu_layer_output,
        layer10_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer10_attention_gpu = RunGpuAttentionFront(
        args.model_path,
        *iq36::find_tensor(index, LayerTensorName(layer10, "attn_output.weight")),
        layer10_core_gate_gpu.attn_gated,
        layer9_run.gpu_layer_output,
        layer10_ffn_norm_weight,
        rms_norm_epsilon,
        args.device_substring,
        args.repeat);

    const auto& layer10_ffn_input = layer10_attention_gpu.attn_post_norm;
    const auto layer10_native_selected_gate_up =
        iq36::matvec_expert_tensor(args.model_path, index,
                                   layer10_selected_gate_up_tensor_name,
                                   layer10_ffn_input, layer10_oracle_expert_ids);
    const auto layer10_native_selected_swiglu =
        iq36::apply_swiglu_from_gate_up(layer10_native_selected_gate_up,
                                        kIntermediateSize, kExpertUsedCount);
    const auto layer10_native_selected_down =
        iq36::matvec_expert_tensor_per_expert_input(
            args.model_path, index, layer10_selected_down_tensor_name,
            layer10_native_selected_swiglu, layer10_oracle_expert_ids);
    const auto layer10_native_weighted =
        iq36::apply_expert_weights(layer10_native_selected_down,
                                   layer10_oracle_weights_norm, kHiddenSize);
    const auto layer10_native_moe_out =
        iq36::aggregate_experts(layer10_native_weighted, kExpertUsedCount, kHiddenSize);
    const auto layer10_native_shared_gate =
        iq36::matvec_tensor(args.model_path, index,
                            layer10_shared_gate_tensor_name, layer10_ffn_input);
    const auto layer10_native_shared_up =
        iq36::matvec_tensor(args.model_path, index,
                            layer10_shared_up_tensor_name, layer10_ffn_input);
    std::vector<float> layer10_native_shared_gate_up;
    layer10_native_shared_gate_up.reserve(
        layer10_native_shared_gate.size() + layer10_native_shared_up.size());
    layer10_native_shared_gate_up.insert(layer10_native_shared_gate_up.end(),
                                        layer10_native_shared_gate.begin(),
                                        layer10_native_shared_gate.end());
    layer10_native_shared_gate_up.insert(layer10_native_shared_gate_up.end(),
                                        layer10_native_shared_up.begin(),
                                        layer10_native_shared_up.end());
    const auto layer10_native_shared_swiglu =
        iq36::apply_swiglu_from_gate_up(layer10_native_shared_gate_up, kIntermediateSize, 1);
    const auto layer10_native_shared_down =
        iq36::matvec_tensor(args.model_path, index, layer10_shared_down_tensor_name,
                            layer10_native_shared_swiglu);
    const auto layer10_native_shared_input_gate =
        iq36::matvec_tensor(args.model_path, index,
                            layer10_shared_input_gate_tensor_name, layer10_ffn_input);
    Require(layer10_native_shared_input_gate.size() == 1,
            "native layer15 shared input gate size mismatch");
    const std::vector<float> layer10_native_shared_sigmoid{
        iq36::sigmoid_scalar(layer10_native_shared_input_gate[0])};
    const auto layer10_native_shared_gated =
        iq36::multiply_by_scalar(layer10_native_shared_down,
                                 layer10_native_shared_sigmoid[0]);
    const auto layer10_native_ffn_out =
        iq36::add_vectors(layer10_native_moe_out, layer10_native_shared_gated);
    const auto layer10_native_layer_output =
        iq36::add_vectors(layer10_attention_gpu.attn_residual, layer10_native_ffn_out);

    const auto layer10_selected_gpu = RunGpuSelectedFfnShell(
        args.model_path, *layer10_selected_gate_up_tensor, *layer10_selected_down_tensor,
        layer10_ffn_input, layer10_oracle_expert_ids, args.device_substring, args.repeat);
    const auto layer10_shared_gpu = RunGpuSharedFfnShell(
        args.model_path, *layer10_shared_gate_tensor, *layer10_shared_up_tensor,
        *layer10_shared_down_tensor, layer10_ffn_input, args.device_substring, args.repeat);
    const auto layer10_tail_gpu = RunGpuShell(
        layer10_shared_input_gate_weights, layer10_ffn_input,
        layer10_selected_gpu.down, layer10_oracle_weights_norm,
        layer10_shared_gpu.down, layer10_attention_gpu.attn_residual,
        args.device_substring, args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    AppendFullAttentionComponentCompare(layer10_full_attention_groups, "l15_attn_post_norm",
                                        layer10_native_attn_post_norm,
                                        layer10_attention_gpu.attn_post_norm,
                                        layer10_oracle_attn_post_norm);
    const auto layer10_k_raw_gpu_vs_cpu = iq36::compare_vectors(
''',
      '''    AppendFullAttentionComponentCompare(layer10_full_attention_groups, "l15_attn_post_norm",
                                        layer10_native_attn_post_norm,
                                        layer10_attention_gpu.attn_post_norm,
                                        layer10_oracle_attn_post_norm);

    std::vector<NamedCompareGroup> layer10_ffn_live_groups;
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_selected_gate_up",
                              layer10_native_selected_gate_up,
                              layer10_selected_gpu.gate_up,
                              layer10_oracle_selected_gate_up);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_selected_swiglu",
                              layer10_native_selected_swiglu,
                              layer10_selected_gpu.swiglu,
                              layer10_oracle_selected_swiglu);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_selected_down",
                              layer10_native_selected_down,
                              layer10_selected_gpu.down,
                              layer10_oracle_selected_down);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_ffn_moe_weighted",
                              layer10_native_weighted,
                              layer10_tail_gpu.weighted,
                              layer10_oracle_weighted);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_ffn_moe_out",
                              layer10_native_moe_out,
                              layer10_tail_gpu.moe_out,
                              layer10_oracle_moe_out);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_shared_down",
                              layer10_native_shared_down,
                              layer10_shared_gpu.down,
                              layer10_oracle_shared_down);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_shared_gate",
                              layer10_native_shared_input_gate,
                              layer10_tail_gpu.shared_gate,
                              layer10_oracle_shared_gate);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_shared_gate_sigmoid",
                              layer10_native_shared_sigmoid,
                              layer10_tail_gpu.shared_gate_sigmoid,
                              layer10_oracle_shared_sigmoid);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_ffn_shexp_gated",
                              layer10_native_shared_gated,
                              layer10_tail_gpu.shared_gated,
                              layer10_oracle_shared_gated);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_ffn_out",
                              layer10_native_ffn_out,
                              layer10_tail_gpu.ffn_out,
                              layer10_oracle_ffn_out);
    AppendCpuGpuOracleCompare(layer10_ffn_live_groups, "l15_layer_output",
                              layer10_native_layer_output,
                              layer10_tail_gpu.layer_output,
                              layer10_oracle_layer_output);

    const auto layer10_k_raw_gpu_vs_cpu = iq36::compare_vectors(
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer10_comparisons_ok =
        layer10_strict_input_ok && layer10_full_component_ok;
    const bool layer10_timing_positive =
''',
      '''    const bool layer10_comparisons_ok =
        layer10_strict_input_ok && layer10_full_component_ok;
    bool layer10_live_ffn_gpu_cpu_ok = true;
    for (const auto& group : layer10_ffn_live_groups) {
      layer10_live_ffn_gpu_cpu_ok =
          layer10_live_ffn_gpu_cpu_ok && ComparePassed(group.gpu_vs_cpu);
    }
    const bool layer10_live_ffn_oracle_magnitude_ok =
        live_ffn_oracle_magnitude_passed(layer10_ffn_live_groups[2].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(layer10_ffn_live_groups[5].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(layer10_ffn_live_groups[9].gpu_vs_oracle) &&
        live_ffn_oracle_magnitude_passed(layer10_ffn_live_groups[10].gpu_vs_oracle);
    const bool layer10_live_layer_output_full_policy_ok =
        ComparePassedFullAttentionComponent(layer10_ffn_live_groups[10].gpu_vs_oracle);
    const bool layer10_live_ffn_lout_ok =
        layer10_live_ffn_gpu_cpu_ok &&
        layer10_live_ffn_oracle_magnitude_ok &&
        layer10_live_layer_output_full_policy_ok;
    const bool layer10_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer10_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer10_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer10_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer10_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0;
''',
      '''        layer10_core_gate_gpu.timing.gate_min_us > 0.0 &&
        layer10_attention_gpu.timing.output_projection_min_us > 0.0 &&
        layer10_attention_gpu.timing.residual_add_min_us > 0.0 &&
        layer10_attention_gpu.timing.ffn_rmsnorm_min_us > 0.0 &&
        layer10_selected_gpu.timing.gate_up_min_us > 0.0 &&
        layer10_selected_gpu.timing.swiglu_min_us > 0.0 &&
        layer10_selected_gpu.timing.down_min_us > 0.0 &&
        layer10_shared_gpu.timing.gate_min_us > 0.0 &&
        layer10_shared_gpu.timing.up_min_us > 0.0 &&
        layer10_shared_gpu.timing.swiglu_min_us > 0.0 &&
        layer10_shared_gpu.timing.down_min_us > 0.0 &&
        layer10_tail_gpu.timing.shell_sum_min_us > 0.0;
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer10_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_attention_gpu.device_name.find(args.device_substring) != std::string::npos;
''',
      '''        layer10_qk_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_v_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_core_gate_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_attention_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_selected_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_shared_gpu.device_name.find(args.device_substring) != std::string::npos &&
        layer10_tail_gpu.device_name.find(args.device_substring) != std::string::npos;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer10_ffn_q6_boundary =
        iq36::find_tensor(index, LayerTensorName(layer10, "ffn_down_exps.weight"))->type == 14 &&
        iq36::find_tensor(index, LayerTensorName(layer10, "ffn_down_shexp.weight"))->type == 14;
    const bool layer10_ok =
''',
      '''    const bool layer10_ffn_down_q4_q6_boundary =
        (layer10_selected_down_tensor->type == 12 || layer10_selected_down_tensor->type == 14) &&
        (layer10_shared_down_tensor->type == 12 || layer10_shared_down_tensor->type == 14) &&
        layer10_selected_gpu.down.size() == kWeightedValueCount &&
        layer10_shared_gpu.down.size() == kHiddenSize &&
        layer10_tail_gpu.layer_output.size() == kHiddenSize;
    const bool layer10_ffn_q6_boundary =
        layer10_selected_down_tensor->type == 14 &&
        layer10_shared_down_tensor->type == 14;
    const bool layer10_ok =
''',
  )
  cpp = replace_once(
      cpp,
      '''        metadata_ok &&
        layer10_comparisons_ok &&
        layer10_timing_positive &&
        layer10_arc_selected &&
        layer10_v_gpu_boundary;
''',
      '''        metadata_ok &&
        layer10_comparisons_ok &&
        layer10_live_ffn_lout_ok &&
        layer10_timing_positive &&
        layer10_arc_selected &&
        layer10_v_gpu_boundary &&
        layer10_ffn_tensor_shapes_ok &&
        layer10_ffn_down_q4_q6_boundary;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer10_attention_sum_min =
        layer10_input_sum_min + layer10_core_output_sum_min;
''',
      '''    const double layer10_attention_sum_min =
        layer10_input_sum_min + layer10_core_output_sum_min;
    const double layer10_ffn_sum_min =
        layer10_selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        layer10_shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        layer10_tail_gpu.timing.shell_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer15_v_projection_boundary\\":\\""
              << (layer10_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer15_ffn_boundary\\":\\"q6_down_pending\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer15_v_projection_boundary\\":\\""
              << (layer10_tensors.v_tensor->type == 14 ? "gpu_q6_raw_matvec" : "gpu_q4x8_matvec") << "\\",";
    std::cout << "\\"layer15_ffn_boundary\\":\\"gpu_live_post_norm_to_q4_q6_down\\",";
    std::cout << "\\"layer15_ffn_input_boundary\\":\\"live_gpu_layer15_post_attention_norm\\",";
    std::cout << "\\"layer15_layer_output_residual_boundary\\":\\"live_gpu_layer15_attention_residual\\",";
    std::cout << "\\"layer15_lout_boundary\\":\\"live_gpu_l_out_15\\",";
    std::cout << "\\"layer15_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer10_selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer15_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer10_shared_down_tensor->type)) << "\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer15_v_projection_device_name\\":\\"" << JsonEscape(layer10_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer15_v_projection_device_name\\":\\"" << JsonEscape(layer10_v_gpu.device_name) << "\\",";
    std::cout << "\\"layer15_selected_ffn_device_name\\":\\"" << JsonEscape(layer10_selected_gpu.device_name) << "\\",";
    std::cout << "\\"layer15_shared_ffn_device_name\\":\\"" << JsonEscape(layer10_shared_gpu.device_name) << "\\",";
    std::cout << "\\"layer15_lout_device_name\\":\\"" << JsonEscape(layer10_tail_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer10_rms_gpu.program_build_ms + layer10_qk_gpu.program_build_ms +
                  layer10_v_gpu.program_build_ms +
                  layer10_core_gate_gpu.program_build_ms +
                  layer10_attention_gpu.program_build_ms)
''',
      '''                  layer10_rms_gpu.program_build_ms + layer10_qk_gpu.program_build_ms +
                  layer10_v_gpu.program_build_ms +
                  layer10_core_gate_gpu.program_build_ms +
                  layer10_attention_gpu.program_build_ms +
                  layer10_selected_gpu.program_build_ms +
                  layer10_shared_gpu.program_build_ms +
                  layer10_tail_gpu.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer10_rms_gpu.build_log + layer10_qk_gpu.build_log +
                            layer10_v_gpu.build_log +
                            layer10_core_gate_gpu.build_log +
                            layer10_attention_gpu.build_log)
''',
      '''                            layer10_rms_gpu.build_log + layer10_qk_gpu.build_log +
                            layer10_v_gpu.build_log +
                            layer10_core_gate_gpu.build_log +
                            layer10_attention_gpu.build_log +
                            layer10_selected_gpu.build_log +
                            layer10_shared_gpu.build_log +
                            layer10_tail_gpu.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer15_full_attention_kernel_sum_min_us\\":"
              << layer10_attention_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_full_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min);
''',
      '''    std::cout << "\\"resident_layer15_full_attention_kernel_sum_min_us\\":"
              << layer10_attention_sum_min << ",";
    std::cout << "\\"resident_layer15_ffn_lout_kernel_sum_min_us\\":"
              << layer10_ffn_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_full_attention_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min) << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_lout_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min + layer8_state_input_sum_min +
                  layer9_state_input_sum_min + layer10_attention_sum_min + layer10_ffn_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteNamedCompareGroups(layer10_strict_groups);
    std::cout << ",\\"l15_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer10_k_raw_gpu_vs_cpu);
    std::cout << "},";
    WriteNamedCompareGroups(layer10_full_attention_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WriteNamedCompareGroups(layer10_strict_groups);
    std::cout << ",\\"l15_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer10_k_raw_gpu_vs_cpu);
    std::cout << "},";
    WriteNamedCompareGroups(layer10_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer10_ffn_live_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer15_v_gpu_boundary\\":"
              << (layer10_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_ffn_q6_boundary\\":"
              << (layer10_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_arc_device_selected\\":"
''',
      '''    std::cout << "\\"layer15_v_gpu_boundary\\":"
              << (layer10_v_gpu_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_ffn_tensor_shapes_ok\\":"
              << (layer10_ffn_tensor_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_ffn_down_q4_q6_boundary\\":"
              << (layer10_ffn_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_live_ffn_gpu_cpu_matches_native\\":"
              << (layer10_live_ffn_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_live_ffn_oracle_magnitude_policy_matches\\":"
              << (layer10_live_ffn_oracle_magnitude_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_live_layer_output_full_policy_matches_oracle\\":"
              << (layer10_live_layer_output_full_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_ffn_lout_handoff_matches\\":"
              << (layer10_live_ffn_lout_ok ? "true" : "false") << ",";
    std::cout << "\\"layer15_lout_boundary_live_gpu\\":true,";
    std::cout << "\\"layer15_ffn_q6_boundary\\":"
              << (layer10_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer15_arc_device_selected\\":"
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
  parser.add_argument("--layer15-conv-history-probe", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "layer5_to_layer15_ffn_lout_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer12_residual_input_boundary") == "live_gpu_l_out_11"
      and probe.get("layer13_residual_input_boundary") == "live_gpu_l_out_12"
      and probe.get("layer14_residual_input_boundary") == "live_gpu_l_out_13"
      and probe.get("layer15_residual_input_boundary") == "live_gpu_l_out_14"
      and probe.get("layer13_qkv_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer13_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer13_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer13_conv_state_boundary") == "captured_conv_state"
      and probe.get("layer14_qkv_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer14_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer14_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer14_conv_state_boundary") == "captured_conv_state"
      and probe.get("layer15_v_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer15_v_projection_boundary") in {"gpu_q4x8_matvec", "gpu_q6_raw_matvec"}
      and probe.get("layer15_ffn_boundary") == "gpu_live_post_norm_to_q4_q6_down"
      and probe.get("layer15_ffn_input_boundary") == "live_gpu_layer15_post_attention_norm"
      and probe.get("layer15_layer_output_residual_boundary") == "live_gpu_layer15_attention_residual"
      and probe.get("layer15_lout_boundary") == "live_gpu_l_out_15"
      and probe.get("layer15_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer15_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def layer14_predecessor_checks(probe: dict[str, Any] | None) -> list[dict[str, Any]]:
  return [
      {"name": "layer14_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer14_state_input_handoff_matches")},
      {"name": "l14_final_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l14_final_output", "gpu_vs_oracle")},
      {"name": "l14_linear_attn_out_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l14_linear_attn_out", "gpu_vs_oracle")},
      {"name": "l14_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l14_attn_output", "gpu_vs_oracle")},
      {"name": "l14_attn_residual_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l14_attn_residual", "gpu_vs_oracle")},
      {"name": "l14_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l14_attn_post_norm", "gpu_vs_oracle")},
      {"name": "l14_selected_gate_up_matches_oracle", "pass": CORE.comparison_passed(probe, "l14_selected_gate_up", "gpu_vs_oracle")},
      {"name": "l14_selected_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l14_selected_down", "gpu_vs_oracle")},
      {"name": "l14_shared_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l14_shared_down", "gpu_vs_oracle")},
      {"name": "l14_ffn_moe_out_matches_oracle", "pass": CORE.comparison_passed(probe, "l14_ffn_moe_out", "gpu_vs_oracle")},
      {"name": "l14_ffn_shexp_gated_matches_oracle", "pass": CORE.comparison_passed(probe, "l14_ffn_shexp_gated", "gpu_vs_oracle")},
      {"name": "l14_ffn_out_matches_oracle", "pass": CORE.comparison_passed(probe, "l14_ffn_out", "gpu_vs_oracle")},
      {"name": "l14_layer_output_lout_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l14_layer_output", "gpu_vs_oracle")},
      {"name": "l14_conv_state_after_matches_native", "pass": CORE.comparison_passed(probe, "l14_conv_state_after", "gpu_vs_cpu")},
      {"name": "l14_recurrent_state_matches_native", "pass": CORE.comparison_passed(probe, "l14_recurrent_state", "gpu_vs_cpu")},
      {"name": "layer14_lout_boundary_live_gpu", "pass": isinstance(probe, dict) and probe.get("layer14_residual_input_boundary") == "live_gpu_l_out_13" and "B390" in str(probe.get("layer14_tail_device_name", ""))},
  ]


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-15 FFN/l_out Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 15 residual input boundary: `{probe.get('layer15_residual_input_boundary')}`",
      f"- layer 15 FFN boundary: `{probe.get('layer15_ffn_boundary')}`",
      f"- layer 15 selected down type: `{probe.get('layer15_selected_down_tensor_type')}`",
      f"- layer 15 shared down type: `{probe.get('layer15_shared_down_tensor_type')}`",
      f"- layer 15 l_out boundary: `{probe.get('layer15_lout_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l15_selected_down",
      "l15_shared_down",
      "l15_ffn_out",
      "l15_layer_output",
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
      f"| layer15_full_attention | {timings.get('resident_layer15_full_attention_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer15_ffn_lout | {timings.get('resident_layer15_ffn_lout_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| through_layer15_lout | {timings.get('resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_lout_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 14, runs layer-15 full attention from live GPU `l_out-14`, then",
      "runs layer-15 FFN and residual output from live GPU layer-15 post-attn",
      "state. This is single-token handoff evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-15 FFN/l_out handoff")

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
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer15-ffn-lout-handoff-probe-{stamp}"
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
  add_layer15_ffn_payloads(layer15_payloads)
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
  local_cpp = out_dir / "gpu_resident_layer15_ffn_lout_handoff_probe.cpp"
  local_cpp.write_text(layer15_ffn_lout_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer15-ffn-lout-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer15_ffn_lout_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer15-ffn-lout-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer15_ffn_lout_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")) and "B390" in str(probe.get("layer15_lout_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
  ]
  checks.extend(layer14_predecessor_checks(probe))
  checks.extend([
      {"name": "layer15_residual_input_from_layer14_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer15_residual_input_from_layer14_live_gpu_lout")},
      {"name": "layer15_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer15_payload_counts_ok")},
      {"name": "layer15_full_attn_input_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer15_full_attn_input_matches_oracle")},
      {"name": "layer15_full_attn_component_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer15_full_attn_component_policy_matches_oracle")},
      {"name": "layer15_full_attn_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer15_full_attn_core_output_matches_oracle")},
      {"name": "layer15_v_gpu_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer15_v_gpu_boundary")},
      {"name": "layer15_ffn_tensor_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer15_ffn_tensor_shapes_ok")},
      {"name": "layer15_ffn_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer15_ffn_down_q4_q6_boundary")},
      {"name": "layer15_live_ffn_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer15_live_ffn_gpu_cpu_matches_native")},
      {"name": "layer15_live_ffn_oracle_magnitude_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer15_live_ffn_oracle_magnitude_policy_matches")},
      {"name": "layer15_live_layer_output_full_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer15_live_layer_output_full_policy_matches_oracle")},
      {"name": "layer15_ffn_lout_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer15_ffn_lout_handoff_matches")},
      {"name": "layer15_lout_boundary_live_gpu", "pass": PRECONV.nested_bool(probe, "checks", "layer15_lout_boundary_live_gpu")},
      {"name": "layer15_arc_device_selected", "pass": PRECONV.nested_bool(probe, "checks", "layer15_arc_device_selected")},
      {"name": "l15_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_residual_input", "gpu_vs_oracle")},
      {"name": "l15_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_attn_norm", "gpu_vs_oracle")},
      {"name": "l15_q_full_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_q_full", "gpu_vs_oracle")},
      {"name": "l15_q_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_q_rope", "gpu_vs_oracle")},
      {"name": "l15_k_rope_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_k_rope", "gpu_vs_oracle")},
      {"name": "l15_v_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_v", "gpu_vs_oracle")},
      {"name": "l15_attn_pregate_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_attn_pregate", "gpu_vs_oracle")},
      {"name": "l15_attn_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_attn_output", "gpu_vs_oracle")},
      {"name": "l15_attn_post_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_attn_post_norm", "gpu_vs_oracle")},
      {"name": "l15_selected_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l15_selected_down", "gpu_vs_oracle")},
      {"name": "l15_shared_down_matches_oracle", "pass": CORE.comparison_passed(probe, "l15_shared_down", "gpu_vs_oracle")},
      {"name": "l15_ffn_out_matches_oracle", "pass": CORE.comparison_passed(probe, "l15_ffn_out", "gpu_vs_oracle")},
      {"name": "l15_layer_output_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l15_layer_output", "gpu_vs_oracle")},
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
      "layer14_full_shell_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer15_full_attention_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
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
          "layer13": str(conv6_path.relative_to(ROOT)),
          "layer9": str(conv7_path.relative_to(ROOT)),
          "layer14": str(conv7_path.relative_to(ROOT)),
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
          "layer3": conv2.get("capture_artifact"),
          "layer4": conv3.get("capture_artifact"),
          "layer5": conv4.get("capture_artifact"),
          "layer7": conv5.get("capture_artifact"),
          "layer8": conv6.get("capture_artifact"),
          "layer13": conv6.get("capture_artifact"),
          "layer9": conv7.get("capture_artifact"),
          "layer14": conv7.get("capture_artifact"),
      },
      "all_history": {
          "layer7": layer7_history,
          "layer11": layer11_history,
          "layer15": layer15_history,
      },
      "ffn_payload_root": str(LAYER15_FFN_PAYLOAD_ROOT.relative_to(ROOT)),
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10],
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
      "tool": "tools/intel-qwen36-gpu-resident-layer15-ffn-lout-handoff-probe.py",
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
      "gpu_resident_layer15_ffn_lout_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer15_full_attn_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer15_full_attn_input_kernel_sum_min_us")),
          ("resident_layer15_full_attn_core_output_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer15_full_attn_core_output_kernel_sum_min_us")),
          ("resident_layer15_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer15_full_attention_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_full_attention_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_full_attention_kernel_sum_min_us")),
          ("resident_layer15_ffn_lout_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer15_ffn_lout_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_lout_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_12_13_14_to_layer15_lout_kernel_sum_min_us")),
          ("layer15_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer15_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer15_q_full_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_q_full", "gpu_vs_oracle", "max_abs_diff")),
          ("layer15_attn_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_attn_output", "gpu_vs_oracle", "max_abs_diff")),
          ("layer15_attn_post_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_attn_post_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("l15_selected_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_selected_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l15_shared_down_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_shared_down", "gpu_vs_oracle", "max_abs_diff")),
          ("l15_ffn_out_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_ffn_out", "gpu_vs_oracle", "max_abs_diff")),
          ("l15_layer_output_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l15_layer_output", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
