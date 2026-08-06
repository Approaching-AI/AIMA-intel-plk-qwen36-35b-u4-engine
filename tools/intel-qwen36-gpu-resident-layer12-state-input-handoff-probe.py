#!/usr/bin/env python3
"""Run the resident GPU layer-12 state/input handoff probe."""

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
L11_FFN_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer11-ffn-lout-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer12-state-input-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"


def load_l11_ffn_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer11_ffn_lout_probe", L11_FFN_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer11 FFN/l_out tool: {L11_FFN_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L11_FFN = load_l11_ffn_tool()
L11 = L11_FFN.L11
CORE = L11_FFN.CORE
L7_INPUT = L11_FFN.L7_INPUT
L8 = L11_FFN.L8
TWO = L11_FFN.TWO
PRECONV = L11_FFN.PRECONV


def replace_once(text: str, old: str, new: str) -> str:
  count = text.count(old)
  if count != 1:
    raise SystemExit(f"expected exactly one source replacement for {old[:80]!r}, found {count}")
  return text.replace(old, new, 1)


def layer12_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L11_FFN.layer11_ffn_lout_probe_cpp(opencl_source)
  for old, new in {
      L11_FFN.SCHEMA_VERSION: SCHEMA_VERSION,
      "layer5_to_layer11_ffn_lout_load_once_run_many":
          "layer5_to_layer12_state_input_load_once_run_many",
      "layer11 full-attention handoff probe expects --layer 5":
          "layer12 state/input handoff probe expects --layer 5",
  }.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer6 = args.layer + 6;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11,
            "layer12 state/input handoff probe expects --layer 5");
''',
      '''    const int layer6 = args.layer + 6;
    const int layer7 = args.layer + 7;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9 && layer5 == 10 && layer6 == 11 && layer7 == 12,
            "layer12 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer6_tensors = ResolveFullAttentionTensorBundle(index, layer6);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer6_tensors = ResolveFullAttentionTensorBundle(index, layer6);
    const auto layer7_tensors = ResolveLayerTensorBundle(index, layer7);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer6_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l6_");
    const auto oracle_attn_residual =
''',
      '''    const auto layer6_oracle = LoadFullAttentionPayloadsPrefixed(args.payload_dir, "l6_");
    const auto layer7_oracle = LoadLayerOraclePayloads(args.payload_dir, "l12");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer6_tail_gpu = RunGpuShell(
        layer6_shared_input_gate_weights, layer6_ffn_input,
        layer6_selected_gpu.down, layer6_oracle_weights_norm,
        layer6_shared_gpu.down, layer6_attention_gpu.attn_residual,
        args.device_substring, args.repeat);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer6_tail_gpu = RunGpuShell(
        layer6_shared_input_gate_weights, layer6_ffn_input,
        layer6_selected_gpu.down, layer6_oracle_weights_norm,
        layer6_shared_gpu.down, layer6_attention_gpu.attn_residual,
        args.device_substring, args.repeat);

    const auto layer7_run = RunResidentLinearLayerShell(
        args, index, layer7_tensors, layer7_oracle,
        layer6_tail_gpu.layer_output, layer6_tail_gpu.layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer6_ok =
        layer6_full_shapes_ok &&
        layer6_payload_counts_ok &&
        metadata_ok &&
        layer6_comparisons_ok &&
        layer6_live_ffn_lout_ok &&
        layer6_timing_positive &&
        layer6_arc_selected &&
        layer6_v_gpu_boundary &&
        layer6_ffn_tensor_shapes_ok &&
        layer6_ffn_down_q4_q6_boundary;
    const bool layer2_timing_positive =
''',
      '''    const bool layer6_ok =
        layer6_full_shapes_ok &&
        layer6_payload_counts_ok &&
        metadata_ok &&
        layer6_comparisons_ok &&
        layer6_live_ffn_lout_ok &&
        layer6_timing_positive &&
        layer6_arc_selected &&
        layer6_v_gpu_boundary &&
        layer6_ffn_tensor_shapes_ok &&
        layer6_ffn_down_q4_q6_boundary;
    const auto find_l7_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer7_run.comparisons.begin(),
              layer7_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer7_run.comparisons.end(),
                  "layer12 comparison missing: " + name);
          return *found;
        };
    const auto& layer7_residual_input = find_l7_group("residual_input");
    const auto& layer7_attn_norm = find_l7_group("attn_norm");
    const auto& layer7_qkv = find_l7_group("linear_attn_qkv_mixed");
    const auto& layer7_conv_output_raw = find_l7_group("conv_output_raw");
    const bool layer7_state_input_gpu_cpu_ok =
        ComparePassed(layer7_residual_input.gpu_vs_cpu) &&
        ComparePassed(layer7_attn_norm.gpu_vs_cpu) &&
        ComparePassed(layer7_qkv.gpu_vs_cpu) &&
        ComparePassed(layer7_conv_output_raw.gpu_vs_cpu) &&
        ComparePassed(layer7_run.conv_state_after_gpu_vs_cpu);
    const bool layer7_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer7_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer7_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer7_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer7_conv_output_raw.gpu_vs_oracle);
    const bool layer7_shapes_ok = ShapesPassed(layer7_run.shape_checks);
    const bool layer7_qkv_q4_q6_boundary =
        (layer7_tensors.qkv_tensor->type == 12 || layer7_tensors.qkv_tensor->type == 14);
    const bool layer7_down_q4_q6_boundary =
        (layer7_tensors.selected_down_tensor->type == 12 ||
         layer7_tensors.selected_down_tensor->type == 14) &&
        (layer7_tensors.shared_down_tensor->type == 12 ||
         layer7_tensors.shared_down_tensor->type == 14);
    const bool layer7_state_input_timing_positive =
        layer7_run.timing.layer_input_rmsnorm_min_us > 0.0 &&
        layer7_run.timing.preconv_to_postconv_kernel_sum_min_us > 0.0;
    const bool layer7_ok =
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
  )
  cpp = replace_once(
      cpp,
      '''        layer4_ok &&
        layer5_ok &&
        layer6_ok &&
        layer2_timing_positive &&
''',
      '''        layer4_ok &&
        layer5_ok &&
        layer6_ok &&
        layer7_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer6_ffn_sum_min =
        layer6_selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        layer6_shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        layer6_tail_gpu.timing.shell_sum_min_us;
''',
      '''    const double layer6_ffn_sum_min =
        layer6_selected_gpu.timing.selected_ffn_kernel_sum_min_us +
        layer6_shared_gpu.timing.shared_ffn_kernel_sum_min_us +
        layer6_tail_gpu.timing.shell_sum_min_us;
    const double layer7_state_input_sum_min =
        layer7_run.timing.layer_input_rmsnorm_min_us +
        layer7_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "," << layer5 << "," << layer6 << "," << layer7 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer11_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer6_shared_down_tensor->type)) << "\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
      '''    std::cout << "\\"layer11_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer6_shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer12_residual_input_boundary\\":\\"live_gpu_l_out_11\\",";
    std::cout << "\\"layer12_qkv_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer7_tensors.qkv_tensor->type)) << "\\",";
    std::cout << "\\"layer12_selected_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer7_tensors.selected_down_tensor->type)) << "\\",";
    std::cout << "\\"layer12_shared_down_tensor_type\\":\\""
              << JsonEscape(iq36::ggml_type_name(layer7_tensors.shared_down_tensor->type)) << "\\",";
    std::cout << "\\"layer12_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"full_attn_core_input_boundary\\":\\"captured_rope_kv_payloads\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer11_lout_device_name\\":\\"" << JsonEscape(layer6_tail_gpu.device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
      '''    std::cout << "\\"layer11_lout_device_name\\":\\"" << JsonEscape(layer6_tail_gpu.device_name) << "\\",";
    std::cout << "\\"layer12_layer_input_device_name\\":\\"" << JsonEscape(layer7_run.layer_input_device_name) << "\\",";
    std::cout << "\\"layer12_preconv_device_name\\":\\"" << JsonEscape(layer7_run.preconv_device_name) << "\\",";
    std::cout << "\\"layer12_tail_device_name\\":\\"" << JsonEscape(layer7_run.tail_device_name) << "\\",";
    std::cout << "\\"layer7_v_tensor_type\\":\\""
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer6_selected_gpu.program_build_ms +
                  layer6_shared_gpu.program_build_ms +
                  layer6_tail_gpu.program_build_ms)
''',
      '''                  layer6_selected_gpu.program_build_ms +
                  layer6_shared_gpu.program_build_ms +
                  layer6_tail_gpu.program_build_ms +
                  layer7_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            layer6_selected_gpu.build_log +
                            layer6_shared_gpu.build_log +
                            layer6_tail_gpu.build_log)
''',
      '''                            layer6_selected_gpu.build_log +
                            layer6_shared_gpu.build_log +
                            layer6_tail_gpu.build_log +
                            layer7_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_to_layer11_lout_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min);
''',
      '''    std::cout << "\\"resident_layer5_6_7_8_9_10_to_layer11_lout_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min) << ",";
    std::cout << "\\"resident_layer12_state_input_kernel_sum_min_us\\":"
              << layer7_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_run.timing.layer_kernel_sum_min_us +
                  layer5_run.timing.layer_kernel_sum_min_us +
                  layer6_attention_sum_min + layer6_ffn_sum_min +
                  layer7_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WriteNamedCompareGroups(layer6_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer6_ffn_live_groups);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WriteNamedCompareGroups(layer6_full_attention_groups);
    std::cout << ",";
    WriteNamedCompareGroups(layer6_ffn_live_groups);
    WritePrefixedCompareGroups("l12", layer7_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l6_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer6_k_raw_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l6_k_raw\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer6_k_raw_gpu_vs_cpu);
    std::cout << "},\\"l12_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer7_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l12_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer7_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer5\\":";
    WriteLayerChecks(layer5_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer5\\":";
    WriteLayerChecks(layer5_run);
    std::cout << ",\\"layer7\\":";
    WriteLayerChecks(layer7_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer11_lout_boundary_live_gpu\\":true,";
    std::cout << "\\"layer11_ffn_q6_boundary\\":"
              << (layer6_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer11_arc_device_selected\\":"
              << (layer6_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer11_lout_boundary_live_gpu\\":true,";
    std::cout << "\\"layer11_ffn_q6_boundary\\":"
              << (layer6_ffn_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer11_arc_device_selected\\":"
              << (layer6_arc_selected ? "true" : "false") << ",";
    std::cout << "\\"layer12_residual_input_from_layer11_live_gpu_lout\\":true,";
    std::cout << "\\"layer12_payload_counts_ok\\":"
              << (layer7_run.payload_counts_ok ? "true" : "false") << ",";
    std::cout << "\\"layer12_shapes_ok\\":"
              << (layer7_shapes_ok ? "true" : "false") << ",";
    std::cout << "\\"layer12_qkv_q4_q6_boundary\\":"
              << (layer7_qkv_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer12_down_q4_q6_boundary\\":"
              << (layer7_down_q4_q6_boundary ? "true" : "false") << ",";
    std::cout << "\\"layer12_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer12_gpu_cpu_matches_native\\":"
              << (layer7_state_input_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer12_state_input_oracle_policy_matches\\":"
              << (layer7_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer12_state_input_handoff_matches\\":"
              << (layer7_ok ? "true" : "false") << ",";
    std::cout << "\\"layer12_full_shell_matches_oracle\\":"
              << (layer7_run.comparisons_passed ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''                  layer4_run.timing_positive && layer5_run.timing_positive &&
                  layer6_timing_positive ? "true" : "false") << ",";
''',
      '''                  layer4_run.timing_positive && layer5_run.timing_positive &&
                  layer6_timing_positive && layer7_state_input_timing_positive ? "true" : "false") << ",";
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
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "layer5_to_layer12_state_input_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("layer11_lout_boundary") == "live_gpu_l_out_11"
      and probe.get("layer12_residual_input_boundary") == "live_gpu_l_out_11"
      and probe.get("layer12_qkv_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer12_selected_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer12_shared_down_tensor_type") in {"Q4_K", "Q6_K"}
      and probe.get("layer12_conv_state_boundary") == "captured_conv_state"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def comparison_passed(probe: dict[str, Any] | None, name: str, lane: str) -> bool:
  return CORE.comparison_passed(probe, name, lane)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-12 State/Input Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 12 residual input boundary: `{probe.get('layer12_residual_input_boundary')}`",
      f"- layer 12 qkv tensor type: `{probe.get('layer12_qkv_tensor_type')}`",
      f"- layer 12 selected/shared down: `{probe.get('layer12_selected_down_tensor_type')}` / `{probe.get('layer12_shared_down_tensor_type')}`",
      f"- layer 12 conv state boundary: `{probe.get('layer12_conv_state_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l12_residual_input",
      "l12_attn_norm",
      "l12_linear_attn_qkv_mixed",
      "l12_conv_output_raw",
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
      f"| layer12_state_input | {timings.get('resident_layer12_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| through_layer12_state_input | {timings.get('resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries closed live GPU layer outputs through",
      "layer 11, then feeds live GPU `l_out-11` into layer 12 RMSNorm, Q4/Q6",
      "QKV, and F32 conv with captured layer-12 conv state. This is captured",
      "single-token state/input evidence, not decode throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-12 state/input handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  layer3 = args.layer + 3
  layer4 = args.layer + 4
  layer5 = args.layer + 5
  layer6 = args.layer + 6
  layer7 = args.layer + 7
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
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer12-state-input-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  payloads2, conv2 = TWO.prefixed_payloads(layer3, conv2_path, "l3")
  payloads3, conv3 = TWO.prefixed_payloads(layer4, conv3_path, "l4")
  payloads4, conv4 = TWO.prefixed_payloads(layer5, conv4_path, "l5")
  payloads5, conv5 = TWO.prefixed_payloads(layer7, conv5_path, "l12")
  layer7_payloads, layer7_history = L7_INPUT.resolve_full_attention_payloads(
      all_history_json, layer2
  )
  CORE.add_layer7_tail_payloads(layer7_payloads)
  L8.add_layer7_ffn_payloads(layer7_payloads)
  layer11_payloads, layer11_history = L11.resolve_prefixed_full_attention_payloads(
      all_history_json, layer6, "l6_"
  )
  L11_FFN.add_layer11_ffn_payloads(layer11_payloads)
  payloads = {
      **payloads0,
      **payloads1,
      **layer7_payloads,
      **payloads2,
      **payloads3,
      **payloads4,
      **layer11_payloads,
      **payloads5,
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
  local_cpp = out_dir / "gpu_resident_layer12_state_input_handoff_probe.cpp"
  local_cpp.write_text(layer12_state_input_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer12-state-input-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer12_state_input_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer12-state-input-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer12_state_input_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")) and "B390" in str(probe.get("layer12_preconv_device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
      {"name": "layer11_ffn_lout_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer11_ffn_lout_handoff_matches")},
      {"name": "layer12_residual_input_from_layer11_live_gpu_lout", "pass": PRECONV.nested_bool(probe, "checks", "layer12_residual_input_from_layer11_live_gpu_lout")},
      {"name": "layer12_payload_counts_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer12_payload_counts_ok")},
      {"name": "layer12_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer12_shapes_ok")},
      {"name": "layer12_qkv_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer12_qkv_q4_q6_boundary")},
      {"name": "layer12_down_q4_q6_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer12_down_q4_q6_boundary")},
      {"name": "layer12_conv_state_input_boundary", "pass": PRECONV.nested_bool(probe, "checks", "layer12_conv_state_input_boundary")},
      {"name": "layer12_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer12_gpu_cpu_matches_native")},
      {"name": "layer12_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer12_state_input_oracle_policy_matches")},
      {"name": "layer12_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer12_state_input_handoff_matches")},
      {"name": "l12_residual_input_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_residual_input", "gpu_vs_oracle")},
      {"name": "l12_attn_norm_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_attn_norm", "gpu_vs_oracle")},
      {"name": "l12_qkv_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_linear_attn_qkv_mixed", "gpu_vs_oracle")},
      {"name": "l12_conv_output_raw_matches_oracle", "pass": CORE.full_attention_comparison_passed(probe, "l12_conv_output_raw", "gpu_vs_oracle")},
      {"name": "gpu_event_timing_positive", "pass": PRECONV.nested_bool(probe, "checks", "gpu_event_timing_positive")},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  comparison_thresholds = {
      "strict_component": CORE.STRICT_COMPARISON_THRESHOLDS,
      "full_attn_component": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer12_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
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
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
          "layer3": conv2.get("capture_artifact"),
          "layer4": conv3.get("capture_artifact"),
          "layer5": conv4.get("capture_artifact"),
          "layer7": conv5.get("capture_artifact"),
      },
      "all_history": {
          "layer7": layer7_history,
          "layer11": layer11_history,
      },
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7],
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
      "tool": "tools/intel-qwen36-gpu-resident-layer12-state-input-handoff-probe.py",
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
      "gpu_resident_layer12_state_input_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer12_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer12_state_input_kernel_sum_min_us")),
          ("resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer5_6_7_8_9_10_11_to_layer12_state_input_kernel_sum_min_us")),
          ("layer12_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer12_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer12_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer12_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l12_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
