#!/usr/bin/env python3
"""Run the resident GPU layer-9 state/input handoff probe."""

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
L8_TOOL = Path(__file__).with_name(
    "intel-qwen36-gpu-resident-layer8-state-input-handoff-probe.py"
)
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-resident-layer9-state-input-handoff-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
DEFAULT_ALL_HISTORY = ROOT / "output/r1-full-attn-all-history-capture-20260627T145615Z/history.json"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
FFN_PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-conv-state-filter-probe-20260630T000001Z/remote-output/payloads"


def load_l8_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_layer8_state_input_probe", L8_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load layer8 state/input tool: {L8_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


L8 = load_l8_tool()
V_Q6 = L8.V_Q6
CORE = L8.CORE
L7_INPUT = L8.L7_INPUT
TWO = L8.TWO
PRECONV = L8.PRECONV


def layer9_state_input_probe_cpp(opencl_source: str) -> str:
  cpp = L8.layer8_state_input_probe_cpp(opencl_source)
  replace_once = L8.replace_once
  replacements = {
      L8.SCHEMA_VERSION: SCHEMA_VERSION,
      "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer8_state_input_load_once_run_many":
          "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer9_state_input_load_once_run_many",
      "layer8 state/input handoff probe expects --layer 5":
          "layer9 state/input handoff probe expects --layer 5",
  }
  for old, new in replacements.items():
    cpp = replace_once(cpp, old, new)
  cpp = replace_once(
      cpp,
      '''    const int layer3 = args.layer + 3;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8,
            "layer9 state/input handoff probe expects --layer 5");
''',
      '''    const int layer3 = args.layer + 3;
    const int layer4 = args.layer + 4;
    Require(layer0 == 5 && layer1 == 6 && layer2 == 7 && layer3 == 8 && layer4 == 9,
            "layer9 state/input handoff probe expects --layer 5");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer3_tensors = ResolveLayerTensorBundle(index, layer3);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
      '''    const auto layer3_tensors = ResolveLayerTensorBundle(index, layer3);
    const auto layer4_tensors = ResolveLayerTensorBundle(index, layer4);
    const auto layer0_oracle = LoadLayerOraclePayloads(args.payload_dir, "l0");
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer3_oracle = LoadLayerOraclePayloads(args.payload_dir, "l3");
    const auto oracle_attn_residual =
''',
      '''    const auto layer3_oracle = LoadLayerOraclePayloads(args.payload_dir, "l3");
    const auto layer4_oracle = LoadLayerOraclePayloads(args.payload_dir, "l4");
    const auto oracle_attn_residual =
''',
  )
  cpp = replace_once(
      cpp,
      '''    const auto layer3_run = RunResidentLinearLayerShell(
        args, index, layer3_tensors, layer3_oracle,
        native_layer_output, tail_gpu.layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
      '''    const auto layer3_run = RunResidentLinearLayerShell(
        args, index, layer3_tensors, layer3_oracle,
        native_layer_output, tail_gpu.layer_output, rms_norm_epsilon);
    const auto layer4_run = RunResidentLinearLayerShell(
        args, index, layer4_tensors, layer4_oracle,
        layer3_run.gpu_layer_output, layer3_run.gpu_layer_output, rms_norm_epsilon);

    std::vector<NamedCompareGroup> strict_groups;
''',
  )
  cpp = replace_once(
      cpp,
      '''    const bool layer3_ok =
        layer3_shapes_ok &&
        layer3_run.payload_counts_ok &&
        layer3_gpu_cpu_ok &&
        layer3_state_input_oracle_policy_ok &&
        layer3_run.timing_positive &&
        layer3_run.arc_selected;
    const bool layer2_timing_positive =
''',
      '''    const bool layer3_ok =
        layer3_shapes_ok &&
        layer3_run.payload_counts_ok &&
        layer3_gpu_cpu_ok &&
        layer3_state_input_oracle_policy_ok &&
        layer3_run.timing_positive &&
        layer3_run.arc_selected;
    const auto find_l4_group =
        [&](const std::string& name) -> const NamedCompareGroup& {
          const auto found = std::find_if(
              layer4_run.comparisons.begin(),
              layer4_run.comparisons.end(),
              [&](const NamedCompareGroup& group) {
                return group.name == name;
              });
          Require(found != layer4_run.comparisons.end(),
                  "layer9 comparison missing: " + name);
          return *found;
        };
    const auto& layer4_residual_input = find_l4_group("residual_input");
    const auto& layer4_attn_norm = find_l4_group("attn_norm");
    const auto& layer4_qkv = find_l4_group("linear_attn_qkv_mixed");
    const auto& layer4_conv_output_raw = find_l4_group("conv_output_raw");
    bool layer4_gpu_cpu_ok = true;
    for (const auto& group : layer4_run.comparisons) {
      layer4_gpu_cpu_ok = layer4_gpu_cpu_ok && ComparePassed(group.gpu_vs_cpu);
    }
    layer4_gpu_cpu_ok =
        layer4_gpu_cpu_ok &&
        ComparePassed(layer4_run.conv_state_after_gpu_vs_cpu) &&
        ComparePassed(layer4_run.recurrent_state_gpu_vs_cpu);
    const bool layer4_state_input_oracle_policy_ok =
        ComparePassedFullAttentionComponent(layer4_residual_input.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer4_attn_norm.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer4_qkv.gpu_vs_oracle) &&
        ComparePassedFullAttentionComponent(layer4_conv_output_raw.gpu_vs_oracle);
    const bool layer4_shapes_ok = ShapesPassed(layer4_run.shape_checks);
    const bool layer4_ok =
        layer4_shapes_ok &&
        layer4_run.payload_counts_ok &&
        layer4_gpu_cpu_ok &&
        layer4_state_input_oracle_policy_ok &&
        layer4_run.timing_positive &&
        layer4_run.arc_selected;
    const bool layer2_timing_positive =
''',
  )
  cpp = replace_once(
      cpp,
      '''        layer2_live_ffn_lout_ok &&
        layer3_ok &&
        layer2_timing_positive &&
''',
      '''        layer2_live_ffn_lout_ok &&
        layer3_ok &&
        layer4_ok &&
        layer2_timing_positive &&
''',
  )
  cpp = replace_once(
      cpp,
      '''    const double layer3_state_input_sum_min =
        layer3_run.timing.layer_input_rmsnorm_min_us +
        layer3_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
      '''    const double layer3_state_input_sum_min =
        layer3_run.timing.layer_input_rmsnorm_min_us +
        layer3_run.timing.preconv_to_postconv_kernel_sum_min_us;
    const double layer4_state_input_sum_min =
        layer4_run.timing.layer_input_rmsnorm_min_us +
        layer4_run.timing.preconv_to_postconv_kernel_sum_min_us;
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "],";
''',
      '''    std::cout << "\\"layers\\":[" << layer0 << "," << layer1 << "," << layer2 << "," << layer3 << "," << layer4 << "],";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer8_residual_input_boundary\\":\\"live_gpu_l_out_7\\",";
    std::cout << "\\"layer8_conv_state_boundary\\":\\"captured_conv_state\\",";''',
      '''    std::cout << "\\"layer8_residual_input_boundary\\":\\"live_gpu_l_out_7\\",";
    std::cout << "\\"layer8_conv_state_boundary\\":\\"captured_conv_state\\",";
    std::cout << "\\"layer9_residual_input_boundary\\":\\"live_gpu_l_out_8\\",";
    std::cout << "\\"layer9_conv_state_boundary\\":\\"captured_conv_state\\",";''',
  )
  cpp = replace_once(
      cpp,
      '''                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  tail_gpu.program_build_ms + layer3_run.program_build_ms)
''',
      '''                  selected_gpu.program_build_ms + shared_gpu.program_build_ms +
                  tail_gpu.program_build_ms + layer3_run.program_build_ms +
                  layer4_run.program_build_ms)
''',
  )
  cpp = replace_once(
      cpp,
      '''                            selected_gpu.build_log + shared_gpu.build_log +
                            tail_gpu.build_log + layer3_run.build_log)
''',
      '''                            selected_gpu.build_log + shared_gpu.build_log +
                            tail_gpu.build_log + layer3_run.build_log +
                            layer4_run.build_log)
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"resident_layer8_state_input_kernel_sum_min_us\\":"
              << layer3_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_to_layer8_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_state_input_sum_min);
''',
      '''    std::cout << "\\"resident_layer8_state_input_kernel_sum_min_us\\":"
              << layer3_state_input_sum_min << ",";
    std::cout << "\\"resident_layer9_state_input_kernel_sum_min_us\\":"
              << layer4_state_input_sum_min << ",";
    std::cout << "\\"resident_layer5_6_7_8_to_layer9_state_input_kernel_sum_min_us\\":"
              << (two_layer_kernel_sum_min + layer2_attention_sum_min +
                  layer2_ffn_sum_min + layer3_run.timing.layer_kernel_sum_min_us +
                  layer4_state_input_sum_min);
''',
  )
  cpp = replace_once(
      cpp,
      '''    WritePrefixedCompareGroups("l3", layer3_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
      '''    WritePrefixedCompareGroups("l3", layer3_run.comparisons, &first_compare);
    WritePrefixedCompareGroups("l4", layer4_run.comparisons, &first_compare);
    std::cout << ",\\"l2_k_raw\\":{\\"gpu_vs_cpu\\":";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "},\\"l3_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer3_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
      '''    std::cout << "},\\"l3_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer3_run.recurrent_state_gpu_vs_cpu);
    std::cout << "},\\"l4_conv_state_after\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer4_run.conv_state_after_gpu_vs_cpu);
    std::cout << "},\\"l4_recurrent_state\\":{\\"gpu_vs_cpu\\":";
    WriteCompare(layer4_run.recurrent_state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\\"checks\\":{";
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << ",\\"layer3\\":";
    WriteLayerChecks(layer3_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
      '''    std::cout << ",\\"layer3\\":";
    WriteLayerChecks(layer3_run);
    std::cout << ",\\"layer4\\":";
    WriteLayerChecks(layer4_run);
    std::cout << ",\\"layer1_residual_input_matches_oracle\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"layer8_state_input_handoff_matches\\":"
              << (layer3_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
      '''    std::cout << "\\"layer8_state_input_handoff_matches\\":"
              << (layer3_ok ? "true" : "false") << ",";
    std::cout << "\\"layer9_residual_input_from_layer8_live_gpu_lout\\":true,";
    std::cout << "\\"layer9_conv_state_input_boundary\\":true,";
    std::cout << "\\"layer9_gpu_cpu_matches_native\\":"
              << (layer4_gpu_cpu_ok ? "true" : "false") << ",";
    std::cout << "\\"layer9_state_input_oracle_policy_matches\\":"
              << (layer4_state_input_oracle_policy_ok ? "true" : "false") << ",";
    std::cout << "\\"layer9_state_input_handoff_matches\\":"
              << (layer4_ok ? "true" : "false") << ",";
    std::cout << "\\"history_kv_state_payloads_present\\":"
''',
  )
  cpp = replace_once(
      cpp,
      '''    std::cout << "\\"gpu_event_timing_positive\\":"
              << (layer0_run.timing_positive && layer1_run.timing_positive &&
                  layer2_timing_positive ? "true" : "false") << ",";
''',
      '''    std::cout << "\\"gpu_event_timing_positive\\":"
              << (layer0_run.timing_positive && layer1_run.timing_positive &&
                  layer2_timing_positive && layer3_run.timing_positive &&
                  layer4_run.timing_positive ? "true" : "false") << ",";
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
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def resident_fields_ok(probe: dict[str, Any] | None, expected_invocations: int) -> bool:
  return (
      isinstance(probe, dict)
      and probe.get("resident_api") == "two_linear_layer_to_full_attention_v_q6_live_ffn_lout_layer9_state_input_load_once_run_many"
      and probe.get("resident_load_count") == 1
      and probe.get("resident_shell_invocations") == expected_invocations
      and probe.get("full_attn_v_projection_boundary") == "gpu_q6_raw_matvec"
      and probe.get("full_attn_ffn_boundary") == "gpu_live_post_norm_to_q6_down"
      and probe.get("ffn_input_boundary") == "live_gpu_post_attention_norm"
      and probe.get("layer_output_residual_boundary") == "live_gpu_attention_residual"
      and probe.get("layer8_residual_input_boundary") == "live_gpu_l_out_7"
      and probe.get("layer8_conv_state_boundary") == "captured_conv_state"
      and probe.get("layer9_residual_input_boundary") == "live_gpu_l_out_8"
      and probe.get("layer9_conv_state_boundary") == "captured_conv_state"
      and PRECONV.nested_bool(probe, "checks", "resident_load_once")
      and PRECONV.nested_bool(probe, "checks", "resident_shell_invocations_positive")
  )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Resident Layer-9 State/Input Handoff Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layers: `{payload.get('layers')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- resident API: `{probe.get('resident_api')}`",
      f"- resident shell invocations: `{probe.get('resident_shell_invocations')}`",
      f"- layer 9 residual input boundary: `{probe.get('layer9_residual_input_boundary')}`",
      f"- layer 9 conv state boundary: `{probe.get('layer9_conv_state_boundary')}`",
      "",
      "| output | comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name in (
      "l4_residual_input",
      "l4_attn_norm",
      "l4_linear_attn_qkv_mixed",
      "l4_conv_output_raw",
  ):
    group = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lane = group.get("gpu_vs_oracle", {}) if isinstance(group, dict) else {}
    lines.append(f"| {name} | gpu_vs_oracle | {lane.get('max_abs_diff')} | {lane.get('rmse')} |")
  lines += [
      "",
      "| kernel group | min us |",
      "|---|---:|",
      f"| layer8_state_input | {timings.get('resident_layer8_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer9_state_input | {timings.get('resident_layer9_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      f"| layer5_6_7_8_to_layer9_state_input | {timings.get('resident_layer5_6_7_8_to_layer9_state_input_kernel_sum_min_us') if isinstance(timings, dict) else None} |",
      "",
      "This target-side process carries layer 5 and layer 6 GPU outputs into",
      "layer 7, computes live GPU layer-7 l_out, then carries layer 8 live GPU",
      "l_out into layer 9's linear-attention shell with captured layer-9 conv",
      "state. This is captured single-token state/input evidence, not decode",
      "throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.resident_invocations <= 0:
    raise SystemExit("--resident-invocations must be positive")
  if args.layer != 5:
    raise SystemExit("--layer must be 5 for the layer-9 state/input handoff")

  layer0 = args.layer
  layer1 = args.layer + 1
  layer2 = args.layer + 2
  layer3 = args.layer + 3
  layer4 = args.layer + 4
  conv0_path = (
      args.conv_history_probe.resolve()
      if args.conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer0).resolve()
  )
  conv1_path = (
      args.next_conv_history_probe.resolve()
      if args.next_conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer1).resolve()
  )
  conv2_path = (
      args.layer8_conv_history_probe.resolve()
      if args.layer8_conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer3).resolve()
  )
  conv3_path = (
      args.layer9_conv_history_probe.resolve()
      if args.layer9_conv_history_probe is not None
      else TWO.latest_conv_history_probe_for_layer(layer4).resolve()
  )
  all_history_json = args.all_history_json.resolve()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-resident-layer9-state-input-handoff-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()

  payloads0, conv0 = TWO.prefixed_payloads(layer0, conv0_path, "l0")
  payloads1, conv1 = TWO.prefixed_payloads(layer1, conv1_path, "l1")
  payloads2, conv2 = TWO.prefixed_payloads(layer3, conv2_path, "l3")
  payloads3, conv3 = TWO.prefixed_payloads(layer4, conv3_path, "l4")
  full_payloads, all_history = L7_INPUT.resolve_full_attention_payloads(all_history_json, layer2)
  CORE.add_layer7_tail_payloads(full_payloads)
  L8.add_layer7_ffn_payloads(full_payloads)
  payloads = {**payloads0, **payloads1, **full_payloads, **payloads2, **payloads3}
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  embedded_opencl_hash = hashlib.sha256(
      (
          opencl_source
          + PRECONV.POSTCONV.ATTENTION.ATTENTION_EXTRA_OPENCL
          + CORE.FULL_ATTN_CORE_EXTRA_OPENCL
      ).encode("utf-8")
  ).hexdigest()
  local_cpp = out_dir / "gpu_resident_layer9_state_input_handoff_probe.cpp"
  local_cpp.write_text(layer9_state_input_probe_cpp(opencl_source), encoding="utf-8")

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-resident-layer9-state-input-handoff-probe-{stamp}"
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
            f"{remote_dir}/tests/gpu_resident_layer9_state_input_handoff_probe.cpp",
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

  executable = f"{remote_dir}/build/iq36-gpu-resident-layer9-state-input-handoff-probe"
  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_resident_layer9_state_input_handoff_probe.cpp')} "
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
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")))},
      {"name": "resident_api_fields_present", "pass": resident_fields_ok(probe, args.resident_invocations)},
      {"name": "layer2_ffn_tensor_shapes_ok", "pass": PRECONV.nested_bool(probe, "checks", "layer2_ffn_tensor_shapes_ok")},
      {"name": "layer2_core_output_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_full_attn_core_output_matches_oracle")},
      {"name": "live_ffn_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer2_live_ffn_gpu_cpu_matches_native")},
      {"name": "live_ffn_oracle_magnitude_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer2_live_ffn_oracle_magnitude_policy_matches")},
      {"name": "live_layer_output_full_policy_matches_oracle", "pass": PRECONV.nested_bool(probe, "checks", "layer2_live_layer_output_full_policy_matches_oracle")},
      {"name": "layer8_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer8_gpu_cpu_matches_native")},
      {"name": "layer8_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer8_state_input_oracle_policy_matches")},
      {"name": "layer8_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer8_state_input_handoff_matches")},
      {"name": "layer9_gpu_cpu_matches_native", "pass": PRECONV.nested_bool(probe, "checks", "layer9_gpu_cpu_matches_native")},
      {"name": "layer9_state_input_oracle_policy_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer9_state_input_oracle_policy_matches")},
      {"name": "layer9_state_input_handoff_matches", "pass": PRECONV.nested_bool(probe, "checks", "layer9_state_input_handoff_matches")},
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
      "live_ffn_oracle_magnitude": {
          "max_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["max_abs_diff"],
          "rmse": CORE.STRICT_COMPARISON_THRESHOLDS["rmse"],
          "mismatch_abs_diff": CORE.STRICT_COMPARISON_THRESHOLDS["mismatch_abs_diff"],
      },
      "layer8_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
      "layer9_state_input_oracle_policy": CORE.FULL_ATTN_COMPARISON_THRESHOLDS,
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
      },
      "conv_history_capture_artifacts": {
          "layer0": conv0.get("capture_artifact"),
          "layer1": conv1.get("capture_artifact"),
          "layer3": conv2.get("capture_artifact"),
          "layer4": conv3.get("capture_artifact"),
      },
      "all_history": all_history,
      "ffn_payload_root": str(FFN_PAYLOAD_ROOT.relative_to(ROOT)),
      "payloads": slim_payloads,
      "layers": [layer0, layer1, layer2, layer3, layer4],
      "resident_invocations": args.resident_invocations,
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "embedded_opencl_source_sha256": embedded_opencl_hash,
      "comparison_thresholds": comparison_thresholds,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-resident-layer9-state-input-handoff-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layers": [layer0, layer1, layer2, layer3, layer4],
      "resident_invocations": args.resident_invocations,
      "conv_history_probes": payload["conv_history_probes"],
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
      "gpu_resident_layer9_state_input_handoff_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("resident_shell_invocations", args.resident_invocations),
          ("resident_layer9_state_input_kernel_sum_min_us", PRECONV.nested_number(timings, "resident_layer9_state_input_kernel_sum_min_us")),
          ("layer9_residual_input_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l4_residual_input", "gpu_vs_oracle", "max_abs_diff")),
          ("layer9_attn_norm_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l4_attn_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("layer9_qkv_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l4_linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("layer9_conv_output_raw_gpu_vs_oracle_max_abs_diff", PRECONV.nested_number(comparisons, "l4_conv_output_raw", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
