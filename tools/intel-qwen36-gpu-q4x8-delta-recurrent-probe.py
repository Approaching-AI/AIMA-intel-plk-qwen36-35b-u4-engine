#!/usr/bin/env python3
"""Run the GPU linear-attention delta recurrent handoff gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-delta-recurrent-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/include/intel_qwen36/gpu_q4x8_matvec.hpp", "include/intel_qwen36/gpu_q4x8_matvec.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
]
PAYLOAD_SPECS = {
    "q_conv_predelta": ("q_conv_predelta.bin", "q_conv_predelta-{layer}__tok15__ord194.bin", 8192),
    "k_conv_predelta": ("k_conv_predelta.bin", "k_conv_predelta-{layer}__tok15__ord196.bin", 8192),
    "v_conv_predelta": ("v_conv_predelta.bin", "v_conv_predelta-{layer}__tok15__ord197.bin", 16384),
    "gate": ("gate.bin", "gate-{layer}__tok15__ord200.bin", 128),
    "beta_sigmoid": ("beta_sigmoid.bin", "beta_sigmoid-{layer}__tok15__ord202.bin", 128),
    "state_predelta": ("state_predelta.bin", "state_predelta-{layer}__tok15__ord203.bin", 2097152),
    "attn_output": ("attn_output.bin", "attn_output-{layer}__tok15__ord204.bin", 16384),
    "z": ("z.bin", "z-{layer}__tok15__ord205.bin", 16384),
    "final_output": ("final_output.bin", "final_output-{layer}__tok15__ord206.bin", 16384),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kLayerCount = 40;
constexpr int kHeadDim = 128;
constexpr int kQueryHeads = 16;
constexpr int kValueHeads = 32;
constexpr int kSourceTokenPosition = 15;
constexpr double kMismatchThreshold = 5e-4;
constexpr double kMaxAbsDiffThreshold = 5e-4;
constexpr double kRmseThreshold = 5e-5;
constexpr double kMinCosine = 0.99999;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 7;
  std::string device_substring = "B390";
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) {
    Die(message);
  }
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += ch; break;
    }
  }
  return out;
}

std::string JoinPath(const std::string& dir, const std::string& name) {
  return (!dir.empty() && dir.back() == '/') ? dir + name : dir + "/" + name;
}

std::string LayerTensorName(int layer, const std::string& suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

float MetadataFloat(const iq36::GgufModelIndex& index,
                    const std::string& key,
                    float fallback) {
  const auto found = index.metadata.find(key);
  if (found == index.metadata.end()) {
    return fallback;
  }
  const auto& value = found->second;
  if (value.kind == iq36::GgufMetadataValue::Kind::kFloat) {
    return static_cast<float>(value.float_value);
  }
  return fallback;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&](const char* name) -> std::string {
      Require(i + 1 < argc, std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--model") args.model_path = value("--model");
    else if (key == "--payload-dir") args.payload_dir = value("--payload-dir");
    else if (key == "--layer") args.layer = std::stoi(value("--layer"));
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer >= 0 && args.layer < kLayerCount, "--layer is out of range");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

bool ComparePassed(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= kMaxAbsDiffThreshold &&
         stats.rmse <= kRmseThreshold &&
         stats.cosine >= kMinCosine;
}

void WriteCompare(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    Require(iq36::find_tensor(index, LayerTensorName(args.layer, "ssm_out.weight")) != nullptr,
            "target layer is not linear attention");
    const auto* norm_tensor = iq36::find_tensor(index, LayerTensorName(args.layer, "ssm_norm.weight"));
    Require(norm_tensor != nullptr, "linear attention norm tensor missing");
    Require(norm_tensor->type == 0 &&
            norm_tensor->dims == std::vector<std::uint64_t>{kHeadDim},
            "linear attention norm tensor shape mismatch");
    const float norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);

    const auto q = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "q_conv_predelta.bin"));
    const auto k = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "k_conv_predelta.bin"));
    const auto v = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "v_conv_predelta.bin"));
    const auto gate = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "gate.bin"));
    const auto beta = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "beta_sigmoid.bin"));
    const auto state = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "state_predelta.bin"));
    const auto oracle_attn = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_output.bin"));
    const auto z = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "z.bin"));
    const auto oracle_final = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "final_output.bin"));
    const auto norm_weight =
        iq36::decode_tensor_row(args.model_path, index, LayerTensorName(args.layer, "ssm_norm.weight"), 0);

    const auto cpu = iq36::run_qwen36_linear_attention_delta_core(
        q, k, v, gate, beta, state, z, norm_weight, norm_epsilon);
    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    const auto gpu = runner.RunLinearAttentionDelta(
        q, k, v, gate, beta, state, z, norm_weight,
        kHeadDim, kQueryHeads, kValueHeads, norm_epsilon, args.repeat);

    const auto attention_cpu_vs_oracle =
        iq36::compare_vectors(cpu.attention_output, oracle_attn, kMismatchThreshold);
    const auto attention_gpu_vs_cpu =
        iq36::compare_vectors(gpu.attention_output, cpu.attention_output, kMismatchThreshold);
    const auto attention_gpu_vs_oracle =
        iq36::compare_vectors(gpu.attention_output, oracle_attn, kMismatchThreshold);
    const auto final_cpu_vs_oracle =
        iq36::compare_vectors(cpu.final_output, oracle_final, kMismatchThreshold);
    const auto final_gpu_vs_cpu =
        iq36::compare_vectors(gpu.final_output, cpu.final_output, kMismatchThreshold);
    const auto final_gpu_vs_oracle =
        iq36::compare_vectors(gpu.final_output, oracle_final, kMismatchThreshold);
    const auto state_gpu_vs_cpu =
        iq36::compare_vectors(gpu.recurrent_state, cpu.recurrent_state, kMismatchThreshold);

    const bool comparisons_passed =
        ComparePassed(attention_cpu_vs_oracle) &&
        ComparePassed(attention_gpu_vs_cpu) &&
        ComparePassed(attention_gpu_vs_oracle) &&
        ComparePassed(final_cpu_vs_oracle) &&
        ComparePassed(final_gpu_vs_cpu) &&
        ComparePassed(final_gpu_vs_oracle) &&
        ComparePassed(state_gpu_vs_cpu);
    const bool timings_positive =
        gpu.timing.delta_min_us > 0.0 && gpu.timing.final_min_us > 0.0;
    const bool checks_passed =
        load_map.ready &&
        runner.device_name().find(args.device_substring) != std::string::npos &&
        comparisons_passed &&
        timings_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-delta-recurrent-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runner.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"norm_epsilon\":" << norm_epsilon << ",";
    std::cout << "\"dimensions\":{";
    std::cout << "\"head_dim\":" << kHeadDim << ",";
    std::cout << "\"query_heads\":" << kQueryHeads << ",";
    std::cout << "\"value_heads\":" << kValueHeads << ",";
    std::cout << "\"recurrent_state_values\":" << state.size();
    std::cout << "},\"thresholds\":{";
    std::cout << "\"mismatch_abs_diff\":" << kMismatchThreshold << ",";
    std::cout << "\"max_abs_diff\":" << kMaxAbsDiffThreshold << ",";
    std::cout << "\"rmse\":" << kRmseThreshold << ",";
    std::cout << "\"min_cosine\":" << kMinCosine;
    std::cout << "},\"timings\":{";
    std::cout << "\"delta_gpu_kernel_min_us\":" << gpu.timing.delta_min_us << ",";
    std::cout << "\"delta_gpu_kernel_mean_us\":" << gpu.timing.delta_mean_us << ",";
    std::cout << "\"delta_global_work_items\":" << gpu.timing.delta_global_work_items << ",";
    std::cout << "\"final_gpu_kernel_min_us\":" << gpu.timing.final_min_us << ",";
    std::cout << "\"final_gpu_kernel_mean_us\":" << gpu.timing.final_mean_us << ",";
    std::cout << "\"final_global_work_items\":" << gpu.timing.final_global_work_items;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"attention_output\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(attention_cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(attention_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(attention_gpu_vs_oracle);
    std::cout << "},\"final_output\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(final_cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(final_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(final_gpu_vs_oracle);
    std::cout << "},\"recurrent_state\":{";
    std::cout << "\"gpu_vs_cpu\":";
    WriteCompare(state_gpu_vs_cpu);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"delta_recurrent_matches_oracle\":" << (comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (timings_positive ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},\"required_checks_passed\":" << (checks_passed ? "true" : "false");
    std::cout << "}\n";
    return checks_passed ? 0 : 3;
  } catch (const std::exception& exc) {
    std::cout << "{\"ok\":false,\"error\":\"" << JsonEscape(exc.what()) << "\"}\n";
    return 2;
  }
}
'''


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
  parser.add_argument("--layer", type=int, default=5)
  parser.add_argument("--repeat", type=int, default=7)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(item) for item in argv)


def cpp_raw_string_literal(value: str) -> str:
  delimiter = "IQ36CL"
  if f"){delimiter}\"" in value:
    raise ValueError(f"OpenCL source contains raw-string delimiter {delimiter}")
  return f'R"{delimiter}({value}){delimiter}"'


def parse_probe_stdout(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return None


def resolve_payloads(layer: int) -> dict[str, dict[str, Any]]:
  payloads: dict[str, dict[str, Any]] = {}
  for name, (stage_name, pattern, size_bytes) in PAYLOAD_SPECS.items():
    path = (PAYLOAD_ROOT / pattern.format(layer=layer)).resolve()
    if not path.exists():
      raise SystemExit(f"delta recurrent payload missing: {path}")
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"delta recurrent payload size mismatch: {path}")
    payloads[name] = {
        "local_path": path,
        "path": str(path.relative_to(ROOT)),
        "sha256": iq36_local.sha256_file(path),
        "size_bytes": size_bytes,
        "stage_name": stage_name,
    }
  return payloads


def nested_bool(obj: dict[str, Any], *keys: str) -> bool:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return False
    current = current.get(key)
  return current is True


def nested_number(obj: dict[str, Any], *keys: str) -> float | None:
  current: Any = obj
  for key in keys:
    if not isinstance(current, dict):
      return None
    current = current.get(key)
  return float(current) if isinstance(current, (int, float)) else None


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Q4-X8 Delta Recurrent Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      "",
      "| output | required comparison | max abs | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name, lane in (
      ("attention_output", "gpu_vs_oracle"),
      ("final_output", "gpu_vs_oracle"),
      ("recurrent_state", "gpu_vs_cpu"),
  ):
    cmp = comparisons.get(name, {}).get(lane, {}) if isinstance(comparisons, dict) else {}
    lines.append(f"| {name} | {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| kernel | min us | mean us |",
      "|---|---:|---:|",
      f"| delta_recurrent | {timings.get('delta_gpu_kernel_min_us')} | {timings.get('delta_gpu_kernel_mean_us')} |",
      f"| final_norm | {timings.get('final_gpu_kernel_min_us')} | {timings.get('final_gpu_kernel_mean_us')} |",
      "",
      "The probe starts from captured postconv prep outputs plus pre-delta",
      "recurrent state and closes the device recurrent update, attention output,",
      "and final gated RMS output. This is component evidence only; it does not",
      "prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-delta-recurrent-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_q4x8_delta_recurrent_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-delta-recurrent-probe-{stamp}"
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
    for local, remote in SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_delta_recurrent_probe.cpp", args.timeout_s))
    for name, payload in payloads.items():
      payload_transfers[name] = iq36_local.copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_delta_recurrent_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-delta-recurrent-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-q4x8-delta-recurrent-probe",
      "--model", args.model,
      "--payload-dir", remote_payload_dir,
      "--layer", str(args.layer),
      "--repeat", str(args.repeat),
      "--device-substring", args.device_substring,
  ]
  run_result = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
              shell_join(run_argv),
          ]),
          args.timeout_s,
      )
      if compile_result.get("returncode") == 0
      else {"cmd": run_argv, "returncode": None, "stdout": "", "stderr": "compile skipped run"}
  )
  probe = parse_probe_stdout(run_result.get("stdout", ""))
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
      {"name": "delta_recurrent_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "delta_recurrent_matches_oracle"))},
      {"name": "gpu_event_timing_positive", "pass": bool(probe and nested_bool(probe, "checks", "gpu_event_timing_positive"))},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  slim_payloads = {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "oracle_bundle": str(args.oracle_bundle.resolve().relative_to(ROOT)),
      "payloads": slim_payloads,
      "layer": args.layer,
      "repeat": args.repeat,
      "engine_shim_header": "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      "engine_shim_source": "engine/src/gpu_q4x8_matvec.cpp",
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-q4x8-delta-recurrent-probe.py",
      "artifact": str(out_dir),
      "host": args.host,
      "remote_dir": remote_dir,
      "layer": args.layer,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
  }
  correctness = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
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
      "gpu_q4x8_delta_recurrent_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("delta_kernel_min_us", nested_number(timings, "delta_gpu_kernel_min_us")),
          ("final_kernel_min_us", nested_number(timings, "final_gpu_kernel_min_us")),
          ("attention_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "attention_output", "gpu_vs_oracle", "max_abs_diff")),
          ("final_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "final_output", "gpu_vs_oracle", "max_abs_diff")),
          ("recurrent_state_gpu_vs_cpu_max_abs_diff", nested_number(comparisons, "recurrent_state", "gpu_vs_cpu", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
