#!/usr/bin/env python3
"""Run the GPU Q4 x8 linear-attention output projection handoff gate."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-output-projection-probe-v3"
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
    ("engine/src/gpu_q4_cpu_order_matvec.cpp", "src/gpu_q4_cpu_order_matvec.cpp"),
]
PAYLOAD_SPECS = {
    "final_output": ("final_output.bin", "final_output-{layer}__tok15__ord206.bin", 16384),
    "linear_attn_out": ("linear_attn_out.bin", "linear_attn_out-{layer}__tok15__ord207.bin", 8192),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kLayerCount = 40;
constexpr int kSourceTokenPosition = 15;
constexpr double kMismatchThreshold = 5e-3;
constexpr double kMaxAbsDiffThreshold = 5e-3;
constexpr double kRmseThreshold = 1e-3;
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

std::vector<std::uint8_t> ReadTensorBytes(std::ifstream& in,
                                          const iq36::GgufTensorInfo& tensor) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.nbytes));
  in.clear();
  in.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(in), "failed to seek tensor payload");
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  Require(in.gcount() == static_cast<std::streamsize>(bytes.size()), "failed to read tensor payload");
  return bytes;
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
    const std::string tensor_name = LayerTensorName(args.layer, "ssm_out.weight");
    const auto* tensor = iq36::find_tensor(index, tensor_name);
    Require(tensor != nullptr, "linear attention output tensor missing");
    Require(tensor->type == 12, "linear attention output tensor is not Q4_K");

    const auto input = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "final_output.bin"));
    const auto oracle = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "linear_attn_out.bin"));
    Require(tensor->dims.size() >= 2, "output projection tensor rank mismatch");
    const std::uint64_t cols = tensor->dims[0];
    std::uint64_t rows = 1;
    for (std::size_t i = 1; i < tensor->dims.size(); ++i) {
      rows *= tensor->dims[i];
    }
    const bool tensor_shape_ok =
        cols == input.size() && rows == oracle.size() && rows % 8 == 0 && cols % 256 == 0;
    const auto cpu = iq36::matvec_tensor(args.model_path, index, tensor_name, input);

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto raw = ReadTensorBytes(model, *tensor);
    const std::uint64_t blocks_per_row = cols / 256;
    const auto packed = iq36::PackQ4Kx8(raw, rows, blocks_per_row);
    const auto q8 = iq36::QuantizeQ8KInputPlanes(input);
    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    iq36::GpuQ4KCpuOrderMatvecRunner cpu_order_runner(args.device_substring);
    const auto cpu_order_handle =
        cpu_order_runner.UploadRawQ4KCpuOrder(raw, rows, blocks_per_row);
    const auto gpu_cpu_order =
        cpu_order_runner.RunResidentRawQ4KCpuOrder(
            cpu_order_handle, q8, args.repeat);
    const auto gpu_rowlane =
        runner.Run(packed, q8.qs, q8.bsums, q8.d, rows, blocks_per_row,
                   args.repeat, iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
    const auto gpu_group8 =
        runner.Run(packed, q8.qs, q8.bsums, q8.d, rows, blocks_per_row,
                   args.repeat, iq36::GpuQ4X8KernelVariant::kGroup8Serial);
    const auto gpu_rowblock16 =
        runner.RunRowblock16(packed, q8.qs, q8.bsums, q8.d, rows,
                             blocks_per_row, args.repeat);
    const auto gpu_rowblock16_cpuorder_finalize =
        runner.RunRowblock16CpuOrderFinalize(
            packed, q8.qs, q8.bsums, q8.d, rows, blocks_per_row, args.repeat);

    const auto cpu_vs_oracle = iq36::compare_vectors(cpu, oracle, kMismatchThreshold);
    const auto cpu_order_gpu_vs_cpu =
        iq36::compare_vectors(gpu_cpu_order.output, cpu, kMismatchThreshold);
    const auto cpu_order_gpu_vs_oracle =
        iq36::compare_vectors(gpu_cpu_order.output, oracle, kMismatchThreshold);
    const auto rowlane_gpu_vs_cpu = iq36::compare_vectors(gpu_rowlane.output, cpu, kMismatchThreshold);
    const auto rowlane_gpu_vs_oracle = iq36::compare_vectors(gpu_rowlane.output, oracle, kMismatchThreshold);
    const auto group8_gpu_vs_cpu = iq36::compare_vectors(gpu_group8.output, cpu, kMismatchThreshold);
    const auto group8_gpu_vs_oracle = iq36::compare_vectors(gpu_group8.output, oracle, kMismatchThreshold);
    const auto rowblock16_gpu_vs_cpu =
        iq36::compare_vectors(gpu_rowblock16.output, cpu, kMismatchThreshold);
    const auto rowblock16_gpu_vs_oracle =
        iq36::compare_vectors(gpu_rowblock16.output, oracle, kMismatchThreshold);
    const auto rowblock16_cpuorder_finalize_gpu_vs_cpu =
        iq36::compare_vectors(
            gpu_rowblock16_cpuorder_finalize.output, cpu, kMismatchThreshold);
    const auto rowblock16_cpuorder_finalize_gpu_vs_oracle =
        iq36::compare_vectors(
            gpu_rowblock16_cpuorder_finalize.output, oracle, kMismatchThreshold);
    const bool comparisons_passed =
        ComparePassed(cpu_vs_oracle) &&
        ComparePassed(rowlane_gpu_vs_cpu) &&
        ComparePassed(rowlane_gpu_vs_oracle);
    const bool cpu_order_comparisons_passed =
        ComparePassed(cpu_vs_oracle) &&
        ComparePassed(cpu_order_gpu_vs_cpu) &&
        ComparePassed(cpu_order_gpu_vs_oracle);
    const bool group8_comparisons_passed =
        ComparePassed(cpu_vs_oracle) &&
        ComparePassed(group8_gpu_vs_cpu) &&
        ComparePassed(group8_gpu_vs_oracle);
    const bool rowblock16_comparisons_passed =
        ComparePassed(cpu_vs_oracle) &&
        ComparePassed(rowblock16_gpu_vs_cpu) &&
        ComparePassed(rowblock16_gpu_vs_oracle);
    const bool rowblock16_cpuorder_finalize_bit_exact =
        rowblock16_cpuorder_finalize_gpu_vs_cpu.same_size &&
        rowblock16_cpuorder_finalize_gpu_vs_cpu.finite &&
        rowblock16_cpuorder_finalize_gpu_vs_cpu.mismatch_count == 0 &&
        rowblock16_cpuorder_finalize_gpu_vs_cpu.max_abs_diff == 0.0 &&
        rowblock16_cpuorder_finalize_gpu_vs_cpu.rmse == 0.0;
    const bool timings_positive = gpu_cpu_order.timing.min_us > 0.0 &&
                                  gpu_rowlane.timing.min_us > 0.0 &&
                                  gpu_group8.timing.min_us > 0.0 &&
                                  gpu_rowblock16.timing.min_us > 0.0 &&
                                  gpu_rowblock16_cpuorder_finalize.timing.min_us > 0.0;
    const bool checks_passed =
        load_map.ready &&
        tensor_shape_ok &&
        gpu_cpu_order.output.size() == oracle.size() &&
        gpu_rowlane.output.size() == oracle.size() &&
        gpu_group8.output.size() == oracle.size() &&
        gpu_rowblock16.output.size() == oracle.size() &&
        gpu_rowblock16_cpuorder_finalize.output.size() == oracle.size() &&
        runner.device_name().find(args.device_substring) != std::string::npos &&
        cpu_order_runner.device_name().find(args.device_substring) !=
            std::string::npos &&
        cpu_order_comparisons_passed &&
        comparisons_passed &&
        group8_comparisons_passed &&
        rowblock16_comparisons_passed &&
        rowblock16_cpuorder_finalize_bit_exact &&
        timings_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-output-projection-probe-v3\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"tensor_name\":\"" << JsonEscape(tensor->name) << "\",";
    std::cout << "\"tensor_type\":\"" << iq36::ggml_type_name(tensor->type) << "\",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"cols\":" << cols << ",";
    std::cout << "\"rows\":" << rows << ",";
    std::cout << "\"blocks_per_row\":" << blocks_per_row << ",";
    std::cout << "\"raw_bytes\":" << raw.size() << ",";
    std::cout << "\"packed_q4k_x8_bytes\":" << packed.size() << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runner.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"cpu_order_output_projection_gpu_kernel_min_us\":" << gpu_cpu_order.timing.min_us << ",";
    std::cout << "\"cpu_order_output_projection_gpu_kernel_mean_us\":" << gpu_cpu_order.timing.mean_us << ",";
    std::cout << "\"cpu_order_output_projection_gpu_effective_raw_gb_s\":" << gpu_cpu_order.timing.effective_raw_gb_s << ",";
    std::cout << "\"output_projection_gpu_kernel_min_us\":" << gpu_rowlane.timing.min_us << ",";
    std::cout << "\"output_projection_gpu_kernel_mean_us\":" << gpu_rowlane.timing.mean_us << ",";
    std::cout << "\"output_projection_gpu_effective_packed_gb_s\":" << gpu_rowlane.timing.effective_packed_gb_s << ",";
    std::cout << "\"global_work_items\":" << gpu_rowlane.timing.global_work_items << ",";
    std::cout << "\"rows_per_work_item\":" << gpu_rowlane.timing.rows_per_work_item << ",";
    std::cout << "\"group8_output_projection_gpu_kernel_min_us\":" << gpu_group8.timing.min_us << ",";
    std::cout << "\"group8_output_projection_gpu_kernel_mean_us\":" << gpu_group8.timing.mean_us << ",";
    std::cout << "\"group8_output_projection_gpu_effective_packed_gb_s\":" << gpu_group8.timing.effective_packed_gb_s << ",";
    std::cout << "\"group8_global_work_items\":" << gpu_group8.timing.global_work_items << ",";
    std::cout << "\"group8_rows_per_work_item\":" << gpu_group8.timing.rows_per_work_item << ",";
    std::cout << "\"rowblock16_output_projection_gpu_kernel_min_us\":" << gpu_rowblock16.timing.min_us << ",";
    std::cout << "\"rowblock16_output_projection_gpu_kernel_mean_us\":" << gpu_rowblock16.timing.mean_us << ",";
    std::cout << "\"rowblock16_output_projection_gpu_effective_packed_gb_s\":" << gpu_rowblock16.timing.effective_packed_gb_s << ",";
    std::cout << "\"rowblock16_global_work_items\":" << gpu_rowblock16.timing.global_work_items << ",";
    std::cout << "\"rowblock16_rows_per_work_item\":" << gpu_rowblock16.timing.rows_per_work_item << ",";
    std::cout << "\"rowblock16_cpuorder_finalize_gpu_kernel_min_us\":" << gpu_rowblock16_cpuorder_finalize.timing.min_us << ",";
    std::cout << "\"rowblock16_cpuorder_finalize_gpu_kernel_mean_us\":" << gpu_rowblock16_cpuorder_finalize.timing.mean_us << ",";
    std::cout << "\"rowblock16_cpuorder_finalize_gpu_effective_packed_gb_s\":" << gpu_rowblock16_cpuorder_finalize.timing.effective_packed_gb_s << ",";
    std::cout << "\"rowblock16_cpuorder_finalize_global_work_items\":" << gpu_rowblock16_cpuorder_finalize.timing.global_work_items;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"linear_attn_out_cpu_order\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(cpu_order_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(cpu_order_gpu_vs_oracle);
    std::cout << "},";
    std::cout << "\"linear_attn_out\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(rowlane_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(rowlane_gpu_vs_oracle);
    std::cout << "},\"linear_attn_out_group8\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(group8_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(group8_gpu_vs_oracle);
    std::cout << "},\"linear_attn_out_rowblock16\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(rowblock16_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(rowblock16_gpu_vs_oracle);
    std::cout << "},\"linear_attn_out_rowblock16_cpuorder_finalize\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(rowblock16_cpuorder_finalize_gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(rowblock16_cpuorder_finalize_gpu_vs_oracle);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"cpu_order_output_projection_matches_oracle\":" << (cpu_order_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"output_projection_matches_oracle\":" << (comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"group8_output_projection_matches_oracle\":" << (group8_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"rowblock16_output_projection_matches_oracle\":" << (rowblock16_comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"rowblock16_cpuorder_finalize_bit_exact_vs_cpu\":" << (rowblock16_cpuorder_finalize_bit_exact ? "true" : "false") << ",";
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
      wildcard = pattern.format(layer=layer)
      wildcard = wildcard.replace("__tok15__ord206.bin", "__tok15__ord*.bin")
      wildcard = wildcard.replace("__tok15__ord207.bin", "__tok15__ord*.bin")
      matches = sorted(PAYLOAD_ROOT.glob(wildcard))
      if len(matches) != 1:
        raise SystemExit(
            f"output projection payload missing or ambiguous: {path} ({len(matches)} wildcard matches)"
        )
      path = matches[0].resolve()
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"output projection payload size mismatch: {path}")
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
  comparison = (
      probe.get("comparisons", {}).get("linear_attn_out", {})
      if isinstance(probe, dict)
      else {}
  )
  cpu_order_comparison = (
      probe.get("comparisons", {}).get("linear_attn_out_cpu_order", {})
      if isinstance(probe, dict)
      else {}
  )
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  group8_comparison = (
      probe.get("comparisons", {}).get("linear_attn_out_group8", {})
      if isinstance(probe, dict)
      else {}
  )
  rowblock16_comparison = (
      probe.get("comparisons", {}).get("linear_attn_out_rowblock16", {})
      if isinstance(probe, dict)
      else {}
  )
  fused_exact_comparison = (
      probe.get("comparisons", {}).get(
          "linear_attn_out_rowblock16_cpuorder_finalize", {})
      if isinstance(probe, dict)
      else {}
  )
  lines = [
      "# GPU Q4-X8 Output Projection Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- tensor: `{probe.get('tensor_name')}` cols `{probe.get('cols')}` rows `{probe.get('rows')}`",
      "",
      "| comparison | max abs | RMSE |",
      "|---|---:|---:|",
  ]
  for lane in ("cpu_vs_oracle", "gpu_vs_cpu", "gpu_vs_oracle"):
    cmp = comparison.get(lane, {}) if isinstance(comparison, dict) else {}
    lines.append(f"| {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  for lane in ("gpu_vs_cpu", "gpu_vs_oracle"):
    cmp = (
        cpu_order_comparison.get(lane, {})
        if isinstance(cpu_order_comparison, dict) else {}
    )
    lines.append(
        f"| cpu_order_{lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  for lane in ("gpu_vs_cpu", "gpu_vs_oracle"):
    cmp = group8_comparison.get(lane, {}) if isinstance(group8_comparison, dict) else {}
    lines.append(f"| group8_{lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  for lane in ("gpu_vs_cpu", "gpu_vs_oracle"):
    cmp = (
        rowblock16_comparison.get(lane, {})
        if isinstance(rowblock16_comparison, dict) else {}
    )
    lines.append(
        f"| rowblock16_{lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  for lane in ("gpu_vs_cpu", "gpu_vs_oracle"):
    cmp = (
        fused_exact_comparison.get(lane, {})
        if isinstance(fused_exact_comparison, dict) else {}
    )
    lines.append(
        f"| rowblock16_cpuorder_finalize_{lane} | "
        f"{cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| kernel | min us | mean us | packed GB/s |",
      "|---|---:|---:|---:|",
      "| cpu_order_output_projection | "
      f"{timings.get('cpu_order_output_projection_gpu_kernel_min_us')} | "
      f"{timings.get('cpu_order_output_projection_gpu_kernel_mean_us')} | "
      f"{timings.get('cpu_order_output_projection_gpu_effective_raw_gb_s')} |",
      "| output_projection | "
      f"{timings.get('output_projection_gpu_kernel_min_us')} | "
      f"{timings.get('output_projection_gpu_kernel_mean_us')} | "
      f"{timings.get('output_projection_gpu_effective_packed_gb_s')} |",
      "| group8_output_projection | "
      f"{timings.get('group8_output_projection_gpu_kernel_min_us')} | "
      f"{timings.get('group8_output_projection_gpu_kernel_mean_us')} | "
      f"{timings.get('group8_output_projection_gpu_effective_packed_gb_s')} |",
      "| rowblock16_output_projection | "
      f"{timings.get('rowblock16_output_projection_gpu_kernel_min_us')} | "
      f"{timings.get('rowblock16_output_projection_gpu_kernel_mean_us')} | "
      f"{timings.get('rowblock16_output_projection_gpu_effective_packed_gb_s')} |",
      "| rowblock16_cpuorder_finalize | "
      f"{timings.get('rowblock16_cpuorder_finalize_gpu_kernel_min_us')} | "
      f"{timings.get('rowblock16_cpuorder_finalize_gpu_kernel_mean_us')} | "
      f"{timings.get('rowblock16_cpuorder_finalize_gpu_effective_packed_gb_s')} |",
      "",
      "The probe starts from captured `final_output` and closes the Q4 x8",
      "`ssm_out.weight` output projection against CPU native and teacher capture.",
      "This is component evidence only; it does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-output-projection-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_q4x8_output_projection_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-output-projection-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_output_projection_probe.cpp", args.timeout_s))
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
          f"{shlex.quote(remote_dir + '/src/gpu_q4_cpu_order_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_output_projection_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-output-projection-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-q4x8-output-projection-probe",
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
      {"name": "cpu_order_output_projection_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "cpu_order_output_projection_matches_oracle"))},
      {"name": "output_projection_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "output_projection_matches_oracle"))},
      {"name": "group8_output_projection_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "group8_output_projection_matches_oracle"))},
      {"name": "rowblock16_output_projection_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "rowblock16_output_projection_matches_oracle"))},
      {"name": "rowblock16_cpuorder_finalize_bit_exact_vs_cpu", "pass": bool(probe and nested_bool(probe, "checks", "rowblock16_cpuorder_finalize_bit_exact_vs_cpu"))},
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
      "tool": "tools/intel-qwen36-gpu-q4x8-output-projection-probe.py",
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
      "gpu_q4x8_output_projection_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("cpu_order_output_projection_kernel_min_us", nested_number(timings, "cpu_order_output_projection_gpu_kernel_min_us")),
          ("cpu_order_output_projection_effective_raw_gb_s", nested_number(timings, "cpu_order_output_projection_gpu_effective_raw_gb_s")),
          ("output_projection_kernel_min_us", nested_number(timings, "output_projection_gpu_kernel_min_us")),
          ("group8_output_projection_kernel_min_us", nested_number(timings, "group8_output_projection_gpu_kernel_min_us")),
          ("rowblock16_output_projection_kernel_min_us", nested_number(timings, "rowblock16_output_projection_gpu_kernel_min_us")),
          ("rowblock16_cpuorder_finalize_kernel_min_us", nested_number(timings, "rowblock16_cpuorder_finalize_gpu_kernel_min_us")),
          ("output_projection_effective_packed_gb_s", nested_number(timings, "output_projection_gpu_effective_packed_gb_s")),
          ("group8_output_projection_effective_packed_gb_s", nested_number(timings, "group8_output_projection_gpu_effective_packed_gb_s")),
          ("rowblock16_output_projection_effective_packed_gb_s", nested_number(timings, "rowblock16_output_projection_gpu_effective_packed_gb_s")),
          ("linear_attn_out_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_out", "gpu_vs_oracle", "max_abs_diff")),
          ("linear_attn_out_gpu_vs_oracle_rmse", nested_number(comparisons, "linear_attn_out", "gpu_vs_oracle", "rmse")),
          ("linear_attn_out_cpu_order_gpu_vs_cpu_max_abs_diff", nested_number(comparisons, "linear_attn_out_cpu_order", "gpu_vs_cpu", "max_abs_diff")),
          ("linear_attn_out_cpu_order_gpu_vs_cpu_rmse", nested_number(comparisons, "linear_attn_out_cpu_order", "gpu_vs_cpu", "rmse")),
          ("linear_attn_out_cpu_order_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_out_cpu_order", "gpu_vs_oracle", "max_abs_diff")),
          ("linear_attn_out_group8_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_out_group8", "gpu_vs_oracle", "max_abs_diff")),
          ("linear_attn_out_group8_gpu_vs_oracle_rmse", nested_number(comparisons, "linear_attn_out_group8", "gpu_vs_oracle", "rmse")),
          ("linear_attn_out_rowblock16_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_out_rowblock16", "gpu_vs_oracle", "max_abs_diff")),
          ("linear_attn_out_rowblock16_gpu_vs_oracle_rmse", nested_number(comparisons, "linear_attn_out_rowblock16", "gpu_vs_oracle", "rmse")),
          ("linear_attn_out_rowblock16_cpuorder_finalize_gpu_vs_cpu_max_abs_diff", nested_number(comparisons, "linear_attn_out_rowblock16_cpuorder_finalize", "gpu_vs_cpu", "max_abs_diff")),
          ("linear_attn_out_rowblock16_cpuorder_finalize_gpu_vs_cpu_rmse", nested_number(comparisons, "linear_attn_out_rowblock16_cpuorder_finalize", "gpu_vs_cpu", "rmse")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
