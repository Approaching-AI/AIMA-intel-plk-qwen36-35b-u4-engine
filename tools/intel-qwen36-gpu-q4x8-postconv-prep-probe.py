#!/usr/bin/env python3
"""Run the GPU Q4 x8 linear-attention postconv prep handoff gate."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-postconv-prep-probe-v0"
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
    "conv_output_raw": ("conv_output_raw.bin", "conv_output_raw-{layer}__tok15__ord191.bin", 32768),
    "conv_output_silu": ("conv_output_silu.bin", "conv_output_silu-{layer}__tok15__ord192.bin", 32768),
    "q_conv": ("q_conv.bin", "q_conv-{layer}__tok15__ord193.bin", 8192),
    "q_conv_predelta": ("q_conv_predelta.bin", "q_conv_predelta-{layer}__tok15__ord194.bin", 8192),
    "k_conv": ("k_conv.bin", "k_conv-{layer}__tok15__ord195.bin", 8192),
    "k_conv_predelta": ("k_conv_predelta.bin", "k_conv_predelta-{layer}__tok15__ord196.bin", 8192),
    "v_conv_predelta": ("v_conv_predelta.bin", "v_conv_predelta-{layer}__tok15__ord197.bin", 16384),
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
constexpr int kQkvMixedSize = 8192;
constexpr int kLinearHeadDim = 128;
constexpr int kLinearQueryHeads = 16;
constexpr int kLinearValueHeads = 32;
constexpr int kSourceTokenPosition = 15;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 7;
  std::string device_substring = "B390";
};

struct PostConvPrep {
  std::vector<float> conv_output_silu;
  std::vector<float> q_conv;
  std::vector<float> k_conv;
  std::vector<float> v_conv_predelta;
  std::vector<float> q_conv_predelta;
  std::vector<float> k_conv_predelta;
};

struct CompareRow {
  std::string name;
  iq36::VectorCompareStats cpu_vs_oracle;
  iq36::VectorCompareStats gpu_vs_cpu;
  iq36::VectorCompareStats gpu_vs_oracle;
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
         stats.max_abs_diff <= 5e-4 &&
         stats.rmse <= 5e-5 &&
         stats.cosine >= 0.99999;
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

std::vector<float> L2NormHeads(const std::vector<float>& input,
                               std::size_t head_dim,
                               float norm_epsilon) {
  Require(head_dim > 0, "head dim must be nonzero");
  Require(input.size() % head_dim == 0, "L2 input size mismatch");
  std::vector<float> output = input;
  const std::size_t heads = input.size() / head_dim;
  for (std::size_t head = 0; head < heads; ++head) {
    const std::size_t base = head * head_dim;
    double sum = 0.0;
    for (std::size_t i = 0; i < head_dim; ++i) {
      const float value = input[base + i];
      sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const float scale =
        1.0f / std::max(std::sqrt(static_cast<float>(sum)), norm_epsilon);
    for (std::size_t i = 0; i < head_dim; ++i) {
      output[base + i] = input[base + i] * scale;
    }
  }
  return output;
}

PostConvPrep RunCpuPostConvPrep(const std::vector<float>& conv_output_raw,
                                float norm_epsilon) {
  constexpr std::size_t kQValues =
      static_cast<std::size_t>(kLinearHeadDim * kLinearQueryHeads);
  constexpr std::size_t kVValues =
      static_cast<std::size_t>(kLinearHeadDim * kLinearValueHeads);
  Require(conv_output_raw.size() == kQValues + kQValues + kVValues,
          "postconv raw size mismatch");
  PostConvPrep prep;
  prep.conv_output_silu.reserve(conv_output_raw.size());
  for (const float value : conv_output_raw) {
    prep.conv_output_silu.push_back(value * iq36::sigmoid_scalar(value));
  }
  prep.q_conv.assign(prep.conv_output_silu.begin(), prep.conv_output_silu.begin() + kQValues);
  prep.k_conv.assign(
      prep.conv_output_silu.begin() + kQValues,
      prep.conv_output_silu.begin() + static_cast<std::ptrdiff_t>(2 * kQValues));
  prep.v_conv_predelta.assign(
      prep.conv_output_silu.begin() + static_cast<std::ptrdiff_t>(2 * kQValues),
      prep.conv_output_silu.end());
  prep.q_conv_predelta = L2NormHeads(prep.q_conv, kLinearHeadDim, norm_epsilon);
  prep.k_conv_predelta = L2NormHeads(prep.k_conv, kLinearHeadDim, norm_epsilon);
  return prep;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    Require(iq36::find_tensor(index, LayerTensorName(args.layer, "ssm_out.weight")) != nullptr,
            "target layer is not linear attention");
    const float norm_epsilon = MetadataFloat(
        index, "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);

    const auto oracle_raw = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_output_raw.bin"));
    const auto oracle_silu = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "conv_output_silu.bin"));
    const auto oracle_q = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "q_conv.bin"));
    const auto oracle_q_norm = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "q_conv_predelta.bin"));
    const auto oracle_k = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "k_conv.bin"));
    const auto oracle_k_norm = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "k_conv_predelta.bin"));
    const auto oracle_v = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "v_conv_predelta.bin"));
    Require(oracle_raw.size() == kQkvMixedSize, "oracle raw size mismatch");

    const auto cpu = RunCpuPostConvPrep(oracle_raw, norm_epsilon);
    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    const auto gpu = runner.RunPostConvPrep(
        oracle_raw,
        kLinearHeadDim,
        kLinearQueryHeads,
        kLinearValueHeads,
        norm_epsilon,
        args.repeat);
    const auto fused = runner.RunPostConvPrepFused(
        oracle_raw,
        kLinearHeadDim,
        kLinearQueryHeads,
        kLinearValueHeads,
        norm_epsilon,
        args.repeat);

    std::vector<CompareRow> rows = {
        {"conv_output_silu",
         iq36::compare_vectors(cpu.conv_output_silu, oracle_silu, 5e-4),
         iq36::compare_vectors(gpu.conv_output_silu, cpu.conv_output_silu, 5e-4),
         iq36::compare_vectors(gpu.conv_output_silu, oracle_silu, 5e-4)},
        {"q_conv",
         iq36::compare_vectors(cpu.q_conv, oracle_q, 5e-4),
         iq36::compare_vectors(gpu.q_conv, cpu.q_conv, 5e-4),
         iq36::compare_vectors(gpu.q_conv, oracle_q, 5e-4)},
        {"q_conv_predelta",
         iq36::compare_vectors(cpu.q_conv_predelta, oracle_q_norm, 5e-4),
         iq36::compare_vectors(gpu.q_conv_predelta, cpu.q_conv_predelta, 5e-4),
         iq36::compare_vectors(gpu.q_conv_predelta, oracle_q_norm, 5e-4)},
        {"k_conv",
         iq36::compare_vectors(cpu.k_conv, oracle_k, 5e-4),
         iq36::compare_vectors(gpu.k_conv, cpu.k_conv, 5e-4),
         iq36::compare_vectors(gpu.k_conv, oracle_k, 5e-4)},
        {"k_conv_predelta",
         iq36::compare_vectors(cpu.k_conv_predelta, oracle_k_norm, 5e-4),
         iq36::compare_vectors(gpu.k_conv_predelta, cpu.k_conv_predelta, 5e-4),
         iq36::compare_vectors(gpu.k_conv_predelta, oracle_k_norm, 5e-4)},
        {"v_conv_predelta",
         iq36::compare_vectors(cpu.v_conv_predelta, oracle_v, 5e-4),
        iq36::compare_vectors(gpu.v_conv_predelta, cpu.v_conv_predelta, 5e-4),
         iq36::compare_vectors(gpu.v_conv_predelta, oracle_v, 5e-4)},
    };
    std::vector<CompareRow> fused_rows = {
        {"conv_output_silu",
         iq36::compare_vectors(cpu.conv_output_silu, oracle_silu, 5e-4),
         iq36::compare_vectors(fused.conv_output_silu, cpu.conv_output_silu, 5e-4),
         iq36::compare_vectors(fused.conv_output_silu, oracle_silu, 5e-4)},
        {"q_conv",
         iq36::compare_vectors(cpu.q_conv, oracle_q, 5e-4),
         iq36::compare_vectors(fused.q_conv, cpu.q_conv, 5e-4),
         iq36::compare_vectors(fused.q_conv, oracle_q, 5e-4)},
        {"q_conv_predelta",
         iq36::compare_vectors(cpu.q_conv_predelta, oracle_q_norm, 5e-4),
         iq36::compare_vectors(fused.q_conv_predelta, cpu.q_conv_predelta, 5e-4),
         iq36::compare_vectors(fused.q_conv_predelta, oracle_q_norm, 5e-4)},
        {"k_conv",
         iq36::compare_vectors(cpu.k_conv, oracle_k, 5e-4),
         iq36::compare_vectors(fused.k_conv, cpu.k_conv, 5e-4),
         iq36::compare_vectors(fused.k_conv, oracle_k, 5e-4)},
        {"k_conv_predelta",
         iq36::compare_vectors(cpu.k_conv_predelta, oracle_k_norm, 5e-4),
         iq36::compare_vectors(fused.k_conv_predelta, cpu.k_conv_predelta, 5e-4),
         iq36::compare_vectors(fused.k_conv_predelta, oracle_k_norm, 5e-4)},
        {"v_conv_predelta",
         iq36::compare_vectors(cpu.v_conv_predelta, oracle_v, 5e-4),
         iq36::compare_vectors(fused.v_conv_predelta, cpu.v_conv_predelta, 5e-4),
         iq36::compare_vectors(fused.v_conv_predelta, oracle_v, 5e-4)},
    };

    bool comparisons_passed = true;
    for (const auto& row : rows) {
      comparisons_passed = comparisons_passed &&
          ComparePassed(row.cpu_vs_oracle) &&
          ComparePassed(row.gpu_vs_cpu) &&
          ComparePassed(row.gpu_vs_oracle);
    }
    bool fused_comparisons_passed = true;
    for (const auto& row : fused_rows) {
      fused_comparisons_passed = fused_comparisons_passed &&
          ComparePassed(row.cpu_vs_oracle) &&
          ComparePassed(row.gpu_vs_cpu) &&
          ComparePassed(row.gpu_vs_oracle);
    }
    const bool timings_positive =
        gpu.timing.silu_split_min_us > 0.0 &&
        (gpu.timing.q_l2_min_us > 0.0 || gpu.timing.k_l2_min_us > 0.0) &&
        fused.timing.fused_min_us > 0.0;
    const bool checks_passed =
        load_map.ready &&
        runner.device_name().find(args.device_substring) != std::string::npos &&
        comparisons_passed &&
        fused_comparisons_passed &&
        timings_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-postconv-prep-probe-v0\",";
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
    std::cout << "\"qkv_mixed_values\":" << kQkvMixedSize << ",";
    std::cout << "\"head_dim\":" << kLinearHeadDim << ",";
    std::cout << "\"query_heads\":" << kLinearQueryHeads << ",";
    std::cout << "\"value_heads\":" << kLinearValueHeads;
    std::cout << "},\"timings\":{";
    std::cout << "\"silu_split_gpu_kernel_min_us\":" << gpu.timing.silu_split_min_us << ",";
    std::cout << "\"silu_split_gpu_kernel_mean_us\":" << gpu.timing.silu_split_mean_us << ",";
    std::cout << "\"silu_split_global_work_items\":" << gpu.timing.silu_split_global_work_items << ",";
    std::cout << "\"q_l2_gpu_kernel_min_us\":" << gpu.timing.q_l2_min_us << ",";
    std::cout << "\"q_l2_gpu_kernel_mean_us\":" << gpu.timing.q_l2_mean_us << ",";
    std::cout << "\"q_l2_global_work_items\":" << gpu.timing.q_l2_global_work_items << ",";
    std::cout << "\"k_l2_gpu_kernel_min_us\":" << gpu.timing.k_l2_min_us << ",";
    std::cout << "\"k_l2_gpu_kernel_mean_us\":" << gpu.timing.k_l2_mean_us << ",";
    std::cout << "\"k_l2_global_work_items\":" << gpu.timing.k_l2_global_work_items;
    std::cout << "},\"fused_timings\":{";
    std::cout << "\"fused_gpu_kernel_min_us\":" << fused.timing.fused_min_us << ",";
    std::cout << "\"fused_gpu_kernel_mean_us\":" << fused.timing.fused_mean_us << ",";
    std::cout << "\"fused_global_work_items\":" << fused.timing.fused_global_work_items;
    std::cout << "},\"comparisons\":{";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (i != 0) std::cout << ",";
      const auto& row = rows[i];
      std::cout << "\"" << JsonEscape(row.name) << "\":{";
      std::cout << "\"cpu_vs_oracle\":";
      WriteCompare(row.cpu_vs_oracle);
      std::cout << ",\"gpu_vs_cpu\":";
      WriteCompare(row.gpu_vs_cpu);
      std::cout << ",\"gpu_vs_oracle\":";
      WriteCompare(row.gpu_vs_oracle);
      std::cout << "}";
    }
    std::cout << "},\"fused_comparisons\":{";
    for (std::size_t i = 0; i < fused_rows.size(); ++i) {
      if (i != 0) std::cout << ",";
      const auto& row = fused_rows[i];
      std::cout << "\"" << JsonEscape(row.name) << "\":{";
      std::cout << "\"cpu_vs_oracle\":";
      WriteCompare(row.cpu_vs_oracle);
      std::cout << ",\"gpu_vs_cpu\":";
      WriteCompare(row.gpu_vs_cpu);
      std::cout << ",\"gpu_vs_oracle\":";
      WriteCompare(row.gpu_vs_oracle);
      std::cout << "}";
    }
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"postconv_prep_matches_oracle\":" << (comparisons_passed ? "true" : "false") << ",";
    std::cout << "\"fused_postconv_prep_matches_oracle\":" << (fused_comparisons_passed ? "true" : "false") << ",";
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
      raise SystemExit(f"postconv prep payload missing: {path}")
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"postconv prep payload size mismatch: {path}")
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
  fused_comparisons = probe.get("fused_comparisons", {}) if isinstance(probe, dict) else {}
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  fused_timings = probe.get("fused_timings", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Q4-X8 Postconv Prep Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      "",
      "| output | GPU vs oracle max abs | GPU vs oracle RMSE |",
      "|---|---:|---:|",
  ]
  for name in (
      "conv_output_silu",
      "q_conv",
      "q_conv_predelta",
      "k_conv",
      "k_conv_predelta",
      "v_conv_predelta",
  ):
    cmp = comparisons.get(name, {}).get("gpu_vs_oracle", {}) if isinstance(comparisons, dict) else {}
    lines.append(f"| {name} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| kernel | min us | mean us |",
      "|---|---:|---:|",
      f"| silu_split | {timings.get('silu_split_gpu_kernel_min_us')} | {timings.get('silu_split_gpu_kernel_mean_us')} |",
      f"| q_l2 | {timings.get('q_l2_gpu_kernel_min_us')} | {timings.get('q_l2_gpu_kernel_mean_us')} |",
      f"| k_l2 | {timings.get('k_l2_gpu_kernel_min_us')} | {timings.get('k_l2_gpu_kernel_mean_us')} |",
      f"| fused_postconv | {fused_timings.get('fused_gpu_kernel_min_us')} | {fused_timings.get('fused_gpu_kernel_mean_us')} |",
      "",
      "| fused output | GPU vs oracle max abs | GPU vs oracle RMSE |",
      "|---|---:|---:|",
  ]
  for name in (
      "conv_output_silu",
      "q_conv",
      "q_conv_predelta",
      "k_conv",
      "k_conv_predelta",
      "v_conv_predelta",
  ):
    cmp = fused_comparisons.get(name, {}).get("gpu_vs_oracle", {}) if isinstance(fused_comparisons, dict) else {}
    lines.append(f"| {name} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "The probe starts from captured `conv_output_raw` and closes the GPU",
      "SiLU/split/Q-K L2-normalization handoff against teacher capture.",
      "It also tests the fused postconv-prep component when that API is present.",
      "Delta recurrent update and final projection remain the next gate.",
      "This is component evidence only; it does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-postconv-prep-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_q4x8_postconv_prep_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-postconv-prep-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_postconv_prep_probe.cpp", args.timeout_s))
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
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_postconv_prep_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-postconv-prep-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-q4x8-postconv-prep-probe",
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
      {"name": "postconv_prep_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "postconv_prep_matches_oracle"))},
      {"name": "fused_postconv_prep_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "fused_postconv_prep_matches_oracle"))},
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
      "tool": "tools/intel-qwen36-gpu-q4x8-postconv-prep-probe.py",
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
  fused_timings = aggregate.get("fused_timings", {}) if isinstance(aggregate.get("fused_timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  fused_comparisons = aggregate.get("fused_comparisons", {}) if isinstance(aggregate.get("fused_comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_q4x8_postconv_prep_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("silu_split_kernel_min_us", nested_number(timings, "silu_split_gpu_kernel_min_us")),
          ("q_l2_kernel_min_us", nested_number(timings, "q_l2_gpu_kernel_min_us")),
          ("k_l2_kernel_min_us", nested_number(timings, "k_l2_gpu_kernel_min_us")),
          ("fused_kernel_min_us", nested_number(fused_timings, "fused_gpu_kernel_min_us")),
          ("conv_output_silu_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "conv_output_silu", "gpu_vs_oracle", "max_abs_diff")),
          ("q_conv_predelta_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "q_conv_predelta", "gpu_vs_oracle", "max_abs_diff")),
          ("k_conv_predelta_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "k_conv_predelta", "gpu_vs_oracle", "max_abs_diff")),
          ("v_conv_predelta_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "v_conv_predelta", "gpu_vs_oracle", "max_abs_diff")),
          ("fused_q_conv_predelta_gpu_vs_oracle_max_abs_diff", nested_number(fused_comparisons, "q_conv_predelta", "gpu_vs_oracle", "max_abs_diff")),
          ("fused_k_conv_predelta_gpu_vs_oracle_max_abs_diff", nested_number(fused_comparisons, "k_conv_predelta", "gpu_vs_oracle", "max_abs_diff")),
          ("fused_v_conv_predelta_gpu_vs_oracle_max_abs_diff", nested_number(fused_comparisons, "v_conv_predelta", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
