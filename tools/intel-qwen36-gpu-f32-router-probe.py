#!/usr/bin/env python3
"""Run the GPU F32 FFN/MoE router handoff gate."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-f32-router-probe-v0"
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
    "attn_post_norm": ("attn_post_norm.bin", "attn_post_norm-{layer}__tok15__ord209.bin", 8192),
    "ffn_moe_logits": ("ffn_moe_logits.bin", "ffn_moe_logits-{layer}__tok15__ord210.bin", 1024),
    "ffn_moe_probs": ("ffn_moe_probs.bin", "ffn_moe_probs-{layer}__tok15__ord211.bin", 1024),
    "ffn_moe_topk": ("ffn_moe_topk.bin", "ffn_moe_topk-{layer}__tok15__ord212.bin", 32),
    "ffn_moe_weights": ("ffn_moe_weights.bin", "ffn_moe_weights-{layer}__tok15__ord213.bin", 32),
    "ffn_moe_weights_norm": ("ffn_moe_weights_norm.bin", "ffn_moe_weights_norm-{layer}__tok15__ord214.bin", 32),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kLayerCount = 40;
constexpr int kHiddenSize = 2048;
constexpr int kExpertCount = 256;
constexpr int kExpertUsedCount = 8;
constexpr int kSourceTokenPosition = 15;
constexpr float kMinWeightSum = 6.103515625e-5f;
constexpr double kLogitsMismatchThreshold = 1e-4;
constexpr double kLogitsMaxAbsDiffThreshold = 1e-4;
constexpr double kLogitsRmseThreshold = 1e-5;
constexpr double kWeightsMismatchThreshold = 2e-5;
constexpr double kWeightsMaxAbsDiffThreshold = 2e-5;
constexpr double kWeightsRmseThreshold = 1e-6;
constexpr double kMinCosine = 0.999999;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 7;
  std::string device_substring = "B390";
};

struct IntCompareStats {
  std::uint64_t lhs_value_count = 0;
  std::uint64_t rhs_value_count = 0;
  std::uint64_t compared_value_count = 0;
  std::uint64_t mismatch_count = 0;
  bool same_size = false;
};

struct RouterResult {
  std::vector<float> probabilities;
  std::vector<std::int32_t> topk;
  std::vector<float> weights;
  std::vector<float> weights_norm;
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

std::vector<float> ReadF32TensorPayload(std::ifstream& in,
                                        const iq36::GgufTensorInfo& tensor,
                                        std::size_t expected_values) {
  Require(tensor.type == 0, "tensor is not F32: " + tensor.name);
  Require(tensor.nbytes == expected_values * sizeof(float), "F32 tensor byte size mismatch");
  const auto bytes = ReadTensorBytes(in, tensor);
  std::vector<float> values(expected_values, 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<std::int32_t> ReadI32VectorFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "i32 vector file could not be opened");
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "i32 vector file size failed");
  Require(static_cast<std::uint64_t>(size) % sizeof(std::int32_t) == 0,
          "i32 vector file size mismatch");
  input.seekg(0, std::ios::beg);
  std::vector<std::int32_t> values(
      static_cast<std::size_t>(size) / sizeof(std::int32_t), 0);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
  Require(static_cast<bool>(input), "i32 vector file read failed");
  return values;
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

std::vector<float> Softmax(const std::vector<float>& logits) {
  Require(!logits.empty(), "softmax input is empty");
  const auto max_it = std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  std::vector<double> exp_values;
  exp_values.reserve(logits.size());
  for (const float value : logits) {
    const double exp_value =
        std::exp(static_cast<double>(value) - static_cast<double>(*max_it));
    exp_values.push_back(exp_value);
    sum += exp_value;
  }
  std::vector<float> probabilities;
  probabilities.reserve(logits.size());
  for (const double value : exp_values) {
    probabilities.push_back(static_cast<float>(value / sum));
  }
  return probabilities;
}

std::vector<std::int32_t> TopKIndices(const std::vector<float>& values, int k) {
  Require(k > 0 && static_cast<std::size_t>(k) <= values.size(), "top-k size mismatch");
  std::vector<std::int32_t> indexes(values.size(), 0);
  std::iota(indexes.begin(), indexes.end(), 0);
  std::partial_sort(
      indexes.begin(),
      indexes.begin() + k,
      indexes.end(),
      [&values](std::int32_t lhs, std::int32_t rhs) {
        const float left = values[static_cast<std::size_t>(lhs)];
        const float right = values[static_cast<std::size_t>(rhs)];
        if (left == right) {
          return lhs < rhs;
        }
        return left > right;
      });
  indexes.resize(static_cast<std::size_t>(k));
  return indexes;
}

std::vector<float> Gather(const std::vector<float>& values,
                          const std::vector<std::int32_t>& indexes) {
  std::vector<float> out;
  out.reserve(indexes.size());
  for (const auto index : indexes) {
    Require(index >= 0 && static_cast<std::size_t>(index) < values.size(),
            "gather index out of range");
    out.push_back(values[static_cast<std::size_t>(index)]);
  }
  return out;
}

std::vector<float> NormalizeWeights(const std::vector<float>& weights) {
  float sum = 0.0f;
  for (const float value : weights) {
    sum += value;
  }
  const float denominator = std::max(sum, kMinWeightSum);
  std::vector<float> out;
  out.reserve(weights.size());
  for (const float value : weights) {
    out.push_back(value / denominator);
  }
  return out;
}

RouterResult RunRouter(const std::vector<float>& logits) {
  RouterResult result;
  result.probabilities = Softmax(logits);
  result.topk = TopKIndices(result.probabilities, kExpertUsedCount);
  result.weights = Gather(result.probabilities, result.topk);
  result.weights_norm = NormalizeWeights(result.weights);
  return result;
}

bool ComparePassed(const iq36::VectorCompareStats& stats,
                   double max_abs,
                   double rmse,
                   double min_cosine) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= max_abs &&
         stats.rmse <= rmse &&
         stats.cosine >= min_cosine;
}

IntCompareStats CompareI32(const std::vector<std::int32_t>& lhs,
                           const std::vector<std::int32_t>& rhs) {
  IntCompareStats stats;
  stats.lhs_value_count = lhs.size();
  stats.rhs_value_count = rhs.size();
  stats.compared_value_count = std::min(lhs.size(), rhs.size());
  stats.same_size = lhs.size() == rhs.size();
  for (std::size_t i = 0; i < stats.compared_value_count; ++i) {
    if (lhs[i] != rhs[i]) {
      ++stats.mismatch_count;
    }
  }
  if (!stats.same_size) {
    stats.mismatch_count +=
        std::max(lhs.size(), rhs.size()) - stats.compared_value_count;
  }
  return stats;
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

void WriteI32Compare(const IntCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"lhs_value_count\":" << stats.lhs_value_count << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rhs_value_count\":" << stats.rhs_value_count << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const std::string tensor_name = LayerTensorName(args.layer, "ffn_gate_inp.weight");
    const auto* tensor = iq36::find_tensor(index, tensor_name);
    Require(tensor != nullptr, "router tensor missing");
    const bool tensor_shape_ok =
        tensor->type == 0 &&
        tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kExpertCount};

    const auto input = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_post_norm.bin"));
    const auto oracle_logits = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_logits.bin"));
    const auto oracle_probs = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_probs.bin"));
    const auto oracle_topk = ReadI32VectorFile(JoinPath(args.payload_dir, "ffn_moe_topk.bin"));
    const auto oracle_weights = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weights.bin"));
    const auto oracle_weights_norm = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_weights_norm.bin"));
    const auto cpu_logits = iq36::matvec_tensor(args.model_path, index, tensor_name, input);
    const auto cpu_router = RunRouter(cpu_logits);

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto weights = ReadF32TensorPayload(
        model, *tensor, static_cast<std::size_t>(kHiddenSize * kExpertCount));
    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    const auto gpu = runner.RunF32Matvec(weights, input, kExpertCount, kHiddenSize, args.repeat);
    const auto gpu_router = RunRouter(gpu.output);

    const auto cpu_logits_vs_oracle =
        iq36::compare_vectors(cpu_logits, oracle_logits, kLogitsMismatchThreshold);
    const auto gpu_logits_vs_cpu =
        iq36::compare_vectors(gpu.output, cpu_logits, kLogitsMismatchThreshold);
    const auto gpu_logits_vs_oracle =
        iq36::compare_vectors(gpu.output, oracle_logits, kLogitsMismatchThreshold);
    const auto cpu_probs_vs_oracle =
        iq36::compare_vectors(cpu_router.probabilities, oracle_probs, kWeightsMismatchThreshold);
    const auto gpu_probs_vs_oracle =
        iq36::compare_vectors(gpu_router.probabilities, oracle_probs, kWeightsMismatchThreshold);
    const auto cpu_weights_vs_oracle =
        iq36::compare_vectors(cpu_router.weights, oracle_weights, kWeightsMismatchThreshold);
    const auto gpu_weights_vs_oracle =
        iq36::compare_vectors(gpu_router.weights, oracle_weights, kWeightsMismatchThreshold);
    const auto cpu_weights_norm_vs_oracle =
        iq36::compare_vectors(cpu_router.weights_norm, oracle_weights_norm, kWeightsMismatchThreshold);
    const auto gpu_weights_norm_vs_oracle =
        iq36::compare_vectors(gpu_router.weights_norm, oracle_weights_norm, kWeightsMismatchThreshold);
    const auto cpu_topk_vs_oracle = CompareI32(cpu_router.topk, oracle_topk);
    const auto gpu_topk_vs_oracle = CompareI32(gpu_router.topk, oracle_topk);
    const auto gpu_topk_vs_cpu = CompareI32(gpu_router.topk, cpu_router.topk);

    const bool comparisons_passed =
        ComparePassed(cpu_logits_vs_oracle, kLogitsMaxAbsDiffThreshold,
                      kLogitsRmseThreshold, kMinCosine) &&
        ComparePassed(gpu_logits_vs_cpu, kLogitsMaxAbsDiffThreshold,
                      kLogitsRmseThreshold, kMinCosine) &&
        ComparePassed(gpu_logits_vs_oracle, kLogitsMaxAbsDiffThreshold,
                      kLogitsRmseThreshold, kMinCosine) &&
        ComparePassed(cpu_probs_vs_oracle, kWeightsMaxAbsDiffThreshold,
                      kWeightsRmseThreshold, kMinCosine) &&
        ComparePassed(gpu_probs_vs_oracle, kWeightsMaxAbsDiffThreshold,
                      kWeightsRmseThreshold, kMinCosine) &&
        ComparePassed(cpu_weights_vs_oracle, kWeightsMaxAbsDiffThreshold,
                      kWeightsRmseThreshold, kMinCosine) &&
        ComparePassed(gpu_weights_vs_oracle, kWeightsMaxAbsDiffThreshold,
                      kWeightsRmseThreshold, kMinCosine) &&
        ComparePassed(cpu_weights_norm_vs_oracle, kWeightsMaxAbsDiffThreshold,
                      kWeightsRmseThreshold, kMinCosine) &&
        ComparePassed(gpu_weights_norm_vs_oracle, kWeightsMaxAbsDiffThreshold,
                      kWeightsRmseThreshold, kMinCosine) &&
        cpu_topk_vs_oracle.same_size &&
        cpu_topk_vs_oracle.mismatch_count == 0 &&
        gpu_topk_vs_oracle.same_size &&
        gpu_topk_vs_oracle.mismatch_count == 0 &&
        gpu_topk_vs_cpu.same_size &&
        gpu_topk_vs_cpu.mismatch_count == 0;
    const bool timings_positive = gpu.timing.min_us > 0.0;
    const bool counts_ok =
        input.size() == kHiddenSize &&
        cpu_logits.size() == kExpertCount &&
        gpu.output.size() == kExpertCount &&
        oracle_logits.size() == kExpertCount &&
        oracle_probs.size() == kExpertCount &&
        oracle_topk.size() == kExpertUsedCount &&
        oracle_weights.size() == kExpertUsedCount &&
        oracle_weights_norm.size() == kExpertUsedCount;
    const bool checks_passed =
        load_map.ready &&
        tensor_shape_ok &&
        counts_ok &&
        runner.device_name().find(args.device_substring) != std::string::npos &&
        comparisons_passed &&
        timings_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-f32-router-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
    std::cout << "\"tensor_name\":\"" << JsonEscape(tensor->name) << "\",";
    std::cout << "\"tensor_type\":\"" << iq36::ggml_type_name(tensor->type) << "\",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"cols\":" << kHiddenSize << ",";
    std::cout << "\"rows\":" << kExpertCount << ",";
    std::cout << "\"raw_bytes\":" << weights.size() * sizeof(float) << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runner.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
    std::cout << "\"router_logits_gpu_kernel_min_us\":" << gpu.timing.min_us << ",";
    std::cout << "\"router_logits_gpu_kernel_mean_us\":" << gpu.timing.mean_us << ",";
    std::cout << "\"router_logits_gpu_effective_weight_gb_s\":" << gpu.timing.effective_weight_gb_s << ",";
    std::cout << "\"global_work_items\":" << gpu.timing.global_work_items;
    std::cout << "},\"comparisons\":{";
    std::cout << "\"logits\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_logits_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(gpu_logits_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(gpu_logits_vs_oracle);
    std::cout << "},\"probabilities\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_probs_vs_oracle);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(gpu_probs_vs_oracle);
    std::cout << "},\"weights\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_weights_vs_oracle);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(gpu_weights_vs_oracle);
    std::cout << "},\"weights_norm\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_weights_norm_vs_oracle);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(gpu_weights_norm_vs_oracle);
    std::cout << "},\"topk\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteI32Compare(cpu_topk_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteI32Compare(gpu_topk_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteI32Compare(gpu_topk_vs_oracle);
    std::cout << "}";
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"counts_ok\":" << (counts_ok ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"router_matches_oracle\":" << (comparisons_passed ? "true" : "false") << ",";
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
      raise SystemExit(f"router payload missing: {path}")
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"router payload size mismatch: {path}")
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
      "# GPU F32 Router Probe",
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
      "| output | required comparison | max abs/mismatches | RMSE |",
      "|---|---|---:|---:|",
  ]
  for name, lane in (
      ("logits", "gpu_vs_oracle"),
      ("probabilities", "gpu_vs_oracle"),
      ("weights", "gpu_vs_oracle"),
      ("weights_norm", "gpu_vs_oracle"),
  ):
    cmp = comparisons.get(name, {}).get(lane, {}) if isinstance(comparisons, dict) else {}
    lines.append(f"| {name} | {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  topk = comparisons.get("topk", {}).get("gpu_vs_oracle", {}) if isinstance(comparisons, dict) else {}
  lines.append(f"| topk | gpu_vs_oracle | {topk.get('mismatch_count')} | n/a |")
  lines += [
      "",
      "| kernel | min us | mean us | weight GB/s |",
      "|---|---:|---:|---:|",
      "| router_logits | "
      f"{timings.get('router_logits_gpu_kernel_min_us')} | "
      f"{timings.get('router_logits_gpu_kernel_mean_us')} | "
      f"{timings.get('router_logits_gpu_effective_weight_gb_s')} |",
      "",
      "The probe starts from captured `attn_post_norm`, computes router logits on",
      "GPU with an F32 matvec, then derives softmax/top-k/weights in the probe and",
      "compares them against teacher capture. This is component evidence only; it",
      "does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-f32-router-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_f32_router_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-f32-router-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_f32_router_probe.cpp", args.timeout_s))
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
          f"{shlex.quote(remote_dir + '/tests/gpu_f32_router_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-f32-router-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-f32-router-probe",
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
      {"name": "router_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "router_matches_oracle"))},
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
      "tool": "tools/intel-qwen36-gpu-f32-router-probe.py",
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
      "gpu_f32_router_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("router_logits_kernel_min_us", nested_number(timings, "router_logits_gpu_kernel_min_us")),
          ("router_logits_effective_weight_gb_s", nested_number(timings, "router_logits_gpu_effective_weight_gb_s")),
          ("logits_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "logits", "gpu_vs_oracle", "max_abs_diff")),
          ("probabilities_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "probabilities", "gpu_vs_oracle", "max_abs_diff")),
          ("weights_norm_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "weights_norm", "gpu_vs_oracle", "max_abs_diff")),
          ("topk_gpu_vs_oracle_mismatches", nested_number(comparisons, "topk", "gpu_vs_oracle", "mismatch_count")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
