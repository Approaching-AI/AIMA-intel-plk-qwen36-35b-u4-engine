#!/usr/bin/env python3
"""Run the GPU Q4 x8 selected-expert gate-up handoff gate."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-selected-gate-up-probe-v1"
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
    "ffn_moe_topk": ("ffn_moe_topk.bin", "ffn_moe_topk-{layer}__tok15__ord212.bin", 32),
    "ffn_moe_gate_up": ("ffn_moe_gate_up.bin", "ffn_moe_gate_up-{layer}__tok15__ord215.bin", 32768),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

	#include <algorithm>
	#include <cstdint>
	#include <cstddef>
	#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
	#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kLayerCount = 40;
constexpr int kHiddenSize = 2048;
constexpr int kExpertCount = 256;
constexpr int kExpertUsedCount = 8;
constexpr int kIntermediateSize = 512;
constexpr int kGateUpRowsPerExpert = kIntermediateSize * 2;
constexpr int kOutputValues = kExpertUsedCount * kGateUpRowsPerExpert;
constexpr int kSharedGateUpRows = kIntermediateSize * 2;
constexpr int kSourceTokenPosition = 15;
constexpr double kMismatchThreshold = 5e-3;
constexpr double kMaxAbsDiffThreshold = 5e-3;
constexpr double kRmseThreshold = 5e-4;
constexpr double kMinCosine = 0.999;
constexpr double kTop16FloorCoveringRatio = 1.0941549767378367;

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

std::vector<std::uint8_t> ReadSelectedExpertRaw(std::ifstream& model,
                                                const iq36::GgufTensorInfo& tensor,
                                                const std::vector<std::int32_t>& expert_ids,
                                                std::uint64_t rows_per_expert,
                                                std::uint64_t row_nbytes) {
  std::vector<std::uint8_t> raw;
  raw.resize(static_cast<std::size_t>(expert_ids.size() * rows_per_expert * row_nbytes));
  for (std::size_t selected = 0; selected < expert_ids.size(); ++selected) {
    const auto expert_id = expert_ids[selected];
    Require(expert_id >= 0 && expert_id < kExpertCount, "selected expert id out of range");
    const std::uint64_t expert_row_base =
        static_cast<std::uint64_t>(expert_id) * rows_per_expert;
    const std::uint64_t source_offset = tensor.absolute_offset + expert_row_base * row_nbytes;
    const std::size_t target_offset =
        selected * static_cast<std::size_t>(rows_per_expert * row_nbytes);
    const std::size_t byte_count = static_cast<std::size_t>(rows_per_expert * row_nbytes);
    model.clear();
    model.seekg(static_cast<std::streamoff>(source_offset), std::ios::beg);
    Require(static_cast<bool>(model), "selected expert slice seek failed");
    model.read(reinterpret_cast<char*>(raw.data() + target_offset),
               static_cast<std::streamsize>(byte_count));
    Require(model.gcount() == static_cast<std::streamsize>(byte_count),
            "selected expert slice read failed");
  }
  return raw;
}

std::vector<std::uint8_t> ReadTensorRaw(std::ifstream& model,
                                        const iq36::GgufTensorInfo& tensor) {
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(tensor.nbytes));
  model.clear();
  model.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(model), "tensor raw seek failed");
  model.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  Require(model.gcount() == static_cast<std::streamsize>(raw.size()),
          "tensor raw read failed");
  return raw;
}

std::vector<float> ConcatFloatVectors(const std::vector<float>& lhs,
                                      const std::vector<float>& rhs) {
  std::vector<float> out;
  out.reserve(lhs.size() + rhs.size());
  out.insert(out.end(), lhs.begin(), lhs.end());
  out.insert(out.end(), rhs.begin(), rhs.end());
  return out;
}

std::vector<std::uint8_t> ConcatRawVectors(const std::vector<std::uint8_t>& first,
                                           const std::vector<std::uint8_t>& second) {
  std::vector<std::uint8_t> out;
  out.reserve(first.size() + second.size());
  out.insert(out.end(), first.begin(), first.end());
  out.insert(out.end(), second.begin(), second.end());
  return out;
}

std::vector<float> SliceFloatVector(const std::vector<float>& values,
                                    std::size_t offset,
                                    std::size_t count) {
  Require(offset <= values.size() && count <= values.size() - offset,
          "float slice out of range");
  return std::vector<float>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                            values.begin() + static_cast<std::ptrdiff_t>(offset + count));
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

void WriteI32Vector(const std::vector<std::int32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

void WriteU32Vector(const std::vector<std::uint32_t>& values) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << values[i];
  }
  std::cout << "]";
}

std::vector<std::int32_t> BuildTop16MaterialExpertIds(
    const std::vector<std::int32_t>& expert_ids) {
  std::vector<std::int32_t> material = expert_ids;
  for (std::int32_t expert = 0;
       expert < kExpertCount && material.size() < 16; ++expert) {
    if (std::find(material.begin(), material.end(), expert) == material.end()) {
      material.push_back(expert);
    }
  }
  Require(material.size() == 16, "top16 material expert count mismatch");
  std::sort(material.begin(), material.end());
  material.erase(std::unique(material.begin(), material.end()), material.end());
  Require(material.size() == 16, "top16 material expert ids not unique");
  return material;
}

std::vector<std::uint32_t> BuildTop16Positions(
    const std::vector<std::int32_t>& expert_ids,
    const std::vector<std::int32_t>& material_ids) {
  std::vector<std::uint32_t> positions;
  positions.reserve(expert_ids.size());
  for (const auto expert_id : expert_ids) {
    const auto it = std::find(material_ids.begin(), material_ids.end(), expert_id);
    Require(it != material_ids.end(), "top16 material missing selected expert");
    positions.push_back(static_cast<std::uint32_t>(
        std::distance(material_ids.begin(), it)));
  }
  return positions;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
	    const auto index = iq36::parse_gguf_model_index(args.model_path);
	    const auto load_map = iq36::validate_qwen36_load_map(index);
	    const std::string tensor_name = LayerTensorName(args.layer, "ffn_gate_up_exps.weight");
	    const auto* tensor = iq36::find_tensor(index, tensor_name);
	    Require(tensor != nullptr, "selected expert gate-up tensor missing");
	    const std::string shared_gate_tensor_name =
	        LayerTensorName(args.layer, "ffn_gate_shexp.weight");
	    const std::string shared_up_tensor_name =
	        LayerTensorName(args.layer, "ffn_up_shexp.weight");
	    const auto* shared_gate_tensor = iq36::find_tensor(index, shared_gate_tensor_name);
	    const auto* shared_up_tensor = iq36::find_tensor(index, shared_up_tensor_name);
	    Require(shared_gate_tensor != nullptr, "shared gate tensor missing");
	    Require(shared_up_tensor != nullptr, "shared up tensor missing");
	    const bool tensor_shape_ok =
	        tensor->type == 12 &&
	        tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kGateUpRowsPerExpert, kExpertCount};
	    const bool shared_tensor_shape_ok =
	        shared_gate_tensor->type == 12 &&
	        shared_up_tensor->type == 12 &&
	        shared_gate_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize} &&
	        shared_up_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kIntermediateSize};

	    const auto input = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_post_norm.bin"));
	    const auto expert_ids = ReadI32VectorFile(JoinPath(args.payload_dir, "ffn_moe_topk.bin"));
	    const auto oracle = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "ffn_moe_gate_up.bin"));
	    const auto cpu = iq36::matvec_expert_tensor(args.model_path, index, tensor_name, input, expert_ids);
	    const auto shared_gate_cpu =
	        iq36::matvec_tensor(args.model_path, index, shared_gate_tensor_name, input);
	    const auto shared_up_cpu =
	        iq36::matvec_tensor(args.model_path, index, shared_up_tensor_name, input);
	    const auto shared_cpu = ConcatFloatVectors(shared_gate_cpu, shared_up_cpu);

	    const std::uint64_t cols = tensor->dims[0];
	    const std::uint64_t rows_per_expert = tensor->dims[1];
	    const std::uint64_t selected_rows = rows_per_expert * expert_ids.size();
	    const std::uint64_t blocks_per_row = cols / 256;
    const std::uint64_t row_nbytes = tensor->nbytes / (tensor->dims[1] * tensor->dims[2]);
    Require(row_nbytes == blocks_per_row * 144, "selected expert Q4 row byte mismatch");
    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
	    const auto selected_raw =
	        ReadSelectedExpertRaw(model, *tensor, expert_ids, rows_per_expert, row_nbytes);
	    const auto top16_material_ids = BuildTop16MaterialExpertIds(expert_ids);
	    const auto top16_positions =
	        BuildTop16Positions(expert_ids, top16_material_ids);
	    const auto top16_material_raw =
	        ReadSelectedExpertRaw(model, *tensor, top16_material_ids,
	                              rows_per_expert, row_nbytes);
	    const auto packed = iq36::PackQ4Kx8(selected_raw, selected_rows, blocks_per_row);
	    const auto top16_material_packed =
	        iq36::PackQ4Kx8(top16_material_raw, rows_per_expert * 16,
	                        blocks_per_row);
	    const auto shared_gate_raw = ReadTensorRaw(model, *shared_gate_tensor);
	    const auto shared_up_raw = ReadTensorRaw(model, *shared_up_tensor);
	    const auto shared_raw = ConcatRawVectors(shared_gate_raw, shared_up_raw);
	    const std::uint64_t shared_rows =
	        shared_gate_tensor->dims[1] + shared_up_tensor->dims[1];
	    Require(shared_rows == kSharedGateUpRows, "shared gate-up row count mismatch");
	    Require(shared_gate_tensor->dims[0] == cols && shared_up_tensor->dims[0] == cols,
	            "shared gate-up column mismatch");
	    Require(shared_gate_tensor->nbytes / shared_gate_tensor->dims[1] == row_nbytes &&
	                shared_up_tensor->nbytes / shared_up_tensor->dims[1] == row_nbytes,
	            "shared gate-up Q4 row byte mismatch");
	    const auto shared_packed = iq36::PackQ4Kx8(shared_raw, shared_rows, blocks_per_row);
	    const auto combined_raw = ConcatRawVectors(selected_raw, shared_raw);
	    const std::uint64_t combined_rows = selected_rows + shared_rows;
	    const auto combined_packed =
	        iq36::PackQ4Kx8(combined_raw, combined_rows, blocks_per_row);
	    const auto q8 = iq36::QuantizeQ8KInputPlanes(input);
	    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
	    const auto gpu = runner.Run(packed, q8.qs, q8.bsums, q8.d, selected_rows,
	                                blocks_per_row, args.repeat,
	                                iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
	    const auto shared_gpu =
	        runner.Run(shared_packed, q8.qs, q8.bsums, q8.d, shared_rows,
	                   blocks_per_row, args.repeat,
	                   iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
	    const auto combined_gpu =
	        runner.Run(combined_packed, q8.qs, q8.bsums, q8.d, combined_rows,
	                   blocks_per_row, args.repeat,
	                   iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
	    std::vector<std::uint64_t> selected_handles;
	    selected_handles.reserve(expert_ids.size());
	    for (const auto expert_id : expert_ids) {
	      const std::vector<std::int32_t> one_expert{expert_id};
	      const auto one_raw =
	          ReadSelectedExpertRaw(model, *tensor, one_expert, rows_per_expert,
	                                row_nbytes);
	      const auto one_packed =
	          iq36::PackQ4Kx8(one_raw, rows_per_expert, blocks_per_row);
	      selected_handles.push_back(
	          runner.UploadPackedQ4X8(one_packed, rows_per_expert,
	                                  blocks_per_row));
	    }
	    const auto shared_handle =
	        runner.UploadPackedQ4X8(shared_packed, shared_rows, blocks_per_row);
	    const auto top16_material_handle =
	        runner.UploadPackedQ4X8(top16_material_packed, rows_per_expert * 16,
	                                blocks_per_row);
	    const auto no_concat_swiglu =
	        runner.RunResidentPackedQ4X8Expert8PlusSharedThenSwiGlu(
	            selected_handles, shared_handle, q8.qs, q8.bsums, q8.d,
	            kIntermediateSize, args.repeat,
	            iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
	    const auto top16_indexed_swiglu =
	        runner.RunResidentPackedQ4X8TopKIndexedExpert8PlusSharedThenSwiGlu(
	            top16_material_handle, shared_handle, top16_positions, q8.qs,
	            q8.bsums, q8.d, kIntermediateSize, args.repeat,
	            iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
	    const auto combined_selected =
	        SliceFloatVector(combined_gpu.output, 0, static_cast<std::size_t>(selected_rows));
	    const auto combined_shared =
	        SliceFloatVector(combined_gpu.output, static_cast<std::size_t>(selected_rows),
	                         static_cast<std::size_t>(shared_rows));
	    const auto oracle_swiglu =
	        iq36::apply_swiglu_from_gate_up(oracle, kIntermediateSize,
	                                        kExpertUsedCount);
	    const auto shared_swiglu_cpu =
	        iq36::apply_swiglu_from_gate_up(shared_cpu, kIntermediateSize, 1);
	    const std::size_t selected_swiglu_values =
	        static_cast<std::size_t>(kIntermediateSize * kExpertUsedCount);
	    const auto no_concat_selected_swiglu =
	        SliceFloatVector(no_concat_swiglu.swiglu, 0, selected_swiglu_values);
	    const auto no_concat_shared_swiglu =
	        SliceFloatVector(no_concat_swiglu.swiglu, selected_swiglu_values,
	                         kIntermediateSize);
	    const auto top16_selected_swiglu =
	        SliceFloatVector(top16_indexed_swiglu.swiglu, 0,
	                         selected_swiglu_values);
	    const auto top16_shared_swiglu =
	        SliceFloatVector(top16_indexed_swiglu.swiglu,
	                         selected_swiglu_values, kIntermediateSize);

	    const auto cpu_vs_oracle = iq36::compare_vectors(cpu, oracle, kMismatchThreshold);
	    const auto gpu_vs_cpu = iq36::compare_vectors(gpu.output, cpu, kMismatchThreshold);
	    const auto gpu_vs_oracle = iq36::compare_vectors(gpu.output, oracle, kMismatchThreshold);
	    const auto shared_gpu_vs_cpu =
	        iq36::compare_vectors(shared_gpu.output, shared_cpu, kMismatchThreshold);
	    const auto combined_selected_vs_oracle =
	        iq36::compare_vectors(combined_selected, oracle, kMismatchThreshold);
	    const auto combined_shared_vs_cpu =
	        iq36::compare_vectors(combined_shared, shared_cpu, kMismatchThreshold);
	    const auto no_concat_selected_vs_oracle_swiglu =
	        iq36::compare_vectors(no_concat_selected_swiglu, oracle_swiglu,
	                              kMismatchThreshold);
	    const auto no_concat_shared_vs_cpu_swiglu =
	        iq36::compare_vectors(no_concat_shared_swiglu, shared_swiglu_cpu,
	                              kMismatchThreshold);
	    const auto top16_selected_vs_oracle_swiglu =
	        iq36::compare_vectors(top16_selected_swiglu, oracle_swiglu,
	                              kMismatchThreshold);
	    const auto top16_shared_vs_cpu_swiglu =
	        iq36::compare_vectors(top16_shared_swiglu, shared_swiglu_cpu,
	                              kMismatchThreshold);
	    const auto top16_vs_no_concat_swiglu =
	        iq36::compare_vectors(top16_indexed_swiglu.swiglu,
	                              no_concat_swiglu.swiglu, kMismatchThreshold);
	    const bool selected_comparisons_passed =
	        ComparePassed(cpu_vs_oracle) && ComparePassed(gpu_vs_cpu) &&
	        ComparePassed(gpu_vs_oracle);
	    const bool shared_comparisons_passed = ComparePassed(shared_gpu_vs_cpu);
	    const bool combined_comparisons_passed =
	        ComparePassed(combined_selected_vs_oracle) &&
	        ComparePassed(combined_shared_vs_cpu);
	    const bool no_concat_swiglu_comparisons_passed =
	        ComparePassed(no_concat_selected_vs_oracle_swiglu) &&
	        ComparePassed(no_concat_shared_vs_cpu_swiglu);
	    const bool top16_indexed_swiglu_comparisons_passed =
	        ComparePassed(top16_selected_vs_oracle_swiglu) &&
	        ComparePassed(top16_shared_vs_cpu_swiglu) &&
	        ComparePassed(top16_vs_no_concat_swiglu);
	    const double top16_indexed_vs_no_concat_shell_speedup =
	        top16_indexed_swiglu.timing.shell_sum_min_us > 0.0
	            ? no_concat_swiglu.timing.shell_sum_min_us /
	                  top16_indexed_swiglu.timing.shell_sum_min_us
	            : 0.0;
	    const double top16_indexed_vs_no_concat_matvec_speedup =
	        top16_indexed_swiglu.timing.matvec.min_us > 0.0
	            ? no_concat_swiglu.timing.matvec.min_us /
	                  top16_indexed_swiglu.timing.matvec.min_us
	            : 0.0;
	    const bool top16_indexed_floor_covering_ratio =
	        top16_indexed_vs_no_concat_shell_speedup >= kTop16FloorCoveringRatio;
	    const bool comparisons_passed =
	        selected_comparisons_passed && shared_comparisons_passed &&
	        combined_comparisons_passed &&
	        no_concat_swiglu_comparisons_passed &&
	        top16_indexed_swiglu_comparisons_passed;
	    const bool counts_ok =
	        input.size() == kHiddenSize &&
	        expert_ids.size() == kExpertUsedCount &&
	        top16_material_ids.size() == 16 &&
	        top16_positions.size() == kExpertUsedCount &&
	        oracle.size() == kOutputValues &&
	        oracle_swiglu.size() == selected_swiglu_values &&
	        cpu.size() == kOutputValues &&
	        shared_cpu.size() == kSharedGateUpRows &&
	        gpu.output.size() == kOutputValues &&
	        shared_gpu.output.size() == kSharedGateUpRows &&
	        combined_gpu.output.size() == selected_rows + shared_rows &&
	        no_concat_swiglu.swiglu.size() ==
	            selected_swiglu_values + kIntermediateSize &&
	        top16_indexed_swiglu.swiglu.size() ==
	            selected_swiglu_values + kIntermediateSize;
	    const bool timings_positive =
	        gpu.timing.min_us > 0.0 && shared_gpu.timing.min_us > 0.0 &&
	        combined_gpu.timing.min_us > 0.0 &&
	        no_concat_swiglu.timing.shell_sum_min_us > 0.0 &&
	        top16_indexed_swiglu.timing.shell_sum_min_us > 0.0;
	    const double separate_min_us = gpu.timing.min_us + shared_gpu.timing.min_us;
	    const double combined_vs_separate_speedup =
	        combined_gpu.timing.min_us > 0.0 ? separate_min_us / combined_gpu.timing.min_us : 0.0;
	    const double combined_vs_selected_only_ratio =
	        combined_gpu.timing.min_us > 0.0 ? gpu.timing.min_us / combined_gpu.timing.min_us : 0.0;
	    const bool checks_passed =
	        load_map.ready &&
	        tensor_shape_ok &&
	        shared_tensor_shape_ok &&
	        counts_ok &&
	        runner.device_name().find(args.device_substring) != std::string::npos &&
	        comparisons_passed &&
	        timings_positive &&
	        top16_indexed_floor_covering_ratio;

    std::cout << std::setprecision(10);
    std::cout << "{";
	    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-selected-gate-up-probe-v1\",";
	    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
	    std::cout << "\"layer\":" << args.layer << ",";
	    std::cout << "\"source_token_position\":" << kSourceTokenPosition << ",";
	    std::cout << "\"tensor_name\":\"" << JsonEscape(tensor->name) << "\",";
	    std::cout << "\"shared_gate_tensor_name\":\""
	              << JsonEscape(shared_gate_tensor->name) << "\",";
	    std::cout << "\"shared_up_tensor_name\":\""
	              << JsonEscape(shared_up_tensor->name) << "\",";
	    std::cout << "\"tensor_type\":\"" << iq36::ggml_type_name(tensor->type) << "\",";
	    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
	    std::cout << "\"shared_tensor_shape_ok\":"
	              << (shared_tensor_shape_ok ? "true" : "false") << ",";
	    std::cout << "\"cols\":" << cols << ",";
	    std::cout << "\"rows_per_expert\":" << rows_per_expert << ",";
	    std::cout << "\"selected_rows\":" << selected_rows << ",";
	    std::cout << "\"shared_rows\":" << shared_rows << ",";
	    std::cout << "\"combined_rows\":" << combined_rows << ",";
	    std::cout << "\"blocks_per_row\":" << blocks_per_row << ",";
	    std::cout << "\"selected_raw_bytes\":" << selected_raw.size() << ",";
	    std::cout << "\"shared_raw_bytes\":" << shared_raw.size() << ",";
	    std::cout << "\"combined_raw_bytes\":" << combined_raw.size() << ",";
	    std::cout << "\"top16_material_raw_bytes\":"
	              << top16_material_raw.size() << ",";
	    std::cout << "\"packed_q4k_x8_bytes\":" << packed.size() << ",";
	    std::cout << "\"shared_packed_q4k_x8_bytes\":" << shared_packed.size() << ",";
	    std::cout << "\"combined_packed_q4k_x8_bytes\":" << combined_packed.size() << ",";
	    std::cout << "\"top16_material_packed_q4k_x8_bytes\":"
	              << top16_material_packed.size() << ",";
	    std::cout << "\"expert_ids\":";
    WriteI32Vector(expert_ids);
    std::cout << ",";
	    std::cout << "\"top16_material_expert_ids\":";
	    WriteI32Vector(top16_material_ids);
	    std::cout << ",";
	    std::cout << "\"top16_selected_positions\":";
	    WriteU32Vector(top16_positions);
	    std::cout << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runner.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"timings\":{";
	    std::cout << "\"selected_gate_up_gpu_kernel_min_us\":" << gpu.timing.min_us << ",";
	    std::cout << "\"selected_gate_up_gpu_kernel_mean_us\":" << gpu.timing.mean_us << ",";
	    std::cout << "\"selected_gate_up_gpu_effective_packed_gb_s\":" << gpu.timing.effective_packed_gb_s << ",";
	    std::cout << "\"shared_gate_up_gpu_kernel_min_us\":" << shared_gpu.timing.min_us << ",";
	    std::cout << "\"shared_gate_up_gpu_kernel_mean_us\":" << shared_gpu.timing.mean_us << ",";
	    std::cout << "\"shared_gate_up_gpu_effective_packed_gb_s\":"
	              << shared_gpu.timing.effective_packed_gb_s << ",";
	    std::cout << "\"combined_gate_up_gpu_kernel_min_us\":"
	              << combined_gpu.timing.min_us << ",";
	    std::cout << "\"combined_gate_up_gpu_kernel_mean_us\":"
	              << combined_gpu.timing.mean_us << ",";
	    std::cout << "\"combined_gate_up_gpu_effective_packed_gb_s\":"
	              << combined_gpu.timing.effective_packed_gb_s << ",";
	    std::cout << "\"no_concat_gateup_swiglu_shell_min_us\":"
	              << no_concat_swiglu.timing.shell_sum_min_us << ",";
	    std::cout << "\"no_concat_gateup_matvec_min_us\":"
	              << no_concat_swiglu.timing.matvec.min_us << ",";
	    std::cout << "\"no_concat_swiglu_min_us\":"
	              << no_concat_swiglu.timing.swiglu_min_us << ",";
	    std::cout << "\"no_concat_gateup_effective_packed_gb_s\":"
	              << no_concat_swiglu.timing.matvec.effective_packed_gb_s << ",";
	    std::cout << "\"top16_indexed_gateup_swiglu_shell_min_us\":"
	              << top16_indexed_swiglu.timing.shell_sum_min_us << ",";
	    std::cout << "\"top16_indexed_gateup_matvec_min_us\":"
	              << top16_indexed_swiglu.timing.matvec.min_us << ",";
	    std::cout << "\"top16_indexed_swiglu_min_us\":"
	              << top16_indexed_swiglu.timing.swiglu_min_us << ",";
	    std::cout << "\"top16_indexed_gateup_effective_packed_gb_s\":"
	              << top16_indexed_swiglu.timing.matvec.effective_packed_gb_s << ",";
	    std::cout << "\"top16_indexed_vs_no_concat_shell_speedup\":"
	              << top16_indexed_vs_no_concat_shell_speedup << ",";
	    std::cout << "\"top16_indexed_vs_no_concat_matvec_speedup\":"
	              << top16_indexed_vs_no_concat_matvec_speedup << ",";
	    std::cout << "\"top16_floor_covering_required_ratio\":"
	              << kTop16FloorCoveringRatio << ",";
	    std::cout << "\"separate_gate_up_gpu_kernel_min_us\":" << separate_min_us << ",";
	    std::cout << "\"combined_vs_separate_speedup\":"
	              << combined_vs_separate_speedup << ",";
	    std::cout << "\"combined_vs_selected_only_ratio\":"
	              << combined_vs_selected_only_ratio << ",";
	    std::cout << "\"global_work_items\":" << gpu.timing.global_work_items << ",";
	    std::cout << "\"shared_global_work_items\":"
	              << shared_gpu.timing.global_work_items << ",";
	    std::cout << "\"combined_global_work_items\":"
	              << combined_gpu.timing.global_work_items << ",";
	    std::cout << "\"rows_per_work_item\":" << gpu.timing.rows_per_work_item;
	    std::cout << "},\"comparisons\":{";
	    std::cout << "\"selected_gate_up\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(gpu_vs_cpu);
	    std::cout << ",\"gpu_vs_oracle\":";
	    WriteCompare(gpu_vs_oracle);
	    std::cout << "}";
	    std::cout << ",\"shared_gate_up\":{";
	    std::cout << "\"gpu_vs_cpu\":";
	    WriteCompare(shared_gpu_vs_cpu);
	    std::cout << "}";
	    std::cout << ",\"combined_gate_up\":{";
	    std::cout << "\"selected_vs_oracle\":";
	    WriteCompare(combined_selected_vs_oracle);
	    std::cout << ",\"shared_vs_cpu\":";
	    WriteCompare(combined_shared_vs_cpu);
	    std::cout << "}";
	    std::cout << ",\"no_concat_gateup_swiglu\":{";
	    std::cout << "\"selected_vs_oracle_swiglu\":";
	    WriteCompare(no_concat_selected_vs_oracle_swiglu);
	    std::cout << ",\"shared_vs_cpu_swiglu\":";
	    WriteCompare(no_concat_shared_vs_cpu_swiglu);
	    std::cout << "}";
	    std::cout << ",\"top16_indexed_gateup_swiglu\":{";
	    std::cout << "\"selected_vs_oracle_swiglu\":";
	    WriteCompare(top16_selected_vs_oracle_swiglu);
	    std::cout << ",\"shared_vs_cpu_swiglu\":";
	    WriteCompare(top16_shared_vs_cpu_swiglu);
	    std::cout << ",\"indexed_vs_no_concat_swiglu\":";
	    WriteCompare(top16_vs_no_concat_swiglu);
	    std::cout << "}";
	    std::cout << "},\"checks\":{";
	    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
	    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
	    std::cout << "\"shared_tensor_shape_ok\":"
	              << (shared_tensor_shape_ok ? "true" : "false") << ",";
	    std::cout << "\"counts_ok\":" << (counts_ok ? "true" : "false") << ",";
	    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
	    std::cout << "\"selected_gate_up_matches_oracle\":"
	              << (selected_comparisons_passed ? "true" : "false") << ",";
	    std::cout << "\"shared_gate_up_matches_cpu\":"
	              << (shared_comparisons_passed ? "true" : "false") << ",";
	    std::cout << "\"combined_gate_up_matches_references\":"
	              << (combined_comparisons_passed ? "true" : "false") << ",";
	    std::cout << "\"no_concat_gateup_swiglu_matches_references\":"
	              << (no_concat_swiglu_comparisons_passed ? "true" : "false") << ",";
	    std::cout << "\"top16_indexed_gateup_swiglu_matches_references\":"
	              << (top16_indexed_swiglu_comparisons_passed ? "true" : "false") << ",";
	    std::cout << "\"top16_indexed_floor_covering_ratio\":"
	              << (top16_indexed_floor_covering_ratio ? "true" : "false") << ",";
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
      raise SystemExit(f"selected gate-up payload missing: {path}")
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"selected gate-up payload size mismatch: {path}")
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
  selected_comparison = (
      probe.get("comparisons", {}).get("selected_gate_up", {})
      if isinstance(probe, dict)
      else {}
  )
  shared_comparison = (
      probe.get("comparisons", {}).get("shared_gate_up", {})
      if isinstance(probe, dict)
      else {}
  )
  combined_comparison = (
      probe.get("comparisons", {}).get("combined_gate_up", {})
      if isinstance(probe, dict)
      else {}
  )
  top16_comparison = (
      probe.get("comparisons", {}).get("top16_indexed_gateup_swiglu", {})
      if isinstance(probe, dict)
      else {}
  )
  timings = probe.get("timings", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Q4-X8 Selected+Shared Gate-Up Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- tensor: `{probe.get('tensor_name')}` selected rows `{probe.get('selected_rows')}`",
      f"- shared rows: `{probe.get('shared_rows')}`; combined rows: `{probe.get('combined_rows')}`",
      f"- expert ids: `{probe.get('expert_ids')}`",
      f"- top16 material ids: `{probe.get('top16_material_expert_ids')}`",
      f"- top16 selected positions: `{probe.get('top16_selected_positions')}`",
      "",
      "| comparison | max abs | RMSE |",
      "|---|---:|---:|",
  ]
  comparison_rows = [
      ("selected_cpu_vs_oracle", selected_comparison.get("cpu_vs_oracle", {})),
      ("selected_gpu_vs_oracle", selected_comparison.get("gpu_vs_oracle", {})),
      ("shared_gpu_vs_cpu", shared_comparison.get("gpu_vs_cpu", {})),
      ("combined_selected_vs_oracle", combined_comparison.get("selected_vs_oracle", {})),
      ("combined_shared_vs_cpu", combined_comparison.get("shared_vs_cpu", {})),
      ("top16_indexed_vs_no_concat_swiglu", top16_comparison.get("indexed_vs_no_concat_swiglu", {})),
      ("top16_selected_vs_oracle_swiglu", top16_comparison.get("selected_vs_oracle_swiglu", {})),
  ]
  for lane, cmp in comparison_rows:
    cmp = cmp if isinstance(cmp, dict) else {}
    lines.append(f"| {lane} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} |")
  lines += [
      "",
      "| kernel | min us | mean us | packed GB/s |",
      "|---|---:|---:|---:|",
      "| selected_gate_up | "
      f"{timings.get('selected_gate_up_gpu_kernel_min_us')} | "
      f"{timings.get('selected_gate_up_gpu_kernel_mean_us')} | "
      f"{timings.get('selected_gate_up_gpu_effective_packed_gb_s')} |",
      "| shared_gate_up | "
      f"{timings.get('shared_gate_up_gpu_kernel_min_us')} | "
      f"{timings.get('shared_gate_up_gpu_kernel_mean_us')} | "
      f"{timings.get('shared_gate_up_gpu_effective_packed_gb_s')} |",
      "| combined_gate_up | "
      f"{timings.get('combined_gate_up_gpu_kernel_min_us')} | "
      f"{timings.get('combined_gate_up_gpu_kernel_mean_us')} | "
      f"{timings.get('combined_gate_up_gpu_effective_packed_gb_s')} |",
      "| no_concat_gateup_swiglu | "
      f"{timings.get('no_concat_gateup_swiglu_shell_min_us')} | "
      f"{timings.get('no_concat_gateup_swiglu_shell_min_us')} | "
      f"{timings.get('no_concat_gateup_effective_packed_gb_s')} |",
      "| top16_indexed_gateup_swiglu | "
      f"{timings.get('top16_indexed_gateup_swiglu_shell_min_us')} | "
      f"{timings.get('top16_indexed_gateup_swiglu_shell_min_us')} | "
      f"{timings.get('top16_indexed_gateup_effective_packed_gb_s')} |",
      "",
      "- separate selected+shared min us: "
      f"`{timings.get('separate_gate_up_gpu_kernel_min_us')}`",
      "- combined/separate kernel speedup: "
      f"`{timings.get('combined_vs_separate_speedup')}`",
      "- top16 indexed/no-concat shell speedup: "
      f"`{timings.get('top16_indexed_vs_no_concat_shell_speedup')}`",
      "- top16 floor-covering required ratio: "
      f"`{timings.get('top16_floor_covering_required_ratio')}`",
      "",
      "The probe concatenates the selected top-k expert rows with shared gate/up rows,",
      "packs the combined matrix into Q4_Kx8, and runs one larger rowlane matvec.",
      "This is component evidence only; it does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-selected-gate-up-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_q4x8_selected_gate_up_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-selected-gate-up-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_selected_gate_up_probe.cpp", args.timeout_s))
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
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_selected_gate_up_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-selected-gate-up-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-q4x8-selected-gate-up-probe",
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
	      {"name": "selected_gate_up_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "selected_gate_up_matches_oracle"))},
	      {"name": "shared_gate_up_matches_cpu", "pass": bool(probe and nested_bool(probe, "checks", "shared_gate_up_matches_cpu"))},
	      {"name": "combined_gate_up_matches_references", "pass": bool(probe and nested_bool(probe, "checks", "combined_gate_up_matches_references"))},
	      {"name": "no_concat_gateup_swiglu_matches_references", "pass": bool(probe and nested_bool(probe, "checks", "no_concat_gateup_swiglu_matches_references"))},
	      {"name": "top16_indexed_gateup_swiglu_matches_references", "pass": bool(probe and nested_bool(probe, "checks", "top16_indexed_gateup_swiglu_matches_references"))},
	      {"name": "top16_indexed_floor_covering_ratio", "pass": bool(probe and nested_bool(probe, "checks", "top16_indexed_floor_covering_ratio"))},
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
      "tool": "tools/intel-qwen36-gpu-q4x8-selected-gate-up-probe.py",
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
	      "gpu_q4x8_selected_shared_gate_up_probe",
	      [
	          ("required_checks_passed", required_checks_passed),
	          ("selected_gate_up_kernel_min_us", nested_number(timings, "selected_gate_up_gpu_kernel_min_us")),
	          ("selected_gate_up_effective_packed_gb_s", nested_number(timings, "selected_gate_up_gpu_effective_packed_gb_s")),
	          ("shared_gate_up_kernel_min_us", nested_number(timings, "shared_gate_up_gpu_kernel_min_us")),
	          ("shared_gate_up_effective_packed_gb_s", nested_number(timings, "shared_gate_up_gpu_effective_packed_gb_s")),
	          ("combined_gate_up_kernel_min_us", nested_number(timings, "combined_gate_up_gpu_kernel_min_us")),
	          ("combined_gate_up_effective_packed_gb_s", nested_number(timings, "combined_gate_up_gpu_effective_packed_gb_s")),
	          ("no_concat_gateup_swiglu_shell_min_us", nested_number(timings, "no_concat_gateup_swiglu_shell_min_us")),
	          ("no_concat_gateup_matvec_min_us", nested_number(timings, "no_concat_gateup_matvec_min_us")),
	          ("no_concat_gateup_effective_packed_gb_s", nested_number(timings, "no_concat_gateup_effective_packed_gb_s")),
	          ("top16_indexed_gateup_swiglu_shell_min_us", nested_number(timings, "top16_indexed_gateup_swiglu_shell_min_us")),
	          ("top16_indexed_gateup_matvec_min_us", nested_number(timings, "top16_indexed_gateup_matvec_min_us")),
	          ("top16_indexed_gateup_effective_packed_gb_s", nested_number(timings, "top16_indexed_gateup_effective_packed_gb_s")),
	          ("top16_indexed_vs_no_concat_shell_speedup", nested_number(timings, "top16_indexed_vs_no_concat_shell_speedup")),
	          ("top16_indexed_vs_no_concat_matvec_speedup", nested_number(timings, "top16_indexed_vs_no_concat_matvec_speedup")),
	          ("top16_floor_covering_required_ratio", nested_number(timings, "top16_floor_covering_required_ratio")),
	          ("separate_gate_up_kernel_min_us", nested_number(timings, "separate_gate_up_gpu_kernel_min_us")),
	          ("combined_vs_separate_speedup", nested_number(timings, "combined_vs_separate_speedup")),
	          ("combined_vs_selected_only_ratio", nested_number(timings, "combined_vs_selected_only_ratio")),
	          ("selected_gate_up_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "selected_gate_up", "gpu_vs_oracle", "max_abs_diff")),
	          ("selected_gate_up_gpu_vs_oracle_rmse", nested_number(comparisons, "selected_gate_up", "gpu_vs_oracle", "rmse")),
	          ("shared_gate_up_gpu_vs_cpu_max_abs_diff", nested_number(comparisons, "shared_gate_up", "gpu_vs_cpu", "max_abs_diff")),
	          ("shared_gate_up_gpu_vs_cpu_rmse", nested_number(comparisons, "shared_gate_up", "gpu_vs_cpu", "rmse")),
	          ("combined_gate_up_selected_vs_oracle_max_abs_diff", nested_number(comparisons, "combined_gate_up", "selected_vs_oracle", "max_abs_diff")),
	          ("combined_gate_up_shared_vs_cpu_max_abs_diff", nested_number(comparisons, "combined_gate_up", "shared_vs_cpu", "max_abs_diff")),
	          ("top16_indexed_vs_no_concat_swiglu_max_abs_diff", nested_number(comparisons, "top16_indexed_gateup_swiglu", "indexed_vs_no_concat_swiglu", "max_abs_diff")),
	          ("top16_indexed_selected_vs_oracle_swiglu_max_abs_diff", nested_number(comparisons, "top16_indexed_gateup_swiglu", "selected_vs_oracle_swiglu", "max_abs_diff")),
	      ],
	  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
