#!/usr/bin/env python3
"""Probe the shared-Q8 linear-preconv qkv/conv regression root.

This is component evidence, not decode or benchmark evidence. It compares the
resident host-Q8 qkv+conv component against the resident F32-input device-Q8
qkv+conv carrier on the target, then combines that result with the seq77
shared-Q8 profile to decide whether the regression is in qkv/conv itself or in
the bundled shared-Q8 alpha/beta/z envelope.
"""

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
SCHEMA_VERSION = "intel-qwen36-linear-preconv-qkv-conv-root-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_OUT_DIR = ROOT / "output/linear-preconv-qkv-conv-root-probe-20260707Tseq90Z"
DEFAULT_SEQ77 = ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z/metrics.json"
DEFAULT_FRONTIER = ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/frontier.json"
PAYLOAD_ROOT = ROOT / "output/r0-boundary-capture-run-20260627T054024Z/remote-output/payloads"
OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/include/intel_qwen36/gpu_q4x8_matvec.hpp", "include/intel_qwen36/gpu_q4x8_matvec.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
]


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

constexpr int kHiddenSize = 2048;
constexpr int kLinearQkvMixedValues = 8192;
constexpr int kLinearConvKernelSize = 4;
constexpr int kLinearConvStateValues =
    (kLinearConvKernelSize - 1) * kLinearQkvMixedValues;
constexpr double kMaxAbsDiffThreshold = 6.0e-3;
constexpr double kRmseThreshold = 7.5e-4;
constexpr double kMinCosine = 0.999;

struct Args {
  std::string model_path;
  std::string payload_dir;
  std::string device_substring = "B390";
  int layer = 12;
  int repeat = 5;
  int trials = 5;
};

struct TimedRun {
  double wall_us = 0.0;
  double q8_quantize_min_us = 0.0;
  double matvec_min_us = 0.0;
  double conv_min_us = 0.0;
  double shell_sum_min_us = 0.0;
};

[[noreturn]] void Die(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool ok, const std::string& message) {
  if (!ok) Die(message);
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

std::vector<std::uint8_t> ReadTensorRaw(std::ifstream& model,
                                        const iq36::GgufTensorInfo& tensor) {
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(tensor.nbytes));
  model.clear();
  model.seekg(static_cast<std::streamoff>(tensor.absolute_offset), std::ios::beg);
  Require(static_cast<bool>(model), "tensor seek failed: " + tensor.name);
  model.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  Require(model.gcount() == static_cast<std::streamsize>(raw.size()),
          "tensor read failed: " + tensor.name);
  return raw;
}

std::vector<float> ReadF32TensorPayload(std::ifstream& model,
                                        const iq36::GgufTensorInfo& tensor,
                                        std::uint64_t values) {
  Require(tensor.type == 0, "expected F32 tensor: " + tensor.name);
  const auto raw = ReadTensorRaw(model, tensor);
  Require(raw.size() == values * sizeof(float), "F32 tensor byte mismatch");
  std::vector<float> out(static_cast<std::size_t>(values), 0.0f);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

std::vector<float> ReadF32File(const std::string& path, std::size_t values) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "payload file open failed");
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size == static_cast<std::streamoff>(values * sizeof(float)),
          "payload file size mismatch");
  input.seekg(0, std::ios::beg);
  std::vector<float> out(values, 0.0f);
  input.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
  Require(static_cast<bool>(input), "payload file read failed");
  return out;
}

std::string FindPayloadFile(const std::string& payload_dir,
                            const std::string& stem,
                            int layer) {
  const std::string prefix = stem + "-" + std::to_string(layer) + "__tok15__ord";
  for (int ord = 0; ord < 2000; ++ord) {
    const std::string path =
        JoinPath(payload_dir, prefix + std::to_string(ord) + ".bin");
    std::ifstream input(path, std::ios::binary);
    if (input.good()) return path;
  }
  Die("payload file not found for " + prefix);
}

double Median(std::vector<double> values) {
  Require(!values.empty(), "median of empty vector");
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double MinValue(const std::vector<double>& values) {
  Require(!values.empty(), "min of empty vector");
  return *std::min_element(values.begin(), values.end());
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
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else if (key == "--layer") args.layer = std::stoi(value("--layer"));
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--trials") args.trials = std::stoi(value("--trials"));
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.payload_dir.empty(), "--payload-dir is required");
  Require(args.layer >= 0 && args.layer < 40, "--layer is out of range");
  Require(args.repeat > 0, "--repeat must be positive");
  Require(args.trials > 0, "--trials must be positive");
  return args;
}

TimedRun HostQ8Run(iq36::GpuQ4X8MatvecRunner& runner,
                   std::uint64_t qkv_q4_handle,
                   std::uint64_t qkv_q6_handle,
                   const iq36::GpuQ8KInputPlanes& q8,
                   std::uint64_t conv_weights_handle,
                   std::uint64_t conv_state_handle,
                   std::uint64_t next_conv_state_handle,
                   bool qkv_is_q4,
                   int repeat,
                   bool readback,
                   std::vector<float>* qkv_out,
                   std::vector<float>* conv_out) {
  const auto begin = std::chrono::steady_clock::now();
  iq36::GpuQ4X8ConvHandoffRun run;
  if (qkv_is_q4) {
    run = runner.RunResidentPackedQ4X8ThenResidentConvState(
        qkv_q4_handle, q8.qs, q8.bsums, q8.d, conv_weights_handle,
        conv_state_handle, kLinearConvKernelSize, repeat,
        iq36::GpuQ4X8KernelVariant::kRowlaneParallel, false,
        next_conv_state_handle, readback, readback);
  } else {
    run = runner.RunResidentRawQ6KThenResidentConvState(
        qkv_q6_handle, q8, conv_weights_handle, conv_state_handle,
        kLinearConvKernelSize, repeat, false, next_conv_state_handle,
        readback, readback);
  }
  const auto end = std::chrono::steady_clock::now();
  TimedRun timed;
  timed.wall_us = std::chrono::duration<double, std::micro>(end - begin).count();
  timed.matvec_min_us = run.timing.matvec.min_us;
  timed.conv_min_us = run.timing.conv_min_us;
  timed.shell_sum_min_us = run.timing.shell_sum_min_us;
  if (readback) {
    *qkv_out = std::move(run.qkv_mixed);
    *conv_out = std::move(run.conv_output_raw);
  }
  return timed;
}

TimedRun DeviceQ8Run(iq36::GpuQ4X8MatvecRunner& runner,
                     std::uint64_t qkv_q4_handle,
                     std::uint64_t qkv_q6_handle,
                     std::uint64_t input_handle,
                     std::uint64_t conv_weights_handle,
                     std::uint64_t conv_state_handle,
                     std::uint64_t next_conv_state_handle,
                     bool qkv_is_q4,
                     int repeat,
                     bool readback,
                     std::vector<float>* qkv_out,
                     std::vector<float>* conv_out) {
  const auto begin = std::chrono::steady_clock::now();
  iq36::GpuQ4X8ConvHandoffRun run;
  if (qkv_is_q4) {
    run = runner.RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState(
        qkv_q4_handle, input_handle, conv_weights_handle, conv_state_handle,
        kLinearConvKernelSize, repeat, iq36::GpuQ4X8KernelVariant::kRowlaneParallel,
        false, next_conv_state_handle, readback, readback);
  } else {
    run = runner.RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState(
        qkv_q6_handle, input_handle, conv_weights_handle, conv_state_handle,
        kLinearConvKernelSize, repeat, false, next_conv_state_handle,
        readback, readback);
  }
  const auto end = std::chrono::steady_clock::now();
  TimedRun timed;
  timed.wall_us = std::chrono::duration<double, std::micro>(end - begin).count();
  timed.q8_quantize_min_us = run.timing.q8_quantize_min_us;
  timed.matvec_min_us = run.timing.matvec.min_us;
  timed.conv_min_us = run.timing.conv_min_us;
  timed.shell_sum_min_us = run.timing.shell_sum_min_us;
  if (readback) {
    *qkv_out = std::move(run.qkv_mixed);
    *conv_out = std::move(run.conv_output_raw);
  }
  return timed;
}

void PrintCompare(const char* name, const iq36::VectorCompareStats& stats) {
  std::cout << "\"" << name << "\":{";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false") << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"cosine\":" << stats.cosine;
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto qkv_name = LayerTensorName(args.layer, "attn_qkv.weight");
    const auto conv_name = LayerTensorName(args.layer, "ssm_conv1d.weight");
    const auto* qkv_tensor = iq36::find_tensor(index, qkv_name);
    const auto* conv_tensor = iq36::find_tensor(index, conv_name);
    Require(qkv_tensor != nullptr, "qkv tensor missing");
    Require(conv_tensor != nullptr, "conv tensor missing");
    Require(qkv_tensor->type == 12 || qkv_tensor->type == 14,
            "qkv tensor must be Q4_K or Q6_K");
    Require(qkv_tensor->dims == std::vector<std::uint64_t>{kHiddenSize, kLinearQkvMixedValues},
            "qkv tensor shape mismatch");
    Require(conv_tensor->type == 0, "conv tensor must be F32");
    Require(conv_tensor->dims == std::vector<std::uint64_t>{kLinearConvKernelSize, kLinearQkvMixedValues},
            "conv tensor shape mismatch");

    const auto payload_path = FindPayloadFile(args.payload_dir, "attn_norm", args.layer);
    const auto attn_norm = ReadF32File(payload_path, kHiddenSize);
    const auto q8 = iq36::QuantizeQ8KInputPlanes(attn_norm);
    const std::uint64_t blocks_per_row = kHiddenSize / 256;
    const bool qkv_is_q4 = qkv_tensor->type == 12;

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "model open failed");
    const auto qkv_raw = ReadTensorRaw(model, *qkv_tensor);
    const auto conv_weights = ReadF32TensorPayload(
        model, *conv_tensor,
        static_cast<std::uint64_t>(kLinearQkvMixedValues * kLinearConvKernelSize));

    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    std::uint64_t qkv_q4_handle = 0;
    std::uint64_t qkv_q6_handle = 0;
    if (qkv_is_q4) {
      const auto packed = iq36::PackQ4Kx8(qkv_raw, kLinearQkvMixedValues,
                                          blocks_per_row);
      qkv_q4_handle =
          runner.UploadPackedQ4X8(packed, kLinearQkvMixedValues, blocks_per_row);
    } else {
      qkv_q6_handle =
          runner.UploadRawQ6K(qkv_raw, kLinearQkvMixedValues, blocks_per_row);
    }
    const auto input_handle = runner.UploadF32Buffer(attn_norm);
    const auto conv_weights_handle =
        runner.UploadConvWeights(conv_weights, kLinearQkvMixedValues,
                                 kLinearConvKernelSize);
    const std::vector<float> conv_state(kLinearConvStateValues, 0.0f);
    const auto conv_state_handle = runner.UploadF32Buffer(conv_state);
    const auto next_conv_state_handle = runner.UploadF32Buffer(conv_state);

    std::vector<float> host_qkv, host_conv, device_qkv, device_conv;
    const auto host_correct = HostQ8Run(
        runner, qkv_q4_handle, qkv_q6_handle, q8, conv_weights_handle,
        conv_state_handle, next_conv_state_handle, qkv_is_q4, 1, true,
        &host_qkv, &host_conv);
    const auto device_correct = DeviceQ8Run(
        runner, qkv_q4_handle, qkv_q6_handle, input_handle, conv_weights_handle,
        conv_state_handle, next_conv_state_handle, qkv_is_q4, 1, true,
        &device_qkv, &device_conv);
    const auto qkv_compare = iq36::compare_vectors(
        host_qkv, device_qkv, kMaxAbsDiffThreshold);
    const auto conv_compare = iq36::compare_vectors(
        host_conv, device_conv, kMaxAbsDiffThreshold);
    const bool correctness_pass =
        qkv_compare.same_size && qkv_compare.finite &&
        qkv_compare.max_abs_diff <= kMaxAbsDiffThreshold &&
        qkv_compare.rmse <= kRmseThreshold && qkv_compare.cosine >= kMinCosine &&
        conv_compare.same_size && conv_compare.finite &&
        conv_compare.max_abs_diff <= kMaxAbsDiffThreshold &&
        conv_compare.rmse <= kRmseThreshold && conv_compare.cosine >= kMinCosine;

    std::vector<double> host_walls, device_walls, host_shells, device_shells;
    std::vector<double> device_q8s;
    host_walls.reserve(static_cast<std::size_t>(args.trials));
    device_walls.reserve(static_cast<std::size_t>(args.trials));
    host_shells.reserve(static_cast<std::size_t>(args.trials));
    device_shells.reserve(static_cast<std::size_t>(args.trials));
    device_q8s.reserve(static_cast<std::size_t>(args.trials));
    std::vector<float> ignored_qkv, ignored_conv;
    for (int i = 0; i < args.trials; ++i) {
      const auto host = HostQ8Run(
          runner, qkv_q4_handle, qkv_q6_handle, q8, conv_weights_handle,
          conv_state_handle, next_conv_state_handle, qkv_is_q4, args.repeat,
          false, &ignored_qkv, &ignored_conv);
      const auto device = DeviceQ8Run(
          runner, qkv_q4_handle, qkv_q6_handle, input_handle,
          conv_weights_handle, conv_state_handle, next_conv_state_handle,
          qkv_is_q4, args.repeat, false, &ignored_qkv, &ignored_conv);
      host_walls.push_back(host.wall_us / static_cast<double>(args.repeat));
      device_walls.push_back(device.wall_us / static_cast<double>(args.repeat));
      host_shells.push_back(host.shell_sum_min_us);
      device_shells.push_back(device.shell_sum_min_us);
      device_q8s.push_back(device.q8_quantize_min_us);
    }

    const double host_wall_min_us = MinValue(host_walls);
    const double device_wall_min_us = MinValue(device_walls);
    const double host_wall_median_us = Median(host_walls);
    const double device_wall_median_us = Median(device_walls);
    const double host_shell_min_us = MinValue(host_shells);
    const double device_shell_min_us = MinValue(device_shells);
    const double wall_speedup =
        device_wall_min_us > 0.0 ? host_wall_min_us / device_wall_min_us : 0.0;
    const double shell_speedup =
        device_shell_min_us > 0.0 ? host_shell_min_us / device_shell_min_us : 0.0;
    const double wall_delta_us = host_wall_min_us - device_wall_min_us;
    const double shell_delta_us = host_shell_min_us - device_shell_min_us;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"target-linear-preconv-qkv-conv-root-probe-v0\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"trials\":" << args.trials << ",";
    std::cout << "\"qkv_tensor_type\":" << qkv_tensor->type << ",";
    std::cout << "\"qkv_tensor_type_name\":\""
              << JsonEscape(iq36::ggml_type_name(qkv_tensor->type)) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"payload\":\"" << JsonEscape(payload_path) << "\",";
    PrintCompare("qkv_compare", qkv_compare);
    std::cout << ",";
    PrintCompare("conv_compare", conv_compare);
    std::cout << ",\"correctness_pass\":" << (correctness_pass ? "true" : "false") << ",";
    std::cout << "\"host_q8_correctness_wall_us\":" << host_correct.wall_us << ",";
    std::cout << "\"device_q8_correctness_wall_us\":" << device_correct.wall_us << ",";
    std::cout << "\"host_q8_wall_min_us_per_call\":" << host_wall_min_us << ",";
    std::cout << "\"device_q8_wall_min_us_per_call\":" << device_wall_min_us << ",";
    std::cout << "\"host_q8_wall_median_us_per_call\":" << host_wall_median_us << ",";
    std::cout << "\"device_q8_wall_median_us_per_call\":" << device_wall_median_us << ",";
    std::cout << "\"host_q8_shell_min_us_per_call\":" << host_shell_min_us << ",";
    std::cout << "\"device_q8_shell_min_us_per_call\":" << device_shell_min_us << ",";
    std::cout << "\"device_q8_quantize_min_us\":" << MinValue(device_q8s) << ",";
    std::cout << "\"wall_speedup_host_over_device\":" << wall_speedup << ",";
    std::cout << "\"shell_speedup_host_over_device\":" << shell_speedup << ",";
    std::cout << "\"wall_delta_us_per_layer\":" << wall_delta_us << ",";
    std::cout << "\"shell_delta_us_per_layer\":" << shell_delta_us << ",";
    std::cout << "\"estimated_40_layer_wall_delta_ms_per_token\":"
              << (wall_delta_us * 40.0 / 1000.0) << ",";
    std::cout << "\"estimated_40_layer_shell_delta_ms_per_token\":"
              << (shell_delta_us * 40.0 / 1000.0);
    std::cout << "}";
    return correctness_pass ? 0 : 2;
  } catch (const std::exception& exc) {
    std::cerr << "probe failed: " << exc.what() << "\n";
    return 1;
  }
}
'''


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _noise_rel(frontier_path: Path) -> float:
  frontier = _load_json(frontier_path)
  no_progress = frontier.get("no_progress") if isinstance(frontier, dict) else None
  noise = no_progress.get("noise") if isinstance(no_progress, dict) else None
  if isinstance(noise, dict):
    return _num(noise.get("rel"))
  value = _num(noise)
  return value if 0.0 < value < 1.0 else value / 100.0


def _profile_delta(seq77: dict[str, Any], key: str) -> float:
  deltas = seq77.get("deltas")
  if not isinstance(deltas, dict):
    derived = seq77.get("derived")
    deltas = derived if isinstance(derived, dict) else None
  if not isinstance(deltas, dict):
    raise SystemExit(f"seq77 metrics lack deltas: {key}")
  return _num(deltas.get(key))


def _source_inputs(paths: list[Path]) -> dict[str, Any]:
  return {
      _display_path(path): {
          "sha256": iq36_local.sha256_file(path),
          "bytes": path.stat().st_size,
      }
      for path in paths
  }


def build_probe_cpp() -> str:
  return PROBE_CPP.replace(
      "@@OPENCL_SOURCE_LITERAL@@",
      json.dumps(OPENCL_SOURCE.read_text(encoding="utf-8")),
  )


def parse_probe(stdout: str) -> dict[str, Any]:
  lines = [line.strip() for line in stdout.splitlines() if line.strip()]
  for line in reversed(lines):
    if line.startswith("{") and line.endswith("}"):
      return json.loads(line)
  raise SystemExit("target probe did not emit JSON")


def run_remote(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
  stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
  remote_dir = f"{args.remote_root.rstrip('/')}/linear-preconv-qkv-conv-root-probe-{stamp}"
  local_cpp = out_dir / "linear_preconv_qkv_conv_root_probe.cpp"
  local_cpp.write_text(build_probe_cpp(), encoding="utf-8")
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  setup = iq36_local.run_target(
      args.host,
      "rm -rf " + shlex.quote(remote_dir) + " && mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "payloads")
      ),
      args.timeout_s,
  )
  transfers = []
  if setup.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(
          iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    transfers.append(
        iq36_local.copy_to(
            args.host,
            local_cpp,
            f"{remote_dir}/tests/linear_preconv_qkv_conv_root_probe.cpp",
            args.timeout_s,
        )
    )
    payload = next(PAYLOAD_ROOT.glob(f"attn_norm-{args.layer}__tok15__ord*.bin"))
    transfers.append(
        iq36_local.copy_to(
            args.host,
            payload,
            f"{remote_dir}/payloads/{payload.name}",
            args.timeout_s,
        )
    )

  compile_result: dict[str, Any] = {"returncode": -1}
  run_result: dict[str, Any] = {"returncode": -1, "stdout": "", "stderr": ""}
  if setup.get("returncode") == 0 and all(t.get("returncode") == 0 for t in transfers):
    executable = f"{remote_dir}/build/linear-preconv-qkv-conv-root-probe"
    compile_cmd = (
        f"source {shlex.quote(args.env_script)} && "
        f"g++ -std=c++20 -O3 -pthread "
        f"-I {shlex.quote(remote_dir + '/include')} "
        f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
        f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
        f"{shlex.quote(remote_dir + '/tests/linear_preconv_qkv_conv_root_probe.cpp')} "
        f"-ldl -o {shlex.quote(executable)}"
      )
    compile_result = iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
    if compile_result.get("returncode") == 0:
      run_cmd = " ".join(
          [
              f"source {shlex.quote(args.env_script)} &&",
              shlex.quote(executable),
              "--model",
              shlex.quote(args.model),
              "--payload-dir",
              shlex.quote(f"{remote_dir}/payloads"),
              "--device-substring",
              shlex.quote(args.device_substring),
              "--layer",
              str(args.layer),
              "--repeat",
              str(args.repeat),
              "--trials",
              str(args.trials),
          ]
      )
      run_result = iq36_local.run_target(args.host, run_cmd, args.timeout_s)

  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  probe = parse_probe(run_result.get("stdout", "")) if run_result.get("returncode") == 0 else {}
  return {
      "remote_dir": remote_dir,
      "local_cpp": _display_path(local_cpp),
      "setup": setup.get("returncode") == 0,
      "transfers": all(t.get("returncode") == 0 for t in transfers),
      "compiled": compile_result.get("returncode") == 0,
      "ran": run_result.get("returncode") == 0,
      "probe": probe,
  }


def compute(args: argparse.Namespace, remote: dict[str, Any]) -> dict[str, Any]:
  seq77 = _load_json(args.seq77_metrics)
  probe = remote.get("probe", {})
  floor_gap_ms = 0.45
  noise = _noise_rel(args.frontier)
  qkv_growth = _profile_delta(seq77, "linear_preconv_qkv_conv_ms_per_token")
  alpha_drop = -_profile_delta(seq77, "linear_preconv_alpha_beta_ms_per_token")
  input_drop = -_profile_delta(seq77, "linear_preconv_input_q8_ms_per_token")
  z_drop = -_profile_delta(seq77, "linear_preconv_z_ms_per_token")
  rebased_qkv_growth = qkv_growth - alpha_drop - input_drop - z_drop
  device_estimated_delta = _num(probe.get("estimated_40_layer_wall_delta_ms_per_token"))
  wall_speedup = _num(probe.get("wall_speedup_host_over_device"))
  shell_speedup = _num(probe.get("shell_speedup_host_over_device"))
  correctness = bool(probe.get("correctness_pass"))
  component_non_growth = correctness and device_estimated_delta >= -floor_gap_ms
  component_floor_covering = correctness and device_estimated_delta >= floor_gap_ms
  qkv_root_is_bundled_envelope = (
      correctness
      and rebased_qkv_growth <= floor_gap_ms
      and alpha_drop + input_drop > qkv_growth * 0.75
  )
  required_checks_passed = (
      bool(remote.get("setup"))
      and bool(remote.get("transfers"))
      and bool(remote.get("compiled"))
      and bool(remote.get("ran"))
      and correctness
      and (component_non_growth or component_floor_covering or qkv_root_is_bundled_envelope)
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "seq77_metrics": {
              "path": _display_path(args.seq77_metrics),
              "sha256": iq36_local.sha256_file(args.seq77_metrics),
          },
          "frontier": {
              "path": _display_path(args.frontier),
              "sha256": iq36_local.sha256_file(args.frontier),
          },
          "source_files": _source_inputs(
              [ROOT / local for local, _ in SOURCE_FILES] + [OPENCL_SOURCE]
          ),
      },
      "remote": remote,
      "seq77_rebased_profile_ms_per_token": {
          "reported_qkv_conv_growth": qkv_growth,
          "embedded_alpha_beta_drop": alpha_drop,
          "embedded_input_q8_drop": input_drop,
          "embedded_z_drop": z_drop,
          "rebased_qkv_conv_growth_after_moved_stages": rebased_qkv_growth,
      },
      "component_probe": probe,
      "derived": {
          "frontier_noise_rel": noise,
          "floor_gap_ms_per_token": floor_gap_ms,
          "component_correct": correctness,
          "component_wall_speedup_host_over_device": wall_speedup,
          "component_shell_speedup_host_over_device": shell_speedup,
          "component_estimated_delta_ms_per_token": device_estimated_delta,
          "component_qkv_conv_non_growth_or_bounded": component_non_growth,
          "component_delta_floor_covering": component_floor_covering,
          "seq77_qkv_growth_is_bundled_envelope": qkv_root_is_bundled_envelope,
          "required_checks_passed": required_checks_passed,
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_probe_allowed": required_checks_passed and component_floor_covering,
          "reason": (
              "The component probe isolates qkv+conv with the resident F32-input "
              "device-Q8 carrier, while seq77 shows the shared-Q8 decode row moved "
              "input_q8 and alpha/beta/z work into qkv_conv accounting."
          ),
          "next_route": (
              "Only wire a new decode row if the isolated qkv+conv component "
              "shows a floor-covering delta; otherwise treat shared-Q8 preconv "
              "as closed until an alpha/beta/z envelope reduction is identified."
          ),
      },
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  iq36_local.write_json(out_dir / "metrics.json", result)
  manifest = {
      "schema_version": f"{SCHEMA_VERSION}-manifest",
      "tool": "tools/intel-qwen36-linear-preconv-qkv-conv-root-gate.py",
      "workstream": WORKSTREAM,
      "artifact": _display_path(out_dir),
      "speedup_claims_allowed": False,
      "required_checks_passed": result["derived"]["required_checks_passed"],
  }
  iq36_local.write_json(out_dir / "manifest.json", manifest)
  d = result["derived"]
  p = result["component_probe"]
  lines = [
      "# Linear Preconv QKV/Conv Root Probe",
      "",
      "This is component evidence, not decode or benchmark evidence.",
      "",
      "## Result",
      "",
      f"- component correct: `{str(d['component_correct']).lower()}`",
      f"- qkv+conv wall speedup host/device: `{d['component_wall_speedup_host_over_device']:.6f}`",
      f"- estimated 40-layer delta: `{d['component_estimated_delta_ms_per_token']:.6f}` ms/token",
      f"- floor-covering component delta: `{str(d['component_delta_floor_covering']).lower()}`",
      f"- seq77 bundled-envelope root: `{str(d['seq77_qkv_growth_is_bundled_envelope']).lower()}`",
      f"- required checks passed: `{str(d['required_checks_passed']).lower()}`",
      "",
      "## Probe",
      "",
      f"- layer: `{p.get('layer')}`",
      f"- qkv type: `{p.get('qkv_tensor_type_name')}`",
      f"- device: `{p.get('device_name')}`",
      f"- remote dir: `{result['remote'].get('remote_dir')}`",
      "",
  ]
  (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--device-substring", default="B390")
  parser.add_argument("--layer", type=int, default=12)
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--trials", type=int, default=5)
  parser.add_argument("--timeout-s", type=int, default=240)
  parser.add_argument("--seq77-metrics", type=Path, default=DEFAULT_SEQ77)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()

  args.out_dir.mkdir(parents=True, exist_ok=True)
  remote = run_remote(args, args.out_dir)
  result = compute(args, remote)
  write_outputs(result, args.out_dir)
  print(json.dumps(result["derived"], indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
