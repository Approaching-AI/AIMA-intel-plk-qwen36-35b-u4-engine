#!/usr/bin/env python3
"""Run the Q4 x8 linear-attention preconv fan-in gate through the GPU shim."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-preconv-fanin-probe-v0"
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
    ("engine/src/gpu_q4_cpu_order_matvec.cpp", "src/gpu_q4_cpu_order_matvec.cpp"),
    ("engine/src/gpu_q4x8_matvec.cpp", "src/gpu_q4x8_matvec.cpp"),
]
PAYLOAD_SPECS = {
    "attn_norm": ("attn_norm.bin", "attn_norm-{layer}__tok15__ord189.bin", 8192),
    "linear_attn_qkv_mixed": ("linear_attn_qkv_mixed.bin", "linear_attn_qkv_mixed-{layer}__tok15__ord190.bin", 32768),
    "alpha": ("alpha.bin", "alpha-{layer}__tok15__ord198.bin", 128),
    "a_softplus": ("a_softplus.bin", "a_softplus-{layer}__tok15__ord199.bin", 128),
    "gate": ("gate.bin", "gate-{layer}__tok15__ord200.bin", 128),
    "beta": ("beta.bin", "beta-{layer}__tok15__ord201.bin", 128),
    "beta_sigmoid": ("beta_sigmoid.bin", "beta_sigmoid-{layer}__tok15__ord202.bin", 128),
    "z": ("z.bin", "z-{layer}__tok15__ord205.bin", 16384),
}


PROBE_CPP = r'''
#include "intel_qwen36/gguf_loader.hpp"
#include "intel_qwen36/gpu_q4x8_matvec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

struct Args {
  std::string model_path;
  std::string payload_dir;
  int layer = 5;
  int repeat = 7;
  std::string device_substring = "B390";
};

struct Projection {
  std::string name;
  std::string suffix;
  std::uint64_t expected_rows = 0;
  std::vector<float> values;
  iq36::GpuQ4X8MatvecTiming timing;
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
  Require(args.layer >= 0, "--layer must be nonnegative");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
}

bool ComparePassed(const iq36::VectorCompareStats& stats) {
  return stats.same_size &&
         stats.finite &&
         stats.mismatch_count == 0 &&
         stats.max_abs_diff <= 5e-3 &&
         stats.rmse <= 1e-3 &&
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

std::vector<float> SoftplusVector(const std::vector<float>& input) {
  std::vector<float> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float x = input[i];
    out[i] = x > 20.0f ? x : std::log1p(std::exp(x));
  }
  return out;
}

std::vector<float> SigmoidVector(const std::vector<float>& input) {
  std::vector<float> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = iq36::sigmoid_scalar(input[i]);
  }
  return out;
}

std::vector<float> MultiplyVectors(const std::vector<float>& lhs,
                                   const std::vector<float>& rhs) {
  Require(lhs.size() == rhs.size(), "multiply vector size mismatch");
  std::vector<float> out(lhs.size());
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    out[i] = lhs[i] * rhs[i];
  }
  return out;
}

Projection RunProjection(const Args& args,
                         const iq36::GgufModelIndex& index,
                         std::ifstream& model,
                         iq36::GpuQ4X8MatvecRunner& runner,
                         const iq36::GpuQ8KInputPlanes& q8,
                         const Projection& spec) {
  Projection out = spec;
  const std::string tensor_name =
      std::string("blk.") + std::to_string(args.layer) + "." + spec.suffix;
  const auto* tensor = iq36::find_tensor(index, tensor_name);
  Require(tensor != nullptr, "projection tensor missing: " + tensor_name);
  Require(tensor->type == 12, "projection tensor is not Q4_K: " + tensor_name);
  Require(tensor->dims == std::vector<std::uint64_t>{2048, spec.expected_rows},
          "projection tensor dims mismatch: " + tensor_name);
  const auto raw = ReadTensorBytes(model, *tensor);
  const auto packed = iq36::PackQ4Kx8(raw, spec.expected_rows, 8);
  const auto run = runner.Run(packed, q8.qs, q8.bsums, q8.d, spec.expected_rows, 8,
                              args.repeat, iq36::GpuQ4X8KernelVariant::kRowlaneParallel);
  out.values = run.output;
  out.timing = run.timing;
  return out;
}

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const auto attn_norm = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "attn_norm.bin"));
    const auto oracle_qkv = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "linear_attn_qkv_mixed.bin"));
    const auto oracle_alpha = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "alpha.bin"));
    const auto oracle_a_softplus = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "a_softplus.bin"));
    const auto oracle_gate = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "gate.bin"));
    const auto oracle_beta = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "beta.bin"));
    const auto oracle_beta_sigmoid = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "beta_sigmoid.bin"));
    const auto oracle_z = iq36::read_f32_vector_file(JoinPath(args.payload_dir, "z.bin"));
    Require(attn_norm.size() == 2048, "attn_norm size mismatch");

    const auto cpu = iq36::run_qwen36_linear_attention_preconv_core(
        args.model_path, index, args.layer, attn_norm);
    const auto q8 = iq36::QuantizeQ8KInputPlanes(attn_norm);
    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");

    std::vector<Projection> specs = {
        {"linear_attn_qkv_mixed", "attn_qkv.weight", 8192},
        {"alpha", "ssm_alpha.weight", 32},
        {"beta", "ssm_beta.weight", 32},
        {"z", "attn_gate.weight", 4096},
    };
    std::map<std::string, Projection> gpu_projection;
    for (const auto& spec : specs) {
      auto projected = RunProjection(args, index, model, runner, q8, spec);
      gpu_projection[projected.name] = std::move(projected);
    }

    const auto ssm_dt = iq36::decode_tensor_row(
        args.model_path, index,
        std::string("blk.") + std::to_string(args.layer) + ".ssm_dt.bias", 0);
    const auto ssm_a = iq36::decode_tensor_row(
        args.model_path, index,
        std::string("blk.") + std::to_string(args.layer) + ".ssm_a", 0);
    const auto gpu_a_softplus = SoftplusVector(iq36::add_vectors(gpu_projection["alpha"].values, ssm_dt));
    const auto gpu_gate = MultiplyVectors(gpu_a_softplus, ssm_a);
    const auto gpu_beta_sigmoid = SigmoidVector(gpu_projection["beta"].values);

    struct CompareRow {
      std::string name;
      iq36::VectorCompareStats cpu_vs_oracle;
      iq36::VectorCompareStats gpu_vs_cpu;
      iq36::VectorCompareStats gpu_vs_oracle;
    };
    std::vector<CompareRow> rows = {
        {"linear_attn_qkv_mixed",
         iq36::compare_vectors(cpu.qkv_mixed, oracle_qkv, 5e-3),
         iq36::compare_vectors(gpu_projection["linear_attn_qkv_mixed"].values, cpu.qkv_mixed, 5e-3),
         iq36::compare_vectors(gpu_projection["linear_attn_qkv_mixed"].values, oracle_qkv, 5e-3)},
        {"alpha",
         iq36::compare_vectors(cpu.alpha, oracle_alpha, 5e-3),
         iq36::compare_vectors(gpu_projection["alpha"].values, cpu.alpha, 5e-3),
         iq36::compare_vectors(gpu_projection["alpha"].values, oracle_alpha, 5e-3)},
        {"a_softplus",
         iq36::compare_vectors(cpu.alpha_softplus, oracle_a_softplus, 5e-3),
         iq36::compare_vectors(gpu_a_softplus, cpu.alpha_softplus, 5e-3),
         iq36::compare_vectors(gpu_a_softplus, oracle_a_softplus, 5e-3)},
        {"gate",
         iq36::compare_vectors(cpu.gate, oracle_gate, 5e-3),
         iq36::compare_vectors(gpu_gate, cpu.gate, 5e-3),
         iq36::compare_vectors(gpu_gate, oracle_gate, 5e-3)},
        {"beta",
         iq36::compare_vectors(cpu.beta, oracle_beta, 5e-3),
         iq36::compare_vectors(gpu_projection["beta"].values, cpu.beta, 5e-3),
         iq36::compare_vectors(gpu_projection["beta"].values, oracle_beta, 5e-3)},
        {"beta_sigmoid",
         iq36::compare_vectors(cpu.beta_sigmoid, oracle_beta_sigmoid, 5e-3),
         iq36::compare_vectors(gpu_beta_sigmoid, cpu.beta_sigmoid, 5e-3),
         iq36::compare_vectors(gpu_beta_sigmoid, oracle_beta_sigmoid, 5e-3)},
        {"z",
         iq36::compare_vectors(cpu.z, oracle_z, 5e-3),
         iq36::compare_vectors(gpu_projection["z"].values, cpu.z, 5e-3),
         iq36::compare_vectors(gpu_projection["z"].values, oracle_z, 5e-3)},
    };
    bool comparisons_passed = true;
    for (const auto& row : rows) {
      comparisons_passed = comparisons_passed &&
          ComparePassed(row.cpu_vs_oracle) &&
          ComparePassed(row.gpu_vs_cpu) &&
          ComparePassed(row.gpu_vs_oracle);
    }
    bool timings_positive = true;
    for (const auto& item : gpu_projection) {
      timings_positive = timings_positive && item.second.timing.min_us > 0.0;
    }
    const bool checks_passed =
        load_map.ready &&
        runner.device_name().find(args.device_substring) != std::string::npos &&
        comparisons_passed &&
        timings_positive;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-preconv-fanin-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runner.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"projection_timings\":{";
    bool first = true;
    for (const auto& item : gpu_projection) {
      if (!first) std::cout << ",";
      first = false;
      const auto& timing = item.second.timing;
      std::cout << "\"" << JsonEscape(item.first) << "\":{";
      std::cout << "\"gpu_kernel_min_us\":" << timing.min_us << ",";
      std::cout << "\"gpu_kernel_mean_us\":" << timing.mean_us << ",";
      std::cout << "\"gpu_effective_packed_gb_s\":" << timing.effective_packed_gb_s << ",";
      std::cout << "\"global_work_items\":" << timing.global_work_items << ",";
      std::cout << "\"rows_per_work_item\":" << timing.rows_per_work_item;
      std::cout << "}";
    }
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
    std::cout << "},\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"comparisons_passed\":" << (comparisons_passed ? "true" : "false") << ",";
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
      payload_name = pattern.format(layer=layer)
      wildcard = payload_name.split("__tok15__ord", 1)[0] + "__tok15__ord*.bin"
      matches = sorted(PAYLOAD_ROOT.glob(wildcard))
      if len(matches) != 1:
        raise SystemExit(f"preconv fan-in payload missing: {path}")
      path = matches[0].resolve()
    if path.stat().st_size != size_bytes:
      raise SystemExit(f"preconv fan-in payload size mismatch: {path}")
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
  timings = probe.get("projection_timings", {}) if isinstance(probe, dict) else {}
  lines = [
      "# GPU Q4-X8 Preconv Fan-In Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      "",
      "| projection | min us | packed GB/s | GPU vs oracle max abs | GPU vs oracle RMSE |",
      "|---|---:|---:|---:|---:|",
  ]
  for name in ("linear_attn_qkv_mixed", "alpha", "beta", "z"):
    timing = timings.get(name, {}) if isinstance(timings, dict) else {}
    cmp = comparisons.get(name, {}).get("gpu_vs_oracle", {}) if isinstance(comparisons, dict) else {}
    lines.append(
        f"| {name} | {timing.get('gpu_kernel_min_us')} | "
        f"{timing.get('gpu_effective_packed_gb_s')} | "
        f"{cmp.get('max_abs_diff')} | {cmp.get('rmse')} |"
    )
  lines += [
      "",
      "Derived checks also cover `a_softplus`, `gate`, and `beta_sigmoid`.",
      "This is component evidence only; it does not prove decode or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-preconv-fanin-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  payloads = resolve_payloads(args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_q4x8_preconv_fanin_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-preconv-fanin-probe-{stamp}"
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
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_preconv_fanin_probe.cpp", args.timeout_s))
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
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_preconv_fanin_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-preconv-fanin-probe')}"
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
      f"{remote_dir}/build/iq36-gpu-q4x8-preconv-fanin-probe",
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
      {"name": "comparisons_passed", "pass": bool(probe and nested_bool(probe, "checks", "comparisons_passed"))},
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
      "tool": "tools/intel-qwen36-gpu-q4x8-preconv-fanin-probe.py",
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
  timings = aggregate.get("projection_timings", {}) if isinstance(aggregate.get("projection_timings"), dict) else {}
  comparisons = aggregate.get("comparisons", {}) if isinstance(aggregate.get("comparisons"), dict) else {}
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_q4x8_preconv_fanin_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("qkv_kernel_min_us", nested_number(timings, "linear_attn_qkv_mixed", "gpu_kernel_min_us")),
          ("alpha_kernel_min_us", nested_number(timings, "alpha", "gpu_kernel_min_us")),
          ("beta_kernel_min_us", nested_number(timings, "beta", "gpu_kernel_min_us")),
          ("z_kernel_min_us", nested_number(timings, "z", "gpu_kernel_min_us")),
          ("qkv_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "linear_attn_qkv_mixed", "gpu_vs_oracle", "max_abs_diff")),
          ("z_gpu_vs_oracle_max_abs_diff", nested_number(comparisons, "z", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
