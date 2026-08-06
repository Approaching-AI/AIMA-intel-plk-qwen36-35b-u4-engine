#!/usr/bin/env python3
"""Run the first Q4 attention/QKV consumer gate through the GPU q4x8 shim."""

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
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-qkv-consumer-probe-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
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
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const char* kQ4X8OpenClSource = @@OPENCL_SOURCE_LITERAL@@;

struct Args {
  std::string model_path;
  std::string input_path;
  std::string oracle_path;
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
    else if (key == "--input") args.input_path = value("--input");
    else if (key == "--oracle") args.oracle_path = value("--oracle");
    else if (key == "--layer") args.layer = std::stoi(value("--layer"));
    else if (key == "--repeat") args.repeat = std::stoi(value("--repeat"));
    else if (key == "--device-substring") args.device_substring = value("--device-substring");
    else Die("unknown argument: " + key);
  }
  Require(!args.model_path.empty(), "--model is required");
  Require(!args.input_path.empty(), "--input is required");
  Require(!args.oracle_path.empty(), "--oracle is required");
  Require(args.layer >= 0, "--layer must be nonnegative");
  Require(args.repeat > 0, "--repeat must be positive");
  return args;
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

void WriteCompare(const iq36::VectorCompareStats& stats) {
  std::cout << "{";
  std::cout << "\"compared_value_count\":" << stats.compared_value_count << ",";
  std::cout << "\"cosine\":" << stats.cosine << ",";
  std::cout << "\"finite\":" << (stats.finite ? "true" : "false") << ",";
  std::cout << "\"finite_pair_count\":" << stats.finite_pair_count << ",";
  std::cout << "\"lhs_l2\":" << stats.lhs_l2 << ",";
  std::cout << "\"lhs_value_count\":" << stats.lhs_value_count << ",";
  std::cout << "\"max_abs_diff\":" << stats.max_abs_diff << ",";
  std::cout << "\"mean_abs_diff\":" << stats.mean_abs_diff << ",";
  std::cout << "\"mismatch_count\":" << stats.mismatch_count << ",";
  std::cout << "\"rhs_l2\":" << stats.rhs_l2 << ",";
  std::cout << "\"rhs_value_count\":" << stats.rhs_value_count << ",";
  std::cout << "\"rmse\":" << stats.rmse << ",";
  std::cout << "\"same_size\":" << (stats.same_size ? "true" : "false");
  std::cout << "}";
}

int main(int argc, char** argv) {
  try {
    constexpr double kMismatchThreshold = 5e-3;
    constexpr double kMaxAbsDiffThreshold = 5e-3;
    constexpr double kRmseThreshold = 1e-3;
    constexpr double kMinCosine = 0.99999;

    const Args args = ParseArgs(argc, argv);
    const auto index = iq36::parse_gguf_model_index(args.model_path);
    const auto load_map = iq36::validate_qwen36_load_map(index);
    const std::string tensor_name =
        std::string("blk.") + std::to_string(args.layer) + ".attn_qkv.weight";
    const auto* tensor = iq36::find_tensor(index, tensor_name);
    Require(tensor != nullptr, "QKV tensor not found");
    const bool tensor_shape_ok =
        tensor->type == 12 && tensor->dims == std::vector<std::uint64_t>{2048, 8192};

    const auto input = iq36::read_f32_vector_file(args.input_path);
    const auto oracle = iq36::read_f32_vector_file(args.oracle_path);
    const auto cpu_native = iq36::matvec_tensor(args.model_path, index, tensor_name, input);

    std::ifstream model(args.model_path, std::ios::binary);
    Require(static_cast<bool>(model), "failed to open model");
    const auto raw = ReadTensorBytes(model, *tensor);
    const std::uint64_t cols = tensor->dims[0];
    std::uint64_t rows = 1;
    for (std::size_t i = 1; i < tensor->dims.size(); ++i) {
      rows *= tensor->dims[i];
    }
    const std::uint64_t blocks_per_row = cols / 256;
    const auto packed = iq36::PackQ4Kx8(raw, rows, blocks_per_row);
    const auto q8 = iq36::QuantizeQ8KInputPlanes(input);
    iq36::GpuQ4X8MatvecRunner runner(args.device_substring, kQ4X8OpenClSource);
    const auto gpu = runner.Run(packed, q8.qs, q8.bsums, q8.d, rows, blocks_per_row,
                                args.repeat, iq36::GpuQ4X8KernelVariant::kRowlaneParallel);

    const auto cpu_vs_oracle = iq36::compare_vectors(cpu_native, oracle, kMismatchThreshold);
    const auto gpu_vs_cpu = iq36::compare_vectors(gpu.output, cpu_native, kMismatchThreshold);
    const auto gpu_vs_oracle = iq36::compare_vectors(gpu.output, oracle, kMismatchThreshold);
    const bool cpu_matches_oracle =
        ComparePassed(cpu_vs_oracle, kMaxAbsDiffThreshold, kRmseThreshold, kMinCosine);
    const bool gpu_matches_cpu =
        ComparePassed(gpu_vs_cpu, kMaxAbsDiffThreshold, kRmseThreshold, kMinCosine);
    const bool gpu_matches_oracle =
        ComparePassed(gpu_vs_oracle, kMaxAbsDiffThreshold, kRmseThreshold, kMinCosine);
    const bool checks_passed =
        load_map.ready &&
        tensor_shape_ok &&
        input.size() == 2048 &&
        oracle.size() == 8192 &&
        cpu_native.size() == 8192 &&
        gpu.output.size() == 8192 &&
        runner.device_name().find(args.device_substring) != std::string::npos &&
        gpu.timing.min_us > 0.0 &&
        cpu_matches_oracle &&
        gpu_matches_cpu &&
        gpu_matches_oracle;

    std::cout << std::setprecision(10);
    std::cout << "{";
    std::cout << "\"schema_version\":\"intel-qwen36-gpu-q4x8-qkv-consumer-probe-v0\",";
    std::cout << "\"model_path\":\"" << JsonEscape(args.model_path) << "\",";
    std::cout << "\"layer\":" << args.layer << ",";
    std::cout << "\"tensor_name\":\"" << JsonEscape(tensor->name) << "\",";
    std::cout << "\"tensor_type\":\"" << iq36::ggml_type_name(tensor->type) << "\",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"cols\":" << cols << ",";
    std::cout << "\"rows\":" << rows << ",";
    std::cout << "\"blocks_per_row\":" << blocks_per_row << ",";
    std::cout << "\"raw_bytes\":" << raw.size() << ",";
    std::cout << "\"packed_q4k_x8_bytes\":" << packed.size() << ",";
    std::cout << "\"input_value_count\":" << input.size() << ",";
    std::cout << "\"oracle_value_count\":" << oracle.size() << ",";
    std::cout << "\"cpu_value_count\":" << cpu_native.size() << ",";
    std::cout << "\"gpu_value_count\":" << gpu.output.size() << ",";
    std::cout << "\"platform_name\":\"" << JsonEscape(runner.platform_name()) << "\",";
    std::cout << "\"device_name\":\"" << JsonEscape(runner.device_name()) << "\",";
    std::cout << "\"program_build_ms\":" << runner.program_build_ms() << ",";
    std::cout << "\"build_log\":\"" << JsonEscape(runner.build_log()) << "\",";
    std::cout << "\"repeat\":" << args.repeat << ",";
    std::cout << "\"gpu_kernel_variant\":\"rowlane_parallel\",";
    std::cout << "\"gpu_kernel_name\":\"q4k_x8_matvec_rowlane\",";
    std::cout << "\"gpu_kernel_min_us\":" << gpu.timing.min_us << ",";
    std::cout << "\"gpu_kernel_mean_us\":" << gpu.timing.mean_us << ",";
    std::cout << "\"gpu_effective_packed_gb_s\":" << gpu.timing.effective_packed_gb_s << ",";
    std::cout << "\"comparisons\":{";
    std::cout << "\"cpu_vs_oracle\":";
    WriteCompare(cpu_vs_oracle);
    std::cout << ",\"gpu_vs_cpu\":";
    WriteCompare(gpu_vs_cpu);
    std::cout << ",\"gpu_vs_oracle\":";
    WriteCompare(gpu_vs_oracle);
    std::cout << "},";
    std::cout << "\"checks\":{";
    std::cout << "\"load_map_ready\":" << (load_map.ready ? "true" : "false") << ",";
    std::cout << "\"arc_device_selected\":" << (runner.device_name().find(args.device_substring) != std::string::npos ? "true" : "false") << ",";
    std::cout << "\"q4_tensor_selected\":" << (tensor->type == 12 ? "true" : "false") << ",";
    std::cout << "\"tensor_shape_ok\":" << (tensor_shape_ok ? "true" : "false") << ",";
    std::cout << "\"cpu_matches_oracle\":" << (cpu_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"gpu_matches_cpu\":" << (gpu_matches_cpu ? "true" : "false") << ",";
    std::cout << "\"gpu_matches_oracle\":" << (gpu_matches_oracle ? "true" : "false") << ",";
    std::cout << "\"gpu_event_timing_positive\":" << (gpu.timing.min_us > 0.0 ? "true" : "false") << ",";
    std::cout << "\"speedup_claims_allowed\":false";
    std::cout << "},";
    std::cout << "\"required_checks_passed\":" << (checks_passed ? "true" : "false");
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: row must be an object")
      rows.append(row)
  return rows


def resolve_payload(bundle: Path, relative: str, expected_size: int) -> Path:
  path = (bundle / relative).resolve()
  if not path.exists():
    raise SystemExit(f"oracle payload missing: {path}")
  if path.stat().st_size != expected_size:
    raise SystemExit(f"oracle payload size mismatch: {path}")
  return path


def resolve_qkv_reference(bundle: Path, layer: int) -> dict[str, Any]:
  bundle = bundle.resolve()
  inputs = load_jsonl(bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(bundle / "boundary-references/outputs.jsonl")
  qkv_input = next(
      (
          row for row in inputs
          if row.get("boundary_type") == "qkv_projection"
          and row.get("layer") == layer
          and row.get("tensor_kind") == "input"
      ),
      None,
  )
  qkv_output = next(
      (
          row for row in outputs
          if row.get("boundary_type") == "qkv_projection"
          and row.get("layer") == layer
          and row.get("tensor_kind") == "output"
      ),
      None,
  )
  if not isinstance(qkv_input, dict) or not isinstance(qkv_output, dict):
    raise SystemExit(f"oracle bundle missing layer {layer} qkv rows")
  input_shape = qkv_input.get("shape_metadata", {})
  output_shape = qkv_output.get("shape_metadata", {})
  if input_shape.get("nbytes") != 8192 or input_shape.get("ne") != [2048, 1, 1, 1]:
    raise SystemExit("oracle qkv input shape mismatch")
  if output_shape.get("nbytes") != 32768 or output_shape.get("ne") != [8192, 1, 1, 1]:
    raise SystemExit("oracle qkv output shape mismatch")
  input_rel = qkv_input.get("reference_input_tensor_path")
  output_rel = qkv_output.get("reference_output_tensor_path")
  if not isinstance(input_rel, str) or not isinstance(output_rel, str):
    raise SystemExit("oracle qkv payload paths missing")
  input_path = resolve_payload(bundle, input_rel, 8192)
  output_path = resolve_payload(bundle, output_rel, 32768)
  return {
      "input_path": input_path,
      "output_path": output_path,
      "input_payload_path": str(input_path.relative_to(ROOT)),
      "output_payload_path": str(output_path.relative_to(ROOT)),
      "input_payload_sha256": iq36_local.sha256_file(input_path),
      "output_payload_sha256": iq36_local.sha256_file(output_path),
      "oracle_bundle": str(bundle.relative_to(ROOT)),
      "policy_id": qkv_output.get("policy_id"),
      "source_prompt_case_id": qkv_input.get("source_prompt_case_id"),
      "source_token_position": qkv_input.get("source_token_position"),
  }


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
  lines = [
      "# GPU Q4-X8 QKV Consumer Probe",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- oracle bundle: `{payload.get('oracle_bundle')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- tensor: `{probe.get('tensor_name')}`",
      f"- engine shim: `{payload.get('engine_shim_source')}`",
      f"- OpenCL source: `{payload.get('opencl_source')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- rowlane min us: `{probe.get('gpu_kernel_min_us')}`",
      f"- rowlane packed GB/s: `{probe.get('gpu_effective_packed_gb_s')}`",
      "",
      "| comparison | max abs | rmse | cosine | mismatch |",
      "|---|---:|---:|---:|---:|",
  ]
  for name in ("cpu_vs_oracle", "gpu_vs_cpu", "gpu_vs_oracle"):
    cmp = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
    lines.append(
        f"| {name} | {cmp.get('max_abs_diff')} | {cmp.get('rmse')} | "
        f"{cmp.get('cosine')} | {cmp.get('mismatch_count')} |"
    )
  lines += [
      "",
      "Decision: this is a first Q4 attention/QKV shim consumer gate. It does",
      "not prove decode, token, or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if args.repeat <= 0:
    raise SystemExit("--repeat must be positive")
  stamp = utc_stamp()
  created_at = iso_now()
  out_dir = args.out_dir or ROOT / f"output/gpu-q4x8-qkv-consumer-probe-{stamp}"
  out_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = out_dir / "raw"
  raw_dir.mkdir()
  ref = resolve_qkv_reference(args.oracle_bundle, args.layer)
  opencl_source = OPENCL_SOURCE.read_text(encoding="utf-8")
  opencl_source_hash = iq36_local.sha256_file(OPENCL_SOURCE)
  local_cpp = out_dir / "gpu_q4x8_qkv_consumer_probe.cpp"
  local_cpp.write_text(
      PROBE_CPP.replace("@@OPENCL_SOURCE_LITERAL@@", cpp_raw_string_literal(opencl_source)),
      encoding="utf-8",
  )

  remote_dir = f"{args.remote_root.rstrip('/')}/gpu-q4x8-qkv-consumer-probe-{stamp}"
  setup = iq36_local.run_target(
      args.host,
      "rm -rf "
      + shlex.quote(remote_dir)
      + " && mkdir -p "
      + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  input_transfer = {"returncode": 1, "stderr": "stage failed", "stdout": ""}
  output_transfer = {"returncode": 1, "stderr": "stage failed", "stdout": ""}
  remote_input = f"{remote_dir}/oracle/{ref['input_path'].name}"
  remote_output = f"{remote_dir}/oracle/{ref['output_path'].name}"
  if setup.get("returncode") == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(iq36_local.copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    transfers.append(iq36_local.copy_to(args.host, local_cpp, f"{remote_dir}/tests/gpu_q4x8_qkv_consumer_probe.cpp", args.timeout_s))
    input_transfer = iq36_local.copy_to(args.host, ref["input_path"], remote_input, args.timeout_s)
    output_transfer = iq36_local.copy_to(args.host, ref["output_path"], remote_output, args.timeout_s)

  compile_cmd = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      (
          "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
          f"-I {shlex.quote(remote_dir + '/include')} "
          f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
          f"{shlex.quote(remote_dir + '/src/gpu_q4x8_matvec.cpp')} "
          f"{shlex.quote(remote_dir + '/tests/gpu_q4x8_qkv_consumer_probe.cpp')} "
          "-ldl -pthread "
          f"-o {shlex.quote(remote_dir + '/build/iq36-gpu-q4x8-qkv-consumer-probe')}"
      ),
  ])
  stage_ok = (
      setup.get("returncode") == 0
      and transfers
      and all(item.get("returncode") == 0 for item in transfers)
      and input_transfer.get("returncode") == 0
      and output_transfer.get("returncode") == 0
  )
  compile_result = (
      iq36_local.run_target(args.host, compile_cmd, args.timeout_s)
      if stage_ok
      else {"cmd": ["stage"], "returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  run_argv = [
      f"{remote_dir}/build/iq36-gpu-q4x8-qkv-consumer-probe",
      "--model", args.model,
      "--input", remote_input,
      "--oracle", remote_output,
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
  iq36_local.write_json(raw_dir / "input-transfer.json", input_transfer)
  iq36_local.write_json(raw_dir / "output-transfer.json", output_transfer)
  iq36_local.write_json(raw_dir / "compile.json", compile_result)
  iq36_local.write_json(raw_dir / "run.json", run_result)
  if probe is not None:
    iq36_local.write_json(out_dir / "probe-result.json", probe)

  checks = [
      {"name": "remote_dir_created", "pass": setup.get("returncode") == 0},
      {"name": "source_files_transferred", "pass": bool(transfers) and all(item.get("returncode") == 0 for item in transfers)},
      {"name": "oracle_input_transferred", "pass": input_transfer.get("returncode") == 0},
      {"name": "oracle_output_transferred", "pass": output_transfer.get("returncode") == 0},
      {"name": "probe_compiled", "pass": compile_result.get("returncode") == 0},
      {"name": "probe_stdout_json_parsed", "pass": isinstance(probe, dict)},
      {"name": "probe_process_succeeded", "pass": run_result.get("returncode") == 0},
      {"name": "arc_b390_selected", "pass": bool(probe and "B390" in str(probe.get("device_name", "")))},
      {"name": "q4_tensor_selected", "pass": bool(probe and probe.get("tensor_type") == "Q4_K")},
      {"name": "cpu_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "cpu_matches_oracle"))},
      {"name": "gpu_matches_cpu", "pass": bool(probe and nested_bool(probe, "checks", "gpu_matches_cpu"))},
      {"name": "gpu_matches_oracle", "pass": bool(probe and nested_bool(probe, "checks", "gpu_matches_oracle"))},
      {"name": "gpu_event_timing_positive", "pass": bool((nested_number(probe or {}, "gpu_kernel_min_us") or 0.0) > 0.0)},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  required_checks_passed = all(item["pass"] for item in checks)
  payload = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "host": args.host,
      "remote_dir": remote_dir,
      "model": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "layer": args.layer,
      "repeat": args.repeat,
      "input_payload_path": ref["input_payload_path"],
      "input_payload_sha256": ref["input_payload_sha256"],
      "output_payload_path": ref["output_payload_path"],
      "output_payload_sha256": ref["output_payload_sha256"],
      "policy_id": ref.get("policy_id"),
      "source_prompt_case_id": ref.get("source_prompt_case_id"),
      "source_token_position": ref.get("source_token_position"),
      "engine_shim_header": "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp",
      "engine_shim_source": "engine/src/gpu_q4x8_matvec.cpp",
      "opencl_source": str(OPENCL_SOURCE.relative_to(ROOT)),
      "opencl_source_sha256": opencl_source_hash,
      "probe": probe,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "recommendation": "use this Q4 QKV consumer gate before decode-loop GPU scheduling",
  }
  manifest = {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": "tools/intel-qwen36-gpu-q4x8-qkv-consumer-probe.py",
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
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "gpu_q4x8_qkv_consumer_probe",
      [
          ("required_checks_passed", required_checks_passed),
          ("gpu_kernel_min_us", aggregate.get("gpu_kernel_min_us")),
          ("gpu_effective_packed_gb_s", aggregate.get("gpu_effective_packed_gb_s")),
          ("cpu_vs_oracle_max_abs_diff", nested_number(aggregate, "comparisons", "cpu_vs_oracle", "max_abs_diff")),
          ("gpu_vs_cpu_max_abs_diff", nested_number(aggregate, "comparisons", "gpu_vs_cpu", "max_abs_diff")),
          ("gpu_vs_oracle_max_abs_diff", nested_number(aggregate, "comparisons", "gpu_vs_oracle", "max_abs_diff")),
      ],
  )
  write_summary(out_dir / "summary.md", payload)
  print(out_dir)
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
