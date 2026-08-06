#!/usr/bin/env python3
"""Run a Q4 x8 z-projection-only probe.

This reuses the preconv fan-in staging/build flow but narrows the generated C++
to the `z` projection (`attn_gate.weight`). It is useful for layers whose QKV
tensor is Q6_K, where the full Q4 fan-in probe cannot run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


sys.dont_write_bytecode = True

BASE_TOOL = Path(__file__).with_name("intel-qwen36-gpu-q4x8-preconv-fanin-probe.py")
SCHEMA_VERSION = "intel-qwen36-gpu-q4x8-z-projection-probe-v0"


def load_base_tool() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_q4x8_preconv_fanin", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise SystemExit(f"failed to load preconv fan-in tool: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base_tool()
BASE_PARSE_ARGS = BASE.parse_args


def strip_variant(argv: list[str]) -> tuple[str, list[str]]:
  variant = "rowlane_parallel"
  result = [argv[0]]
  index = 1
  while index < len(argv):
    item = argv[index]
    if item == "--variant":
      if index + 1 >= len(argv):
        raise SystemExit("--variant requires a value")
      variant = argv[index + 1]
      index += 2
      continue
    if item.startswith("--variant="):
      variant = item.split("=", 1)[1]
      index += 1
      continue
    result.append(item)
    index += 1
  if variant not in {"rowlane_parallel", "group8_serial"}:
    raise SystemExit("--variant must be rowlane_parallel or group8_serial")
  return variant, result


def z_only_cpp(variant: str = "rowlane_parallel") -> str:
  kernel_variant = {
      "rowlane_parallel": "kRowlaneParallel",
      "group8_serial": "kGroup8Serial",
  }[variant]
  cpp = BASE.PROBE_CPP.replace(BASE.SCHEMA_VERSION, SCHEMA_VERSION)
  cpp = BASE.PROBE_CPP.replace(
      '''    std::vector<Projection> specs = {
        {"linear_attn_qkv_mixed", "attn_qkv.weight", 8192},
        {"alpha", "ssm_alpha.weight", 32},
        {"beta", "ssm_beta.weight", 32},
        {"z", "attn_gate.weight", 4096},
    };
''',
      '''    std::vector<Projection> specs = {
        {"z", "attn_gate.weight", 4096},
    };
''',
  ).replace(BASE.SCHEMA_VERSION, SCHEMA_VERSION)
  cpp = cpp.replace(
      '''    const auto ssm_dt = iq36::decode_tensor_row(
        args.model_path, index,
        std::string("blk.") + std::to_string(args.layer) + ".ssm_dt.bias", 0);
    const auto ssm_a = iq36::decode_tensor_row(
        args.model_path, index,
        std::string("blk.") + std::to_string(args.layer) + ".ssm_a", 0);
    const auto gpu_a_softplus = SoftplusVector(iq36::add_vectors(gpu_projection["alpha"].values, ssm_dt));
    const auto gpu_gate = MultiplyVectors(gpu_a_softplus, ssm_a);
    const auto gpu_beta_sigmoid = SigmoidVector(gpu_projection["beta"].values);

''',
      "",
  )
  cpp = cpp.replace(
      "iq36::GpuQ4X8KernelVariant::kRowlaneParallel",
      f"iq36::GpuQ4X8KernelVariant::{kernel_variant}",
  )
  cpp = cpp.replace(
      '''    std::cout << "\\"repeat\\":" << args.repeat << ",";
''',
      f'''    std::cout << "\\"repeat\\":" << args.repeat << ",";
    std::cout << "\\"kernel_variant\\":\\"{variant}\\",";
''',
  )
  start = cpp.index("    std::vector<CompareRow> rows = {")
  end = cpp.index("    bool comparisons_passed = true;", start)
  rows = '''    std::vector<CompareRow> rows = {
        {"z",
         iq36::compare_vectors(cpu.z, oracle_z, 5e-3),
         iq36::compare_vectors(gpu_projection["z"].values, cpu.z, 5e-3),
         iq36::compare_vectors(gpu_projection["z"].values, oracle_z, 5e-3)},
    };
'''
  return cpp[:start] + rows + cpp[end:]


def parse_args() -> Any:
  args = BASE_PARSE_ARGS()
  if args.out_dir is None:
    stamp = BASE.utc_stamp()
    args.out_dir = BASE.ROOT / f"output/gpu-q4x8-z-projection-probe-{stamp}"
  return args


def write_summary(path: Path, payload: dict[str, Any]) -> None:
  probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
  timings = probe.get("projection_timings", {}) if isinstance(probe, dict) else {}
  comparisons = probe.get("comparisons", {}) if isinstance(probe, dict) else {}
  z_timing = timings.get("z", {}) if isinstance(timings, dict) else {}
  z_cmp = comparisons.get("z", {}) if isinstance(comparisons, dict) else {}
  gpu_vs_cpu = z_cmp.get("gpu_vs_cpu", {}) if isinstance(z_cmp, dict) else {}
  gpu_vs_oracle = z_cmp.get("gpu_vs_oracle", {}) if isinstance(z_cmp, dict) else {}
  lines = [
      "# GPU Q4-X8 Z Projection Probe",
      "",
      f"- workstream: `{BASE.WORKSTREAM}`",
      f"- host: `{payload.get('host')}`",
      f"- model: `{payload.get('model')}`",
      f"- layer: `{payload.get('layer')}`",
      f"- required checks passed: `{str(payload.get('required_checks_passed')).lower()}`",
      "- speedup claims allowed: `false`",
      f"- platform/device: `{probe.get('platform_name')}` / `{probe.get('device_name')}`",
      f"- kernel variant: `{probe.get('kernel_variant')}`",
      f"- min us: `{z_timing.get('gpu_kernel_min_us') if isinstance(z_timing, dict) else None}`",
      "",
      "| comparison | max abs | RMSE |",
      "|---|---:|---:|",
      f"| gpu_vs_cpu | {gpu_vs_cpu.get('max_abs_diff')} | {gpu_vs_cpu.get('rmse')} |",
      f"| gpu_vs_oracle | {gpu_vs_oracle.get('max_abs_diff')} | {gpu_vs_oracle.get('rmse')} |",
      "",
      "This is a component projection diagnostic only; it does not prove decode",
      "or model throughput.",
      "",
  ]
  path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  variant, forwarded_argv = strip_variant(sys.argv)
  BASE.SCHEMA_VERSION = SCHEMA_VERSION
  BASE.PROBE_CPP = z_only_cpp(variant)
  BASE.parse_args = parse_args
  BASE.write_summary = write_summary
  old_argv = sys.argv
  try:
    sys.argv = forwarded_argv
    return BASE.main()
  finally:
    sys.argv = old_argv


if __name__ == "__main__":
  raise SystemExit(main())
