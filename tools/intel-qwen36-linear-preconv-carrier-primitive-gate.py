#!/usr/bin/env python3
"""Gate resident-input linear-preconv carrier primitive plumbing.

This is source-contract evidence, not runtime speed evidence. It proves the
seq71 gap has a current engine primitive: a resident F32 input handle can feed
device-side Q8 quantization, resident Q4/Q6 qkv projection, and resident
conv-state handoff without a host-Q8 bridge. Decode remains default-off until
RunGpuPreConvFront is explicitly wired to consume resident attention-norm
handles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-linear-preconv-carrier-primitive-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GENERATED_DECODE_SOURCE = (
    ROOT / "output/r2-gpu-decode-smoke-20260706T110836Z/r2_gpu_decode_smoke.cpp"
)
DEFAULT_SEQ71 = ROOT / "output/linear-preconv-carrier-gap-gate-20260706Tseq71Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/linear-preconv-carrier-primitive-gate-20260706Tseq72Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(text: str, pattern: str, label: str) -> dict[str, Any]:
  match = re.search(pattern, text, re.S)
  return {
      "label": label,
      "present": match is not None,
      "line": text.count("\n", 0, match.start()) + 1 if match else None,
  }


def _absent_check(text: str, pattern: str, label: str) -> dict[str, Any]:
  match = re.search(pattern, text, re.S)
  return {
      "label": label,
      "absent": match is None,
      "line": text.count("\n", 0, match.start()) + 1 if match else None,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  header = args.gpu_header.read_text(encoding="utf-8")
  source = args.gpu_source.read_text(encoding="utf-8")
  decode_source = args.decode_source.read_text(encoding="utf-8")
  generated_decode_source = args.generated_decode_source.read_text(encoding="utf-8")
  seq71 = _load_json(args.seq71_metrics)

  checks = [
      _check(
          header,
          r"struct GpuQ4X8ConvHandoffTiming\s*\{[^}]*"
          r"double q8_quantize_min_us = 0\.0;[^}]*"
          r"double shell_sum_min_us = 0\.0;[^}]*"
          r"std::uint64_t q8_quantize_global_work_items = 0;",
          "conv_handoff_timing_records_device_q8_quantize",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState"
          r"\([^;]*std::uint64_t input_handle[^;]*"
          r"bool readback_conv_output = true\)",
          "public_q6_resident_input_preconv_api_exists",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState"
          r"\([^;]*std::uint64_t input_handle[^;]*"
          r"GpuQ4X8KernelVariant variant[^;]*bool readback_conv_output = true\)",
          "public_q4_resident_input_preconv_api_exists",
      ),
      _check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState"
          r"\([^)]*\)\s*\{.*?RunQ8QuantizeKernel\(\s*input\.buffer,"
          r".*?RunQ6K(?:SelectedRowstripeKernel|Kernel)\("
          r".*?kernel_conv_",
          "impl_q6_quantizes_resident_input_then_qkv_conv",
      ),
      _check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState"
          r"\([^)]*\)\s*\{.*?RunQ8QuantizeWithBsumsKernel\(\s*input\.buffer,"
          r".*?RunHandoffKernels\(",
          "impl_q4_quantizes_resident_input_then_qkv_conv",
      ),
      _check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState"
          r"\([^)]*\)\s*\{.*?run\.conv_output_handle = RegisterF32BufferAlias",
          "impl_q6_returns_resident_conv_output_handle",
      ),
      _check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState"
          r"\([^)]*\)\s*\{.*?run\.conv_output_handle = RegisterF32BufferAlias",
          "impl_q4_returns_resident_conv_output_handle",
      ),
      _check(
          source,
          r"GpuQ4X8MatvecRunner::\s*"
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState"
          r"\([^)]*\)\s*\{[^}]*impl_->"
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState",
          "public_wrapper_forwards_q6_preconv_primitive",
      ),
      _check(
          source,
          r"GpuQ4X8MatvecRunner::\s*"
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState"
          r"\([^)]*\)\s*\{[^}]*impl_\s*->"
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState",
          "public_wrapper_forwards_q4_preconv_primitive",
      ),
  ]
  no_host_q8_bridge_checks = [
      _absent_check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState"
          r"\([^)]*\)\s*\{(?:(?!return run;).)*clEnqueueWriteBuffer\(queue_, q8_",
          "q6_preconv_primitive_has_no_host_q8_write",
      ),
      _absent_check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState"
          r"\([^)]*\)\s*\{(?:(?!return run;).)*clEnqueueWriteBuffer\(queue_, q8_",
          "q4_preconv_primitive_has_no_host_q8_write",
      ),
  ]
  decode_call_checks = [
      _absent_check(
          decode_source,
          r"RunF32InputHandleDeviceQ8ThenResident(?:RawQ6K|PackedQ4X8)"
          r"ThenResidentConvState",
          "decode_generator_does_not_call_preconv_primitive",
      ),
      _absent_check(
          generated_decode_source,
          r"RunF32InputHandleDeviceQ8ThenResident(?:RawQ6K|PackedQ4X8)"
          r"ThenResidentConvState",
          "generated_decode_does_not_call_preconv_primitive",
      ),
  ]

  all_checks_pass = all(check["present"] for check in checks)
  no_host_q8_bridge = all(check["absent"] for check in no_host_q8_bridge_checks)
  decode_default_off = all(check["absent"] for check in decode_call_checks)
  seq71_gap_was_real = bool(
      seq71.get("derived", {}).get("linear_preconv_gap_blocks_existing_carrier")
  )
  primitive_ready = (
      all_checks_pass
      and no_host_q8_bridge
      and decode_default_off
      and seq71_gap_was_real
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "gpu_header": {
              "path": _display_path(args.gpu_header),
              "sha256": _sha256(args.gpu_header),
          },
          "gpu_source": {
              "path": _display_path(args.gpu_source),
              "sha256": _sha256(args.gpu_source),
          },
          "decode_source": {
              "path": _display_path(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "generated_decode_source": {
              "path": _display_path(args.generated_decode_source),
              "sha256": _sha256(args.generated_decode_source),
          },
          "seq71_metrics": {
              "path": _display_path(args.seq71_metrics),
              "sha256": _sha256(args.seq71_metrics),
          },
      },
      "checks": checks,
      "no_host_q8_bridge_checks": no_host_q8_bridge_checks,
      "decode_default_off_checks": decode_call_checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "no_host_q8_bridge_in_primitive": no_host_q8_bridge,
          "decode_default_off": decode_default_off,
          "seq71_gap_was_real": seq71_gap_was_real,
          "primitive_ready_for_preconv_wiring": primitive_ready,
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "The engine now exposes resident-input device-Q8 Q4/Q6 "
              "qkv+conv-state primitives that avoid a host-Q8 bridge and return "
              "resident conv-output handles."
          ),
          "next_route": (
              "Wire RunGpuPreConvFront to accept a resident attention-norm "
              "handle and call these primitives behind an opt-in carrier gate; "
              "then run a compile/top-1 explore before any speed promotion."
          ),
      },
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": f"{SCHEMA_VERSION}-manifest",
      "tool": "tools/intel-qwen36-linear-preconv-carrier-primitive-gate.py",
      "workstream": WORKSTREAM,
      "artifact": _display_path(out_dir),
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  d = result["derived"]
  lines = [
      "# Linear Preconv Carrier Primitive Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- contract checks pass: `{str(d['all_contract_checks_pass']).lower()}`",
      f"- no host-Q8 bridge in primitive: `{str(d['no_host_q8_bridge_in_primitive']).lower()}`",
      f"- decode default-off: `{str(d['decode_default_off']).lower()}`",
      f"- seq71 gap was real: `{str(d['seq71_gap_was_real']).lower()}`",
      f"- primitive ready for preconv wiring: `{str(d['primitive_ready_for_preconv_wiring']).lower()}`",
      "",
      "## Verdict",
      "",
      result["verdict"]["reason"],
      "",
      result["verdict"]["next_route"],
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gpu-header", type=Path, default=DEFAULT_GPU_HEADER)
  parser.add_argument("--gpu-source", type=Path, default=DEFAULT_GPU_SOURCE)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument(
      "--generated-decode-source", type=Path, default=DEFAULT_GENERATED_DECODE_SOURCE
  )
  parser.add_argument("--seq71-metrics", type=Path, default=DEFAULT_SEQ71)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  d = result["derived"]
  print("linear preconv carrier primitive gate")
  print(f"  artifact: {out_dir}")
  print(
      "  contract checks: "
      f"{d['all_contract_checks_pass']} ; no host-Q8 bridge: "
      f"{d['no_host_q8_bridge_in_primitive']}"
  )
  print(f"  primitive ready: {d['primitive_ready_for_preconv_wiring']}")
  print(f"  verdict: {result['verdict']['next_route']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
