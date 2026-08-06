#!/usr/bin/env python3
"""Gate the shared-device-Q8 linear-preconv primitive.

This is source-contract evidence, not benchmark evidence. Seq73 established
that qkv-only preconv wiring is not promotable because alpha/beta/z still need
the same Q8 planes. Seq74 made the resident attention-norm handle available.
This gate proves the engine now has a same-runner bundle primitive that
quantizes that resident F32 input once and fans the resulting device-Q8 buffers
into qkv+conv and alpha/beta/z.
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
SCHEMA_VERSION = "intel-qwen36-linear-preconv-shared-q8-primitive-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL_SOURCE = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GENERATED_DECODE_SOURCE = (
    ROOT
    / "output/r2-gpu-decode-smoke-20260706Tseq74-attn-norm-handle-generate/r2_gpu_decode_smoke.cpp"
)
DEFAULT_SEQ73 = (
    ROOT / "output/linear-preconv-carrier-bundle-gate-20260706Tseq73Z/metrics.json"
)
DEFAULT_SEQ74 = (
    ROOT / "output/linear-preconv-attn-norm-handle-gate-20260706Tseq74Z/metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/linear-preconv-shared-q8-primitive-gate-20260706Tseq75Z"
)


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


def _function_body(text: str, name: str) -> str:
  index = text.find(name)
  if index < 0:
    return ""
  brace = text.find("{", index)
  if brace < 0:
    return ""
  depth = 0
  for pos in range(brace, len(text)):
    char = text[pos]
    if char == "{":
      depth += 1
    elif char == "}":
      depth -= 1
      if depth == 0:
        return text[brace : pos + 1]
  return ""


def _body_check(body: str, pattern: str, label: str) -> dict[str, Any]:
  return _check(body, pattern, label)


def _body_count(body: str, needle: str, label: str) -> dict[str, Any]:
  return {
      "label": label,
      "count": body.count(needle),
      "present": body.count(needle) > 0,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  header = args.gpu_header.read_text(encoding="utf-8")
  source = args.gpu_source.read_text(encoding="utf-8")
  opencl = args.opencl_source.read_text(encoding="utf-8")
  decode_source = args.decode_source.read_text(encoding="utf-8")
  generated_decode_source = args.generated_decode_source.read_text(encoding="utf-8")
  seq73 = _load_json(args.seq73_metrics)
  seq74 = _load_json(args.seq74_metrics)

  q6_body = _function_body(
      source,
      "RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder",
  )
  q4_body = _function_body(
      source,
      "RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder",
  )

  public_api_checks = [
      _check(
          header,
          r"struct GpuLinearPreconvSharedQ8Timing \{[^}]*alpha_beta_z",
          "shared_q8_preconv_timing_struct_present",
      ),
      _check(
          header,
          r"std::uint64_t UploadRawQ4KCpuOrder\([^;]*"
          r"std::uint64_t blocks_per_row\)",
          "gpu_q4x8_runner_can_upload_raw_q4_cpu_order",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ4KCpuOrder"
          r"\([^;]*std::uint64_t input_handle[^;]*bool readback_output = true\)",
          "standalone_raw_q4_cpu_order_resident_input_api_present",
      ),
      _check(
          header,
          r"RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder"
          r"\([^;]*std::uint64_t alpha_beta_z_handle[^;]*bool readback_alpha_beta_z = true\)",
          "shared_q8_q6_qkv_conv_plus_alpha_beta_z_api_present",
      ),
      _check(
          header,
          r"RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder"
          r"\([^;]*std::uint64_t alpha_beta_z_handle[^;]*GpuQ4X8KernelVariant variant[^;]*"
          r"bool readback_alpha_beta_z = true\)",
          "shared_q8_q4_qkv_conv_plus_alpha_beta_z_api_present",
      ),
  ]
  kernel_checks = [
      _check(
          opencl,
          r"__kernel void q4k_cpu_order_matvec\(",
          "q4_cpu_order_kernel_available_in_q4x8_program",
      ),
      _check(
          source,
          r'kernel_q4_cpu_order_ = CreateNamedKernel\("q4k_cpu_order_matvec"\);',
          "q4_cpu_order_kernel_created_in_shared_runner",
      ),
      _check(
          source,
          r"std::unordered_map<std::uint64_t, ResidentRawQ4KCpuOrder>\s+"
          r"resident_raw_q4_cpu_order_;",
          "raw_q4_cpu_order_residency_in_shared_runner",
      ),
      _check(
          source,
          r"GpuQ4KCpuOrderMatvecTiming RunQ4KCpuOrderKernel\(",
          "raw_q4_cpu_order_kernel_launcher_present",
      ),
  ]
  q6_bundle_checks = [
      _body_count(
          q6_body,
          "RunQ8QuantizeWithBsumsKernel(",
          "q6_bundle_quantizes_once_with_bsums",
      ),
      _body_check(
          q6_body,
          r"(RunQ6KSelectedRowstripeKernel|RunQ6KKernel)\([^;]*q8_qs_buffer[^;]*q8_d_buffer",
          "q6_bundle_feeds_shared_q8_to_qkv",
      ),
      _body_check(
          q6_body,
          r"kernel_conv_[^;]*qkv_buffer",
          "q6_bundle_feeds_qkv_to_resident_conv",
      ),
      _body_check(
          q6_body,
          r"RunQ4KCpuOrderKernel\([^;]*q8_qs_buffer,\s*q8_bsums_buffer,\s*q8_d_buffer",
          "q6_bundle_feeds_same_q8_to_alpha_beta_z",
      ),
  ]
  q4_bundle_checks = [
      _body_count(
          q4_body,
          "RunQ8QuantizeWithBsumsKernel(",
          "q4_bundle_quantizes_once_with_bsums",
      ),
      _body_check(
          q4_body,
          r"RunHandoffKernels\([^;]*q8_qs_buffer,\s*q8_bsums_buffer,\s*q8_d_buffer",
          "q4_bundle_feeds_shared_q8_to_qkv_conv",
      ),
      _body_check(
          q4_body,
          r"RunQ4KCpuOrderKernel\([^;]*q8_qs_buffer,\s*q8_bsums_buffer,\s*q8_d_buffer",
          "q4_bundle_feeds_same_q8_to_alpha_beta_z",
      ),
  ]
  wrapper_checks = [
      _check(
          source,
          r"GpuQ4X8MatvecRunner::UploadRawQ4KCpuOrder\([^)]*\)\s*\{[^}]*"
          r"impl_->UploadRawQ4KCpuOrder",
          "upload_raw_q4_cpu_order_wrapper_present",
      ),
      _check(
          source,
          r"GpuQ4X8MatvecRunner::RunF32InputHandleDeviceQ8ThenResidentRawQ4KCpuOrder"
          r"\([^)]*\)\s*\{[^}]*impl_->RunF32InputHandleDeviceQ8ThenResidentRawQ4KCpuOrder",
          "standalone_raw_q4_cpu_order_wrapper_present",
      ),
      _check(
          source,
          r"GpuQ4X8MatvecRunner::\s*"
          r"RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder"
          r"\([^)]*\)\s*\{[^}]*impl_",
          "shared_q8_q6_bundle_wrapper_present",
      ),
      _check(
          source,
          r"GpuQ4X8MatvecRunner::\s*"
          r"RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder"
          r"\([^)]*\)\s*\{[^}]*impl_",
          "shared_q8_q4_bundle_wrapper_present",
      ),
  ]
  default_off_checks = [
      _absent_check(
          decode_source + "\n" + generated_decode_source,
          r"RunF32InputHandleSharedDeviceQ8ThenResident.*ResidentRawQ4KCpuOrder",
          "decode_does_not_consume_shared_q8_bundle_yet",
      ),
      _absent_check(
          decode_source + "\n" + generated_decode_source,
          r"IQ36_LINEAR_PRECONV_SHARED_Q8|IQ36_LINEAR_PRECONV_RESIDENT_INPUT",
          "no_decode_gate_for_shared_q8_preconv_yet",
      ),
  ]

  q6_quantize_once = (
      q6_bundle_checks[0]["count"] == 1
      if isinstance(q6_bundle_checks[0].get("count"), int)
      else False
  )
  q4_quantize_once = (
      q4_bundle_checks[0]["count"] == 1
      if isinstance(q4_bundle_checks[0].get("count"), int)
      else False
  )
  seq73_required = bool(
      seq73.get("derived", {}).get("shared_device_q8_preconv_bundle_required")
  )
  seq74_ready = bool(
      seq74.get("derived", {}).get("attn_norm_handle_carrier_precondition_ready")
  )
  all_public_api_checks_present = all(check["present"] for check in public_api_checks)
  all_kernel_checks_present = all(check["present"] for check in kernel_checks)
  all_q6_bundle_checks_present = (
      q6_quantize_once and all(check["present"] for check in q6_bundle_checks[1:])
  )
  all_q4_bundle_checks_present = (
      q4_quantize_once and all(check["present"] for check in q4_bundle_checks[1:])
  )
  all_wrapper_checks_present = all(check["present"] for check in wrapper_checks)
  decode_default_off = all(check["absent"] for check in default_off_checks)
  primitive_ready = all(
      [
          seq73_required,
          seq74_ready,
          all_public_api_checks_present,
          all_kernel_checks_present,
          all_q6_bundle_checks_present,
          all_q4_bundle_checks_present,
          all_wrapper_checks_present,
          decode_default_off,
      ]
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
          "opencl_source": {
              "path": _display_path(args.opencl_source),
              "sha256": _sha256(args.opencl_source),
          },
          "decode_source": {
              "path": _display_path(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "generated_decode_source": {
              "path": _display_path(args.generated_decode_source),
              "sha256": _sha256(args.generated_decode_source),
          },
          "seq73_metrics": {
              "path": _display_path(args.seq73_metrics),
              "sha256": _sha256(args.seq73_metrics),
          },
          "seq74_metrics": {
              "path": _display_path(args.seq74_metrics),
              "sha256": _sha256(args.seq74_metrics),
          },
      },
      "public_api_checks": public_api_checks,
      "kernel_checks": kernel_checks,
      "q6_bundle_checks": q6_bundle_checks,
      "q4_bundle_checks": q4_bundle_checks,
      "wrapper_checks": wrapper_checks,
      "default_off_checks": default_off_checks,
      "derived": {
          "seq73_shared_device_q8_preconv_bundle_required": seq73_required,
          "seq74_attn_norm_handle_carrier_precondition_ready": seq74_ready,
          "all_public_api_checks_present": all_public_api_checks_present,
          "all_kernel_checks_present": all_kernel_checks_present,
          "q6_bundle_quantizes_once_with_bsums": q6_quantize_once,
          "q4_bundle_quantizes_once_with_bsums": q4_quantize_once,
          "all_q6_bundle_checks_present": all_q6_bundle_checks_present,
          "all_q4_bundle_checks_present": all_q4_bundle_checks_present,
          "all_wrapper_checks_present": all_wrapper_checks_present,
          "decode_default_off": decode_default_off,
          "shared_device_q8_preconv_bundle_primitive_ready": primitive_ready,
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "The engine has a same-runner shared-device-Q8 preconv primitive, "
              "but the decode harness still does not call it."
          ),
          "next_route": (
              "Wire RunGpuPreConvFront to consume LayerInputRmsNormRun.attn_norm_handle "
              "and call the shared-device-Q8 bundle behind an opt-in gate."
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
      "tool": "tools/intel-qwen36-linear-preconv-shared-q8-primitive-gate.py",
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
      "# Linear Preconv Shared-Q8 Primitive Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      f"- public API checks present: `{str(d['all_public_api_checks_present']).lower()}`",
      f"- kernel checks present: `{str(d['all_kernel_checks_present']).lower()}`",
      f"- Q6 bundle checks present: `{str(d['all_q6_bundle_checks_present']).lower()}`",
      f"- Q4 bundle checks present: `{str(d['all_q4_bundle_checks_present']).lower()}`",
      f"- wrapper checks present: `{str(d['all_wrapper_checks_present']).lower()}`",
      f"- decode default-off: `{str(d['decode_default_off']).lower()}`",
      f"- primitive ready: `{str(d['shared_device_q8_preconv_bundle_primitive_ready']).lower()}`",
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
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL_SOURCE)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument(
      "--generated-decode-source", type=Path, default=DEFAULT_GENERATED_DECODE_SOURCE
  )
  parser.add_argument("--seq73-metrics", type=Path, default=DEFAULT_SEQ73)
  parser.add_argument("--seq74-metrics", type=Path, default=DEFAULT_SEQ74)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  d = result["derived"]
  print("linear preconv shared-Q8 primitive gate")
  print(f"  artifact: {out_dir}")
  print(
      "  primitive ready: "
      f"{d['shared_device_q8_preconv_bundle_primitive_ready']} ; "
      f"decode default-off: {d['decode_default_off']}"
  )
  print(f"  verdict: {result['verdict']['next_route']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
