#!/usr/bin/env python3
"""Gate qkv-only linear-preconv carrier wiring.

This is source-contract evidence, not benchmark evidence. Seq72 added a
resident-input device-Q8 -> resident qkv+conv-state primitive, but the live
preconv front fans one host-Q8 quantization into qkv and alpha/beta/z paths.
Wiring only qkv to the seq72 primitive would leave the host-Q8 bridge for the
other projections; wiring each projection with the current resident-input
primitives would quantize the same resident F32 input multiple times. The next
carrier unit therefore needs a shared-Q8/bundled preconv primitive, not a
qkv-only decode wire.
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
SCHEMA_VERSION = "intel-qwen36-linear-preconv-carrier-bundle-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GENERATED_DECODE_SOURCE = (
    ROOT / "output/r2-gpu-decode-smoke-20260706T110836Z/r2_gpu_decode_smoke.cpp"
)
DEFAULT_GENERATE_RESULT = (
    ROOT / "output/r2-gpu-decode-smoke-20260706T110836Z/result.json"
)
DEFAULT_SEQ72 = (
    ROOT / "output/linear-preconv-carrier-primitive-gate-20260706Tseq72Z/metrics.json"
)
DEFAULT_OUT_DIR = ROOT / "output/linear-preconv-carrier-bundle-gate-20260706Tseq73Z"


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


def _line_of(text: str, needle: str) -> int | None:
  index = text.find(needle)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def compute(args: argparse.Namespace) -> dict[str, Any]:
  header = args.gpu_header.read_text(encoding="utf-8")
  decode_source = args.decode_source.read_text(encoding="utf-8")
  generated_decode_source = args.generated_decode_source.read_text(encoding="utf-8")
  generate_result = _load_json(args.generate_result)
  seq72 = _load_json(args.seq72_metrics)

  preconv_fanout_checks = [
      _check(
          generated_decode_source,
          r"PreConvFrontRun RunGpuPreConvFront\([^)]*"
          r"const std::vector<float>& attn_norm",
          "preconv_api_consumes_host_attn_norm_vector",
      ),
      _check(
          generated_decode_source,
          r"const auto q8 = iq36::QuantizeQ8KInputPlanes\(attn_norm\);",
          "preconv_front_quantizes_attn_norm_once_on_host",
      ),
      _check(
          generated_decode_source,
          r"RunResidentPackedQ4X8ThenResidentConvState\(\s*"
          r"resident_qkv_handle,\s*q8\.qs,\s*q8\.bsums,\s*q8\.d",
          "q4_qkv_conv_state_consumes_shared_host_q8",
      ),
      _check(
          generated_decode_source,
          r"RunResidentRawQ6KThenResidentConvState\(\s*"
          r"resident_qkv_handle,\s*q8,",
          "q6_qkv_conv_state_consumes_shared_host_q8",
      ),
      _check(
          generated_decode_source,
          r"RunResidentRawQ4KCpuOrder\(\s*alpha_beta_z_handle,\s*q8,\s*repeat\)",
          "fused_alpha_beta_z_path_consumes_shared_host_q8",
      ),
      _check(
          generated_decode_source,
          r"RunResidentPackedQ4X8\(\s*alpha_beta_handle,\s*"
          r"q8\.qs,\s*q8\.bsums,\s*q8\.d",
          "packed_alpha_beta_path_consumes_shared_host_q8",
      ),
      _check(
          generated_decode_source,
          r"RunProjectionFromTensor\(\s*runner,\s*model,\s*"
          r"alpha_tensor,\s*q8,\s*kLinearValueHeads",
          "fallback_alpha_path_consumes_shared_host_q8",
      ),
      _check(
          generated_decode_source,
          r"RunProjectionFromTensor\(\s*runner,\s*model,\s*"
          r"beta_tensor,\s*q8,\s*kLinearValueHeads",
          "fallback_beta_path_consumes_shared_host_q8",
      ),
      _check(
          generated_decode_source,
          r"RunProjectionFromTensor\(\s*runner,\s*model,\s*"
          r"z_tensor,\s*q8,\s*kLinearVValues",
          "fallback_z_path_consumes_shared_host_q8",
      ),
      _check(
          generated_decode_source,
          r"const auto q8 = iq36::QuantizeQ8KInputPlanes\(\s*"
          r"layer_input_gpu\.attn_norm\s*\);",
          "post_preconv_cpu_order_z_requantizes_host_attn_norm",
      ),
  ]
  resident_input_api_checks = [
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8\([^;]*"
          r"std::uint64_t input_handle[^;]*bool readback_output = true\)",
          "standalone_q4_resident_input_device_q8_projection_api_exists",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6K\([^;]*"
          r"std::uint64_t input_handle[^;]*bool readback_output = true\)",
          "standalone_q6_resident_input_device_q8_projection_api_exists",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ThenResidentConvState"
          r"\([^;]*std::uint64_t input_handle",
          "seq72_q4_qkv_conv_resident_input_api_exists",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6KThenResidentConvState"
          r"\([^;]*std::uint64_t input_handle",
          "seq72_q6_qkv_conv_resident_input_api_exists",
      ),
  ]
  missing_bundle_checks = [
      _absent_check(
          header,
          r"\bGpuDeviceQ8(?:Planes|Handle)\b",
          "no_public_reusable_device_q8_planes_or_handle_type",
      ),
      _absent_check(
          header,
          r"\b(?:device_q8|q8)_(?:planes_)?handle\b",
          "no_public_q8_handle_parameter",
      ),
      _absent_check(
          header,
          r"\b(?:Upload|Create|Register)(?:Device)?Q8(?:Planes|Handle)\b",
          "no_public_upload_or_register_q8_handle_api",
      ),
      _absent_check(
          generated_decode_source,
          r"\battn_norm_handle\b",
          "generated_preconv_has_no_attn_norm_handle_carrier",
      ),
      _absent_check(
          decode_source + "\n" + generated_decode_source,
          r"IQ36_LINEAR_PRECONV_RESIDENT_INPUT",
          "no_decode_gate_for_qkv_only_resident_input_preconv",
      ),
  ]

  all_preconv_fanout_checks_present = all(
      check["present"] for check in preconv_fanout_checks
  )
  resident_input_apis_present = all(
      check["present"] for check in resident_input_api_checks
  )
  no_shared_device_q8_bundle = all(
      check["absent"] for check in missing_bundle_checks
  )
  seq72_ready = bool(
      seq72.get("derived", {}).get("primitive_ready_for_preconv_wiring")
  )
  qkv_only_leaves_host_q8_bridge = (
      seq72_ready and all_preconv_fanout_checks_present
  )
  naive_all_projection_wiring_duplicates_device_q8 = (
      resident_input_apis_present and no_shared_device_q8_bundle
  )
  bundle_required = (
      qkv_only_leaves_host_q8_bridge
      and naive_all_projection_wiring_duplicates_device_q8
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "gpu_header": {
              "path": _display_path(args.gpu_header),
              "sha256": _sha256(args.gpu_header),
          },
          "decode_source": {
              "path": _display_path(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "generated_decode_source": {
              "path": _display_path(args.generated_decode_source),
              "sha256": _sha256(args.generated_decode_source),
          },
          "generate_result": {
              "path": _display_path(args.generate_result),
              "sha256": _sha256(args.generate_result),
              "source_sha": generate_result.get("source_sha"),
              "generated_cpp": generate_result.get("generated_cpp"),
          },
          "seq72_metrics": {
              "path": _display_path(args.seq72_metrics),
              "sha256": _sha256(args.seq72_metrics),
          },
      },
      "preconv_host_q8_fanout_checks": preconv_fanout_checks,
      "resident_input_api_checks": resident_input_api_checks,
      "missing_bundle_checks": missing_bundle_checks,
      "derived": {
          "seq72_primitive_ready_for_preconv_wiring": seq72_ready,
          "all_preconv_fanout_checks_present": all_preconv_fanout_checks_present,
          "resident_input_projection_apis_present": resident_input_apis_present,
          "no_shared_device_q8_bundle_api": no_shared_device_q8_bundle,
          "qkv_only_preconv_wiring_leaves_host_q8_bridge": (
              qkv_only_leaves_host_q8_bridge
          ),
          "naive_all_projection_wiring_duplicates_device_q8": (
              naive_all_projection_wiring_duplicates_device_q8
          ),
          "qkv_only_preconv_wiring_promotable": False,
          "shared_device_q8_preconv_bundle_required": bundle_required,
          "preconv_signature_line": _line_of(
              generated_decode_source, "PreConvFrontRun RunGpuPreConvFront("
          ),
          "host_q8_quantize_line": _line_of(
              generated_decode_source,
              "const auto q8 = iq36::QuantizeQ8KInputPlanes(attn_norm);",
          ),
          "post_preconv_cpu_order_z_quantize_line": _line_of(
              generated_decode_source,
              "const auto q8 = iq36::QuantizeQ8KInputPlanes(layer_input_gpu.attn_norm);",
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "qkv_only_preconv_wiring_promotable": False,
          "reason": (
              "Seq72 makes qkv+conv resident-input wiring possible, but the "
              "current preconv front still shares one host-Q8 quantization "
              "across qkv and alpha/beta/z paths. Qkv-only wiring would leave "
              "the host-Q8 bridge in place for the remaining projections, and "
              "the current APIs do not expose a reusable device-Q8 handle for "
              "a quantize-once preconv bundle."
          ),
          "next_route": (
              "Add a shared-device-Q8/bundled linear-preconv carrier that "
              "quantizes the resident attention-norm handle once and feeds "
              "qkv+conv plus alpha/beta/z consumers, then wire it behind an "
              "opt-in decode gate; otherwise switch to DPAS beyond current Q4 "
              "occupancy bounds or a materially different non-atomic "
              "down-to-tail route."
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
      "tool": "tools/intel-qwen36-linear-preconv-carrier-bundle-gate.py",
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
      "# Linear Preconv Carrier Bundle Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- seq72 primitive ready: `{str(d['seq72_primitive_ready_for_preconv_wiring']).lower()}`",
      f"- preconv host-Q8 fanout present: `{str(d['all_preconv_fanout_checks_present']).lower()}`",
      f"- resident-input projection APIs present: `{str(d['resident_input_projection_apis_present']).lower()}`",
      f"- shared device-Q8 bundle API absent: `{str(d['no_shared_device_q8_bundle_api']).lower()}`",
      f"- qkv-only wiring leaves host-Q8 bridge: `{str(d['qkv_only_preconv_wiring_leaves_host_q8_bridge']).lower()}`",
      f"- naive all-projection wiring duplicates device-Q8: `{str(d['naive_all_projection_wiring_duplicates_device_q8']).lower()}`",
      f"- shared device-Q8 preconv bundle required: `{str(d['shared_device_q8_preconv_bundle_required']).lower()}`",
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
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument(
      "--generated-decode-source", type=Path, default=DEFAULT_GENERATED_DECODE_SOURCE
  )
  parser.add_argument("--generate-result", type=Path, default=DEFAULT_GENERATE_RESULT)
  parser.add_argument("--seq72-metrics", type=Path, default=DEFAULT_SEQ72)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  d = result["derived"]
  print("linear preconv carrier bundle gate")
  print(f"  artifact: {out_dir}")
  print(
      "  fanout present: "
      f"{d['all_preconv_fanout_checks_present']} ; shared Q8 API absent: "
      f"{d['no_shared_device_q8_bundle_api']}"
  )
  print(
      "  qkv-only promotable: "
      f"{d['qkv_only_preconv_wiring_promotable']} ; bundle required: "
      f"{d['shared_device_q8_preconv_bundle_required']}"
  )
  print(f"  verdict: {result['verdict']['next_route']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
