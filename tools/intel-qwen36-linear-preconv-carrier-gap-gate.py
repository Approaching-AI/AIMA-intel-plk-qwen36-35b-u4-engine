#!/usr/bin/env python3
"""Gate the full-carrier linear-preconv gap.

This is source-contract evidence, not benchmark evidence. It proves the
current carrier primitives stop before the linear qkv+conv preconv boundary:
resident F32 input handles can feed device-Q8 Q4/Q6 projections, and resident
qkv projections can feed resident conv-state, but there is no single
resident-input -> device-Q8 -> resident qkv -> resident conv-state handoff.
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
SCHEMA_VERSION = "intel-qwen36-linear-preconv-carrier-gap-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GENERATED_DECODE_SOURCE = (
    ROOT / "output/r2-gpu-decode-smoke-20260706T110836Z/r2_gpu_decode_smoke.cpp"
)
DEFAULT_GENERATE_RESULT = (
    ROOT / "output/r2-gpu-decode-smoke-20260706T110836Z/result.json"
)
DEFAULT_SEQ54 = ROOT / "output/resident-hidden-carrier-gate-20260706Tseq54Z/metrics.json"
DEFAULT_SEQ59 = ROOT / "output/device-q8-q4-no-readback-contract-gate-20260706Tseq59Z/metrics.json"
DEFAULT_SEQ60 = ROOT / "output/device-q8-q6-no-readback-contract-gate-20260706Tseq60Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/linear-preconv-carrier-gap-gate-20260706Tseq71Z"


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
  source = args.gpu_source.read_text(encoding="utf-8")
  decode_source = args.decode_source.read_text(encoding="utf-8")
  generated_decode_source = args.generated_decode_source.read_text(encoding="utf-8")
  generate_result = _load_json(args.generate_result)
  seq54 = _load_json(args.seq54_metrics)
  seq59 = _load_json(args.seq59_metrics)
  seq60 = _load_json(args.seq60_metrics)

  positive_checks = [
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8\([^;]*"
          r"std::uint64_t input_handle[^;]*bool readback_output = true\)",
          "q4_resident_input_device_q8_projection_primitive_exists",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentRawQ6K\([^;]*"
          r"std::uint64_t input_handle[^;]*bool readback_output = true\)",
          "q6_resident_input_device_q8_projection_primitive_exists",
      ),
      _check(
          header,
          r"RunResidentPackedQ4X8ThenResidentConvState\([^;]*"
          r"const std::vector<std::int8_t>& q8_qs[^;]*"
          r"const std::vector<std::int16_t>& q8_bsums[^;]*"
          r"const std::vector<float>& q8_d",
          "q4_resident_conv_state_api_requires_host_q8_planes",
      ),
      _check(
          header,
          r"RunResidentRawQ6KThenResidentConvState\([^;]*"
          r"const GpuQ8KInputPlanes& q8",
          "q6_resident_conv_state_api_requires_host_q8_planes",
      ),
      _check(
          source,
          r"RunResidentPackedQ4X8ThenResidentConvState\([^)]*\)\s*\{.*?"
          r"ValidateQ8InputPlanes\(q8_qs, q8_bsums, q8_d",
          "q4_conv_state_impl_uploads_validated_host_q8_planes",
      ),
      _check(
          source,
          r"RunResidentRawQ6KThenResidentConvState\([^)]*\)\s*\{.*?"
          r"ValidateQ6KQ8InputPlanes\(q8, resident\.blocks_per_row, repeat\)",
          "q6_conv_state_impl_uploads_validated_host_q8_planes",
      ),
      _check(
          source,
          r"GpuQ4X8ConvHandoffRun\s+RunResidentRawQ6KThenResidentConvState",
          "q6_resident_qkv_to_conv_state_impl_exists",
      ),
      _check(
          generated_decode_source,
          r"RunGpuPreConvFront\([^)]*const std::vector<float>& attn_norm",
          "decode_preconv_api_consumes_host_attn_norm_vector",
      ),
      _check(
          generated_decode_source,
          r"const auto q8 = iq36::QuantizeQ8KInputPlanes\(attn_norm\);",
          "decode_preconv_quantizes_attn_norm_on_host",
      ),
      _check(
          generated_decode_source,
          r"runner\.RunResidentRawQ6KThenResidentConvState\(\s*"
          r"resident_qkv_handle, q8, resident_conv_weights_handle",
          "decode_q6_preconv_resident_conv_state_still_uses_host_q8",
      ),
      _check(
          generated_decode_source,
          r"runner\.RunResidentPackedQ4X8ThenResidentConvState\(\s*"
          r"resident_qkv_handle,\s*q8\.qs,\s*q8\.bsums,\s*q8\.d",
          "decode_q4_preconv_resident_conv_state_still_uses_host_q8",
      ),
      _check(
          generated_decode_source,
          r"preconv_gpu = RunGpuPreConvFront\([^;]*"
          r"layer_input_gpu\.attn_norm",
          "live_linear_layer_passes_host_attn_norm_to_preconv",
      ),
  ]
  absent_checks = [
      _absent_check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResident(?:PackedQ4X8|RawQ6K)"
          r"ThenResidentConv",
          "no_public_resident_input_device_q8_then_conv_handoff",
      ),
      _absent_check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResident(?:PackedQ4X8|RawQ6K)"
          r"ThenResidentConv",
          "no_impl_resident_input_device_q8_then_conv_handoff",
      ),
      _absent_check(
          decode_source,
          r"RunF32InputHandleDeviceQ8ThenResident(?:PackedQ4X8|RawQ6K)"
          r"ThenResidentConv",
          "decode_does_not_call_resident_input_device_q8_then_conv",
      ),
  ]
  all_positive_checks_pass = all(check["present"] for check in positive_checks)
  all_absent_checks_pass = all(check["absent"] for check in absent_checks)
  seq54_requires_carrier = bool(
      seq54.get("derived", {}).get(
          "resident_hidden_state_carrier_or_down_tail_fusion_required"
      )
  )
  seq59_ready = bool(
      seq59.get("derived", {}).get("primitive_ready_for_carrier_wiring")
  )
  seq60_ready = bool(
      seq60.get("derived", {}).get("primitive_ready_for_carrier_wiring")
  )
  preconv_gap_blocks_existing_carrier = (
      all_positive_checks_pass
      and all_absent_checks_pass
      and seq54_requires_carrier
      and seq59_ready
      and seq60_ready
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
          "generate_result": {
              "path": _display_path(args.generate_result),
              "sha256": _sha256(args.generate_result),
              "source_sha": generate_result.get("source_sha"),
              "generated_cpp": generate_result.get("generated_cpp"),
          },
          "seq54_metrics": {
              "path": _display_path(args.seq54_metrics),
              "sha256": _sha256(args.seq54_metrics),
          },
          "seq59_metrics": {
              "path": _display_path(args.seq59_metrics),
              "sha256": _sha256(args.seq59_metrics),
          },
          "seq60_metrics": {
              "path": _display_path(args.seq60_metrics),
              "sha256": _sha256(args.seq60_metrics),
          },
      },
      "source_contract_checks": positive_checks,
      "missing_api_checks": absent_checks,
      "derived": {
          "all_source_contract_checks_present": all_positive_checks_pass,
          "all_missing_api_checks_pass": all_absent_checks_pass,
          "seq54_requires_carrier_or_fusion": seq54_requires_carrier,
          "seq59_q4_device_q8_projection_ready": seq59_ready,
          "seq60_q6_device_q8_projection_ready": seq60_ready,
          "linear_preconv_gap_blocks_existing_carrier": (
              preconv_gap_blocks_existing_carrier
          ),
          "preconv_signature_line": _line_of(
              generated_decode_source, "PreConvFrontRun RunGpuPreConvFront("
          ),
          "host_preconv_call_line": _line_of(
              generated_decode_source, "layer_input_gpu.attn_norm"
          ),
          "q6_conv_state_header_line": _line_of(
              header, "RunResidentRawQ6KThenResidentConvState("
          ),
          "q4_conv_state_header_line": _line_of(
              header, "RunResidentPackedQ4X8ThenResidentConvState("
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "full_carrier_wirable_with_existing_primitives": False,
          "reason": (
              "The current full-carrier path reaches resident-input "
              "device-Q8 Q4/Q6 projection primitives, and it reaches resident "
              "qkv-to-conv-state primitives, but the qkv+conv preconv boundary "
              "still requires host Q8 planes from a host attn_norm vector. "
              "Existing primitives cannot be composed without a host vector "
              "between them."
          ),
          "next_route": (
              "A full-carrier continuation must add a resident-input "
              "device-Q8 then resident qkv+conv-state handoff for the linear "
              "preconv path, then wire RunGpuPreConvFront to consume the "
              "resident attention-norm handle. Otherwise switch to DPAS beyond "
              "the current Q4 occupancy bounds or a materially different "
              "non-atomic down-to-tail design."
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
      "tool": "tools/intel-qwen36-linear-preconv-carrier-gap-gate.py",
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
      "# Linear Preconv Carrier Gap Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- source contracts present: `{str(d['all_source_contract_checks_present']).lower()}`",
      f"- missing API checks pass: `{str(d['all_missing_api_checks_pass']).lower()}`",
      f"- seq54 carrier/fusion requirement: `{str(d['seq54_requires_carrier_or_fusion']).lower()}`",
      f"- Q4 resident-input device-Q8 projection ready: `{str(d['seq59_q4_device_q8_projection_ready']).lower()}`",
      f"- Q6 resident-input device-Q8 projection ready: `{str(d['seq60_q6_device_q8_projection_ready']).lower()}`",
      f"- preconv gap blocks existing carrier: `{str(d['linear_preconv_gap_blocks_existing_carrier']).lower()}`",
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
  parser.add_argument("--generate-result", type=Path, default=DEFAULT_GENERATE_RESULT)
  parser.add_argument("--seq54-metrics", type=Path, default=DEFAULT_SEQ54)
  parser.add_argument("--seq59-metrics", type=Path, default=DEFAULT_SEQ59)
  parser.add_argument("--seq60-metrics", type=Path, default=DEFAULT_SEQ60)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  d = result["derived"]
  print("linear preconv carrier gap gate")
  print(f"  artifact: {out_dir}")
  print(
      "  source checks: "
      f"{d['all_source_contract_checks_present']} ; missing-api checks: "
      f"{d['all_missing_api_checks_pass']}"
  )
  print(
      "  preconv gap blocks existing carrier: "
      f"{d['linear_preconv_gap_blocks_existing_carrier']}"
  )
  print(f"  verdict: {result['verdict']['next_route']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
