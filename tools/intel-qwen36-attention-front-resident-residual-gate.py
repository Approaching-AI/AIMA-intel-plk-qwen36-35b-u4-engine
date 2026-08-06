#!/usr/bin/env python3
"""Gate attention-front resident residual-input carrier wiring.

This is source-contract evidence, not benchmark evidence. It proves decode can
opt into passing the previous resident tail layer-output handle into the next
attention-front residual add/RMSNorm handoff, while default decode remains
unchanged. Runtime speed evidence must come from a separate decode-smoke row.
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
SCHEMA_VERSION = "intel-qwen36-attention-front-resident-residual-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ54 = ROOT / "output/resident-hidden-carrier-gate-20260706Tseq54Z/metrics.json"
DEFAULT_SEQ63 = ROOT / "output/ffn-tail-resident-input-decode-gate-20260706Tseq63Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/attention-front-resident-residual-gate-20260706Tseq64Z"


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


def compute(args: argparse.Namespace) -> dict[str, Any]:
  header = args.gpu_header.read_text(encoding="utf-8")
  source = args.gpu_source.read_text(encoding="utf-8")
  decode_source = args.decode_source.read_text(encoding="utf-8")
  seq54 = _load_json(args.seq54_metrics)
  seq63 = _load_json(args.seq63_metrics)

  checks = [
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm\([^;]*"
          r"std::uint64_t residual_input_handle = 0\)",
          "public_linear_final_handoff_accepts_default_off_residual_handle",
      ),
      _check(
          header,
          r"RunResidentPackedQ4X8ThenResidentResidualRmsNorm\([^;]*"
          r"std::uint64_t residual_input_handle = 0\)",
          "public_attention_front_handoff_accepts_default_off_residual_handle",
      ),
      _check(
          header,
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\([^;]*"
          r"std::uint64_t residual_input_handle = 0\)",
          "public_full_core_handoff_accepts_default_off_residual_handle",
      ),
      _check(
          source,
          r"RunResidentPackedQ4X8ThenResidentResidualRmsNorm\([^)]*"
          r"std::uint64_t residual_input_handle = 0\)\s*\{.*?"
          r"ResidentF32BufferForHandle\(residual_input_handle\).*?"
          r"resident_residual_input->buffer.*?"
          r"if \(resident_residual_input == nullptr\) \{\s*"
          r"Check\(api_\.clEnqueueWriteBuffer",
          "impl_attention_front_handoff_uses_resident_residual_or_host_write",
      ),
      _check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm\([^)]*"
          r"std::uint64_t residual_input_handle = 0\)\s*\{.*?"
          r"ResidentF32BufferForHandle\(residual_input_handle\).*?"
          r"resident_residual_input->buffer.*?"
          r"if \(resident_residual_input == nullptr\) \{\s*"
          r"Check\(api_\.clEnqueueWriteBuffer",
          "impl_linear_final_handoff_uses_resident_residual_or_host_write",
      ),
      _check(
          source,
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\([^)]*"
          r"std::uint64_t residual_input_handle = 0\)\s*\{.*?"
          r"ResidentF32BufferForHandle\(residual_input_handle\).*?"
          r"resident_residual_input->buffer.*?"
          r"if \(resident_residual_input == nullptr\) \{\s*"
          r"Check\(api_\.clEnqueueWriteBuffer",
          "impl_full_core_handoff_uses_resident_residual_or_host_write",
      ),
      _check(
          source,
          r"residual_input_handle\).*?"
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm\(",
          "public_linear_final_wrapper_forwards_residual_handle",
      ),
      _check(
          source,
          r"residual_input_handle\).*?"
          r"RunResidentPackedQ4X8ThenResidentResidualRmsNorm\(",
          "public_attention_front_wrapper_forwards_residual_handle",
      ),
      _check(
          source,
          r"residual_input_handle\).*?"
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\(",
          "public_full_core_wrapper_forwards_residual_handle",
      ),
      _check(
          decode_source,
          r"bool g_decode_attention_front_resident_residual_input = false;",
          "decode_env_gate_defaults_off",
      ),
      _check(
          decode_source,
          r"std::getenv\(\"IQ36_ATTENTION_FRONT_RESIDENT_RESIDUAL_INPUT\"\) != nullptr",
          "decode_env_gate_reads_explicit_env",
      ),
      _check(
          decode_source,
          r"DecodeKeepPrevLayerOutputHandle\(\).*?"
          r"g_decode_attention_front_resident_residual_input",
          "decode_keeps_prev_tail_output_for_attention_residual_route",
      ),
      _check(
          decode_source,
          r"const std::uint64_t attention_residual_input_handle =\s*"
          r"g_decode_attention_front_resident_residual_input\s*\?\s*"
          r"g_decode_prev_layer_output_handle\s*:\s*0;",
          "decode_layer_selects_previous_resident_layer_output_handle",
      ),
      _check(
          decode_source,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm\([^;]*"
          r"residual_input_handle\);",
          "decode_linear_final_device_q8_handoff_passes_residual_handle",
      ),
      _check(
          decode_source,
          r"RunResidentPackedQ4X8ThenResidentResidualRmsNorm\([^;]*"
          r"resident_residual_input_handle\);",
          "decode_attention_front_handoff_passes_residual_handle",
      ),
      _check(
          decode_source,
          r"RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm\([^;]*"
          r"attention_residual_input_handle\);",
          "decode_full_core_handoff_passes_residual_handle",
      ),
      _check(
          decode_source,
          r"attention_front_resident_residual_input_enabled",
          "decode_stdout_reports_env_gate",
      ),
      _check(
          decode_source,
          r"\"IQ36_ATTENTION_FRONT_RESIDENT_RESIDUAL_INPUT\"",
          "remote_command_propagates_env_gate",
      ),
  ]

  all_checks_pass = all(check["present"] for check in checks)
  seq54_requires_carrier = bool(
      seq54.get("derived", {}).get(
          "resident_hidden_state_carrier_or_down_tail_fusion_required"
      )
  )
  seq63_ready = bool(
      seq63.get("derived", {}).get("decode_path_ready_for_explore")
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
          "seq54_metrics": {
              "path": _display_path(args.seq54_metrics),
              "sha256": _sha256(args.seq54_metrics),
          },
          "seq63_metrics": {
              "path": _display_path(args.seq63_metrics),
              "sha256": _sha256(args.seq63_metrics),
          },
      },
      "checks": checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "seq54_requires_resident_carrier_or_fusion": seq54_requires_carrier,
          "seq63_ffn_tail_resident_input_ready": seq63_ready,
          "default_behavior_preserved": all_checks_pass,
          "decode_path_ready_for_explore": (
              all_checks_pass and seq54_requires_carrier and seq63_ready
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "Decode can now opt into using the previous resident tail "
              "layer-output handle as the next attention-front residual input "
              "via IQ36_ATTENTION_FRONT_RESIDENT_RESIDUAL_INPUT. This is "
              "default-off carrier wiring, not speed evidence."
          ),
          "next_route": (
              "Run one noqueue decode-smoke explore row with the accepted "
              "selected/shared Q4+Q6 stack plus this env gate. Promote only if "
              "it clears the frontier noise band and passes the normal "
              "correctness ladder."
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
      "tool": "tools/intel-qwen36-attention-front-resident-residual-gate.py",
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
      "# Attention-Front Resident Residual Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- contract checks pass: `{str(d['all_contract_checks_pass']).lower()}`",
      f"- default behavior preserved: `{str(d['default_behavior_preserved']).lower()}`",
      f"- decode path ready for explore: `{str(d['decode_path_ready_for_explore']).lower()}`",
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
  parser.add_argument("--seq54-metrics", type=Path, default=DEFAULT_SEQ54)
  parser.add_argument("--seq63-metrics", type=Path, default=DEFAULT_SEQ63)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("attention-front resident residual gate")
  print(f"  artifact: {_display_path(out_dir)}")
  print(f"  contract checks: {derived['all_contract_checks_pass']}")
  print(f"  decode path ready for explore: {derived['decode_path_ready_for_explore']}")
  print(f"  speedup claims allowed: {result['verdict']['speedup_claims_allowed']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
