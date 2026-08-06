#!/usr/bin/env python3
"""Gate resident F32-input device-Q8 Q4 no-readback plumbing.

This is source-contract evidence, not runtime speed evidence. It proves a
resident F32 hidden/input handle can feed device-side Q8 quantization and a
resident Q4_K x8 projection while producing a resident output handle without
host readback. Existing decode remains default-readback until broader qkv,
preconv, selected/shared FFN, and tail consumers are wired.
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
SCHEMA_VERSION = "intel-qwen36-device-q8-q4-no-readback-contract-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ57 = ROOT / "output/f32matvec-no-readback-contract-gate-20260706Tseq57Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/device-q8-q4-no-readback-contract-gate-20260706Tseq59Z"


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
  seq57 = _load_json(args.seq57_metrics)

  checks = [
      _check(
          header,
          r"struct GpuDeviceQ8Q4X8MatvecRun\s*\{[^}]*"
          r"std::uint64_t output_handle = 0;[^}]*"
          r"bool output_host_valid = true;",
          "device_q8_q4_run_records_output_handle_and_host_validity",
      ),
      _check(
          header,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8\([^;]*"
          r"bool readback_output = true\)",
          "public_device_q8_q4_api_defaults_to_readback",
      ),
      _check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8\([^)]*"
          r"bool readback_output = true\)",
          "impl_device_q8_q4_api_defaults_to_readback",
      ),
      _check(
          source,
          r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8\([^)]*"
          r"\)\s*\{.*?run\.output_host_valid = readback_output;",
          "impl_marks_device_q8_q4_host_vector_validity",
      ),
      _check(
          source,
          r"run\.output_host_valid = readback_output;\s*"
          r"if \(readback_output\) \{\s*run\.output\.assign",
          "impl_allocates_device_q8_q4_host_output_only_when_requested",
      ),
      _check(
          source,
          r"RunQ8QuantizeWithBsumsKernel\(\s*input\.buffer,\s*block_count,",
          "impl_quantizes_resident_input_handle_on_device",
      ),
      _check(
          source,
          r"f32_input_q4_scratch_output_,\s*"
          r"static_cast<std::size_t>\(resident\.rows\) \* sizeof\(float\),\s*"
          r"kClMemReadWrite",
          "impl_device_q8_q4_output_backing_buffer_is_gpu_readable",
      ),
      _check(
          source,
          r"if \(readback_output && !read_in_kernel\) \{\s*"
          r"Check\(api_\.clEnqueueReadBuffer\(queue_, output_buffer",
          "impl_guards_device_q8_q4_output_readback",
      ),
      _check(
          source,
          r"run\.output_handle = RegisterF32BufferAlias\(\s*"
          r"&f32_input_q4_output_alias_handle_, output_buffer",
          "impl_registers_device_q8_q4_output_resident_alias",
      ),
      _check(
          source,
          r"std::uint64_t f32_input_q4_output_alias_handle_ = 0;",
          "impl_has_device_q8_q4_output_alias_slot",
      ),
      _check(
          source,
          r"handle, input_handle, repeat, variant, readback_output\);",
          "public_wrapper_forwards_readback_flag",
      ),
  ]
  decode_call_present = re.search(
      r"RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8\(",
      decode_source,
  ) is not None
  all_checks_pass = all(check["present"] for check in checks)
  seq57_ready = bool(
      seq57.get("derived", {}).get("primitive_ready_for_carrier_wiring")
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
          "seq57_metrics": {
              "path": _display_path(args.seq57_metrics),
              "sha256": _sha256(args.seq57_metrics),
          },
      },
      "checks": checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "decode_call_present": decode_call_present,
          "default_behavior_preserved": all_checks_pass and not decode_call_present,
          "seq57_f32matvec_no_readback_ready": seq57_ready,
          "primitive_ready_for_carrier_wiring": (
              all_checks_pass and not decode_call_present and seq57_ready
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "Resident F32 input handles can now feed device-side Q8 "
              "quantization plus resident Q4 projection and return a resident "
              "output handle without host readback."
          ),
          "next_route": (
              "Wire this primitive into real consumers only as part of a broader "
              "carrier: qkv/preconv and selected/shared FFN must consume resident "
              "normalized handles, and FFN tail/attention residuals must stop "
              "requiring host vectors before no-readback decode is admissible."
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
      "tool": "tools/intel-qwen36-device-q8-q4-no-readback-contract-gate.py",
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
      "# Device-Q8 Q4 No-Readback Contract Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- contract checks pass: `{str(d['all_contract_checks_pass']).lower()}`",
      f"- decode call present: `{str(d['decode_call_present']).lower()}`",
      f"- default behavior preserved: `{str(d['default_behavior_preserved']).lower()}`",
      f"- primitive ready for carrier wiring: `{str(d['primitive_ready_for_carrier_wiring']).lower()}`",
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
  parser.add_argument("--seq57-metrics", type=Path, default=DEFAULT_SEQ57)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("device-Q8 Q4 no-readback contract gate")
  print(f"all_contract_checks_pass={derived['all_contract_checks_pass']}")
  print(f"default_behavior_preserved={derived['default_behavior_preserved']}")
  print(
      "primitive_ready_for_carrier_wiring="
      f"{derived['primitive_ready_for_carrier_wiring']}"
  )
  print(f"artifact={_display_path(out_dir)}")
  return 0 if derived["primitive_ready_for_carrier_wiring"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
