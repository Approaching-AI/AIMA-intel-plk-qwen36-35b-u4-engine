#!/usr/bin/env python3
"""Gate resident-input F32 matvec no-readback plumbing for the carrier route.

This is source-contract evidence, not runtime speed evidence. It proves the
resident F32 matvec-from-input-handle path can produce a resident output handle
without host readback, while existing decode keeps default readback because CPU
router top-k still needs host logits.
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
SCHEMA_VERSION = "intel-qwen36-f32matvec-no-readback-contract-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ56 = ROOT / "output/rmsnorm-no-readback-contract-gate-20260706Tseq56Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/f32matvec-no-readback-contract-gate-20260706Tseq57Z"


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
  seq56 = _load_json(args.seq56_metrics)

  checks = [
      _check(
          header,
          r"struct GpuF32MatvecRun\s*\{[^}]*std::uint64_t output_handle = 0;"
          r"[^}]*bool output_host_valid = true;",
          "f32matvec_run_records_output_handle_and_host_validity",
      ),
      _check(
          header,
          r"RunResidentF32MatvecFromInputHandle\([^;]*"
          r"bool readback_output = true\)",
          "public_f32matvec_handle_api_defaults_to_readback",
      ),
      _check(
          source,
          r"RunResidentF32MatvecFromInputHandle\([^)]*"
          r"bool readback_output = true\)",
          "impl_f32matvec_handle_api_defaults_to_readback",
      ),
      _check(
          source,
          r"RunResidentF32MatvecFromInputHandle\([^)]*\)\s*\{.*?"
          r"run\.output_host_valid = readback_output;\s*"
          r"if \(readback_output\) \{\s*run\.output\.assign",
          "impl_allocates_host_output_only_when_requested",
      ),
      _check(
          source,
          r"out_buffer = EnsureScratchBuffer\(\s*"
          r"resident_f32_matvec_scratch_output_",
          "impl_uses_scratch_owned_output_buffer",
      ),
      _check(
          source,
          r"resident_f32_matvec_scratch_output_,\s*"
          r"static_cast<std::size_t>\(resident\.rows\) \* sizeof\(float\),\s*"
          r"kClMemReadWrite",
          "impl_f32matvec_output_backing_buffer_is_gpu_readable",
      ),
      _check(
          source,
          r"ReleaseScratchBuffer\(resident_f32_matvec_scratch_output_\);",
          "impl_releases_scratch_owned_output_buffer",
      ),
      _check(
          source,
          r"if \(readback_output\) \{\s*Check\(api_\.clEnqueueReadBuffer"
          r"\(\s*queue_, out_buffer",
          "impl_guards_f32matvec_output_readback",
      ),
      _check(
          source,
          r"run\.output_handle = RegisterF32BufferAlias\(\s*"
          r"&resident_f32_matvec_output_alias_handle_, out_buffer",
          "impl_registers_f32matvec_output_resident_alias",
      ),
      _check(
          source,
          r"std::uint64_t resident_f32_matvec_output_alias_handle_ = 0;",
          "impl_has_f32matvec_output_alias_slot",
      ),
      _check(
          source,
          r"handle, input_handle, repeat, readback_output\);",
          "public_wrapper_forwards_readback_flag",
      ),
  ]
  decode_call_uses_explicit_no_readback = re.search(
      r"RunResidentF32MatvecFromInputHandle\([^;]*,\s*false\)",
      decode_source,
      re.S,
  ) is not None
  all_checks_pass = all(check["present"] for check in checks)
  seq56_ready = bool(
      seq56.get("derived", {}).get("primitive_ready_for_carrier_wiring")
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
          "seq56_metrics": {
              "path": _display_path(args.seq56_metrics),
              "sha256": _sha256(args.seq56_metrics),
          },
      },
      "checks": checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "decode_call_uses_explicit_no_readback": decode_call_uses_explicit_no_readback,
          "default_behavior_preserved": all_checks_pass
          and not decode_call_uses_explicit_no_readback,
          "seq56_rmsnorm_no_readback_ready": seq56_ready,
          "primitive_ready_for_carrier_wiring": (
              all_checks_pass and not decode_call_uses_explicit_no_readback
              and seq56_ready
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "Resident F32 matvec from an input handle can now produce a "
              "resident output handle without host readback, while existing "
              "decode keeps default readback for CPU router top-k."
          ),
          "next_route": (
              "The carrier still needs GPU-side consumers for F32 matvec output "
              "or a GPU top-k/router path; selected/shared FFN and qkv/preconv "
              "must consume resident normalized/input handles before no-readback "
              "decode can be enabled."
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
      "tool": "tools/intel-qwen36-f32matvec-no-readback-contract-gate.py",
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
      "# F32 Matvec No-Readback Contract Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- contract checks pass: `{str(d['all_contract_checks_pass']).lower()}`",
      f"- decode explicitly disables readback: `{str(d['decode_call_uses_explicit_no_readback']).lower()}`",
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
  parser.add_argument("--seq56-metrics", type=Path, default=DEFAULT_SEQ56)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("f32matvec no-readback contract gate")
  print(f"  artifact: {out_dir}")
  print(f"  contract checks: {derived['all_contract_checks_pass']}")
  print(
      "  decode explicit no-readback: "
      f"{derived['decode_call_uses_explicit_no_readback']}"
  )
  print(f"  verdict: {result['verdict']['reason']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
