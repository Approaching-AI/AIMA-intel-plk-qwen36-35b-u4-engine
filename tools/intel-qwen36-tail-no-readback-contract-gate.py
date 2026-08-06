#!/usr/bin/env python3
"""Gate the FFN-tail no-readback primitive for the carrier route.

This checks source shape only. It proves the resident down-handle FFN-tail API
can produce a layer-output handle without host readback, while preserving the
default readback behavior and keeping decode unwired until the hidden-state
carrier exists.
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
SCHEMA_VERSION = "intel-qwen36-tail-no-readback-contract-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ54 = ROOT / "output/resident-hidden-carrier-gate-20260706Tseq54Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/tail-no-readback-contract-gate-20260706Tseq55Z"


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

  checks = [
      _check(
          header,
          r"struct GpuFfnTailRun\s*\{[^}]*bool layer_output_host_valid = true;",
          "run_result_records_host_validity",
      ),
      _check(
          header,
          r"RunFfnTailFromDownHandle\([^;]*bool readback_layer_output = true\)",
          "single_down_handle_public_api_defaults_to_readback",
      ),
      _check(
          header,
          r"RunFfnTailFromDownHandles\([^;]*bool readback_layer_output = true\)",
          "dual_down_handles_public_api_defaults_to_readback",
      ),
      _check(
          source,
          r"RunFfnTailFromDownHandle\([^)]*std::uint64_t ffn_shexp_handle = 0,\s*"
          r"bool readback_layer_output = true\)",
          "impl_keeps_default_readback_behavior",
      ),
      _check(
          source,
          r"run\.layer_output_host_valid = readback_layer_output;",
          "impl_marks_host_vector_validity",
      ),
      _check(
          source,
          r"if \(readback_layer_output\) \{\s*run\.layer_output\.assign",
          "impl_allocates_host_output_only_when_readback_requested",
      ),
      _check(
          source,
          r"if \(readback_layer_output\) \{\s*Check\(api_\.clEnqueueReadBuffer"
          r"\(queue_, layer_output_buffer",
          "impl_guards_ffn_tail_layer_output_readback",
      ),
      _check(
          source,
          r"RunFfnTailFromDownHandle\([^)]*"
          r"bool readback_layer_output = true\)\s*\{.*?"
          r"EnsureScratchBuffer\(\s*scratch, values \* sizeof\(float\), "
          r"kClMemReadWrite, name\);.*?"
          r"make_write\(ffn_tail_scratch_layer_output_",
          "impl_layer_output_backing_buffer_is_gpu_readable",
      ),
      _check(
          source,
          r"ffn_shexp_handle, readback_layer_output\);",
          "dual_handle_wrapper_forwards_readback_flag",
      ),
      _check(
          source,
          r"repeat, 0,\s*readback_layer_output\);",
          "single_handle_wrapper_forwards_readback_flag",
      ),
  ]
  decode_uses_no_readback = "readback_layer_output" in decode_source
  all_checks_pass = all(check["present"] for check in checks)
  seq54_requires_carrier = bool(
      seq54.get("derived", {}).get(
          "resident_hidden_state_carrier_or_down_tail_fusion_required"
      )
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
      },
      "checks": checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "decode_loop_uses_no_readback_flag": decode_uses_no_readback,
          "default_behavior_preserved": all_checks_pass and not decode_uses_no_readback,
          "seq54_requires_carrier": seq54_requires_carrier,
          "primitive_ready_for_carrier_wiring": (
              all_checks_pass and not decode_uses_no_readback and seq54_requires_carrier
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "The resident down-handle FFN-tail primitive can now skip the host "
              "layer-output readback and still return a resident layer-output "
              "handle, but existing decode calls keep the default readback path. "
              "This is carrier plumbing only."
          ),
          "next_route": (
              "Wire the primitive only after a resident hidden-state carrier "
              "can pass the layer-output handle through next-layer RMSNorm, "
              "attention residual, FFN norm/router input, selected/shared FFN, "
              "and FFN tail without requiring the host layer-output vector."
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
      "tool": "tools/intel-qwen36-tail-no-readback-contract-gate.py",
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
      "# Tail No-Readback Contract Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- contract checks pass: `{str(d['all_contract_checks_pass']).lower()}`",
      f"- decode loop uses no-readback flag: `{str(d['decode_loop_uses_no_readback_flag']).lower()}`",
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
  parser.add_argument("--seq54-metrics", type=Path, default=DEFAULT_SEQ54)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("tail no-readback contract gate")
  print(f"  artifact: {out_dir}")
  print(f"  contract checks: {derived['all_contract_checks_pass']}")
  print(f"  decode no-readback enabled: {derived['decode_loop_uses_no_readback_flag']}")
  print(f"  verdict: {result['verdict']['reason']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
