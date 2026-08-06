#!/usr/bin/env python3
"""Gate FFN-tail resident-input carrier plumbing.

This is source-contract evidence, not runtime speed evidence. It proves the
runner has a default-off FFN-tail API that consumes resident FFN norm,
selected-down, shared-down, and attention-residual handles, computes the shared
gate through the resident F32 matvec path without host readback, and can return
a resident layer-output handle without host readback. Existing decode remains
unchanged until the broader carrier is wired.
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
SCHEMA_VERSION = "intel-qwen36-ffn-tail-resident-input-contract-gate-v0"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ60 = ROOT / "output/device-q8-q6-no-readback-contract-gate-20260706Tseq60Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/ffn-tail-resident-input-contract-gate-20260706Tseq61Z"


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
  seq60 = _load_json(args.seq60_metrics)

  checks = [
      _check(
          header,
          r"RunFfnTailFromDownHandlesResidentInputs\([^;]*"
          r"bool readback_layer_output = true\)",
          "public_tail_resident_input_api_defaults_to_readback",
      ),
      _check(
          source,
          r"RunFfnTailFromDownHandlesResidentInputs\([^)]*"
          r"bool readback_layer_output = true\)",
          "impl_tail_resident_input_api_defaults_to_readback",
      ),
      _check(
          source,
          r"RunFfnTailFromDownHandlesResidentInputs\([^)]*\)\s*\{.*?"
          r"RunResidentF32MatvecFromInputHandle\([^;]*repeat,\s*false\)",
          "impl_computes_shared_gate_without_host_readback",
      ),
      _check(
          source,
          r"const auto& shared_gate_buffer =\s*"
          r"ResidentF32BufferForHandle\(shared_gate\.output_handle\)",
          "impl_uses_resident_shared_gate_scalar",
      ),
      _check(
          source,
          r"ResidentF32BufferForHandle\(ffn_moe_down_handle\).*?"
          r"ResidentF32BufferForHandle\(ffn_shexp_handle\).*?"
          r"ResidentF32BufferForHandle\(attn_residual_handle\)",
          "impl_consumes_resident_down_shared_and_residual_handles",
      ),
      _check(
          source,
          r"run\.layer_output_host_valid = readback_layer_output;\s*"
          r"if \(readback_layer_output\) \{\s*run\.layer_output\.assign",
          "impl_allocates_tail_host_output_only_when_requested",
      ),
      _check(
          source,
          r"if \(readback_layer_output\) \{\s*"
          r"Check\(api_\.clEnqueueReadBuffer\(queue_, layer_output_buffer",
          "impl_guards_tail_layer_output_readback",
      ),
      _check(
          source,
          r"run\.layer_output_handle = RegisterF32BufferAlias\(\s*"
          r"&ffn_tail_layer_output_alias_handle_, layer_output_buffer",
          "impl_registers_tail_layer_output_resident_alias",
      ),
      _check(
          source,
          r"shared_gate_matvec_handle, attn_post_norm_handle, ffn_moe_down_handle,"
          r"\s*weights_norm, ffn_shexp_handle, attn_residual_handle",
          "public_wrapper_forwards_resident_handles",
      ),
  ]
  decode_call_present = re.search(
      r"RunFfnTailFromDownHandlesResidentInputs\(",
      decode_source,
  ) is not None
  all_checks_pass = all(check["present"] for check in checks)
  seq60_ready = bool(
      seq60.get("derived", {}).get("primitive_ready_for_carrier_wiring")
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
          "seq60_metrics": {
              "path": _display_path(args.seq60_metrics),
              "sha256": _sha256(args.seq60_metrics),
          },
      },
      "checks": checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "decode_call_present": decode_call_present,
          "default_behavior_preserved": all_checks_pass and not decode_call_present,
          "seq60_device_q8_q6_ready": seq60_ready,
          "primitive_ready_for_carrier_wiring": (
              all_checks_pass and not decode_call_present and seq60_ready
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "FFN tail can now consume resident FFN-norm, selected-down, "
              "shared-down, and attention-residual handles, with the shared "
              "gate computed through resident F32 matvec without readback."
          ),
          "next_route": (
              "The remaining carrier wiring must pass resident handles from "
              "attention-front/layer-input RMSNorm into qkv/preconv, "
              "selected/shared FFN, and this resident-input tail path; decode "
              "must stay default-readback until that whole boundary exists."
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
      "tool": "tools/intel-qwen36-ffn-tail-resident-input-contract-gate.py",
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
      "# FFN-Tail Resident-Input Contract Gate",
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
  parser.add_argument("--seq60-metrics", type=Path, default=DEFAULT_SEQ60)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("FFN-tail resident-input contract gate")
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
