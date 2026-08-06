#!/usr/bin/env python3
"""Gate resident attention-norm handle carry-through for linear preconv.

This is source/generate evidence only. It proves the decode generator now
preserves the resident F32 handle produced by layer-input RMSNorm in
LayerInputRmsNormRun. No decode path consumes the handle yet; seq73 still
requires a shared-device-Q8 preconv bundle before a speed path is admissible.
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
SCHEMA_VERSION = "intel-qwen36-linear-preconv-attn-norm-handle-gate-v0"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GENERATED_DECODE_SOURCE = (
    ROOT
    / "output/r2-gpu-decode-smoke-20260706Tseq74-attn-norm-handle-generate/r2_gpu_decode_smoke.cpp"
)
DEFAULT_GENERATE_RESULT = (
    ROOT
    / "output/r2-gpu-decode-smoke-20260706Tseq74-attn-norm-handle-generate/result.json"
)
DEFAULT_SEQ73 = (
    ROOT / "output/linear-preconv-carrier-bundle-gate-20260706Tseq73Z/metrics.json"
)
DEFAULT_OUT_DIR = ROOT / "output/linear-preconv-attn-norm-handle-gate-20260706Tseq74Z"


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
  decode_source = args.decode_source.read_text(encoding="utf-8")
  generated_decode_source = args.generated_decode_source.read_text(encoding="utf-8")
  generate_result = _load_json(args.generate_result)
  seq73 = _load_json(args.seq73_metrics)

  source_checks = [
      _check(
          decode_source,
          r"struct LayerInputRmsNormRun \{\s*"
          r"std::vector<float> attn_norm;\s*"
          r"std::uint64_t attn_norm_handle = 0;",
          "generator_adds_layer_input_attn_norm_handle_field",
      ),
      _check(
          decode_source,
          r"run\.attn_norm = std::move\(rms\.output\);\s*"
          r"run\.attn_norm_handle = rms\.output_handle;",
          "generator_records_rmsnorm_output_handle",
      ),
      _check(
          generated_decode_source,
          r"struct LayerInputRmsNormRun \{\s*"
          r"std::vector<float> attn_norm;\s*"
          r"std::uint64_t attn_norm_handle = 0;",
          "generated_cpp_has_layer_input_attn_norm_handle_field",
      ),
      _check(
          generated_decode_source,
          r"run\.attn_norm = std::move\(rms\.output\);\s*"
          r"run\.attn_norm_handle = rms\.output_handle;",
          "generated_cpp_records_rmsnorm_output_handle",
      ),
  ]
  default_off_checks = [
      _absent_check(
          generated_decode_source,
          r"RunGpuPreConvFront\([^)]*attn_norm_handle",
          "preconv_call_does_not_consume_attn_norm_handle_yet",
      ),
      _absent_check(
          decode_source + "\n" + generated_decode_source,
          r"IQ36_LINEAR_PRECONV_RESIDENT_INPUT",
          "no_decode_gate_for_linear_preconv_resident_input_yet",
      ),
  ]

  source_ready = all(check["present"] for check in source_checks)
  decode_default_off = all(check["absent"] for check in default_off_checks)
  seq73_bundle_required = bool(
      seq73.get("derived", {}).get("shared_device_q8_preconv_bundle_required")
  )
  carrier_precondition_ready = (
      source_ready and decode_default_off and seq73_bundle_required
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
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
              "generate_only": generate_result.get("generate_only"),
          },
          "seq73_metrics": {
              "path": _display_path(args.seq73_metrics),
              "sha256": _sha256(args.seq73_metrics),
          },
      },
      "source_checks": source_checks,
      "default_off_checks": default_off_checks,
      "derived": {
          "all_source_checks_present": source_ready,
          "decode_default_off": decode_default_off,
          "seq73_shared_device_q8_preconv_bundle_required": seq73_bundle_required,
          "attn_norm_handle_carrier_precondition_ready": carrier_precondition_ready,
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "Layer-input RMSNorm now preserves its resident output handle in "
              "the generated decode harness, but linear preconv still does not "
              "consume it."
          ),
          "next_route": (
              "Use this handle only after adding the shared-device-Q8 "
              "linear-preconv bundle required by seq73."
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
      "tool": "tools/intel-qwen36-linear-preconv-attn-norm-handle-gate.py",
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
      "# Linear Preconv Attention-Norm Handle Gate",
      "",
      "This is source/generate evidence, not runtime speed evidence.",
      "",
      f"- source checks present: `{str(d['all_source_checks_present']).lower()}`",
      f"- decode default-off: `{str(d['decode_default_off']).lower()}`",
      f"- seq73 bundle required: `{str(d['seq73_shared_device_q8_preconv_bundle_required']).lower()}`",
      f"- carrier precondition ready: `{str(d['attn_norm_handle_carrier_precondition_ready']).lower()}`",
      "",
      result["verdict"]["reason"],
      "",
      result["verdict"]["next_route"],
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument(
      "--generated-decode-source", type=Path, default=DEFAULT_GENERATED_DECODE_SOURCE
  )
  parser.add_argument("--generate-result", type=Path, default=DEFAULT_GENERATE_RESULT)
  parser.add_argument("--seq73-metrics", type=Path, default=DEFAULT_SEQ73)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  d = result["derived"]
  print("linear preconv attention-norm handle gate")
  print(f"  artifact: {out_dir}")
  print(
      "  source checks: "
      f"{d['all_source_checks_present']} ; decode default-off: "
      f"{d['decode_default_off']}"
  )
  print(
      "  carrier precondition ready: "
      f"{d['attn_norm_handle_carrier_precondition_ready']}"
  )
  print(f"  verdict: {result['verdict']['next_route']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
