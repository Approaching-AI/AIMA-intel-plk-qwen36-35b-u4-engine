#!/usr/bin/env python3
"""Gate decode wiring for FFN-tail resident-input carrier.

This is source-contract evidence, not runtime speed evidence. It proves the
decode harness has an env-gated path that calls the seq61 resident-input
FFN-tail primitive only when the FFN norm handle, attention residual handle,
selected/shared down handles, and resident F32 shared-gate weights all exist.
The path is default-off and must not be counted as a speed row without an
explore/promotion run.
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
SCHEMA_VERSION = "intel-qwen36-ffn-tail-resident-input-decode-gate-v0"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ61 = (
    ROOT
    / "output/ffn-tail-resident-input-contract-gate-20260706Tseq61Z/metrics.json"
)
DEFAULT_SEQ62 = (
    ROOT
    / "output/q6-selected-shared-defer-contract-gate-20260706Tseq62Z/metrics.json"
)
DEFAULT_OUT_DIR = ROOT / "output/ffn-tail-resident-input-decode-gate-20260706Tseq63Z"


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
  source = args.decode_source.read_text(encoding="utf-8")
  seq61 = _load_json(args.seq61_metrics)
  seq62 = _load_json(args.seq62_metrics)
  checks = [
      _check(
          source,
          r"bool g_decode_ffn_tail_resident_input = false;",
          "decode_env_gate_defaults_off",
      ),
      _check(
          source,
          r"g_decode_ffn_tail_resident_input =\s*"
          r"std::getenv\(\"IQ36_FFN_TAIL_RESIDENT_INPUT\"\) != nullptr;",
          "decode_env_gate_reads_iq36_ffn_tail_resident_input",
      ),
      _check(
          source,
          r"\"IQ36_FFN_TAIL_RESIDENT_INPUT\"",
          "remote_env_propagates_iq36_ffn_tail_resident_input",
      ),
      _check(
          source,
          r"std::uint64_t ffn_residual_handle_for_tail =\s*"
          r"attention_gpu\.attn_residual_handle;",
          "linear_attention_residual_handle_threaded_to_tail",
      ),
      _check(
          source,
          r"std::uint64_t ffn_residual_handle_for_tail = "
          r"handoff\.residual_handle;",
          "full_attention_handoff_residual_handle_threaded_to_tail",
      ),
      _check(
          source,
          r"ffn_residual_handle_for_tail = 0;\s*"
          r"ffn_norm_handle_for_tail = 0;",
          "diagnostic_shadow_paths_clear_tail_handles",
      ),
      _check(
          source,
          r"RunGpuHybridFfnTail\([^)]*"
          r"std::uint64_t ffn_input_handle = 0,\s*"
          r"std::uint64_t attention_residual_handle = 0\)",
          "tail_helper_accepts_optional_resident_handles",
      ),
      _check(
          source,
          r"attention_residual_used != &attention_residual\) \{\s*"
          r"attention_residual_handle = 0;",
          "tail_helper_clears_residual_handle_after_substitution",
      ),
      _check(
          source,
          r"const bool use_resident_input_tail =\s*"
          r"g_decode_ffn_tail_resident_input &&\s*"
          r"ffn_input_handle != 0 &&\s*"
          r"attention_residual_handle != 0 &&\s*"
          r"g_decode_resident_f32_weights != nullptr;",
          "resident_input_tail_requires_all_handles_and_f32_weights",
      ),
      _check(
          source,
          r"g_decode_resident_f32_weights->TensorHandle\(\s*"
          r"model, \*ffn_tensors\.shared_input_gate_tensor, 1,\s*"
          r"kHiddenSize\)",
          "resident_input_tail_uploads_shared_gate_f32_handle",
      ),
      _check(
          source,
          r"RunFfnTailFromDownHandlesResidentInputs\(\s*"
          r"shared_gate_handle, ffn_input_handle,\s*"
          r"selected_gpu\.down_handle, router\.normalized_weights,\s*"
          r"shared_gpu\.down_handle, attention_residual_handle",
          "resident_input_tail_calls_seq61_primitive",
      ),
      _check(
          source,
          r"ffn_tail_resident_input_enabled",
          "target_stdout_reports_resident_input_tail_env",
      ),
  ]
  all_checks_pass = all(check["present"] for check in checks)
  seq61_ready = bool(
      seq61.get("derived", {}).get("primitive_ready_for_carrier_wiring")
  )
  seq62_ready = bool(
      seq62.get("derived", {}).get("primitive_ready_for_carrier_wiring")
  )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "decode_source": {
              "path": _display_path(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "seq61_metrics": {
              "path": _display_path(args.seq61_metrics),
              "sha256": _sha256(args.seq61_metrics),
          },
          "seq62_metrics": {
              "path": _display_path(args.seq62_metrics),
              "sha256": _sha256(args.seq62_metrics),
          },
      },
      "checks": checks,
      "derived": {
          "all_contract_checks_pass": all_checks_pass,
          "seq61_resident_tail_ready": seq61_ready,
          "seq62_q6_defer_ready": seq62_ready,
          "default_behavior_preserved": all_checks_pass,
          "decode_path_ready_for_explore": (
              all_checks_pass and seq61_ready and seq62_ready
          ),
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": False,
          "reason": (
              "Decode can now opt into the resident-input FFN-tail path via "
              "IQ36_FFN_TAIL_RESIDENT_INPUT, but default decode remains "
              "unchanged and no token-emitting measurement is recorded here."
          ),
          "next_route": (
              "The next admissible step is one explore row with the full "
              "selected+shared Q4/Q6 stack plus this env gate, then either "
              "promotion evidence if it clears noise or a rejection record."
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
      "tool": "tools/intel-qwen36-ffn-tail-resident-input-decode-gate.py",
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
      "# FFN-Tail Resident-Input Decode Gate",
      "",
      "This is source-contract evidence, not runtime speed evidence.",
      "",
      "## Checks",
      "",
      f"- contract checks pass: `{str(d['all_contract_checks_pass']).lower()}`",
      f"- seq61 resident tail ready: `{str(d['seq61_resident_tail_ready']).lower()}`",
      f"- seq62 Q6 defer ready: `{str(d['seq62_q6_defer_ready']).lower()}`",
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
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq61-metrics", type=Path, default=DEFAULT_SEQ61)
  parser.add_argument("--seq62-metrics", type=Path, default=DEFAULT_SEQ62)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  write_outputs(result, args.out_dir)
  d = result["derived"]
  print("FFN-tail resident-input decode gate")
  print(f"all_contract_checks_pass={d['all_contract_checks_pass']}")
  print(f"default_behavior_preserved={d['default_behavior_preserved']}")
  print(f"decode_path_ready_for_explore={d['decode_path_ready_for_explore']}")
  print(f"artifact={_display_path(args.out_dir)}")
  if not d["decode_path_ready_for_explore"]:
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
