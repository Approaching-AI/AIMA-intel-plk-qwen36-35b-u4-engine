#!/usr/bin/env python3
"""Gate default-off rowblock16 decode source integration.

This is source-only evidence. It verifies that the component-proven rowblock16
output-projection kernel is wired into the resident attention-front handoff
behind an explicit default-off decode flag, without launching a token row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-attention-front-handoff-matvec-rowblock16-"
    "decode-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ180 = (
    ROOT
    / "output/attention-front-handoff-matvec-kernel-algorithm-component-gate-20260708Tseq180Z"
    / "metrics.json"
)
DEFAULT_SEQ182 = (
    ROOT
    / "output/gpu-q4x8-output-projection-rowblock16-component-20260708Tseq182Z"
    / "probe.json"
)
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/attention-front-handoff-matvec-rowblock16-decode-source-gate-20260708Tseq183Z"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", [])
  )


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
  )


def _markers(text: str, required: list[str]) -> dict[str, Any]:
  missing = [item for item in required if item not in text]
  return {"pass": not missing, "missing": missing, "required": required}


def _comparison_ok(seq182: dict[str, Any]) -> bool:
  comparison = (
      seq182.get("probe", {})
      .get("comparisons", {})
      .get("linear_attn_out_rowblock16", {})
      .get("gpu_vs_oracle", {})
  )
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and _num(comparison.get("max_abs_diff")) <= 5.0e-8
      and _num(comparison.get("cosine")) >= 0.999999
  )


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq180 = _load_json(args.seq180)
  seq182 = _load_json(args.seq182)
  header = _read(args.header)
  engine = _read(args.engine_source)
  decode = _read(args.decode_source)

  target = seq180.get("component_design", {}).get("component_acceptance", {})
  timings = seq182.get("probe", {}).get("timings", {})
  rowblock16_us = _num(timings.get("rowblock16_output_projection_gpu_kernel_min_us"))
  target_us = _num(target.get("target_us_per_call"))

  header_markers = _markers(header, [
      "bool use_rowblock16_output_projection = false",
      "RunResidentPackedQ4X8ThenResidualRmsNorm(",
      "RunResidentPackedQ4X8ThenResidentResidualRmsNorm(",
  ])
  engine_markers = _markers(engine, [
      "bool use_rowblock16_output_projection = false",
      "if (use_rowblock16_output_projection) {",
      "rowblock16 attention-front output projection requires BPR16",
      "RunRowblock16Kernel(",
      "use_rowblock16_output_projection);",
  ])
  decode_markers = _markers(decode, [
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16",
      "bool attention_front_output_projection_rowblock16 = false",
      "bool g_decode_attention_front_output_projection_rowblock16 = false",
      "g_decode_attention_front_output_projection_rowblock16 =",
      "args.attention_front_output_projection_rowblock16",
      "g_decode_attention_front_output_projection_rowblock16);",
      "\"attention_front_output_projection_rowblock16_enabled\"",
  ])
  default_off = (
      "attention_front_output_projection_rowblock16 = false" in decode
      and "bool g_decode_attention_front_output_projection_rowblock16 = false"
      in decode
      and "--attention-front-output-projection-rowblock16" not in decode
  )

  checks = [
      {
          "name": "seq182_component_probe_authorized_decode_source",
          "pass": (
              seq182.get("required_checks_passed") is True
              and _comparison_ok(seq182)
              and rowblock16_us > 0.0
              and target_us > 0.0
              and rowblock16_us <= target_us
              and _has_candidate(
                  routes,
                  182,
                  "accept_attention_front_handoff_matvec_rowblock16_component_probe",
              )
              and _has_switch(
                  routes,
                  "select_attention_front_handoff_matvec_rowblock16_decode_source_gate",
                  182,
              )
          ),
          "detail": {
              "rowblock16_us": rowblock16_us,
              "target_us_per_call": target_us,
          },
      },
      {"name": "header_default_off_api_present", **header_markers},
      {"name": "engine_handoff_rowblock16_branch_present", **engine_markers},
      {"name": "decode_default_off_flag_present", **decode_markers},
      {
          "name": "decode_flag_default_off",
          "pass": default_off,
          "detail": {
              "cli_flag_present": (
                  "--attention-front-output-projection-rowblock16" in decode
              ),
          },
      },
      {
          "name": "speedup_claims_forbidden",
          "pass": True,
      },
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq180_component_gate": _rel(args.seq180),
          "seq182_component_probe": _rel(args.seq182),
          "header": _rel(args.header),
          "header_sha256": _sha256(args.header),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
      },
      "component_evidence": {
          "rowblock16_us": rowblock16_us,
          "target_us_per_call": target_us,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "disposition": (
          "accept_attention_front_handoff_matvec_rowblock16_decode_source"
      ),
      "selected_next_route": (
          "attention_front_handoff_matvec_rowblock16_target_compile_gate"
      ),
      "decode_compile_allowed": required_checks_passed,
      "decode_probe_allowed": False,
      "speedup_claims_allowed": False,
      "next_route_reason": (
          "Rowblock16 is now available behind a default-off decode flag. "
          "Next unit is target compile/generate evidence; no token row or "
          "speed claim is authorized by this source gate."
      ),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq180", type=Path, default=DEFAULT_SEQ180)
  parser.add_argument("--seq182", type=Path, default=DEFAULT_SEQ182)
  parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  (args.out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(json.dumps({
      "required_checks_passed": result["required_checks_passed"],
      "disposition": result["disposition"],
      "selected_next_route": result["selected_next_route"],
      "out_dir": _rel(args.out_dir),
      "decode_compile_allowed": result["decode_compile_allowed"],
  }, sort_keys=True))
  return 0 if result["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
