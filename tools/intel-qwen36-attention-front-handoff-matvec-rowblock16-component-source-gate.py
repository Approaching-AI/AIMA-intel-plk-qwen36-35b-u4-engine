#!/usr/bin/env python3
"""Gate rowblock16 output-projection component source wiring.

This is source-only evidence. It verifies that the rowblock16 component path is
wired for a component probe and that no decode route has been enabled.
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
    "component-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ180 = (
    ROOT
    / "output/attention-front-handoff-matvec-kernel-algorithm-component-gate-20260708Tseq180Z"
    / "metrics.json"
)
DEFAULT_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_ENGINE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_PROBE = ROOT / "tools/intel-qwen36-gpu-q4x8-output-projection-probe.py"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/attention-front-handoff-matvec-rowblock16-component-source-gate-20260708Tseq181Z"
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
  for row in routes.get("candidate_history", []):
    if (
        isinstance(row, dict)
        and row.get("seq") == seq
        and row.get("disposition") == disposition
    ):
      return True
  return False


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _markers(text: str, required: list[str]) -> dict[str, Any]:
  missing = [item for item in required if item not in text]
  return {"pass": not missing, "missing": missing, "required": required}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq180 = _load_json(args.seq180)
  header = _read(args.header)
  engine = _read(args.engine_source)
  opencl = _read(args.opencl_source)
  probe = _read(args.probe_source)
  decode = _read(args.decode_source)

  selected_next = "attention_front_handoff_matvec_rowblock16_component_probe_gate"
  target = seq180.get("component_design", {}).get("component_acceptance", {})
  opencl_markers = _markers(opencl, [
      "q4k_x8_matvec_rowblock16_reduce",
      "__local float partial[16]",
      "get_group_id(0)",
      "blocks_per_row != 16U",
  ])
  engine_markers = _markers(engine, [
      "kernel_rowblock16_",
      "CreateNamedKernel(\"q4k_x8_matvec_rowblock16_reduce\")",
      "RunRowblock16Kernel",
      "RunRowblock16(",
      "rowblock16 Q4 matvec requires blocks_per_row == 16",
  ])
  header_markers = _markers(header, ["GpuQ4X8MatvecRun RunRowblock16("])
  probe_markers = _markers(probe, [
      "SCHEMA_VERSION = \"intel-qwen36-gpu-q4x8-output-projection-probe-v1\"",
      "runner.RunRowblock16",
      "linear_attn_out_rowblock16",
      "rowblock16_output_projection_gpu_kernel_min_us",
      "rowblock16_output_projection_matches_oracle",
  ])
  no_decode_markers = [
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16",
      "attention_front_output_projection_rowblock16",
  ]
  present_decode = [marker for marker in no_decode_markers if marker in decode]

  checks = [
      {
          "name": "seq180_selected_this_source_gate",
          "pass": (
              seq180.get("required_checks_passed") is True
              and seq180.get("selected_next_route")
              == "attention_front_handoff_matvec_rowblock16_component_source_gate"
              and _has_candidate(
                  routes,
                  180,
                  "select_attention_front_handoff_matvec_rowblock16_component_source_gate",
              )
              and _has_switch(
                  routes,
                  "select_attention_front_handoff_matvec_rowblock16_component_source_gate",
                  180,
              )
          ),
          "detail": {
              "seq180_disposition": seq180.get("disposition"),
              "seq180_selected_next_route": seq180.get("selected_next_route"),
          },
      },
      {"name": "opencl_rowblock16_kernel_present", **opencl_markers},
      {"name": "engine_rowblock16_component_api_present", **engine_markers},
      {"name": "header_rowblock16_component_api_present", **header_markers},
      {"name": "component_probe_reports_rowblock16_lane", **probe_markers},
      {
          "name": "decode_path_not_enabled",
          "pass": not present_decode,
          "present_decode_markers": present_decode,
      },
      {
          "name": "component_target_preserved",
          "pass": (
              _num(target.get("current_us_per_call")) > 0.0
              and _num(target.get("target_us_per_call")) > 0.0
              and _num(target.get("target_us_per_call"))
              < _num(target.get("current_us_per_call"))
          ),
          "detail": target,
      },
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq180_component_gate": _rel(args.seq180),
          "header": _rel(args.header),
          "header_sha256": _sha256(args.header),
          "engine_source": _rel(args.engine_source),
          "engine_source_sha256": _sha256(args.engine_source),
          "opencl_source": _rel(args.opencl_source),
          "opencl_source_sha256": _sha256(args.opencl_source),
          "probe_source": _rel(args.probe_source),
          "probe_source_sha256": _sha256(args.probe_source),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "disposition": (
          "accept_attention_front_handoff_matvec_rowblock16_component_source"
      ),
      "selected_next_route": selected_next,
      "component_probe_allowed": required_checks_passed,
      "decode_probe_allowed": False,
      "speedup_claims_allowed": False,
      "next_route_reason": (
          "Rowblock16 source wiring is present only for component probing. "
          "Next unit is a target component probe; decode remains forbidden "
          "until rowblock16 matches current rowlane and meets the seq180 target."
      ),
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq180", type=Path, default=DEFAULT_SEQ180)
  parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL)
  parser.add_argument("--probe-source", type=Path, default=DEFAULT_PROBE)
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
      "component_probe_allowed": result["component_probe_allowed"],
  }, sort_keys=True))
  return 0 if result["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
