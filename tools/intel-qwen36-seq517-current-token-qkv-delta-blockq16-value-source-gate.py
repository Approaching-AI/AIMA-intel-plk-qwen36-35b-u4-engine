#!/usr/bin/env python3
"""Close the block-q16 value-source route unless a real product source exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-seq517-current-token-qkv-delta-blockq16-"
    "value-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ516 = (
    ROOT
    / "output/seq516-current-token-qkv-delta-blockq16-distribution-fix-gate-20260709Tseq516Z"
    / "metrics.json"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq517-current-token-qkv-delta-blockq16-value-source-gate-20260709Tseq517Z"
)

SELECTED_NEXT_ROUTE = "router_prompt_distribution_route_switch_gate"
REQUIRED_CLOSED_ROUTES = {
    "router_math_static_or_lagged_qkv_delta_predictors",
    "router_math_live_round_or_selected_affine_qkv_delta_approximation",
    "selected_layer_input_recursive_source_value_chase",
    "qkv_delta_producer_mapped_replacement_overlay_as_product_fix",
}


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


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _line_of(text: str, pattern: str, *, regex: bool = True) -> int | None:
  if regex:
    match = re.search(pattern, text, flags=re.S | re.M)
    if match is None:
      return None
    return text.count("\n", 0, match.start()) + 1
  index = text.find(pattern)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def _present(text: str, label: str, pattern: str, *,
             regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "present": line is not None, "line": line}


def _absent(text: str, label: str, pattern: str, *,
            regex: bool = True) -> dict[str, Any]:
  line = _line_of(text, pattern, regex=regex)
  return {"label": label, "absent": line is None, "line": line}


def _all_present(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("present") is True for row in rows)


def _all_absent(rows: list[dict[str, Any]]) -> bool:
  return all(row.get("absent") is True for row in rows)


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _function_body(source: str) -> str:
  match = re.search(
      r"std::uint64_t DecodeRouterQkvDeltaBlockQ16SourceHandle\(.*?"
      r"\n}\n\nstd::uint64_t DecodeElapsedNs",
      source,
      flags=re.S)
  return match.group(0) if match else ""


def _blockq16_source_state(source: str) -> dict[str, Any]:
  body = _function_body(source)
  zero_value_checks = [
      _present(
          body,
          "zero_selected_q_delta_vector",
          r"std::vector<std::int16_t>\s+selected_q_delta"
          r"\(selected_indices\.size\(\),\s*0\);"),
      _present(
          body,
          "zero_block_scale_vector",
          r"std::vector<float>\s+block_scales"
          r"\(\(kHiddenSize \+ 63\) / 64,\s*0\.0f\);"),
  ]
  real_source_absent_checks = [
      _absent(body, "no_selected_q_delta_assignment",
              r"selected_q_delta\[[^\]]+\]\s*="),
      _absent(body, "no_block_scales_assignment",
              r"block_scales\[[^\]]+\]\s*="),
      _absent(body, "no_quantized_delta_loop",
              "std::round(delta / scale)", regex=False),
      _absent(body, "no_shadow_trace_in_product_handle",
              "g_decode_cpu_shadow_trace", regex=False),
  ]
  cpu_shadow_incompatible_checks = [
      _present(
          source,
          "blockq16_rejects_cpu_shadow_state",
          r"args\.router_qkv_delta_blockq16_source\).*?"
          r"!args\.cpu_shadow_state_each_token.*?"
          r"args\.cpu_shadow_layer_input_delta_layers\.empty\(\).*?"
          r"IQ36_ROUTER_QKV_DELTA_BLOCKQ16_SOURCE is incompatible "
          r"with CPU-shadow values"),
  ]
  return {
      "function_found": bool(body),
      "zero_value_checks": zero_value_checks,
      "real_source_absent_checks": real_source_absent_checks,
      "cpu_shadow_incompatible_checks": cpu_shadow_incompatible_checks,
      "zero_noop_and_no_real_source": (
          bool(body)
          and _all_present(zero_value_checks)
          and _all_absent(real_source_absent_checks)),
      "cpu_shadow_incompatible": _all_present(cpu_shadow_incompatible_checks),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq516 = _load_json(args.seq516)
  source_text = _read(args.decode_source)
  source_state = _blockq16_source_state(source_text)
  rejected_names = _rejected_names(rejected)
  missing_closed = sorted(REQUIRED_CLOSED_ROUTES - rejected_names)

  checks = [
      {
          "name": "seq516_selected_value_source_gate",
          "pass": (
              seq516.get("required_checks_passed") is True
              and seq516.get("disposition")
              == "accept_blockq16_distribution_fix_select_value_source"
              and seq516.get("selected_next_route")
              == "router_prompt_all_linear_current_token_qkv_delta_blockq16_value_source_gate"
              and _has_candidate(
                  routes, 516,
                  "accept_blockq16_distribution_fix_select_value_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_current_token_qkv_delta_blockq16_value_source_gate",
                  516)),
      },
      {
          "name": "blockq16_product_path_has_no_real_value_source",
          "pass": source_state["zero_noop_and_no_real_source"],
          "detail": source_state,
      },
      {
          "name": "blockq16_contract_forbids_cpu_shadow_values",
          "pass": source_state["cpu_shadow_incompatible"],
          "detail": source_state["cpu_shadow_incompatible_checks"],
      },
      {
          "name": "known_value_source_substitutes_are_closed",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq516": _rel(args.seq516),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
      },
      "source_state": source_state,
      "closed_value_source_substitutes": sorted(REQUIRED_CLOSED_ROUTES),
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "router_distribution_allowed": False,
      "decode_probe_allowed": False,
      "disposition": (
          "reject_blockq16_value_source_no_product_source_select_route_switch"
          if required else
          "block_before_blockq16_value_source_route_switch"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_value_source_gate"),
      "next_route_reason": (
          "The block-q16 correction still has no legal product value source: "
          "its product path emits zero q_delta/block scales, CPU-shadow values "
          "are explicitly incompatible, and known static/lagged/rounding/"
          "affine/recursive/producer-mapped substitutes are closed. Close this "
          "route unless a new non-shadow value-source proof is introduced; the "
          "next unit is a router-distribution route switch."
          if required else
          "Value-source closure is not proven; do not switch routes or run "
          "speed rows from this state."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  lines = [
      "# Seq517 Current-Token QKV-Delta Block-Q16 Value Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control/correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq516", type=Path, default=DEFAULT_SEQ516)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
