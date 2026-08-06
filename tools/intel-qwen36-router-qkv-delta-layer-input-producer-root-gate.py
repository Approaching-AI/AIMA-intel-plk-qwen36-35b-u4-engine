#!/usr/bin/env python3
"""Gate the qkv-delta layer-input producer root route.

This is route-control evidence. It consumes the prior full-attention residual
diagnostics plus the all-linear qkv-delta lower bound and selects the next
source unit without launching a token row.
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
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-router-qkv-delta-layer-input-producer-root-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ270 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-input-20260708Tseq270Z"
)
DEFAULT_SEQ271 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-norm-only-20260708Tseq271Z"
)
DEFAULT_SEQ272 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-residual-only-20260708Tseq272Z"
)
DEFAULT_SEQ273 = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-fullattn-q4cpuorder-prev-boundary-20260708Tseq273Z"
)
DEFAULT_SEQ291 = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-all-linear-qkvcol-delta-blockq16-top512-20260708Tseq291Z"
)
DEFAULT_SEQ297 = (
    ROOT
    / "output/router-qkv-delta-current-token-value-source-gate-20260708Tseq297Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-root-gate-20260708Tseq298Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
PRIOR_FULL_ATTENTION_PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
DECODE_TOKENS = 8
TOPK = 512
HIDDEN_SIZE = 2048
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99


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


def _summary_metric(summary: str, label: str) -> float:
  match = re.search(rf"^- {re.escape(label)}: `([^`]+)`", summary, re.M)
  if match is None:
    return 0.0
  try:
    return float(match.group(1))
  except ValueError:
    return 0.0


def _summary_bool(summary: str, label: str) -> bool:
  match = re.search(rf"^- {re.escape(label)}: `([^`]+)`", summary, re.M)
  return bool(match and match.group(1).lower() == "true")


def _artifact_summary(path: Path) -> dict[str, Any]:
  summary_path = path / "SUMMARY.md"
  result_path = path / "result.json"
  summary = _read(summary_path)
  result = _load_json(result_path)
  smoke = result.get("smoke")
  smoke = smoke if isinstance(smoke, dict) else {}
  ladder = smoke.get("distribution_ladder")
  ladder = ladder if isinstance(ladder, dict) else {}
  return {
      "artifact": _rel(path),
      "summary_sha256": _sha256(summary_path),
      "result_sha256": _sha256(result_path),
      "required_checks_passed": _summary_bool(summary, "required checks passed"),
      "distribution_ladder_passed": _summary_bool(
          summary, "distribution ladder passed"),
      "max_kld": _summary_metric(summary, "distribution max KLD"),
      "top1_rate": _summary_metric(summary, "distribution top-1 rate"),
      "min_logit_cosine": _summary_metric(
          summary, "distribution min logit cosine"),
      "case_id": smoke.get("case_id"),
      "cpu_shadow_ffn_input_layers": int(_num(
          smoke.get("cpu_shadow_ffn_input_layers"))),
      "cpu_shadow_ffn_input_layer_ids": smoke.get("cpu_shadow_ffn_input_layer_ids"),
      "cpu_shadow_layer_input_delta_values": int(_num(
          smoke.get("cpu_shadow_layer_input_delta_values"))),
      "top1_match_count": int(_num(ladder.get("top1_match_count"))),
  }


def _pass_distribution(row: dict[str, Any]) -> bool:
  return (
      row["required_checks_passed"] is True
      and row["distribution_ladder_passed"] is True
      and row["max_kld"] < KLD_THRESHOLD
      and row["top1_rate"] >= TOP1_THRESHOLD
  )


def _fail_distribution(row: dict[str, Any]) -> bool:
  return row["max_kld"] >= KLD_THRESHOLD


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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _source_state(text: str) -> dict[str, Any]:
  present = [
      _present(text, "ffn_residual_source_hook_present",
               "attention_residual_used", regex=False),
      _present(text, "ffn_residual_handle_for_tail_present",
               "ffn_residual_handle_for_tail", regex=False),
      _present(text, "full_attention_residual_handle_present",
               "attention_gpu.attn_residual_handle", regex=False),
      _present(text, "layer_output_handle_loop_present",
               "DecodeCarrierLayerOutputHandleLoopActive", regex=False),
      _present(text, "full_attention_core_history_handle_present",
               "resident_hidden_state_carrier_full_attention_core_history_handle",
               regex=False),
      _present(text, "all_linear_qkv_delta_target_present",
               "g_decode_cpu_shadow_layer_input_delta_layers", regex=False),
  ]
  absent = [
      _absent(text, "no_full_attention_residual_product_source_yet",
              "DecodeRouterQkvDeltaFullAttentionResidualSource", regex=False),
      _absent(text, "no_layer_input_producer_source_flag_yet",
              "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE", regex=False),
  ]
  return {
      "producer_hook_primitives_present": _all_present(present),
      "producer_source_absent": _all_absent(absent),
      "present_checks": present,
      "absent_checks": absent,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq297 = _load_json(args.seq297)
  decode_source = _read(args.decode_source)
  rejected_names = _rejected_names(rejected)
  source = _source_state(decode_source)
  artifacts = {
      "seq270_full_ffn_input": _artifact_summary(args.seq270),
      "seq271_norm_only": _artifact_summary(args.seq271),
      "seq272_residual_only": _artifact_summary(args.seq272),
      "seq273_q4_cpu_order_projection": _artifact_summary(args.seq273),
      "seq291_all_linear_top512_code": _artifact_summary(args.seq291),
  }
  required_closed = {
      "router_math_carrier_loop_or_q4_cpu_order_full_attention_projection_residual_fix",
      "router_math_split_full_attention_projection_arithmetic_residual_fix",
      "router_math_live_round_or_selected_affine_qkv_delta_approximation",
  }
  missing_closed = sorted(required_closed - rejected_names)
  top512_values = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS * TOPK

  checks = [
      {
          "name": "seq297_selected_layer_input_producer_root_gate",
          "pass": (
              seq297.get("required_checks_passed") is True
              and seq297.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_layer_input_producer_root_gate"
              and _has_candidate(
                  routes, 297,
                  "reject_missing_product_value_source_select_layer_input_producer_root_gate")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_layer_input_producer_root_gate",
                  297)
          ),
      },
      {
          "name": "prior_full_attention_residual_is_passing_root_signal",
          "pass": (
              _pass_distribution(artifacts["seq270_full_ffn_input"])
              and _pass_distribution(artifacts["seq272_residual_only"])
              and artifacts["seq270_full_ffn_input"]["cpu_shadow_ffn_input_layers"]
                  == len(PRIOR_FULL_ATTENTION_PRODUCER_LAYERS) * DECODE_TOKENS
              and artifacts["seq272_residual_only"]["cpu_shadow_ffn_input_layers"]
                  == len(PRIOR_FULL_ATTENTION_PRODUCER_LAYERS) * DECODE_TOKENS
          ),
          "detail": {
              "producer_layers": PRIOR_FULL_ATTENTION_PRODUCER_LAYERS,
              "seq270": artifacts["seq270_full_ffn_input"],
              "seq272": artifacts["seq272_residual_only"],
          },
      },
      {
          "name": "non_residual_product_like_roots_remain_closed",
          "pass": (
              _fail_distribution(artifacts["seq271_norm_only"])
              and _fail_distribution(artifacts["seq273_q4_cpu_order_projection"])
              and not missing_closed
          ),
          "detail": {
              "seq271_norm_only": artifacts["seq271_norm_only"],
              "seq273_q4_cpu_order_projection": (
                  artifacts["seq273_q4_cpu_order_projection"]),
              "missing_closed_routes": missing_closed,
          },
      },
      {
          "name": "all_linear_top512_code_bound_still_requires_coverage",
          "pass": (
              _pass_distribution(artifacts["seq291_all_linear_top512_code"])
              and artifacts["seq291_all_linear_top512_code"]
                  ["cpu_shadow_layer_input_delta_values"] == top512_values
          ),
          "detail": {
              "seq291": artifacts["seq291_all_linear_top512_code"],
              "top512_values_expected": top512_values,
              "all_linear_layers": ALL_LINEAR_LAYERS,
          },
      },
      {
          "name": "source_has_hooks_but_no_product_producer_source",
          "pass": (
              source["producer_hook_primitives_present"]
              and source["producer_source_absent"]),
          "detail": source,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
          "seq270": _rel(args.seq270),
          "seq271": _rel(args.seq271),
          "seq272": _rel(args.seq272),
          "seq273": _rel(args.seq273),
          "seq291": _rel(args.seq291),
          "seq297_gate": _rel(args.seq297),
      },
      "artifacts": artifacts,
      "source": source,
      "producer_root": {
          "root": "prior_full_attention_ffn_residual_input",
          "producer_layers": PRIOR_FULL_ATTENTION_PRODUCER_LAYERS,
          "producer_layer_count": len(PRIOR_FULL_ATTENTION_PRODUCER_LAYERS),
          "decode_tokens": DECODE_TOKENS,
          "root_values": len(PRIOR_FULL_ATTENTION_PRODUCER_LAYERS)
              * DECODE_TOKENS * HIDDEN_SIZE,
      },
      "coverage_bound": {
          "all_linear_layers": ALL_LINEAR_LAYERS,
          "topk": TOPK,
          "hidden_size": HIDDEN_SIZE,
          "decode_tokens": DECODE_TOKENS,
          "top512_values_expected": top512_values,
          "top512_fraction_per_layer": TOPK / HIDDEN_SIZE,
      },
      "checks": checks,
      "required_checks_passed": required,
      "component_product_source_present": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "select_full_attention_residual_layer_input_producer_source_gate"
          if required else
          "block_before_full_attention_residual_layer_input_producer_source_gate"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_layer_input_producer_source_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_layer_input_producer_root_fix_gate"
      ),
      "next_route_reason": (
          "The layer-input producer root is the prior full-attention FFN "
          "residual input. Residual-only CPU-shadow substitution passes while "
          "norm-only and q4 CPU-order projection roots fail; the source has "
          "resident handle hooks but no product producer source. Add a "
          "default-off source gate for this producer before any decode or "
          "router distribution row."
          if required else
          "Layer-input producer root evidence is inconsistent; fix this gate "
          "before source or token rows."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  lines = [
      "# Router QKV Delta Layer-Input Producer Root Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- producer root: `{metrics['producer_root']['root']}`",
      f"- producer layers: `{metrics['producer_root']['producer_layers']}`",
      f"- top512 values: `{metrics['coverage_bound']['top512_values_expected']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control evidence. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq270", type=Path, default=DEFAULT_SEQ270)
  parser.add_argument("--seq271", type=Path, default=DEFAULT_SEQ271)
  parser.add_argument("--seq272", type=Path, default=DEFAULT_SEQ272)
  parser.add_argument("--seq273", type=Path, default=DEFAULT_SEQ273)
  parser.add_argument("--seq291", type=Path, default=DEFAULT_SEQ291)
  parser.add_argument("--seq297", type=Path, default=DEFAULT_SEQ297)
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
