#!/usr/bin/env python3
"""Route-control gate after qkv-delta product-consumer distribution rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = (
    "intel-qwen36-router-qkv-delta-product-consumer-distribution-fix-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ316 = (
    ROOT
    / "output/router-qkv-delta-product-consumer-router-distribution-gate-20260708Tseq316Z"
    / "metrics.json"
)
DEFAULT_MATH_RESIDUAL = (
    ROOT
    / "output/r2-gpu-router-math-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-residual-only-20260708Tseq272Z"
    / "result.json"
)
DEFAULT_CODE_RESIDUAL = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-residual-only-20260708Tseq317diagZ"
    / "result.json"
)
DEFAULT_CODE_LAYER35 = (
    ROOT
    / "output/r2-gpu-router-code-distribution-rowblock16-26mask-double-swiglu-shadow-prev-full-ffn-residual-only-layer35-20260708Tseq317diagZ"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-product-consumer-distribution-fix-gate-20260708Tseq317Z"
)

KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


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


def _smoke(path: Path) -> dict[str, Any]:
  payload = _load_json(path)
  smoke = payload.get("smoke")
  return smoke if isinstance(smoke, dict) else payload


def _dist_summary(path: Path) -> dict[str, Any]:
  smoke = _smoke(path)
  dist = smoke.get("distribution_ladder")
  dist = dist if isinstance(dist, dict) else {}
  return {
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "cpu_shadow_state_each_token_enabled": smoke.get(
          "cpu_shadow_state_each_token_enabled"),
      "cpu_shadow_ffn_input_layers": smoke.get("cpu_shadow_ffn_input_layers"),
      "cpu_shadow_ffn_input_layer_ids": smoke.get(
          "cpu_shadow_ffn_input_layer_ids"),
      "cpu_shadow_ffn_input_residual_only_enabled": smoke.get(
          "cpu_shadow_ffn_input_residual_only_enabled"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_match_count": dist.get("top1_match_count"),
      "position_count": dist.get("position_count"),
  }


def _dist_pass(row: dict[str, Any]) -> bool:
  return (
      row.get("distribution_required_checks_passed") is True
      and _num(row.get("max_kld")) <= KLD_THRESHOLD
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
  )


def _dist_fail(row: dict[str, Any]) -> bool:
  return (
      row.get("distribution_required_checks_passed") is False
      or _num(row.get("max_kld")) > KLD_THRESHOLD
      or _num(row.get("top1_rate")) < TOP1_THRESHOLD
  )


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq316 = _load_json(args.seq316)
  math_residual = _dist_summary(args.math_residual)
  code_residual = _dist_summary(args.code_residual)
  code_layer35 = _dist_summary(args.code_layer35)

  seq316_runs = seq316.get("runs")
  seq316_runs = seq316_runs if isinstance(seq316_runs, list) else []
  seq316_ready = all(
      isinstance(row, dict)
      and row.get("summary", {}).get("producer_layers") == 72
      and row.get("summary", {}).get("producer_values") == 147456
      and row.get("summary", {}).get("producer_misses") == 0
      and row.get("summary", {}).get("product_layers") == 216
      and row.get("summary", {}).get("product_values") == 110592
      and row.get("summary", {}).get("product_misses") == 24
      for row in seq316_runs
  )
  seq316_failed_distribution = any(
      isinstance(row, dict)
      and (
          _num(row.get("summary", {}).get("distribution", {}).get("max_kld"))
          > KLD_THRESHOLD
          or _num(row.get("summary", {}).get("distribution", {}).get("top1_rate"))
          < TOP1_THRESHOLD
      )
      for row in seq316_runs
  )

  checks = [
      {
          "name": "seq316_selected_distribution_fix_gate",
          "pass": (
              seq316.get("required_checks_passed") is False
              and seq316.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_product_consumer_distribution_fix_gate"
              and seq316_ready
              and seq316_failed_distribution
              and _has_candidate(
                  routes, 316,
                  "reject_qkv_delta_product_consumer_router_distribution")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_product_consumer_distribution_fix_gate",
                  316)
          ),
      },
      {
          "name": "all_full_attention_residual_only_passes_math_and_code",
          "pass": (
              _dist_pass(math_residual)
              and _dist_pass(code_residual)
              and math_residual.get("cpu_shadow_ffn_input_layers") == 72
              and code_residual.get("cpu_shadow_ffn_input_layers") == 72
              and math_residual.get("cpu_shadow_ffn_input_residual_only_enabled")
                  is True
              and code_residual.get("cpu_shadow_ffn_input_residual_only_enabled")
                  is True
          ),
          "detail": {
              "math_residual": math_residual,
              "code_residual": code_residual,
          },
      },
      {
          "name": "layer35_only_is_not_enough_for_router_code",
          "pass": (
              _dist_fail(code_layer35)
              and code_layer35.get("cpu_shadow_ffn_input_layers") == 8
              and code_layer35.get("cpu_shadow_ffn_input_layer_ids") == [35]
          ),
          "detail": {"code_layer35": code_layer35},
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq316": _rel(args.seq316),
          "math_residual": _rel(args.math_residual),
          "code_residual": _rel(args.code_residual),
          "code_layer35": _rel(args.code_layer35),
      },
      "seq316": {
          "ready_counters": seq316_ready,
          "failed_distribution": seq316_failed_distribution,
          "runs": [
              row.get("summary", {}) for row in seq316_runs
              if isinstance(row, dict)
          ],
      },
      "residual_rows": {
          "math_residual": math_residual,
          "code_residual": code_residual,
          "code_layer35": code_layer35,
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "close_qkv_delta_product_consumer_source_select_full_attention_residual_product_source"
          if required else
          "block_before_distribution_fix_route_switch"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_product_source_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_product_consumer_distribution_fix_gate"
      ),
      "next_route_reason": (
          "The qkv-delta product consumer has ready counters but does not "
          "reproduce the passing distribution signal. The source-of-truth "
          "diagnostic is full-attention FFN residual input on all nine prior "
          "full-attention layers; layer35 alone is insufficient for router_code. "
          "Switch to a full-attention residual product source gate before any "
          "speed or long-context promotion."
          if required else
          "Distribution-fix evidence is inconsistent; do not switch routes yet."
      ),
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
  code = metrics["residual_rows"]["code_residual"]
  layer35 = metrics["residual_rows"]["code_layer35"]
  lines = [
      "# Router QKV Delta Product Consumer Distribution Fix Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- code residual max KLD: `{code.get('max_kld')}`",
      f"- code layer35-only max KLD: `{layer35.get('max_kld')}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
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
  parser.add_argument("--seq316", type=Path, default=DEFAULT_SEQ316)
  parser.add_argument("--math-residual", type=Path,
                      default=DEFAULT_MATH_RESIDUAL)
  parser.add_argument("--code-residual", type=Path,
                      default=DEFAULT_CODE_RESIDUAL)
  parser.add_argument("--code-layer35", type=Path,
                      default=DEFAULT_CODE_LAYER35)
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
