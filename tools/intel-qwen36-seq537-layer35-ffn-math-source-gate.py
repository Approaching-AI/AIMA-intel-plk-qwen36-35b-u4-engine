#!/usr/bin/env python3
"""Classify the layer-35 FFN math source gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-seq537-layer35-ffn-math-source-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ536 = (
    ROOT
    / "output/seq536-layer36-layer-input-source-gate-20260709Tseq536Z"
    / "metrics.json"
)
DEFAULT_MATH = (
    ROOT
    / "output/seq537-layer35-ffn-math-source-math-20260709Tseq537Z"
    / "result.json"
)
DEFAULT_CODE = (
    ROOT
    / "output/seq537-layer35-ffn-math-source-code-20260709Tseq537Z"
    / "result.json"
)
DEFAULT_FALLBACK_MATH = (
    ROOT
    / "output/seq537-layer35-cpu-fallback-math-20260709Tseq537Z"
    / "result.json"
)
DEFAULT_FALLBACK_CODE = (
    ROOT
    / "output/seq537-layer35-cpu-fallback-code-20260709Tseq537Z"
    / "result.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/seq537-layer35-ffn-math-source-gate-20260709Tseq537Z"
)

CURRENT_ROUTE = "router_prompt_distribution_layer35_ffn_math_source_gate"
NEXT_ROUTE = "router_prompt_distribution_layer35_ffn_input_source_gate"
REJECTED_ROUTE = "router_prompt_distribution_layer35_ffn_math_source"
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SOURCE_LAYER = 35
COSINE_THRESHOLD = 0.9999
MAX_GPU_CPU_MATH_ABS = 2.0e-5


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


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


def _load_smoke(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  payload = _load_json(path)
  if isinstance(payload, dict) and isinstance(payload.get("smoke"), dict):
    return payload, payload["smoke"]
  if isinstance(payload, dict):
    return payload, payload
  raise TypeError(f"{path} does not contain a JSON object")


def _dist(smoke: dict[str, Any]) -> dict[str, Any]:
  dist = smoke.get("distribution_ladder")
  return dist if isinstance(dist, dict) else {}


def _failed_steps(smoke: dict[str, Any]) -> list[dict[str, Any]]:
  steps = _dist(smoke).get("steps")
  steps = steps if isinstance(steps, list) else []
  return [
      step for step in steps
      if isinstance(step, dict)
      and isinstance(step.get("token_index"), int)
      and (_num(step.get("kld")) > KLD_THRESHOLD
           or step.get("top1_matches") is False)
  ]


def _table_step(smoke: dict[str, Any], table: str, token_index: int) -> dict[str, Any]:
  rows = smoke.get(table)
  rows = rows if isinstance(rows, list) else []
  for row in rows:
    if isinstance(row, dict) and row.get("token_index") == token_index:
      return row
  return {}


def _layer(step: dict[str, Any], layer: int) -> dict[str, Any]:
  rows = step.get("layers")
  rows = rows if isinstance(rows, list) else []
  for row in rows:
    if isinstance(row, dict) and row.get("layer") == layer:
      return row
  return {}


def _base_row(path: Path, label: str) -> dict[str, Any]:
  payload, smoke = _load_smoke(path)
  dist = _dist(smoke)
  entries: list[dict[str, Any]] = []
  for step in _failed_steps(smoke):
    token_index = int(step["token_index"])
    live = _layer(_table_step(smoke, "ffn_live_math_diff_by_step", token_index),
                  SOURCE_LAYER)
    residual = _layer(
        _table_step(smoke, "residual_source_diff_by_step", token_index),
        SOURCE_LAYER)
    entries.append({
        "token_index": token_index,
        "kld": step.get("kld"),
        "top1_matches": step.get("top1_matches"),
        "gpu_output_vs_cpu_ffn_cosine":
            live.get("gpu_output_vs_cpu_ffn_cosine"),
        "gpu_output_vs_cpu_ffn_max_abs_diff":
            live.get("gpu_output_vs_cpu_ffn_max_abs_diff"),
        "ffn_norm_gpu_vs_cpu_max_abs_diff":
            live.get("ffn_norm_gpu_vs_cpu_max_abs_diff"),
        "layer35_ffn_input_cosine": residual.get("ffn_input_cosine"),
        "layer35_ffn_input_max_abs_diff": residual.get("ffn_input_max_abs_diff"),
        "layer35_layer_output_cosine": residual.get("residual_cosine"),
        "layer35_layer_output_max_abs_diff":
            residual.get("residual_max_abs_diff"),
    })

  def min_metric(name: str, default: float = 1.0) -> float:
    return min((_num(row.get(name), default) for row in entries), default=default)

  def max_metric(name: str) -> float:
    return max((_num(row.get(name)) for row in entries), default=0.0)

  mismatch_count = sum(
      1 for row in entries
      if _num(row.get("gpu_output_vs_cpu_ffn_max_abs_diff")) >
      MAX_GPU_CPU_MATH_ABS or _num(row.get("gpu_output_vs_cpu_ffn_cosine"), 1.0)
      < COSINE_THRESHOLD)
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "full_attention_state_diff_enabled": smoke.get(
          "full_attention_state_diff_enabled"),
      "diagnostic_layer_range": payload.get("diagnostic_layer_range"),
      "failed_step_count": len(entries),
      "failed_steps": entries,
      "max_gpu_output_vs_cpu_ffn_abs_diff":
          max_metric("gpu_output_vs_cpu_ffn_max_abs_diff"),
      "min_gpu_output_vs_cpu_ffn_cosine":
          min_metric("gpu_output_vs_cpu_ffn_cosine"),
      "live_ffn_math_mismatch_failed_step_count": mismatch_count,
      "min_layer35_ffn_input_cosine":
          min_metric("layer35_ffn_input_cosine"),
      "max_layer35_ffn_input_abs_diff":
          max_metric("layer35_ffn_input_max_abs_diff"),
  }


def _fallback_row(path: Path, label: str, base_max_kld: float) -> dict[str, Any]:
  payload, smoke = _load_smoke(path)
  dist = _dist(smoke)
  failed = _failed_steps(smoke)
  max_kld = _num(dist.get("max_kld"))
  return {
      "label": label,
      "path": _rel(path),
      "case_id": smoke.get("case_id"),
      "target_returncode": payload.get("target", {}).get(
          "run", {}).get("returncode"),
      "required_checks_passed": smoke.get("required_checks_passed"),
      "distribution_required_checks_passed": dist.get("required_checks_passed"),
      "max_kld": dist.get("max_kld"),
      "mean_kld": dist.get("mean_kld"),
      "top1_rate": dist.get("top1_rate"),
      "top1_pass": dist.get("top1_pass"),
      "kld_pass": dist.get("kld_pass"),
      "cpu_layer_fallback_layer_ids": smoke.get("cpu_layer_fallback_layer_ids"),
      "cpu_layer_fallback_layers": smoke.get("cpu_layer_fallback_layers"),
      "failed_step_count": len(failed),
      "failed_steps": [
          {
              "token_index": step.get("token_index"),
              "kld": step.get("kld"),
              "top1_matches": step.get("top1_matches"),
          }
          for step in failed
      ],
      "max_kld_delta_vs_base": max_kld - base_max_kld,
  }


def _base_row_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and row.get("full_attention_state_diff_enabled") is True
      and row.get("diagnostic_layer_range") == "35:37")


def _fallback_row_ran(row: dict[str, Any]) -> bool:
  return (
      row.get("target_returncode") == 2
      and row.get("distribution_required_checks_passed") is False
      and row.get("kld_pass") is False
      and row.get("top1_pass") is True
      and _num(row.get("top1_rate")) >= TOP1_THRESHOLD
      and _num(row.get("max_kld")) > KLD_THRESHOLD
      and row.get("cpu_layer_fallback_layer_ids") == [SOURCE_LAYER])


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq536 = _load_json(args.seq536)
  math_row = _base_row(args.math, "seq537_layer35_ffn_math_source_math")
  code_row = _base_row(args.code, "seq537_layer35_ffn_math_source_code")
  fallback_math = _fallback_row(
      args.fallback_math, "seq537_layer35_cpu_fallback_math",
      _num(math_row.get("max_kld")))
  fallback_code = _fallback_row(
      args.fallback_code, "seq537_layer35_cpu_fallback_code",
      _num(code_row.get("max_kld")))

  fallback_still_fails = (
      _fallback_row_ran(fallback_math)
      and _fallback_row_ran(fallback_code))
  ffn_input_drift_remains = (
      min(_num(math_row.get("min_layer35_ffn_input_cosine"), 1.0),
          _num(code_row.get("min_layer35_ffn_input_cosine"), 1.0)) <
      COSINE_THRESHOLD)

  checks = [
      {
          "name": "seq536_selected_layer35_ffn_math_source_route",
          "pass": (
              seq536.get("required_checks_passed") is True
              and seq536.get("selected_next_route") == CURRENT_ROUTE
              and _has_candidate(
                  routes, 536,
                  "select_layer35_ffn_math_source_due_live_ffn_math_mismatch")
              and _has_switch(
                  routes,
                  "select_router_prompt_distribution_layer35_ffn_math_source_gate",
                  536)),
      },
      {
          "name": "layer35_ffn_live_math_rows_target_ran",
          "pass": _base_row_ran(math_row) and _base_row_ran(code_row),
          "detail": {"math": math_row, "code": code_row},
      },
      {
          "name": "layer35_cpu_fallback_does_not_clear_distribution",
          "pass": fallback_still_fails,
          "detail": {
              "math": fallback_math,
              "code": fallback_code,
          },
      },
      {
          "name": "layer35_ffn_input_drift_remains_after_rejecting_math",
          "pass": ffn_input_drift_remains,
          "detail": {
              "cosine_threshold": COSINE_THRESHOLD,
              "math_min_layer35_ffn_input_cosine":
                  math_row.get("min_layer35_ffn_input_cosine"),
              "code_min_layer35_ffn_input_cosine":
                  code_row.get("min_layer35_ffn_input_cosine"),
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq536": _rel(args.seq536),
          "math": _rel(args.math),
          "code": _rel(args.code),
          "fallback_math": _rel(args.fallback_math),
          "fallback_code": _rel(args.fallback_code),
      },
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "layer35_ffn_math_source_gate_allowed": required,
      "layer35_ffn_input_source_gate_allowed": required,
      "rows": {
          "math": math_row,
          "code": code_row,
          "fallback_math": fallback_math,
          "fallback_code": fallback_code,
      },
      "combined": {
          "base_failed_step_count":
              math_row["failed_step_count"] + code_row["failed_step_count"],
          "fallback_failed_step_count":
              fallback_math["failed_step_count"] +
              fallback_code["failed_step_count"],
          "live_ffn_math_mismatch_failed_step_count":
              math_row["live_ffn_math_mismatch_failed_step_count"] +
              code_row["live_ffn_math_mismatch_failed_step_count"],
          "max_live_gpu_output_vs_cpu_ffn_abs_diff":
              max(_num(math_row.get("max_gpu_output_vs_cpu_ffn_abs_diff")),
                  _num(code_row.get("max_gpu_output_vs_cpu_ffn_abs_diff"))),
          "min_layer35_ffn_input_cosine":
              min(_num(math_row.get("min_layer35_ffn_input_cosine"), 1.0),
                  _num(code_row.get("min_layer35_ffn_input_cosine"), 1.0)),
          "math_fallback_max_kld": fallback_math.get("max_kld"),
          "code_fallback_max_kld": fallback_code.get("max_kld"),
      },
      "disposition": (
          "reject_layer35_ffn_math_source_select_layer35_ffn_input_source"
          if required else
          "block_layer35_ffn_math_source_inconsistent_evidence"),
      "rejected_route": REJECTED_ROUTE if required else None,
      "selected_next_route": NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Layer35 CPU fallback does not clear router math/code distribution "
          "and worsens code max KLD, while layer35 FFN-input drift remains on "
          "failed steps. Reject layer35 FFN math as the correction source and "
          "attribute layer35 FFN input before product correction, speed "
          "promotion, or long-context rows."
          if required else
          "Layer35 FFN math source evidence is inconsistent; do not switch "
          "routes or run speed/promotion/long-context rows."),
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
  rows = metrics["rows"]
  lines = [
      "# Seq537 Layer35 FFN Math Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- rejected_route: `{metrics['rejected_route']}`",
      f"- failed_checks: `{failed}`",
      f"- live_ffn_math_mismatch_failed_step_count: `{metrics['combined']['live_ffn_math_mismatch_failed_step_count']}`",
      f"- min_layer35_ffn_input_cosine: `{metrics['combined']['min_layer35_ffn_input_cosine']}`",
      "",
      "## Evidence",
      "",
      f"- base math/code max KLD: `{rows['math']['max_kld']}` / `{rows['code']['max_kld']}`",
      f"- layer35 CPU fallback math/code max KLD: `{rows['fallback_math']['max_kld']}` / `{rows['fallback_code']['max_kld']}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is correctness-route evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq536", type=Path, default=DEFAULT_SEQ536)
  parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
  parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
  parser.add_argument("--fallback-math", type=Path, default=DEFAULT_FALLBACK_MATH)
  parser.add_argument("--fallback-code", type=Path, default=DEFAULT_FALLBACK_CODE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
