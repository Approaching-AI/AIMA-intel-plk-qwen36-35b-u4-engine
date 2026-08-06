#!/usr/bin/env python3
"""Classify the block-q16 distribution failure and select the value-source fix."""

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
    "intel-qwen36-seq516-current-token-qkv-delta-blockq16-"
    "distribution-fix-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ507 = (
    ROOT
    / "output/seq507-current-token-qkv-delta-design-gate-20260709Tseq507Z"
    / "metrics.json"
)
DEFAULT_SEQ515 = (
    ROOT
    / "output/seq515-current-token-qkv-delta-blockq16-router-distribution-gate-20260709Tseq515Z"
    / "metrics.json"
)
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq516-current-token-qkv-delta-blockq16-distribution-fix-gate-20260709Tseq516Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
DECODE_TOKENS = 8
TOPK = 512
EXPECTED_LAYERS = len(ALL_LINEAR_LAYERS) * DECODE_TOKENS
EXPECTED_VALUES = EXPECTED_LAYERS * TOPK
KLD_THRESHOLD = 0.005
TOP1_THRESHOLD = 0.99
SELECTED_NEXT_ROUTE = (
    "router_prompt_all_linear_current_token_qkv_delta_blockq16_value_source_gate"
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


def _dist(row: dict[str, Any]) -> dict[str, Any]:
  dist = row.get("distribution")
  return dist if isinstance(dist, dict) else {}


def _seq515_rows(seq515: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for run in seq515.get("runs", []) or []:
    if not isinstance(run, dict):
      continue
    summary = run.get("summary")
    if isinstance(summary, dict):
      rows.append(summary)
  return rows


def _seq515_clean_counters(rows: list[dict[str, Any]]) -> bool:
  return len(rows) == 2 and all(
      row.get("blockq16_layers") == EXPECTED_LAYERS
      and row.get("blockq16_values") == EXPECTED_VALUES
      and row.get("blockq16_misses") == 0
      and row.get("blockq16_ready") is True
      and row.get("cpu_shadow_state_each_token_enabled") is False
      and row.get("cpu_shadow_layer_input_layers") == 0
      and row.get("cpu_shadow_attention_output_layers") == 0
      for row in rows)


def _seq515_distribution_failed(rows: list[dict[str, Any]]) -> bool:
  return len(rows) == 2 and any(
      _num(_dist(row).get("max_kld")) > KLD_THRESHOLD
      or _num(_dist(row).get("top1_rate"), 1.0) < TOP1_THRESHOLD
      for row in rows)


def _case_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [
      {
          "case_id": row.get("case_id"),
          "max_kld": _dist(row).get("max_kld"),
          "top1_rate": _dist(row).get("top1_rate"),
          "min_logits_cosine": _dist(row).get("min_logits_cosine"),
          "blockq16_layers": row.get("blockq16_layers"),
          "blockq16_values": row.get("blockq16_values"),
          "blockq16_misses": row.get("blockq16_misses"),
          "cpu_shadow_state_each_token_enabled": row.get(
              "cpu_shadow_state_each_token_enabled"),
      }
      for row in rows
  ]


def _lower_bound(seq507: dict[str, Any]) -> dict[str, Any]:
  shape = seq507.get("correction_shape")
  shape = shape if isinstance(shape, dict) else {}
  return {
      "required_checks_passed": seq507.get("required_checks_passed"),
      "selected_next_route": seq507.get("selected_next_route"),
      "layers": shape.get("layers"),
      "layer_count": shape.get("layer_count"),
      "topk": shape.get("topk"),
      "selector": shape.get("selector"),
      "value_mode": shape.get("value_mode"),
      "required_values": shape.get("required_values"),
  }


def _lower_bound_ready(row: dict[str, Any]) -> bool:
  return (
      row.get("required_checks_passed") is True
      and row.get("selected_next_route")
      == "router_prompt_all_linear_current_token_qkv_delta_blockq16_source_contract_gate"
      and row.get("layers") == ALL_LINEAR_LAYERS
      and row.get("layer_count") == len(ALL_LINEAR_LAYERS)
      and row.get("topk") == TOPK
      and row.get("selector") == "linear_qkv_col_abs"
      and row.get("value_mode") == "shadow_delta_block_q16"
      and row.get("required_values") == EXPECTED_VALUES)


def _function_body(source: str) -> str:
  match = re.search(
      r"std::uint64_t DecodeRouterQkvDeltaBlockQ16SourceHandle\(.*?"
      r"\n}\n\nstd::uint64_t DecodeElapsedNs",
      source,
      flags=re.S)
  return match.group(0) if match else ""


def _source_shape(source: str) -> dict[str, Any]:
  body = _function_body(source)
  value_source_checks = [
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
      _present(
          body,
          "blockq16_overlay_launch",
          "RunRouterQkvDeltaBlockQ16Overlay", regex=False),
      _present(
          body,
          "entry_group_live_input_upload",
          "runner.UploadF32Buffer(live_layer_input)", regex=False),
  ]
  missing_value_source_checks = [
      _absent(body, "no_shadow_value_source_in_product_handle",
              "g_decode_cpu_shadow_trace", regex=False),
      _absent(body, "no_live_delta_quantization",
              "block_max_abs_delta", regex=False),
      _absent(body, "no_selected_q_delta_assignment",
              r"selected_q_delta\[[^\]]+\]\s*="),
      _absent(body, "no_block_scale_assignment",
              r"block_scales\[[^\]]+\]\s*="),
  ]
  return {
      "function_found": bool(body),
      "value_source_checks": value_source_checks,
      "missing_value_source_checks": missing_value_source_checks,
      "zero_delta_noop_overlay": (
          bool(body)
          and _all_present(value_source_checks)
          and _all_absent(missing_value_source_checks)),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq507 = _load_json(args.seq507)
  seq515 = _load_json(args.seq515)
  source_text = _read(args.decode_source)

  seq515_rows = _seq515_rows(seq515)
  lower_bound = _lower_bound(seq507)
  source_shape = _source_shape(source_text)

  checks = [
      {
          "name": "seq515_distribution_fix_gate_selected",
          "pass": (
              seq515.get("disposition")
              == "reject_current_token_qkv_delta_blockq16_router_distribution"
              and seq515.get("selected_next_route")
              == "router_prompt_all_linear_current_token_qkv_delta_blockq16_distribution_fix_gate"
              and _has_candidate(
                  routes, 515,
                  "reject_current_token_qkv_delta_blockq16_router_distribution")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_current_token_qkv_delta_blockq16_distribution_fix_gate",
                  515)),
      },
      {
          "name": "seq515_counters_are_clean",
          "pass": _seq515_clean_counters(seq515_rows),
          "detail": _case_summary(seq515_rows),
      },
      {
          "name": "seq515_distribution_still_fails",
          "pass": _seq515_distribution_failed(seq515_rows),
          "detail": _case_summary(seq515_rows),
      },
      {
          "name": "seq507_lower_bound_requires_real_blockq16_values",
          "pass": _lower_bound_ready(lower_bound),
          "detail": lower_bound,
      },
      {
          "name": "current_product_blockq16_overlay_is_zero_delta_noop",
          "pass": source_shape.get("zero_delta_noop_overlay") is True,
          "detail": source_shape,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq507_design": _rel(args.seq507),
          "seq515_distribution": _rel(args.seq515),
          "decode_source": _rel(args.decode_source),
          "decode_source_sha256": _sha256(args.decode_source),
      },
      "seq515_distribution": {
          "rows": _case_summary(seq515_rows),
          "router_distribution_passed": seq515.get(
              "router_distribution_passed"),
          "required_checks_passed": seq515.get("required_checks_passed"),
      },
      "lower_bound": lower_bound,
      "source_shape": source_shape,
      "checks": checks,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "router_distribution_allowed": False,
      "decode_probe_allowed": False,
      "disposition": (
          "accept_blockq16_distribution_fix_select_value_source"
          if required else
          "reject_blockq16_distribution_fix_classification"),
      "selected_next_route": (
          SELECTED_NEXT_ROUTE if required else
          "router_prompt_all_linear_current_token_qkv_delta_blockq16_distribution_fix_gate"),
      "next_route_reason": (
          "Coverage is fixed: seq515 has all-30 block-q16 counters with zero "
          "misses. Distribution still fails because the product overlay is a "
          "zero-delta no-op; the next unit must source real q_delta/block-scale "
          "values for the selected qkv-column block-q16 correction."
          if required else
          "The distribution-fix classification is incomplete; do not launch "
          "speed or long-context rows."),
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
      "# Seq516 Current-Token QKV-Delta Block-Q16 Distribution Fix Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      "",
      "## Seq515 Rows",
      "",
  ]
  for row in metrics["seq515_distribution"]["rows"]:
    lines.extend([
        f"- {row['case_id']}: max KLD `{row['max_kld']}`, top1 `{row['top1_rate']}`, counters `{row['blockq16_layers']}` / `{row['blockq16_values']}` / `{row['blockq16_misses']}`",
    ])
  lines.extend([
      "",
      "## Classification",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control/correctness evidence only. It is not a speed claim.",
      "",
  ])
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq507", type=Path, default=DEFAULT_SEQ507)
  parser.add_argument("--seq515", type=Path, default=DEFAULT_SEQ515)
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
