#!/usr/bin/env python3
"""Gate the qkv-delta component implementation-source route.

This is a route-control/source preflight. It consumes the source-only component
probe and verifies that the next source change cannot honestly be a mask,
rounding, static predictor, or contract-bit flip. Passing this gate selects the
current-token value-source gate before any decode or router distribution row.
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
SCHEMA_VERSION = (
    "intel-qwen36-router-qkv-delta-component-implementation-source-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SEQ292 = (
    ROOT / "output/router-qkv-delta-source-gate-20260708Tseq292Z/metrics.json"
)
DEFAULT_SEQ295 = (
    ROOT / "output/router-qkv-delta-component-probe-gate-20260708Tseq295Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "output/router-qkv-delta-component-implementation-source-gate-20260708Tseq296Z"
)

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
TOPK = 512
HIDDEN_SIZE = 8192


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
  current_contract = [
      _present(text, "component_contract_struct",
               "DecodeRouterQkvDeltaComponentContract", regex=False),
      _present(text, "component_ready_function",
               "DecodeRouterQkvDeltaComponentReady", regex=False),
      _present(text, "source_only_guard",
               "IQ36_ROUTER_QKV_DELTA_COMPONENT_SOURCE is source-gate only",
               regex=False),
      _present(text, "contract_product_field_false",
               "bool product_owned_source = false;", regex=False),
      _present(text, "contract_cpu_shadow_free_field_false",
               "bool cpu_shadow_free = false;", regex=False),
      _present(text, "contract_host_sync_free_field_false",
               "bool host_sync_free = false;", regex=False),
  ]
  fake_impl_absent = [
      _absent(text, "no_contract_bit_flip_product_source_true",
              "product_owned_source = true", regex=False),
      _absent(text, "no_contract_bit_flip_cpu_shadow_free_true",
              "cpu_shadow_free = true", regex=False),
      _absent(text, "no_contract_bit_flip_host_sync_free_true",
              "host_sync_free = true", regex=False),
      _absent(text, "no_named_product_source_helper_yet",
              "DecodeRouterQkvDeltaProductValueSource", regex=False),
      _absent(text, "no_decode_runtime_product_correction_yet",
              "router_qkv_delta_product_values", regex=False),
  ]
  shadow_bound_present = [
      _present(text, "cpu_shadow_delta_diagnostic_present",
               "g_decode_cpu_shadow_layer_input_delta_layers", regex=False),
      _present(text, "linear_qkv_selector_present",
               "linear_qkv_col_abs", regex=False),
      _present(text, "shadow_delta_applier_present",
               "DecodeApplyTopKShadowDelta", regex=False),
  ]
  return {
      "current_contract_present": _all_present(current_contract),
      "fake_product_source_absent": _all_absent(fake_impl_absent),
      "shadow_bound_diagnostic_still_present": _all_present(shadow_bound_present),
      "current_contract_checks": current_contract,
      "fake_impl_absent_checks": fake_impl_absent,
      "shadow_bound_checks": shadow_bound_present,
  }


def _seq292_shape(seq292: dict[str, Any]) -> dict[str, Any]:
  shape = seq292.get("correction_shape")
  return shape if isinstance(shape, dict) else {}


def _seq295_shape(seq295: dict[str, Any]) -> dict[str, Any]:
  shape = seq295.get("correction_shape")
  return shape if isinstance(shape, dict) else {}


def _shape_passes(shape: dict[str, Any]) -> bool:
  return (
      shape.get("layers") == ALL_LINEAR_LAYERS
      and shape.get("top512_values_expected") == 122880
      and _num(shape.get("top512_fraction_per_layer")) == TOPK / HIDDEN_SIZE
  )


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq292 = _load_json(args.seq292)
  seq295 = _load_json(args.seq295)
  decode_source = _read(args.decode_source)
  source = _source_state(decode_source)
  rejected_names = _rejected_names(rejected)
  required_closed = {
      "router_math_static_or_lagged_qkv_delta_predictors",
      "router_math_live_round_or_selected_affine_qkv_delta_approximation",
      "router_math_split_full_attention_projection_arithmetic_residual_fix",
  }
  missing_closed = sorted(required_closed - rejected_names)
  seq292_shape = _seq292_shape(seq292)
  seq295_shape = _seq295_shape(seq295)
  top512_values = len(ALL_LINEAR_LAYERS) * 8 * TOPK

  checks = [
      {
          "name": "seq295_selected_implementation_source_gate",
          "pass": (
              seq295.get("required_checks_passed") is True
              and seq295.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_component_implementation_source_gate"
              and seq295.get("component_product_source_present") is False
              and seq295.get("decode_probe_allowed") is False
              and _has_candidate(
                  routes, 295,
                  "reject_source_only_component_as_product_probe_select_implementation_source")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_component_implementation_source_gate",
                  295)
          ),
      },
      {
          "name": "top512_math_code_lower_bound_preserved",
          "pass": (
              seq292.get("required_checks_passed") is True
              and _shape_passes(seq292_shape)
              and _shape_passes(seq295_shape)
              and top512_values == 122880),
          "detail": {
              "seq292_shape": seq292_shape,
              "seq295_shape": seq295_shape,
              "expected_values": top512_values,
          },
      },
      {
          "name": "closed_non_value_sources_not_reopened",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
      {
          "name": "source_has_not_faked_product_owned_contract",
          "pass": (
              source["current_contract_present"]
              and source["fake_product_source_absent"]
              and source["shadow_bound_diagnostic_still_present"]),
          "detail": source,
      },
      {
          "name": "implementation_must_source_current_token_values",
          "pass": (
              seq292.get("product_approximations_failed") is True
              or any(
                  isinstance(row, dict)
                  and row.get("name") == "product_approximations_failed"
                  and row.get("pass") is True
                  for row in seq292.get("checks", [])
              )
          ),
          "detail": {
              "closed_classes": sorted(required_closed),
              "value_source_rule": (
                  "top512 mask/selector evidence is necessary but not "
                  "sufficient; the implementation must produce current-token "
                  "values without CPU shadow or host sync."
              ),
          },
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
          "seq292_source_gate": _rel(args.seq292),
          "seq295_component_probe": _rel(args.seq295),
      },
      "source": source,
      "correction_shape": {
          "layers": ALL_LINEAR_LAYERS,
          "layer_count": len(ALL_LINEAR_LAYERS),
          "topk": TOPK,
          "hidden_size": HIDDEN_SIZE,
          "decode_tokens": 8,
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
          "reject_fake_product_source_select_current_token_value_source_gate"
          if required else
          "block_before_current_token_value_source_gate"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_current_token_value_source_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_component_implementation_source_fix_gate"
      ),
      "next_route_reason": (
          "The component implementation-source gate is constrained to a "
          "current-token value source: the top512 qkv selector is only the "
          "selection rule, and closed routes already reject static, lagged, "
          "rounded, affine, and split-projection substitutes. The next source "
          "unit must produce current-token values without CPU shadow or host sync."
          if required else
          "Implementation-source evidence is inconsistent; fix the route gate "
          "before adding source or running decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  lines = [
      "# Router QKV Delta Component Implementation Source Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- component_product_source_present: `{str(metrics['component_product_source_present']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- top512 values: `{metrics['correction_shape']['top512_values_expected']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is route-control/source preflight. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--seq292", type=Path, default=DEFAULT_SEQ292)
  parser.add_argument("--seq295", type=Path, default=DEFAULT_SEQ295)
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
