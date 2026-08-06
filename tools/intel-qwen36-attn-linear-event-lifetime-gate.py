#!/usr/bin/env python3
"""Gate the next attention/full-core/linear-preconv event-lifetime route.

This is source-contract and route-control evidence, not speed evidence.  It
checks whether the current tree already has the resident handles needed for a
single broad proof, while preventing a blind rerun of the closed isolated
linear-final, residual-input, and shared-Q8 preconv routes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-attn-linear-event-lifetime-gate-v0"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_ROUTE_SELECTION = ROOT / "output/route-selection-gate-20260707Tseq82Z/metrics.json"
DEFAULT_ATTN_LINEAR_BUDGET = ROOT / "output/attn-linear-budget-20260707Tseq82Z/budget.json"
DEFAULT_SHARED_Q8_GATE = ROOT / "output/linear-preconv-shared-q8-profile-gate-20260706Tseq77Z/metrics.json"
DEFAULT_DECODE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_ENGINE_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_ENGINE_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_OUT_DIR = ROOT / "output/attn-linear-event-lifetime-gate-20260707Tseq83Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8")


def _check_markers(name: str, text: str, markers: list[str]) -> dict[str, Any]:
  missing = [marker for marker in markers if marker not in text]
  return {
      "name": name,
      "pass": not missing,
      "missing": missing,
      "marker_count": len(markers),
  }


def _closed_routes(rejected: dict[str, Any], pattern: str) -> list[str]:
  rows = rejected.get("rejected")
  if not isinstance(rows, list):
    return []
  rx = re.compile(pattern, re.IGNORECASE)
  routes: list[str] = []
  for row in rows:
    if not isinstance(row, dict):
      continue
    haystack = " ".join(
        str(row.get(key, ""))
        for key in ("route", "class", "reason", "runtime_cleanup")
    )
    if rx.search(haystack):
      route = row.get("route")
      if isinstance(route, str):
        routes.append(route)
  return routes


def _frontier_gap(frontier: dict[str, Any]) -> dict[str, float]:
  budget = frontier.get("goal_budget")
  budget = budget if isinstance(budget, dict) else {}
  per_token = budget.get("per_token_ms")
  per_token = per_token if isinstance(per_token, dict) else {}
  verdict = budget.get("verdict")
  verdict = verdict if isinstance(verdict, dict) else {}
  wall = _num(per_token.get("wall"))
  floor_budget = _num(verdict.get("floor_budget_ms_per_token"))
  return {
      "wall_ms_per_token": wall,
      "floor_budget_ms_per_token": floor_budget,
      "floor_gap_ms_per_token": max(0.0, wall - floor_budget),
      "floor_tps": _num(verdict.get("floor_tps")),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  rejected = _load_json(args.rejected)
  route_selection = _load_json(args.route_selection)
  attn_budget = _load_json(args.attn_linear_budget)
  shared_q8 = _load_json(args.shared_q8_gate)
  decode = _read(args.decode_source)
  engine_header = _read(args.engine_header)
  engine_source = _read(args.engine_source)
  opencl = _read(args.opencl_source)
  all_source = "\n".join([decode, engine_header, engine_source, opencl])

  frontier_gap = _frontier_gap(frontier)
  gap_subset = attn_budget.get("same_source_gap_upper_bound_ms_per_token")
  gap_subset = gap_subset if isinstance(gap_subset, dict) else {}
  gap_sum = _num(gap_subset.get("attention_fullcore_linear_sum"))

  closed = {
      "linear_final_device_q8": _closed_routes(
          rejected, r"linear[_ -]final|final[_ -]output.*device[_ -]q8"
      ),
      "attention_residual_input": _closed_routes(
          rejected, r"attention[_ -]front.*resident[_ -]residual|residual[_ -]handle"
      ),
      "linear_preconv_shared_q8": _closed_routes(
          rejected, r"shared[_ -]?q8.*preconv|preconv.*shared[_ -]?q8"
      ),
      "isolated_finish_or_lifetime": _closed_routes(
          rejected, r"finish|readback|buffer[-_ ]?lifecycle|local.*carrier"
      ),
  }

  source_contracts = [
      _check_markers(
          "linear_delta_final_output_handle_available",
          all_source,
          [
              "std::uint64_t final_output_handle = 0;",
              "bool readback_final_output",
              "linear_final_output_handle = resident_delta.final_output_handle;",
              "IQ36_LINEAR_FINAL_DEVICE_Q8_HANDOFF",
              "RunF32InputHandleDeviceQ8ThenResidentPackedQ4X8ResidualRmsNorm",
          ],
      ),
      _check_markers(
          "attention_front_resident_outputs_available",
          all_source,
          [
              "RunGpuAttentionFrontFromInputHandle",
              "attn_residual_handle",
              "attn_post_norm_handle",
              "resident_residual_input_handle",
              "g_decode_attention_front_handoff_layers",
          ],
      ),
      _check_markers(
          "full_core_attention_front_handoff_available",
          all_source,
          [
              "g_decode_resident_full_core_attention_front_handoff",
              "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNorm",
              "g_decode_full_core_attention_front_handoff_layers",
              "full_core_attention_front_handoff_kernel_us",
          ],
      ),
      _check_markers(
          "linear_preconv_attn_norm_handle_available",
          all_source,
          [
              "std::uint64_t attn_norm_handle = 0;",
              "RunGpuPreConvFront(",
              "IQ36_LINEAR_PRECONV_SHARED_Q8",
              "RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder",
              "RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder",
          ],
      ),
      _check_markers(
          "previous_layer_output_handle_available",
          all_source,
          [
              "g_decode_prev_layer_output_handle",
              "DecodeKeepPrevLayerOutputHandle",
              "RunRmsNormHiddenResidentInputResidentWeight",
              "IQ36_ATTENTION_FRONT_RESIDENT_RESIDUAL_INPUT",
              "--resident-tail-output-rmsnorm-input",
          ],
      ),
      _check_markers(
          "default_off_diagnostic_flags_available",
          all_source,
          [
              "g_decode_linear_final_device_q8_handoff = false",
              "g_decode_resident_attention_front_handoff = false",
              "g_decode_resident_full_core_attention_front_handoff = false",
              "g_decode_linear_preconv_shared_q8 = false",
          ],
      ),
  ]

  broad_route_markers = [
      "IQ36_ATTENTION_LINEAR_EVENT_LIFETIME",
      "g_decode_attention_linear_event_lifetime",
      "--attention-linear-event-lifetime",
  ]
  broad_route_present = any(marker in all_source for marker in broad_route_markers)

  shared_q8_verdict = shared_q8.get("verdict")
  shared_q8_verdict = shared_q8_verdict if isinstance(shared_q8_verdict, dict) else {}
  shared_q8_closed = shared_q8_verdict.get("shared_q8_profile_closes_speed_route") is True

  checks = [
      {
          "name": "seq82_selected_event_lifetime_route",
          "pass": route_selection.get("selected_next_route")
          == "attention_linear_event_lifetime_proof",
      },
      {
          "name": "gap_sum_can_clear_frontier_floor_gap",
          "pass": gap_sum >= frontier_gap["floor_gap_ms_per_token"] > 0.0,
      },
      {
          "name": "closed_isolated_routes_acknowledged",
          "pass": all(bool(value) for value in closed.values()) and shared_q8_closed,
      },
      {
          "name": "resident_handle_source_contracts_present",
          "pass": all(item["pass"] for item in source_contracts),
      },
      {
          "name": "single_broad_route_not_yet_wired",
          "pass": not broad_route_present,
      },
  ]
  required = all(item["pass"] for item in checks)

  if required:
    disposition = "event_lifetime_source_contract_ready_needs_combined_decode_wiring"
    next_action = (
        "Wire one default-off combined event-lifetime route for the accepted "
        "stack, then run same-source 8-token baseline/candidate explores. The "
        "candidate must clear the 0.45 ms/token floor gap and must not grow "
        "attention-front wall. Treat IQ36_LINEAR_FINAL_DEVICE_Q8_HANDOFF and "
        "IQ36_LINEAR_PRECONV_SHARED_Q8 as closed unless the combined proof also "
        "shows their specific regressions are gone."
    )
  elif broad_route_present:
    disposition = "event_lifetime_broad_route_present_needs_target_probe"
    next_action = (
        "Run the broad route as a same-source explore pair and gate it against "
        "attention-front non-growth before considering promotion."
    )
  else:
    disposition = "event_lifetime_contract_missing"
    next_action = "Fix the missing source-contract checks before any target run."

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "checks": checks,
      "disposition": disposition,
      "next_action": next_action,
      "inputs": {
          "frontier": _rel(args.frontier),
          "rejected": _rel(args.rejected),
          "route_selection": _rel(args.route_selection),
          "attn_linear_budget": _rel(args.attn_linear_budget),
          "shared_q8_gate": _rel(args.shared_q8_gate),
          "decode_source": _rel(args.decode_source),
          "engine_header": _rel(args.engine_header),
          "engine_source": _rel(args.engine_source),
          "opencl_source": _rel(args.opencl_source),
      },
      "frontier": frontier_gap,
      "gap_upper_bound_ms_per_token": {
          "attention_front": _num(gap_subset.get("attention_front")),
          "full_core": _num(gap_subset.get("full_core")),
          "linear_preconv": _num(gap_subset.get("linear_preconv")),
          "attention_fullcore_linear_sum": gap_sum,
      },
      "closed_route_counts": {key: len(value) for key, value in closed.items()},
      "closed_route_examples": {key: value[-3:] for key, value in closed.items()},
      "source_contracts": source_contracts,
      "broad_route_present": broad_route_present,
      "broad_route_markers": broad_route_markers,
      "shared_q8_profile_closes_speed_route": shared_q8_closed,
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  (out_dir / "manifest.json").write_text(
      json.dumps(
          {
              "schema_version": payload["schema_version"],
              "workstream": payload["workstream"],
              "metrics": "metrics.json",
              "required_checks_passed": payload["required_checks_passed"],
              "speedup_claims_allowed": False,
          },
          indent=2,
          sort_keys=True,
      )
      + "\n",
      encoding="utf-8",
  )
  failed = [item["name"] for item in payload["checks"] if item["pass"] is not True]
  frontier = payload["frontier"]
  gaps = payload["gap_upper_bound_ms_per_token"]
  lines = [
      "# Attention/Linear Event-Lifetime Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- frontier floor gap: `{frontier['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- attention/full-core/linear gap sum: `{gaps['attention_fullcore_linear_sum']:.3f}` ms/token",
      f"- broad route already wired: `{str(payload['broad_route_present']).lower()}`",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is source-contract/route evidence only. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--route-selection", type=Path, default=DEFAULT_ROUTE_SELECTION)
  parser.add_argument("--attn-linear-budget", type=Path, default=DEFAULT_ATTN_LINEAR_BUDGET)
  parser.add_argument("--shared-q8-gate", type=Path, default=DEFAULT_SHARED_Q8_GATE)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE)
  parser.add_argument("--engine-header", type=Path, default=DEFAULT_ENGINE_HEADER)
  parser.add_argument("--engine-source", type=Path, default=DEFAULT_ENGINE_SOURCE)
  parser.add_argument("--opencl-source", type=Path, default=DEFAULT_OPENCL)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
