#!/usr/bin/env python3
"""Gate local tail-output handoffs against the resident hidden-state carrier.

This is route-selection evidence, not benchmark evidence. It reads the current
decode harness shape plus the seq53 drain accounting and records whether
another local tail/RMSNorm/residual handoff can remove the measured Q6 drain.
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
SCHEMA_VERSION = "intel-qwen36-resident-hidden-carrier-gate-v0"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GPU_HEADER = ROOT / "engine/include/intel_qwen36/gpu_q4x8_matvec.hpp"
DEFAULT_REJECTED = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/rejected-routes.json"
)
DEFAULT_SEQ53_METRICS = ROOT / "output/q6-defer-drain-budget-20260706Tseq53Z/metrics.json"
DEFAULT_OUT_DIR = ROOT / "output/resident-hidden-carrier-gate-20260706Tseq54Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_of(text: str, needle: str) -> int | None:
  index = text.find(needle)
  if index < 0:
    return None
  return text.count("\n", 0, index) + 1


def _regex_check(text: str, pattern: str, *, label: str) -> dict[str, Any]:
  match = re.search(pattern, text, re.S)
  return {
      "label": label,
      "present": match is not None,
      "line": text.count("\n", 0, match.start()) + 1 if match else None,
  }


def _closed_route(rejected: dict[str, Any], route: str) -> dict[str, Any]:
  rows = rejected.get("rejected")
  if not isinstance(rows, list):
    return {"route": route, "present": False}
  for row in rows:
    if isinstance(row, dict) and row.get("route") == route:
      return {
          "route": route,
          "present": True,
          "class": row.get("class"),
          "evidence": row.get("evidence"),
      }
  return {"route": route, "present": False}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  decode_source = args.decode_source.read_text(encoding="utf-8")
  gpu_header = args.gpu_header.read_text(encoding="utf-8")
  rejected = _load_json(args.rejected_routes)
  seq53 = _load_json(args.seq53_metrics)
  derived53 = seq53["derived"]

  code_checks = [
      _regex_check(
          decode_source,
          r"RunGpuHybridLinearLayerLive\([^)]*const std::vector<float>& residual",
          label="linear_layer_api_requires_host_residual_vector",
      ),
      _regex_check(
          decode_source,
          r"RunGpuHybridFullAttentionLayerLive\([^)]*const std::vector<float>& residual",
          label="full_attention_layer_api_requires_host_residual_vector",
      ),
      _regex_check(
          decode_source,
          r"RunGpuAttentionFrontFromInputHandle\([^)]*const std::vector<float>& residual_input",
          label="attention_front_handle_path_still_requires_host_residual",
      ),
      _regex_check(
          decode_source,
          r"RunGpuHybridFfnTail\([^)]*const std::vector<float>& ffn_input[^)]*"
          r"const std::vector<float>& attention_residual[^)]*std::uint64_t ffn_input_handle",
          label="ffn_tail_api_requires_host_ffn_norm_and_attention_residual",
      ),
      _regex_check(
          decode_source,
          r"RunGpuSelectedFfnShell\([^;]*\*ffn_input_used",
          label="selected_ffn_still_consumes_host_ffn_norm_vector",
      ),
      _regex_check(
          decode_source,
          r"RunGpuSharedFfnShell\([^;]*\*ffn_input_used",
          label="shared_ffn_still_consumes_host_ffn_norm_vector",
      ),
      _regex_check(
          decode_source,
          r"ffn_input_handle != 0\s*\?\s*g_decode_shared_q4_runner->RunResidentF32MatvecFromInputHandle",
          label="ffn_input_handle_scope_is_gpu_router_only",
      ),
      _regex_check(
          decode_source,
          r"g_decode_resident_tail_output_rmsnorm_input\s*\?\s*g_decode_prev_layer_output_handle\s*:\s*0",
          label="tail_output_handle_scope_is_layer_input_rmsnorm_only",
      ),
      _regex_check(
          gpu_header,
          r"RunFfnTailFromDownHandles\([^)]*const std::vector<float>& attn_post_norm[^)]*"
          r"const std::vector<float>& attn_residual",
          label="runner_tail_down_handles_api_requires_host_tail_vectors",
      ),
      _regex_check(
          gpu_header,
          r"RunRmsNormHiddenResidentInputResidentWeight\([^)]*std::uint64_t input_handle",
          label="resident_input_rmsnorm_primitive_exists_but_is_not_full_carrier",
      ),
  ]
  all_code_checks_pass = all(check["present"] for check in code_checks)

  closed_names = [
      "gpu_resident_norm_weights_tail_output_input",
      "gpu_tail_output_residual_handle_attention_front",
      "gpu_attention_residual_tail_handoff",
      "gpu_attention_residual_readback_skip_tail_handle",
      "gpu_tail_output_rmsnorm_input_current_best_retest",
      "gpu_q6_defer_tail_read_drain_noqueue",
      "gpu_q6_defer_tail_rmsnorm_input_noqueue",
      "gpu_q6_defer_finish_without_tail_drain_elimination",
  ]
  closed_routes = [_closed_route(rejected, route) for route in closed_names]
  all_closed_routes_present = all(route["present"] for route in closed_routes)

  tail_drain_elimination_clears_floor = bool(
      derived53.get("tail_drain_elimination_clears_floor")
  )
  local_handoff_closed = all_code_checks_pass and all_closed_routes_present
  requires_carrier_or_fusion = local_handoff_closed and tail_drain_elimination_clears_floor

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "decode_source": {
              "path": _display_path(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "gpu_header": {
              "path": _display_path(args.gpu_header),
              "sha256": _sha256(args.gpu_header),
          },
          "rejected_routes": {
              "path": _display_path(args.rejected_routes),
              "sha256": _sha256(args.rejected_routes),
          },
          "seq53_metrics": {
              "path": _display_path(args.seq53_metrics),
              "sha256": _sha256(args.seq53_metrics),
          },
      },
      "seq53_drain_accounting": {
          "selected_down_wait_saved_ms_per_token": derived53[
              "selected_down_wait_saved_ms_per_token"
          ],
          "selected_ffn_saved_ms_per_token": derived53[
              "selected_ffn_saved_ms_per_token"
          ],
          "ffn_tail_growth_ms_per_token": derived53[
              "ffn_tail_growth_ms_per_token"
          ],
          "promotion_delta_pct_vs_current_best": derived53[
              "promotion_delta_pct_vs_current_best"
          ],
          "frontier_noise_pct": derived53["frontier_noise_pct"],
          "projected_tps_without_tail_growth": derived53[
              "projected_tps_without_tail_growth"
          ],
          "tail_drain_elimination_clears_floor": tail_drain_elimination_clears_floor,
      },
      "code_contract_checks": code_checks,
      "closed_local_handoff_routes": closed_routes,
      "derived": {
          "all_code_contract_checks_present": all_code_checks_pass,
          "all_required_local_handoff_closures_present": all_closed_routes_present,
          "local_handoff_closed_without_hidden_state_carrier": local_handoff_closed,
          "resident_hidden_state_carrier_or_down_tail_fusion_required": (
              requires_carrier_or_fusion
          ),
          "tail_output_handle_set_line": _line_of(
              decode_source, "g_decode_prev_layer_output_handle = resident_tail.layer_output_handle"
          ),
          "tail_output_handle_rmsnorm_use_line": _line_of(
              decode_source, "g_decode_prev_layer_output_handle\n          : 0"
          ),
      },
      "verdict": {
          "local_tail_handoff_promotable": False,
          "speedup_claims_allowed": False,
          "reason": (
              "Seq53 shows the Q6 wait can be removed only if the tail drain is "
              "removed, but the live harness still carries hidden state through "
              "host vectors at the layer, attention-front, selected/shared FFN, "
              "and FFN-tail boundaries. The existing tail-output handle feeds "
              "only next-layer RMSNorm. Local tail/RMSNorm/residual handoffs are "
              "already closed by rejected-route evidence."
          ),
          "next_route": (
              "Do not spend another probe on a local tail-output, RMSNorm-input, "
              "residual-handle, or read-as-drain variant. Continue only with a "
              "resident hidden-state carrier contract that propagates the layer "
              "output handle through layer input RMSNorm, attention residual, "
              "FFN norm/router input, selected/shared FFN, and FFN tail; or with "
              "a true selected/shared down-to-tail fusion that removes the drain."
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
      "tool": "tools/intel-qwen36-resident-hidden-carrier-gate.py",
      "workstream": WORKSTREAM,
      "artifact": _display_path(out_dir),
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  d = result["derived"]
  s = result["seq53_drain_accounting"]
  lines = [
      "# Resident Hidden Carrier Gate",
      "",
      "This is route-selection evidence over current source shape and closed-route records.",
      "",
      "## Seq53 Drain Signal",
      "",
      f"- selected down wait saved: `{s['selected_down_wait_saved_ms_per_token']:.3f}` ms/token",
      f"- FFN-tail growth: `{s['ffn_tail_growth_ms_per_token']:.3f}` ms/token",
      f"- promotion delta: `{s['promotion_delta_pct_vs_current_best']:.3f}%` "
      f"inside `{s['frontier_noise_pct']:.3f}%` noise",
      f"- projected speed without tail growth: `{s['projected_tps_without_tail_growth']:.3f}` tok/s",
      "",
      "## Source Contract",
      "",
      f"- code checks present: `{str(d['all_code_contract_checks_present']).lower()}`",
      f"- local handoff closures present: `{str(d['all_required_local_handoff_closures_present']).lower()}`",
      f"- local handoffs closed without carrier: `{str(d['local_handoff_closed_without_hidden_state_carrier']).lower()}`",
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
  parser.add_argument("--gpu-header", type=Path, default=DEFAULT_GPU_HEADER)
  parser.add_argument("--rejected-routes", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq53-metrics", type=Path, default=DEFAULT_SEQ53_METRICS)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
  result = compute(args)
  write_outputs(result, out_dir)
  derived = result["derived"]
  print("resident hidden carrier gate")
  print(f"  artifact: {out_dir}")
  print(
      "  code checks: "
      f"{derived['all_code_contract_checks_present']} ; closed routes: "
      f"{derived['all_required_local_handoff_closures_present']}"
  )
  print(f"  verdict: {result['verdict']['next_route']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
