#!/usr/bin/env python3
"""Select the next product-decode route from the active kill-number."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-product-decode-route-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km"
FRONTIER = ACTIVE / "frontier.json"
ACCEPTANCE = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/"
    "acceptance-matrix.json")
NATIVE_CONSENSUS = (
    ROOT / "output/native-consensus-gate-20260712Tseq730cleanZ/result.json")
Q4_STREAM = (
    ROOT / "output/gpu-q4x8-qmatvec-ffn-gateup-full-"
    "20260702T225500Z/probe-result.json")
Q6_STREAM = (
    ROOT / "output/q6-rowstripe16-58gbps-gate-"
    "20260711Tseq658cleanZ/gate.json")
STRICT_ACTIVE_WEIGHT_GB = 1.975676544
KV_BYTES_PER_CONTEXT_TOKEN = 20_480
CONTEXT = 1024
MIN_PERSISTENT_SCHEDULE_OVERHEAD_MS = 1.5


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/product-decode-route-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  required = [FRONTIER, ACCEPTANCE, NATIVE_CONSENSUS, Q4_STREAM, Q6_STREAM]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  frontier = load_json(FRONTIER)
  acceptance = load_json(ACCEPTANCE)
  consensus = load_json(NATIVE_CONSENSUS)
  q4 = load_json(Q4_STREAM)
  q6 = load_json(Q6_STREAM)
  anchor = frontier.get("goal_anchor", {})
  budget = frontier.get("goal_budget", {})
  budget_verdict = budget.get("verdict", {})
  per_token = budget.get("per_token_ms", {})
  product_floor = float(
      acceptance["bootstrap_targets"]["decode_tokens_s"]["1024"])
  product_wall_ms = 1000.0 / product_floor
  strict_gb_per_token = (
      STRICT_ACTIVE_WEIGHT_GB +
      CONTEXT * KV_BYTES_PER_CONTEXT_TOKEN / 1e9)
  target_gb_s = strict_gb_per_token * product_floor
  q4_gb_s = float(q4.get("gpu_effective_packed_gb_s", 0.0))
  q6_gb_s = float(
      q6.get("rowstripe_variant", {}).get("gpu_effective_packed_gb_s", 0.0))
  kernel_ms = float(per_token.get("gpu_kernel_busy_floor", float("nan")))
  wall_ms = float(per_token.get("wall", float("nan")))
  overhead_ms = float(per_token.get("non_kernel_overhead", float("nan")))
  minimum_kernel_cut_ms = kernel_ms - product_wall_ms
  total_wall_cut_ms = wall_ms - product_wall_ms
  kernel_schedule_cap_ms = product_wall_ms - MIN_PERSISTENT_SCHEDULE_OVERHEAD_MS
  top_stages = budget.get("top_stage_walls_ms_per_token", [])
  top_three = top_stages[:3]
  top_three_ms = sum(float(row.get("ms_per_token", 0.0)) for row in top_three)

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("three_case_cross_reference_exact_decode_passed",
            consensus.get("required_checks_passed") is True and
            consensus.get("git", {}).get("dirty") is False and
            len(consensus.get("rows", [])) == 3 and
            all(row.get("candidate_exact_reference_match") is True
                for row in consensus.get("rows", []))),
      check("frontier_uses_active_product_floor",
            anchor.get("active_product_decode_floor_tps") == product_floor and
            budget_verdict.get("floor_tps") == product_floor),
      check("overhead_only_route_is_arithmetically_closed",
            budget_verdict.get(
                "can_reach_floor_without_kernel_work") is False and
            float(budget_verdict.get(
                "overhead_only_ceiling_tok_s", float("inf"))) < product_floor,
            overhead_only_ceiling_tok_s=budget_verdict.get(
                "overhead_only_ceiling_tok_s")),
      check("current_kernel_cut_is_structural",
            finite(kernel_ms) and minimum_kernel_cut_ms > 16.0 and
            float(budget_verdict.get(
                "min_kernel_time_cut_pct_needed", 0.0)) >= 0.44,
            current_kernel_ms=kernel_ms,
            minimum_kernel_cut_ms=minimum_kernel_cut_ms,
            minimum_kernel_cut_fraction=budget_verdict.get(
                "min_kernel_time_cut_pct_needed")),
      check("top_three_stage_families_cover_required_cut",
            len(top_three) == 3 and top_three_ms > minimum_kernel_cut_ms,
            top_three=top_three,
            top_three_ms=top_three_ms),
      check("real_q4_and_q6_stream_carriers_clear_target_bandwidth",
            q4_gb_s >= target_gb_s and q6_gb_s >= target_gb_s,
            q4_gb_s=q4_gb_s, q6_gb_s=q6_gb_s,
            target_gb_s=target_gb_s),
  ]
  passed = all(row["pass"] for row in checks)
  created_at = iso_now()
  selected_route = {
      "id": "resident_packed_full_token_schedule_v5",
      "class": "persistent_whole_token_packed_stream_schedule",
      "reason": (
          "The accepted decode carrier is exact on fit/validation/test "
          "reference-consensus cases, but current kernels alone take "
          f"{kernel_ms:.3f} ms/token versus the {product_wall_ms:.3f} ms "
          "entire product budget. Overhead-only work tops out below the floor. "
          "Real Q4 and Q6 carriers already exceed the strict target bandwidth, "
          "so the next route must preserve their layouts while scheduling the "
          "whole token persistently and eliminating inter-stage host drains."
      ),
      "scope": [
          "selected_ffn",
          "linear_preconv",
          "attention_front",
          "remaining_resident_token_stages",
      ],
      "admission": {
          "full_token_kernel_schedule_ms_max": kernel_schedule_cap_ms,
          "residual_host_submit_overhead_ms_max":
              MIN_PERSISTENT_SCHEDULE_OVERHEAD_MS,
          "full_token_wall_ms_max": product_wall_ms,
          "strict_stream_bandwidth_gb_s_min": target_gb_s,
          "component_and_consensus_correctness_unchanged": True,
      },
      "closed_work": (
          "No isolated launch/read/finish/setup microcut, single-stage local-size "
          "variant, or codec sweep can enter before a whole-token schedule "
          "design demonstrates this admission budget."
      ),
  }
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "git": state,
      "inputs": {
          "frontier": str(FRONTIER.relative_to(ROOT)),
          "acceptance": str(ACCEPTANCE.relative_to(ROOT)),
          "native_consensus": str(NATIVE_CONSENSUS.relative_to(ROOT)),
          "q4_stream": str(Q4_STREAM.relative_to(ROOT)),
          "q6_stream": str(Q6_STREAM.relative_to(ROOT)),
      },
      "budget": {
          "product_floor_tokens_s": product_floor,
          "product_wall_ms_per_token": product_wall_ms,
          "current_wall_ms_per_token": wall_ms,
          "current_kernel_ms_per_token": kernel_ms,
          "current_non_kernel_overhead_ms_per_token": overhead_ms,
          "minimum_kernel_cut_ms": minimum_kernel_cut_ms,
          "total_wall_cut_ms": total_wall_cut_ms,
          "strict_gb_per_token": strict_gb_per_token,
          "strict_target_gb_s": target_gb_s,
          "q4_measured_gb_s": q4_gb_s,
          "q6_measured_gb_s": q6_gb_s,
      },
      "checks": checks,
      "selected_route": selected_route,
      "required_checks_passed": passed,
      "disposition": (
          "select_resident_packed_full_token_schedule"
          if passed else "reject_product_decode_route_selection"),
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "checks": checks,
      "required_checks_passed": passed,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out),
      "git": state,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  (out / "summary.md").write_text("\n".join([
      "# Product decode route gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- current wall / kernel: `{wall_ms:.3f} / {kernel_ms:.3f} ms/token`",
      f"- product wall: `{product_wall_ms:.3f} ms/token`",
      f"- overhead-only ceiling: `{budget_verdict.get('overhead_only_ceiling_tok_s')} tok/s`",
      f"- minimum current-kernel cut: `{minimum_kernel_cut_ms:.3f} ms` / "
      f"`{float(budget_verdict.get('min_kernel_time_cut_pct_needed')) * 100:.2f}%`",
      f"- strict target bandwidth: `{target_gb_s:.3f} GB/s`",
      f"- measured Q4 / Q6 carriers: `{q4_gb_s:.3f} / {q6_gb_s:.3f} GB/s`",
      f"- selected route: `{selected_route['id']}`", "",
      "This selects an implementation route; it is not a product speed claim.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out),
      "pass": passed,
      "selected_route": selected_route["id"],
      "kernel_cut_pct": float(budget_verdict.get(
          "min_kernel_time_cut_pct_needed")) * 100.0,
      "target_gb_s": target_gb_s,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
