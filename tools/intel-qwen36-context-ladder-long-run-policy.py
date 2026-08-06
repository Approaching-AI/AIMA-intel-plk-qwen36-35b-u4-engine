#!/usr/bin/env python3
"""Create a policy artifact for expensive post-8k context-ladder runs.

The context-ladder diagnostic is intentionally expensive after 8k. This tool
turns the next-bucket decision into a reproducible artifact instead of leaving
it as prose in STATUS.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-context-ladder-long-run-policy-v0"
POLICY_ID = "q6_pair_post_16k_32k_explicit_long_jobs_v0"
DEFAULT_ROLLUP = ROOT / "output/context-ladder-rollup-20260629T004840Z"
EXPECTED_ROUTE = "post_r1_20260628T054920Z_dense_q6_pair_dot_flags"
OBSERVED_COUNTS = [1024, 2048, 4096, 8192, 16384]
NEXT_BUCKET_TOKENS = 32768
NEXT_BUCKET_LABEL = "032k"
DEFERRED_BUCKETS = [65536, 102400, 131072, 262144]
TIMEOUT_S = 90000


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--rollup-dir", type=Path, default=DEFAULT_ROLLUP)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=TIMEOUT_S)
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def rel(path: Path) -> str:
  return path.resolve().relative_to(ROOT).as_posix()


def rows_by_kind(series: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  out: dict[str, list[dict[str, Any]]] = {}
  for row in series:
    kind = row.get("kind")
    if not isinstance(kind, str):
      continue
    out.setdefault(kind, []).append(row)
  for rows in out.values():
    rows.sort(key=lambda row: int(row.get("prompt_token_count", -1)))
  return out


def project_next_bucket(kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
  if len(rows) < 2:
    return {
        "kind": kind,
        "projectable": False,
        "reason": "insufficient_observed_buckets",
      }
  prev = rows[-2]
  latest = rows[-1]
  prev_ns = int(prev["prompt_prefill_ns"])
  latest_ns = int(latest["prompt_prefill_ns"])
  ratio = latest_ns / prev_ns
  projected_ns = int(round(latest_ns * ratio))
  projected_s = projected_ns / 1_000_000_000
  return {
      "basis": {
          "latest_bucket_tokens": latest.get("prompt_token_count"),
          "latest_prefill_ns": latest_ns,
          "previous_bucket_tokens": prev.get("prompt_token_count"),
          "previous_prefill_ns": prev_ns,
      },
      "case_id": f"{'sentinel' if kind == 'sentinel_retrieval' else 'prefill_shape'}_{NEXT_BUCKET_LABEL}",
      "kind": kind,
      "latest_adjacent_ratio": ratio,
      "projectable": True,
      "projected_prefill_ns": projected_ns,
      "projected_wall_hours": projected_s / 3600,
      "projected_wall_seconds": projected_s,
  }


def command_for(case_id: str, timeout_s: int) -> str:
  return (
      "python3 tools/intel-qwen36-context-ladder-native-diagnostic.py "
      f"--case-id {case_id} --dense-q6-pair-dot --timeout-s {timeout_s}"
  )


def build_summary(payload: dict[str, Any]) -> str:
  policy = payload["policy"]
  lines = [
      "# Context Ladder Long-Run Policy",
      "",
      f"- policy id: `{policy['policy_id']}`",
      f"- decision: `{policy['decision']}`",
      f"- accepted rollup: `{policy['accepted_rollup']}`",
      f"- next bucket: `{policy['next_bucket']['label']}`",
      f"- timeout seconds: `{policy['next_bucket']['timeout_s']}`",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| case | projected hours | command |",
      "|---|---:|---|",
  ]
  projections = {
      item["case_id"]: item for item in policy["next_bucket"]["projections"]
      if item.get("projectable")
  }
  for job in policy["next_bucket"]["jobs"]:
    projection = projections.get(job["case_id"], {})
    hours = projection.get("projected_wall_hours")
    hours_text = f"{hours:.2f}" if isinstance(hours, (float, int)) else ""
    lines.append(f"| `{job['case_id']}` | {hours_text} | `{job['command']}` |")
  lines += [
      "",
      "This policy schedules the next ladder bucket as explicit isolated long",
      "jobs. It is not benchmark evidence and must not be used as a speedup",
      "claim.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.timeout_s < 3600:
    raise SystemExit("--timeout-s must be at least 3600 for long-run policy")
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/context-ladder-long-run-policy-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  rollup_dir = args.rollup_dir.resolve()
  rollup_path = rollup_dir / "rollup.json"
  correctness_path = rollup_dir / "correctness.json"
  if not rollup_path.exists() or not correctness_path.exists():
    raise SystemExit(f"rollup artifact missing required files: {rollup_dir}")
  rollup_payload = load_json(rollup_path)
  rollup_correctness = load_json(correctness_path)
  rollup = rollup_payload.get("rollup", {})
  artifacts = rollup.get("artifacts", [])
  series = rollup.get("series", [])
  if not isinstance(artifacts, list) or not isinstance(series, list):
    raise SystemExit("rollup payload missing artifacts or series")

  grouped = rows_by_kind([row for row in series if isinstance(row, dict)])
  projections = [
      project_next_bucket(kind, grouped.get(kind, []))
      for kind in ("sentinel_retrieval", "prefill_shape")
  ]
  jobs = [
      {
          "case_id": f"sentinel_{NEXT_BUCKET_LABEL}",
          "command": command_for(f"sentinel_{NEXT_BUCKET_LABEL}", args.timeout_s),
          "kind": "sentinel_retrieval",
          "process_policy": "isolated_single_case_target_process",
      },
      {
          "case_id": f"prefill_shape_{NEXT_BUCKET_LABEL}",
          "command": command_for(f"prefill_shape_{NEXT_BUCKET_LABEL}", args.timeout_s),
          "kind": "prefill_shape",
          "process_policy": "isolated_single_case_target_process",
      },
  ]
  commands = [job["command"] for job in jobs]
  post_run_rollup_command = (
      "python3 tools/intel-qwen36-context-ladder-rollup.py "
      + " ".join(artifact["artifact"] for artifact in artifacts)
      + f" <sentinel_{NEXT_BUCKET_LABEL}_artifact>"
      + f" <prefill_shape_{NEXT_BUCKET_LABEL}_artifact>"
  )

  artifact_routes = {
      artifact.get("route") for artifact in artifacts if isinstance(artifact, dict)
  }
  artifact_prefix_modes = {
      artifact.get("prefix_cache_enabled")
      for artifact in artifacts
      if isinstance(artifact, dict)
  }
  artifact_case_isolation = {
      artifact.get("case_process_isolation")
      for artifact in artifacts
      if isinstance(artifact, dict)
  }
  counts_by_kind = {
      kind: [row.get("prompt_token_count") for row in rows]
      for kind, rows in grouped.items()
  }
  max_projected_s = max(
      (
          projection.get("projected_wall_seconds", 0)
          for projection in projections
          if projection.get("projectable")
      ),
      default=0,
  )
  checks = [
      {
          "name": "accepted_rollup_checks_passed",
          "pass": rollup_correctness.get("required_checks_passed") is True
          and rollup_payload.get("required_checks_passed") is True,
      },
      {
          "name": "accepted_rollup_forbids_speedup_claims",
          "pass": rollup_correctness.get("speedup_claims_allowed") is False
          and rollup_payload.get("speedup_claims_allowed") is False,
      },
      {
          "name": "accepted_rollup_route_is_q6_pair",
          "pass": artifact_routes == {EXPECTED_ROUTE},
          "routes": sorted(str(route) for route in artifact_routes),
      },
      {
          "name": "accepted_rollup_is_cold_no_prefix",
          "pass": artifact_prefix_modes == {False},
          "prefix_cache_enabled": sorted(str(value) for value in artifact_prefix_modes),
      },
      {
          "name": "accepted_rollup_uses_case_process_isolation",
          "pass": artifact_case_isolation == {True},
          "case_process_isolation": sorted(str(value) for value in artifact_case_isolation),
      },
      {
          "name": "accepted_rollup_has_observed_counts_for_both_kinds",
          "pass": (
              counts_by_kind.get("sentinel_retrieval") == OBSERVED_COUNTS
              and counts_by_kind.get("prefill_shape") == OBSERVED_COUNTS
          ),
          "counts_by_kind": counts_by_kind,
      },
      {
          "name": "next_bucket_is_policy_bucket_only",
          "pass": NEXT_BUCKET_TOKENS == 32768 and DEFERRED_BUCKETS[0] == 65536,
          "deferred_buckets": DEFERRED_BUCKETS,
          "next_bucket_tokens": NEXT_BUCKET_TOKENS,
      },
      {
          "name": "next_jobs_are_single_case_isolated",
          "pass": all(job["process_policy"] == "isolated_single_case_target_process" for job in jobs)
          and all("--combined-process" not in command for command in commands),
      },
      {
          "name": "next_jobs_keep_q6_pair_route",
          "pass": all("--dense-q6-pair-dot" in command for command in commands),
      },
      {
          "name": "timeout_has_projection_margin",
          "pass": max_projected_s > 0 and args.timeout_s >= int(max_projected_s * 1.5),
          "max_projected_seconds": max_projected_s,
          "timeout_s": args.timeout_s,
      },
      {
          "name": "higher_buckets_deferred_until_policy_bucket_rollup",
          "pass": bool(DEFERRED_BUCKETS),
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  policy = {
      "accepted_rollup": rel(rollup_dir),
      "deferred_buckets": DEFERRED_BUCKETS,
      "decision": f"run_{NEXT_BUCKET_LABEL}_next_as_explicit_isolated_long_jobs_only",
      "full_context_ladder_claim_allowed": False,
      "next_bucket": {
          "jobs": jobs,
          "label": NEXT_BUCKET_LABEL,
          "post_run_rollup_command": post_run_rollup_command,
          "projections": projections,
          "timeout_s": args.timeout_s,
          "tokens": NEXT_BUCKET_TOKENS,
      },
      "policy_id": POLICY_ID,
      "policy_status": "closed" if required_checks_passed else "open_failed_checks",
      "prefix_cache_enabled": False,
      "route": EXPECTED_ROUTE,
      "run_policy_gate_closed": required_checks_passed,
      "speedup_claims_allowed": False,
      "start_conditions": [
          "target is reserved for long context jobs",
          "no concurrent micro-route experiment is using the target",
          "run one case per target process",
          "keep max_new_tokens=1 and prefix cache disabled",
          f"roll up {NEXT_BUCKET_LABEL} only after both {NEXT_BUCKET_LABEL} "
          "case artifacts pass required checks",
      ],
      "stop_before": f"064k_or_larger_until_{NEXT_BUCKET_LABEL}_rollup_is_accepted",
      "workstream": WORKSTREAM,
  }
  payload = {
      "created_at": created_at,
      "policy": policy,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-context-ladder-long-run-policy.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "policy.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "context_ladder_long_run_policy",
      "policy_id": POLICY_ID,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "context_ladder_long_run_policy",
      [
          ("next_bucket_tokens", NEXT_BUCKET_TOKENS),
          ("deferred_bucket_count", len(DEFERRED_BUCKETS)),
          ("job_count", len(jobs)),
          ("max_projected_seconds", max_projected_s),
          ("timeout_s", args.timeout_s),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
