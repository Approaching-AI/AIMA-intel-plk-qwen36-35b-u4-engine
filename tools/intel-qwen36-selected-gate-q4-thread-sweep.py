#!/usr/bin/env python3
"""Capture a selected gate/up Q4 direct-dot thread-count sweep.

This is route-diagnostic evidence only. It runs the native candidate generator
with the current post-R1 route stack and varies ``--selected-expert-ffn-threads``
while keeping the selected gate/up Q4 direct-dot route enabled. The artifact
records deterministic token/top-k agreement and the fused selected gate/up
profile rows for each thread count.
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
SCHEMA_VERSION = "intel-qwen36-selected-gate-q4-thread-sweep-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
CANDIDATE_TOOL = ROOT / "tools/intel-qwen36-r1-native-candidate-jsonl.py"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_thread_counts(value: str) -> list[int]:
  counts: list[int] = []
  seen: set[int] = set()
  for item in value.split(","):
    item = item.strip()
    if not item:
      continue
    count = int(item)
    if count < 1 or count > 256:
      raise argparse.ArgumentTypeError("thread counts must be 1..256")
    if count not in seen:
      counts.append(count)
      seen.add(count)
  if not counts:
    raise argparse.ArgumentTypeError("at least one thread count is required")
  return counts


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--max-new-tokens", type=int, default=1)
  parser.add_argument("--warmup-runs", type=int, default=0)
  parser.add_argument("--timed-runs", type=int, default=1)
  parser.add_argument(
      "--threads",
      type=parse_thread_counts,
      default=parse_thread_counts("1,2,4,8,16,32"),
  )
  parser.add_argument(
      "--case-id",
      action="append",
      default=[],
      help="Case id to run. Repeatable. Defaults to short_math_001.",
  )
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def rel(path: Path) -> str:
  return path.resolve().relative_to(ROOT).as_posix()


def profile_rows(stdout: dict[str, Any], op: str) -> list[dict[str, Any]]:
  rows = stdout.get("matvec_profile", [])
  if not isinstance(rows, list):
    return []
  return [
      row
      for row in rows
      if isinstance(row, dict) and row.get("op") == op
  ]


def aggregate_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
  total_ns = sum(row.get("total_ns", 0) for row in rows if isinstance(row.get("total_ns"), int))
  call_count = sum(row.get("call_count", 0) for row in rows if isinstance(row.get("call_count"), int))
  row_count = sum(row.get("row_count", 0) for row in rows if isinstance(row.get("row_count"), int))
  max_ns = max(
      [row.get("max_ns", 0) for row in rows if isinstance(row.get("max_ns"), int)]
      or [0]
  )
  return {
      "call_count": call_count,
      "max_ns": max_ns,
      "profile_row_count": len(rows),
      "row_count": row_count,
      "total_ns": total_ns,
      "weighted_average_ns": total_ns // call_count if call_count else 0,
  }


def case_signatures(stdout: dict[str, Any]) -> dict[str, dict[str, Any]]:
  signatures: dict[str, dict[str, Any]] = {}
  for case in stdout.get("cases", []):
    if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
      continue
    signatures[case["case_id"]] = {
        "first_token_top_logprob_id_signature": case.get(
            "first_token_top_logprob_id_signature"
        ),
        "generated_token_ids": case.get("generated_token_ids"),
        "prompt_token_count": case.get("prompt_token_count"),
    }
  return signatures


def timed_total_ns(stdout: dict[str, Any]) -> int | None:
  timed_runs = stdout.get("timed_runs", [])
  if not timed_runs or not isinstance(timed_runs[0], dict):
    return None
  value = timed_runs[0].get("total_ns")
  return value if isinstance(value, int) else None


def run_candidate(args: argparse.Namespace, out_dir: Path, threads: int) -> dict[str, Any]:
  candidate_out_dir = out_dir / f"thread-{threads}"
  cmd = [
      "python3",
      str(CANDIDATE_TOOL),
      "--host",
      args.host,
      "--model",
      args.model,
      "--env-script",
      args.env_script,
      "--remote-root",
      args.remote_root,
      "--out-dir",
      str(candidate_out_dir),
      "--timeout-s",
      str(args.timeout_s),
      "--max-new-tokens",
      str(args.max_new_tokens),
      "--warmup-runs",
      str(args.warmup_runs),
      "--timed-runs",
      str(args.timed_runs),
      "--resident-cache",
      "--profile-matvec",
      "--lm-head-top-k",
      "--lm-head-threads",
      "16",
      "--expert-slice-matvec",
      "--expert-slice-threads",
      "16",
      "--dense-matvec",
      "--dense-matvec-threads",
      "16",
      "--dense-matvec-min-rows",
      "256",
      "--shared-parallel-executor",
      "--selected-expert-ffn",
      "--selected-expert-ffn-threads",
      str(threads),
      "--selected-gate-q4-direct-dot",
  ]
  for case_id in args.case_id:
    cmd += ["--case-id", case_id]

  process = iq36_local.run(cmd, args.timeout_s + 120)
  stdout = load_json(candidate_out_dir / "native-candidate-stdout.json")
  correctness = load_json(candidate_out_dir / "correctness.json")
  gate = load_json(candidate_out_dir / "gate" / "gate.json")
  gate_rows = profile_rows(stdout, "selected_expert_ffn_gate_swiglu_q4direct")
  down_rows = profile_rows(stdout, "selected_expert_ffn_down_aggregate")
  return {
      "artifact": rel(candidate_out_dir),
      "case_signatures": case_signatures(stdout),
      "cmd": cmd,
      "correctness": correctness,
      "generated_case_count": (
          len(stdout.get("cases", [])) if isinstance(stdout.get("cases"), list) else 0
      ),
      "gate_closed": gate.get("r1_native_correctness_gate", {}).get(
          "r1_native_correctness_gate_closed"
      ),
      "process": process,
      "q4direct_gate_profile": aggregate_profile(gate_rows),
      "q4direct_gate_profile_top": gate_rows[:10],
      "selected_down_profile": aggregate_profile(down_rows),
      "selected_expert_ffn_enabled": stdout.get("selected_expert_ffn_enabled"),
      "selected_expert_ffn_threads": stdout.get("selected_expert_ffn_threads"),
      "selected_gate_q4_direct_dot_enabled": stdout.get(
          "selected_gate_q4_direct_dot_enabled"
      ),
      "thread_count": threads,
      "timed_total_ns": timed_total_ns(stdout),
  }


def signatures_match(results: list[dict[str, Any]]) -> bool:
  if not results:
    return False
  first = results[0].get("case_signatures")
  return all(result.get("case_signatures") == first for result in results)


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# Selected Gate Q4 Thread Sweep",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- case ids: `{payload['case_ids']}`",
      f"- max new tokens: {payload['max_new_tokens']}",
      f"- thread counts: `{payload['thread_counts']}`",
      f"- signatures match: `{str(payload['diagnostic']['signatures_match']).lower()}`",
      f"- speedup claims allowed: `{str(payload['speedup_claims_allowed']).lower()}`",
      "",
      "| threads | artifact | timed total ns | q4direct total ns | q4direct avg ns | down total ns |",
      "|---:|---|---:|---:|---:|---:|",
  ]
  for result in payload["results"]:
    gate_profile = result.get("q4direct_gate_profile", {})
    down_profile = result.get("selected_down_profile", {})
    lines.append(
        "| {threads} | `{artifact}` | {timed} | {gate_total} | {gate_avg} | {down_total} |".format(
            threads=result.get("thread_count"),
            artifact=result.get("artifact"),
            timed=result.get("timed_total_ns"),
            gate_total=gate_profile.get("total_ns"),
            gate_avg=gate_profile.get("weighted_average_ns"),
            down_total=down_profile.get("total_ns"),
        )
    )
  lines += [
      "",
      "This artifact is route-diagnostic evidence only. It records raw timings",
      "from narrow runs and does not promote a speedup claim.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.max_new_tokens < 1 or args.max_new_tokens > 16:
    raise SystemExit("--max-new-tokens must be 1..16")
  if args.warmup_runs < 0 or args.warmup_runs > 8:
    raise SystemExit("--warmup-runs must be 0..8")
  if args.timed_runs < 1 or args.timed_runs > 8:
    raise SystemExit("--timed-runs must be 1..8")
  if not args.case_id:
    args.case_id = ["short_math_001"]

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/selected-gate-q4-thread-sweep-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  results = [run_candidate(args, out_dir, threads) for threads in args.threads]
  all_candidates_ran = all(result["process"].get("returncode") == 0 for result in results)
  all_checks_passed = all(
      result["correctness"].get("required_checks_passed") is True for result in results
  )
  all_routes_enabled = all(
      result.get("selected_expert_ffn_enabled") is True
      and result.get("selected_expert_ffn_threads") == result.get("thread_count")
      and result.get("selected_gate_q4_direct_dot_enabled") is True
      for result in results
  )
  all_profiles_present = all(
      result.get("q4direct_gate_profile", {}).get("profile_row_count") == 40
      and result.get("q4direct_gate_profile", {}).get("total_ns", 0) > 0
      for result in results
  )
  deterministic_signatures = signatures_match(results)

  diagnostic = {
      "all_candidate_checks_passed": all_checks_passed,
      "all_candidate_runs_executed": all_candidates_ran,
      "all_q4direct_profiles_present": all_profiles_present,
      "all_routes_enabled": all_routes_enabled,
      "signatures_match": deterministic_signatures,
  }
  checks = [
      {"name": "candidate_runs_executed", "pass": all_candidates_ran},
      {"name": "candidate_checks_passed", "pass": all_checks_passed},
      {"name": "selected_gate_q4_routes_enabled", "pass": all_routes_enabled},
      {"name": "selected_gate_q4_profiles_present", "pass": all_profiles_present},
      {"name": "thread_signatures_match", "pass": deterministic_signatures},
      {"name": "speedup_claims_forbidden", "pass": True},
  ]
  payload = {
      "case_ids": args.case_id,
      "created_at": created_at,
      "diagnostic": diagnostic,
      "host": args.host,
      "max_new_tokens": args.max_new_tokens,
      "model_path": args.model,
      "results": results,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "thread_counts": args.threads,
      "timed_runs": args.timed_runs,
      "tool": "tools/intel-qwen36-selected-gate-q4-thread-sweep.py",
      "warmup_runs": args.warmup_runs,
      "workstream": WORKSTREAM,
  }

  iq36_local.write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "case_ids": args.case_id,
      "host": args.host,
      "max_new_tokens": args.max_new_tokens,
      "model_path": args.model,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "thread_counts": args.threads,
      "tool": "tools/intel-qwen36-selected-gate-q4-thread-sweep.py",
      "workstream": WORKSTREAM,
  })
  iq36_local.write_json(out_dir / "diagnostic.json", payload)
  iq36_local.write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "selected_gate_q4_thread_sweep",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  metrics: list[tuple[str, Any]] = [
      ("thread_count_count", len(args.threads)),
      ("signatures_match", deterministic_signatures),
      ("speedup_claims_allowed", False),
  ]
  for result in results:
    thread = result["thread_count"]
    gate_profile = result.get("q4direct_gate_profile", {})
    down_profile = result.get("selected_down_profile", {})
    metrics += [
        (f"thread_{thread}_timed_total_ns", result.get("timed_total_ns")),
        (f"thread_{thread}_q4direct_gate_total_ns", gate_profile.get("total_ns")),
        (
            f"thread_{thread}_q4direct_gate_average_ns",
            gate_profile.get("weighted_average_ns"),
        ),
        (f"thread_{thread}_selected_down_total_ns", down_profile.get("total_ns")),
    ]
  iq36_local.write_metric(out_dir / "metrics.jsonl", "selected_gate_q4_thread_sweep", metrics)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")

  print(f"selected gate Q4 thread sweep output: {out_dir}")
  return 0 if all(check["pass"] for check in checks) else 1


if __name__ == "__main__":
  raise SystemExit(main())
