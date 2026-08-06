#!/usr/bin/env python3
"""Validate native per-token SSE streaming semantics on the target."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import iq36_local


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r3-streaming-smoke-v0"
EXPECTED_SENTINEL_001K_SIGNATURE = [271, 198, 21134, 3054, 3437]
DEFAULT_ORACLE_BUNDLE_DIR = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
REQUIRED_ORACLE_BUNDLE_FILES = (
    "manifest.json",
    "correctness.json",
    "token-topk-references.jsonl",
    "teacher-forced-distribution-references.jsonl",
    "boundary-references/inputs.jsonl",
    "boundary-references/outputs.jsonl",
)
ZERO_ORACLE_BUNDLE_STATS = {
    "boundary_input_rows": 0,
    "boundary_output_rows": 0,
    "teacher_forced_distribution_rows": 0,
    "token_topk_rows": 0,
}


def load_context_helper() -> Any:
  path = ROOT / "tools/intel-qwen36-context-ladder-native-diagnostic.py"
  spec = importlib.util.spec_from_file_location("iq36_context_diag", path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"could not load context helper: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CTX = load_context_helper()


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=CTX.DEFAULT_HOST)
  parser.add_argument("--model", default=CTX.DEFAULT_MODEL)
  parser.add_argument("--env-script", default=CTX.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=CTX.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--token-id-refs", type=Path, default=CTX.DEFAULT_TOKEN_ID_REFS)
  parser.add_argument("--oracle-bundle-dir", type=Path, default=DEFAULT_ORACLE_BUNDLE_DIR)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--case-id", default="sentinel_001k")
  parser.add_argument("--max-new-tokens", type=int, default=4)
  parser.add_argument("--warmup-runs", type=int, default=0)
  parser.add_argument("--timed-runs", type=int, default=1)
  parser.add_argument("--no-resident-cache", dest="resident_cache", action="store_false")
  parser.add_argument("--no-resident-harness-load", dest="resident_harness_load", action="store_false")
  parser.set_defaults(resident_cache=True, resident_harness_load=True)
  parser.add_argument("--profile-matvec", action="store_true")
  parser.add_argument("--dense-q6-pair-dot", action="store_true", default=True)
  parser.add_argument("--selected-expert-down-q4-pair-dot", action="store_true")
  parser.add_argument("--selected-expert-down-q6-pair-dot", action="store_true", default=True)
  parser.add_argument("--q4-plane-layout", action="store_true", default=True)
  return parser.parse_args()


def count_nonempty_lines(path: Path) -> int:
  count = 0
  with path.open("r", encoding="utf-8") as fh:
    for line in fh:
      if line.strip():
        count += 1
  return count


def oracle_bundle_stats(bundle_dir: Path) -> dict[str, int]:
  if not bundle_dir.is_dir():
    raise SystemExit(f"oracle bundle dir missing: {bundle_dir}")
  missing = [
      relative
      for relative in REQUIRED_ORACLE_BUNDLE_FILES
      if not (bundle_dir / relative).is_file()
  ]
  if missing:
    raise SystemExit(f"oracle bundle missing required files: {missing}")
  return {
      "boundary_input_rows": count_nonempty_lines(
          bundle_dir / "boundary-references/inputs.jsonl"
      ),
      "boundary_output_rows": count_nonempty_lines(
          bundle_dir / "boundary-references/outputs.jsonl"
      ),
      "teacher_forced_distribution_rows": count_nonempty_lines(
          bundle_dir / "teacher-forced-distribution-references.jsonl"
      ),
      "token_topk_rows": count_nonempty_lines(
          bundle_dir / "token-topk-references.jsonl"
      ),
  }


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def parse_sse(stdout: str) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  for block in stdout.replace("\r\n", "\n").split("\n\n"):
    block = block.strip()
    if not block:
      continue
    event_name = ""
    data_lines: list[str] = []
    for line in block.splitlines():
      if line.startswith("event:"):
        event_name = line.split(":", 1)[1].strip()
      elif line.startswith("data:"):
        data_lines.append(line.split(":", 1)[1].strip())
    if not event_name or not data_lines:
      events.append({"event": "parse_error", "raw": block})
      continue
    data = json.loads("".join(data_lines))
    if not isinstance(data, dict):
      raise SystemExit("SSE data payload is not an object")
    data["event"] = event_name
    events.append(data)
  return events


def oracle_bundle_stage(
    host: str,
    bundle_dir: Path,
    remote_bundle_dir: str,
    timeout_s: int,
) -> dict[str, Any]:
  mkdir = iq36_local.run_target(
      host,
      "mkdir -p "
      + " ".join(
          shlex.quote(path)
          for path in (remote_bundle_dir, remote_bundle_dir + "/boundary-references")
      ),
      timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  if mkdir.get("returncode") == 0:
    for relative in REQUIRED_ORACLE_BUNDLE_FILES:
      transfers.append(
          iq36_local.copy_to(
              host,
              bundle_dir / relative,
              f"{remote_bundle_dir}/{relative}",
              timeout_s,
          )
      )
  return {
      "mkdir": mkdir,
      "remote_dir": remote_bundle_dir,
      "required_files": list(REQUIRED_ORACLE_BUNDLE_FILES),
      "transfers": transfers,
  }


def build_run_command(
    remote_dir: str,
    remote_token_dir: str,
    model: str,
    max_new_tokens: int,
    route_args: list[str],
    case_id: str,
) -> str:
  argv = [
      remote_dir + "/build/iq36-native-candidate-jsonl",
      model,
      remote_token_dir,
      str(max_new_tokens),
  ] + route_args + ["--stream-sse-events", case_id]
  return " ".join(shlex.quote(item) for item in argv)


def main() -> int:
  args = parse_args()
  if args.max_new_tokens < 1 or args.max_new_tokens > 32:
    raise SystemExit("--max-new-tokens must be 1..32 for streaming smoke")

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r3-streaming-smoke-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  token_dir = out_dir / "token-input"
  remote_dir = f"{args.remote_root}/r3-streaming-smoke-{stamp}"
  remote_token_dir = f"{remote_dir}/tokens"
  remote_oracle_bundle_dir = f"{remote_dir}/oracle-bundle"

  case_ids = [args.case_id]
  rows = CTX.selected_rows(args.token_id_refs, case_ids)
  token_manifest = CTX.prepare_token_inputs(rows, token_dir)
  expected_oracle_stats = (
      oracle_bundle_stats(args.oracle_bundle_dir)
      if args.resident_harness_load else dict(ZERO_ORACLE_BUNDLE_STATS)
  )
  mkdir, source_transfers = CTX.source_stage(args.host, remote_dir, args.timeout_s)
  token_transfers: list[dict[str, Any]] = []
  if mkdir.get("returncode") == 0 and all(
      item.get("returncode") == 0 for item in source_transfers
  ):
    token_transfers = CTX.token_stage(
        args.host,
        token_dir,
        remote_token_dir,
        token_manifest,
        args.timeout_s,
    )
  oracle_stage: dict[str, Any] = {
      "mkdir": {"returncode": 0, "stdout": "", "stderr": "disabled"},
      "remote_dir": None,
      "required_files": list(REQUIRED_ORACLE_BUNDLE_FILES),
      "transfers": [],
  }
  if (
      args.resident_harness_load
      and mkdir.get("returncode") == 0
      and all(item.get("returncode") == 0 for item in source_transfers)
      and bool(token_transfers)
      and all(item.get("returncode") == 0 for item in token_transfers)
  ):
    oracle_stage = oracle_bundle_stage(
        args.host,
        args.oracle_bundle_dir,
        remote_oracle_bundle_dir,
        args.timeout_s,
    )

  oracle_staged = (
      not args.resident_harness_load
      or (
          oracle_stage.get("mkdir", {}).get("returncode") == 0
          and bool(oracle_stage.get("transfers"))
          and all(
              item.get("returncode") == 0
              for item in oracle_stage.get("transfers", [])
          )
      )
  )

  staged = (
      mkdir.get("returncode") == 0
      and all(item.get("returncode") == 0 for item in source_transfers)
      and bool(token_transfers)
      and all(item.get("returncode") == 0 for item in token_transfers)
      and oracle_staged
  )
  build = (
      iq36_local.run_target(
          args.host,
          f"bash -lc {shlex.quote(CTX.build_command(remote_dir, args.env_script))}",
          args.timeout_s,
      )
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )

  route_args = CTX.accepted_route_args(args)
  if args.resident_harness_load:
    route_args += ["--resident-harness-bundle", remote_oracle_bundle_dir]
  run_command = build_run_command(
      remote_dir,
      remote_token_dir,
      args.model,
      args.max_new_tokens,
      route_args,
      args.case_id,
  )
  run = (
      iq36_local.run_target(args.host, run_command, args.timeout_s)
      if build.get("returncode") == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )
  stdout = str(run.get("stdout", ""))
  events = parse_sse(stdout) if run.get("returncode") == 0 else []
  token_events = [event for event in events if event.get("event") == "token"]
  done_events = [event for event in events if event.get("event") == "done"]
  done = done_events[-1] if done_events else {}

  token_ids = [event.get("token_id") for event in token_events]
  generated_indexes = [event.get("generated_index") for event in token_events]
  session_event_indexes = [
      event.get("resident_session_event_index") for event in token_events
  ]
  session_ids = [event.get("resident_session_id") for event in token_events]
  first_signature = (
      token_events[0].get("top_logprob_id_signature")
      if token_events else None
  )
  done_cases = done.get("cases") if isinstance(done, dict) else None
  done_case = done_cases[0] if isinstance(done_cases, list) and done_cases else {}
  done_generated = done_case.get("generated_token_ids", [])
  done_oracle_stats = done.get("resident_harness_oracle_bundle_stats")
  done_session_id = done.get("resident_session_id")
  done_session_token_count = done.get("resident_session_token_count")
  expected_indexes = list(range(len(token_events)))
  resident_event_api = "ResidentHarness"

  checks = [
      {"name": "token_id_refs_present", "pass": args.token_id_refs.exists()},
      {
          "name": "oracle_bundle_files_present",
          "pass": not args.resident_harness_load
          or all((args.oracle_bundle_dir / relative).is_file()
                 for relative in REQUIRED_ORACLE_BUNDLE_FILES),
      },
      {
          "name": "oracle_bundle_stats_counted",
          "pass": not args.resident_harness_load
          or all(value > 0 for value in expected_oracle_stats.values()),
      },
      {"name": "selected_case_present", "pass": len(rows) == 1},
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers)
          and all(item.get("returncode") == 0 for item in source_transfers),
      },
      {
          "name": "token_inputs_transferred",
          "pass": bool(token_transfers)
          and all(item.get("returncode") == 0 for item in token_transfers),
      },
      {
          "name": "resident_oracle_bundle_transferred",
          "pass": not args.resident_harness_load
          or (
              oracle_stage.get("mkdir", {}).get("returncode") == 0
              and bool(oracle_stage.get("transfers"))
              and all(
                  item.get("returncode") == 0
                  for item in oracle_stage.get("transfers", [])
              )
          ),
      },
      {"name": "target_native_runner_built", "pass": build.get("returncode") == 0},
      {"name": "target_streaming_runner_ran", "pass": run.get("returncode") == 0},
      {"name": "sse_events_parsed", "pass": bool(events) and not any(event.get("event") == "parse_error" for event in events)},
      {"name": "token_events_present", "pass": len(token_events) == args.max_new_tokens},
      {"name": "done_event_present", "pass": len(done_events) == 1},
      {"name": "token_indexes_contiguous", "pass": generated_indexes == expected_indexes},
      {
          "name": "done_matches_token_events",
          "pass": done_generated == token_ids,
      },
      {
          "name": "resident_event_api_recorded",
          "pass": bool(token_events)
          and all(event.get("resident_event_api") == resident_event_api
                  for event in token_events)
          and done.get("resident_event_api") == resident_event_api,
      },
      {
          "name": "resident_streaming_session_recorded",
          "pass": bool(token_events)
          and session_event_indexes == expected_indexes
          and all(session_id == done_session_id for session_id in session_ids)
          and isinstance(done_session_id, str)
          and bool(done_session_id)
          and done_session_token_count == len(token_events),
      },
      {
          "name": "route_flags_recorded",
          "pass": done.get("q4_plane_layout_enabled") is args.q4_plane_layout
          and done.get("dense_q6_pair_dot_enabled") is args.dense_q6_pair_dot
          and done.get("selected_expert_down_q6_pair_dot_enabled")
          is args.selected_expert_down_q6_pair_dot,
      },
      {
          "name": "resident_harness_loaded",
          "pass": not args.resident_harness_load
          or done.get("resident_harness_loaded") is True,
      },
      {
          "name": "resident_harness_stats_match",
          "pass": not args.resident_harness_load
          or done_oracle_stats == expected_oracle_stats,
      },
      {
          "name": "sentinel_001k_first_token_stable",
          "pass": args.case_id != "sentinel_001k" or (
              bool(token_events)
              and token_events[0].get("token_id") == 271
              and first_signature == EXPECTED_SENTINEL_001K_SIGNATURE
          ),
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)

  payload = {
      "case_id": args.case_id,
      "created_at": created_at,
      "done_event": done,
      "events": events,
      "max_new_tokens": args.max_new_tokens,
      "remote_dir": remote_dir,
      "remote_oracle_bundle_dir": (
          remote_oracle_bundle_dir if args.resident_harness_load else None
      ),
      "required_checks_passed": required_checks_passed,
      "resident_harness_expected_oracle_bundle_stats": expected_oracle_stats,
      "resident_harness_load_enabled": args.resident_harness_load,
      "route_args": route_args + ["--stream-sse-events"],
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "token_events": token_events,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "case_id": args.case_id,
      "max_new_tokens": args.max_new_tokens,
      "required_checks_passed": required_checks_passed,
      "resident_harness_load_enabled": args.resident_harness_load,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r3-streaming-smoke.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "streaming-smoke.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r3_streaming_smoke",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "source_files": CTX.SOURCE_FILES,
      "source_transfers": source_transfers,
      "oracle_bundle_stage": oracle_stage,
      "oracle_bundle_expected_stats": expected_oracle_stats,
      "token_transfers": token_transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "run.json", {
      "cmd": run_command,
      "returncode": run.get("returncode"),
      "stderr_tail": str(run.get("stderr", ""))[-4000:],
      "stdout_size_bytes": len(stdout.encode("utf-8")),
  })
  write_jsonl(out_dir / "events.jsonl", events)
  iq36_local.write_metric(
      out_dir / "metrics.jsonl",
      "r3_streaming_smoke",
      [
          ("token_event_count", len(token_events)),
          ("done_event_count", len(done_events)),
          ("resident_harness_loaded", done.get("resident_harness_loaded") is True),
          ("resident_harness_stats_match", done_oracle_stats == expected_oracle_stats),
          ("required_checks_passed", required_checks_passed),
          ("speedup_claims_allowed", False),
      ],
  )
  (out_dir / "summary.md").write_text(
      "\n".join([
          "# R3 Streaming Smoke",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- case: `{args.case_id}`",
          f"- token events: `{len(token_events)}`",
          f"- done events: `{len(done_events)}`",
          f"- resident harness loaded: `{str(done.get('resident_harness_loaded') is True).lower()}`",
          f"- resident harness stats match: `{str(done_oracle_stats == expected_oracle_stats).lower()}`",
          f"- required checks passed: `{str(required_checks_passed).lower()}`",
          f"- speedup claims allowed: `false`",
          "",
          "This artifact validates per-token SSE event shape only. It is not a",
          "throughput benchmark or speedup claim.",
          "",
      ]),
      encoding="utf-8",
  )
  print(str(out_dir.relative_to(ROOT)))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
