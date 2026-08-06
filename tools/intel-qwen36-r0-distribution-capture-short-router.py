#!/usr/bin/env python3
"""Capture the six short/router teacher-forced distribution rows."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-distribution-capture-short-router-v0"
SMOKE_TOOL = ROOT / "tools/intel-qwen36-r0-distribution-capture-smoke.py"
QUEUE_PATH = ROOT / "output/r0-oracle-capture-queue-20260626T074119Z/teacher-forced-distribution-tasks.jsonl"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default="local")
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--request-timeout-s", type=int, default=900)
  parser.add_argument("--ready-timeout-s", type=int, default=420)
  parser.add_argument("--poll-interval-s", type=int, default=2)
  parser.add_argument("--port-base", type=int, default=18150)
  parser.add_argument(
      "--max-cases",
      type=int,
      default=6,
      help="Maximum short/router cases to capture. Default captures all six.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-distribution-capture-short-router-<UTC>.",
  )
  return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    value = json.load(handle)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      text = line.strip()
      if not text:
        continue
      try:
        value = json.loads(text)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def short_router_tasks() -> list[dict[str, Any]]:
  rows = load_jsonl(QUEUE_PATH)
  selected = [
      row for row in rows
      if row.get("prompt_set") in {"short", "router-stability"}
  ]
  if len(selected) < 6:
    raise SystemExit("capture queue does not contain six short/router distribution tasks")
  return selected[:6]


def run_smoke(
    *,
    args: argparse.Namespace,
    case_dir: Path,
    case_id: str,
    max_new_tokens: int,
    port: int,
) -> dict[str, Any]:
  cmd = [
      "python3",
      str(SMOKE_TOOL),
      "--host",
      args.host,
      "--case-id",
      case_id,
      "--max-new-tokens",
      str(max_new_tokens),
      "--port",
      str(port),
      "--timeout-s",
      str(args.timeout_s),
      "--request-timeout-s",
      str(args.request_timeout_s),
      "--ready-timeout-s",
      str(args.ready_timeout_s),
      "--poll-interval-s",
      str(args.poll_interval_s),
      "--out-dir",
      str(case_dir),
  ]
  result = subprocess.run(
      cmd,
      cwd=ROOT,
      check=False,
      capture_output=True,
      text=True,
      encoding="utf-8",
      errors="replace",
      timeout=args.timeout_s + 120,
  )
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stderr": result.stderr,
      "stdout": result.stdout,
  }


def normalize_row(row: dict[str, Any], *, task: dict[str, Any], source_path: Path) -> dict[str, Any]:
  copied = dict(row)
  requested_tokens = int(task["required_output_token_counts"][0])
  copied["bundle_jsonl_path"] = "teacher-forced-distribution-references.jsonl"
  copied["capture_mode"] = "current_target_llama_server_completion_probabilities_short_router_subset"
  copied["capture_status"] = "captured_short_router_subset"
  copied["queue_task_id"] = task["task_id"]
  copied["requested_output_token_count"] = requested_tokens
  copied["required_output_token_counts"] = task["required_output_token_counts"]
  copied["source_smoke_jsonl"] = str(source_path.resolve().relative_to(ROOT))
  copied["stopped_before_request_limit"] = (
      len(copied.get("distribution_positions", [])) < requested_tokens
  )
  copied["limitations"] = {
      "full_acceptance_context_ladder": False,
      "not_a_full_r0_oracle_bundle": True,
      "not_a_per_boundary_tensor_bundle": True,
      "short_router_subset_only": True,
  }
  copied["schema_version"] = SCHEMA_VERSION
  return copied


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# R0 Short/Router Distribution Capture",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- captured cases: {payload['captured_case_count']}",
      f"- total distribution positions: {payload['total_distribution_positions']}",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This fills the short/router distribution subset only. It is not a full",
      "acceptance-ladder distribution bundle and not a per-boundary oracle bundle.",
      "",
      "| Case | Requested tokens | Captured positions | Stopped early |",
      "|---|---:|---:|---:|",
  ]
  for case in payload["case_results"]:
    lines.append(
        f"| `{case['case_id']}` | {case['requested_tokens']} | "
        f"{case['captured_positions']} | "
        f"`{str(case['stopped_before_request_limit']).lower()}` |"
    )
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-distribution-capture-short-router-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  tasks = short_router_tasks()[:args.max_cases]
  if not tasks:
    raise SystemExit("no tasks selected")

  rows = []
  case_results = []
  command_results = []
  for index, task in enumerate(tasks):
    case_id = str(task["case_id"])
    token_counts = task.get("required_output_token_counts")
    if not isinstance(token_counts, list) or len(token_counts) != 1:
      raise SystemExit(f"{case_id}: expected one required output token count")
    max_new_tokens = int(token_counts[0])
    case_dir = out_dir / f"case-{index + 1:02d}-{case_id}"
    result = run_smoke(
        args=args,
        case_dir=case_dir,
        case_id=case_id,
        max_new_tokens=max_new_tokens,
        port=args.port_base + index,
    )
    command_results.append({
        "case_id": case_id,
        "returncode": result["returncode"],
        "stderr_path": str((case_dir / "invoke.stderr").relative_to(ROOT)),
        "stdout_path": str((case_dir / "invoke.stdout").relative_to(ROOT)),
      })
    (case_dir / "invoke.stdout").write_text(result["stdout"], encoding="utf-8")
    (case_dir / "invoke.stderr").write_text(result["stderr"], encoding="utf-8")
    correctness_path = case_dir / "correctness.json"
    smoke_jsonl_path = case_dir / "distribution-smoke.jsonl"
    if result["returncode"] != 0 or not correctness_path.is_file() or not smoke_jsonl_path.is_file():
      case_results.append({
          "captured_positions": 0,
          "case_id": case_id,
          "correctness_passed": False,
          "distribution_row_valid": False,
          "requested_tokens": max_new_tokens,
          "stopped_before_request_limit": False,
      })
      continue
    correctness = load_json(correctness_path)
    smoke_rows = load_jsonl(smoke_jsonl_path)
    smoke_row = smoke_rows[0] if smoke_rows else {}
    normalized = normalize_row(smoke_row, task=task, source_path=smoke_jsonl_path)
    rows.append(normalized)
    captured_positions = len(normalized.get("distribution_positions", []))
    distribution_row_valid = (
        normalized.get("workstream") == WORKSTREAM
        and normalized.get("request_status") == 200
        and 0 < captured_positions <= max_new_tokens
        and all(
            position.get("top_logprobs")
            for position in normalized.get("distribution_positions", [])
        )
    )
    case_results.append({
        "captured_positions": captured_positions,
        "case_id": case_id,
        "correctness_passed": correctness.get("required_checks_passed") is True,
        "distribution_row_valid": distribution_row_valid,
        "requested_tokens": max_new_tokens,
        "stopped_before_request_limit": captured_positions < max_new_tokens,
    })

  out_jsonl = out_dir / "teacher-forced-distribution-short-router.jsonl"
  with out_jsonl.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
  total_positions = sum(len(row.get("distribution_positions", [])) for row in rows)
  checks = [
      {
          "name": "all_selected_cases_completed",
          "pass": len(rows) == len(tasks) and all(result["returncode"] == 0 for result in command_results),
          "captured": len(rows),
          "selected": len(tasks),
      },
      {
          "name": "all_case_correctness_passed",
          "pass": all(case["correctness_passed"] for case in case_results),
      },
      {
          "name": "all_case_distribution_rows_valid",
          "pass": all(case["distribution_row_valid"] for case in case_results),
      },
      {
          "name": "captured_positions_within_requested_limit",
          "pass": all(
              0 < case["captured_positions"] <= case["requested_tokens"]
              for case in case_results
          ),
      },
      {
          "name": "rows_are_current_workstream",
          "pass": all(row.get("workstream") == WORKSTREAM for row in rows),
      },
      {
          "name": "top_logprobs_present",
          "pass": bool(rows)
          and all(
              position.get("top_logprobs")
              for row in rows
              for position in row.get("distribution_positions", [])
          ),
      },
      {
          "name": "subset_does_not_claim_full_bundle",
          "pass": all(
              row.get("limitations", {}).get("short_router_subset_only") is True
              and row.get("limitations", {}).get("not_a_full_r0_oracle_bundle") is True
              for row in rows
          ),
      },
      {
          "name": "oracle_gate_remains_open",
          "pass": True,
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  payload = {
      "captured_case_count": len(rows),
      "case_results": case_results,
      "command_results": command_results,
      "created_at": created_at,
      "host": args.host,
      "r0_oracle_gate_closed": False,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "selected_case_count": len(tasks),
      "stopped_before_request_count": sum(
          1 for case in case_results if case["stopped_before_request_limit"]
      ),
      "teacher_forced_distribution_jsonl": str(out_jsonl.relative_to(ROOT)),
      "total_distribution_positions": total_positions,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-distribution-capture-short-router.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "capture.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_distribution_capture_short_router",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in (
        ("captured_case_count", len(rows)),
        ("selected_case_count", len(tasks)),
        ("total_distribution_positions", total_positions),
        ("stopped_before_request_count", payload["stopped_before_request_count"]),
        ("required_checks_passed", required_checks_passed),
        ("r0_oracle_gate_closed", False),
    ):
      handle.write(json.dumps({
          "metric": metric,
          "phase": "r0_distribution_capture_short_router",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"short/router distribution capture output: {out_dir}")
  if not required_checks_passed:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
