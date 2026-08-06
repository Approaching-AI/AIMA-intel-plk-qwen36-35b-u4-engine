#!/usr/bin/env python3
"""Run the patched llama.cpp R0 boundary capture executable on target."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-boundary-capture-run-v0"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
REMOTE_CAPTURE_ROOT = "/home/intel/intel-qwen36-r0/captures"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--case-id", default="short_math_001")
  parser.add_argument("--threads", type=int, default=1)
  parser.add_argument("--n-ctx", type=int, default=32)
  parser.add_argument("--ngl", type=int, default=0)
  parser.add_argument(
      "--max-tensors",
      type=int,
      default=0,
      help="Pass through to capture executable. 0 means unlimited.",
  )
  parser.add_argument(
      "--filter",
      action="append",
      default=[],
      help="Additional tensor-name regex passed through to the capture executable.",
  )
  parser.add_argument(
      "--timeout-s",
      type=int,
      default=3600,
      help="Local timeout for the target capture run.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-boundary-capture-run-<UTC>.",
  )
  return parser.parse_args()


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_number}: expected JSON object")
      rows.append(value)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def run(cmd: list[str], *, timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "command": cmd,
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr + f"\nlocal timeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_script: str, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_script, timeout_s)


def parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key:
      values[key.strip()] = value.strip()
  return values


def find_prompt_row(case_id: str, materialized_prompts_path: Path) -> dict[str, Any]:
  for row in load_jsonl(materialized_prompts_path):
    if row.get("case_id") == case_id:
      return row
  raise SystemExit(f"{materialized_prompts_path}: missing case_id {case_id}")


def copy_staged_output(host: str, staged_out_dir: str, local_dir: Path) -> dict[str, Any]:
  local_dir.mkdir(parents=True, exist_ok=True)
  return iq36_local.copy_tree_from(host, staged_out_dir, local_dir, 600)


def analyze_capture(remote_copy_dir: Path) -> dict[str, Any]:
  summary_path = remote_copy_dir / "capture-summary.json"
  topk_path = remote_copy_dir / "sampler-topk.json"
  tensor_jsonl_path = remote_copy_dir / "tensor-dumps.jsonl"
  payload_dir = remote_copy_dir / "payloads"
  summary = load_json(summary_path) if summary_path.exists() else {}
  topk = load_json(topk_path) if topk_path.exists() else {}
  tensor_rows = load_jsonl(tensor_jsonl_path) if tensor_jsonl_path.exists() else []
  payload_paths = sorted(payload_dir.glob("*.bin")) if payload_dir.exists() else []
  observed_positions = sorted({
      row.get("observed_token_position")
      for row in tensor_rows
      if isinstance(row.get("observed_token_position"), int)
  })
  tensor_names = sorted({
      row.get("tensor_name")
      for row in tensor_rows
      if isinstance(row.get("tensor_name"), str)
  })
  payload_bytes_total = sum(path.stat().st_size for path in payload_paths)
  tensor_bytes_total = sum(
      int(row.get("nbytes", 0))
      for row in tensor_rows
      if isinstance(row.get("nbytes"), int)
  )
  return {
      "capture_summary_present": summary_path.exists(),
      "captured_tensor_count": summary.get("captured_tensor_count"),
      "logits_present": summary.get("logits_present"),
      "observed_positions": observed_positions,
      "payload_bytes_total": payload_bytes_total,
      "payload_file_count": len(payload_paths),
      "prompt_token_count": summary.get("prompt_token_count"),
      "sampler_topk_present": topk_path.exists(),
      "source_token_position": summary.get("source_token_position"),
      "tensor_bytes_total": tensor_bytes_total,
      "tensor_dump_jsonl_present": tensor_jsonl_path.exists(),
      "tensor_jsonl_row_count": len(tensor_rows),
      "unique_tensor_name_count": len(tensor_names),
      "unique_tensor_names_sample": tensor_names[:40],
  }


def build_summary(payload: dict[str, Any]) -> str:
  run_route = payload["capture_run"]
  analysis = payload["capture_analysis"]
  lines = [
      "# R0 Boundary Capture Run",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- executable: `{run_route['executable_path']}`",
      f"- remote output: `{run_route['remote_out_dir']}`",
      f"- case id: `{run_route['case_id']}`",
      f"- source token position: `{run_route['source_token_position']}`",
      f"- return code: `{run_route['returncode']}`",
      f"- captured tensors: `{analysis['captured_tensor_count']}`",
      f"- tensor JSONL rows: `{analysis['tensor_jsonl_row_count']}`",
      f"- payload files: `{analysis['payload_file_count']}`",
      f"- route status: `{payload['route_status']}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This artifact runs the first locked-model boundary tensor capture route.",
      "It is capture evidence only until mapped into the required oracle bundle",
      "input/output JSONLs and validated with the full bundle validator.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-boundary-capture-run-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  remote_copy_dir = out_dir / "remote-output"
  raw_dir.mkdir(parents=True, exist_ok=True)

  build_path = latest("r0-boundary-capture-build-*", "build.json")
  if build_path is None:
    raise SystemExit("no latest boundary capture build artifact found")
  build = load_json(build_path)
  build_correctness = load_json(build_path.parent / "correctness.json")
  executable_path = build.get("build_route", {}).get("executable_path")
  if not isinstance(executable_path, str) or not executable_path:
    raise SystemExit("latest boundary capture build missing executable path")

  queue_path = latest("r0-oracle-capture-queue-*", "capture-queue.json")
  if queue_path is None:
    raise SystemExit("no latest oracle capture queue artifact found")
  queue = load_json(queue_path)
  boundary_source = queue.get("boundary_capture_source", {})
  source_token_position = boundary_source.get("source_token_position")
  if not isinstance(source_token_position, int):
    raise SystemExit("latest oracle capture queue missing source token position")
  if boundary_source.get("case_id") != args.case_id:
    raise SystemExit(
        f"requested case {args.case_id} does not match queue case "
        f"{boundary_source.get('case_id')}"
    )

  materialized_path = latest(
      "r0-oracle-prompt-materialization-*",
      "materialized-prompts.jsonl",
  )
  if materialized_path is None:
    raise SystemExit("no latest prompt materialization artifact found")
  prompt_row = find_prompt_row(args.case_id, materialized_path)
  remote_prompt_path = prompt_row.get("remote_prompt_path")
  if not isinstance(remote_prompt_path, str) or not remote_prompt_path:
    raise SystemExit("prompt materialization row missing remote_prompt_path")

  remote_out_dir = f"{REMOTE_CAPTURE_ROOT}/boundary-capture-run-{stamp}"
  cmd_parts = [
      shlex.quote(executable_path),
      "--model",
      shlex.quote(MODEL_PATH),
      "--prompt-file",
      shlex.quote(remote_prompt_path),
      "--out-dir",
      shlex.quote(remote_out_dir),
      "--case-id",
      shlex.quote(args.case_id),
      "--source-token-position",
      str(source_token_position),
      "--threads",
      str(args.threads),
      "--n-ctx",
      str(args.n_ctx),
      "--ngl",
      str(args.ngl),
  ]
  if args.max_tensors > 0:
    cmd_parts.extend(["--max-tensors", str(args.max_tensors)])
  for extra_filter in args.filter:
    cmd_parts.extend(["--filter", shlex.quote(extra_filter)])
  capture_command = " ".join(cmd_parts)
  remote_script = "\n".join([
      "set -u",
      f"exe={shlex.quote(executable_path)}",
      f"model={shlex.quote(MODEL_PATH)}",
      f"prompt={shlex.quote(remote_prompt_path)}",
      f"out={shlex.quote(remote_out_dir)}",
      "rm -rf \"$out\"",
      "mkdir -p \"$out\"",
      "test -x \"$exe\" || { echo executable_missing=true; exit 127; }",
      "test -f \"$model\" || { echo model_missing=true; exit 127; }",
      "test -f \"$prompt\" || { echo prompt_missing=true; exit 127; }",
      f"{capture_command}",
      "rc=$?",
      "printf 'remote_returncode=%s\\n' \"$rc\"",
      "printf 'remote_out_dir=%s\\n' \"$out\"",
      "printf 'capture_summary_present='; test -f \"$out/capture-summary.json\" && echo true || echo false",
      "printf 'sampler_topk_present='; test -f \"$out/sampler-topk.json\" && echo true || echo false",
      "printf 'tensor_dump_jsonl_present='; test -f \"$out/tensor-dumps.jsonl\" && echo true || echo false",
      "printf 'tensor_dump_row_count='; test -f \"$out/tensor-dumps.jsonl\" && wc -l < \"$out/tensor-dumps.jsonl\" || echo 0",
      "printf 'payload_file_count='; test -d \"$out/payloads\" && find \"$out/payloads\" -type f -name '*.bin' | wc -l || echo 0",
      "exit \"$rc\"",
  ])
  capture = run_target(args.host, remote_script, timeout_s=args.timeout_s)
  (raw_dir / "capture.stdout").write_text(capture["stdout"], encoding="utf-8")
  (raw_dir / "capture.stderr").write_text(capture["stderr"], encoding="utf-8")

  copy_result = {
      "command": [],
      "returncode": 127,
      "stdout": "",
      "stderr": "capture did not run successfully; staged output not copied",
      "timed_out": False,
  }
  if capture["returncode"] == 0:
    copy_result = copy_staged_output(args.host, remote_out_dir, remote_copy_dir)
  (raw_dir / "copy.stdout").write_text(copy_result["stdout"], encoding="utf-8")
  (raw_dir / "copy.stderr").write_text(copy_result["stderr"], encoding="utf-8")

  analysis = analyze_capture(remote_copy_dir) if copy_result["returncode"] == 0 else {
      "capture_summary_present": False,
      "captured_tensor_count": 0,
      "logits_present": False,
      "observed_positions": [],
      "payload_bytes_total": 0,
      "payload_file_count": 0,
      "prompt_token_count": None,
      "sampler_topk_present": False,
      "source_token_position": None,
      "tensor_bytes_total": 0,
      "tensor_dump_jsonl_present": False,
      "tensor_jsonl_row_count": 0,
      "unique_tensor_name_count": 0,
      "unique_tensor_names_sample": [],
  }
  remote_values = parse_key_values(capture["stdout"])
  route_status = (
      "boundary_capture_run_succeeded"
      if capture["returncode"] == 0
      and copy_result["returncode"] == 0
      and analysis.get("captured_tensor_count", 0) > 0
      and analysis.get("tensor_jsonl_row_count", 0) > 0
      and analysis.get("payload_file_count", 0) == analysis.get("tensor_jsonl_row_count", -1)
      and analysis.get("observed_positions") == [source_token_position]
      else "boundary_capture_run_failed"
  )
  payload = {
      "capture_analysis": analysis,
      "capture_run": {
          "case_id": args.case_id,
          "executable_path": executable_path,
          "max_tensors": args.max_tensors,
          "extra_filters": args.filter,
          "model": {
              "path": MODEL_PATH,
              "sha256": MODEL_SHA256,
          },
          "n_ctx": args.n_ctx,
          "ngl": args.ngl,
          "prompt_observed_tokens": prompt_row.get("observed_prompt_tokens"),
          "prompt_remote_path": remote_prompt_path,
          "remote_out_dir": remote_values.get("remote_out_dir", remote_out_dir),
          "returncode": capture["returncode"],
          "source_token_position": source_token_position,
          "threads": args.threads,
          "timed_out": capture["timed_out"],
      },
      "created_at": created_at,
      "evidence": {
          "build": rel(build_path.parent),
          "capture_queue": rel(queue_path.parent),
          "materialized_prompts": rel(materialized_path.parent),
          "raw_dir": rel(raw_dir),
          "remote_output_copy": rel(remote_copy_dir) if remote_copy_dir.exists() else None,
      },
      "host": args.host,
      "r0_oracle_gate_closed": False,
      "route_status": route_status,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "latest_build_available",
          "pass": build.get("route_status") == "boundary_capture_executable_built"
          and build_correctness.get("required_checks_passed") is True,
      },
      {
          "name": "queue_source_matches_case",
          "pass": boundary_source.get("case_id") == args.case_id
          and source_token_position == 15,
      },
      {
          "name": "prompt_materialization_matches_queue",
          "pass": prompt_row.get("observed_prompt_tokens")
          == boundary_source.get("prompt_token_count")
          and prompt_row.get("remote_prompt_path") == remote_prompt_path,
      },
      {
          "name": "capture_command_succeeded",
          "pass": capture["returncode"] == 0 and capture["timed_out"] is False,
      },
      {
          "name": "remote_output_copied",
          "pass": copy_result["returncode"] == 0 and copy_result["timed_out"] is False,
      },
      {
          "name": "capture_outputs_present",
          "pass": analysis.get("capture_summary_present") is True
          and analysis.get("sampler_topk_present") is True
          and analysis.get("tensor_dump_jsonl_present") is True,
      },
      {
          "name": "tensor_payloads_match_jsonl",
          "pass": analysis.get("tensor_jsonl_row_count", 0) > 0
          and analysis.get("payload_file_count")
          == analysis.get("tensor_jsonl_row_count"),
      },
      {
          "name": "capture_position_matches_queue",
          "pass": analysis.get("source_token_position") == source_token_position
          and analysis.get("observed_positions") == [source_token_position],
      },
      {
          "name": "capture_run_does_not_close_oracle_gate",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  correctness = {
      "checks": checks,
      "gate": "r0_boundary_capture_run",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-boundary-capture-run.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "capture-run.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("capture_returncode", capture["returncode"]),
        ("copy_returncode", copy_result["returncode"]),
        ("captured_tensor_count", analysis.get("captured_tensor_count", 0)),
        ("tensor_jsonl_row_count", analysis.get("tensor_jsonl_row_count", 0)),
        ("payload_file_count", analysis.get("payload_file_count", 0)),
        ("payload_bytes_total", analysis.get("payload_bytes_total", 0)),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_boundary_capture_run",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"boundary capture run output: {out_dir}")
  print(f"route_status={route_status}")
  print(f"captured_tensor_count={analysis.get('captured_tensor_count')}")
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
