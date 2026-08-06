#!/usr/bin/env python3
"""Run one llama.cpp denominator candidate row on the target."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess

import iq36_local
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
LLAMA_BENCH = "/home/intel/llama-cpp/llama-b9518/llama-bench"
INTEL_ENV = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--output-tokens", type=int, default=1)
  parser.add_argument(
      "--mode",
      choices=("separate", "paired"),
      default="separate",
      help="Use separate -p/-n rows or paired -pg prompt,gen row.",
  )
  parser.add_argument("--repetitions", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--ubatch-size", type=int, default=512)
  parser.add_argument("--threads", type=int, default=16)
  parser.add_argument("--device", default="Vulkan0")
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--poll-interval-s", type=int, default=15)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-run")
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-llama-denominator-<UTC>.",
  )
  return parser.parse_args()


def run(
    cmd: list[str],
    *,
    timeout_s: int,
    input_text: str | None = None,
) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        input=input_text,
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


def run_target(host: str, remote_command: str, *, timeout_s: int, input_text: str | None = None) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s, input_text=input_text)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def remote_put_text(host: str, remote_path: str, text: str) -> dict[str, Any]:
  return run_target(
      host,
      f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && "
      f"cat > {shlex.quote(remote_path)} && chmod +x {shlex.quote(remote_path)}",
      timeout_s=30,
      input_text=text,
  )


def remote_read_file(host: str, remote_path: str, *, timeout_s: int = 30) -> dict[str, Any]:
  return run_target(
      host,
      f"test -f {shlex.quote(remote_path)} && cat {shlex.quote(remote_path)}",
      timeout_s=timeout_s,
  )


def fetch_remote_file(host: str, remote_dir: str, remote_name: str, local_dir: Path) -> str:
  result = remote_read_file(host, f"{remote_dir.rstrip('/')}/{remote_name}")
  local_path = local_dir / remote_name
  if result["returncode"] == 0:
    local_path.write_text(result["stdout"], encoding="utf-8")
    return result["stdout"]
  local_path.write_text("", encoding="utf-8")
  (local_dir / f"{remote_name}.fetch.stderr").write_text(result["stderr"], encoding="utf-8")
  return ""


def benchmark_command(args: argparse.Namespace) -> str:
  if args.mode == "paired":
    prompt_gen = f"-pg {args.bucket},{args.output_tokens} "
  else:
    prompt_gen = f"-p {args.bucket} -n {args.output_tokens} "
  return (
      f"{shlex.quote(LLAMA_BENCH)} "
      f"-m {shlex.quote(MODEL_PATH)} "
      f"{prompt_gen}"
      f"-r {args.repetitions} "
      f"-b {args.batch_size} "
      f"-ub {args.ubatch_size} "
      f"-t {args.threads} "
      f"-dev {shlex.quote(args.device)} "
      "-ngl -1 "
      "-ctk f16 "
      "-ctv f16 "
      "--no-warmup "
      "-o json"
  )


def launch_remote(args: argparse.Namespace, *, created_at: str, raw_dir: Path) -> dict[str, Any]:
  stamp = created_at.replace("-", "").replace(":", "")
  remote_dir = f"{args.remote_root.rstrip('/')}/llama-denominator-{stamp}"
  remote_files_dir = raw_dir / "remote"
  remote_files_dir.mkdir(parents=True, exist_ok=True)
  command = benchmark_command(args)
  run_script = "\n".join([
      "#!/usr/bin/env bash",
      "set -u",
      f"source {shlex.quote(INTEL_ENV)} >/tmp/iq36-llama-env.log 2>&1 || true",
      "export INTEL_FORCE_PROBE=b080",
      "printf 'remote_run_started_at='",
      "date -u +%Y-%m-%dT%H:%M:%SZ",
      f"printf 'benchmark_command=%s\\n' {shlex.quote(command)}",
      command,
      "status=$?",
      "printf 'remote_run_finished_at='",
      "date -u +%Y-%m-%dT%H:%M:%SZ",
      "exit \"$status\"",
      "",
  ])
  launch_script = "\n".join([
      "#!/usr/bin/env bash",
      "set -u",
      "date -u +%Y-%m-%dT%H:%M:%SZ > started_at",
      "./run.sh > stdout.log 2> stderr.log",
      "status=$?",
      "printf '%s\\n' \"$status\" > exitcode",
      "date -u +%Y-%m-%dT%H:%M:%SZ > finished_at",
      "exit \"$status\"",
      "",
  ])
  run_put = remote_put_text(args.host, f"{remote_dir}/run.sh", run_script)
  launch_put = remote_put_text(args.host, f"{remote_dir}/launch.sh", launch_script)
  launch = run_target(
      args.host,
      (
          f"cd {shlex.quote(remote_dir)} || exit 1; "
          "nohup ./launch.sh >/dev/null 2>&1 < /dev/null & "
          "pid=$!; echo \"$pid\" > launcher.pid; echo \"$pid\""
      ),
      timeout_s=30,
  )
  polls: list[dict[str, Any]] = []
  start = time.monotonic()
  exit_code_text = ""
  while True:
    poll = run_target(
        args.host,
        (
            f"cd {shlex.quote(remote_dir)} || exit 1; "
            "if test -f exitcode; then printf 'done '; cat exitcode; else echo running; fi"
        ),
        timeout_s=30,
    )
    polls.append({
        "at": iso_now(),
        "returncode": poll["returncode"],
        "stderr": poll["stderr"],
        "stdout": poll["stdout"].strip(),
    })
    if poll["stdout"].startswith("done"):
      exit_code_text = poll["stdout"].replace("done", "", 1).strip()
      break
    if time.monotonic() - start > args.timeout_s:
      exit_code_text = "124"
      escaped_command = command.replace("'", "'\"'\"'")
      run_target(
          args.host,
          (
              "pkill -TERM -f "
              f"{shlex.quote(escaped_command)} || true; "
              "sleep 2; "
              "pkill -KILL -f "
              f"{shlex.quote(escaped_command)} || true"
          ),
          timeout_s=30,
      )
      for _ in range(5):
        final_poll = run_target(
            args.host,
            (
                f"cd {shlex.quote(remote_dir)} || exit 1; "
                "if test -f exitcode; then printf 'done '; cat exitcode; else echo running; fi"
            ),
            timeout_s=30,
        )
        polls.append({
            "at": iso_now(),
            "returncode": final_poll["returncode"],
            "stderr": final_poll["stderr"],
            "stdout": final_poll["stdout"].strip(),
        })
        if final_poll["stdout"].startswith("done"):
          exit_code_text = "124"
          break
        time.sleep(2)
      break
    time.sleep(args.poll_interval_s)

  fetched: dict[str, str] = {}
  for remote_name in (
      "stdout.log",
      "stderr.log",
      "run.sh",
      "launch.sh",
      "started_at",
      "finished_at",
      "exitcode",
      "launcher.pid",
  ):
    fetched[remote_name] = fetch_remote_file(args.host, remote_dir, remote_name, remote_files_dir)
  return {
      "benchmark_command": command,
      "exit_code_text": exit_code_text,
      "launch": launch,
      "launch_script": launch_put,
      "polls": polls,
      "remote_dir": remote_dir,
      "remote_files": str(remote_files_dir),
      "run_script": run_put,
      "stderr": fetched.get("stderr.log", ""),
      "stdout": fetched.get("stdout.log", ""),
  }


def extract_json_value(text: str) -> Any | None:
  decoder = json.JSONDecoder()
  for index, char in enumerate(text):
    if char not in "[{":
      continue
    try:
      value, _ = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
      continue
    if isinstance(value, (list, dict)):
      return value
  return None


def normalize_rows(value: Any) -> list[dict[str, Any]]:
  if isinstance(value, list):
    return [row for row in value if isinstance(row, dict)]
  if isinstance(value, dict):
    return [value]
  return []


def parse_llama_rows(
    stdout: str,
    *,
    bucket: int,
    output_tokens: int,
) -> dict[str, Any]:
  value = extract_json_value(stdout)
  rows = normalize_rows(value)
  parse_status = "parsed" if rows else "missing_json"
  prefill_tokens_s = None
  decode_tokens_s = None
  paired_tokens_s = None
  row_summaries = []
  for row in rows:
    test_name = str(row.get("test", ""))
    n_prompt = row.get("n_prompt")
    n_gen = row.get("n_gen")
    avg_ts = row.get("avg_ts")
    summary = {
        "avg_ts": avg_ts,
        "n_gen": n_gen,
        "n_prompt": n_prompt,
        "test": test_name,
    }
    row_summaries.append(summary)
    if isinstance(avg_ts, (int, float)):
      if n_prompt == bucket and (test_name.startswith("pp") or n_gen == 0):
        prefill_tokens_s = float(avg_ts)
      if n_gen == output_tokens and (test_name.startswith("tg") or n_prompt == 0):
        decode_tokens_s = float(avg_ts)
      if n_prompt == bucket and n_gen == output_tokens:
        paired_tokens_s = float(avg_ts)
  return {
      "decode_tokens_s": decode_tokens_s,
      "json_row_count": len(rows),
      "paired_tokens_s": paired_tokens_s,
      "parse_status": parse_status,
      "prefill_tokens_s": prefill_tokens_s,
      "rows": row_summaries,
  }


def build_summary(payload: dict[str, Any]) -> str:
  row = payload["row"]
  lines = [
      "# R0 llama.cpp Denominator Candidate Row",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- bucket: {row['bucket']}",
      f"- output tokens: {row['output_tokens_requested']}",
      f"- route label: `{payload['route_label']}`",
      f"- return code: {row['raw']['returncode']}",
      f"- parse status: `{row['parse_status']}`",
      f"- parsed JSON rows: {row['json_row_count']}",
      f"- prefill tok/s: {row.get('prefill_tokens_s')}",
      f"- decode tok/s: {row.get('decode_tokens_s')}",
      f"- paired prompt+gen tok/s: {row.get('paired_tokens_s')}",
      f"- R0 denominator gate closed: `{str(payload['r0_denominator_gate_closed']).lower()}`",
      "",
      "This records one llama.cpp denominator-candidate row. A smoke row",
      "proves the route can load and emit metrics; only a valid 262144 row or",
      "an explicit unavailable-lane policy can resolve the R0 denominator item.",
      "",
  ]
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or (ROOT / f"output/r0-llama-denominator-{stamp}")
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  remote = launch_remote(args, created_at=created_at, raw_dir=raw_dir)
  stdout = remote["stdout"]
  stderr = remote["stderr"]
  (raw_dir / "llama-denominator.stdout").write_text(stdout, encoding="utf-8")
  (raw_dir / "llama-denominator.stderr").write_text(stderr, encoding="utf-8")
  try:
    returncode = int(remote["exit_code_text"])
  except ValueError:
    returncode = 124
  parsed = parse_llama_rows(stdout, bucket=args.bucket, output_tokens=args.output_tokens)
  route_label = (
      "candidate_262144_paired"
      if args.bucket == 262144 and args.mode == "paired"
      else "candidate_262144"
      if args.bucket == 262144
      else f"smoke_{args.mode}"
  )
  row = {
      "bucket": args.bucket,
      "cache_state": "cold_no_prefix_process_per_prompt",
      "decode_tokens_s": parsed["decode_tokens_s"],
      "device": args.device,
      "input_tokens": args.bucket,
      "json_row_count": parsed["json_row_count"],
      "llama_bench_mode": args.mode,
      "model_path": MODEL_PATH,
      "model_sha256": MODEL_SHA256,
      "output_tokens_requested": args.output_tokens,
      "paired_tokens_s": parsed["paired_tokens_s"],
      "parse_status": parsed["parse_status"],
      "phase": "llama_cpp_denominator_candidate",
      "prefill_tokens_s": parsed["prefill_tokens_s"],
      "raw": {
          "command": remote["benchmark_command"],
          "launch_mode": "local-background",
          "remote": {
              "launch": remote["launch"],
              "launch_script": remote["launch_script"],
              "polls": remote["polls"],
              "remote_dir": remote["remote_dir"],
              "remote_files": remote["remote_files"],
              "run_script": remote["run_script"],
          },
          "returncode": returncode,
          "stderr": str((raw_dir / "llama-denominator.stderr").relative_to(ROOT)),
          "stdout": str((raw_dir / "llama-denominator.stdout").relative_to(ROOT)),
      },
      "rows": parsed["rows"],
  }
  payload = {
      "created_at": created_at,
      "host": args.host,
      "r0_denominator_gate_closed": False,
      "route_label": route_label,
      "row": row,
      "schema_version": "intel-qwen36-r0-llama-denominator-row-v0",
      "workstream": WORKSTREAM,
  }
  checks = [
      {"name": "local_process_completed", "pass": returncode == 0, "returncode": returncode},
      {"name": "json_metrics_parsed", "pass": parsed["parse_status"] == "parsed"},
      {"name": "json_row_count_positive", "pass": parsed["json_row_count"] > 0},
      {"name": "input_bucket_recorded", "pass": row["input_tokens"] == args.bucket},
      {
          "name": "prefill_tokens_s_parsed",
          "pass": row["prefill_tokens_s"] is not None or args.mode == "paired",
          "value": row["prefill_tokens_s"],
      },
      {
          "name": "decode_tokens_s_parsed",
          "pass": row["decode_tokens_s"] is not None or args.mode == "paired",
          "value": row["decode_tokens_s"],
      },
      {
          "name": "paired_tokens_s_parsed",
          "pass": row["paired_tokens_s"] is not None or args.mode == "separate",
          "value": row["paired_tokens_s"],
      },
      {"name": "does_not_close_r0_denominator", "pass": True},
  ]
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "route_label": route_label,
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-llama-denominator-run.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "row.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_llama_denominator_candidate_row",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": payload["schema_version"],
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric in ("prefill_tokens_s", "decode_tokens_s", "paired_tokens_s", "json_row_count"):
      fh.write(json.dumps({
          "bucket": args.bucket,
          "metric": metric,
          "phase": "llama_cpp_denominator_candidate",
          "value": row.get(metric),
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"llama denominator row output: {out_dir}")


if __name__ == "__main__":
  main()
