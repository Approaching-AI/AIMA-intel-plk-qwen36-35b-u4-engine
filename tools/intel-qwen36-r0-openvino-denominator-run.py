#!/usr/bin/env python3
"""Run one current-target OpenVINO denominator bucket and parse the result."""

from __future__ import annotations

import argparse
import json
import math
import re
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
OPENVINO_DIR = "/home/intel/ov"
OPENVINO_MODEL = "/home/intel/Qwen3.6-35B-A3B-ov"
PROMPT_DIR = "/home/intel/ov/prompts"
DEFAULT_PROMPT_FILE = "prompt_10_256Kin_512out_r1.txt"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
  parser.add_argument("--bucket", type=int, default=262144)
  parser.add_argument("--output-tokens", type=int, default=512)
  parser.add_argument("--num-iter", type=int, default=1)
  parser.add_argument("--num-warmup", type=int, default=0)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument(
      "--launch-mode",
      choices=("direct", "background"),
      default="direct",
      help="Run in the foreground or launch a local background job and poll logs.",
  )
  parser.add_argument(
      "--poll-interval-s",
      type=int,
      default=15,
      help="Polling interval for --launch-mode background.",
  )
  parser.add_argument(
      "--staging-root",
      "--remote-root",
      dest="remote_root",
      metavar="PATH",
      default="/home/intel/intel-qwen36-run",
      help="Local staging parent for background launch artifacts.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-openvino-denominator-<UTC>.",
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


def first_float(text: str) -> float | None:
  match = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
  return float(match.group(0)) if match else None


def parse_openvino_stdout(stdout: str, *, bucket: int, output_tokens: int, prompt_file: str) -> dict[str, Any]:
  row: dict[str, Any] = {
      "bucket": bucket,
      "cache_state": "cold_no_prefix_process_per_prompt",
      "output_tokens_requested": output_tokens,
      "parse_status": "parsed",
      "parse_warnings": [],
      "phase": "openvino_denominator",
      "prompt_file": prompt_file,
      "token_correctness": "not_checked_openvino_denominator",
  }
  for line in stdout.splitlines():
    stripped = line.strip()
    if stripped.startswith("openvino runtime version:"):
      row["runtime_version"] = stripped
    elif stripped.startswith("Number of images:"):
      match = re.search(r"Prompt token size:\s*([0-9]+)", stripped)
      if match:
        row["prompt_tokens"] = int(match.group(1))
        row["input_tokens"] = int(match.group(1))
    elif stripped.startswith("Output token size:"):
      value = first_float(stripped)
      if value is not None:
        row["output_tokens"] = int(value)
    elif stripped.startswith("Load time:"):
      row["load_ms"] = first_float(stripped)
    elif stripped.startswith("Generate time:"):
      row["generate_ms"] = first_float(stripped)
    elif stripped.startswith("Tokenization time:"):
      row["tokenization_ms"] = first_float(stripped)
    elif stripped.startswith("Detokenization time:"):
      row["detokenization_ms"] = first_float(stripped)
    elif stripped.startswith("Embeddings preparation time:"):
      row["embeddings_ms"] = first_float(stripped)
    elif stripped.startswith("TTFT:"):
      row["ttft_ms"] = first_float(stripped)
    elif stripped.startswith("TPOT:"):
      row["tpot_ms"] = first_float(stripped)
    elif stripped.startswith("Throughput"):
      row["decode_tokens_s"] = first_float(stripped)

  prompt_tokens = row.get("prompt_tokens")
  ttft_ms = row.get("ttft_ms")
  if isinstance(prompt_tokens, int) and isinstance(ttft_ms, float) and ttft_ms > 0:
    row["prefill_tokens_s"] = prompt_tokens / (ttft_ms / 1000.0)
  if row.get("output_tokens") != output_tokens:
    row["parse_warnings"].append("reported output token count differs from requested output tokens")
  required = [
      "prompt_tokens",
      "output_tokens",
      "ttft_ms",
      "tpot_ms",
      "decode_tokens_s",
      "prefill_tokens_s",
  ]
  missing = [key for key in required if key not in row]
  if missing:
    row["parse_status"] = "missing_fields"
    row["parse_warnings"].append("missing fields: " + ",".join(missing))
  return row


def benchmark_command(args: argparse.Namespace) -> str:
  prompt_path = f"{PROMPT_DIR.rstrip('/')}/{args.prompt_file}"
  return (
      f"cd {shlex.quote(OPENVINO_DIR)} && "
      ". openvino_env/bin/activate && "
      "python benchmark_vlm_new.py "
      f"-pf {shlex.quote(prompt_path)} "
      f"-m {shlex.quote(OPENVINO_MODEL)} "
      f"-d {shlex.quote(args.device)} "
      f"-nw {args.num_warmup} "
      f"-n {args.num_iter} "
      f"-mt {args.output_tokens}"
  )


def remote_put_text(args: argparse.Namespace, remote_path: str, text: str) -> dict[str, Any]:
  quoted_path = shlex.quote(remote_path)
  parent = shlex.quote(str(Path(remote_path).parent))
  return run_target(
      args.host,
      f"mkdir -p {parent} && cat > {quoted_path} && chmod +x {quoted_path}",
      timeout_s=30,
      input_text=text,
  )


def remote_read_file(args: argparse.Namespace, remote_path: str, *, timeout_s: int = 30) -> dict[str, Any]:
  quoted_path = shlex.quote(remote_path)
  return run_target(
      args.host,
      f"test -f {quoted_path} && cat {quoted_path}",
      timeout_s=timeout_s,
  )


def fetch_remote_file(
    args: argparse.Namespace,
    *,
    remote_dir: str,
    remote_name: str,
    local_dir: Path,
) -> str:
  result = remote_read_file(args, f"{remote_dir.rstrip('/')}/{remote_name}")
  local_path = local_dir / remote_name
  if result["returncode"] == 0:
    local_path.write_text(result["stdout"], encoding="utf-8")
    return result["stdout"]
  local_path.write_text("", encoding="utf-8")
  (local_dir / f"{remote_name}.fetch.stderr").write_text(result["stderr"], encoding="utf-8")
  return ""


def launch_remote_background(
    args: argparse.Namespace,
    *,
    created_at: str,
    raw_dir: Path,
    remote_command: str,
) -> dict[str, Any]:
  stamp = created_at.replace("-", "").replace(":", "")
  remote_dir = f"{args.remote_root.rstrip('/')}/openvino-denominator-{stamp}"
  remote_files_dir = raw_dir / "remote"
  remote_files_dir.mkdir(parents=True, exist_ok=True)

  run_script = "\n".join(
      [
          "#!/usr/bin/env bash",
          "set -u",
          "set -o pipefail",
          f"echo remote_run_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) >&2",
          f"echo benchmark_command={shlex.quote(remote_command)} >&2",
          remote_command,
          "",
      ]
  )
  launch_script = "\n".join(
      [
          "#!/usr/bin/env bash",
          "set -u",
          "set -o pipefail",
          "echo $$ > launcher.pid",
          "date -u +%Y-%m-%dT%H:%M:%SZ > started_at",
          "./run.sh > stdout.log 2> stderr.log",
          "rc=$?",
          "printf '%s\\n' \"$rc\" > exitcode.tmp",
          "mv exitcode.tmp exitcode",
          "date -u +%Y-%m-%dT%H:%M:%SZ > finished_at",
          "exit \"$rc\"",
          "",
      ]
  )

  setup_run = remote_put_text(args, f"{remote_dir}/run.sh", run_script)
  setup_launch = remote_put_text(args, f"{remote_dir}/launch.sh", launch_script)
  setup = {
      "launch_script": setup_launch,
      "remote_dir": remote_dir,
      "run_script": setup_run,
  }
  if setup_run["returncode"] != 0 or setup_launch["returncode"] != 0:
    return {
        "command": remote_command,
        "returncode": 126,
        "stdout": "",
        "stderr": setup_run["stderr"] + setup_launch["stderr"],
        "timed_out": False,
        "remote": setup,
    }

  launch_cmd = (
      f"cd {shlex.quote(remote_dir)} || exit 1; "
      "nohup ./launch.sh >/dev/null 2>&1 < /dev/null & "
      "pid=$!; echo \"$pid\" > launcher.pid; echo \"$pid\""
  )
  launch = run_target(args.host, launch_cmd, timeout_s=30)
  setup["launch"] = launch
  if launch["returncode"] != 0:
    return {
        "command": remote_command,
        "returncode": 127,
        "stdout": "",
        "stderr": launch["stderr"],
        "timed_out": False,
        "remote": setup,
    }

  deadline = time.monotonic() + args.timeout_s
  polls: list[dict[str, Any]] = []
  exitcode: int | None = None
  timed_out = False
  poll_cmd = (
      f"if test -f {shlex.quote(remote_dir)}/exitcode; then "
      f"printf 'done '; cat {shlex.quote(remote_dir)}/exitcode; "
      f"elif test -f {shlex.quote(remote_dir)}/launcher.pid "
      f"&& kill -0 $(cat {shlex.quote(remote_dir)}/launcher.pid) 2>/dev/null; then "
      "printf 'running'; "
      "else printf 'missing'; fi"
  )
  while True:
    poll = run_target(args.host, poll_cmd, timeout_s=30)
    polls.append(
        {
            "at": iso_now(),
            "returncode": poll["returncode"],
            "stderr": poll["stderr"],
            "stdout": poll["stdout"].strip(),
        }
    )
    stdout = poll["stdout"].strip()
    if poll["returncode"] == 0 and stdout.startswith("done "):
      try:
        exitcode = int(stdout.split(None, 1)[1].strip())
      except (IndexError, ValueError):
        exitcode = 125
      break
    if time.monotonic() >= deadline:
      timed_out = True
      kill_cmd = (
          f"if test -f {shlex.quote(remote_dir)}/launcher.pid; then "
          f"kill $(cat {shlex.quote(remote_dir)}/launcher.pid) 2>/dev/null || true; fi"
      )
      setup["timeout_kill"] = run_target(args.host, kill_cmd, timeout_s=30)
      break
    time.sleep(max(1, args.poll_interval_s))

  stdout_log = fetch_remote_file(
      args,
      remote_dir=remote_dir,
      remote_name="stdout.log",
      local_dir=remote_files_dir,
  )
  stderr_log = fetch_remote_file(
      args,
      remote_dir=remote_dir,
      remote_name="stderr.log",
      local_dir=remote_files_dir,
  )
  for remote_name in (
      "run.sh",
      "launch.sh",
      "started_at",
      "finished_at",
      "exitcode",
      "launcher.pid",
  ):
    fetch_remote_file(
        args,
        remote_dir=remote_dir,
        remote_name=remote_name,
        local_dir=remote_files_dir,
    )

  return {
      "command": remote_command,
      "returncode": 124 if timed_out else (exitcode if exitcode is not None else 125),
      "stdout": stdout_log,
      "stderr": stderr_log,
      "timed_out": timed_out,
      "remote": {
          **setup,
          "polls": polls,
          "remote_dir": remote_dir,
          "remote_files": str(remote_files_dir),
      },
  }


def finite_metric(row: dict[str, Any], key: str) -> bool:
  value = row.get(key)
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_summary(payload: dict[str, Any]) -> str:
  row = payload["row"]
  return "\n".join(
      [
          "# OpenVINO Denominator Row",
          "",
          f"- workstream: `{WORKSTREAM}`",
          f"- route label: `{payload['route_label']}`",
          f"- host: `{payload['host']}`",
          f"- launch mode: `{payload['launch_mode']}`",
          f"- prompt file: `{row['prompt_file']}`",
          f"- prompt tokens: {row.get('prompt_tokens')}",
          f"- output tokens: {row.get('output_tokens')}",
          f"- TTFT ms: {row.get('ttft_ms')}",
          f"- prefill tok/s: {row.get('prefill_tokens_s')}",
          f"- decode tok/s: {row.get('decode_tokens_s')}",
          f"- TPOT ms: {row.get('tpot_ms')}",
          f"- parse status: `{row.get('parse_status')}`",
          "",
          "This records one current-target denominator row. It does not by itself",
          "close the full denominator or R0 performance gate.",
          "",
      ]
  )


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "")
    out_dir = ROOT / f"output/r0-openvino-denominator-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  remote_command = benchmark_command(args)
  if args.launch_mode == "background":
    result = launch_remote_background(
        args,
        created_at=created_at,
        raw_dir=raw_dir,
        remote_command=remote_command,
    )
  else:
    result = run_target(
        args.host,
        remote_command,
        timeout_s=args.timeout_s,
    )
  (raw_dir / "openvino-denominator.stdout").write_text(result["stdout"], encoding="utf-8")
  (raw_dir / "openvino-denominator.stderr").write_text(result["stderr"], encoding="utf-8")
  row = parse_openvino_stdout(
      result["stdout"],
      bucket=args.bucket,
      output_tokens=args.output_tokens,
      prompt_file=args.prompt_file,
  )
  if "CL_OUT_OF_RESOURCES" in result["stderr"]:
    row["failure_class"] = "openvino_gpu_cl_out_of_resources"
  elif "RuntimeError:" in result["stderr"]:
    row["failure_class"] = "openvino_runtime_error"
  row["raw"] = {
      "command": remote_command,
      "launch_mode": args.launch_mode,
      "remote": result.get("remote"),
      "returncode": result["returncode"],
      "stderr": str(raw_dir / "openvino-denominator.stderr"),
      "stdout": str(raw_dir / "openvino-denominator.stdout"),
      "timed_out": result["timed_out"],
  }
  parse_ok = result["returncode"] == 0 and row.get("parse_status") == "parsed"
  route_label = "diagnostic" if parse_ok else "rejected"
  payload = {
      "created_at": created_at,
      "host": args.host,
      "launch_mode": args.launch_mode,
      "model": {
          "openvino_path": OPENVINO_MODEL,
          "prompt_dir": PROMPT_DIR,
      },
      "r0_denominator_gate_closed": False,
      "remote_command": remote_command,
      "route_label": route_label,
      "row": row,
      "schema_version": "intel-qwen36-r0-openvino-denominator-row-v0",
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "launch_mode": args.launch_mode,
      "prompt_file": args.prompt_file,
      "route_label": route_label,
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-openvino-denominator-run.py",
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    metric = {
        "bucket": row.get("bucket"),
        "decode_tokens_s": row.get("decode_tokens_s"),
        "input_tokens": row.get("prompt_tokens"),
        "output_tokens": row.get("output_tokens"),
        "parse_status": row.get("parse_status"),
        "phase": "openvino_denominator",
        "prefill_tokens_s": row.get("prefill_tokens_s"),
        "raw": row.get("raw"),
        "route_label": route_label,
        "tpot_ms": row.get("tpot_ms"),
        "ttft_ms": row.get("ttft_ms"),
    }
    fh.write(json.dumps(metric, sort_keys=True) + "\n")
  write_json(out_dir / "correctness.json", {
      "checks": [
          {
              "finite_metrics": all(
                  finite_metric(row, key)
                  for key in ("ttft_ms", "prefill_tokens_s", "decode_tokens_s", "tpot_ms")
              ),
              "parse_status": row.get("parse_status"),
              "prompt_file": row.get("prompt_file"),
              "prompt_tokens_match_bucket": row.get("prompt_tokens") == args.bucket,
              "returncode": result["returncode"],
              "timed_out": result["timed_out"],
          }
      ],
      "gate": "current_target_openvino_denominator_row",
      "required_checks_passed": parse_ok and row.get("prompt_tokens") == args.bucket,
      "route_label": route_label,
      "token_correctness": "not_checked_denominator_only",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "notes": "single denominator row only",
      "route_label": route_label,
  })
  write_json(out_dir / "row.json", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"openvino denominator row output: {out_dir}")
  if not parse_ok:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
