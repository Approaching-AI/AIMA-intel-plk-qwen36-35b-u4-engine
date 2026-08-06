#!/usr/bin/env python3
"""Capture distribution rows from materialized oracle prompts."""

from __future__ import annotations

import argparse
import hashlib
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
SCHEMA_VERSION = "intel-qwen36-r0-distribution-capture-materialized-v0"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
LLAMA_SERVER = "/home/intel/llama-cpp/llama-b9518/llama-server"
LLAMA_TOKENIZE = "/home/intel/llama-cpp/llama-b9518/llama-tokenize"
DEFAULT_CASES = ("sentinel_001k", "prefill_shape_001k")


REMOTE_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
LLAMA_SERVER = "/home/intel/llama-cpp/llama-b9518/llama-server"
LLAMA_TOKENIZE = "/home/intel/llama-cpp/llama-b9518/llama-tokenize"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def http_json(url, payload=None, timeout_s=30):
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            text = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(text) if text else None, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, text
    except Exception as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def parse_token_ids(stdout):
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if text.startswith("[") and text.endswith("]"):
            value = json.loads(text)
            if isinstance(value, list) and all(isinstance(item, int) for item in value):
                return value
    raise RuntimeError("could not parse llama-tokenize token ids")


def token_sha256(token_ids):
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def top_logprob_id_signature(top_logprobs):
    if not isinstance(top_logprobs, list):
        return []
    return [
        item["id"]
        for item in top_logprobs
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    ]


def normalize_top_logprobs(top_logprobs):
    if not isinstance(top_logprobs, list):
        return []
    normalized = []
    for item in top_logprobs:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "bytes": item.get("bytes") if isinstance(item.get("bytes"), list) else [],
            "id": item.get("id"),
            "logprob": item.get("logprob"),
            "token": item.get("token") if isinstance(item.get("token"), str) else "",
        })
    return normalized


def normalize_position(item, position, prompt_token_count):
    top_logprobs = normalize_top_logprobs(item.get("top_logprobs"))
    return {
        "context_token_count": prompt_token_count + position,
        "position": position,
        "reference_token_id": item.get("id"),
        "reference_token_logprob": item.get("logprob"),
        "reference_token_text": item.get("token") if isinstance(item.get("token"), str) else "",
        "top1_id": top_logprobs[0].get("id") if top_logprobs else None,
        "top_logprob_id_signature": top_logprob_id_signature(top_logprobs),
        "top_logprobs": top_logprobs,
    }


def main():
    cwd = Path.cwd()
    config = json.loads((cwd / "input.json").read_text(encoding="utf-8"))
    prompt_path = Path(config["remote_prompt_path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    port = int(config["port"])
    max_new_tokens = int(config["max_new_tokens"])
    n_probs = int(config["n_probs"])
    ctx_size = int(config["ctx_size"])
    request_timeout_s = int(config["request_timeout_s"])
    ready_timeout_s = int(config["ready_timeout_s"])
    env = dict(os.environ)
    env["INTEL_FORCE_PROBE"] = "b080"
    tokenized = subprocess.run(
        [
            LLAMA_TOKENIZE,
            "-m",
            MODEL_PATH,
            "-f",
            str(prompt_path),
            "--ids",
            "--log-disable",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
        env=env,
    )
    (cwd / "llama-tokenize.stdout").write_text(tokenized.stdout, encoding="utf-8")
    (cwd / "llama-tokenize.stderr").write_text(tokenized.stderr, encoding="utf-8")
    token_ids = parse_token_ids(tokenized.stdout)
    command = [
        LLAMA_SERVER,
        "-m",
        MODEL_PATH,
        "-c",
        str(ctx_size),
        "-n",
        str(max_new_tokens),
        "-ngl",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "-np",
        "1",
        "--log-file",
        str(cwd / "llama_server.log"),
        "--log-colors",
        "off",
    ]
    write_json(cwd / "llama_server_command.json", {"command": command})
    stdout = (cwd / "llama_server.stdout").open("w", encoding="utf-8")
    stderr = (cwd / "llama_server.stderr").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        stdout=stdout,
        stderr=stderr,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    server_record = {"attempts": [], "ready": False}
    try:
        started = time.monotonic()
        for attempt in range(1, ready_timeout_s + 1):
            if proc.poll() is not None:
                server_record.update({
                    "duration_s": time.monotonic() - started,
                    "reason": f"server exited with {proc.returncode}",
                })
                break
            status, value, text = http_json(f"http://127.0.0.1:{port}/health", timeout_s=5)
            server_record["attempts"].append({"attempt": attempt, "status": status, "text": text[:200]})
            if status == 200:
                server_record.update({"duration_s": time.monotonic() - started, "ready": True})
                write_json(cwd / "llama_server_health.json", value)
                break
            time.sleep(1)
        write_json(cwd / "llama_server_ready.json", server_record)
        if not server_record["ready"]:
            write_json(cwd / "result.json", {"ok": False, "server": server_record})
            return 2
        request = {
            "cache_prompt": False,
            "n_predict": max_new_tokens,
            "n_probs": n_probs,
            "prompt": prompt,
            "seed": 0,
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
        }
        write_json(cwd / "completion_request.json", {
            **request,
            "prompt": {
                "bytes": len(prompt.encode("utf-8")),
                "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            },
        })
        req_started = time.monotonic()
        status, value, text = http_json(
            f"http://127.0.0.1:{port}/completion",
            request,
            timeout_s=request_timeout_s,
        )
        duration_s = time.monotonic() - req_started
        (cwd / "completion_response.raw").write_text(text, encoding="utf-8")
        if not isinstance(value, dict):
            value = {}
        write_json(cwd / "completion_response.json", value)
        probabilities = value.get("completion_probabilities")
        if not isinstance(probabilities, list):
            probabilities = []
        positions = [
            normalize_position(item, position, len(token_ids))
            for position, item in enumerate(probabilities)
            if isinstance(item, dict)
        ]
        generated_ids = [
            item.get("id")
            for item in probabilities
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        ]
        row = {
            "bundle_jsonl_path": "teacher-forced-distribution-references.jsonl",
            "capture_mode": "current_target_llama_server_materialized_prompt_distribution",
            "capture_status": "captured_materialized_prompt_distribution_subset",
            "case_id": config["case_id"],
            "distribution_positions": positions,
            "generated_token_count": len(generated_ids),
            "generated_token_ids": generated_ids,
            "kind": config.get("kind"),
            "limitations": {
                "full_acceptance_context_ladder": False,
                "materialized_prompt_subset_only": True,
                "not_a_full_r0_oracle_bundle": True,
                "not_a_per_boundary_tensor_bundle": True,
            },
            "materialized_prompt_path": config.get("materialized_prompt_path"),
            "prompt_set": config.get("prompt_set"),
            "prompt_token_count": len(token_ids),
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": token_sha256(token_ids),
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "queue_task_id": config.get("queue_task_id"),
            "remote_prompt_path": str(prompt_path),
            "request_duration_s": duration_s,
            "request_status": status,
            "requested_output_token_count": max_new_tokens,
            "required_output_token_counts": config.get("required_output_token_counts"),
            "schema_version": "intel-qwen36-r0-distribution-capture-materialized-v0",
            "source_reference_runtime": "llama.cpp CPU server",
            "stopped_before_request_limit": len(generated_ids) < max_new_tokens,
            "suite": config.get("suite"),
            "target_prompt_tokens": config.get("target_prompt_tokens"),
            "workstream": WORKSTREAM,
        }
        with (cwd / "teacher-forced-distribution-materialized.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        checks = [
            {"name": "request_status_ok", "pass": status == 200, "status": status},
            {"name": "completion_probabilities_present", "pass": bool(probabilities), "count": len(probabilities)},
            {"name": "generated_token_count_positive", "pass": len(generated_ids) > 0, "count": len(generated_ids)},
            {"name": "generated_token_count_within_request", "pass": len(generated_ids) <= max_new_tokens, "count": len(generated_ids), "max_new_tokens": max_new_tokens},
            {"name": "top_logprobs_present", "pass": bool(positions) and all(pos["top_logprobs"] for pos in positions)},
            {"name": "top1_matches_reference", "pass": bool(positions) and all(pos["top1_id"] == pos["reference_token_id"] for pos in positions)},
            {"name": "prompt_token_count_matches_materialization", "pass": len(token_ids) == config.get("observed_prompt_tokens"), "count": len(token_ids), "expected": config.get("observed_prompt_tokens")},
        ]
        result = {
            "checks": checks,
            "ok": all(check["pass"] for check in checks),
            "row": row,
            "server": server_record,
        }
        write_json(cwd / "result.json", result)
        return 0 if result["ok"] else 3
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        stdout.close()
        stderr.close()
        write_json(cwd / "llama_server_returncode.json", {"returncode": proc.returncode})


if __name__ == "__main__":
    raise SystemExit(main())
'''


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--case-id", action="append", default=[])
  parser.add_argument("--output-token-count", type=int, default=512)
  parser.add_argument("--n-probs", type=int, default=5)
  parser.add_argument("--port-base", type=int, default=18680)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--poll-interval-s", type=int, default=5)
  parser.add_argument("--request-timeout-s", type=int, default=2400)
  parser.add_argument("--ready-timeout-s", type=int, default=420)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-run")
  parser.add_argument(
      "--materialization-dir",
      type=Path,
      default=ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z",
  )
  parser.add_argument(
      "--queue-dir",
      type=Path,
      default=ROOT / "output/r0-oracle-capture-queue-20260626T074119Z",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-distribution-capture-materialized-<UTC>.",
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
        row = json.loads(text)
      except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: expected object")
      rows.append(row)
  return rows


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def remote_put_text(host: str, remote_path: str, text: str) -> dict[str, Any]:
  return run_target(
      host,
      f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && "
      f"cat > {shlex.quote(remote_path)}",
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


def ctx_size_for_request(prompt_tokens: int, output_tokens: int) -> int:
  return min(262144, max(1024, prompt_tokens + output_tokens + 128))


def selected_rows(args: argparse.Namespace) -> list[tuple[dict[str, Any], dict[str, Any]]]:
  materialized = load_jsonl(args.materialization_dir / "materialized-prompts.jsonl")
  by_case = {row["case_id"]: row for row in materialized}
  tasks = load_jsonl(args.queue_dir / "teacher-forced-distribution-tasks.jsonl")
  tasks_by_case = {row["case_id"]: row for row in tasks}
  case_ids = args.case_id or list(DEFAULT_CASES)
  missing = [case_id for case_id in case_ids if case_id not in by_case or case_id not in tasks_by_case]
  if missing:
    raise SystemExit(f"case ids missing from materialization or queue: {missing}")
  selected = []
  for case_id in case_ids:
    task = tasks_by_case[case_id]
    token_counts = task.get("required_output_token_counts")
    if not isinstance(token_counts, list) or args.output_token_count not in token_counts:
      raise SystemExit(f"{case_id}: output token count {args.output_token_count} not in required counts {token_counts}")
    selected.append((by_case[case_id], task))
  return selected


def launch_case(
    *,
    args: argparse.Namespace,
    case_dir: Path,
    created_at: str,
    index: int,
    row: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
  remote_dir = (
      f"{args.remote_root.rstrip('/')}/distribution-materialized-"
      f"{created_at.replace('-', '').replace(':', '')}-{index:02d}-{row['case_id']}"
  )
  raw_remote_dir = case_dir / "raw" / "remote"
  raw_remote_dir.mkdir(parents=True, exist_ok=True)
  observed_tokens = int(row["observed_prompt_tokens"])
  remote_input = {
      "case_id": row["case_id"],
      "ctx_size": ctx_size_for_request(observed_tokens, args.output_token_count),
      "kind": row.get("kind"),
      "materialized_prompt_path": row.get("materialized_prompt_path"),
      "max_new_tokens": args.output_token_count,
      "n_probs": args.n_probs,
      "observed_prompt_tokens": observed_tokens,
      "port": args.port_base + index,
      "prompt_set": row.get("prompt_set"),
      "queue_task_id": task.get("task_id"),
      "ready_timeout_s": args.ready_timeout_s,
      "remote_prompt_path": row["remote_prompt_path"],
      "request_timeout_s": args.request_timeout_s,
      "required_output_token_counts": task.get("required_output_token_counts"),
      "suite": row.get("suite"),
      "target_prompt_tokens": row.get("target_prompt_tokens"),
  }
  input_put = remote_put_text(
      args.host,
      f"{remote_dir}/input.json",
      json.dumps(remote_input, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
  )
  runner_put = remote_put_text(args.host, f"{remote_dir}/run.py", REMOTE_RUNNER)
  chmod = run_target(args.host, f"chmod +x {shlex.quote(remote_dir + '/run.py')}", timeout_s=30)
  launch_script = "\n".join([
      "#!/usr/bin/env bash",
      "set -u",
      "date -u +%Y-%m-%dT%H:%M:%SZ > started_at",
      "python3 run.py > runner.stdout 2> runner.stderr",
      "status=$?",
      "printf '%s\\n' \"$status\" > exitcode",
      "date -u +%Y-%m-%dT%H:%M:%SZ > finished_at",
      "exit \"$status\"",
      "",
  ])
  launch_put = remote_put_text(args.host, f"{remote_dir}/launch.sh", launch_script)
  chmod_launch = run_target(args.host, f"chmod +x {shlex.quote(remote_dir + '/launch.sh')}", timeout_s=30)
  launch = run_target(
      args.host,
      (
          f"cd {shlex.quote(remote_dir)} || exit 1; "
          "nohup ./launch.sh >/dev/null 2>&1 < /dev/null & "
          "pid=$!; echo \"$pid\" > launcher.pid; echo \"$pid\""
      ),
      timeout_s=30,
  )
  polls = []
  start = time.monotonic()
  timed_out = False
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
      break
    if time.monotonic() - start > args.timeout_s:
      timed_out = True
      run_target(
          args.host,
          (
              f"pkill -TERM -f {shlex.quote(LLAMA_SERVER + ' -m ' + MODEL_PATH)} || true; "
              "sleep 2; "
              f"pkill -KILL -f {shlex.quote(LLAMA_SERVER + ' -m ' + MODEL_PATH)} || true"
          ),
          timeout_s=30,
      )
      break
    time.sleep(args.poll_interval_s)
  fetched = {}
  for remote_name in (
      "input.json",
      "run.py",
      "launch.sh",
      "started_at",
      "finished_at",
      "exitcode",
      "launcher.pid",
      "runner.stdout",
      "runner.stderr",
      "llama-tokenize.stdout",
      "llama-tokenize.stderr",
      "llama_server_command.json",
      "llama_server_health.json",
      "llama_server_ready.json",
      "llama_server_returncode.json",
      "llama_server.stdout",
      "llama_server.stderr",
      "llama_server.log",
      "completion_request.json",
      "completion_response.raw",
      "completion_response.json",
      "teacher-forced-distribution-materialized.jsonl",
      "result.json",
  ):
    fetched[remote_name] = fetch_remote_file(args.host, remote_dir, remote_name, raw_remote_dir)
  return {
      "chmod": chmod,
      "chmod_launch": chmod_launch,
      "fetched_files": sorted(fetched),
      "input_put": input_put,
      "launch": launch,
      "launch_put": launch_put,
      "polls": polls,
      "remote_dir": remote_dir,
      "runner_put": runner_put,
      "timed_out": timed_out,
  }


def load_case_json(case_dir: Path, name: str) -> dict[str, Any]:
  path = case_dir / "raw" / "remote" / name
  if not path.is_file() or not path.read_text(encoding="utf-8").strip():
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# R0 Materialized-Prompt Distribution Capture",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- captured cases: {payload['captured_case_count']}",
      f"- requested output tokens: {payload['requested_output_token_count']}",
      f"- total distribution positions: {payload['total_distribution_positions']}",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This captures a materialized-prompt distribution subset. It is not the",
      "full acceptance distribution bundle and not a per-boundary oracle bundle.",
      "",
      "| Case | Prompt tokens | Captured positions | Request status |",
      "|---|---:|---:|---:|",
  ]
  for case in payload["case_results"]:
    lines.append(
        f"| `{case['case_id']}` | {case['prompt_token_count']} | "
        f"{case['captured_positions']} | {case['request_status']} |"
    )
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  if args.output_token_count <= 0:
    raise SystemExit("--output-token-count must be positive")
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-distribution-capture-materialized-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  selections = selected_rows(args)
  rows = []
  case_results = []
  command_results = []
  for index, (source_row, task) in enumerate(selections):
    case_id = source_row["case_id"]
    case_dir = out_dir / f"case-{index + 1:02d}-{case_id}"
    remote = launch_case(
        args=args,
        case_dir=case_dir,
        created_at=created_at,
        index=index,
        row=source_row,
        task=task,
    )
    result = load_case_json(case_dir, "result.json")
    row = result.get("row", {}) if isinstance(result.get("row"), dict) else {}
    positions = row.get("distribution_positions", [])
    case_results.append({
        "captured_positions": len(positions) if isinstance(positions, list) else 0,
        "case_id": case_id,
        "expected_prompt_token_count": source_row.get("observed_prompt_tokens"),
        "prompt_token_count": row.get("prompt_token_count"),
        "request_status": row.get("request_status"),
        "result_ok": result.get("ok") is True,
        "stopped_before_request_limit": row.get("stopped_before_request_limit"),
        "top_logprobs_present": bool(positions)
        and all(position.get("top_logprobs") for position in positions),
    })
    command_results.append({
        "case_id": case_id,
        "remote_dir": remote["remote_dir"],
        "timed_out": remote["timed_out"],
    })
    if row:
      rows.append(row)
  out_jsonl = out_dir / "teacher-forced-distribution-materialized.jsonl"
  with out_jsonl.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
  total_positions = sum(
      len(row.get("distribution_positions", []))
      for row in rows
      if isinstance(row.get("distribution_positions"), list)
  )
  checks = [
      {
          "name": "all_selected_cases_completed",
          "pass": len(rows) == len(case_results) == len(selections)
          and all(case["result_ok"] for case in case_results),
          "captured": len(rows),
          "selected": len(selections),
      },
      {
          "name": "all_request_status_ok",
          "pass": all(case["request_status"] == 200 for case in case_results),
      },
      {
          "name": "all_prompt_counts_match_materialization",
          "pass": all(
              case["prompt_token_count"] == case["expected_prompt_token_count"]
              for case in case_results
          ),
      },
      {
          "name": "all_captured_positions_within_request",
          "pass": all(
              0 < case["captured_positions"] <= args.output_token_count
              for case in case_results
          ),
      },
      {
          "name": "top_logprobs_present",
          "pass": all(case["top_logprobs_present"] for case in case_results),
      },
      {
          "name": "rows_are_current_workstream",
          "pass": all(row.get("workstream") == WORKSTREAM for row in rows),
      },
      {
          "name": "subset_does_not_claim_full_bundle",
          "pass": all(
              row.get("limitations", {}).get("materialized_prompt_subset_only") is True
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
      "materialization_dir": str(args.materialization_dir.resolve().relative_to(ROOT)),
      "queue_dir": str(args.queue_dir.resolve().relative_to(ROOT)),
      "r0_oracle_gate_closed": False,
      "requested_output_token_count": args.output_token_count,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "teacher_forced_distribution_jsonl": str(out_jsonl.relative_to(ROOT)),
      "total_distribution_positions": total_positions,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-distribution-capture-materialized.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "capture.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_distribution_capture_materialized",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in (
        ("captured_case_count", len(rows)),
        ("requested_output_token_count", args.output_token_count),
        ("total_distribution_positions", total_positions),
        ("required_checks_passed", required_checks_passed),
        ("r0_oracle_gate_closed", False),
    ):
      handle.write(json.dumps({
          "metric": metric,
          "phase": "r0_distribution_capture_materialized",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"materialized distribution capture output: {out_dir}")
  if not required_checks_passed:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
