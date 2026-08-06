#!/usr/bin/env python3
"""Capture one current-target llama.cpp distribution smoke row."""

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
SCHEMA_VERSION = "intel-qwen36-r0-distribution-capture-smoke-v0"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
LLAMA_SERVER = "/home/intel/llama-cpp/llama-b9518/llama-server"
LLAMA_TOKENIZE = "/home/intel/llama-cpp/llama-b9518/llama-tokenize"


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


def normalize_position(item, position, prompt_token_count):
    top_logprobs = item.get("top_logprobs") if isinstance(item.get("top_logprobs"), list) else []
    return {
        "context_token_count": prompt_token_count + position,
        "position": position,
        "reference_token_id": item.get("id"),
        "reference_token_logprob": item.get("logprob"),
        "reference_token_text": item.get("token") if isinstance(item.get("token"), str) else "",
        "top1_id": top_logprobs[0].get("id") if top_logprobs and isinstance(top_logprobs[0], dict) else None,
        "top_logprob_id_signature": top_logprob_id_signature(top_logprobs),
        "top_logprobs": top_logprobs,
    }


def main():
    cwd = Path.cwd()
    config = json.loads((cwd / "input.json").read_text(encoding="utf-8"))
    prompt = config["prompt"]
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
            "--stdin",
            "--ids",
            "--log-disable",
        ],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=120,
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
    server_record = {
        "attempts": [],
        "ready": False,
    }
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
        write_json(cwd / "completion_request.json", request)
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
            "capture_mode": "current_target_llama_server_completion_probabilities_smoke",
            "capture_status": "captured_partial_smoke",
            "case_id": config["case_id"],
            "distribution_positions": positions,
            "generated_token_count": len(generated_ids),
            "generated_token_ids": generated_ids,
            "limitations": {
                "full_acceptance_context_ladder": False,
                "not_a_full_r0_oracle_bundle": True,
                "not_a_per_boundary_tensor_bundle": True,
                "smoke_only": True,
            },
            "prompt_set": config.get("prompt_set"),
            "prompt_token_count": len(token_ids),
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": token_sha256(token_ids),
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request_duration_s": duration_s,
            "request_status": status,
            "schema_version": "intel-qwen36-r0-distribution-capture-smoke-v0",
            "source_reference_runtime": "llama.cpp CPU server",
            "suite": config.get("suite"),
            "suite_manifest_name": config.get("suite_manifest_name"),
            "stopped_before_request_limit": len(generated_ids) < max_new_tokens,
            "target_max_new_tokens": max_new_tokens,
            "workstream": WORKSTREAM,
        }
        with (cwd / "distribution-smoke.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        checks = [
            {"name": "request_status_ok", "pass": status == 200, "status": status},
            {"name": "completion_probabilities_present", "pass": bool(probabilities), "count": len(probabilities)},
            {"name": "generated_token_count_positive", "pass": len(generated_ids) > 0, "count": len(generated_ids)},
            {"name": "generated_token_count_within_request", "pass": len(generated_ids) <= max_new_tokens, "count": len(generated_ids), "max_new_tokens": max_new_tokens},
            {"name": "top_logprobs_present", "pass": bool(positions) and all(pos["top_logprobs"] for pos in positions)},
            {"name": "top1_matches_reference", "pass": bool(positions) and all(pos["top1_id"] == pos["reference_token_id"] for pos in positions)},
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
  parser.add_argument("--case-id", default="short_math_001")
  parser.add_argument("--max-new-tokens", type=int, default=1)
  parser.add_argument("--n-probs", type=int, default=5)
  parser.add_argument("--ctx-size", type=int, default=1024)
  parser.add_argument("--port", type=int, default=18143)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--poll-interval-s", type=int, default=10)
  parser.add_argument("--request-timeout-s", type=int, default=240)
  parser.add_argument("--ready-timeout-s", type=int, default=420)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-run")
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-distribution-capture-smoke-<UTC>.",
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


def prompt_rows() -> list[dict[str, Any]]:
  base = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts"
  rows = []
  for name in ("deterministic-greedy.jsonl", "router-stability.jsonl"):
    rows.extend(load_jsonl(base / name))
  return rows


def select_prompt(case_id: str) -> dict[str, Any]:
  for row in prompt_rows():
    if row.get("id") == case_id:
      return row
  raise SystemExit(f"case not found in short/router prompts: {case_id}")


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


def sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def launch_remote(
    *,
    args: argparse.Namespace,
    created_at: str,
    out_dir: Path,
    prompt_row: dict[str, Any],
) -> dict[str, Any]:
  stamp = created_at.replace("-", "").replace(":", "")
  remote_dir = f"{args.remote_root.rstrip('/')}/distribution-capture-smoke-{stamp}"
  raw_remote_dir = out_dir / "raw" / "remote"
  raw_remote_dir.mkdir(parents=True, exist_ok=True)
  remote_input = {
      "case_id": prompt_row["id"],
      "ctx_size": args.ctx_size,
      "max_new_tokens": args.max_new_tokens,
      "n_probs": args.n_probs,
      "port": args.port,
      "prompt": prompt_row["prompt"],
      "prompt_set": prompt_row.get("prompt_set"),
      "ready_timeout_s": args.ready_timeout_s,
      "request_timeout_s": args.request_timeout_s,
      "suite": prompt_row.get("suite"),
      "suite_manifest_name": prompt_row.get("suite"),
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
      "distribution-smoke.jsonl",
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


def load_remote_json(raw_remote_dir: Path, name: str) -> dict[str, Any]:
  path = raw_remote_dir / name
  if not path.is_file() or not path.read_text(encoding="utf-8").strip():
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def build_summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  checks = payload["checks"]
  lines = [
      "# R0 Distribution Capture Smoke",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- case: `{payload['case_id']}`",
      f"- max new tokens: {payload['max_new_tokens']}",
      f"- request status: `{result.get('row', {}).get('request_status')}`",
      f"- generated positions: {result.get('row', {}).get('generated_token_count')}",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This is a one-row smoke capture for the distribution route. It is not a",
      "full acceptance distribution bundle and not a per-boundary oracle bundle.",
      "",
      "## Checks",
      "",
  ]
  for check in checks:
    lines.append(f"- `{check['name']}`: `{str(check['pass']).lower()}`")
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  if args.max_new_tokens <= 0:
    raise SystemExit("--max-new-tokens must be positive")
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-distribution-capture-smoke-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  prompt_row = select_prompt(args.case_id)
  remote = launch_remote(args=args, created_at=created_at, out_dir=out_dir, prompt_row=prompt_row)
  raw_remote_dir = out_dir / "raw" / "remote"
  result = load_remote_json(raw_remote_dir, "result.json")
  row = result.get("row", {}) if isinstance(result.get("row"), dict) else {}
  checks = [
      {
          "name": "remote_run_completed",
          "pass": remote["timed_out"] is False
          and (raw_remote_dir / "exitcode").read_text(encoding="utf-8").strip() == "0",
      },
      {
          "name": "result_ok",
          "pass": result.get("ok") is True,
      },
      {
          "name": "current_workstream_row",
          "pass": row.get("workstream") == WORKSTREAM,
      },
      {
          "name": "generated_token_count_positive",
          "pass": isinstance(row.get("generated_token_count"), int)
          and row.get("generated_token_count") > 0,
          "count": row.get("generated_token_count"),
      },
      {
          "name": "generated_token_count_within_request",
          "pass": isinstance(row.get("generated_token_count"), int)
          and row.get("generated_token_count") <= args.max_new_tokens,
          "count": row.get("generated_token_count"),
          "max_new_tokens": args.max_new_tokens,
      },
      {
          "name": "top_logprobs_present",
          "pass": bool(row.get("distribution_positions"))
          and all(position.get("top_logprobs") for position in row["distribution_positions"]),
      },
      {
          "name": "smoke_does_not_claim_full_bundle",
          "pass": row.get("limitations", {}).get("smoke_only") is True
          and row.get("limitations", {}).get("not_a_full_r0_oracle_bundle") is True,
      },
      {
          "name": "oracle_gate_remains_open",
          "pass": True,
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  distribution_smoke_path = out_dir / "distribution-smoke.jsonl"
  remote_distribution = raw_remote_dir / "distribution-smoke.jsonl"
  if remote_distribution.is_file():
    distribution_smoke_path.write_text(
        remote_distribution.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
  else:
    distribution_smoke_path.write_text("", encoding="utf-8")
  payload = {
      "case_id": args.case_id,
      "created_at": created_at,
      "distribution_smoke_jsonl": str(distribution_smoke_path.relative_to(ROOT)),
      "host": args.host,
      "max_new_tokens": args.max_new_tokens,
      "model": {
          "path": MODEL_PATH,
          "sha256": MODEL_SHA256,
          "batch_size": 1,
      },
      "n_probs": args.n_probs,
      "r0_oracle_gate_closed": False,
      "remote": {
          "remote_dir": remote["remote_dir"],
          "timed_out": remote["timed_out"],
      },
      "required_checks_passed": required_checks_passed,
      "result": result,
      "checks": checks,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-distribution-capture-smoke.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "smoke.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_distribution_capture_smoke",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for metric, value in (
        ("generated_token_count", row.get("generated_token_count")),
        ("max_new_tokens", args.max_new_tokens),
        ("request_status", row.get("request_status")),
        ("required_checks_passed", required_checks_passed),
        ("r0_oracle_gate_closed", False),
    ):
      handle.write(json.dumps({
          "metric": metric,
          "phase": "r0_distribution_capture_smoke",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"distribution capture smoke output: {out_dir}")
  if not required_checks_passed:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
