#!/usr/bin/env python3
"""Materialize oracle prompt specs and verify active-tokenizer counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-oracle-prompt-materialization-v0"
DEFAULT_HOST = "local"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
LLAMA_TOKENIZE = "/home/intel/llama-cpp/llama-b9518/llama-tokenize"
PROMPT_BASE = ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km"
PROMPT_MANIFEST = PROMPT_BASE / "prompt-suites.json"
QUEUE_PATH = ROOT / "output/r0-oracle-capture-queue-20260626T074119Z/token-topk-tasks.jsonl"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-run")
  parser.add_argument("--count-timeout-s", type=int, default=300)
  parser.add_argument("--adjustment-limit", type=int, default=8)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-oracle-prompt-materialization-<UTC>.",
  )
  return parser.parse_args()


def run(
    cmd: list[str],
    *,
    input_text: str | None = None,
    timeout_s: int = 60,
) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        input=input_text,
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
        "stderr": stderr + f"\ntimeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_command: str, *, timeout_s: int = 60) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    value = json.load(handle)
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
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


def sha256_text(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def prompt_rows_by_id() -> dict[str, dict[str, Any]]:
  manifest = load_json(PROMPT_MANIFEST)
  rows: dict[str, dict[str, Any]] = {}
  for suite_name, suite in manifest.get("suites", {}).items():
    path_value = suite.get("path")
    if not isinstance(path_value, str):
      raise SystemExit(f"prompt suite {suite_name} missing path")
    for row in load_jsonl(PROMPT_BASE / path_value):
      row_id = row.get("id")
      if not isinstance(row_id, str) or not row_id:
        raise SystemExit(f"{path_value}: prompt row missing id")
      if row_id in rows:
        raise SystemExit(f"duplicate prompt row id: {row_id}")
      rows[row_id] = row
  return rows


def token_topk_case_order() -> list[str]:
  rows = load_jsonl(QUEUE_PATH)
  return [str(row["case_id"]) for row in rows]


def one_token_filler(token_count: int) -> str:
  if token_count < 0:
    raise ValueError("token_count must be nonnegative")
  if token_count == 0:
    return ""
  return "a" + (" a" * (token_count - 1))


def stable_words(seed: str, count: int) -> str:
  words = [
      "anchor",
      "vector",
      "matrix",
      "kernel",
      "ledger",
      "cache",
      "route",
      "signal",
      "window",
      "tensor",
      "token",
      "driver",
  ]
  selected = []
  for index in range(count):
    digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
    selected.append(words[digest[0] % len(words)])
  return " ".join(selected)


def build_prefill_prompt(row: dict[str, Any], fill_tokens: int) -> str:
  generator = row.get("generator", {})
  template = generator.get("line_template", "IQ36 prefill line {line}.")
  seed = generator.get("seed", row["id"])
  template_lines = []
  for line in range(1, 9):
    template_lines.append(str(template).format(line=line))
  parts = [
      f"IQ36 prefill shape prompt {row['id']}.",
      f"seed: {seed}",
      *template_lines,
      one_token_filler(fill_tokens),
  ]
  return "\n".join(part for part in parts if part)


def build_sentinel_prompt(row: dict[str, Any], fill_tokens: int) -> str:
  generator = row.get("generator", {})
  seed = str(generator.get("seed", row["id"]))
  sentinel_key = str(generator["sentinel_key"])
  sentinel_value = str(generator["sentinel_value"])
  before_tokens = fill_tokens // 2
  after_tokens = fill_tokens - before_tokens
  before = one_token_filler(before_tokens)
  after = one_token_filler(after_tokens)
  parts = [
      f"IQ36 sentinel retrieval prompt {row['id']}.",
      f"seed: {seed}",
      stable_words(f"{seed}:before", 32),
      before,
      f"{sentinel_key}: {sentinel_value}",
      after,
      stable_words(f"{seed}:after", 32),
      f"Question: {row['query']}",
      "Answer only the value.",
  ]
  return "\n".join(part for part in parts if part)


def build_generated_prompt(row: dict[str, Any], fill_tokens: int) -> str:
  if row.get("kind") == "prefill_shape":
    return build_prefill_prompt(row, fill_tokens)
  if row.get("kind") == "sentinel_retrieval":
    return build_sentinel_prompt(row, fill_tokens)
  raise SystemExit(f"{row['id']}: unsupported generated kind {row.get('kind')}")


def parse_token_count(stdout: str) -> int | None:
  for line in reversed(stdout.splitlines()):
    text = line.strip()
    prefix = "Total number of tokens:"
    if text.startswith(prefix):
      try:
        return int(text[len(prefix):].strip())
      except ValueError:
        return None
    try:
      return int(text)
    except ValueError:
      continue
  return None


def stage_file(host: str, local_path: Path, target_path: str, *,
               timeout_s: int) -> dict[str, Any]:
  mkdir = run_target(
      host,
      f"mkdir -p {shlex.quote(str(Path(target_path).parent))}",
      timeout_s=30,
  )
  if mkdir["returncode"] != 0:
    return {"mkdir": mkdir, "returncode": mkdir["returncode"]}
  copy = iq36_local.copy_to(host, local_path, target_path, timeout_s)
  return {"copy": copy, "mkdir": mkdir, "returncode": copy["returncode"]}


def remote_count_tokens(
    *,
    args: argparse.Namespace,
    local_path: Path,
    remote_path: str,
) -> dict[str, Any]:
  put = stage_file(
      args.host,
      local_path,
      remote_path,
      timeout_s=args.count_timeout_s,
  )
  if put["returncode"] != 0:
    return {
        "ok": False,
        "put": put,
        "token_count": None,
    }
  command = (
      "set -o pipefail; "
      f"{shlex.quote(LLAMA_TOKENIZE)} "
      f"-m {shlex.quote(MODEL_PATH)} "
      f"-f {shlex.quote(remote_path)} "
      "--show-count --log-disable "
      "| tail -n 1"
  )
  count = run_target(args.host, f"bash -lc {shlex.quote(command)}", timeout_s=args.count_timeout_s)
  return {
      "count_command": count["command"],
      "ok": count["returncode"] == 0 and parse_token_count(count["stdout"]) is not None,
      "put": put,
      "returncode": count["returncode"],
      "stderr": count["stderr"][-2000:],
      "stdout": count["stdout"][-2000:],
      "timed_out": count["timed_out"],
      "token_count": parse_token_count(count["stdout"]),
  }


def materialize_exact_generated(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    remote_dir: str,
    row: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], int]:
  target = row.get("target_prompt_tokens")
  if not isinstance(target, int) or target <= 0:
    raise SystemExit(f"{row['id']}: generated row missing positive target_prompt_tokens")
  tmp_dir = out_dir / "tmp"
  tmp_dir.mkdir(parents=True, exist_ok=True)
  fill_tokens = target
  attempts: list[dict[str, Any]] = []
  prompt_text = ""
  for attempt in range(1, args.adjustment_limit + 1):
    if fill_tokens < 0:
      raise SystemExit(f"{row['id']}: negative filler token count during adjustment")
    prompt_text = build_generated_prompt(row, fill_tokens)
    local_tmp = tmp_dir / f"{row['id']}.candidate.txt"
    local_tmp.write_text(prompt_text, encoding="utf-8")
    count_result = remote_count_tokens(
        args=args,
        local_path=local_tmp,
        remote_path=f"{remote_dir}/tmp/{row['id']}.candidate.txt",
    )
    observed = count_result.get("token_count")
    attempts.append({
        "attempt": attempt,
        "fill_tokens": fill_tokens,
        "observed_token_count": observed,
        "ok": count_result.get("ok") is True,
        "remote_path": f"{remote_dir}/tmp/{row['id']}.candidate.txt",
        "returncode": count_result.get("returncode"),
        "stderr_tail": count_result.get("stderr", ""),
        "stdout_tail": count_result.get("stdout", ""),
    })
    if observed == target:
      return prompt_text, attempts, fill_tokens
    if not isinstance(observed, int):
      raise SystemExit(f"{row['id']}: tokenizer count failed")
    fill_tokens += target - observed
  raise SystemExit(f"{row['id']}: could not materialize exact token count after attempts")


def materialize_row(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    remote_dir: str,
    row: dict[str, Any],
) -> dict[str, Any]:
  prompt_dir = out_dir / "prompts"
  prompt_dir.mkdir(parents=True, exist_ok=True)
  case_id = row["id"]
  attempts: list[dict[str, Any]] = []
  fill_tokens: int | None = None
  if row.get("kind") == "token_exact":
    prompt_text = row.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text:
      raise SystemExit(f"{case_id}: token_exact row missing prompt")
  else:
    prompt_text, attempts, fill_tokens = materialize_exact_generated(
        args=args,
        out_dir=out_dir,
        remote_dir=remote_dir,
        row=row,
    )
  local_path = prompt_dir / f"{case_id}.txt"
  local_path.write_text(prompt_text, encoding="utf-8")
  remote_path = f"{remote_dir}/prompts/{case_id}.txt"
  count_result = remote_count_tokens(args=args, local_path=local_path, remote_path=remote_path)
  observed = count_result.get("token_count")
  target = row.get("target_prompt_tokens")
  exact_target = target is None or observed == target
  return {
      "bytes": len(prompt_text.encode("utf-8")),
      "case_id": case_id,
      "exact_target_prompt_tokens": exact_target,
      "generator_type": row.get("generator", {}).get("type"),
      "kind": row.get("kind"),
      "materialization_attempts": attempts,
      "materialized_prompt_path": str(local_path.relative_to(ROOT)),
      "model": {
          "path": MODEL_PATH,
          "sha256": MODEL_SHA256,
          "batch_size": 1,
      },
      "observed_prompt_tokens": observed,
      "padding_token_count": fill_tokens,
      "prompt_set": row.get("prompt_set"),
      "prompt_utf8_sha256": sha256_text(prompt_text),
      "prompt_file_sha256": sha256_file(local_path),
      "remote_prompt_path": remote_path,
      "schema_version": SCHEMA_VERSION,
      "suite": row.get("suite"),
      "target_prompt_tokens": target,
      "tokenizer": "llama.cpp llama-tokenize",
      "tokenizer_count_result": {
          "ok": count_result.get("ok") is True,
          "returncode": count_result.get("returncode"),
          "stderr_tail": count_result.get("stderr", ""),
          "stdout_tail": count_result.get("stdout", ""),
          "timed_out": count_result.get("timed_out"),
      },
      "workstream": WORKSTREAM,
  }


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# R0 Oracle Prompt Materialization",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- prompt rows: {payload['prompt_row_count']}",
      f"- generated rows: {payload['generated_prompt_row_count']}",
      f"- exact generated rows: {payload['exact_generated_prompt_row_count']}",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This materializes prompt payloads and verifies active llama.cpp tokenizer",
      "counts. It is not a token/top-k bundle, distribution bundle, or",
      "per-boundary oracle bundle.",
      "",
      "| Prompt set | Rows | Exact target rows |",
      "|---|---:|---:|",
  ]
  for prompt_set, item in sorted(payload["prompt_set_counts"].items()):
    lines.append(f"| `{prompt_set}` | {item['rows']} | {item['exact_target_rows']} |")
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-oracle-prompt-materialization-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root.rstrip('/')}/oracle-prompt-materialization-{stamp}"

  rows_by_id = prompt_rows_by_id()
  case_order = token_topk_case_order()
  rows = []
  for case_id in case_order:
    if case_id not in rows_by_id:
      raise SystemExit(f"token/top-k queue references unknown prompt row: {case_id}")
    rows.append(materialize_row(args=args, out_dir=out_dir, remote_dir=remote_dir, row=rows_by_id[case_id]))

  generated_rows = [row for row in rows if row.get("target_prompt_tokens") is not None]
  prompt_set_counts: dict[str, dict[str, int]] = {}
  for row in rows:
    prompt_set = str(row.get("prompt_set"))
    item = prompt_set_counts.setdefault(prompt_set, {"exact_target_rows": 0, "rows": 0})
    item["rows"] += 1
    if row.get("exact_target_prompt_tokens") is True and row.get("target_prompt_tokens") is not None:
      item["exact_target_rows"] += 1

  checks = [
      {
          "name": "all_token_topk_queue_prompts_materialized",
          "pass": len(rows) == 26 and [row["case_id"] for row in rows] == case_order,
          "count": len(rows),
      },
      {
          "name": "all_generated_rows_exact_target_tokens",
          "pass": len(generated_rows) == 20
          and all(row.get("exact_target_prompt_tokens") is True for row in generated_rows),
          "count": len(generated_rows),
      },
      {
          "name": "token_exact_rows_counted",
          "pass": len(rows) - len(generated_rows) == 6
          and all(
              isinstance(row.get("observed_prompt_tokens"), int)
              for row in rows
              if row.get("target_prompt_tokens") is None
          ),
      },
      {
          "name": "all_rows_have_prompt_hashes",
          "pass": all(row.get("prompt_utf8_sha256") and row.get("prompt_file_sha256") for row in rows),
      },
      {
          "name": "oracle_gate_remains_open",
          "pass": True,
      },
  ]
  required_checks_passed = all(check["pass"] for check in checks)
  payload = {
      "created_at": created_at,
      "exact_generated_prompt_row_count": sum(
          1 for row in generated_rows if row.get("exact_target_prompt_tokens") is True
      ),
      "generated_prompt_row_count": len(generated_rows),
      "host": args.host,
      "model": {
          "path": MODEL_PATH,
          "sha256": MODEL_SHA256,
          "batch_size": 1,
      },
      "prompt_row_count": len(rows),
      "prompt_set_counts": prompt_set_counts,
      "r0_oracle_gate_closed": False,
      "remote_dir": remote_dir,
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_jsonl(out_dir / "materialized-prompts.jsonl", rows)
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-oracle-prompt-materialize.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "materialization.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r0_oracle_prompt_materialization",
      "required_checks_passed": required_checks_passed,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps({
          "case_id": row["case_id"],
          "metric": "observed_prompt_tokens",
          "phase": "r0_oracle_prompt_materialization",
          "prompt_set": row.get("prompt_set"),
          "target_prompt_tokens": row.get("target_prompt_tokens"),
          "value": row.get("observed_prompt_tokens"),
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"oracle prompt materialization output: {out_dir}")
  if not required_checks_passed:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
