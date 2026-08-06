#!/usr/bin/env python3
"""Materialize the locked fresh corpus with the target model tokenizer only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = (
    "intel-qwen36-state-conditioned-head-correction-token-materialize-v0"
)
CURRENT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "corpus_token_materialization_gate"
)
SELECTED_NEXT_ROUTE = (
    "router_prompt_distribution_state_conditioned_head_correction_"
    "fit_observable_source_gate"
)
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
TOKENIZER_PATH = "/home/intel/llama-cpp/llama-b9518/llama-tokenize"


REMOTE_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def parse_token_ids(stdout: str) -> list[int]:
  for line in reversed(stdout.splitlines()):
    text = line.strip()
    if text.startswith("[") and text.endswith("]"):
      value = json.loads(text)
      if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return value
  raise RuntimeError("could not parse llama-tokenize token ids")


def main() -> int:
  contract_path = Path(sys.argv[1])
  model_path = sys.argv[2]
  tokenizer_path = sys.argv[3]
  contract = json.loads(contract_path.read_text(encoding="utf-8"))
  work_dir = contract_path.parent / "prompts"
  work_dir.mkdir(parents=True, exist_ok=True)
  env = dict(os.environ)
  env["INTEL_FORCE_PROBE"] = "b080"
  rows = []
  for prompt in contract["prompts"]:
    prompt_path = work_dir / f"{prompt['id']}.txt"
    prompt_path.write_text(prompt["prompt"], encoding="utf-8")
    command = [
        tokenizer_path, "-m", model_path, "-f", str(prompt_path),
        "--ids", "--log-disable",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=180, check=False, env=env)
    row = {
        "id": prompt["id"],
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
    }
    if completed.returncode == 0:
      row["token_ids"] = parse_token_ids(completed.stdout)
    rows.append(row)
    if completed.returncode != 0:
      break
  print(json.dumps({
      "schema_version": "intel-qwen36-remote-tokenize-v0",
      "model_path": model_path,
      "tokenizer_path": tokenizer_path,
      "rows": rows,
  }, sort_keys=True))
  return 0 if len(rows) == len(contract["prompts"]) and all(
      row["returncode"] == 0 for row in rows) else 2


if __name__ == "__main__":
  raise SystemExit(main())
'''


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _parse_stdout(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    text = line.strip()
    if text.startswith("{") and text.endswith("}"):
      value = json.loads(text)
      if isinstance(value, dict):
        return value
  raise json.JSONDecodeError("no JSON result in remote stdout", stdout, 0)


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _short_result(result: dict[str, Any]) -> dict[str, Any]:
  return {
      "cmd": result.get("cmd"),
      "returncode": result.get("returncode"),
      "stderr": result.get("stderr"),
  }


def _materialize(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
  args.out_dir.mkdir(parents=True, exist_ok=True)
  token_dir = args.out_dir / "token-input"
  token_dir.mkdir(parents=True, exist_ok=True)
  local_contract = args.out_dir / "locked-corpus.json"
  local_runner = args.out_dir / "remote-tokenize.py"
  local_contract.write_text(
      json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  local_runner.write_text(REMOTE_RUNNER, encoding="utf-8")
  remote_dir = f"{args.remote_root}/seq571-state-conditioned-head-correction-token-input"
  mkdir = iq36_local.run_target(
      args.host, f"mkdir -p {shlex.quote(remote_dir)}", args.timeout_s)
  contract_transfer = (
      iq36_local.copy_to(
          args.host, local_contract, f"{remote_dir}/locked-corpus.json",
          args.timeout_s)
      if mkdir.get("returncode") == 0 else
      {"returncode": 1, "stderr": "remote mkdir failed", "stdout": ""})
  runner_transfer = (
      iq36_local.copy_to(
          args.host, local_runner, f"{remote_dir}/remote-tokenize.py",
          args.timeout_s)
      if contract_transfer.get("returncode") == 0 else
      {"returncode": 1, "stderr": "contract transfer failed", "stdout": ""})
  remote_command = " ".join([
      "python3",
      shlex.quote(f"{remote_dir}/remote-tokenize.py"),
      shlex.quote(f"{remote_dir}/locked-corpus.json"),
      shlex.quote(args.model),
      shlex.quote(args.tokenizer),
  ])
  run = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(remote_command)}", args.timeout_s)
      if runner_transfer.get("returncode") == 0 else
      {"returncode": 1, "stderr": "runner transfer failed", "stdout": ""})
  parsed: dict[str, Any] = {}
  parse_error = None
  try:
    parsed = _parse_stdout(run.get("stdout", ""))
  except json.JSONDecodeError as exc:
    parse_error = str(exc)
  prompt_by_id = {
      row["id"]: row for row in contract.get("prompts", [])
      if isinstance(row, dict) and isinstance(row.get("id"), str)
  }
  token_rows = []
  cases_lines = []
  for row in parsed.get("rows", []):
    if not isinstance(row, dict):
      continue
    case_id = row.get("id")
    token_ids = row.get("token_ids")
    prompt = prompt_by_id.get(case_id)
    if not isinstance(case_id, str) or not isinstance(token_ids, list) \
        or prompt is None:
      continue
    raw = b"".join(struct.pack("<I", int(token_id)) for token_id in token_ids)
    token_path = token_dir / f"{case_id}.tokens.u32"
    token_path.write_bytes(raw)
    token_sha = _sha256_bytes(raw)
    prompt_sha = _sha256_bytes(prompt["prompt"].encode("utf-8"))
    cases_lines.append("\t".join([
        case_id, str(len(token_ids)), token_sha[:16], str(token_ids[0]),
        str(token_ids[-1]), token_path.name,
    ]))
    token_rows.append({
        "id": case_id,
        "domain": prompt["domain"],
        "split": prompt["split"],
        "prompt_sha256": prompt_sha,
        "token_count": len(token_ids),
        "first_token_id": token_ids[0],
        "last_token_id": token_ids[-1],
        "token_file": _rel(token_path),
        "token_sha256": token_sha,
    })
  (token_dir / "cases.tsv").write_text(
      "\n".join(cases_lines) + ("\n" if cases_lines else ""),
      encoding="utf-8")
  (args.out_dir / "token-manifest.json").write_text(
      json.dumps(token_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return {
      "remote": {
          "dir": remote_dir,
          "mkdir": _short_result(mkdir),
          "contract_transfer": _short_result(contract_transfer),
          "runner_transfer": _short_result(runner_transfer),
          "run": _short_result(run),
          "command": remote_command,
          "parse_error": parse_error,
          "model_path": parsed.get("model_path"),
          "tokenizer_path": parsed.get("tokenizer_path"),
          "row_count": len(parsed.get("rows", [])),
      },
      "token_rows": token_rows,
      "token_dir": _rel(token_dir),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  predecessor = _load(args.predecessor)
  contract = _load(args.contract)
  predecessor_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("token_materialization_allowed") is True
      and predecessor.get("decode_row_allowed") is False
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and predecessor.get("inputs", {}).get("contract_sha256")
      == _sha256_bytes(args.contract.read_bytes())
      and _has_candidate(routes, 570, CURRENT_ROUTE)
      and _has_switch(
          routes, 570,
          "select_router_prompt_distribution_state_conditioned_head_"
          "correction_corpus_token_materialization_gate"))
  materialized = (
      _materialize(args, contract) if predecessor_selects else {
          "remote": {}, "token_rows": [], "token_dir": None})
  token_rows = materialized["token_rows"]
  expected_prompts = contract.get("prompts", [])
  expected_ids = [row.get("id") for row in expected_prompts
                  if isinstance(row, dict)]
  actual_ids = [row.get("id") for row in token_rows]
  token_shape_passes = (
      len(token_rows) == 24
      and actual_ids == expected_ids
      and all(isinstance(row.get("token_count"), int)
              and 1 <= row["token_count"] <= 512
              and isinstance(row.get("first_token_id"), int)
              and isinstance(row.get("last_token_id"), int)
              for row in token_rows))
  files_pass = all(
      (ROOT / row["token_file"]).is_file()
      and _sha256_bytes((ROOT / row["token_file"]).read_bytes())
      == row["token_sha256"]
      for row in token_rows)
  remote = materialized["remote"]
  tokenizer_only = (
      remote.get("run", {}).get("returncode") == 0
      and remote.get("parse_error") is None
      and remote.get("row_count") == 24
      and remote.get("model_path") == args.model
      and remote.get("tokenizer_path") == args.tokenizer
      and args.tokenizer in remote.get("command", "")
      and "llama-server" not in remote.get("command", "")
      and "decode" not in remote.get("command", ""))
  checks = [
      {"name": "seq570_selected_token_materialization_gate",
       "pass": predecessor_selects},
      {"name": "remote_command_used_locked_model_tokenizer_only",
       "pass": tokenizer_only},
      {"name": "all_24_locked_cases_materialized_in_contract_order",
       "pass": token_shape_passes},
      {"name": "all_u32_token_files_match_recorded_hashes",
       "pass": files_pass},
      {"name": "cases_tsv_and_manifest_exist",
       "pass": (args.out_dir / "token-input/cases.tsv").is_file()
               and (args.out_dir / "token-manifest.json").is_file()},
  ]
  required = all(bool(row["pass"]) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": _rel(args.routes),
          "predecessor": _rel(args.predecessor),
          "contract": _rel(args.contract),
          "contract_sha256": _sha256_bytes(args.contract.read_bytes()),
          "host": args.host,
          "model_path": args.model,
          "tokenizer_path": args.tokenizer,
      },
      "materialization": materialized,
      "checks": checks,
      "required_checks_passed": required,
      "fit_observable_source_gate_allowed": required,
      "decode_row_allowed": False,
      "model_fit_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          "accept_locked_corpus_token_materialization"
          if required else "reject_locked_corpus_token_materialization"),
      "selected_next_route": SELECTED_NEXT_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "The target llama tokenizer produced ordered, hashed u32 token files "
          "for all 24 locked prompts without launching a model server or decode. "
          "Add and source-gate the fit-observable top8 diagnostic next; fit, "
          "validation, and test rows remain blocked."
          if required else
          "Fix target tokenizer execution, case order, token bounds, or hashes "
          "before any decode or fitting."),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  rows = metrics["materialization"]["token_rows"]
  counts = [row["token_count"] for row in rows]
  lines = [
      f"# Seq{metrics['sequence']} State-Conditioned Corpus Token Materialization",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- tokenized cases: `{len(rows)}`",
      f"- token-count min/max: `{min(counts) if counts else None}` / `{max(counts) if counts else None}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is tokenizer evidence only. No prompt was decoded.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=571)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq570-state-conditioned-head-correction-corpus-gate-20260710Tseq570Z/metrics.json")
  parser.add_argument(
      "--contract", type=Path,
      default=ACTIVE / "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-corpus-2026-07-10.json")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default="local")
  parser.add_argument("--model", default=MODEL_PATH)
  parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-gpu")
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq571-state-conditioned-head-correction-token-input-20260710Tseq571Z")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "fit_observable_source_gate_allowed": metrics["fit_observable_source_gate_allowed"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
