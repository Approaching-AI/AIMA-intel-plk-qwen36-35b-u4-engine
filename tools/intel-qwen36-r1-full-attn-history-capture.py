#!/usr/bin/env python3
"""Capture the short_math_001 L3 full-attention token history.

The canonical R0 oracle bundle captures one source token position. Full
attention core validation for token 15 also needs the layer-3 K/V history for
tokens 0..15. This tool reruns the existing patched boundary-capture
executable for each short prompt token and copies only the L3 tensors needed
for an R1 full-attention core compare.
"""

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
SCHEMA_VERSION = "intel-qwen36-r1-full-attn-history-capture-v0"
DEFAULT_HOST = "local"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
MODEL_SHA256 = "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
CASE_ID = "short_math_001"
TOKEN_COUNT = 16
MAX_TENSORS = 130

SELECTED_SPECS = {
    "q_rope": {"size_bytes": 16384, "value_count": 4096},
    "k_rope": {"size_bytes": 2048, "value_count": 512},
    "v": {"size_bytes": 2048, "value_count": 512},
    "attn_pregate": {"size_bytes": 16384, "value_count": 4096},
    "attn_gated": {"size_bytes": 16384, "value_count": 4096},
    "attn_output": {"size_bytes": 8192, "value_count": 2048},
}


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--out-dir", type=Path, default=None)
  return parser.parse_args()


def run(cmd: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    return {
        "cmd": cmd,
        "returncode": 124,
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") + "\ntimeout",
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
        "timed_out": True,
    }
  return {
      "cmd": cmd,
      "returncode": proc.returncode,
      "stderr": proc.stderr,
      "stdout": proc.stdout,
      "timed_out": False,
  }


def run_target(host: str, remote_command: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def copy_from(host: str, remote_path: str, local_path: Path, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_from(host, remote_path, local_path, timeout_s)


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


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def find_prompt_row(materialized_path: Path) -> dict[str, Any]:
  for row in load_jsonl(materialized_path):
    if row.get("case_id") == CASE_ID:
      return row
  raise SystemExit(f"{materialized_path}: missing {CASE_ID}")


def parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" in line:
      key, value = line.split("=", 1)
      values[key.strip()] = value.strip()
  return values


REMOTE_SELECT_SCRIPT = r'''
import hashlib
import json
import shutil
import sys
from pathlib import Path

raw = Path(sys.argv[1])
selected = Path(sys.argv[2])
token = int(sys.argv[3])
selected.mkdir(parents=True, exist_ok=True)

want = {
    ("Qcur-3", "ROPE"): "q_rope",
    ("Kcur-3", "ROPE"): "k_rope",
    ("Vcur-3", "RESHAPE"): "v",
    ("attn_pregate-3", "RESHAPE"): "attn_pregate",
    ("attn_gated-3", "MUL"): "attn_gated",
    ("attn_output-3", "MUL_MAT"): "attn_output",
}
records = {}
rows = []
with (raw / "tensor-dumps.jsonl").open("r", encoding="utf-8") as fh:
  for line in fh:
    row = json.loads(line)
    rows.append(row)
    key = (row.get("tensor_name"), row.get("tensor_op"))
    name = want.get(key)
    if name is None or name in records:
      continue
    src = raw / row["payload_path"]
    dst_name = f"tok{token:02d}_{name}.bin"
    dst = selected / dst_name
    shutil.copy2(src, dst)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    records[name] = {
        "nbytes": row.get("nbytes"),
        "original_payload_path": row.get("payload_path"),
        "selected_name": dst_name,
        "sha256": digest,
        "tensor_name": row.get("tensor_name"),
        "tensor_op": row.get("tensor_op"),
    }

missing = sorted(set(want.values()) - set(records))
summary = json.loads((raw / "capture-summary.json").read_text(encoding="utf-8"))
manifest = {
    "capture_summary": summary,
    "missing": missing,
    "selected": records,
    "tensor_jsonl_row_count": len(rows),
    "token_position": token,
}
(selected / f"tok{token:02d}-selected-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if missing:
  raise SystemExit("missing selected tensors: " + ",".join(missing))
'''


def capture_one(
    host: str,
    executable: str,
    prompt_path: str,
    remote_base: str,
    token_position: int,
    timeout_s: int,
) -> dict[str, Any]:
  raw_dir = f"{remote_base}/tok{token_position:02d}/raw"
  selected_dir = f"{remote_base}/tok{token_position:02d}/selected"
  command = " ".join([
      shlex.quote(executable),
      "--model",
      shlex.quote(MODEL_PATH),
      "--prompt-file",
      shlex.quote(prompt_path),
      "--out-dir",
      shlex.quote(raw_dir),
      "--case-id",
      shlex.quote(CASE_ID),
      "--source-token-position",
      str(token_position),
      "--threads",
      "1",
      "--n-ctx",
      "32",
      "--ngl",
      "0",
      "--max-tensors",
      str(MAX_TENSORS),
  ])
  remote_script = "\n".join([
      "set -euo pipefail",
      f"raw={shlex.quote(raw_dir)}",
      f"selected={shlex.quote(selected_dir)}",
      "rm -rf \"$raw\" \"$selected\"",
      "mkdir -p \"$raw\" \"$selected\"",
      f"{command} > \"$raw/run.stdout\" 2> \"$raw/run.stderr\"",
      "python3 - \"$raw\" \"$selected\" "
      f"{token_position} <<'PY'",
      REMOTE_SELECT_SCRIPT,
      "PY",
      f"printf 'token_position={token_position}\\n'",
      "printf 'remote_raw_dir=%s\\n' \"$raw\"",
      "printf 'remote_selected_dir=%s\\n' \"$selected\"",
      "printf 'selected_file_count='; find \"$selected\" -type f | wc -l",
  ])
  result = run_target(host, remote_script, timeout_s)
  values = parse_key_values(result.get("stdout", ""))
  result["remote_raw_dir"] = values.get("remote_raw_dir", raw_dir)
  result["remote_selected_dir"] = values.get("remote_selected_dir", selected_dir)
  result["selected_file_count"] = int(values.get("selected_file_count", "0"))
  result["token_position"] = token_position
  return result


def build_history(local_selected_root: Path) -> dict[str, Any]:
  tokens: list[dict[str, Any]] = []
  for token in range(TOKEN_COUNT):
    token_dir = local_selected_root / f"tok{token:02d}" / "selected"
    manifest_path = token_dir / f"tok{token:02d}-selected-manifest.json"
    manifest = load_json(manifest_path)
    selected = manifest.get("selected", {})
    if set(selected) != set(SELECTED_SPECS):
      raise SystemExit(f"token {token}: selected tensor set mismatch")
    token_payloads: dict[str, Any] = {}
    for name, spec in SELECTED_SPECS.items():
      entry = selected[name]
      path = token_dir / entry["selected_name"]
      if not path.exists():
        raise SystemExit(f"token {token}: missing selected payload {name}")
      if path.stat().st_size != spec["size_bytes"]:
        raise SystemExit(f"token {token}: selected payload size mismatch {name}")
      token_payloads[name] = {
          "path": str(path.relative_to(ROOT)),
          "sha256": sha256_file(path),
          "size_bytes": spec["size_bytes"],
          "tensor_name": entry["tensor_name"],
          "tensor_op": entry["tensor_op"],
          "value_count": spec["value_count"],
      }
    summary = manifest.get("capture_summary", {})
    tokens.append({
        "captured_tensor_count": summary.get("captured_tensor_count"),
        "payloads": token_payloads,
        "prompt_token_count": summary.get("prompt_token_count"),
        "source_token_position": summary.get("source_token_position"),
        "tensor_jsonl_row_count": manifest.get("tensor_jsonl_row_count"),
    })
  return {
      "history_token_count": len(tokens),
      "layer_index": 3,
      "payload_value_counts": {
          name: spec["value_count"] for name, spec in SELECTED_SPECS.items()
      },
      "source_prompt_case_id": CASE_ID,
      "source_token_positions": list(range(TOKEN_COUNT)),
      "tokens": tokens,
  }


def build_summary(payload: dict[str, Any]) -> str:
  history = payload["full_attn_history_capture"]
  lines = [
      "# R1 Full-Attention History Capture",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model']['path']}`",
      f"- case id: `{history['source_prompt_case_id']}`",
      f"- layer index: {history['layer_index']}",
      f"- source token positions: `{history['source_token_positions']}`",
      f"- history token count: {history['history_token_count']}",
      f"- capture passed: `{str(payload['full_attn_history_capture_passed']).lower()}`",
      "",
      "This artifact captures only the L3 Q/K/V and attention output-side",
      "payloads needed to validate the token-15 full-attention core. It is not",
      "native candidate JSONL evidence and does not close R1.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-full-attn-history-capture-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  selected_root = out_dir / "selected"

  build_path = latest("r0-boundary-capture-build-*", "build.json")
  if build_path is None:
    raise SystemExit("no boundary capture build artifact found")
  build = load_json(build_path)
  executable = build.get("build_route", {}).get("executable_path")
  if not isinstance(executable, str) or not executable:
    raise SystemExit("boundary capture build missing executable path")

  materialized_path = latest(
      "r0-oracle-prompt-materialization-*",
      "materialized-prompts.jsonl",
  )
  if materialized_path is None:
    raise SystemExit("no prompt materialization artifact found")
  prompt_row = find_prompt_row(materialized_path)
  prompt_path = prompt_row.get("remote_prompt_path")
  if not isinstance(prompt_path, str) or not prompt_path:
    raise SystemExit("prompt materialization missing remote_prompt_path")
  if prompt_row.get("observed_prompt_tokens") != TOKEN_COUNT:
    raise SystemExit("short_math_001 prompt token count changed")

  remote_base = f"{args.remote_root}/full-attn-history-capture-{stamp}"
  captures: list[dict[str, Any]] = []
  copy_results: list[dict[str, Any]] = []
  for token in range(TOKEN_COUNT):
    capture = capture_one(
        args.host,
        executable,
        prompt_path,
        remote_base,
        token,
        args.timeout_s,
    )
    captures.append(capture)
    if capture["returncode"] == 0:
      local_selected_dir = selected_root / f"tok{token:02d}" / "selected"
      local_selected_dir.mkdir(parents=True, exist_ok=True)
      copy_result = copy_from(
          args.host,
          f"{capture['remote_selected_dir']}/.",
          local_selected_dir,
          args.timeout_s,
      )
    else:
      copy_result = {
          "cmd": [],
          "returncode": 1,
          "stderr": "capture failed",
          "stdout": "",
          "timed_out": False,
      }
    copy_results.append(copy_result)

  history = (
      build_history(selected_root)
      if all(item.get("returncode") == 0 for item in copy_results)
      else {
          "history_token_count": 0,
          "layer_index": 3,
          "payload_value_counts": {},
          "source_prompt_case_id": CASE_ID,
          "source_token_positions": [],
          "tokens": [],
      }
  )
  passed = (
      len(captures) == TOKEN_COUNT
      and all(item.get("returncode") == 0 for item in captures)
      and all(item.get("timed_out") is False for item in captures)
      and all(item.get("selected_file_count") == len(SELECTED_SPECS) + 1 for item in captures)
      and all(item.get("returncode") == 0 for item in copy_results)
      and history.get("history_token_count") == TOKEN_COUNT
      and all(
          token.get("source_token_position") == index
          and token.get("prompt_token_count") == TOKEN_COUNT
          and set(token.get("payloads", {})) == set(SELECTED_SPECS)
          for index, token in enumerate(history.get("tokens", []))
      )
  )

  payload = {
      "capture_build_artifact": str(build_path.parent.relative_to(ROOT)),
      "created_at": created_at,
      "full_attn_history_capture": history,
      "full_attn_history_capture_passed": passed,
      "host": args.host,
      "model": {
          "path": MODEL_PATH,
          "sha256": MODEL_SHA256,
      },
      "prompt_materialization_artifact": str(materialized_path.parent.relative_to(ROOT)),
      "remote_base": remote_base,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_captures": captures,
      "target_copies": copy_results,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "case_id": CASE_ID,
      "host": args.host,
      "model": payload["model"],
      "remote_base": remote_base,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-full-attn-history-capture.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "history.json", payload)
  checks = [
      {"name": "capture_executable_available", "pass": isinstance(executable, str)},
      {"name": "prompt_materialization_has_16_tokens", "pass": prompt_row.get("observed_prompt_tokens") == TOKEN_COUNT},
      {"name": "all_token_captures_succeeded", "pass": all(item.get("returncode") == 0 for item in captures)},
      {"name": "all_selected_payload_sets_copied", "pass": all(item.get("returncode") == 0 for item in copy_results)},
      {"name": "history_payloads_complete", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "full_attn_history_capture_passed": passed,
      "gate": "r1_full_attn_history_capture",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("full_attn_history_capture_passed", passed),
        ("history_token_count", history.get("history_token_count")),
        ("selected_payload_count", sum(len(token.get("payloads", {})) for token in history.get("tokens", []))),
        ("r1_native_correctness_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_full_attn_history_capture",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 full-attn history capture output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
