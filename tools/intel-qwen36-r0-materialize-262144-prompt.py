#!/usr/bin/env python3
"""Materialize an exact 262144-token OpenVINO denominator prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_HOST = "local"
TARGET_TOKENS = 262144
PROMPT_NAME = "prompt_10_256Kin_512out_r1.txt"
TARGET_PROMPT_DIR = "/home/intel/ov/prompts"
OPENVINO_MODEL = "/home/intel/Qwen3.6-35B-A3B-ov"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
  parser.add_argument("--prompt-name", default=PROMPT_NAME)
  parser.add_argument("--install-target", action="store_true")
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-materialize-262144-prompt-<UTC>.",
  )
  return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


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


def count_tokens_local(host: str, text: str, timeout_s: int = 120) -> dict[str, Any]:
  command = (
      "cd /home/intel/ov && . openvino_env/bin/activate && "
      "python -c \"import sys, openvino_genai as ov; "
      "text=sys.stdin.read(); "
      f"tok=ov.Tokenizer('{OPENVINO_MODEL}'); "
      "ids=tok.encode(text).input_ids; "
      "print(ids.get_shape()[1] if hasattr(ids,'get_shape') else len(ids))\""
  )
  result = iq36_local.run_target(
      host,
      command,
      timeout_s,
      input_text=text,
  )
  try:
    count = int(result["stdout"].strip())
  except ValueError:
    count = None
  result["token_count"] = count
  return result


def materialize_prompt(target_tokens: int) -> str:
  if target_tokens < 1:
    raise SystemExit("--target-tokens must be positive")
  return "a" + (" a" * (target_tokens - 1))


def install_prompt(host: str, prompt_path: Path, prompt_name: str) -> dict[str, Any]:
  staged_tmp = f"/tmp/{prompt_name}.tmp"
  target_path = f"{TARGET_PROMPT_DIR}/{prompt_name}"
  copy = iq36_local.copy_to(host, prompt_path, staged_tmp, 60)
  if copy["returncode"] != 0:
    return {"installed": False, "copy": copy}
  move = iq36_local.run_target(
      host,
      f"mkdir -p {TARGET_PROMPT_DIR} && mv {staged_tmp} {target_path} && "
      f"sha256sum {target_path} && stat -c '%s %n' {target_path}",
      60,
  )
  return {
      "installed": move["returncode"] == 0,
      "target_path": target_path,
      "rollback": f"rm -f {target_path}",
      "copy": copy,
      "verify": move,
  }


def build_summary(result: dict[str, Any]) -> str:
  correctness = result["correctness"]
  install = result.get("target_install")
  lines = [
      "# R0 262144 prompt materialization",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- prompt name: `{result['prompt']['name']}`",
      f"- target tokens: {result['prompt']['target_tokens']}",
      f"- tokenizer count: {correctness['tokenizer_count']}",
      f"- exact token count: `{str(correctness['exact_token_count']).lower()}`",
      f"- local path: `{result['prompt']['local_path']}`",
  ]
  if install:
    lines.append(f"- installed on target: `{str(install['installed']).lower()}`")
    lines.append(f"- target path: `{install.get('target_path')}`")
    lines.append(f"- rollback: `{install.get('rollback')}`")
  else:
    lines.append("- installed on target: `false`")
  lines.extend(
      [
          "",
          "This only prepares denominator input. It does not run the denominator",
          "benchmark or close R0.",
          "",
      ]
  )
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  created_at = iso_now()
  out_dir = args.out_dir
  if out_dir is None:
    stamp = created_at.replace("-", "").replace(":", "")
    out_dir = ROOT / f"output/r0-materialize-262144-prompt-{stamp}"
  out_dir = out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  prompt_text = materialize_prompt(args.target_tokens)
  prompt_path = out_dir / args.prompt_name
  prompt_path.write_text(prompt_text, encoding="utf-8")
  local_sha = file_sha256(prompt_path)
  tokenizer_result = count_tokens_local(args.host, prompt_text)
  exact = tokenizer_result.get("token_count") == args.target_tokens
  install = None
  if args.install_target:
    if not exact:
      raise SystemExit("refusing to install prompt with non-exact token count")
    install = install_prompt(args.host, prompt_path, args.prompt_name)

  result = {
      "correctness": {
          "exact_token_count": exact,
          "tokenizer_count": tokenizer_result.get("token_count"),
          "tokenizer_model": OPENVINO_MODEL,
          "tokenizer_result": tokenizer_result,
      },
      "created_at": created_at,
      "host": args.host,
      "prompt": {
          "bytes": len(prompt_text.encode("utf-8")),
          "local_path": str(prompt_path),
          "name": args.prompt_name,
          "sha256": local_sha,
          "target_tokens": args.target_tokens,
      },
      "r0_denominator_gate_closed": False,
      "schema_version": "intel-qwen36-r0-materialized-denominator-prompt-v0",
      "target_install": install,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "materialized-prompt.json", result)
  (out_dir / "summary.md").write_text(build_summary(result), encoding="utf-8")
  print(f"materialized prompt output: {out_dir}")
  if not exact:
    raise SystemExit(1)
  if args.install_target and not install.get("installed"):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
