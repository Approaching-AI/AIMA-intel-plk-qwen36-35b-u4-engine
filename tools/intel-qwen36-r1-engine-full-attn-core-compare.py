#!/usr/bin/env python3
"""Build and run the engine-side L3 full-attention core compare.

This validates the first full-attention layer's causal attention core for
``short_math_001`` token position 15 using captured token 0..15 L3 K/V
history. It is component evidence only: it consumes captured oracle K/V
history, does not update KV cache itself, does not emit native candidate JSONL
rows, does not close R1, and does not allow speed claims.
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
SCHEMA_VERSION = "intel-qwen36-r1-engine-full-attn-core-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-full-attn-core-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
TOKEN_COUNT = 16

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/full_attn_core_compare.cpp", "tests/full_attn_core_compare.cpp"),
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--history-json", type=Path, default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=360)
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
    }
  return {
      "cmd": cmd,
      "returncode": proc.returncode,
      "stderr": proc.stderr,
      "stdout": proc.stdout,
  }


def run_target(host: str, remote_command: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def copy_to(host: str, local_path: Path, remote_path: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_to(host, local_path, remote_path, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def latest_history_json() -> Path:
  paths = sorted((ROOT / "output").glob("r1-full-attn-history-capture-*/history.json"))
  if not paths:
    raise SystemExit("no full-attn history capture artifact found")
  return paths[-1]


def payload_entry(
    history_json: Path,
    token_index: int,
    payload_name: str,
    stage_name: str,
    token_payloads: dict[str, Any],
) -> dict[str, Any]:
  entry = token_payloads.get(payload_name)
  if not isinstance(entry, dict):
    raise SystemExit(f"{history_json}: token {token_index} missing {payload_name}")
  local_path = (ROOT / entry["path"]).resolve()
  if not local_path.exists():
    raise SystemExit(f"{history_json}: missing payload file {local_path}")
  if local_path.stat().st_size != entry.get("size_bytes"):
    raise SystemExit(f"{history_json}: size mismatch for {local_path}")
  digest = sha256_file(local_path)
  if digest != entry.get("sha256"):
    raise SystemExit(f"{history_json}: sha256 mismatch for {local_path}")
  return {
      "local_path": local_path,
      "path": str(local_path.relative_to(ROOT)),
      "sha256": digest,
      "size_bytes": entry.get("size_bytes"),
      "stage_name": stage_name,
      "tensor_name": entry.get("tensor_name"),
      "tensor_op": entry.get("tensor_op"),
      "token_position": token_index,
      "value_count": entry.get("value_count"),
  }


def resolve_history(history_json: Path) -> dict[str, Any]:
  history_json = history_json.resolve()
  payload = load_json(history_json)
  if payload.get("schema_version") != "intel-qwen36-r1-full-attn-history-capture-v0":
    raise SystemExit(f"{history_json}: unexpected history schema")
  if payload.get("full_attn_history_capture_passed") is not True:
    raise SystemExit(f"{history_json}: history capture did not pass")
  history = payload.get("full_attn_history_capture", {})
  tokens = history.get("tokens", [])
  if history.get("history_token_count") != TOKEN_COUNT or len(tokens) != TOKEN_COUNT:
    raise SystemExit(f"{history_json}: expected {TOKEN_COUNT} token history")
  if history.get("source_prompt_case_id") != "short_math_001":
    raise SystemExit(f"{history_json}: unexpected prompt case")

  staged: dict[str, dict[str, Any]] = {}
  for token_index, token in enumerate(tokens):
    if token.get("source_token_position") != token_index:
      raise SystemExit(f"{history_json}: token position mismatch at {token_index}")
    token_payloads = token.get("payloads", {})
    prefix = f"tok{token_index:02d}"
    for payload_name in ("k_rope", "v"):
      stage_name = f"{prefix}_{payload_name}.bin"
      staged[stage_name] = payload_entry(
          history_json, token_index, payload_name, stage_name, token_payloads)
    if token_index == 15:
      for payload_name in ("q_rope", "attn_pregate"):
        stage_name = f"{prefix}_{payload_name}.bin"
        staged[stage_name] = payload_entry(
            history_json, token_index, payload_name, stage_name, token_payloads)

  return {
      "created_at": payload.get("created_at"),
      "history_artifact": str(history_json.parent.relative_to(ROOT)),
      "history_json": str(history_json.relative_to(ROOT)),
      "history_token_count": history.get("history_token_count"),
      "payloads": staged,
      "remote_base": payload.get("remote_base"),
      "source_prompt_case_id": history.get("source_prompt_case_id"),
      "source_token_positions": history.get("source_token_positions"),
  }


def slim_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
      name: {key: value for key, value in payload.items() if key != "local_path"}
      for name, payload in payloads.items()
  }


def comparison_passed(comparison: dict[str, Any]) -> bool:
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0
      and comparison.get("max_abs_diff") <= 5e-3
      and comparison.get("rmse") <= 1e-3
      and comparison.get("cosine") >= 0.99999
  )


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  selected_mode = parsed.get("selected_mode")
  modes = parsed.get("modes", {})
  selected = modes.get(selected_mode, {}) if isinstance(modes, dict) else {}
  attention = parsed.get("attention_parameters", {})
  metadata = parsed.get("metadata", {})
  tensors = parsed.get("tensors", {})
  history_vectors = parsed.get("history_vectors", {})
  oracle_vector = parsed.get("oracle_vector", {})
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("layer_index") == 3
      and parsed.get("prompt_case_id") == "short_math_001"
      and parsed.get("source_token_position") == 15
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and selected_mode in ("f32_source_payload", "fp16_kv_cache")
      and selected.get("passed") is True
      and comparison_passed(selected.get("comparison", {}))
      and attention.get("history_token_count") == TOKEN_COUNT
      and attention.get("head_dim") == 256
      and attention.get("q_head_count") == 16
      and attention.get("kv_head_count") == 2
      and attention.get("gqa_group") == 8
      and abs(attention.get("attention_scale", 0.0) - 0.0625) < 1e-12
      and metadata.get("ok") is True
      and metadata.get("full_attention_interval") == 4
      and metadata.get("head_count") == 16
      and metadata.get("head_count_kv") == 2
      and metadata.get("key_length") == 256
      and metadata.get("value_length") == 256
      and tensors.get("shape_ok") is True
      and history_vectors.get("q_rope", {}).get("count") == 4096
      and history_vectors.get("k_history", {}).get("count") == 8192
      and history_vectors.get("v_history", {}).get("count") == 8192
      and oracle_vector.get("count") == 4096
      and all(
          item.get("finite") is True and item.get("nonzero") is True
          for item in (
              history_vectors.get("q_rope", {}),
              history_vectors.get("k_history", {}),
              history_vectors.get("v_history", {}),
              oracle_vector,
              selected.get("native_vector", {}),
          )
      )
  )


def comparison_summary(parsed: dict[str, Any]) -> dict[str, Any]:
  selected_mode = parsed.get("selected_mode")
  selected = parsed.get("modes", {}).get(selected_mode, {})
  comparison = selected.get("comparison", {})
  return {
      "cosine": comparison.get("cosine"),
      "max_abs_diff": comparison.get("max_abs_diff"),
      "mean_abs_diff": comparison.get("mean_abs_diff"),
      "mismatch_count": comparison.get("mismatch_count"),
      "rmse": comparison.get("rmse"),
      "selected_mode": selected_mode,
  }


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_full_attn_core_compare"]
  comparison = state.get("comparison", {})
  rows = [
      ("engine_full_attn_core_compare_passed", payload["engine_full_attn_core_compare_passed"]),
      ("history_token_count", state.get("history_token_count")),
      ("staged_payload_count", state.get("staged_payload_count")),
      ("selected_mode", comparison.get("selected_mode")),
      ("max_abs_diff", comparison.get("max_abs_diff")),
      ("rmse", comparison.get("rmse")),
      ("cosine", comparison.get("cosine")),
      ("mismatch_count", comparison.get("mismatch_count")),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_full_attn_core_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_full_attn_core_compare"]
  comparison = state["comparison"]
  lines = [
      "# R1 Engine Full-Attention Core Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- history artifact: `{state['history_artifact']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- history token count: {state['history_token_count']}",
      f"- selected mode: `{comparison.get('selected_mode')}`",
      f"- max abs diff: {comparison.get('max_abs_diff')}",
      f"- rmse: {comparison.get('rmse')}",
      f"- cosine: {comparison.get('cosine')}",
      f"- full-attention core compare passed: `{str(payload['engine_full_attn_core_compare_passed']).lower()}`",
      "",
      "This artifact validates causal full attention from captured L3 K/V",
      "history to token-15 pregate attention state. It does not yet validate",
      "native KV-cache updates, native candidate JSONL, or token replay.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-full-attn-core-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-full-attn-core-compare-{stamp}"
  remote_payload_dir = f"{remote_dir}/history"
  history_json = args.history_json.resolve() if args.history_json else latest_history_json()
  ref = resolve_history(history_json)

  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "history")
      ),
      args.timeout_s,
  )
  source_transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      name: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for name in ref["payloads"]
  }
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      source_transfers.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    for name, payload in ref["payloads"].items():
      payload_transfers[name] = copy_to(
          args.host,
          payload["local_path"],
          f"{remote_payload_dir}/{payload['stage_name']}",
          args.timeout_s,
      )

  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in source_transfers)
      and all(item["returncode"] == 0 for item in payload_transfers.values())
  )
  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/full_attn_core_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-full-attn-core-compare')}",
  ])
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-full-attn-core-compare"),
      shlex.quote(args.model),
      shlex.quote(remote_payload_dir),
  ])
  compare = (
      run_target(args.host, compare_command, args.timeout_s)
      if build["returncode"] == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )

  parsed: dict[str, Any] = {}
  parse_error = None
  if compare.get("stdout"):
    try:
      parsed = json.loads(compare["stdout"])
    except json.JSONDecodeError as exc:
      parse_error = str(exc)

  passed = bool(parsed) and compare_passed(parsed, build, compare, args.model)
  state = {
      "attention_parameters": parsed.get("attention_parameters", {}) if parsed else {},
      "boundary_type": "full_attention_core",
      "comparison": comparison_summary(parsed) if parsed else {},
      "engine_stdout_schema_version": ENGINE_STDOUT_SCHEMA,
      "history_artifact": ref["history_artifact"],
      "history_capture_created_at": ref["created_at"],
      "history_json": ref["history_json"],
      "history_token_count": ref["history_token_count"],
      "layer_index": 3,
      "metadata": parsed.get("metadata", {}) if parsed else {},
      "modes": parsed.get("modes", {}) if parsed else {},
      "payloads": slim_payloads(ref["payloads"]),
      "source_prompt_case_id": ref["source_prompt_case_id"],
      "source_token_position": 15,
      "source_token_positions": ref["source_token_positions"],
      "staged_payload_count": len(ref["payloads"]),
      "target_build_returncode": build.get("returncode"),
      "target_compare_returncode": compare.get("returncode"),
      "tensors": parsed.get("tensors", {}) if parsed else {},
  }

  payload = {
      "created_at": created_at,
      "engine_full_attn_core_compare": state,
      "engine_full_attn_core_compare_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "parse_error": parse_error,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_build": build,
      "target_compare": compare,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "history_artifact": ref["history_artifact"],
      "host": args.host,
      "model_path": args.model,
      "payloads": slim_payloads(ref["payloads"]),
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-full-attn-core-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "payload_transfers": payload_transfers,
      "remote_dir": remote_dir,
      "remote_payload_dir": remote_payload_dir,
      "source_files": SOURCE_FILES,
      "source_transfers": source_transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(
      out_dir / "full-attn-core-stdout.json",
      parsed if parsed else {"parse_error": parse_error},
  )
  write_json(out_dir / "compare.json", payload)
  checks = [
      {"name": "history_capture_available", "pass": ref["history_token_count"] == TOKEN_COUNT},
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(
              item.get("returncode") == 0 for item in source_transfers
          ),
      },
      {
          "name": "history_payloads_transferred",
          "pass": all(item.get("returncode") == 0 for item in payload_transfers.values()),
      },
      {"name": "target_engine_full_attn_core_compare_built", "pass": build.get("returncode") == 0},
      {"name": "target_engine_full_attn_core_compare_ran", "pass": compare.get("returncode") == 0},
      {"name": "target_engine_full_attn_core_compare_output_parsed", "pass": bool(parsed)},
      {"name": "full_attention_core_matches_oracle_pregate", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "engine_full_attn_core_compare_passed": passed,
      "gate": "r1_engine_full_attn_core_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine full-attn core compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
