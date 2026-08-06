#!/usr/bin/env python3
"""Audit the stock-F16 versus requested-U8 product KV-cache route.

This is a correctness-first route gate, not a performance benchmark.  It runs
the real text-only product path in isolated workers and stops the U8 route at
the first exact greedy-token failure.  Long-context timing is deliberately not
spent when the short correctness guard already fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-kv-precision-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_CONTRACT = (
    ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json")
PROMPT = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/prefill_shape_002k.txt")
EXPECTED_INPUT_TOKENS = 2048
OUTPUT_TOKENS = 20


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
  parser.add_argument("--model-contract", type=Path, default=MODEL_CONTRACT)
  parser.add_argument("--prompt", type=Path, default=PROMPT)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.out_dir is None and args.worker_config is None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-kv-precision-{stamp}"
  if args.timeout_s <= 0:
    parser.error("timeout-s must be positive")
  return args


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def git_state(out_dir: Path) -> dict[str, Any]:
  def git(*args: str) -> str:
    run = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = git("status", "--porcelain").splitlines()
  try:
    relative_out = str(out_dir.relative_to(ROOT))
  except ValueError:
    relative_out = ""
  dirty = [row for row in dirty if not relative_out or relative_out not in row]
  return {
      "commit": git("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def locked_file_rows(
    model_dir: Path, contract: dict[str, Any],
) -> list[dict[str, Any]]:
  rows = []
  locked = contract.get("product_model", {}).get("locked_files", {})
  for name, expected in sorted(locked.items()):
    path = model_dir / name
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    rows.append({
        "bytes": size,
        "expected_bytes": expected.get("bytes"),
        "expected_sha256": expected.get("sha256"),
        "exists": exists,
        "name": name,
        "pass": (
            exists and size == expected.get("bytes") and
            digest == expected.get("sha256")),
        "path": str(path),
        "sha256": digest,
    })
  return rows


def scalar(metric: Any) -> float:
  return float(metric.mean)


def worker_main(config_path: Path) -> int:
  if Path(sys.prefix).resolve() != OV_PYTHON.parent.parent.resolve():
    raise SystemExit(f"worker requires {OV_PYTHON}, observed {sys.executable}")

  import numpy as np
  import openvino as ov
  import openvino_genai as ov_genai

  cfg = load_json(config_path)
  model_dir = Path(cfg["model_dir"])
  prompt_path = Path(cfg["prompt"])
  prompt = prompt_path.read_text(encoding="utf-8")
  scheduler = ov_genai.SchedulerConfig()
  scheduler.enable_prefix_caching = False
  scheduler.max_num_batched_tokens = sys.maxsize
  properties: dict[str, Any] = {
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
      "scheduler_config": scheduler,
  }
  requested_precision = cfg.get("kv_cache_precision")
  if requested_precision is not None:
    properties["KV_CACHE_PRECISION"] = requested_precision

  source = ov.Core().read_model(
      str(model_dir / "openvino_language_model.xml"))
  embedded_precision = str(
      source.get_rt_info(["runtime_options", "KV_CACHE_PRECISION"]).value)

  started = dt.datetime.now(dt.timezone.utc).isoformat()
  pipe = ov_genai.VLMPipeline(
      str(model_dir), cfg["device"], **properties)
  tokenizer = pipe.get_tokenizer()
  prompt_ids = np.asarray(tokenizer.encode(prompt).input_ids.data).reshape(-1)

  warmup = ov_genai.GenerationConfig()
  warmup.max_new_tokens = 4
  warmup.ignore_eos = True
  warmup.apply_chat_template = False
  warmup.do_sample = False
  warmup.num_beams = 1
  pipe.generate(prompt, generation_config=warmup)

  generation = ov_genai.GenerationConfig()
  generation.max_new_tokens = int(cfg["output_tokens"])
  generation.ignore_eos = True
  generation.apply_chat_template = False
  generation.do_sample = False
  generation.num_beams = 1
  result = pipe.generate(prompt, generation_config=generation)
  perf = result.perf_metrics
  decoded = str(result.texts[0])
  decoded_ids = np.asarray(
      tokenizer.encode(decoded).input_ids.data).reshape(-1).astype(np.int64)
  payload = {
      "apply_chat_template": False,
      "decode_tokens_s": scalar(perf.get_throughput()),
      "decoded_text": decoded,
      "decoded_text_sha256": hashlib.sha256(
          decoded.encode("utf-8")).hexdigest(),
      "decoded_token_ids": [int(value) for value in decoded_ids],
      "device": cfg["device"],
      "do_sample": False,
      "embedded_model_kv_cache_precision": embedded_precision,
      "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "input_tokens": int(perf.get_num_input_tokens()),
      "kv_cache_precision_override": requested_precision,
      "mode": cfg["mode"],
      "num_beams": 1,
      "openvino_genai_version": ov_genai.__version__,
      "openvino_runtime_version": ov.get_version(),
      "output_tokens": int(perf.get_num_generated_tokens()),
      "prefix_caching": False,
      "prompt_file_sha256": sha256_file(prompt_path),
      "prompt_token_count": int(prompt_ids.size),
      "prompt_token_ids_sha256": hashlib.sha256(
          np.asarray(prompt_ids, dtype="<u4").tobytes()).hexdigest(),
      "requested_output_tokens": int(cfg["output_tokens"]),
      "started_at": started,
      "tpot_ms": scalar(perf.get_tpot()),
      "ttft_ms": scalar(perf.get_ttft()),
  }
  write_json(Path(cfg["result_path"]), payload)
  print(json.dumps({
      "event": "worker_complete",
      "mode": cfg["mode"],
      "text_sha256": payload["decoded_text_sha256"],
      "tpot_ms": payload["tpot_ms"],
  }, sort_keys=True), flush=True)
  return 0


def run_worker(
    name: str, raw: Path, base: dict[str, Any], timeout_s: int,
    kv_cache_precision: str | None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
  worker_dir = raw / name
  worker_dir.mkdir()
  result_path = worker_dir / "result.json"
  config = {
      **base,
      "kv_cache_precision": kv_cache_precision,
      "mode": name,
      "result_path": str(result_path),
  }
  config_path = worker_dir / "config.json"
  write_json(config_path, config)
  command = [
      str(OV_PYTHON), str(Path(__file__).resolve()),
      "--worker-config", str(config_path),
  ]
  env = os.environ.copy()
  env.update({
      "NEO_CACHE_DIR": str(worker_dir / "neo-cache"),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  (worker_dir / "neo-cache").mkdir()
  try:
    run = subprocess.run(
        command, cwd=ROOT, env=env, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
  except subprocess.TimeoutExpired as exc:
    run = subprocess.CompletedProcess(
        command, 124, str(exc.stdout or ""), str(exc.stderr or ""))
  (worker_dir / "stdout").write_text(run.stdout, encoding="utf-8")
  (worker_dir / "stderr").write_text(run.stderr, encoding="utf-8")
  write_json(worker_dir / "command.json", {
      "command": command,
      "environment": {key: env[key] for key in (
          "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": run.returncode,
  })
  result = load_json(result_path) if result_path.is_file() else {}
  return run, result


def main() -> int:
  args = parse_args()
  if args.worker_config is not None:
    return worker_main(args.worker_config.resolve())

  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = [args.model_contract, args.prompt, OV_PYTHON]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  git = git_state(out)
  contract = load_json(args.model_contract)
  locked_files = locked_file_rows(args.model_dir, contract)
  base = {
      "device": args.device,
      "model_dir": str(args.model_dir.resolve()),
      "output_tokens": OUTPUT_TOKENS,
      "prompt": str(args.prompt.resolve()),
  }
  stock_run, stock = run_worker(
      "stock", raw, base, args.timeout_s, None)
  primary_run, primary = run_worker(
      "u8-primary", raw, base, args.timeout_s, "u8")
  confirm_run, confirm = run_worker(
      "u8-confirm", raw, base, args.timeout_s, "u8")

  runtime = contract["runtime_contract"]["baseline"]
  results = (stock, primary, confirm)
  exact_counts = all(
      row.get("input_tokens") == EXPECTED_INPUT_TOKENS and
      row.get("prompt_token_count") == EXPECTED_INPUT_TOKENS and
      row.get("output_tokens") == OUTPUT_TOKENS
      for row in results)
  stock_hash = stock.get("decoded_text_sha256")
  primary_hash = primary.get("decoded_text_sha256")
  confirm_hash = confirm.get("decoded_text_sha256")
  both_u8_mismatch = bool(
      stock_hash and primary_hash and confirm_hash and
      primary_hash != stock_hash and confirm_hash != stock_hash)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("all_locked_model_files_match_contract",
            bool(locked_files) and all(row["pass"] for row in locked_files),
            rows=locked_files),
      check("isolated_stock_and_u8_workers_complete",
            stock_run.returncode == primary_run.returncode ==
            confirm_run.returncode == 0,
            returncodes=[stock_run.returncode, primary_run.returncode,
                         confirm_run.returncode]),
      check("workers_use_locked_runtime",
            all(row.get("openvino_runtime_version") ==
                runtime["openvino_runtime_version"] and
                row.get("openvino_genai_version") ==
                runtime["openvino_genai_version"] for row in results),
            versions=[{
                "runtime": row.get("openvino_runtime_version"),
                "genai": row.get("openvino_genai_version"),
            } for row in results]),
      check("locked_model_embeds_f16_kv_cache",
            all(row.get("embedded_model_kv_cache_precision") == "f16"
                for row in results),
            observed=[row.get("embedded_model_kv_cache_precision")
                      for row in results]),
      check("exact_2k_input_and_20_output_counts", exact_counts,
            observed=[{
                "input": row.get("input_tokens"),
                "output": row.get("output_tokens"),
                "prompt": row.get("prompt_token_count"),
            } for row in results]),
      check("cold_prefix_and_chat_template_controls_hold",
            all(row.get("prefix_caching") is False and
                row.get("apply_chat_template") is False and
                row.get("do_sample") is False and row.get("num_beams") == 1
                for row in results)),
      check("u8_override_executes_twice",
            primary.get("kv_cache_precision_override") == "u8" and
            confirm.get("kv_cache_precision_override") == "u8"),
      check("u8_override_fails_exact_greedy_text_contract_twice",
            both_u8_mismatch,
            stock_sha256=stock_hash, primary_sha256=primary_hash,
            confirm_sha256=confirm_hash),
  ]
  passed = all(row["pass"] for row in checks)
  correctness = {
      "checks": checks,
      "claim": "reject_simple_u8_kv_property_before_long_context_timing",
      "required_checks_passed": passed,
      "schema_version": SCHEMA,
      "u8_product_correctness_pass": not both_u8_mismatch,
      "workstream": WORKSTREAM,
  }
  performance = {
      "claim_boundary": "diagnostic_only_correctness_failed",
      "product_speedup_claim": False,
      "rows": [{
          "decode_tokens_s": row.get("decode_tokens_s"),
          "mode": row.get("mode"),
          "tpot_ms": row.get("tpot_ms"),
          "ttft_ms": row.get("ttft_ms"),
      } for row in results],
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
  }
  manifest = {
      "command": sys.argv,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "model_contract": str(args.model_contract.resolve()),
      "model_dir": str(args.model_dir.resolve()),
      "product_speedup_claim": False,
      "prompt": str(args.prompt.resolve()),
      "required_checks_passed": passed,
      "schema_version": SCHEMA,
      "tool": str(Path(__file__).resolve()),
      "workstream": WORKSTREAM,
  }
  smoothness = {
      "not_applicable": True,
      "reason": "the simple U8 route fails the exact 2k greedy guard",
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
  }
  write_json(out / "correctness.json", correctness)
  write_json(out / "manifest.json", manifest)
  write_json(out / "performance.json", performance)
  write_json(out / "smoothness.json", smoothness)
  write_jsonl(out / "metrics.jsonl", [
      {"kind": "correctness", **correctness},
      {"kind": "performance", **performance},
  ])
  summary = [
      "# OpenVINO KV-cache precision gate",
      "",
      f"- required checks: **{'PASS' if passed else 'FAIL'}**",
      "- locked model KV-cache precision: `f16`",
      f"- stock text SHA256: `{stock_hash}`",
      f"- U8 primary text SHA256: `{primary_hash}`",
      f"- U8 confirm text SHA256: `{confirm_hash}`",
      f"- U8 exact product correctness: `{not both_u8_mismatch}`",
      "- product speedup claim: `false`",
      "",
      "The candidate U8 property is rejected before long-context timing when "
      "both isolated U8 workers differ from the deterministic stock output. "
      "Timing rows are diagnostics only.",
      "",
  ]
  (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out),
      "required_checks_passed": passed,
      "u8_product_correctness_pass": not both_u8_mismatch,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
