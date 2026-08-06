#!/usr/bin/env python3
"""Measure an exact-prompt OpenVINO denominator matrix with one resident load."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
DEFAULT_MATERIALIZATION = (
    ROOT / "output/r0-oracle-prompt-materialization-20260626T082201Z")
DEFAULT_OPENVINO_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
DEFAULT_OPENVINO_MODEL = Path("/home/intel/Qwen3.6-35B-A3B-ov")
DEFAULT_LLAMA_TOKENIZE = Path("/home/intel/llama-cpp/llama-b9518/llama-tokenize")
DEFAULT_GGUF_MODEL = Path(
    "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
DEFAULT_FILLER_DIR = ROOT / "output/r0-openvino-exact-filler-prompts-20260711"
DEFAULT_BUCKETS = (
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400, 131072)
CASE_SUFFIX = {
    1024: "001k",
    2048: "002k",
    4096: "004k",
    8192: "008k",
    16384: "016k",
    32768: "032k",
    65536: "064k",
    102400: "100k",
    131072: "128k",
    262144: "256k",
}


WORKER = r'''#!/usr/bin/env python3
import json
import sys
import time
import traceback
from pathlib import Path

import openvino_genai as ov_genai
from openvino import get_version


def scalar(metric):
  return float(metric.mean)


def main():
  config_path = Path(sys.argv[1])
  cfg = json.loads(config_path.read_text(encoding="utf-8"))
  scheduler = ov_genai.SchedulerConfig()
  scheduler.enable_prefix_caching = False
  scheduler.max_num_batched_tokens = sys.maxsize
  load_started = time.perf_counter()
  pipe = ov_genai.VLMPipeline(
      cfg["model"], cfg["device"], scheduler_config=scheduler,
      DYNAMIC_QUANTIZATION_GROUP_SIZE=256)
  load_wall_ms = (time.perf_counter() - load_started) * 1000.0
  tokenizer = pipe.get_tokenizer()
  generation = ov_genai.GenerationConfig()
  generation.max_new_tokens = cfg["output_tokens"]
  generation.ignore_eos = cfg["ignore_eos"]
  generation.apply_chat_template = False
  rows = []
  failures = []
  for spec in cfg["prompts"]:
    prompt_path = Path(spec["path"])
    prompt = prompt_path.read_text(encoding="utf-8")
    encoded = tokenizer.encode(prompt).input_ids
    observed = int(encoded.get_shape()[1])
    print(json.dumps({"event": "prompt", "case_id": spec["case_id"],
                      "observed_tokens": observed}), flush=True)
    try:
      for _ in range(cfg["num_warmup"]):
        pipe.generate(prompt, generation_config=generation)
      for iteration in range(1, cfg["num_iter"] + 1):
        wall_started = time.perf_counter()
        result = pipe.generate(prompt, generation_config=generation)
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        perf = result.perf_metrics
        ttft_ms = scalar(perf.get_ttft())
        tpot_ms = scalar(perf.get_tpot())
        actual_input_tokens = int(perf.get_num_input_tokens())
        row = {
            "bucket": spec["bucket"],
            "case_id": spec["case_id"],
            "decode_tokens_s": scalar(perf.get_throughput()),
            "detokenization_ms": scalar(perf.get_detokenization_duration()),
            "embeddings_ms": scalar(perf.get_prepare_embeddings_duration()),
            "generate_ms": scalar(perf.get_generate_duration()),
            "input_tokens": actual_input_tokens,
            "iteration": iteration,
            "load_ms": float(perf.get_load_time()),
            "output_tokens": int(perf.get_num_generated_tokens()),
            "prefill_tokens_s": actual_input_tokens / (ttft_ms / 1000.0),
            "prompt_file_sha256": spec["sha256"],
            "prompt_path": str(prompt_path),
            "prompt_set": spec["prompt_set"],
            "tokenization_ms": scalar(perf.get_tokenization_duration()),
            "tokenizer_input_tokens": observed,
            "tpot_ms": tpot_ms,
            "ttft_ms": ttft_ms,
            "wall_ms": wall_ms,
        }
        rows.append(row)
        print(json.dumps({"event": "row", **row}), flush=True)
    except Exception as exc:
      failure = {
          "bucket": spec["bucket"],
          "case_id": spec["case_id"],
          "error": repr(exc),
          "traceback": traceback.format_exc(),
      }
      failures.append(failure)
      print(json.dumps({"event": "failure", **failure}), flush=True)
  payload = {
      "apply_chat_template": False,
      "device": cfg["device"],
      "failures": failures,
      "load_wall_ms": load_wall_ms,
      "openvino_genai_version": ov_genai.__version__,
      "openvino_runtime_version": get_version(),
      "prefix_caching": False,
      "rows": rows,
  }
  Path(cfg["result_path"]).write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return 0 if not failures else 2


if __name__ == "__main__":
  raise SystemExit(main())
'''


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_buckets(value: str) -> tuple[int, ...]:
  buckets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
  unknown = sorted(set(buckets) - set(CASE_SUFFIX))
  if not buckets or unknown:
    raise argparse.ArgumentTypeError(f"invalid buckets: {unknown or value}")
  return buckets


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--materialization-dir", type=Path,
                      default=DEFAULT_MATERIALIZATION)
  parser.add_argument("--prompt-set",
                      choices=("prefill_shape", "sentinel", "filler", "both"),
                      default="prefill_shape")
  parser.add_argument("--buckets", type=parse_buckets,
                      default=DEFAULT_BUCKETS)
  parser.add_argument("--output-tokens", type=int, default=512)
  parser.add_argument("--num-warmup", type=int, default=1)
  parser.add_argument("--num-iter", type=int, default=3)
  parser.add_argument("--device", default="GPU")
  parser.add_argument(
      "--respect-eos", action="store_true",
      help="Allow EOS to shorten a row; fixed-length lanes ignore EOS by default.")
  parser.add_argument("--openvino-python", type=Path,
                      default=DEFAULT_OPENVINO_PYTHON)
  parser.add_argument("--model", type=Path, default=DEFAULT_OPENVINO_MODEL)
  parser.add_argument("--llama-tokenize", type=Path,
                      default=DEFAULT_LLAMA_TOKENIZE)
  parser.add_argument("--gguf-model", type=Path, default=DEFAULT_GGUF_MODEL)
  parser.add_argument("--filler-dir", type=Path, default=DEFAULT_FILLER_DIR)
  parser.add_argument("--timeout-s", type=int, default=7200)
  parser.add_argument("--plan-only", action="store_true")
  parser.add_argument("--out-dir", type=Path)
  return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: expected object")
      rows.append(row)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state() -> dict[str, Any]:
  def command(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""
  dirty = command("status", "--porcelain")
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def parse_llama_count(stdout: str) -> int | None:
  for line in reversed(stdout.splitlines()):
    prefix = "Total number of tokens:"
    text = line.strip()
    if text.startswith(prefix):
      try:
        return int(text[len(prefix):].strip())
      except ValueError:
        return None
  return None


def select_filler_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
  args.filler_dir.mkdir(parents=True, exist_ok=True)
  selected = []
  for bucket in args.buckets:
    case_id = f"filler_{CASE_SUFFIX[bucket]}"
    path = args.filler_dir / f"{case_id}.txt"
    path.write_text("a" + (" a" * (bucket - 1)), encoding="utf-8")
    count = subprocess.run(
        [str(args.llama_tokenize), "-m", str(args.gguf_model),
         "-f", str(path), "--show-count", "--log-disable"],
        check=False, capture_output=True, text=True, timeout=300)
    observed = parse_llama_count(count.stdout)
    if count.returncode != 0 or observed != bucket:
      raise SystemExit(
          f"{case_id}: GGUF tokenizer count {observed}, expected {bucket}")
    selected.append({
        "bucket": bucket,
        "case_id": case_id,
        "expected_tokens": bucket,
        "gguf_token_count": observed,
        "path": str(path.resolve()),
        "prompt_set": "filler",
        "sha256": sha256_file(path),
    })
  return selected


def select_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
  if args.prompt_set == "filler":
    return select_filler_prompts(args)
  rows = load_jsonl(args.materialization_dir / "materialized-prompts.jsonl")
  by_case = {row.get("case_id"): row for row in rows}
  prefixes = ("prefill_shape", "sentinel") if args.prompt_set == "both" else (
      args.prompt_set,)
  selected = []
  for bucket in args.buckets:
    for prefix in prefixes:
      case_id = f"{prefix}_{CASE_SUFFIX[bucket]}"
      source = by_case.get(case_id)
      if source is None:
        raise SystemExit(f"materialization missing {case_id}")
      path = ROOT / str(source["materialized_prompt_path"])
      if not path.is_file():
        raise SystemExit(f"prompt missing: {path}")
      if source.get("observed_prompt_tokens") != bucket:
        raise SystemExit(f"{case_id}: source token count mismatch")
      digest = sha256_file(path)
      if digest != source.get("prompt_file_sha256"):
        raise SystemExit(f"{case_id}: prompt digest mismatch")
      selected.append({
          "bucket": bucket,
          "case_id": case_id,
          "expected_tokens": bucket,
          "path": str(path),
          "prompt_set": prefix,
          "sha256": digest,
      })
  return selected


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def median(rows: list[dict[str, Any]], key: str) -> float | None:
  values = [float(row[key]) for row in rows if finite(row.get(key))]
  return statistics.median(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  grouped: dict[int, list[dict[str, Any]]] = {}
  for row in rows:
    grouped.setdefault(int(row["bucket"]), []).append(row)
  result = []
  for bucket, bucket_rows in sorted(grouped.items()):
    result.append({
        "bucket": bucket,
        "decode_tokens_s_median": median(bucket_rows, "decode_tokens_s"),
        "prefill_tokens_s_median": median(bucket_rows, "prefill_tokens_s"),
        "row_count": len(bucket_rows),
        "tpot_ms_median": median(bucket_rows, "tpot_ms"),
        "ttft_ms_median": median(bucket_rows, "ttft_ms"),
    })
  return result


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")


def build_summary(payload: dict[str, Any]) -> str:
  lines = [
      "# OpenVINO exact-prompt denominator matrix",
      "",
      f"- required_checks_passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- prompt set: `{payload['config']['prompt_set']}`",
      f"- output tokens: `{payload['config']['output_tokens']}`",
      f"- ignore EOS: `{str(payload['config']['ignore_eos']).lower()}`",
      f"- warmup / measured per prompt: `{payload['config']['num_warmup']} / {payload['config']['num_iter']}`",
      f"- resident model load: `{payload['worker'].get('load_wall_ms')}` ms",
      "",
      "| bucket | rows | prefill median tok/s | decode median tok/s | TTFT median ms | TPOT median ms |",
      "|---:|---:|---:|---:|---:|---:|",
  ]
  for row in payload["bucket_summaries"]:
    lines.append(
        f"| {row['bucket']} | {row['row_count']} | "
        f"{row['prefill_tokens_s_median']:.6f} | "
        f"{row['decode_tokens_s_median']:.6f} | "
        f"{row['ttft_ms_median']:.3f} | {row['tpot_ms_median']:.3f} |")
  lines.extend([
      "",
      "This is an OpenVINO denominator artifact, not a native speedup or",
      "correctness claim.",
      "",
  ])
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  if args.output_tokens < 1 or args.num_warmup < 0 or args.num_iter < 1:
    raise SystemExit("invalid output-token/warmup/iteration count")
  prompts = select_prompts(args)
  plan = {
      "apply_chat_template": False,
      "buckets": list(args.buckets),
      "device": args.device,
      "ignore_eos": not args.respect_eos,
      "model": str(args.model.resolve()),
      "num_iter": args.num_iter,
      "num_warmup": args.num_warmup,
      "output_tokens": args.output_tokens,
      "prompt_set": args.prompt_set,
      "prompts": prompts,
  }
  if args.plan_only:
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0

  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (args.out_dir or ROOT / f"output/r0-openvino-denominator-matrix-{stamp}").resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  result_path = out_dir / "worker-result.json"
  worker_path = raw_dir / "openvino-matrix-worker.py"
  worker_path.write_text(WORKER, encoding="utf-8")
  worker_config = {**plan, "result_path": str(result_path)}
  config_path = out_dir / "worker-config.json"
  write_json(config_path, worker_config)
  command = [str(args.openvino_python), str(worker_path), str(config_path)]
  try:
    process = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=args.timeout_s)
    timed_out = False
  except subprocess.TimeoutExpired as exc:
    process = subprocess.CompletedProcess(
        command, 124,
        exc.stdout if isinstance(exc.stdout, str) else "",
        exc.stderr if isinstance(exc.stderr, str) else "")
    timed_out = True
  (raw_dir / "worker.stdout").write_text(process.stdout, encoding="utf-8")
  (raw_dir / "worker.stderr").write_text(process.stderr, encoding="utf-8")
  worker: dict[str, Any] = {}
  if result_path.is_file():
    worker = json.loads(result_path.read_text(encoding="utf-8"))
  rows = worker.get("rows", []) if isinstance(worker, dict) else []
  expected_rows = len(prompts) * args.num_iter
  prompt_by_case = {row["case_id"]: row for row in prompts}
  exact_counts = all(
      row.get("input_tokens") == prompt_by_case.get(row.get("case_id"), {}).get(
          "expected_tokens") for row in rows)
  finite_rows = all(
      all(finite(row.get(key)) for key in (
          "decode_tokens_s", "prefill_tokens_s", "tpot_ms", "ttft_ms"))
      for row in rows)
  output_counts = all(row.get("output_tokens") == args.output_tokens for row in rows)
  checks = [
      {"name": "worker_returncode", "pass": process.returncode == 0,
       "value": process.returncode},
      {"name": "worker_not_timed_out", "pass": not timed_out},
      {"name": "all_prompt_token_counts_exact", "pass": exact_counts},
      {"name": "all_metric_rows_finite", "pass": finite_rows},
      {"name": "all_output_counts_match", "pass": output_counts},
      {"name": "all_expected_rows_present", "pass": len(rows) == expected_rows,
       "expected": expected_rows, "observed": len(rows)},
      {"name": "prefix_caching_disabled",
       "pass": worker.get("prefix_caching") is False},
      {"name": "chat_template_disabled",
       "pass": worker.get("apply_chat_template") is False},
      {"name": "no_worker_failures", "pass": not worker.get("failures")},
  ]
  required = all(bool(check["pass"]) for check in checks)
  route_label = "diagnostic" if required else "rejected"
  bucket_summaries = summarize(rows)
  payload = {
      "bucket_summaries": bucket_summaries,
      "config": plan,
      "created_at": created_at,
      "git": git_state(),
      "host": {"hostname": platform.node(), "kernel": platform.release()},
      "required_checks_passed": required,
      "route_label": route_label,
      "schema_version": "intel-qwen36-r0-openvino-denominator-matrix-v0",
      "worker": worker,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": payload["git"],
      "route_label": route_label,
      "schema_version": payload["schema_version"],
      "tool": "tools/intel-qwen36-r0-openvino-denominator-matrix.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "same_host_openvino_exact_prompt_denominator_matrix",
      "required_checks_passed": required,
      "route_label": route_label,
      "token_correctness": "not_checked_denominator_only",
      "workstream": WORKSTREAM,
  })
  smooth_rows = []
  for previous, current in zip(bucket_summaries, bucket_summaries[1:]):
    smooth_rows.append({
        "from_bucket": previous["bucket"],
        "to_bucket": current["bucket"],
        "tpot_ratio": (
            current["tpot_ms_median"] / previous["tpot_ms_median"]),
    })
  write_json(out_dir / "smoothness.json", {
      "applicable": True,
      "adjacent_tpot": smooth_rows,
      "notes": "OpenVINO denominator slope only; not native product smoothness",
      "route_label": route_label,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps({
          **row,
          "cache_state": "cold_no_prefix_model_resident",
          "phase": "openvino_denominator",
          "route_label": route_label,
      }, sort_keys=True) + "\n")
  write_json(out_dir / "matrix.json", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required,
      "row_count": len(rows),
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
