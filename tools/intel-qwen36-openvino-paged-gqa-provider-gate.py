#!/usr/bin/env python3
"""Gate the exact-128k product paged-GQA provider source and event rate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-paged-gqa-provider-gate-v0"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OV_MODEL = Path("/home/intel/Qwen3.6-35B-A3B-ov")
OV_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
OV_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
PROMPT_128K = ROOT / (
    "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/prefill_shape_128k.txt")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
TRACE_TARGET = "iq36-opencl-dispatch-trace"
TRACE_LIBRARY = BUILD_DIR / "iq36-opencl-dispatch-trace.so"
OCLOC = Path("/usr/bin/ocloc")
ATTENTION_CAP_MS = 28.250
NOISE_MAX = 0.005


WORKER = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import openvino_genai as ov_genai
from openvino import get_version


def scalar(metric):
  return float(metric.mean)


def main():
  cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  scheduler = ov_genai.SchedulerConfig()
  scheduler.enable_prefix_caching = False
  scheduler.max_num_batched_tokens = sys.maxsize
  load_started = time.perf_counter()
  pipe = ov_genai.VLMPipeline(
      cfg["model"], cfg["device"], scheduler_config=scheduler,
      DYNAMIC_QUANTIZATION_GROUP_SIZE=256, PERF_COUNT=True)
  load_wall_ms = (time.perf_counter() - load_started) * 1000.0
  prompt = Path(cfg["prompt_path"]).read_text(encoding="utf-8")
  tokenizer = pipe.get_tokenizer()
  tokenizer_input_tokens = int(tokenizer.encode(prompt).input_ids.get_shape()[1])
  generation = ov_genai.GenerationConfig()
  generation.max_new_tokens = int(cfg["output_tokens"])
  generation.ignore_eos = True
  generation.apply_chat_template = False
  Path(cfg["trace_marker"]).write_text(
      f"input{tokenizer_input_tokens}_output{cfg['output_tokens']}\n",
      encoding="utf-8")
  started = time.perf_counter()
  result = pipe.generate(prompt, generation_config=generation)
  wall_ms = (time.perf_counter() - started) * 1000.0
  perf = result.perf_metrics
  payload = {
      "apply_chat_template": False,
      "decode_tokens_s": scalar(perf.get_throughput()),
      "device": cfg["device"],
      "generated_tokens": int(perf.get_num_generated_tokens()),
      "ignore_eos": True,
      "input_tokens": int(perf.get_num_input_tokens()),
      "load_wall_ms": load_wall_ms,
      "openvino_genai_version": ov_genai.__version__,
      "openvino_runtime_version": get_version(),
      "prefix_caching": False,
      "prefill_tokens_s": int(perf.get_num_input_tokens()) / (
          scalar(perf.get_ttft()) / 1000.0),
      "scheduler": {
          "enable_prefix_caching": False,
          "max_num_batched_tokens": sys.maxsize,
      },
      "tokenizer_input_tokens": tokenizer_input_tokens,
      "tpot_ms": scalar(perf.get_tpot()),
      "ttft_ms": scalar(perf.get_ttft()),
      "wall_ms": wall_ms,
  }
  Path(cfg["result_path"]).write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  print(json.dumps({
      "event": "complete", "input_tokens": payload["input_tokens"],
      "generated_tokens": payload["generated_tokens"],
      "tpot_ms": payload["tpot_ms"], "wall_ms": wall_ms,
  }), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
'''


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--prompt-path", type=Path, default=PROMPT_128K)
  parser.add_argument("--expected-input-tokens", type=int, default=131072)
  parser.add_argument("--output-tokens", type=int, default=4)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--model", type=Path, default=OV_MODEL)
  parser.add_argument("--openvino-python", type=Path, default=OV_PYTHON)
  parser.add_argument("--timeout-s", type=int, default=1800)
  return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_state(out_dir: Path) -> dict[str, Any]:
  def command(*args: str) -> str:
    run = subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return run.stdout.strip() if run.returncode == 0 else ""

  dirty = command("status", "--porcelain").splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [line for line in dirty if not out_rel or out_rel not in line]
  return {
      "commit": command("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def build_trace(raw: Path) -> tuple[bool, dict[str, Any]]:
  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ]
  configure = subprocess.run(
      configure_command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TRACE_TARGET, "-j8"]
  build = subprocess.run(
      build_command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=600)
  payload = {
      "build": {"command": build_command, "returncode": build.returncode,
                "stderr": build.stderr, "stdout": build.stdout},
      "configure": {"command": configure_command,
                    "returncode": configure.returncode,
                    "stderr": configure.stderr, "stdout": configure.stdout},
      "library": str(TRACE_LIBRARY),
  }
  write_json(raw / "trace-build.json", payload)
  return bool(
      configure.returncode == 0 and build.returncode == 0
      and TRACE_LIBRARY.is_file()), payload


def disassemble(binary: Path, dump: Path) -> dict[str, Any]:
  dump.mkdir(parents=True, exist_ok=False)
  command = [str(OCLOC), "disasm", "-file", str(binary), "-dump", str(dump)]
  run = subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=300)
  write_json(dump / "command.json", {
      "command": command, "returncode": run.returncode,
      "stderr": run.stderr, "stdout": run.stdout,
  })
  ze_info = dump / ".ze_info"
  return {
      "returncode": run.returncode,
      "ze_info": (ze_info.read_text(errors="replace")
                  if ze_info.is_file() else ""),
  }


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  cache = raw / "neo-cache"
  raw.mkdir(parents=True, exist_ok=False)
  cache.mkdir()
  git = git_state(out_dir)
  trace_build_ok, _ = build_trace(raw)

  worker_path = raw / "openvino-paged-gqa-worker.py"
  worker_path.write_text(WORKER)
  worker_result_path = raw / "worker-result.json"
  config = {
      "device": args.device,
      "model": str(args.model.resolve()),
      "output_tokens": args.output_tokens,
      "prompt_path": str(args.prompt_path.resolve()),
      "result_path": str(worker_result_path),
      "trace_marker": str(raw / "trace-active"),
  }
  config_path = raw / "worker-config.json"
  write_json(config_path, config)
  command = [str(args.openvino_python), str(worker_path), str(config_path)]
  env = os.environ.copy()
  env.update({
      "IQ36_OPENCL_TRACE_FILTER": "single_token",
      "IQ36_OPENCL_TRACE_MARKER": str(raw / "trace-active"),
      "IQ36_OPENCL_TRACE_PATH": str(raw / "dispatch-trace.jsonl"),
      "IQ36_OPENCL_TRACE_TIMING": "1",
      "LD_AUDIT": str(TRACE_LIBRARY),
      "NEO_CACHE_DIR": str(cache),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024 * 1024 * 1024),
      "NEO_CACHE_PERSISTENT": "1",
  })
  worker = (
      subprocess.run(
          command, cwd=ROOT, env=env, check=False, capture_output=True,
          text=True, encoding="utf-8", errors="replace",
          timeout=args.timeout_s)
      if trace_build_ok else subprocess.CompletedProcess(
          command, 1, "", "dispatch trace build failed"))
  (raw / "worker.stdout").write_text(worker.stdout)
  (raw / "worker.stderr").write_text(worker.stderr)
  write_json(raw / "worker-command.json", {
      "command": command,
      "environment": {key: env[key] for key in (
          "IQ36_OPENCL_TRACE_FILTER", "IQ36_OPENCL_TRACE_MARKER",
          "IQ36_OPENCL_TRACE_PATH", "IQ36_OPENCL_TRACE_TIMING", "LD_AUDIT",
          "NEO_CACHE_DIR", "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "returncode": worker.returncode,
  })
  worker_result = (
      json.loads(worker_result_path.read_text())
      if worker_result_path.is_file() else {})
  trace_path = raw / "dispatch-trace.jsonl"
  trace_rows = []
  if trace_path.is_file():
    for line in trace_path.read_text().splitlines():
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(row, dict): trace_rows.append(row)

  captures = []
  for binary in sorted(cache.rglob("*.cl_cache")):
    data = binary.read_bytes()
    if b"paged_attention_opt__single_token" not in data:
      continue
    digest = sha256(binary)
    disasm = disassemble(binary, raw / "paged-disassembly" / digest[:16])
    names = sorted(set(
        token.decode("ascii", errors="ignore")
        for token in re.findall(
            rb"paged_attention_opt__[A-Za-z0-9_]+", data)))
    captures.append({
        "disassembly_returncode": disasm["returncode"],
        "kernel_names": names,
        "relative_path": str(binary.relative_to(cache)),
        "sha256": digest,
        "size_bytes": binary.stat().st_size,
        "ze_info_has_all_names": all(name in disasm["ze_info"] for name in names),
    })

  main_rows = [row for row in trace_rows if "finalization" not in row.get("kernel", "")]
  final_rows = [row for row in trace_rows if "finalization" in row.get("kernel", "")]
  trace_shape_ok = bool(len(trace_rows) == 60 and len(main_rows) == 30
                        and len(final_rows) == 30)
  layers = []
  pairing_ok = trace_shape_ok
  if trace_shape_ok:
    for index in range(0, len(trace_rows), 2):
      main = trace_rows[index]
      final = trace_rows[index + 1]
      main_name = str(main.get("kernel", ""))
      final_name = str(final.get("kernel", ""))
      main_match = re.search(r"(\d+__sa)$", main_name)
      final_match = re.search(r"(\d+__sa)$", final_name)
      pair_ok = bool(
          "finalization" not in main_name
          and "finalization" in final_name
          and main_match is not None and final_match is not None
          and main_match.group(1) == final_match.group(1)
          and main.get("status") == 0 and final.get("status") == 0
          and main.get("timing_status") == 0 and final.get("timing_status") == 0)
      pairing_ok = pairing_ok and pair_ok
      layers.append({
          "final_kernel": final_name,
          "final_us": float(final.get("duration_ns", 0)) / 1000.0,
          "main_kernel": main_name,
          "main_us": float(main.get("duration_ns", 0)) / 1000.0,
          "pair_ok": pair_ok,
          "total_us": (
              float(main.get("duration_ns", 0))
              + float(final.get("duration_ns", 0))) / 1000.0,
      })
  token_rows = []
  if pairing_ok and len(layers) == 30:
    for token_index in range(3):
      token_layers = layers[token_index * 10:(token_index + 1) * 10]
      token_rows.append({
          "attention_sum_ms": sum(row["total_us"] for row in token_layers) / 1000.0,
          "layer_max_us": max(row["total_us"] for row in token_layers),
          "layer_median_us": statistics.median(
              row["total_us"] for row in token_layers),
          "layer_min_us": min(row["total_us"] for row in token_layers),
          "token_index": token_index,
      })
  repeat_ms = token_rows[1]["attention_sum_ms"] if len(token_rows) == 3 else 1e9
  confirm_ms = token_rows[2]["attention_sum_ms"] if len(token_rows) == 3 else 1e9
  spread = abs(repeat_ms - confirm_ms) / min(repeat_ms, confirm_ms)
  rate_pass = bool(
      repeat_ms <= ATTENTION_CAP_MS and confirm_ms <= ATTENTION_CAP_MS
      and spread <= NOISE_MAX)
  captured_names = {
      name for capture in captures for name in capture["kernel_names"]}
  traced_names = {str(row.get("kernel", "")) for row in trace_rows}
  mapping_pass = bool(
      captures and traced_names and traced_names <= captured_names
      and all(row["disassembly_returncode"] == 0
              and row["ze_info_has_all_names"] for row in captures))
  metadata_pass = bool(
      pairing_ok and all(row.get("args") for row in trace_rows)
      and all(row.get("global_size") and row.get("local_size")
              for row in trace_rows)
      and all(int(row.get("duration_ns", 0)) > 0 for row in trace_rows))
  shape_pass = bool(
      args.expected_input_tokens == 131072 and args.output_tokens == 4
      and args.prompt_path.resolve() == PROMPT_128K.resolve()
      and worker_result.get("tokenizer_input_tokens") == 131072
      and worker_result.get("input_tokens") == 131072
      and worker_result.get("generated_tokens") == 4)
  source_pass = bool(
      OV_SOURCE.is_dir() and args.model.resolve() == OV_MODEL.resolve()
      and args.openvino_python.resolve() == OV_PYTHON.resolve())
  checks = [
      {"name": "repository_clean_at_gate", "pass": not git["dirty"],
       "dirty_paths": git["dirty_paths"]},
      {"name": "pinned_provider_source", "pass": source_pass,
       "openvino_commit": OV_COMMIT},
      {"name": "dispatch_trace_build", "pass": trace_build_ok},
      {"name": "worker_execution", "pass": worker.returncode == 0},
      {"name": "exact_product_shape", "pass": shape_pass},
      {"name": "paged_dispatch_shape_and_pairing", "pass": pairing_ok},
      {"name": "paged_binary_dispatch_mapping", "pass": mapping_pass},
      {"name": "paged_dispatch_metadata", "pass": metadata_pass},
      {"name": "paged_attention_repeat_confirm_rate", "pass": rate_pass},
  ]
  required = all(bool(check["pass"]) for check in checks)
  result = {
      "attention_cap_ms": ATTENTION_CAP_MS,
      "binary_captures": captures,
      "confirm_attention_ms": confirm_ms,
      "layer_rows": layers,
      "noise_max": NOISE_MAX,
      "paired_spread": spread,
      "rate_pass": rate_pass,
      "repeat_attention_ms": repeat_ms,
      "token_rows": token_rows,
      "trace_kernel_names": sorted(traced_names),
      "trace_rows": len(trace_rows),
      "worker": worker_result,
  }
  payload = {
      "checks": checks,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "required_checks_passed": required,
      "result": result,
      "route_label": "provider_source_admitted" if required else "rejected",
      "schema_version": SCHEMA,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"], "git": git,
      "prompt": {"path": str(args.prompt_path), "sha256": sha256(args.prompt_path)},
      "required_checks_passed": required,
      "route_label": payload["route_label"], "schema_version": SCHEMA,
      "tool": str(Path(__file__).relative_to(ROOT)), "workstream": WORKSTREAM,
  })
  write_json(out_dir / "correctness.json", {
      "applicable": False,
      "reason": "provider source/rate gate; native replay numeric proof remains open",
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
  })
  with (out_dir / "metrics.jsonl").open("w") as handle:
    for row in token_rows:
      handle.write(json.dumps({
          **row, "attention_cap_ms": ATTENTION_CAP_MS,
          "route_label": payload["route_label"],
          "speedup_claims_allowed": False,
      }, sort_keys=True) + "\n")
  write_json(out_dir / "smoothness.json", {
      "paired_spread": spread, "paired_spread_max": NOISE_MAX,
      "required_checks_passed": spread <= NOISE_MAX,
  })
  summary = [
      "# Exact-128k product paged-GQA provider gate", "",
      f"- required checks passed: `{str(required).lower()}`",
      f"- traced main / finalization: `{len(main_rows)} / {len(final_rows)}`",
      f"- repeat / confirm attention: `{repeat_ms} / {confirm_ms} ms`",
      f"- paired spread: `{spread}`",
      f"- attention cap: `{ATTENTION_CAP_MS} ms`",
      f"- rate pass: `{str(rate_pass).lower()}`",
      "- native replay / product speed admitted: `false / false`", "",
      "This gate admits only a captured paged-GQA provider source when the",
      "pre-registered event-rate and noise gates pass. Native replay numeric,",
      "integration, output-512, and product correctness remain separate.", "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary))
  print(json.dumps({
      "confirm_attention_ms": confirm_ms,
      "out_dir": str(out_dir.relative_to(ROOT)),
      "paired_spread": spread,
      "repeat_attention_ms": repeat_ms,
      "required_checks_passed": required,
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
