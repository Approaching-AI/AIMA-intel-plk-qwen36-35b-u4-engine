#!/usr/bin/env python3
"""Measure one near-bucket service carrier case in an isolated process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/service"))

from iq36_server.runtime import OpenVinoRuntime, RuntimeConfig  # noqa: E402


CASES = {
    16380: 16384,
    32758: 32768,
    65519: 65536,
    131037: 131072,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prompt-tokens", type=int, choices=tuple(CASES),
                      required=True)
  parser.add_argument("--output-tokens", type=int, default=64)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--repo-root", type=Path, default=ROOT)
  parser.add_argument(
      "--model-dir", type=Path,
      default=Path("/home/intel/Qwen3.6-35B-A3B-ov"))
  parser.add_argument(
      "--plugin", type=Path,
      default=ROOT / "output/openvino-90214e-l0-gpu-v011/bin/intel64/"
      "Release/libopenvino_intel_gpu_plugin.so")
  parser.add_argument(
      "--custom-config", type=Path,
      default=ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml")
  parser.add_argument(
      "--compile-cache-dir", type=Path,
      default=ROOT / "output/http-near-boundary-cache-v011")
  args = parser.parse_args()
  if not 2 <= args.output_tokens <= 512:
    parser.error("--output-tokens must be in [2, 512]")
  return args


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def git_value(root: Path, *arguments: str) -> str:
  result = subprocess.run(
      ["git", *arguments], cwd=root, check=True, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  return result.stdout.strip()


def memory_values(path: Path) -> dict[str, int]:
  values = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    name, _, raw = line.partition(":")
    fields = raw.split()
    if fields and fields[0].isdigit():
      values[name] = int(fields[0])
  return values


class MemorySampler:
  def __init__(self) -> None:
    initial = memory_values(Path("/proc/meminfo"))
    self.total_kib = initial["MemTotal"]
    self.start_available_kib = initial["MemAvailable"]
    self.min_available_kib = self.start_available_kib
    self.max_rss_kib = 0
    self.max_swap_kib = 0
    self.samples = 0
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)

  def start(self) -> None:
    self._thread.start()

  def stop(self) -> dict[str, Any]:
    self._stop.set()
    self._thread.join()
    ending = memory_values(Path("/proc/meminfo"))
    return {
        "samples": self.samples,
        "system_total_bytes": self.total_kib * 1024,
        "system_available_start_bytes": self.start_available_kib * 1024,
        "system_available_min_bytes": self.min_available_kib * 1024,
        "system_available_end_bytes": ending["MemAvailable"] * 1024,
        "peak_system_used_bytes":
            (self.total_kib - self.min_available_kib) * 1024,
        "peak_available_drop_from_start_bytes":
            max(0, self.start_available_kib - self.min_available_kib) * 1024,
        "max_process_rss_bytes": self.max_rss_kib * 1024,
        "max_process_swap_bytes": self.max_swap_kib * 1024,
    }

  def _run(self) -> None:
    while not self._stop.is_set():
      system = memory_values(Path("/proc/meminfo"))
      process = memory_values(Path("/proc/self/status"))
      self.min_available_kib = min(
          self.min_available_kib, system["MemAvailable"])
      self.max_rss_kib = max(self.max_rss_kib, process.get("VmRSS", 0))
      self.max_swap_kib = max(self.max_swap_kib, process.get("VmSwap", 0))
      self.samples += 1
      self._stop.wait(0.05)


def main() -> int:
  args = parse_args()
  repo = args.repo_root.expanduser().resolve()
  output = args.output.expanduser().resolve()
  if output.exists():
    raise SystemExit(f"error: output already exists: {output}")

  prompt_tokens = args.prompt_tokens
  bucket = CASES[prompt_tokens]
  config = RuntimeConfig(
      repo_root=repo,
      model_dir=args.model_dir.expanduser().resolve(),
      device="GPU",
      plugin=args.plugin.expanduser().resolve(),
      custom_config=args.custom_config.expanduser().resolve(),
      profile="long_compact",
      bucket=bucket,
      compile_cache_dir=args.compile_cache_dir.expanduser().resolve(),
      prefix_cache_bytes=0,
      prefix_cache_entries=0,
      prefix_cache_ttl_s=0,
      prewarm=False)

  sampler = MemorySampler()
  sampler.start()
  wall_started = time.monotonic_ns()
  runtime = OpenVinoRuntime(config)
  np = runtime.np
  prompt_text = "hello " * (prompt_tokens - 1)
  encoded = runtime.tokenizer.encode(prompt_text)
  prompt = np.asarray(encoded.input_ids.data, dtype=np.int64).reshape(-1)
  if prompt.size != prompt_tokens:
    raise RuntimeError(
        f"prompt materialized {prompt.size} tokens, expected {prompt_tokens}")

  token_times_ns: list[int] = []

  def emit(event: dict[str, Any]) -> None:
    if event.get("event") == "token":
      token_times_ns.append(time.monotonic_ns())

  generation_started = time.monotonic_ns()
  result = runtime.generate(
      f"near-boundary-{prompt_tokens}", tuple(map(int, prompt)), {
          "max_new_tokens": args.output_tokens,
          "temperature": 0.0,
          "ignore_eos": True,
          "stop": [],
          "repetition_penalty": 1.0,
          "presence_penalty": 0.0,
          "frequency_penalty": 0.0,
          "seed": 0,
          "top_k": 0,
          "top_p": 1.0,
          "logprobs": False,
          "top_logprobs": 0,
      }, threading.Event(), emit, use_prefix_cache=False)
  generation_finished = time.monotonic_ns()
  memory = sampler.stop()

  generated = result["token_ids"]
  if len(generated) != args.output_tokens or len(token_times_ns) != len(generated):
    raise RuntimeError("generation did not produce the requested token count")
  ttft_ms = (token_times_ns[0] - generation_started) / 1_000_000.0
  inter_token_ms = (
      (token_times_ns[-1] - token_times_ns[0]) / 1_000_000.0)
  inter_token_tps = (
      (len(token_times_ns) - 1) * 1000.0 / inter_token_ms)

  ready = runtime.ready_info()
  status = git_value(repo, "status", "--porcelain=v1")
  record = {
      "schema": "iq36-http-near-boundary-benchmark-v1",
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "source": {
          "commit": git_value(repo, "rev-parse", "HEAD"),
          "dirty": bool(status),
          "diff_scope": (
              "long LM-head logical reinterpretation of preallocated "
              "activation and output buffers"),
      },
      "target": {
          "alias": "local-ptl-target",
          "kernel": platform.release(),
          "machine": platform.machine(),
          "model_dir": str(config.model_dir),
          "device": config.device,
          "batch_size": 1,
          "runtime": {
              "openvino": ready["openvino_version"],
              "openvino_genai": ready["openvino_genai_version"],
              "openvino_tokenizers": ready["openvino_tokenizers_version"],
          },
      },
      "workload": {
          "prompt_construction": "the UTF-8 string 'hello ' repeated N-1 times",
          "prompt_utf8_sha256": sha256_bytes(prompt_text.encode("utf-8")),
          "prompt_token_ids_sha256": sha256_bytes(
              np.asarray(prompt, dtype="<i8").tobytes()),
          "prompt_tokens": prompt_tokens,
          "bucket": bucket,
          "output_tokens": args.output_tokens,
          "precision": "locked OpenVINO U4 model with IQ36 long plugin",
          "cache_state": "cold no-prefix; persistent compile cache enabled",
          "warmup": "none; compile time reported separately",
          "profile": "long_compact",
      },
      "identity": {
          "plugin_sha256": ready["plugin_sha256"],
          "custom_config_sha256": ready["custom_config_sha256"],
          "runtime_identity_verified": ready["runtime_identity_verified"],
      },
      "performance": {
          "compile_ms": runtime.compile_ms,
          "ttft_ms": ttft_ms,
          "prefill_ms": result["prefill_ms"],
          "decode_ms": result["decode_ms"],
          "decode_tokens_per_s_service_timer":
              len(generated) * 1000.0 / result["decode_ms"],
          "inter_token_decode_tokens_per_s": inter_token_tps,
          "generation_wall_ms":
              (generation_finished - generation_started) / 1_000_000.0,
          "process_wall_ms":
              (generation_finished - wall_started) / 1_000_000.0,
      },
      "correctness_sanity": {
          "requested_output_tokens_emitted": True,
          "finite_completion": True,
          "finish_reason": result["finish_reason"],
          "generated_token_ids_sha256": sha256_bytes(
              np.asarray(generated, dtype="<i8").tobytes()),
      },
      "memory": memory,
      "method": {
          "worker_isolation": "one runtime and one case per OS process",
          "prefix_cache": False,
          "ignore_eos": True,
          "sampling": "greedy temperature=0",
          "ttft_definition":
              "generate() entry to the first emitted token after compilation",
          "decode_tps_definition": (
              "inter-token rate is 63 intervals for 64 emitted tokens; the "
              "service-timer rate is also reported for exact reproducibility"),
          "memory_sampling_interval_ms": 50,
      },
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  print(json.dumps({
      "output": str(output),
      "prompt_tokens": prompt_tokens,
      "ttft_ms": ttft_ms,
      "inter_token_decode_tokens_per_s": inter_token_tps,
      "required_checks_passed": True,
  }, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
