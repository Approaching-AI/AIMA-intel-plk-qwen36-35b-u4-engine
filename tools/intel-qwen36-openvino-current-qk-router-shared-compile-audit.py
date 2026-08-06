#!/usr/bin/env python3
"""Audit the successful seq2207 compile without repeating GPU work.

Seq2207 compiled the intended graph and exited before InferRequest creation,
but its wrapper inspected flag fields emitted only by the normal inference
return path.  This zero-GPU audit binds the immutable raw worker config,
environment, result, runtime census, and memory evidence instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-current-qk-router-shared-compile-audit-v1")
COMPILE_DIR = ROOT / (
    "output/openvino-current-qk-router-shared-compile-"
    "20260731Tseq2207-clean")
COMPILE_RESULT = COMPILE_DIR / "result.json"
WORKER_CONFIG = COMPILE_DIR / "raw/worker/worker-config.json"
WORKER_RESULT = COMPILE_DIR / "raw/worker/worker-result.json"
MODEL_IDENTITY = COMPILE_DIR / "model-identity.json"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2206/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_COMPILE_RESULT_SHA256 = (
    "e8ac2dc58084d72eb10d813bc6d5fb614e45e6d87467c1f8074fe8acbed6f725")
EXPECTED_WORKER_CONFIG_SHA256 = (
    "ed3f7dbcb8125683a730524c575822d62eadd7219cf242c8e8028c3face2aa08")
EXPECTED_WORKER_RESULT_SHA256 = (
    "470bd85934f0333bc2493bc6dba1526df1542148b0b2faecb7fa41f7370dea68")
EXPECTED_MODEL_IDENTITY_SHA256 = (
    "f6a27f8da62474492721219cdaf8228a6a0a378cffcc037235f0f79f3d48d762")
EXPECTED_PLUGIN_SHA256 = (
    "3ffcacbd4f7b1ab10e9a461b28c7385a86ec9c530f4af03495c5fb3dbba239f5")
COMPILE_COMMIT = "959c523dfec177a5e60e6291b134ae0f7751e753"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, text=True, capture_output=True, check=False)


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def runtime_names(census: dict[str, Any], prefix: str) -> list[str]:
  return sorted(
      str(row.get("name")) for row in census.get("attention_rows", [])
      if row.get("layer_type") == "CustomGPUPrimitive" and
      str(row.get("name", "")).startswith(prefix))


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required_paths = (
      COMPILE_RESULT, WORKER_CONFIG, WORKER_RESULT, MODEL_IDENTITY, PLUGIN)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing compile-audit inputs: " + ", ".join(missing))

  compile_gate = load_json(COMPILE_RESULT)
  config = load_json(WORKER_CONFIG)
  worker_result = load_json(WORKER_RESULT)
  model_identity = load_json(MODEL_IDENTITY)
  worker = compile_gate.get("worker") or {}
  environment = worker.get("environment") or {}
  source = worker_result.get("source_summary") or {}
  census = worker_result.get("runtime_census") or {}
  monitor = worker.get("monitor") or {}
  memory_guard = worker.get("memory_guard") or {}
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  status = run(["git", "status", "--porcelain"]).stdout.splitlines()
  ancestor = run([
      "git", "merge-base", "--is-ancestor", COMPILE_COMMIT, head])
  failed = [
      row.get("name") for row in compile_gate.get("checks", [])
      if row.get("pass") is False]
  passed = [
      row.get("name") for row in compile_gate.get("checks", [])
      if row.get("pass") is True]
  expected_attention_names = sorted(
      f"iq36_hot_attention_layer{layer}" for layer in LAYERS)
  expected_qk_names = sorted(
      f"iq36_qk_rope_layout_layer{layer}" for layer in LAYERS)
  qk_layers = sorted(
      int(row["layer"]) for row in source.get("qk_rope_layout_rewrites", [])
      if isinstance(row, dict) and isinstance(row.get("layer"), int))
  available_start = available_memory_bytes()

  checks = [
      check("repository_is_clean_pushed_and_contains_compile_commit",
            not status and head == origin_main and ancestor.returncode == 0,
            head=head, origin_main=origin_main, status=status,
            compile_commit_is_ancestor=ancestor.returncode == 0),
      check("seq2207_evidence_files_are_immutable",
            sha256(COMPILE_RESULT) == EXPECTED_COMPILE_RESULT_SHA256 and
            sha256(WORKER_CONFIG) == EXPECTED_WORKER_CONFIG_SHA256 and
            sha256(WORKER_RESULT) == EXPECTED_WORKER_RESULT_SHA256 and
            sha256(MODEL_IDENTITY) == EXPECTED_MODEL_IDENTITY_SHA256 and
            sha256(PLUGIN) == EXPECTED_PLUGIN_SHA256,
            compile_result_sha256=sha256(COMPILE_RESULT),
            worker_config_sha256=sha256(WORKER_CONFIG),
            worker_result_sha256=sha256(WORKER_RESULT),
            model_identity_sha256=sha256(MODEL_IDENTITY),
            plugin_sha256=sha256(PLUGIN)),
      check("wrapper_failure_is_exactly_early_return_flag_echo",
            compile_gate.get("required_checks_passed") is False and
            compile_gate.get("verdict") ==
                "repair_current_qk_router_shared_candidate_compile" and
            len(passed) == 13 and
            failed == ["candidate_flags_are_exact_and_mutually_exclusive"],
            passed_check_count=len(passed), failed_checks=failed),
      check("raw_config_binds_exact_candidate_only_flags",
            config.get("mode") == "candidate" and
            config.get("candidate_path") == "hot_cold_custom" and
            config.get("candidate_gpu_plugin") == str(PLUGIN) and
            config.get("compile_only") is True and
            config.get("instantiate_only") is False and
            config.get("fuse_qk_rope_layout") is True and
            config.get("fuse_router_shared_triple") is True and
            config.get("fuse_fixed_fc") is False and
            config.get("fixed_fc_manager_direct") is False and
            config.get("capture_execution_census") is True and
            config.get("bucket") == 2048 and
            config.get("output_tokens") == 130 and
            config.get("target_layers") == list(LAYERS)),
      check("worker_environment_enables_only_router_shared_route",
            environment.get("IQ36_ROUTER_SHARED_TRIPLE") == "1" and
            "IQ36_FIXED_FC_MANAGER_SCOPE" not in environment and
            environment.get("IQ36_LM_HEAD_I8Q1") == "1" and
            environment.get("IQ36_LM_HEAD_I8Q1_GATED_EXACT") == "1",
            relevant_environment={
                key: value for key, value in environment.items()
                if key.startswith("IQ36_")}),
      check("actual_compile_completed_without_request_or_inference",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            worker_result.get("compile_only") is True and
            worker_result.get("instantiate_only") is False and
            worker_result.get("worker_created_infer_request") is False and
            worker_result.get("worker_executed_inference") is False and
            "generated_token_ids" not in worker_result and
            "state_schema_after" not in worker_result),
      check("actual_source_and_runtime_census_are_exact",
            worker_result.get("candidate_gpu_plugin_sha256") ==
                EXPECTED_PLUGIN_SHA256 and
            source.get("fuse_qk_rope_layout") is True and
            source.get("qk_rope_layout_rewrite_count") == len(LAYERS) and
            qk_layers == list(LAYERS) and
            source.get("custom_count_after") == len(LAYERS) and
            source.get("stock_sdpa_count_after") == 0 and
            census.get("qk_rope_layout_custom_count") == len(LAYERS) and
            census.get("hot_attention_custom_count") == len(LAYERS) and
            census.get("linear_conv_custom_count") == 30 and
            census.get("fixed_fc_custom_count") == 0 and
            census.get("stock_sdpa_like_count") == 0 and
            runtime_names(census, "iq36_hot_attention_layer") ==
                expected_attention_names and
            runtime_names(census, "iq36_qk_rope_layout_layer") ==
                expected_qk_names,
            qk_layers=qk_layers),
      check("compile_duration_is_finite",
            isinstance(worker_result.get("language_compile_ms"), (int, float))
            and math.isfinite(float(worker_result["language_compile_ms"]))
            and float(worker_result["language_compile_ms"]) > 0.0,
            language_compile_ms=worker_result.get("language_compile_ms")),
      check("seq2207_memory_scope_and_guard_hold",
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True and
            (worker.get("worker_transient_scope") or {}).get(
                "resource_limits_changed") is False and
            memory_guard.get("tripped") is False and
            int(monitor.get("process_rss_peak_bytes", -1)) >= 0 and
            int(monitor.get("process_swap_peak_bytes", -1)) >= 0 and
            int(monitor.get("system_available_min_bytes", 0)) >= 4 * 1024**3,
            monitor=monitor,
            worker_transient_scope=worker.get("worker_transient_scope")),
      check("model_identity_remains_locked",
            model_identity.get("required_checks_passed") is True,
            model_dir=model_identity.get("model_dir")),
      check("audit_launches_no_compiler_gpu_model_or_inference_worker", True,
            compiler_builds=0, graph_compiles=0, gpu_contexts=0,
            model_workers=0, infer_requests=0, inference_workers=0),
      check("executed_shared_fc_census_remains_deferred", True,
            expected_next_execution_counts={
                "FullyConnectedCompressed": 291,
                "IQ36QKRopeLayout": 10,
                "IQ36ExactPhaseDualCohortHotAttentionGQA": 10,
                "shared_triples": 40,
                "unfused_router_gates": 40,
            }),
  ]
  available_end = available_memory_bytes()
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_current_qk_router_shared_output130_correctness_worker"
      if required else
      "repair_seq2207_compile_evidence")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_output130_correctness_worker_admitted": required,
      "performance_worker_admitted": False,
      "formal_performance_admitted": False,
      "seq2207_compile_repeated": False,
      "workers": {
          "compiler_builds": 0,
          "graph_compiles": 0,
          "gpu_contexts": 0,
          "model_workers": 0,
          "infer_requests": 0,
          "inference_workers": 0,
      },
      "memory": {
          "start_available_bytes": available_start,
          "end_available_bytes": available_end,
      },
      "compile_evidence": {
          "result": display(COMPILE_RESULT),
          "worker_config": display(WORKER_CONFIG),
          "worker_result": display(WORKER_RESULT),
          "model_identity": display(MODEL_IDENTITY),
          "language_compile_ms": worker_result.get("language_compile_ms"),
          "peak_rss_bytes": monitor.get("process_rss_peak_bytes"),
          "peak_swap_bytes": monitor.get("process_swap_peak_bytes"),
          "minimum_available_bytes": monitor.get(
              "system_available_min_bytes"),
      },
      "checks": checks,
      "next_action": {
          "route": "current_qk_router_shared_candidate_output130_correctness",
          "requirements": [
              "push one exact candidate-only correctness/census gate",
              "run one fresh-scope 2k/output130 InferRequest worker",
              "require accepted logits/tokens/provider and no OOM",
              "require execution census FC/QK/attention/shared/router "
              "291/10/10/40/40",
              "only a pass may fund one control-candidate point block",
          ],
      },
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "inputs": {
          display(path): sha256(path)
          for path in required_paths
      },
      "workers": result["workers"],
  })
  report = f"""# Seq2207 current-bundle compile evidence audit

Verdict: **{verdict}**. Required checks:
`{str(required).lower()}`.

The existing seq2207 worker compiled the intended candidate graph in
`{float(worker_result.get('language_compile_ms', 0.0)):.3f} ms`. Its only
wrapper failure was reading flags from the compile-only early-return result;
the immutable raw config and worker environment bind Q/K plus router-shared
on and both fixed-FC routes off. The runtime graph retains Q/K, dual-attention,
and linear custom owner counts `10/10/30`.

This audit repeated no compile and launched no GPU or model worker. Seq2207
peak RSS/swap was `{int(monitor.get('process_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_swap_peak_bytes', 0))} B`, with no OOM or guard.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": required,
      "seq2207_compile_repeated": False,
      "language_compile_ms": worker_result.get("language_compile_ms"),
      "gpu_workers": 0,
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
