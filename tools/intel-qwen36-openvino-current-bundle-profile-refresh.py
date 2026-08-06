#!/usr/bin/env python3
"""Profile one accepted product timing worker without changing its provider.

The source worker must be a completed candidate timing row from the product
gate.  This tool preserves its bucket, graph, plugin, teacher-forced token
prefix, and either accepted LM-head timing contract: full-logit gated-exact or
compact token-only.  It changes only the requested diagnostic output length
and enables OpenVINO host-time plus final execution profiling.  Exactly one
guarded, transient-scope worker is launched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-openvino-current-bundle-profile-refresh-v1"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
DEFAULT_SOURCE = ROOT / (
    "output/openvino-lm-head-gated-exact-count25-"
    "20260723Tseq2121-all10-128k-o512-abba1/raw/"
    "sentinel_128k/block00/candidate-b1")


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PRODUCT = load_module("iq36_current_bundle_product", PRODUCT_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--source-worker-dir", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--output-tokens", type=int, default=3)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--plan-only", action="store_true")
  args = parser.parse_args()
  if args.output_tokens < 2:
    parser.error("output-tokens must be at least two")
  if args.timeout_s <= 0 or args.poll_interval_s <= 0:
    parser.error("timeout and poll interval must be positive")
  if (args.abort_below_available_gib < 0 or args.min_available_gib < 0 or
      args.abort_below_available_gib > args.min_available_gib):
    parser.error("invalid memory thresholds")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  return {"commit": commit, "dirty": bool(status), "status": status}


def other_product_workers() -> list[dict[str, Any]]:
  rows = []
  for proc in Path("/proc").iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
      continue
    try:
      command = (proc / "cmdline").read_bytes().replace(
          b"\0", b" ").decode("utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
      continue
    if (PRODUCT_TOOL.name in command and "--worker-config" in command):
      rows.append({"pid": int(proc.name), "command": command.strip()})
  return rows


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def config_delta(
    base: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
  return {
      key: {"base": base.get(key), "effective": candidate.get(key)}
      for key in sorted(set(base) | set(candidate))
      if base.get(key) != candidate.get(key)
  }


def provider_contract(row: dict[str, Any]) -> str | None:
  if (
      row.get("lm_head_i8q1_greedy_local2") is True and
      row.get("lm_head_token_only_feedback") is True):
    return "compact_direct_token_only"
  if (
      row.get("lm_head_i8q1_gated_exact") is True and
      row.get("lm_head_i8q1_greedy_local2") is False and
      row.get("lm_head_token_only_feedback") is False):
    return "full_logit_gated_exact"
  return None


def executed_provider_contract(result: dict[str, Any]) -> tuple[str | None, list[str]]:
  trace = result.get("lm_head_i8q1_trace") or {}
  selections = trace.get("selection_rows") or []
  providers = sorted({
      str(row.get("provider", "")) for row in selections})
  requested = provider_contract(result)
  if (
      requested == "full_logit_gated_exact" and selections and
      result.get("timing_token_output") is False and
      all(row.get("token_only") is False for row in selections) and
      all("gated_exact" in provider for provider in providers)):
    return requested, providers
  if (
      requested == "compact_direct_token_only" and selections and
      result.get("timing_token_output") is True and
      all(row.get("token_only") is True for row in selections) and
      all("compact" in provider and "gated_exact" not in provider
          for provider in providers)):
    return requested, providers
  return None, providers


def profile_rollup(census: dict[str, Any]) -> dict[str, Any]:
  totals: defaultdict[str, float] = defaultdict(float)
  counts: defaultdict[str, int] = defaultdict(int)
  maxima: defaultdict[str, float] = defaultdict(float)
  for row in census.get("retained_rows", []):
    node_type = str(row.get("node_type", ""))
    real_time = float(row.get("real_time_us", 0.0))
    totals[node_type] += real_time
    counts[node_type] += 1
    maxima[node_type] = max(maxima[node_type], real_time)
  ranked = sorted(
      ({
          "node_type": node_type,
          "executed_count": counts[node_type],
          "raw_real_time_us_nonadditive": totals[node_type],
          "max_row_real_time_us": maxima[node_type],
      } for node_type in totals),
      key=lambda row: (-row["raw_real_time_us_nonadditive"],
                       row["node_type"]))
  top_rows = [{
      key: row.get(key) for key in (
          "node_name", "node_type", "exec_type", "real_time_us", "cpu_time_us",
          "status")
  } for row in census.get("top_rows", [])[:32]]
  return {
      "dominant_retained_node_type": (
          ranked[0]["node_type"] if ranked else None),
      "ranked_retained_node_types_nonadditive": ranked,
      "top_execution_rows_nonadditive": top_rows,
      "profile_time_is_direct_savings_evidence": False,
  }


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  source_dir = args.source_worker_dir.resolve()

  source_config_path = source_dir / "worker-config.json"
  source_result_path = source_dir / "worker-result.json"
  source_run_path = source_dir / "run.json"
  for path in (source_config_path, source_result_path, source_run_path):
    if not path.is_file():
      raise SystemExit(f"missing source worker evidence: {path}")
  source_config = load_json(source_config_path)
  source_result = load_json(source_result_path)
  source_run = load_json(source_run_path)
  source_bucket = int(source_config.get("bucket") or 0)
  source_provider = provider_contract(source_config)
  result_provider, source_providers = executed_provider_contract(source_result)
  raw = (
      out_dir / "raw" / f"bucket_{source_bucket:06d}" /
      "profile" / "candidate")
  plugin = Path(str(source_config["candidate_gpu_plugin"])).resolve()
  if not plugin.is_file():
    raise SystemExit(f"missing candidate plugin: {plugin}")

  expected_tokens = [
      int(value) for value in source_result["generated_token_ids"][
          :args.output_tokens]]
  if len(expected_tokens) != args.output_tokens:
    raise SystemExit("source worker does not cover requested output tokens")
  reference_path = out_dir / "teacher-reference.json"

  config = dict(source_config)
  config.update({
      "capture_execution_census": True,
      "capture_lm_head_hidden": False,
      "capture_logits": False,
      "capture_prefill_profiles": False,
      "checkpoint_steps": list(range(args.output_tokens)),
      "host_time_profiling": 2,
      "output_tokens": args.output_tokens,
      "reference_result": str(reference_path.resolve()),
  })
  git = git_state()
  concurrent = other_product_workers()
  delta = config_delta(source_config, config)
  allowed_delta = {
      "capture_execution_census",
      "capture_lm_head_hidden",
      "capture_logits",
      "capture_prefill_profiles",
      "checkpoint_steps",
      "host_time_profiling",
      "output_tokens",
      "reference_result",
  }
  plan_checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("source_worker_completed_without_oom",
            source_run.get("returncode") == 0 and
            source_run.get("timed_out") is False and
            source_run.get("oom_observed") is False and
            (source_run.get("memory_guard") or {}).get("tripped") is False),
      check("source_is_candidate_timing_bundle",
            source_bucket >= 2048 and
            source_config.get("mode") == "candidate" and
            source_config.get("purpose") == "paired_product_timing"),
      check("source_selects_one_supported_exact_timing_provider",
            source_provider is not None and
            result_provider == source_provider,
            requested=source_provider, executed=result_provider,
            providers=source_providers),
      check("source_plugin_identity_is_exact",
            PRODUCT.sha256_file(plugin) ==
                source_result.get("candidate_gpu_plugin_sha256"),
            plugin=str(plugin),
            sha256=PRODUCT.sha256_file(plugin)),
      check("profile_changes_only_capture_length_and_reference_fields",
            set(delta) <= allowed_delta and
            {"host_time_profiling", "output_tokens", "reference_result"} <=
                set(delta) and
            config.get("host_time_profiling") == 2 and
            config.get("output_tokens") == args.output_tokens,
            delta=delta, allowed_delta=sorted(allowed_delta)),
      check("no_concurrent_product_worker_at_gate",
            not concurrent, concurrent=concurrent),
      check("one_serial_transient_scope_worker_is_pre_registered",
            args.min_available_gib == 8.0 and
            args.abort_below_available_gib == 4.0,
            worker_count=1, stock_worker_count=0,
            workers_concurrent=False, worker_transient_scope=True),
  ]
  plan_passed = all(row["pass"] for row in plan_checks)
  if args.plan_only:
    out_dir.mkdir(parents=True, exist_ok=False)
    plan_payload = {
        "schema_version": SCHEMA,
        "git": git,
        "required_checks_passed": plan_passed,
        "checks": plan_checks,
        "source_worker": str(source_dir),
        "source_worker_result_sha256": PRODUCT.sha256_file(source_result_path),
        "out_dir": str(out_dir),
        "candidate_gpu_plugin": str(plugin),
        "candidate_gpu_plugin_sha256": PRODUCT.sha256_file(plugin),
        "bucket": config.get("bucket"),
        "purpose": config.get("purpose"),
        "source_provider_contract": source_provider,
        "source_executed_provider_contract": result_provider,
        "source_lm_head_providers": source_providers,
        "output_tokens": config.get("output_tokens"),
        "expected_tokens": expected_tokens,
        "capture_execution_census": config.get("capture_execution_census"),
        "host_time_profiling": config.get("host_time_profiling"),
        "lm_head_i8q1_gated_exact_requested": config.get(
            "lm_head_i8q1_gated_exact"),
        "lm_head_i8q1_greedy_local2": config.get(
            "lm_head_i8q1_greedy_local2"),
        "lm_head_token_only_feedback": config.get(
            "lm_head_token_only_feedback"),
        "worker_count": 1,
        "stock_worker_count": 0,
        "workers_concurrent": False,
        "gpu_workers_launched": 0,
        "memory_preflight_gib": args.min_available_gib,
        "memory_abort_gib": args.abort_below_available_gib,
        "worker_transient_scope": True,
        "config_delta": delta,
    }
    write_json(out_dir / "plan.json", plan_payload)
    write_json(out_dir / "manifest.json", {
        "schema_version": SCHEMA,
        "git": git,
        "plan_only": True,
        "source_worker": str(source_dir),
        "source_worker_result_sha256": PRODUCT.sha256_file(source_result_path),
        "candidate_gpu_plugin_sha256": PRODUCT.sha256_file(plugin),
        "gpu_workers": 0,
        "planned_gpu_workers": 1,
        "workers_concurrent": False,
        "worker_transient_scope": True,
    })
    print(json.dumps({
        "artifact": str(out_dir),
        "required_checks_passed": plan_passed,
        "bucket": source_bucket,
        "timing_provider_contract": source_provider,
        "gpu_workers_launched": 0,
        "planned_gpu_workers": 1,
    }, sort_keys=True), flush=True)
    return 0 if plan_passed else 2

  out_dir.mkdir(parents=True, exist_ok=False)
  write_json(reference_path, {"generated_token_ids": expected_tokens})
  if concurrent:
    raise RuntimeError(f"concurrent product worker detected: {concurrent}")

  worker_args = SimpleNamespace(
      abort_below_available_gib=args.abort_below_available_gib,
      candidate_gpu_plugin=plugin,
      candidate_impls_cache_capacity=source_config.get(
          "candidate_impls_cache_capacity"),
      custom_config=Path(str(source_config["custom_config"])).resolve(),
      device=str(source_config.get("device", "GPU")),
      min_available_gib=args.min_available_gib,
      model_dir=Path(str(source_config["model_dir"])).resolve(),
      openvino_python=PRODUCT.OV_PYTHON,
      pack_gdn_state=bool(source_config.get("pack_gdn_state", False)),
      poll_interval_s=args.poll_interval_s,
      prime_candidate_exact_decode_shape=bool(
          source_config.get("prime_candidate_exact_decode_shape", False)),
      resume=False,
      timeout_s=args.timeout_s,
      worker_transient_scope=True,
  )
  worker = PRODUCT.run_worker(worker_args, raw, config)
  result = worker.get("result") or {}
  census = result.get("execution_census") or {}
  rollup = profile_rollup(census)
  actual_provider, providers = executed_provider_contract(result)

  checks = plan_checks + [
      check("single_worker_completed_without_oom",
            worker.get("returncode") == 0 and
            not worker.get("timed_out") and
            not worker.get("oom_observed") and
            not (worker.get("memory_guard") or {}).get("tripped"),
            worker={key: worker.get(key) for key in (
                "returncode", "timed_out", "oom_observed", "memory_guard",
                "monitor")}),
      check("profile_preserves_teacher_forced_token_prefix",
            result.get("teacher_forced_from_stock") is True and
            result.get("generated_token_ids") == expected_tokens,
            expected=expected_tokens,
            actual=result.get("generated_token_ids")),
      check("profile_uses_exact_source_plugin",
            result.get("candidate_gpu_plugin_sha256") ==
            source_result.get("candidate_gpu_plugin_sha256")),
      check("profile_preserves_exact_source_timing_provider",
            actual_provider == source_provider,
            expected=source_provider, actual=actual_provider,
            providers=providers),
      check("execution_profile_is_present",
            bool(census.get("executed_type_counts")) and
            bool(census.get("top_rows")) and
            rollup["dominant_retained_node_type"] is not None),
      check("profile_rows_are_attribution_only",
            rollup["profile_time_is_direct_savings_evidence"] is False),
  ]
  passed = all(row["pass"] for row in checks)
  metrics = {
      "schema_version": SCHEMA,
      "required_checks_passed": passed,
      "git": git,
      "source_worker": str(source_dir),
      "source_worker_result_sha256": PRODUCT.sha256_file(source_result_path),
      "candidate_gpu_plugin": str(plugin),
      "candidate_gpu_plugin_sha256": PRODUCT.sha256_file(plugin),
      "bucket": source_bucket,
      "timing_provider_contract": actual_provider,
      "output_tokens": args.output_tokens,
      "gpu_workers_launched": 1,
      "stock_workers_launched": 0,
      "concurrent_workers_at_launch": concurrent,
      "worker": {key: value for key, value in worker.items()
                 if key != "result"},
      "worker_result_summary": {
          "generated_token_ids": result.get("generated_token_ids"),
          "generated_token_ids_sha256": result.get(
              "generated_token_ids_sha256"),
          "prefill_tokens_s": result.get("prefill_tokens_s"),
          "decode_tokens_s": result.get("decode_tokens_s"),
          "decode_wall_ms": result.get("decode_wall_ms"),
          "source_summary": result.get("source_summary"),
          "executed_type_counts": census.get("executed_type_counts"),
          "lm_head_providers": providers,
      },
      "profile_rollup": rollup,
      "checks": checks,
  }
  write_json(out_dir / "metrics.json", metrics)
  write_json(out_dir / "manifest.json", {
      "schema_version": SCHEMA,
      "git": git,
      "source_worker": str(source_dir),
      "bucket": source_bucket,
      "timing_provider_contract": source_provider,
      "candidate_gpu_plugin": str(plugin),
      "candidate_gpu_plugin_sha256": PRODUCT.sha256_file(plugin),
      "output_tokens": args.output_tokens,
      "host_time_profiling": 2,
      "capture_execution_census": True,
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "worker_transient_scope": True,
  })
  dominant = rollup["dominant_retained_node_type"] or "n/a"
  (out_dir / "summary.md").write_text(
      "# Current timing-bundle profile refresh\n\n"
      f"Required checks passed: `{str(passed).lower()}`. Exactly one guarded "
      "candidate worker ran; no stock or concurrent worker was launched.\n\n"
      f"Bucket/provider: `{source_bucket}` / `{actual_provider}`. Dominant "
      f"retained node type: `{dominant}`. Profile times are "
      "non-additive attribution telemetry, not direct savings evidence. "
      "Use `metrics.json` to select and source-bound the next kernel route.\n",
      encoding="utf-8")
  print(json.dumps({
      "artifact": str(out_dir),
      "required_checks_passed": passed,
      "bucket": source_bucket,
      "timing_provider_contract": actual_provider,
      "dominant_retained_node_type": dominant,
      "returncode": worker.get("returncode"),
      "oom_observed": worker.get("oom_observed"),
      "memory_guard_tripped": (
          worker.get("memory_guard") or {}).get("tripped"),
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
