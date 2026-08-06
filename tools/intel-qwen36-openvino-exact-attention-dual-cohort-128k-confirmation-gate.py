#!/usr/bin/env python3
"""Confirm the 128k dual-cohort lane with eight paired ABBA blocks."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-dual-cohort-128k-"
    "confirmation-gate-v1")
WALL_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-exact-attention-dual-cohort-"
    "128k-wall-gate.py")
SEED = ROOT / (
    "output/openvino-exact-attention-dual-cohort-128k-wall-"
    "20260723Tseq2139-clean")
SEED_GATE = SEED / "gate.json"
SEED_STOCK_CONFIG = SEED / "raw/stock-a1/worker-config.json"
SEED_CANDIDATE_CONFIG = SEED / "raw/candidate-b1/worker-config.json"
EXPECTED_SEED_COMMIT = "4ccfd566a5f791da4c8a6e0684adeb39211c61f4"
PAIRED_BLOCKS = 8
TARGET_RATIO = 1.10


def load_module(name: str, path: Path) -> ModuleType:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


WALL = load_module("iq36_dual_cohort_wall", WALL_TOOL)
PRODUCT = WALL.PRODUCT


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def base_config(path: Path) -> dict[str, Any]:
  cfg = PRODUCT.load_json(path)
  for key in ("raw", "result", "candidate_gpu_plugin", "custom_config",
              "device", "model_dir", "pack_gdn_state",
              "prime_candidate_exact_decode_shape",
              "candidate_impls_cache_capacity"):
    cfg.pop(key, None)
  return cfg


def run_is_exact(run: dict[str, Any]) -> bool:
  result = run.get("result") or {}
  if run.get("returncode") != 0 or run.get("timed_out") or (
      run.get("oom_observed") is not False) or (
      (run.get("memory_guard") or {}).get("tripped") is not False):
    return False
  if result.get("generated_token_ids_sha256") != WALL.EXPECTED_TOKEN_SHA256:
    return False
  if result.get("mode") == "candidate":
    return (
        WALL.exact_dual_runtime(result)
        and WALL.candidate_timing_isolation(result))
  return WALL.stock_timing_isolation(result)


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  if out.exists() and not args.resume:
    raise SystemExit(f"output already exists: {out}")
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=args.resume)

  required_paths = (
      WALL_TOOL, SEED_GATE, SEED_STOCK_CONFIG, SEED_CANDIDATE_CONFIG,
      WALL.PRODUCT_TOOL, WALL.GRAPH_TOOL, WALL.KERNEL, WALL.CONFIG,
      WALL.SHORT_GATE, WALL.ACCEPTED_CORRECTNESS, WALL.REFERENCE_STOCK,
      WALL.PROMPT, WALL.PLUGIN, PRODUCT.ACCEPTANCE, PRODUCT.MODEL_CONTRACT,
      PRODUCT.MODEL_DIR)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit(
        "missing dual-cohort confirmation inputs: " + ", ".join(missing))

  git = PRODUCT.BOOT.git_state(out)
  seed = PRODUCT.load_json(SEED_GATE)
  seed_sources = seed.get("sources") or {}
  seed_runs = seed.get("workers") or {}
  seed_performance = seed.get("performance") or {}
  seed_bound = bool(
      seed.get("required_checks_passed") is True
      and seed.get("multiblock_confirmation_admitted") is True
      and seed.get("verdict") ==
          "admit_exact_attention_dual_cohort_128k_multiblock_confirmation"
      and seed.get("git", {}).get("commit") == EXPECTED_SEED_COMMIT
      and seed.get("git", {}).get("dirty") is False
      and seed_performance.get("paired_block_count") == 1
      and seed_performance.get("absolute_floors", {}).get("pass") is True
      and len(seed_runs) == 4
      and all(run_is_exact(run) for run in seed_runs.values())
      and seed_sources.get(PRODUCT.relative(WALL.PRODUCT_TOOL)) ==
          PRODUCT.sha256_file(WALL.PRODUCT_TOOL)
      and seed_sources.get(PRODUCT.relative(WALL.GRAPH_TOOL)) ==
          PRODUCT.sha256_file(WALL.GRAPH_TOOL)
      and seed_sources.get(PRODUCT.relative(WALL.KERNEL)) ==
          PRODUCT.sha256_file(WALL.KERNEL)
      and seed_sources.get(PRODUCT.relative(WALL.CONFIG)) ==
          PRODUCT.sha256_file(WALL.CONFIG)
      and seed_sources.get(PRODUCT.relative(WALL.SHORT_GATE)) ==
          PRODUCT.sha256_file(WALL.SHORT_GATE)
      and seed.get("plugin", {}).get("sha256") ==
          WALL.EXPECTED_PLUGIN_SHA256)

  stock_config = base_config(SEED_STOCK_CONFIG)
  candidate_config = base_config(SEED_CANDIDATE_CONFIG)
  template_bound = bool(
      stock_config.get("purpose") == "paired_product_timing"
      and stock_config.get("mode") == "stock"
      and stock_config.get("candidate_path") == "stock_sdpa"
      and stock_config.get("reference_result") ==
          str(WALL.REFERENCE_STOCK.resolve())
      and stock_config.get("output_tokens") == WALL.OUTPUT_TOKENS
      and stock_config.get("exact_phase_dual_cohort") is True
      and candidate_config.get("purpose") == "paired_product_timing"
      and candidate_config.get("mode") == "candidate"
      and candidate_config.get("candidate_path") == "hot_cold_custom"
      and candidate_config.get("reference_result") ==
          str(WALL.REFERENCE_STOCK.resolve())
      and candidate_config.get("output_tokens") == WALL.OUTPUT_TOKENS
      and candidate_config.get("exact_phase_dual_cohort") is True
      and candidate_config.get("timing_token_output") is True)

  worker_args = SimpleNamespace(
      abort_below_available_gib=WALL.MEMORY_STOP_GIB,
      candidate_gpu_plugin=WALL.PLUGIN,
      candidate_impls_cache_capacity=None,
      custom_config=WALL.CONFIG,
      device="GPU",
      min_available_gib=WALL.PREFLIGHT_GIB,
      model_dir=PRODUCT.MODEL_DIR,
      openvino_python=PRODUCT.OV_PYTHON,
      pack_gdn_state=False,
      poll_interval_s=1.0,
      prime_candidate_exact_decode_shape=False,
      resume=args.resume,
      timeout_s=args.timeout_s,
  )
  schedule = (
      ("stock-a1", "stock"),
      ("candidate-b1", "candidate"),
      ("candidate-b2", "candidate"),
      ("stock-a2", "stock"),
  )
  all_runs = [seed_runs[label] for label, _ in schedule]
  blocks = list(seed_performance.get("blocks") or [])
  new_runs: dict[str, dict[str, Any]] = {}
  stopped_reason = None
  for block_index in range(1, PAIRED_BLOCKS):
    block_runs: dict[str, dict[str, Any]] = {}
    for label, mode in schedule:
      config = stock_config if mode == "stock" else candidate_config
      worker_dir = raw / f"block{block_index:02d}" / label
      run = PRODUCT.run_worker(worker_args, worker_dir, dict(config))
      run["worker"] = PRODUCT.relative(worker_dir)
      block_runs[label] = run
      new_runs[f"block{block_index:02d}/{label}"] = run
      all_runs.append(run)
      if not run_is_exact(run):
        stopped_reason = f"block{block_index:02d}/{label} failed"
        break
    if stopped_reason:
      break
    block = PRODUCT.block_summary(block_index, block_runs)
    blocks.append(block)
    PRODUCT.write_json(
        raw / f"block{block_index:02d}/block-summary.json", block)

  complete = (
      stopped_reason is None and len(blocks) == PAIRED_BLOCKS
      and len(all_runs) == 4 * PAIRED_BLOCKS)
  case = {
      "bucket": WALL.BUCKET,
      "candidate_path": "hot_cold_custom",
      "case_id": "sentinel_128k",
  }
  acceptance = PRODUCT.load_json(PRODUCT.ACCEPTANCE)
  performance = (
      PRODUCT.performance_for_case(case, blocks, acceptance)
      if blocks else {})
  memory = (
      PRODUCT.memory_rollup(all_runs, acceptance)
      if all_runs else {"checks": [], "required_checks_passed": False})
  smoothness = (
      PRODUCT.smoothness_rollup([performance], all_runs, acceptance)
      if performance else {
          "checks": [], "required_checks_passed": False})
  inference = performance.get("phase_inference") or {}
  inference_pass = bool(
      complete
      and all(
          inference.get(phase, {}).get("sample_count_pass") is True
          and inference.get(phase, {}).get("rate_pass") is True
          and math.isfinite(float(
              inference.get(phase, {}).get(
                  "lower_confidence_bound_ratio", math.nan)))
          and float(inference[phase]["lower_confidence_bound_ratio"]) >=
              TARGET_RATIO
          for phase in ("prefill_tokens_s", "decode_tokens_s", "total_rate")))
  absolute_pass = bool(
      performance.get("absolute_floors", {}).get("pass") is True)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("clean_seq2139_seed_block_is_exactly_bound", seed_bound),
      check("seed_worker_configs_are_reused_without_semantic_change",
            template_bound),
      check("eight_abba_blocks_execute_strictly_serially", complete,
            paired_blocks=len(blocks), worker_count=len(all_runs),
            stopped_reason=stopped_reason),
      check("all_32_timing_rows_preserve_tokens_and_isolation",
            complete and all(run_is_exact(run) for run in all_runs)),
      check("paired_one_sided_95pct_lcbs_clear_1p10",
            inference_pass, phase_inference=inference),
      check("candidate_absolute_prefill_and_decode_floors_pass",
            absolute_pass,
            absolute_floors=performance.get("absolute_floors")),
      check("memory_and_smoothness_checks_pass",
            memory.get("required_checks_passed") is True
            and smoothness.get("required_checks_passed") is True,
            memory_checks=memory.get("checks"),
            smoothness_checks=smoothness.get("checks")),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "confirm_exact_attention_dual_cohort_128k_lane"
      if required else
      "repair_exact_attention_dual_cohort_128k_confirmation")
  model_identity = PRODUCT.BOOT.capture_model_identity(
      PRODUCT.MODEL_DIR.resolve(), PRODUCT.MODEL_CONTRACT.resolve())
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "lane_128k_confirmed": required,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
      "checks": checks,
      "seed_gate": PRODUCT.relative(SEED_GATE),
      "seed_block_count": 1,
      "new_block_count": max(0, len(blocks) - 1),
      "paired_block_count": len(blocks),
      "worker_count": len(all_runs),
      "new_workers": new_runs,
      "performance": performance,
      "memory": memory,
      "smoothness": smoothness,
      "model_identity": model_identity,
      "next_route": (
          "openvino_exact_attention_dual_cohort_complete_product_matrix"
          if required else
          "profile_or_switch_exact_attention_dual_cohort_product_route"),
      "sources": {
          PRODUCT.relative(path): PRODUCT.sha256_file(path)
          for path in (
              WALL_TOOL, SEED_GATE, SEED_STOCK_CONFIG,
              SEED_CANDIDATE_CONFIG, WALL.PRODUCT_TOOL, WALL.GRAPH_TOOL,
              WALL.KERNEL, WALL.CONFIG, WALL.SHORT_GATE,
              WALL.ACCEPTED_CORRECTNESS, WALL.REFERENCE_STOCK)
      },
  }
  PRODUCT.write_json(out / "gate.json", payload)
  PRODUCT.write_json(out / "performance.json", performance)
  PRODUCT.write_json(out / "memory.json", memory)
  PRODUCT.write_json(out / "smoothness.json", smoothness)
  PRODUCT.write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "seed_gate": PRODUCT.relative(SEED_GATE),
      "files": [
          "gate.json", "performance.json", "memory.json",
          "smoothness.json", "summary.md",
      ],
  })
  floors = performance.get("absolute_floors") or {}
  (out / "summary.md").write_text(
      "\n".join([
          "# Exact-attention dual-cohort 128k confirmation",
          "",
          f"- verdict: `{verdict}`",
          f"- paired blocks / workers: `{len(blocks)} / {len(all_runs)}`",
          f"- candidate prefill/decode: "
          f"`{floors.get('prefill_median')} / "
          f"{floors.get('decode_median')} tok/s`",
          f"- prefill/decode/total LCB: "
          f"`{inference.get('prefill_tokens_s', {}).get('lower_confidence_bound_ratio')} / "
          f"{inference.get('decode_tokens_s', {}).get('lower_confidence_bound_ratio')} / "
          f"{inference.get('total_rate', {}).get('lower_confidence_bound_ratio')}`",
          "- complete product speedup claim: `false`",
          "",
      ]), encoding="utf-8")
  print(json.dumps({
      "output": PRODUCT.relative(out),
      "verdict": verdict,
      "required_checks_passed": required,
      "paired_block_count": len(blocks),
      "worker_count": len(all_runs),
      "candidate_prefill_tokens_s": floors.get("prefill_median"),
      "candidate_decode_tokens_s": floors.get("decode_median"),
      "prefill_lcb": inference.get(
          "prefill_tokens_s", {}).get("lower_confidence_bound_ratio"),
      "decode_lcb": inference.get(
          "decode_tokens_s", {}).get("lower_confidence_bound_ratio"),
      "total_lcb": inference.get(
          "total_rate", {}).get("lower_confidence_bound_ratio"),
      "speedup_claims_allowed": False,
  }, sort_keys=True), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
