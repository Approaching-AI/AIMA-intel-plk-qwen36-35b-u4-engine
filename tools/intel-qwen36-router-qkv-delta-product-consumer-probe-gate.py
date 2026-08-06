#!/usr/bin/env python3
"""Probe product-owned qkv-delta consumer counters on target."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


SCHEMA_VERSION = "intel-qwen36-router-qkv-delta-product-consumer-probe-gate-v0"

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ311 = (
    ROOT
    / "output/router-qkv-delta-product-consumer-target-compile-gate-20260708Tseq311Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r1-engine-seed-prompt-input-check-20260627T155328Z/token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-product-consumer-probe-gate-20260708Tseq312Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
DECODE_TOKENS = 1
EXPECTED_PRODUCER_VALUES = len(PRODUCER_LAYERS) * 2048 * DECODE_TOKENS
EXPECTED_PRODUCT_LAYERS = len(ALL_LINEAR_LAYERS) - 3
EXPECTED_PRODUCT_VALUES = EXPECTED_PRODUCT_LAYERS * 512 * DECODE_TOKENS
EXPECTED_PRODUCT_MISSES = 3
SOURCE_ONLY_GUARD = (
    "IQ36_ROUTER_QKV_DELTA_DEVICE_SPARSE_OVERLAY_SOURCE is source-gate only"
)


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _has_candidate(routes: dict[str, Any], seq: int, disposition: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("disposition") == disposition
      for row in routes.get("candidate_history", [])
  )


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("decision") == decision
      and _num(row.get("seq_covered")) >= seq_covered
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", [])
  )


def _parse_stdout_json(run: dict[str, Any]) -> dict[str, Any]:
  stdout = str(run.get("stdout") or "").strip()
  if not stdout:
    return {}
  first_line = stdout.splitlines()[0]
  parsed = json.loads(first_line)
  return parsed if isinstance(parsed, dict) else {}


def _run_probe(args: argparse.Namespace, binary: str,
               remote_token_dir: str) -> dict[str, Any]:
  run_flags = [
      "--model", shlex.quote(args.model),
      "--token-dir", shlex.quote(remote_token_dir),
      "--case-id", "short_math_001",
      "--device-substring", "B390",
      "--repeat", "1",
      "--decode-tokens", str(DECODE_TOKENS),
      "--lm-head-threads", "16",
      "--shared-q4-runner",
      "--resident-q4-weights",
      "--resident-selected-q4-experts",
      "--resident-selected-q6-experts",
      "--resident-selected-q6-sorted-cache",
      "--resident-selected-q6-rowstripe",
      "--resident-selected-cache-topk", "16",
      "--resident-shared-q6-down",
      "--resident-full-attention-v-q6",
      "--resident-linear-q6-qkv",
      "--resident-q4-cpu-order-z",
      "--resident-linear-conv-weights",
      "--resident-linear-state",
      "--resident-postconv-delta-handoff",
      "--resident-norm-weights",
      "--resident-tail-output-rmsnorm-input",
      "--resident-gate-up-swiglu-handoff",
      "--resident-attention-front-handoff",
      "--resident-full-core-attention-front-handoff",
      "--gpu-router",
      "--gpu-lm-head-q6",
      "--opencl-double-swiglu",
  ]
  run_parts = [
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([
          "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE=1",
          "IQ36_ROUTER_QKV_DELTA_PRODUCT_CONSUMER_SOURCE=1",
          shlex.quote(binary),
          *run_flags,
      ]),
  ]
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(' && '.join(run_parts))}",
      args.timeout_s)


def _smoke_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  keys = [
      "schema_version",
      "case_id",
      "decode_continuation_output_tokens",
      "router_qkv_delta_layer_input_producer_source_enabled",
      "router_qkv_delta_layer_input_producer_source_layers",
      "router_qkv_delta_layer_input_producer_source_values",
      "router_qkv_delta_layer_input_producer_source_misses",
      "router_qkv_delta_layer_input_producer_source_ready",
      "router_qkv_delta_product_consumer_source_enabled",
      "router_qkv_delta_product_consumer_source_layers",
      "router_qkv_delta_product_consumer_source_values",
      "router_qkv_delta_product_consumer_source_misses",
      "router_qkv_delta_product_consumer_source_ready",
      "resident_tail_output_rmsnorm_input_enabled",
      "cpu_shadow_state_each_token_enabled",
      "cpu_shadow_ffn_input_each_token_enabled",
      "cpu_shadow_layer_input_layers",
      "cpu_shadow_attention_output_layers",
      "top1_matches_native",
      "top1_match_count",
      "required_checks_passed",
      "speedup_claims_allowed",
      "gpu_hybrid_decode_tok_s",
  ]
  return {key: smoke.get(key) for key in keys}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq311 = _load_json(args.seq311)
  compile_summary = seq311.get("compile_summary")
  compile_summary = compile_summary if isinstance(compile_summary, dict) else {}
  binary = compile_summary.get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq311 compile binary missing")
  token_cache = iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)
  probe_run = (
      _run_probe(args, binary, str(token_cache.get("dir")))
      if token_cache.get("ok") is True else
      {"returncode": 125, "stdout": "", "stderr": "token staging failed"}
  )
  smoke = _parse_stdout_json(probe_run)
  run_text = str(probe_run.get("stdout") or "") + str(probe_run.get("stderr") or "")

  checks = [
      {
          "name": "seq311_selected_product_consumer_probe_gate",
          "pass": (
              seq311.get("required_checks_passed") is True
              and seq311.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_product_consumer_probe_gate"
              and _has_candidate(
                  routes, 311,
                  "accept_qkv_delta_product_consumer_target_compile")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_product_consumer_probe_gate",
                  311)
          ),
      },
      {
          "name": "token_inputs_staged",
          "pass": token_cache.get("ok") is True,
          "detail": {
              "cache_hit": token_cache.get("hit"),
              "key": token_cache.get("key"),
              "dir": token_cache.get("dir"),
          },
      },
      {
          "name": "target_binary_emitted_probe_json",
          "pass": (
              smoke.get("schema_version")
              == "intel-qwen36-r2-gpu-decode-smoke-v0"
              and probe_run.get("returncode") in (0, 2)
          ),
          "detail": {
              "returncode": probe_run.get("returncode"),
              "stdout_bytes": len(str(probe_run.get("stdout") or "")),
              "stderr_bytes": len(str(probe_run.get("stderr") or "")),
          },
      },
      {
          "name": "producer_counters_ready",
          "pass": (
              smoke.get("router_qkv_delta_layer_input_producer_source_enabled")
              is True
              and smoke.get("router_qkv_delta_layer_input_producer_source_layers")
              == len(PRODUCER_LAYERS) * DECODE_TOKENS
              and smoke.get("router_qkv_delta_layer_input_producer_source_values")
              == EXPECTED_PRODUCER_VALUES
              and smoke.get("router_qkv_delta_layer_input_producer_source_misses")
              == 0
              and smoke.get("router_qkv_delta_layer_input_producer_source_ready")
              is True
          ),
          "detail": {
              "expected_values": EXPECTED_PRODUCER_VALUES,
              "smoke": _smoke_summary(smoke),
          },
      },
      {
          "name": "product_consumer_counters_ready_with_first_block_miss",
          "pass": (
              smoke.get("router_qkv_delta_product_consumer_source_enabled")
              is True
              and smoke.get("router_qkv_delta_product_consumer_source_layers")
              == EXPECTED_PRODUCT_LAYERS
              and smoke.get("router_qkv_delta_product_consumer_source_values")
              == EXPECTED_PRODUCT_VALUES
              and smoke.get("router_qkv_delta_product_consumer_source_misses")
              == EXPECTED_PRODUCT_MISSES
              and smoke.get("router_qkv_delta_product_consumer_source_ready")
              is True
          ),
          "detail": {
              "expected_layers": EXPECTED_PRODUCT_LAYERS,
              "expected_values": EXPECTED_PRODUCT_VALUES,
              "expected_misses": EXPECTED_PRODUCT_MISSES,
              "note": "Layers 0..2 have no prior resident producer handle in this source shape.",
          },
      },
      {
          "name": "product_consumer_probe_is_cpu_shadow_free",
          "pass": (
              smoke.get("cpu_shadow_state_each_token_enabled") is False
              and smoke.get("cpu_shadow_ffn_input_each_token_enabled") is False
              and smoke.get("cpu_shadow_layer_input_layers") == 0
              and smoke.get("cpu_shadow_attention_output_layers") == 0
          ),
      },
      {
          "name": "probe_keeps_greedy_top1_and_no_speed_claim",
          "pass": (
              smoke.get("top1_matches_native") is True
              and smoke.get("top1_match_count") == DECODE_TOKENS
              and smoke.get("speedup_claims_allowed") is False
          ),
      },
      {
          "name": "source_only_overlay_guard_not_hit",
          "pass": SOURCE_ONLY_GUARD not in run_text,
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq311_target_compile": _rel(args.seq311),
          "host": args.host,
          "model": args.model,
          "env_script": args.env_script,
          "binary": binary,
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "probe_run": {
          "cmd": probe_run.get("cmd"),
          "returncode": probe_run.get("returncode"),
          "stdout_bytes": len(str(probe_run.get("stdout") or "")),
          "stderr_bytes": len(str(probe_run.get("stderr") or "")),
      },
      "smoke_summary": _smoke_summary(smoke),
      "product_consumer_probe_passed": required,
      "checks": checks,
      "required_checks_passed": required,
      "decode_probe_allowed": required,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_qkv_delta_product_consumer_probe_with_first_block_miss"
          if required else
          "block_before_qkv_delta_product_consumer_decode_gate"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_product_consumer_decode_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_product_consumer_probe_fix_gate"
      ),
      "next_route_reason": (
          "The target product-consumer probe runs one token without CPU-shadow "
          "values, preserves greedy top-1, and emits resident producer plus "
          "product-consumer counters. The first three all-linear layers miss "
          "because this source has no prior producer handle for layers 0..2; "
          "record that explicitly and run the 8-token decode gate before any "
          "router distribution row or speed promotion."
          if required else
          "The product-consumer target probe did not pass; fix probe/source "
          "before decode or router distribution rows."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [
      row["name"] for row in metrics["checks"]
      if row.get("pass") is not True
  ]
  smoke = metrics["smoke_summary"]
  lines = [
      "# Router QKV Delta Product Consumer Probe Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- producer source values: `{smoke.get('router_qkv_delta_layer_input_producer_source_values')}`",
      f"- product consumer values: `{smoke.get('router_qkv_delta_product_consumer_source_values')}`",
      f"- product consumer misses: `{smoke.get('router_qkv_delta_product_consumer_source_misses')}`",
      f"- top1_matches_native: `{str(smoke.get('top1_matches_native')).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is a one-token guarded probe. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq311", type=Path, default=DEFAULT_SEQ311)
  parser.add_argument("--token-input-dir", type=Path,
                      default=DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "decode_probe_allowed": metrics["decode_probe_allowed"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
