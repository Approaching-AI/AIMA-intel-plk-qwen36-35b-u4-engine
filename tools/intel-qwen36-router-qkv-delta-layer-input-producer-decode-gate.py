#!/usr/bin/env python3
"""Run an 8-token decode gate for the product-owned layer-input producer."""

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


SCHEMA_VERSION = (
    "intel-qwen36-router-qkv-delta-layer-input-producer-decode-gate-v0"
)
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ304 = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-implementation-probe-gate-20260708Tseq304Z"
    / "metrics.json"
)
DEFAULT_TOKEN_INPUT_DIR = (
    ROOT
    / "output/r1-engine-seed-prompt-input-check-20260627T155328Z/token-input"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-decode-gate-20260708Tseq305Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
HIDDEN_SIZE = 2048
DECODE_TOKENS = 8
EXPECTED_LAYERS = len(PRODUCER_LAYERS) * DECODE_TOKENS
EXPECTED_VALUES = EXPECTED_LAYERS * HIDDEN_SIZE


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


def _smoke_from_stdout(run: dict[str, Any]) -> dict[str, Any]:
  stdout = str(run.get("stdout") or "").strip()
  if not stdout:
    return {}
  parsed = json.loads(stdout.splitlines()[0])
  return parsed if isinstance(parsed, dict) else {}


def _run_decode(args: argparse.Namespace, binary: str,
                remote_token_dir: str) -> dict[str, Any]:
  flags = [
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
      "--resident-gate-up-swiglu-handoff",
      "--resident-attention-front-handoff",
      "--resident-full-core-attention-front-handoff",
      "--gpu-router",
      "--gpu-lm-head-q6",
      "--opencl-double-swiglu",
  ]
  remote_script = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([
          "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE=1",
          shlex.quote(binary),
          *flags,
      ]),
  ])
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(remote_script)}", args.timeout_s)


def _summary(smoke: dict[str, Any]) -> dict[str, Any]:
  keys = [
      "case_id",
      "decode_continuation_output_tokens",
      "gpu_hybrid_decode_tok_s",
      "router_qkv_delta_layer_input_producer_source_layers",
      "router_qkv_delta_layer_input_producer_source_values",
      "router_qkv_delta_layer_input_producer_source_misses",
      "router_qkv_delta_layer_input_producer_source_ready",
      "cpu_shadow_state_each_token_enabled",
      "cpu_shadow_ffn_input_each_token_enabled",
      "cpu_shadow_layer_input_layers",
      "cpu_shadow_attention_output_layers",
      "top1_matches_native",
      "top1_match_count",
      "greedy_prefix_match_count",
      "required_checks_passed",
      "speedup_claims_allowed",
  ]
  return {key: smoke.get(key) for key in keys}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq304 = _load_json(args.seq304)
  binary = seq304.get("inputs", {}).get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq304 binary missing")
  token_cache = iq36_local.ensure_cached_tokens(
      args.host, f"{args.remote_root}/cache", args.token_input_dir,
      args.timeout_s)
  run = (
      _run_decode(args, binary, str(token_cache.get("dir")))
      if token_cache.get("ok") is True else
      {"returncode": 125, "stdout": "", "stderr": "token staging failed"}
  )
  smoke = _smoke_from_stdout(run)
  checks = [
      {
          "name": "seq304_selected_decode_gate",
          "pass": (
              seq304.get("required_checks_passed") is True
              and seq304.get("selected_next_route")
              == "router_prompt_full_attention_residual_layer_input_producer_decode_gate"
              and _has_candidate(
                  routes, 304,
                  "accept_full_attention_residual_layer_input_producer_implementation_probe")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_residual_layer_input_producer_decode_gate",
                  304)
          ),
      },
      {
          "name": "target_binary_emitted_8_token_json",
          "pass": (
              smoke.get("schema_version")
              == "intel-qwen36-r2-gpu-decode-smoke-v0"
              and run.get("returncode") in (0, 2)
              and smoke.get("decode_continuation_output_tokens") == DECODE_TOKENS
          ),
          "detail": {
              "returncode": run.get("returncode"),
              "stdout_bytes": len(str(run.get("stdout") or "")),
              "stderr_bytes": len(str(run.get("stderr") or "")),
          },
      },
      {
          "name": "product_producer_counters_ready_for_8_tokens",
          "pass": (
              smoke.get("router_qkv_delta_layer_input_producer_source_layers")
              == EXPECTED_LAYERS
              and smoke.get("router_qkv_delta_layer_input_producer_source_values")
              == EXPECTED_VALUES
              and smoke.get("router_qkv_delta_layer_input_producer_source_misses")
              == 0
              and smoke.get("router_qkv_delta_layer_input_producer_source_ready")
              is True
          ),
          "detail": {
              "expected_layers": EXPECTED_LAYERS,
              "expected_values": EXPECTED_VALUES,
              "observed_layers": smoke.get(
                  "router_qkv_delta_layer_input_producer_source_layers"),
              "observed_values": smoke.get(
                  "router_qkv_delta_layer_input_producer_source_values"),
              "observed_misses": smoke.get(
                  "router_qkv_delta_layer_input_producer_source_misses"),
          },
      },
      {
          "name": "producer_source_is_cpu_shadow_free",
          "pass": (
              smoke.get("cpu_shadow_state_each_token_enabled") is False
              and smoke.get("cpu_shadow_ffn_input_each_token_enabled") is False
              and smoke.get("cpu_shadow_layer_input_layers") == 0
              and smoke.get("cpu_shadow_attention_output_layers") == 0
          ),
      },
      {
          "name": "greedy_decode_preserved_no_speed_claim",
          "pass": (
              smoke.get("top1_matches_native") is True
              and smoke.get("top1_match_count") == DECODE_TOKENS
              and smoke.get("greedy_prefix_match_count") == DECODE_TOKENS
              and smoke.get("speedup_claims_allowed") is False
          ),
          "detail": {
              "required_checks_passed": smoke.get("required_checks_passed"),
              "gpu_hybrid_decode_tok_s": smoke.get("gpu_hybrid_decode_tok_s"),
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq304_probe": _rel(args.seq304),
          "token_input_dir": _rel(args.token_input_dir),
          "host": args.host,
          "model": args.model,
          "binary": binary,
      },
      "token_cache": {
          "ok": token_cache.get("ok"),
          "hit": token_cache.get("hit"),
          "key": token_cache.get("key"),
          "dir": token_cache.get("dir"),
      },
      "probe_run": {
          "cmd": run.get("cmd"),
          "returncode": run.get("returncode"),
          "stdout_bytes": len(str(run.get("stdout") or "")),
          "stderr": run.get("stderr"),
      },
      "smoke_summary": _summary(smoke),
      "producer_counter_expectation": {
          "producer_layers": PRODUCER_LAYERS,
          "decode_tokens": DECODE_TOKENS,
          "hidden_size": HIDDEN_SIZE,
          "expected_layers": EXPECTED_LAYERS,
          "expected_values": EXPECTED_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "decode_correctness_passed": required,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_residual_layer_input_producer_decode_gate"
          if required else
          "reject_full_attention_residual_layer_input_producer_decode_gate"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_product_source_from_layer_input_producer_source_gate"
          if required else
          "router_prompt_full_attention_residual_layer_input_producer_decode_fix_gate"
      ),
      "next_route_reason": (
          "The product-owned producer source is stable across the 8-token "
          "decode row and preserves greedy correctness. Next wire the "
          "all-linear qkv-delta product source to consume these producer "
          "handles before router distribution or speed promotion."
          if required else
          "The 8-token producer decode gate failed; fix producer source "
          "correctness before any qkv-delta consumer or distribution row."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  summary = metrics["smoke_summary"]
  lines = [
      "# Router QKV Delta Layer-Input Producer Decode Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- source layers: `{summary.get('router_qkv_delta_layer_input_producer_source_layers')}`",
      f"- source values: `{summary.get('router_qkv_delta_layer_input_producer_source_values')}`",
      f"- source misses: `{summary.get('router_qkv_delta_layer_input_producer_source_misses')}`",
      f"- top1 match count: `{summary.get('top1_match_count')}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is an 8-token decode/correctness gate. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq304", type=Path, default=DEFAULT_SEQ304)
  parser.add_argument("--token-input-dir", type=Path, default=DEFAULT_TOKEN_INPUT_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--timeout-s", type=int, default=900)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "decode_correctness_passed": metrics["decode_correctness_passed"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
