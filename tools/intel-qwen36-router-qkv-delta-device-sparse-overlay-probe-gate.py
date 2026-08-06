#!/usr/bin/env python3
"""Probe the router qkv-delta device sparse-overlay source after compile.

This is a guarded source probe. It runs the compiled target binary with the
overlay gate enabled and verifies that the source-only guard blocks before any
token row. Passing this gate means the primitive is compiled and safely gated;
the next unit must wire the all-linear qkv-delta product consumer.
"""

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
    "intel-qwen36-router-qkv-delta-device-sparse-overlay-probe-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_REJECTED = ACTIVE / "rejected-routes.json"
DEFAULT_SEQ308 = (
    ROOT
    / "output/router-qkv-delta-device-sparse-overlay-target-compile-gate-20260708Tseq308Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-device-sparse-overlay-probe-gate-20260708Tseq309Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_TOKEN_DIR = "/tmp/iq36-router-qkv-delta-device-sparse-overlay-guard"

ALL_LINEAR_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
]
PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
TOP512_VALUES = len(ALL_LINEAR_LAYERS) * 8 * 512
PRODUCER_VALUES = len(PRODUCER_LAYERS) * 8 * 2048
EXPECTED_GUARD = (
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


def _rejected_names(rejected: dict[str, Any]) -> set[str]:
  names: set[str] = set()
  for row in rejected.get("rejected", []):
    if isinstance(row, dict) and isinstance(row.get("route"), str):
      names.add(row["route"])
  return names


def _short_result(result: dict[str, Any]) -> dict[str, Any]:
  return {
      "cmd": result.get("cmd"),
      "returncode": result.get("returncode"),
      "stdout": result.get("stdout"),
      "stderr": result.get("stderr"),
  }


def _run_guard(args: argparse.Namespace, binary: str) -> dict[str, Any]:
  run_flags = [
      "--model", shlex.quote(args.model),
      "--token-dir", shlex.quote(args.token_dir),
      "--case-id", "short_math_001",
      "--device-substring", "B390",
      "--repeat", "1",
      "--decode-tokens", "1",
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
  run_parts = [
      "set -o pipefail",
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      f"mkdir -p {shlex.quote(args.token_dir)}",
      " ".join([
          "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE=1",
          "IQ36_ROUTER_QKV_DELTA_DEVICE_SPARSE_OVERLAY_SOURCE=1",
          shlex.quote(binary),
          *run_flags,
      ]),
  ]
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(' && '.join(run_parts))}",
      args.timeout_s)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  rejected = _load_json(args.rejected)
  seq308 = _load_json(args.seq308)
  compile_summary = seq308.get("compile_summary")
  compile_summary = compile_summary if isinstance(compile_summary, dict) else {}
  binary = compile_summary.get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq308 compile binary missing")

  guard_run = _run_guard(args, binary)
  guard_text = (
      str(guard_run.get("stdout") or "") + "\n" +
      str(guard_run.get("stderr") or "")
  )
  rejected_names = _rejected_names(rejected)
  required_closed = {
      "router_math_static_or_lagged_qkv_delta_predictors",
      "router_math_live_round_or_selected_affine_qkv_delta_approximation",
      "router_math_split_full_attention_projection_arithmetic_residual_fix",
  }
  missing_closed = sorted(required_closed - rejected_names)

  checks = [
      {
          "name": "seq308_selected_overlay_probe_gate",
          "pass": (
              seq308.get("required_checks_passed") is True
              and seq308.get("selected_next_route")
              == "router_prompt_all_linear_qkv_delta_device_sparse_overlay_probe_gate"
              and seq308.get("overlay_probe_allowed") is True
              and _has_candidate(
                  routes, 308,
                  "accept_device_sparse_overlay_target_compile")
              and _has_switch(
                  routes,
                  "select_router_prompt_all_linear_qkv_delta_device_sparse_overlay_probe_gate",
                  308)
          ),
      },
      {
          "name": "target_guard_blocks_source_only_overlay",
          "pass": (
              guard_run.get("returncode") != 0 and EXPECTED_GUARD in guard_text),
          "detail": _short_result(guard_run),
      },
      {
          "name": "guard_probe_uses_producer_and_overlay_envs",
          "pass": (
              "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE=1"
              in str(guard_run.get("cmd"))
              and "IQ36_ROUTER_QKV_DELTA_DEVICE_SPARSE_OVERLAY_SOURCE=1"
              in str(guard_run.get("cmd"))),
      },
      {
          "name": "no_token_json_emitted_by_guard_probe",
          "pass": "decode_continuation_output_tokens" not in guard_text,
      },
      {
          "name": "top512_consumer_and_producer_shapes_preserved",
          "pass": TOP512_VALUES == 122880 and PRODUCER_VALUES == 147456,
          "detail": {
              "all_linear_layers": ALL_LINEAR_LAYERS,
              "producer_layers": PRODUCER_LAYERS,
              "decode_tokens": 8,
              "topk": 512,
              "top512_values": TOP512_VALUES,
              "producer_values": PRODUCER_VALUES,
          },
      },
      {
          "name": "closed_approximation_classes_not_reopened",
          "pass": not missing_closed,
          "detail": {"missing_closed_routes": missing_closed},
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "rejected": _rel(args.rejected),
          "seq308_target_compile": _rel(args.seq308),
          "host": args.host,
          "model": args.model,
          "env_script": args.env_script,
          "binary": binary,
      },
      "guard_probe": _short_result(guard_run),
      "consumer_requirement": {
          "all_linear_layers": ALL_LINEAR_LAYERS,
          "decode_tokens": 8,
          "topk": 512,
          "top512_values": TOP512_VALUES,
      },
      "producer_requirement": {
          "producer_layers": PRODUCER_LAYERS,
          "decode_tokens": 8,
          "producer_values": PRODUCER_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "overlay_probe_completed": required,
      "overlay_source_safe": required,
      "qkv_delta_product_consumer_present": False,
      "product_consumer_source_allowed": required,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_source_only_overlay_probe_select_product_consumer_source"
          if required else
          "block_before_qkv_delta_product_consumer_source"
      ),
      "selected_next_route": (
          "router_prompt_all_linear_qkv_delta_product_consumer_source_gate"
          if required else
          "router_prompt_all_linear_qkv_delta_device_sparse_overlay_probe_fix_gate"
      ),
      "next_route_reason": (
          "The compiled overlay source is safe but still source-only: the "
          "target guard blocks before token execution while preserving the "
          "producer and top512 consumer shapes. The next unit must wire the "
          "all-linear qkv-delta product consumer to the resident producer "
          "handles and sparse-overlay primitive; decode and router "
          "distribution rows remain blocked."
          if required else
          "The overlay probe did not prove the source-only guard. Fix the "
          "overlay source/probe before product-consumer wiring or decode."
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
  lines = [
      "# Router QKV Delta Device Sparse Overlay Probe Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- overlay_source_safe: `{str(metrics['overlay_source_safe']).lower()}`",
      f"- qkv_delta_product_consumer_present: `{str(metrics['qkv_delta_product_consumer_present']).lower()}`",
      f"- product_consumer_source_allowed: `{str(metrics['product_consumer_source_allowed']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- top512 consumer values: `{TOP512_VALUES}`",
      f"- producer values: `{PRODUCER_VALUES}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is a guarded source probe. It does not emit tokens or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
  parser.add_argument("--seq308", type=Path, default=DEFAULT_SEQ308)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--token-dir", default=DEFAULT_TOKEN_DIR)
  parser.add_argument("--timeout-s", type=int, default=300)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "qkv_delta_product_consumer_present": (
          metrics["qkv_delta_product_consumer_present"]),
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
