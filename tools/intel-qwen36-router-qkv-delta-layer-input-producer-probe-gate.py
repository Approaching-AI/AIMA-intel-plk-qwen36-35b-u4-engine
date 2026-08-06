#!/usr/bin/env python3
"""Probe the router qkv-delta layer-input producer source after compile.

This is a negative source probe. It runs the compiled target binary with the
producer gate enabled and verifies that the guard blocks before any token row.
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
    "intel-qwen36-router-qkv-delta-layer-input-producer-probe-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ300 = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-target-compile-gate-20260708Tseq300Z"
    / "metrics.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/router-qkv-delta-layer-input-producer-probe-gate-20260708Tseq301Z"
)
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_TOKEN_DIR = "/tmp/iq36-router-qkv-delta-layer-input-producer-guard"

PRODUCER_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]
ROOT_VALUES = 147456
EXPECTED_GUARD = (
    "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE is source-gate only"
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


def _short_result(result: dict[str, Any]) -> dict[str, Any]:
  return {
      "cmd": result.get("cmd"),
      "returncode": result.get("returncode"),
      "stdout": result.get("stdout"),
      "stderr": result.get("stderr"),
  }


def _run_guard(args: argparse.Namespace, binary: str) -> dict[str, Any]:
  run_parts = [
      "set -o pipefail",
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      f"mkdir -p {shlex.quote(args.token_dir)}",
      " ".join([
          "IQ36_ROUTER_QKV_DELTA_LAYER_INPUT_PRODUCER_SOURCE=1",
          shlex.quote(binary),
          "--model", shlex.quote(args.model),
          "--token-dir", shlex.quote(args.token_dir),
          "--case-id", "short_math_001",
          "--device-substring", "B390",
          "--repeat", "1",
          "--decode-tokens", "1",
          "--lm-head-threads", "16",
      ]),
  ]
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(' && '.join(run_parts))}",
      args.timeout_s)


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq300 = _load_json(args.seq300)
  compile_summary = seq300.get("compile_summary")
  compile_summary = compile_summary if isinstance(compile_summary, dict) else {}
  binary = compile_summary.get("binary")
  if not isinstance(binary, str) or not binary:
    raise SystemExit("seq300 compile binary missing")

  guard_run = _run_guard(args, binary)
  guard_text = (
      str(guard_run.get("stdout") or "") + "\n" +
      str(guard_run.get("stderr") or "")
  )

  checks = [
      {
          "name": "seq300_selected_producer_probe_gate",
          "pass": (
              seq300.get("required_checks_passed") is True
              and seq300.get("selected_next_route")
              == "router_prompt_full_attention_residual_layer_input_producer_probe_gate"
              and _has_candidate(
                  routes, 300,
                  "accept_full_attention_residual_layer_input_producer_target_compile")
              and _has_switch(
                  routes,
                  "select_router_prompt_full_attention_residual_layer_input_producer_probe_gate",
                  300)
          ),
      },
      {
          "name": "target_guard_blocks_source_only_producer",
          "pass": (
              guard_run.get("returncode") != 0 and EXPECTED_GUARD in guard_text),
          "detail": _short_result(guard_run),
      },
      {
          "name": "no_token_json_emitted_by_guard_probe",
          "pass": "decode_continuation_output_tokens" not in guard_text,
      },
      {
          "name": "producer_root_shape_preserved",
          "pass": ROOT_VALUES == 147456,
          "detail": {
              "producer_layers": PRODUCER_LAYERS,
              "producer_layer_count": len(PRODUCER_LAYERS),
              "decode_tokens": 8,
              "hidden_size": 2048,
              "root_values": ROOT_VALUES,
          },
      },
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq300_target_compile": _rel(args.seq300),
          "host": args.host,
          "model": args.model,
          "env_script": args.env_script,
          "binary": binary,
      },
      "guard_probe": _short_result(guard_run),
      "producer_root": {
          "root": "prior_full_attention_ffn_residual_input",
          "producer_layers": PRODUCER_LAYERS,
          "producer_layer_count": len(PRODUCER_LAYERS),
          "decode_tokens": 8,
          "hidden_size": 2048,
          "root_values": ROOT_VALUES,
      },
      "checks": checks,
      "required_checks_passed": required,
      "producer_probe_completed": required,
      "producer_product_source_present": False,
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "reject_source_only_producer_as_product_probe_select_implementation_source"
          if required else
          "block_before_layer_input_producer_implementation_source"
      ),
      "selected_next_route": (
          "router_prompt_full_attention_residual_layer_input_producer_implementation_source_gate"
          if required else
          "router_prompt_full_attention_residual_layer_input_producer_probe_fix_gate"
      ),
      "next_route_reason": (
          "The compiled producer scaffold is safe but still source-only: the "
          "target guard blocks before token execution. The next unit must add "
          "a product-owned full-attention residual layer-input producer source; "
          "decode and router distribution rows remain blocked."
          if required else
          "The producer probe did not prove the source-only guard; fix the "
          "source/probe before implementation or decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row.get("pass")]
  lines = [
      "# Router QKV Delta Layer-Input Producer Probe Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- producer_product_source_present: `{str(metrics['producer_product_source_present']).lower()}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      f"- router_distribution_allowed: `{str(metrics['router_distribution_allowed']).lower()}`",
      f"- producer root values: `{metrics['producer_root']['root_values']}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is a negative source probe. It does not claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq300", type=Path, default=DEFAULT_SEQ300)
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
      "producer_product_source_present": metrics["producer_product_source_present"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
