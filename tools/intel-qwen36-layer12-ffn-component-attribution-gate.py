#!/usr/bin/env python3
"""Run one token to attribute the layer-12 FFN source boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


ONE_TOKEN_TOOL = (
    ROOT / "tools/intel-qwen36-fused-exact-linear-projection-one-token-probe-gate.py"
)
SCHEMA_VERSION = "intel-qwen36-layer12-ffn-component-attribution-gate-v0"
CURRENT_ROUTE = (
    "router_prompt_distribution_layer12_ffn_component_attribution_gate"
)
LOCAL_MATH_ROUTE = (
    "router_prompt_distribution_layer12_ffn_local_math_source_gate"
)
INPUT_SOURCE_ROUTE = (
    "router_prompt_distribution_layer12_ffn_input_source_gate"
)
SOURCE_LAYER = 12
COSINE_THRESHOLD = 0.9999
MAX_GPU_CPU_MATH_ABS = 2.0e-5


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


ONE = _load_module(ONE_TOKEN_TOOL, "iq36_layer12_ffn_one_token_gate")


def _run_probe(args: argparse.Namespace, binary: str,
               remote_token_dir: str) -> dict[str, Any]:
  run_flags = [
      "--model", args.model,
      "--token-dir", remote_token_dir,
      "--case-id", "router_math_reason_001",
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
      "--full-attention-state-diff",
      "--teacher-force-native-tokens",
      "--distribution-ladder",
      "--diagnostic-layer-range", "12:13",
      "--diagnostic-token-limit", "1",
  ]
  env = [
      "IQ36_OPENCL_NO_QUEUE_PROFILING=1",
      "IQ36_SELECTED_SHARED_Q4_GATEUP_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q4_DOWN_COMBINED=1",
      "IQ36_SELECTED_SHARED_Q6_DOWN_COMBINED=1",
      "IQ36_DEFER_FFN_DOWN_FINISH_BUNDLE=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16=1",
      "IQ36_ATTENTION_FRONT_OUTPUT_PROJECTION_ROWBLOCK16_LAYERS="
      + ONE.ROWBLOCK16_26MASK,
      "IQ36_INPUT_RMSNORM_SERIAL_REDUCTION_LAYERS=" + ONE.LINEAR_LAYER_CSV,
      "IQ36_LINEAR_OUTPUT_PROJECTION_CPU_ORDER_LAYERS=" + ONE.LINEAR_LAYER_CSV,
      "IQ36_LINEAR_OUTPUT_PROJECTION_ROWBLOCK16_CPUORDER_FINALIZE=1",
      "IQ36_LINEAR_FINAL_DEVICE_Q8_HANDOFF=1",
  ]
  command = " ".join(
      [*env, shlex.quote(binary),
       *(shlex.quote(value) for value in run_flags)])
  shell = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      command,
  ])
  return iq36_local.run_target(
      args.host, f"bash -lc {shlex.quote(shell)}", args.timeout_s)


def _step(smoke: dict[str, Any], table: str) -> dict[str, Any]:
  rows = smoke.get(table, [])
  if not isinstance(rows, list):
    return {}
  for row in rows:
    if isinstance(row, dict) and row.get("token_index") == 0:
      return row
  return {}


def _layer(step: dict[str, Any], layer: int) -> dict[str, Any]:
  rows = step.get("layers", [])
  if not isinstance(rows, list):
    return {}
  for row in rows:
    if isinstance(row, dict) and row.get("layer") == layer:
      return row
  return {}


def _component_summary(smoke: dict[str, Any]) -> dict[str, Any]:
  source = _layer(_step(smoke, "ffn_component_source_diff_by_step"),
                  SOURCE_LAYER)
  live = _layer(_step(smoke, "ffn_live_math_diff_by_step"), SOURCE_LAYER)
  boundary = _layer(_step(smoke, "layer_boundary_diff_by_step"), SOURCE_LAYER)
  return {
      "layer": SOURCE_LAYER,
      "ffn_input_max_abs_diff": source.get("ffn_input_max_abs_diff"),
      "ffn_norm_native_gpu_max_abs_diff": source.get(
          "ffn_norm_max_abs_diff"),
      "router_ids_match_native": source.get("router_ids_match"),
      "router_weight_max_abs_diff": source.get("router_weight_max_abs_diff"),
      "selected_gate_up_native_gpu_max_abs_diff": source.get(
          "selected_gate_up_max_abs_diff"),
      "selected_swiglu_native_gpu_max_abs_diff": source.get(
          "selected_swiglu_max_abs_diff"),
      "selected_down_native_gpu_max_abs_diff": source.get(
          "selected_down_max_abs_diff"),
      "shared_down_native_gpu_max_abs_diff": source.get(
          "shared_down_max_abs_diff"),
      "ffn_out_native_gpu_max_abs_diff": source.get("ffn_out_max_abs_diff"),
      "layer_output_native_gpu_max_abs_diff": boundary.get(
          "output_max_abs_diff"),
      "ffn_norm_gpu_vs_cpu_max_abs_diff": live.get(
          "ffn_norm_gpu_vs_cpu_max_abs_diff"),
      "ffn_norm_gpu_vs_cpu_cosine": live.get(
          "ffn_norm_gpu_vs_cpu_cosine"),
      "selected_down_gpu_vs_cpu_max_abs_diff": live.get(
          "selected_down_gpu_vs_cpu_max_abs_diff"),
      "selected_down_gpu_vs_cpu_cosine": live.get(
          "selected_down_gpu_vs_cpu_cosine"),
      "shared_down_gpu_vs_cpu_max_abs_diff": live.get(
          "shared_down_gpu_vs_cpu_max_abs_diff"),
      "shared_down_gpu_vs_cpu_cosine": live.get(
          "shared_down_gpu_vs_cpu_cosine"),
      "layer_output_from_live_downs_vs_gpu_output_max_abs_diff": live.get(
          "layer_output_from_live_downs_vs_gpu_output_max_abs_diff"),
      "layer_output_from_live_downs_vs_gpu_output_cosine": live.get(
          "layer_output_from_live_downs_vs_gpu_output_cosine"),
      "gpu_output_vs_cpu_ffn_available": live.get(
          "gpu_output_vs_cpu_ffn_available"),
      "gpu_output_vs_cpu_ffn_max_abs_diff": live.get(
          "gpu_output_vs_cpu_ffn_max_abs_diff"),
      "gpu_output_vs_cpu_ffn_cosine": live.get(
          "gpu_output_vs_cpu_ffn_cosine"),
  }


def _number(value: Any, default: float) -> float:
  return float(value) if isinstance(value, (int, float)) else default


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = ONE._load(args.routes)  # noqa: SLF001
  predecessor = ONE._load(args.predecessor)  # noqa: SLF001
  contract = predecessor.get("diagnostic_contract", {})
  binary = str(contract.get("binary", ""))
  binary_key = contract.get("binary_key")
  reused_target_run = args.reuse_existing
  if reused_target_run:
    run = ONE._load(args.out_dir / "raw-run.json")  # noqa: SLF001
    smoke = ONE._load(args.out_dir / "smoke.json")  # noqa: SLF001
    prior_metrics = ONE._load(args.out_dir / "metrics.json")  # noqa: SLF001
    cache_check = next(
        (row for row in prior_metrics.get("checks", [])
         if isinstance(row, dict)
         and row.get("name") == "router_token_inputs_are_cached"), {})
    token_cache = dict(cache_check.get("detail", {}))
    token_cache["ok"] = cache_check.get("pass") is True
  else:
    token_cache = iq36_local.ensure_cached_tokens(
        args.host, f"{args.remote_root}/cache", args.token_input_dir,
        args.timeout_s)
    run = (
        _run_probe(args, binary, str(token_cache.get("dir")))
        if token_cache.get("ok") is True and binary else
        {"returncode": 125, "stdout": "", "stderr": "token staging failed"}
    )
    smoke = ONE._parse_stdout(run)  # noqa: SLF001
  smoke_summary = ONE._summary(smoke)  # noqa: SLF001
  component = _component_summary(smoke)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  if not reused_target_run:
    iq36_local.write_json(args.out_dir / "raw-run.json", run)
    if smoke:
      iq36_local.write_json(args.out_dir / "smoke.json", smoke)

  route_selects = (
      predecessor.get("required_checks_passed") is True
      and predecessor.get("one_token_diagnostic_allowed") is True
      and predecessor.get("selected_next_route") == CURRENT_ROUTE
      and contract.get("decode_tokens") == 1
      and contract.get("diagnostic_layer_range") == "12:13"
      and contract.get("diagnostic_token_limit") == 1
      and binary_key == args.expected_binary_key
      and ONE._has_candidate(routes, 594, CURRENT_ROUTE)  # noqa: SLF001
      and ONE._has_switch(  # noqa: SLF001
          routes, 594,
          "select_router_prompt_distribution_layer12_ffn_component_"
          "attribution_gate"))
  execution_complete = (
      run.get("returncode") in (0, 2)
      and smoke_summary.get("schema_version")
      == "intel-qwen36-r2-gpu-decode-smoke-v0"
      and smoke_summary.get("case_id") == "router_math_reason_001"
      and smoke_summary.get("decode_tokens_per_session") == 1
      and smoke.get("diagnostic_layer_start") == 12
      and smoke.get("diagnostic_layer_end") == 13
      and smoke.get("diagnostic_token_limit") == 1)
  selectors_pass = (
      smoke_summary.get("linear_output_projection_rowblock16_cpuorder_finalize")
      is True
      and smoke_summary.get("linear_output_projection_cpu_order_layers")
      == ONE.LINEAR_LAYERS
      and smoke_summary.get("input_rmsnorm_serial_reduction_layers")
      == ONE.LINEAR_LAYERS
      and smoke_summary.get("linear_final_device_q8_handoff_enabled") is True)
  trace_complete = (
      component.get("layer") == SOURCE_LAYER
      and isinstance(component.get("ffn_input_max_abs_diff"), (int, float))
      and component.get("gpu_output_vs_cpu_ffn_available") is True
      and isinstance(
          component.get("gpu_output_vs_cpu_ffn_max_abs_diff"), (int, float))
      and isinstance(component.get("gpu_output_vs_cpu_ffn_cosine"),
                     (int, float))
      and isinstance(component.get("layer_output_native_gpu_max_abs_diff"),
                     (int, float)))
  local_math_mismatch = (
      _number(component.get("gpu_output_vs_cpu_ffn_max_abs_diff"), 0.0)
      > MAX_GPU_CPU_MATH_ABS
      or _number(component.get("gpu_output_vs_cpu_ffn_cosine"), 1.0)
      < COSINE_THRESHOLD)
  measurement_complete = (
      route_selects and token_cache.get("ok") is True and execution_complete
      and selectors_pass and trace_complete)
  selected_next = (
      LOCAL_MATH_ROUTE if local_math_mismatch else INPUT_SOURCE_ROUTE)
  if not measurement_complete:
    selected_next = CURRENT_ROUTE
  checks = [
      {"name": "seq594_selected_one_layer12_ffn_trace",
       "pass": route_selects},
      {"name": "router_token_inputs_are_cached",
       "pass": token_cache.get("ok") is True,
       "detail": {
           "reused_target_run": reused_target_run,
           "hit": token_cache.get("hit"),
           "key": token_cache.get("key"),
           "dir": token_cache.get("dir"),
       }},
      {"name": "target_binary_emitted_one_token_with_layer12_window",
       "pass": execution_complete,
       "detail": {"returncode": run.get("returncode")}},
      {"name": "all_linear_exact_selectors_remained_unchanged",
       "pass": selectors_pass},
      {"name": "layer12_component_and_same_input_live_math_trace_complete",
       "pass": trace_complete,
       "detail": component},
  ]
  if measurement_complete and local_math_mismatch:
    disposition = "select_layer12_ffn_local_math_source"
    reason = (
        "The layer-12 GPU FFN output differs materially from a CPU FFN "
        "recompute on the same live GPU input. Source-gate the local FFN "
        "component chain before any implementation or distribution row."
    )
  elif measurement_complete:
    disposition = "reject_layer12_ffn_local_math_select_input_source"
    reason = (
        "The layer-12 GPU FFN output matches the CPU FFN recompute on the "
        "same live GPU input inside the existing `2e-5`/`0.9999` source "
        "threshold. Attribute the incoming layer-12 FFN state next; do not "
        "change FFN kernels."
    )
  else:
    disposition = "block_incomplete_layer12_ffn_component_attribution"
    reason = (
        "Repair token staging, exact selector activation, or the layer-12 "
        "trace without changing decode math, then rerun this one row."
    )
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "inputs": {
          "routes": ONE._rel(args.routes),  # noqa: SLF001
          "predecessor": ONE._rel(args.predecessor),  # noqa: SLF001
          "token_input_dir": ONE._rel(args.token_input_dir),  # noqa: SLF001
          "host": args.host,
          "model": args.model,
          "binary": binary,
          "binary_key": binary_key,
          "target_run_reused": reused_target_run,
      },
      "run": {
          "returncode": run.get("returncode"),
          "stdout_bytes": len(str(run.get("stdout") or "")),
          "stderr_bytes": len(str(run.get("stderr") or "")),
      },
      "smoke_summary": smoke_summary,
      "component_summary": component,
      "thresholds": {
          "gpu_cpu_ffn_max_abs": MAX_GPU_CPU_MATH_ABS,
          "gpu_cpu_ffn_cosine_min": COSINE_THRESHOLD,
      },
      "checks": checks,
      "measurement_complete": measurement_complete,
      "required_checks_passed": measurement_complete,
      "local_ffn_math_mismatch": (
          local_math_mismatch if measurement_complete else None),
      "decode_probe_allowed": False,
      "router_distribution_allowed": False,
      "speed_probe_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": disposition,
      "selected_next_route": selected_next,
      "next_route_reason": reason,
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  component = metrics["component_summary"]
  lines = [
      f"# Seq{metrics['sequence']} Layer-12 FFN Component Attribution",
      "",
      f"- measurement_complete: `{str(metrics['measurement_complete']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- local_ffn_math_mismatch: `{metrics['local_ffn_math_mismatch']}`",
      f"- same-input GPU-vs-CPU FFN output max abs: "
      f"`{component.get('gpu_output_vs_cpu_ffn_max_abs_diff')}`",
      f"- same-input GPU-vs-CPU FFN output cosine: "
      f"`{component.get('gpu_output_vs_cpu_ffn_cosine')}`",
      f"- native-vs-GPU full layer output max abs: "
      f"`{component.get('layer_output_native_gpu_max_abs_diff')}`",
      f"- failed_checks: `{failed}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is one-token correctness-source evidence only, not a speed row.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=595)
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--predecessor", type=Path,
      default=ROOT / "output/seq594-fused-exact-linear-projection-route-close-gate-20260710Tseq594Z/metrics.json")
  parser.add_argument(
      "--token-input-dir", type=Path,
      default=ROOT / "output/r2-gpu-acceptance-matrix-router-both-rowblock16-26mask-20260708Tseq222Z/token-input")
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / "output/seq595-layer12-ffn-component-attribution-gate-20260710Tseq595Z")
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default="local")
  parser.add_argument(
      "--model",
      default="/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
  parser.add_argument(
      "--env-script",
      default="/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default="/home/intel/intel-qwen36-gpu")
  parser.add_argument("--expected-binary-key",
                      default="5553e5fbb1dc5aea9ae2d0fe")
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument(
      "--reuse-existing", action="store_true",
      help="Reclassify the existing raw-run/smoke artifact without target work")
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "measurement_complete": metrics["measurement_complete"],
      "disposition": metrics["disposition"],
      "local_ffn_math_mismatch": metrics["local_ffn_math_mismatch"],
      "target_run_reused": metrics["inputs"]["target_run_reused"],
      "component_summary": metrics["component_summary"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": ONE._rel(args.out_dir),  # noqa: SLF001
  }, sort_keys=True))
  return 0 if metrics["measurement_complete"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
