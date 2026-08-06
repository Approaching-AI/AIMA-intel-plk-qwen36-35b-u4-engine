#!/usr/bin/env python3
"""Target-compile the full-attention core/history implementation path."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    "intel-qwen36-resident-hidden-state-carrier-full-attention-core-history-"
    "implementation-target-compile-gate-v0"
)

DEFAULT_FRONTIER = ACTIVE / "frontier.json"
DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ147 = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-source-gate-20260707Tseq147Z"
    / "metrics.json"
)
DEFAULT_GENERATE_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-generate-only-20260707Tseq147Z"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/resident-hidden-state-carrier-full-attention-core-history-implementation-target-compile-gate-20260707Tseq148Z"
)
DEFAULT_HOST = "local"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-gpu"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _load_decode_smoke_module() -> Any:
  path = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
  spec = importlib.util.spec_from_file_location("iq36_decode_smoke", path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load decode smoke module: {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _has_switch(routes: dict[str, Any], decision: str, seq_covered: int) -> bool:
  for row in routes.get("switch_decisions", []):
    if (
        isinstance(row, dict)
        and row.get("decision") == decision
        and _num(row.get("seq_covered")) >= seq_covered
        and row.get("resolved") is True
    ):
      return True
  return False


def _frontier_state(frontier: dict[str, Any]) -> dict[str, Any]:
  anchor = frontier.get("goal_anchor")
  anchor = anchor if isinstance(anchor, dict) else {}
  no_progress = frontier.get("no_progress")
  no_progress = no_progress if isinstance(no_progress, dict) else {}
  noise = no_progress.get("noise")
  noise = noise if isinstance(noise, dict) else {}
  return {
      "current_best_tps": _num(anchor.get("current_best_tps")),
      "floor_tps": _num(anchor.get("same_host_vulkan_floor_tps")),
      "noise_rel": _num(noise.get("rel")),
      "runs_since_significant_improvement": no_progress.get(
          "runs_since_significant_improvement"),
      "hard_stall_breached": no_progress.get("hard_stall_breached"),
      "review_recorded_for_current_best": no_progress.get(
          "review_recorded_for_current_best"),
  }


def _last_command(result: dict[str, Any], key: str) -> dict[str, Any]:
  value = result.get(key)
  return value if isinstance(value, dict) else {}


def compute(args: argparse.Namespace) -> dict[str, Any]:
  frontier = _load_json(args.frontier)
  routes = _load_json(args.routes)
  seq147 = _load_json(args.seq147)
  generate_result_path = args.generate_dir / "result.json"
  generated_cpp_path = args.generate_dir / "r2_gpu_decode_smoke.cpp"
  generate_result = _load_json(generate_result_path)
  generated_cpp = generated_cpp_path.read_text(encoding="utf-8")
  smoke = _load_decode_smoke_module()
  cache_root = f"{args.remote_root}/cache"

  compile_result = iq36_local.ensure_cached_binary(
      args.host,
      cache_root,
      smoke.SOURCE_FILES,
      ROOT,
      generated_cpp_path,
      "tests/r2_gpu_decode_smoke.cpp",
      lambda remote_dir: smoke.build_command(remote_dir, args.env_script),
      "build/r2-gpu-decode-smoke",
      args.timeout_s,
  )
  frontier_state = _frontier_state(frontier)
  build = _last_command(compile_result, "build")
  publish = _last_command(compile_result, "publish")

  manifest_checks = {
      "generate_only": generate_result.get("generate_only") is True,
      "resident_hidden_state_carrier": (
          generate_result.get("resident_hidden_state_carrier") is True),
      "resident_hidden_state_carrier_preconv_bundle": (
          generate_result.get("resident_hidden_state_carrier_preconv_bundle")
          is True),
      "resident_hidden_state_carrier_selected_shared_tail": (
          generate_result.get("resident_hidden_state_carrier_selected_shared_tail")
          is True),
      "resident_hidden_state_carrier_layer_output_handle_loop": (
          generate_result.get("resident_hidden_state_carrier_layer_output_handle_loop")
          is True),
      "resident_hidden_state_carrier_full_attention_qkv_handle": (
          generate_result.get("resident_hidden_state_carrier_full_attention_qkv_handle")
          is True),
      "resident_hidden_state_carrier_full_attention_core_history_handle": (
          generate_result.get(
              "resident_hidden_state_carrier_full_attention_core_history_handle")
          is True),
      "resident_full_core_attention_front_handoff": (
          generate_result.get("resident_full_core_attention_front_handoff") is True),
      "resident_full_attention_v_q6": (
          generate_result.get("resident_full_attention_v_q6") is True),
      "no_smoke_json": not (args.generate_dir / "smoke.json").exists(),
  }
  generated_checks = {
      "source_only_runtime_guard_absent": (
          "full-attention core/history resident boundary is source-gate only"
          not in generated_cpp),
      "qk_norm_rope_path_present": (
          "RunFullAttentionQkNormRopeFromHandles" in generated_cpp),
      "history_append_path_present": (
          "BuildFullAttentionHistoryFromHandle" in generated_cpp),
      "full_core_from_handles_path_present": (
          "RunFullAttentionCoreGateThenResidentPackedQ4X8ResidualRmsNormFromHandles"
          in generated_cpp),
      "contract_ready_helper_present": (
          "DecodeFullAttentionCoreHistoryResidentBoundaryReady" in generated_cpp),
  }

  checks = [
      {
          "name": "seq147_selected_implementation_target_compile_gate",
          "pass": (
              seq147.get("required_checks_passed") is True
              and seq147.get("selected_next_route")
              == "resident_hidden_state_carrier_full_attention_core_history_implementation_target_compile_gate"
              and _has_switch(
                  routes,
                  "accept_full_attention_core_history_implementation_source_switch_to_target_compile_gate",
                  147,
              )
          ),
      },
      {
          "name": "generate_only_manifest_is_implementation_not_decode_row",
          "pass": all(manifest_checks.values()),
          "detail": manifest_checks,
      },
      {
          "name": "generated_cpp_implementation_shape_present",
          "pass": all(generated_checks.values()),
          "detail": generated_checks,
      },
      {
          "name": "target_binary_compile_or_cache_hit_passed",
          "pass": compile_result.get("ok") is True,
          "detail": {
              "cache_hit": compile_result.get("hit"),
              "key": compile_result.get("key"),
              "binary": compile_result.get("binary"),
              "build_returncode": build.get("returncode"),
              "publish_returncode": publish.get("returncode"),
          },
      },
      {
          "name": "frontier_still_below_floor_no_speed_claim",
          "pass": frontier_state["current_best_tps"] < frontier_state["floor_tps"],
          "detail": frontier_state,
      },
  ]
  required_checks_passed = all(bool(row.get("pass")) for row in checks)

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "frontier": _rel(args.frontier),
          "routes": _rel(args.routes),
          "seq147_source_gate": _rel(args.seq147),
          "generate_only_result": _rel(generate_result_path),
          "generated_cpp": _rel(generated_cpp_path),
          "host": args.host,
          "env_script": args.env_script,
          "remote_root": args.remote_root,
      },
      "frontier": frontier_state,
      "compile": compile_result,
      "compile_summary": {
          "ok": compile_result.get("ok"),
          "cache_hit": compile_result.get("hit"),
          "key": compile_result.get("key"),
          "binary": compile_result.get("binary"),
          "build_returncode": build.get("returncode"),
          "publish_returncode": publish.get("returncode"),
      },
      "generated": generated_checks,
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "decode_probe_allowed": required_checks_passed,
      "component_probe_allowed": False,
      "speedup_claims_allowed": False,
      "disposition": (
          "accept_full_attention_core_history_implementation_target_compile"
          if required_checks_passed
          else "reject_full_attention_core_history_implementation_target_compile"
      ),
      "selected_next_route": (
          "resident_hidden_state_carrier_full_attention_core_history_implementation_decode_gate"
          if required_checks_passed
          else "resident_hidden_state_carrier_full_attention_core_history_implementation_compile_fix_gate"
      ),
      "next_route_reason": (
          "The implementation path compiles on the target without launching a "
          "token row. The next admissible unit is a correctness/distribution "
          "decode gate; no speed row is allowed before that evidence."
          if required_checks_passed
          else "The implementation path did not compile on the target. Fix "
               "compile errors before correctness or speed decode."
      ),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  lines = [
      "# Resident Hidden-State Carrier Full-Attention Core/History Implementation Target Compile Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- decode_probe_allowed: `{str(metrics['decode_probe_allowed']).lower()}`",
      "",
      metrics["next_route_reason"],
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq147", type=Path, default=DEFAULT_SEQ147)
  parser.add_argument("--generate-dir", type=Path, default=DEFAULT_GENERATE_DIR)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
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
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
