#!/usr/bin/env python3
"""Gate opt-in decode use of the shared-device-Q8 linear-preconv bundle.

This is compile/top-1 evidence for the opt-in path, not benchmark evidence.
Seq75 added the same-runner primitive; this gate proves the generated decode
loop can consume LayerInputRmsNormRun.attn_norm_handle, enter the shared-Q8
preconv path, and produce the same top-1 token on the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-linear-preconv-shared-q8-decode-gate-v0"
DEFAULT_DECODE_SOURCE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_GPU_SOURCE = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_GENERATED_DECODE_SOURCE = (
    ROOT
    / "output/r2-gpu-linear-preconv-shared-q8-smoke-20260706Tseq76-r4Z/r2_gpu_decode_smoke.cpp"
)
DEFAULT_RUNTIME_RESULT = (
    ROOT / "output/r2-gpu-linear-preconv-shared-q8-smoke-20260706Tseq76-r4Z/result.json"
)
DEFAULT_SEQ75 = (
    ROOT / "output/linear-preconv-shared-q8-primitive-gate-20260706Tseq75Z/metrics.json"
)
DEFAULT_OUT_DIR = ROOT / "output/linear-preconv-shared-q8-decode-gate-20260706Tseq76Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(text: str, pattern: str, label: str) -> dict[str, Any]:
  match = re.search(pattern, text, re.S)
  return {
      "label": label,
      "present": match is not None,
      "line": text.count("\n", 0, match.start()) + 1 if match else None,
  }


def _bool_check(label: str, value: bool, detail: Any = None) -> dict[str, Any]:
  out: dict[str, Any] = {"label": label, "pass": bool(value)}
  if detail is not None:
    out["detail"] = detail
  return out


def _function_body(text: str, name: str) -> str:
  index = text.find(name)
  if index < 0:
    return ""
  brace = text.find("{", index)
  if brace < 0:
    return ""
  depth = 0
  for pos in range(brace, len(text)):
    char = text[pos]
    if char == "{":
      depth += 1
    elif char == "}":
      depth -= 1
      if depth == 0:
        return text[brace : pos + 1]
  return ""


def _check_named(result: dict[str, Any], name: str) -> bool:
  for check in result.get("checks", []):
    if check.get("name") == name:
      return check.get("pass") is True
  return False


def compute(args: argparse.Namespace) -> dict[str, Any]:
  decode_source = args.decode_source.read_text(encoding="utf-8")
  gpu_source = args.gpu_source.read_text(encoding="utf-8")
  generated = args.generated_decode_source.read_text(encoding="utf-8")
  runtime = _load_json(args.runtime_result)
  smoke = runtime.get("smoke", {})
  seq75 = _load_json(args.seq75_metrics)
  preconv_body = _function_body(generated, "RunGpuPreConvFront")
  shared_branch_index = preconv_body.find("if (use_shared_q8_preconv)")
  host_q8_index = preconv_body.find("QuantizeQ8KInputPlanes(attn_norm)")

  source_checks = [
      _check(
          decode_source,
          r"bool linear_preconv_shared_q8 = false;",
          "decode_args_has_shared_q8_flag",
      ),
      _check(
          decode_source,
          r'key == "--linear-preconv-shared-q8"',
          "decode_runtime_parses_shared_q8_flag",
      ),
      _check(
          decode_source,
          r'os\.environ\.get\("IQ36_LINEAR_PRECONV_SHARED_Q8"\)',
          "python_driver_uses_env_opt_in_without_argparse_growth",
      ),
      _check(
          decode_source,
          r"if args\.linear_preconv_shared_q8:\s*parts\.append\(\"--linear-preconv-shared-q8\"\)",
          "remote_run_command_forwards_shared_q8_flag",
      ),
      _check(
          decode_source,
          r'"linear_preconv_shared_q8": args\.linear_preconv_shared_q8',
          "manifest_records_shared_q8_flag",
      ),
      _check(
          generated,
          r"const std::vector<float>& attn_norm,\s*"
          r"std::uint64_t attn_norm_handle,\s*"
          r"bool use_shared_q8_preconv,",
          "generated_preconv_signature_consumes_attn_norm_handle",
      ),
      _check(
          generated,
          r"layer_input_gpu\.attn_norm,\s*"
          r"layer_input_gpu\.attn_norm_handle,\s*g_decode_linear_preconv_shared_q8",
          "generated_live_call_passes_attn_norm_handle_and_gate",
      ),
      _check(
          preconv_body,
          r"RunF32InputHandleSharedDeviceQ8ThenResidentRawQ6KConvStateAndResidentRawQ4KCpuOrder",
          "shared_branch_calls_q6_bundle",
      ),
      _check(
          preconv_body,
          r"RunF32InputHandleSharedDeviceQ8ThenResidentPackedQ4X8ConvStateAndResidentRawQ4KCpuOrder",
          "shared_branch_calls_q4_bundle",
      ),
      _check(
          generated,
          r"!g_decode_linear_preconv_shared_q8\s*&&\s*!\(g_decode_linear_abz_cpu_order_fused",
          "generic_z_rerun_excludes_shared_q8_path",
      ),
      _check(
          gpu_source,
          r"GpuRmsNormRun RunRmsNormHiddenResidentWeight[\s\S]*"
          r"RegisterF32BufferAlias\(\s*&rmsnorm_hidden_output_alias_handle_",
          "resident_weight_rmsnorm_registers_output_handle",
      ),
  ]
  ordering_checks = [
      _bool_check(
          "shared_branch_precedes_host_q8_quantize",
          shared_branch_index >= 0
          and host_q8_index >= 0
          and shared_branch_index < host_q8_index,
          {
              "shared_branch_index": shared_branch_index,
              "host_q8_index": host_q8_index,
          },
      ),
  ]
  runtime_checks = [
      _bool_check(
          "target_binary_built",
          _check_named(runtime, "target_binary_built"),
      ),
      _bool_check(
          "target_stdout_parsed",
          _check_named(runtime, "target_stdout_parsed"),
      ),
      _bool_check(
          "linear_preconv_shared_q8_enabled_in_smoke",
          smoke.get("linear_preconv_shared_q8_enabled") is True,
      ),
      _bool_check(
          "top1_matches_native",
          smoke.get("top1_matches_native") is True,
          {
              "gpu_generated_token_ids": smoke.get("gpu_generated_token_ids"),
              "native_reference_token_ids": smoke.get("native_reference_token_ids"),
          },
      ),
      _bool_check(
          "resident_q4_cpu_order_z_uses_shared_runner",
          smoke.get("resident_q4_cpu_order_z_runner_program_build_ms") == 0,
          smoke.get("resident_q4_cpu_order_z_runner_program_build_ms"),
      ),
      _bool_check(
          "linear_preconv_host_q8_bridge_removed",
          smoke.get("linear_preconv_kernel_profile_us", {}).get("host_q8_bridge") == 0
          and smoke.get("linear_preconv_wall_profile_ns", {}).get("input_q8") == 0,
          {
              "kernel": smoke.get("linear_preconv_kernel_profile_us", {}).get(
                  "host_q8_bridge"
              ),
              "wall": smoke.get("linear_preconv_wall_profile_ns", {}).get("input_q8"),
          },
      ),
      _bool_check(
          "separate_q4_cpu_order_z_rerun_removed",
          smoke.get("q4_cpu_order_z_layers") == 0,
          smoke.get("q4_cpu_order_z_layers"),
      ),
      _bool_check(
          "resident_alpha_beta_z_handles_hit",
          smoke.get("resident_q4_cpu_order_z_handles", 0) > 0
          and smoke.get("resident_q4_cpu_order_z_hits", 0) > 0,
          {
              "handles": smoke.get("resident_q4_cpu_order_z_handles"),
              "hits": smoke.get("resident_q4_cpu_order_z_hits"),
          },
      ),
  ]

  all_source_checks = all(check["present"] for check in source_checks)
  all_ordering_checks = all(check["pass"] for check in ordering_checks)
  all_runtime_checks = all(check["pass"] for check in runtime_checks)
  seq75_ready = bool(
      seq75.get("derived", {}).get("shared_device_q8_preconv_bundle_primitive_ready")
  )
  compile_top1_ready = (
      all_source_checks and all_ordering_checks and all_runtime_checks and seq75_ready
  )

  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "decode_source": {
              "path": _display_path(args.decode_source),
              "sha256": _sha256(args.decode_source),
          },
          "gpu_source": {
              "path": _display_path(args.gpu_source),
              "sha256": _sha256(args.gpu_source),
          },
          "generated_decode_source": {
              "path": _display_path(args.generated_decode_source),
              "sha256": _sha256(args.generated_decode_source),
          },
          "runtime_result": {
              "path": _display_path(args.runtime_result),
              "sha256": _sha256(args.runtime_result),
              "source_sha": runtime.get("source_sha"),
              "required_checks_passed": runtime.get("required_checks_passed"),
          },
          "seq75_metrics": {
              "path": _display_path(args.seq75_metrics),
              "sha256": _sha256(args.seq75_metrics),
          },
      },
      "source_checks": source_checks,
      "ordering_checks": ordering_checks,
      "runtime_checks": runtime_checks,
      "runtime_observation": {
          "target_binary_ran_check": _check_named(runtime, "target_binary_ran"),
          "required_checks_passed": smoke.get("required_checks_passed"),
          "topk_ids_match_native": smoke.get("topk_ids_match_native"),
          "gpu_hybrid_decode_tok_s": smoke.get("gpu_hybrid_decode_tok_s"),
          "linear_preconv_wall_profile_ns": smoke.get(
              "linear_preconv_wall_profile_ns"
          ),
      },
      "derived": {
          "all_source_checks_present": all_source_checks,
          "all_ordering_checks_pass": all_ordering_checks,
          "all_runtime_checks_pass": all_runtime_checks,
          "seq75_shared_q8_primitive_ready": seq75_ready,
          "shared_q8_decode_compile_top1_ready": compile_top1_ready,
      },
      "verdict": {
          "speedup_claims_allowed": False,
          "decode_speed_path_enabled": True,
          "reason": (
              "The opt-in shared-Q8 linear-preconv path compiled on target, "
              "entered the generated decode loop, removed the host-Q8 bridge "
              "and separate z rerun from linear preconv, and matched native "
              "top-1. Full free-run top-k remains diagnostic-only."
          ),
          "next_route": (
              "Measure paired speed/confirm and distribution only if the "
              "frontier stall gate admits this route; otherwise use the row "
              "as correctness/compile evidence and move to the next kernel-side cut."
          ),
      },
  }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  manifest = {
      "schema_version": f"{SCHEMA_VERSION}-manifest",
      "tool": "tools/intel-qwen36-linear-preconv-shared-q8-decode-gate.py",
      "workstream": WORKSTREAM,
      "artifact": _display_path(out_dir),
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  d = result["derived"]
  obs = result["runtime_observation"]
  lines = [
      "# Linear Preconv Shared-Q8 Decode Gate",
      "",
      "This is opt-in compile/top-1 evidence, not benchmark evidence.",
      "",
      f"- source checks present: `{str(d['all_source_checks_present']).lower()}`",
      f"- ordering checks pass: `{str(d['all_ordering_checks_pass']).lower()}`",
      f"- runtime checks pass: `{str(d['all_runtime_checks_pass']).lower()}`",
      f"- seq75 primitive ready: `{str(d['seq75_shared_q8_primitive_ready']).lower()}`",
      f"- compile/top-1 ready: `{str(d['shared_q8_decode_compile_top1_ready']).lower()}`",
      f"- target_binary_ran check: `{str(obs['target_binary_ran_check']).lower()}`",
      f"- required checks passed: `{str(obs['required_checks_passed']).lower()}`",
      f"- top-k ids match native: `{str(obs['topk_ids_match_native']).lower()}`",
      f"- observed tok/s: `{obs['gpu_hybrid_decode_tok_s']}`",
      "",
      result["verdict"]["reason"],
      "",
      result["verdict"]["next_route"],
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--decode-source", type=Path, default=DEFAULT_DECODE_SOURCE)
  parser.add_argument("--gpu-source", type=Path, default=DEFAULT_GPU_SOURCE)
  parser.add_argument(
      "--generated-decode-source", type=Path, default=DEFAULT_GENERATED_DECODE_SOURCE
  )
  parser.add_argument("--runtime-result", type=Path, default=DEFAULT_RUNTIME_RESULT)
  parser.add_argument("--seq75-metrics", type=Path, default=DEFAULT_SEQ75)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  result = compute(args)
  write_outputs(result, args.out_dir)
  return 0 if result["derived"]["shared_q8_decode_compile_top1_ready"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
