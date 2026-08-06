#!/usr/bin/env python3
"""Classify seq497 GPU attention-norm math to the shared RMSNorm scale kernel."""

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
SEQ497_GATE = (
    ROOT
    / "tools/intel-qwen36-seq497-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-linear-z-source-gap-gate.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-seq498-attn-norm-math-shared-scale-kernel-gate-v0"
)

DEFAULT_ROUTES = ACTIVE / "routes-ledger.json"
DEFAULT_SEQ497 = (
    ROOT
    / "output/seq497-projection-q8-input-sensitivity-linear-input-source-ffn-delta-selected-gate-up-input-sensitivity-ffn-norm-input-sensitivity-attention-output-projection-q8-input-sensitivity-linear-z-source-gap-gate-20260709Tseq497Z"
    / "metrics.json"
)
DEFAULT_OPENCL = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
DEFAULT_RUNNER = ROOT / "engine/src/gpu_q4x8_matvec.cpp"
DEFAULT_SMOKE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_OUT_DIR = (
    ROOT
    / "output/seq498-attn-norm-math-shared-scale-kernel-gate-20260709Tseq498Z"
)

COSINE_THRESHOLD = 0.9999
MATERIAL_ABS_EPS = 1.0e-12


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module {name}: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


SEQ497 = _load_module(SEQ497_GATE, "iq36_seq497_gate")
CURRENT_ROUTE = SEQ497.ATTN_NORM_MATH_ROUTE
SCALE_KERNEL_ROUTE = CURRENT_ROUTE.replace(
    "_linear_z_source_attn_norm_math_gap_gate",
    "_linear_z_source_attn_norm_scale_kernel_gap_gate")


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
  return float(value) if isinstance(value, (int, float)) else default


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


def _body_for(source: str, name: str) -> str:
  start = source.find(name)
  if start < 0:
    return ""
  brace = source.find("{", start)
  if brace < 0:
    return ""
  depth = 0
  for pos in range(brace, len(source)):
    ch = source[pos]
    if ch == "{":
      depth += 1
    elif ch == "}":
      depth -= 1
      if depth == 0:
        return source[brace:pos + 1]
  return ""


def _all_contains(source: str, snippets: list[str]) -> bool:
  return all(snippet in source for snippet in snippets)


def _method_shape(source: str, name: str) -> dict[str, Any]:
  body = _body_for(source, name)
  return {
      "found": bool(body),
      "uses_scale_kernel": "kernel_rmsnorm_hidden_scale_" in body,
      "uses_apply_kernel": "kernel_rmsnorm_hidden_apply_scale_" in body,
      "enqueues_scale": (
          "clEnqueueNDRangeKernel(\n                queue_, kernel_rmsnorm_hidden_scale_"
          in body
          or "clEnqueueNDRangeKernel(queue_, kernel_rmsnorm_hidden_scale_"
          in body),
      "enqueues_apply": (
          "clEnqueueNDRangeKernel(\n                queue_, kernel_rmsnorm_hidden_apply_scale_"
          in body
          or "clEnqueueNDRangeKernel(queue_, kernel_rmsnorm_hidden_apply_scale_"
          in body),
      "uses_standalone_serial_kernel": "kernel_rmsnorm_hidden_" in body
          and "kernel_rmsnorm_hidden_scale_" not in body,
  }


def _source_shape(opencl_source: str, runner_source: str,
                  smoke_source: str) -> dict[str, Any]:
  serial_kernel = _body_for(opencl_source, "__kernel void rms_norm_hidden_f32")
  scale_kernel = _body_for(opencl_source, "__kernel void rms_norm_hidden_scale_f32")
  apply_kernel = _body_for(opencl_source,
                           "__kernel void rms_norm_hidden_apply_scale_f32")
  methods = {
      name: _method_shape(runner_source, name)
      for name in [
          "GpuRmsNormRun RunRmsNormHidden(",
          "GpuRmsNormRun RunRmsNormHiddenResidentWeight(",
          "GpuRmsNormRun RunRmsNormHiddenResidentInputResidentWeight(",
      ]
  }
  layer_input_shared_runner = _all_contains(smoke_source, [
      "g_decode_shared_q4_runner != nullptr",
      "RunRmsNormHiddenResidentInputResidentWeight(",
      "RunRmsNormHiddenResidentWeight(",
      "RunRmsNormHidden(",
      "return run;",
  ])
  cpu_sqrt_flag_scope = _all_contains(smoke_source, [
      "--opencl-cpu-sqrt-norm",
      "Diagnostic: use CPU-shaped 1/sqrt normalization scales instead of OpenCL rsqrt.",
      "if opencl_cpu_sqrt_norm:",
      "const float scale = rsqrt(mean_square + epsilon);",
      "const float scale = 1.0f / sqrt(mean_square + epsilon);",
  ])
  return {
      "standalone_serial_kernel": {
          "found": bool(serial_kernel),
          "serial_sum": "for (uint i = 0; i < hidden_size; ++i)" in serial_kernel,
          "uses_rsqrt_scale": (
              "const float scale = rsqrt(mean_square + epsilon);" in serial_kernel),
      },
      "shared_scale_kernel": {
          "found": bool(scale_kernel),
          "local_partial_reduction": _all_contains(scale_kernel, [
              "__local float partial[256];",
              "const uint chunk = (hidden_size + local_size - 1U) / local_size;",
              "partial[lid] = sum_squares;",
              "for (uint i = 0; i < local_size; ++i)",
              "total += partial[i];",
          ]),
          "scale_out_rsqrt": (
              "scale_out[0] = rsqrt(mean_square + epsilon);" in scale_kernel),
      },
      "shared_apply_kernel": {
          "found": bool(apply_kernel),
          "applies_single_scale": (
              "output[index] = input[index] * scale_in[0] * weight[index];"
              in apply_kernel),
      },
      "runner_methods": methods,
      "layer_input_rmsnorm_uses_shared_runner": layer_input_shared_runner,
      "cpu_sqrt_flag_scope": {
          "exists": cpu_sqrt_flag_scope,
          "targets_standalone_hidden_scale_text": (
              "const float scale = rsqrt(mean_square + epsilon);" in smoke_source),
          "has_no_shared_scale_out_rewrite": (
              "scale_out[0] = rsqrt(mean_square + epsilon);" not in smoke_source
              and "scale_out[0] = 1.0f / sqrt(mean_square + epsilon);"
              not in smoke_source),
      },
  }


def _source_shape_pass(shape: dict[str, Any]) -> bool:
  serial = shape["standalone_serial_kernel"]
  scale = shape["shared_scale_kernel"]
  apply = shape["shared_apply_kernel"]
  methods = shape["runner_methods"]
  return (
      serial["found"]
      and serial["serial_sum"]
      and serial["uses_rsqrt_scale"]
      and scale["found"]
      and scale["local_partial_reduction"]
      and scale["scale_out_rsqrt"]
      and apply["found"]
      and apply["applies_single_scale"]
      and all(
          method["found"]
          and method["uses_scale_kernel"]
          and method["uses_apply_kernel"]
          and method["enqueues_scale"]
          and method["enqueues_apply"]
          and not method["uses_standalone_serial_kernel"]
          for method in methods.values()
      )
      and shape["layer_input_rmsnorm_uses_shared_runner"]
      and shape["cpu_sqrt_flag_scope"]["exists"]
      and shape["cpu_sqrt_flag_scope"]["targets_standalone_hidden_scale_text"]
      and shape["cpu_sqrt_flag_scope"]["has_no_shared_scale_out_rewrite"]
  )


def _seq497_evidence(seq497: dict[str, Any]) -> dict[str, Any]:
  c = seq497.get("min_cosines", {})
  a = seq497.get("max_abs_diffs", {})
  q8 = seq497.get("q8_mismatches", {})
  return {
      "final_projection_cosine": _num(c.get("final_projection"), 1.0),
      "z_projection_cosine": _num(c.get("z_projection"), 1.0),
      "boundary_input_abs": _num(a.get("boundary_input")),
      "attn_norm_from_gpu_input_abs": _num(a.get("attn_norm_from_gpu_input")),
      "gpu_attn_norm_vs_cpu_cosine": _num(
          c.get("gpu_attn_norm_vs_cpu"), 1.0),
      "gpu_attn_norm_vs_cpu_abs": _num(a.get("gpu_attn_norm_vs_cpu")),
      "qkv_from_gpu_attn_norm_abs": _num(a.get("qkv_from_gpu_attn_norm")),
      "z_from_gpu_attn_norm_abs": _num(a.get("z_from_gpu_attn_norm")),
      "gpu_z_vs_cpu_cosine": _num(c.get("gpu_z_vs_cpu"), 1.0),
      "gpu_z_vs_cpu_abs": _num(a.get("gpu_z_vs_cpu")),
      "final_q8_qs": int(_num(q8.get("final_qs"))),
      "z_q8_qs": int(_num(q8.get("z_qs"))),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load_json(args.routes)
  seq497 = _load_json(args.seq497)
  classification = str(seq497.get("diagnostic_classification"))
  scale_kernel_classification = classification.replace(
      "_attn_norm_math_gap", "_attn_norm_scale_kernel_gap")
  preconditions_pass = (
      seq497.get("required_checks_passed") is True
      and seq497.get("selected_next_route") == CURRENT_ROUTE
      and seq497.get("attn_norm_math_gap") is True
      and classification.endswith("_attn_norm_math_gap")
      and _has_candidate(routes, 497, str(seq497.get("disposition")))
      and _has_switch(routes, f"select_{CURRENT_ROUTE}", 497)
  )
  evidence = _seq497_evidence(seq497)
  seq497_reproduced = (
      evidence["final_projection_cosine"] >= COSINE_THRESHOLD
      and evidence["z_projection_cosine"] >= COSINE_THRESHOLD
      and evidence["boundary_input_abs"] == 0.0
      and evidence["attn_norm_from_gpu_input_abs"] == 0.0
      and evidence["gpu_attn_norm_vs_cpu_cosine"] >= COSINE_THRESHOLD
      and evidence["gpu_attn_norm_vs_cpu_abs"] > MATERIAL_ABS_EPS
      and evidence["qkv_from_gpu_attn_norm_abs"] > MATERIAL_ABS_EPS
      and evidence["z_from_gpu_attn_norm_abs"] > MATERIAL_ABS_EPS
      and evidence["gpu_z_vs_cpu_cosine"] >= COSINE_THRESHOLD
      and evidence["gpu_z_vs_cpu_abs"] == 0.0
      and evidence["final_q8_qs"] > 0
      and evidence["final_q8_qs"] == evidence["z_q8_qs"]
  )
  shape = _source_shape(
      args.opencl.read_text(encoding="utf-8"),
      args.runner.read_text(encoding="utf-8"),
      args.smoke.read_text(encoding="utf-8"),
  )
  source_shape_pass = _source_shape_pass(shape)
  scale_kernel_gap = preconditions_pass and seq497_reproduced and source_shape_pass
  checks = [
      {"name": "seq497_attn_norm_math_gate", "pass": preconditions_pass},
      {"name": "seq497_attn_norm_math_evidence_reproduced",
       "pass": seq497_reproduced,
       "detail": evidence},
      {"name": "shared_rmsnorm_scale_kernel_shape",
       "pass": source_shape_pass,
       "detail": shape},
      {"name": "attn_norm_math_classified_to_shared_scale_kernel",
       "pass": scale_kernel_gap,
       "detail": {
           "scale_kernel_gap": scale_kernel_gap,
           "selected_next_route": SCALE_KERNEL_ROUTE,
       }},
  ]
  required = all(bool(row.get("pass")) for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "inputs": {
          "routes": _rel(args.routes),
          "seq497": _rel(args.seq497),
          "opencl_source": _rel(args.opencl),
          "runner_source": _rel(args.runner),
          "decode_smoke_source": _rel(args.smoke),
      },
      "checks": checks,
      "required_checks_passed": required,
      "diagnostic_classification": (
          scale_kernel_classification if required
          else f"{classification}_unclassified"),
      "attn_norm_scale_kernel_gap": scale_kernel_gap,
      "seq497_evidence": evidence,
      "source_shape": shape,
      "speedup_claims_allowed": False,
      "disposition": (
          f"accept_{classification}_shared_scale_kernel_classification"
          if required else
          f"block_{classification}_shared_scale_kernel_classification"
      ),
      "selected_next_route": SCALE_KERNEL_ROUTE if required else CURRENT_ROUTE,
      "next_route_reason": (
          "Seq497 proves the live input and CPU RMSNorm replay are exact while "
          "the GPU attention norm drifts and z/qkv inherit that drift. The "
          "active layer-input path routes through the shared Q4 runner's "
          "two-stage hidden RMSNorm scale/apply kernels, whose scale stage uses "
          "a local-partial reduction and writes rsqrt(mean_square + epsilon). "
          "The existing --opencl-cpu-sqrt-norm diagnostic rewrites the standalone "
          "hidden RMSNorm scale text, not the shared scale_out kernel. Root the "
          "shared RMSNorm scale kernel next."
          if required else
          "Shared RMSNorm scale-kernel attribution is incomplete; keep the "
          "attention-norm math gate open."
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
  e = metrics["seq497_evidence"]
  lines = [
      "# Seq498 Attention-Norm Math Shared Scale-Kernel Gate",
      "",
      f"- required_checks_passed: `{str(metrics['required_checks_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- diagnostic_classification: `{metrics['diagnostic_classification']}`",
      f"- final/z projection cosines: `{e['final_projection_cosine']}` / `{e['z_projection_cosine']}`",
      f"- boundary/cpu-norm/gpu-norm/z/gpu-z max abs: `{e['boundary_input_abs']}` / `{e['attn_norm_from_gpu_input_abs']}` / `{e['gpu_attn_norm_vs_cpu_abs']}` / `{e['z_from_gpu_attn_norm_abs']}` / `{e['gpu_z_vs_cpu_abs']}`",
      f"- qkv_from_gpu_attn_norm max abs: `{e['qkv_from_gpu_attn_norm_abs']}`",
      f"- q8 final/z qs: `{e['final_q8_qs']}` / `{e['z_q8_qs']}`",
      f"- attn_norm_scale_kernel_gap: `{str(metrics['attn_norm_scale_kernel_gap']).lower()}`",
      f"- failed_checks: `{failed}`",
      f"- speedup_claims_allowed: `{str(metrics['speedup_claims_allowed']).lower()}`",
      "",
      metrics["next_route_reason"],
      "",
      "This is distribution/correctness evidence only. It is not a speed claim.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
  parser.add_argument("--seq497", type=Path, default=DEFAULT_SEQ497)
  parser.add_argument("--opencl", type=Path, default=DEFAULT_OPENCL)
  parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
  parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "diagnostic_classification": metrics["diagnostic_classification"],
      "disposition": metrics["disposition"],
      "out_dir": _rel(args.out_dir),
      "required_checks_passed": metrics["required_checks_passed"],
      "selected_next_route": metrics["selected_next_route"],
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
