#!/usr/bin/env python3
"""Run ADR 0010's exact grouped-sparse Q4_K routed-MoE killer gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-onednn-grouped-q4k-moe-component-gate-v1"
BASE_GATE_PATH = ROOT / "tools/intel-qwen36-onednn-q4k-bucket-component-gate.py"
COMPONENT_SOURCE = ROOT / "engine/tools/onednn_grouped_q4k_moe_component.cpp"
DEFAULT_CAPTURE = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-20260711Tseq646cleanZ/"
    "raw/capture")
DEFAULT_ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")


def load_base_gate() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_onednn_q4k_gate", BASE_GATE_PATH)
  if spec is None or spec.loader is None:
    raise SystemExit(f"could not import {BASE_GATE_PATH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base_gate()


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=BASE.DEFAULT_MODEL)
  parser.add_argument("--census", type=Path, default=BASE.DEFAULT_CENSUS)
  parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
  parser.add_argument("--tensor-index", type=Path,
                      default=BASE.DEFAULT_TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path,
                      default=BASE.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=BASE.DEFAULT_CXX)
  parser.add_argument("--onednn-source", type=Path,
                      default=BASE.DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path,
                      default=DEFAULT_ONEDNN_BUILD)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=11)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.warmup <= 0 or args.repeat <= 0 or args.timeout_s <= 0:
    parser.error("warmup, repeat, and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/onednn-grouped-q4k-moe-component-gate-{stamp}"
  return args


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def compare_checks(prefix: str, comparison: dict[str, Any],
                   count: int) -> list[dict[str, Any]]:
  mismatch_count = comparison.get("mismatch_count")
  max_abs_diff = float(comparison.get("max_abs_diff", float("inf")))
  return [
      check(f"all_{count}_{prefix}_values_compared",
            comparison.get("compared_value_count") == count),
      check(f"zero_{prefix}_values_above_5e_3",
            mismatch_count == 0 and max_abs_diff <= 5e-3,
            max_abs_diff=max_abs_diff, mismatch_count=mismatch_count),
  ]


def disposition(correctness_passed: bool, performance_passed: bool) -> str:
  if correctness_passed and performance_passed:
    return "admit_grouped_sparse_exact_q4k_for_native_kernel_extraction"
  if not correctness_passed and not performance_passed:
    return "reject_grouped_sparse_exact_q4k_on_correctness_and_cap"
  if not correctness_passed:
    return "reject_grouped_sparse_exact_q4k_on_correctness"
  return "reject_grouped_sparse_exact_q4k_above_complete_cap"


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  required_paths = [
      args.model,
      args.census / "result.json",
      args.census / "layer-shapes.jsonl",
      args.census / "router-assignments.jsonl",
      args.capture / "tensor-dumps.jsonl",
      args.tensor_index,
      args.env_script,
      args.cxx,
      args.onednn_source,
      args.onednn_build,
      args.onednn_build / "src/libdnnl.so",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      BASE_GATE_PATH,
      COMPONENT_SOURCE,
      ROOT / "engine/tools/onednn_q4k_bucket_component.cpp",
  ]
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))
  if BASE.sha256_file(args.model) != BASE.MODEL_SHA256:
    raise SystemExit("locked model hash mismatch")

  created_at = iso_now()
  census_result, shape, assignments = BASE.selected_shape(args.census)
  budget = BASE.derive_budget(shape, True)
  tensors = BASE.tensor_rows(args.tensor_index)
  metadata, payloads = BASE.captured_payloads(args.capture, True)
  topk_name = f"ffn_moe_topk-{BASE.LAYER}"
  topk_stride = int(metadata[topk_name]["nb"][1])
  router_ids_match = (
      BASE.captured_router_ids(payloads[topk_name], topk_stride) ==
      assignments["expert_ids_by_token"])
  source_commit = BASE.git_output(args.onednn_source, "rev-parse", "HEAD")
  source_dirty = bool(BASE.git_output(args.onednn_source, "status", "--porcelain"))

  binary = raw_dir / "onednn-grouped-q4k-moe-component"
  build_result = BASE.shell_run([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", str(COMPONENT_SOURCE),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(binary),
  ], args.env_script, args.timeout_s)
  BASE.write_run_logs(raw_dir, "build", build_result)

  command = [
      str(binary), "--model", str(args.model),
      "--weight-offset", str(tensors["gate_up"]["absolute_offset"]),
      "--weight-bytes", str(tensors["gate_up"]["nbytes"]),
      "--input", str(payloads[f"attn_post_norm-{BASE.LAYER}"]),
      "--topk", str(payloads[topk_name]),
      "--topk-stride", str(topk_stride),
      "--oracle", str(payloads[f"ffn_moe_swiglu-{BASE.LAYER}"]),
      "--warmup", str(args.warmup), "--repeat", str(args.repeat),
      "--kernel-cap-us", str(budget["kernel_cap_us"]),
      "--down-weight-offset", str(tensors["down"]["absolute_offset"]),
      "--down-weight-bytes", str(tensors["down"]["nbytes"]),
      "--router-weights",
      str(payloads[f"ffn_moe_weights_norm-{BASE.LAYER}"]),
      "--down-oracle", str(payloads[f"ffn_moe_down-{BASE.LAYER}"]),
      "--moe-oracle", str(payloads[f"ffn_moe_out-{BASE.LAYER}"]),
  ]
  component_result = (
      BASE.shell_run(command, args.env_script, args.timeout_s)
      if build_result["returncode"] == 0 else
      {"command": command, "returncode": 125, "stderr": "build failed",
       "stdout": "", "timed_out": False})
  BASE.write_run_logs(raw_dir, "component", component_result)
  probe = BASE.parse_probe(component_result)

  evidence_checks = [
      check("locked_census_gate_passed",
            census_result.get("required_checks_passed") is True),
      check("pinned_onednn_source_commit",
            source_commit == BASE.ONEDNN_COMMIT,
            observed=source_commit, required=BASE.ONEDNN_COMMIT),
      check("pinned_onednn_source_clean", not source_dirty),
      check("locked_capture_payload_hashes_match", True),
      check("captured_router_ids_match_seq639", router_ids_match),
      check("component_build_passed", build_result["returncode"] == 0),
      check("component_execution_completed",
            component_result["returncode"] in (0, 2) and bool(probe)),
      check("runtime_onednn_hash_matches_source",
            probe.get("onednn_version", {}).get("hash") == BASE.ONEDNN_COMMIT),
      check("arc_b390_selected", "B390" in str(probe.get("device_name"))),
      check("real_grouped_shape_preserved",
            probe.get("active_experts") == shape["active_expert_count"] and
            probe.get("assignment_count") == shape["assignment_count"] and
            probe.get("max_group_size") == 361),
      check("three_grouped_microkernels_selected",
            probe.get("implementations_pass") is True and
            len(probe.get("implementations", [])) == 3 and
            all("grouped_gemm:micro" in str(value)
                for value in probe.get("implementations", []))),
      check("exact_floating_q4k_residual_selected",
            probe.get("exact_floating_residual") is True),
      check("locked_f16_grouped_value_type",
            probe.get("source_type") == "f16"),
      check("lossless_active_and_resident_q4_repack",
            probe.get("repack_pass") is True and
            probe.get("active_q4_code_count") == 698_351_616 and
            probe.get("resident_q4_code_count") == 805_306_368 and
            probe.get("repack_mismatch_count") == 0),
      check("speedup_claims_forbidden", True),
  ]
  swiglu = probe.get("compare", {})
  weighted_down = probe.get("weighted_down_compare", {})
  routed_output = probe.get("moe_compare", {})
  correctness_checks = [
      check("exact_q4k_component_correctness_passed",
            probe.get("correctness_pass") is True),
      *compare_checks("swiglu", swiglu, 4_194_304),
      *compare_checks("weighted_down", weighted_down, 16_777_216),
      *compare_checks("routed_output", routed_output, 2_097_152),
  ]
  performance_checks = [
      check("complete_runtime_below_cap",
            probe.get("performance_pass") is True and
            float(probe.get("minimum_us", float("inf"))) <=
            float(budget["kernel_cap_us"])),
  ]
  evidence_passed = all(row["pass"] for row in evidence_checks)
  correctness_passed = all(row["pass"] for row in correctness_checks)
  performance_passed = all(row["pass"] for row in performance_checks)
  required_passed = evidence_passed and correctness_passed and performance_passed
  route_disposition = disposition(correctness_passed, performance_passed)
  cap_fraction = (
      float(probe["minimum_us"]) / float(budget["kernel_cap_us"])
      if "minimum_us" in probe else None)

  runtime_boundary = {
      "excluded": ["one-time resident all-256-expert Q4_K repack"],
      "included": [
          "expert-major F32-to-F16 gather and exact group-32 input sums",
          "two grouped F16-by-U4 gate/up microkernels",
          "floating Q4_K affine-min residual and F16 SwiGLU",
          "one grouped F16-by-U4 down microkernel",
          "floating down residual and normalized router weighting",
          "deterministic inverse scatter, submission, and final queue drain",
      ],
      "reason": "ADR 0010 complete layer-27 killer boundary",
  }
  result = {
      "budget": budget,
      "case_id": BASE.CASE_ID,
      "checks": evidence_checks + correctness_checks + performance_checks,
      "correctness_checks_passed": correctness_passed,
      "created_at": created_at,
      "disposition": route_disposition,
      "evidence_checks_passed": evidence_passed,
      "git": BASE.git_state(),
      "layer": BASE.LAYER,
      "performance_checks_passed": performance_passed,
      "probe": probe,
      "required_checks_passed": required_passed,
      "runtime_boundary": runtime_boundary,
      "schema_version": SCHEMA_VERSION,
      "sources": {
          "capture": str(args.capture.relative_to(ROOT)),
          "capture_payload_sha256": BASE.ROUTED_PAYLOAD_SHA256,
          "census": str(args.census.relative_to(ROOT)),
          "component": str(COMPONENT_SOURCE.relative_to(ROOT)),
          "component_sha256": BASE.sha256_file(COMPONENT_SOURCE),
          "model_path": str(args.model),
          "model_sha256": BASE.MODEL_SHA256,
          "onednn_build": str(args.onednn_build),
          "onednn_commit": source_commit,
          "onednn_source": str(args.onednn_source),
      },
      "speedup_claims_allowed": False,
      "tensors": tensors,
      "tile_tokens": BASE.TILE_TOKENS,
      "workstream": WORKSTREAM,
  }
  BASE.write_json(out_dir / "result.json", result)
  BASE.write_json(out_dir / "correctness.json", {
      "checks": evidence_checks + correctness_checks,
      "correctness_checks_passed": correctness_passed,
      "evidence_checks_passed": evidence_passed,
      "routed_output_comparison": routed_output,
      "swiglu_comparison": swiglu,
      "weighted_down_comparison": weighted_down,
  })
  BASE.write_json(out_dir / "capture-metadata.json", {
      "case_id": BASE.CASE_ID,
      "layer": BASE.LAYER,
      "payload_sha256": BASE.ROUTED_PAYLOAD_SHA256,
      "router_ids_match_seq639": router_ids_match,
      "tensors": list(metadata.values()),
      "tile_tokens": BASE.TILE_TOKENS,
  })
  BASE.write_json(out_dir / "smoothness.json", {
      "applicable": False,
      "reason": "single real 1024-token grouped routed-MoE killer boundary",
  })
  BASE.write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "git": result["git"],
      "schema_version": SCHEMA_VERSION,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  metrics = [
      {"metric": "complete_component_minimum_us",
       "value": probe.get("minimum_us")},
      {"metric": "complete_component_median_us",
       "value": probe.get("median_us")},
      {"metric": "component_cap_us", "value": budget["kernel_cap_us"]},
      {"metric": "cap_fraction", "value": cap_fraction},
      {"metric": "swiglu_max_abs_diff", "value": swiglu.get("max_abs_diff")},
      {"metric": "swiglu_mismatch_count",
       "value": swiglu.get("mismatch_count")},
      {"metric": "weighted_down_max_abs_diff",
       "value": weighted_down.get("max_abs_diff")},
      {"metric": "routed_output_max_abs_diff",
       "value": routed_output.get("max_abs_diff")},
      {"metric": "required_checks_passed", "value": required_passed},
  ]
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in metrics:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  minimum_us = probe.get("minimum_us", "unavailable")
  median_us = probe.get("median_us", "unavailable")
  summary = [
      "# Exact grouped-sparse Q4_K routed-MoE killer gate",
      "",
      f"- case/layer: `{BASE.CASE_ID}` / `{BASE.LAYER}`",
      f"- active experts / assignments / max group: "
      f"`{probe.get('active_experts')}` / `{probe.get('assignment_count')}` / "
      f"`{probe.get('max_group_size')}`",
      f"- grouped implementations: `{probe.get('implementations')}`",
      f"- SwiGLU max abs / mismatches: `{swiglu.get('max_abs_diff')}` / "
      f"`{swiglu.get('mismatch_count')}`",
      f"- complete runtime minimum / median: `{minimum_us} / {median_us} us`",
      f"- complete cap / fraction: `{float(budget['kernel_cap_us']):.3f} us` / "
      f"`{cap_fraction:.3f}`" if cap_fraction is not None else
      f"- complete cap / fraction: `{float(budget['kernel_cap_us']):.3f} us` / unavailable",
      f"- required checks passed: `{str(required_passed).lower()}`",
      f"- disposition: `{route_disposition}`",
      "",
      "The timer includes all dynamic gather, three grouped sparse matrix cores,",
      "exact floating Q4_K residuals, SwiGLU, router weighting, inverse scatter,",
      "submission, and queue drain. Only one-time resident repack is excluded.",
      "This route-closing component result is not a native prefill or product",
      "speed claim; oneDNN remains a development generator/reference only.",
      "",
  ]
  (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "disposition": route_disposition,
      "minimum_us": probe.get("minimum_us"),
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required_passed,
  }, sort_keys=True))
  return 0 if required_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
