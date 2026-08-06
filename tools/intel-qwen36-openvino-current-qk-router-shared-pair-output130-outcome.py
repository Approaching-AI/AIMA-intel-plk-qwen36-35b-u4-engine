#!/usr/bin/env python3
"""Classify seq2215 and gate one decomposed-GLU shared-pair repair.

This audit creates no GPU context and loads no model. It rechecks all 130
immutable logits, corrects the pair execution topology to FC -40 / GLU +40,
and admits only one source change: keep the N=1024 horizontal FC pair while
skipping GLUFusion under the existing candidate-only pair switch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-current-qk-router-shared-pair-"
    "output130-outcome-v1")
SEQ2215 = ROOT / (
    "output/openvino-current-qk-router-shared-pair-output130-correctness-"
    "20260731Tseq2215-clean")
RESULT = SEQ2215 / "result.json"
WORKER_CONFIG = SEQ2215 / "raw/candidate/worker-config.json"
WORKER_RESULT = SEQ2215 / "raw/candidate/worker-result.json"
MANIFEST = SEQ2215 / "manifest.json"
PLAN_GATE = ROOT / (
    "output/openvino-current-qk-router-shared-pair-output130-plan-"
    "20260731Tseq2214-clean/result.json")
REFERENCE_ROOT = ROOT / (
    "output/openvino-2k-gated-exact-timing-abba1-"
    "20260731Tseq2183-clean/raw/sentinel_002k/correctness")
REFERENCE_CANDIDATE = REFERENCE_ROOT / "candidate/worker-result.json"
REFERENCE_STOCK = REFERENCE_ROOT / "stock/worker-result.json"
QK_GATE = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-output512-correctness-"
    "20260731Tseq2200-clean/result.json")
PAIR_COMPONENT = ROOT / (
    "output/openvino-fixed-fc-plugin-phase-provider-"
    "20260718Tseq1429-m1024-optin-manager-t1-t128-t1-cleancommit/"
    "metrics.json")
PAIR_GRAPH = ROOT / (
    "output/openvino-fixed-fc-phase-provider-full-graph-"
    "20260718Tseq1430-m1024-optin-native-manager-t1-cleancommit/"
    "metrics.json")
PAIR_PATCH = ROOT / "engine/openvino/iq36-current-router-shared-pair.patch"
NO_GLU_PATCH = ROOT / (
    "engine/openvino/iq36-current-router-shared-pair-no-glu.patch")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
TRANSFORM_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
FC_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2212/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
ACCEPTED_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED_SHA256 = {
    RESULT: "c21e42f2282392bd7cb587ff152a66c576b4700b9a51f6bbdcc9cf6683cf0a20",
    WORKER_CONFIG: "cba922079b7f797cb2b37c380dee03dc0e107c4e06211c29015e1b78e2ea8f84",
    WORKER_RESULT: "39bd292a75c6d18daf39e3d3386c2d21da02a4a5486eae24374677471551d8e5",
    MANIFEST: "580fdf76ed3e6ceedfbf52e28b59cc38aaa210f627ef7b4d3ecb62fc918b0af5",
    PLAN_GATE: "11ea1f0ba96a9d45192cc40971ffd653862a39ed8442b0e0409034bcdbef700f",
    REFERENCE_CANDIDATE: "fa6a4aacdd45251c6818b467477794688754ffc7c5fa744ad9fb22e4961523b3",
    REFERENCE_STOCK: "c327d633b0a6c75320d577bbe555e992303f85da3de800be7b8d70536f7d5215",
    QK_GATE: "ff862015c9cec1aad4fb1c7efa8aa519927417361b480d90d50a95c9292512df",
    PAIR_COMPONENT: "49558a4adde45bb5dfa706d41965b640a5fc30ff1e34dd20cd2d4187313f56fa",
    PAIR_GRAPH: "cfc5da0caf846c21e9098c27e39227d814fe35f6f48812237e258647d50e05d4",
    PAIR_PATCH: "092e1b3d23277cd1ab34577fc26f594efcfb0a837d72904b28b64ae01af36d3a",
    NO_GLU_PATCH: "af1ead7982f2149268637c758502c7f6db81d5cdf2b0cbba905d2c47bddf524e",
    PRODUCT_TOOL: "baa6cb5591766eb91dcb1456d0195216f10a4fafb9477fc3a357f8eb98a8c3b1",
    TRANSFORM_SOURCE: "abbe70c6ed19abce6e6ae7ee586072436b9e3efdd8aaed3bdd3adeec09d73055",
    FC_SOURCE: "1944c1af859c2ccd416a481da8d0bd336bbe39ad9a4bca0aed9ea56182b7996f",
    PLUGIN: "9165f6aa9c31f43b7554c65161e2534bf42ff250b5ad97b51c83e37e2d51ffcd",
    ACCEPTED_PLUGIN: "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985",
}
SEQ2215_COMMIT = "d9f409acff4d10c7d2c97dd4ce0ee10b378a9ed5"
PLAN_TOOL_SHA256 = (
    "3d1c7989e6cca72b5c415f4504e44cf85f88b5231022eb246039158b2771d562")
OUTPUT_TOKENS = 130
KLD_MAX = 0.005
MEMORY_STOP_BYTES = 4 * 1024**3


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def run(
    command: list[str], cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, text=True, capture_output=True, check=False)


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def checkpoints(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
  return {
      int(row["step"]): row
      for row in result.get("distribution_checkpoints", [])
      if isinstance(row, dict) and isinstance(row.get("step"), int)
  }


def checkpoint_path(row: dict[str, Any]) -> Path:
  path = Path(str(row["file"]))
  return path if path.is_absolute() else ROOT / path


def logsumexp(values: np.ndarray) -> float:
  maximum = float(np.max(values))
  return maximum + math.log(float(np.exp(values - maximum).sum()))


def kld(reference: np.ndarray, candidate: np.ndarray) -> float:
  left = reference.astype(np.float64, copy=False)
  right = candidate.astype(np.float64, copy=False)
  left_log = left - logsumexp(left)
  right_log = right - logsumexp(right)
  probability = np.exp(left_log)
  return float(np.sum(probability * (left_log - right_log)))


def count_delta(
    baseline: dict[str, int], observed: dict[str, int],
) -> dict[str, int]:
  return {
      key: int(observed.get(key, 0) - baseline.get(key, 0))
      for key in sorted(set(baseline) | set(observed))
      if observed.get(key, 0) != baseline.get(key, 0)
  }


def named_check(metrics: dict[str, Any], name: str) -> dict[str, Any]:
  return next(
      row for row in metrics.get("checks", [])
      if isinstance(row, dict) and row.get("name") == name)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  missing = [display(path) for path in EXPECTED_SHA256 if not path.is_file()]
  if missing:
    raise SystemExit("missing seq2215 outcome inputs: " + ", ".join(missing))

  observed_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  result = load_json(RESULT)
  worker_result = load_json(WORKER_RESULT)
  plan = load_json(PLAN_GATE)
  accepted = load_json(REFERENCE_CANDIDATE)
  stock = load_json(REFERENCE_STOCK)
  qk_gate = load_json(QK_GATE)
  pair_component = load_json(PAIR_COMPONENT)
  pair_graph = load_json(PAIR_GRAPH)
  worker = result.get("worker") or {}
  monitor = worker.get("monitor") or {}
  guard = worker.get("memory_guard") or {}
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  status = run(["git", "status", "--porcelain"]).stdout.splitlines()
  ancestor = run(
      ["git", "merge-base", "--is-ancestor", SEQ2215_COMMIT, head])
  patch_forward = run(
      ["git", "apply", "--check", str(NO_GLU_PATCH)], SOURCE_TREE)
  patch_reverse = run(
      ["git", "apply", "--reverse", "--check", str(NO_GLU_PATCH)],
      SOURCE_TREE)

  pair_rows = checkpoints(worker_result)
  accepted_rows = checkpoints(accepted)
  stock_rows = checkpoints(stock)
  distribution_rows: list[dict[str, Any]] = []
  invalid_checkpoint_steps: list[int] = []
  for step in range(OUTPUT_TOKENS):
    pair_row = pair_rows.get(step)
    accepted_row = accepted_rows.get(step)
    stock_row = stock_rows.get(step)
    if pair_row is None or accepted_row is None or stock_row is None:
      invalid_checkpoint_steps.append(step)
      continue
    rows = (pair_row, accepted_row, stock_row)
    paths = tuple(checkpoint_path(row) for row in rows)
    if any(not path.is_file() for path in paths):
      invalid_checkpoint_steps.append(step)
      continue
    if any(
        row.get("sha256") != sha256(path)
        for row, path in zip(rows, paths, strict=True)):
      invalid_checkpoint_steps.append(step)
      continue
    pair_logits, accepted_logits, stock_logits = (
        np.fromfile(path, dtype=np.float32) for path in paths)
    if not (
        pair_logits.shape == accepted_logits.shape == stock_logits.shape ==
        (248320,) and
        np.isfinite(pair_logits).all() and
        np.isfinite(accepted_logits).all() and
        np.isfinite(stock_logits).all()):
      invalid_checkpoint_steps.append(step)
      continue
    distribution_rows.append({
        "step": step,
        "accepted_to_pair_kld": kld(accepted_logits, pair_logits),
        "stock_to_pair_kld": kld(stock_logits, pair_logits),
        "stock_to_accepted_kld": kld(stock_logits, accepted_logits),
        "max_abs_vs_accepted": float(
            np.max(np.abs(pair_logits - accepted_logits))),
        "mean_abs_vs_accepted": float(
            np.mean(np.abs(pair_logits - accepted_logits))),
        "top1_pair": int(np.argmax(pair_logits)),
        "top1_accepted": int(np.argmax(accepted_logits)),
        "bitwise_equal": bool(np.array_equal(pair_logits, accepted_logits)),
    })

  worst = max(
      distribution_rows,
      key=lambda row: float(row["accepted_to_pair_kld"]))
  top1_mismatches = [
      row for row in distribution_rows
      if row["top1_pair"] != row["top1_accepted"]]
  kld_failure_steps = [
      int(row["step"]) for row in distribution_rows
      if float(row["accepted_to_pair_kld"]) > KLD_MAX]
  bitwise_matches = sum(row["bitwise_equal"] for row in distribution_rows)
  accepted_tokens = [
      int(value) for value in accepted["generated_token_ids"][:OUTPUT_TOKENS]]
  pair_tokens = [
      int(value)
      for value in worker_result.get("generated_token_ids", [])[:OUTPUT_TOKENS]]
  token_mismatches = [
      {"step": step, "accepted": expected, "pair": actual}
      for step, (expected, actual) in enumerate(
          zip(accepted_tokens, pair_tokens, strict=False))
      if expected != actual
  ]

  observed_counts = (
      (worker_result.get("execution_census") or {}).get(
          "executed_type_counts") or {})
  qk_counts = (
      (qk_gate.get("execution") or {}).get("executed_type_counts") or {})
  census_delta = count_delta(qk_counts, observed_counts)
  expected_delta = {
      "FullyConnectedCompressed": -40,
      "GLU": 40,
  }
  profile = (result.get("execution") or {}).get("runtime_profile") or {}
  pair_decode_check = named_check(
      pair_component, "all_decode_outputs_are_bit_exact_to_stock")
  pair_runtime_check = named_check(
      pair_graph, "minimal_execution_runs_expected_fixed_fc_provider_set")
  transform_text = TRANSFORM_SOURCE.read_text(encoding="utf-8")
  no_glu_patch_text = NO_GLU_PATCH.read_text(encoding="utf-8")

  checks = [
      check("repository_is_clean_pushed_and_contains_seq2215",
            not status and head == origin_main and ancestor.returncode == 0,
            head=head, origin_main=origin_main, status=status,
            seq2215_is_ancestor=ancestor.returncode == 0),
      check("all_outcome_inputs_have_exact_hashes",
            all(
                observed_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                display(path): digest
                for path, digest in observed_hashes.items()}),
      check("seq2214_plan_admitted_only_the_executed_worker",
            plan.get("required_checks_passed") is True and
            plan.get("verdict") ==
                "admit_one_bound_current_qk_router_shared_pair_output130_"
                "worker" and
            plan.get("gpu_workers_launched") == 0 and
            plan.get("tool_sha256") == PLAN_TOOL_SHA256 and
            plan.get("git", {}).get("commit") == SEQ2215_COMMIT),
      check("seq2215_worker_completed_once_without_oom",
            worker.get("returncode") == 0 and
            worker.get("timed_out") is False and
            worker.get("oom_observed") is False and
            guard.get("tripped") is False and
            int(monitor.get("system_available_min_bytes", 0)) >=
                MEMORY_STOP_BYTES and
            (worker.get("worker_transient_scope") or {}).get("enabled")
                is True,
            monitor=monitor),
      check("qk_only_remains_bitwise_exact_output512",
            qk_gate.get("required_checks_passed") is True and
            qk_gate.get("correctness", {}).get(
                "bitwise_checkpoint_count") == 512 and
            qk_gate.get("correctness", {}).get(
                "current_carrier_relative", {}).get("max_kld") == 0.0),
      check("all_40_pairs_activate_with_exact_glu_topology",
            census_delta == expected_delta and
            observed_counts.get("FullyConnectedCompressed") == 331 and
            observed_counts.get("GLU") == 40 and
            profile.get("fused_shared_pair_count") == 40 and
            profile.get("unfused_pair_original_count") == 0 and
            profile.get("unfused_scalar_shared_gate_count") == 40 and
            profile.get("unfused_router_gate_count") == 40,
            census_delta=census_delta, runtime_profile=profile),
      check("pair_with_glu_fails_complete_numeric_boundary",
            not invalid_checkpoint_steps and
            len(distribution_rows) == OUTPUT_TOKENS and
            bitwise_matches == 0 and
            float(worst["accepted_to_pair_kld"]) > KLD_MAX and
            int(worst["step"]) == 37 and
            [row["step"] for row in top1_mismatches] == [41, 42] and
            token_mismatches == [
                {"step": 41, "accepted": 1049, "pair": 5141},
                {"step": 42, "accepted": 8211, "pair": 5435},
            ],
            invalid_checkpoint_steps=invalid_checkpoint_steps,
            bitwise_matches=bitwise_matches, worst=worst,
            top1_mismatches=top1_mismatches,
            token_mismatches=token_mismatches),
      check("prefill_and_early_t1_pass_before_recurrent_error_grows",
            distribution_rows[0]["accepted_to_pair_kld"] < KLD_MAX and
            distribution_rows[2]["accepted_to_pair_kld"] < KLD_MAX and
            kld_failure_steps and kld_failure_steps[0] == 26,
            step0=distribution_rows[0], step2=distribution_rows[2],
            first_kld_failure_step=(
                kld_failure_steps[0] if kld_failure_steps else None),
            kld_failure_steps=kld_failure_steps),
      check("pair_fc_has_distinct_exact_t1_component_evidence",
            pair_component.get("required_checks_passed") is True and
            pair_component.get("rollup", {}).get(
                "decode_compared_rows") == 320 and
            pair_decode_check.get("pass") is True and
            pair_graph.get("required_checks_passed") is True and
            pair_runtime_check.get("pass") is True and
            pair_runtime_check.get("compressed_fc_count") == 331 and
            pair_runtime_check.get("manager_selection_count") == 0,
            pair_decode_rows=pair_component.get(
                "rollup", {}).get("decode_compared_rows"),
            pair_graph_runtime=pair_runtime_check),
      check("one_default_off_no_glu_patch_is_forward_only",
            patch_forward.returncode == 0 and
            patch_reverse.returncode != 0 and
            no_glu_patch_text.count("diff --git ") == 1 and
            "transformations_pipeline.cpp" in no_glu_patch_text and
            'std::getenv("IQ36_ROUTER_SHARED_PAIR")' in no_glu_patch_text and
            "-        manager.register_pass<ov::pass::GLUFusion>();" in
                no_glu_patch_text and
            transform_text.count(
                "manager.register_pass<ov::pass::GLUFusion>();") == 1,
            forward_check={
                "returncode": patch_forward.returncode,
                "stdout": patch_forward.stdout,
                "stderr": patch_forward.stderr,
            },
            reverse_check={
                "returncode": patch_reverse.returncode,
                "stdout": patch_reverse.stdout,
                "stderr": patch_reverse.stderr,
            }),
      check("no_gpu_compiler_or_model_worker_ran", True,
            gpu_contexts=0, compilers=0, model_workers=0),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_pair_with_glu_admit_one_decomposed_glu_patch_and_serial_build"
      if passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": {
          "head": head,
          "origin_main": origin_main,
          "dirty": bool(status),
          "dirty_paths": status,
      },
      "verdict": verdict,
      "required_checks_passed": passed,
      "pair_with_glu_closed": passed,
      "point_or_formal_performance_admitted": False,
      "source_patch_admitted": passed,
      "serial_plugin_build_admitted": passed,
      "gpu_worker_admitted": False,
      "model_worker_admitted": False,
      "correctness": {
          "row_count": len(distribution_rows),
          "bitwise_matches": bitwise_matches,
          "max_accepted_to_pair_kld": float(
              worst["accepted_to_pair_kld"]),
          "max_kld_step": int(worst["step"]),
          "top1_mismatches": top1_mismatches,
          "token_mismatches": token_mismatches,
          "step0_accepted_to_pair_kld": float(
              distribution_rows[0]["accepted_to_pair_kld"]),
          "step2_accepted_to_pair_kld": float(
              distribution_rows[2]["accepted_to_pair_kld"]),
          "first_kld_failure_step": (
              kld_failure_steps[0] if kld_failure_steps else None),
      },
      "execution": {
          "observed_counts": observed_counts,
          "delta_vs_qk_only": census_delta,
      },
      "next_route": {
          "route": "current_qk_router_shared_pair_decomposed_glu",
          "source_contract": {
              "pair_fusion_stays_enabled": True,
              "fused_widths": [512, 512],
              "glu_fusion_skipped_only_under_pair_switch": True,
              "shared_scalar_gate_stays_independent": True,
              "router_stays_independent": True,
              "expected_fully_connected_compressed": 331,
              "expected_glu": 0,
              "expected_split_topology_delta": {
                  "Crop": 40,
                  "Multiply": 40,
                  "VariadicSplit": 40,
              },
          },
          "why_distinct": (
              "seq2215 proves the pair itself activates exactly, but the only "
              "non-FC census delta is 40 GLU nodes and complete recurrent "
              "numerics fail; the successor keeps the funded pair and changes "
              "only that downstream arithmetic boundary"),
          "requirements": [
              "apply the admitted incremental patch exactly once",
              "build only the GPU plugin at -j1 under the 8/4-GiB guard",
              "compile without an InferRequest before correctness",
              "require output130 correctness before any timing",
          ],
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "git": metrics["git"],
      "inputs": {
          display(path): digest for path, digest in observed_hashes.items()},
      "gpu_contexts": 0,
      "compilers": 0,
      "model_workers": 0,
  })
  report = f"""# N=1024 shared-pair output130 outcome

Verdict: **{verdict}**. Required checks:
`{str(passed).lower()}`.

The pair executes with FC/GLU delta `-40/+40` versus Q/K-only, but matches
`{bitwise_matches}/{OUTPUT_TOKENS}` accepted logits. Accepted-relative KLD
first exceeds `{KLD_MAX}` at step
`{kld_failure_steps[0] if kld_failure_steps else None}` and peaks at
`{float(worst['accepted_to_pair_kld']):.9f}` on step `{int(worst['step'])}`.
Token mismatches occur at steps 41 and 42. No timing is admitted.

The only admitted successor keeps the N=1024 FC pair and skips GLUFusion under
the existing candidate-only pair environment. This audit runs no compiler,
GPU context, model, request, or inference worker.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "max_kld": metrics["correctness"]["max_accepted_to_pair_kld"],
      "max_kld_step": metrics["correctness"]["max_kld_step"],
      "first_kld_failure_step": metrics["correctness"][
          "first_kld_failure_step"],
      "token_mismatches": token_mismatches,
      "gpu_workers_launched": 0,
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
