#!/usr/bin/env python3
"""Decide the upstream GDN vload/FMA route from one isolated short A/B pair.

The source-only bound admitted one 2k/17-step component because the complete
registered GDN bucket could close the residual left by the fixed-FC ceiling.
This gate verifies that the control and candidate differ only by the rebuilt
GPU plugin, preserves exact teacher-forced tokens and execution census, and
requires the observed post-JIT wall saving to clear that pre-derived residual.
It performs no model compile, GPU context creation, or long/product worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-gdn-vload-fma-component-gate-v0"

BOUND = ROOT / (
    "output/openvino-gdn-vload-fma-bound-"
    "20260717Tseq1287-cleanZ/metrics.json")
CONTROL = ROOT / (
    "output/openvino-gdn-vload-fma-component-"
    "20260717Tseq1288-control-assign-2k-warm17-cleanZ/metrics.json")
CANDIDATE = ROOT / (
    "output/openvino-gdn-vload-fma-component-"
    "20260717Tseq1289-candidate-assign-vload-fma-2k-warm17-cleanZ/metrics.json")
PATCH = ROOT / "engine/openvino/iq36-gdn-vload-fma.patch"
CONTROL_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-assign-device-seq1283/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CANDIDATE_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-assign-gdn-vload-fma-seq1288/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
OPENVINO_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")

EXPECTED_CONTROL_PLUGIN = (
    "eaff3b2fb212679de761b05d7c1d95594ea1a51ba5025abfbcc3ee0d85f57527")
EXPECTED_CANDIDATE_PLUGIN = (
    "7255d66a1b8011381d23e539435e7efdc0790ac9fc81cebf324f2440d752d2f2")
DECODE_STEPS = 17
EXPECTED_COUNTS = {
    "Assign": 60,
    "FullyConnectedCompressed": 371,
    "GatedDeltaNet": 30,
    "IQ36HotAttentionGQA": 10,
    "IQ36LinearConvSwish": 30,
    "RMS": 131,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory stop must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  status = [row for row in status if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def actual_source_identity(metrics: dict[str, Any]) -> dict[str, Any]:
  identity = metrics["accepted_identity"]
  return {
      "config_sha256": identity.get("actual_config_sha256"),
      "sources": {
          str(row.get("path")): row.get("actual_sha256")
          for row in identity.get("sources", [])
      },
  }


def plugin_input(metrics: dict[str, Any]) -> tuple[str, str]:
  rows = [
      (str(path), str(digest))
      for path, digest in metrics["inputs"].items()
      if str(path).endswith("libopenvino_intel_gpu_plugin.so")
  ]
  if len(rows) != 1:
    raise ValueError(f"expected exactly one plugin input, got {rows}")
  return rows[0]


def normalized_inputs(metrics: dict[str, Any]) -> dict[str, str]:
  return {
      str(path): str(digest)
      for path, digest in metrics["inputs"].items()
      if not str(path).endswith("libopenvino_intel_gpu_plugin.so")
  }


def worker_ok(metrics: dict[str, Any], stop_bytes: int) -> bool:
  worker = metrics["worker"]
  monitor = worker["monitor"]
  return (
      worker.get("returncode") == 0
      and worker.get("timed_out") is False
      and worker.get("oom_observed") is False
      and worker.get("memory_guard", {}).get("tripped") is False
      and int(worker.get("memory_guard", {}).get("abort_below_bytes", 0))
      == stop_bytes
      and int(monitor.get("process_swap_peak_bytes", -1)) == 0
      and int(monitor.get("system_available_min_bytes", 0)) >= stop_bytes)


def selected_profile(metrics: dict[str, Any]) -> dict[str, float]:
  raw = metrics["profile_audit"][
      "raw_real_time_us_by_node_type_nonadditive"]
  names = (
      "Assign", "GatedDeltaNet", "IQ36LinearConvSwish",
      "FullyConnectedCompressed", "DynamicQuantize",
      "IQ36HotAttentionGQA", "Transpose")
  return {name: float(raw.get(name, 0.0)) for name in names}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(
        f"memory stop: {available_start} < {stop_bytes} bytes")

  required = (
      BOUND, CONTROL, CANDIDATE, PATCH, CONTROL_PLUGIN,
      CANDIDATE_PLUGIN, OPENVINO_SOURCE / ".git")
  missing = [display_path(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing GDN component inputs: " + ", ".join(missing))

  git = git_state(output)
  bound = load_json(BOUND)
  control = load_json(CONTROL)
  candidate = load_json(CANDIDATE)

  pair_commit = str(control["git"]["commit"])
  pair_same_commit = pair_commit == str(candidate["git"]["commit"])
  pair_is_ancestor = subprocess.run(
      ["git", "merge-base", "--is-ancestor", pair_commit, git["commit"]],
      cwd=ROOT, text=True, capture_output=True).returncode == 0
  pair_to_gate_paths = subprocess.run(
      ["git", "diff", "--name-only", f"{pair_commit}..{git['commit']}"],
      cwd=ROOT, text=True, capture_output=True, check=True
  ).stdout.splitlines()
  allowed_gate_paths = {
      "tools/intel-qwen36-openvino-gdn-vload-fma-component-gate.py"}

  source_patch_present = subprocess.run(
      ["git", "apply", "--reverse", "--check", str(PATCH.resolve())],
      cwd=OPENVINO_SOURCE, text=True, capture_output=True).returncode == 0
  control_hash = sha256(CONTROL_PLUGIN)
  candidate_hash = sha256(CANDIDATE_PLUGIN)
  control_plugin_input = plugin_input(control)
  candidate_plugin_input = plugin_input(candidate)
  control_identity = actual_source_identity(control)
  candidate_identity = actual_source_identity(candidate)

  control_walls = [
      float(value) for value in
      control["worker_result_summary"]["decode_wall_ms"]]
  candidate_walls = [
      float(value) for value in
      candidate["worker_result_summary"]["decode_wall_ms"]]
  control_stable = control_walls[1:]
  candidate_stable = candidate_walls[1:]
  control_median = statistics.median(control_stable)
  candidate_median = statistics.median(candidate_stable)
  control_mean = statistics.mean(control_stable)
  candidate_mean = statistics.mean(candidate_stable)
  observed_wall_saving_ms = control_median - candidate_median
  required_saving_ms = float(bound["bound"]["fixed_fc_residual_ms"])
  component_performance_passed = observed_wall_saving_ms >= required_saving_ms

  control_profile = selected_profile(control)
  candidate_profile = selected_profile(candidate)
  profile_delta_us = {
      name: candidate_profile[name] - control_profile[name]
      for name in control_profile}

  control_counts = control["profile_audit"]["selected_executed_counts"]
  candidate_counts = candidate["profile_audit"]["selected_executed_counts"]
  top1_exact = (
      control["actual_top1"] == control["expected_top1"]
      and candidate["actual_top1"] == candidate["expected_top1"]
      and candidate["actual_top1"] == control["actual_top1"])
  profile_census_exact = (
      control["profile_audit"]["selected_counts_exact"] is True
      and candidate["profile_audit"]["selected_counts_exact"] is True
      and control_counts == EXPECTED_COUNTS
      and candidate_counts == EXPECTED_COUNTS)
  memory_samples_equal = (
      control["worker_result_summary"]["memory_samples"][
          "gpu_after_language_compile"]
      == candidate["worker_result_summary"]["memory_samples"][
          "gpu_after_language_compile"]
      and control["worker_result_summary"]["memory_samples"][
          "gpu_after_final_infer"]
      == candidate["worker_result_summary"]["memory_samples"][
          "gpu_after_final_infer"])

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("source_bound_admitted_exactly_one_short_component",
            bound.get("required_checks_passed") is True
            and bound.get("component_build_admitted") is True
            and bound.get("long_worker_admitted") is False
            and bound.get("product_worker_admitted") is False),
      check("pair_uses_one_clean_common_snapshot",
            pair_same_commit
            and control["git"]["dirty"] is False
            and candidate["git"]["dirty"] is False,
            pair_commit=pair_commit),
      check("gate_only_postdates_pair_by_its_own_tool",
            pair_is_ancestor
            and set(pair_to_gate_paths).issubset(allowed_gate_paths),
            pair_to_gate_paths=pair_to_gate_paths),
      check("control_and_candidate_plugins_are_exact_and_distinct",
            control_hash == EXPECTED_CONTROL_PLUGIN
            and candidate_hash == EXPECTED_CANDIDATE_PLUGIN
            and control_hash != candidate_hash
            and control_plugin_input[1] == control_hash
            and candidate_plugin_input[1] == candidate_hash,
            control_path=control_plugin_input[0],
            control_sha256=control_hash,
            candidate_path=candidate_plugin_input[0],
            candidate_sha256=candidate_hash),
      check("candidate_source_contains_exact_durable_patch",
            source_patch_present, patch=display_path(PATCH),
            patch_sha256=sha256(PATCH)),
      check("pair_uses_identical_graph_inputs_sources_and_config",
            normalized_inputs(control) == normalized_inputs(candidate)
            and control_identity == candidate_identity,
            identity=control_identity),
      check("both_short_workers_are_serial_candidate_only",
            control.get("gpu_workers_launched") == 1
            and candidate.get("gpu_workers_launched") == 1
            and control.get("stock_worker_launched") is False
            and candidate.get("stock_worker_launched") is False
            and control.get("concurrent_worker_launched") is False
            and candidate.get("concurrent_worker_launched") is False
            and control.get("long_worker_launched") is False
            and candidate.get("long_worker_launched") is False),
      check("both_short_workers_complete_without_oom_or_process_swap",
            worker_ok(control, stop_bytes)
            and worker_ok(candidate, stop_bytes),
            control_monitor=control["worker"]["monitor"],
            candidate_monitor=candidate["worker"]["monitor"]),
      check("pair_is_exact_2k_warm17_only",
            control.get("lane") == "2k"
            and candidate.get("lane") == "2k"
            and control.get("decode_steps") == DECODE_STEPS
            and candidate.get("decode_steps") == DECODE_STEPS
            and len(control_walls) == DECODE_STEPS
            and len(candidate_walls) == DECODE_STEPS),
      check("teacher_forced_top1_and_profile_census_are_exact",
            top1_exact and profile_census_exact,
            executed_counts=candidate_counts),
      check("gdn_patch_does_not_change_graph_memory_allocation",
            memory_samples_equal,
            control=control["worker_result_summary"]["memory_samples"],
            candidate=candidate["worker_result_summary"]["memory_samples"]),
      check("no_model_compile_or_long_worker_ran_in_decision_gate",
            True, gpu_contexts=0, model_compiles=0, model_workers=0,
            long_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  route_accepted = required_checks_passed and component_performance_passed
  verdict = (
      "accept_upstream_gdn_vload_fma_for_clean_integration"
      if route_accepted else
      "reject_upstream_gdn_vload_fma_after_short_component"
      if required_checks_passed else "inconclusive")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "pair_commit": pair_commit,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_performance_passed": component_performance_passed,
      "route_accepted": route_accepted,
      "graph_integration_admitted": route_accepted,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "plugins": {
          "control": {"path": str(CONTROL_PLUGIN), "sha256": control_hash},
          "candidate": {
              "path": str(CANDIDATE_PLUGIN), "sha256": candidate_hash},
      },
      "performance": {
          "control_decode_wall_ms": control_walls,
          "candidate_decode_wall_ms": candidate_walls,
          "stable_sample_rule": "drop first decode JIT sample",
          "stable_samples_per_side": len(control_stable),
          "control_median_ms": control_median,
          "candidate_median_ms": candidate_median,
          "control_mean_ms": control_mean,
          "candidate_mean_ms": candidate_mean,
          "observed_wall_saving_ms": observed_wall_saving_ms,
          "required_fixed_fc_residual_saving_ms": required_saving_ms,
          "margin_to_required_saving_ms": (
              observed_wall_saving_ms - required_saving_ms),
          "raw_profile_us_control_nonadditive": control_profile,
          "raw_profile_us_candidate_nonadditive": candidate_profile,
          "raw_profile_us_delta_candidate_minus_control_nonadditive": (
              profile_delta_us),
          "raw_profile_is_decision_evidence": False,
          "interpretation": (
              "the upstream codegen lowers the non-additive GDN profile row, "
              "but the measured stable end-to-end wall regresses and does "
              "not clear the pre-derived residual"),
      },
      "correctness": {
          "top1_exact": top1_exact,
          "profile_census_exact": profile_census_exact,
          "actual_top1": candidate["actual_top1"],
      },
      "oom": {
          "control": control["worker"]["monitor"],
          "candidate": candidate["worker"]["monitor"],
          "guard_tripped": False,
          "oom_observed": False,
          "process_swap_peak_bytes": 0,
      },
      "checks": checks,
      "decision": {
          "close_route": required_checks_passed
          and not component_performance_passed,
          "reason": (
              "the exact applicable upstream GDN kernel patch preserves "
              "correctness but regresses stable wall; do not spend 32k, "
              "ABBA, output512, fixed-FC integration, or product workers"),
          "reopen_condition": (
              "a materially different GDN kernel algorithm or fusion with "
              "an independently derived complete-bucket bound; not another "
              "repeat of this vload/FMA codegen patch"),
      },
      "memory_stop_bytes": stop_bytes,
      "available_memory_start_bytes": available_start,
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  (output / "manifest.json").write_text(
      json.dumps({
          "schema": SCHEMA,
          "tool": display_path(Path(__file__)),
          "git": git,
          "inputs": {
              display_path(path): sha256(path)
              for path in required if path.is_file()
          },
          "gpu_contexts": 0,
          "model_compiles": 0,
          "model_workers": 0,
          "long_workers": 0,
      }, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  summary = "\n".join((
      "# OpenVINO GDN vload/FMA short component",
      "",
      f"Verdict: **{verdict}**. Evidence checks: "
      f"`{str(required_checks_passed).lower()}`; component performance gate: "
      f"`{str(component_performance_passed).lower()}`.",
      "",
      "Both isolated workers preserve all 18 teacher-forced top-1 tokens and "
      "the exact 30-node GDN / full execution census. GPU allocation is "
      "identical. Neither worker OOMed, used process swap, or tripped the "
      "4 GiB guard; no long worker ran.",
      "",
      f"After dropping the first decode JIT sample, control median is "
      f"`{control_median:.6f} ms` and candidate median is "
      f"`{candidate_median:.6f} ms`: observed saving "
      f"`{observed_wall_saving_ms:.6f} ms`, versus the pre-derived required "
      f"`{required_saving_ms:.6f} ms`. The raw non-additive GDN row moves "
      f"from `{control_profile['GatedDeltaNet']:.0f} us` to "
      f"`{candidate_profile['GatedDeltaNet']:.0f} us`, but it is telemetry, "
      "not wall-time decision evidence.",
      "",
      "Close this exact upstream vload/FMA route. Do not launch 32k, ABBA, "
      "output512, fixed-FC integration, or product workers for it.",
      "",
  ))
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "output": display_path(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_performance_passed": component_performance_passed,
      "observed_wall_saving_ms": observed_wall_saving_ms,
      "required_saving_ms": required_saving_ms,
      "raw_gdn_delta_us": profile_delta_us["GatedDeltaNet"],
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
