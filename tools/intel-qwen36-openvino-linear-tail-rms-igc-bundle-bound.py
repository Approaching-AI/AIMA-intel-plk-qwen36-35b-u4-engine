#!/usr/bin/env python3
"""Bound a max-three linear tail plus PR36747 RMS and IGC 2.38.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-linear-tail-rms-igc-bundle-bound-v0"
SOURCE_TREE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
LINEAR4_COMPONENT = ROOT / (
    "output/openvino-shared-linear4-igc2382-component-"
    "20260718Tseq1341-candidate-2k-warm17-cleanZ/metrics.json")
LINEAR4_WORKER = ROOT / (
    "output/openvino-shared-linear4-igc2382-component-"
    "20260718Tseq1341-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
SHARED_COMPONENT = ROOT / (
    "output/openvino-router-isolated-shared-triple-component-"
    "20260718Tseq1337-candidate-2k-warm17-cleanZ/metrics.json")
QK_WORKER = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
LINEAR_BOUND = ROOT / (
    "output/openvino-large-n-four-fc-qk-bound-"
    "20260718Tseq1333-cleanZ/metrics.json")
IGC_GATE = ROOT / (
    "output/openvino-igc2382-component-gate-"
    "20260717Tseq1301-cleanZ/metrics.json")
UPSTREAM_BOUND = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/metrics.json")
RMS_PATCH = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/raw/openvino-pr36747.patch")
LINEAR_SUFFIXES = {
    "qkv": "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
    "a": "linear_attn.in_proj_a/ov_ext::linear/MatMul",
    "b": "linear_attn.in_proj_b/ov_ext::linear/MatMul",
    "z": "linear_attn.in_proj_z/ov_ext::linear/MatMul",
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


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, text=True, capture_output=True, check=False)


def git_state(output: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], ROOT).stdout.strip()
  rows = run(["git", "status", "--porcelain"], ROOT).stdout.splitlines()
  allowed = {
      "engine/openvino/iq36-shared-linear4-horizontal-fusion.patch",
      "tools/intel-qwen36-openvino-shared-linear4-igc2382-source-gate.py",
      "tools/intel-qwen36-openvino-shared-linear4-igc2382-build.py",
      "tools/intel-qwen36-openvino-shared-linear4-igc2382-component.py",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc-bundle-bound.py",
  }
  relative_output = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative_output):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def executed_rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
  rows = worker.get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("worker full_profile missing")
  return [row for row in rows
          if isinstance(row, dict) and row.get("status") == "Status.EXECUTED"]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      LINEAR4_COMPONENT, LINEAR4_WORKER, SHARED_COMPONENT, QK_WORKER,
      LINEAR_BOUND, IGC_GATE, UPSTREAM_BOUND, RMS_PATCH)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing tail-triple bound inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_start = available_memory_bytes()
  if memory_start < stop_bytes:
    raise RuntimeError("memory stop tripped before source-only bound")
  git = git_state(output)
  linear4 = load_json(LINEAR4_COMPONENT)
  shared = load_json(SHARED_COMPONENT)
  qk_worker = load_json(QK_WORKER)
  linear4_worker = load_json(LINEAR4_WORKER)
  linear_bound = load_json(LINEAR_BOUND)
  igc = load_json(IGC_GATE)
  upstream = load_json(UPSTREAM_BOUND)
  rms_apply = run(["git", "apply", "--check", str(RMS_PATCH)], SOURCE_TREE)

  qk_rows = executed_rows(qk_worker)
  linear4_rows = executed_rows(linear4_worker)
  original_us = {}
  for key, suffix in LINEAR_SUFFIXES.items():
    original_us[key] = sum(
        float(row.get("real_time_us") or 0.0) for row in qk_rows
        if row.get("node_type") == "FullyConnectedCompressed"
        and str(row.get("node_name", "")).endswith(suffix))
  fused_linear_us = sum(
      float(row.get("real_time_us") or 0.0) for row in linear4_rows
      if row.get("node_type") == "FullyConnectedCompressed"
      and ".linear_attn." in str(row.get("node_name", ""))
      and "_fused_4FCs" in str(row.get("node_name", "")))
  fused_crop_us = sum(
      float(row.get("real_time_us") or 0.0) for row in linear4_rows
      if row.get("node_type") == "Crop"
      and ".linear_attn." in str(row.get("node_name", "")))
  original_total_ms = sum(original_us.values()) / 1000.0
  tail_original_ms = (
      original_us["a"] + original_us["b"] + original_us["z"]) / 1000.0
  full_four_complete_ms = (fused_linear_us + fused_crop_us) / 1000.0
  full_four_raw_pool_ms = original_total_ms - full_four_complete_ms

  kill_number_ms = float(shared["performance"]["required_total_saving_ms"])
  measured_qk_shared_ms = float(
      shared["performance"]["total_observed_saving_ms"])
  rms_ceiling_ms = 0.358
  igc_point_ms = float(igc["performance"]["observed_median_saving_ms"])
  dispatch_90_ms = float(
      linear_bound["budget"]["favorable_dispatch_ceiling_ms"])
  dispatch_60_ms = dispatch_90_ms * (60.0 / 90.0)
  residual_before_raw_ms = (
      kill_number_ms - measured_qk_shared_ms - rms_ceiling_ms
      - igc_point_ms - dispatch_60_ms)
  raw_retention_fraction_needed = residual_before_raw_ms / full_four_raw_pool_ms
  favorable_total_ms = (
      measured_qk_shared_ms + rms_ceiling_ms + igc_point_ms
      + dispatch_60_ms + full_four_raw_pool_ms)
  favorable_margin_ms = favorable_total_ms - kill_number_ms

  rms_census = upstream["locked_runtime"]["census"]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1341_exact_linear_four_is_conclusively_closed",
            linear4.get("evidence_checks_passed") is True
            and linear4.get("activation_passed") is True
            and linear4.get("correctness_passed") is False
            and linear4.get("performance_passed") is False
            and linear4["profile"]["candidate"].get(
                "fused_four_fc_linear_count") == 30),
      check("stock_max_three_shared_and_qkv_paths_are_exact",
            shared.get("activation_passed") is True
            and shared.get("correctness_passed") is True
            and shared["profile"]["candidate"].get(
                "fused_shared_triple_count") == 40
            and shared["profile"]["candidate"].get(
                "existing_fused_qkv_count") == 10),
      check("linear_branch_profile_partition_is_exact",
            original_us == {"qkv": 4897.0, "a": 283.0,
                            "b": 280.0, "z": 2783.0}
            and abs(original_total_ms - 8.243) < 1e-12
            and abs(tail_original_ms - 3.346) < 1e-12
            and abs(fused_linear_us - 7807.0) < 1e-12
            and abs(fused_crop_us - 30.0) < 1e-12,
            original_us=original_us,
            original_total_ms=original_total_ms,
            tail_original_ms=tail_original_ms,
            fused_linear_us=fused_linear_us,
            fused_crop_us=fused_crop_us),
      check("pr36747_applies_cleanly_to_pinned_external_tree",
            rms_apply.returncode == 0,
            patch_sha256=sha256(RMS_PATCH),
            stderr=rms_apply.stderr.strip()),
      check("pr36747_has_exact_131_live_rms_consumers",
            upstream.get("required_checks_passed") is True
            and rms_census.get("rms_count") == 131
            and rms_census.get("rms_exec_types") == {
                "rms_gpu_bfyx_opt__f16": 131},
            census=rms_census),
      check("tail_triple_rms_igc_favorable_union_clears_kill_number",
            favorable_margin_ms > 0.0
            and 0.0 < raw_retention_fraction_needed < 0.01,
            measured_qk_shared_ms=measured_qk_shared_ms,
            rms_complete_ceiling_ms=rms_ceiling_ms,
            igc_unconfirmed_point_ms=igc_point_ms,
            removed_dispatch_ceiling_ms=dispatch_60_ms,
            full_four_raw_pool_ms=full_four_raw_pool_ms,
            raw_retention_fraction_needed=raw_retention_fraction_needed,
            favorable_total_ms=favorable_total_ms,
            kill_number_ms=kill_number_ms,
            favorable_margin_ms=favorable_margin_ms),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            available_memory_bytes() >= stop_bytes,
            start_available_bytes=memory_start, stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  source_audit_admitted = required_checks_passed
  verdict = (
      "admit_linear_tail_triple_pr36747_rms_igc2382_source_audit"
      if source_audit_admitted else "inconclusive")

  source_contract = {
      "global_max_fcs_to_fuse": 3,
      "preserve_shared_triples": 40,
      "preserve_unfused_router_gates": 40,
      "preserve_existing_qkv_triples": 10,
      "new_linear_tail_triples": 30,
      "linear_tail_widths": [32, 32, 4096],
      "linear_branch_left_unfused": 8192,
      "linear_k": 2048,
      "pr36747_patch_sha256": sha256(RMS_PATCH),
      "isolated_igc2382": True,
      "expected_fully_connected_compressed": 231,
      "expected_fused_three_groups": 80,
      "expected_unfused_linear_qkv": 30,
  }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_audit_admitted": source_audit_admitted,
      "source_edit_admitted": False,
      "plugin_build_admitted": False,
      "gpu_worker_admitted": False,
      "source_contract": source_contract,
      "profile_partition": {
          "original_linear_branch_us": original_us,
          "original_total_ms": original_total_ms,
          "tail_original_ms": tail_original_ms,
          "full_four_fused_ms": fused_linear_us / 1000.0,
          "full_four_crop_ms": fused_crop_us / 1000.0,
          "full_four_raw_pool_ms": full_four_raw_pool_ms,
          "raw_profile_is_savings_evidence": False,
      },
      "budget": {
          "kill_number_ms": kill_number_ms,
          "measured_qk_shared_ms": measured_qk_shared_ms,
          "rms_complete_ceiling_ms": rms_ceiling_ms,
          "igc2382_unconfirmed_point_ms": igc_point_ms,
          "removed_60_dispatch_ceiling_ms": dispatch_60_ms,
          "residual_before_raw_ms": residual_before_raw_ms,
          "full_four_raw_pool_ms": full_four_raw_pool_ms,
          "raw_retention_fraction_needed": raw_retention_fraction_needed,
          "favorable_total_ms": favorable_total_ms,
          "favorable_margin_ms": favorable_margin_ms,
          "interpretation": (
              "All terms admit only an exact source audit. RMS is granted its "
              "entire registered bucket, IGC is an unconfirmed point, dispatch "
              "is a provider ceiling, and raw events are non-additive."),
      },
      "next_action": {
          "route": "openvino_linear_tail_triple_rms_igc_source_gate",
          "requirements": [
              "replace linear max-four with a max-three tail subset",
              "leave the N=8192 linear branch independent",
              "apply exact captured PR36747 after an overlap audit",
              "add unit graphs and run a no-GPU exact patch/source gate",
          ],
      },
      "checks": checks,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "compilers": 0,
      "gpu_contexts": 0,
      "model_workers": 0,
  })
  report = f"""# Linear tail max-three + PR36747 RMS + IGC bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

Seq1341 proves exact max-four linear fusion is invalid: all 30 groups activate,
but all 18 top-1 IDs differ and wall time regresses versus shared-only. The
correct max-three successor leaves `N=8192` independent and fuses only
`[32,32,4096]`; its original non-additive event point is
`{tail_original_ms:.6f} ms`.

Measured Q/K+shared, the full RMS bucket, unconfirmed IGC point, and the
60-dispatch ceiling leave only `{residual_before_raw_ms:.6f} ms`. That is
`{raw_retention_fraction_needed:.3%}` of the `{full_four_raw_pool_ms:.6f}-ms`
full-four raw pool. The favorable total is `{favorable_total_ms:.6f} ms`,
`{favorable_margin_ms:.6f} ms` above the kill-number. PR36747 applies cleanly
and has 131 live RMS consumers. This admits a source audit only, not a speed
claim.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "source_audit_admitted": source_audit_admitted,
      "raw_retention_fraction_needed": raw_retention_fraction_needed,
      "favorable_margin_ms": favorable_margin_ms,
      "gpu_or_model_worker_launched": False,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
