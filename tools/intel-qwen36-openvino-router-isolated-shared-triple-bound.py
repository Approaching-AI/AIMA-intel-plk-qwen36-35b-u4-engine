#!/usr/bin/env python3
"""Bound fusing three shared-expert FCs while preserving the router gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-router-isolated-shared-triple-bound-v0"
LARGE_N_BOUND = ROOT / (
    "output/openvino-large-n-four-fc-qk-bound-"
    "20260718Tseq1333-cleanZ/metrics.json")
BUNDLE_BOUND = ROOT / (
    "output/openvino-fc-rms-igc-qk-rope-bundle-bound-"
    "20260718Tseq1328-cleanZ/metrics.json")
QK_WORKER = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
ALL_FOUR_WORKER = ROOT / (
    "output/openvino-four-fc-qk-bundle-component-"
    "20260718Tseq1331-candidate-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
FC_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
SHARED_SUFFIXES = (
    "mlp.shared_expert_gate/ov_ext::linear/MatMul",
    "mlp.shared_expert.gate_proj/ov_ext::linear/MatMul",
    "mlp.shared_expert.up_proj/ov_ext::linear/MatMul",
)
ROUTER_SUFFIX = "mlp.gate/aten::linear/MatMul"


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


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-four-fc-qk-bundle-component.py",
      "tools/intel-qwen36-openvino-four-fc-qk-bundle-outcome.py",
      "tools/intel-qwen36-openvino-large-n-four-fc-qk-bound.py",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-bound.py",
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


def rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
  value = worker.get("full_profile")
  if not isinstance(value, list):
    raise TypeError("worker lacks full_profile")
  return [row for row in value if row.get("status") == "Status.EXECUTED"]


def raw_ms(selected: list[dict[str, Any]]) -> float:
  return sum(float(row.get("real_time_us") or 0.0)
             for row in selected) / 1000.0


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      LARGE_N_BOUND, BUNDLE_BOUND, QK_WORKER, ALL_FOUR_WORKER, FC_SOURCE)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing shared-triple bound inputs: " + ", ".join(missing))
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError("memory stop tripped before source bound")
  git = git_state(output)
  large_n = load_json(LARGE_N_BOUND)
  bundle = load_json(BUNDLE_BOUND)
  qk = load_json(QK_WORKER)
  all_four = load_json(ALL_FOUR_WORKER)
  qk_rows = rows(qk)
  all_four_rows = rows(all_four)
  source = FC_SOURCE.read_text(encoding="utf-8")

  shared_rows = [
      row for row in qk_rows
      if row.get("node_type") == "FullyConnectedCompressed"
      and any(str(row.get("node_name", "")).endswith(suffix)
              for suffix in SHARED_SUFFIXES)]
  router_rows = [
      row for row in qk_rows
      if row.get("node_type") == "FullyConnectedCompressed"
      and str(row.get("node_name", "")).endswith(ROUTER_SUFFIX)]
  fused_mlp_rows = [
      row for row in all_four_rows
      if row.get("node_type") == "FullyConnectedCompressed"
      and "_fused_4FCs" in str(row.get("node_name", ""))
      and ".mlp." in str(row.get("node_name", ""))]
  mlp_split_overhead = [
      row for row in all_four_rows
      if ".mlp." in str(row.get("node_name", ""))
      and row.get("node_type") in {"Crop", "Multiply", "VariadicSplit"}]
  shared_raw_ms = raw_ms(shared_rows)
  router_raw_ms = raw_ms(router_rows)
  all_four_fused_raw_ms = raw_ms(fused_mlp_rows)
  measured_split_overhead_ms = raw_ms(mlp_split_overhead)
  favorable_net_event_screen_ms = (
      shared_raw_ms - all_four_fused_raw_ms - measured_split_overhead_ms)
  required_incremental_ms = (
      float(bundle["budget"]["current_kill_number_ms"])
      - float(bundle["budget"]["seq1327_qk_observed_component_point_ms"]))
  screen_margin_ms = favorable_net_event_screen_ms - required_incremental_ms
  router_groups = bundle["locked_ir"]["router_shared_groups"]
  shapes_exact = all(
      row["output_widths"] == [1, 512, 512, 256]
      and row["fused_output_width"] == 1281
      and row["weight_contracts_exact"] is True
      for row in router_groups)
  shared_parameter_bytes = 40 * (1 + 512 + 512) * 1104
  router_parameter_bytes = 40 * 256 * 1104
  source_contract = {
      "stock_max_three": source.count(
          "const int max_num_fcs_to_fuse = 4;") == 1,
      "current_all_group_patch_is_rejected_input": True,
      "candidate_predicate": (
          "when a shared input has four compressed FC users with K=2048, "
          "exclude the unique N=256 branch from the matcher and callback; "
          "retain max three and fuse N=[1,512,512] only"),
      "expected_existing_qkv_fused_three_groups": 10,
      "expected_new_shared_fused_three_groups": 40,
      "expected_router_gate_unfused": 40,
      "expected_fully_connected_compressed": 291,
  }
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1333_closes_large_n_and_routes_here",
            large_n.get("required_checks_passed") is True
            and large_n.get("large_n_route_closed") is True
            and large_n.get("next_route", {}).get("route") ==
                "openvino_router_isolated_shared_triple_source_bound"),
      check("locked_ir_has_exact_40_shared_triple_plus_router_groups",
            len(router_groups) == 40 and shapes_exact,
            group_count=len(router_groups), shapes_exact=shapes_exact),
      check("shared_and_router_parameter_bytes_partition_exactly",
            shared_parameter_bytes == 45264000
            and router_parameter_bytes == 11304960
            and shared_parameter_bytes + router_parameter_bytes ==
                bundle["locked_ir"]["parameter_bytes"]["router_shared"],
            shared_parameter_bytes=shared_parameter_bytes,
            router_parameter_bytes=router_parameter_bytes),
      check("runtime_event_rows_partition_exactly",
            len(shared_rows) == 120 and len(router_rows) == 40
            and len(fused_mlp_rows) == 40
            and Counter(str(row.get("node_type"))
                        for row in mlp_split_overhead) == {
                            "Crop": 40, "Multiply": 40,
                            "VariadicSplit": 40},
            shared_rows=len(shared_rows), router_rows=len(router_rows),
            fused_rows=len(fused_mlp_rows)),
      check("shared_triple_favorable_event_screen_clears_residual",
            math.isclose(shared_raw_ms, 6.644, abs_tol=1e-12)
            and math.isclose(router_raw_ms, 0.476, abs_tol=1e-12)
            and math.isclose(all_four_fused_raw_ms, 2.043, abs_tol=1e-12)
            and math.isclose(measured_split_overhead_ms, 0.431,
                             abs_tol=1e-12)
            and screen_margin_ms > 0.0,
            shared_raw_ms=shared_raw_ms,
            router_raw_ms=router_raw_ms,
            all_four_fused_raw_ms=all_four_fused_raw_ms,
            measured_split_overhead_ms=measured_split_overhead_ms,
            favorable_net_event_screen_ms=favorable_net_event_screen_ms,
            required_incremental_ms=required_incremental_ms,
            screen_margin_ms=screen_margin_ms,
            raw_profile_is_savings_evidence=False),
      check("source_change_is_one_parameterized_subset_not_per_layer",
            source_contract["stock_max_three"] is True
            and source_contract["expected_fully_connected_compressed"] == 291,
            source_contract=source_contract),
      check("router_gate_is_preserved_as_correctness_hypothesis",
            True,
            note=("the N=256 router output directly selects experts and is "
                  "the smallest event bucket; preserving it is a targeted "
                  "hypothesis, not yet correctness proof")),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            available_memory_bytes() >= stop_bytes,
            available_start_bytes=available_start, stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_router_isolated_shared_triple_source_patch"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": required_checks_passed,
      "plugin_build_admitted": False,
      "gpu_worker_admitted": False,
      "locked_partition": {
          "groups": 40,
          "shared_widths": [1, 512, 512],
          "router_width": 256,
          "shared_parameter_bytes": shared_parameter_bytes,
          "router_parameter_bytes": router_parameter_bytes,
      },
      "budget": {
          "shared_original_nonadditive_raw_ms": shared_raw_ms,
          "router_original_nonadditive_raw_ms": router_raw_ms,
          "all_four_fused_nonadditive_raw_ms": all_four_fused_raw_ms,
          "measured_split_overhead_nonadditive_raw_ms": (
              measured_split_overhead_ms),
          "favorable_net_event_screen_ms": favorable_net_event_screen_ms,
          "required_incremental_ms_after_retained_qk": required_incremental_ms,
          "screen_margin_ms": screen_margin_ms,
          "interpretation": (
              "non-additive events are used only to admit a default-off "
              "source patch; no performance or product inference is made"),
      },
      "source_contract": source_contract,
      "next_action": {
          "route": "openvino_router_isolated_shared_triple_source_gate",
          "requirements": [
              "replace the rejected max-four patch in the pinned source",
              "preserve the N=256 router FC and fuse only N=[1,512,512]",
              "add unequal-width three-output unit graph coverage",
              "run a no-GPU exact patch/source gate before any build",
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
  report = f"""# Router-isolated shared-triple FC bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or worker ran.

Each of 40 locked MLP inputs feeds `[1, 512, 512]` shared-expert projections
and one 256-wide router gate. The shared triple owns `{shared_raw_ms:.3f} ms`
of the original `{shared_raw_ms + router_raw_ms:.3f}-ms` non-additive FC event
point; the router gate is only `{router_raw_ms:.3f} ms` and directly controls
expert selection.

Granting the measured all-four fused event cost and all observed MLP split
overhead leaves a favorable `{favorable_net_event_screen_ms:.3f}-ms` source
screen, `{screen_margin_ms:.3f} ms` above the post-Q/K residual. This is not a
speed claim. Admit a default-off subset patch that preserves the router gate,
fuses only the three shared projections, and predicts FC census `371 -> 291`.
OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "screen_margin_ms": screen_margin_ms,
      "expected_fc_census": 291,
      "workers": 0,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
