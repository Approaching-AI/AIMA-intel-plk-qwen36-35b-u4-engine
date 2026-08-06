#!/usr/bin/env python3
"""Bound shared-triple + exact linear-four + IGC 2.38.2 before edits."""

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
SCHEMA = "intel-qwen36-openvino-shared-linear4-igc2382-bundle-bound-v0"
SHARED_COMPONENT = ROOT / (
    "output/openvino-router-isolated-shared-triple-component-"
    "20260718Tseq1337-candidate-2k-warm17-cleanZ/metrics.json")
LINEAR_BOUND = ROOT / (
    "output/openvino-large-n-four-fc-qk-bound-"
    "20260718Tseq1333-cleanZ/metrics.json")
GROUP_AUDIT = ROOT / (
    "output/openvino-fc-rms-igc-qk-rope-bundle-bound-"
    "20260718Tseq1328-cleanZ/metrics.json")
IGC_GATE = ROOT / (
    "output/openvino-igc2382-component-gate-"
    "20260717Tseq1301-cleanZ/metrics.json")


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
      "engine/openvino/iq36-router-isolated-shared-triple-fusion.patch",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-source-gate.py",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-build.py",
      "tools/intel-qwen36-openvino-router-isolated-shared-triple-component.py",
      "tools/intel-qwen36-openvino-shared-linear4-igc2382-bundle-bound.py",
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


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (SHARED_COMPONENT, LINEAR_BOUND, GROUP_AUDIT, IGC_GATE)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing combined-bound inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_start = available_memory_bytes()
  if memory_start < stop_bytes:
    raise RuntimeError("memory stop tripped before source-only bound")
  git = git_state(output)
  shared = load_json(SHARED_COMPONENT)
  linear = load_json(LINEAR_BOUND)
  groups = load_json(GROUP_AUDIT)
  igc = load_json(IGC_GATE)

  linear_groups = groups["locked_ir"]["linear_attention_groups"]
  widths_exact = all(
      row.get("output_widths") == [8192, 32, 32, 4096]
      and row.get("shared_input_exact") is True
      and row.get("weight_contracts_exact") is True
      for row in linear_groups)
  k2048_exact = all(
      contract["constants"]["weight"]["shape"][1:] == [32, 64]
      for row in linear_groups for contract in row["contracts"])
  expected_linear_layers = [layer for layer in range(40)
                            if layer % 4 != 3]
  layers_exact = [int(row["layer"]) for row in linear_groups] == (
      expected_linear_layers)

  kill_number_ms = float(shared["performance"]["required_total_saving_ms"])
  measured_qk_shared_ms = float(
      shared["performance"]["total_observed_saving_ms"])
  linear_favorable_ms = float(
      linear["budget"]["favorable_incremental_ceiling_ms"])
  igc_point_ms = float(igc["performance"]["observed_median_saving_ms"])
  residual_after_shared_ms = kill_number_ms - measured_qk_shared_ms
  residual_after_linear_ms = residual_after_shared_ms - linear_favorable_ms
  combined_favorable_ms = (
      measured_qk_shared_ms + linear_favorable_ms + igc_point_ms)
  combined_margin_ms = combined_favorable_ms - kill_number_ms
  igc_retention_fraction_needed = residual_after_linear_ms / igc_point_ms

  isolated_igc = next(
      row["isolated_igc"] for row in igc.get("checks", [])
      if isinstance(row, dict) and "isolated_igc" in row)
  expected_igc_hashes = {
      "/tmp/iq36-igc-2.38.2-root/usr/local/lib/libigc.so.2":
          "ff0cc269af1b2f843521b9207c54370fddab25caa404b1322cbdb4598452da33",
      "/tmp/iq36-igc-2.38.2-root/usr/local/lib/libigdfcl.so.2":
          "edd0cc3c73fee76ce156b8a8281d5a747f2634bc81a95da0ca1af9e72abd8de2",
      "/tmp/iq36-igc-2.38.2-root/usr/local/lib/libopencl-clang2.so.17":
          "5ad86d1aa4c4b92ca5ff96cbe2ca96d888b5afc5517e3c23b1772983c4dec63b",
  }
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1337_shared_triple_is_exact_but_below_kill_number",
            shared.get("evidence_checks_passed") is True
            and shared.get("activation_passed") is True
            and shared.get("correctness_passed") is True
            and shared.get("performance_passed") is False
            and shared["profile"]["candidate"].get(
                "fused_shared_triple_count") == 40
            and shared["profile"]["candidate"].get(
                "unfused_router_gate_count") == 40
            and shared["profile"]["candidate"].get(
                "fused_four_fc_count") == 0),
      check("linear_four_group_contract_is_exact_all_30_layers",
            len(linear_groups) == 30 and widths_exact and k2048_exact
            and layers_exact,
            groups=len(linear_groups), widths=[8192, 32, 32, 4096],
            collapsed_k=2048),
      check("seq1333_linear_ceiling_is_complete_and_source_only",
            linear.get("required_checks_passed") is True
            and linear.get("large_n_route_closed") is True
            and linear.get("plugin_build_admitted") is False
            and linear["budget"].get("removed_dispatches") == 90
            and abs(linear_favorable_ms - 0.7031331417624522) < 1e-12),
      check("igc2382_is_exact_retained_bundle_ingredient",
            igc.get("required_checks_passed") is True
            and igc.get("decision", {}).get(
                "retain_as_bundle_ingredient") is True
            and isolated_igc.get("library_dir") ==
                "/tmp/iq36-igc-2.38.2-root/usr/local/lib"
            and isolated_igc.get("libraries") == expected_igc_hashes),
      check("smallest_shared_linear_igc_favorable_union_clears_kill_number",
            combined_margin_ms > 0.0
            and 0.0 < igc_retention_fraction_needed <= 1.0,
            measured_qk_shared_ms=measured_qk_shared_ms,
            linear_favorable_ceiling_ms=linear_favorable_ms,
            igc_unconfirmed_point_ms=igc_point_ms,
            combined_favorable_ms=combined_favorable_ms,
            kill_number_ms=kill_number_ms,
            margin_ms=combined_margin_ms,
            igc_retention_fraction_needed=igc_retention_fraction_needed),
      check("rms_is_not_required_for_smallest_favorable_union",
            combined_margin_ms > 0.0,
            rms_included=False,
            note="park PR36747 until the smaller linear-plus-IGC bundle is measured"),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            available_memory_bytes() >= stop_bytes,
            start_available_bytes=memory_start, stop_bytes=stop_bytes),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  source_edit_admitted = required_checks_passed
  verdict = (
      "admit_exact_linear_four_plus_shared_triple_igc2382_source_patch"
      if source_edit_admitted else "inconclusive")

  source_contract = {
      "preserve_router_isolated_shared_triples": 40,
      "preserve_unfused_router_gates": 40,
      "preserve_existing_qkv_triples": 10,
      "new_exact_linear_four_groups": 30,
      "linear_four_widths": [8192, 32, 32, 4096],
      "linear_four_k": 2048,
      "global_max_four_allowed": False,
      "expected_fully_connected_compressed": 201,
      "expected_fused_shared_triples": 40,
      "expected_fused_linear_fours": 30,
      "expected_unfused_router_gates": 40,
  }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": source_edit_admitted,
      "plugin_build_admitted": False,
      "gpu_worker_admitted": False,
      "source_contract": source_contract,
      "budget": {
          "kill_number_ms": kill_number_ms,
          "measured_qk_shared_ms": measured_qk_shared_ms,
          "residual_after_measured_shared_ms": residual_after_shared_ms,
          "linear_favorable_ceiling_ms": linear_favorable_ms,
          "residual_after_linear_ceiling_ms": residual_after_linear_ms,
          "igc2382_unconfirmed_median_point_ms": igc_point_ms,
          "igc_retention_fraction_needed": igc_retention_fraction_needed,
          "combined_favorable_ms": combined_favorable_ms,
          "combined_margin_ms": combined_margin_ms,
          "interpretation": (
              "shared/QK is measured component evidence; linear is a favorable "
              "non-additive source ceiling and IGC is an unconfirmed median "
              "point. Their union admits only a source patch, not a speed claim."),
      },
      "igc2382": isolated_igc,
      "next_action": {
          "route": "openvino_shared_linear4_igc2382_source_gate",
          "requirements": [
              "extend the existing subset helper only for exact linear widths",
              "retain the router-isolated shared triple and stock max-three",
              "add exact unit coverage for both subset cases",
              "run a no-GPU patch/source gate before any build",
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
  report = f"""# Shared-triple + exact linear-four + IGC 2.38.2 bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler, GPU context, or model
worker ran.

Seq1337 measures exact Q/K plus router-isolated shared triples at
`{measured_qk_shared_ms:.6f} ms` total point saving, leaving
`{residual_after_shared_ms:.6f} ms`. The exact 30 linear four-way groups have a
favorable source ceiling of `{linear_favorable_ms:.6f} ms`; isolated IGC
2.38.2 contributes an unconfirmed `{igc_point_ms:.6f}-ms` median point.

The smallest favorable union is `{combined_favorable_ms:.6f} ms`, only
`{combined_margin_ms:.6f} ms` above the `{kill_number_ms:.6f}-ms` kill-number;
it needs at least `{igc_retention_fraction_needed:.3%}` of the IGC point after
the linear ceiling. This admits one exact source patch only. PR36747 RMS remains
parked. These cross-artifact values are not product inference or a speed claim.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "source_edit_admitted": source_edit_admitted,
      "combined_favorable_ms": combined_favorable_ms,
      "combined_margin_ms": combined_margin_ms,
      "gpu_or_model_worker_launched": False,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
