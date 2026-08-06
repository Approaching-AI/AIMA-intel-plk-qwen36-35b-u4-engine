#!/usr/bin/env python3
"""Admit one default-off Q/K plus N=1024 shared-pair source/build cut.

The N=[1,512,512] triple is closed by seq2210.  This zero-GPU gate admits a
distinct successor only: fuse the two N=512 shared-expert bulk projections,
leave the scalar shared gate and router independent, and keep the rejected
fixed-FC manager disabled.
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


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-current-qk-router-shared-pair-source-gate-v1")
SOURCE_TREE = Path("/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
FC_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
PAIR_PATCH = ROOT / "engine/openvino/iq36-current-router-shared-pair.patch"
TRIPLE_PATCH = ROOT / "engine/openvino/iq36-current-router-shared-triple.patch"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
OUTCOME = ROOT / (
    "output/openvino-current-qk-router-shared-output130-outcome-"
    "20260731Tseq2210-clean/metrics.json")
QK_POINT = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-abba-precheck-"
    "20260731Tseq2198-clean/result.json")
QK_FORMAL = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-formal-abba8-"
    "20260731Tseq2202-clean/result.json")
PAIR_COMPONENT = ROOT / (
    "output/openvino-fixed-fc-plugin-phase-provider-"
    "20260718Tseq1429-m1024-optin-manager-t1-t128-t1-cleancommit/"
    "metrics.json")
PAIR_GRAPH = ROOT / (
    "output/openvino-fixed-fc-phase-provider-full-graph-"
    "20260718Tseq1430-m1024-optin-native-manager-t1-cleancommit/"
    "metrics.json")
PINNED_SOURCE_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
EXPECTED_SHA256 = {
    FC_SOURCE: "3bb3485f4ef6303f9c34966d170d683ddf7b9e52d131836ec07e08af02de7bd3",
    PAIR_PATCH: "092e1b3d23277cd1ab34577fc26f594efcfb0a837d72904b28b64ae01af36d3a",
    TRIPLE_PATCH: "ae013a8a610de89d6f8b48971e7238b240db31d2d1d832fce328a6a4290f4420",
    PRODUCT_TOOL: "baa6cb5591766eb91dcb1456d0195216f10a4fafb9477fc3a357f8eb98a8c3b1",
    OUTCOME: "3f0db4100cf52bed448cf06dcbc61615c34136bd31fa9a2450f86fb966121c58",
    QK_FORMAL: "ab3132c45efa7d67cce04befac93f34b2d6ee5563533c325cfaa7c0e66b2d06f",
    PAIR_COMPONENT: "49558a4adde45bb5dfa706d41965b640a5fc30ff1e34dd20cd2d4187313f56fa",
    PAIR_GRAPH: "cfc5da0caf846c21e9098c27e39227d814fe35f6f48812237e258647d50e05d4",
}
MEMORY_STOP_BYTES = 4 * 1024**3
DECODE_TOTAL_POINT_TARGET = 1.02


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


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def named_check(metrics: dict[str, Any], name: str) -> dict[str, Any]:
  return next(
      row for row in metrics.get("checks", [])
      if isinstance(row, dict) and row.get("name") == name)


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = tuple(EXPECTED_SHA256) + (QK_POINT,)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing shared-pair source inputs: " + ", ".join(missing))

  observed_hashes = {path: sha256(path) for path in EXPECTED_SHA256}
  outcome = load_json(OUTCOME)
  qk_point = load_json(QK_POINT)
  qk_formal = load_json(QK_FORMAL)
  pair_component = load_json(PAIR_COMPONENT)
  pair_graph = load_json(PAIR_GRAPH)
  pair_decode_check = named_check(
      pair_component, "all_decode_outputs_are_bit_exact_to_stock")
  pair_runtime_check = named_check(
      pair_graph, "minimal_execution_runs_expected_fixed_fc_provider_set")
  pair_t1_rows = [
      row for row in pair_component["rollup"]["timing_by_case"]
      if int(row["tokens"]) == 1]
  conservative_pair_saving_ms = min(
      (float(row["stock_wall_us_median"]) -
       float(row["candidate_wall_us_median"])) / 1000.0
      for row in pair_t1_rows)

  control_tail = sum(
      float(qk_point["tails"][name]["p50_ms"])
      for name in ("control-a1", "control-a2")) / 2.0
  qk_tail = sum(
      float(qk_point["tails"][name]["p50_ms"])
      for name in ("qk-b1", "qk-b2")) / 2.0
  target_tail = control_tail / DECODE_TOTAL_POINT_TARGET
  required_pair_realization_ms = max(0.0, qk_tail - target_tail)
  funding_multiple = (
      conservative_pair_saving_ms / required_pair_realization_ms
      if required_pair_realization_ms > 0.0 else math.inf)

  source_commit = run(
      ["git", "rev-parse", "HEAD"], SOURCE_TREE).stdout.strip()
  apply_check = run(
      ["git", "apply", "--check", str(PAIR_PATCH)], SOURCE_TREE)
  reverse_check = run(
      ["git", "apply", "--reverse", "--check", str(PAIR_PATCH)],
      SOURCE_TREE)
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin_main = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  status = run(["git", "status", "--porcelain"]).stdout.splitlines()
  product_text = PRODUCT_TOOL.read_text(encoding="utf-8")
  pair_text = PAIR_PATCH.read_text(encoding="utf-8")
  pair_added_text = "\n".join(
      line[1:] for line in pair_text.splitlines()
      if line.startswith("+") and not line.startswith("+++"))
  start_available = available_memory_bytes()

  qk_inference = qk_formal["phase_inference"]
  checks = [
      check("repository_is_clean_and_pushed_at_gate",
            not status and head == origin_main,
            head=head, origin_main=origin_main, status=status),
      check("all_bound_inputs_have_exact_hashes",
            all(
                observed_hashes[path] == expected
                for path, expected in EXPECTED_SHA256.items()),
            observed={
                display(path): digest
                for path, digest in observed_hashes.items()}),
      check("current_cumulative_source_is_exact",
            source_commit == PINNED_SOURCE_COMMIT and
            sha256(FC_SOURCE) == EXPECTED_SHA256[FC_SOURCE],
            source_commit=source_commit,
            source_sha256=sha256(FC_SOURCE)),
      check("seq2210_closes_only_n1025_and_admits_pair_source",
            outcome.get("required_checks_passed") is True and
            outcome.get("verdict") ==
                "reject_n1025_triple_admit_n1024_stock_pair_source_gate" and
            outcome.get("triple_route_closed") is True and
            outcome.get("n1024_stock_pair_source_gate_admitted") is True and
            outcome["correctness"]["max_kld_step"] == 41 and
            outcome["correctness"]["max_accepted_to_triple_kld"] > 0.005 and
            outcome["next_route"]["source_contract"] == {
                "expected_fully_connected_compressed": 331,
                "fixed_fc_manager_enabled": False,
                "fused_output_width": 1024,
                "fused_widths": [512, 512],
                "router_stays_independent": True,
                "router_width": 256,
                "shared_scalar_gate_stays_independent": True,
                "shared_scalar_gate_width": 1,
            }),
      check("n1024_pair_t1_evidence_is_exact_and_manager_free",
            pair_component.get("required_checks_passed") is True and
            pair_component["rollup"]["decode_compared_rows"] == 320 and
            pair_decode_check.get("pass") is True and
            pair_graph.get("required_checks_passed") is True and
            pair_runtime_check.get("pass") is True and
            pair_runtime_check.get("compressed_fc_count") == 331 and
            pair_runtime_check.get("manager_selection_count") == 0 and
            pair_runtime_check.get("manager_prepack_count") == 0,
            decode_compared_rows=pair_component[
                "rollup"]["decode_compared_rows"],
            pair_runtime=pair_runtime_check),
      check("qk_is_exact_and_requires_only_bounded_pair_realization",
            qk_formal.get("verdict") ==
                "reject_stock_half_qk_rope_after_formal_incremental_inference"
            and qk_formal["correctness"]["bitwise_checkpoint_count"] == 512
            and float(qk_inference["prefill_tokens_s"][
                "lower_confidence_bound_ratio"]) < 1.005
            and float(qk_inference["decode_tokens_s"][
                "lower_confidence_bound_ratio"]) >= 1.005
            and conservative_pair_saving_ms > required_pair_realization_ms
            and funding_multiple > 1.0,
            control_tail_ms=control_tail,
            qk_tail_ms=qk_tail,
            target_tail_ms=target_tail,
            required_pair_realization_ms=required_pair_realization_ms,
            conservative_t1_pair_component_saving_ms=(
                conservative_pair_saving_ms),
            funding_multiple=funding_multiple),
      check("incremental_patch_is_default_off_exact_and_applies_once",
            apply_check.returncode == 0 and reverse_check.returncode != 0 and
            pair_text.count("diff --git ") == 1 and
            "fc_horizontal_fusion.cpp" in pair_text and
            "IQ36_ROUTER_SHARED_PAIR" in pair_text and
            "enabled_routes != 1" in pair_text and
            "router_shared_pair_enabled()) &&" in pair_text and
            "n == 512" in pair_text and
            "expand_locked_zero_points_to_u8" not in pair_text,
            patch_sha256=sha256(PAIR_PATCH),
            apply_stderr=apply_check.stderr.strip(),
            reverse_stderr=reverse_check.stderr.strip()),
      check("runtime_switch_is_candidate_only_and_mutually_exclusive",
            "--fuse-router-shared-pair" in product_text and
            product_text.count("IQ36_ROUTER_SHARED_PAIR") == 3 and
            product_text.count('"fuse_router_shared_pair"') >= 5 and
            "router-shared triple and pair are mutually exclusive" in
                product_text and
            "router-shared fusion leaked into a fixed-FC route" in
                product_text),
      check("successor_is_not_fixed_manager_or_triple_repeat",
            outcome["next_route"]["source_contract"][
                "fixed_fc_manager_enabled"] is False and
            "[1,512,512]" not in pair_added_text and
            "n != 256" not in pair_added_text and
            "IQ36_FIXED_FC_MANAGER_SCOPE" not in pair_added_text,
            triple_repeated=False,
            fixed_manager_enabled=False),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_stop_never_tripped",
            start_available >= MEMORY_STOP_BYTES,
            available_bytes=start_available,
            stop_bytes=MEMORY_STOP_BYTES),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_current_qk_router_shared_pair_patch_and_serial_build"
      if passed else "inconclusive")
  result = {
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
      "source_patch_admitted": passed,
      "serial_plugin_build_admitted": passed,
      "gpu_worker_admitted": False,
      "model_worker_admitted": False,
      "source_contract": outcome["next_route"]["source_contract"],
      "budget": {
          "control_tail_ms": control_tail,
          "qk_tail_ms": qk_tail,
          "point_target": DECODE_TOTAL_POINT_TARGET,
          "required_pair_realization_ms": required_pair_realization_ms,
          "conservative_t1_pair_component_saving_ms": (
              conservative_pair_saving_ms),
          "funding_multiple": funding_multiple,
          "interpretation": (
              "same-lane component evidence funds one source/build cut only; "
              "no end-to-end speed or additivity claim is admitted"),
      },
      "next_action": {
          "route": "current_qk_router_shared_n1024_stock_pair",
          "requirements": [
              "apply the incremental patch exactly once",
              "build the Intel GPU plugin at -j1 into a new isolated carrier",
              "do not create a GPU context during the build",
              "compile and output130-correctness gate before point timing",
          ],
      },
      "checks": checks,
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "git": result["git"],
      "inputs": {
          display(path): digest for path, digest in observed_hashes.items()},
      "compilers": 0,
      "gpu_contexts": 0,
      "model_workers": 0,
  })
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "required_pair_realization_ms": required_pair_realization_ms,
      "conservative_pair_component_saving_ms": conservative_pair_saving_ms,
      "gpu_workers_launched": 0,
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
