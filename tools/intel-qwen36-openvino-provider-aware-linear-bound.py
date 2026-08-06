#!/usr/bin/env python3
"""Admit or close the provider-aware FC-to-linear-state boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-provider-aware-linear-bound-v0"
MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
FC_COMPONENT = REPO / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ/metrics.json")
PROFILE_RESULT = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1136-"
    "dq-subgroup-32k-warm17-cleanZ/raw/32k/candidate/worker-result.json")
REAL_CARRIER = REPO / (
    "output/level-zero-real-carrier-gate-20260712Tseq738cleanZ/result.json")
GRAPH_SOURCE = REPO / "tools/intel_qwen36_openvino_hot_cold_attention.py"
LINEAR_SOURCE = REPO / "engine/openvino/custom/iq36_linear_conv_swish.cl"
REJECTED = REPO / "doc/active" / WS / "rejected-routes.json"

FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
LINEAR_LAYERS = tuple(
    layer for layer in range(40) if layer not in FULL_ATTENTION_LAYERS)
FC_STOCK_MS = 11.020
GDN_MS = 1.319
LINEAR_CONV_MS = 0.193
KILL_NUMBER_MS = 2.837
GDN_HEADS = 32
GDN_KEY = 128
GDN_VALUE = 128
F16_BYTES = 2


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state() -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  return {"commit": commit, "dirty": bool(status), "status": status}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def display_path(path: Path) -> str:
  try:
    return str(path.relative_to(REPO))
  except ValueError:
    return str(path)


def edge_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers = root.find("layers")
  edges = root.find("edges")
  if layers is None or edges is None:
    raise ValueError("locked IR is missing layers or edges")
  name_to_id = {
      row.attrib.get("name", ""): row.attrib["id"] for row in layers}
  edge_pairs = {
      (row.attrib["from-layer"], row.attrib["to-layer"]) for row in edges}
  rows = []
  for layer in LINEAR_LAYERS:
    prefix = f"__module.model.model.language_model.layers.{layer}.linear_attn/"
    names = (
        f"__module.model.model.language_model.layers.{layer}."
        "linear_attn.in_proj_qkv/ov_ext::linear/MatMul",
        prefix + "aten::transpose/Transpose",
        prefix + "aten::cat/Concat",
        prefix + "aten::_convolution/GroupConvolution",
        prefix + "aten::slice/Slice_2",
        prefix + "aten::silu/Swish",
        prefix + "aten::transpose/Transpose_1",
        prefix + "aten::split_with_sizes/VariadicSplit",
    )
    ids = [name_to_id.get(name) for name in names]
    direct = [
        ids[index] is not None and ids[index + 1] is not None and
        (str(ids[index]), str(ids[index + 1])) in edge_pairs
        for index in range(len(ids) - 1)]
    rows.append({
        "layer": layer,
        "names_present": all(value is not None for value in ids),
        "direct_chain": all(direct),
        "node_ids": ids,
    })
  return {
      "linear_layers": list(LINEAR_LAYERS),
      "rows": rows,
      "all_names_present": all(row["names_present"] for row in rows),
      "all_direct_chains": all(row["direct_chain"] for row in rows),
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      MODEL_XML, FC_COMPONENT, PROFILE_RESULT, REAL_CARRIER, GRAPH_SOURCE,
      LINEAR_SOURCE, REJECTED)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing bound inputs: " + ", ".join(missing))

  git = git_state()
  fc = load_json(FC_COMPONENT)
  profile = load_json(PROFILE_RESULT)
  carrier = load_json(REAL_CARRIER)
  rejected = load_json(REJECTED)
  graph_source = GRAPH_SOURCE.read_text(encoding="utf-8")
  linear_source = LINEAR_SOURCE.read_text(encoding="utf-8")
  graph = edge_audit()
  sample_memory("after-locked-ir-audit", stop_bytes, memory)

  aggregate = fc["aggregate"]
  fc_schedule_ms = float(aggregate["dominant_ms"])
  fc_saving_ms = FC_STOCK_MS - fc_schedule_ms
  adjacent_ms = GDN_MS + LINEAR_CONV_MS
  required_adjacent_saving_ms = KILL_NUMBER_MS - fc_saving_ms
  adjacent_target_ms = adjacent_ms - required_adjacent_saving_ms
  adjacent_reduction = required_adjacent_saving_ms / adjacent_ms
  combined_current_ms = FC_STOCK_MS + adjacent_ms
  combined_target_ms = combined_current_ms - KILL_NUMBER_MS
  perfect_union_saving_ms = fc_saving_ms + adjacent_ms

  # Each linear layer must at least read and publish its 32x128x128 F16
  # recurrent state once. This intentionally ignores all arithmetic and every
  # other input/output, making it a favorable physical feasibility floor.
  state_bytes_per_layer = GDN_HEADS * GDN_KEY * GDN_VALUE * F16_BYTES
  state_rw_bytes = len(LINEAR_LAYERS) * state_bytes_per_layer * 2
  state_rw_required_gbps = state_rw_bytes / adjacent_target_ms / 1_000_000
  proven_carrier_gbps = float(
      carrier["admission"]["paired_kernel_gb_s_min"])

  source_summary = profile.get("source_summary", {})
  replacements = source_summary.get("linear_conv_replacements", [])
  replacement_layers = tuple(
      sorted(int(row["layer"]) for row in replacements))
  full_profile = profile.get("full_profile", [])
  executed = Counter(
      str(row.get("node_type")) for row in full_profile
      if row.get("status") == "Status.EXECUTED")
  rejected_row = next(
      (row for row in rejected.get("rejected", [])
       if row.get("route") ==
       "openvino_qkv_transpose_to_gdn_adjacent_fusion_v28a"), {})

  component_admission = (
      perfect_union_saving_ms > KILL_NUMBER_MS and
      0.0 < adjacent_target_ms < adjacent_ms and
      state_rw_required_gbps < proven_carrier_gbps)
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("fixed_fc_component_is_clean_complete_ceiling",
            fc.get("required_checks_passed") is True and
            fc.get("route_stop_proven") is True and
            int(aggregate["dominant_bytes"]) == 770_901_120 and
            int(aggregate["remaining_bytes_not_charged"]) == 0,
            artifact=str(FC_COMPONENT.relative_to(REPO))),
      check("locked_ir_has_all_thirty_direct_fc_conv_chains",
            graph["all_names_present"] and graph["all_direct_chains"] and
            len(graph["rows"]) == 30),
      check("accepted_graph_fuses_all_thirty_conv_state_boundaries",
            source_summary.get("linear_conv_replacement_count") == 30 and
            source_summary.get("linear_conv_custom_count_after") == 30 and
            replacement_layers == LINEAR_LAYERS,
            replacement_layers=list(replacement_layers)),
      check("accepted_fusion_bypasses_both_materialized_conv_transposes",
            "input_transpose.input_value(0)" in graph_source and
            "output_transpose.output(0).replace" in graph_source and
            "FC-output transpose" in linear_source),
      check("executed_profile_has_exact_linear_boundary_counts",
            executed["FullyConnectedCompressed"] == 371 and
            executed["IQ36LinearConvSwish"] == 30 and
            executed["GatedDeltaNet"] == 30,
            counts=dict(executed)),
      check("prior_qkv_to_gdn_spelling_requires_broader_provider_boundary",
            "broader provider-aware FC-to-conv/GDN boundary" in
            str(rejected_row.get("reopen_condition", "")),
            route=rejected_row),
      check("combined_boundary_has_arithmetic_headroom",
            perfect_union_saving_ms > KILL_NUMBER_MS,
            perfect_union_saving_ms=perfect_union_saving_ms,
            kill_number_ms=KILL_NUMBER_MS),
      check("mandatory_state_traffic_is_below_proven_physical_carrier",
            state_rw_required_gbps < proven_carrier_gbps,
            required_gbps=state_rw_required_gbps,
            proven_carrier_gbps=proven_carrier_gbps),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_real_provider_aware_component_only"
      if required_checks_passed and component_admission else
      "reject_provider_aware_boundary_before_component")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "component_admission_passed": component_admission,
      "long_worker_admitted": False,
      "graph_integration_admitted": False,
      "boundary": {
          "current_ms_per_token": {
              "non_lm_fc": FC_STOCK_MS,
              "gated_delta_net": GDN_MS,
              "linear_conv_state_swish": LINEAR_CONV_MS,
              "combined": combined_current_ms,
          },
          "optimistic_fc_schedule_ms_per_token": fc_schedule_ms,
          "optimistic_fc_saving_ms_per_token": fc_saving_ms,
          "kill_number_ms_per_token": KILL_NUMBER_MS,
          "required_adjacent_saving_ms_per_token":
              required_adjacent_saving_ms,
          "adjacent_target_ms_per_token": adjacent_target_ms,
          "required_adjacent_reduction": adjacent_reduction,
          "combined_target_ms_per_token": combined_target_ms,
          "perfect_union_saving_ms_per_token": perfect_union_saving_ms,
      },
      "physical_floor": {
          "state_bytes_per_layer": state_bytes_per_layer,
          "all_layer_state_read_write_bytes": state_rw_bytes,
          "state_read_write_required_gbps": state_rw_required_gbps,
          "proven_paired_carrier_floor_gbps": proven_carrier_gbps,
          "optimism": [
              "charges only one read and one write of recurrent F16 state",
              "charges no GDN arithmetic, qkv/gate/beta traffic, or launch cost",
              "uses the full adjacent residual as the state-traffic budget",
          ],
      },
      "locked_ir_audit": graph,
      "executed_profile_counts": dict(executed),
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {
          str(path.relative_to(REPO) if path.is_relative_to(REPO) else path):
              sha256(path) for path in required
      },
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  summary = f"""# OpenVINO provider-aware linear boundary

Verdict: **{verdict}**. Required evidence checks:
`{str(required_checks_passed).lower()}`.

Seq1233's maximally optimistic FC schedule saves
`{fc_saving_ms:.6f} ms/token`, so the adjacent all-30 linear boundary must save another
`{required_adjacent_saving_ms:.6f} ms/token`. Its current GDN plus fused
conv/state/SiLU envelope is `{adjacent_ms:.3f} ms/token`; the component target
is therefore at most `{adjacent_target_ms:.6f} ms/token`, a
`{adjacent_reduction * 100:.2f}%` cut.

The locked IR contains all 30 direct FC -> transpose -> conv -> SiLU ->
transpose -> split chains, while the accepted custom conv already bypasses both
conv transposes. A successor must preserve seq1233's fixed-shape FC carrier and
the exact GDN recurrence; it may not repeat the closed generic direct/tiled
QKV-to-GDN kernels.

Even an ideal component must read and write `{state_rw_bytes:,}` bytes of F16
recurrent state per token. Fitting only that traffic into the target requires
`{state_rw_required_gbps:.3f} GB/s`, below the clean real-carrier floor of
`{proven_carrier_gbps:.3f} GB/s`. This admits exactly one real layer-0 component
capture/proof. It does not admit graph integration, a 32k row, ABBA, or
output512. No worker was launched by this bound gate.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "required_adjacent_saving_ms": required_adjacent_saving_ms,
      "adjacent_target_ms": adjacent_target_ms,
      "state_rw_required_gbps": state_rw_required_gbps,
  }, separators=(",", ":")))
  return 0 if required_checks_passed and component_admission else 2


if __name__ == "__main__":
  raise SystemExit(main())
