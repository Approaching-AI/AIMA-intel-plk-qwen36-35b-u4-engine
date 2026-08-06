#!/usr/bin/env python3
"""Bound one fused Q/K layout plus partial-RoPE producer.

The locked full-attention graph materializes two Q/K transposes, four partial
rotary slices, two native RoPE kernels, and two rotary/tail concats per layer
before the accepted custom attention.  This source-only gate proves that exact
ten-layer boundary and prices replacing its 100 dispatches with ten
parameterized two-output custom operations.  It creates no compiler, GPU
context, or model worker.
"""

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


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-qk-rope-layout-bound-v0"
MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
CONTROL = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
POST_IGC = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/metrics.json")
OUTPUT_BOUND = ROOT / (
    "output/openvino-attention-output-gate-fusion-bound-"
    "20260717Tseq1311c-cleanZ/metrics.json")
PROJECTION_BOUND = ROOT / (
    "output/openvino-full-attention-projection-consumer-bound-"
    "20260715Tseq1238-cleanZ/metrics.json")
GATED_DQ_OUTCOME = ROOT / (
    "output/openvino-attention-gated-dq-outcome-"
    "20260717Tseq1322-cleanZ/metrics.json")
FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))


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
  raise RuntimeError("MemAvailable missing")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {"tools/intel-qwen36-openvino-qk-rope-layout-bound.py"}
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
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def port_shapes(node: ET.Element, section: str) -> list[list[int]]:
  parent = node.find(section)
  if parent is None:
    return []
  return [[int(dim.text) for dim in port.findall("dim")]
          for port in parent.findall("port")]


def locked_ir_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layer_root = root.find("layers")
  edge_root = root.find("edges")
  if layer_root is None or edge_root is None:
    raise ValueError("locked IR lacks layers/edges")
  layers = {int(node.attrib["id"]): node for node in layer_root}
  by_name = {str(node.attrib.get("name")): node_id
             for node_id, node in layers.items()}
  outgoing: dict[int, set[int]] = {node_id: set() for node_id in layers}
  incoming: dict[int, list[int]] = {node_id: [] for node_id in layers}
  for edge in edge_root:
    source = int(edge.attrib["from-layer"])
    target = int(edge.attrib["to-layer"])
    outgoing[source].add(target)
    incoming[target].append(source)

  def member(layer: int, suffix: str) -> str:
    return (
        "__module.model.model.language_model.layers."
        f"{layer}.self_attn/{suffix}")

  rows = []
  for layer in FULL_ATTENTION_LAYERS:
    # Layer 39 was serialized V/K/Q rather than Q/K/V.  Its exporter suffixes
    # differ while the tensor and arithmetic contracts remain identical.
    suffix = ({
        "q_transpose": "aten::transpose/Transpose_2",
        "q_rotary_slice": "aten::slice/Slice_4",
        "q_tail_slice": "aten::slice/Slice_7",
        "q_cos_multiply": "aten::mul/Multiply_2",
        "q_sin_multiply": "aten::mul/Multiply_3",
        "q_rope": "aten::add/Add_1",
        "q_concat": "aten::cat/Concat_5",
        "k_transpose": "aten::transpose/Transpose_1",
        "k_rotary_slice": "aten::slice/Slice",
        "k_tail_slice": "aten::slice/Slice_3",
        "k_cos_multiply": "aten::mul/Multiply",
        "k_sin_multiply": "aten::mul/Multiply_1",
        "k_rope": "aten::add/Add",
        "k_concat": "aten::cat/Concat_2",
    } if layer == 39 else {
        "q_transpose": "aten::transpose/Transpose",
        "q_rotary_slice": "aten::slice/Slice",
        "q_tail_slice": "aten::slice/Slice_3",
        "q_cos_multiply": "aten::mul/Multiply",
        "q_sin_multiply": "aten::mul/Multiply_1",
        "q_rope": "aten::add/Add",
        "q_concat": "aten::cat/Concat_1",
        "k_transpose": "aten::transpose/Transpose_1",
        "k_rotary_slice": "aten::slice/Slice_4",
        "k_tail_slice": "aten::slice/Slice_7",
        "k_cos_multiply": "aten::mul/Multiply_2",
        "k_sin_multiply": "aten::mul/Multiply_3",
        "k_rope": "aten::add/Add_1",
        "k_concat": "aten::cat/Concat_3",
    })
    names = {
        "q_norm": (
            "__module.model.model.language_model.layers."
            f"{layer}.self_attn.q_norm/aten::mul/Multiply_1"),
        "k_norm": (
            "__module.model.model.language_model.layers."
            f"{layer}.self_attn.k_norm/aten::mul/Multiply_1"),
        **{key: member(layer, value) for key, value in suffix.items()},
    }
    ids = {key: by_name.get(value) for key, value in names.items()}
    direct_pairs = (
        ("q_norm", "q_transpose"),
        ("k_norm", "k_transpose"),
        ("q_transpose", "q_rotary_slice"),
        ("q_transpose", "q_tail_slice"),
        ("k_transpose", "k_rotary_slice"),
        ("k_transpose", "k_tail_slice"),
        ("q_rotary_slice", "q_cos_multiply"),
        ("q_cos_multiply", "q_rope"),
        ("q_sin_multiply", "q_rope"),
        ("q_rope", "q_concat"),
        ("q_tail_slice", "q_concat"),
        ("k_rotary_slice", "k_cos_multiply"),
        ("k_cos_multiply", "k_rope"),
        ("k_sin_multiply", "k_rope"),
        ("k_rope", "k_concat"),
        ("k_tail_slice", "k_concat"),
    )
    direct = {
        f"{source}->{target}": (
            ids[source] is not None and ids[target] is not None and
            int(ids[target]) in outgoing[int(ids[source])])
        for source, target in direct_pairs
    }
    shared_cos = False
    shared_sin = False
    if all(ids[key] is not None for key in (
        "q_cos_multiply", "k_cos_multiply",
        "q_sin_multiply", "k_sin_multiply")):
      q_cos_inputs = incoming[int(ids["q_cos_multiply"])]
      k_cos_inputs = incoming[int(ids["k_cos_multiply"])]
      q_sin_inputs = incoming[int(ids["q_sin_multiply"])]
      k_sin_inputs = incoming[int(ids["k_sin_multiply"])]
      shared_cos = bool(set(q_cos_inputs) & set(k_cos_inputs))
      shared_sin = bool(set(q_sin_inputs) & set(k_sin_inputs))
    shapes = {
        key: port_shapes(layers[int(node_id)], "output")
        for key, node_id in ids.items() if node_id is not None}
    shape_exact = (
        shapes.get("q_norm") == [[-1, -1, 16, 256]]
        and shapes.get("k_norm") == [[-1, -1, 2, 256]]
        and shapes.get("q_transpose") == [[-1, 16, -1, 256]]
        and shapes.get("k_transpose") == [[-1, 2, -1, 256]]
        and shapes.get("q_rope") == [[-1, 16, -1, 64]]
        and shapes.get("k_rope") == [[-1, 2, -1, 64]]
        and shapes.get("q_concat") == [[-1, 16, -1, 256]]
        and shapes.get("k_concat") == [[-1, 2, -1, 256]])
    rows.append({
        "layer": layer,
        "names": names,
        "names_present": all(value is not None for value in ids.values()),
        "direct_edges": direct,
        "shared_cos_input": shared_cos,
        "shared_sin_input": shared_sin,
        "shapes": shapes,
        "shape_exact": shape_exact,
    })
  exact = all(
      row["names_present"] and all(row["direct_edges"].values())
      and row["shared_cos_input"] and row["shared_sin_input"]
      and row["shape_exact"] for row in rows)
  return {"rows": rows, "exact_qk_rope_layout_contract": exact}


def runtime_audit(control: dict[str, Any]) -> dict[str, Any]:
  rows = control.get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("retained control has no full_profile")
  counts: Counter[str] = Counter()
  selected = []
  for row in rows:
    if row.get("status") != "Status.EXECUTED":
      continue
    name = str(row.get("node_name", ""))
    if not any(f"layers.{layer}.self_attn/" in name
               for layer in FULL_ATTENTION_LAYERS):
      continue
    kind = None
    layer = next(
        (value for value in FULL_ATTENTION_LAYERS
         if f"layers.{value}.self_attn/" in name), None)
    if layer is None:
      continue
    q_transpose = ("/aten::transpose/Transpose_2"
                   if layer == 39 else "/aten::transpose/Transpose")
    q_concat = ("/aten::cat/Concat_5"
                if layer == 39 else "/aten::cat/Concat_1")
    k_concat = ("/aten::cat/Concat_2"
                if layer == 39 else "/aten::cat/Concat_3")
    if row.get("node_type") == "Transpose" and name.endswith(
        (q_transpose, "/aten::transpose/Transpose_1")):
      kind = "q_k_transpose"
    elif row.get("node_type") == "StridedSlice" and name.endswith(
        ("/aten::slice/Slice", "/aten::slice/Slice_3",
         "/aten::slice/Slice_4", "/aten::slice/Slice_7")):
      kind = "q_k_rotary_slice"
    elif row.get("node_type") == "RoPE" and name.endswith(
        ("/aten::add/Add", "/aten::add/Add_1")):
      kind = "q_k_rope"
    elif row.get("node_type") == "Concat" and name.endswith(
        (q_concat, k_concat)):
      kind = "q_k_rotary_concat"
    if kind is not None:
      counts[kind] += 1
      selected.append(row)
  expected = {
      "q_k_transpose": 20,
      "q_k_rotary_slice": 40,
      "q_k_rope": 20,
      "q_k_rotary_concat": 20,
  }
  return {
      "counts": dict(counts),
      "expected_counts": expected,
      "counts_exact": dict(counts) == expected,
      "selected_executed_rows": len(selected),
      "raw_profile_time_is_savings_evidence": False,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  required = (
      MODEL_XML, CONTROL, POST_IGC, OUTPUT_BOUND,
      PROJECTION_BOUND, GATED_DQ_OUTCOME)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing Q/K RoPE bound inputs: " + ", ".join(missing))
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory = [{"stage": "start", "available_bytes": available_memory_bytes()}]
  git = git_state(output)
  control = load_json(CONTROL)
  post_igc = load_json(POST_IGC)
  output_bound = load_json(OUTPUT_BOUND)
  projection = load_json(PROJECTION_BOUND)
  gated_dq = load_json(GATED_DQ_OUTCOME)
  ir = locked_ir_audit()
  runtime = runtime_audit(control)
  memory.append({"stage": "after-audit",
                 "available_bytes": available_memory_bytes()})

  existing_dispatches = runtime["selected_executed_rows"]
  replacement_dispatches = len(FULL_ATTENTION_LAYERS)
  removed_dispatches = existing_dispatches - replacement_dispatches
  enqueue_us = float(output_bound["budget"]["max_enqueue_us_per_dispatch"])
  set_args_us = float(
      output_bound["budget"]["set_arguments_per_boundary_dispatch_us"])
  provider_ceiling_ms = removed_dispatches * (enqueue_us + set_args_us) / 1000.0
  prior_union_ms = float(
      post_igc["budget"]["favorable_rms_plus_igc_union_ms"])
  residual_ms = float(post_igc["budget"]["residual_after_fixed_fc_ms"])
  required_saving_ms = float(
      post_igc["budget"]["favorable_union_shortfall_ms"])
  expanded_union_ms = prior_union_ms + provider_ceiling_ms
  expanded_margin_ms = expanded_union_ms - residual_ms
  source_fundable = provider_ceiling_ms >= required_saving_ms

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("locked_ir_has_exact_ten_qk_rope_layout_chains",
            ir["exact_qk_rope_layout_contract"], audit=ir),
      check("retained_control_executes_exact_qk_boundary",
            runtime["counts_exact"] and existing_dispatches == 100,
            audit=runtime),
      check("prior_complete_boundary_was_rejected_only_on_old_residual",
            projection.get("required_checks_passed") is True
            and projection.get("component_admitted") is False
            and projection.get("verdict") ==
                "reject_full_attention_projection_consumer_before_source"),
      check("recent_attention_epilogue_and_gated_dq_routes_are_closed",
            gated_dq.get("evidence_checks_passed") is True
            and gated_dq.get("route_closed") is True
            and gated_dq.get("verdict") ==
                "reject_attention_gated_dq_after_component"),
      check("replacement_is_dispatch_subtractive",
            existing_dispatches == 100 and replacement_dispatches == 10
            and removed_dispatches == 90),
      check("independent_provider_ceiling_clears_bundle_shortfall",
            source_fundable,
            provider_ceiling_ms=provider_ceiling_ms,
            required_incremental_saving_ms=required_saving_ms,
            expanded_union_margin_ms=expanded_margin_ms),
      check("profile_times_are_not_added_as_savings",
            runtime["raw_profile_time_is_savings_evidence"] is False),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            memory=memory),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_qk_rope_layout_source_and_one_short_candidate"
      if required_checks_passed and source_fundable else
      "reject_qk_rope_layout_before_source"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": required_checks_passed and source_fundable,
      "candidate_workers_admitted_after_exact_rewrite_audit": (
          1 if required_checks_passed and source_fundable else 0),
      "additional_control_worker_admitted": False,
      "compiler_admitted": False,
      "gpu_worker_launched": False,
      "long_worker_admitted": False,
      "locked_ir_audit": ir,
      "runtime_audit": runtime,
      "budget": {
          "existing_boundary_dispatches": existing_dispatches,
          "replacement_dispatches": replacement_dispatches,
          "removed_dispatches": removed_dispatches,
          "max_enqueue_us_per_dispatch": enqueue_us,
          "set_arguments_per_boundary_dispatch_us": set_args_us,
          "favorable_qk_rope_layout_provider_ceiling_ms": provider_ceiling_ms,
          "seq1302_favorable_rms_plus_igc_union_ms": prior_union_ms,
          "seq1302_shortfall_ms": required_saving_ms,
          "residual_after_fixed_fc_ms": residual_ms,
          "expanded_favorable_union_ms": expanded_union_ms,
          "expanded_union_margin_ms": expanded_margin_ms,
          "interpretation": (
              "source admission only; the Q/K producer must independently "
              "save the seq1302 shortfall, and the eventual FC/RMS/IGC/QK "
              "bundle must be rebuilt and measured together"),
      },
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {display(path): sha256(path) for path in required},
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": metrics["inputs"],
      "compiler_invocations": 0,
      "gpu_contexts": 0,
      "model_workers": 0,
  })
  report = f"""# Q/K RoPE-layout producer bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`. No compiler or GPU worker ran.

The locked IR and retained seq1304 runtime match ten identical boundaries:
20 Q/K transposes, 40 rotary/tail slices, 20 native RoPE dispatches, and 20
rotary/tail concats. A parameterized two-output producer can consume the
token-major Q/K RMS outputs plus their shared cosine/sine inputs and emit the
same head-major F32 graph values for the accepted custom attention.

Replacing 100 existing dispatches with ten producers removes 90. At the same
favorable provider rate used by seq1311c, the source ceiling is
`{provider_ceiling_ms:.7f} ms/token`, independently above the
`{required_saving_ms:.7f}-ms` RMS-plus-IGC shortfall. Added only for source
admission, the favorable union reaches `{expanded_union_ms:.7f} ms/token`
versus the `{residual_ms:.7f}-ms` fixed-FC residual.

Admit an exact no-GPU rewrite audit and then at most one candidate-only short
worker. Raw profile times are not savings evidence, and any eventual bundle
must be rebuilt and measured together. OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "provider_ceiling_ms": provider_ceiling_ms,
      "required_incremental_saving_ms": required_saving_ms,
      "gpu_worker_launched": False,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
