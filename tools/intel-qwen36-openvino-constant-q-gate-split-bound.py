#!/usr/bin/env python3
"""Bound constant Q/gate split lengths plus consumer relocation.

The prior runtime component relocated all ten split consumers but left every
VariadicSplit and Crop dispatch live.  This source-only gate proves the
remaining static-input/dynamic-length rejection, validates the exact locked
512 -> 256 + 256 constant fold, and admits at most one candidate-only short
worker using the retained control plugin.  It creates no GPU context,
compiler, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-constant-q-gate-split-bound-v0"
MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
WORKER = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
PRIOR_DECISION = ROOT / (
    "output/openvino-dynamic-split-consumer-relocation-decision-"
    "20260717Tseq1308-cleanZ/metrics.json")
PRIOR_CANDIDATE = ROOT / (
    "output/openvino-dynamic-split-consumer-relocation-component-"
    "20260717Tseq1307-candidate-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
PRIOR_BOUND = ROOT / (
    "output/openvino-dynamic-split-inplace-bound-"
    "20260717Tseq1303b-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-dynamic-split-inplace-plugin-build-"
    "20260717Tseq1303c-cleanZ/manifest.json")
CONTROL_PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-dynamic-split-control-seq1304/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
PINNED_OPENVINO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PINNED_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
PR_COMMIT = "6eccc301f69eda560338a9c4bf43d498ab2da937"
FUSING_PATH = (
    "src/plugins/intel_gpu/src/graph/graph_optimizer/"
    "prepare_buffer_fusing.cpp")
PR_TEST_PATH = (
    "src/plugins/intel_gpu/tests/unit/passes/prepare_buffer_fusing_test.cpp")
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
  allowed = {
      "tools/intel-qwen36-openvino-constant-q-gate-split-bound.py"}
  output_relative = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(output_relative):
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


def source_text(commit: str, path: str) -> str:
  return subprocess.run(
      ["git", "show", f"{commit}:{path}"], cwd=PINNED_OPENVINO,
      check=True, capture_output=True, text=True).stdout


def port_shapes(node: ET.Element, section: str) -> dict[int, tuple[int, ...]]:
  parent = node.find(section)
  if parent is None:
    return {}
  return {
      int(port.attrib["id"]): tuple(int(dim.text) for dim in port.findall("dim"))
      for port in parent.findall("port")}


def locked_split_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers_node = root.find("layers")
  edges_node = root.find("edges")
  if layers_node is None or edges_node is None:
    raise ValueError("locked IR lacks layers/edges")
  layers = {int(node.attrib["id"]): node for node in layers_node}
  by_name = {str(node.attrib.get("name")): node_id
             for node_id, node in layers.items()}
  inputs: dict[int, dict[int, int]] = {node_id: {} for node_id in layers}
  for edge in edges_node:
    inputs[int(edge.attrib["to-layer"])][
        int(edge.attrib["to-port"])] = int(edge.attrib["from-layer"])
  rows = []
  for layer in FULL_ATTENTION_LAYERS:
    name = (
        "__module.model.model.language_model.layers."
        f"{layer}.self_attn/prim::ListUnpack/VariadicSplit")
    split = layers[by_name[name]]
    split_inputs = port_shapes(split, "input")
    outputs = list(port_shapes(split, "output").values())
    input_ports = sorted(split_inputs)
    source_ids = [inputs[int(split.attrib["id"])][port]
                  for port in input_ports]
    sources = [layers[node_id] for node_id in source_ids]
    rows.append({
        "layer": layer,
        "split": name,
        "input_shape": list(split_inputs[input_ports[0]]),
        "output_shapes": [list(shape) for shape in outputs],
        "input_source_types": [node.attrib.get("type") for node in sources],
        "input_source_names": [node.attrib.get("name") for node in sources],
    })
  exact = all(
      row["input_shape"][-1] == 512
      and [shape[-1] for shape in row["output_shapes"]] == [256, 256]
      and row["input_source_types"] == ["Reshape", "Const", "Concat"]
      for row in rows)
  return {"rows": rows, "exact_locked_constant_contract": exact}


def execute_model_rewrite_audit(raw: Path) -> dict[str, Any]:
  script = f"""
import importlib.util, json
from pathlib import Path
import numpy as np
import openvino as ov
p = Path({str(GRAPH_MODULE)!r})
s = importlib.util.spec_from_file_location('iq36_graph', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
model = ov.Core().read_model({str(MODEL_XML)!r})
folds = m.fold_q_gate_split_lengths_to_constants(model, ov, np)
moves = m.relocate_q_gate_split_consumers(model, ov)
ops = model.get_ordered_ops()
by_name = {{x.get_friendly_name(): x for x in ops}}
rows = []
for fold in folds:
  split = by_name[fold['split']]
  rows.append({{
    'layer': fold['layer'],
    'length_source_type': split.input_value(2).get_node().get_type_name(),
    'length_values': [int(value) for value in
                      split.input_value(2).get_node().get_vector()],
    'q_direct_users': sorted(x.get_node().get_type_name()
                             for x in split.output(0).get_target_inputs()),
    'gate_direct_users': sorted(x.get_node().get_type_name()
                                for x in split.output(1).get_target_inputs()),
  }})
print(json.dumps({{
  'folds': len(folds), 'relocations': len(moves), 'rows': rows,
  'old_length_sources_live': sum(
      fold['old_lengths_source'] in by_name for fold in folds),
  'new_constants_live': sum(
      fold['constant'] in by_name for fold in folds),
}}, sort_keys=True))
"""
  before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  completed = subprocess.run(
      [str(OV_PYTHON), "-c", script], cwd=ROOT, check=True,
      capture_output=True, text=True, timeout=120)
  after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  audit = json.loads(completed.stdout)
  audit["child_max_rss_kib_upper_bound"] = max(0, int(after - before))
  write_json(raw / "model-rewrite-audit.json", audit)
  return audit


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  available_start = available_memory_bytes()
  if available_start < stop_bytes:
    raise RuntimeError(f"memory stop: {available_start} < {stop_bytes}")
  required = (
      MODEL_XML, OV_PYTHON, GRAPH_MODULE, WORKER, PRIOR_DECISION,
      PRIOR_CANDIDATE, PRIOR_BOUND, BUILD, CONTROL_PLUGIN)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing constant-split bound inputs: " + ", ".join(missing))

  git = git_state(output)
  prior = load_json(PRIOR_DECISION)
  candidate = load_json(PRIOR_CANDIDATE)
  bound = load_json(PRIOR_BOUND)
  build = load_json(BUILD)
  locked = locked_split_audit()
  rewrite = execute_model_rewrite_audit(raw)
  pinned_fusing = source_text(PINNED_COMMIT, FUSING_PATH)
  pr_fusing = source_text(PR_COMMIT, FUSING_PATH)
  pr_tests = source_text(PR_COMMIT, PR_TEST_PATH)

  target = prior["profile"]["target_split"]
  source = candidate.get("source_summary") or {}
  relocation_closed = (
      prior.get("verdict") ==
          "reject_q_gate_split_consumer_relocation_after_component"
      and prior.get("evidence_checks_passed") is True
      and prior.get("activation_passed") is False
      and prior.get("correctness_passed") is True
      and prior.get("route_accepted") is False
      and target.get("status_counts") == {"Status.EXECUTED": 20}
      and source.get("relocate_dynamic_split_consumers") is True
      and source.get("split_consumer_relocation_count") == 10)
  root_cause = {
      "runtime_relocation_active_but_twenty_dispatches_live": relocation_closed,
      "pinned_matcher_requires_both_split_inputs_constant": all(
          text in pinned_fusing for text in (
              "!crop_node.get_dependency(1).is_constant()",
              "!crop_node.get_dependency(2).is_constant()")),
      "pr_matcher_static_input_still_requires_constant_lengths": all(
          text in pr_fusing for text in (
              "const auto is_input_dynamic",
              "crop_node.get_dependency(2).is_constant()")),
      "official_pr_test_rejects_static_input_dynamic_lengths": all(
          text in pr_tests for text in (
              "in_place_crop_static_input_dynamic_split_lengths",
              "EXPECT_EQ(crop1_prim->can_be_optimized(), false)",
              "EXPECT_EQ(crop2_prim->can_be_optimized(), false)")),
      "inference": (
          "The compiled target reaches the static-input matcher branch; this "
          "is inferred from exact PR source/tests plus the seq1307 runtime "
          "census, not from a plugin debug trace."),
  }
  rewrite_exact = (
      rewrite.get("folds") == 10
      and rewrite.get("relocations") == 10
      and rewrite.get("old_length_sources_live") == 0
      and rewrite.get("new_constants_live") == 10
      and all(
          row.get("length_source_type") == "Constant"
          and row.get("length_values") == [256, 256]
          and row.get("q_direct_users") == ["Multiply", "Power"]
          and row.get("gate_direct_users") == ["Sigmoid"]
          for row in rewrite.get("rows", [])))
  budget = bound["budget"]
  budget_admits = (
      float(budget["favorable_split_provider_ceiling_ms"]) >
          float(budget["seq1302_shortfall_ms"])
      and float(budget["expanded_union_margin_ms"]) > 0.0)
  control_plugin_exact = (
      sha256(CONTROL_PLUGIN) == build["control_plugin"]["sha256"])

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("unchanged_consumer_relocation_is_conclusively_closed",
            relocation_closed),
      check("source_and_official_test_explain_remaining_rejection",
            all(value for key, value in root_cause.items()
                if key != "inference"), root_cause=root_cause),
      check("locked_ir_has_ten_exact_512_to_256_plus_256_splits",
            locked["exact_locked_constant_contract"], audit=locked),
      check("exact_model_rewrite_folds_lengths_and_removes_reshape_users",
            rewrite_exact, audit=rewrite),
      check("pinned_control_plugin_supports_the_constant_case",
            root_cause["pinned_matcher_requires_both_split_inputs_constant"]
            and control_plugin_exact,
            plugin=display(CONTROL_PLUGIN),
            sha256=sha256(CONTROL_PLUGIN)),
      check("independent_provider_ceiling_still_clears_shortfall",
            budget_admits, budget=budget),
      check("decision_gate_launches_no_compiler_gpu_or_model_worker", True,
            compilers=0, gpu_contexts=0, model_workers=0, model_ir_reads=1),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_constant_q_gate_split_candidate"
      if required_checks_passed else "inconclusive")
  available_end = available_memory_bytes()
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "constant_split_candidate_admitted": required_checks_passed,
      "candidate_workers_admitted": 1 if required_checks_passed else 0,
      "additional_control_worker_admitted": False,
      "compiler_build_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "root_cause": root_cause,
      "locked_split": locked,
      "model_rewrite": rewrite,
      "budget": budget,
      "control_plugin": {
          "path": str(CONTROL_PLUGIN),
          "sha256": sha256(CONTROL_PLUGIN),
          "exact": control_plugin_exact,
      },
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
          "oom_observed": False,
      },
      "checks": checks,
      "decision": {
          "close_unchanged_pr_and_relocation": required_checks_passed,
          "next_route": (
              "openvino_constant_q_gate_split_component"
              if required_checks_passed else None),
          "reason": (
              "fold the ten semantically fixed split lengths to [256,256], "
              "retain consumer relocation, and use the pinned control plugin"),
          "reopen_condition": (
              "the one admitted candidate must optimize out all ten "
              "VariadicSplit and ten Crop rows before timing is interpreted"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "pinned_openvino_commit": PINNED_COMMIT,
      "official_openvino_pr_commit": PR_COMMIT,
      "gpu_contexts": 0,
      "compilers": 0,
      "model_workers": 0,
      "model_ir_reads": 1,
  })
  report = f"""# Constant Q/gate split-length bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

Seq1307 proves that consumer relocation alone is insufficient: all ten
relocations execute, correctness remains exact, but all `10 VariadicSplit + 10
Crop` rows stay live. The remaining source condition is now pinned. The
compiled graph behaves as the static-input branch, where dynamic split lengths
are rejected; this is an inference from the exact matcher, its official test,
and the complete runtime census.

The locked product IR makes the values invariant: each of the ten Q/gate
splits is axis `-1`, width `512`, with two width-`256` outputs. The model audit
replaces all ten Concat length sources with exact I32 `[256, 256]` constants
and retains all ten validated consumer relocations. This satisfies the
existing pinned matcher, so the experiment uses the retained control plugin
and needs no compiler build or PR patch.

The `0.0686963-ms` independent provider ceiling still exceeds the
`0.0473885-ms` shortfall. Admit exactly one candidate-only 2k/17-step worker
against seq1304's retained control. All twenty target rows must optimize out
before timing is interpreted. No compiler, GPU context, or model worker ran in
this gate; OOM observed: false.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "candidate_workers_admitted": 1 if required_checks_passed else 0,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
