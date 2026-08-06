#!/usr/bin/env python3
"""Close PR36362 alone and bound the exact Q/gate consumer relocation.

The first short component pair proved that the upstream patch does not fire on
the locked graph because both dynamic split outputs immediately feed Reshape.
This source-only gate verifies the failed activation, proves the two exact
reshape relocations, and admits at most one candidate-only short worker.  It
creates no GPU context, compiler, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-dynamic-split-consumer-relocation-bound-v0")
MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
GRAPH_MODULE = ROOT / "tools/intel_qwen36_openvino_hot_cold_attention.py"
WORKER = ROOT / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
PATCH = ROOT / "engine/openvino/iq36-dynamic-split-length-inplace.patch"
BOUND = ROOT / (
    "output/openvino-dynamic-split-inplace-bound-"
    "20260717Tseq1303b-cleanZ/metrics.json")
BUILD = ROOT / (
    "output/openvino-dynamic-split-inplace-plugin-build-"
    "20260717Tseq1303c-cleanZ/manifest.json")
CONTROL_ROOT = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate")
CONTROL = CONTROL_ROOT / "worker-result.json"
CANDIDATE_ROOT = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1305-candidate-2k-warm17-cleanZ")
CANDIDATE = CANDIDATE_ROOT / "metrics.json"
CANDIDATE_WORKER = CANDIDATE_ROOT / "raw/2k/candidate/worker-result.json"
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
EXPECTED_CORE_COUNTS = {
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
      "tools/intel-qwen36-openvino-dynamic-split-consumer-relocation-bound.py"}
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


def profile_rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
  rows = worker.get("full_profile")
  if isinstance(rows, list):
    return rows
  phases = worker.get("phases", [])
  if phases and isinstance(phases[-1], dict):
    rows = phases[-1].get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("worker has no final full profile")
  return rows


def profile_audit(worker: dict[str, Any]) -> dict[str, Any]:
  executed = [row for row in profile_rows(worker)
              if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row.get("node_type")) for row in executed)
  split_rows = [
      row for row in executed
      if "self_attn/prim::ListUnpack/VariadicSplit.out" in
         str(row.get("node_name", ""))]
  split_counts = Counter(str(row.get("node_type")) for row in split_rows)
  return {
      "core_counts": {
          name: int(counts.get(name, 0)) for name in EXPECTED_CORE_COUNTS},
      "split_counts": dict(sorted(split_counts.items())),
      "split_dispatches": len(split_rows),
      "split_raw_real_time_us_nonadditive": sum(
          float(row.get("real_time_us", 0.0)) for row in split_rows),
      "split_exec_types": sorted({
          str(row.get("exec_type")) for row in split_rows}),
  }


def stable_walls(worker: dict[str, Any]) -> list[float]:
  walls = [float(row["wall_ms_diagnostic"])
           for row in worker.get("phases", [])[1:]]
  if len(walls) != 17 or not all(math.isfinite(value) and value > 0.0
                                 for value in walls):
    raise ValueError("worker does not have 17 finite decode walls")
  return walls[1:]


def port_shapes(node: ET.Element, section: str) -> dict[int, tuple[int, ...]]:
  parent = node.find(section)
  if parent is None:
    return {}
  return {
      int(port.attrib["id"]): tuple(int(dim.text) for dim in port.findall("dim"))
      for port in parent.findall("port")}


def locked_relocation_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers_node = root.find("layers")
  edges_node = root.find("edges")
  if layers_node is None or edges_node is None:
    raise ValueError("locked IR lacks layers/edges")
  layers = {int(node.attrib["id"]): node for node in layers_node}
  by_name = {str(node.attrib.get("name")): node_id
             for node_id, node in layers.items()}
  outgoing: dict[int, list[tuple[int, int, int]]] = {
      node_id: [] for node_id in layers}
  for edge in edges_node:
    outgoing[int(edge.attrib["from-layer"])].append((
        int(edge.attrib["from-port"]), int(edge.attrib["to-layer"]),
        int(edge.attrib["to-port"])))
  rows = []
  for layer in FULL_ATTENTION_LAYERS:
    name = (
        "__module.model.model.language_model.layers."
        f"{layer}.self_attn/prim::ListUnpack/VariadicSplit")
    split_id = by_name[name]
    split = layers[split_id]
    outputs = port_shapes(split, "output")
    output_ports = sorted(outputs)
    split_users = {}
    for output_index, port in enumerate(output_ports):
      users = [(target, target_port) for source_port, target, target_port
               in outgoing[split_id] if source_port == port]
      if len(users) != 1:
        raise ValueError(f"unexpected split fanout: layer {layer} output {port}")
      split_users[output_index] = users[0][0]
    q_reshape_id = split_users[0]
    gate_reshape_id = split_users[1]
    q_reshape = layers[q_reshape_id]
    gate_reshape = layers[gate_reshape_id]
    q_out = next(iter(port_shapes(q_reshape, "output").values()))
    gate_out = next(iter(port_shapes(gate_reshape, "output").values()))
    q_downstream = sorted(
        str(layers[target].attrib.get("type"))
        for _, target, _ in outgoing[q_reshape_id])
    gate_downstream_ids = [target for _, target, _ in outgoing[gate_reshape_id]]
    if len(gate_downstream_ids) != 1:
      raise ValueError(f"unexpected gate reshape fanout at layer {layer}")
    sigmoid_id = gate_downstream_ids[0]
    sigmoid = layers[sigmoid_id]
    sigmoid_users = sorted(
        str(layers[target].attrib.get("type"))
        for _, target, _ in outgoing[sigmoid_id])
    rows.append({
        "layer": layer,
        "split": name,
        "split_output_shapes": list(outputs.values()),
        "direct_user_types": [
            q_reshape.attrib.get("type"), gate_reshape.attrib.get("type")],
        "q_reshape": q_reshape.attrib.get("name"),
        "q_reshape_output_shape": q_out,
        "q_is_shape_preserving": q_out == outputs[output_ports[0]],
        "q_post_relocation_user_types": q_downstream,
        "gate_reshape": gate_reshape.attrib.get("name"),
        "gate_reshape_output_shape": gate_out,
        "gate_user_type": sigmoid.attrib.get("type"),
        "gate_post_relocation_user_types": [sigmoid.attrib.get("type")],
        "sigmoid_original_user_types": sigmoid_users,
    })
  exact = all(
      row["direct_user_types"] == ["Reshape", "Reshape"]
      and row["q_is_shape_preserving"] is True
      and row["q_post_relocation_user_types"] == ["Multiply", "Power"]
      and row["gate_user_type"] == "Sigmoid"
      and row["gate_post_relocation_user_types"] == ["Sigmoid"]
      and row["sigmoid_original_user_types"] == ["Multiply"]
      for row in rows)
  return {"rows": rows, "exact_relocation_contract": exact}


def source_text(commit: str, path: str) -> str:
  return subprocess.run(
      ["git", "show", f"{commit}:{path}"], cwd=PINNED_OPENVINO,
      check=True, capture_output=True, text=True).stdout


def execute_model_rewrite_audit(raw: Path) -> dict[str, Any]:
  script = f"""
import importlib.util, json
from pathlib import Path
import openvino as ov
p = Path({str(GRAPH_MODULE)!r})
s = importlib.util.spec_from_file_location('iq36_graph', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
model = ov.Core().read_model({str(MODEL_XML)!r})
rows = m.relocate_q_gate_split_consumers(model, ov)
ops = model.get_ordered_ops()
print(json.dumps({{
  'relocations': len(rows),
  'old_q_reshape_live': sum(r['q_bypassed_reshape'] == x.get_friendly_name() for r in rows for x in ops),
  'old_gate_reshape_live': sum(r['gate_moved_reshape'] == x.get_friendly_name() for r in rows for x in ops),
  'old_gate_sigmoid_live': sum(r['gate_replaced_sigmoid'] == x.get_friendly_name() for r in rows for x in ops),
  'new_sigmoid': sum(x.get_friendly_name().startswith('iq36_split_gate_sigmoid_layer') for x in ops),
  'new_reshape': sum(x.get_friendly_name().startswith('iq36_split_gate_reshape_layer') for x in ops),
  'first': rows[0], 'last': rows[-1]
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
      MODEL_XML, OV_PYTHON, GRAPH_MODULE, WORKER, PATCH, BOUND, BUILD,
      CONTROL, CANDIDATE, CANDIDATE_WORKER)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing relocation-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  bound = load_json(BOUND)
  build = load_json(BUILD)
  control = load_json(CONTROL)
  candidate_metrics = load_json(CANDIDATE)
  candidate = load_json(CANDIDATE_WORKER)
  control_profile = profile_audit(control)
  candidate_profile = profile_audit(candidate)
  control_top1 = [int(row["top1"]) for row in control["phases"]]
  candidate_top1 = [int(row["top1"]) for row in candidate["phases"]]
  expected_top1 = [int(value) for value in candidate_metrics["expected_top1"]]
  control_stable = stable_walls(control)
  candidate_stable = stable_walls(candidate)
  control_median = statistics.median(control_stable)
  candidate_median = statistics.median(candidate_stable)

  pinned_fusing = source_text(PINNED_COMMIT, FUSING_PATH)
  pr_tests = source_text(PR_COMMIT, PR_TEST_PATH)
  source_root_cause = {
      "pinned_matcher_rejects_dynamic_reshape_user": all(text in pinned_fusing
          for text in (
              "if (user->is_type<reshape>())",
              "if (node.is_dynamic() && node.get_users().size() != 1)",
              "!reshape_node.is_runtime_propagatable_padding()")),
      "official_pr_test_optimizes_nonreshape_crop": (
          "EXPECT_EQ(crop1_prim->can_be_optimized(), true)" in pr_tests),
      "official_pr_test_rejects_reshape_crop": (
          "EXPECT_EQ(crop2_prim->can_be_optimized(), false)" in pr_tests),
      "pr_commit": PR_COMMIT,
  }
  locked = locked_relocation_audit()
  rewrite = execute_model_rewrite_audit(raw)
  rewrite_exact = (
      rewrite.get("relocations") == 10
      and rewrite.get("old_q_reshape_live") == 0
      and rewrite.get("old_gate_reshape_live") == 0
      and rewrite.get("old_gate_sigmoid_live") == 0
      and rewrite.get("new_sigmoid") == 10
      and rewrite.get("new_reshape") == 10)
  candidate_monitor = candidate_metrics["worker"]["monitor"]
  component_safe = (
      candidate_metrics["worker"]["returncode"] == 0
      and candidate_metrics["worker"]["timed_out"] is False
      and candidate_metrics["worker"]["memory_guard"]["tripped"] is False
      and candidate_metrics["worker"]["oom_observed"] is False
      and int(candidate_monitor["system_available_min_bytes"]) >= stop_bytes
      and int(control["memory_samples"]["after_language_compile"]) >= stop_bytes)
  plugin_pair_exact = (
      control.get("candidate_gpu_plugin_sha256") ==
          build["control_plugin"]["sha256"]
      and candidate.get("candidate_gpu_plugin_sha256") ==
          build["candidate_plugin"]["sha256"])
  core_census_exact = (
      control_profile["core_counts"] == EXPECTED_CORE_COUNTS
      and candidate_profile["core_counts"] == EXPECTED_CORE_COUNTS)
  no_activation = (
      control_profile["split_counts"] == {"Crop": 10, "VariadicSplit": 10}
      and candidate_profile["split_counts"] ==
          {"Crop": 10, "VariadicSplit": 10})
  budget = bound["budget"]
  expanded_margin = float(budget["expanded_union_margin_ms"])

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("seq1303b_source_admission_is_pinned",
            bound.get("required_checks_passed") is True
            and expanded_margin > 0.0),
      check("isolated_plugin_pair_uses_exact_build_identities",
            plugin_pair_exact),
      check("both_raw_workers_complete_exact_teacher_forced_path",
            control_top1 == expected_top1 == candidate_top1
            and all(row.get("logits_finite") is True
                    for row in control["phases"] + candidate["phases"])),
      check("both_workers_preserve_core_execution_census",
            core_census_exact,
            control=control_profile["core_counts"],
            candidate=candidate_profile["core_counts"]),
      check("pr36362_alone_does_not_activate_locked_split_crop",
            no_activation,
            control=control_profile,
            candidate=candidate_profile),
      check("component_workers_remain_serial_and_above_stop_without_oom",
            component_safe,
            control_after_compile_available_bytes=
                control["memory_samples"]["after_language_compile"],
            candidate_monitor=candidate_monitor,
            note=("the control worker completed, but its outer packaging hit "
                  "the now-fixed frontier schema KeyError after the worker; "
                  "raw correctness/profile/memory evidence is retained")),
      check("source_and_official_test_explain_reshape_rejection",
            all(source_root_cause.values()), root_cause=source_root_cause),
      check("locked_graph_has_ten_exact_commutable_relocations",
            locked["exact_relocation_contract"], audit=locked),
      check("exact_model_rewrite_removes_old_users_and_validates",
            rewrite_exact, audit=rewrite),
      check("relocation_keeps_the_registered_independent_budget",
            float(budget["favorable_split_provider_ceiling_ms"]) >
                float(budget["seq1302_shortfall_ms"])
            and expanded_margin > 0.0,
            budget=budget),
      check("decision_gate_launches_no_compiler_gpu_or_model_worker", True,
            compilers=0, gpu_contexts=0, model_workers=0,
            model_ir_reads=1),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_q_gate_split_consumer_relocation_candidate"
      if required_checks_passed else "inconclusive")
  available_end = available_memory_bytes()
  performance = {
      "stable_sample_rule": "drop first decode JIT sample",
      "control_median_ms": control_median,
      "candidate_median_ms": candidate_median,
      "observed_median_saving_ms": control_median - candidate_median,
      "control_mean_ms": statistics.mean(control_stable),
      "candidate_mean_ms": statistics.mean(candidate_stable),
      "raw_pair_is_speed_evidence": False,
      "reason": "the PR-alone candidate leaves every target dispatch live",
  }
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "pr36362_alone_closed": required_checks_passed,
      "relocation_candidate_admitted": required_checks_passed,
      "additional_control_worker_admitted": False,
      "candidate_workers_admitted": 1 if required_checks_passed else 0,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "component": {
          "control_profile": control_profile,
          "candidate_profile": candidate_profile,
          "performance": performance,
          "top1": candidate_top1,
      },
      "root_cause": source_root_cause,
      "locked_relocation": locked,
      "model_rewrite": rewrite,
      "budget": budget,
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
          "candidate_monitor": candidate_monitor,
          "oom_observed": False,
      },
      "checks": checks,
      "decision": {
          "close_pr36362_alone": required_checks_passed,
          "next_route": (
              "openvino_q_gate_split_consumer_relocation_component"
              if required_checks_passed else None),
          "reason": (
              "bypass the shape-preserving Q reshape and commute the gate "
              "reshape after Sigmoid, then require direct proof that all ten "
              "split and ten Crop dispatches optimize out"),
          "reopen_condition": (
              "none for PR36362 alone or a repeat of the unchanged graph; "
              "the one admitted relocation candidate must change the exact "
              "runtime census before timing can be interpreted"),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "official_openvino_pr_commit": PR_COMMIT,
      "gpu_contexts": 0,
      "compilers": 0,
      "model_workers": 0,
      "model_ir_reads": 1,
  })
  report = f"""# Q/gate split consumer-relocation bound

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

The exact PR36362-only pair is correct but does not activate: control and
candidate both execute `10 VariadicSplit + 10 Crop` rows. After dropping each
side's first decode JIT sample, the diagnostic medians are
`{control_median:.6f} -> {candidate_median:.6f} ms`; this is not speed evidence
because the intended graph cut never occurred. Close PR36362 on the unchanged
graph without a repeat.

The root cause is exact. Every locked split output immediately feeds Reshape,
and the pinned crop matcher rejects the dynamic reshape-user shape that the
official PR test also marks non-optimizable. For all ten layers, the Q reshape
is shape preserving and feeds only Power/Multiply, while the gate reshape is a
pure flatten immediately before Sigmoid. Bypass the first and commute the second
after Sigmoid. The exact OpenVINO model rewrite validates with ten old Q
reshapes, ten old gate reshapes, and ten old gate Sigmoids removed from the live
graph, replaced by ten relocated Sigmoid/Reshape pairs.

The relocation retains seq1303b's `0.0686963-ms` independent provider ceiling
and `0.0213078-ms` expanded-union margin. Admit one candidate-only 2k/17-step
worker against the existing control. It must optimize out all twenty target
rows and preserve all 18 top-1 tokens plus the core census. No compiler, GPU
context, or model worker ran in this decision gate.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "pr36362_alone_closed": required_checks_passed,
      "relocation_candidate_admitted": required_checks_passed,
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
