#!/usr/bin/env python3
"""Audit new upstream FC/DQ capabilities before any compiler or GPU work.

The gate combines immutable oneDNN strategy patches, live OpenVINO pull-request
metadata, the locked five-cohort FC ceiling, and the stored provider trace.  It
answers only whether a capability published after the pinned runtime supplies a
complete source-derived path below the 8.183-ms non-LM FC budget.  It never
invokes a compiler, creates an OpenCL/Level Zero context, or starts a model
worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-upstream-fc-capability-bound-v1"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
FIXED_COMPONENT = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/metrics.json")
PROVIDER_TRACE = ROOT / (
    "output/openvino-hot-cold-product-"
    "20260715Tseq1212-onednn-gemm-selection-trace-2k-o4-dirtyZ/raw/"
    "sentinel_002k/correctness/candidate/worker.stdout")
LOCKED_MODEL_XML = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
RUNTIME_GRAPH = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1150-fixed-2k-dq-census-cleanZ/raw/2k/candidate/"
    "runtime-graph.xml")
PROVIDER_BOUND = ROOT / (
    "output/openvino-full-attention-projection-consumer-bound-"
    "20260715Tseq1238-cleanZ/metrics.json")
PINNED_OPENVINO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
GATED_MLP_TRANSFORM = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/plugin/transformations/fuse_gated_mlp.cpp")
GATED_MLP_OP = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/plugin/transformations/op/gated_mlp.cpp")
GATED_MLP_PIPELINE = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
GATED_MLP_REF = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/thirdparty/onednn_gpu/src/gpu/intel/"
    "gated_mlp/ref.hpp")
SHARED_DQ_PATCH = ROOT / "engine/openvino/iq36-shared-dynamic-quantize.patch"
SUBGROUP_DQ_PATCH = ROOT / (
    "engine/openvino/iq36-dynamic-quantize-subgroup64.patch")

FHS_COMMIT = "c83564ade4721a250d1e9cc6b5e7aedc764bbbb9"
FOOS_COMMIT = "5797885300663280e5cd713c255f25d85bb59313"
FHS_PATCH_URL = (
    "https://github.com/uxlfoundation/oneDNN/commit/" + FHS_COMMIT + ".patch")
FOOS_PATCH_URL = (
    "https://github.com/uxlfoundation/oneDNN/commit/" + FOOS_COMMIT + ".patch")

PULLS = {
    "onednn_fhs_nnn": ("uxlfoundation", "oneDNN", 5300),
    "onednn_foos_nnn": ("uxlfoundation", "oneDNN", 5357),
    "openvino_dq_consistency": ("openvinotoolkit", "openvino", 36078),
    "openvino_int4_nnn_shared_weights": (
        "openvinotoolkit", "openvino", 36845),
    "openvino_nontransposed_f16_policy": (
        "openvinotoolkit", "openvino", 36437),
    "openvino_sink_unsqueeze_eltwise": (
        "openvinotoolkit", "openvino", 36879),
    "openvino_preserve_weight_sharing": (
        "openvinotoolkit", "openvino", 36905),
    "openvino_shared_dq": ("openvinotoolkit", "openvino", 36866),
    "openvino_subgroup64_dq": ("openvinotoolkit", "openvino", 36867),
    "openvino_skip_dq_dgpu": ("openvinotoolkit", "openvino", 36907),
    "openvino_revert_dq_consistency": (
        "openvinotoolkit", "openvino", 36927),
    "onednn_gated_mlp_inplace": ("uxlfoundation", "oneDNN", 5413),
}

EXPECTED_COHORTS = (
    ("linear_in_fused_m12352_k2048", 12352, 2048, 30),
    ("full_qkv_fused_m9216_k2048", 9216, 2048, 10),
    ("mlp_input_fused_m1281_k2048", 1281, 2048, 40),
    ("m2048_k4096", 2048, 4096, 40),
    ("m2048_k512", 2048, 512, 40),
)
FHS_PUBLISHED_GFLOPS = (379.27, 402.244, 355.471, 391.531)
FHS_PUBLISHED_ROWS = (
    {"m": 3072, "k": 3072, "nnn_gflops": 379.27,
     "tnn_gflops": 398.171},
    {"m": 3072, "k": 8192, "nnn_gflops": 402.244,
     "tnn_gflops": 325.958},
    {"m": 9216, "k": 3072, "nnn_gflops": 355.471,
     "tnn_gflops": 411.595},
    {"m": 16384, "k": 3072, "nnn_gflops": 391.531,
     "tnn_gflops": 417.853},
)
PHYSICAL_CARRIER_GBPS = 106.52460857
FC_TARGET_MS = 8.183
FC_STOCK_MS = 11.020
KILL_NUMBER_MS = 2.837


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--network-timeout-s", type=float, default=30.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0 or args.network_timeout_s <= 0.0:
    parser.error("memory and network timeouts must be positive")
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
    return str(path)


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


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


def fetch(
    url: str, destination: Path, timeout_s: float, *, accept: str,
) -> bytes:
  request = urllib.request.Request(
      url, headers={"Accept": accept, "User-Agent": "intel-qwen36-source-bound"})
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    value = response.read()
  destination.write_bytes(value)
  return value


def fetch_pull(
    owner: str, repo: str, number: int, raw: Path, timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
  url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
  value = fetch(
      url, raw / f"{owner}-{repo}-pr{number}.json", timeout_s,
      accept="application/vnd.github+json")
  payload = json.loads(value)
  if not isinstance(payload, dict) or payload.get("number") != number:
    raise ValueError(f"unexpected pull-request response: {url}")
  summary = {
      "number": number,
      "title": payload.get("title"),
      "html_url": payload.get("html_url"),
      "state": payload.get("state"),
      "draft": payload.get("draft"),
      "created_at": payload.get("created_at"),
      "updated_at": payload.get("updated_at"),
      "merged_at": payload.get("merged_at"),
      "head_sha": payload.get("head", {}).get("sha"),
      "body_sha256": sha256_bytes(str(payload.get("body", "")).encode()),
  }
  return payload, summary


def fhs_covers(m: int, _k: int) -> bool:
  return m <= 288 or 289 <= m <= 1728 or m >= 1729


def foos_covers(m: int, k: int) -> bool:
  return (
      m <= 1536
      or (2048 <= m <= 5375 and 4864 <= k <= 11520)
      or (2048 <= m <= 4096 and k >= 11521)
      or m >= 5376)


def parse_ir(
    path: Path,
) -> tuple[
    dict[int, ET.Element],
    dict[int, dict[int, tuple[int, int]]],
]:
  root = ET.parse(path).getroot()
  layers_node = root.find("layers")
  edges_node = root.find("edges")
  if layers_node is None or edges_node is None:
    raise ValueError(f"missing IR layers/edges: {path}")
  layers = {int(node.attrib["id"]): node for node in layers_node}
  incoming: dict[int, dict[int, tuple[int, int]]] = {
      node_id: {} for node_id in layers}
  for edge in edges_node:
    incoming[int(edge.attrib["to-layer"])][int(edge.attrib["to-port"])] = (
        int(edge.attrib["from-layer"]), int(edge.attrib["from-port"]))
  return layers, incoming


def port_shapes(node: ET.Element, section: str) -> dict[int, tuple[int, ...]]:
  value = node.find(section)
  if value is None:
    return {}
  return {
      int(port.attrib["id"]): tuple(int(dim.text or "-1")
                                    for dim in port.findall("dim"))
      for port in value.findall("port")}


def output_shape(
    layers: dict[int, ET.Element], node_id: int, port: int | None = None,
) -> tuple[int, ...]:
  shapes = port_shapes(layers[node_id], "output")
  if port is not None and port in shapes:
    return shapes[port]
  return next(iter(shapes.values()), ())


def unsqueeze_axis(
    input_shape: tuple[int, ...], output_shape_value: tuple[int, ...],
) -> int:
  if len(output_shape_value) != len(input_shape) + 1:
    return -1
  for axis, value in enumerate(output_shape_value):
    if value != 1:
      continue
    without_axis = output_shape_value[:axis] + output_shape_value[axis + 1:]
    if all(left < 0 or right < 0 or left == right
           for left, right in zip(input_shape, without_axis)):
      return axis
  return -1


def audit_locked_graph(path: Path) -> dict[str, Any]:
  layers, incoming = parse_ir(path)

  def node_type(node_id: int | None) -> str:
    if node_id is None:
      return ""
    return str(layers[node_id].attrib.get("type", ""))

  def source(node_id: int, port: int = 0) -> int | None:
    value = incoming.get(node_id, {}).get(port)
    return None if value is None else value[0]

  def fusable(node_id: int | None) -> bool:
    return any(label in node_type(node_id)
               for label in ("FullyConnected", "MatMul", "Transpose"))

  binary_arithmetic = {
      "Add", "Subtract", "Multiply", "Divide", "Maximum", "Minimum",
      "Power", "SquaredDifference", "Mod", "FloorMod"}
  sink_matches: list[dict[str, Any]] = []
  for node_id, node in layers.items():
    if node_type(node_id) not in binary_arithmetic:
      continue
    if len(incoming[node_id]) != 2:
      continue
    result_shape = output_shape(layers, node_id)
    for input_port in sorted(incoming[node_id]):
      reshape_id, _ = incoming[node_id][input_port]
      if node_type(reshape_id) not in ("Reshape", "Unsqueeze"):
        continue
      reshape_inputs = port_shapes(layers[reshape_id], "input")
      axis = unsqueeze_axis(
          reshape_inputs.get(0, ()), output_shape(layers, reshape_id))
      if axis < 0:
        continue
      other_port = 1 - input_port
      if other_port not in incoming[node_id]:
        continue
      other_id, other_output_port = incoming[node_id][other_port]
      other_shape = output_shape(layers, other_id, other_output_port)
      if (len(other_shape) != len(result_shape) or axis >= len(other_shape)
          or other_shape[axis] != 1):
        continue
      producer_id = source(reshape_id)
      is_fusable = fusable(producer_id)
      if not is_fusable and node_type(producer_id) == "Add":
        is_fusable = any(fusable(parent_id)
                         for parent_id, _ in incoming[producer_id].values())
      if not is_fusable:
        continue
      sink_matches.append({
          "node_id": node_id,
          "node_name": node.attrib.get("name"),
          "node_type": node_type(node_id),
          "producer_id": producer_id,
          "producer_type": node_type(producer_id),
          "axis": axis,
      })
      break

  weight_convert_users: dict[int, set[int]] = defaultdict(set)
  weight_parameter_matmuls: set[int] = set()
  weight_constant_matmuls: set[int] = set()
  matmul_ids = [node_id for node_id in layers
                if node_type(node_id) == "MatMul"]
  weight_matmul_ids = [
      node_id for node_id in matmul_ids
      if "rotary_emb/aten::matmul" not in str(
          layers[node_id].attrib.get("name", ""))]
  for matmul_id in weight_matmul_ids:
    weight_id = source(matmul_id, 1)
    if weight_id is None:
      continue
    stack = [weight_id]
    visited: set[int] = set()
    while stack:
      ancestor_id = stack.pop()
      if ancestor_id in visited:
        continue
      visited.add(ancestor_id)
      ancestor_type = node_type(ancestor_id)
      if ancestor_type == "Convert":
        weight_convert_users[ancestor_id].add(matmul_id)
      elif ancestor_type == "Parameter":
        weight_parameter_matmuls.add(matmul_id)
      elif ancestor_type == "Const":
        weight_constant_matmuls.add(matmul_id)
      stack.extend(parent_id for parent_id, _
                   in incoming[ancestor_id].values())
  shared_weight_converts = {
      node_id: users for node_id, users in weight_convert_users.items()
      if len(users) > 1}

  gated_mlp_rows: list[dict[str, Any]] = []
  for down_id in matmul_ids:
    multiply_id = source(down_id)
    if node_type(multiply_id) != "Multiply":
      continue
    parents = [source(multiply_id, port) for port in (0, 1)]
    for swish_id, up_id in (parents, list(reversed(parents))):
      if node_type(swish_id) != "Swish" or node_type(up_id) != "MatMul":
        continue
      gate_id = source(swish_id)
      if node_type(gate_id) != "MatMul":
        continue
      if source(gate_id) != source(up_id):
        continue
      ids = (gate_id, up_id, down_id)
      weight_ranks = [len(port_shapes(layers[value], "input").get(1, ()))
                      for value in ids]
      intermediate_shape = output_shape(layers, gate_id)
      gated_mlp_rows.append({
          "down_id": down_id,
          "down_name": layers[down_id].attrib.get("name"),
          "gate_id": gate_id,
          "up_id": up_id,
          "weight_ranks": weight_ranks,
          "rank2_weights": weight_ranks == [2, 2, 2],
          "intermediate_elements_per_token": (
              intermediate_shape[-1] if intermediate_shape else -1),
      })
      break

  return {
      "layer_count": len(layers),
      "matmul_count": len(matmul_ids),
      "weight_matmul_count": len(weight_matmul_ids),
      "sink_unsqueeze_exact_match_count": len(sink_matches),
      "sink_unsqueeze_exact_matches": sink_matches,
      "weight_convert_ancestor_count": len(weight_convert_users),
      "shared_weight_convert_count": len(shared_weight_converts),
      "shared_weight_convert_user_counts": sorted(
          len(users) for users in shared_weight_converts.values()),
      "weight_parameter_matmul_count": len(weight_parameter_matmuls),
      "weight_constant_matmul_count": len(weight_constant_matmuls),
      "gated_mlp_structural_match_count": len(gated_mlp_rows),
      "gated_mlp_rank2_match_count": sum(
          bool(row["rank2_weights"]) for row in gated_mlp_rows),
      "gated_mlp_grouped_rank3_match_count": sum(
          row["weight_ranks"] == [3, 3, 3] for row in gated_mlp_rows),
      "gated_mlp_rank2_intermediate_elements_per_token": sum(
          int(row["intermediate_elements_per_token"])
          for row in gated_mlp_rows if row["rank2_weights"]),
      "gated_mlp_rows": gated_mlp_rows,
  }


def audit_runtime_graph(path: Path) -> dict[str, Any]:
  layers, _ = parse_ir(path)
  shared = {"gate": 0, "up": 0, "down": 0}
  gated_mlp_count = 0
  for node in layers.values():
    node_type_value = str(node.attrib.get("type", ""))
    name = str(node.attrib.get("name", ""))
    if "GatedMLP" in node_type_value or "gated_mlp" in name.lower():
      gated_mlp_count += 1
    if node_type_value != "FullyConnected" or ".shared_expert." not in name:
      continue
    for label in shared:
      if f".{label}_proj/" in name:
        shared[label] += 1
  return {
      "layer_count": len(layers),
      "gated_mlp_count": gated_mlp_count,
      "shared_expert_fc_counts": shared,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  raw = output / "raw"
  raw.mkdir()
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required_paths = (
      STATUS, ROUTES, REJECTED, FIXED_COMPONENT, PROVIDER_TRACE,
      LOCKED_MODEL_XML, RUNTIME_GRAPH, PROVIDER_BOUND, GATED_MLP_TRANSFORM,
      GATED_MLP_OP, GATED_MLP_PIPELINE, GATED_MLP_REF, SHARED_DQ_PATCH,
      SUBGROUP_DQ_PATCH)
  missing = [display_path(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing upstream-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  fixed = load_json(FIXED_COMPONENT)
  provider_bound = load_json(PROVIDER_BOUND)
  routes = load_json(ROUTES)
  rejected = load_json(REJECTED)
  status_text = STATUS.read_text(encoding="utf-8")
  status_flat = re.sub(r"\s+", " ", status_text)
  trace_text = PROVIDER_TRACE.read_text(encoding="utf-8", errors="replace")
  shared_patch = SHARED_DQ_PATCH.read_text(encoding="utf-8")
  subgroup_patch = SUBGROUP_DQ_PATCH.read_text(encoding="utf-8")
  gated_mlp_transform = GATED_MLP_TRANSFORM.read_text(encoding="utf-8")
  gated_mlp_op = GATED_MLP_OP.read_text(encoding="utf-8")
  gated_mlp_pipeline = GATED_MLP_PIPELINE.read_text(encoding="utf-8")
  gated_mlp_ref = GATED_MLP_REF.read_text(encoding="utf-8")
  locked_graph = audit_locked_graph(LOCKED_MODEL_XML)
  runtime_graph = audit_runtime_graph(RUNTIME_GRAPH)
  sample_memory("after-local-evidence", stop_bytes, memory)

  pulls: dict[str, dict[str, Any]] = {}
  pull_payloads: dict[str, dict[str, Any]] = {}
  for label, (owner, repo, number) in PULLS.items():
    payload, summary = fetch_pull(
        owner, repo, number, raw, args.network_timeout_s)
    pull_payloads[label] = payload
    pulls[label] = summary

  fhs_patch_bytes = fetch(
      FHS_PATCH_URL, raw / f"onednn-{FHS_COMMIT}.patch",
      args.network_timeout_s, accept="text/plain")
  foos_patch_bytes = fetch(
      FOOS_PATCH_URL, raw / f"onednn-{FOOS_COMMIT}.patch",
      args.network_timeout_s, accept="text/plain")
  sink_head = str(pulls["openvino_sink_unsqueeze_eltwise"]["head_sha"])
  sink_source_bytes = fetch(
      "https://raw.githubusercontent.com/openvinotoolkit/openvino/"
      f"{sink_head}/src/plugins/intel_gpu/src/plugin/transformations/"
      "sink_unsqueeze_through_eltwise.cpp",
      raw / f"openvino-{sink_head}-sink-unsqueeze-through-eltwise.cpp",
      args.network_timeout_s, accept="text/plain")
  gated_mlp_patch_bytes = fetch(
      "https://patch-diff.githubusercontent.com/raw/uxlfoundation/oneDNN/"
      "pull/5413.patch",
      raw / "uxlfoundation-oneDNN-pr5413.patch",
      args.network_timeout_s, accept="text/plain")
  fhs_patch = fhs_patch_bytes.decode("utf-8", errors="replace")
  foos_patch = foos_patch_bytes.decode("utf-8", errors="replace")
  sink_source = sink_source_bytes.decode("utf-8", errors="replace")
  gated_mlp_patch = gated_mlp_patch_bytes.decode("utf-8", errors="replace")
  sample_memory("after-upstream-evidence", stop_bytes, memory)

  fhs_added = [
      line for line in fhs_patch.splitlines()
      if line.startswith("+") and "{'G', \"gemm\", {\"F\", \"H\", \"S\"}" in line
      and '{"N", "N", "N"}' in line]
  foos_added = [
      line for line in foos_patch.splitlines()
      if line.startswith("+") and "{'G', \"gemm\", {\"[FO]\", \"O\", \"S\"}" in line
      and '{"N", "N", "N"}' in line]
  fhs_ranges_exact = (
      len(fhs_added) == 3
      and any("{288, 1, -1}" in line for line in fhs_added)
      and any("{289, 1, -1}" in line and "{1728, 1, -1}" in line
              for line in fhs_added)
      and any("{1729, 1, -1}" in line and "{-1, 1, -1}" in line
              for line in fhs_added))
  foos_ranges_exact = (
      len(foos_added) == 4
      and any("{1536, 1, -1}" in line for line in foos_added)
      and any("{2048, 1, 4864}" in line and "{5375, 1, 11520}" in line
              for line in foos_added)
      and any("{2048, 1, 11521}" in line and "{4096, 1, -1}" in line
              for line in foos_added)
      and any("{5376, 1, -1}" in line for line in foos_added))

  fixed_rows = {
      str(row["name"]): row for row in fixed.get("cohorts", [])
      if isinstance(row, dict) and int(row.get("count", 0)) > 0}
  cohort_rows: list[dict[str, Any]] = []
  for name, m, k, count in EXPECTED_COHORTS:
    stored = fixed_rows.get(name, {})
    exact = (
        stored.get("m") == m and stored.get("k") == k
        and stored.get("count") == count)
    if not exact:
      raise ValueError(f"fixed cohort changed: {name}")
    operations = 2 * m * k * count
    weights = m * k * count
    cohort_rows.append({
        "name": name, "m": m, "n": 1, "k": k, "count": count,
        "operations": operations,
        "parameter_bytes": int(stored["cohort_bytes"]),
        "fixed_component_ms": float(stored["cohort_ms"]),
        "fhs_nnn_covered": fhs_covers(m, k),
        "foos_nnn_covered": foos_covers(m, k),
        "weight_elements": weights,
    })

  total_ops = sum(int(row["operations"]) for row in cohort_rows)
  total_weights = sum(int(row["weight_elements"]) for row in cohort_rows)
  total_outputs = sum(
      int(row["m"]) * int(row["count"]) for row in cohort_rows)
  u4_bytes = total_weights / 2
  group64_metadata_bytes = total_weights / 64 * 2.5
  per_output_metadata_bytes = total_outputs * 2.5
  extra_group64_bytes = group64_metadata_bytes - per_output_metadata_bytes
  required_gflops = total_ops / (FC_TARGET_MS / 1000) / 1e9
  fhs_compute_ms = {
      str(rate): total_ops / (rate * 1e9) * 1000
      for rate in FHS_PUBLISHED_GFLOPS}
  extra_group64_serial_ms = (
      extra_group64_bytes / (PHYSICAL_CARRIER_GBPS * 1e9) * 1000)
  published_fast_ms = min(fhs_compute_ms.values())
  published_slow_ms = max(fhs_compute_ms.values())
  fhs_envelope_low_ms = published_fast_ms
  fhs_envelope_high_ms = published_slow_ms + extra_group64_serial_ms

  foos_covered_ops = sum(
      int(row["operations"]) for row in cohort_rows
      if row["foos_nnn_covered"])
  foos_uncovered_ops = total_ops - foos_covered_ops
  foos_uncovered_fixed_ms = sum(
      float(row["fixed_component_ms"]) for row in cohort_rows
      if not row["foos_nnn_covered"])
  foos_covered_budget_ms = FC_TARGET_MS - foos_uncovered_fixed_ms
  foos_required_covered_gflops = (
      foos_covered_ops / (foos_covered_budget_ms / 1000) / 1e9)

  trace_fhs_t_shapes = sorted({
      (int(m), int(k)) for m, k in re.findall(
          r"selection:m:(\d+),n:1,k:(\d+).*entry:G gemm FHS T", trace_text)
  })
  trace_fhs_group64 = (
      "scalea[g1x64,H] gemm [fH]H[SH] T" in trace_text)
  trace_decode_foos = bool(re.search(
      r"selection:m:\d+,n:1,k:\d+.*entry:G gemm \[FO\]OS", trace_text))

  fhs_body = str(pull_payloads["onednn_fhs_nnn"].get("body", ""))
  fhs_perf_exact = all(str(value) in fhs_body for value in FHS_PUBLISHED_GFLOPS)
  foos_body = str(pull_payloads["onednn_foos_nnn"].get("body", ""))
  dq_body = str(pull_payloads["openvino_dq_consistency"].get("body", ""))
  nnn_body = str(
      pull_payloads["openvino_int4_nnn_shared_weights"].get("body", ""))
  nontransposed_policy_body = str(
      pull_payloads["openvino_nontransposed_f16_policy"].get("body", ""))
  sink_body = str(
      pull_payloads["openvino_sink_unsqueeze_eltwise"].get("body", ""))
  weight_sharing_body = str(
      pull_payloads["openvino_preserve_weight_sharing"].get("body", ""))
  share_body = str(pull_payloads["openvino_shared_dq"].get("body", ""))
  subgroup_body = str(pull_payloads["openvino_subgroup64_dq"].get("body", ""))
  dgpu_body = str(pull_payloads["openvino_skip_dq_dgpu"].get("body", ""))
  revert_body = str(
      pull_payloads["openvino_revert_dq_consistency"].get("body", ""))
  gated_mlp_body = str(
      pull_payloads["onednn_gated_mlp_inplace"].get("body", ""))

  fhs_published_rows = [
      {**row,
       "nnn_to_tnn_ratio": row["nnn_gflops"] / row["tnn_gflops"],
       "nnn_faster_than_tnn": row["nnn_gflops"] > row["tnn_gflops"]}
      for row in FHS_PUBLISHED_ROWS]
  fhs_positive_rows = [
      row for row in fhs_published_rows if row["nnn_faster_than_tnn"]]
  fhs_automatic_policy_cohorts = [
      row["name"] for row in cohort_rows if int(row["k"]) >= 8192]

  provider_ceiling = provider_bound["overlap_free_ceiling"]
  remaining_required_ms = float(
      provider_bound["budget"]["remaining_required_ms_per_token"])
  all_graph_provider_ceiling_ms = float(
      provider_ceiling["provider_launch_ceiling_ms_per_token"])
  small_tensor_gbps = float(provider_ceiling["small_tensor_gbps"])
  gated_mlp_saved_traffic_bytes = (
      int(locked_graph["gated_mlp_rank2_intermediate_elements_per_token"])
      * 4)
  gated_mlp_saved_traffic_ms = (
      gated_mlp_saved_traffic_bytes / (small_tensor_gbps * 1e9) * 1000)
  gated_mlp_impossible_ceiling_ms = (
      all_graph_provider_ceiling_ms + gated_mlp_saved_traffic_ms)
  gated_mlp_residual_shortfall_ms = (
      remaining_required_ms - gated_mlp_impossible_ceiling_ms)

  sink_source_contract_exact = all(value in sink_source for value in (
      "BinaryElementwiseArithmetic", "unsqueeze_axis",
      "FullyConnected", "MatMul", "Transpose"))
  gated_mlp_local_contract_exact = (
      "GPU_DEBUG_VALUE_OR(config.get_disable_gated_mlp_fusion(), true)"
      in gated_mlp_pipeline
      and "rank().compatible(2)" in gated_mlp_op
      and "wrap_type<ov::op::v0::MatMul>" in gated_mlp_transform
      and "get_up_dst_md(up_dst_md)" in gated_mlp_ref)
  gated_mlp_patch_contract_exact = all(value in gated_mlp_patch for value in (
      "reduce memory usage via binary_mul(dst)",
      "reads_dst_buffer = true", "get_up_dst_md(up_dst_md)"))

  fhs_complete_bound = False
  foos_complete_bound = False
  admitted = fhs_complete_bound or foos_complete_bound
  active_route = routes.get("active_route", {}).get("id")
  closed_routes = {
      row.get("route") for row in rejected.get("rejected", [])
      if isinstance(row, dict)}

  candidates = {
      "fhs_nnn_nontransposed": {
          "upstream_pr": pulls["onednn_fhs_nnn"],
          "commit": FHS_COMMIT,
          "patch_url": FHS_PATCH_URL,
          "patch_sha256": sha256_bytes(fhs_patch_bytes),
          "strategy_rows": len(fhs_added),
          "covered_cohorts": [
              row["name"] for row in cohort_rows if row["fhs_nnn_covered"]],
          "covered_cohort_count": sum(
              bool(row["fhs_nnn_covered"]) for row in cohort_rows),
          "published_gflops": list(FHS_PUBLISHED_GFLOPS),
          "published_rows": fhs_published_rows,
          "published_nnn_faster_rows": fhs_positive_rows,
          "published_case_contract": (
              "representative Xe2 FHS NNN n=1 with per-output-channel "
              "weight scale and zero point; not the five exact group64 cohorts"),
          "required_effective_gflops": required_gflops,
          "published_compute_ms_by_gflops": fhs_compute_ms,
          "group64_metadata_bytes": group64_metadata_bytes,
          "per_output_metadata_bytes": per_output_metadata_bytes,
          "extra_group64_metadata_bytes": extra_group64_bytes,
          "extra_group64_serial_ms_at_physical_carrier": (
              extra_group64_serial_ms),
          "source_evidence_envelope_ms": {
              "optimistic_published_compute_only": fhs_envelope_low_ms,
              "conservative_slowest_published_plus_serial_metadata": (
                  fhs_envelope_high_ms),
              "target": FC_TARGET_MS,
              "straddles_target": (
                  fhs_envelope_low_ms < FC_TARGET_MS < fhs_envelope_high_ms),
          },
          "current_decode_already_uses_transposed_fhs_group64": (
              trace_fhs_group64),
          "current_decode_fhs_t_shapes": [
              {"m": m, "n": 1, "k": k} for m, k in trace_fhs_t_shapes],
          "openvino_nontransposed_int4_scope": (
              "PR 36845 enables Parameter/shared-weight compilation; it does "
              "not publish a decode cut for the locked Constant-weight graph"),
          "openvino_automatic_nontransposed_policy": {
              "upstream_pr": pulls["openvino_nontransposed_f16_policy"],
              "contract": (
                  "uncompressed f16, K >= 8192, and thin M or N; locked "
                  "cohorts are compressed U4 with K <= 4096"),
              "eligible_locked_cohorts": fhs_automatic_policy_cohorts,
          },
          "locked_weight_parameter_matmuls": (
              locked_graph["weight_parameter_matmul_count"]),
          "locked_weight_constant_matmuls": (
              locked_graph["weight_constant_matmul_count"]),
          "complete_source_bound": fhs_complete_bound,
          "missing_evidence": [
              "exact five-cohort group64 timing or a source-derived upper bound",
              "direct-consumer accounting that avoids materialized Crop/split work",
              "evidence that NNN improves rather than merely matches current TNN",
          ],
          "disposition": "reject_current_upstream_consumer_no_locked_cohort",
      },
      "foos_nnn_shared_dq": {
          "upstream_pr": pulls["onednn_foos_nnn"],
          "commit": FOOS_COMMIT,
          "patch_url": FOOS_PATCH_URL,
          "patch_sha256": sha256_bytes(foos_patch_bytes),
          "strategy_rows": len(foos_added),
          "covered_cohorts": [
              row["name"] for row in cohort_rows if row["foos_nnn_covered"]],
          "uncovered_cohorts": [
              row["name"] for row in cohort_rows
              if not row["foos_nnn_covered"]],
          "covered_operations_fraction": foos_covered_ops / total_ops,
          "uncovered_fixed_component_ms": foos_uncovered_fixed_ms,
          "remaining_covered_budget_ms_before_dq": foos_covered_budget_ms,
          "required_covered_gflops_before_dq": foos_required_covered_gflops,
          "published_claim": "up to approximately 1.5-2x; chart only",
          "standalone_dq_required_for_onednn_by_pr36078": (
              "oneDNN FC" in dq_body and "keep standalone DQ" in dq_body),
          "current_decode_selects_foos": trace_decode_foos,
          "dq_policy_is_in_upstream_flux": (
              pulls["openvino_revert_dq_consistency"]["state"] == "open"),
          "complete_source_bound": foos_complete_bound,
          "missing_evidence": [
              "two uncovered M=2048 cohorts",
              "exact group64 FC timing instead of a graphical speedup claim",
              "standalone shared-DQ dispatch and arithmetic cost",
              "settled OpenVINO runtime-skip policy",
          ],
          "disposition": "lower_rank_track_only_not_admitted",
      },
      "upstream_dq_share_and_subgroup64": {
          "shared_pr": pulls["openvino_shared_dq"],
          "subgroup_pr": pulls["openvino_subgroup64_dq"],
          "local_shared_patch": display_path(SHARED_DQ_PATCH),
          "local_subgroup_patch": display_path(SUBGROUP_DQ_PATCH),
          "already_present_in_accepted_carrier": True,
          "disposition": "not_a_new_cut",
      },
      "openvino_sink_unsqueeze_eltwise": {
          "upstream_pr": pulls["openvino_sink_unsqueeze_eltwise"],
          "source_sha256": sha256_bytes(sink_source_bytes),
          "source_contract_exact": sink_source_contract_exact,
          "locked_ir_exact_match_count": (
              locked_graph["sink_unsqueeze_exact_match_count"]),
          "locked_ir_exact_matches": (
              locked_graph["sink_unsqueeze_exact_matches"]),
          "disposition": "reject_zero_locked_graph_matches",
      },
      "openvino_preserve_weight_sharing": {
          "upstream_pr": pulls["openvino_preserve_weight_sharing"],
          "locked_weight_convert_ancestor_count": (
              locked_graph["weight_convert_ancestor_count"]),
          "shared_weight_convert_count": (
              locked_graph["shared_weight_convert_count"]),
          "weight_parameter_matmul_count": (
              locked_graph["weight_parameter_matmul_count"]),
          "disposition": "reject_zero_shared_weight_convert_fanout",
      },
      "onednn_gated_mlp_inplace": {
          "upstream_pr": pulls["onednn_gated_mlp_inplace"],
          "patch_sha256": sha256_bytes(gated_mlp_patch_bytes),
          "local_contract_exact": gated_mlp_local_contract_exact,
          "patch_contract_exact": gated_mlp_patch_contract_exact,
          "fusion_default_disabled": True,
          "locked_structural_matches": (
              locked_graph["gated_mlp_structural_match_count"]),
          "locked_rank2_matches": (
              locked_graph["gated_mlp_rank2_match_count"]),
          "locked_grouped_rank3_matches": (
              locked_graph["gated_mlp_grouped_rank3_match_count"]),
          "current_runtime_gated_mlp_count": (
              runtime_graph["gated_mlp_count"]),
          "current_runtime_shared_expert_fc_counts": (
              runtime_graph["shared_expert_fc_counts"]),
          "saved_intermediate_traffic_bytes_per_token": (
              gated_mlp_saved_traffic_bytes),
          "saved_intermediate_traffic_ms_at_small_tensor_rate": (
              gated_mlp_saved_traffic_ms),
          "all_graph_provider_ceiling_ms": all_graph_provider_ceiling_ms,
          "impossible_combined_ceiling_ms": gated_mlp_impossible_ceiling_ms,
          "remaining_required_ms": remaining_required_ms,
          "residual_shortfall_ms": gated_mlp_residual_shortfall_ms,
          "bound_rule": (
              "assign every stored graph provider launch/set-argument cost "
              "plus the complete removed intermediate traffic to GatedMLP; "
              "the primitive still executes three nested GEMMs"),
          "disposition": "reject_below_remaining_aggregate_cut",
      },
  }

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("active_owner_gate_is_current",
            active_route == "openvino_locked_target_owner_contract_decision",
            active_route=active_route),
      check("status_requires_complete_new_capability_bound",
            "independently verified new capability with a complete "
            "source-derived bound" in status_flat),
      check("fixed_component_boundary_is_exact",
            fixed.get("aggregate", {}).get("non_lm_fc_bytes") == 770_901_120
            and abs(float(fixed.get("aggregate", {}).get(
                "dominant_ms", 0.0)) - 8.86764) < 1.0e-5),
      check("fixed_fc_rejection_is_registered",
            "openvino_fixed_shape_decode_u4_f16_microkernel_v28n"
            in closed_routes),
      check("current_decode_is_fhs_t_group64", trace_fhs_group64,
            shapes=len(trace_fhs_t_shapes)),
      check("fhs_upstream_strategy_ranges_exact", fhs_ranges_exact,
            rows=len(fhs_added)),
      check("foos_upstream_strategy_ranges_exact", foos_ranges_exact,
            rows=len(foos_added)),
      check("fhs_published_numbers_found_in_primary_source", fhs_perf_exact),
      check("fhs_published_comparison_rows_exact",
            all(str(row["tnn_gflops"]) in fhs_body
                and str(row["nnn_gflops"]) in fhs_body
                for row in FHS_PUBLISHED_ROWS)
            and len(fhs_positive_rows) == 1
            and int(fhs_positive_rows[0]["k"]) == 8192,
            nnn_faster_rows=fhs_positive_rows),
      check("fhs_covers_all_five_cohorts",
            all(bool(row["fhs_nnn_covered"]) for row in cohort_rows)),
      check("fhs_source_evidence_straddles_budget",
            fhs_envelope_low_ms < FC_TARGET_MS < fhs_envelope_high_ms,
            optimistic_ms=fhs_envelope_low_ms,
            conservative_ms=fhs_envelope_high_ms,
            target_ms=FC_TARGET_MS),
      check("openvino_nontransposed_policy_excludes_locked_cohorts",
            pulls["openvino_nontransposed_f16_policy"]["merged_at"] is not None
            and "K >= 8192" in nontransposed_policy_body
            and "non-compressed f16" in nontransposed_policy_body
            and not fhs_automatic_policy_cohorts,
            eligible_locked_cohorts=fhs_automatic_policy_cohorts),
      check("locked_graph_has_no_parameter_weight_matmul",
            locked_graph["weight_parameter_matmul_count"] == 0,
            constant_weight_matmuls=(
                locked_graph["weight_constant_matmul_count"])),
      check("foos_does_not_cover_complete_boundary",
            sum(bool(row["foos_nnn_covered"]) for row in cohort_rows) == 3,
            uncovered=[row["name"] for row in cohort_rows
                       if not row["foos_nnn_covered"]]),
      check("foos_primary_source_has_only_graphical_speed_claim",
            "1.5-2x" in foos_body and "user-attachments" in foos_body),
      check("openvino_nnn_scope_is_shared_weight_parameters",
            "Parameter" in nnn_body and "shared weights" in nnn_body
            and "format_tag::ab" in nnn_body),
      check("upstream_shared_dq_duplicates_local_capability",
            "deduplicate identical DQ" in share_body
            and "shared_dynamic_quantize" in shared_patch),
      check("upstream_subgroup64_dq_duplicates_local_capability",
            "GS=64" in subgroup_body
            and "sub_group_reduce_max" in subgroup_patch),
      check("onednn_requires_standalone_dq_under_merged_policy",
            pulls["openvino_dq_consistency"]["merged_at"] is not None
            and "oneDNN FC" in dq_body and "keep standalone DQ" in dq_body),
      check("upstream_dq_policy_has_unmerged_followups",
            "57%" in dgpu_body and "Reverts #36078" in revert_body),
      check("sink_unsqueeze_source_contract_exact",
            sink_source_contract_exact
            and "unit-dimension unsqueeze" in sink_body),
      check("sink_unsqueeze_has_zero_locked_matches",
            locked_graph["sink_unsqueeze_exact_match_count"] == 0,
            matches=locked_graph["sink_unsqueeze_exact_matches"]),
      check("weight_sharing_fix_has_zero_locked_fanout",
            "shared by several MatMuls" in weight_sharing_body
            and locked_graph["shared_weight_convert_count"] == 0,
            weight_convert_ancestors=(
                locked_graph["weight_convert_ancestor_count"])),
      check("gated_mlp_local_and_patch_contracts_exact",
            gated_mlp_local_contract_exact
            and gated_mlp_patch_contract_exact
            and "~100% performance compared to 3-GEMM" in gated_mlp_body),
      check("gated_mlp_locked_topology_exact",
            locked_graph["gated_mlp_structural_match_count"] == 80
            and locked_graph["gated_mlp_rank2_match_count"] == 40
            and locked_graph["gated_mlp_grouped_rank3_match_count"] == 40
            and runtime_graph["gated_mlp_count"] == 0
            and runtime_graph["shared_expert_fc_counts"]
            == {"gate": 40, "up": 40, "down": 40}),
      check("gated_mlp_impossible_ceiling_below_remaining_cut",
            gated_mlp_impossible_ceiling_ms < remaining_required_ms
            and gated_mlp_residual_shortfall_ms > 0.0,
            impossible_ceiling_ms=gated_mlp_impossible_ceiling_ms,
            remaining_required_ms=remaining_required_ms,
            residual_shortfall_ms=gated_mlp_residual_shortfall_ms),
      check("no_new_capability_meets_reopen_contract", not admitted),
  ]
  required_checks_passed = all(bool(row["pass"]) for row in checks)

  verdict = {
      "required_checks_passed": required_checks_passed,
      "new_capability_admitted": admitted,
      "active_route": active_route,
      "next_route": active_route,
      "highest_rank_track": "independently_verified_new_capability_source_bound",
      "highest_rank_reopen_trigger": (
          "exact group64 five-cohort upper bound below 8.183 ms including "
          "direct-consumer materialization and provider overhead"),
      "reason": (
          "The upstream OpenVINO consumer selects non-transposed weights only "
          "for uncompressed f16 K>=8192, while every locked U4 cohort has "
          "K<=4096 and Constant weights. The new reshape/eltwise pass has zero "
          "locked-IR matches, the weight-sharing fix has zero shared Convert "
          "fanout, and the maximally optimistic GatedMLP provider+traffic "
          "ceiling remains below the residual cut. No current upstream route "
          "authorizes a compiler or GPU probe."),
      "compiler_invocations": 0,
      "gpu_contexts_created": 0,
      "model_workers_started": 0,
  }

  sample_memory("complete", stop_bytes, memory)
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "inputs": {
          display_path(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
          for path in required_paths},
      "upstream_snapshot": pulls,
      "cohorts": cohort_rows,
      "graph_audits": {
          "locked_model": locked_graph,
          "stored_runtime": runtime_graph,
      },
      "registered_provider_boundary": {
          "remaining_required_ms": remaining_required_ms,
          "all_graph_provider_ceiling_ms": all_graph_provider_ceiling_ms,
          "small_tensor_gbps": small_tensor_gbps,
      },
      "arithmetic": {
          "total_operations_per_token": total_ops,
          "weight_elements_per_token": total_weights,
          "u4_weight_bytes": u4_bytes,
          "group64_metadata_bytes": group64_metadata_bytes,
          "total_parameter_bytes": u4_bytes + group64_metadata_bytes,
          "required_effective_gflops": required_gflops,
          "fc_stock_ms": FC_STOCK_MS,
          "fc_target_ms": FC_TARGET_MS,
          "kill_number_ms": KILL_NUMBER_MS,
      },
      "candidates": candidates,
      "checks": checks,
      "verdict": verdict,
      "memory": {
          "stop_bytes": stop_bytes,
          "samples": memory,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory),
      },
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  manifest = {
      "schema": SCHEMA,
      "created_at": metrics["created_at"],
      "git": git,
      "required_checks_passed": required_checks_passed,
      "new_capability_admitted": admitted,
      "metrics": "metrics.json",
      "raw_files": sorted(path.name for path in raw.iterdir()),
  }
  (output / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  summary = f"""# Upstream FC capability source bound

- result: `{'PASS' if required_checks_passed else 'FAIL'}`
- new capability admitted: `{str(admitted).lower()}`
- current decode provider: FHS TNN with group-64 weights
- exact five-cohort requirement: `{required_gflops:.6f} GFLOPS`, `{FC_TARGET_MS:.3f} ms/token`
- FHS NNN catalog: covers `5/5`; OpenVINO automatic consumer: `0/5` (locked U4 K <= 4096, policy requires uncompressed f16 K >= 8192)
- FHS NNN published evidence envelope: `{fhs_envelope_low_ms:.6f}..{fhs_envelope_high_ms:.6f} ms`; only the K=8192 row beats TNN
- [FO]OS NNN: covers `3/5`; uncovered fixed cohorts already cost `{foos_uncovered_fixed_ms:.6f} ms` before standalone DQ
- sink-unsqueeze eltwise exact locked matches: `{locked_graph['sink_unsqueeze_exact_match_count']}`
- shared-weight Convert fanout: `{locked_graph['shared_weight_convert_count']}`
- GatedMLP rank2/grouped matches: `{locked_graph['gated_mlp_rank2_match_count']}/{locked_graph['gated_mlp_grouped_rank3_match_count']}`; impossible provider+traffic ceiling `{gated_mlp_impossible_ceiling_ms:.6f} ms` versus `{remaining_required_ms:.6f} ms`
- highest-rank track: `independently_verified_new_capability_source_bound`
- reopen trigger: exact group-64 five-cohort upper bound below `8.183 ms`, including direct-consumer/provider overhead
- compiler/GPU/model workers: `0/0/0`
- minimum available memory: `{metrics['memory']['minimum_available_bytes']} B`

The owner-contract gate remains active. FHS NNN is real in oneDNN, but the
upstream OpenVINO performance policy excludes every locked cohort. The other
fresh reshape, weight-sharing, and GatedMLP changes either have zero exact
consumer matches or stay below the residual aggregate cut. This source record
does not authorize a build or GPU probe.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")

  print(json.dumps({
      "schema": SCHEMA,
      "required_checks_passed": required_checks_passed,
      "new_capability_admitted": admitted,
      "output": display_path(output),
      "minimum_available_bytes": metrics["memory"]["minimum_available_bytes"],
  }, sort_keys=True))
  return 0 if required_checks_passed and not admitted else 1


if __name__ == "__main__":
  raise SystemExit(main())
