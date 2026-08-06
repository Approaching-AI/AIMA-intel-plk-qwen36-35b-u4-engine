#!/usr/bin/env python3
"""Bound fresh graph successors after the post-IGC opportunity watch.

This source-only gate matches OpenVINO PR 36362 against the ten locked
full-attention Q/gate VariadicSplit chains.  It also closes the neighboring
QKV transpose/crop and Broadcast titles, and records grouped-GEMM post-ops as
prefill-only.  It invokes no compiler, GPU context, or model worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-dynamic-split-inplace-bound-v0"

MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
MODEL_BIN = MODEL_XML.with_suffix(".bin")
PINNED_OPENVINO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PINNED_CROP = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/graph/crop.cpp")
PINNED_FUSING = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/graph/graph_optimizer/"
    "prepare_buffer_fusing.cpp")
PINNED_PRIMITIVE = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/graph/primitive_inst.cpp")
PINNED_MOE = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/"
    "moe_3gemm_swiglu_opt.cpp")

SEQ1238 = ROOT / (
    "output/openvino-full-attention-projection-consumer-bound-"
    "20260715Tseq1238-cleanZ/metrics.json")
SEQ1299 = ROOT / (
    "output/openvino-igc2382-component-"
    "20260717Tseq1299-control-igc2344-2k-warm17-cleanZ/"
    "raw/2k/candidate/worker-result.json")
SEQ1302 = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/metrics.json")
SEQ1136 = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1136-dq-subgroup-32k-warm17-cleanZ/"
    "raw/32k/candidate/worker-result.json")

FULL_ATTENTION_LAYERS = tuple(range(3, 40, 4))
OPENVINO_PULLS = {
    "dynamic_split_inplace": 36362,
    "qkv_permute_crop": 36336,
    "broadcast_optimization": 36343,
    "grouped_gemm_post_ops": 35924,
}
EXPECTED_TITLES = {
    36362: (
        "[GPU] Make variadic_split with dynamic split_length able to do "
        "inplace crop"),
    36336: (
        "[GPU] Eliminate unnecessary Permute and Crop primitive execution "
        "after QKV Split"),
    36343: "[GPU] Broadcast optimization",
    35924: "[GPU] Add post_ops support for grouped_gemm",
}
ONEDNN_GROUPED_POST_OPS = (
    "https://api.github.com/repos/uxlfoundation/oneDNN/pulls/5535")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--network-timeout-s", type=float, default=30.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.network_timeout_s <= 0.0 or args.memory_stop_gib <= 0.0:
    parser.error("timeouts and memory stop must be positive")
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


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable missing from /proc/meminfo")


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
      capture_output=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True, text=True,
      capture_output=True).stdout.splitlines()
  allowed = {
      "tools/intel-qwen36-openvino-dynamic-split-inplace-bound.py",
  }
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  dirty = []
  for row in rows:
    path = row[3:]
    if relative and path.startswith(relative):
      continue
    if path in allowed:
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


def fetch(
    url: str, destination: Path, timeout_s: float,
    accept: str = "application/vnd.github+json",
) -> bytes:
  request = urllib.request.Request(
      url, headers={"Accept": accept,
                    "User-Agent": "intel-qwen36-dynamic-split-bound"})
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    value = response.read()
  destination.write_bytes(value)
  return value


def fetch_json(url: str, destination: Path, timeout_s: float) -> dict[str, Any]:
  value = json.loads(fetch(url, destination, timeout_s))
  if not isinstance(value, dict):
    raise TypeError(f"expected object from {url}")
  return value


def pull_summary(payload: dict[str, Any], patch: bytes) -> dict[str, Any]:
  return {
      "number": payload.get("number"),
      "title": payload.get("title"),
      "state": payload.get("state"),
      "draft": payload.get("draft"),
      "merged_at": payload.get("merged_at"),
      "updated_at": payload.get("updated_at"),
      "head_sha": payload.get("head", {}).get("sha"),
      "base_sha": payload.get("base", {}).get("sha"),
      "html_url": payload.get("html_url"),
      "changed_files": payload.get("changed_files"),
      "additions": payload.get("additions"),
      "deletions": payload.get("deletions"),
      "body_sha256": sha256_bytes(str(payload.get("body", "")).encode()),
      "patch_sha256": sha256_bytes(patch),
      "patch_bytes": len(patch),
  }


def port_shapes(node: ET.Element, section: str) -> dict[int, tuple[int, ...]]:
  value = node.find(section)
  if value is None:
    return {}
  return {
      int(port.attrib["id"]): tuple(
          int(dim.text or "-1") for dim in port.findall("dim"))
      for port in value.findall("port")
  }


def const_i64(node: ET.Element, stream: Any) -> tuple[int, ...] | None:
  if node.attrib.get("type") != "Const":
    return None
  data = node.find("data")
  if data is None or data.attrib.get("element_type") != "i64":
    return None
  offset = int(data.attrib["offset"])
  size = int(data.attrib["size"])
  stream.seek(offset)
  value = stream.read(size)
  return struct.unpack("<" + "q" * (size // 8), value)


def locked_split_audit() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  layers_node = root.find("layers")
  edges_node = root.find("edges")
  if layers_node is None or edges_node is None:
    raise ValueError("locked IR lacks layers/edges")
  layers = {int(node.attrib["id"]): node for node in layers_node}
  by_name = {str(node.attrib.get("name", "")): int(node.attrib["id"])
             for node in layers_node}
  incoming: dict[int, dict[int, tuple[int, int]]] = {
      node_id: {} for node_id in layers}
  outgoing: dict[int, list[tuple[int, int, int]]] = {
      node_id: [] for node_id in layers}
  for edge in edges_node:
    source = int(edge.attrib["from-layer"])
    source_port = int(edge.attrib["from-port"])
    target = int(edge.attrib["to-layer"])
    target_port = int(edge.attrib["to-port"])
    incoming[target][target_port] = (source, source_port)
    outgoing[source].append((source_port, target, target_port))

  rows = []
  with MODEL_BIN.open("rb") as bin_stream:
    for layer in FULL_ATTENTION_LAYERS:
      name = (
          "__module.model.model.language_model.layers."
          f"{layer}.self_attn/prim::ListUnpack/VariadicSplit")
      node_id = by_name.get(name)
      if node_id is None:
        rows.append({"layer": layer, "present": False})
        continue
      node = layers[node_id]
      data_id = incoming[node_id][0][0]
      axis_id = incoming[node_id][1][0]
      lengths_id = incoming[node_id][2][0]
      data_shape = next(iter(port_shapes(layers[data_id], "output").values()))
      lengths_shape = next(
          iter(port_shapes(layers[lengths_id], "output").values()))
      output_shapes = list(port_shapes(node, "output").values())
      users = sorted(
          str(layers[target].attrib.get("type", ""))
          for _, target, _ in outgoing[node_id])
      rows.append({
          "layer": layer,
          "present": True,
          "node_id": node_id,
          "node_type": node.attrib.get("type"),
          "data_source_type": layers[data_id].attrib.get("type"),
          "data_shape": data_shape,
          "data_dynamic": any(dim < 0 for dim in data_shape),
          "axis_source_type": layers[axis_id].attrib.get("type"),
          "axis_value": const_i64(layers[axis_id], bin_stream),
          "split_lengths_source_id": lengths_id,
          "split_lengths_source_type": layers[lengths_id].attrib.get("type"),
          "split_lengths_shape": lengths_shape,
          "split_lengths_layout_static": all(dim >= 0 for dim in lengths_shape),
          "split_lengths_content_constant":
              layers[lengths_id].attrib.get("type") == "Const",
          "output_shapes": output_shapes,
          "user_types": users,
      })
  exact = all(
      row.get("present") is True
      and row.get("node_type") == "VariadicSplit"
      and row.get("data_source_type") == "Reshape"
      and row.get("data_shape") == (-1, -1, 16, 512)
      and row.get("data_dynamic") is True
      and row.get("axis_source_type") == "Const"
      and row.get("axis_value") == (-1,)
      and row.get("split_lengths_source_type") == "Concat"
      and row.get("split_lengths_shape") == (2,)
      and row.get("split_lengths_layout_static") is True
      and row.get("split_lengths_content_constant") is False
      and row.get("output_shapes") == [
          (-1, -1, 16, 256), (-1, -1, 16, 256)]
      and row.get("user_types") == ["Reshape", "Reshape"]
      for row in rows)
  return {
      "layers": list(FULL_ATTENTION_LAYERS),
      "rows": rows,
      "exact_dynamic_split_length_contract": exact,
      "shared_split_lengths_source_ids": sorted({
          int(row["split_lengths_source_id"])
          for row in rows if row.get("present")}),
  }


def profile_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
  rows = value.get("full_profile")
  if isinstance(rows, list):
    return rows
  phases = value.get("phases", [])
  if phases and isinstance(phases[0], dict):
    rows = phases[0].get("full_profile")
  if not isinstance(rows, list):
    raise TypeError("stored worker has no full profile")
  return rows


def runtime_audit(decode: dict[str, Any], prefill: dict[str, Any]) -> dict[str, Any]:
  decode_rows = profile_rows(decode)
  executed = [row for row in decode_rows
              if row.get("status") == "Status.EXECUTED"]
  split = [row for row in executed
           if "self_attn/prim::ListUnpack/VariadicSplit.out" in str(
               row.get("node_name", ""))
           and row.get("node_type") in ("VariadicSplit", "Crop")]
  split_counts = Counter(str(row.get("node_type")) for row in split)
  broadcasts = [row for row in executed if row.get("node_type") == "Broadcast"]
  decode_moe = [row for row in executed
                if row.get("node_type") == "MOE3GemmFusedCompressed"]
  prefill_rows = profile_rows(prefill)
  prefill_moe = [row for row in prefill_rows
                 if row.get("status") == "Status.EXECUTED"
                 and row.get("node_type") == "MOE3GemmFusedCompressed"]
  return {
      "decode_q_gate_split": {
          "counts": dict(sorted(split_counts.items())),
          "dispatches": len(split),
          "raw_real_time_us_nonadditive": sum(
              float(row.get("real_time_us", 0.0)) for row in split),
          "exec_types": sorted({str(row.get("exec_type")) for row in split}),
          "names": sorted(str(row.get("node_name")) for row in split),
      },
      "decode_broadcast": {
          "count": len(broadcasts),
          "total_real_time_us": sum(
              float(row.get("real_time_us", 0.0)) for row in broadcasts),
          "exec_types": dict(sorted(Counter(
              str(row.get("exec_type")) for row in broadcasts).items())),
      },
      "decode_moe": {
          "count": len(decode_moe),
          "total_real_time_us_nonadditive": sum(
              float(row.get("real_time_us", 0.0)) for row in decode_moe),
          "exec_types": dict(sorted(Counter(
              str(row.get("exec_type")) for row in decode_moe).items())),
      },
      "prefill_moe": {
          "count": len(prefill_moe),
          "total_real_time_us_nonadditive": sum(
              float(row.get("real_time_us", 0.0)) for row in prefill_moe),
          "exec_types": dict(sorted(Counter(
              str(row.get("exec_type")) for row in prefill_moe).items())),
      },
  }


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
      MODEL_XML, MODEL_BIN, PINNED_CROP, PINNED_FUSING, PINNED_PRIMITIVE,
      PINNED_MOE, SEQ1238, SEQ1299, SEQ1302, SEQ1136)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing dynamic-split inputs: " + ", ".join(missing))

  git = git_state(output)
  seq1238 = load_json(SEQ1238)
  seq1302 = load_json(SEQ1302)
  runtime = runtime_audit(load_json(SEQ1299), load_json(SEQ1136))
  locked = locked_split_audit()

  pulls: dict[str, dict[str, Any]] = {}
  pull_text: dict[str, dict[str, str]] = {}
  raw_files: list[Path] = []
  for name, number in OPENVINO_PULLS.items():
    json_path = raw / f"openvino-pr{number}.json"
    patch_path = raw / f"openvino-pr{number}.patch"
    payload = fetch_json(
        f"https://api.github.com/repos/openvinotoolkit/openvino/pulls/{number}",
        json_path, args.network_timeout_s)
    patch = fetch(
        f"https://github.com/openvinotoolkit/openvino/pull/{number}.patch",
        patch_path, args.network_timeout_s, "application/vnd.github.patch")
    pulls[name] = pull_summary(payload, patch)
    pull_text[name] = {
        "body": str(payload.get("body", "")),
        "patch": patch.decode("utf-8", errors="replace"),
    }
    raw_files.extend((json_path, patch_path))

  onednn_path = raw / "onednn-pr5535.json"
  onednn = fetch_json(
      ONEDNN_GROUPED_POST_OPS, onednn_path, args.network_timeout_s)
  raw_files.append(onednn_path)

  split_patch_bytes = (raw / "openvino-pr36362.patch").read_bytes()
  apply_check = subprocess.run(
      ["git", "apply", "--check", "-"], cwd=PINNED_OPENVINO,
      input=split_patch_bytes, capture_output=True)
  pinned_crop = PINNED_CROP.read_text(encoding="utf-8")
  pinned_fusing = PINNED_FUSING.read_text(encoding="utf-8")
  pinned_moe = PINNED_MOE.read_text(encoding="utf-8")

  host = seq1238["overlap_free_ceiling"]
  split_dispatches = int(runtime["decode_q_gate_split"]["dispatches"])
  set_args_per_boundary_us = (
      float(host["all_graph_set_arguments_us_charged_to_boundary"])
      / int(host["boundary_executed_launches"]))
  enqueue_per_dispatch_us = float(host["max_steady_enqueue_us_per_dispatch"])
  favorable_per_dispatch_us = set_args_per_boundary_us + enqueue_per_dispatch_us
  split_provider_ceiling_ms = split_dispatches * favorable_per_dispatch_us / 1000.0

  budget1302 = seq1302["budget"]
  residual_ms = float(budget1302["residual_after_fixed_fc_ms"])
  prior_union_ms = float(budget1302["favorable_rms_plus_igc_union_ms"])
  prior_shortfall_ms = float(budget1302["favorable_union_shortfall_ms"])
  expanded_union_ms = prior_union_ms + split_provider_ceiling_ms
  expanded_margin_ms = expanded_union_ms - residual_ms

  dynamic_patch = pull_text["dynamic_split_inplace"]["patch"]
  qkv_patch = pull_text["qkv_permute_crop"]["patch"]
  broadcast_patch = pull_text["broadcast_optimization"]["patch"]
  grouped_patch = pull_text["grouped_gemm_post_ops"]["patch"]
  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("registered_prior_bounds_are_exact",
            seq1238.get("required_checks_passed") is True
            and seq1238.get("verdict") ==
                "reject_full_attention_projection_consumer_before_source"
            and seq1302.get("required_checks_passed") is True
            and seq1302.get("verdict") ==
                "retain_rms_and_igc_release_watch_no_build"),
      check("official_pull_identities_refresh_exactly",
            all(pulls[name]["number"] == number
                    and pulls[name]["title"] == EXPECTED_TITLES[number]
                    and pulls[name]["head_sha"]
                    and pulls[name]["patch_bytes"] > 0
                    for name, number in OPENVINO_PULLS.items())),
      check("locked_ir_has_ten_exact_dynamic_split_length_consumers",
            locked["exact_dynamic_split_length_contract"]
            and locked["shared_split_lengths_source_ids"] == [1266],
            audit=locked),
      check("accepted_decode_executes_exact_twenty_split_crop_dispatches",
            runtime["decode_q_gate_split"]["counts"] == {
                "Crop": 10, "VariadicSplit": 10}
            and split_dispatches == 20
            and runtime["decode_q_gate_split"]["exec_types"] == [
                "generic_eltwise_ref__f16"],
            runtime=runtime["decode_q_gate_split"]),
      check("pr36362_changes_the_exact_rejecting_dynamic_contract",
            "is_input_dynamic" in dynamic_patch
            and "get_input_layout(2).is_static()" in dynamic_patch
            and "do_runtime_in_place_crop" in dynamic_patch
            and "input2 is not constant" in pinned_fusing
            and "get_dependency(2).is_constant()" in pinned_fusing
            and "output_shapes = shape_infer" in pinned_crop,
            full_upstream_patch_applies_to_pinned=apply_check.returncode == 0,
            full_patch_apply_stderr=apply_check.stderr.decode(
                "utf-8", errors="replace").strip(),
            note=("the complete upstream patch needs a three-production-file "
                  "context backport; the exact semantic hunk is absent")),
      check("dynamic_split_ceiling_clears_seq1302_bundle_shortfall",
            prior_shortfall_ms > 0.0
            and split_provider_ceiling_ms > prior_shortfall_ms
            and expanded_margin_ms > 0.0,
            prior_shortfall_ms=prior_shortfall_ms,
            split_dispatches=split_dispatches,
            set_arguments_per_boundary_dispatch_us=set_args_per_boundary_us,
            max_enqueue_us_per_dispatch=enqueue_per_dispatch_us,
            favorable_split_provider_ceiling_ms=split_provider_ceiling_ms,
            expanded_union_ms=expanded_union_ms,
            residual_ms=residual_ms,
            expanded_margin_ms=expanded_margin_ms,
            note=("admission ceiling only; raw PERF_COUNT is non-additive and "
                  "promotion requires an isolated serial short pair")),
      check("pr36336_qkv_matchers_have_zero_locked_match",
            "wrap_type<ov::op::v1::Split>" in qkv_patch
            and "QKVSplitReshapeMatcher" in qkv_patch
            and all(row["node_type"] == "VariadicSplit"
                    for row in locked["rows"])
            and all(row["data_source_type"] == "Reshape"
                    for row in locked["rows"])
            and all(row["output_shapes"][0][-1] == 256
                    for row in locked["rows"]),
            reason=("locked nodes are Reshape->VariadicSplit on axis -1 with "
                    "two width-256 outputs, not Transpose->Split or the "
                    "static-one QKV crop patterns")),
      check("pr36343_broadcast_has_no_steady_gpu_consumer",
            "broadcast_gpu" in broadcast_patch
            and runtime["decode_broadcast"]["count"] == 5
            and runtime["decode_broadcast"]["total_real_time_us"] == 0.0
            and all(name.startswith("broadcast_cpu_impl")
                    for name in runtime["decode_broadcast"]["exec_types"]),
            runtime=runtime["decode_broadcast"]),
      check("pr35924_is_exact_prefill_only_moe_successor",
            "grouped_gemm_prefill_swiglu" in grouped_patch
            and "post-ops" in grouped_patch
            and "exec_prefill_grouped_gemm" in pinned_moe
            and "if (token_num == 1)" in pinned_moe
            and "return exec_single_token" in pinned_moe
            and runtime["decode_moe"]["count"] == 40
            and runtime["prefill_moe"]["count"] == 40
            and onednn.get("number") == 5535
            and onednn.get("merged_at") is not None,
            decode=runtime["decode_moe"],
            prefill=runtime["prefill_moe"],
            onednn_dependency={
                "state": onednn.get("state"),
                "merged_at": onednn.get("merged_at"),
                "head_sha": onednn.get("head", {}).get("sha"),
            }),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_compiles=0,
            model_workers=0, long_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  probe_admitted = required_checks_passed and expanded_margin_ms > 0.0
  verdict = (
      "admit_one_isolated_dynamic_split_inplace_plugin_probe"
      if probe_admitted else "inconclusive")
  available_end = available_memory_bytes()

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "source_edit_admitted": probe_admitted,
      "isolated_plugin_build_admitted": probe_admitted,
      "serial_short_pair_admitted": probe_admitted,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "official_openvino": pulls,
      "official_onednn_pr5535": {
          "number": onednn.get("number"),
          "title": onednn.get("title"),
          "state": onednn.get("state"),
          "merged_at": onednn.get("merged_at"),
          "head_sha": onednn.get("head", {}).get("sha"),
          "html_url": onednn.get("html_url"),
      },
      "locked_ir": locked,
      "runtime": runtime,
      "budget": {
          "residual_after_fixed_fc_ms": residual_ms,
          "seq1302_favorable_rms_plus_igc_union_ms": prior_union_ms,
          "seq1302_shortfall_ms": prior_shortfall_ms,
          "split_dispatches": split_dispatches,
          "set_arguments_per_boundary_dispatch_us": set_args_per_boundary_us,
          "max_enqueue_us_per_dispatch": enqueue_per_dispatch_us,
          "favorable_split_provider_ceiling_ms": split_provider_ceiling_ms,
          "expanded_favorable_union_ms": expanded_union_ms,
          "expanded_union_margin_ms": expanded_margin_ms,
          "interpretation": (
              "PR36362 is admitted only because its exact independent "
              "twenty-dispatch provider ceiling closes seq1302's small "
              "bundle shortfall; this is not measured saving"),
      },
      "checks": checks,
      "directions": [
          {
              "rank": 1,
              "route": "openvino_pr36362_dynamic_split_length_inplace",
              "status": "isolated_short_probe_admitted",
              "next_gate": (
                  "backport only the three production files, build one "
                  "isolated plugin with bounded parallelism, prove the split "
                  "census change and exact short correctness, then compare a "
                  "serial control/candidate pair"),
          },
          {
              "rank": 2,
              "route": "openvino_pr35924_grouped_gemm_post_ops",
              "status": "exact_prefill_only_successor_parked",
              "next_gate": (
                  "retain for the long-prefill lane after decode is funded; "
                  "token_num==1 returns through exec_single_token first"),
          },
          {
              "rank": 3,
              "route": "openvino_pr36336_qkv_permute_crop",
              "status": "zero_locked_match",
          },
          {
              "rank": 4,
              "route": "openvino_pr36343_broadcast_optimization",
              "status": "cpu_zero_time_consumers_only",
          },
      ],
      "memory": {
          "stop_bytes": stop_bytes,
          "available_start_bytes": available_start,
          "available_end_bytes": available_end,
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git,
      "inputs": {display(path): sha256(path) for path in required},
      "official_snapshots": {
          display(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
          for path in raw_files
      },
      "compilers": 0,
      "gpu_contexts": 0,
      "model_compiles": 0,
      "model_workers": 0,
      "long_workers": 0,
  })
  report = "\n".join((
      "# Dynamic split in-place successor bound",
      "",
      f"Verdict: **{verdict}**. Required checks: "
      f"`{str(required_checks_passed).lower()}`. No compiler or worker ran.",
      "",
      "OpenVINO PR 36362 has an exact locked consumer. Every one of the ten "
      "full-attention Q/gate splits has dynamic data, a constant axis, a "
      "static two-entry split-length layout with dynamic contents, and two "
      "Reshape users. The accepted decode still executes ten VariadicSplit "
      "and ten Crop dispatches because the pinned optimizer rejects every "
      "non-constant split-length input.",
      "",
      f"Seq1302 is short by `{prior_shortfall_ms:.7f} ms`. Charging the exact "
      f"twenty dispatches the stored favorable provider ceiling gives "
      f"`{split_provider_ceiling_ms:.7f} ms`; the expanded source union is "
      f"`{expanded_union_ms:.7f} ms` versus `{residual_ms:.7f} ms`, a "
      f"`{expanded_margin_ms:.7f}-ms` admission margin. This is an upper bound, "
      "not a speed claim; only one isolated serial short pair is admitted.",
      "",
      "PR 36336 has no locked matcher shape, and the five executed Broadcast "
      "nodes are CPU implementations with zero recorded GPU time. PR 35924 "
      "does touch the exact MoE provider, but only after token_num>1 enters "
      "exec_prefill_grouped_gemm; retain it as a prefill-only successor.",
      "",
      f"Available memory stayed `{available_start} -> {available_end}` bytes; "
      "no compiler, GPU context, model compile, model worker, OOM, or restart "
      "occurred.",
      "",
  ))
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "output": display(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "split_dispatches": split_dispatches,
      "split_provider_ceiling_ms": split_provider_ceiling_ms,
      "expanded_union_ms": expanded_union_ms,
      "expanded_margin_ms": expanded_margin_ms,
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
