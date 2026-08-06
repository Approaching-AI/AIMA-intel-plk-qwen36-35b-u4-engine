#!/usr/bin/env python3
"""Bound the upstream device-memory Assign fix before a candidate build.

OpenVINO PR 36851 fixes a Qwen3.6 dGPU allocation bug: a GPU producer whose
only consumer is Assign was forced to write into usm_host because Assign is
registered as a CPU implementation even though it only enqueues a USM copy.
The accepted IQ36 carrier already aliases the 30 conv and 30 SSM/GDN variable
updates, but its pinned runtime predates the producer-allocation fix.  This
gate verifies the exact upstream source contract, the locked runtime graph,
the accepted alias contract, and a non-overlapping FC plus recurrent-state
bound.  It does not invoke a compiler, create a GPU context, or load the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-assign-device-memory-bound-v0"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
LINEAR_BOUND = ROOT / (
    "output/openvino-provider-aware-linear-bound-"
    "20260715Tseq1235b-cleanZ/metrics.json")
FIXED_FC = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/metrics.json")
RUNTIME_GRAPH = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1150-fixed-2k-dq-census-cleanZ/raw/2k/candidate/"
    "runtime-graph.xml")
ACCEPTED_MANIFEST = ROOT / (
    "output/openvino-hot-cold-product-"
    "20260715Tseq1204-alias-fused-linear-state-32k-o64-cleanZ/"
    "manifest.json")
ALIAS_PATCH = ROOT / "engine/openvino/iq36-level-zero-linear-state-alias.patch"
PINNED_OPENVINO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")

UPSTREAM_COMMIT = "b308ae2b0af83d3c6fd275409957b68728f02525"
UPSTREAM_PR = 36851
UPSTREAM_PATCH_URL = (
    "https://github.com/openvinotoolkit/openvino/commit/"
    f"{UPSTREAM_COMMIT}.patch")
UPSTREAM_PR_URL = (
    "https://api.github.com/repos/openvinotoolkit/openvino/pulls/"
    f"{UPSTREAM_PR}")

PINNED_SHA = "90214e5be052438cec5617ed3ea7e37df1538f68"
CURRENT_TPOT_MS = 29.748
TPOT_CAP_MS = 26.911
KILL_NUMBER_MS = CURRENT_TPOT_MS - TPOT_CAP_MS


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--network-timeout-s", type=float, default=30.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0 or args.network_timeout_s <= 0.0:
    parser.error("memory and network limits must be positive")
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


def fetch(url: str, timeout_s: float, *, accept: str) -> bytes:
  request = urllib.request.Request(
      url,
      headers={
          "Accept": accept,
          "User-Agent": "intel-qwen36-assign-device-memory-bound",
      })
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    return response.read()


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def parse_runtime_assigns(path: Path) -> dict[str, Any]:
  root = ET.parse(path).getroot()
  layers_node = root.find("layers")
  edges_node = root.find("edges")
  if layers_node is None or edges_node is None:
    raise ValueError(f"missing runtime graph layers/edges: {path}")
  layers = {int(node.attrib["id"]): node for node in layers_node}
  incoming: dict[int, list[int]] = defaultdict(list)
  for edge in edges_node:
    incoming[int(edge.attrib["to-layer"])].append(
        int(edge.attrib["from-layer"]))

  rows: list[dict[str, Any]] = []
  for node_id, node in layers.items():
    if str(node.attrib.get("type", "")).lower() != "assign":
      continue
    for producer_id in incoming[node_id]:
      producer = layers[producer_id]
      data = producer.find("data")
      rows.append({
          "assign_id": node_id,
          "assign_name": node.attrib.get("name"),
          "producer_id": producer_id,
          "producer_name": producer.attrib.get("name"),
          "producer_type": producer.attrib.get("type"),
          "producer_primitive": (
              data.attrib.get("primitiveType") if data is not None else None),
          "producer_exec_time": (
              data.attrib.get("execTimeMcs") if data is not None else None),
      })
  counts = Counter(str(row["producer_type"]) for row in rows)
  return {
      "layer_count": len(layers),
      "assign_count": len(rows),
      "producer_type_counts": dict(sorted(counts.items())),
      "gdn_ocl_ref_count": sum(
          row["producer_type"] == "GatedDeltaNet"
          and "gated_delta_net::ref" in str(row["producer_primitive"])
          for row in rows),
      "rows": rows,
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

  required = (
      STATUS, ROUTES, REJECTED, LINEAR_BOUND, FIXED_FC, RUNTIME_GRAPH,
      ACCEPTED_MANIFEST, ALIAS_PATCH,
      PINNED_OPENVINO / ".git",
  )
  missing = [display_path(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing assign-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  status_text = STATUS.read_text(encoding="utf-8")
  routes = load_json(ROUTES)
  rejected = load_json(REJECTED)
  linear = load_json(LINEAR_BOUND)
  fixed = load_json(FIXED_FC)
  accepted = load_json(ACCEPTED_MANIFEST)
  alias_patch = ALIAS_PATCH.read_text(encoding="utf-8")
  runtime = parse_runtime_assigns(RUNTIME_GRAPH)
  sample_memory("after-local-evidence", stop_bytes, memory)

  patch_bytes = fetch(
      UPSTREAM_PATCH_URL, args.network_timeout_s, accept="text/plain")
  pr_bytes = fetch(
      UPSTREAM_PR_URL, args.network_timeout_s,
      accept="application/vnd.github+json")
  (raw / f"openvino-{UPSTREAM_COMMIT}.patch").write_bytes(patch_bytes)
  (raw / f"openvinotoolkit-openvino-pr{UPSTREAM_PR}.json").write_bytes(
      pr_bytes)
  patch_text = patch_bytes.decode("utf-8", errors="replace")
  pr = json.loads(pr_bytes)
  if not isinstance(pr, dict):
    raise TypeError("unexpected upstream PR response")
  sample_memory("after-upstream-evidence", stop_bytes, memory)

  pinned_head = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=PINNED_OPENVINO, text=True,
      capture_output=True, check=True).stdout.strip()
  pinned_assign = subprocess.run(
      ["git", "show", f"{PINNED_SHA}:src/plugins/intel_gpu/src/graph/"
       "impls/cpu/assign.cpp"],
      cwd=PINNED_OPENVINO, text=True, capture_output=True,
      check=True).stdout
  pinned_primitive = subprocess.run(
      ["git", "show", f"{PINNED_SHA}:src/plugins/intel_gpu/src/graph/"
       "primitive_inst.cpp"],
      cwd=PINNED_OPENVINO, text=True, capture_output=True,
      check=True).stdout

  boundary = linear["boundary"]
  physical = linear["physical_floor"]
  current_gdn_ms = float(
      boundary["current_ms_per_token"]["gated_delta_net"])
  retained_conv_ms = float(
      boundary["current_ms_per_token"]["linear_conv_state_swish"])
  current_adjacent_ms = current_gdn_ms + retained_conv_ms
  fixed_fc_saving_ms = float(
      boundary["optimistic_fc_saving_ms_per_token"])
  adjacent_target_ms = float(boundary["adjacent_target_ms_per_token"])
  state_bytes = int(physical["all_layer_state_read_write_bytes"])
  device_carrier_gbps = float(physical["proven_paired_carrier_floor_gbps"])

  projected_device_state_ms = (
      state_bytes / (device_carrier_gbps * 1e9) * 1000)
  projected_adjacent_ms = projected_device_state_ms + retained_conv_ms
  projected_adjacent_saving_ms = current_adjacent_ms - projected_adjacent_ms
  projected_union_saving_ms = (
      fixed_fc_saving_ms + projected_adjacent_saving_ms)
  projected_tpot_ms = CURRENT_TPOT_MS - projected_union_saving_ms
  tpot_margin_ms = TPOT_CAP_MS - projected_tpot_ms

  closed_routes = {
      row.get("route") for row in rejected.get("rejected", [])
      if isinstance(row, dict)}
  pr_body = str(pr.get("body", ""))
  pr_summary = {
      "number": pr.get("number"),
      "title": pr.get("title"),
      "html_url": pr.get("html_url"),
      "state": pr.get("state"),
      "merged_at": pr.get("merged_at"),
      "head_sha": pr.get("head", {}).get("sha"),
      "body_sha256": sha256_bytes(pr_body.encode()),
  }

  source_contract_exact = all(value in patch_text for value in (
      "qwen3.6 model",
      "gated_delta_net_ref",
      "usm_host",
      "virtual bool requires_lockable_input() const { return is_cpu(); }",
      "bool requires_lockable_input() const override { return false; }",
      "return impl->requires_lockable_input()",
      "assign_producer_output_is_not_lockable",
      "allocation_type::usm_host",
  ))
  alias_contract_exact = all(value in alias_patch for value in (
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN",
      "cache_params.past.conv.",
      "cache_params.past.ssm.",
      "QueueTypes::in_order",
      "variable.set_memory(input_memory",
  ))
  upstream_is_new = (
      "requires_lockable_input" not in pinned_assign
      and "requires_lockable_input" not in pinned_primitive)

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("active_owner_gate_is_current",
            routes.get("active_route", {}).get("id")
            == "openvino_locked_target_owner_contract_decision"),
      check("status_requires_independent_new_capability_bound",
            "independently verified new capability" in re.sub(
                r"\s+", " ", status_text)),
      check("pinned_runtime_commit_is_exact", pinned_head == PINNED_SHA,
            observed=pinned_head, expected=PINNED_SHA),
      check("upstream_assign_device_memory_source_contract_is_exact",
            source_contract_exact, upstream_commit=UPSTREAM_COMMIT,
            upstream_pr=UPSTREAM_PR),
      check("upstream_pr_is_the_qwen36_assign_fix",
            pr.get("number") == UPSTREAM_PR
            and "assign" in str(pr.get("title", "")).lower()
            and "qwen3.6" in pr_body
            and "gated_delta_net_ref" in pr_body),
      check("capability_is_absent_from_pinned_runtime", upstream_is_new),
      check("locked_runtime_assign_topology_is_exact",
            runtime["assign_count"] == 60
            and runtime["producer_type_counts"]
            == {"GatedDeltaNet": 30, "Reshape": 30}
            and runtime["gdn_ocl_ref_count"] == 30,
            assign_count=runtime["assign_count"],
            producer_type_counts=runtime["producer_type_counts"],
            gdn_ocl_ref_count=runtime["gdn_ocl_ref_count"]),
      check("accepted_alias_contract_is_exact",
            alias_contract_exact
            and accepted.get("alias_linear_state_assign") is True
            and accepted.get("fuse_linear_conv_state") is True),
      check("fixed_fc_boundary_is_exact",
            fixed.get("required_checks_passed") is True
            and fixed.get("verdict") == "reject_before_graph_integration"
            and "openvino_fixed_shape_decode_u4_f16_microkernel_v28n"
            in closed_routes),
      check("recurrent_state_boundary_is_exact",
            state_bytes == 62_914_560
            and abs(current_gdn_ms - 1.319) < 1e-12
            and abs(retained_conv_ms - 0.193) < 1e-12
            and abs(device_carrier_gbps - 106.524608569878) < 1e-9),
      check("device_memory_adjacent_bound_clears_target",
            projected_adjacent_ms < adjacent_target_ms,
            projected_adjacent_ms=projected_adjacent_ms,
            adjacent_target_ms=adjacent_target_ms),
      check("nonoverlapping_fc_state_union_clears_kill_number",
            projected_union_saving_ms > KILL_NUMBER_MS,
            projected_union_saving_ms=projected_union_saving_ms,
            kill_number_ms=KILL_NUMBER_MS,
            margin_ms=projected_union_saving_ms - KILL_NUMBER_MS),
      check("projected_tpot_clears_absolute_cap",
            projected_tpot_ms < TPOT_CAP_MS,
            projected_tpot_ms=projected_tpot_ms,
            tpot_cap_ms=TPOT_CAP_MS,
            margin_ms=tpot_margin_ms),
      check("no_compiler_gpu_or_model_worker_ran",
            True, compiler_workers=0, gpu_workers=0, model_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  admitted = required_checks_passed
  verdict = (
      "admit_assign_device_memory_component_only"
      if admitted else "reject_assign_device_memory_before_build")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "upstream": {
          "commit": UPSTREAM_COMMIT,
          "commit_url": (
              "https://github.com/openvinotoolkit/openvino/commit/"
              f"{UPSTREAM_COMMIT}"),
          "pull_request": pr_summary,
          "patch_sha256": sha256_bytes(patch_bytes),
          "source_contract_exact": source_contract_exact,
          "absent_from_pinned_runtime": upstream_is_new,
      },
      "accepted_carrier": {
          "manifest": display_path(ACCEPTED_MANIFEST),
          "alias_linear_state_assign": accepted.get(
              "alias_linear_state_assign"),
          "fuse_linear_conv_state": accepted.get(
              "fuse_linear_conv_state"),
          "alias_patch": display_path(ALIAS_PATCH),
          "alias_contract_exact": alias_contract_exact,
      },
      "runtime_graph": runtime,
      "bound": {
          "current_tpot_ms": CURRENT_TPOT_MS,
          "tpot_cap_ms": TPOT_CAP_MS,
          "kill_number_ms": KILL_NUMBER_MS,
          "fixed_fc_saving_ms": fixed_fc_saving_ms,
          "current_gdn_ms": current_gdn_ms,
          "retained_linear_conv_state_swish_ms": retained_conv_ms,
          "current_adjacent_ms": current_adjacent_ms,
          "state_read_write_bytes": state_bytes,
          "proven_device_carrier_gbps": device_carrier_gbps,
          "projected_device_state_ms": projected_device_state_ms,
          "projected_adjacent_ms": projected_adjacent_ms,
          "adjacent_target_ms": adjacent_target_ms,
          "projected_adjacent_saving_ms": projected_adjacent_saving_ms,
          "projected_nonoverlapping_union_saving_ms": (
              projected_union_saving_ms),
          "projected_tpot_ms": projected_tpot_ms,
          "tpot_margin_ms": tpot_margin_ms,
          "interpretation": (
              "component admission only: preserve all current conv time, "
              "price only exact GDN recurrent-state read/write bytes at the "
              "previously proven device carrier, and add the non-overlapping "
              "fixed-FC component saving; this is not integration, product, "
              "ABBA, or speedup evidence"),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "component_build_admitted": admitted,
      "graph_integration_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "verdict": verdict,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  (output / "manifest.json").write_text(
      json.dumps({
          "schema": SCHEMA,
          "tool": display_path(Path(__file__)),
          "git": git,
          "inputs": [display_path(path) for path in required],
          "upstream_commit": UPSTREAM_COMMIT,
          "upstream_pr": UPSTREAM_PR,
          "memory_stop_bytes": stop_bytes,
      }, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  summary = "\n".join((
      "# OpenVINO Assign producer device-memory bound",
      "",
      f"Verdict: **{verdict}**. Required checks: "
      f"`{str(required_checks_passed).lower()}`. No compiler, GPU context, or "
      "model worker ran.",
      "",
      f"The locked runtime has `{runtime['assign_count']}` Assign edges: "
      f"`{runtime['producer_type_counts'].get('GatedDeltaNet', 0)}` GDN and "
      f"`{runtime['producer_type_counts'].get('Reshape', 0)}` conv-state "
      "producers. The accepted state alias removes their copies, while the "
      "pinned allocator still forces producer output to `usm_host`. Upstream "
      f"PR #{UPSTREAM_PR} adds the missing device-memory allocation contract "
      "and explicitly identifies Qwen3.6 GDN as the motivating failure.",
      "",
      f"Pricing the exact `{state_bytes:,}` recurrent-state bytes at the "
      f"proven `{device_carrier_gbps:.3f} GB/s` device carrier gives "
      f"`{projected_device_state_ms:.6f} ms/token`. Retaining the complete "
      f"`{retained_conv_ms:.3f} ms/token` conv/state/SiLU bucket gives a "
      f"`{projected_adjacent_ms:.6f} ms/token` adjacent bound versus "
      f"`{adjacent_target_ms:.6f} ms/token`.",
      "",
      f"The non-overlapping union with seq1233's fixed-FC saving is "
      f"`{projected_union_saving_ms:.6f} ms/token`, above the "
      f"`{KILL_NUMBER_MS:.6f} ms/token` kill-number by "
      f"`{projected_union_saving_ms - KILL_NUMBER_MS:.6f} ms/token`. The "
      f"projected TPOT is `{projected_tpot_ms:.6f} ms/token` versus the "
      f"`{TPOT_CAP_MS:.6f}` cap. This admits one serial short component only; "
      "it is not a speed claim and does not admit graph integration, a long "
      "row, ABBA, or output512.",
      "",
  ))
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "output": display_path(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "projected_adjacent_ms": projected_adjacent_ms,
      "projected_union_saving_ms": projected_union_saving_ms,
      "projected_tpot_ms": projected_tpot_ms,
      "tpot_margin_ms": tpot_margin_ms,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
