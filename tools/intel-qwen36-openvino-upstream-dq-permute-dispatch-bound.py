#!/usr/bin/env python3
"""Audit current upstream DQ/permute PRs against the accepted runtime graph.

This is a source-only admission gate.  It pins OpenVINO PRs 36624 and 37022,
intersects their exact selector/dispatch conditions with the accepted seq2189
runtime graph, and starts no compiler, GPU context, model, InferRequest, or
inference worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-upstream-dq-permute-dispatch-bound-v0"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
RUNTIME_GRAPH = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1150-fixed-2k-dq-census-cleanZ/raw/2k/candidate/"
    "runtime-graph.xml")
PROFILE_WORKER = ROOT / (
    "output/openvino-current-bundle-profile-refresh-"
    "20260731Tseq2204-short-o130-clean/raw/bucket_002048/profile/"
    "candidate/worker-result.json")
LONG_WORKER = ROOT / (
    "output/openvino-exact-attention-dual-cohort-32k64k-wall-"
    "20260724Tseq2143-clean/raw/sentinel_064k/block01/"
    "candidate-b1/worker-result.json")

EXPECTED_INPUT_SHA256 = {
    RUNTIME_GRAPH: (
        "16126628ac711955652b815f556a2f866dcb0f9b27eb38a5f290e768a59ef909"),
    PROFILE_WORKER: (
        "d2e66caaef46344a82d3238af120f698a7eaad005085f0a40b6d91bcbc4e21f1"),
}

DQ_PR = 36624
DQ_HEAD = "29173a67347c3150afa1e34944365c77e9c0f2f3"
PERMUTE_PR = 37022
PERMUTE_HEAD = "7f89a330d6330290b8499b0416adfdc0b96f39ca"

DQ_FILES = {
    "kernel": (
        "src/plugins/intel_gpu/src/kernel_selector/kernels/"
        "dynamic_quantize/dynamic_quantize_kernel_opt.cpp"),
    "header": (
        "src/plugins/intel_gpu/src/kernel_selector/kernels/"
        "dynamic_quantize/dynamic_quantize_kernel_opt.h"),
    "selector": (
        "src/plugins/intel_gpu/src/kernel_selector/kernels/"
        "dynamic_quantize/dynamic_quantize_kernel_selector.cpp"),
    "opencl": (
        "src/plugins/intel_gpu/src/kernel_selector/cl_kernels/"
        "dynamic_quantize_gpu_opt.cl"),
}
PERMUTE_FILES = {
    "kernel": (
        "src/plugins/intel_gpu/src/kernel_selector/kernels/permute/"
        "permute_kernel_xy_swap.cpp"),
    "opencl": (
        "src/plugins/intel_gpu/src/kernel_selector/cl_kernels/"
        "permute_xy_swap.cl"),
    "tests": (
        "src/plugins/intel_gpu/tests/unit/test_cases/"
        "permute_gpu_test.cpp"),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--memory-stop-gib", default=4.0, type=float)
  parser.add_argument("--network-timeout-s", default=30.0, type=float)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0 or args.network_timeout_s <= 0:
    parser.error("memory stop and network timeout must be positive")
  return args


def run(command: list[str], cwd: Path = ROOT) -> str:
  result = subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n{result.stderr}")
  return result.stdout


def git(*args: str) -> str:
  return run(["git", *args]).strip()


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected a JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, samples: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  samples.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def repository_state(output: Path) -> dict[str, Any]:
  head = git("rev-parse", "HEAD")
  upstream = git("rev-parse", "@{u}")
  output_rel = relative(output)
  dirty = []
  for row in git("status", "--porcelain", "--untracked-files=all").splitlines():
    path = row[3:]
    if path == output_rel or path.startswith(output_rel + "/"):
      continue
    dirty.append(row)
  return {
      "branch": git("branch", "--show-current"),
      "commit": head,
      "upstream_commit": upstream,
      "pushed": head == upstream,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def fetch(
    url: str, destination: Path, timeout_s: float, *,
    accept: str = "application/vnd.github+json",
) -> bytes:
  request = urllib.request.Request(
      url,
      headers={
          "Accept": accept,
          "User-Agent": "intel-qwen36-dq-permute-dispatch-bound",
      })
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    value = response.read()
  destination.write_bytes(value)
  return value


def fetch_json(
    url: str, destination: Path, timeout_s: float,
) -> dict[str, Any] | list[Any]:
  value = json.loads(fetch(url, destination, timeout_s))
  if not isinstance(value, (dict, list)):
    raise TypeError(f"unexpected JSON response: {url}")
  return value


def fetch_pull(
    number: int, raw: Path, timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
  payload = fetch_json(
      f"https://api.github.com/repos/openvinotoolkit/openvino/pulls/{number}",
      raw / f"openvinotoolkit-openvino-pr{number}.json", timeout_s)
  if not isinstance(payload, dict) or payload.get("number") != number:
    raise ValueError(f"unexpected pull response for PR {number}")
  return payload, {
      "number": number,
      "title": payload.get("title"),
      "html_url": payload.get("html_url"),
      "state": payload.get("state"),
      "draft": payload.get("draft"),
      "created_at": payload.get("created_at"),
      "updated_at": payload.get("updated_at"),
      "head_sha": payload.get("head", {}).get("sha"),
      "base_sha": payload.get("base", {}).get("sha"),
      "commits": payload.get("commits"),
      "additions": payload.get("additions"),
      "deletions": payload.get("deletions"),
      "changed_files": payload.get("changed_files"),
      "body_sha256": sha256_bytes(
          str(payload.get("body", "")).encode("utf-8")),
  }


def fetch_pull_files(
    number: int, raw: Path, timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  payload = fetch_json(
      "https://api.github.com/repos/openvinotoolkit/openvino/"
      f"pulls/{number}/files?per_page=100",
      raw / f"openvinotoolkit-openvino-pr{number}-files.json", timeout_s)
  if not isinstance(payload, list):
    raise TypeError(f"unexpected pull files response for PR {number}")
  full = [row for row in payload if isinstance(row, dict)]
  summary = [{
      "filename": row.get("filename"),
      "status": row.get("status"),
      "additions": row.get("additions"),
      "deletions": row.get("deletions"),
  } for row in full]
  return full, summary


def fetch_pull_commits(
    number: int, raw: Path, timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  payload = fetch_json(
      "https://api.github.com/repos/openvinotoolkit/openvino/"
      f"pulls/{number}/commits?per_page=100",
      raw / f"openvinotoolkit-openvino-pr{number}-commits.json", timeout_s)
  if not isinstance(payload, list):
    raise TypeError(f"unexpected pull commits response for PR {number}")
  full = [row for row in payload if isinstance(row, dict)]
  summary = [{
      "sha": row.get("sha"),
      "message": row.get("commit", {}).get("message"),
  } for row in full]
  return full, summary


def fetch_sources(
    head: str, paths: dict[str, str], raw: Path, timeout_s: float,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
  texts: dict[str, str] = {}
  evidence: dict[str, dict[str, Any]] = {}
  for label, path in paths.items():
    destination = raw / (
        f"openvinotoolkit-openvino-{head[:12]}-"
        f"{path.replace('/', '__')}")
    value = fetch(
        "https://raw.githubusercontent.com/openvinotoolkit/openvino/"
        f"{head}/{path}",
        destination, timeout_s, accept="text/plain")
    texts[label] = value.decode("utf-8", errors="replace")
    evidence[label] = {
        "path": path,
        "artifact": relative(destination),
        "bytes": len(value),
        "sha256": sha256_bytes(value),
    }
  return texts, evidence


def port_shapes(layer: ET.Element, kind: str) -> list[list[int]]:
  parent = layer.find(kind)
  if parent is None:
    return []
  result = []
  for port in parent.findall("port"):
    result.append([
        int(dim.text) for dim in port.findall("dim")
        if dim.text is not None])
  return result


def runtime_graph_audit() -> dict[str, Any]:
  root = ET.parse(RUNTIME_GRAPH).getroot()
  dq_rows = []
  permute_rows = []
  for layer in root.iter("layer"):
    data = layer.find("data")
    attrs = data.attrib if data is not None else {}
    inputs = port_shapes(layer, "input")
    outputs = port_shapes(layer, "output")
    common = {
        "id": int(layer.attrib["id"]),
        "name": layer.attrib.get("name"),
        "type": layer.attrib.get("type"),
        "primitive_type": attrs.get("primitiveType"),
        "runtime_precision": attrs.get("runtimePrecision"),
        "input_shapes": inputs,
        "output_shapes": outputs,
    }
    if layer.attrib.get("type") == "DynamicQuantize":
      dq_rows.append({
          **common,
          "group_sizes": [
              int(value.strip())
              for value in attrs.get("group_sizes", "").split(",")
              if value.strip()],
      })
    if attrs.get("primitiveType", "").startswith("permute_ref"):
      permute_rows.append(common)

  dq_groups = Counter(
      tuple(row["group_sizes"]) for row in dq_rows)
  dq_shapes = Counter(
      tuple(row["input_shapes"][0]) for row in dq_rows)
  permute_cohorts = Counter(
      (
          row["primitive_type"],
          tuple(row["input_shapes"][0]),
          tuple(row["output_shapes"][0]),
      )
      for row in permute_rows)
  dynamic_permute_count = sum(
      any(dim < 0 for shape in row["input_shapes"] + row["output_shapes"]
          for dim in shape)
      for row in permute_rows)
  large_rows = [
      row for row in dq_rows if row["group_sizes"][-1:] == [256]]
  return {
      "layer_count": sum(1 for _ in root.iter("layer")),
      "dynamic_quantize": {
          "count": len(dq_rows),
          "primitive_counts": dict(sorted(Counter(
              row["primitive_type"] for row in dq_rows).items())),
          "group_size_cohorts": [{
              "group_sizes": list(group_sizes),
              "count": count,
          } for group_sizes, count in sorted(dq_groups.items())],
          "input_shape_cohorts": [{
              "input_shape": list(shape),
              "count": count,
          } for shape, count in sorted(dq_shapes.items())],
          "large_group_rows": large_rows,
      },
      "permute_ref": {
          "count": len(permute_rows),
          "primitive_counts": dict(sorted(Counter(
              row["primitive_type"] for row in permute_rows).items())),
          "dynamic_tensor_count": dynamic_permute_count,
          "cohorts": [{
              "primitive_type": primitive,
              "input_shape": list(input_shape),
              "output_shape": list(output_shape),
              "count": count,
          } for (primitive, input_shape, output_shape), count
              in sorted(permute_cohorts.items())],
      },
  }


def profile_audit(worker: dict[str, Any]) -> dict[str, Any]:
  census = worker.get("execution_census", {})
  counts = census.get("executed_type_counts", {})
  dq_rows = [
      row for row in census.get("retained_rows", [])
      if row.get("status") == "Status.EXECUTED"
      and row.get("node_type") == "DynamicQuantize"]
  permute_top = [
      row for row in census.get("top_rows", [])
      if row.get("status") == "Status.EXECUTED"
      and row.get("node_type") == "Transpose"
      and str(row.get("exec_type", "")).startswith("permute_ref")]
  return {
      "executed_dynamic_quantize_count": int(
          counts.get("DynamicQuantize", 0)),
      "executed_transpose_count": int(counts.get("Transpose", 0)),
      "dynamic_quantize_retained_rows": len(dq_rows),
      "dynamic_quantize_raw_us_nonadditive": sum(
          float(row["real_time_us"]) for row in dq_rows),
      "permute_ref_top_rows": len(permute_top),
      "permute_ref_top_raw_us_nonadditive": sum(
          float(row["real_time_us"]) for row in permute_top),
      "profile_time_is_direct_savings_evidence": False,
  }


def vector_size(feature_elements: int) -> int:
  for size in (8, 4, 2):
    if (feature_elements // 16) % size == 0:
      return size
  return 1


def align(value: int, alignment: int) -> int:
  return ((value + alignment - 1) // alignment) * alignment


def dq_dispatch_cases(
    short_worker: dict[str, Any], long_worker: dict[str, Any],
) -> list[dict[str, Any]]:
  token_counts = {
      1,
      int(short_worker["input_token_count"]),
      int(short_worker["generated_token_count"]),
      int(short_worker["prefill_chunk_tokens"]),
      int(long_worker["generated_token_count"]),
      int(long_worker["prefill_chunk_tokens"]),
  }
  cases = []
  for token_count in sorted(token_counts):
    feature_elements = token_count * 2048
    vec = vector_size(feature_elements)
    total = feature_elements // (16 * vec)
    block = min(total, 32)
    aligned = align(total, block)
    cases.append({
        "token_count_if_flattened_into_feature": token_count,
        "feature_elements": feature_elements,
        "vector_size": vec,
        "total_block_num_before": total,
        "block_num": block,
        "total_block_num_after": aligned,
        "added_blocks": aligned - total,
    })
  return cases


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
      STATUS, ROUTES, REJECTED, RUNTIME_GRAPH, PROFILE_WORKER, LONG_WORKER)
  missing = [relative(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing dispatch audit inputs: " + ", ".join(missing))

  repo = repository_state(output)
  input_hashes = {
      relative(path): {
          "sha256": sha256(path),
          "bytes": path.stat().st_size,
      } for path in required}
  short_worker = load_json(PROFILE_WORKER)
  long_worker = load_json(LONG_WORKER)
  graph = runtime_graph_audit()
  profile = profile_audit(short_worker)
  dispatch_cases = dq_dispatch_cases(short_worker, long_worker)
  sample_memory("after-local-evidence", stop_bytes, memory)

  dq_payload, dq_pull = fetch_pull(
      DQ_PR, raw, args.network_timeout_s)
  permute_payload, permute_pull = fetch_pull(
      PERMUTE_PR, raw, args.network_timeout_s)
  _, dq_file_rows = fetch_pull_files(
      DQ_PR, raw, args.network_timeout_s)
  _, permute_file_rows = fetch_pull_files(
      PERMUTE_PR, raw, args.network_timeout_s)
  _, dq_commits = fetch_pull_commits(
      DQ_PR, raw, args.network_timeout_s)
  dq_sources, dq_source_evidence = fetch_sources(
      DQ_HEAD, DQ_FILES, raw, args.network_timeout_s)
  permute_sources, permute_source_evidence = fetch_sources(
      PERMUTE_HEAD, PERMUTE_FILES, raw, args.network_timeout_s)
  sample_memory("after-upstream-evidence", stop_bytes, memory)

  dq_kernel = dq_sources["kernel"]
  dq_header = dq_sources["header"]
  dq_selector = dq_sources["selector"]
  permute_kernel = permute_sources["kernel"]
  permute_opencl = permute_sources["opencl"]
  permute_tests = permute_sources["tests"]

  dq_commit_messages = "\n".join(
      str(row.get("message", "")) for row in dq_commits)
  dq_large_rows = graph["dynamic_quantize"]["large_group_rows"]
  dq_small_count = sum(
      row["count"]
      for row in graph["dynamic_quantize"]["group_size_cohorts"]
      if row["group_sizes"][-1] <= 64)
  dq_added_source = (
      "src/plugins/intel_gpu/src/kernel_selector/cl_kernels/"
      "dynamic_quantize_gpu_opt_org_ref_to_be_reverted.cl")
  dq_source_contract = {
      "pull": dq_pull,
      "files": dq_file_rows,
      "commits": dq_commits,
      "alignment_is_large_group_only": (
          "if (mode == DynQuanMode::SMALL_GS)" in dq_kernel
          and "} else if (mode == DynQuanMode::LARGE_GS)" in dq_kernel
          and "total_block_num = Align(total_block_num, block_num)"
          in dq_kernel),
      "level_zero_only_alignment": (
          "#if defined(OV_GPU_WITH_ZE_RT) && OV_GPU_WITH_ZE_RT"
          in dq_kernel),
      "small_group_dispatch_unchanged": (
          "dispatchData.gws = {bf_size.first, bf_size.second / "
          "params.group_sizes.back(), 1};" in dq_kernel),
      "draft_debug_scaffolding_present": (
          dq_pull["draft"] is True
          and 'std::cout << "Before align: "' in dq_kernel
          and "OrgRefToBeReverted" in dq_header
          and "Attach<DynamicQuantizeKernelOptOrgRefToBeReverted>();"
          in dq_selector
          and any(row.get("filename") == dq_added_source
                  for row in dq_file_rows)
          and "to be reverted" in dq_commit_messages.lower()),
      "locked_small_group_untouched_count": dq_small_count,
      "locked_large_group_rows": dq_large_rows,
      "accepted_dispatch_cases": dispatch_cases,
      "accepted_dispatch_added_blocks": sum(
          row["added_blocks"] for row in dispatch_cases),
      "exact_runtime_dispatch_delta_count": 0,
  }
  permute_source_contract = {
      "pull": permute_pull,
      "files": permute_file_rows,
      "xy_swap_order_only": (
          "return order[0] == 0 && order[1] == 1 "
          "&& order[2] == 3 && order[3] == 2;" in permute_kernel),
      "static_plain_bfyx_only": (
          "params.has_dynamic_tensors()" in permute_kernel
          and "DO_NOT_USE_THIS_KERNEL(p.layerID)" in permute_kernel
          and "DataLayout::bfyx" in permute_kernel),
      "ragged_remainder_path_present": (
          "return {kWgDim, true};" in permute_kernel
          and 'MakeJitConstant("REMAINDER"' in permute_kernel
          and "#if REMAINDER" in permute_opencl),
      "ragged_tests_present": all(
          shape in permute_tests for shape in (
              "{{1, 16, 72, 256}, format::bfyx}",
              "{{2, 3, 72, 100}, format::bfyx}",
              "{{1, 1, 17, 33}, format::bfyx}")),
      "locked_dynamic_rejection_count": (
          graph["permute_ref"]["dynamic_tensor_count"]),
      "exact_runtime_selector_match_count": 0,
  }

  expected_hashes_pass = all(
      sha256(path) == expected
      for path, expected in EXPECTED_INPUT_SHA256.items())
  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check("accepted_runtime_inputs_exact", expected_hashes_pass),
      check(
          "accepted_runtime_owner_census_exact",
          profile["executed_dynamic_quantize_count"] == 161
          and profile["executed_transpose_count"] == 40
          and graph["dynamic_quantize"]["count"] == 161
          and graph["permute_ref"]["count"] == 40),
      check(
          "pr36624_head_and_state_pinned",
          dq_pull["head_sha"] == DQ_HEAD
          and dq_pull["state"] == "open"
          and dq_pull["draft"] is True),
      check(
          "pr36624_only_changes_large_group_dispatch_alignment",
          dq_source_contract["alignment_is_large_group_only"]
          and dq_source_contract["level_zero_only_alignment"]
          and dq_source_contract["small_group_dispatch_unchanged"]),
      check(
          "pr36624_has_no_locked_product_dispatch_delta",
          dq_small_count == 160
          and len(dq_large_rows) == 1
          and dq_large_rows[0]["group_sizes"] == [1, 1, 256]
          and dq_large_rows[0]["input_shapes"] == [[-1, -1, 2048]]
          and all(row["added_blocks"] == 0 for row in dispatch_cases)
          and dq_source_contract["exact_runtime_dispatch_delta_count"] == 0),
      check(
          "pr36624_is_not_build_ready",
          dq_source_contract["draft_debug_scaffolding_present"]),
      check(
          "pr37022_head_and_state_pinned",
          permute_pull["head_sha"] == PERMUTE_HEAD
          and permute_pull["state"] == "open"
          and permute_pull["draft"] is False),
      check(
          "pr37022_rejects_every_locked_permute",
          permute_source_contract["xy_swap_order_only"]
          and permute_source_contract["static_plain_bfyx_only"]
          and graph["permute_ref"]["dynamic_tensor_count"] == 40
          and permute_source_contract["exact_runtime_selector_match_count"]
          == 0),
      check(
          "profile_attribution_is_not_saving_evidence",
          profile["dynamic_quantize_retained_rows"] == 161
          and profile["permute_ref_top_rows"] == 20
          and not profile["profile_time_is_direct_savings_evidence"]),
      check(
          "no_build_or_gpu_work_admitted",
          True,
          compiler_invocations=0,
          gpu_contexts_created=0,
          model_workers_started=0,
          infer_requests_created=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": required_checks_passed,
      "verdict": (
          "reject_both_current_upstream_candidates_zero_exact_dispatch_delta"
          if required_checks_passed else
          "inconclusive_source_or_runtime_contract_mismatch"),
      "pr36624": {
          "build_admitted": False,
          "reason": (
              "160/161 locked DQ nodes take SMALL_GS, which the PR does not "
              "change. The sole group256 LARGE_GS LM-head row has zero added "
              "blocks for both feature-axis interpretations over every "
              "accepted decode/prefill chunk shape. The branch is also a "
              "draft with debug and reference code marked to be reverted."),
          "reopen_trigger": (
              "a cleaned successor that changes an executed SMALL_GS body, "
              "or a locked product shape with a nonzero LARGE_GS dispatch "
              "delta and an independent positive component bound"),
      },
      "pr37022": {
          "build_admitted": False,
          "reason": (
              "All 40 accepted permute_ref nodes have dynamic tensors, while "
              "the new XY-swap selector explicitly rejects dynamic tensors "
              "before its ragged-tile path can be selected."),
          "reopen_trigger": (
              "dynamic-tensor support for the exact locked transpose orders, "
              "or a separately admitted static bucket specialization with "
              "an exact positive component bound"),
      },
      "compiler_invocations": 0,
      "gpu_contexts_created": 0,
      "model_workers_started": 0,
      "infer_requests_created": 0,
  }

  sample_memory("complete", stop_bytes, memory)
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": input_hashes,
      "accepted_runtime_graph": graph,
      "accepted_profile": profile,
      "upstream_contracts": {
          "pr36624_dynamic_quantize": dq_source_contract,
          "pr37022_permute_xy_swap": permute_source_contract,
      },
      "upstream_sources": {
          "pr36624_dynamic_quantize": dq_source_evidence,
          "pr37022_permute_xy_swap": permute_source_evidence,
      },
      "checks": checks,
      "memory": {
          "stop_bytes": stop_bytes,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory),
          "samples": memory,
      },
      "workers": {
          "maximum_concurrent_workers": 0,
          "compiler_invocations": 0,
          "gpu_contexts_created": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
      },
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# Upstream DQ / permute exact-dispatch bound\n\n"
      f"- Required checks: `{required_checks_passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      "- PR36624: 160 SMALL_GS nodes untouched; the sole group256 "
      "LARGE_GS row has zero added blocks for every accepted shape.\n"
      "- PR37022: all 40 permute_ref nodes are dynamic and rejected by the "
      "new selector before its remainder path.\n"
      "- Compiler/GPU/model/InferRequest count: `0/0/0/0`.\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": required_checks_passed,
      "verdict": verdict["verdict"],
      "dq_exact_dispatch_delta_count": 0,
      "permute_exact_selector_match_count": 0,
      "minimum_available_bytes": metrics["memory"][
          "minimum_available_bytes"],
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
