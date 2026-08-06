#!/usr/bin/env python3
"""Audit upstream GatedMLP micro_horz without compiling or using the GPU.

This gate pins oneDNN PR 5059 and OpenVINO PR 36139, intersects their source
contracts with the locked Qwen3.6 IR and the accepted seq2189 runtime profile,
and decides whether the pair supplies enough exact evidence to authorize a
build.  It intentionally starts no compiler, model worker, InferRequest, or GPU
context.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-upstream-gated-mlp-micro-horz-bound-v0"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
LOCKED_MODEL_XML = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
PINNED_OPENVINO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
PINNED_ONEDNN = (
    PINNED_OPENVINO / "src/plugins/intel_gpu/thirdparty/onednn_gpu")
LOCAL_PIPELINE = PINNED_OPENVINO / (
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
LOCAL_GMLP_LIST = PINNED_ONEDNN / "src/gpu/gpu_gated_mlp_list.cpp"
LOCAL_GMLP_REF = PINNED_ONEDNN / "src/gpu/intel/gated_mlp/ref.hpp"
OLD_AUDIT_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-upstream-fc-capability-bound.py")
SEQ1281 = ROOT / (
    "output/openvino-upstream-fc-capability-bound-"
    "20260717Tseq1281-cleanZ/metrics.json")
SEQ2202 = ROOT / (
    "output/openvino-qk-rope-layout-stock-half-formal-abba8-"
    "20260731Tseq2202-clean/result.json")
SEQ2204 = ROOT / (
    "output/openvino-current-bundle-profile-refresh-"
    "20260731Tseq2204-short-o130-clean/metrics.json")
SEQ2204_WORKER = ROOT / (
    "output/openvino-current-bundle-profile-refresh-"
    "20260731Tseq2204-short-o130-clean/raw/bucket_002048/profile/"
    "candidate/worker-result.json")

ONEDNN_PR = 5059
ONEDNN_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
OPENVINO_PR = 36139
OPENVINO_HEAD = "5b44316ef578066aab360eda7193cb708ba5f679"
OPENVINO_PR_ONEDNN_SHA = "6569fb284ea8e5ec628090a3d1d400485eed84b5"
PINNED_OPENVINO_SHA = "90214e5be052438cec5617ed3ea7e37df1538f68"
PINNED_ONEDNN_SHA = "20db47e2d3c4df1b66e93bed2e97d30da175512d"

ONEDNN_FILES = {
    "impl_list": "src/gpu/gpu_gated_mlp_list.cpp",
    "micro_cl": "src/gpu/intel/gated_mlp/micro_horz.cl",
    "micro_cpp": "src/gpu/intel/gated_mlp/micro_horz.cpp",
    "micro_hpp": "src/gpu/intel/gated_mlp/micro_horz.hpp",
    "tests": "tests/gtests/internals/test_gated_mlp.cpp",
}
OPENVINO_FILES = {
    "wrapper": (
        "src/plugins/intel_gpu/src/graph/impls/onednn/"
        "gated_mlp_onednn.cpp"),
    "dynamic_quantize": (
        "src/plugins/intel_gpu/src/plugin/transformations/"
        "dynamic_quantize_gated_mlp.cpp"),
    "fusion": (
        "src/plugins/intel_gpu/src/plugin/transformations/"
        "fuse_gated_mlp.cpp"),
    "op": (
        "src/plugins/intel_gpu/src/plugin/transformations/op/gated_mlp.cpp"),
    "pipeline": (
        "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp"),
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--network-timeout-s", type=float, default=30.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0 or args.network_timeout_s <= 0.0:
    parser.error("memory and network timeouts must be positive")
  return args


def run(command: list[str], cwd: Path = ROOT) -> str:
  result = subprocess.run(
      command, cwd=cwd, text=True, capture_output=True, check=False,
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n{result.stderr}")
  return result.stdout


def git(cwd: Path, *args: str) -> str:
  return run(["git", *args], cwd=cwd).strip()


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
    label: str, stop_bytes: int, samples: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  samples.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def repository_state(output: Path) -> dict[str, Any]:
  head = git(ROOT, "rev-parse", "HEAD")
  upstream = git(ROOT, "rev-parse", "@{u}")
  branch = git(ROOT, "branch", "--show-current")
  output_rel = display(output)
  dirty: list[str] = []
  for row in git(ROOT, "status", "--porcelain", "--untracked-files=all"
                 ).splitlines():
    path = row[3:]
    if path == output_rel or path.startswith(output_rel + "/"):
      continue
    dirty.append(row)
  return {
      "branch": branch,
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
          "User-Agent": "intel-qwen36-gated-mlp-source-bound",
      })
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    value = response.read()
  destination.write_bytes(value)
  return value


def fetch_json(
    url: str, destination: Path, timeout_s: float,
) -> dict[str, Any] | list[Any]:
  value = fetch(url, destination, timeout_s)
  payload = json.loads(value)
  if not isinstance(payload, (dict, list)):
    raise TypeError(f"unexpected JSON response: {url}")
  return payload


def fetch_pull(
    owner: str, repo: str, number: int, raw: Path, timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
  payload = fetch_json(
      f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}",
      raw / f"{owner}-{repo}-pr{number}.json", timeout_s)
  if not isinstance(payload, dict) or payload.get("number") != number:
    raise ValueError(f"unexpected pull request response for {number}")
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
      "head_repo": payload.get("head", {}).get("repo", {}).get("full_name"),
      "base_sha": payload.get("base", {}).get("sha"),
      "body_sha256": sha256_bytes(str(payload.get("body", "")).encode()),
  }
  return payload, summary


def fetch_pull_files(
    owner: str, repo: str, number: int, raw: Path, timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  payload = fetch_json(
      f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files"
      "?per_page=100",
      raw / f"{owner}-{repo}-pr{number}-files.json", timeout_s)
  if not isinstance(payload, list):
    raise TypeError(f"unexpected pull file response for {number}")
  rows = [{
      "filename": row.get("filename"),
      "status": row.get("status"),
      "additions": row.get("additions"),
      "deletions": row.get("deletions"),
  } for row in payload if isinstance(row, dict)]
  return [row for row in payload if isinstance(row, dict)], rows


def fetch_sources(
    owner: str, repo: str, head: str, paths: dict[str, str],
    raw: Path, timeout_s: float,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
  texts: dict[str, str] = {}
  evidence: dict[str, dict[str, Any]] = {}
  for label, source_path in paths.items():
    destination = raw / (
        f"{owner}-{repo}-{head[:12]}-{source_path.replace('/', '__')}")
    value = fetch(
        f"https://raw.githubusercontent.com/{owner}/{repo}/{head}/"
        f"{source_path}",
        destination, timeout_s, accept="text/plain")
    texts[label] = value.decode("utf-8", errors="replace")
    evidence[label] = {
        "path": source_path,
        "artifact": display(destination),
        "sha256": sha256_bytes(value),
        "bytes": len(value),
    }
  return texts, evidence


def load_old_audit() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_upstream_fc_capability_bound", OLD_AUDIT_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {OLD_AUDIT_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  for name in ("audit_locked_graph", "parse_ir", "port_shapes"):
    if not hasattr(module, name):
      raise RuntimeError(f"old audit helper missing {name}")
  return module


def locked_shape_audit(module: Any) -> dict[str, Any]:
  locked = module.audit_locked_graph(LOCKED_MODEL_XML)
  layers, _ = module.parse_ir(LOCKED_MODEL_XML)
  cohorts: Counter[tuple[Any, ...]] = Counter()
  for row in locked["gated_mlp_rows"]:
    if not row["rank2_weights"]:
      continue
    ids = [row["gate_id"], row["up_id"], row["down_id"]]
    weight_shapes = tuple(
        module.port_shapes(layers[node_id], "input").get(1, ())
        for node_id in ids)
    source_shape = module.port_shapes(
        layers[row["gate_id"]], "input").get(0, ())
    cohorts[(source_shape, weight_shapes)] += 1
  locked["rank2_shape_cohorts"] = [{
      "source_shape": list(source_shape),
      "weight_shapes": [list(shape) for shape in weight_shapes],
      "count": count,
  } for (source_shape, weight_shapes), count in sorted(cohorts.items())]
  return locked


def current_profile_audit(worker: dict[str, Any]) -> dict[str, Any]:
  census = worker.get("execution_census", {})
  retained = census.get("retained_rows", [])
  if not isinstance(retained, list):
    raise TypeError("seq2204 retained profile rows are missing")
  labels = ("gate", "up", "down")
  target_rows: list[dict[str, Any]] = []
  counts = Counter()
  for row in retained:
    if not isinstance(row, dict):
      continue
    name = str(row.get("node_name", ""))
    if (row.get("status") != "Status.EXECUTED"
        or row.get("node_type") != "FullyConnectedCompressed"
        or ".shared_expert." not in name):
      continue
    for label in labels:
      if f".{label}_proj/" in name:
        counts[label] += 1
        target_rows.append(row)
        break
  times = [float(row["real_time_us"]) for row in target_rows]
  executed = census.get("executed_type_counts", {})
  return {
      "gated_mlp_count": int(executed.get("GatedMLP", 0)),
      "shared_expert_fc_counts": {
          label: int(counts[label]) for label in labels},
      "shared_expert_fc_profile": {
          "count": len(times),
          "raw_real_time_us_nonadditive": sum(times),
          "median_row_real_time_us": statistics.median(times),
          "minimum_row_real_time_us": min(times),
          "maximum_row_real_time_us": max(times),
          "direct_savings_evidence": False,
      },
      "executed_type_counts": executed,
  }


def strip_cpp_comments(value: str) -> str:
  return re.sub(
      r"//[^\n]*|/\*.*?\*/", "", value,
      flags=re.MULTILINE | re.DOTALL)


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
      STATUS, ROUTES, REJECTED, LOCKED_MODEL_XML, LOCAL_PIPELINE,
      LOCAL_GMLP_LIST, LOCAL_GMLP_REF, OLD_AUDIT_TOOL, SEQ1281, SEQ2202,
      SEQ2204, SEQ2204_WORKER)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing gated-MLP audit inputs: " + ", ".join(missing))

  repo = repository_state(output)
  seq1281 = load_json(SEQ1281)
  seq2202 = load_json(SEQ2202)
  seq2204 = load_json(SEQ2204)
  profile_worker = load_json(SEQ2204_WORKER)
  routes = load_json(ROUTES)
  rejected = load_json(REJECTED)
  old_audit = load_old_audit()
  locked = locked_shape_audit(old_audit)
  current = current_profile_audit(profile_worker)
  sample_memory("after-local-evidence", stop_bytes, memory)

  onednn_payload, onednn_pull = fetch_pull(
      "uxlfoundation", "oneDNN", ONEDNN_PR, raw, args.network_timeout_s)
  openvino_payload, openvino_pull = fetch_pull(
      "openvinotoolkit", "openvino", OPENVINO_PR, raw,
      args.network_timeout_s)
  _, onednn_file_rows = fetch_pull_files(
      "uxlfoundation", "oneDNN", ONEDNN_PR, raw, args.network_timeout_s)
  _, openvino_file_rows = fetch_pull_files(
      "openvinotoolkit", "openvino", OPENVINO_PR, raw,
      args.network_timeout_s)
  onednn_sources, onednn_source_evidence = fetch_sources(
      "uxlfoundation", "oneDNN", ONEDNN_HEAD, ONEDNN_FILES, raw,
      args.network_timeout_s)
  openvino_sources, openvino_source_evidence = fetch_sources(
      "openvinotoolkit", "openvino", OPENVINO_HEAD, OPENVINO_FILES, raw,
      args.network_timeout_s)

  head_repo = str(openvino_pull["head_repo"])
  submodule_payload = fetch_json(
      f"https://api.github.com/repos/{head_repo}/contents/"
      "src/plugins/intel_gpu/thirdparty/onednn_gpu"
      f"?ref={OPENVINO_HEAD}",
      raw / "openvino-pr36139-onednn-submodule.json",
      args.network_timeout_s)
  if not isinstance(submodule_payload, dict):
    raise TypeError("OpenVINO PR oneDNN submodule response is not an object")
  integration_list_bytes = fetch(
      "https://raw.githubusercontent.com/uxlfoundation/oneDNN/"
      f"{OPENVINO_PR_ONEDNN_SHA}/src/gpu/gpu_gated_mlp_list.cpp",
      raw / (
          "onednn-openvino-pr36139-submodule-"
          f"{OPENVINO_PR_ONEDNN_SHA[:12]}-gmlp-list.cpp"),
      args.network_timeout_s, accept="text/plain")
  integration_list = integration_list_bytes.decode(
      "utf-8", errors="replace")
  sample_memory("after-upstream-evidence", stop_bytes, memory)

  local_pipeline = LOCAL_PIPELINE.read_text(encoding="utf-8")
  local_list = LOCAL_GMLP_LIST.read_text(encoding="utf-8")
  micro_list = onednn_sources["impl_list"]
  micro_cpp = onednn_sources["micro_cpp"]
  micro_cl = onednn_sources["micro_cl"]
  tests = onednn_sources["tests"]
  tests_active = strip_cpp_comments(tests)
  tests_active_suite = tests_active.split(
      "INSTANTIATE_TEST_SUITE_P", maxsplit=1)[-1]
  ov_wrapper = openvino_sources["wrapper"]
  ov_dq = openvino_sources["dynamic_quantize"]
  ov_fusion = openvino_sources["fusion"]
  ov_pipeline = openvino_sources["pipeline"]

  micro_index = micro_list.find(
      "GPU_INSTANCE_INTEL(intel::gated_mlp::micro_horz_t)")
  ref_index = micro_list.find(
      "GPU_INSTANCE_INTEL(intel::gated_mlp::ref_t)")
  integration_micro_index = integration_list.find(
      "GPU_INSTANCE_INTEL(intel::gated_mlp::micro_horz_t)")
  integration_ref_index = integration_list.find(
      "GPU_INSTANCE_INTEL(intel::gated_mlp::ref_t)")
  ugemm_calls = len(re.findall(r"\bugemm_wgu\s*\(", micro_cl))
  active_test_rows = len(re.findall(
      r"\bmlp_dims_t\s*\{", tests_active_suite))
  exact_decode_test_rows = len(re.findall(
      r"\bmlp_dims_t\s*\{\s*1\s*,\s*2048\s*,\s*512\s*,\s*64\s*,\s*64",
      tests_active_suite))
  exact_prefill_test_rows = len(re.findall(
      r"\bmlp_dims_t\s*\{\s*2048\s*,\s*2048\s*,\s*512\s*,\s*64\s*,\s*64",
      tests_active_suite))
  active_large_test_rows = len(re.findall(
      r"\bmlp_dims_t\s*\{\s*1024\s*,\s*4096\s*,\s*27392\s*,\s*128"
      r"\s*,\s*128",
      tests_active_suite))

  rank2_elements = int(
      locked["gated_mlp_rank2_intermediate_elements_per_token"])
  current_intermediate_bytes = rank2_elements * 12
  micro_intermediate_bytes = rank2_elements * 4
  maximum_removed_bytes = current_intermediate_bytes - micro_intermediate_bytes
  old_candidate = seq1281["candidates"]["onednn_gated_mlp_inplace"]
  old_rate_gbps = (
      float(old_candidate["saved_intermediate_traffic_bytes_per_token"])
      / (float(old_candidate[
          "saved_intermediate_traffic_ms_at_small_tensor_rate"]) / 1000)
      / 1e9)
  maximum_removed_ms = maximum_removed_bytes / (old_rate_gbps * 1e9) * 1000
  prefill_lcb = float(seq2202["phase_inference"]["prefill_tokens_s"][
      "lower_confidence_bound_ratio"])
  qk_target = float(seq2202["target_ratio"])
  qk_lcb_ratio_gap = qk_target - prefill_lcb

  closed = {
      str(row.get("route")): row
      for row in rejected.get("rejected", [])
      if isinstance(row, dict)}
  prior_route = closed.get("openvino_upstream_gated_mlp_inplace_v29h", {})
  onednn_body = str(onednn_payload.get("body", ""))
  openvino_body = str(openvino_payload.get("body", ""))
  exact_shape_performance_published = False
  complete_positive_bound = False
  product_build_admitted = False
  component_build_admitted = False

  source_contract = {
      "pinned_runtime": {
          "openvino_commit": git(PINNED_OPENVINO, "rev-parse", "HEAD"),
          "onednn_commit": git(PINNED_ONEDNN, "rev-parse", "HEAD"),
          "fusion_default_disabled": (
              "GPU_DEBUG_VALUE_OR(config.get_disable_gated_mlp_fusion(), true)"
              in local_pipeline),
          "onednn_impls": ["ref"],
          "micro_horz_present": (
              (PINNED_ONEDNN / "src/gpu/intel/gated_mlp/micro_horz.cpp"
               ).is_file()),
      },
      "onednn_pr5059": {
          "pull": onednn_pull,
          "files": onednn_file_rows,
          "micro_horz_precedes_ref": (
              0 <= micro_index < ref_index),
          "ptl_xe2_config": [32, 16, 1, 1],
          "gate_up_ugemm_calls": ugemm_calls,
          "nested_down_gemm": (
              "gemm_down_->execute(down_ctx)" in micro_cpp),
          "f16_intermediate_scratch": (
              "key_matmul_src_trans" in micro_cpp
              and "tile_store(S_tile_dst, tmp_reduce_mem" in micro_cl),
          "active_test_rows": active_test_rows,
          "active_large_u4_group128_rows": active_large_test_rows,
          "exact_locked_decode_u4_group64_rows": exact_decode_test_rows,
          "exact_locked_prefill_u4_group64_rows": exact_prefill_test_rows,
          "performance_optimizations_ongoing": (
              "perf optimizations ongoing" in onednn_body),
          "published_exact_locked_shape_performance": (
              exact_shape_performance_published),
      },
      "openvino_pr36139": {
          "pull": openvino_pull,
          "files": openvino_file_rows,
          "body_mentions_support_and_dynamic_quantization": (
              "gated_mlp micro_horz" in openvino_body
              and "dynamic quantization" in openvino_body.lower()),
          "fusion_default_disabled": (
              "GPU_DEBUG_VALUE_OR(config.get_disable_gated_mlp_fusion(), true)"
              in ov_pipeline),
          "gmlp_dynamic_quantization_default_disabled": (
              "config.get_dynamic_quantize_gated_mlp(), false"
              in ov_pipeline),
          "compressed_u4_i4_supported": (
              "ov::element::u4" in ov_fusion
              and "ov::element::i4" in ov_fusion),
          "grouped_weights_flattened_and_scales_transposed": (
              "Flatten to 2D for GatedMLP/oneDNN" in ov_fusion
              and 'create_transpose(info->scale, "scale_transpose")'
              in ov_fusion),
          "wrapper_forces_flat_2d": "#define FORCE_FLAT_2D 0" in ov_wrapper,
          "dynamic_quantize_pass_present": (
              "DynamicQuantizeGatedMLP::DynamicQuantizeGatedMLP" in ov_dq),
          "submodule_commit": submodule_payload.get("sha"),
          "submodule_ref_precedes_micro_horz": (
              0 <= integration_ref_index < integration_micro_index),
          "submodule_contains_pr5059_head": (
              submodule_payload.get("sha") == ONEDNN_HEAD),
      },
      "arithmetic_schedule": {
          "gate_up_microkernel_ugemm_count": ugemm_calls,
          "nested_down_gemm_count": 1,
          "total_gemm_count": ugemm_calls + 1,
          "reduces_three_gemm_arithmetic_schedule": ugemm_calls + 1 < 3,
          "materializes_f16_product_before_down": True,
      },
  }

  checks = [
      check("repository_clean_and_pushed_at_gate",
            not repo["dirty"] and repo["pushed"] and repo["branch"] == "main",
            **repo),
      check("pinned_runtime_identity_exact",
            source_contract["pinned_runtime"]["openvino_commit"]
            == PINNED_OPENVINO_SHA
            and source_contract["pinned_runtime"]["onednn_commit"]
            == PINNED_ONEDNN_SHA),
      check("pinned_runtime_has_ref_only_and_default_off",
            source_contract["pinned_runtime"]["fusion_default_disabled"]
            and not source_contract["pinned_runtime"]["micro_horz_present"]
            and "micro_horz_t" not in local_list
            and "ref_t" in local_list),
      check("locked_rank2_shared_expert_contract_exact",
            locked["gated_mlp_structural_match_count"] == 80
            and locked["gated_mlp_rank2_match_count"] == 40
            and locked["gated_mlp_grouped_rank3_match_count"] == 40
            and locked["rank2_shape_cohorts"] == [{
                "source_shape": [-1, 2048],
                "weight_shapes": [
                    [512, 2048], [512, 2048], [2048, 512]],
                "count": 40,
            }],
            shape_cohorts=locked["rank2_shape_cohorts"]),
      check("accepted_carrier_runtime_owner_exact",
            current["gated_mlp_count"] == 0
            and current["shared_expert_fc_counts"]
            == {"gate": 40, "up": 40, "down": 40}
            and current["shared_expert_fc_profile"]["count"] == 120
            and current["shared_expert_fc_profile"][
                "raw_real_time_us_nonadditive"] == 1554.0),
      check("upstream_heads_are_pinned_open_prs",
            onednn_pull["head_sha"] == ONEDNN_HEAD
            and onednn_pull["state"] == "open"
            and openvino_pull["head_sha"] == OPENVINO_HEAD
            and openvino_pull["state"] == "open"),
      check("pr5059_selects_micro_horz_first_on_ptl",
            source_contract["onednn_pr5059"]["micro_horz_precedes_ref"]
            and "{32, 16, 1, 1}" in micro_cpp
            and "gpu_arch_t::xe_hpg" in micro_cpp),
      check("pr5059_retains_three_gemm_and_f16_scratch_schedule",
            ugemm_calls == 2
            and source_contract["onednn_pr5059"]["nested_down_gemm"]
            and source_contract["onednn_pr5059"]["f16_intermediate_scratch"]
            and not source_contract["arithmetic_schedule"][
                "reduces_three_gemm_arithmetic_schedule"]),
      check("pr5059_has_no_active_exact_locked_shape_row",
            active_test_rows == 1
            and active_large_test_rows == 1
            and exact_decode_test_rows == 0
            and exact_prefill_test_rows == 0),
      check("pr5059_declares_performance_work_incomplete",
            source_contract["onednn_pr5059"][
                "performance_optimizations_ongoing"]
            and not exact_shape_performance_published),
      check("pr36139_supports_locked_compression_but_defaults_off",
            source_contract["openvino_pr36139"][
                "compressed_u4_i4_supported"]
            and source_contract["openvino_pr36139"][
                "grouped_weights_flattened_and_scales_transposed"]
            and source_contract["openvino_pr36139"][
                "fusion_default_disabled"]
            and source_contract["openvino_pr36139"][
                "gmlp_dynamic_quantization_default_disabled"]),
      check("pr36139_does_not_select_pr5059_micro_horz",
            source_contract["openvino_pr36139"]["submodule_commit"]
            == OPENVINO_PR_ONEDNN_SHA
            and source_contract["openvino_pr36139"][
                "submodule_ref_precedes_micro_horz"]
            and not source_contract["openvino_pr36139"][
                "submodule_contains_pr5059_head"]),
      check("prior_three_gemm_route_rejection_is_registered",
            prior_route.get("class")
            == "three_gemm_scratch_alias_below_residual_aggregate_cut"
            and "reduces the three-GEMM arithmetic schedule"
            in str(prior_route.get("reopen_condition", ""))),
      check("source_ceiling_is_not_positive_saving_evidence",
            maximum_removed_bytes == 163_840
            and maximum_removed_ms > 0.0
            and not complete_positive_bound),
      check("no_build_or_gpu_work_is_admitted",
            not product_build_admitted and not component_build_admitted),
  ]
  required_checks_passed = all(bool(row["pass"]) for row in checks)

  verdict = {
      "required_checks_passed": required_checks_passed,
      "verdict": (
          "reject_current_micro_horz_build_watch_for_locked_shape_"
          "performance_successor"),
      "product_build_admitted": product_build_admitted,
      "component_build_admitted": component_build_admitted,
      "watch_route_registered": True,
      "reason": (
          "PR 5059 is structurally applicable to all 40 rank-2 shared-expert "
          "chains, but its selected PTL body still executes two gate/up "
          "ugemms plus one nested down GEMM and materializes an f16 product. "
          "Its only active test is a different U4 group-128 large shape, the "
          "author states that performance work is ongoing, and PR 36139 "
          "still points to an older oneDNN revision that ranks ref before "
          "micro_horz. Combining two open branches without an exact locked-"
          "shape positive component bound is not admitted."),
      "reopen_trigger": (
          "a PR5059 performance successor or an exact PTL U4 group64 "
          "MB=1/2048, IC=2048, OC=512 component result with a paired positive "
          "saving bound, followed by an OpenVINO integration revision that "
          "actually selects that oneDNN body"),
      "qk_bundle_note": (
          "The independent Q/K route misses its prefill LCB target by only "
          f"{qk_lcb_ratio_gap:.12f}x, so a future positive GatedMLP component "
          "could fund a preregistered bundle; the current source ceiling is "
          "not such positive evidence."),
      "compiler_invocations": 0,
      "gpu_contexts_created": 0,
      "model_workers_started": 0,
      "infer_requests_created": 0,
  }

  sample_memory("complete", stop_bytes, memory)
  created_at = datetime.now(timezone.utc).isoformat()
  raw_files = sorted(path for path in raw.iterdir() if path.is_file())
  metrics = {
      "schema": SCHEMA,
      "created_at": created_at,
      "workstream": WS,
      "git": repo,
      "inputs": {
          display(path): {
              "sha256": sha256(path),
              "bytes": path.stat().st_size,
          } for path in required
      },
      "upstream_raw": {
          display(path): {
              "sha256": sha256(path),
              "bytes": path.stat().st_size,
          } for path in raw_files
      },
      "upstream_sources": {
          "onednn": onednn_source_evidence,
          "openvino": openvino_source_evidence,
      },
      "locked_graph": locked,
      "accepted_carrier_profile": current,
      "source_contract": source_contract,
      "bound": {
          "current_materialized_intermediate_bytes_per_token": (
              current_intermediate_bytes),
          "micro_horz_intermediate_bytes_per_token": micro_intermediate_bytes,
          "maximum_removed_intermediate_bytes_per_token": maximum_removed_bytes,
          "registered_small_tensor_gbps": old_rate_gbps,
          "maximum_removed_traffic_ms_per_token": maximum_removed_ms,
          "profile_target_fc_raw_us_nonadditive": current[
              "shared_expert_fc_profile"]["raw_real_time_us_nonadditive"],
          "profile_time_is_direct_savings_evidence": False,
          "complete_positive_source_bound": complete_positive_bound,
          "qk_prefill_lcb_ratio": prefill_lcb,
          "qk_prefill_target_ratio": qk_target,
          "qk_prefill_lcb_ratio_gap": qk_lcb_ratio_gap,
      },
      "checks": checks,
      "verdict": verdict,
      "memory": {
          "stop_bytes": stop_bytes,
          "samples": memory,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory),
      },
      "workers": {
          "compiler_invocations": 0,
          "gpu_contexts_created": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
          "maximum_concurrent_workers": 0,
      },
  }
  write_json(output / "metrics.json", metrics)
  manifest = {
      "schema": SCHEMA,
      "created_at": created_at,
      "git": repo,
      "required_checks_passed": required_checks_passed,
      "verdict": verdict["verdict"],
      "metrics": "metrics.json",
      "report": "report.md",
      "raw_files": [path.name for path in raw_files],
  }
  write_json(output / "manifest.json", manifest)

  report = f"""# Upstream gated-MLP micro_horz source bound

- result: `{'PASS' if required_checks_passed else 'FAIL'}`
- verdict: `{verdict['verdict']}`
- locked matches: rank-2 shared expert `40`; grouped rank-3 routed `40`
- accepted runtime owner: GatedMLP `0`; shared gate/up/down FC `40/40/40`
- shared gate/up/down profile: `120` rows, raw non-additive `1554 us`
- PR 5059 arithmetic: `2` gate/up ugemms + `1` nested down GEMM
- PR 5059 intermediate: f16 product scratch remains materialized
- active upstream tests: `{active_test_rows}`; exact locked U4/group64 rows: `0`
- maximum removed intermediate traffic: `{maximum_removed_bytes} B/token`,
  `{maximum_removed_ms:.9f} ms/token` at the registered small-tensor rate
- Q/K prefill LCB gap: `{qk_lcb_ratio_gap:.12f}x`
- PR 36139 oneDNN pointer: `{OPENVINO_PR_ONEDNN_SHA[:12]}`; ref precedes
  micro_horz and PR 5059 is not selected
- product/component build admitted: `false/false`
- compiler/GPU/model/InferRequest: `0/0/0/0`
- minimum available memory: `{metrics['memory']['minimum_available_bytes']} B`

The upstream route is real enough to watch but not ready to spend a build.
It does not satisfy the registered three-GEMM reopen condition, publishes no
exact locked-shape performance result, and requires combining two open branches
whose integration pointer still selects the reference implementation first.
Reopen on a performance successor or an exact paired PTL U4 group64 component
result for MB=1/2048, IC=2048, OC=512.
"""
  (output / "report.md").write_text(report, encoding="utf-8")

  print(json.dumps({
      "schema": SCHEMA,
      "required_checks_passed": required_checks_passed,
      "verdict": verdict["verdict"],
      "product_build_admitted": product_build_admitted,
      "component_build_admitted": component_build_admitted,
      "output": display(output),
      "minimum_available_bytes": metrics["memory"][
          "minimum_available_bytes"],
  }, sort_keys=True))
  return 0 if required_checks_passed and not product_build_admitted else 1


if __name__ == "__main__":
  raise SystemExit(main())
