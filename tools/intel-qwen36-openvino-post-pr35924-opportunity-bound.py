#!/usr/bin/env python3
"""Close current upstream candidates and select one hardware-limit route.

This is a source-only routing gate.  It refreshes five official pull requests,
intersects them with the accepted PTL/Xe3 U4 dispatch and locked IR, and then
compares the remaining FC/DQ ceilings with the measured LM-head exact-fallback
traffic floor.  It starts no compiler, GPU context, model compile, InferRequest,
or inference worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
SCHEMA = "intel-qwen36-openvino-post-pr35924-opportunity-bound-v0"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
DQ_BOUND = ROOT / (
    "output/openvino-upstream-dq-permute-dispatch-bound-"
    "20260731Tseq2222-clean/metrics.json")
TARGET_BOUND = ROOT / (
    "output/onednn-gmlp-pr5681-strategy-bound-"
    "20260731Tseq2226-clean/metrics.json")
FC_BOUND = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/metrics.json")
PROFILE = ROOT / (
    "output/openvino-current-bundle-profile-refresh-"
    "20260731Tseq2204-short-o130-clean/metrics.json")
LM_FALLBACK_BOUND = ROOT / (
    "output/openvino-lm-head-gated-exact-fallback-bound-"
    "20260731Tseq2186-clean/result.json")
LM_FALLBACK_COMPONENT = ROOT / (
    "output/openvino-lm-head-gated-exact-component-"
    "20260731Tseq2187-clean/result.json")
LM_TOPK_COMPONENT = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-component-"
    "20260731Tseq2188-clean/result.json")
PROVIDER_TRACE = ROOT / (
    "output/openvino-hot-cold-product-"
    "20260715Tseq1212-onednn-gemm-selection-trace-2k-o4-dirtyZ/raw/"
    "sentinel_002k/correctness/candidate/worker.stdout")
LOCKED_MODEL_XML = Path(
    "/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")

EXPECTED_INPUT_SHA256 = {
    DQ_BOUND:
        "275ecd2ce685d2b55f20954199ba1cf0618dca5c43341f3c24521647cc329cf8",
    TARGET_BOUND:
        "7038b6638a49010f8f430cb225a9a3c5b1bca7254c144214107c1e42a36b589a",
    FC_BOUND:
        "e0e93bd9f18fab2ef89acee0b947aa78f8f3eb6b87173f5b0f0866309db012d4",
    PROFILE:
        "58eab16e483a06248e818a4aeaa0adeea666b38aa7b355eab6cc345a1e782ae9",
    LM_FALLBACK_BOUND:
        "6bed3a9f24917433d51559c62d3dec222abaa9654c16ad1443b96b98b0936be7",
    LM_FALLBACK_COMPONENT:
        "caf1814a1786f74e637b5aa398455bac64a831d4dd5fa22557a7def0919d9a73",
    LM_TOPK_COMPONENT:
        "8d5cda1698fbfca2b814dc4e263671660705880e1ae182a3a3026e0c84737102",
    PROVIDER_TRACE:
        "84cd23bc0fce9cf2c90131b47c34217df6f5b3c39f85bc0f5a2f6633c1c1d14b",
    LOCKED_MODEL_XML:
        "fae1047f6a758ded4fab95f5faee9bf68f92b4433d778496bd9d44efa51cdbb0",
}

PULLS = {
    "onednn_gemm_fixes": {
        "owner": "uxlfoundation",
        "repo": "oneDNN",
        "number": 5634,
        "title": "[GPU] Gemm fixes",
        "head": "b2ad2b363075057430e1d87d75ddf4ab0601785b",
    },
    "onednn_xe3p_ohs": {
        "owner": "uxlfoundation",
        "repo": "oneDNN",
        "number": 5713,
        "title": "[GPU] gemm: add Xe3p OHS TNN strategies, n<=128",
        "head": "8f5d81794fc263f3d3ac57dbe17bbb528f226f0b",
    },
    "onednn_strategy_suppression": {
        "owner": "uxlfoundation",
        "repo": "oneDNN",
        "number": 5681,
        "title": "xe: gemm: jit: generator: prevent unexpected strategy supression",
        "head": "d68d4194763918b85e8f72181f3aede151396aa6",
    },
    "openvino_dq_kernel_sharing": {
        "owner": "openvinotoolkit",
        "repo": "openvino",
        "number": 37140,
        "title": "[GPU] Fix dynamic_quantize kernel sharing between different IFM sizes",
        "head": "f02f16c9b0b7b966c3e0e91876db255197951d7a",
    },
    "openvino_mamba2_fusion": {
        "owner": "openvinotoolkit",
        "repo": "openvino",
        "number": 36412,
        "title": "[POC] [DO NOT REVIEW] Mamba2 Fusion",
        "head": "0708a91bb31e8131632592d8c7715e4caa10f1a9",
    },
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


def display(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


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
  output_rel = display(output)
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
          "User-Agent": "intel-qwen36-post-pr35924-opportunity-bound",
      })
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    value = response.read()
  destination.write_bytes(value)
  return value


def fetch_pull(
    name: str, spec: dict[str, Any], raw: Path, timeout_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, list[Path]]:
  owner = str(spec["owner"])
  repo = str(spec["repo"])
  number = int(spec["number"])
  stem = f"{owner}-{repo}-pr{number}"
  metadata_path = raw / f"{stem}.json"
  files_path = raw / f"{stem}-files.json"
  patch_path = raw / f"{stem}.patch"
  base = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
  metadata_bytes = fetch(base, metadata_path, timeout_s)
  files_bytes = fetch(base + "/files?per_page=100", files_path, timeout_s)
  patch_bytes = fetch(
      f"https://github.com/{owner}/{repo}/pull/{number}.patch",
      patch_path, timeout_s, accept="application/vnd.github.patch")
  metadata = json.loads(metadata_bytes)
  files = json.loads(files_bytes)
  if not isinstance(metadata, dict) or metadata.get("number") != number:
    raise TypeError(f"unexpected metadata for {name}")
  if not isinstance(files, list):
    raise TypeError(f"unexpected file list for {name}")
  return (
      metadata,
      [row for row in files if isinstance(row, dict)],
      patch_bytes.decode("utf-8", errors="replace"),
      [metadata_path, files_path, patch_path],
  )


def pull_summary(
    metadata: dict[str, Any], files: list[dict[str, Any]], patch: str,
) -> dict[str, Any]:
  return {
      "number": metadata.get("number"),
      "title": metadata.get("title"),
      "html_url": metadata.get("html_url"),
      "state": metadata.get("state"),
      "draft": metadata.get("draft"),
      "created_at": metadata.get("created_at"),
      "updated_at": metadata.get("updated_at"),
      "head_sha": metadata.get("head", {}).get("sha"),
      "base_sha": metadata.get("base", {}).get("sha"),
      "commits": metadata.get("commits"),
      "additions": metadata.get("additions"),
      "deletions": metadata.get("deletions"),
      "changed_files": metadata.get("changed_files"),
      "body_sha256": sha256_bytes(
          str(metadata.get("body", "")).encode("utf-8")),
      "patch_sha256": sha256_bytes(patch.encode("utf-8")),
      "patch_bytes": len(patch.encode("utf-8")),
      "files": [{
          "filename": row.get("filename"),
          "status": row.get("status"),
          "additions": row.get("additions"),
          "deletions": row.get("deletions"),
      } for row in files],
  }


def added_catalog_lines(patch: str) -> list[str]:
  return [
      line[1:] for line in patch.splitlines()
      if line.startswith("+{{") and not line.startswith("+++")
  ]


def provider_census(text: str) -> dict[str, Any]:
  expression = re.compile(
      r"selection:m:(\d+),n:(\d+),k:(\d+),rank:0,entry:(.*)$")
  rows: dict[tuple[int, int, int, str], dict[str, Any]] = {}
  for line in text.splitlines():
    match = expression.search(line)
    if not match:
      continue
    m, n, k = (int(match.group(index)) for index in (1, 2, 3))
    entry = match.group(4)
    rows[(m, n, k, entry)] = {
        "m": m, "n": n, "k": k, "entry": entry,
    }
  values = list(rows.values())
  decode = [row for row in values
            if row["n"] == 1 and "G gemm FHS" in row["entry"]]
  prefill = [row for row in values
             if row["n"] > 1 and "[FO]OS" in row["entry"]]
  return {
      "unique_rank0_rows": len(values),
      "decode_fhs_rows": decode,
      "decode_k_values": sorted({row["k"] for row in decode}),
      "decode_m_values": sorted({row["m"] for row in decode}),
      "decode_k_remainder_mod128_count":
          sum(row["k"] % 128 != 0 for row in decode),
      "decode_non_kchain1_marker_count":
          sum(bool(re.search(r"\bkc(?:2|4|8|16)\b", row["entry"]))
              for row in decode),
      "prefill_foos_row_count": len(prefill),
  }


def locked_loop_census(path: Path) -> dict[str, Any]:
  root = ET.parse(path).getroot()
  rows = []
  for layer in root.iter("layer"):
    if layer.get("type") != "Loop":
      continue
    input_node = layer.find("./input")
    output_node = layer.find("./output")
    body_layers = layer.findall("./body/layers/layer")
    body_types = Counter(
        str(body_layer.get("type")) for body_layer in body_layers)
    rows.append({
        "id": layer.get("id"),
        "name": layer.get("name"),
        "input_count": (
            len(input_node.findall("./port")) if input_node is not None else 0),
        "output_count": (
            len(output_node.findall("./port"))
            if output_node is not None else 0),
        "body_type_counts": dict(sorted(body_types.items())),
    })
  return {
      "loop_count": len(rows),
      "input_count_histogram": dict(sorted(Counter(
          row["input_count"] for row in rows).items())),
      "output_count_histogram": dict(sorted(Counter(
          row["output_count"] for row in rows).items())),
      "mamba2_matcher_input_count": 8,
      "arity_match_count": sum(row["input_count"] == 8 for row in rows),
      "rows": rows,
  }


def profile_census(profile: dict[str, Any]) -> dict[str, Any]:
  rows = profile["profile_rollup"]["ranked_retained_node_types_nonadditive"]
  wanted = {
      "DynamicQuantize",
      "FullyConnectedCompressed",
      "IQ36ExactPhaseDualCohortHotAttentionGQA",
      "IQ36LinearConvSwish",
      "MOE3GemmFusedCompressed",
  }
  return {
      str(row["node_type"]): {
          "executed_count": int(row["executed_count"]),
          "raw_real_time_us_nonadditive":
              float(row["raw_real_time_us_nonadditive"]),
      }
      for row in rows if row.get("node_type") in wanted
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_samples: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory_samples)

  required_inputs = (
      STATUS, ROUTES, REJECTED, *EXPECTED_INPUT_SHA256.keys())
  missing = [display(path) for path in required_inputs if not path.is_file()]
  if missing:
    raise SystemExit("missing opportunity-bound inputs: " + ", ".join(missing))
  input_hashes = {path: sha256(path) for path in required_inputs}
  exact_input_hashes = all(
      input_hashes[path] == expected
      for path, expected in EXPECTED_INPUT_SHA256.items())

  git_state = repository_state(output)
  dq = load_json(DQ_BOUND)
  target = load_json(TARGET_BOUND)
  fc = load_json(FC_BOUND)
  profile = load_json(PROFILE)
  fallback = load_json(LM_FALLBACK_BOUND)
  fallback_component = load_json(LM_FALLBACK_COMPONENT)
  topk_component = load_json(LM_TOPK_COMPONENT)
  provider = provider_census(PROVIDER_TRACE.read_text(encoding="utf-8"))
  loops = locked_loop_census(LOCKED_MODEL_XML)
  retained_profile = profile_census(profile)

  fetched: dict[str, dict[str, Any]] = {}
  private: dict[str, dict[str, Any]] = {}
  snapshots: list[Path] = []
  for name, spec in PULLS.items():
    metadata, files, patch, paths = fetch_pull(
        name, spec, raw, args.network_timeout_s)
    fetched[name] = pull_summary(metadata, files, patch)
    private[name] = {
        "body": str(metadata.get("body", "")),
        "patch": patch,
        "added_catalog": added_catalog_lines(patch),
    }
    snapshots.extend(paths)
    sample_memory(f"fetched_{name}", stop_bytes, memory_samples)

  pr5634 = private["onednn_gemm_fixes"]
  pr5713 = private["onednn_xe3p_ohs"]
  pr5681 = private["onednn_strategy_suppression"]
  pr37140 = private["openvino_dq_kernel_sharing"]
  pr36412 = private["openvino_mamba2_fusion"]

  dq_runtime = dq["accepted_runtime_graph"]["dynamic_quantize"]
  group64_count = sum(
      int(row["count"]) for row in dq_runtime["group_size_cohorts"]
      if 64 in row["group_sizes"])
  dq_patch_files = {
      str(row["filename"])
      for row in fetched["openvino_dq_kernel_sharing"]["files"]
  }
  mamba_files = {
      str(row["filename"])
      for row in fetched["openvino_mamba2_fusion"]["files"]
  }

  pr5634_exact_catalog_rows = [
      row for row in pr5634["added_catalog"]
      if '{"F", "H", "S"}' in row and '"FHS"' in row
  ]
  pr5713_non_ohs_rows = [
      row for row in pr5713["added_catalog"]
      if '{"O", "H", "S"}' not in row
  ]
  pr5681_selected_strategy = (
      "at32 am128 aB wg 2x1x8 ikr xaf st vav hi pt sr br "
      "sb128 bk0 bm0 nmk sys")
  official_heads_exact = all(
      fetched[name]["number"] == spec["number"]
      and fetched[name]["title"] == spec["title"]
      and fetched[name]["head_sha"] == spec["head"]
      and fetched[name]["state"] == "open"
      and fetched[name]["patch_bytes"] > 0
      for name, spec in PULLS.items())

  fc_aggregate = fc["aggregate"]
  fallback_rows = fallback["worker_rows"]
  slow_count = min(int(row["mode"]["slow_count"]) for row in fallback_rows)
  interval_count = min(
      int(row["interval_count_after_skip"]) for row in fallback_rows)
  fallback_rate = slow_count / interval_count
  slow_increment_ms = float(
      fallback["traffic_bound"]["minimum_observed_slow_increment_ms"])
  gross_mean_avoidable_ms = fallback_rate * slow_increment_ms
  required_saving_ms = float(fallback["required_saving_ms"])
  p95_slow_budget = math.floor(0.05 * interval_count)
  certificates_needed_for_p95 = slow_count - p95_slow_budget
  certificate_fraction_of_slow = certificates_needed_for_p95 / slow_count
  full_scan_bytes = int(
      fallback["traffic_bound"]["mandatory_matvec_bytes"])
  traffic_floor_ms = float(fallback["traffic_bound"]["traffic_floor_ms"])
  repeat_gbps = float(
      fallback_component["repeat_audit"]["stage_profile"]["matvec"][
          "median_gb_s"])
  confirm_gbps = float(
      fallback_component["confirm_audit"]["stage_profile"]["matvec"][
          "median_gb_s"])
  topk_lcb_us = min(
      float(topk_component["repeat_audit"]["performance_inference"][
          "lower_confidence_bound_saving_us"]),
      float(topk_component["confirm_audit"]["performance_inference"][
          "lower_confidence_bound_saving_us"]))

  upstream_analysis = {
      "onednn_pr5634": {
          "added_catalog_row_count": len(pr5634["added_catalog"]),
          "accepted_fhs_u4_added_catalog_row_count":
              len(pr5634_exact_catalog_rows),
          "product_decode_k_values": provider["decode_k_values"],
          "product_decode_k_remainder_mod128_count":
              provider["decode_k_remainder_mod128_count"],
          "patch_has_int4_remainder_fix":
              "sub-byte increments" in pr5634["body"]
              and "Ai_remIncrCopy" in pr5634["patch"],
          "patch_has_early_dequant_remainder_fix":
              "earlyDequantizeA()" in pr5634["patch"]
              and "earlyDequantizeB()" in pr5634["patch"],
          "patch_has_xe3p_only_atomic_fix":
              "hw == HW::Xe3p" in pr5634["patch"],
          "performance_route_admitted": False,
          "reason": (
              "all selected decode K values are multiples of the observed "
              "k128 provider tile; catalog edits add no accepted F/H/S FHS "
              "row, while the remaining changes are remainder correctness or "
              "Xe3p/NVL-P resource fixes"),
      },
      "onednn_pr5713": {
          "added_catalog_row_count": len(pr5713["added_catalog"]),
          "non_ohs_added_catalog_row_count": len(pr5713_non_ohs_rows),
          "target_is_ptl_generic_xe3":
              target["architecture_dispatch"]["ptl_maps_to_generic_xe3"]
              and target["architecture_dispatch"][
                  "generic_xe3_maps_to_core_xe3"],
          "patch_is_xe3p_nvlp":
              "Xe3p OHS TNN" in pr5713["patch"]
              and "NVLP tuning" in pr5713["body"],
          "performance_route_admitted": False,
          "reason": (
              "the patch is an OHS Xe3p/NVL-P catalog addition; the locked "
              "target is PTL/Core Xe3 and the accepted U4 provider is FHS"),
      },
      "onednn_pr5681": {
          "added_catalog_row_count": len(pr5681["added_catalog"]),
          "accepted_rank0_strategy_present_in_patch":
              pr5681_selected_strategy in pr5681["patch"],
          "accepted_decode_non_kchain1_marker_count":
              provider["decode_non_kchain1_marker_count"],
          "prior_architecture_control_intersection":
              target["pr5681_intersection"][
                  "architecture_control_intersection"],
          "prior_changes_architecture_control":
              target["pr5681_intersection"]["changes_architecture_control"],
          "performance_route_admitted": False,
          "reason": (
              "the accepted rank-0 generic FHS strategy is absent from the "
              "changed catalog and uses no kChain>1 marker; the prior exact "
              "audit also found no architecture-control intersection. The PR "
              "publishes no complete locked-shape positive bound"),
      },
      "openvino_pr37140": {
          "locked_dynamic_quantize_count": dq_runtime["count"],
          "locked_group64_count": group64_count,
          "locked_primitive_counts": dq_runtime["primitive_counts"],
          "changed_opencl_kernel_body_file_count": sum(
              "/cl_kernels/" in path for path in dq_patch_files),
          "cache_identity_fields_present":
              "hash_combine(seed, innermost_size)" in pr37140["patch"]
              and "innermost_size == rhs_casted.innermost_size"
                  in pr37140["patch"],
          "jit_value_source_relocated":
              "params.fc_ifm_size = primitive->innermost_size"
                  in pr37140["patch"],
          "performance_route_admitted": False,
          "correctness_watch": True,
          "reason": (
              "the change makes resolved IFM part of primitive/cache identity "
              "and relocates the same JIT value; it changes no OpenCL kernel "
              "body. The accepted graph already executes all 161 DQs "
              "correctly, so this is a correctness watch, not a steady cut"),
      },
      "openvino_pr36412": {
          "locked_loop_count": loops["loop_count"],
          "locked_loop_input_count_histogram":
              loops["input_count_histogram"],
          "matcher_required_input_count": loops[
              "mamba2_matcher_input_count"],
          "locked_arity_match_count": loops["arity_match_count"],
          "changed_gpu_plugin_file_count": sum(
              "/intel_gpu/" in path for path in mamba_files),
          "changed_cpu_pipeline_file_count": sum(
              path.endswith(
                  "src/plugins/intel_cpu/src/transformations/"
                  "transformation_pipeline.cpp")
              for path in mamba_files),
          "matcher_exact_arity_literal_present":
              "loop->get_input_size() != 8" in pr36412["patch"],
          "performance_route_admitted": False,
          "reason": (
              "the locked IR has 30 nine-input Gated-Delta-style Loops, while "
              "the Mamba2 matcher requires eight inputs. The PR registers only "
              "the CPU transformation pipeline and adds no GPU implementation"),
      },
  }

  checks = [
      check("repository_clean_and_pushed_at_gate",
            not git_state["dirty"] and git_state["pushed"],
            git=git_state),
      check("registered_inputs_match_exact_hashes",
            exact_input_hashes,
            mismatches={
                display(path): {
                    "expected": expected,
                    "actual": input_hashes[path],
                }
                for path, expected in EXPECTED_INPUT_SHA256.items()
                if input_hashes[path] != expected
            }),
      check("official_pull_heads_are_exact_and_open",
            official_heads_exact),
      check("accepted_target_and_dispatch_evidence_is_closed",
            dq["verdict"]["required_checks_passed"] is True
            and target["verdict"]["required_checks_passed"] is True
            and provider["decode_k_values"] == [512, 2048, 4096]
            and provider["decode_fhs_rows"]
            and provider["prefill_foos_row_count"] > 0),
      check("pr5634_has_no_accepted_steady_speed_intersection",
            provider["decode_k_remainder_mod128_count"] == 0
            and len(pr5634_exact_catalog_rows) == 0
            and upstream_analysis["onednn_pr5634"][
                "patch_has_xe3p_only_atomic_fix"]),
      check("pr5713_is_xe3p_ohs_not_ptl_xe3_fhs",
            not pr5713_non_ohs_rows
            and upstream_analysis["onednn_pr5713"]["patch_is_xe3p_nvlp"]
            and upstream_analysis["onednn_pr5713"][
                "target_is_ptl_generic_xe3"]),
      check("pr5681_has_no_complete_selected_provider_bound",
            pr5681_selected_strategy not in pr5681["patch"]
            and provider["decode_non_kchain1_marker_count"] == 0
            and not target["pr5681_intersection"][
                "architecture_control_intersection"]
            and target["pr5681_intersection"][
                "changes_architecture_control"] is False),
      check("pr37140_is_cache_correctness_not_kernel_body_speed",
            upstream_analysis["openvino_pr37140"][
                "changed_opencl_kernel_body_file_count"] == 0
            and upstream_analysis["openvino_pr37140"][
                "cache_identity_fields_present"]
            and upstream_analysis["openvino_pr37140"][
                "jit_value_source_relocated"]
            and dq_runtime["count"] == 161
            and group64_count == 160),
      check("pr36412_has_zero_locked_match_and_zero_gpu_body",
            loops["loop_count"] == 30
            and loops["arity_match_count"] == 0
            and loops["input_count_histogram"] == {9: 30}
            and upstream_analysis["openvino_pr36412"][
                "changed_gpu_plugin_file_count"] == 0
            and upstream_analysis["openvino_pr36412"][
                "changed_cpu_pipeline_file_count"] == 1),
      check("fc_dq_profile_is_not_direct_saving_and_fc_ceiling_misses",
            profile["profile_rollup"][
                "profile_time_is_direct_savings_evidence"] is False
            and float(fc_aggregate["optimistic_saving_ms"])
                < float(fc_aggregate["kill_number_ms"])),
      check("full_i8_fallback_is_at_hardware_traffic_limit",
            fallback["required_checks_passed"] is True
            and fallback_component["required_checks_passed"] is True
            and full_scan_bytes == 509_552_640
            and min(repeat_gbps, confirm_gbps)
                > float(fallback["traffic_bound"]["bandwidth_lcb_gb_s"])
            and topk_component["required_checks_passed"] is True
            and topk_lcb_us > required_saving_ms * 1000.0),
      check("certificate_route_has_floor_sized_avoidable_work",
            slow_count == 50
            and interval_count == 495
            and gross_mean_avoidable_ms > required_saving_ms
            and certificates_needed_for_p95 > 0
            and certificates_needed_for_p95 <= slow_count),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compiler_invocations=0, gpu_contexts=0, model_compiles=0,
            infer_requests=0, model_workers=0, product_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "reject_current_upstream_select_lm_head_exact_token_certificate_bound"
      if required_checks_passed else "inconclusive")
  sample_memory("end", stop_bytes, memory_samples)

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git_state,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "speedup_claims_allowed": False,
      "source_edit_admitted": False,
      "compiler_build_admitted": False,
      "plugin_build_admitted": False,
      "gpu_component_admitted": False,
      "model_worker_admitted": False,
      "official_pulls": fetched,
      "accepted_dispatch": provider,
      "locked_loops": loops,
      "retained_profile": retained_profile,
      "upstream_analysis": upstream_analysis,
      "route_comparison": {
          "fc": {
              "non_lm_fc_bytes": int(fc_aggregate["non_lm_fc_bytes"]),
              "stock_ms": float(fc_aggregate["stock_ms"]),
              "target_ms": float(fc_aggregate["target_ms"]),
              "kill_number_ms": float(fc_aggregate["kill_number_ms"]),
              "optimistic_fixed_schedule_ms":
                  float(fc_aggregate["dominant_ms"]),
              "optimistic_saving_ms":
                  float(fc_aggregate["optimistic_saving_ms"]),
              "shortfall_ms": (
                  float(fc_aggregate["kill_number_ms"])
                  - float(fc_aggregate["optimistic_saving_ms"])),
              "status": "closed_fixed_strategy_family_misses_even_optimistically",
          },
          "dynamic_quantize": {
              "executed_count": dq_runtime["count"],
              "group64_count": group64_count,
              "profile_raw_us_nonadditive":
                  retained_profile["DynamicQuantize"][
                      "raw_real_time_us_nonadditive"],
              "profile_is_direct_saving_evidence": False,
              "status": "no_new_kernel_body_or_complete_positive_bound",
          },
          "lm_head_exact_fallback": {
              "observed_slow_intervals": slow_count,
              "observed_intervals": interval_count,
              "observed_fallback_rate": fallback_rate,
              "minimum_slow_increment_ms": slow_increment_ms,
              "gross_mean_avoidable_ms": gross_mean_avoidable_ms,
              "required_product_saving_ms": required_saving_ms,
              "gross_multiple_over_required":
                  gross_mean_avoidable_ms / required_saving_ms,
              "full_scan_bytes": full_scan_bytes,
              "traffic_floor_ms": traffic_floor_ms,
              "repeat_matvec_median_gb_s": repeat_gbps,
              "confirm_matvec_median_gb_s": confirm_gbps,
              "parallel_topk_saving_lcb_us": topk_lcb_us,
              "p95_slow_event_budget": p95_slow_budget,
              "minimum_slow_events_to_certify_for_p95":
                  certificates_needed_for_p95,
              "minimum_certificate_fraction_of_slow_events":
                  certificate_fraction_of_slow,
              "status": "selected_for_offline_certificate_bound",
              "interpretation": (
                  "the full I8 fallback is already a bandwidth-limit scan; "
                  "the next movable boundary is proving the exact token "
                  "without reading all 509.553 MB, with full scan retained as "
                  "a correctness fallback when certification fails"),
          },
      },
      "selected_direction": {
          "rank": 1,
          "route": "lm_head_exact_token_certificate_bound",
          "admission": "offline_captured_hidden_only",
          "next_gate": (
              "derive a conservative per-row or per-block Q1 residual bound; "
              "on captured real hiddens require exact top1 for every certified "
              "row, zero false certificates, and enough coverage to justify "
              "a 2k slow-event capture before any plugin or product worker"),
          "forbidden_claim": (
              "captured empirical top1 agreement alone is not a certificate "
              "and does not authorize a speedup claim"),
      },
      "parked_directions": [
          {
              "rank": 2,
              "route": "dynamic_quantize_local_kernel_roofline",
              "trigger": (
                  "a source-derived executed group64 body with a complete "
                  "positive component bound, not raw profile attribution"),
          },
          {
              "rank": 3,
              "route": "fc_new_algorithm_not_fixed_strategy_neighbor",
              "trigger": (
                  "a new dataflow or compression algorithm whose complete "
                  "bound beats the 0.68464-ms optimistic fixed-schedule "
                  "shortfall; do not reopen neighboring catalog tiles"),
          },
      ],
      "checks": checks,
      "workers": {
          "compiler_invocations": 0,
          "gpu_contexts": 0,
          "model_compiles": 0,
          "infer_requests": 0,
          "model_workers": 0,
          "product_workers": 0,
          "workers_concurrent": False,
          "oom_observed": False,
      },
      "memory": {
          "stop_bytes": stop_bytes,
          "samples": memory_samples,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory_samples),
      },
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "git": git_state,
      "inputs": {
          display(path): {
              "bytes": path.stat().st_size,
              "sha256": input_hashes[path],
          }
          for path in required_inputs
      },
      "official_snapshots": {
          display(path): {
              "bytes": path.stat().st_size,
              "sha256": sha256(path),
          }
          for path in snapshots
      },
      "compiler_invocations": 0,
      "gpu_contexts": 0,
      "model_compiles": 0,
      "infer_requests": 0,
      "model_workers": 0,
      "product_workers": 0,
  })

  report = "\n".join((
      "# Post-PR35924 opportunity bound",
      "",
      f"Verdict: **{verdict}**. Required checks: "
      f"`{str(required_checks_passed).lower()}`. No build or worker ran.",
      "",
      "The current upstream set supplies no admitted PTL/Xe3 U4 steady-speed "
      "cut. oneDNN PR5634 is remainder/correctness plus Xe3p/NVL-P repair; "
      "all accepted decode K values are k128-exact. PR5713 is Xe3p OHS. "
      "PR5681 does not change the accepted rank-0 generic FHS strategy and "
      "publishes no complete locked-shape bound. OpenVINO PR37140 fixes DQ "
      "cache identity without changing an OpenCL kernel body.",
      "",
      "The locked model does contain 30 Loops, but every one has nine inputs; "
      "PR36412's Mamba2 matcher requires eight. It is registered only in the "
      "CPU transformation pipeline and adds no GPU implementation, so it is "
      "not a product route.",
      "",
      f"The fixed-schedule FC ceiling saves only "
      f"`{float(fc_aggregate['optimistic_saving_ms']):.6f} ms` versus "
      f"`{float(fc_aggregate['kill_number_ms']):.6f} ms` required. By "
      f"contrast, the exact LM-head fallback scans `{full_scan_bytes:,} B` "
      f"at `{repeat_gbps:.3f}/{confirm_gbps:.3f} GB/s`, already at the "
      "traffic ruler. Its identical `50/495` slow-event pattern exposes up "
      f"to `{gross_mean_avoidable_ms:.6f} ms/token` gross mean work, "
      f"`{gross_mean_avoidable_ms / required_saving_ms:.2f}x` the registered "
      "cut.",
      "",
      "Select an offline exact-token certificate bound: use conservative Q1 "
      "residual bounds to prove that no unseen row can beat the exact "
      "candidate, and retain the full I8 scan whenever proof fails. At least "
      f"`{certificates_needed_for_p95}/50` observed slow events would need "
      "certification to push the fallback population below the 5% p95 tail. "
      "Empirical agreement alone is not proof and does not authorize a "
      "speedup claim.",
      "",
      f"Available memory never fell below "
      f"`{metrics['memory']['minimum_available_bytes']:,} B`; no compiler, "
      "GPU context, model worker, OOM, or restart occurred.",
      "",
  ))
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "output": display(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "upstream_routes_admitted": 0,
      "selected_route": "lm_head_exact_token_certificate_bound",
      "gross_mean_avoidable_ms": gross_mean_avoidable_ms,
      "gross_multiple_over_required":
          gross_mean_avoidable_ms / required_saving_ms,
      "minimum_slow_events_to_certify_for_p95":
          certificates_needed_for_p95,
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
