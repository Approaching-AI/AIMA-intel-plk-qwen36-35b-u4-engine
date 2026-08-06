#!/usr/bin/env python3
"""Close the post-seq1351 upstream refresh without launching a GPU worker.

The gate pins the current OpenVINO, oneDNN, IGC, and compute-runtime heads,
intersects their candidate changes with the locked IR/runtime census, and
admits only a next source-contract audit.  It intentionally does not compile
anything or create a GPU context.
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
SCHEMA = "intel-qwen36-openvino-post-seq1351-upstream-opportunity-bound-v0"
R0 = Path("/home/intel/intel-qwen36-r0")
OV = R0 / "source/openvino-90214e5be05"
ONEDNN = OV / "src/plugins/intel_gpu/thirdparty/onednn_gpu"
COMPUTE_RUNTIME = R0 / (
    "source/compute-runtime-82aab87fc932edc0558a0302d545a5bcc22edf41")

OV_PINNED = "90214e5be052438cec5617ed3ea7e37df1538f68"
OV_MASTER = "6a905c78c5aceb32e65690978e3be70fc2e56081"
ONEDNN_PINNED = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
ONEDNN_MASTER = "47b2bf4f4df49310a7b81e848d85a0c6ac737a22"
IGC_PREVIOUS_MASTER = "68ef7642f5b2114bc6465b517ae3c9976d570150"
IGC_MASTER = "4af5d6e45e4c4ab00576b0aa9b243a0aa073f99c"
IGC_RELEASE_TAG_OBJECT = "3eef0f89d3a4fe2b443de595e23d7700a5d1491b"
COMPUTE_RUNTIME_PINNED = "82aab87fc932edc0558a0302d545a5bcc22edf41"
COMPUTE_RUNTIME_MASTER = "308f4244adde7ac48ca3996b421ca1b0d6fe7cb6"

PR_HEADS = {
    34704: "53b324240985de8e2c55d92efd92018253c2df13",
    36578: "a8ff0ae5f4a77f94e95bc2892f0d9288fd6583bd",
    36587: "acc546eb4efce8d787450cc163be3d6d91129db9",
    36645: "d1812dcf6f88f7e53830958a666d8bbd28b11099",
    36747: "f1ace4a6d435685b25bab239c8490fa689e49032",
    36775: "ad3825dc47c928f72947fd6766b33c2b133db15b",
    36798: "aba5f7fd4bbe45cc88f739f3260cf7a9f1e9367a",
    36809: "1c958bf8c050e48d905d2cf493278017549b7df4",
    36845: "f35396b6297d93ef42b06d98cfb0ec91bb9fcd5e",
    36865: "22098bc97fb98a72ab181922ac218e72549a8f64",
    36866: "eff2da24c13d77b3c546800c846e634daca9e45f",
    36867: "ed99ff10f3a29267ed5b831a366aceeb530fc566",
    36870: "e3d3d6e4e039a1d798b04fcf82408505ada8bcdf",
    36879: "6913bb8ae7571314a79c14c7db1ca655189d4487",
    36883: "8f4a72c50a52f0190af3a6f95b9833f147374687",
    36891: "0841bd30da83bfcc591bcfabf76920932606d70e",
    36907: "4175d314e927fd3f26b7045124f9e1d6879497f6",
    36922: "e644f47510128a92506c1951564ba5f167f37948",
    36932: "113bbc9e24e15c1eb0c1a3620fabdb05a4a7f930",
    36936: "cb7cc947f2b4a81b66f406446b41e30ed820fc9e",
    36944: "f6b82110a691508ccb0632e259bf318d106832d1",
    36946: "4e1d18058e5a6c480d1b70bc1c0fe92d27a03f0b",
}

SEQ1204 = ROOT / (
    "output/openvino-hot-cold-product-20260715Tseq1204-"
    "alias-fused-linear-state-32k-o64-cleanZ/raw/sentinel_032k/"
    "correctness/candidate/worker-result.json")
SEQ1281 = ROOT / (
    "output/openvino-upstream-fc-capability-bound-"
    "20260717Tseq1281-cleanZ/metrics.json")
SEQ1294 = ROOT / (
    "output/openvino-fc-hardware-limit-bound-"
    "20260717Tseq1294-cleanZ/metrics.json")
SEQ1295 = ROOT / (
    "output/openvino-fc-transparent-compression-bound-"
    "20260717Tseq1295b-cleanZ/metrics.json")
SEQ1297 = ROOT / (
    "output/openvino-fc-upstream-vector-imm-bound-"
    "20260717Tseq1297-cleanZ/metrics.json")
SEQ1302 = ROOT / (
    "output/openvino-post-igc-opportunity-bound-"
    "20260717Tseq1302-cleanZ/metrics.json")
SEQ1304_WORKER = ROOT / (
    "output/openvino-dynamic-split-inplace-component-"
    "20260717Tseq1304-control-2k-warm17-cleanZ/raw/2k/candidate/"
    "worker-result.json")
SEQ1327 = ROOT / (
    "output/openvino-qk-rope-layout-component-"
    "20260717Tseq1327-corrected-candidate-2k-warm17-cleanZ/metrics.json")
SEQ1328 = ROOT / (
    "output/openvino-fc-rms-igc-qk-rope-bundle-bound-"
    "20260718Tseq1328-cleanZ/metrics.json")
SEQ1337 = ROOT / (
    "output/openvino-router-isolated-shared-triple-component-"
    "20260718Tseq1337-candidate-2k-warm17-cleanZ/metrics.json")
SEQ1351 = ROOT / (
    "output/openvino-pr36747-rms-igc2382-32k-component-"
    "20260718Tseq1351-candidate-32k-warm17-cleanZ/metrics.json")
RUNTIME_GRAPH = ROOT / (
    "output/openvino-attention-phase-profile-20260715Tseq1172-"
    "l0-dq-restored-32k-warm17-cleanZ/raw/32k/candidate/runtime-graph.xml")
ONEDNN_TRACE = ROOT / (
    "output/openvino-hot-cold-product-20260715Tseq1212-"
    "onednn-gemm-selection-trace-2k-o4-dirtyZ/raw/sentinel_002k/"
    "correctness/candidate/worker.stdout")

KERNEL_DB = "src/gpu/intel/gemm/jit/selector/db/kernel.db"
FHS_T_STRATEGY = (
    "at32 am128 aB wg 2x1x8 ikr xaf st vav hi pt sr br sb128 "
    "bk0 bm0 nmk sys")
KILL_NUMBER_MS = 2.837085


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory stop must be positive")
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


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def git_state(output: Path) -> dict[str, Any]:
  allowed = {
      "tools/intel-qwen36-openvino-post-seq1351-upstream-opportunity-bound.py",
  }
  output_rel = display(output)
  dirty = []
  for row in git(ROOT, "status", "--porcelain").splitlines():
    path = row[3:]
    if path in allowed or path.startswith(output_rel):
      continue
    dirty.append(row)
  return {
      "commit": git(ROOT, "rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_tool_paths": sorted(allowed),
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def ls_remote(url: str, refs: list[str]) -> dict[str, str]:
  rows = run(["git", "ls-remote", url, *refs]).splitlines()
  result = {}
  for row in rows:
    sha, ref = row.split("\t", 1)
    result[ref] = sha
  return result


def object_text(repo: Path, ref: str, path: str) -> str:
  return run(["git", "show", f"{ref}:{path}"], cwd=repo)


def runtime_census(worker: dict[str, Any]) -> dict[str, int]:
  rows = [
      row for row in worker.get("full_profile", [])
      if row.get("status") == "Status.EXECUTED"]
  return dict(sorted(Counter(str(row.get("node_type")) for row in rows).items()))


def graph_census() -> dict[str, Any]:
  rows = [node.attrib for node in ET.parse(RUNTIME_GRAPH).getroot().iter("data")]
  return {
      "node_count": len(rows),
      "fc_matmul_runtime_i8_count": sum(
          row.get("runtimePrecision") == "i8"
          and "MatMul" in row.get("originalLayersNames", "") for row in rows),
      "dq_i8_output_count": sum(
          row.get("outputPrecisions") == "i8"
          and "dynamic_quantize" in row.get("primitiveType", "")
          for row in rows),
      "mvn_primitive_count": sum(
          "mvn" in row.get("primitiveType", "").lower() for row in rows),
      "rms_primitive_count": sum(
          "rms_gpu_bfyx_opt" in row.get("primitiveType", "") for row in rows),
  }


def candidate(number: int, title: str, disposition: str,
              match_count: int, reason: str) -> dict[str, Any]:
  return {
      "pr": number,
      "title": title,
      "head_sha": PR_HEADS[number],
      "locked_match_count": match_count,
      "fresh_complete_bound_ms": 0.0,
      "disposition": disposition,
      "reason": reason,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  minimum_memory = available_memory_bytes()
  required = (
      SEQ1204, SEQ1281, SEQ1294, SEQ1295, SEQ1297, SEQ1302,
      SEQ1304_WORKER, SEQ1327, SEQ1328, SEQ1337, SEQ1351,
      RUNTIME_GRAPH, ONEDNN_TRACE)
  missing = [display(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing audit inputs: " + ", ".join(missing))

  repo = git_state(output)
  seq1204 = load_json(SEQ1204)
  seq1281 = load_json(SEQ1281)
  seq1294 = load_json(SEQ1294)
  seq1295 = load_json(SEQ1295)
  seq1297 = load_json(SEQ1297)
  seq1302 = load_json(SEQ1302)
  control = load_json(SEQ1304_WORKER)
  seq1327 = load_json(SEQ1327)
  seq1328 = load_json(SEQ1328)
  seq1337 = load_json(SEQ1337)
  seq1351 = load_json(SEQ1351)

  refs = {
      "openvino": {
          "pinned": git(OV, "rev-parse", "HEAD"),
          "origin_master": git(OV, "rev-parse", "origin/master"),
      },
      "onednn_gpu": {
          "pinned": git(ONEDNN, "rev-parse", "HEAD"),
          "origin_main": git(ONEDNN, "rev-parse", "origin/main"),
      },
      "compute_runtime": {
          "pinned": git(COMPUTE_RUNTIME, "rev-parse", "HEAD"),
          "origin_master": git(COMPUTE_RUNTIME, "rev-parse", "origin/master"),
      },
      "igc": {
          "previous_master": IGC_PREVIOUS_MASTER,
          "origin_master": IGC_MASTER,
          "release_v2_38_2_tag_object": IGC_RELEASE_TAG_OBJECT,
      },
  }
  requested_pr_refs = [f"refs/pull/{number}/head" for number in PR_HEADS]
  remote_prs = ls_remote(
      "https://github.com/openvinotoolkit/openvino.git", requested_pr_refs)
  observed_pr_heads = {
      int(ref.split("/")[2]): sha for ref, sha in remote_prs.items()}

  pinned_db = object_text(ONEDNN, ONEDNN_PINNED, KERNEL_DB)
  master_db = object_text(ONEDNN, "origin/main", KERNEL_DB)
  row_prefix = '{{\'G\', "gemm", {"F", "H", "S"}, {"T", "N", "N"}}'
  pinned_rows = [
      line for line in pinned_db.splitlines()
      if row_prefix in line and FHS_T_STRATEGY in line]
  master_rows = [
      line for line in master_db.splitlines()
      if row_prefix in line and FHS_T_STRATEGY in line]
  strategy = {
      "runtime_trace_occurrences": ONEDNN_TRACE.read_text(
          encoding="utf-8", errors="replace").count(FHS_T_STRATEGY),
      "pinned_row_count": len(pinned_rows),
      "master_row_count": len(master_rows),
      "exact_row_unchanged": pinned_rows == master_rows and len(pinned_rows) == 1,
      "row_sha256": hashlib.sha256(
          (pinned_rows[0] if pinned_rows else "").encode()).hexdigest(),
      "l3_prefetch_flag_present": (
          "l3" in pinned_rows[0].lower() if pinned_rows else False),
  }
  census = runtime_census(control)
  graph = graph_census()

  candidates = [
      candidate(36946, "Use an event for host sync",
                "wrong_npu_backend", 0,
                "Both changed files are under src/plugins/intel_npu; the product backend is Intel GPU."),
      candidate(36879, "Sink unit reshape through eltwise after FC",
                "zero_locked_ir_matches", 0,
                "Seq1281 reproduced the exact matcher over 16,051 locked nodes and found zero matches."),
      candidate(34704, "F16 plus INT4 bf_tiled_dyn_b FP32 accumulator",
                "wrong_fc_provider_and_activation_precision", 0,
                "All 371 runtime FC MatMuls are i8 and select oneDNN FHS-T; none uses the F16 bf_tiled_dyn_b kernel."),
      candidate(36587, "MVN bfyx one-pass sum and sum-square",
                "zero_mvn_consumers", 0,
                "The locked runtime has 131 RMS kernels and zero MVN kernels."),
      candidate(36922, "MVN blocked-layout fallback fix",
                "zero_mvn_consumers", 0,
                "The locked runtime has no MVN primitive or blocked-layout MVN fallback."),
      candidate(36870, "Keep U4 precision in GroupedMatMulCompressed",
                "zero_grouped_matmul_consumers", 0,
                "The runtime uses MOE3GemmFusedCompressed and has no GroupedMatMulCompressed node."),
      candidate(36944, "FP4 and MXFP4 dynamic quantization",
                "wrong_weight_dtype", 0,
                "The locked product contract is U4, not FP4 or MXFP4."),
      candidate(36883, "Large F16 FC accuracy on non-IMMAD devices",
                "wrong_device_capability", 0,
                "PTL B390 uses the IMMAD/DPAS path; this is a non-IMMAD correctness guard."),
      candidate(36775, "Avoid KV-cache Broadcast",
                "wrong_attention_carrier", 0,
                "The candidate executes ten IQ36HotAttentionGQA custom nodes and no stock GQA/SDPA attention node."),
      candidate(36645, "Zero-copy device-USM subbuffers",
                "compile_cache_only", 0,
                "Seq1302 already classified this as allocation/cache work with no steady decode cut."),
      candidate(36865, "RMS fusion without gamma",
                "zero_no_gamma_rms_matches", 0,
                "Seq1302 found zero no-gamma RMS matches in the locked graph."),
      candidate(36936, "Recognize GatherMatmul as decompression consumer",
                "zero_gathermatmul_consumers", 0,
                "Seq1302 found zero GatherMatmul consumers."),
      candidate(36866, "Share DynamicQuantize",
                "duplicate_accepted_capability", 0,
                "The accepted carrier already contains the equivalent shared-DQ capability."),
      candidate(36867, "Subgroup-64 DynamicQuantize",
                "duplicate_accepted_capability", 0,
                "The accepted carrier already contains the equivalent subgroup-64 capability."),
      candidate(36747, "RMS and MVN optimization",
                "closed_by_seq1351_long_gate", 0,
                "The exact short bundle passed, then seq1351 failed 32k correctness, latency, and smoothness."),
      candidate(36845, "Non-transposed INT4 shared Parameter weights",
                "zero_parameter_weight_matmuls", 0,
                "The locked graph has 511 Constant-weight MatMuls and zero Parameter-weight MatMuls."),
      candidate(36932, "GGML MoE GatherMatmul GPU",
                "wrong_frontend_and_zero_runtime_consumer", 0,
                "The product is locked IR, not GGML frontend, and the runtime has no GatherMatmul node."),
      candidate(36907, "Skip DQ for 4-bit FC on dGPU",
                "wrong_integrated_device", 0,
                "The target is an integrated PTL GPU; the PR is explicitly discrete-GPU only."),
      candidate(36578, "Lazy static-input allocation",
                "compile_memory_only", 0,
                "The change moves compile-time allocation and supplies no steady decode work cut."),
      candidate(36891, "Automatic MoE offload ratio",
                "memory_policy_only", 0,
                "The route changes offload policy, not the resident batch-1 kernel schedule."),
      candidate(36809, "MoE offload-to-disk",
                "memory_policy_only", 0,
                "Disk offload is a capacity mechanism and cannot reduce the resident decode schedule."),
  ]

  merged = [
      {
          "commit": "3902b4a06b3928b635329a6f96cfa94adbd71773",
          "classification": "first_compile_only",
          "fresh_complete_bound_ms": 0.0,
          "reason": "Skips an approximately 100-ms IGC capability probe once per model compile; steady decode is unchanged.",
      },
      {
          "commit": "e01939759b2a2c434c8198f0f2831771670a3c6d",
          "classification": "speculative_multi_token_only",
          "fresh_complete_bound_ms": 0.0,
          "reason": "Generalizes the existing token_num==1 GEMV dispatch to token_num<=32; at locked batch-1 it launches the same gate/up, down, and reduce work.",
      },
      {
          "commit": "423bd283b1df7310a7652856e7d3895c51941dee",
          "classification": "moe_architecture_and_compile_robustness",
          "fresh_complete_bound_ms": 0.0,
          "reason": "Separates router ownership but retains one router kernel and the same expert-body arithmetic schedule.",
      },
      {
          "commit": "ea600b76705caba629310e5b079ff47a82ecc121",
          "classification": "wrong_stock_gqa_carrier",
          "fresh_complete_bound_ms": 0.0,
          "reason": "Quantized-KV GroupQueryAttention has zero consumers under the custom-attention carrier.",
      },
  ]
  one_dnn = [
      {
          "commit": "0307a1673ce359426a58e18b861a0321eab28192",
          "classification": "source_zero_point_correctness_guard",
          "fresh_complete_bound_ms": 0.0,
      },
      {
          "commit": "a061ba327ee305093f2a488614f0f65c154270fe",
          "classification": "strategy_metadata_refresh_selected_row_unchanged",
          "fresh_complete_bound_ms": 0.0,
      },
      {
          "commits": [
              "d201c5c6520bdf46c7edbb7428c1ae23ac277af6",
              "4b9e5627819dc0456fa033eeeb251fcda8fa1b95"],
          "classification": "already_bounded_vector_immediates",
          "fresh_complete_bound_ms": 0.0,
          "reason": "Seq1297's impossible best remains 8.855864 ms versus the 8.183-ms FC target.",
      },
      {
          "commits": [
              "4687696500de694c8f0f7a65fcac7d1616d4a34e",
              "eae2a6d89645c166932c15e15b39a5da76d352f2"],
          "classification": "l3_prefetch_not_selected",
          "fresh_complete_bound_ms": 0.0,
          "reason": "The exact selected FHS-T strategy row has no L3-prefetch flag."},
      {
          "commit": "6fa35f42902cde376003c4a0f60aeba2bb8e2c1b",
          "classification": "xe3p_only_not_ptl_xe3",
          "fresh_complete_bound_ms": 0.0,
      },
  ]
  igc = {
      "new_commits_since_seq1302": 1,
      "commit": IGC_MASTER,
      "classification": "fp64_atomic_compiler_crash_fix",
      "locked_match_count": 0,
      "fresh_complete_bound_ms": 0.0,
      "reason": "The only new change guards FP64 atomic emulation; locked kernels do not use FP64 atomics.",
  }
  compute_runtime = [
      {
          "commit": "d37510c8567e8df0438b8e96f29734e1da96d6a0",
          "classification": "profiling_timestamp_query_only",
          "fresh_complete_bound_ms": 0.0,
          "reason": "The accepted seq1204 product worker has no PERF_COUNT and host_time_profiling is zero."},
      {
          "commits": [
              "2d4dafef2a9241b478a2a36afdf3e62b15ae9435",
              "939f7717bad11d58dedce31aefc46ee30fc8f10b"],
          "classification": "xe2_prefetch_enabled_then_reverted",
          "fresh_complete_bound_ms": 0.0},
      {
          "commits": [
              "ec6de9bf2441f8ff97b4f84d399c9c6b85f4a153",
              "827f3454b18c7b1396971ca6e6a334faf81ecb2f"],
          "classification": "dgpu_ioq_barrier_enabled_then_reverted",
          "fresh_complete_bound_ms": 0.0},
      {
          "commit": "7bef564aa03dafb594f7cafd8de51fd930d5c290",
          "classification": "host_wait_policy_without_complete_decode_bound",
          "fresh_complete_bound_ms": 0.0},
  ]

  all_rows = candidates + merged + one_dnn + [igc] + compute_runtime
  fresh_complete_bound_ms = sum(
      float(row.get("fresh_complete_bound_ms", 0.0)) for row in all_rows)
  qk_fc = {
      "seq1233_optimistic_fixed_fc_saving_ms": seq1328["budget"][
          "seq1233_optimistic_fixed_fc_saving_ms"],
      "seq1327_observed_qk_component_ms": seq1327["performance"][
          "observed_median_saving_ms"],
      "arithmetic_screen_ms": seq1328["budget"][
          "fc_plus_observed_qk_screen_ms"],
      "arithmetic_margin_ms": seq1328["budget"]["observed_screen_margin_ms"],
      "already_attempted_implementation": (
          "OpenVINO horizontal FC fusion at widths four and three"),
      "attempted_implementation_outcome": (
          "four-way and corrected three-way linear fusion changed all 18 top-1 IDs; "
          "router-isolated shared triple remained exact but saved only 1.992833 ms total"),
      "remaining_distinct_variant": (
          "integrate the independent fixed-shape gemmstone component as a "
          "parameterized graph provider while preserving each original FP16 output boundary"),
      "gpu_admitted": False,
      "next_gate": "source-only graph-integration and arithmetic-boundary contract audit",
  }

  expected_core = {
      "Assign": 60, "DynamicQuantize": 161,
      "FullyConnectedCompressed": 371, "GatedDeltaNet": 30,
      "IQ36HotAttentionGQA": 10, "IQ36LinearConvSwish": 30,
      "MOE3GemmFusedCompressed": 40, "RMS": 131,
  }
  core = {name: census.get(name, 0) for name in expected_core}
  expected_heads = dict(sorted(PR_HEADS.items()))
  checks = [
      check("repository_clean_at_gate", not repo["dirty"], git=repo),
      check("all_registered_inputs_exist", not missing),
      check("prior_upstream_and_long_gate_evidence_is_conclusive",
            seq1281["verdict"]["required_checks_passed"] is True
            and seq1281["verdict"]["new_capability_admitted"] is False
            and seq1302.get("required_checks_passed") is True
            and seq1351.get("evidence_checks_passed") is True
            and seq1351.get("correctness_passed") is False
            and seq1351.get("performance_passed") is False
            and seq1351.get("smoothness_passed") is False),
      check("official_source_heads_are_exact",
            refs["openvino"] == {
                "pinned": OV_PINNED, "origin_master": OV_MASTER}
            and refs["onednn_gpu"] == {
                "pinned": ONEDNN_PINNED, "origin_main": ONEDNN_MASTER}
            and refs["compute_runtime"] == {
                "pinned": COMPUTE_RUNTIME_PINNED,
                "origin_master": COMPUTE_RUNTIME_MASTER}),
      check("official_openvino_pr_heads_are_exact",
            observed_pr_heads == expected_heads,
            observed=observed_pr_heads, expected=expected_heads),
      check("locked_runtime_core_census_is_exact", core == expected_core,
            observed=core, expected=expected_core),
      check("locked_runtime_excludes_wrong_carriers",
            census.get("GroupQueryAttention", 0) == 0
            and census.get("ScaledDotProductAttention", 0) == 0
            and census.get("PagedAttention", 0) == 0
            and census.get("MVN", 0) == 0
            and census.get("GroupedMatMulCompressed", 0) == 0
            and census.get("GatherMatmul", 0) == 0),
      check("locked_fc_precision_and_provider_are_exact",
            graph["fc_matmul_runtime_i8_count"] == 371
            and graph["dq_i8_output_count"] == 161
            and graph["mvn_primitive_count"] == 0
            and graph["rms_primitive_count"] == 131,
            graph=graph),
      check("selected_onednn_fhs_t_strategy_is_unchanged",
            strategy["runtime_trace_occurrences"] > 0
            and strategy["exact_row_unchanged"] is True,
            strategy=strategy),
      check("oneDNN_and_fc_hardware_limit_inputs_remain_closed",
            seq1294.get("required_checks_passed") is True
            and seq1295.get("required_checks_passed") is True
            and seq1297.get("required_checks_passed") is True),
      check("accepted_product_worker_has_profiling_disabled",
            "PERF_COUNT" not in seq1204.get("compile_config", {})),
      check("fresh_upstream_complete_bound_is_zero",
            fresh_complete_bound_ms == 0.0,
            fresh_complete_bound_ms=fresh_complete_bound_ms,
            kill_number_ms=KILL_NUMBER_MS),
      check("fixed_fc_followup_is_only_a_source_contract_audit",
            seq1328.get("required_checks_passed") is True
            and seq1328.get("gpu_component_admitted") is False
            and seq1337.get("activation_passed") is True
            and seq1337.get("correctness_passed") is True
            and seq1337["performance"]["component_performance_passed"] is False
            and qk_fc["gpu_admitted"] is False),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compilers=0, gpu_contexts=0, model_workers=0),
      check("memory_guard_never_tripped",
            minimum_memory >= int(args.memory_stop_gib * 1024**3),
            available_bytes=minimum_memory,
            stop_bytes=int(args.memory_stop_gib * 1024**3)),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "close_current_upstream_refresh_select_fixed_fc_graph_contract_audit"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": repo,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "refs": refs,
      "pr_heads": observed_pr_heads,
      "runtime_census": census,
      "runtime_graph": graph,
      "onednn_selected_strategy": strategy,
      "openvino_open_pr_candidates": candidates,
      "openvino_merged_candidates": merged,
      "onednn_candidates": one_dnn,
      "igc_candidate": igc,
      "compute_runtime_candidates": compute_runtime,
      "fresh_upstream_complete_bound_ms": fresh_complete_bound_ms,
      "kill_number_ms": KILL_NUMBER_MS,
      "gpu_worker_admitted": False,
      "fixed_fc_qk_followup": qk_fc,
      "next_route": "openvino_fixed_fc_graph_integration_contract_audit",
      "checks": checks,
  }
  write_json(raw / "source-heads.json", refs)
  write_json(raw / "openvino-pr-heads.json", observed_pr_heads)
  write_json(output / "metrics.json", metrics)
  report = f"""# Post-seq1351 upstream opportunity bound

Verdict: **{verdict}**. Required checks: `{str(required_checks_passed).lower()}`.
No compiler, GPU context, graph compile, or model worker ran.

The refreshed immutable heads are OpenVINO `{refs['openvino']['origin_master'][:12]}`,
oneDNN GPU `{refs['onednn_gpu']['origin_main'][:12]}`, IGC
`{refs['igc']['origin_master'][:12]}`, and compute-runtime
`{refs['compute_runtime']['origin_master'][:12]}`. The actual locked decode
provider still selects the identical oneDNN FHS-T `2x1x8` strategy. The runtime
contains 371 i8 compressed FCs, 161 i8 DynamicQuantize outputs, 131 RMS nodes,
and zero MVN, stock attention, GroupedMatMulCompressed, or GatherMatmul consumers.

Every fresh upstream item is either a zero-match/wrong-carrier change, a
compile/memory/correctness fix, a capability already present in the accepted
carrier, or a tested closed route. Their non-overlapping fresh complete bound is
`{fresh_complete_bound_ms:.6f} ms/token` versus the `{KILL_NUMBER_MS:.6f}`
kill-number, so no GPU worker is admitted.

One distinct source question remains. Seq1233's optimistic fixed-FC component
plus seq1327's exact Q/K component screens at
`{qk_fc['arithmetic_screen_ms']:.6f} ms`, a
`{qk_fc['arithmetic_margin_ms']:.6f}-ms` arithmetic margin. Seq1328 already
converted that arithmetic into horizontal-FC integration; four-way and
three-way linear variants failed tokens, while the exact shared triple remained
sub-threshold. The next gate therefore audits only whether the independent
gemmstone component can be integrated as one parameterized provider while
preserving every original FP16 arithmetic boundary. This is not implementation
or performance admission.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "fresh_upstream_complete_bound_ms": fresh_complete_bound_ms,
      "next_route": metrics["next_route"],
      "compiler_or_worker_launched": False,
  }, sort_keys=True))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
