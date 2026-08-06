#!/usr/bin/env python3
"""Bound oneDNN PR5716 against the locked grouped-MoE consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
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
SCHEMA = "intel-qwen36-onednn-pr5716-grouped-eltwise-bound-v0"
MODEL_XML = Path("/home/intel/Qwen3.6-35B-A3B-ov/openvino_language_model.xml")
PINNED_OV_MOE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/"
    "moe_3gemm_swiglu_opt.cpp")
SEQ1303 = ROOT / (
    "output/openvino-dynamic-split-inplace-bound-"
    "20260717Tseq1303b-cleanZ/metrics.json")
SEQ1303_SHA256 = (
    "a36988fc090b99eee56288ce0ac59ca6be5a4ec4f38f9f1d6bd43425c947ee29")
SEQ2204 = ROOT / (
    "output/openvino-current-bundle-profile-refresh-"
    "20260731Tseq2204-short-o130-clean/raw/bucket_002048/profile/"
    "candidate/worker-result.json")
SEQ2204_SHA256 = (
    "d2e66caaef46344a82d3238af120f698a7eaad005085f0a40b6d91bcbc4e21f1")
PR5716_API = "https://api.github.com/repos/uxlfoundation/oneDNN/pulls/5716"
PR5716_DIFF = PR5716_API
PR35924_API = (
    "https://api.github.com/repos/openvinotoolkit/openvino/pulls/35924")
PR35924_DIFF = PR35924_API
PR5535_API = "https://api.github.com/repos/uxlfoundation/oneDNN/pulls/5535"
EXPECTED = {
    "pr5716": {
        "number": 5716,
        "state": "open",
        "draft": False,
        "title": "[GPU] Enable all eltwise algorithms in grouped GEMM",
        "head": "bda707e1cafa1f650558b85faba1483e53eef07c",
        "base": "b7bfe8bf3c252d31bcbf93d30b617897a1e447be",
        "changed_files": 7,
    },
    "pr35924": {
        "number": 35924,
        "state": "open",
        "draft": False,
        "title": "[GPU] Add post_ops support for grouped_gemm",
        "head": "5cf601a51ce1dbb5a223c08a41c126e46ddf5628",
    },
    "pr5535": {
        "number": 5535,
        "state": "closed",
        "title": "grouped matmul CPU and GPU ref to support all eltwise post-ops",
        "head": "0b7e9c1f83491a685b67e20f1f8fff2dc9ce7331",
        "merged": True,
    },
}
PR5716_FILES = {
    "src/gpu/intel/include/eltwise.h",
    "src/gpu/intel/include/eltwise_bwd_body.h",
    "src/gpu/intel/include/eltwise_fwd_body.h",
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_post_ops_gen.cpp",
    "src/gpu/intel/matmul/grouped_post_ops_gen.hpp",
    "tests/benchdnn/inputs/matmul/harness_matmul_grouped_2dby3d",
}
MEMORY_STOP_BYTES = 4 * 1024**3


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--network-timeout-s", default=30.0, type=float)
  args = parser.parse_args()
  if args.network_timeout_s <= 0:
    parser.error("network timeout must be positive")
  return args


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
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


def proc_mem_available() -> int:
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  return 0


def fetch(url: str, timeout_s: float, accept: str) -> bytes:
  request = urllib.request.Request(
      url,
      headers={
          "Accept": accept,
          "User-Agent": "intel-qwen36-source-bound",
      })
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    return response.read()


def fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
  value = json.loads(fetch(
      url, timeout_s, "application/vnd.github+json").decode("utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected object from {url}")
  return value


def pull_summary(pull: dict[str, Any]) -> dict[str, Any]:
  return {
      "number": pull.get("number"),
      "state": pull.get("state"),
      "draft": pull.get("draft"),
      "title": pull.get("title"),
      "head_sha": pull.get("head", {}).get("sha"),
      "base_sha": pull.get("base", {}).get("sha"),
      "created_at": pull.get("created_at"),
      "updated_at": pull.get("updated_at"),
      "merged_at": pull.get("merged_at"),
      "commits": pull.get("commits"),
      "additions": pull.get("additions"),
      "deletions": pull.get("deletions"),
      "changed_files": pull.get("changed_files"),
      "html_url": pull.get("html_url"),
  }


def raw_url(repo: str, revision: str, path: str) -> str:
  return f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def model_activation_census() -> dict[str, Any]:
  root = ET.parse(MODEL_XML).getroot()
  rows = []
  for layer in root.findall(".//layer"):
    name = str(layer.get("name", ""))
    layer_type = str(layer.get("type", ""))
    if layer_type in ("Swish", "Gelu"):
      rows.append({"name": name, "type": layer_type})
  counts = Counter(row["type"] for row in rows)
  return {
      "all_counts": dict(sorted(counts.items())),
      "routed_expert_silu_count": sum(
          "mlp.experts.act_fn/aten::silu/Swish" in row["name"]
          for row in rows),
      "shared_expert_silu_count": sum(
          "mlp.shared_expert.act_fn/aten::silu/Swish" in row["name"]
          for row in rows),
      "routed_or_shared_gelu_count": sum(
          row["type"] == "Gelu"
          and ("mlp.experts." in row["name"]
               or "mlp.shared_expert." in row["name"])
          for row in rows),
  }


def find_seq1303_check(metrics: dict[str, Any], name: str) -> dict[str, Any]:
  rows = [
      row for row in metrics.get("checks", [])
      if isinstance(row, dict) and row.get("name") == name]
  if len(rows) != 1:
    raise RuntimeError(f"expected one seq1303 check named {name}")
  return rows[0]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  raw = output / "raw"
  raw.mkdir()
  required = (MODEL_XML, PINNED_OV_MOE, SEQ1303, SEQ2204)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing PR5716 audit inputs: " + ", ".join(missing))

  memory_start = proc_mem_available()
  if memory_start < MEMORY_STOP_BYTES:
    raise SystemExit("available memory below 4 GiB before source audit")
  repo = repository_state(output)
  seq1303 = load_json(SEQ1303)
  seq2204 = load_json(SEQ2204)

  pull5716 = fetch_json(PR5716_API, args.network_timeout_s)
  pull35924 = fetch_json(PR35924_API, args.network_timeout_s)
  pull5535 = fetch_json(PR5535_API, args.network_timeout_s)
  diff5716 = fetch(
      PR5716_DIFF, args.network_timeout_s,
      "application/vnd.github.v3.diff").decode("utf-8", errors="replace")
  diff35924 = fetch(
      PR35924_DIFF, args.network_timeout_s,
      "application/vnd.github.v3.diff").decode("utf-8", errors="replace")
  summary5716 = pull_summary(pull5716)
  summary35924 = pull_summary(pull35924)
  summary5535 = pull_summary(pull5535)
  (raw / "pr5716.json").write_text(
      json.dumps(pull5716, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  (raw / "pr5716.diff").write_text(diff5716, encoding="utf-8")
  (raw / "pr35924.json").write_text(
      json.dumps(pull35924, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  (raw / "pr35924.diff").write_text(diff35924, encoding="utf-8")
  (raw / "pr5535.json").write_text(
      json.dumps(pull5535, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")

  grouped_path = "src/gpu/intel/matmul/grouped_post_ops_gen.cpp"
  base_grouped = fetch(
      raw_url("uxlfoundation/oneDNN", EXPECTED["pr5716"]["base"],
              grouped_path),
      args.network_timeout_s, "text/plain").decode(
          "utf-8", errors="replace")
  head_grouped = fetch(
      raw_url("uxlfoundation/oneDNN", EXPECTED["pr5716"]["head"],
              grouped_path),
      args.network_timeout_s, "text/plain").decode(
          "utf-8", errors="replace")
  (raw / "pr5716-base-grouped_post_ops_gen.cpp").write_text(
      base_grouped, encoding="utf-8")
  (raw / "pr5716-head-grouped_post_ops_gen.cpp").write_text(
      head_grouped, encoding="utf-8")
  head_fwd = fetch(
      raw_url("uxlfoundation/oneDNN", EXPECTED["pr5716"]["head"],
              "src/gpu/intel/include/eltwise_fwd_body.h"),
      args.network_timeout_s, "text/plain").decode(
          "utf-8", errors="replace")
  (raw / "pr5716-head-eltwise_fwd_body.h").write_text(
      head_fwd, encoding="utf-8")

  changed_files = set(re.findall(
      r"^diff --git a/(.+?) b/", diff5716, flags=re.MULTILINE))
  added_test_lines = [
      line[1:] for line in diff5716.splitlines()
      if line.startswith("+") and not line.startswith("+++")
      and "--attr-post-ops=" in line]
  new_test_algorithms = sorted(set(re.findall(
      r"eltwise_[a-z0-9_]+", "\n".join(added_test_lines))))
  model = model_activation_census()
  prior_grouped = find_seq1303_check(
      seq1303, "pr35924_is_exact_prefill_only_moe_successor")
  current_moe_count = int(
      seq2204.get("execution_census", {}).get(
          "executed_type_counts", {}).get("MOE3GemmFusedCompressed", 0))
  pinned_moe = PINNED_OV_MOE.read_text(encoding="utf-8")
  base_supports_locked_swish = bool(
      "e.eltwise.alg == alg_kind::eltwise_swish" in base_grouped
      and "eltwise_apply_%d(v) ((%s) * ((v) / "
      "(1.0f + exp(-(%s) * (v)))))" in base_grouped)
  head_refactors_swish_same_expression = bool(
      "is_eltwise_alg_supported(e.eltwise.alg)" in head_grouped
      and "fwd_eltwise_common" in head_grouped
      and "case eltwise_swish" in head_fwd
      and "return s / (POST_OP_LITERAL(1.) + exp(-alpha * s));"
      in head_fwd)
  pr35924_is_only_product_consumer = bool(
      "This eliminates the separate grouped_gemm_prefill_swiglu OCL kernel"
      in diff35924
      and "append_eltwise" in diff35924
      and "append_binary" in diff35924
      and "exec_prefill_grouped_gemm" in pinned_moe
      and "if (token_num == 1)" in pinned_moe
      and "return exec_single_token" in pinned_moe
      and prior_grouped.get("pass") is True)
  pull_identity_ok = bool(
      summary5716["number"] == EXPECTED["pr5716"]["number"]
      and summary5716["state"] == EXPECTED["pr5716"]["state"]
      and summary5716["draft"] == EXPECTED["pr5716"]["draft"]
      and summary5716["title"] == EXPECTED["pr5716"]["title"]
      and summary5716["head_sha"] == EXPECTED["pr5716"]["head"]
      and summary5716["base_sha"] == EXPECTED["pr5716"]["base"]
      and summary5716["changed_files"]
      == EXPECTED["pr5716"]["changed_files"]
      and summary35924["number"] == EXPECTED["pr35924"]["number"]
      and summary35924["state"] == EXPECTED["pr35924"]["state"]
      and summary35924["draft"] == EXPECTED["pr35924"]["draft"]
      and summary35924["title"] == EXPECTED["pr35924"]["title"]
      and summary35924["head_sha"] == EXPECTED["pr35924"]["head"]
      and summary5535["number"] == EXPECTED["pr5535"]["number"]
      and summary5535["state"] == EXPECTED["pr5535"]["state"]
      and summary5535["title"] == EXPECTED["pr5535"]["title"]
      and summary5535["head_sha"] == EXPECTED["pr5535"]["head"]
      and bool(summary5535["merged_at"]) == EXPECTED["pr5535"]["merged"])
  memory_end = proc_mem_available()

  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "retained_product_evidence_identity_exact",
          sha256(SEQ1303) == SEQ1303_SHA256
          and sha256(SEQ2204) == SEQ2204_SHA256,
          seq1303_sha256=sha256(SEQ1303),
          seq2204_sha256=sha256(SEQ2204)),
      check(
          "live_upstream_identities_exact",
          pull_identity_ok,
          pr5716=summary5716,
          pr35924=summary35924,
          pr5535=summary5535),
      check(
          "pr5716_changed_surface_exact",
          changed_files == PR5716_FILES,
          changed_files=sorted(changed_files)),
      check(
          "locked_moe_activation_is_swish_not_new_gelu",
          model["routed_expert_silu_count"] == 40
          and model["shared_expert_silu_count"] == 40
          and model["routed_or_shared_gelu_count"] == 0,
          census=model),
      check(
          "pr5716_base_already_supports_locked_swish",
          base_supports_locked_swish,
          base_supports_locked_swish=base_supports_locked_swish,
          newly_tested_algorithms=new_test_algorithms),
      check(
          "pr5716_refactors_locked_swish_but_unlocks_no_product_algorithm",
          head_refactors_swish_same_expression
          and set(new_test_algorithms)
          == {"eltwise_gelu_erf", "eltwise_gelu_tanh"},
          refactors_same_swish_expression=head_refactors_swish_same_expression,
          newly_tested_algorithms=new_test_algorithms),
      check(
          "accepted_carrier_has_exact_moe_owner_but_no_grouped_postop_binding",
          current_moe_count == 40
          and prior_grouped.get("decode", {}).get("count") == 40
          and prior_grouped.get("prefill", {}).get("count") == 40
          and pr35924_is_only_product_consumer,
          current_moe_count=current_moe_count,
          prior_grouped=prior_grouped,
          pr35924_is_only_product_consumer=pr35924_is_only_product_consumer),
      check(
          "no_standalone_dispatch_or_materialization_delta",
          base_supports_locked_swish and pr35924_is_only_product_consumer,
          pr5716_new_locked_algorithm_count=0,
          current_product_grouped_postop_bindings=0,
          product_change_owner="OpenVINO PR35924 prefill path"),
      check(
          "source_audit_used_no_compiler_gpu_model_or_infer_request",
          True,
          compilers_started=0,
          gpu_contexts_created=0,
          gpu_kernels_executed=0,
          model_workers_started=0,
          infer_requests_created=0),
      check(
          "memory_stop_held",
          min(memory_start, memory_end) >= MEMORY_STOP_BYTES,
          available_start_bytes=memory_start,
          available_end_bytes=memory_end,
          stop_bytes=MEMORY_STOP_BYTES),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": passed,
      "pr5716_standalone_build_admitted": False,
      "pr35924_prefill_source_bound_admitted": passed,
      "product_build_admitted": False,
      "verdict": (
          "reject_pr5716_standalone_retain_pr35924_prefill_fusion"
          if passed else
          "hold_pr5716_for_source_identity_failure"),
      "reason": (
          "The locked routed and shared experts use Swish, which the PR5716 "
          "base already accepts. PR5716 broadens algorithms and refactors the "
          "same Swish expression, but the accepted carrier has no grouped "
          "post-op binding. Only OpenVINO PR35924 creates that prefill-only "
          "consumer and removes the standalone SwiGLU kernel."),
      "next_action": (
          "derive a zero-build PR35924 exact prefill component traffic and "
          "launch bound; keep decode unchanged and compile only if the bound "
          "can clear a registered prefill cut"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": {
          "model_xml": {
              "path": str(MODEL_XML), "sha256": sha256(MODEL_XML)},
          "pinned_openvino_moe": {
              "path": str(PINNED_OV_MOE), "sha256": sha256(PINNED_OV_MOE)},
          "seq1303": {
              "path": relative(SEQ1303), "sha256": sha256(SEQ1303)},
          "seq2204": {
              "path": relative(SEQ2204), "sha256": sha256(SEQ2204)},
      },
      "upstream": {
          "pr5716": summary5716,
          "pr35924": summary35924,
          "pr5535": summary5535,
          "pr5716_changed_files": sorted(changed_files),
          "pr5716_new_test_algorithms": new_test_algorithms,
      },
      "locked_consumer": {
          "model_activation_census": model,
          "current_moe_count": current_moe_count,
          "prior_grouped_evidence": prior_grouped,
          "base_supports_locked_swish": base_supports_locked_swish,
          "head_refactors_swish_same_expression":
              head_refactors_swish_same_expression,
          "pr35924_is_only_product_consumer":
              pr35924_is_only_product_consumer,
      },
      "process_census": {
          "compilers_started": 0,
          "gpu_contexts_created": 0,
          "gpu_kernels_executed": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
          "product_builds": 0,
      },
      "memory": {
          "available_start_bytes": memory_start,
          "available_end_bytes": memory_end,
          "stop_bytes": MEMORY_STOP_BYTES,
      },
      "checks": checks,
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# oneDNN PR5716 grouped-eltwise bound\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Locked routed/shared Swish / GELU: "
      f"`{model['routed_expert_silu_count']}/"
      f"{model['shared_expert_silu_count']}/"
      f"{model['routed_or_shared_gelu_count']}`\n"
      f"- Newly tested algorithms: `{new_test_algorithms}`\n"
      f"- Current exact MoE owners: `{current_moe_count}`\n"
      "- PR5716 standalone compiler/GPU/model/InferRequest: `0/0/0/0`\n"
      "- Next: source-bound PR35924's prefill-only standalone SwiGLU removal\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": passed,
      "verdict": verdict["verdict"],
      "locked_routed_swish": model["routed_expert_silu_count"],
      "locked_shared_swish": model["shared_expert_silu_count"],
      "locked_moe_gelu": model["routed_or_shared_gelu_count"],
      "newly_tested_algorithms": new_test_algorithms,
      "current_moe_count": current_moe_count,
      "pr5716_standalone_build_admitted": False,
      "pr35924_prefill_source_bound_admitted": passed,
      "minimum_available_bytes": min(memory_start, memory_end),
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
