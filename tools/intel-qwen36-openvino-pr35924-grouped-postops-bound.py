#!/usr/bin/env python3
"""Bound OpenVINO PR35924's locked 2k prefill traffic opportunity.

This gate performs no checkout, source edit, configure, compile, GPU work,
model load, or inference.  It binds the exact upstream delta to the accepted
grouped-MoE consumer, registers a fresh 1.005x prefill cut, and decides whether
one isolated serial candidate-plugin build is worth preparing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-grouped-postops-bound-v0"

MODEL_CONTRACT = (
    ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json")
SEQ2230 = ROOT / (
    "output/onednn-pr5716-grouped-eltwise-bound-"
    "20260731Tseq2230-clean/metrics.json")
SEQ2193 = ROOT / (
    "output/openvino-lm-head-parallel-block-topk-2k-abba8-"
    "20260731Tseq2193-clean/gate.json")
SEQ2151 = ROOT / (
    "output/openvino-exact-attention-two-workgroup-traffic-"
    "20260724Tseq2151-clean/result.json")
PINNED_OV_REPO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
MOE_REL = (
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/"
    "moe_3gemm_swiglu_opt.cpp")
PINNED_MOE = PINNED_OV_REPO / MOE_REL

EXPECTED_SHA256 = {
    MODEL_CONTRACT: (
        "c9616cf79e96f5e628a2425198b8f9ea67c703ddcb379df1012ebe8843cbfd48"),
    SEQ2230: (
        "c83973486a9486ecfdf94bd7c190c8124e0d9e3768dcbdae378f4804c524146b"),
    SEQ2193: (
        "c125f51dde39d6080ed1b4a8698cb3864874fcf31e3acb5a38fffbae9c86ceee"),
    SEQ2151: (
        "ef1e96dc63e884f688e1057f19261fc844f24b1476e601573fd25b93a6fc9e5b"),
    PINNED_MOE: (
        "d388d8034526c2a3f438a62ff7a5f5be7df060ba911683248960d08dfc92c855"),
}

PR35924_API = (
    "https://api.github.com/repos/openvinotoolkit/openvino/pulls/35924")
PR5535_API = "https://api.github.com/repos/uxlfoundation/oneDNN/pulls/5535"
EXPECTED_PR35924 = {
    "number": 35924,
    "state": "open",
    "draft": False,
    "title": "[GPU] Add post_ops support for grouped_gemm",
    "head": "5cf601a51ce1dbb5a223c08a41c126e46ddf5628",
    "base": "337f0f63bf5b03fcc0a6d555288eae5e8e0e2f3b",
    "commits": 3,
    "changed_files": 1,
}
EXPECTED_PR5535 = {
    "number": 5535,
    "state": "closed",
    "title": "grouped matmul CPU and GPU ref to support all eltwise post-ops",
    "head": "0b7e9c1f83491a685b67e20f1f8fff2dc9ce7331",
    "merged": True,
}

TARGET_PREFILL_RATIO = 1.005
EXPECTED_INPUT_TOKENS = 2048
EXPECTED_LAYER_COUNT = 40
EXPECTED_ACTIVE_EXPERTS = 8
EXPECTED_INTERMEDIATE_SIZE = 512
F16_BYTES = 2
ELIMINATED_F16_TRANSFERS_PER_ELEMENT = 2
PREFLIGHT_BYTES = 8 * 1024**3
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


def run(
    command: list[str], cwd: Path = ROOT, *, require_success: bool = True,
) -> subprocess.CompletedProcess[str]:
  result = subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  if require_success and result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n{result.stderr}")
  return result


def git(*args: str, cwd: Path = ROOT) -> str:
  return run(["git", *args], cwd=cwd).stdout.strip()


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
  for row in git(
      "status", "--porcelain", "--untracked-files=all").splitlines():
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


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(
      encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


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


def raw_url(repo: str, revision: str, path: str) -> str:
  return f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"


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


def function_body(source: str, signature: str) -> str:
  start = source.index(signature)
  opening = source.index("{", start)
  depth = 0
  for index in range(opening, len(source)):
    if source[index] == "{":
      depth += 1
    elif source[index] == "}":
      depth -= 1
      if depth == 0:
        return source[start:index + 1]
  raise RuntimeError(f"unterminated function: {signature}")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  raw = output / "raw"
  raw.mkdir()

  required = tuple(EXPECTED_SHA256)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing PR35924 bound inputs: " + ", ".join(missing))
  memory_samples = [{"label": "start",
                     "available_bytes": available_memory_bytes()}]
  if memory_samples[-1]["available_bytes"] < MEMORY_STOP_BYTES:
    raise SystemExit("available memory below 4 GiB before source audit")

  repo = repository_state(output)
  contract = load_json(MODEL_CONTRACT)
  seq2230 = load_json(SEQ2230)
  seq2193 = load_json(SEQ2193)
  seq2151 = load_json(SEQ2151)
  pinned_source = PINNED_MOE.read_text(encoding="utf-8")

  pull35924 = fetch_json(PR35924_API, args.network_timeout_s)
  pull5535 = fetch_json(PR5535_API, args.network_timeout_s)
  diff35924 = fetch(
      PR35924_API, args.network_timeout_s,
      "application/vnd.github.v3.diff").decode("utf-8", errors="replace")
  summary35924 = pull_summary(pull35924)
  summary5535 = pull_summary(pull5535)
  base_source = fetch(
      raw_url(
          "openvinotoolkit/openvino", EXPECTED_PR35924["base"], MOE_REL),
      args.network_timeout_s, "text/plain").decode(
          "utf-8", errors="replace")
  head_source = fetch(
      raw_url(
          "openvinotoolkit/openvino", EXPECTED_PR35924["head"], MOE_REL),
      args.network_timeout_s, "text/plain").decode(
          "utf-8", errors="replace")
  write_json(raw / "pr35924.json", pull35924)
  write_json(raw / "pr5535.json", pull5535)
  (raw / "pr35924.diff").write_text(diff35924, encoding="utf-8")
  (raw / "pr35924-base-moe.cpp").write_text(
      base_source, encoding="utf-8")
  (raw / "pr35924-head-moe.cpp").write_text(
      head_source, encoding="utf-8")

  apply_check = run(
      ["git", "apply", "--check", str(raw / "pr35924.diff")],
      cwd=PINNED_OV_REPO, require_success=False)
  changed_files = set(re.findall(
      r"^diff --git a/(.+?) b/", diff35924, flags=re.MULTILINE))
  base_prefill = function_body(
      base_source, "cldnn::event::ptr exec_prefill_grouped_gemm")
  head_prefill = function_body(
      head_source, "cldnn::event::ptr exec_prefill_grouped_gemm")
  pinned_prefill = function_body(
      pinned_source, "cldnn::event::ptr exec_prefill_grouped_gemm")
  decode_branch_pattern = re.compile(
      r"if \(token_num == 1\) \{\s*"
      r"return exec_single_token\(\{topk_event\}, instance, scratch\);\s*\}")
  pinned_decode = decode_branch_pattern.findall(pinned_source)

  architecture = contract["product_model"]["architecture"]
  layers = int(architecture["layers"])
  active_experts = int(architecture["active_experts"])
  intermediate_size = int(architecture["moe_intermediate_size"])
  performance = seq2193.get("performance", [])
  if not isinstance(performance, list) or len(performance) != 1:
    raise RuntimeError("seq2193 must contain exactly one performance case")
  carrier = performance[0]
  input_tokens = int(seq2193["config"]["cases"][0]["expected_tokens"])
  candidate_prefill_rate = float(
      carrier["absolute_floors"]["prefill_median"])
  candidate_prefill_wall_ms = (
      input_tokens / candidate_prefill_rate * 1000.0)
  required_fraction = 1.0 - 1.0 / TARGET_PREFILL_RATIO
  required_total_cut_ms = candidate_prefill_wall_ms * required_fraction
  required_per_layer_cut_ms = required_total_cut_ms / layers

  total_gathered_tokens = input_tokens * active_experts
  intermediate_elements_per_layer = (
      total_gathered_tokens * intermediate_size)
  eliminated_bytes_per_element = (
      F16_BYTES * ELIMINATED_F16_TRANSFERS_PER_ELEMENT)
  eliminated_bytes_per_layer = (
      intermediate_elements_per_layer * eliminated_bytes_per_element)
  eliminated_bytes_all_layers = eliminated_bytes_per_layer * layers
  removed_launches_all_layers = layers
  bandwidth_point_gb_s = float(seq2151["bandwidth_point_gb_s"])
  bandwidth_lcb_gb_s = float(seq2151["bandwidth_lcb_gb_s"])
  fast_ruler_gb_s = max(bandwidth_point_gb_s, bandwidth_lcb_gb_s)
  traffic_equivalent_ms = (
      eliminated_bytes_all_layers / (fast_ruler_gb_s * 1e9) * 1000.0)
  traffic_equivalent_per_layer_ms = traffic_equivalent_ms / layers
  opportunity_multiple = traffic_equivalent_ms / required_total_cut_ms
  opportunity_margin_ms = traffic_equivalent_ms - required_total_cut_ms

  prior_grouped = seq2230["locked_consumer"]["prior_grouped_evidence"]
  pull_identity_exact = bool(
      summary35924["number"] == EXPECTED_PR35924["number"]
      and summary35924["state"] == EXPECTED_PR35924["state"]
      and summary35924["draft"] == EXPECTED_PR35924["draft"]
      and summary35924["title"] == EXPECTED_PR35924["title"]
      and summary35924["head_sha"] == EXPECTED_PR35924["head"]
      and summary35924["base_sha"] == EXPECTED_PR35924["base"]
      and summary35924["commits"] == EXPECTED_PR35924["commits"]
      and summary35924["changed_files"]
      == EXPECTED_PR35924["changed_files"]
      and summary5535["number"] == EXPECTED_PR5535["number"]
      and summary5535["state"] == EXPECTED_PR5535["state"]
      and summary5535["title"] == EXPECTED_PR5535["title"]
      and summary5535["head_sha"] == EXPECTED_PR5535["head"]
      and bool(summary5535["merged_at"]) == EXPECTED_PR5535["merged"])
  source_delta_exact = bool(
      changed_files == {MOE_REL}
      and "*grouped_gemm_prefill_swiglu" in base_prefill
      and "*grouped_gemm_prefill_swiglu" in pinned_prefill
      and "*grouped_gemm_prefill_swiglu" not in head_prefill
      and "gate_po.append_eltwise" in head_source
      and "gate_po.append_binary" in head_source
      and "DNNL_ARG_ATTR_MULTIPLE_POST_OP" in head_prefill
      and head_prefill.find("gk.up_prim.execute")
      < head_prefill.find("gk.gate_prim.execute")
      and len(pinned_decode) == 1
      and "token_num == 1" not in diff35924
      and "exec_single_token" not in diff35924)
  prior_f16_exact = bool(
      prior_grouped.get("pass") is True
      and prior_grouped.get("decode", {}).get("count") == layers
      and prior_grouped.get("prefill", {}).get("count") == layers
      and prior_grouped.get("decode", {}).get("exec_types")
      == {"ocl::moe::moe_3gemm_swiglu_opt___f16": layers}
      and prior_grouped.get("prefill", {}).get("exec_types")
      == {"ocl::moe::moe_3gemm_swiglu_opt___f16": layers})

  memory_samples.append({
      "label": "complete", "available_bytes": available_memory_bytes()})
  hashes = {relative(path): sha256(path) for path in required}
  hashes_exact = all(
      hashes[relative(path)] == expected
      for path, expected in EXPECTED_SHA256.items())
  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "frozen_evidence_and_locked_source_hashes_exact",
          hashes_exact, hashes=hashes),
      check(
          "live_upstream_identities_exact",
          pull_identity_exact,
          pr35924=summary35924, pr5535=summary5535),
      check(
          "pr35924_is_one_product_file_and_applies_to_accepted_source",
          changed_files == {MOE_REL} and apply_check.returncode == 0,
          changed_files=sorted(changed_files),
          apply_check_returncode=apply_check.returncode,
          apply_check_stderr=apply_check.stderr.strip()),
      check(
          "exact_prefill_fusion_and_decode_noninterference_bound",
          source_delta_exact,
          base_has_separate_swiglu=(
              "*grouped_gemm_prefill_swiglu" in base_prefill),
          head_has_separate_swiglu=(
              "*grouped_gemm_prefill_swiglu" in head_prefill),
          head_has_grouped_postops=(
              "gate_po.append_eltwise" in head_source
              and "gate_po.append_binary" in head_source),
          accepted_decode_branch_exact=len(pinned_decode) == 1,
          patch_touches_decode_branch=(
              "token_num == 1" in diff35924
              or "exec_single_token" in diff35924)),
      check(
          "locked_shape_and_f16_owner_exact",
          layers == EXPECTED_LAYER_COUNT
          and active_experts == EXPECTED_ACTIVE_EXPERTS
          and intermediate_size == EXPECTED_INTERMEDIATE_SIZE
          and input_tokens == EXPECTED_INPUT_TOKENS
          and prior_f16_exact,
          layers=layers, input_tokens=input_tokens,
          active_experts=active_experts,
          intermediate_size=intermediate_size,
          prior_grouped=prior_grouped),
      check(
          "fresh_incremental_prefill_kill_number_registered",
          carrier["case_id"] == "prefill_shape_002k"
          and carrier["paired_block_count"] == 8
          and TARGET_PREFILL_RATIO == 1.005
          and candidate_prefill_rate > 0
          and required_total_cut_ms > 0,
          case_id=carrier["case_id"],
          paired_block_count=carrier["paired_block_count"],
          candidate_prefill_rate_tokens_s=candidate_prefill_rate,
          candidate_prefill_wall_ms=candidate_prefill_wall_ms,
          target_ratio=TARGET_PREFILL_RATIO,
          required_total_cut_ms=required_total_cut_ms,
          required_per_layer_cut_ms=required_per_layer_cut_ms),
      check(
          "exact_eliminated_traffic_and_launch_count_bound",
          intermediate_elements_per_layer == 8_388_608
          and eliminated_bytes_per_layer == 33_554_432
          and eliminated_bytes_all_layers == 1_342_177_280
          and removed_launches_all_layers == 40,
          total_gathered_tokens=total_gathered_tokens,
          intermediate_elements_per_layer=intermediate_elements_per_layer,
          eliminated_bytes_per_element=eliminated_bytes_per_element,
          eliminated_bytes_per_layer=eliminated_bytes_per_layer,
          eliminated_bytes_all_layers=eliminated_bytes_all_layers,
          removed_launches_all_layers=removed_launches_all_layers),
      check(
          "fast_ruler_traffic_equivalent_clears_cut_by_two_x",
          seq2151["required_checks_passed"] is True
          and opportunity_multiple >= 2.0,
          bandwidth_point_gb_s=bandwidth_point_gb_s,
          bandwidth_lcb_gb_s=bandwidth_lcb_gb_s,
          fast_ruler_gb_s=fast_ruler_gb_s,
          traffic_equivalent_ms=traffic_equivalent_ms,
          traffic_equivalent_per_layer_ms=(
              traffic_equivalent_per_layer_ms),
          required_total_cut_ms=required_total_cut_ms,
          opportunity_multiple=opportunity_multiple,
          opportunity_margin_ms=opportunity_margin_ms,
          interpretation=(
              "A scale bound for build admission, not measured or guaranteed "
              "product saving; the removed launch also has no assigned time.")),
      check(
          "source_bound_used_no_checkout_edit_build_gpu_model_or_inference",
          True,
          checkouts_created=0, source_files_edited=0,
          configure_invocations=0, compiler_invocations=0,
          gpu_contexts_created=0, model_workers_started=0,
          infer_requests_created=0),
      check(
          "memory_guards_hold",
          memory_samples[0]["available_bytes"] >= PREFLIGHT_BYTES
          and min(row["available_bytes"] for row in memory_samples)
          >= MEMORY_STOP_BYTES,
          preflight_bytes=PREFLIGHT_BYTES,
          stop_bytes=MEMORY_STOP_BYTES,
          samples=memory_samples),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": passed,
      "isolated_serial_candidate_plugin_build_admitted": passed,
      "product_build_or_speed_claim_admitted": False,
      "verdict": (
          "admit_one_pr35924_isolated_serial_candidate_plugin_build"
          if passed else
          "hold_pr35924_for_source_or_arithmetic_mismatch"),
      "reason": (
          "At the exact 2k grouped-MoE shape, PR35924 removes two mandatory "
          "F16 transfers for 8,388,608 elements in each of 40 layers and one "
          "standalone prefill launch per layer, while the single-token branch "
          "is unchanged. The 1.342-GB traffic delta is 2.47x the fresh "
          "1.005x prefill cut when expressed at the retained fast bandwidth "
          "ruler. This funds one isolated serial candidate build, not a "
          "performance claim."),
      "next_action": (
          "materialize the exact one-file patch in an isolated accepted-"
          "carrier source, bind its version and diff, then build only the "
          "candidate GPU plugin at -j1 under 8-GiB preflight / 4-GiB abort"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": {
          relative(path): {
              "sha256": hashes[relative(path)],
              "bytes": path.stat().st_size,
          } for path in required},
      "upstream": {
          "pr35924": summary35924,
          "pr5535": summary5535,
          "changed_files": sorted(changed_files),
          "apply_check": {
              "repository": str(PINNED_OV_REPO),
              "returncode": apply_check.returncode,
              "stdout": apply_check.stdout.strip(),
              "stderr": apply_check.stderr.strip(),
          },
      },
      "locked_shape": {
          "layers": layers,
          "input_tokens": input_tokens,
          "active_experts": active_experts,
          "intermediate_size": intermediate_size,
          "activation_bytes": F16_BYTES,
          "prior_grouped_execution": prior_grouped,
      },
      "registered_prefill_cut": {
          "case_id": carrier["case_id"],
          "paired_block_count": carrier["paired_block_count"],
          "candidate_prefill_rate_tokens_s": candidate_prefill_rate,
          "candidate_prefill_wall_ms": candidate_prefill_wall_ms,
          "target_ratio": TARGET_PREFILL_RATIO,
          "required_wall_fraction": required_fraction,
          "required_total_cut_ms": required_total_cut_ms,
          "required_per_layer_cut_ms": required_per_layer_cut_ms,
      },
      "traffic_and_launch_bound": {
          "total_gathered_tokens_per_layer": total_gathered_tokens,
          "intermediate_elements_per_layer": (
              intermediate_elements_per_layer),
          "eliminated_transfers": [
              "standalone gate GEMM destination write",
              "standalone SwiGLU gate-input read",
          ],
          "eliminated_bytes_per_element": eliminated_bytes_per_element,
          "eliminated_bytes_per_layer": eliminated_bytes_per_layer,
          "eliminated_bytes_all_layers": eliminated_bytes_all_layers,
          "removed_launches_all_layers": removed_launches_all_layers,
          "decode_launch_delta": 0,
          "bandwidth_point_gb_s": bandwidth_point_gb_s,
          "bandwidth_lcb_gb_s": bandwidth_lcb_gb_s,
          "fast_ruler_gb_s": fast_ruler_gb_s,
          "traffic_equivalent_ms": traffic_equivalent_ms,
          "traffic_equivalent_per_layer_ms": (
              traffic_equivalent_per_layer_ms),
          "opportunity_multiple": opportunity_multiple,
          "opportunity_margin_ms": opportunity_margin_ms,
          "claim_boundary": (
              "source and traffic scale bound only; no measured saving"),
      },
      "checks": checks,
      "process_census": {
          "checkouts_created": 0,
          "source_files_edited": 0,
          "configure_invocations": 0,
          "compiler_invocations": 0,
          "gpu_contexts_created": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
      },
      "memory": {
          "preflight_bytes": PREFLIGHT_BYTES,
          "stop_bytes": MEMORY_STOP_BYTES,
          "minimum_available_bytes": min(
              row["available_bytes"] for row in memory_samples),
          "samples": memory_samples,
      },
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# OpenVINO PR35924 grouped-postops bound\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Fresh 1.005x prefill cut: `{required_total_cut_ms:.6f} ms` "
      f"total / `{required_per_layer_cut_ms * 1000.0:.3f} us/layer`\n"
      f"- Exact eliminated traffic: `{eliminated_bytes_all_layers}` bytes "
      f"and `{removed_launches_all_layers}` prefill launches\n"
      f"- Fast-ruler traffic equivalent: `{traffic_equivalent_ms:.6f} ms` "
      f"(`{opportunity_multiple:.3f}x` the cut)\n"
      "- Decode delta by source branch: `0`; compiler/GPU/model/"
      "InferRequest: `0/0/0/0`\n"
      "- This is build admission, not measured saving or a product claim.\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": passed,
      "verdict": verdict["verdict"],
      "required_total_cut_ms": required_total_cut_ms,
      "eliminated_bytes_all_layers": eliminated_bytes_all_layers,
      "removed_launches_all_layers": removed_launches_all_layers,
      "traffic_equivalent_ms": traffic_equivalent_ms,
      "opportunity_multiple": opportunity_multiple,
      "minimum_available_bytes": metrics["memory"][
          "minimum_available_bytes"],
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
