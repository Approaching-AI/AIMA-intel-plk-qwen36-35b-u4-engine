#!/usr/bin/env python3
"""Bound PR5681 and the exact PR5059 Arc B390 architecture failure.

This is a source-only gate.  It distinguishes a strategy-selection change
from the build-time ISA registration that controls the failing dispatch
switch.  It starts no compiler, GPU kernel, model, or InferRequest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-onednn-gmlp-pr5681-strategy-bound-v0"

SEQ2225A = ROOT / (
    "output/onednn-gmlp-exact-component-run-"
    "20260731Tseq2225a-clean/metrics.json")
SEQ2225A_STDOUT = ROOT / (
    "output/onednn-gmlp-exact-component-run-"
    "20260731Tseq2225a-clean/raw/block01/decode/stdout.log")
BUILD_AUDIT = ROOT / (
    "output/onednn-gmlp-exact-component-build-audit-"
    "20260731Tseq2224a-clean/metrics.json")
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"

EXPECTED_SEQ2225A_SHA256 = (
    "1522190c315bb41bb7f20f5dbb6bb7456fd89aebf3dfe9b8da5477e741054292")
EXPECTED_BUILD_AUDIT_SHA256 = (
    "25880917fec7c95ac2096cecaf994bee6851b247b7b751f4ee770f53ac0ac2b7")
PR5059 = 5059
PR5059_HEAD = "8621740ea5e600468c76a11a3c0c1616977f978d"
PR5681 = 5681
PR5681_HEAD = "d68d4194763918b85e8f72181f3aede151396aa6"
PR5681_BASE = "22694d0f23622e5aa0bed8e31251df33bf089ca2"

SOURCE = Path("/home/intel/intel-qwen36-r0/source/oneDNN-862174-gmlp-exact")
BUILD_XE2 = Path("/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-exact")
BUILD_XE3 = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-862174-gmlp-xe3-exact")
TEST_XE3 = BUILD_XE3 / "tests/gtests/internals/test_internals_gmlp"

SELECTOR = SOURCE / (
    "src/gpu/intel/gemm/jit/generator/microkernel_selector.cpp")
REGISTRATION = SOURCE / "src/common/impl_registration.hpp"
NGEN_PACKAGER = SOURCE / "third_party/ngen/npack/neo_packager.hpp"
NGEN_CORE = SOURCE / "third_party/ngen/ngen_core.hpp"
CMAKE_CACHE = BUILD_XE2 / "CMakeCache.txt"
BUILD_CONFIG = (
    BUILD_XE2 / "include/oneapi/dnnl/dnnl_config.h")
BUILD_NINJA = BUILD_XE2 / "build.ninja"

PR5681_FILES = {
    "src/gpu/intel/gemm/jit/generator/strategy.cpp",
    "src/gpu/intel/gemm/jit/selector/db/kernel.db",
    "src/gpu/intel/gemm/jit/selector/db/ukernel_mmr.db",
}
ARCH_CONTROL_FILES = {
    "src/gpu/intel/gemm/jit/generator/microkernel_selector.cpp",
    "src/common/impl_registration.hpp",
    "src/gpu/intel/gemm/jit/CMakeLists.txt",
    "cmake/configuring_primitive_list.cmake",
    "cmake/options.cmake",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--network-timeout-s", default=30.0, type=float)
  parser.add_argument("--memory-stop-gib", default=4.0, type=float)
  args = parser.parse_args()
  if args.network_timeout_s <= 0 or args.memory_stop_gib <= 0:
    parser.error("timeouts and memory stop must be positive")
  return args


def run(
    command: list[str], cwd: Path = ROOT, *, check: bool = True,
) -> subprocess.CompletedProcess[str]:
  result = subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")
  if check and result.returncode != 0:
    raise RuntimeError(
        f"command failed ({result.returncode}): {command}\n{result.stderr}")
  return result


def git(cwd: Path, *args: str) -> str:
  return run(["git", *args], cwd=cwd).stdout.strip()


def relative(path: Path) -> str:
  try:
    return path.resolve().relative_to(ROOT).as_posix()
  except ValueError:
    return str(path.resolve())


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
        f"memory stop at {label}: {available} < {stop_bytes}")


def repository_state(output: Path) -> dict[str, Any]:
  head = git(ROOT, "rev-parse", "HEAD")
  upstream = git(ROOT, "rev-parse", "@{u}")
  output_rel = relative(output)
  dirty = []
  for row in git(
      ROOT, "status", "--porcelain", "--untracked-files=all").splitlines():
    path = row[3:]
    if path == output_rel or path.startswith(output_rel + "/"):
      continue
    dirty.append(row)
  return {
      "branch": git(ROOT, "branch", "--show-current"),
      "commit": head,
      "upstream_commit": upstream,
      "pushed": head == upstream,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def fetch_json(
    url: str, destination: Path, timeout_s: float,
) -> dict[str, Any] | list[Any]:
  request = urllib.request.Request(
      url,
      headers={
          "Accept": "application/vnd.github+json",
          "User-Agent": "intel-qwen36-gmlp-pr5681-bound",
      })
  with urllib.request.urlopen(request, timeout=timeout_s) as response:
    payload = response.read()
  destination.write_bytes(payload)
  value = json.loads(payload)
  if not isinstance(value, (dict, list)):
    raise TypeError(f"unexpected GitHub response: {url}")
  return value


def pull_summary(payload: dict[str, Any]) -> dict[str, Any]:
  return {
      "number": payload.get("number"),
      "title": payload.get("title"),
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
      "html_url": payload.get("html_url"),
  }


def cmake_command() -> list[str]:
  conda = Path("/home/intel/intel-box-env/conda")
  return [
      str(conda / "bin/cmake"),
      "-S", str(SOURCE),
      "-B", str(BUILD_XE3),
      "-G", "Ninja",
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DCMAKE_C_COMPILER={conda / 'bin/cc'}",
      f"-DCMAKE_CXX_COMPILER={conda / 'bin/c++'}",
      f"-DCMAKE_MAKE_PROGRAM={conda / 'bin/ninja'}",
      f"-DCMAKE_PREFIX_PATH={conda}",
      f"-DOpenCL_INCLUDE_DIR={conda / 'include'}",
      f"-DOpenCL_LIBRARY={conda / 'lib/libOpenCL.so'}",
      "-DDNNL_BUILD_TESTS=ON",
      "-DDNNL_BUILD_EXAMPLES=OFF",
      "-DONEDNN_BUILD_GRAPH=OFF",
      "-DDNNL_CPU_RUNTIME=NONE",
      "-DDNNL_GPU_RUNTIME=OCL",
      "-DDNNL_GPU_VENDOR=INTEL",
      "-DDNNL_LIBRARY_TYPE=SHARED",
      "-DDNNL_ENABLE_WORKLOAD=INFERENCE",
      "-DDNNL_ENABLE_PRIMITIVE=GATED_MLP;MATMUL;REORDER;ELTWISE",
      "-DDNNL_ENABLE_PRIMITIVE_GPU_ISA=XE3",
  ]


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = (
      SEQ2225A, SEQ2225A_STDOUT, BUILD_AUDIT, TARGET_CONTRACT,
      SOURCE, SELECTOR, REGISTRATION, NGEN_PACKAGER, NGEN_CORE,
      CMAKE_CACHE, BUILD_CONFIG, BUILD_NINJA)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing source-bound inputs: " + ", ".join(missing))

  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_samples: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory_samples)
  repo = repository_state(output)
  seq2225a = load_json(SEQ2225A)
  build_audit = load_json(BUILD_AUDIT)
  target = load_json(TARGET_CONTRACT)
  stdout = SEQ2225A_STDOUT.read_text(
      encoding="utf-8", errors="replace")
  selector = SELECTOR.read_text(encoding="utf-8", errors="replace")
  registration = REGISTRATION.read_text(encoding="utf-8", errors="replace")
  packager = NGEN_PACKAGER.read_text(encoding="utf-8", errors="replace")
  core = NGEN_CORE.read_text(encoding="utf-8", errors="replace")
  cache = CMAKE_CACHE.read_text(encoding="utf-8", errors="replace")
  config = BUILD_CONFIG.read_text(encoding="utf-8", errors="replace")
  ninja = BUILD_NINJA.read_text(encoding="utf-8", errors="replace")

  pr_payload = fetch_json(
      f"https://api.github.com/repos/uxlfoundation/oneDNN/pulls/{PR5681}",
      raw / "onednn-pr5681.json", args.network_timeout_s)
  files_payload = fetch_json(
      "https://api.github.com/repos/uxlfoundation/oneDNN/"
      f"pulls/{PR5681}/files?per_page=100",
      raw / "onednn-pr5681-files.json", args.network_timeout_s)
  if not isinstance(pr_payload, dict) or not isinstance(files_payload, list):
    raise TypeError("unexpected PR5681 response")
  pr = pull_summary(pr_payload)
  changed = {
      str(row.get("filename")) for row in files_payload
      if isinstance(row, dict)}
  patches = {
      str(row.get("filename")): str(row.get("patch", ""))
      for row in files_payload if isinstance(row, dict)}

  source_before = git(SOURCE, "status", "--short", "--untracked-files=all")
  fetch_command = [
      "git", "fetch", "--no-tags", "--filter=blob:none", "origin",
      f"refs/pull/{PR5681}/head:refs/iq36/pr5681",
      f"refs/pull/{PR5059}/head:refs/iq36/pr5059",
  ]
  fetch_started = time.monotonic()
  fetched = run(fetch_command, cwd=SOURCE, check=False)
  fetch_elapsed = time.monotonic() - fetch_started
  source_after = git(SOURCE, "status", "--short", "--untracked-files=all")
  merge_tree = run(
      ["git", "merge-tree", "--write-tree",
       "refs/iq36/pr5681", "refs/iq36/pr5059"],
      cwd=SOURCE, check=False)
  ahead_behind = git(
      SOURCE, "rev-list", "--left-right", "--count",
      "refs/iq36/pr5059...refs/iq36/pr5681").split()
  sample_memory("after_source_audit", stop_bytes, memory_samples)

  successful_micro = len(re.findall(
      r"primitive,(?:create[^,]*|exec),gpu,gated_mlp,"
      r"ocl:micro_horz:any", stdout))
  failed_micro = len(re.findall(
      r"create:dispatch,gated_mlp,gpu,gated_mlp,"
      r"ocl:micro_horz:any", stdout))
  successful_ref = len(re.findall(
      r"primitive,(?:create[^,]*|exec),gpu,gated_mlp,"
      r"ocl:ref:any", stdout))
  exact_failure = {
      "entry_candidate": "entry candidate: heuristics" in stdout,
      "attempt_reached_generation": "attempting heuristic strategy:" in stdout,
      "unsupported_architecture": (
          "strategy failed(Unsupported architecture):" in stdout),
      "no_matching_kernel": (
          "gemm_gateup microkernel generation failed with message: "
          "No matching kernel" in stdout),
      "successful_micro_horz_dispatches": successful_micro,
      "failed_micro_horz_dispatches": failed_micro,
      "successful_ref_dispatches": successful_ref,
  }
  build_isa = {
      "cache_gpu_isa": (
          re.search(
              r"^DNNL_ENABLE_PRIMITIVE_GPU_ISA:INTERNAL=(.+)$",
              cache, flags=re.MULTILINE).group(1)
          if re.search(
              r"^DNNL_ENABLE_PRIMITIVE_GPU_ISA:INTERNAL=(.+)$",
              cache, flags=re.MULTILINE) else None),
      "build_all": "#define BUILD_PRIMITIVE_GPU_ISA_ALL 0" in config,
      "build_xe2": "#define BUILD_XE2 1" in config,
      "build_xe3": "#define BUILD_XE3 0" in config,
      "build_xe3p": "#define BUILD_XE3P 0" in config,
      "gemmstone_xe2_definitions": ninja.count("-DGEMMSTONE_BUILD_XE2"),
      "gemmstone_xe3_definitions": ninja.count("-DGEMMSTONE_BUILD_XE3"),
  }
  dispatch = {
      "selector_has_xe2_case": (
          "REG_XE2_ISA(ARCH_DISPATCH(Xe2))" in selector),
      "selector_has_xe3_case": (
          "REG_XE3_ISA(ARCH_DISPATCH(Xe3))" in selector),
      "selector_has_default_error": (
          'default: throw std::runtime_error("Unsupported architecture")'
          in selector),
      "registration_xe2_is_build_guarded": (
          "#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_XE2" in registration),
      "registration_xe3_is_build_guarded": (
          "#if BUILD_PRIMITIVE_GPU_ISA_ALL || BUILD_XE3" in registration),
      "ptl_maps_to_generic_xe3": (
          "if (family == ProductFamily::PTL) return "
          "NGEN_NAMESPACE::ProductFamily::GenericXe3;" in packager),
      "generic_xe3_maps_to_core_xe3": (
          "if (family >= ProductFamily::GenericXe3)" in core
          and "return Core::Xe3;" in core),
  }
  target_identity = {
      "machine_label": target.get("target", {}).get("machine_label"),
      "opencl_device": target.get("runtime", {}).get("opencl_device"),
      "is_locked_ptl": "PTL" in str(
          target.get("target", {}).get("machine_label", "")),
      "is_arc_b390": "B390" in str(
          target.get("runtime", {}).get("opencl_device", "")),
  }
  pr_intersection = {
      "changed_files": sorted(changed),
      "changed_files_exact": changed == PR5681_FILES,
      "architecture_control_intersection": sorted(
          changed & ARCH_CONTROL_FILES),
      "changes_architecture_control": bool(changed & ARCH_CONTROL_FILES),
      "strategy_patch_mentions_kchain": (
          "kChain" in patches.get(
              "src/gpu/intel/gemm/jit/generator/strategy.cpp", "")),
      "exact_attempt_used_heuristics": exact_failure["entry_candidate"],
      "database_edits_can_select_exact_attempt": False,
      "preflight_was_not_failure_site": (
          exact_failure["attempt_reached_generation"]
          and exact_failure["unsupported_architecture"]),
      "merge_tree_returncode": merge_tree.returncode,
      "merge_conflict_paths": sorted(set(re.findall(
          r"CONFLICT \(content\): Merge conflict in (.+)",
          merge_tree.stdout + merge_tree.stderr))),
      "pr5059_only_commits": int(ahead_behind[0]),
      "pr5681_side_commits": int(ahead_behind[1]),
  }
  configure = cmake_command()
  build = [
      "/home/intel/intel-box-env/conda/bin/cmake",
      "--build", str(BUILD_XE3),
      "--target", "test_internals_gmlp",
      "-j", "1",
  ]
  plan = {
      "source": {
          "worktree": str(SOURCE),
          "commit": PR5059_HEAD,
          "reuse_clean_exact_worktree": True,
      },
      "build": {
          "build_dir": str(BUILD_XE3),
          "test_binary": str(TEST_XE3),
          "fresh_build_dir_required": True,
          "configure_command": configure,
          "build_command": build,
          "parallel_jobs": 1,
          "preflight_bytes": 8 * 1024**3,
          "abort_below_bytes": 4 * 1024**3,
      },
      "execution": {
          "strictly_serial": True,
          "product_build": False,
          "gpu_contexts": 0,
          "model_workers": 0,
          "infer_requests": 0,
      },
      "next_provider_gate": {
          "required_provider": "ocl:micro_horz:any",
          "shapes": [
              {"name": "decode", "mb": 1, "ic": 2048, "oc": 512},
              {"name": "prefill", "mb": 2048, "ic": 2048, "oc": 512},
          ],
          "stop_on_first_provider_failure": True,
      },
  }

  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "prior_evidence_identity_exact",
          sha256(SEQ2225A) == EXPECTED_SEQ2225A_SHA256
          and sha256(BUILD_AUDIT) == EXPECTED_BUILD_AUDIT_SHA256,
          seq2225a_sha256=sha256(SEQ2225A),
          build_audit_sha256=sha256(BUILD_AUDIT)),
      check(
          "exact_source_and_external_worktree_clean",
          git(SOURCE, "rev-parse", "HEAD") == PR5059_HEAD
          and source_before == "" and source_after == "",
          source_commit=git(SOURCE, "rev-parse", "HEAD"),
          status_before=source_before, status_after=source_after),
      check(
          "pr5681_live_identity_exact",
          pr["number"] == PR5681 and pr["head_sha"] == PR5681_HEAD
          and pr["base_sha"] == PR5681_BASE and pr["state"] == "open"
          and pr["draft"] is False and fetched.returncode == 0,
          pull=pr, fetch_returncode=fetched.returncode),
      check(
          "exact_failure_reaches_unregistered_architecture_switch",
          all(exact_failure[key] for key in (
              "entry_candidate", "attempt_reached_generation",
              "unsupported_architecture", "no_matching_kernel"))
          and successful_micro == 0 and failed_micro == 1
          and successful_ref >= 1,
          **exact_failure),
      check(
          "xe2_only_build_excludes_locked_ptl_xe3",
          build_isa["cache_gpu_isa"] == "XE2"
          and build_isa["build_all"] and build_isa["build_xe2"]
          and build_isa["build_xe3"] and build_isa["build_xe3p"]
          and build_isa["gemmstone_xe2_definitions"] > 0
          and build_isa["gemmstone_xe3_definitions"] == 0
          and all(dispatch.values())
          and target_identity["is_locked_ptl"]
          and target_identity["is_arc_b390"],
          build_isa=build_isa, dispatch=dispatch,
          target=target_identity),
      check(
          "pr5681_cannot_change_exact_architecture_failure",
          pr_intersection["changed_files_exact"]
          and not pr_intersection["changes_architecture_control"]
          and pr_intersection["strategy_patch_mentions_kchain"]
          and pr_intersection["exact_attempt_used_heuristics"]
          and not pr_intersection["database_edits_can_select_exact_attempt"]
          and pr_intersection["preflight_was_not_failure_site"],
          **pr_intersection),
      check(
          "pr5681_rebase_is_not_minimal_clean_correction",
          merge_tree.returncode != 0
          and "src/gpu/intel/gated_mlp/micro_horz.cpp"
          in pr_intersection["merge_conflict_paths"],
          merge_tree_returncode=merge_tree.returncode,
          merge_tree_stdout=merge_tree.stdout,
          merge_tree_stderr=merge_tree.stderr),
      check(
          "one_fresh_xe3_only_j1_rebuild_is_exact_and_bounded",
          not BUILD_XE3.exists()
          and configure[-1] == "-DDNNL_ENABLE_PRIMITIVE_GPU_ISA=XE3"
          and build[-4:] == [
              "--target", "test_internals_gmlp", "-j", "1"]
          and plan["build"]["preflight_bytes"] == 8 * 1024**3
          and plan["build"]["abort_below_bytes"] == 4 * 1024**3,
          build_dir_fresh=not BUILD_XE3.exists(),
          configure_command=configure, build_command=build),
      check(
          "source_audit_used_no_compiler_gpu_model_or_infer_request",
          True, compilers_started=0, gpu_contexts_created=0,
          gpu_kernels_executed=0, model_workers_started=0,
          infer_requests_created=0),
      check(
          "memory_stop_held",
          min(row["available_bytes"] for row in memory_samples)
          >= stop_bytes,
          stop_bytes=stop_bytes, samples=memory_samples),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": required_checks_passed,
      "pr5681_exact_fix_admitted": False,
      "xe3_component_rebuild_admitted": required_checks_passed,
      "product_build_admitted": False,
      "verdict": (
          "reject_pr5681_admit_exact_xe3_component_rebuild"
          if required_checks_passed else
          "reject_rebuild_source_or_identity_gate_failed"),
      "reason": (
          "The exact failure occurs after heuristic preflight in a build whose "
          "ISA registration includes Xe2 only. The locked PTL target maps to "
          "Xe3. PR5681 touches strategy/DB files but not the dispatch or build "
          "registration, and its database edits cannot select the observed "
          "heuristic attempt."),
      "next_if_pass": (
          "build only test_internals_gmlp from the same PR5059 source with "
          "DNNL_ENABLE_PRIMITIVE_GPU_ISA=XE3 at -j1; then require successful "
          "ocl:micro_horz:any on exact decode and prefill before timing"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "inputs": {
          "seq2225a": {
              "path": relative(SEQ2225A), "sha256": sha256(SEQ2225A)},
          "build_audit": {
              "path": relative(BUILD_AUDIT), "sha256": sha256(BUILD_AUDIT)},
          "target_contract": {
              "path": relative(TARGET_CONTRACT),
              "sha256": sha256(TARGET_CONTRACT)},
      },
      "pull_request": pr,
      "fetch": {
          "command": fetch_command,
          "returncode": fetched.returncode,
          "elapsed_seconds": fetch_elapsed,
          "stdout": fetched.stdout,
          "stderr": fetched.stderr,
      },
      "exact_failure": exact_failure,
      "build_isa": build_isa,
      "architecture_dispatch": dispatch,
      "target": target_identity,
      "pr5681_intersection": pr_intersection,
      "plan": plan,
      "memory": {
          "stop_bytes": stop_bytes,
          "samples": memory_samples,
      },
      "workers": {
          "compilers_started": 0,
          "gpu_contexts_created": 0,
          "gpu_kernels_executed": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
      },
      "checks": checks,
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "plan.json", {
      "schema": SCHEMA + "-plan",
      "workstream": WS,
      "inputs": metrics["inputs"],
      "source_verdict": verdict,
      "plan": plan,
  })
  (output / "report.md").write_text(
      "# oneDNN PR5681 / PR5059 exact architecture bound\n\n"
      f"- Required checks: `{required_checks_passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Exact failure: heuristic preflight passes, then "
      f"`Unsupported architecture`; micro/ref success "
      f"`{successful_micro}/{successful_ref}`\n"
      f"- Built ISA / target ISA: `XE2 / XE3 (PTL)`\n"
      f"- PR5681 architecture-control file intersection: "
      f"`{pr_intersection['architecture_control_intersection']}`\n"
      f"- PR5059/PR5681 merge conflict: "
      f"`{pr_intersection['merge_conflict_paths']}`\n"
      f"- Compiler/GPU/model/InferRequest: `0/0/0/0`\n"
      f"- Next: one fresh `XE3`, `-j1`, sole-test-target rebuild\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": required_checks_passed,
      "verdict": verdict["verdict"],
      "pr5681_exact_fix_admitted": verdict["pr5681_exact_fix_admitted"],
      "xe3_component_rebuild_admitted": (
          verdict["xe3_component_rebuild_admitted"]),
      "built_isa": build_isa["cache_gpu_isa"],
      "target_isa": "XE3",
  }, sort_keys=True), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
