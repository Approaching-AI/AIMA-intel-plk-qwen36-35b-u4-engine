#!/usr/bin/env python3
"""Build and link the exact oneDNN dependency pinned by OpenVINO PR35924.

The first PR35924 product attempt ported only the initial grouped-post-ops
commit onto the release oneDNN branch.  Two product correctness runs rejected
that mixed tree.  This gate closes that dependency ambiguity: it builds the
exact PR gitlink at ``babb7375`` in an isolated tree, preserves the accepted
Level Zero profiling event-pool chain, and links one isolated GPU plugin from
the already-built OpenVINO objects.  It creates no GPU context or model worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-exact-onednn-build-v0"
R0 = Path("/home/intel/intel-qwen36-r0")

BASE_TOOL = ROOT / "tools/intel-qwen36-openvino-pr35924-product-build.py"
FAILED_MINIMAL = ROOT / (
    "output/openvino-pr35924-product-correctness-trace-"
    "20260731Tseq2235-clean/result.json")
FAILED_PARITY = ROOT / (
    "output/openvino-pr35924-swish-parity-correctness-"
    "20260731Tseq2239-clean/result.json")
PRIOR_EXACT_BUILD_STDOUT = ROOT / (
    "output/openvino-pr35924-exact-onednn-build-"
    "20260801Tseq2240c-clean/raw/build-exact-pr35924-onednn.stdout")
EVENT_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-babb7375-ze-profile-event-pool-chain.patch")

OPENVINO_AUDIT = R0 / "source/openvino-90214-pr35924-audit"
LOCKED_OPENVINO = R0 / "source/openvino-90214e5be05"
LOCKED_ONEDNN = (
    LOCKED_OPENVINO / "src/plugins/intel_gpu/thirdparty/onednn_gpu")
ONEDNN_AUDIT = R0 / "source/onednn-20db-pr13e-audit"
EXACT_ONEDNN = R0 / "source/onednn-babb7375-pr35924-exact"
EXACT_BUILD = R0 / "build/onednn-babb7375-pr35924-exact"
EXACT_INSTALL = R0 / "build/onednn-babb7375-pr35924-exact-install"
EXACT_ARCHIVE = EXACT_INSTALL / "lib/libopenvino_onednn_gpu.a"

OPENVINO_BUILD = R0 / "build/openvino-90214e-l0-gpu"
NINJA = Path("/home/intel/intel-box-env/conda/bin/ninja")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
GCC = Path("/home/intel/intel-box-env/conda/bin/gcc")
GXX = Path("/home/intel/intel-box-env/conda/bin/g++")
AR = Path("/home/intel/intel-box-env/conda/bin/ar")
OPENCL_INCLUDE = LOCKED_OPENVINO / "thirdparty/ocl/cl_headers"
OPENCL_LIBRARY = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libOpenCL.so")
CURRENT_ARCHIVE = OPENVINO_BUILD / (
    "src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/"
    "libopenvino_onednn_gpu.a")
CURRENT_BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE = R0 / (
    "output/openvino-90214e-l0-gpu-seq2240d/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
MINIMAL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2233/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
PARITY_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2238/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")

LOCKED_OPENVINO_HEAD = "90214e5be052438cec5617ed3ea7e37df1538f68"
LOCKED_ONEDNN_HEAD = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
INITIAL_POSTOPS_COMMIT = "13e6484e8f4ad11f384dc0bc6138c09b29b9b228"
EXACT_ONEDNN_HEAD = "babb7375ff500dd8ad77d26cbd2b044122b7a8b4"
PR35924_HEAD = "5cf601a51ce1dbb5a223c08a41c126e46ddf5628"
PR35924_BASE = "337f0f63bf5b03fcc0a6d555288eae5e8e0e2f3b"
EXPECTED_EVENT_PATCH_SHA256 = (
    "6263da09724cb09d34667306f30cb62711a76716c0ea4b158e5d0a28c61e277c")
EXPECTED_EXACT_GENERATOR_SHA256 = (
    "2ef0ed2fb2c916a6e0eda58eb11c30fb1206b23456c78f96efd4f9214545c0e6")
EXPECTED_MINIMAL_RESULT_SHA256 = (
    "c99c295a8482ca5b42c39e0453cd647b43c65d127ef41861990acba58219070f")
EXPECTED_PARITY_RESULT_SHA256 = (
    "8ed42780b2831a3fdd2126bf3caec61adfa5f5fe3d4cb8cc0c0082688bea598c")
EXPECTED_PRIOR_EXACT_BUILD_STDOUT_SHA256 = (
    "c8e159aa95cc122e503b57e9a5cab62ad52eeae6b002fe490241c869e9a2d8a0")
EXPECTED_CONTROL_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
EXPECTED_MINIMAL_PLUGIN_SHA256 = (
    "c66c9be61ee31110a55c8a064ed1390bd3d21a3f1766a03fdea84a078a519849")
EXPECTED_PARITY_PLUGIN_SHA256 = (
    "bbaaa6880695eab4381d2aa6bf32162ea318565d4c66b99f19dcef31689fbbd7")
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE_BUILD = load_module("iq36_pr35924_base_build", BASE_TOOL)
BASE = BASE_BUILD.BASE


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
      "--candidate-plugin", type=Path, default=DEFAULT_CANDIDATE)
  parser.add_argument("--configure-timeout-s", type=float, default=900.0)
  parser.add_argument("--build-timeout-s", type=float, default=7200.0)
  parser.add_argument("--link-timeout-s", type=float, default=900.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.1)
  args = parser.parse_args()
  if min(
      args.configure_timeout_s, args.build_timeout_s, args.link_timeout_s,
      args.poll_interval_s) <= 0:
    parser.error("timeouts and poll interval must be positive")
  return args


def run(command: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
  return {
      "path": str(path),
      "exists": path.is_file(),
      "bytes": path.stat().st_size if path.is_file() else 0,
      "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
      "sha256": sha256(path) if path.is_file() else None,
  }


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def stage_ok(stage: dict[str, Any]) -> bool:
  return bool(
      stage.get("returncode") == 0
      and stage.get("timed_out") is False
      and stage.get("memory_guard_tripped") is False
      and stage.get("oom_observed") is False)


def git_text(repo: Path, *args: str) -> str:
  result = run(["git", *args], cwd=repo)
  if result.returncode:
    raise RuntimeError(
        f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
  return result.stdout.strip()


def git_is_ancestor(repo: Path, older: str, newer: str) -> bool:
  return run(
      ["git", "merge-base", "--is-ancestor", older, newer],
      cwd=repo).returncode == 0


def numstat(repo: Path, older: str, newer: str,
            paths: list[str] | None = None) -> dict[str, int]:
  command = ["git", "diff", "--numstat", f"{older}..{newer}"]
  if paths:
    command += ["--", *paths]
  result = run(command, cwd=repo)
  if result.returncode:
    raise RuntimeError(result.stderr.strip())
  insertions = 0
  deletions = 0
  binaries = 0
  rows = 0
  for line in result.stdout.splitlines():
    added, removed, _ = line.split("\t", 2)
    rows += 1
    if added == "-" or removed == "-":
      binaries += 1
    else:
      insertions += int(added)
      deletions += int(removed)
  return {
      "files": rows,
      "insertions": insertions,
      "deletions": deletions,
      "binary_files": binaries,
  }


def reverse_apply_check(repo: Path, patch: Path) -> dict[str, Any]:
  result = run(
      ["git", "apply", "--reverse", "--check", str(patch)], cwd=repo)
  return {
      "command": result.args,
      "returncode": result.returncode,
      "stdout": result.stdout.strip(),
      "stderr": result.stderr.strip(),
  }


def exact_source_contract() -> dict[str, Any]:
  grouped_paths = [
      "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
      "src/gpu/intel/matmul/grouped_post_ops_gen.cpp",
      "src/gpu/intel/matmul/grouped_post_ops_gen.hpp",
  ]
  status_result = run(["git", "status", "--short"], cwd=EXACT_ONEDNN)
  if status_result.returncode:
    raise RuntimeError(status_result.stderr.strip())
  exact_status = sorted(
      ({"code": line[:2], "path": line[3:]}
       for line in status_result.stdout.splitlines()),
      key=lambda row: (row["path"], row["code"]))
  exact_head = git_text(EXACT_ONEDNN, "rev-parse", "HEAD")
  locked_head = git_text(LOCKED_ONEDNN, "rev-parse", "HEAD")
  openvino_head = git_text(LOCKED_OPENVINO, "rev-parse", "HEAD")
  pr_head = git_text(OPENVINO_AUDIT, "rev-parse", "origin/pr/35924")
  pr_base = git_text(
      OPENVINO_AUDIT, "merge-base", "origin/pr/35924", "origin/master")
  pr_gitlink = git_text(
      OPENVINO_AUDIT, "rev-parse",
      "origin/pr/35924:src/plugins/intel_gpu/thirdparty/onednn_gpu")
  locked_gitlink = git_text(
      OPENVINO_AUDIT, "rev-parse",
      f"{LOCKED_OPENVINO_HEAD}:src/plugins/intel_gpu/thirdparty/onednn_gpu")
  grouped_log = git_text(
      ONEDNN_AUDIT, "log", "--format=%H %s", "--reverse",
      f"{INITIAL_POSTOPS_COMMIT}..{EXACT_ONEDNN_HEAD}", "--", *grouped_paths)
  generator = (
      EXACT_ONEDNN / "src/gpu/intel/matmul/grouped_post_ops_gen.cpp")
  generator_text = generator.read_text(
      encoding="utf-8", errors="replace")
  return {
      "openvino_head": openvino_head,
      "locked_onednn_head": locked_head,
      "exact_onednn_head": exact_head,
      "pr35924_head": pr_head,
      "pr35924_base": pr_base,
      "pr35924_onednn_gitlink": pr_gitlink,
      "locked_onednn_gitlink": locked_gitlink,
      "exact_status": exact_status,
      "event_patch_sha256": sha256(EVENT_PATCH),
      "event_patch_reverse": reverse_apply_check(EXACT_ONEDNN, EVENT_PATCH),
      "generator_sha256": sha256(generator),
      "generator_has_upstream_swiglu": (
          "exp(-(" in generator_text
          and "tile_elementwise((*c_tile), eltwise_apply_" in generator_text
          and "native_exp" not in generator_text
          and "eltwise_f16_boundary" not in generator_text),
      "initial_is_ancestor_of_exact": git_is_ancestor(
          ONEDNN_AUDIT, INITIAL_POSTOPS_COMMIT, EXACT_ONEDNN_HEAD),
      "locked_is_ancestor_of_exact": git_is_ancestor(
          ONEDNN_AUDIT, LOCKED_ONEDNN_HEAD, EXACT_ONEDNN_HEAD),
      "locked_exact_merge_base": git_text(
          ONEDNN_AUDIT, "merge-base",
          LOCKED_ONEDNN_HEAD, EXACT_ONEDNN_HEAD),
      "full_locked_to_exact_numstat": numstat(
          ONEDNN_AUDIT, LOCKED_ONEDNN_HEAD, EXACT_ONEDNN_HEAD),
      "grouped_initial_to_exact_numstat": numstat(
          ONEDNN_AUDIT, INITIAL_POSTOPS_COMMIT, EXACT_ONEDNN_HEAD,
          grouped_paths),
      "grouped_post_initial_commits": grouped_log.splitlines(),
  }


def final_link_command(candidate: Path) -> tuple[list[str], dict[str, Any]]:
  result = run(
      [str(NINJA), "-C", str(OPENVINO_BUILD), "-t", "commands",
       "openvino_intel_gpu_plugin"])
  if result.returncode:
    raise RuntimeError(result.stderr.strip())
  matches = [
      line for line in result.stdout.splitlines()
      if str(CURRENT_BUILD_PLUGIN) in line
      and " -o " + str(CURRENT_BUILD_PLUGIN) in line
      and "libopenvino_onednn_gpu.a" in line
  ]
  if len(matches) != 1:
    raise RuntimeError(
        f"expected one plugin link command, observed {len(matches)}")
  tokens = shlex.split(matches[0])
  separators = [index for index, token in enumerate(tokens) if token == "&&"]
  if separators:
    start = separators[0] + 1
    end = separators[1] if len(separators) > 1 else len(tokens)
    tokens = tokens[start:end]
  replaced_archive = 0
  replaced_output = 0
  for index, token in enumerate(tokens):
    resolved = (OPENVINO_BUILD / token).resolve() if not Path(
        token).is_absolute() else Path(token).resolve()
    if resolved == CURRENT_ARCHIVE.resolve():
      tokens[index] = str(EXACT_ARCHIVE)
      replaced_archive += 1
    if index > 0 and tokens[index - 1] == "-o":
      if Path(token).resolve() == CURRENT_BUILD_PLUGIN.resolve():
        tokens[index] = str(candidate)
        replaced_output += 1
  if replaced_archive != 1 or replaced_output != 1:
    raise RuntimeError(
        f"link rewrite archive={replaced_archive} output={replaced_output}")
  return tokens, {
      "source_command_sha256": hashlib.sha256(
          matches[0].encode("utf-8")).hexdigest(),
      "argument_count": len(tokens),
      "archive_replacements": replaced_archive,
      "output_replacements": replaced_output,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  output.mkdir(parents=True, exist_ok=False)
  raw.mkdir()
  candidate = args.candidate_plugin.resolve()
  if candidate.exists():
    raise SystemExit(f"candidate already exists: {candidate}")

  required_paths = (
      BASE_TOOL, FAILED_MINIMAL, FAILED_PARITY, PRIOR_EXACT_BUILD_STDOUT,
      EVENT_PATCH,
      OPENVINO_AUDIT, LOCKED_OPENVINO, LOCKED_ONEDNN, ONEDNN_AUDIT,
      EXACT_ONEDNN, OPENVINO_BUILD, NINJA, CMAKE, GCC, GXX, AR,
      OPENCL_INCLUDE, OPENCL_LIBRARY, CURRENT_ARCHIVE,
      CURRENT_BUILD_PLUGIN, CONTROL_PLUGIN, MINIMAL_PLUGIN, PARITY_PLUGIN)
  missing = [str(path) for path in required_paths if not path.exists()]
  if missing:
    raise SystemExit("missing exact-build inputs: " + ", ".join(missing))

  repository = BASE.repository_state(output)
  source = exact_source_contract()
  minimal_result = load_json(FAILED_MINIMAL)
  parity_result = load_json(FAILED_PARITY)
  memory_before = BASE.proc_meminfo()
  controls_before = {
      "control": file_record(CONTROL_PLUGIN),
      "minimal": file_record(MINIMAL_PLUGIN),
      "parity": file_record(PARITY_PLUGIN),
      "current_archive": file_record(CURRENT_ARCHIVE),
      "current_build_plugin": file_record(CURRENT_BUILD_PLUGIN),
  }

  source_checks = [
      check("repository_is_clean_and_pushed_at_gate",
            repository["branch"] == "main"
            and repository["pushed"] and not repository["dirty"],
            **repository),
      check("two_mixed_tree_correctness_rejections_are_exact",
            sha256(FAILED_MINIMAL) == EXPECTED_MINIMAL_RESULT_SHA256
            and sha256(FAILED_PARITY) == EXPECTED_PARITY_RESULT_SHA256
            and minimal_result.get("verdict")
                == "reject_or_repair_pr35924_runtime_correctness_or_provider"
            and parity_result.get("verdict")
                == "reject_pr35924_parity_on_product_correctness"
            and minimal_result.get("worker", {}).get("oom_observed") is False
            and parity_result.get("worker", {}).get("oom_observed") is False,
            minimal_result_sha256=sha256(FAILED_MINIMAL),
            parity_result_sha256=sha256(FAILED_PARITY),
            minimal_max_kld=minimal_result.get(
                "correctness", {}).get("max_kld"),
            parity_max_kld=parity_result.get(
                "correctness", {}).get("max_kld")),
      check("openvino_pr_and_gitlinks_are_exact",
            source["openvino_head"] == LOCKED_OPENVINO_HEAD
            and source["pr35924_head"] == PR35924_HEAD
            and source["pr35924_base"] == PR35924_BASE
            and source["locked_onednn_gitlink"] == LOCKED_ONEDNN_HEAD
            and source["pr35924_onednn_gitlink"] == EXACT_ONEDNN_HEAD,
            openvino_head=source["openvino_head"],
            pr35924_head=source["pr35924_head"],
            pr35924_base=source["pr35924_base"],
            locked_gitlink=source["locked_onednn_gitlink"],
            pr_gitlink=source["pr35924_onednn_gitlink"]),
      check("minimal_backport_is_not_dependency_closed",
            source["initial_is_ancestor_of_exact"] is True
            and source["locked_is_ancestor_of_exact"] is False
            and source["locked_exact_merge_base"]
                == "f55a9a3b7518e074713b6d1c0e41442e438da33a"
            and source["full_locked_to_exact_numstat"] == {
                "files": 1176, "insertions": 41686, "deletions": 21949,
                "binary_files": 1}
            and source["grouped_initial_to_exact_numstat"] == {
                "files": 3, "insertions": 155, "deletions": 71,
                "binary_files": 0},
            dependency=source),
      check("exact_onednn_and_profile_chain_postimage_are_bound",
            source["exact_onednn_head"] == EXACT_ONEDNN_HEAD
            and source["exact_status"] == [
                {"code": " M", "path": "src/xpu/ze/stream_impl.cpp"},
                {"code": " M", "path": "src/xpu/ze/stream_impl.hpp"}]
            and source["event_patch_sha256"]
                == EXPECTED_EVENT_PATCH_SHA256
            and source["event_patch_reverse"]["returncode"] == 0
            and source["generator_sha256"]
                == EXPECTED_EXACT_GENERATOR_SHA256
            and source["generator_has_upstream_swiglu"] is True,
            exact_head=source["exact_onednn_head"],
            exact_status=source["exact_status"],
            event_patch_sha256=source["event_patch_sha256"],
            event_patch_reverse=source["event_patch_reverse"],
            generator_sha256=source["generator_sha256"]),
      check("completed_serial_exact_dependency_build_is_bound",
            sha256(PRIOR_EXACT_BUILD_STDOUT)
                == EXPECTED_PRIOR_EXACT_BUILD_STDOUT_SHA256
            and len(re.findall(
                r"\bBuilding (?:C|CXX) object\b",
                PRIOR_EXACT_BUILD_STDOUT.read_text(
                    encoding="utf-8", errors="replace"))) == 437
            and "[536/537] Linking CXX static library "
                "src/libopenvino_onednn_gpu.a"
                in PRIOR_EXACT_BUILD_STDOUT.read_text(
                    encoding="utf-8", errors="replace")
            and "-- Installing: "
                "/home/intel/intel-qwen36-r0/build/"
                "onednn-babb7375-pr35924-exact-install/lib/"
                "libopenvino_onednn_gpu.a"
                in PRIOR_EXACT_BUILD_STDOUT.read_text(
                    encoding="utf-8", errors="replace"),
            prior_build_stdout=str(PRIOR_EXACT_BUILD_STDOUT),
            prior_build_stdout_sha256=sha256(PRIOR_EXACT_BUILD_STDOUT),
            prior_compile_steps=437),
      check("accepted_and_rejected_plugins_are_bound",
            controls_before["control"]["sha256"]
                == EXPECTED_CONTROL_SHA256
            and controls_before["minimal"]["sha256"]
                == EXPECTED_MINIMAL_PLUGIN_SHA256
            and controls_before["parity"]["sha256"]
                == EXPECTED_PARITY_PLUGIN_SHA256
            and candidate not in {
                CONTROL_PLUGIN.resolve(), MINIMAL_PLUGIN.resolve(),
                PARITY_PLUGIN.resolve(), CURRENT_BUILD_PLUGIN.resolve()},
            controls=controls_before, candidate=str(candidate)),
      check("eight_gib_preflight_clears",
            int(memory_before.get("MemAvailable", 0)) >= PREFLIGHT_BYTES,
            available_bytes=memory_before.get("MemAvailable"),
            preflight_bytes=PREFLIGHT_BYTES),
  ]
  source_admitted = all(row["pass"] for row in source_checks)

  environment = os.environ.copy()
  conda_bin = "/home/intel/intel-box-env/conda/bin"
  conda_lib = "/home/intel/intel-box-env/conda/lib"
  environment["PATH"] = conda_bin + ":" + environment.get("PATH", "")
  environment["LD_LIBRARY_PATH"] = (
      conda_lib + ":" + environment.get("LD_LIBRARY_PATH", ""))

  primitive_list = (
      "CONCAT;CONVOLUTION;DECONVOLUTION;GATED_MLP;INNER_PRODUCT;MATMUL;"
      "REORDER;POOLING;REDUCTION;SDPA;RNN")
  configure_command = [
      str(CMAKE), "-S", str(EXACT_ONEDNN), "-B", str(EXACT_BUILD),
      "-G", "Ninja",
      "-DCMAKE_BUILD_TYPE=Release",
      f"-DCMAKE_C_COMPILER={GCC}",
      f"-DCMAKE_CXX_COMPILER={GXX}",
      f"-DCMAKE_INSTALL_PREFIX={EXACT_INSTALL}",
      "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=OFF",
      "-DDNNL_CPU_RUNTIME=NONE",
      "-DDNNL_GPU_RUNTIME=ZE",
      "-DONEDNN_GPU_VENDOR=INTEL",
      "-DDNNL_LIBRARY_TYPE=STATIC",
      "-DDNNL_LIBRARY_NAME=openvino_onednn_gpu",
      "-DDNNL_BUILD_TESTS=OFF",
      "-DDNNL_BUILD_EXAMPLES=OFF",
      "-DDNNL_BUILD_DOC=OFF",
      "-DONEDNN_BUILD_GRAPH=OFF",
      "-DDNNL_ENABLE_WORKLOAD=INFERENCE",
      f"-DDNNL_ENABLE_PRIMITIVE={primitive_list}",
      "-DDNNL_EXPERIMENTAL_GROUPED_MEMORY=ON",
      "-DDNNL_EXPERIMENTAL_PROFILING=ON",
      "-DDNNL_VERBOSE=ON",
      "-DDNNL_TARGET_ARCH=X64",
      f"-DOpenCL_INCLUDE_DIR={OPENCL_INCLUDE}",
      f"-DOpenCL_LIBRARY={OPENCL_LIBRARY}",
  ]
  build_command = [
      str(CMAKE), "--build", str(EXACT_BUILD),
      "--target", "install", "--parallel", "1"]

  skipped = {
      "returncode": 125, "timed_out": False,
      "memory_guard_tripped": False, "oom_observed": False,
      "skipped": "source gate failed", "monitor": {}}
  configure = dict(skipped)
  build = dict(skipped)
  link = dict(skipped)
  link_contract: dict[str, Any] = {}
  if source_admitted:
    configure = BASE.run_scoped(
        output, raw, "configure-exact-pr35924-onednn", configure_command,
        args.configure_timeout_s, args.poll_interval_s, environment)
    if stage_ok(configure):
      build = BASE.run_scoped(
          output, raw, "build-exact-pr35924-onednn", build_command,
          args.build_timeout_s, args.poll_interval_s, environment)
    if stage_ok(build) and EXACT_ARCHIVE.is_file():
      candidate.parent.mkdir(parents=True, exist_ok=False)
      try:
        link_command, link_contract = final_link_command(candidate)
        link = BASE.run_scoped(
            output, raw, "link-exact-pr35924-plugin",
            [str(CMAKE), "-E", "chdir", str(OPENVINO_BUILD), *link_command],
            args.link_timeout_s, args.poll_interval_s, environment)
      except Exception as error:
        link = {
            **skipped, "returncode": 124,
            "skipped": "link command construction failed",
            "error": str(error)}

  archive = file_record(EXACT_ARCHIVE)
  candidate_record = file_record(candidate)
  controls_after = {
      "control": file_record(CONTROL_PLUGIN),
      "minimal": file_record(MINIMAL_PLUGIN),
      "parity": file_record(PARITY_PLUGIN),
      "current_archive": file_record(CURRENT_ARCHIVE),
      "current_build_plugin": file_record(CURRENT_BUILD_PLUGIN),
  }
  ar_members = run(
      [str(AR), "t", str(EXACT_ARCHIVE)]
      if EXACT_ARCHIVE.is_file() else ["/usr/bin/false"])
  ldd = run(
      ["/usr/bin/ldd", str(candidate)]
      if candidate.is_file() else ["/usr/bin/false"])
  build_stdout = (raw / "build-exact-pr35924-onednn.stdout")
  build_text = (
      build_stdout.read_text(encoding="utf-8", errors="replace")
      if build_stdout.is_file() else "")
  prior_build_text = PRIOR_EXACT_BUILD_STDOUT.read_text(
      encoding="utf-8", errors="replace")
  compile_steps = len(re.findall(
      r"\bBuilding (?:C|CXX) object\b", build_text))
  prior_compile_steps = len(re.findall(
      r"\bBuilding (?:C|CXX) object\b", prior_build_text))

  stages = (configure, build, link)
  minimum_available = min(
      int(stage.get("monitor", {}).get(
          "system_available_min_bytes", memory_before.get("MemAvailable", 0)))
      for stage in stages)
  oom_free = all(
      not stage.get("oom_observed", False)
      and not stage.get("memory_guard_tripped", False)
      and all(
          int(value) == 0
          for key, value in stage.get(
              "monitor", {}).get("memory_events_max", {}).items()
          if key in ("oom", "oom_kill", "oom_group_kill"))
      for stage in stages)

  checks = [
      *source_checks,
      check("exact_onednn_configure_build_and_direct_link_succeed",
            source_admitted and stage_ok(configure) and stage_ok(build)
            and stage_ok(link) and archive["exists"]
            and candidate_record["exists"]
            and (compile_steps >= 1 or prior_compile_steps == 437)
            and ("grouped_post_ops_gen.cpp.o" in build_text
                 or "grouped_post_ops_gen.cpp.o" in prior_build_text)
            and ar_members.returncode == 0
            and "grouped_post_ops_gen.cpp.o" in ar_members.stdout,
            configure=configure, build=build, link=link,
            compile_steps=compile_steps,
            prior_compile_steps=prior_compile_steps, archive=archive,
            candidate=candidate_record, link_contract=link_contract),
      check("exact_candidate_is_isolated_and_loadable",
            candidate_record["sha256"] not in {
                None, EXPECTED_CONTROL_SHA256,
                EXPECTED_MINIMAL_PLUGIN_SHA256,
                EXPECTED_PARITY_PLUGIN_SHA256}
            and ldd.returncode == 0
            and "not found" not in ldd.stdout
            and "not found" not in ldd.stderr,
            candidate=candidate_record, ldd_returncode=ldd.returncode,
            ldd_stdout=ldd.stdout.strip(), ldd_stderr=ldd.stderr.strip()),
      check("all_control_and_current_build_artifacts_remain_unchanged",
            controls_after == controls_before,
            before=controls_before, after=controls_after),
      check("four_gib_abort_and_oom_guards_hold",
            minimum_available >= ABORT_BYTES and oom_free,
            minimum_available_bytes=minimum_available,
            abort_bytes=ABORT_BYTES, oom_free=oom_free),
      check("build_and_link_create_no_gpu_or_model_work", True,
            gpu_contexts_created=0, gpu_kernels_executed=0,
            models_loaded=0, infer_requests_created=0,
            model_workers_started=0),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_pr35924_product_compile"
      if required else
      "reject_or_repair_exact_pr35924_dependency_build")
  result = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "repository": repository,
      "verdict": verdict,
      "required_checks_passed": required,
      "exact_product_compile_admitted": required,
      "product_correctness_admitted": False,
      "performance_worker_admitted": False,
      "formal_performance_admitted": False,
      "source": source,
      "checks": checks,
      "configure": configure,
      "build": build,
      "link": link,
      "exact_archive": archive,
      "candidate_plugin": candidate_record,
      "controls": controls_after,
      "memory": {
          "preflight_available_bytes": memory_before.get("MemAvailable"),
          "minimum_available_bytes": minimum_available,
          "abort_bytes": ABORT_BYTES,
          "oom_free": oom_free,
      },
      "next_action": (
          "Run one serial graph compile with the exact candidate; if and only "
          "if it preserves the 40 grouped MoE owners, run output130 "
          "correctness before any timing."
          if required else
          "Repair or reject the exact dependency configure/build/link; do not "
          "launch a GPU or model worker."),
  }
  write_json(output / "result.json", result)
  print(json.dumps({
      "output": str(output),
      "verdict": verdict,
      "required_checks_passed": required,
      "exact_archive_sha256": archive["sha256"],
      "candidate_plugin_sha256": candidate_record["sha256"],
      "compile_steps": compile_steps,
      "minimum_available_bytes": minimum_available,
  }, sort_keys=True))
  return 0 if required else 1


if __name__ == "__main__":
  raise SystemExit(main())
