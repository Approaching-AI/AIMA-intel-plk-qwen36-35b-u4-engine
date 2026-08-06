#!/usr/bin/env python3
"""Build the source-bound PR35924 grouped-postops candidate plugin.

The source preparation is deliberately external to this tool: the durable
oneDNN Swish-postops backport and exact OpenVINO PR35924 patch must already be
materialized.  This gate binds those exact postimages, regenerates the existing
build once, and builds only ``openvino_intel_gpu_plugin`` at parallelism one.
It creates no GPU context, model worker, InferRequest, or inference worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-product-build-v0"
BASE_TOOL = ROOT / "tools/intel-qwen36-onednn-gmlp-exact-component-build.py"
ADMISSION = ROOT / (
    "output/openvino-pr35924-grouped-postops-bound-"
    "20260731Tseq2231-clean/metrics.json")
EXPECTED_ADMISSION_SHA256 = (
    "b75181257a81b124e2309cbb7baebd309242fcd65a7338d4e3084aae88583258")

R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
ONEDNN = SOURCE_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu"
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
ONEDNN_BUILD = (
    BUILD_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu_build")
ONEDNN_INSTALL_LIB = (
    BUILD_TREE / "src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/"
    "libopenvino_onednn_gpu.a")
BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2233/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
NM = Path("/home/intel/intel-box-env/conda/bin/nm")

ONEDNN_PATCH = (
    ROOT / "engine/openvino/iq36-onednn-grouped-postops-swish.patch")
OPENVINO_PATCH = (
    ROOT / "engine/openvino/iq36-pr35924-grouped-postops.patch")
EXPECTED_PATCH_SHA256 = {
    ONEDNN_PATCH: (
        "732a0c75bb5622e58683db070e09029ed7278a7c7993633e8f1df27e8c047a9a"),
    OPENVINO_PATCH: (
        "6f205f856a0118c0a43bb7914131a3f8edee148f279c9e2b0e7cf967ca8c8350"),
}
OPENVINO_HEAD = "90214e5be052438cec5617ed3ea7e37df1538f68"
ONEDNN_HEAD = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
ONEDNN_ORIGIN_COMMIT = "13e6484e8f4ad11f384dc0bc6138c09b29b9b228"

MOE_REL = (
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/"
    "moe_3gemm_swiglu_opt.cpp")
ONEDNN_RELS = (
    "src/gpu/intel/matmul/grouped_micro_gemm.cl",
    "src/gpu/intel/matmul/grouped_micro_gemm.cpp",
    "src/gpu/intel/matmul/grouped_micro_gemm.hpp",
    "src/gpu/intel/matmul/grouped_post_ops_gen.cpp",
    "src/gpu/intel/matmul/grouped_post_ops_gen.hpp",
)
EXPECTED_POST_SHA256 = {
    ONEDNN / ONEDNN_RELS[0]: (
        "30ba346c84d2f96074f6eec18f597376636c1e660718e7caffb5fd64ea76361a"),
    ONEDNN / ONEDNN_RELS[1]: (
        "9dfd92ed5f081e4704561c9b9586f4c551fd7b8e552c65dab16024f55ab0e83c"),
    ONEDNN / ONEDNN_RELS[2]: (
        "2fcdca92efa5be500d77c03f9d5780e7972a7322bad3492865e2bc1f77b6a4f9"),
    ONEDNN / ONEDNN_RELS[3]: (
        "a14ddbccf6090a95df81ba2a7d56fdea1cef2db1d1b89ea997186abcfe632694"),
    ONEDNN / ONEDNN_RELS[4]: (
        "6469ac21ebeb56c82faa7c60bb472ebf5e3bb7604d2759b49fa99c1265e53877"),
    SOURCE_TREE / MOE_REL: (
        "f43628afcc8b244760c7edb64ae5d57a73a84fefb2f327fa76bc59617158b137"),
}
EXPECTED_CONTROL_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")
PREFLIGHT_BYTES = 8 * 1024**3
ABORT_BYTES = 4 * 1024**3


def load_base() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_scoped_build_base", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import scoped build helper: {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_base()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
      "--candidate-plugin", type=Path, default=DEFAULT_CANDIDATE_PLUGIN)
  parser.add_argument("--configure-timeout-s", default=600.0, type=float)
  parser.add_argument("--build-timeout-s", default=3600.0, type=float)
  parser.add_argument("--poll-interval-s", default=0.1, type=float)
  args = parser.parse_args()
  if (args.configure_timeout_s <= 0 or args.build_timeout_s <= 0
      or args.poll_interval_s <= 0):
    parser.error("timeouts and poll interval must be positive")
  return args


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
    command: list[str], cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")


def git(cwd: Path, *args: str) -> str:
  result = run(["git", *args], cwd=cwd)
  if result.returncode != 0:
    raise RuntimeError(
        f"git failed ({result.returncode}): {args}\n{result.stderr}")
  return result.stdout.strip()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def file_record(path: Path) -> dict[str, Any]:
  return {
      "path": str(path),
      "exists": path.is_file(),
      "bytes": path.stat().st_size if path.is_file() else 0,
      "sha256": BASE.sha256(path) if path.is_file() else None,
      "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
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


def target_status(repo: Path, paths: tuple[str, ...]) -> list[str]:
  output = git(
      repo, "status", "--short", "--untracked-files=all", "--", *paths)
  return output.splitlines() if output else []


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


def source_contract() -> dict[str, Any]:
  onednn_files = {str(path): file_record(path)
                  for path in EXPECTED_POST_SHA256 if path.is_relative_to(ONEDNN)}
  moe = file_record(SOURCE_TREE / MOE_REL)
  grouped_cpp = (ONEDNN / ONEDNN_RELS[1]).read_text(
      encoding="utf-8", errors="replace")
  grouped_cl = (ONEDNN / ONEDNN_RELS[0]).read_text(
      encoding="utf-8", errors="replace")
  postops_cpp = (ONEDNN / ONEDNN_RELS[3]).read_text(
      encoding="utf-8", errors="replace")
  moe_text = (SOURCE_TREE / MOE_REL).read_text(
      encoding="utf-8", errors="replace")
  grouped_prefill = function_body(
      moe_text, "cldnn::event::ptr exec_prefill_grouped_gemm")
  return {
      "openvino_head": git(SOURCE_TREE, "rev-parse", "HEAD"),
      "onednn_head": git(ONEDNN, "rev-parse", "HEAD"),
      "onednn_origin_commit_present": (
          run(["git", "cat-file", "-e", f"{ONEDNN_ORIGIN_COMMIT}^{{commit}}"],
              cwd=ONEDNN).returncode == 0),
      "patches": {
          str(path): file_record(path) for path in EXPECTED_PATCH_SHA256},
      "postimages": {**onednn_files, str(SOURCE_TREE / MOE_REL): moe},
      "onednn_target_status": target_status(ONEDNN, ONEDNN_RELS),
      "openvino_target_status": target_status(SOURCE_TREE, (MOE_REL,)),
      "onednn_reverse_apply": reverse_apply_check(ONEDNN, ONEDNN_PATCH),
      "openvino_reverse_apply": reverse_apply_check(
          SOURCE_TREE, OPENVINO_PATCH),
      "onednn_postops_exact": (
          "with_post_op_ = !attr()->post_ops_.has_default_values()"
          in grouped_cpp
          and "generate_post_ops_microgemm_header" in grouped_cpp
          and "apply_post_ops_chain(&c_tile" in grouped_cl
          and "e.eltwise.alg == alg_kind::eltwise_swish" in postops_cpp
          and "po_kind_t::binary_grouped_scale" in postops_cpp),
      "openvino_fusion_exact": (
          "gate_po.append_eltwise" in moe_text
          and "gate_po.append_binary" in moe_text
          and "DNNL_ARG_ATTR_MULTIPLE_POST_OP" in grouped_prefill
          and "*grouped_gemm_prefill_swiglu" not in grouped_prefill
          and grouped_prefill.find("gk.up_prim.execute")
          < grouped_prefill.find("gk.gate_prim.execute")),
      "decode_branch_exact": (
          moe_text.count("if (token_num == 1)") == 1
          and moe_text.count(
              "return exec_single_token({topk_event}, instance, scratch);")
          == 1),
  }


def source_hashes_exact(contract: dict[str, Any]) -> bool:
  for path, expected in EXPECTED_POST_SHA256.items():
    row = contract["postimages"].get(str(path), {})
    if row.get("sha256") != expected:
      return False
  return True


def patch_hashes_exact(contract: dict[str, Any]) -> bool:
  for path, expected in EXPECTED_PATCH_SHA256.items():
    row = contract["patches"].get(str(path), {})
    if row.get("sha256") != expected:
      return False
  return True


def stage_ok(stage: dict[str, Any]) -> bool:
  return bool(
      stage.get("returncode") == 0
      and not stage.get("timed_out", False)
      and not stage.get("memory_guard_tripped", False)
      and not stage.get("oom_observed", False))


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  raw = output / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  candidate_plugin = args.candidate_plugin.resolve()
  if candidate_plugin.exists():
    raise SystemExit(
        f"isolated candidate plugin already exists: {candidate_plugin}")

  required = (
      BASE_TOOL, ADMISSION, SOURCE_TREE, ONEDNN, BUILD_TREE, BUILD_PLUGIN,
      ONEDNN_BUILD, ONEDNN_INSTALL_LIB, CONTROL_PLUGIN, CMAKE, NM,
      ONEDNN_PATCH, OPENVINO_PATCH,
      *EXPECTED_POST_SHA256)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing PR35924 build inputs: " + ", ".join(missing))

  repo = BASE.repository_state(output)
  admission = load_json(ADMISSION)
  source_before = source_contract()
  onednn_library_before = file_record(ONEDNN_INSTALL_LIB)
  build_before = file_record(BUILD_PLUGIN)
  control_before = file_record(CONTROL_PLUGIN)
  memory_before = BASE.proc_meminfo()
  environment = os.environ.copy()
  conda_bin = "/home/intel/intel-box-env/conda/bin"
  conda_lib = "/home/intel/intel-box-env/conda/lib"
  environment["PATH"] = conda_bin + ":" + environment.get("PATH", "")
  environment["LD_LIBRARY_PATH"] = (
      conda_lib + ":" + environment.get("LD_LIBRARY_PATH", ""))

  source_checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "seq2231_admission_identity_exact",
          BASE.sha256(ADMISSION) == EXPECTED_ADMISSION_SHA256
          and admission["verdict"]["required_checks_passed"] is True
          and admission["verdict"][
              "isolated_serial_candidate_plugin_build_admitted"] is True
          and admission["verdict"][
              "product_build_or_speed_claim_admitted"] is False,
          admission_sha256=BASE.sha256(ADMISSION)),
      check(
          "durable_patch_hashes_and_heads_exact",
          patch_hashes_exact(source_before)
          and source_before["openvino_head"] == OPENVINO_HEAD
          and source_before["onednn_head"] == ONEDNN_HEAD
          and source_before["onednn_origin_commit_present"],
          openvino_head=source_before["openvino_head"],
          onednn_head=source_before["onednn_head"],
          patches=source_before["patches"]),
      check(
          "exact_materialized_postimages_bound",
          source_hashes_exact(source_before)
          and source_before["onednn_reverse_apply"]["returncode"] == 0
          and source_before["openvino_reverse_apply"]["returncode"] == 0,
          postimages=source_before["postimages"],
          onednn_target_status=source_before["onednn_target_status"],
          openvino_target_status=source_before["openvino_target_status"],
          onednn_reverse_apply=source_before["onednn_reverse_apply"],
          openvino_reverse_apply=source_before["openvino_reverse_apply"]),
      check(
          "swish_binary_grouped_postops_backport_exact",
          source_before["onednn_postops_exact"],
          origin_commit=ONEDNN_ORIGIN_COMMIT),
      check(
          "pr35924_prefill_fusion_and_decode_branch_exact",
          source_before["openvino_fusion_exact"]
          and source_before["decode_branch_exact"]),
      check(
          "accepted_control_plugin_exact_and_isolated",
          control_before["sha256"] == EXPECTED_CONTROL_SHA256
          and candidate_plugin != CONTROL_PLUGIN.resolve()
          and candidate_plugin != BUILD_PLUGIN.resolve(),
          control_plugin=control_before,
          candidate_plugin=str(candidate_plugin)),
      check(
          "eight_gib_preflight_clears",
          int(memory_before.get("MemAvailable", 0)) >= PREFLIGHT_BYTES,
          available_bytes=memory_before.get("MemAvailable"),
          preflight_bytes=PREFLIGHT_BYTES),
  ]
  source_admitted = all(row["pass"] for row in source_checks)

  configure_command = [
      str(CMAKE), "-S", str(ONEDNN), "-B", str(ONEDNN_BUILD)]
  onednn_build_command = [
      str(CMAKE), "--build", str(ONEDNN_BUILD),
      "--target", "install", "--parallel", "1"]
  plugin_build_command = [
      str(CMAKE), "--build", str(BUILD_TREE),
      "--target", "openvino_intel_gpu_plugin", "--parallel", "1"]
  configure: dict[str, Any] = {
      "returncode": 125, "skipped": "source gate failed", "monitor": {}}
  onednn_build: dict[str, Any] = {
      "returncode": 125, "skipped": "source gate failed", "monitor": {}}
  plugin_build: dict[str, Any] = {
      "returncode": 125, "skipped": "source gate failed", "monitor": {}}
  if source_admitted:
    configure = BASE.run_scoped(
        output, raw, "configure-onednn-pr35924", configure_command,
        args.configure_timeout_s, args.poll_interval_s, environment)
    if stage_ok(configure):
      onednn_build = BASE.run_scoped(
          output, raw, "build-onednn-pr35924", onednn_build_command,
          args.build_timeout_s, args.poll_interval_s, environment)
    if stage_ok(onednn_build):
      plugin_build = BASE.run_scoped(
          output, raw, "build-plugin-pr35924", plugin_build_command,
          args.build_timeout_s, args.poll_interval_s, environment)

  source_after = source_contract()
  onednn_library_after = file_record(ONEDNN_INSTALL_LIB)
  build_after = file_record(BUILD_PLUGIN)
  control_after = file_record(CONTROL_PLUGIN)
  build_stdout_paths = (
      raw / "build-onednn-pr35924.stdout",
      raw / "build-plugin-pr35924.stdout",
  )
  build_stdout = "\n".join(
      path.read_text(encoding="utf-8", errors="replace")
      for path in build_stdout_paths if path.is_file())
  compile_steps = len(re.findall(
      r"\bBuilding (?:C|CXX) object\b", build_stdout))
  build_ninja = (ONEDNN_BUILD / "build.ninja").read_text(
      encoding="utf-8", errors="replace")
  candidate = {
      "path": str(candidate_plugin),
      "exists": False,
      "bytes": 0,
      "sha256": None,
  }
  build_succeeded = bool(
      stage_ok(configure) and stage_ok(onednn_build)
      and stage_ok(plugin_build)
      and onednn_library_after["sha256"]
      != onednn_library_before["sha256"]
      and onednn_library_after["mtime_ns"]
      > onednn_library_before["mtime_ns"]
      and build_after["sha256"] != build_before["sha256"]
      and build_after["mtime_ns"] > build_before["mtime_ns"]
      and build_after["bytes"] > 0)
  if build_succeeded:
    candidate_plugin.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(BUILD_PLUGIN, candidate_plugin)
    candidate = file_record(candidate_plugin)

  links = (
      run(["/usr/bin/ldd", str(candidate_plugin)])
      if candidate_plugin.is_file() else
      subprocess.CompletedProcess(
          ["/usr/bin/ldd", str(candidate_plugin)], 125, "", "missing"))
  symbols = (
      run([str(NM), "-C", str(ONEDNN_INSTALL_LIB)])
      if ONEDNN_INSTALL_LIB.is_file() else
      subprocess.CompletedProcess(
          [str(NM), "-C", str(ONEDNN_INSTALL_LIB)], 125, "", "missing"))
  link_text = (links.stdout + links.stderr).lower()
  configure_monitor = configure.get("monitor", {})
  onednn_monitor = onednn_build.get("monitor", {})
  plugin_monitor = plugin_build.get("monitor", {})
  all_events = (
      configure_monitor.get("memory_events_max", {}),
      onednn_monitor.get("memory_events_max", {}),
      plugin_monitor.get("memory_events_max", {}),
  )
  minimum_available = min(
      int(configure_monitor.get(
          "system_available_min_bytes", memory_before["MemAvailable"])),
      int(onednn_monitor.get(
          "system_available_min_bytes", memory_before["MemAvailable"])),
      int(plugin_monitor.get(
          "system_available_min_bytes", memory_before["MemAvailable"])))
  build_checks = [
      check(
          "configure_registers_new_grouped_postops_source",
          stage_ok(configure)
          and build_ninja.count("grouped_post_ops_gen.cpp") >= 1,
          configure=configure,
          ninja_occurrences=build_ninja.count("grouped_post_ops_gen.cpp")),
      check(
          "sole_serial_onednn_and_gpu_plugin_builds_succeed",
          build_succeeded
          and onednn_build_command[-2:] == ["--parallel", "1"]
          and plugin_build_command[-2:] == ["--parallel", "1"],
          onednn_build=onednn_build,
          plugin_build=plugin_build,
          compile_steps=compile_steps,
          onednn_library_before=onednn_library_before,
          onednn_library_after=onednn_library_after,
          build_plugin_before=build_before,
          build_plugin_after=build_after),
      check(
          "four_gib_abort_and_oom_guards_hold",
          minimum_available >= ABORT_BYTES
          and all(int(row.get("oom", 0)) == 0
                  and int(row.get("oom_kill", 0)) == 0
                  for row in all_events),
          minimum_available_bytes=minimum_available,
          abort_bytes=ABORT_BYTES,
          configure_memory_events=all_events[0],
          onednn_build_memory_events=all_events[1],
          plugin_build_memory_events=all_events[2]),
      check(
          "candidate_is_copied_to_isolated_carrier",
          candidate.get("sha256") == build_after["sha256"]
          and candidate.get("sha256") not in (
              build_before["sha256"], control_before["sha256"])
          and candidate.get("bytes") == build_after["bytes"],
          candidate_plugin=candidate),
      check(
          "candidate_link_map_is_complete",
          links.returncode == 0
          and "libopenvino.so" in link_text
          and "libopencl.so" in link_text
          and "not found" not in link_text,
          link_returncode=links.returncode,
          link_stdout=links.stdout,
          link_stderr=links.stderr),
      check(
          "installed_onednn_contains_grouped_postops_generator_symbols",
          symbols.returncode == 0
          and "generate_post_ops_microgemm_header" in symbols.stdout
          and "check_post_op_chain" in symbols.stdout,
          symbol_returncode=symbols.returncode,
          grouped_postops_symbol_lines=[
              line for line in symbols.stdout.splitlines()
              if ("generate_post_ops_microgemm_header" in line
                  or "check_post_op_chain" in line)]),
      check(
          "source_and_accepted_control_are_unchanged_by_build",
          source_hashes_exact(source_after)
          and source_after["onednn_target_status"]
          == source_before["onednn_target_status"]
          and source_after["openvino_target_status"]
          == source_before["openvino_target_status"]
          and control_after["sha256"] == control_before["sha256"],
          source_before=source_before,
          source_after=source_after,
          control_before=control_before,
          control_after=control_after),
      check(
          "no_gpu_context_model_or_infer_request_ran",
          True, gpu_contexts_created=0, gpu_kernels_executed=0,
          model_workers_started=0, infer_requests_created=0,
          inference_workers_started=0),
  ]
  passed = source_admitted and all(row["pass"] for row in build_checks)
  verdict = {
      "required_checks_passed": passed,
      "compile_only_graph_gate_admitted": passed,
      "inference_admitted": False,
      "performance_claim_admitted": False,
      "verdict": (
          "admit_pr35924_plugin_for_compile_only_graph_gate"
          if passed else
          "repair_pr35924_source_port_or_serial_build"),
      "next_action": (
          "compile the exact 2k product graph with the isolated candidate "
          "plugin, require all 40 grouped-MoE owners and the grouped GEMM "
          "post-op provider, then admit one inference only if compilation "
          "and provider binding pass"),
  }
  metrics = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "git": repo,
      "admission": {
          "path": BASE.relative(ADMISSION),
          "sha256": BASE.sha256(ADMISSION),
      },
      "source_before": source_before,
      "source_after": source_after,
      "source_checks": source_checks,
      "configure": configure,
      "onednn_build": {
          **onednn_build,
          "target": "install",
          "parallel_jobs": 1,
      },
      "plugin_build": {
          **plugin_build,
          "target": "openvino_intel_gpu_plugin",
          "parallel_jobs": 1,
          "compile_steps": compile_steps,
      },
      "onednn_library_before": onednn_library_before,
      "onednn_library_after": onednn_library_after,
      "build_plugin_before": build_before,
      "build_plugin_after": build_after,
      "control_plugin_before": control_before,
      "control_plugin_after": control_after,
      "candidate_plugin": candidate,
      "link_map": {
          "command": links.args,
          "returncode": links.returncode,
          "stdout": links.stdout,
          "stderr": links.stderr,
      },
      "memory": {
          "initial": memory_before,
          "preflight_bytes": PREFLIGHT_BYTES,
          "abort_bytes": ABORT_BYTES,
          "minimum_available_bytes": minimum_available,
      },
      "workers": {
          "maximum_concurrent_workers": 1,
          "configure_invocations": int(source_admitted),
          "onednn_builds": int(source_admitted and stage_ok(configure)),
          "plugin_builds": int(stage_ok(onednn_build)),
          "gpu_contexts_created": 0,
          "gpu_kernels_executed": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
          "inference_workers_started": 0,
      },
      "checks": source_checks + build_checks,
      "verdict": verdict,
  }
  write_json(output / "metrics.json", metrics)
  (output / "report.md").write_text(
      "# OpenVINO PR35924 candidate plugin build\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Configure/build elapsed: "
      f"`{configure.get('elapsed_seconds', 0):.3f}/"
      f"{onednn_build.get('elapsed_seconds', 0):.3f}/"
      f"{plugin_build.get('elapsed_seconds', 0):.3f} s`\n"
      f"- Compile steps / parallel jobs: `{compile_steps}/1`\n"
      f"- Candidate SHA256: `{candidate.get('sha256')}`\n"
      f"- Minimum available memory: `{minimum_available}` bytes\n"
      "- GPU/model/InferRequest/inference workers: `0/0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": BASE.relative(output),
      "required_checks_passed": passed,
      "verdict": verdict["verdict"],
      "configure_returncode": configure.get("returncode"),
      "onednn_build_returncode": onednn_build.get("returncode"),
      "plugin_build_returncode": plugin_build.get("returncode"),
      "onednn_build_elapsed_seconds": onednn_build.get("elapsed_seconds"),
      "plugin_build_elapsed_seconds": plugin_build.get("elapsed_seconds"),
      "compile_steps": compile_steps,
      "candidate_plugin_sha256": candidate.get("sha256"),
      "minimum_available_bytes": minimum_available,
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
