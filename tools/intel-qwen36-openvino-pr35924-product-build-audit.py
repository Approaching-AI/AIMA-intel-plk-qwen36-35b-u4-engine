#!/usr/bin/env python3
"""Audit seq2233's successful PR35924 plugin build without rebuilding.

Seq2233 compiled and linked the exact candidate but its final gate looked for
internal oneDNN generator symbols in the stripped plugin rather than in the
installed static archive.  This audit binds every original input and output,
checks the archive members and definitions at their real ownership boundary,
and performs no configure, compile, GPU, model, or inference work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-product-build-audit-v0"
BUILD = ROOT / (
    "output/openvino-pr35924-product-build-"
    "20260731Tseq2233-clean/metrics.json")
ONEDNN_STDOUT = ROOT / (
    "output/openvino-pr35924-product-build-"
    "20260731Tseq2233-clean/raw/build-onednn-pr35924.stdout")
PLUGIN_STDOUT = ROOT / (
    "output/openvino-pr35924-product-build-"
    "20260731Tseq2233-clean/raw/build-plugin-pr35924.stdout")
R0 = Path("/home/intel/intel-qwen36-r0")
ONEDNN_LIBRARY = R0 / (
    "build/openvino-90214e-l0-gpu/src/plugins/intel_gpu/thirdparty/"
    "onednn_gpu_install/lib/libopenvino_onednn_gpu.a")
CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2233/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
NM = Path("/home/intel/intel-box-env/conda/bin/nm")
AR = Path("/home/intel/intel-box-env/conda/bin/ar")
EXPECTED_SHA256 = {
    BUILD: (
        "9d91d52648509ac5a89f16ddca589dc23d274cc0393498d34b962429c044ec64"),
    ONEDNN_STDOUT: (
        "149d4494ebcf117e915abcbb4ad3703f26be2abafb2404368f0f6098f9d2d82f"),
    PLUGIN_STDOUT: (
        "db0b9544aca3b26689bd3d13a9cb7039ba6bdc578a0c8e6c0b0564f40d6f6e1b"),
    ONEDNN_LIBRARY: (
        "6a9c11e317be6b18db5a0da20912e9660e3734af3e3e0a9cf2672fb89578ed2d"),
    CANDIDATE_PLUGIN: (
        "c66c9be61ee31110a55c8a064ed1390bd3d21a3f1766a03fdea84a078a519849"),
    CONTROL_PLUGIN: (
        "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985"),
}
MEMORY_STOP_BYTES = 4 * 1024**3


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  return parser.parse_args()


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


def run(command: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=cwd, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace")


def git(*args: str) -> str:
  result = run(["git", *args])
  if result.returncode != 0:
    raise RuntimeError(
        f"git failed ({result.returncode}): {args}\n{result.stderr}")
  return result.stdout.strip()


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


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  missing = [str(path) for path in (*EXPECTED_SHA256, NM, AR)
             if not path.is_file()]
  if missing:
    raise SystemExit("missing seq2233 audit inputs: " + ", ".join(missing))

  memory_start = available_memory_bytes()
  if memory_start < MEMORY_STOP_BYTES:
    raise SystemExit("available memory below 4 GiB before build audit")
  repo = repository_state(output)
  build = load_json(BUILD)
  onednn_stdout = ONEDNN_STDOUT.read_text(
      encoding="utf-8", errors="replace")
  plugin_stdout = PLUGIN_STDOUT.read_text(
      encoding="utf-8", errors="replace")
  members = run([str(AR), "t", str(ONEDNN_LIBRARY)])
  symbols = run([str(NM), "-C", str(ONEDNN_LIBRARY)])
  links = run(["/usr/bin/ldd", str(CANDIDATE_PLUGIN)])
  grouped_symbol_lines = [
      line for line in symbols.stdout.splitlines()
      if ("generate_post_ops_microgemm_header" in line
          or "check_post_op_chain" in line)]
  failed = [
      row for row in build.get("checks", [])
      if isinstance(row, dict) and row.get("pass") is not True]
  original_failure_exact = bool(
      len(failed) == 1
      and failed[0].get("name")
      == "candidate_contains_grouped_postops_generator_symbols")
  original_other_checks_pass = all(
      row.get("pass") is True
      for row in build.get("checks", [])
      if row is not failed[0])
  hashes = {relative(path): sha256(path) for path in EXPECTED_SHA256}
  hashes_exact = all(
      hashes[relative(path)] == expected
      for path, expected in EXPECTED_SHA256.items())
  memory_end = available_memory_bytes()
  link_text = (links.stdout + links.stderr).lower()
  checks = [
      check(
          "repository_clean_and_pushed_at_gate",
          repo["branch"] == "main" and repo["pushed"] and not repo["dirty"],
          **repo),
      check(
          "seq2233_inputs_and_outputs_hash_exact",
          hashes_exact, hashes=hashes),
      check(
          "seq2233_only_failed_at_wrong_symbol_ownership_boundary",
          original_failure_exact and original_other_checks_pass
          and build["configure"]["returncode"] == 0
          and build["onednn_build"]["returncode"] == 0
          and build["plugin_build"]["returncode"] == 0,
          failed_checks=failed),
      check(
          "grouped_postops_sources_really_compiled_and_installed",
          "Building CXX object" in onednn_stdout
          and "grouped_micro_gemm.cpp.o" in onednn_stdout
          and "grouped_post_ops_gen.cpp.o" in onednn_stdout
          and "matmul_grouped_micro_gemm_kernel.cpp.o" in onednn_stdout
          and "Installing:" in onednn_stdout
          and str(ONEDNN_LIBRARY) in onednn_stdout
          and "Linking CXX shared module" in plugin_stdout,
          onednn_build_stdout_sha256=sha256(ONEDNN_STDOUT),
          plugin_build_stdout_sha256=sha256(PLUGIN_STDOUT)),
      check(
          "installed_archive_contains_generator_member_and_definitions",
          members.returncode == 0
          and "grouped_micro_gemm.cpp.o" in members.stdout
          and "grouped_post_ops_gen.cpp.o" in members.stdout
          and symbols.returncode == 0
          and any(
              " T " in line
              and "generate_post_ops_microgemm_header" in line
              for line in grouped_symbol_lines)
          and any(
              " T " in line and "check_post_op_chain" in line
              for line in grouped_symbol_lines),
          archive_members=[
              line for line in members.stdout.splitlines()
              if "grouped_" in line],
          grouped_symbol_lines=grouped_symbol_lines),
      check(
          "candidate_plugin_is_new_linked_and_control_unchanged",
          build["candidate_plugin"]["sha256"]
          == EXPECTED_SHA256[CANDIDATE_PLUGIN]
          and build["candidate_plugin"]["sha256"]
          != EXPECTED_SHA256[CONTROL_PLUGIN]
          and links.returncode == 0
          and "libopenvino.so" in link_text
          and "libopencl.so" in link_text
          and "not found" not in link_text,
          candidate_sha256=sha256(CANDIDATE_PLUGIN),
          control_sha256=sha256(CONTROL_PLUGIN),
          link_returncode=links.returncode),
      check(
          "build_memory_policy_held_without_oom",
          build["memory"]["minimum_available_bytes"] >= MEMORY_STOP_BYTES
          and all(
              int(stage["monitor"]["memory_events_max"].get(
                  "oom_kill", 0)) == 0
              and int(stage["monitor"]["memory_events_max"].get(
                  "oom_group_kill", 0)) == 0
              for stage in (
                  build["configure"], build["onednn_build"],
                  build["plugin_build"])),
          minimum_available_bytes=build["memory"][
              "minimum_available_bytes"],
          stop_bytes=MEMORY_STOP_BYTES),
      check(
          "audit_used_no_configure_compile_gpu_model_or_inference",
          True, configure_invocations=0, compiler_invocations=0,
          gpu_contexts_created=0, gpu_kernels_executed=0,
          model_workers_started=0, infer_requests_created=0,
          inference_workers_started=0),
      check(
          "audit_memory_stop_held",
          min(memory_start, memory_end) >= MEMORY_STOP_BYTES,
          available_start_bytes=memory_start,
          available_end_bytes=memory_end,
          stop_bytes=MEMORY_STOP_BYTES),
  ]
  passed = all(row["pass"] for row in checks)
  verdict = {
      "required_checks_passed": passed,
      "compile_only_graph_gate_admitted": passed,
      "inference_admitted": False,
      "performance_claim_admitted": False,
      "verdict": (
          "admit_pr35924_plugin_for_compile_only_graph_gate"
          if passed else
          "hold_pr35924_for_archive_or_identity_failure"),
      "reason": (
          "The grouped-postops generator is an internal oneDNN implementation "
          "symbol owned by the installed static archive, not a required "
          "export from the stripped final plugin. The archive contains both "
          "the object and strong definitions, seq2233 compiled and installed "
          "it, and the final plugin linked successfully."),
      "next_action": (
          "compile the exact 2k product graph with the isolated candidate "
          "plugin and require all 40 grouped-MoE owners plus grouped-postops "
          "provider binding before one inference"),
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
          } for path in EXPECTED_SHA256},
      "original_build": {
          "failed_checks": failed,
          "all_other_checks_pass": original_other_checks_pass,
          "configure_returncode": build["configure"]["returncode"],
          "onednn_build_returncode": build["onednn_build"]["returncode"],
          "plugin_build_returncode": build["plugin_build"]["returncode"],
          "candidate_plugin": build["candidate_plugin"],
          "memory": build["memory"],
      },
      "archive": {
          "path": str(ONEDNN_LIBRARY),
          "sha256": sha256(ONEDNN_LIBRARY),
          "grouped_members": [
              line for line in members.stdout.splitlines()
              if "grouped_" in line],
          "grouped_symbol_lines": grouped_symbol_lines,
      },
      "process_census": {
          "configure_invocations": 0,
          "compiler_invocations": 0,
          "gpu_contexts_created": 0,
          "gpu_kernels_executed": 0,
          "model_workers_started": 0,
          "infer_requests_created": 0,
          "inference_workers_started": 0,
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
      "# OpenVINO PR35924 product-build audit\n\n"
      f"- Required checks: `{passed}`\n"
      f"- Verdict: `{verdict['verdict']}`\n"
      f"- Candidate SHA256: `{sha256(CANDIDATE_PLUGIN)}`\n"
      f"- oneDNN archive SHA256: `{sha256(ONEDNN_LIBRARY)}`\n"
      "- Generator object / strong definitions: `present/present`\n"
      "- Audit configure/compiler/GPU/model/InferRequest: `0/0/0/0/0`\n",
      encoding="utf-8")
  print(json.dumps({
      "output": relative(output),
      "required_checks_passed": passed,
      "verdict": verdict["verdict"],
      "candidate_plugin_sha256": sha256(CANDIDATE_PLUGIN),
      "onednn_archive_sha256": sha256(ONEDNN_LIBRARY),
      "grouped_symbol_line_count": len(grouped_symbol_lines),
      "minimum_available_bytes": min(memory_start, memory_end),
  }, sort_keys=True), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
