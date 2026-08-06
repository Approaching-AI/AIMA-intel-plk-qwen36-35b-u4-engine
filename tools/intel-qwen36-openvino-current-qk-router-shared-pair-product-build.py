#!/usr/bin/env python3
"""Apply and serial-build the admitted current Q/K plus N=1024 pair cut.

This is a compiler-only boundary.  It applies the incremental default-off
pair patch once, builds only ``openvino_intel_gpu_plugin`` at ``-j1``, and
copies the result to an isolated carrier.  It creates no GPU context, loads no
model, and runs no inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-current-qk-router-shared-pair-product-build-v1")
COMMON_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-current-qk-router-shared-product-build.py")
SOURCE_GATE = ROOT / (
    "output/openvino-current-qk-router-shared-pair-source-gate-"
    "20260731Tseq2211a-clean/result.json")
PAIR_PATCH = ROOT / "engine/openvino/iq36-current-router-shared-pair.patch"
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
SOURCE_REL = Path(
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
SOURCE = SOURCE_TREE / SOURCE_REL
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
ACCEPTED_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2212/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
PINNED_SOURCE_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
SOURCE_GATE_COMMIT = "a1b85febb41da489a0d84db71be151640afc473b"
EXPECTED_SOURCE_SHA256 = (
    "3bb3485f4ef6303f9c34966d170d683ddf7b9e52d131836ec07e08af02de7bd3")
EXPECTED_SOURCE_GATE_SHA256 = (
    "d581bada552d6780f6657e491d863ded603aae026cacfb2f9b249cd0c308ce08")
EXPECTED_PAIR_PATCH_SHA256 = (
    "092e1b3d23277cd1ab34577fc26f594efcfb0a837d72904b28b64ae01af36d3a")
EXPECTED_PRODUCT_TOOL_SHA256 = (
    "baa6cb5591766eb91dcb1456d0195216f10a4fafb9477fc3a357f8eb98a8c3b1")
EXPECTED_BUILD_PLUGIN_SHA256 = (
    "3ffcacbd4f7b1ab10e9a461b28c7385a86ec9c530f4af03495c5fb3dbba239f5")
EXPECTED_ACCEPTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


COMMON = load_module("iq36_shared_pair_build_common", COMMON_TOOL)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--candidate-plugin", type=Path, default=DEFAULT_CANDIDATE_PLUGIN)
  parser.add_argument("--timeout-s", type=float, default=900.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument("--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument("--poll-interval-s", type=float, default=0.1)
  args = parser.parse_args()
  if args.timeout_s <= 0.0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if (args.min_available_gib <= 0.0 or
      args.abort_below_available_gib <= 0.0 or
      args.abort_below_available_gib > args.min_available_gib):
    parser.error("memory thresholds are invalid")
  return args


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
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path.resolve())


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  candidate_plugin = args.candidate_plugin.resolve()
  if output.exists():
    raise SystemExit(f"output already exists: {output}")
  if candidate_plugin.exists():
    raise SystemExit(
        f"isolated candidate plugin already exists: {candidate_plugin}")
  (output / "raw").mkdir(parents=True, exist_ok=False)

  required = (
      COMMON_TOOL, SOURCE_GATE, PAIR_PATCH, PRODUCT_TOOL, SOURCE_TREE, SOURCE,
      BUILD_TREE, BUILD_PLUGIN, ACCEPTED_PLUGIN, CMAKE)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing shared-pair build inputs: " + ", ".join(missing))

  head = COMMON.run(["git", "rev-parse", "HEAD"])["stdout"].strip()
  origin_main = COMMON.run(
      ["git", "rev-parse", "origin/main"])["stdout"].strip()
  repo_status = COMMON.run(
      ["git", "status", "--porcelain"])["stdout"].splitlines()
  gate_ancestor = COMMON.run(
      ["git", "merge-base", "--is-ancestor", SOURCE_GATE_COMMIT, head])
  source_commit = COMMON.run(
      ["git", "rev-parse", "HEAD"], SOURCE_TREE)["stdout"].strip()
  forward_check = COMMON.run(
      ["git", "apply", "--check", str(PAIR_PATCH)], SOURCE_TREE)
  reverse_check_before = COMMON.run(
      ["git", "apply", "--reverse", "--check", str(PAIR_PATCH)],
      SOURCE_TREE)
  gate = load_json(SOURCE_GATE)
  source_before = COMMON.source_record()
  build_before = COMMON.plugin_record(BUILD_PLUGIN)
  accepted_before = COMMON.plugin_record(ACCEPTED_PLUGIN)
  start_memory = COMMON.meminfo()
  preflight_bytes = int(args.min_available_gib * 1024**3)
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  patch_text = PAIR_PATCH.read_text(encoding="utf-8")
  product_text = PRODUCT_TOOL.read_text(encoding="utf-8")

  source_checks = [
      check("repository_is_clean_pushed_and_contains_source_gate",
            not repo_status and head == origin_main and
            gate_ancestor["returncode"] == 0,
            head=head, origin_main=origin_main, status=repo_status,
            source_gate_is_ancestor=gate_ancestor["returncode"] == 0),
      check("seq2211a_admits_one_patch_and_serial_build_only",
            sha256(SOURCE_GATE) == EXPECTED_SOURCE_GATE_SHA256 and
            gate.get("required_checks_passed") is True and
            gate.get("verdict") ==
                "admit_one_current_qk_router_shared_pair_patch_and_"
                "serial_build" and
            gate.get("source_patch_admitted") is True and
            gate.get("serial_plugin_build_admitted") is True and
            gate.get("gpu_worker_admitted") is False and
            gate.get("model_worker_admitted") is False and
            gate.get("git", {}).get("head") == SOURCE_GATE_COMMIT,
            source_gate_sha256=sha256(SOURCE_GATE),
            source_gate_verdict=gate.get("verdict")),
      check("current_cumulative_source_is_exact_preimage",
            source_commit == PINNED_SOURCE_COMMIT and
            source_before["sha256"] == EXPECTED_SOURCE_SHA256,
            source_commit=source_commit, source_before=source_before),
      check("incremental_pair_patch_is_forward_only_and_pinned",
            sha256(PAIR_PATCH) == EXPECTED_PAIR_PATCH_SHA256 and
            forward_check["returncode"] == 0 and
            reverse_check_before["returncode"] != 0 and
            patch_text.count("diff --git ") == 1 and
            "IQ36_ROUTER_SHARED_PAIR" in patch_text and
            "enabled_routes != 1" in patch_text and
            "n == 512" in patch_text,
            patch_sha256=sha256(PAIR_PATCH),
            forward_check=forward_check,
            reverse_check_before=reverse_check_before),
      check("candidate_only_runtime_switch_contract_is_pinned",
            sha256(PRODUCT_TOOL) == EXPECTED_PRODUCT_TOOL_SHA256 and
            "--fuse-router-shared-pair" in product_text and
            product_text.count("IQ36_ROUTER_SHARED_PAIR") == 3 and
            "router-shared triple and pair are mutually exclusive" in
                product_text and
            "router-shared fusion leaked into a fixed-FC route" in
                product_text,
            product_tool_sha256=sha256(PRODUCT_TOOL)),
      check("mutable_build_base_and_accepted_carrier_are_exact",
            build_before["sha256"] == EXPECTED_BUILD_PLUGIN_SHA256 and
            accepted_before["sha256"] == EXPECTED_ACCEPTED_PLUGIN_SHA256,
            build_plugin_before=build_before,
            accepted_plugin_before=accepted_before),
      check("eight_gib_preflight_is_available",
            int(start_memory.get("MemAvailable", 0)) >= preflight_bytes,
            available_bytes=start_memory.get("MemAvailable"),
            preflight_bytes=preflight_bytes),
      check("isolated_output_cannot_overwrite_existing_carriers",
            candidate_plugin not in (
                BUILD_PLUGIN.resolve(), ACCEPTED_PLUGIN.resolve()) and
            not candidate_plugin.exists(),
            candidate_plugin=str(candidate_plugin)),
      check("no_gpu_context_model_or_inference_worker_is_admitted", True,
            gpu_contexts=0, model_workers=0, infer_requests=0,
            inference_workers=0),
  ]
  source_apply_admitted = all(row["pass"] for row in source_checks)
  apply_result = (
      COMMON.run(["git", "apply", str(PAIR_PATCH)], SOURCE_TREE)
      if source_apply_admitted else {
          "command": ["git", "apply", str(PAIR_PATCH)],
          "returncode": 125,
          "stdout": "",
          "stderr": "source gate failed",
          "skipped": True,
      })

  source_after = COMMON.source_record()
  reverse_check_after = COMMON.run(
      ["git", "apply", "--reverse", "--check", str(PAIR_PATCH)],
      SOURCE_TREE)
  forward_check_after = COMMON.run(
      ["git", "apply", "--check", str(PAIR_PATCH)], SOURCE_TREE)
  source_text = SOURCE.read_text(encoding="utf-8")
  apply_checks = [
      check("one_incremental_pair_patch_application_succeeds",
            source_apply_admitted and apply_result["returncode"] == 0 and
            source_after["sha256"] != source_before["sha256"],
            apply_result=apply_result,
            source_before_sha256=source_before["sha256"],
            source_after_sha256=source_after["sha256"]),
      check("applied_source_is_reverse_only_and_structurally_exact",
            reverse_check_after["returncode"] == 0 and
            forward_check_after["returncode"] != 0 and
            source_text.count("bool fixed_m1024_scope_enabled()") == 1 and
            source_text.count("bool router_shared_triple_enabled()") == 1 and
            source_text.count("bool router_shared_pair_enabled()") == 1 and
            source_text.count(
                'std::getenv("IQ36_ROUTER_SHARED_PAIR")') == 1 and
            source_text.count("enabled_routes != 1") == 1 and
            source_text.count(
                "fixed_m1024_scope_enabled() || "
                "router_shared_pair_enabled()") == 1 and
            source_text.count(
                "router_shared_triple_enabled() ? 3 : 2") == 1,
            reverse_check_after=reverse_check_after,
            forward_check_after=forward_check_after,
            source_after=source_after),
  ]
  build_admitted = source_apply_admitted and all(
      row["pass"] for row in apply_checks)
  build_command = [
      str(CMAKE), "--build", str(BUILD_TREE),
      "--target", "openvino_intel_gpu_plugin", "--parallel", "1",
  ]
  build = (
      COMMON.monitor_build(
          build_command, output, args.timeout_s, abort_bytes,
          args.poll_interval_s)
      if build_admitted else {
          "command": build_command,
          "returncode": 125,
          "elapsed_seconds": 0.0,
          "timed_out": False,
          "memory_guard_tripped": False,
          "oom_observed": False,
          "monitor": {},
          "stdout": None,
          "stderr": None,
          "skipped": "source application gate failed",
      })

  build_after = COMMON.plugin_record(BUILD_PLUGIN)
  accepted_after = COMMON.plugin_record(ACCEPTED_PLUGIN)
  monitor = build.get("monitor", {})
  build_succeeded = (
      build["returncode"] == 0 and
      not build["timed_out"] and
      not build["memory_guard_tripped"] and
      not build["oom_observed"] and
      build_after["sha256"] != build_before["sha256"] and
      build_after["size_bytes"] > 0 and
      build_after["mtime_ns"] > build_before["mtime_ns"])
  candidate: dict[str, Any] = {
      "path": str(candidate_plugin),
      "sha256": None,
      "size_bytes": None,
      "mtime_ns": None,
  }
  if build_admitted and build_succeeded:
    candidate_plugin.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(BUILD_PLUGIN, candidate_plugin)
    candidate = COMMON.plugin_record(candidate_plugin)

  links = (
      COMMON.run(["ldd", str(candidate_plugin)])
      if candidate_plugin.is_file() else {
          "command": ["ldd", str(candidate_plugin)],
          "returncode": 125,
          "stdout": "",
          "stderr": "candidate plugin missing",
      })
  lower_links = (links["stdout"] + links["stderr"]).lower()
  build_checks = [
      check("one_serial_incremental_gpu_plugin_build_succeeds",
            build_admitted and build_succeeded, build=build),
      check("four_gib_abort_guard_holds_without_oom",
            not build["memory_guard_tripped"] and
            not build["oom_observed"] and
            int(monitor.get("system_available_min_bytes", 0)) >= abort_bytes,
            abort_bytes=abort_bytes, monitor=monitor),
      check("new_plugin_is_copied_to_isolated_seq2212_carrier",
            candidate.get("sha256") == build_after["sha256"] and
            candidate.get("size_bytes") == build_after["size_bytes"] and
            candidate.get("sha256") not in (
                EXPECTED_BUILD_PLUGIN_SHA256,
                EXPECTED_ACCEPTED_PLUGIN_SHA256),
            candidate_plugin=candidate,
            build_plugin_after=build_after),
      check("accepted_seq2189_carrier_remains_bitwise_immutable",
            accepted_after == accepted_before,
            accepted_plugin_before=accepted_before,
            accepted_plugin_after=accepted_after),
      check("isolated_plugin_link_map_is_complete",
            links["returncode"] == 0 and
            "libopenvino.so" in lower_links and
            "libopencl.so" in lower_links and
            "not found" not in lower_links,
            link_map=links),
      check("no_gpu_context_model_or_inference_worker_ran", True,
            compiler_builds=1 if build_admitted else 0, gpu_contexts=0,
            model_workers=0, infer_requests=0, inference_workers=0,
            workers_concurrent=False),
  ]
  passed = (
      source_apply_admitted and
      all(row["pass"] for row in apply_checks) and
      all(row["pass"] for row in build_checks))
  verdict = (
      "admit_current_qk_router_shared_pair_plugin_for_compile_gate"
      if passed else "repair_current_qk_router_shared_pair_source_or_build")
  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "verdict": verdict,
      "required_checks_passed": passed,
      "candidate_only_graph_compile_admitted": passed,
      "infer_request_admitted": False,
      "inference_admitted": False,
      "performance_claim_admitted": False,
      "repository": {
          "head": head,
          "origin_main": origin_main,
          "dirty": bool(repo_status),
          "status": repo_status,
      },
      "source_gate": {
          "path": display(SOURCE_GATE),
          "sha256": sha256(SOURCE_GATE),
          "verdict": gate.get("verdict"),
      },
      "source_before": source_before,
      "source_after_apply": source_after,
      "source_checks": source_checks,
      "source_apply": apply_result,
      "apply_checks": apply_checks,
      "build": build,
      "build_plugin_before": build_before,
      "build_plugin_after": build_after,
      "accepted_plugin_before": accepted_before,
      "accepted_plugin_after": accepted_after,
      "candidate_plugin": candidate,
      "link_map": links,
      "build_checks": build_checks,
      "workers": {
          "compiler_builds": 1 if build_admitted else 0,
          "gpu_contexts": 0,
          "model_workers": 0,
          "infer_requests": 0,
          "inference_workers": 0,
          "workers_concurrent": False,
      },
      "next_action": {
          "route": "current_qk_router_shared_pair_compile_gate",
          "requirements": [
              "push a candidate-only compile gate before loading the plugin",
              "enable Q/K plus pair and keep triple/fixed manager off",
              "require QK/attention/linear owners 10/10/30",
              "create no InferRequest until compile evidence passes",
              "require FC/shared-pair/router 331/40/40 at correctness",
          ],
      },
      "checks": source_checks + apply_checks + build_checks,
  }
  write_json(output / "result.json", result)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "tool": display(Path(__file__)),
      "tool_sha256": sha256(Path(__file__)),
      "repository": result["repository"],
      "inputs": {
          display(SOURCE_GATE): sha256(SOURCE_GATE),
          display(PAIR_PATCH): sha256(PAIR_PATCH),
          display(PRODUCT_TOOL): sha256(PRODUCT_TOOL),
          str(SOURCE): source_after["sha256"],
          str(ACCEPTED_PLUGIN): accepted_after["sha256"],
      },
      "candidate_plugin": candidate,
      "compiler_builds": result["workers"]["compiler_builds"],
      "gpu_contexts": 0,
      "model_workers": 0,
      "infer_requests": 0,
      "inference_workers": 0,
  })
  report = f"""# Current Q/K plus N=1024 shared-pair serial build

Verdict: **{verdict}**. Required checks:
`{str(passed).lower()}`.

Seq2211a admitted exactly one incremental source application and one serial
plugin build. The patch changed FC transformation source
`{source_before['sha256']}` to `{source_after['sha256']}`. The `-j1` build
completed in `{float(build['elapsed_seconds']):.3f} s` and produced isolated
candidate `{candidate.get('sha256')}`.

Peak process-group RSS/swap was
`{int(monitor.get('process_group_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_group_swap_peak_bytes', 0))} B`; minimum available
memory was `{int(monitor.get('system_available_min_bytes', 0))} B`. No guard
or OOM fired. The accepted seq2189 plugin remained bitwise unchanged. No GPU
context, model worker, InferRequest, or inference worker ran.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "elapsed_seconds": build["elapsed_seconds"],
      "candidate_plugin_sha256": candidate.get("sha256"),
      "source_after_sha256": source_after["sha256"],
      "peak_rss_bytes": monitor.get("process_group_rss_peak_bytes"),
      "minimum_available_bytes": monitor.get("system_available_min_bytes"),
      "oom_observed": build["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
