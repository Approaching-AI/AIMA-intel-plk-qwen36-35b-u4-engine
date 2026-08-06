#!/usr/bin/env python3
"""Apply and serial-build the admitted N=1024 pair decomposed-GLU cut.

This compiler-only boundary applies the incremental default-off GLU patch
once, builds only ``openvino_intel_gpu_plugin`` at ``-j1``, and copies the
result to an isolated carrier. It creates no GPU context, loads no model, and
runs no inference.
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
    "intel-qwen36-openvino-current-qk-router-shared-pair-no-glu-"
    "product-build-v1")
COMMON_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-current-qk-router-shared-"
    "product-build.py")
OUTCOME_GATE = ROOT / (
    "output/openvino-current-qk-router-shared-pair-output130-outcome-"
    "20260731Tseq2216-clean/metrics.json")
NO_GLU_PATCH = ROOT / (
    "engine/openvino/iq36-current-router-shared-pair-no-glu.patch")
PRODUCT_TOOL = ROOT / "tools/intel-qwen36-openvino-hot-cold-product-gate.py"
R0 = Path("/home/intel/intel-qwen36-r0")
SOURCE_TREE = R0 / "source/openvino-90214e5be05"
TRANSFORM_REL = Path(
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
TRANSFORM_SOURCE = SOURCE_TREE / TRANSFORM_REL
FC_SOURCE = SOURCE_TREE / (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
BUILD_TREE = R0 / "build/openvino-90214e-l0-gpu"
BUILD_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2109/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
PAIR_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2212/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
ACCEPTED_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
DEFAULT_CANDIDATE_PLUGIN = R0 / (
    "output/openvino-90214e-l0-gpu-seq2217/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
PINNED_SOURCE_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
OUTCOME_GATE_COMMIT = "847e2c20cc4e2096c081208eb86e1ae8315816d1"
EXPECTED_OUTCOME_SHA256 = (
    "e1ddf6eaa7efcffc8ee7299556ea8790ec2745807480c36e7216f023951638fa")
EXPECTED_NO_GLU_PATCH_SHA256 = (
    "af1ead7982f2149268637c758502c7f6db81d5cdf2b0cbba905d2c47bddf524e")
EXPECTED_PRODUCT_TOOL_SHA256 = (
    "baa6cb5591766eb91dcb1456d0195216f10a4fafb9477fc3a357f8eb98a8c3b1")
EXPECTED_TRANSFORM_SOURCE_SHA256 = (
    "abbe70c6ed19abce6e6ae7ee586072436b9e3efdd8aaed3bdd3adeec09d73055")
EXPECTED_FC_SOURCE_SHA256 = (
    "1944c1af859c2ccd416a481da8d0bd336bbe39ad9a4bca0aed9ea56182b7996f")
EXPECTED_BUILD_PLUGIN_SHA256 = (
    "9165f6aa9c31f43b7554c65161e2534bf42ff250b5ad97b51c83e37e2d51ffcd")
EXPECTED_PAIR_PLUGIN_SHA256 = EXPECTED_BUILD_PLUGIN_SHA256
EXPECTED_ACCEPTED_PLUGIN_SHA256 = (
    "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985")


def load_module(name: str, path: Path) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


COMMON = load_module("iq36_pair_no_glu_build_common", COMMON_TOOL)
for required_common_name in (
    "meminfo", "monitor_build", "plugin_record", "run", "source_record"):
  if not hasattr(COMMON, required_common_name):
    raise RuntimeError(
        f"{COMMON_TOOL} lacks required helper {required_common_name}")


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


def source_record(path: Path, relative: Path) -> dict[str, Any]:
  stat = path.stat()
  status = COMMON.run(
      ["git", "status", "--short", "--", str(relative)], SOURCE_TREE)
  return {
      "path": str(path),
      "sha256": sha256(path),
      "size_bytes": stat.st_size,
      "mtime_ns": stat.st_mtime_ns,
      "target_status": status,
  }


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
      COMMON_TOOL, OUTCOME_GATE, NO_GLU_PATCH, PRODUCT_TOOL, SOURCE_TREE,
      TRANSFORM_SOURCE, FC_SOURCE, BUILD_TREE, BUILD_PLUGIN, PAIR_PLUGIN,
      ACCEPTED_PLUGIN, CMAKE)
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit(
        "missing pair-no-GLU build inputs: " + ", ".join(missing))

  head = COMMON.run(["git", "rev-parse", "HEAD"])["stdout"].strip()
  origin_main = COMMON.run(
      ["git", "rev-parse", "origin/main"])["stdout"].strip()
  repo_status = COMMON.run(
      ["git", "status", "--porcelain"])["stdout"].splitlines()
  gate_ancestor = COMMON.run(
      ["git", "merge-base", "--is-ancestor", OUTCOME_GATE_COMMIT, head])
  source_commit = COMMON.run(
      ["git", "rev-parse", "HEAD"], SOURCE_TREE)["stdout"].strip()
  forward_check = COMMON.run(
      ["git", "apply", "--check", str(NO_GLU_PATCH)], SOURCE_TREE)
  reverse_check_before = COMMON.run(
      ["git", "apply", "--reverse", "--check", str(NO_GLU_PATCH)],
      SOURCE_TREE)
  outcome = load_json(OUTCOME_GATE)
  transform_before = source_record(TRANSFORM_SOURCE, TRANSFORM_REL)
  fc_before = COMMON.source_record()
  build_before = COMMON.plugin_record(BUILD_PLUGIN)
  pair_before = COMMON.plugin_record(PAIR_PLUGIN)
  accepted_before = COMMON.plugin_record(ACCEPTED_PLUGIN)
  start_memory = COMMON.meminfo()
  preflight_bytes = int(args.min_available_gib * 1024**3)
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  patch_text = NO_GLU_PATCH.read_text(encoding="utf-8")
  product_text = PRODUCT_TOOL.read_text(encoding="utf-8")

  source_checks = [
      check("repository_is_clean_pushed_and_contains_outcome_gate",
            not repo_status and head == origin_main and
            gate_ancestor["returncode"] == 0,
            head=head, origin_main=origin_main, status=repo_status,
            outcome_gate_is_ancestor=gate_ancestor["returncode"] == 0),
      check("seq2216_admits_one_patch_and_serial_build_only",
            sha256(OUTCOME_GATE) == EXPECTED_OUTCOME_SHA256 and
            outcome.get("required_checks_passed") is True and
            outcome.get("verdict") ==
                "reject_pair_with_glu_admit_one_decomposed_glu_patch_and_"
                "serial_build" and
            outcome.get("pair_with_glu_closed") is True and
            outcome.get("source_patch_admitted") is True and
            outcome.get("serial_plugin_build_admitted") is True and
            outcome.get("gpu_worker_admitted") is False and
            outcome.get("model_worker_admitted") is False and
            outcome.get("git", {}).get("head") == OUTCOME_GATE_COMMIT,
            outcome_gate_sha256=sha256(OUTCOME_GATE),
            outcome_verdict=outcome.get("verdict")),
      check("current_cumulative_sources_are_exact_preimages",
            source_commit == PINNED_SOURCE_COMMIT and
            transform_before["sha256"] ==
                EXPECTED_TRANSFORM_SOURCE_SHA256 and
            fc_before["sha256"] == EXPECTED_FC_SOURCE_SHA256,
            source_commit=source_commit,
            transform_before=transform_before, fc_before=fc_before),
      check("incremental_no_glu_patch_is_forward_only_and_pinned",
            sha256(NO_GLU_PATCH) == EXPECTED_NO_GLU_PATCH_SHA256 and
            forward_check["returncode"] == 0 and
            reverse_check_before["returncode"] != 0 and
            patch_text.count("diff --git ") == 1 and
            "transformations_pipeline.cpp" in patch_text and
            'std::getenv("IQ36_ROUTER_SHARED_PAIR")' in patch_text and
            "-        manager.register_pass<ov::pass::GLUFusion>();" in
                patch_text,
            patch_sha256=sha256(NO_GLU_PATCH),
            forward_check=forward_check,
            reverse_check_before=reverse_check_before),
      check("candidate_pair_runtime_switch_contract_is_pinned",
            sha256(PRODUCT_TOOL) == EXPECTED_PRODUCT_TOOL_SHA256 and
            "--fuse-router-shared-pair" in product_text and
            product_text.count("IQ36_ROUTER_SHARED_PAIR") == 3,
            product_tool_sha256=sha256(PRODUCT_TOOL)),
      check("mutable_build_pair_and_accepted_carriers_are_exact",
            build_before["sha256"] == EXPECTED_BUILD_PLUGIN_SHA256 and
            pair_before["sha256"] == EXPECTED_PAIR_PLUGIN_SHA256 and
            accepted_before["sha256"] == EXPECTED_ACCEPTED_PLUGIN_SHA256,
            build_plugin_before=build_before, pair_plugin_before=pair_before,
            accepted_plugin_before=accepted_before),
      check("eight_gib_preflight_is_available",
            int(start_memory.get("MemAvailable", 0)) >= preflight_bytes,
            available_bytes=start_memory.get("MemAvailable"),
            preflight_bytes=preflight_bytes),
      check("isolated_output_cannot_overwrite_existing_carriers",
            candidate_plugin not in (
                BUILD_PLUGIN.resolve(), PAIR_PLUGIN.resolve(),
                ACCEPTED_PLUGIN.resolve()) and
            not candidate_plugin.exists(),
            candidate_plugin=str(candidate_plugin)),
      check("no_gpu_context_model_or_inference_worker_is_admitted", True,
            gpu_contexts=0, model_workers=0, infer_requests=0,
            inference_workers=0),
  ]
  source_apply_admitted = all(row["pass"] for row in source_checks)
  apply_result = (
      COMMON.run(["git", "apply", str(NO_GLU_PATCH)], SOURCE_TREE)
      if source_apply_admitted else {
          "command": ["git", "apply", str(NO_GLU_PATCH)],
          "returncode": 125,
          "stdout": "",
          "stderr": "source gate failed",
          "skipped": True,
      })

  transform_after = source_record(TRANSFORM_SOURCE, TRANSFORM_REL)
  fc_after = COMMON.source_record()
  reverse_check_after = COMMON.run(
      ["git", "apply", "--reverse", "--check", str(NO_GLU_PATCH)],
      SOURCE_TREE)
  forward_check_after = COMMON.run(
      ["git", "apply", "--check", str(NO_GLU_PATCH)], SOURCE_TREE)
  source_text = TRANSFORM_SOURCE.read_text(encoding="utf-8")
  apply_checks = [
      check("one_incremental_no_glu_patch_application_succeeds",
            source_apply_admitted and apply_result["returncode"] == 0 and
            transform_after["sha256"] != transform_before["sha256"],
            apply_result=apply_result,
            transform_before_sha256=transform_before["sha256"],
            transform_after_sha256=transform_after["sha256"]),
      check("applied_source_is_reverse_only_and_structurally_exact",
            reverse_check_after["returncode"] == 0 and
            forward_check_after["returncode"] != 0 and
            source_text.count("#include <cstdlib>") == 1 and
            source_text.count(
                'std::getenv("IQ36_ROUTER_SHARED_PAIR")') == 1 and
            source_text.count(
                "manager.register_pass<ov::pass::GLUFusion>();") == 1 and
            fc_after == fc_before,
            reverse_check_after=reverse_check_after,
            forward_check_after=forward_check_after,
            transform_after=transform_after, fc_after=fc_after),
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
  pair_after = COMMON.plugin_record(PAIR_PLUGIN)
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
      check("new_plugin_is_copied_to_isolated_seq2217_carrier",
            candidate.get("sha256") == build_after["sha256"] and
            candidate.get("size_bytes") == build_after["size_bytes"] and
            candidate.get("sha256") not in (
                EXPECTED_BUILD_PLUGIN_SHA256,
                EXPECTED_ACCEPTED_PLUGIN_SHA256),
            candidate_plugin=candidate,
            build_plugin_after=build_after),
      check("seq2212_and_accepted_carriers_remain_bitwise_immutable",
            pair_after == pair_before and accepted_after == accepted_before,
            pair_plugin_before=pair_before, pair_plugin_after=pair_after,
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
      "admit_pair_decomposed_glu_plugin_for_compile_gate"
      if passed else "repair_pair_decomposed_glu_source_or_build")
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
      "outcome_gate": {
          "path": display(OUTCOME_GATE),
          "sha256": sha256(OUTCOME_GATE),
          "verdict": outcome.get("verdict"),
      },
      "transform_before": transform_before,
      "transform_after_apply": transform_after,
      "fc_source": fc_after,
      "source_checks": source_checks,
      "source_apply": apply_result,
      "apply_checks": apply_checks,
      "build": build,
      "build_plugin_before": build_before,
      "build_plugin_after": build_after,
      "pair_plugin_before": pair_before,
      "pair_plugin_after": pair_after,
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
          "route": "pair_decomposed_glu_compile_gate",
          "requirements": [
              "push a candidate-only compile gate before loading the plugin",
              "enable Q/K plus pair and keep triple/fixed manager off",
              "require pair FC331 with GLU0 and split topology",
              "create no InferRequest until compile evidence passes",
              "require output130 exact correctness before timing",
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
          display(OUTCOME_GATE): sha256(OUTCOME_GATE),
          display(NO_GLU_PATCH): sha256(NO_GLU_PATCH),
          display(PRODUCT_TOOL): sha256(PRODUCT_TOOL),
          str(TRANSFORM_SOURCE): transform_after["sha256"],
          str(FC_SOURCE): fc_after["sha256"],
          str(PAIR_PLUGIN): pair_after["sha256"],
          str(ACCEPTED_PLUGIN): accepted_after["sha256"],
      },
      "candidate_plugin": candidate,
      "compiler_builds": result["workers"]["compiler_builds"],
      "gpu_contexts": 0,
      "model_workers": 0,
      "infer_requests": 0,
      "inference_workers": 0,
  })
  report = f"""# N=1024 pair decomposed-GLU serial build

Verdict: **{verdict}**. Required checks:
`{str(passed).lower()}`.

Seq2216 admitted exactly one incremental source application and one serial
plugin build. The transformation source changed from
`{transform_before['sha256']}` to `{transform_after['sha256']}`. The `-j1`
build completed in `{float(build['elapsed_seconds']):.3f} s` and produced
isolated candidate `{candidate.get('sha256')}`.

Peak process-group RSS/swap was
`{int(monitor.get('process_group_rss_peak_bytes', 0))}/`
`{int(monitor.get('process_group_swap_peak_bytes', 0))} B`; minimum available
memory was `{int(monitor.get('system_available_min_bytes', 0))} B`. No guard
or OOM fired. Seq2212 and accepted seq2189 plugins remained bitwise unchanged.
No GPU context, model worker, InferRequest, or inference worker ran.
"""
  (output / "report.md").write_text(report, encoding="utf-8")
  print(json.dumps({
      "artifact": display(output),
      "verdict": verdict,
      "required_checks_passed": passed,
      "elapsed_seconds": build["elapsed_seconds"],
      "candidate_plugin_sha256": candidate.get("sha256"),
      "transform_after_sha256": transform_after["sha256"],
      "peak_rss_bytes": monitor.get("process_group_rss_peak_bytes"),
      "minimum_available_bytes": monitor.get("system_available_min_bytes"),
      "oom_observed": build["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
