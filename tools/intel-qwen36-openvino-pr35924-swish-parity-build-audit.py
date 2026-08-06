#!/usr/bin/env python3
"""Audit seq2236 after its compile-step counter false negative."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-pr35924-swish-parity-build-audit-v0"
BUILD_METRICS = ROOT / (
    "output/openvino-pr35924-swish-parity-build-"
    "20260731Tseq2236-clean/metrics.json")
ONEDNN_STDOUT = ROOT / (
    "output/openvino-pr35924-swish-parity-build-"
    "20260731Tseq2236-clean/raw/build-onednn-pr35924-parity.stdout")
PLUGIN_STDOUT = ROOT / (
    "output/openvino-pr35924-swish-parity-build-"
    "20260731Tseq2236-clean/raw/build-plugin-pr35924-parity.stdout")
PARITY_PATCH = ROOT / (
    "engine/openvino/"
    "iq36-onednn-grouped-postops-openvino-swish-parity.patch")
GENERATOR = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu/"
    "src/gpu/intel/matmul/grouped_post_ops_gen.cpp")
ARCHIVE = Path(
    "/home/intel/intel-qwen36-r0/build/openvino-90214e-l0-gpu/"
    "src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/"
    "libopenvino_onednn_gpu.a")
CANDIDATE = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2236/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
SEQ2233 = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2233/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
CONTROL = Path(
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2189/bin/intel64/Release/"
    "libopenvino_intel_gpu_plugin.so")
EXPECTED = {
    BUILD_METRICS:
        "80daaf01a008ef4e7de614bd75b185c412aa8bb06329756b2681c78185c86145",
    ONEDNN_STDOUT:
        "a0c9ace6f1dc60b9430cb471e0dce54f89dca6bcca1b4eccbe31e884fd7a1317",
    PLUGIN_STDOUT:
        "5d28261ff026b92bd5d43a2ae3033c5da7a68b08d013ef45bd3395f1119f58cb",
    PARITY_PATCH:
        "13d6e22aca83ef93cb79991d2eab6d3b1931a2a0a2e1a67de6e644d4a1e9bf0f",
    GENERATOR:
        "9c3fef12c5d0f8ea8706f0da0280bb92a3f344274aea83e861c45faa1d6eb19c",
    ARCHIVE:
        "5208f431ece909a82732a496e51117a2fcae899db22f9deae1a2be4ba36e660f",
    CANDIDATE:
        "7827246114e095dca42887458a9bdbc505635e22ea5d3ba6a32c990ff555dcda",
    SEQ2233:
        "c66c9be61ee31110a55c8a064ed1390bd3d21a3f1766a03fdea84a078a519849",
    CONTROL:
        "9832adffa8bf3fd7b013c47d9d2abefa66987c5fbde350c9669fce58c697b985",
}
ABORT_BYTES = 4 * 1024**3


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  return parser.parse_args()


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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True)


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  missing = [str(path) for path in EXPECTED if not path.is_file()]
  if missing:
    raise SystemExit("missing parity audit inputs: " + ", ".join(missing))

  status = run(["git", "status", "--porcelain"]).stdout.splitlines()
  head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
  origin = run(["git", "rev-parse", "origin/main"]).stdout.strip()
  metrics = load_json(BUILD_METRICS)
  failed_checks = [
      row for row in metrics.get("checks", [])
      if row.get("pass") is not True]
  build_check = next(
      (row for row in failed_checks
       if row.get("name") ==
           "sole_serial_onednn_and_plugin_builds_succeed"), {})
  onednn_stdout = ONEDNN_STDOUT.read_text(
      encoding="utf-8", errors="replace")
  plugin_stdout = PLUGIN_STDOUT.read_text(
      encoding="utf-8", errors="replace")
  parity_reverse = run([
      "git", "-C", str(GENERATOR.parents[4]), "apply", "--reverse",
      "--check", str(PARITY_PATCH)])
  hashes = {str(path): sha256(path) for path in EXPECTED}
  stages = (
      build_check.get("onednn_build") or {},
      build_check.get("plugin_build") or {})

  checks = [
      check("repository_clean_and_pushed_at_audit",
            not status and head == origin,
            commit=head, upstream_commit=origin, dirty_paths=status),
      check("seq2236_artifacts_are_hash_exact",
            all(hashes[str(path)] == expected
                for path, expected in EXPECTED.items()),
            hashes=hashes),
      check("only_failure_is_counter_expectation",
            len(failed_checks) == 1 and
            build_check.get("compile_steps") == 1 and
            all(stage.get("returncode") == 0 and
                stage.get("timed_out") is False and
                stage.get("memory_guard_tripped") is False and
                stage.get("oom_observed") is False
                for stage in stages) and
            build_check.get("archive_after", {}).get("sha256") ==
                EXPECTED[ARCHIVE] and
            build_check.get("build_after", {}).get("sha256") ==
                EXPECTED[CANDIDATE],
            failed_checks=failed_checks),
      check("one_source_compile_plus_two_links_are_real",
            "Building CXX object" in onednn_stdout and
            "grouped_post_ops_gen.cpp.o" in onednn_stdout and
            "Linking CXX static library" in onednn_stdout and
            "Linking CXX shared module" in plugin_stdout,
            note=(
                "compile_steps intentionally counts compiler invocations; "
                "the prior >=2 expectation incorrectly counted linking")),
      check("parity_source_is_materialized",
            parity_reverse.returncode == 0 and
            "convert_float(convert_half(v))" in
                GENERATOR.read_text(encoding="utf-8") and
            "native_exp(-(" in GENERATOR.read_text(encoding="utf-8"),
            reverse_apply_returncode=parity_reverse.returncode,
            reverse_apply_stderr=parity_reverse.stderr.strip()),
      check("memory_and_oom_guards_held",
            int(metrics.get("minimum_available_bytes", 0)) >= ABORT_BYTES and
            all(not stage.get("oom_observed", False) and
                not stage.get("memory_guard_tripped", False)
                for stage in stages),
            minimum_available_bytes=metrics.get(
                "minimum_available_bytes"),
            abort_bytes=ABORT_BYTES),
      check("audit_uses_no_compile_gpu_model_or_inference", True,
            configure_invocations=0, compiler_invocations=0,
            gpu_contexts_created=0, model_workers_started=0,
            infer_requests_created=0, inference_workers_started=0),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_seq2236_for_one_output130_correctness_worker"
      if required else
      "repair_seq2236_parity_build_or_audit")
  payload = {
      "schema": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_output130_correctness_worker_admitted": required,
      "performance_worker_admitted": False,
      "checks": checks,
      "candidate_plugin": {
          "path": str(CANDIDATE), "sha256": sha256(CANDIDATE)},
      "next_action": (
          "run one candidate-only output130 correctness worker; only exact "
          "tokens plus KLD/top-1 and owner-census pass may admit timing"),
  }
  write_json(output / "metrics.json", payload)
  (output / "report.md").write_text(
      "# PR35924 Swish-parity build audit\n\n"
      f"Verdict: **{verdict}**. Required checks: "
      f"`{str(required).lower()}`.\n\n"
      "Seq2236 compiled the sole changed generator object, linked the oneDNN "
      "archive, and relinked the GPU plugin. The build gate's only failure "
      "was expecting two compiler invocations when one source file changed; "
      "linking is not a compiler invocation. No build was repeated.\n",
      encoding="utf-8")
  print(json.dumps({
      "artifact": str(output.relative_to(ROOT)),
      "verdict": verdict,
      "required_checks_passed": required,
      "candidate_plugin_sha256": sha256(CANDIDATE),
  }, separators=(",", ":")), flush=True)
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
