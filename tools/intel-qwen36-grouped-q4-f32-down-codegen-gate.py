#!/usr/bin/env python3
"""Generate one pinned grouped-Q4 down binary with an F32 destination."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-grouped-q4-f32-down-codegen-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
PARENT_GATE_PATH = ROOT / "tools/intel-qwen36-grouped-s8-u4-prefill-gate.py"


def load_parent() -> Any:
  spec = importlib.util.spec_from_file_location("iq36_q4_parent", PARENT_GATE_PATH)
  if spec is None or spec.loader is None:
    raise SystemExit(f"could not import {PARENT_GATE_PATH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


PARENT = load_parent()
BASE = PARENT.BASE


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=BASE.DEFAULT_MODEL)
  parser.add_argument("--capture", type=Path, default=PARENT.DEFAULT_CAPTURE)
  parser.add_argument("--tensor-index", type=Path,
                      default=BASE.DEFAULT_TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path,
                      default=BASE.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=BASE.DEFAULT_CXX)
  parser.add_argument("--onednn-source", type=Path,
                      default=BASE.DEFAULT_ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path,
                      default=PARENT.DEFAULT_ONEDNN_BUILD)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--skip-onednn-build", action="store_true")
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.jobs, args.timeout_s) <= 0:
    parser.error("jobs and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/grouped-q4-f32-down-codegen-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(root: Path, *args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=root, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output(ROOT, "status", "--porcelain")
  return {"commit": git_output(ROOT, "rev-parse", "HEAD"),
          "dirty": bool(dirty), "dirty_paths": dirty.splitlines()}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = [
      args.model, args.capture / "tensor-dumps.jsonl", args.tensor_index,
      args.env_script, args.cxx, args.onednn_source, args.onednn_build,
      args.onednn_build / "src/libdnnl.so",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      PARENT.PREP_SOURCE, PARENT.ONEDNN_PATCH, PARENT_GATE_PATH]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))
  if BASE.sha256_file(args.model) != BASE.MODEL_SHA256:
    raise SystemExit("locked model hash mismatch")

  created_at = iso_now()
  state = git_state()
  source_commit = git_output(args.onednn_source, "rev-parse", "HEAD")
  source_diff = PARENT.git_bytes(
      args.onednn_source, "diff", "--unified=0", "--",
      *PARENT.PATCHED_ONEDNN_PATHS)
  source_status = PARENT.git_bytes(
      args.onednn_source, "status", "--porcelain").decode("utf-8")
  expected_status = sorted(
      f" M {path}" for path in PARENT.PATCHED_ONEDNN_PATHS)
  patch_exact = (
      source_diff == PARENT.ONEDNN_PATCH.read_bytes() and
      sorted(source_status.splitlines()) == expected_status)

  build_command = [
      "cmake", "--build", str(args.onednn_build), "--target", "dnnl",
      "-j", str(args.jobs)]
  build = ({"command": build_command, "returncode": 0, "stdout": "",
            "stderr": "skipped by request", "timed_out": False}
           if args.skip_onednn_build else
           PARENT.shell_run_env(
               build_command, args.env_script, args.timeout_s))
  BASE.write_run_logs(raw, "onednn-build", build)

  generator = raw / "offline-prepack-generator"
  generator_build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", str(PARENT.PREP_SOURCE),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(generator)]
  generator_build = (
      PARENT.shell_run_env(
          generator_build_command, args.env_script, args.timeout_s)
      if build["returncode"] == 0 else
      PARENT.failed_run(generator_build_command, "oneDNN build failed"))
  BASE.write_run_logs(raw, "generator-build", generator_build)

  tensors = BASE.tensor_rows(args.tensor_index)
  metadata, payloads = BASE.captured_payloads(args.capture, True)
  topk_stride = int(metadata[f"ffn_moe_topk-{BASE.LAYER}"]["nb"][1])
  generator_command = PARENT.generator_command(
      generator, args, tensors, payloads, topk_stride,
      PARENT.ADR_0015_KERNEL_CAP_US)
  prefix = raw / "q4-down-f32"
  generation = (
      PARENT.shell_run_env(generator_command, args.env_script, args.timeout_s, {
          "DNNL_PRIMITIVE_CACHE_CAPACITY": "0",
          "IQ36_GENERATE_S8_GROUPED": "1",
          "IQ36_GROUPED_F32_DOWN_OUTPUT": "1",
          "IQ36_GROUPED_FUSED_KIND": "down",
          "IQ36_DUMP_FUSED_PROGRAM_PREFIX": str(prefix),
          "IQ36_EXIT_AFTER_FUSED_DUMP": "1"})
      if generator_build["returncode"] == 0 else
      PARENT.failed_run(generator_command, "generator build failed"))
  BASE.write_run_logs(raw, "generate-q4-down-f32", generation)
  binary = raw / "q4-down-f32.0.bin"
  binary_bytes = binary.stat().st_size if binary.is_file() else -1
  binary_sha256 = sha256(binary) if binary_bytes > 0 else None

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("locked_model_path_and_hash",
            args.model.resolve() == BASE.DEFAULT_MODEL.resolve() and
            BASE.sha256_file(args.model) == BASE.MODEL_SHA256),
      check("pinned_onednn_source_and_exact_patch",
            source_commit == BASE.ONEDNN_COMMIT and patch_exact,
            source_commit=source_commit, patch_exact=patch_exact),
      check("generator_builds", generator_build["returncode"] == 0),
      check("f32_down_binary_generated",
            generation["returncode"] == 0 and binary_bytes > 0,
            binary_bytes=binary_bytes, sha256=binary_sha256),
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state, "model": str(args.model),
      "onednn_commit": source_commit,
      "binary": str(binary), "binary_bytes": binary_bytes,
      "binary_sha256": binary_sha256,
      "destination_type": "f32", "fused_kind": "down",
      "checks": checks, "required_checks_passed": passed,
      "disposition": (
          "accept_q4_f32_down_codegen_capability"
          if passed else "reject_q4_f32_down_codegen_capability"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed, "speedup_claims_allowed": False})
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out), "git": state,
      "required_checks_passed": passed, "speedup_claims_allowed": False})
  (out / "summary.md").write_text("\n".join([
      "# Grouped Q4 F32 down codegen", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- binary bytes: `{binary_bytes}`",
      f"- binary SHA-256: `{binary_sha256}`", "",
      "This proves offline F32-destination binary generation only. Component,",
      "token, context, and product-speed gates remain open.", "",
  ]), encoding="utf-8")
  print(json.dumps({"artifact": str(out), "pass": passed,
                    "binary": str(binary), "sha256": binary_sha256},
                   sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
