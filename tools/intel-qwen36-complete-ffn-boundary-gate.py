#!/usr/bin/env python3
"""Capture and validate the fixed layer-27 complete FFN boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-complete-ffn-boundary-gate-v0"
LAYER = 27
TOKENS = 1024
HIDDEN = 2048
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
TOKEN_INPUT = ROOT / (
    "output/r2-native-matrix-20260629T011942Z/token-input/"
    "prefill_shape_008k.tokens.u32")
PRIOR_CAPTURE = ROOT / (
    "output/onednn-q4k-routed-moe-component-gate-"
    "20260711Tseq646cleanZ/raw/capture")
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
LLAMA_COMMIT = "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
LLAMA_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/"
    "llama-qwen36-boundary-capture-noflash-20260629T234151Z")
CAPTURE_SOURCE = ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"
TOKEN_SHA256 = "8a3554ce47f204926f29b898eee2dd17d3f849f73ab8094c05b4f96a17b35ad8"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--tokens", type=Path, default=TOKEN_INPUT)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--llama-source", type=Path, default=LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=LLAMA_BUILD)
  parser.add_argument("--threads", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  args = parser.parse_args()
  if args.threads <= 0 or args.timeout_s <= 0:
    parser.error("threads and timeout-s must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/complete-ffn-boundary-gate-{stamp}"
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise RuntimeError(f"expected JSON object: {path}")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise RuntimeError(f"expected JSON object: {path}:{line_number}")
    rows.append(value)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(root: Path, *parts: str) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=root, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False,
        timeout=timeout_s, encoding="utf-8", errors="replace")
    return {
        "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command, "returncode": 124,
        "stdout": error.stdout or "", "stderr": error.stderr or "",
        "timed_out": True,
    }


def shell_run(command: list[str], args: argparse.Namespace) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0 && " +
      shlex.join(command))
  return run(["bash", "-lc", shell], args.timeout_s)


def write_run(raw: Path, label: str, row: dict[str, Any]) -> None:
  (raw / f"{label}.command.json").write_text(
      json.dumps({
          "command": row["command"], "returncode": row["returncode"],
          "timed_out": row["timed_out"],
      }, indent=2) + "\n", encoding="utf-8")
  (raw / f"{label}.stdout").write_text(
      str(row["stdout"]), encoding="utf-8")
  (raw / f"{label}.stderr").write_text(
      str(row["stderr"]), encoding="utf-8")


def read_f32(path: Path) -> array[float]:
  values = array("f")
  with path.open("rb") as handle:
    values.fromfile(handle, path.stat().st_size // values.itemsize)
  if sys.byteorder != "little":
    values.byteswap()
  return values


def compare_generated(
    observed: array[float], count: int, expected_at: Callable[[int], float],
) -> dict[str, Any]:
  if len(observed) != count:
    return {"finite": False, "count": len(observed), "expected_count": count}
  sum_expected2 = 0.0
  sum_observed2 = 0.0
  sum_diff2 = 0.0
  dot = 0.0
  max_abs = 0.0
  finite = True
  for index, observed_value in enumerate(observed):
    expected = expected_at(index)
    if not math.isfinite(expected) or not math.isfinite(observed_value):
      finite = False
      continue
    difference = float(observed_value) - expected
    max_abs = max(max_abs, abs(difference))
    sum_diff2 += difference * difference
    sum_expected2 += expected * expected
    sum_observed2 += float(observed_value) * float(observed_value)
    dot += expected * float(observed_value)
  relative_l2 = (
      math.sqrt(sum_diff2 / sum_expected2) if sum_expected2 > 0 else math.inf)
  cosine = (
      dot / math.sqrt(sum_expected2 * sum_observed2)
      if sum_expected2 > 0 and sum_observed2 > 0 else 0.0)
  return {
      "count": count, "finite": finite, "max_abs_diff": max_abs,
      "relative_l2": relative_l2, "cosine": cosine,
      "pass": finite and relative_l2 <= 2e-6 and cosine >= 0.999999,
  }


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  capture = raw / "capture"
  raw.mkdir(parents=True, exist_ok=False)

  required = [
      args.model, args.tokens, args.env_script, args.cxx, args.llama_source,
      args.llama_build, CAPTURE_SOURCE, MODEL_CONTRACT,
      PRIOR_CAPTURE / "tensor-dumps.jsonl",
      args.llama_build / "bin/libllama.so.0.0.1",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing inputs: " + ", ".join(missing))

  contract = load_json(MODEL_CONTRACT)
  commit = git_output(ROOT, "rev-parse", "HEAD")
  dirty = git_output(ROOT, "status", "--porcelain")
  llama_commit = git_output(args.llama_source, "rev-parse", "HEAD")
  llama_boundary_dirty = git_output(
      args.llama_source, "status", "--porcelain", "--",
      "src/models/qwen3next.cpp")
  model_size = args.model.stat().st_size
  token_hash = sha256_file(args.tokens)

  binary = raw / "complete-ffn-capture"
  library_dir = args.llama_build / "bin"
  build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DGGML_BACKEND_SHARED", "-DGGML_SHARED", "-DGGML_USE_CPU",
      "-DLLAMA_SHARED", f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}", str(CAPTURE_SOURCE),
      f"-L{library_dir}", f"-Wl,-rpath,{library_dir}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1", "-Wl,-l:libggml-base.so.0.13.1",
      "-fopenmp", "-pthread", "-o", str(binary),
  ]
  build = shell_run(build_command, args)
  write_run(raw, "build", build)

  capture_command = [
      str(binary), "--model", str(args.model),
      "--token-ids-file", str(args.tokens), "--binary-u32-token-file",
      "--token-count", str(TOKENS), "--batch-all", "--component-layer",
      str(LAYER), "--component-through-ffn-out", "--out-dir", str(capture),
      "--case-id", "prefill_shape_008k_tile1024_layer27_complete_ffn",
      "--threads", str(args.threads), "--n-ctx", "2048", "--ngl", "0",
      "--top-k", "1", "--predicts-generated-position", "0",
  ]
  capture_run = (
      shell_run(capture_command, args) if build["returncode"] == 0 else
      {"command": capture_command, "returncode": 125, "stdout": "",
       "stderr": "capture build failed", "timed_out": False})
  write_run(raw, "capture", capture_run)

  rows = (
      load_jsonl(capture / "tensor-dumps.jsonl")
      if capture_run["returncode"] == 0 else [])
  by_name = {str(row.get("tensor_name")): row for row in rows}
  expected = {
      f"attn_post_norm-{LAYER}": ("f32", [HIDDEN, TOKENS, 1, 1]),
      f"ffn_moe_topk-{LAYER}": ("i32", [8, TOKENS, 1, 1]),
      f"ffn_moe_swiglu-{LAYER}": ("f32", [512, 8, TOKENS, 1]),
      f"ffn_moe_weights_norm-{LAYER}": ("f32", [8, TOKENS, 1, 1]),
      f"ffn_moe_down-{LAYER}": ("f32", [HIDDEN, 8, TOKENS, 1]),
      f"ffn_moe_out-{LAYER}": ("f32", [HIDDEN, TOKENS, 1, 1]),
      f"ffn_shexp-{LAYER}": ("f32", [HIDDEN, TOKENS, 1, 1]),
      f"shared_expert_gate-{LAYER}": ("f32", [1, TOKENS, 1, 1]),
      f"shared_expert_gate_sigmoid-{LAYER}": ("f32", [1, TOKENS, 1, 1]),
      f"ffn_shexp_gated-{LAYER}": ("f32", [HIDDEN, TOKENS, 1, 1]),
      f"ffn_out-{LAYER}": ("f32", [HIDDEN, TOKENS, 1, 1]),
  }
  metadata_ok = set(by_name) == set(expected)
  payloads: dict[str, Path] = {}
  payload_hashes: dict[str, str] = {}
  if metadata_ok:
    for name, (tensor_type, shape) in expected.items():
      row = by_name[name]
      path = capture / str(row.get("payload_path"))
      row_ok = (
          row.get("tensor_type") == tensor_type and row.get("ne") == shape and
          path.is_file() and path.stat().st_size == int(row.get("nbytes", -1)))
      metadata_ok = metadata_ok and row_ok
      if row_ok:
        payloads[name] = path
        payload_hashes[name] = sha256_file(path)

  prior_rows = {
      str(row["tensor_name"]): row
      for row in load_jsonl(PRIOR_CAPTURE / "tensor-dumps.jsonl")}
  prior_hashes = {
      name: sha256_file(PRIOR_CAPTURE / str(row["payload_path"]))
      for name, row in prior_rows.items()
  }
  routed_names = set(prior_hashes)
  routed_identity_ok = (
      routed_names.issubset(payload_hashes) and
      all(payload_hashes[name] == value
          for name, value in prior_hashes.items()))

  algebra: dict[str, Any] = {}
  if metadata_ok:
    moe = read_f32(payloads[f"ffn_moe_out-{LAYER}"])
    shared = read_f32(payloads[f"ffn_shexp-{LAYER}"])
    gate = read_f32(payloads[f"shared_expert_gate-{LAYER}"])
    sigmoid = read_f32(
        payloads[f"shared_expert_gate_sigmoid-{LAYER}"])
    shared_gated = read_f32(payloads[f"ffn_shexp_gated-{LAYER}"])
    final = read_f32(payloads[f"ffn_out-{LAYER}"])
    sigmoid_compare = compare_generated(
        sigmoid, TOKENS,
        lambda index: 1.0 / (1.0 + math.exp(-float(gate[index]))))
    shared_compare = compare_generated(
        shared_gated, TOKENS * HIDDEN,
        lambda index: float(shared[index]) * float(sigmoid[index // HIDDEN]))
    final_compare = compare_generated(
        final, TOKENS * HIDDEN,
        lambda index: float(moe[index]) + float(shared_gated[index]))
    algebra = {
        "sigmoid": sigmoid_compare,
        "shared_gate_apply": shared_compare,
        "routed_plus_shared": final_compare,
    }

  checks = [
      check("repository_clean_at_gate", dirty == "", dirty_paths=dirty.splitlines()),
      check("pinned_llama_boundary_source_exact",
            llama_commit == LLAMA_COMMIT and llama_boundary_dirty == "",
            expected=LLAMA_COMMIT, observed=llama_commit,
            boundary_dirty_paths=llama_boundary_dirty.splitlines()),
      check("locked_model_path_and_size", str(args.model.resolve()) ==
            contract["model"]["gguf_model_path"] and model_size ==
            int(contract["model"]["gguf_model_size_bytes"]),
            observed_size=model_size),
      check("locked_token_input_hash", token_hash == TOKEN_SHA256,
            observed=token_hash, expected=TOKEN_SHA256),
      check("contract_requires_shared_expert_and_moe_residual",
            "shared_expert" in contract["boundary_types"] and
            "moe_residual" in contract["boundary_types"]),
      check("capture_source_builds", build["returncode"] == 0),
      check("complete_ffn_capture_executes", capture_run["returncode"] == 0),
      check("exact_eleven_tensor_boundary_captured", metadata_ok,
            observed_names=sorted(by_name), expected_names=sorted(expected)),
      check("routed_prefix_is_byte_identical_to_seq646", routed_identity_ok,
            compared_names=sorted(routed_names)),
      check("shared_gate_sigmoid_algebra", algebra.get("sigmoid", {}).get("pass") is True,
            metrics=algebra.get("sigmoid", {})),
      check("shared_gate_apply_algebra", algebra.get("shared_gate_apply", {}).get("pass") is True,
            metrics=algebra.get("shared_gate_apply", {})),
      check("final_ffn_out_add_algebra", algebra.get("routed_plus_shared", {}).get("pass") is True,
            metrics=algebra.get("routed_plus_shared", {})),
  ]
  passed = all(row["pass"] for row in checks)
  disposition = (
      "accept_layer27_complete_ffn_capture_for_microkernel_source_gate"
      if passed else "reject_incomplete_or_inconsistent_complete_ffn_capture")
  selected = (
      "native_prefill_f16_u4_active_expert_microkernel_complete_ffn_source_gate"
      if passed else "repair_complete_ffn_capture_boundary")
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "commit": commit,
      "evaluation_completed": capture_run["returncode"] == 0,
      "required_checks_passed": passed,
      "disposition": disposition,
      "selected_next_route": selected,
      "checks": checks,
      "algebra": algebra,
      "capture": {
          "case_id": "prefill_shape_008k_tile1024_layer27_complete_ffn",
          "layer": LAYER, "tokens": TOKENS, "tensor_count": len(rows),
          "payload_sha256": payload_hashes,
          "prior_routed_capture": str(PRIOR_CAPTURE.relative_to(ROOT)),
          "oracle_end": "ffn_out",
      },
      "contract": {
          "complete_ffn_input": "attn_post_norm",
          "routed_output": "ffn_moe_out",
          "shared_output": "ffn_shexp_gated",
          "complete_ffn_output": "ffn_out",
          "speedup_claims_allowed": False,
      },
  }
  (out / "result.json").write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out / "summary.md").write_text("\n".join([
      "# Complete FFN boundary gate", "",
      f"- disposition: `{disposition}`",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- captured tensors: `{len(rows)}`",
      f"- routed prefix byte-identical to seq646: `{str(routed_identity_ok).lower()}`",
      f"- final-add relative L2: `{algebra.get('routed_plus_shared', {}).get('relative_l2')}`",
      f"- selected next route: `{selected}`", "",
      "This is a boundary/correctness artifact, not a performance claim.", "",
  ]), encoding="utf-8")
  print(json.dumps(result, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
