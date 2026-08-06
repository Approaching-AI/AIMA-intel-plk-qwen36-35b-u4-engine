#!/usr/bin/env python3
"""Build and run the engine-side L0 MoE residual compare."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-engine-moe-residual-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-moe-residual-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/moe_residual_compare.cpp", "tests/moe_residual_compare.cpp"),
]

PAYLOAD_ORDER = [
    "attn_post_norm",
    "ffn_moe_down",
    "weights_norm",
    "ffn_moe_weighted",
    "ffn_shexp",
    "shared_gate",
    "shared_gate_sigmoid",
    "ffn_shexp_gated",
    "ffn_out",
    "attn_residual",
    "moe_residual",
]

EXPECTED_VECTOR_COUNTS = {
    "attn_post_norm": 2048,
    "attn_residual": 2048,
    "ffn_moe_down": 16384,
    "ffn_moe_weighted": 16384,
    "ffn_out": 2048,
    "ffn_shexp": 2048,
    "ffn_shexp_gated": 2048,
    "moe_residual": 2048,
    "oracle_moe_residual": 2048,
    "shared_gate": 1,
    "shared_gate_sigmoid": 1,
    "weights_norm": 8,
}

COMPARISON_KEYS = [
    "ffn_moe_weighted",
    "shared_gate",
    "shared_gate_sigmoid",
    "ffn_shexp_gated",
    "ffn_out",
    "moe_residual",
]


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--env-script", default=DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=DEFAULT_REMOTE_ROOT)
  parser.add_argument("--oracle-bundle", type=Path, default=DEFAULT_ORACLE_BUNDLE)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=300)
  return parser.parse_args()


def run(cmd: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    return {
        "cmd": cmd,
        "returncode": 124,
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") + "\ntimeout",
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
    }
  return {
      "cmd": cmd,
      "returncode": proc.returncode,
      "stderr": proc.stderr,
      "stdout": proc.stdout,
  }


def run_target(host: str, remote_command: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_command, timeout_s)


def copy_to(host: str, local_path: Path, remote_path: str, timeout_s: int) -> dict[str, Any]:
  return iq36_local.copy_to(host, local_path, remote_path, timeout_s)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open("r", encoding="utf-8") as fh:
    for line_number, line in enumerate(fh, start=1):
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number}: row must be an object")
      rows.append(row)
  return rows


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def resolve_payload(oracle_bundle: Path, relative_payload: str, expected_size: int) -> Path:
  payload_path = (oracle_bundle / relative_payload).resolve()
  if not payload_path.exists():
    raise SystemExit(f"oracle MoE residual payload missing: {payload_path}")
  if payload_path.stat().st_size != expected_size:
    raise SystemExit(f"oracle MoE residual payload size mismatch: {payload_path}")
  return payload_path


def payload_record(path: Path) -> dict[str, Any]:
  return {
      "path": str(path.relative_to(ROOT)),
      "sha256": sha256_file(path),
      "size_bytes": path.stat().st_size,
      "_path": path,
  }


def _find_row(
    rows: list[dict[str, Any]],
    boundary_type: str,
    tensor_kind: str,
    layer: int,
) -> dict[str, Any]:
  row = next(
      (
          item for item in rows
          if item.get("boundary_type") == boundary_type
          and item.get("layer") == layer
          and item.get("tensor_kind") == tensor_kind
      ),
      None,
  )
  if not isinstance(row, dict):
    raise SystemExit(f"oracle bundle missing L{layer} {boundary_type} {tensor_kind} row")
  return row


def _require_shape(row: dict[str, Any], nbytes: int, ne: list[int], tensor_name: str, op: str) -> None:
  shape = row.get("shape_metadata", {})
  if shape.get("nbytes") != nbytes:
    raise SystemExit(f"oracle {tensor_name} nbytes mismatch")
  if shape.get("ne") != ne:
    raise SystemExit(f"oracle {tensor_name} shape mismatch")
  if shape.get("tensor_name") != tensor_name:
    raise SystemExit(f"oracle {tensor_name} tensor name mismatch")
  if shape.get("tensor_op") != op:
    raise SystemExit(f"oracle {tensor_name} tensor op mismatch")


def resolve_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")
  router_output = _find_row(outputs, "router_topk", "output", 0)
  selected_down_output = _find_row(outputs, "selected_expert_down", "output", 0)
  shared_input = _find_row(inputs, "shared_expert", "input", 0)
  shared_output = _find_row(outputs, "shared_expert", "output", 0)
  moe_input = _find_row(inputs, "moe_residual", "input", 0)
  moe_output = _find_row(outputs, "moe_residual", "output", 0)

  _require_shape(shared_input, 8192, [2048, 1, 1, 1], "attn_post_norm-0", "MUL")
  _require_shape(
      selected_down_output,
      65536,
      [2048, 8, 1, 1],
      "ffn_moe_down-0",
      "MUL_MAT_ID",
  )
  _require_shape(shared_output, 8192, [2048, 1, 1, 1], "ffn_shexp-0", "MUL_MAT")
  _require_shape(moe_input, 8192, [2048, 1, 1, 1], "ffn_out-0", "ADD")
  _require_shape(moe_output, 8192, [2048, 1, 1, 1], "moe_residual-0", "ADD")

  router_paths = router_output.get("reference_output_tensor_paths", {})
  selected_paths = selected_down_output.get("reference_output_tensor_paths", {})
  shared_paths = shared_output.get("reference_output_tensor_paths", {})
  moe_output_paths = moe_output.get("reference_output_tensor_paths", {})
  if not isinstance(router_paths, dict) or "ffn_moe_weights_norm-0" not in router_paths:
    raise SystemExit("oracle L0 router normalized weights missing")
  if not isinstance(selected_paths, dict) or "ffn_moe_weighted-0" not in selected_paths:
    raise SystemExit("oracle L0 selected expert weighted output missing")
  for required in (
      "ffn_shexp-0",
      "shared_expert_gate-0",
      "shared_expert_gate_sigmoid-0",
      "ffn_shexp_gated-0",
  ):
    if not isinstance(shared_paths, dict) or required not in shared_paths:
      raise SystemExit(f"oracle L0 shared expert side output missing: {required}")
  if not isinstance(moe_output_paths, dict) or "attn_residual-0" not in moe_output_paths:
    raise SystemExit("oracle L0 MoE residual attn residual input missing")

  source_case = moe_output.get("source_prompt_case_id")
  source_token = moe_output.get("source_token_position")
  for row in (router_output, selected_down_output, shared_input, shared_output, moe_input):
    if (
        row.get("source_prompt_case_id") != source_case
        or row.get("source_token_position") != source_token
    ):
      raise SystemExit("oracle L0 MoE residual source case/token mismatch")

  payload_specs = {
      "attn_post_norm": (shared_input["reference_input_tensor_path"], 8192),
      "ffn_moe_down": (selected_paths["ffn_moe_down-0"], 65536),
      "weights_norm": (router_paths["ffn_moe_weights_norm-0"], 32),
      "ffn_moe_weighted": (selected_paths["ffn_moe_weighted-0"], 65536),
      "ffn_shexp": (shared_paths["ffn_shexp-0"], 8192),
      "shared_gate": (shared_paths["shared_expert_gate-0"], 4),
      "shared_gate_sigmoid": (shared_paths["shared_expert_gate_sigmoid-0"], 4),
      "ffn_shexp_gated": (shared_paths["ffn_shexp_gated-0"], 8192),
      "ffn_out": (moe_input["reference_input_tensor_path"], 8192),
      "attn_residual": (moe_output_paths["attn_residual-0"], 8192),
      "moe_residual": (moe_output["reference_output_tensor_path"], 8192),
  }
  payloads = {
      label: payload_record(resolve_payload(oracle_bundle, rel_path, size))
      for label, (rel_path, size) in payload_specs.items()
  }
  return {
      "boundary_type": "moe_residual",
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "payloads": payloads,
      "source_prompt_case_id": source_case,
      "source_token_position": source_token,
  }


def comparison_passed(comparison: dict[str, Any]) -> bool:
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0
      and comparison.get("max_abs_diff") <= 5e-3
      and comparison.get("rmse") <= 5e-4
      and comparison.get("cosine") >= 0.999
  )


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  tensor = parsed.get("tensor", {})
  vector_names = {
      "attn_post_norm": "attn_post_norm_vector",
      "attn_residual": "attn_residual_vector",
      "ffn_moe_down": "ffn_moe_down_vector",
      "ffn_moe_weighted": "ffn_moe_weighted_vector",
      "ffn_out": "ffn_out_vector",
      "ffn_shexp": "ffn_shexp_vector",
      "ffn_shexp_gated": "ffn_shexp_gated_vector",
      "moe_residual": "moe_residual_vector",
      "oracle_moe_residual": "oracle_moe_residual_vector",
      "shared_gate": "shared_gate_vector",
      "shared_gate_sigmoid": "shared_gate_sigmoid_vector",
      "weights_norm": "weights_norm_vector",
  }
  vectors_ok = True
  for label, parsed_key in vector_names.items():
    vector = parsed.get(parsed_key, {})
    vectors_ok = vectors_ok and vector.get("count") == EXPECTED_VECTOR_COUNTS[label]
    vectors_ok = vectors_ok and vector.get("finite") is True
    vectors_ok = vectors_ok and vector.get("nonzero") is True
  comparisons_ok = all(
      comparison_passed(parsed.get(f"{label}_comparison", {}))
      for label in COMPARISON_KEYS
  )
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and tensor.get("name") == "blk.0.ffn_gate_inp_shexp.weight"
      and tensor.get("type_name") == "F32"
      and tensor.get("dims") == [2048]
      and tensor.get("shape_ok") is True
      and vectors_ok
      and comparisons_ok
  )


def contract_payloads(ref: dict[str, Any]) -> dict[str, dict[str, Any]]:
  result = {}
  for label, record in ref["payloads"].items():
    result[label] = {
        "path": record["path"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }
  return result


def comparison_summary(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
  result = {}
  for label in COMPARISON_KEYS:
    comparison = parsed.get(f"{label}_comparison", {})
    result[label] = {
        "max_abs_diff": comparison.get("max_abs_diff"),
        "mean_abs_diff": comparison.get("mean_abs_diff"),
        "rmse": comparison.get("rmse"),
        "cosine": comparison.get("cosine"),
        "mismatch_count": comparison.get("mismatch_count"),
    }
  return result


def vector_counts(parsed: dict[str, Any]) -> dict[str, Any]:
  parsed_keys = {
      "attn_post_norm": "attn_post_norm_vector",
      "attn_residual": "attn_residual_vector",
      "ffn_moe_down": "ffn_moe_down_vector",
      "ffn_moe_weighted": "ffn_moe_weighted_vector",
      "ffn_out": "ffn_out_vector",
      "ffn_shexp": "ffn_shexp_vector",
      "ffn_shexp_gated": "ffn_shexp_gated_vector",
      "moe_residual": "moe_residual_vector",
      "oracle_moe_residual": "oracle_moe_residual_vector",
      "shared_gate": "shared_gate_vector",
      "shared_gate_sigmoid": "shared_gate_sigmoid_vector",
      "weights_norm": "weights_norm_vector",
  }
  return {
      label: parsed.get(parsed_key, {}).get("count")
      for label, parsed_key in parsed_keys.items()
  }


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_moe_residual_compare"]
  rows = [
      ("engine_moe_residual_compare_passed", payload["engine_moe_residual_compare_passed"]),
      ("ffn_out_max_abs_diff", state["comparisons"]["ffn_out"]["max_abs_diff"]),
      ("ffn_out_rmse", state["comparisons"]["ffn_out"]["rmse"]),
      ("moe_residual_max_abs_diff", state["comparisons"]["moe_residual"]["max_abs_diff"]),
      ("moe_residual_rmse", state["comparisons"]["moe_residual"]["rmse"]),
      ("moe_residual_cosine", state["comparisons"]["moe_residual"]["cosine"]),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_moe_residual_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_moe_residual_compare"]
  ffn_out = state["comparisons"]["ffn_out"]
  moe_residual = state["comparisons"]["moe_residual"]
  lines = [
      "# R1 Engine MoE Residual Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- shared gate tensor: `{state.get('tensor_name')}` {state.get('tensor_type')} {state.get('tensor_dims')}",
      f"- ffn_out max abs diff: {ffn_out.get('max_abs_diff')}",
      f"- ffn_out RMSE: {ffn_out.get('rmse')}",
      f"- moe_residual max abs diff: {moe_residual.get('max_abs_diff')}",
      f"- moe_residual RMSE: {moe_residual.get('rmse')}",
      f"- moe_residual cosine: {moe_residual.get('cosine')}",
      f"- MoE residual compare passed: `{str(payload['engine_moe_residual_compare_passed']).lower()}`",
      "",
      "This artifact records whether the engine-side C++ MoE weighting,",
      "shared-expert gating, FFN output add, and post-FFN residual add reproduce",
      "the L0 `ffn_out-0` and derived `moe_residual-0` tensors for the locked",
      "prompt token. It does not run the full model loop, emit native candidate",
      "token rows, or allow speedup claims.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-moe-residual-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-moe-residual-compare-{stamp}"
  ref = resolve_reference(args.oracle_bundle)

  remote_payloads = {
      label: f"{remote_dir}/oracle/{ref['payloads'][label]['_path'].name}"
      for label in PAYLOAD_ORDER
  }
  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      label: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for label in PAYLOAD_ORDER
  }
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      transfers.append(copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s))
    for label in PAYLOAD_ORDER:
      payload_transfers[label] = copy_to(
          args.host,
          ref["payloads"][label]["_path"],
          remote_payloads[label],
          args.timeout_s,
      )

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/moe_residual_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-moe-residual-compare')}",
  ])
  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in transfers)
      and all(item["returncode"] == 0 for item in payload_transfers.values())
  )
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-moe-residual-compare"),
      shlex.quote(args.model),
      *(shlex.quote(remote_payloads[label]) for label in PAYLOAD_ORDER),
  ])
  compare = (
      run_target(args.host, compare_command, args.timeout_s)
      if build["returncode"] == 0
      else {"returncode": 1, "stdout": "", "stderr": "build failed"}
  )

  parsed: dict[str, Any] = {}
  parse_error = None
  if compare.get("stdout"):
    try:
      parsed = json.loads(compare["stdout"])
    except json.JSONDecodeError as exc:
      parse_error = str(exc)

  passed = bool(parsed) and compare_passed(parsed, build, compare, args.model)
  payload = {
      "created_at": created_at,
      "engine_moe_residual_compare": {
          "boundary_type": ref["boundary_type"],
          "comparisons": comparison_summary(parsed) if parsed else {},
          "engine_stdout_schema_version": parsed.get("schema_version") if parsed else None,
          "payloads": contract_payloads(ref),
          "source_prompt_case_id": ref["source_prompt_case_id"],
          "source_token_position": ref["source_token_position"],
          "target_build_returncode": build.get("returncode"),
          "target_compare_returncode": compare.get("returncode"),
          "tensor_dims": parsed.get("tensor", {}).get("dims") if parsed else None,
          "tensor_name": parsed.get("tensor", {}).get("name") if parsed else None,
          "tensor_type": parsed.get("tensor", {}).get("type_name") if parsed else None,
          "vector_counts": vector_counts(parsed) if parsed else {},
      },
      "engine_moe_residual_compare_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "parse_error": parse_error,
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "target_build": {
          "returncode": build.get("returncode"),
          "stderr": build.get("stderr"),
          "stdout": build.get("stdout"),
      },
      "target_compare": {
          "returncode": compare.get("returncode"),
          "stderr": compare.get("stderr"),
          "stdout": compare.get("stdout"),
      },
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "payloads": contract_payloads(ref),
      "remote_dir": remote_dir,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-moe-residual-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "payload_transfers": payload_transfers,
      "remote_dir": remote_dir,
      "remote_payloads": remote_payloads,
      "source_files": SOURCE_FILES,
      "transfer_results": transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(out_dir / "moe-residual-stdout.json", parsed if parsed else {"parse_error": parse_error})
  write_json(out_dir / "compare.json", payload)
  checks = [
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {"name": "source_files_transferred", "pass": all(item.get("returncode") == 0 for item in transfers)},
      {"name": "target_engine_moe_residual_compare_built", "pass": build.get("returncode") == 0},
      {"name": "target_engine_moe_residual_compare_ran", "pass": compare.get("returncode") == 0},
      {"name": "target_engine_moe_residual_compare_output_parsed", "pass": bool(parsed)},
      {"name": "moe_residual_matches_oracle_payload", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  checks.extend(
      {"name": f"oracle_moe_residual_{label}_payload_transferred", "pass": payload_transfers[label].get("returncode") == 0}
      for label in PAYLOAD_ORDER
  )
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "engine_moe_residual_compare_passed": passed,
      "gate": "r1_engine_moe_residual_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": passed,
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine moe residual compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
