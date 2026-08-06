#!/usr/bin/env python3
"""Build and run the engine-side 40-layer external-attention loop shell compare."""

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
SCHEMA_VERSION = "intel-qwen36-r1-engine-loop-shell-compare-v0"
ENGINE_STDOUT_SCHEMA = "intel-qwen36-engine-loop-shell-compare-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_ENV_SCRIPT = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
DEFAULT_REMOTE_ROOT = "/home/intel/intel-qwen36-r1"
DEFAULT_ORACLE_BUNDLE = ROOT / "oracle/r0-oracle-bundle-20260627T060028Z"
LAYER_COUNT = 40

SOURCE_FILES = [
    ("engine/include/intel_qwen36/gguf_loader.hpp", "include/intel_qwen36/gguf_loader.hpp"),
    ("engine/src/gguf_loader.cpp", "src/gguf_loader.cpp"),
    ("engine/tests/loop_shell_compare.cpp", "tests/loop_shell_compare.cpp"),
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
  parser.add_argument("--timeout-s", type=int, default=1800)
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


def sha256_json(value: Any) -> str:
  data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(data).hexdigest()


def payload_record(path: Path) -> dict[str, Any]:
  return {
      "path": str(path.relative_to(ROOT)),
      "sha256": sha256_file(path),
      "size_bytes": path.stat().st_size,
      "_path": path,
  }


def resolve_payload(oracle_bundle: Path, relative_payload: str, expected_size: int) -> Path:
  payload_path = (oracle_bundle / relative_payload).resolve()
  if not payload_path.exists():
    raise SystemExit(f"oracle loop shell payload missing: {payload_path}")
  if payload_path.stat().st_size != expected_size:
    raise SystemExit(f"oracle loop shell payload size mismatch: {payload_path}")
  return payload_path


def _find_row(
    rows: list[dict[str, Any]],
    boundary_type: str,
    tensor_kind: str,
    layer: int | None = None,
) -> dict[str, Any]:
  row = next(
      (
          item for item in rows
          if item.get("boundary_type") == boundary_type
          and item.get("tensor_kind") == tensor_kind
          and (layer is None or item.get("layer") == layer)
      ),
      None,
  )
  if not isinstance(row, dict):
    label = "global" if layer is None else f"L{layer}"
    raise SystemExit(f"oracle bundle missing {label} {boundary_type} {tensor_kind} row")
  return row


def _reference_path(row: dict[str, Any], kind: str) -> str:
  key = f"reference_{kind}_tensor_path"
  value = row.get(key)
  if not isinstance(value, str):
    raise SystemExit(f"oracle row missing {key}")
  return value


def _side_path(row: dict[str, Any], kind: str, label: str) -> str:
  key = f"reference_{kind}_tensor_paths"
  values = row.get(key)
  if not isinstance(values, dict) or label not in values:
    raise SystemExit(f"oracle row missing {label} in {key}")
  return values[label]


def resolve_reference(oracle_bundle: Path) -> dict[str, Any]:
  oracle_bundle = oracle_bundle.resolve()
  inputs = load_jsonl(oracle_bundle / "boundary-references/inputs.jsonl")
  outputs = load_jsonl(oracle_bundle / "boundary-references/outputs.jsonl")

  layer_input = _find_row(inputs, "layer_input_rmsnorm", "input", 0)
  residual_input = payload_record(
      resolve_payload(oracle_bundle, _reference_path(layer_input, "input"), 8192)
  )

  layers = []
  source_case = layer_input.get("source_prompt_case_id")
  source_token = layer_input.get("source_token_position")
  for layer in range(LAYER_COUNT):
    attn_projection_input = _find_row(
        inputs, "attention_output_projection", "input", layer
    )
    attn_projection_output = _find_row(
        outputs, "attention_output_projection", "output", layer
    )
    layer_residual_input = _find_row(inputs, "layer_input_rmsnorm", "input", layer)
    attn_residual = _find_row(outputs, "post_attention_residual", "output", layer)
    router = _find_row(outputs, "router_topk", "output", layer)
    layer_output = _find_row(outputs, "moe_residual", "output", layer)
    for row in (
        layer_residual_input,
        attn_projection_input,
        attn_projection_output,
        attn_residual,
        router,
        layer_output,
    ):
      if (
          row.get("source_prompt_case_id") != source_case
          or row.get("source_token_position") != source_token
      ):
        raise SystemExit(f"oracle loop shell L{layer} source case/token mismatch")

    projection_name = attn_projection_input.get("shape_metadata", {}).get("tensor_name")
    output_paths = attn_projection_output.get("reference_output_tensor_paths", {})
    if not isinstance(projection_name, str):
      raise SystemExit(f"oracle loop shell L{layer} projection input name missing")
    if projection_name.startswith("final_output-"):
      attention_output_label = f"linear_attn_out-{layer}"
      layer_kind = "linear_ssm"
      attention_output_size = 8192
    elif projection_name.startswith("attn_gated-"):
      attention_output_label = f"attn_output-{layer}"
      layer_kind = "full_attention"
      attention_output_size = 8192
    else:
      raise SystemExit(f"oracle loop shell L{layer} unexpected projection input: {projection_name}")
    if attention_output_label not in output_paths:
      raise SystemExit(f"oracle loop shell L{layer} missing {attention_output_label}")

    layers.append({
        "layer_index": layer,
        "layer_kind": layer_kind,
        "attention_projection_input_name": projection_name,
        "residual_input": payload_record(resolve_payload(
            oracle_bundle,
            _reference_path(layer_residual_input, "input"),
            8192,
        )),
        "attention_projection_input": payload_record(resolve_payload(
            oracle_bundle,
            _reference_path(attn_projection_input, "input"),
            16384,
        )),
        "attention_output": payload_record(resolve_payload(
            oracle_bundle,
            output_paths[attention_output_label],
            attention_output_size,
        )),
        "attention_residual": payload_record(resolve_payload(
            oracle_bundle,
            _reference_path(attn_residual, "output"),
            8192,
        )),
        "topk": payload_record(resolve_payload(
            oracle_bundle,
            _side_path(router, "output", f"ffn_moe_topk-{layer}"),
            32,
        )),
        "layer_output": payload_record(resolve_payload(
            oracle_bundle,
            _reference_path(layer_output, "output"),
            8192,
        )),
    })

  final_norm = _find_row(outputs, "final_norm", "output")
  lm_head = _find_row(outputs, "lm_head", "output")
  sampler = _find_row(outputs, "sampler", "output")
  globals_payloads = {
      "result_norm": payload_record(resolve_payload(
          oracle_bundle, _reference_path(final_norm, "output"), 8192
      )),
      "result_output": payload_record(resolve_payload(
          oracle_bundle, _reference_path(lm_head, "output"), 993280
      )),
      "sampler_topk": payload_record(resolve_payload(
          oracle_bundle, _reference_path(sampler, "output"), 380
      )),
  }
  return {
      "oracle_bundle": str(oracle_bundle.relative_to(ROOT)),
      "residual_input": residual_input,
      "layers": layers,
      "globals": globals_payloads,
      "source_prompt_case_id": source_case,
      "source_token_position": source_token,
  }


def strip_private_paths(value: Any) -> Any:
  if isinstance(value, dict):
    return {
        key: strip_private_paths(item)
        for key, item in value.items()
        if key != "_path"
    }
  if isinstance(value, list):
    return [strip_private_paths(item) for item in value]
  return value


def contract_payloads(ref: dict[str, Any], layer_payload_manifest: dict[str, Any]) -> dict[str, Any]:
  return {
      "residual_input": strip_private_paths(ref["residual_input"]),
      "globals": strip_private_paths(ref["globals"]),
      "layer_payload_manifest_path": layer_payload_manifest["path"],
      "layer_payload_manifest_sha256": layer_payload_manifest["sha256"],
      "layer_payload_counts": {
          "residual_input": LAYER_COUNT,
          "attention_projection_input": LAYER_COUNT,
          "attention_output": LAYER_COUNT,
          "attention_residual": LAYER_COUNT,
          "topk": LAYER_COUNT,
          "layer_output": LAYER_COUNT,
      },
  }


def summarize_layers(parsed: dict[str, Any]) -> dict[str, Any]:
  layers = parsed.get("layers", [])
  if not isinstance(layers, list):
    layers = []
  selected = []
  for layer_index in (0, 3, 39):
    if layer_index < len(layers) and isinstance(layers[layer_index], dict):
      layer = layers[layer_index]
      selected.append({
          "layer_index": layer_index,
          "attention_output_max_abs_diff": layer.get("attention_output_comparison", {}).get("max_abs_diff"),
          "attention_residual_max_abs_diff": layer.get("attention_residual_comparison", {}).get("max_abs_diff"),
          "layer_output_max_abs_diff": layer.get("layer_output_comparison", {}).get("max_abs_diff"),
          "topk_mismatch_count": layer.get("topk_comparison", {}).get("mismatch_count"),
      })
  return {
      "selected_layers": selected,
      "topk_mismatch_total": parsed.get("topk_mismatch_total"),
      "max_attention_output_abs_diff": parsed.get("max_attention_output_abs_diff"),
      "max_attention_residual_abs_diff": parsed.get("max_attention_residual_abs_diff"),
      "max_layer_output_abs_diff": parsed.get("max_layer_output_abs_diff"),
      "final_norm": summarize_comparison(parsed.get("final_norm_comparison", {})),
      "logits": summarize_comparison(parsed.get("logits_comparison", {})),
      "sampler": {
          "mismatch_count": parsed.get("sampler_comparison", {}).get("mismatch_count"),
          "native_top_token_id": (
              parsed.get("native_sampler_topk", [{}])[0].get("token_id")
              if isinstance(parsed.get("native_sampler_topk"), list)
              and parsed.get("native_sampler_topk")
              else None
          ),
          "oracle_top_token_id": (
              parsed.get("expected_sampler_topk", [{}])[0].get("token_id")
              if isinstance(parsed.get("expected_sampler_topk"), list)
              and parsed.get("expected_sampler_topk")
              else None
          ),
      },
  }


def summarize_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
  return {
      "cosine": comparison.get("cosine"),
      "max_abs_diff": comparison.get("max_abs_diff"),
      "mean_abs_diff": comparison.get("mean_abs_diff"),
      "mismatch_count": comparison.get("mismatch_count"),
      "rmse": comparison.get("rmse"),
  }


def vector_counts(parsed: dict[str, Any]) -> dict[str, Any]:
  result = {}
  for key, value in parsed.items():
    if isinstance(value, dict) and key.endswith("_vector") and "count" in value:
      result[key.removesuffix("_vector")] = value.get("count")
  return result


def compare_passed(parsed: dict[str, Any], build: dict[str, Any], compare: dict[str, Any], model_path: str) -> bool:
  return (
      build.get("returncode") == 0
      and compare.get("returncode") == 0
      and parsed.get("schema_version") == ENGINE_STDOUT_SCHEMA
      and parsed.get("model_path") == model_path
      and parsed.get("load_map_ready") is True
      and parsed.get("passed") is True
      and parsed.get("layer_count") == LAYER_COUNT
      and parsed.get("residual_mode") == "teacher_forced_oracle"
      and parsed.get("topk_mismatch_total") == 0
      and parsed.get("sampler_comparison", {}).get("mismatch_count") == 0
  )


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
  state = payload["engine_loop_shell_compare"]
  comparisons = state["comparisons"]
  rows = [
      ("engine_loop_shell_compare_passed", payload["engine_loop_shell_compare_passed"]),
      ("topk_mismatch_total", comparisons["topk_mismatch_total"]),
      ("max_attention_output_abs_diff", comparisons["max_attention_output_abs_diff"]),
      ("max_attention_residual_abs_diff", comparisons["max_attention_residual_abs_diff"]),
      ("max_layer_output_abs_diff", comparisons["max_layer_output_abs_diff"]),
      ("final_norm_max_abs_diff", comparisons["final_norm"]["max_abs_diff"]),
      ("logits_max_abs_diff", comparisons["logits"]["max_abs_diff"]),
      ("sampler_token_id_mismatch_count", comparisons["sampler"]["mismatch_count"]),
      ("r1_native_correctness_gate_closed", False),
  ]
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_engine_loop_shell_compare",
          "value": value,
      }, sort_keys=True) + "\n")


def build_summary(payload: dict[str, Any]) -> str:
  state = payload["engine_loop_shell_compare"]
  comparisons = state["comparisons"]
  lines = [
      "# R1 Engine Loop Shell Compare",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- oracle bundle: `{payload['oracle_bundle']}`",
      f"- source case/token: `{state['source_prompt_case_id']}` token position {state['source_token_position']}",
      f"- layer count: {state['layer_count']}",
      f"- residual mode: `{state['residual_mode']}`",
      f"- top-k mismatch total: {comparisons['topk_mismatch_total']}",
      f"- max attention output abs diff: {comparisons['max_attention_output_abs_diff']}",
      f"- max attention residual abs diff: {comparisons['max_attention_residual_abs_diff']}",
      f"- max layer output abs diff: {comparisons['max_layer_output_abs_diff']}",
      f"- final norm max abs diff: {comparisons['final_norm']['max_abs_diff']}",
      f"- logits max abs diff: {comparisons['logits']['max_abs_diff']}",
      f"- sampler mismatch count: {comparisons['sampler']['mismatch_count']}",
      f"- Loop shell compare passed: `{str(payload['engine_loop_shell_compare_passed']).lower()}`",
      "",
      "This artifact runs the 40-layer engine shell with oracle residual and",
      "attention projection inputs in teacher-forced mode. It validates O(1)",
      "layer-shell reuse, FFN/MoE composition, final norm, LM head, and",
      "deterministic sampler top-k, but it does not implement attention/SSM",
      "or KV state updates and does not close native token correctness.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-engine-loop-shell-compare-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  remote_dir = f"{args.remote_root}/engine-loop-shell-compare-{stamp}"
  ref = resolve_reference(args.oracle_bundle)

  layer_payload_manifest_local = {
      "schema_version": f"{SCHEMA_VERSION}.layer-payloads",
      "layers": strip_private_paths(ref["layers"]),
  }
  layer_payload_manifest_path = out_dir / "layer-payloads.json"
  write_json(layer_payload_manifest_path, layer_payload_manifest_local)
  layer_payload_manifest_record = {
      "path": str(layer_payload_manifest_path.relative_to(ROOT)),
      "sha256": sha256_json(layer_payload_manifest_local),
      "size_bytes": layer_payload_manifest_path.stat().st_size,
  }

  remote_payloads: dict[str, str] = {
      "residual_input": f"{remote_dir}/oracle/residual_input.bin",
      "result_norm": f"{remote_dir}/oracle/result_norm.bin",
      "result_output": f"{remote_dir}/oracle/result_output.bin",
      "sampler_topk": f"{remote_dir}/oracle/sampler-topk.json",
  }
  for layer in range(LAYER_COUNT):
    suffix = f"{layer:02d}"
    remote_payloads[f"residual_input_{suffix}"] = (
        f"{remote_dir}/oracle/residual_input_{suffix}.bin"
    )
    remote_payloads[f"attention_projection_input_{suffix}"] = (
        f"{remote_dir}/oracle/attention_projection_input_{suffix}.bin"
    )
    remote_payloads[f"attention_output_{suffix}"] = (
        f"{remote_dir}/oracle/attention_output_{suffix}.bin"
    )
    remote_payloads[f"attention_residual_{suffix}"] = (
        f"{remote_dir}/oracle/attention_residual_{suffix}.bin"
    )
    remote_payloads[f"topk_{suffix}"] = f"{remote_dir}/oracle/topk_{suffix}.bin"
    remote_payloads[f"layer_output_{suffix}"] = (
        f"{remote_dir}/oracle/layer_output_{suffix}.bin"
    )

  mkdir = run_target(
      args.host,
      "mkdir -p " + " ".join(
          shlex.quote(f"{remote_dir}/{subdir}")
          for subdir in ("include/intel_qwen36", "src", "tests", "build", "oracle")
      ),
      args.timeout_s,
  )
  source_transfers: list[dict[str, Any]] = []
  payload_transfers: dict[str, dict[str, Any]] = {
      label: {"returncode": 1, "stdout": "", "stderr": "stage failed"}
      for label in remote_payloads
  }
  if mkdir["returncode"] == 0:
    for local, remote in SOURCE_FILES:
      source_transfers.append(
          copy_to(args.host, ROOT / local, f"{remote_dir}/{remote}", args.timeout_s)
      )
    payload_transfers["residual_input"] = copy_to(
        args.host, ref["residual_input"]["_path"], remote_payloads["residual_input"], args.timeout_s
    )
    payload_transfers["result_norm"] = copy_to(
        args.host, ref["globals"]["result_norm"]["_path"], remote_payloads["result_norm"], args.timeout_s
    )
    payload_transfers["result_output"] = copy_to(
        args.host, ref["globals"]["result_output"]["_path"], remote_payloads["result_output"], args.timeout_s
    )
    payload_transfers["sampler_topk"] = copy_to(
        args.host, ref["globals"]["sampler_topk"]["_path"], remote_payloads["sampler_topk"], args.timeout_s
    )
    for layer in range(LAYER_COUNT):
      suffix = f"{layer:02d}"
      layer_ref = ref["layers"][layer]
      for local_key, remote_key in (
          ("residual_input", f"residual_input_{suffix}"),
          ("attention_projection_input", f"attention_projection_input_{suffix}"),
          ("attention_output", f"attention_output_{suffix}"),
          ("attention_residual", f"attention_residual_{suffix}"),
          ("topk", f"topk_{suffix}"),
          ("layer_output", f"layer_output_{suffix}"),
      ):
        payload_transfers[remote_key] = copy_to(
            args.host,
            layer_ref[local_key]["_path"],
            remote_payloads[remote_key],
            args.timeout_s,
        )

  build_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      "g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic "
      f"-I {shlex.quote(remote_dir + '/include')} "
      f"{shlex.quote(remote_dir + '/src/gguf_loader.cpp')} "
      f"{shlex.quote(remote_dir + '/tests/loop_shell_compare.cpp')} "
      f"-o {shlex.quote(remote_dir + '/build/iq36-loop-shell-compare')}",
  ])
  staged = (
      mkdir["returncode"] == 0
      and all(item["returncode"] == 0 for item in source_transfers)
      and all(item["returncode"] == 0 for item in payload_transfers.values())
  )
  build = (
      run_target(args.host, f"bash -lc {shlex.quote(build_command)}", args.timeout_s)
      if staged else {"returncode": 1, "stdout": "", "stderr": "stage failed"}
  )
  compare_command = " ".join([
      shlex.quote(remote_dir + "/build/iq36-loop-shell-compare"),
      shlex.quote(args.model),
      shlex.quote(remote_dir + "/oracle"),
      shlex.quote(remote_payloads["sampler_topk"]),
      "--teacher-forced-residuals",
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
  state = {
      "boundary_type": "loop_shell",
      "comparisons": summarize_layers(parsed) if parsed else {},
      "engine_stdout_schema_version": ENGINE_STDOUT_SCHEMA,
      "input_payload_path": ref["residual_input"]["path"],
      "input_payload_sha256": ref["residual_input"]["sha256"],
      "input_payload_size_bytes": ref["residual_input"]["size_bytes"],
      "layer_count": LAYER_COUNT,
      "output_payload_path": ref["globals"]["result_output"]["path"],
      "output_payload_sha256": ref["globals"]["result_output"]["sha256"],
      "output_payload_size_bytes": ref["globals"]["result_output"]["size_bytes"],
      "payloads": contract_payloads(ref, layer_payload_manifest_record),
      "residual_mode": parsed.get("residual_mode") if parsed else "teacher_forced_oracle",
      "source_prompt_case_id": ref["source_prompt_case_id"],
      "source_token_position": ref["source_token_position"],
      "target_build_returncode": build.get("returncode"),
      "target_compare_returncode": compare.get("returncode"),
      "vector_counts": vector_counts(parsed) if parsed else {},
  }
  payload = {
      "created_at": created_at,
      "engine_loop_shell_compare": state,
      "engine_loop_shell_compare_passed": passed,
      "host": args.host,
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "input_payload_path": state["input_payload_path"],
      "input_payload_sha256": state["input_payload_sha256"],
      "model_path": args.model,
      "oracle_bundle": ref["oracle_bundle"],
      "output_payload_path": state["output_payload_path"],
      "output_payload_sha256": state["output_payload_sha256"],
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-engine-loop-shell-compare.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "stage.json", {
      "mkdir": mkdir,
      "payload_transfer_count": len(payload_transfers),
      "payload_transfers": payload_transfers,
      "remote_dir": remote_dir,
      "source_files": SOURCE_FILES,
      "source_transfers": source_transfers,
  })
  write_json(out_dir / "build.json", build)
  write_json(
      out_dir / "loop-shell-stdout.json",
      parsed if parsed else {
          "parse_error": parse_error,
          "raw_stdout": compare.get("stdout", ""),
          "schema_version": ENGINE_STDOUT_SCHEMA,
      },
  )
  write_json(out_dir / "compare.json", {
      **payload,
      "parse_error": parse_error,
      "target_build": build,
      "target_compare": compare,
  })
  checks = [
      {"name": "target_stage_created", "pass": mkdir.get("returncode") == 0},
      {
          "name": "source_files_transferred",
          "pass": bool(source_transfers) and all(
              item.get("returncode") == 0 for item in source_transfers
          ),
      },
      {
          "name": "oracle_loop_shell_payloads_transferred",
          "pass": all(item.get("returncode") == 0 for item in payload_transfers.values()),
      },
      {"name": "target_engine_loop_shell_compare_built", "pass": build.get("returncode") == 0},
      {"name": "target_engine_loop_shell_compare_ran", "pass": compare.get("returncode") == 0},
      {"name": "target_engine_loop_shell_compare_output_parsed", "pass": bool(parsed)},
      {"name": "loop_shell_matches_oracle_payloads", "pass": passed},
      {"name": "does_not_close_native_token_correctness", "pass": True},
  ]
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "engine_loop_shell_compare_passed": passed,
      "gate": "r1_engine_loop_shell_compare",
      "r1_native_correctness_gate_closed": False,
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "speedup_claims_allowed": False,
      "workstream": WORKSTREAM,
  })
  write_metrics(out_dir / "metrics.jsonl", payload)
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 engine loop shell compare output: {out_dir}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
