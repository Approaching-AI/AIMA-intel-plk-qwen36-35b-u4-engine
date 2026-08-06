#!/usr/bin/env python3
"""Audit one fixed OpenVINO-to-packed-backend semantic state import."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-reference-state-import-gate-v0"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
OV_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
DEFAULT_TOKENS = (
    ROOT
    / "output/seq571-state-conditioned-head-correction-token-input-20260710Tseq571Z"
    / "token-input/fresh_code_03.tokens.u32"
)


WORKER = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov


STATE_RE = re.compile(
    r"cache_params\.past\.(conv|ssm|key|value)\.(\d+)"
    r"cache_params\.present\.\1\.\2")
LINEAR_GLOBAL_LAYERS = tuple(
    layer for layer in range(40) if (layer + 1) % 4 != 0)
FULL_GLOBAL_LAYERS = tuple(
    layer for layer in range(40) if (layer + 1) % 4 == 0)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def make_inputs(embedding, token_ids, start: int, total: int):
  ids = np.asarray([token_ids], dtype=np.int64)
  embedded = embedding({"input": ids})[embedding.output(0)]
  count = len(token_ids)
  positions = np.arange(start, start + count, dtype=np.int64)
  return {
      "attention_mask": np.ones((1, total), dtype=np.int64),
      "inputs_embeds": np.asarray(embedded, dtype=np.float32),
      "position_ids": np.tile(positions, (4, 1)).reshape(4, 1, count),
      "beam_idx": np.zeros((1,), dtype=np.int32),
  }


def write_native_state(state, state_dir: Path):
  name = state.name
  match = STATE_RE.fullmatch(name)
  if match is None:
    raise RuntimeError(f"unrecognized state name: {name}")
  kind = match.group(1)
  logical_layer = int(match.group(2))
  value = np.array(state.state.data, dtype=np.float32, copy=True)
  original_shape = list(value.shape)
  if kind == "conv":
    if original_shape != [1, 8192, 4]:
      raise RuntimeError(f"unexpected conv shape: {original_shape}")
    global_layer = LINEAR_GLOBAL_LAYERS[logical_layer]
    native_kind = "linear_conv"
    # The converted graph concatenates four cached values with new QKV and
    # assigns the last four. Native causal-conv state stores the preceding
    # kernel_size-1 values, so the fixed mapping drops the oldest slot.
    native = value[0, :, 1:4]
    transform = "drop_oldest_of_four_channel_major"
  elif kind == "ssm":
    if original_shape != [1, 32, 128, 128]:
      raise RuntimeError(f"unexpected SSM shape: {original_shape}")
    global_layer = LINEAR_GLOBAL_LAYERS[logical_layer]
    native_kind = "linear_recurrent"
    # OpenVINO stores the recurrent matrix as [value_head, key, value]: its
    # graph reduces state * k[:, None] over the key axis and updates with the
    # outer product k[:, None] * delta[None, :]. The native delta core indexes
    # state as [value_head, value, key], so the last two axes must be swapped.
    native = np.transpose(value[0], (0, 2, 1))
    transform = "transpose_key_value_to_value_key"
  else:
    prefix_tokens = int(value.shape[2]) if value.ndim == 4 else -1
    if original_shape != [1, 2, prefix_tokens, 256]:
      raise RuntimeError(f"unexpected {kind} shape: {original_shape}")
    global_layer = FULL_GLOBAL_LAYERS[logical_layer]
    native_kind = "full_k" if kind == "key" else "full_v"
    native = np.transpose(value[0], (1, 0, 2))
    transform = "head_major_to_token_major"
  native = np.ascontiguousarray(native, dtype="<f4")
  path = state_dir / f"{native_kind}_{global_layer}.f32"
  native.tofile(path)
  return {
      "byte_count": path.stat().st_size,
      "file": path.name,
      "global_layer": global_layer,
      "logical_layer": logical_layer,
      "native_kind": native_kind,
      "native_shape": list(native.shape),
      "openvino_name": name,
      "openvino_shape": original_shape,
      "sha256": sha256(path),
      "transform": transform,
  }


def top8(logits):
  indices = np.argpartition(logits, -8)[-8:]
  indices = sorted(indices, key=lambda index: float(logits[index]), reverse=True)
  return [{"id": int(index), "value": float(logits[index])} for index in indices]


def main():
  cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  state_dir = Path(cfg["state_dir"])
  state_dir.mkdir(parents=True, exist_ok=True)
  tokens = np.fromfile(cfg["token_file"], dtype="<u4").astype(np.int64)
  if len(tokens) < 2:
    raise RuntimeError("token file must contain at least two tokens")

  core = ov.Core()
  embedding_model = core.read_model(str(Path(cfg["ov_dir"]) /
                                        "openvino_text_embeddings_model.xml"))
  embedding = core.compile_model(
      embedding_model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
  language_model = core.read_model(str(Path(cfg["ov_dir"]) /
                                       "openvino_language_model.xml"))
  compile_config = {
      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
      "PERFORMANCE_HINT": "LATENCY",
  }
  compile_started = time.perf_counter()
  language = core.compile_model(language_model, cfg["device"], compile_config)
  compile_wall_ms = (time.perf_counter() - compile_started) * 1000.0
  request = language.create_infer_request()
  request.reset_state()

  prefix = tokens[:-1]
  prefix_started = time.perf_counter()
  request.infer(make_inputs(embedding, prefix, 0, len(prefix)))
  prefix_wall_ms = (time.perf_counter() - prefix_started) * 1000.0
  state_manifest = sorted(
      (write_native_state(state, state_dir) for state in request.query_state()),
      key=lambda row: (row["global_layer"], row["native_kind"]),
  )

  final_started = time.perf_counter()
  outputs = request.infer(make_inputs(
      embedding, tokens[-1:], len(prefix), len(tokens)))
  final_wall_ms = (time.perf_counter() - final_started) * 1000.0
  logits = np.asarray(next(iter(outputs.values())), dtype=np.float32)[0, -1]
  logits_path = state_dir / "openvino_logits.f32"
  np.ascontiguousarray(logits, dtype="<f4").tofile(logits_path)

  counts = {
      kind: sum(row["native_kind"] == kind for row in state_manifest)
      for kind in ("linear_conv", "linear_recurrent", "full_k", "full_v")
  }
  required = counts == {
      "linear_conv": 30, "linear_recurrent": 30,
      "full_k": 10, "full_v": 10,
  }
  result = {
      "compile_config": compile_config,
      "compile_wall_ms": compile_wall_ms,
      "device": cfg["device"],
      "final_token_wall_ms": final_wall_ms,
      "logits": {
          "byte_count": logits_path.stat().st_size,
          "file": logits_path.name,
          "sha256": sha256(logits_path),
          "shape": list(logits.shape),
          "top8": top8(logits),
      },
      "openvino_version": ov.get_version(),
      "prefix_tokens": len(prefix),
      "prefix_wall_ms": prefix_wall_ms,
      "required_checks_passed": required,
      "state_counts": counts,
      "state_manifest": state_manifest,
      "teacher_token_id": int(tokens[-1]),
  }
  Path(cfg["result_path"]).write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
'''


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKENS)
  parser.add_argument("--device", default="GPU")
  parser.add_argument("--timeout-s", type=int, default=7200)
  return parser.parse_args()


def run(
    command: list[str], timeout: int,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout, env=environment)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [line for line in dirty if not out_rel or out_rel not in line]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def distribution_rows(native: dict[str, Any]) -> list[dict[str, Any]]:
  return [
      {"comparison": name, **(native.get(field) or {})}
      for name, field in (
          ("CPU GGUF vs OpenVINO", "cpu_vs_openvino"),
          ("CPU GGUF vs imported native", "cpu_vs_imported_native"),
          ("OpenVINO vs imported native", "openvino_vs_imported_native"),
      )
  ]


def state_family_rows(native: dict[str, Any]) -> list[dict[str, Any]]:
  rows = native.get("state_comparisons") or []
  result = []
  for kind in ("linear_conv", "linear_recurrent", "full_k", "full_v"):
    numerics = [row["numeric"] for row in rows if row.get("kind") == kind]
    if not numerics:
      continue
    result.append({
        "kind": kind,
        "layer_count": len(numerics),
        "max_relative_l2": max(float(row["relative_l2"]) for row in numerics),
        "min_cosine": min(float(row["cosine"]) for row in numerics),
        "median_cosine": statistics.median(
            float(row["cosine"]) for row in numerics),
    })
  return result


def summary(payload: dict[str, Any]) -> str:
  lines = [
      "# Reference-state import audit",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- route label: `{payload['route_label']}`",
      f"- prompt prefix tokens: `{payload['worker'].get('prefix_tokens')}`",
      "- product speedup claim: `forbidden`",
      "",
      "| distribution | KLD | logits cosine | top-1 | gate |",
      "|---|---:|---:|:---:|:---:|",
  ]
  for row in payload["distribution_rows"]:
    lines.append(
        f"| {row['comparison']} | {row.get('kld', float('nan')):.6f} | "
        f"{row.get('logits_cosine', float('nan')):.6f} | "
        f"{'pass' if row.get('top1_matches') else 'fail'} | "
        f"{'pass' if row.get('required_checks_passed') else 'fail'} |")
  lines += [
      "",
      "| imported state family | layers | min / median cosine | max relL2 |",
      "|---|---:|---:|---:|",
  ]
  for row in payload["state_family_rows"]:
    lines.append(
        f"| {row['kind']} | {row['layer_count']} | "
        f"{row['min_cosine']:.6f} / {row['median_cosine']:.6f} | "
        f"{row['max_relative_l2']:.6f} |")
  lines += [
      "",
      "This gate validates only one imported semantic decode state. It neither",
      "provides native prefill nor authorizes a product performance claim.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  generated_dir = out_dir / "generated"
  raw_dir.mkdir(parents=True, exist_ok=False)
  generated_dir.mkdir(parents=True, exist_ok=True)
  git = git_state(out_dir)
  created_at = dt.datetime.now(dt.timezone.utc).isoformat()

  worker_path = raw_dir / "openvino-state-worker.py"
  worker_path.write_text(WORKER)
  worker_result_path = raw_dir / "worker-result.json"

  compile_command = [
      "ocloc", "compile", "-file",
      str(ROOT / "engine/gpu/opencl/q4x8_matvec.cl"),
      "-device", "0xb080", "-output", "iq36_q4x8_all",
      "-out_dir", str(generated_dir), "-output_no_suffix",
      "--format", "zebin", "-options",
      "-cl-std=CL3.0 -D IQ36_USE_INTEGER_DOT=1", "-q",
  ]
  compile_run = run(compile_command, 300)
  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release",
  ]
  configure_run = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target",
      "iq36-packed-token-level-zero-backend-smoke", "-j8",
  ]
  build_run = run(build_command, 600)
  write_json(raw_dir / "build.json", {
      "compile": {"command": compile_command, "returncode": compile_run.returncode,
                  "stdout": compile_run.stdout, "stderr": compile_run.stderr},
      "configure": {"command": configure_command,
                    "returncode": configure_run.returncode,
                    "stdout": configure_run.stdout, "stderr": configure_run.stderr},
      "build": {"command": build_command, "returncode": build_run.returncode,
                "stdout": build_run.stdout, "stderr": build_run.stderr},
  })
  executable = BUILD_DIR / "iq36-packed-token-level-zero-backend-smoke"
  build_ok = all((
      compile_run.returncode == 0, configure_run.returncode == 0,
      build_run.returncode == 0, executable.is_file(),
      (generated_dir / "iq36_q4x8_all.bin").is_file(),
  ))

  worker: dict[str, Any] = {}
  native: dict[str, Any] = {}
  worker_returncode = -1
  native_returncode = -1
  with tempfile.TemporaryDirectory(prefix="iq36-reference-state-") as temporary:
    state_dir = Path(temporary)
    worker_config = {
        "device": args.device,
        "ov_dir": str(OV_DIR),
        "result_path": str(worker_result_path),
        "state_dir": str(state_dir),
        "token_file": str(args.token_file.resolve()),
    }
    write_json(raw_dir / "worker-config.json", {
        **worker_config, "state_dir": "<ephemeral-derived-state>",
    })
    worker_command = [str(OV_PYTHON), str(worker_path), str(raw_dir / "worker-config.runtime.json")]
    write_json(raw_dir / "worker-config.runtime.json", worker_config)
    worker_run = run(worker_command, args.timeout_s)
    worker_returncode = worker_run.returncode
    (raw_dir / "worker.stdout").write_text(worker_run.stdout)
    (raw_dir / "worker.stderr").write_text(worker_run.stderr)
    (raw_dir / "worker-config.runtime.json").unlink(missing_ok=True)
    if worker_result_path.is_file():
      worker = json.loads(worker_result_path.read_text())

    if build_ok and worker_returncode == 0:
      native_command = [
          str(executable), str(MODEL),
          str(generated_dir / "iq36_q4x8_all.bin"),
          str(args.token_file.resolve()), "--import-state", str(state_dir),
      ]
      environment = os.environ.copy()
      environment["IQ36_INT8_BLOCK32_KV_GQA"] = "1"
      native_run = run(native_command, args.timeout_s, environment)
      native_returncode = native_run.returncode
      (raw_dir / "native.stdout").write_text(native_run.stdout)
      (raw_dir / "native.stderr").write_text(native_run.stderr)
      write_json(raw_dir / "native-command.json", {
          "command": [*native_command[:-1], "<ephemeral-derived-state>"],
          "environment": {"IQ36_INT8_BLOCK32_KV_GQA": "1"},
          "returncode": native_returncode,
      })
      native = parse_last_json(native_run.stdout)

  manifest_rows = worker.get("state_manifest") or []
  fixed_transforms = {
      "linear_conv": "drop_oldest_of_four_channel_major",
      "linear_recurrent": "transpose_key_value_to_value_key",
      "full_k": "head_major_to_token_major",
      "full_v": "head_major_to_token_major",
  }
  transforms_ok = bool(manifest_rows) and all(
      row.get("transform") == fixed_transforms.get(row.get("native_kind"))
      for row in manifest_rows)
  state_rows = native.get("state_comparisons") or []
  checks = [
      {"name": "repository_clean_at_gate", "pass": not git["dirty"],
       "dirty_paths": git["dirty_paths"]},
      {"name": "target_module_and_smoke_build", "pass": build_ok},
      {"name": "openvino_state_capture", "pass": (
          worker_returncode == 0 and worker.get("required_checks_passed") is True)},
      {"name": "exact_80_state_mapping", "pass": (
          worker.get("state_counts") == {
              "linear_conv": 30, "linear_recurrent": 30,
              "full_k": 10, "full_v": 10,
          } and len(manifest_rows) == 80)},
      {"name": "single_pre_registered_transform", "pass": transforms_ok},
      {"name": "all_native_state_comparisons_finite", "pass": (
          len(state_rows) == 80 and all(
              row.get("numeric", {}).get("finite") is True for row in state_rows))},
      {"name": "teacher_forced_distribution_triangle", "pass": (
          native_returncode == 0 and native.get("required_checks_passed") is True)},
      {"name": "product_speedup_not_claimed", "pass": (
          native.get("speedup_claims_allowed") is False)},
  ]
  required = all(bool(check["pass"]) for check in checks)
  payload = {
      "checks": checks,
      "created_at": created_at,
      "distribution_rows": distribution_rows(native),
      "git": git,
      "native": native,
      "native_returncode": native_returncode,
      "required_checks_passed": required,
      "route_label": "candidate" if required else "rejected",
      "schema_version": SCHEMA,
      "speedup_claims_allowed": False,
      "state_family_rows": state_family_rows(native),
      "worker": worker,
      "worker_returncode": worker_returncode,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": created_at,
      "git": git,
      "required_checks_passed": required,
      "route_label": payload["route_label"],
      "schema_version": SCHEMA,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "correctness_applicable": True,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "state_semantics": "openvino_reference_import",
  })
  with (out_dir / "state-comparisons.jsonl").open("w") as handle:
    for row in state_rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(summary(payload))
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "required_checks_passed": required,
      "route_label": payload["route_label"],
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
