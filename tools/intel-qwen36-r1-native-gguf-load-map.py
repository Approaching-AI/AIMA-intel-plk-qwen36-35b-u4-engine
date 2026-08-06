#!/usr/bin/env python3
"""Build a native GGUF load map for the locked Qwen3.6 model.

This is an R1 prerequisite artifact for the native token loop. It reads the
locked GGUF header and tensor table on the target host, validates the
Qwen35MoE layer pattern, and emits mmap-ready tensor metadata. It does not run
inference and does not create native candidate token rows.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r1-native-gguf-load-map-v0"
DEFAULT_HOST = "local"
DEFAULT_MODEL = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
EXPECTED_SIZE_BYTES = 21166755168

GGML_TYPES = {
    0: {"name": "F32", "block_size": 1, "type_size": 4},
    12: {"name": "Q4_K", "block_size": 256, "type_size": 144},
    14: {"name": "Q6_K", "block_size": 256, "type_size": 210},
}

NON_LAYER_EXPECTED = {
    "output.weight": {"dims": [2048, 248320], "type": 14},
    "output_norm.weight": {"dims": [2048], "type": 0},
    "token_embd.weight": {"dims": [2048, 248320], "type": 12},
}

COMMON_LAYER_EXPECTED = {
    "attn_norm.weight": {"dims": [2048], "type": 0},
    "ffn_down_exps.weight": {"dims": [512, 2048, 256], "types": [12, 14]},
    "ffn_down_shexp.weight": {"dims": [512, 2048], "types": [12, 14]},
    "ffn_gate_inp.weight": {"dims": [2048, 256], "type": 0},
    "ffn_gate_inp_shexp.weight": {"dims": [2048], "type": 0},
    "ffn_gate_shexp.weight": {"dims": [2048, 512], "type": 12},
    "ffn_gate_up_exps.weight": {"dims": [2048, 1024, 256], "type": 12},
    "ffn_up_shexp.weight": {"dims": [2048, 512], "type": 12},
    "post_attention_norm.weight": {"dims": [2048], "type": 0},
}

LINEAR_SSM_EXPECTED = {
    "attn_gate.weight": {"dims": [2048, 4096], "type": 12},
    "attn_qkv.weight": {"dims": [2048, 8192], "types": [12, 14]},
    "ssm_a": {"dims": [32], "type": 0},
    "ssm_alpha.weight": {"dims": [2048, 32], "type": 12},
    "ssm_beta.weight": {"dims": [2048, 32], "type": 12},
    "ssm_conv1d.weight": {"dims": [4, 8192], "type": 0},
    "ssm_dt.bias": {"dims": [32], "type": 0},
    "ssm_norm.weight": {"dims": [128], "type": 0},
    "ssm_out.weight": {"dims": [4096, 2048], "type": 12},
}

FULL_ATTENTION_EXPECTED = {
    "attn_k.weight": {"dims": [2048, 512], "type": 12},
    "attn_k_norm.weight": {"dims": [256], "type": 0},
    "attn_output.weight": {"dims": [4096, 2048], "type": 12},
    "attn_q.weight": {"dims": [2048, 8192], "type": 12},
    "attn_q_norm.weight": {"dims": [256], "type": 0},
    "attn_v.weight": {"dims": [2048, 512], "types": [12, 14]},
}


REMOTE_SCRIPT = r'''
import json
import os
import re
import struct
import sys

MODEL_PATH = sys.argv[1]

VALUE_TYPES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32",
    5: "int32", 6: "float32", 7: "bool", 8: "string", 9: "array",
    10: "uint64", 11: "int64", 12: "float64",
}

def u32(fh):
    return struct.unpack("<I", fh.read(4))[0]

def u64(fh):
    return struct.unpack("<Q", fh.read(8))[0]

def read_string(fh):
    size = u64(fh)
    return fh.read(size).decode("utf-8", "replace")

def read_scalar(fh, value_type):
    if value_type == 0: return struct.unpack("<B", fh.read(1))[0]
    if value_type == 1: return struct.unpack("<b", fh.read(1))[0]
    if value_type == 2: return struct.unpack("<H", fh.read(2))[0]
    if value_type == 3: return struct.unpack("<h", fh.read(2))[0]
    if value_type == 4: return struct.unpack("<I", fh.read(4))[0]
    if value_type == 5: return struct.unpack("<i", fh.read(4))[0]
    if value_type == 6: return struct.unpack("<f", fh.read(4))[0]
    if value_type == 7: return bool(struct.unpack("<?", fh.read(1))[0])
    if value_type == 8: return read_string(fh)
    if value_type == 10: return struct.unpack("<Q", fh.read(8))[0]
    if value_type == 11: return struct.unpack("<q", fh.read(8))[0]
    if value_type == 12: return struct.unpack("<d", fh.read(8))[0]
    raise ValueError(f"unsupported scalar type {value_type}")

def skip_scalar(fh, value_type):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type in sizes:
        fh.seek(sizes[value_type], os.SEEK_CUR)
        return
    if value_type == 8:
        size = u64(fh)
        fh.seek(size, os.SEEK_CUR)
        return
    raise ValueError(f"unsupported scalar type {value_type}")

def read_value(fh, value_type):
    if value_type != 9:
        return read_scalar(fh, value_type)
    element_type = u32(fh)
    length = u64(fh)
    if length <= 64 and element_type != 9:
        return [read_scalar(fh, element_type) for _ in range(length)]
    for _ in range(length):
        skip_scalar(fh, element_type)
    return {
        "array_type": VALUE_TYPES.get(element_type, str(element_type)),
        "length": length,
        "omitted": True,
    }

stat = os.stat(MODEL_PATH)
metadata = {}
tensors = []
with open(MODEL_PATH, "rb") as fh:
    magic = fh.read(4)
    version = u32(fh)
    tensor_count = u64(fh)
    metadata_kv_count = u64(fh)
    for _ in range(metadata_kv_count):
        key = read_string(fh)
        value_type = u32(fh)
        metadata[key] = read_value(fh, value_type)
    for index in range(tensor_count):
        name = read_string(fh)
        ndims = u32(fh)
        dims = [u64(fh) for _ in range(ndims)]
        tensor_type = u32(fh)
        offset = u64(fh)
        tensors.append({
            "dims": dims,
            "index": index,
            "name": name,
            "offset": offset,
            "type": tensor_type,
        })
    tensor_info_end = fh.tell()

alignment = int(metadata.get("general.alignment", 32))
data_section_offset = ((tensor_info_end + alignment - 1) // alignment) * alignment

layer_re = re.compile(r"^blk\.(\d+)\.(.+)$")
for row in tensors:
    match = layer_re.match(row["name"])
    if match:
        row["layer_index"] = int(match.group(1))
        row["suffix"] = match.group(2)
    row["absolute_offset"] = data_section_offset + row["offset"]

print(json.dumps({
    "data_section_offset": data_section_offset,
    "file_size_bytes": stat.st_size,
    "header": {
        "magic": magic.decode("ascii", "replace"),
        "metadata_kv_count": metadata_kv_count,
        "tensor_count": tensor_count,
        "version": version,
    },
    "metadata": metadata,
    "model_path": MODEL_PATH,
    "tensors": tensors,
}, sort_keys=True))
'''


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument("--timeout-s", type=int, default=120)
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


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for row in rows:
      fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def tensor_nbytes(tensor: dict[str, Any]) -> int | None:
  traits = GGML_TYPES.get(tensor.get("type"))
  if traits is None:
    return None
  elements = math.prod(tensor.get("dims", []))
  blocks = math.ceil(elements / traits["block_size"])
  return blocks * traits["type_size"]


def annotate_tensors(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
  annotated = []
  for row in tensors:
    out = dict(row)
    traits = GGML_TYPES.get(out.get("type"), {})
    out["ggml_type_name"] = traits.get("name", f"unknown_{out.get('type')}")
    nbytes = tensor_nbytes(out)
    out["nbytes"] = nbytes
    if isinstance(nbytes, int) and isinstance(out.get("absolute_offset"), int):
      out["absolute_end"] = out["absolute_offset"] + nbytes
    annotated.append(out)
  return annotated


def expected_full_attention_layers(interval: int, layers: int) -> list[int]:
  return [index for index in range(layers) if (index + 1) % interval == 0]


def group_layers(tensors: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
  layers: dict[int, dict[str, dict[str, Any]]] = {}
  for tensor in tensors:
    layer_index = tensor.get("layer_index")
    suffix = tensor.get("suffix")
    if isinstance(layer_index, int) and isinstance(suffix, str):
      layers.setdefault(layer_index, {})[suffix] = tensor
  return layers


def check_tensor(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
  expected_types = expected["types"] if "types" in expected else [expected["type"]]
  return (
      isinstance(actual, dict)
      and actual.get("dims") == expected["dims"]
      and actual.get("type") in expected_types
      and isinstance(actual.get("absolute_offset"), int)
      and isinstance(actual.get("nbytes"), int)
  )


def layer_kind(layer_index: int, full_layers: set[int]) -> str:
  return "full_attention" if layer_index in full_layers else "linear_ssm"


def validate_load_map(remote: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  metadata = remote["metadata"]
  tensors = remote["tensors"]
  layers = group_layers(tensors)
  block_count = int(metadata.get("qwen35moe.block_count", -1))
  interval = int(metadata.get("qwen35moe.full_attention_interval", -1))
  full_layers = set(expected_full_attention_layers(interval, block_count))
  non_layer = {row["name"]: row for row in tensors if "layer_index" not in row}
  checks: list[dict[str, Any]] = []

  def add(name: str, ok: bool, **extra: Any) -> None:
    checks.append({"name": name, "pass": bool(ok), **extra})

  header = remote["header"]
  add("locked_model_path", remote.get("model_path") == DEFAULT_MODEL, path=remote.get("model_path"))
  add(
      "locked_model_size",
      remote.get("file_size_bytes") == EXPECTED_SIZE_BYTES,
      file_size_bytes=remote.get("file_size_bytes"),
  )
  add("gguf_v3_header", header.get("magic") == "GGUF" and header.get("version") == 3)
  add("tensor_count", header.get("tensor_count") == 693, tensor_count=header.get("tensor_count"))
  add(
      "metadata_kv_count",
      header.get("metadata_kv_count") == 45,
      metadata_kv_count=header.get("metadata_kv_count"),
  )
  add("architecture", metadata.get("general.architecture") == "qwen35moe")
  add("file_type_q4_k_m", metadata.get("general.file_type") == 15)
  add("context_length", metadata.get("qwen35moe.context_length") == 262144)
  add("block_count", block_count == 40, block_count=block_count)
  add("embedding_length", metadata.get("qwen35moe.embedding_length") == 2048)
  add("head_count", metadata.get("qwen35moe.attention.head_count") == 16)
  add("kv_head_count", metadata.get("qwen35moe.attention.head_count_kv") == 2)
  add("expert_count", metadata.get("qwen35moe.expert_count") == 256)
  add("expert_used_count", metadata.get("qwen35moe.expert_used_count") == 8)
  add("full_attention_interval", interval == 4, interval=interval)
  add("layer_index_count", sorted(layers) == list(range(40)), layer_indexes=sorted(layers))
  add(
      "full_attention_layer_indexes",
      sorted(full_layers) == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
      full_attention_layers=sorted(full_layers),
  )
  add(
      "non_layer_tensor_set",
      sorted(non_layer) == sorted(NON_LAYER_EXPECTED),
      non_layer_tensors=sorted(non_layer),
  )
  for name, expected in NON_LAYER_EXPECTED.items():
    add(f"non_layer_tensor:{name}", check_tensor(non_layer.get(name), expected))

  linear_layers = []
  full_attention_layers = []
  layer_summaries = []
  for index in range(40):
    tensors_by_suffix = layers.get(index, {})
    kind = layer_kind(index, full_layers)
    expected = dict(COMMON_LAYER_EXPECTED)
    if kind == "full_attention":
      expected.update(FULL_ATTENTION_EXPECTED)
      full_attention_layers.append(index)
    else:
      expected.update(LINEAR_SSM_EXPECTED)
      linear_layers.append(index)
    suffixes = sorted(tensors_by_suffix)
    expected_suffixes = sorted(expected)
    add(
        f"layer_{index:02d}_suffix_set",
        suffixes == expected_suffixes,
        kind=kind,
        suffix_count=len(suffixes),
    )
    dims_types_ok = all(
        check_tensor(tensors_by_suffix.get(suffix), expected_value)
        for suffix, expected_value in expected.items()
    )
    add(f"layer_{index:02d}_dims_types", dims_types_ok, kind=kind)
    layer_summaries.append({
        "kind": kind,
        "layer_index": index,
        "tensor_count": len(suffixes),
    })

  type_counts: dict[str, int] = {}
  type_bytes: dict[str, int] = {}
  for tensor in tensors:
    type_name = tensor["ggml_type_name"]
    type_counts[type_name] = type_counts.get(type_name, 0) + 1
    if isinstance(tensor.get("nbytes"), int):
      type_bytes[type_name] = type_bytes.get(type_name, 0) + tensor["nbytes"]

  summary = {
      "full_attention_layer_count": len(full_attention_layers),
      "full_attention_layers": full_attention_layers,
      "layer_summaries": layer_summaries,
      "linear_ssm_layer_count": len(linear_layers),
      "linear_ssm_layers": linear_layers,
      "native_gguf_load_map_ready": all(check["pass"] for check in checks),
      "tensor_count": len(tensors),
      "tensor_type_counts": type_counts,
      "tensor_type_nbytes": type_bytes,
  }
  return checks, summary


def build_summary(payload: dict[str, Any]) -> str:
  load_map = payload["native_gguf_load_map"]
  lines = [
      "# R1 Native GGUF Load Map",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- model: `{payload['model_path']}`",
      f"- tensors: {load_map['tensor_count']}",
      f"- linear/SSM layers: {load_map['linear_ssm_layer_count']}",
      f"- full-attention layers: {load_map['full_attention_layer_count']}",
      f"- load map ready: `{str(load_map['native_gguf_load_map_ready']).lower()}`",
      "",
      "This artifact is a native model-load prerequisite only. It does not",
      "run inference, generate token candidates, or allow speedup claims.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = (
      ROOT / f"output/r1-native-gguf-load-map-{stamp}"
      if args.out_dir is None
      else args.out_dir
  ).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  remote_command = (
      "python3 - "
      + shlex.quote(args.model)
      + " <<'PY'\n"
      + REMOTE_SCRIPT
      + "\nPY"
  )
  result = run_target(args.host, remote_command, args.timeout_s)
  write_json(out_dir / "remote-command.json", result)
  if result["returncode"] != 0:
    write_json(out_dir / "correctness.json", {
        "checks": [{
            "name": "remote_gguf_parse",
            "pass": False,
            "returncode": result["returncode"],
        }],
        "native_gguf_load_map_ready": False,
        "required_checks_passed": False,
        "schema_version": SCHEMA_VERSION,
        "workstream": WORKSTREAM,
    })
    print(f"r1 native gguf load map output: {out_dir}")
    return 1

  remote = json.loads(result["stdout"])
  remote["tensors"] = annotate_tensors(remote["tensors"])
  checks, load_map_summary = validate_load_map(remote)
  payload = {
      "created_at": created_at,
      "data_section_offset": remote["data_section_offset"],
      "file_size_bytes": remote["file_size_bytes"],
      "header": remote["header"],
      "host": args.host,
      "metadata": {
          key: remote["metadata"].get(key)
          for key in sorted(remote["metadata"])
          if key.startswith(("general.", "qwen35moe.", "tokenizer.ggml.model"))
      },
      "model_path": remote["model_path"],
      "native_gguf_load_map": {
          **load_map_summary,
          "r1_native_correctness_gate_closed": False,
          "speedup_claims_allowed": False,
      },
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }

  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "host": args.host,
      "model_path": args.model,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r1-native-gguf-load-map.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "load-map.json", payload)
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "gate": "r1_native_gguf_load_map",
      "native_gguf_load_map_ready": load_map_summary["native_gguf_load_map_ready"],
      "required_checks_passed": all(check["pass"] for check in checks),
      "r1_native_correctness_gate_closed": False,
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  })
  write_jsonl(out_dir / "tensor-index.jsonl", remote["tensors"])
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("tensor_count", load_map_summary["tensor_count"]),
        ("linear_ssm_layer_count", load_map_summary["linear_ssm_layer_count"]),
        ("full_attention_layer_count", load_map_summary["full_attention_layer_count"]),
        ("native_gguf_load_map_ready", load_map_summary["native_gguf_load_map_ready"]),
        ("r1_native_correctness_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r1_native_gguf_load_map",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"r1 native gguf load map output: {out_dir}")
  return 0 if load_map_summary["native_gguf_load_map_ready"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
