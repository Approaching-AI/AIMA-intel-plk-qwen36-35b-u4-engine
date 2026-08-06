#!/usr/bin/env python3
"""Capture R0 target and model facts for the locked intel-qwen36 workstream.

The script is intentionally read-only. It is designed to run on the Intel PTL
target host and print a JSON payload that can be stored under output/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf"
DEFAULT_EXPECTED_SHA256 = (
    "d42becf903ed7093d438c0d9f44afc136756b6b8f766121e9066fb888a2dc36e"
)

PACKAGE_NAMES = [
    "intel-driver-compiler-npu",
    "intel-fw-npu",
    "intel-level-zero-npu",
    "libze-intel-gpu1",
    "libze1",
    "intel-opencl-icd",
    "intel-level-zero-gpu",
    "openvino",
    "openvino-libs",
]

GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGUF_METADATA_KEYS = {
    "general.architecture",
    "general.basename",
    "general.file_type",
    "general.name",
    "general.quantization_version",
    "tokenizer.ggml.model",
}

GGUF_METADATA_PREFIXES = (
    "qwen3.",
    "qwen3moe.",
    "qwen35moe.",
)


def run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
  except Exception as exc:  # pragma: no cover - diagnostic path.
    return {"cmd": cmd, "error": repr(exc)}


def parse_key_value_colon(text: str) -> dict[str, str]:
  result = {}
  for line in text.splitlines():
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    result[key.strip()] = value.strip()
  return result


def parse_os_release() -> dict[str, str]:
  path = Path("/etc/os-release")
  if not path.exists():
    return {}
  result = {}
  for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if "=" not in line or line.startswith("#"):
      continue
    key, value = line.split("=", 1)
    result[key] = value.strip().strip('"')
  return result


def parse_meminfo() -> dict[str, str]:
  path = Path("/proc/meminfo")
  if not path.exists():
    return {}
  return parse_key_value_colon(path.read_text(encoding="utf-8", errors="replace"))


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    while True:
      chunk = fh.read(16 * 1024 * 1024)
      if not chunk:
        return digest.hexdigest()
      digest.update(chunk)


def read_u32(fh: Any) -> int:
  return struct.unpack("<I", fh.read(4))[0]


def read_u64(fh: Any) -> int:
  return struct.unpack("<Q", fh.read(8))[0]


def read_string(fh: Any) -> str:
  size = read_u64(fh)
  return fh.read(size).decode("utf-8", errors="replace")


def read_scalar(fh: Any, value_type: int) -> Any:
  if value_type == 0:
    return struct.unpack("<B", fh.read(1))[0]
  if value_type == 1:
    return struct.unpack("<b", fh.read(1))[0]
  if value_type == 2:
    return struct.unpack("<H", fh.read(2))[0]
  if value_type == 3:
    return struct.unpack("<h", fh.read(2))[0]
  if value_type == 4:
    return struct.unpack("<I", fh.read(4))[0]
  if value_type == 5:
    return struct.unpack("<i", fh.read(4))[0]
  if value_type == 6:
    return struct.unpack("<f", fh.read(4))[0]
  if value_type == 7:
    return bool(struct.unpack("<?", fh.read(1))[0])
  if value_type == 8:
    return read_string(fh)
  if value_type == 10:
    return struct.unpack("<Q", fh.read(8))[0]
  if value_type == 11:
    return struct.unpack("<q", fh.read(8))[0]
  if value_type == 12:
    return struct.unpack("<d", fh.read(8))[0]
  raise ValueError(f"unsupported GGUF scalar type {value_type}")


def skip_scalar(fh: Any, value_type: int) -> None:
  sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
  if value_type in sizes:
    fh.seek(sizes[value_type], os.SEEK_CUR)
    return
  if value_type == 8:
    size = read_u64(fh)
    fh.seek(size, os.SEEK_CUR)
    return
  raise ValueError(f"unsupported GGUF scalar type {value_type}")


def read_metadata_value(fh: Any, value_type: int) -> Any:
  if value_type != 9:
    return read_scalar(fh, value_type)

  element_type = read_u32(fh)
  length = read_u64(fh)
  if length <= 64 and element_type != 9:
    return [read_scalar(fh, element_type) for _ in range(length)]

  for _ in range(length):
    skip_scalar(fh, element_type)
  return {
      "array_type": GGUF_VALUE_TYPES.get(element_type, element_type),
      "length": length,
      "omitted": True,
  }


def gguf_metadata(path: Path) -> dict[str, Any]:
  result: dict[str, Any] = {}
  with path.open("rb") as fh:
    magic = fh.read(4)
    if magic != b"GGUF":
      return {"error": "not_gguf", "magic_hex": magic.hex()}

    version = read_u32(fh)
    tensor_count = read_u64(fh)
    metadata_kv_count = read_u64(fh)
    result["_header"] = {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
    }

    for _ in range(metadata_kv_count):
      key = read_string(fh)
      value_type = read_u32(fh)
      value = read_metadata_value(fh, value_type)
      if key in GGUF_METADATA_KEYS or key.startswith(GGUF_METADATA_PREFIXES):
        if isinstance(value, str) and len(value) > 240:
          value = f"{value[:240]}..."
        result[key] = value
  return result


def package_versions() -> list[str]:
  versions = []
  for name in PACKAGE_NAMES:
    query = run(["dpkg-query", "-W", "-f=${Package} ${Version}\\n", name])
    if query.get("returncode") == 0 and query.get("stdout"):
      versions.append(query["stdout"])
  return versions


def model_facts(path: Path, expected_sha256: str, skip_sha256: bool) -> dict[str, Any]:
  facts: dict[str, Any] = {
      "path": str(path),
      "exists": path.exists(),
      "expected_sha256": expected_sha256,
      "sha256_checked": not skip_sha256,
  }
  if not path.exists():
    facts["sha256_matches_expected"] = False
    return facts

  stat = path.stat()
  facts.update({
      "size_bytes": stat.st_size,
      "size_gib": round(stat.st_size / (1024**3), 4),
      "mtime_epoch": int(stat.st_mtime),
      "gguf_metadata": gguf_metadata(path),
  })

  if skip_sha256:
    facts["sha256_matches_expected"] = None
    return facts

  sha256 = sha256_file(path)
  facts["sha256"] = sha256
  facts["sha256_matches_expected"] = sha256 == expected_sha256
  return facts


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
  parser.add_argument("--expected-sha256", default=DEFAULT_EXPECTED_SHA256)
  parser.add_argument("--skip-sha256", action="store_true")
  args = parser.parse_args()

  lscpu = run(["lscpu"])
  lscpu_fields = parse_key_value_colon(lscpu.get("stdout", ""))
  meminfo = parse_meminfo()
  payload = {
      "schema_version": "0.1",
      "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
      "captured_at_utc": datetime.now(timezone.utc).isoformat(),
      "hostname": socket.gethostname(),
      "fqdn": socket.getfqdn(),
      "platform": {
          "system": platform.system(),
          "release": platform.release(),
          "machine": platform.machine(),
          "python": platform.python_version(),
      },
      "os_release": parse_os_release(),
      "cpu": {
          "model_name": lscpu_fields.get("Model name"),
          "logical_cpus": os.cpu_count(),
          "threads_per_core": lscpu_fields.get("Thread(s) per core"),
          "cores_per_socket": lscpu_fields.get("Core(s) per socket"),
          "sockets": lscpu_fields.get("Socket(s)"),
          "lscpu": lscpu,
      },
      "memory": {
          "MemTotal": meminfo.get("MemTotal"),
          "SwapTotal": meminfo.get("SwapTotal"),
      },
      "storage": {
          "root_df": run(["df", "-B1", "--output=source,size,avail,target", "/"]),
      },
      "runtime": {
          "lspci_intel": run(
              ["bash", "-lc", "lspci | grep -Ei 'Intel|VGA|3D|Processing|NPU|Display'"],
          ),
          "dri_nodes": run(["bash", "-lc", "ls -l /dev/dri 2>/dev/null || true"]),
          "clinfo_list": run(["bash", "-lc", "command -v clinfo >/dev/null && clinfo -l || true"]),
          "clinfo_summary": run(
              [
                  "bash",
                  "-lc",
                  "command -v clinfo >/dev/null && "
                  "clinfo | grep -E '^[[:space:]]*(Platform Name|Device Name|Device Version|Driver Version|Max compute units|Global memory size|Local memory size|Max clock frequency)' || true",
              ],
              timeout=60,
          ),
          "ze_info_head": run(
              ["bash", "-lc", "command -v ze_info >/dev/null && ze_info 2>/dev/null | head -n 160 || true"],
          ),
          "vainfo_head": run(
              ["bash", "-lc", "command -v vainfo >/dev/null && vainfo 2>/dev/null | head -n 80 || true"],
          ),
          "packages": package_versions(),
      },
      "model": model_facts(Path(args.model_path), args.expected_sha256, args.skip_sha256),
  }
  print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
