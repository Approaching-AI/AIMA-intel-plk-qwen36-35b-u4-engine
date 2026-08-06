from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MODEL_CONTRACT_RELATIVE = Path(
    "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json")
MODEL_CONTRACT_SHA256 = (
    "c9616cf79e96f5e628a2425198b8f9ea67c703ddcb379df1012ebe8843cbfd48")
EXPECTED_MODEL_FINGERPRINT = (
    "eb05132e47fe0fd1dc42fa3082e7241696ed1449dec246a3cc14bef4af21d7ec")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _verify_files(
    model_dir: Path, locked_files: Mapping[str, Mapping[str, Any]],
    *, full_hash: bool,
) -> tuple[list[dict[str, Any]], str | None]:
  rows: list[dict[str, Any]] = []
  fingerprint_rows = []
  for name, expected in locked_files.items():
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
      raise RuntimeError(f"unsafe model-contract file name: {name}")
    path = model_dir / relative
    if not path.is_file():
      raise RuntimeError(f"locked model file is missing: {name}")
    observed_bytes = path.stat().st_size
    expected_bytes = int(expected["bytes"])
    if observed_bytes != expected_bytes:
      raise RuntimeError(
          f"locked model size mismatch for {name}: expected "
          f"{expected_bytes}, observed {observed_bytes}")
    expected_sha = str(expected["sha256"])
    observed_sha = sha256_file(path) if full_hash else None
    if full_hash and observed_sha != expected_sha:
      raise RuntimeError(
          f"locked model SHA-256 mismatch for {name}: expected "
          f"{expected_sha}, observed {observed_sha}")
    rows.append({
        "file": name, "bytes": observed_bytes,
        "sha256_verified": bool(full_hash),
    })
    if full_hash:
      fingerprint_rows.append((name, observed_sha, observed_bytes))
  fingerprint = None
  if full_hash:
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_rows, sort_keys=True).encode()).hexdigest()
  return rows, fingerprint


def verify_model_identity(
    model_dir: Path, contract_path: Path, mode: str,
) -> dict[str, Any]:
  started = time.perf_counter_ns()
  if mode == "off":
    return {
        "mode": mode, "sha256_verified": False,
        "model_fingerprint": None,
        "expected_model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
        "files_verified": 0, "bytes_verified": 0,
        "elapsed_ms": 0.0,
    }
  observed_contract = sha256_file(contract_path)
  if observed_contract != MODEL_CONTRACT_SHA256:
    raise RuntimeError(
        "model contract fingerprint mismatch: expected "
        f"{MODEL_CONTRACT_SHA256}, observed {observed_contract}")
  try:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    locked_files = contract["product_model"]["locked_files"]
  except (KeyError, TypeError, json.JSONDecodeError) as error:
    raise RuntimeError("model contract has an invalid locked-files shape") from error
  if not isinstance(locked_files, dict) or not locked_files:
    raise RuntimeError("model contract has no locked files")
  rows, fingerprint = _verify_files(
      model_dir, locked_files, full_hash=mode == "full")
  if mode == "full" and fingerprint != EXPECTED_MODEL_FINGERPRINT:
    raise RuntimeError(
        "locked model aggregate fingerprint mismatch: expected "
        f"{EXPECTED_MODEL_FINGERPRINT}, observed {fingerprint}")
  return {
      "mode": mode,
      "sha256_verified": mode == "full",
      "model_fingerprint": fingerprint,
      "expected_model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
      "files_verified": len(rows),
      "bytes_verified": sum(int(row["bytes"]) for row in rows),
      "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
  }
