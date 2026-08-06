#!/usr/bin/env python3
"""Shared local-experiment helpers for the locked Intel target.

The repository now lives on the experiment machine itself.  Drivers therefore
execute target commands directly and stage files with local copies; no network
transport configuration is part of the experiment loop.

The first ``target`` argument on the execution and copy helpers is retained so
older command lines and artifact schemas remain readable.  It is deliberately
restricted to this machine and is never used as a network address.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable


LOCAL_TARGET = "local"
_LOCAL_TARGET_ALIASES = frozenset({
    LOCAL_TARGET,
    "localhost",
    "127.0.0.1",
    "::1",
    socket.gethostname(),
    socket.getfqdn(),
})


def _require_local_target(target: str) -> None:
  if target not in _LOCAL_TARGET_ALIASES:
    raise ValueError(
        f"target must be this local machine ({LOCAL_TARGET!r}); got {target!r}"
    )


def run(cmd: list[str], timeout_s: int, *,
        input_text: str | None = None) -> dict[str, Any]:
  """Run a subprocess, capturing stdout/stderr; never raises on nonzero exit."""
  try:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=input_text,
        timeout=timeout_s,
        check=False,
    )
  except subprocess.TimeoutExpired as exc:
    return {
        "cmd": cmd,
        "command": cmd,
        "returncode": 124,
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") + "\ntimeout",
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
        "timed_out": True,
    }
  return {
      "cmd": cmd,
      "command": cmd,
      "returncode": proc.returncode,
      "stderr": proc.stderr,
      "stdout": proc.stdout,
      "timed_out": False,
  }


def run_target(target: str, command: str, timeout_s: int, *,
               input_text: str | None = None) -> dict[str, Any]:
  """Run a shell command directly on the local experiment machine."""
  _require_local_target(target)
  return run(["bash", "-lc", command], timeout_s, input_text=input_text)


def copy_to(target: str, local_path: Path, target_path: str,
            timeout_s: int) -> dict[str, Any]:
  """Copy one local file into a target staging path on this machine."""
  _require_local_target(target)
  return run(["cp", "--", str(local_path), target_path], timeout_s)


def copy_from(target: str, target_path: str, local_path: Path,
              timeout_s: int) -> dict[str, Any]:
  """Copy one file from a target staging path on this machine."""
  _require_local_target(target)
  return run(["cp", "--", target_path, str(local_path)], timeout_s)


def copy_tree_to(target: str, local_dir: Path, target_dir: str,
                 timeout_s: int) -> dict[str, Any]:
  """Copy a directory's contents into a local target staging directory."""
  _require_local_target(target)
  return run(["cp", "-a", f"{local_dir}/.", target_dir], timeout_s)


def copy_tree_from(target: str, target_dir: str, local_dir: Path,
                   timeout_s: int) -> dict[str, Any]:
  """Copy a target staging directory's contents into a local directory."""
  _require_local_target(target)
  return run(["cp", "-a", f"{target_dir}/.", str(local_dir)], timeout_s)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def write_metric(path: Path, phase: str, rows: Iterable[tuple[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    for metric, value in rows:
      fh.write(
          json.dumps({"metric": metric, "phase": phase, "value": value}, sort_keys=True)
          + "\n"
      )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
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


def build_summary(title: str, fields: Iterable[tuple[str, Any]], body: str = "") -> str:
  lines = [f"# {title}", ""]
  lines += [f"- {name}: `{value}`" for name, value in fields]
  if body:
    lines += ["", body]
  return "\n".join(lines) + "\n"


# result.json top level is the config record; these keys are per-run volatile
# or derived and must not enter the config identity hash.  The historical
# remote_* keys remain because old evidence and generated drivers use them as
# staging-path field names even though staging is now local.
CONFIG_VOLATILE_KEYS = frozenset({
    "created_at", "generated_cpp", "remote_dir", "remote_token_dir",
    "checks", "required_checks_passed", "token_stream_required_checks_passed",
    "target", "parse_error", "smoke", "token_stream_event_count",
    "token_stream_jsonl", "explore", "cache",
})


def config_sha(manifest: dict[str, Any]) -> str:
  config = {k: v for k, v in manifest.items() if k not in CONFIG_VOLATILE_KEYS}
  canon = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
  return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def source_fingerprint(source_files: Iterable[tuple[str, str]], root: Path,
                       extra_bytes: bytes = b"") -> str:
  """Fingerprint the staged source closure (engine files + generated TU)."""
  digest = hashlib.sha256()
  for local, staged in source_files:
    digest.update(staged.encode("utf-8"))
    digest.update((root / local).read_bytes())
  digest.update(extra_bytes)
  return digest.hexdigest()


def ensure_cached_binary(
    target: str,
    cache_root: str,
    source_files: list[tuple[str, str]],
    root: Path,
    generated_cpp: Path,
    generated_cpp_remote_rel: str,
    build_command_for: Callable[[str], str],
    built_binary_rel: str,
    timeout_s: int,
) -> dict[str, Any]:
  """Return a locally cached binary for this exact source/build closure."""
  _require_local_target(target)
  cpp_bytes = generated_cpp.read_bytes()
  digest = hashlib.sha256()
  digest.update(source_fingerprint(source_files, root, cpp_bytes).encode("utf-8"))
  digest.update(build_command_for("__CACHE_SRC__").encode("utf-8"))
  key = digest.hexdigest()[:24]
  bin_path = f"{cache_root}/bin/{key}"
  src_dir = f"{cache_root}/src/{key}"

  result: dict[str, Any] = {"key": key, "binary": bin_path, "src_dir": src_dir}
  probe = run_target(target, f"test -x {shlex.quote(bin_path)}", timeout_s)
  result["probe"] = probe
  if probe.get("returncode") == 0:
    result.update({"hit": True, "ok": True})
    return result

  result["hit"] = False
  subdirs = sorted({str(Path(staged).parent) for _, staged in source_files}
                   | {str(Path(generated_cpp_remote_rel).parent), "build"})
  mkdir = run_target(
      target,
      "mkdir -p " + " ".join(
          shlex.quote(f"{src_dir}/{sub}") for sub in subdirs
      ) + " " + shlex.quote(f"{cache_root}/bin"),
      timeout_s,
  )
  result["mkdir"] = mkdir
  if mkdir.get("returncode") != 0:
    result["ok"] = False
    return result

  transfers = []
  for local, staged in source_files:
    transfers.append(copy_to(target, root / local, f"{src_dir}/{staged}", timeout_s))
  transfers.append(
      copy_to(target, generated_cpp, f"{src_dir}/{generated_cpp_remote_rel}", timeout_s)
  )
  result["transfers"] = transfers
  if not all(transfer.get("returncode") == 0 for transfer in transfers):
    result["ok"] = False
    return result

  build = run_target(
      target,
      f"bash -lc {shlex.quote(build_command_for(src_dir))}",
      timeout_s,
  )
  result["build"] = build
  if build.get("returncode") != 0:
    result["ok"] = False
    return result

  publish = run_target(
      target,
      f"cp {shlex.quote(src_dir + '/' + built_binary_rel)} "
      f"{shlex.quote(bin_path + '.tmp')} "
      f"&& mv {shlex.quote(bin_path + '.tmp')} {shlex.quote(bin_path)}",
      timeout_s,
  )
  result["publish"] = publish
  result["ok"] = publish.get("returncode") == 0
  return result


def ensure_cached_tokens(target: str, cache_root: str, token_dir: Path,
                         timeout_s: int) -> dict[str, Any]:
  """Stage a token-input directory once per content hash on this machine."""
  _require_local_target(target)
  digest = hashlib.sha256()
  for path in sorted(path for path in token_dir.rglob("*") if path.is_file()):
    digest.update(str(path.relative_to(token_dir)).encode("utf-8"))
    digest.update(path.read_bytes())
  key = digest.hexdigest()[:24]
  staged_dir = f"{cache_root}/tokens/{key}"
  result: dict[str, Any] = {"key": key, "dir": staged_dir}

  probe = run_target(target, f"test -f {shlex.quote(staged_dir + '/.complete')}", timeout_s)
  result["probe"] = probe
  if probe.get("returncode") == 0:
    result.update({"hit": True, "ok": True})
    return result

  result["hit"] = False
  mkdir = run_target(target, f"mkdir -p {shlex.quote(staged_dir)}", timeout_s)
  result["mkdir"] = mkdir
  if mkdir.get("returncode") != 0:
    result["ok"] = False
    return result
  transfer = copy_tree_to(target, token_dir, staged_dir, timeout_s)
  result["transfer"] = transfer
  if transfer.get("returncode") != 0:
    result["ok"] = False
    return result
  seal = run_target(target, f"touch {shlex.quote(staged_dir + '/.complete')}", timeout_s)
  result["seal"] = seal
  result["ok"] = seal.get("returncode") == 0
  return result
