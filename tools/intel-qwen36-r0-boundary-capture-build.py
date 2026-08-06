#!/usr/bin/env python3
"""Build the patched llama.cpp R0 boundary capture executable on target."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-boundary-capture-build-v0"
DEFAULT_HOST = "local"
EXPECTED_SHA = "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
REMOTE_SOURCE_DIR = f"/home/intel/intel-qwen36-r0/source/llama.cpp-{EXPECTED_SHA}"
REMOTE_BUILD_ROOT = "/home/intel/intel-qwen36-r0/build"
INTEL_ENV = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
TARGET = "llama-qwen36-boundary-capture"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-boundary-capture-build-<UTC>.",
  )
  parser.add_argument(
      "--keep-existing-build-dir",
      action="store_true",
      help="Do not remove the generated build dir if it already exists.",
  )
  return parser.parse_args()


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def run(cmd: list[str], *, timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "command": cmd,
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr + f"\nlocal timeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_script: str, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_script, timeout_s)


def parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key:
      values[key.strip()] = value.strip()
  return values


def build_summary(payload: dict[str, Any]) -> str:
  route = payload["build_route"]
  lines = [
      "# R0 Boundary Capture Build",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- build dir: `{route['remote_build_dir']}`",
      f"- configure return code: `{route['configure_returncode']}`",
      f"- build return code: `{route['build_returncode']}`",
      f"- executable present: `{str(route['executable_present']).lower()}`",
      f"- route status: `{payload['route_status']}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This artifact builds the dedicated capture executable only. It does not",
      "run the model, dump oracle tensors, create an oracle bundle, or close R0.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-boundary-capture-build-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  patch_route_path = latest(
      "r0-boundary-capture-instrumentation-patch-*",
      "patch-route.json",
  )
  if patch_route_path is None:
    raise SystemExit("no latest boundary capture instrumentation patch artifact found")
  patch_route = load_json(patch_route_path)
  patch_correctness = load_json(patch_route_path.parent / "correctness.json")
  remote_build_dir = f"{REMOTE_BUILD_ROOT}/{TARGET}-{stamp}"
  setup_line = "" if args.keep_existing_build_dir else f"rm -rf {shlex.quote(remote_build_dir)}"

  configure_script = "\n".join([
      "set -u",
      f"cd {shlex.quote(REMOTE_SOURCE_DIR)}",
      f"test -f {shlex.quote(INTEL_ENV)} && . {shlex.quote(INTEL_ENV)}",
      "printf 'source_rev_parse='; git rev-parse HEAD",
      "printf 'source_status_short_count='; git status --short | wc -l",
      "printf 'source_status_short='; git status --short | tr '\\n' ';'; printf '\\n'",
      f"mkdir -p {shlex.quote(REMOTE_BUILD_ROOT)}",
      setup_line,
      f"cmake -S {shlex.quote(REMOTE_SOURCE_DIR)} -B {shlex.quote(remote_build_dir)} -G Ninja "
      "-DCMAKE_BUILD_TYPE=Release "
      "-DLLAMA_BUILD_TOOLS=ON "
      "-DLLAMA_BUILD_SERVER=OFF "
      "-DLLAMA_BUILD_EXAMPLES=OFF "
      "-DLLAMA_BUILD_TESTS=OFF "
      "-DLLAMA_BUILD_APP=OFF",
  ])
  configure = run_target(args.host, configure_script, timeout_s=180)
  (raw_dir / "configure.stdout").write_text(configure["stdout"], encoding="utf-8")
  (raw_dir / "configure.stderr").write_text(configure["stderr"], encoding="utf-8")

  build_script = "\n".join([
      "set -u",
      f"test -f {shlex.quote(INTEL_ENV)} && . {shlex.quote(INTEL_ENV)}",
      f"cmake --build {shlex.quote(remote_build_dir)} --target {TARGET} -j 2",
      f"printf 'executable_present='; test -x {shlex.quote(remote_build_dir + '/bin/' + TARGET)} && echo true || echo false",
      f"printf 'executable_path={remote_build_dir}/bin/{TARGET}\\n'",
  ])
  build = run_target(args.host, build_script, timeout_s=900)
  (raw_dir / "build.stdout").write_text(build["stdout"], encoding="utf-8")
  (raw_dir / "build.stderr").write_text(build["stderr"], encoding="utf-8")

  help_script = "\n".join([
      "set -u",
      f"exe={shlex.quote(remote_build_dir + '/bin/' + TARGET)}",
      "if test -x \"$exe\"; then \"$exe\" --help >/tmp/iq36-capture-help.stdout 2>/tmp/iq36-capture-help.stderr; rc=$?; else rc=127; fi",
      "printf 'help_returncode=%s\\n' \"$rc\"",
      "cat /tmp/iq36-capture-help.stdout 2>/dev/null || true",
      "cat /tmp/iq36-capture-help.stderr 2>/dev/null || true",
  ])
  help_result = run_target(args.host, help_script, timeout_s=30)
  (raw_dir / "help.stdout").write_text(help_result["stdout"], encoding="utf-8")
  (raw_dir / "help.stderr").write_text(help_result["stderr"], encoding="utf-8")

  configure_values = parse_key_values(configure["stdout"])
  build_values = parse_key_values(build["stdout"])
  help_values = parse_key_values(help_result["stdout"])
  build_route = {
      "build_returncode": build["returncode"],
      "build_timed_out": build["timed_out"],
      "configure_returncode": configure["returncode"],
      "configure_timed_out": configure["timed_out"],
      "executable_path": build_values.get("executable_path"),
      "executable_present": build_values.get("executable_present") == "true",
      "help_returncode": int(help_values.get("help_returncode", "-1")),
      "keep_existing_build_dir": args.keep_existing_build_dir,
      "remote_build_dir": remote_build_dir,
      "target": TARGET,
  }
  route_status = (
      "boundary_capture_executable_built"
      if build_route["configure_returncode"] == 0
      and build_route["build_returncode"] == 0
      and build_route["executable_present"] is True
      and build_route["help_returncode"] == 0
      else "boundary_capture_build_failed"
  )
  payload = {
      "build_route": build_route,
      "created_at": created_at,
      "evidence": {
          "patch_route": rel(patch_route_path.parent),
          "raw_dir": rel(raw_dir),
      },
      "host": args.host,
      "r0_oracle_gate_closed": False,
      "route_status": route_status,
      "schema_version": SCHEMA_VERSION,
      "source_state": {
          "source_rev_parse": configure_values.get("source_rev_parse"),
          "source_status_short": configure_values.get("source_status_short", ""),
          "source_status_short_count": int(configure_values.get("source_status_short_count", "-1")),
      },
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "latest_patch_route_applied",
          "pass": patch_route.get("route_status") == "target_source_patched_ready_for_build"
          and patch_correctness.get("required_checks_passed") is True,
      },
      {
          "name": "source_contains_expected_patch_only",
          "pass": payload["source_state"]["source_rev_parse"] == EXPECTED_SHA
          and payload["source_state"]["source_status_short_count"] == 2
          and "tools/qwen36-boundary-capture" in payload["source_state"]["source_status_short"]
          and "tools/CMakeLists.txt" in payload["source_state"]["source_status_short"],
      },
      {
          "name": "cmake_configure_succeeded",
          "pass": build_route["configure_returncode"] == 0 and build_route["configure_timed_out"] is False,
      },
      {
          "name": "capture_target_build_succeeded",
          "pass": build_route["build_returncode"] == 0
          and build_route["build_timed_out"] is False
          and build_route["executable_present"] is True,
      },
      {
          "name": "capture_executable_help_succeeded",
          "pass": build_route["help_returncode"] == 0,
      },
      {
          "name": "build_does_not_close_oracle_gate",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  correctness = {
      "checks": checks,
      "gate": "r0_boundary_capture_build",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-boundary-capture-build.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "build.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("configure_returncode", build_route["configure_returncode"]),
        ("build_returncode", build_route["build_returncode"]),
        ("executable_present", build_route["executable_present"]),
        ("help_returncode", build_route["help_returncode"]),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_boundary_capture_build",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"boundary capture build output: {out_dir}")
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
