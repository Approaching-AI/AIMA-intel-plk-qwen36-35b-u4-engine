#!/usr/bin/env python3
"""Resolve and optionally stage the llama.cpp source build route for R0."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess

import iq36_local
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-llama-source-build-route-v0"
DEFAULT_HOST = "local"
UPSTREAM_REPO = "https://github.com/ggml-org/llama.cpp"
UPSTREAM_GIT = "https://github.com/ggml-org/llama.cpp.git"
UPSTREAM_API = "https://api.github.com/repos/ggml-org/llama.cpp"
INTEL_ENV = "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh"
REMOTE_STAGE_ROOT = "/home/intel/intel-qwen36-r0/source"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument(
      "--stage-target-source",
      action="store_true",
      help="Clone/fetch the exact upstream llama.cpp commit into target user space.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-llama-source-build-route-<UTC>.",
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


def bool_text(value: Any) -> bool:
  return value == "true"


def resolve_upstream_commit(short_sha: str, raw_dir: Path) -> dict[str, Any]:
  url = f"{UPSTREAM_API}/commits/{short_sha}"
  request = urllib.request.Request(
      url,
      headers={
          "Accept": "application/vnd.github+json",
          "User-Agent": "intel-qwen36-r0-source-route",
      },
  )
  try:
    with urllib.request.urlopen(request, timeout=30) as response:
      status = response.status
      body = response.read().decode("utf-8", errors="replace")
  except Exception as exc:  # noqa: BLE001 - artifact should record the failure.
    (raw_dir / "upstream_commit_error.txt").write_text(str(exc), encoding="utf-8")
    return {
        "api_status": None,
        "commit_message_first_line": None,
        "commit_short": short_sha,
        "html_url": f"{UPSTREAM_REPO}/commit/{short_sha}",
        "resolved": False,
        "sha": None,
        "url": url,
    }
  (raw_dir / "upstream_commit.json").write_text(body, encoding="utf-8")
  data = json.loads(body)
  message = data.get("commit", {}).get("message", "")
  if not isinstance(message, str):
    message = ""
  sha = data.get("sha")
  return {
      "api_status": status,
      "archive_url": f"{UPSTREAM_REPO}/archive/{sha}.tar.gz" if isinstance(sha, str) else None,
      "commit_message_first_line": message.splitlines()[0] if message else None,
      "commit_short": short_sha,
      "html_url": data.get("html_url") or f"{UPSTREAM_REPO}/commit/{short_sha}",
      "resolved": status == 200 and isinstance(sha, str) and sha.startswith(short_sha),
      "sha": sha,
      "url": url,
  }


def stage_target_source(host: str, full_sha: str, raw_dir: Path) -> dict[str, Any]:
  stage_dir = f"{REMOTE_STAGE_ROOT}/llama.cpp-{full_sha}"
  tmp_dir = f"{stage_dir}.tmp"
  script = "\n".join([
      "set -u",
      f"sha={shlex.quote(full_sha)}",
      f"stage_root={shlex.quote(REMOTE_STAGE_ROOT)}",
      f"stage_dir={shlex.quote(stage_dir)}",
      f"tmp_dir={shlex.quote(tmp_dir)}",
      "mkdir -p \"$stage_root\"",
      "if test -d \"$stage_dir/.git\"; then",
      "  printf 'stage_status=already_present\\n'",
      "else",
      "  rm -rf \"$tmp_dir\"",
      "  git init \"$tmp_dir\"",
      "  cd \"$tmp_dir\"",
      f"  git remote add origin {shlex.quote(UPSTREAM_GIT)}",
      "  git fetch --depth 1 origin \"$sha\"",
      "  git checkout --detach FETCH_HEAD",
      "  cd \"$stage_root\"",
      "  mv \"$tmp_dir\" \"$stage_dir\"",
      "  printf 'stage_status=created\\n'",
      "fi",
      "cd \"$stage_dir\"",
      "printf 'stage_dir=%s\\n' \"$stage_dir\"",
      "printf 'source_rev_parse='; git rev-parse HEAD",
      "printf 'source_describe='; git describe --tags --always --dirty 2>/dev/null || true",
      "printf 'cmakelists_present='; test -f CMakeLists.txt && echo true || echo false",
      "printf 'llama_cpp_file_present='; test -f src/llama.cpp && echo true || echo false",
      "printf 'ggml_dir_present='; test -d ggml && echo true || echo false",
      "printf 'stage_file_count='; find . -maxdepth 2 -type f | wc -l",
      f"printf 'intel_env_present='; test -f {shlex.quote(INTEL_ENV)} && echo true || echo false",
      f"if test -f {shlex.quote(INTEL_ENV)}; then . {shlex.quote(INTEL_ENV)}; fi",
      "printf 'cmake_present_after_env='; command -v cmake >/dev/null 2>&1 && echo true || echo false",
      "printf 'gxx_present_after_env='; command -v g++ >/dev/null 2>&1 && echo true || echo false",
      "printf 'ninja_present_after_env='; command -v ninja >/dev/null 2>&1 && echo true || echo false",
  ])
  result = run_target(host, script, timeout_s=240)
  (raw_dir / "target_source_stage.stdout").write_text(result["stdout"], encoding="utf-8")
  (raw_dir / "target_source_stage.stderr").write_text(result["stderr"], encoding="utf-8")
  values = parse_key_values(result["stdout"])
  return {
      "cmakelists_present": bool_text(values.get("cmakelists_present")),
      "cmake_present_after_env": bool_text(values.get("cmake_present_after_env")),
      "ggml_dir_present": bool_text(values.get("ggml_dir_present")),
      "gxx_present_after_env": bool_text(values.get("gxx_present_after_env")),
      "intel_env_present": bool_text(values.get("intel_env_present")),
      "llama_cpp_file_present": bool_text(values.get("llama_cpp_file_present")),
      "ninja_present_after_env": bool_text(values.get("ninja_present_after_env")),
      "returncode": result["returncode"],
      "source_describe": values.get("source_describe"),
      "source_rev_parse": values.get("source_rev_parse"),
      "stage_dir": values.get("stage_dir") or stage_dir,
      "stage_file_count": int(values["stage_file_count"]) if values.get("stage_file_count", "").isdigit() else None,
      "stage_status": values.get("stage_status"),
      "timed_out": result["timed_out"],
  }


def build_summary(payload: dict[str, Any]) -> str:
  source = payload["source_route"]
  stage = payload["target_source_stage"]
  lines = [
      "# R0 llama.cpp Source Build Route",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- target llama build: `{source['target_runtime_version']['raw_version_line']}`",
      f"- upstream commit resolved: `{str(source['upstream_commit']['resolved']).lower()}`",
      f"- upstream commit: `{source['upstream_commit']['sha']}`",
      f"- stage attempted: `{str(stage['attempted']).lower()}`",
      f"- stage status: `{stage.get('stage_status')}`",
      f"- stage dir: `{stage.get('stage_dir')}`",
      f"- source ready for instrumentation: `{str(stage['source_ready_for_instrumentation']).lower()}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This artifact identifies and stages the source route only. It does not",
      "add instrumentation, build a runtime, dump tensors, or create an oracle bundle.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-llama-source-build-route-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  boundary_preflight_path = latest(
      "r0-boundary-capture-route-preflight-*",
      "preflight.json",
  )
  if boundary_preflight_path is None:
    raise SystemExit("no boundary capture route preflight artifact found under output/")
  boundary_preflight = load_json(boundary_preflight_path)
  target = boundary_preflight.get("target_footholds", {})
  runtime_version = target.get("llama_runtime_version", {})
  if not isinstance(runtime_version, dict):
    raise SystemExit("latest boundary route preflight lacks llama runtime version")
  commit_short = runtime_version.get("commit_short")
  if not isinstance(commit_short, str) or not commit_short:
    raise SystemExit("latest boundary route preflight lacks commit_short")
  upstream = resolve_upstream_commit(commit_short, raw_dir)
  stage_result: dict[str, Any] = {
      "attempted": args.stage_target_source,
      "source_ready_for_instrumentation": False,
      "stage_dir": None,
      "stage_status": None,
  }
  if args.stage_target_source and upstream.get("resolved") is True:
    sha = upstream.get("sha")
    if not isinstance(sha, str):
      raise SystemExit("upstream commit resolved without full SHA")
    stage_result.update(stage_target_source(args.host, sha, raw_dir))
    stage_result["attempted"] = True
    stage_result["source_ready_for_instrumentation"] = (
        stage_result.get("returncode") == 0
        and stage_result.get("source_rev_parse") == sha
        and stage_result.get("cmakelists_present") is True
        and stage_result.get("llama_cpp_file_present") is True
        and stage_result.get("ggml_dir_present") is True
    )

  source_route = {
      "build_env_script": INTEL_ENV,
      "instrumentation_route_status": (
          "source_staged_ready_for_instrumentation"
          if stage_result.get("source_ready_for_instrumentation") is True
          else "source_route_identified_not_staged"
          if not args.stage_target_source
          else "source_stage_failed"
      ),
      "required_next_steps": [
          "add gated llama.cpp graph/boundary instrumentation for the queued short_math_001 source token",
          "build the instrumented runtime with the existing Intel env toolchain",
          "run the instrumented runtime against the locked GGUF to emit boundary input/output tensor payloads",
          "assemble and validate the full oracle bundle before resident harness load",
      ],
      "target_runtime_version": runtime_version,
      "upstream_commit": upstream,
      "upstream_repo": UPSTREAM_REPO,
  }
  payload = {
      "created_at": created_at,
      "evidence": {
          "boundary_capture_route_preflight": rel(boundary_preflight_path.parent),
          "raw_dir": rel(raw_dir),
      },
      "host": args.host,
      "r0_oracle_gate_closed": False,
      "schema_version": SCHEMA_VERSION,
      "source_route": source_route,
      "target_source_stage": stage_result,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "latest_boundary_route_preflight_available",
          "pass": boundary_preflight.get("schema_version")
          == "intel-qwen36-r0-boundary-capture-route-preflight-v0"
          and boundary_preflight.get("route_decision", {}).get("r0_oracle_gate_closed")
          is False,
      },
      {
          "name": "target_runtime_commit_matches_llama_b9518",
          "pass": runtime_version.get("build_number") == 9518
          and runtime_version.get("commit_short") == "7c158fbb4",
          "target_runtime_version": runtime_version,
      },
      {
          "name": "official_upstream_commit_resolved",
          "pass": upstream.get("resolved") is True
          and isinstance(upstream.get("sha"), str)
          and upstream["sha"].startswith("7c158fbb4"),
          "upstream_commit": upstream,
      },
      {
          "name": "intel_env_toolchain_available",
          "pass": target.get("intel_env_build_tools_present") is True,
      },
      {
          "name": "target_source_staged_when_requested",
          "pass": (
              stage_result.get("source_ready_for_instrumentation") is True
              if args.stage_target_source
              else stage_result.get("attempted") is False
          ),
          "target_source_stage": stage_result,
      },
      {
          "name": "source_route_does_not_close_oracle_gate",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  correctness = {
      "checks": checks,
      "gate": "r0_llama_source_build_route",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-llama-source-build-route.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "source-route.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("upstream_commit_resolved", upstream.get("resolved") is True),
        ("target_source_stage_attempted", args.stage_target_source),
        (
            "source_ready_for_instrumentation",
            stage_result.get("source_ready_for_instrumentation") is True,
        ),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_llama_source_build_route",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"llama source build route output: {out_dir}")
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
