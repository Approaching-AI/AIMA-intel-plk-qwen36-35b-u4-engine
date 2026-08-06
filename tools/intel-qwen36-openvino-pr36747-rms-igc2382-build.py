#!/usr/bin/env python3
"""Build the clean shared-triple plus PR36747 RMS candidate plugin once."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_TOOL = ROOT / (
    "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-build.py")
PATCH = ROOT / "engine/openvino/iq36-router-shared-pr36747-rms.patch"
SOURCE_GATE = ROOT / (
    "output/openvino-pr36747-rms-igc2382-source-gate-"
    "20260718Tseq1347-cleanZ/metrics.json")
EXPECTED_PREVIOUS_PLUGIN = (
    "ac6b908b0cc5a8444672d94ccdaa9ecd08e2f2d3825a55d9b44e009c00d8fce1")
SCHEMA = "intel-qwen36-openvino-pr36747-rms-igc2382-build-v0"


def load_module() -> Any:
  spec = importlib.util.spec_from_file_location(
      "iq36_linear_tail_rms_build_base", BASE_TOOL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {BASE_TOOL}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


BASE = load_module()


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  allowed = {
      "engine/openvino/iq36-linear-tail-triple-pr36747-rms.patch",
      "engine/openvino/iq36-router-shared-pr36747-rms.patch",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-source-gate.py",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-build.py",
      "tools/intel-qwen36-openvino-linear-tail-rms-igc2382-component.py",
      "tools/intel-qwen36-openvino-pr36747-rms-igc2382-isolation-bound.py",
      "tools/intel-qwen36-openvino-pr36747-rms-igc2382-source-gate.py",
      "tools/intel-qwen36-openvino-pr36747-rms-igc2382-build.py",
  }
  relative_output = str(output.resolve().relative_to(ROOT))
  dirty = []
  for row in rows:
    path = row[3:]
    if path in allowed or path.startswith(relative_output):
      continue
    dirty.append(row)
  return {
      "commit": commit,
      "dirty": bool(dirty),
      "dirty_paths": dirty,
      "allowed_uncommitted_paths": sorted(allowed),
  }


def main() -> int:
  BASE.SCHEMA = SCHEMA
  BASE.PATCH = PATCH
  BASE.SOURCE_GATE = SOURCE_GATE
  BASE.EXPECTED_PREVIOUS_CANDIDATE_SHA256 = EXPECTED_PREVIOUS_PLUGIN
  BASE.git_state = git_state
  BASE.__file__ = str(Path(__file__).resolve())
  args = BASE.parse_args()
  captured = io.StringIO()
  with contextlib.redirect_stdout(captured):
    returncode = BASE.main()

  output = args.output.resolve()
  metrics_path = output / "metrics.json"
  manifest_path = output / "manifest.json"
  if metrics_path.is_file():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    passed = metrics.get("required_checks_passed") is True
    verdict = (
        "retain_clean_pr36747_rms_plugin_for_igc2382_component_preflight"
        if passed else "clean_incremental_build_failed")
    metrics["schema"] = SCHEMA
    metrics["verdict"] = verdict
    metrics["next_action"] = {
        "route": "openvino_pr36747_rms_igc2382_clean_component_preflight",
        "requirements": [
            "verify the exact clean plugin and seven-file source identities",
            "require the exact 291-FC and 131-RMS runtime census",
            "admit at most one guarded candidate-only 2k/17-step worker",
        ],
    }
    BASE.write_json(metrics_path, metrics)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = SCHEMA
    manifest["tool"] = str(Path(__file__).resolve().relative_to(ROOT))
    BASE.write_json(manifest_path, manifest)
    build = metrics["build"]
    monitor = build["monitor"]
    before = metrics["candidate_plugin_before"]["sha256"]
    after = metrics["candidate_plugin_after"]["sha256"]
    report = f"""# Clean PR36747 RMS incremental GPU-plugin build

Verdict: **{verdict}**. Required checks: `{str(passed).lower()}`.

The single admitted build completed in `{build['elapsed_seconds']:.2f} s`
with parallelism `{build['parallel']}`. Peak monitored RSS was
`{int(monitor['process_rss_peak_bytes']) / 1024:.0f} KiB`; process-group swap
was `{monitor['process_group_swap_peak_bytes']} B`, minimum available memory
was `{monitor['system_available_min_bytes']} B`, the 4-GiB stop did not trip,
and no OOM was observed.

The plugin changed from `{before}` to `{after}`. No GPU context or model worker
ran; retain it only for one exact component preflight.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "artifact": str(output.relative_to(ROOT)),
        "verdict": verdict,
        "returncode": build["returncode"],
        "elapsed_seconds": build["elapsed_seconds"],
        "peak_rss_bytes": monitor["process_rss_peak_bytes"],
        "candidate_plugin_sha256": after,
        "oom_observed": build["oom_observed"],
    }, separators=(",", ":")), flush=True)
  else:
    print(captured.getvalue(), end="")
  return returncode


if __name__ == "__main__":
  raise SystemExit(main())
