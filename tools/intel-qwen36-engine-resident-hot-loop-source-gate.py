#!/usr/bin/env python3
"""Verify the default-off engine resident GPU hot-loop source cut.

This is source evidence only. It checks that seq95's selected route now has an
engine-owned hot-loop API, that decode-smoke can opt into it through an
environment gate, and that the generated source no longer keeps the resident
loop call as an inline lambda callback. It does not run tokens or claim speed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
SCHEMA_VERSION = "intel-qwen36-engine-resident-hot-loop-source-gate-v0"
DEFAULT_SEQ95 = (
    ROOT / "output/resident-decode-loop-overhead-root-gate-20260707Tseq95Z/metrics.json"
)
DEFAULT_GENERATE_ONLY = (
    ROOT / "output/engine-resident-gpu-hot-loop-generate-only-20260707Tseq96Z/result.json"
)
DEFAULT_GENERATED_CPP = (
    ROOT / "output/engine-resident-gpu-hot-loop-generate-only-20260707Tseq96Z/r2_gpu_decode_smoke.cpp"
)
DEFAULT_RESIDENT_HEADER = ROOT / "engine/include/intel_qwen36/resident_harness.hpp"
DEFAULT_RESIDENT_SOURCE = ROOT / "engine/src/resident_harness.cpp"
DEFAULT_DECODE_SMOKE = ROOT / "tools/intel-qwen36-r2-gpu-decode-smoke.py"
DEFAULT_SELF_TEST = ROOT / "engine/tests/self_test.cpp"
DEFAULT_OUT_DIR = ROOT / "output/engine-resident-gpu-hot-loop-source-gate-20260707Tseq96Z"


def _load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _num(value: Any) -> float:
  return float(value) if isinstance(value, (int, float)) else 0.0


def _marker(text: str, pattern: str) -> bool:
  return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def _source_markers(
    header: str, source: str, smoke: str, self_test: str, generated_cpp: str
) -> dict[str, list[dict[str, Any]]]:
  engine_markers = [
      {
          "name": "resident_gpu_hot_decode_loop_class_present",
          "pass": _marker(header, r"class\s+ResidentGpuHotDecodeLoop\b"),
      },
      {
          "name": "resident_gpu_hot_decode_loop_constructor_present",
          "pass": _marker(
              source,
              r"ResidentGpuHotDecodeLoop::ResidentGpuHotDecodeLoop"
          ),
      },
      {
          "name": "typed_runtime_adapter_present",
          "pass": _marker(
              header,
              r"ResidentGpuHotDecodeLoopRuntimeAdapter<\s*TokenFn,\s*DoneFn\s*>"
          ),
      },
      {
          "name": "hot_loop_calls_runtime_decode_token",
          "pass": "runtime.decode_token(step)" in header,
      },
      {
          "name": "hot_loop_calls_runtime_finish_session",
          "pass": "runtime.finish_session(done_context)" in header,
      },
  ]
  smoke_markers = [
      {
          "name": "env_default_off_switch_present",
          "pass": "IQ36_ENGINE_RESIDENT_GPU_HOT_LOOP" in smoke,
      },
      {
          "name": "no_cli_flag_added_for_hot_loop",
          "pass": "--engine-resident-gpu-hot-loop" not in smoke
          and "--resident-gpu-hot-loop" not in smoke,
      },
      {
          "name": "token_lambda_extracted_to_named_value",
          "pass": "auto resident_decode_token =" in smoke,
      },
      {
          "name": "done_lambda_extracted_to_named_value",
          "pass": "auto resident_decode_done =" in smoke,
      },
      {
          "name": "hot_loop_runtime_adapter_used",
          "pass": "make_resident_gpu_hot_decode_loop_runtime" in smoke,
      },
      {
          "name": "hot_loop_branch_uses_engine_api",
          "pass": "ResidentGpuHotDecodeLoop resident_gpu_hot_loop" in smoke,
      },
      {
          "name": "legacy_inline_resident_loop_lambda_removed",
          "pass": not _marker(
              smoke,
              r"resident_decode_loop\.run\(\s*std::cout,\s*resident_loop_config,\s*\[&\]",
          ),
      },
      {
          "name": "smoke_summary_reports_hot_loop_api",
          "pass": "engine_resident_gpu_hot_loop_enabled" in smoke
          and "resident_gpu_hot_loop_api" in smoke,
      },
  ]
  generated_markers = [
      {
          "name": "generated_source_hot_loop_env_parse",
          "pass": "IQ36_ENGINE_RESIDENT_GPU_HOT_LOOP" in generated_cpp,
      },
      {
          "name": "generated_source_hot_loop_engine_api",
          "pass": "ResidentGpuHotDecodeLoop resident_gpu_hot_loop" in generated_cpp,
      },
      {
          "name": "generated_source_runtime_adapter",
          "pass": "make_resident_gpu_hot_decode_loop_runtime" in generated_cpp,
      },
      {
          "name": "generated_source_legacy_inline_lambda_removed",
          "pass": not _marker(
              generated_cpp,
              r"resident_decode_loop\.run\(\s*std::cout,\s*resident_loop_config,\s*\[&\]",
          ),
      },
  ]
  test_markers = [
      {
          "name": "self_test_covers_hot_loop",
          "pass": "ResidentGpuHotDecodeLoop hot_loop" in self_test
          and "make_resident_gpu_hot_decode_loop_runtime" in self_test,
      },
  ]
  return {
      "engine": engine_markers,
      "decode_smoke": smoke_markers,
      "generated_cpp": generated_markers,
      "self_test": test_markers,
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  seq95 = _load_json(args.seq95)
  generate_only = _load_json(args.generate_only)
  header = args.resident_header.read_text(encoding="utf-8")
  source = args.resident_source.read_text(encoding="utf-8")
  smoke = args.decode_smoke.read_text(encoding="utf-8")
  self_test = args.self_test.read_text(encoding="utf-8")
  generated_cpp = args.generated_cpp.read_text(encoding="utf-8")
  markers = _source_markers(header, source, smoke, self_test, generated_cpp)
  marker_groups_passed = {
      group: all(row["pass"] for row in rows)
      for group, rows in markers.items()
  }
  seq95_speed_profile = seq95.get("speed_profile")
  seq95_speed_profile = (
      seq95_speed_profile if isinstance(seq95_speed_profile, dict) else {}
  )
  checks = [
      {
          "name": "seq95_selected_engine_hot_loop_extraction",
          "pass": seq95.get("required_checks_passed") is True
          and seq95.get("selected_next_route")
          == "engine_resident_gpu_hot_loop_extraction",
      },
      {
          "name": "seq95_target_bucket_still_floor_covering_source_cut",
          "pass": _num(seq95_speed_profile.get("unprofiled_ms_per_token"))
          >= _num(seq95_speed_profile.get("floor_gap_ms_per_token"))
          and 0.0 < _num(seq95_speed_profile.get("required_fraction_of_unprofiled")) <= 1.0,
      },
      {
          "name": "generate_only_artifact_is_source_only",
          "pass": generate_only.get("generate_only") is True
          and generate_only.get("engine_resident_gpu_hot_loop") is True
          and generate_only.get("generated_cpp") == _rel(args.generated_cpp),
      },
      {
          "name": "engine_hot_loop_source_markers_passed",
          "pass": marker_groups_passed["engine"],
          "detail": markers["engine"],
      },
      {
          "name": "decode_smoke_default_off_hot_loop_markers_passed",
          "pass": marker_groups_passed["decode_smoke"],
          "detail": markers["decode_smoke"],
      },
      {
          "name": "generated_cpp_hot_loop_markers_passed",
          "pass": marker_groups_passed["generated_cpp"],
          "detail": markers["generated_cpp"],
      },
      {
          "name": "self_test_hot_loop_marker_present",
          "pass": marker_groups_passed["self_test"],
          "detail": markers["self_test"],
      },
  ]
  required = all(row["pass"] for row in checks)
  return {
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
      "required_checks_passed": required,
      "speedup_claims_allowed": False,
      "disposition": (
          "engine_resident_gpu_hot_loop_source_cut_ready_for_explore"
          if required
          else "engine_resident_gpu_hot_loop_source_gate_failed"
      ),
      "selected_next_route": (
          "engine_resident_gpu_hot_loop_explore"
          if required
          else "fix_engine_resident_gpu_hot_loop_source"
      ),
      "next_action": (
          "Run the default-off hot-loop path as `decode-smoke --explore` with "
          "`IQ36_ENGINE_RESIDENT_GPU_HOT_LOOP=1` on the accepted frontier flags. "
          "Do not promote without confirm plus paired teacher-forced distribution "
          "evidence outside the 0.50% noise band."
          if required
          else "Fix failed source markers before any token-emitting run."
      ),
      "inputs": {
          "seq95": _rel(args.seq95),
          "generate_only": _rel(args.generate_only),
          "generated_cpp": _rel(args.generated_cpp),
          "resident_header": _rel(args.resident_header),
          "resident_source": _rel(args.resident_source),
          "decode_smoke": _rel(args.decode_smoke),
          "self_test": _rel(args.self_test),
      },
      "marker_groups_passed": marker_groups_passed,
      "markers": markers,
      "checks": checks,
      "seq95_budget": {
          "floor_gap_ms_per_token": _num(
              seq95_speed_profile.get("floor_gap_ms_per_token")),
          "unprofiled_ms_per_token": _num(
              seq95_speed_profile.get("unprofiled_ms_per_token")),
          "required_fraction_of_unprofiled": _num(
              seq95_speed_profile.get("required_fraction_of_unprofiled")),
          "gpu_loop_bookkeeping_ms_per_token": _num(
              seq95_speed_profile.get("gpu_loop_bookkeeping_ms_per_token")),
      },
  }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
  out_dir.mkdir(parents=True, exist_ok=False)
  (out_dir / "metrics.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  manifest = {
      "schema_version": payload["schema_version"],
      "workstream": payload["workstream"],
      "tool": "tools/intel-qwen36-engine-resident-hot-loop-source-gate.py",
      "inputs": payload["inputs"],
      "selected_next_route": payload["selected_next_route"],
      "speedup_claims_allowed": False,
  }
  (out_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in payload["checks"] if not row["pass"]]
  budget = payload["seq95_budget"]
  lines = [
      "# Engine Resident GPU Hot-Loop Source Gate",
      "",
      f"- required checks passed: `{str(payload['required_checks_passed']).lower()}`",
      f"- disposition: `{payload['disposition']}`",
      f"- selected next route: `{payload['selected_next_route']}`",
      f"- frontier gap: `{budget['floor_gap_ms_per_token']:.3f}` ms/token",
      f"- generated-callback unprofiled bucket: `{budget['unprofiled_ms_per_token']:.6f}` ms/token",
      f"- loop-bookkeeping profile bucket: `{budget['gpu_loop_bookkeeping_ms_per_token']:.6f}` ms/token",
      f"- failed checks: `{failed}`",
      "",
      payload["next_action"],
      "",
      "This is source evidence only. It does not run tokens or claim speed.",
      "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seq95", type=Path, default=DEFAULT_SEQ95)
  parser.add_argument("--generate-only", type=Path, default=DEFAULT_GENERATE_ONLY)
  parser.add_argument("--generated-cpp", type=Path, default=DEFAULT_GENERATED_CPP)
  parser.add_argument("--resident-header", type=Path, default=DEFAULT_RESIDENT_HEADER)
  parser.add_argument("--resident-source", type=Path, default=DEFAULT_RESIDENT_SOURCE)
  parser.add_argument("--decode-smoke", type=Path, default=DEFAULT_DECODE_SMOKE)
  parser.add_argument("--self-test", type=Path, default=DEFAULT_SELF_TEST)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  payload = compute(args)
  write_outputs(args.out_dir, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0 if payload["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
