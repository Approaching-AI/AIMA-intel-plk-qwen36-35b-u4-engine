#!/usr/bin/env python3
"""Gate the source contract for the packed whole-token decode schedule.

The gate compiles the native schedule entry, derives its byte census from the
locked GGUF, and binds that census to the active product budget and accepted
consensus-decode evidence. It is a source/design feasibility gate, not a timed
product-decode or speedup claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-packed-token-schedule-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
BUILD_DIR = ROOT / "build/engine"
SMOKE_TARGET = "iq36-packed-token-schedule-smoke"
ROUTE_GATE = (
    ROOT / "output/product-decode-route-gate-20260712Tseq731cleanZ/result.json")
STATE_BUDGET = (
    ROOT / "output/packed-token-state-budget-gate-20260712Tseq734cleanZ/result.json")
CONSENSUS_GATE = (
    ROOT / "output/native-consensus-gate-20260712Tseq730cleanZ/result.json")
SOURCE_PATHS = [
    ROOT / "engine/include/intel_qwen36/packed_token_schedule.hpp",
    ROOT / "engine/src/packed_token_schedule.cpp",
    ROOT / "engine/tools/packed_token_schedule_smoke.cpp",
    ROOT / "engine/CMakeLists.txt",
    ROOT / "engine/boundaries.json",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--build-dir", type=Path, default=BUILD_DIR)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/packed-token-schedule-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(command: list[str], timeout_s: int = 180) -> dict[str, Any]:
  try:
    proc = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
  except subprocess.TimeoutExpired as exc:
    return {
        "command": command,
        "returncode": 124,
        "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
        "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") +
            "\ntimeout",
    }
  return {
      "command": command,
      "returncode": proc.returncode,
      "stdout": proc.stdout,
      "stderr": proc.stderr,
  }


def git_output(*args: str) -> str:
  result = run(["git", *args], timeout_s=30)
  return result["stdout"].strip() if result["returncode"] == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def source_digest(paths: list[Path]) -> tuple[str, dict[str, str]]:
  aggregate = hashlib.sha256()
  per_file: dict[str, str] = {}
  for path in paths:
    payload = path.read_bytes()
    relative = str(path.relative_to(ROOT))
    digest = hashlib.sha256(payload).hexdigest()
    per_file[relative] = digest
    aggregate.update(relative.encode("utf-8"))
    aggregate.update(b"\0")
    aggregate.update(payload)
  return aggregate.hexdigest(), per_file


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  out.mkdir(parents=True, exist_ok=False)
  required = [args.model, args.cmake, ROUTE_GATE, STATE_BUDGET,
              CONSENSUS_GATE, *SOURCE_PATHS]
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  route = load_json(ROUTE_GATE)
  state_budget = load_json(STATE_BUDGET)
  consensus = load_json(CONSENSUS_GATE)
  configure = run([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B",
      str(args.build_dir), "-DCMAKE_BUILD_TYPE=Release",
  ])
  build = run([
      str(args.cmake), "--build", str(args.build_dir), "--target",
      SMOKE_TARGET, "-j", "8",
  ]) if configure["returncode"] == 0 else {
      "command": [], "returncode": 125, "stdout": "", "stderr":
          "configure failed",
  }
  binary = args.build_dir / SMOKE_TARGET
  smoke_run = run([str(binary), str(args.model)], timeout_s=60) \
      if build["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "", "stderr":
              "build failed",
      }
  try:
    smoke = json.loads(smoke_run["stdout"].strip())
  except (json.JSONDecodeError, TypeError):
    smoke = {}
  link_map = run(["ldd", str(binary)], timeout_s=30) \
      if binary.is_file() else {
          "command": [], "returncode": 125, "stdout": "", "stderr":
              "binary missing",
      }
  source_sha256, source_files = source_digest(SOURCE_PATHS)

  selected_route = route.get("selected_route", {})
  admission = state_budget.get("admission", {})
  budget = route.get("budget", {})
  q4_gb_s = float(budget.get("q4_measured_gb_s", 0.0))
  q6_gb_s = float(budget.get("q6_measured_gb_s", 0.0))
  q4_bytes = int(smoke.get("q4_stream_bytes_per_token", 0))
  q6_bytes = int(smoke.get("q6_stream_bytes_per_token", 0))
  f32_bytes = int(smoke.get("f32_stream_bytes_per_token", 0))
  state_read_bytes = int(
      smoke.get("resident_state_read_bytes_per_token", 0))
  state_write_bytes = int(
      smoke.get("resident_state_write_bytes_per_token", 0))
  strict_target_gb_s = float(
      admission.get("strict_stream_bandwidth_gb_s_min", 0.0))
  kernel_cap_ms = float(
      admission.get("kernel_schedule_ms_max", 0.0))
  q4_carrier_ms = q4_bytes / 1e6 / q4_gb_s if q4_gb_s > 0 else math.inf
  q6_carrier_ms = q6_bytes / 1e6 / q6_gb_s if q6_gb_s > 0 else math.inf
  other_carrier_ms = (
      (f32_bytes + state_read_bytes + state_write_bytes) /
      1e6 / strict_target_gb_s
      if strict_target_gb_s > 0 else math.inf)
  carrier_envelope_ms = q4_carrier_ms + q6_carrier_ms + other_carrier_ms
  carrier_margin_ms = kernel_cap_ms - carrier_envelope_ms
  kernel_required_gb_s = float(
      smoke.get("kernel_stream_bandwidth_gb_s_min", math.inf))
  lower_link_map = (link_map.get("stdout", "") +
                    link_map.get("stderr", "")).lower()

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("source_configure_and_build", configure["returncode"] == 0 and
            build["returncode"] == 0,
            configure_returncode=configure["returncode"],
            build_returncode=build["returncode"]),
      check("locked_model_schedule_smoke", smoke_run["returncode"] == 0 and
            smoke.get("required_checks_passed") is True,
            returncode=smoke_run["returncode"]),
      check("locked_gguf_tensor_coverage",
            smoke.get("covered_tensor_count") == 693 and
            smoke.get("active_weight_bytes_per_token") == 1_975_676_544 and
            smoke.get("resident_state_read_bytes_per_token") == 86_835_200 and
            smoke.get("resident_state_write_bytes_per_token") == 65_884_160 and
            smoke.get("strict_stream_bytes_per_token") == 2_128_395_904),
      check("o1_layer_source_shape",
            smoke.get("linear_layer_count") == 30 and
            smoke.get("full_attention_layer_count") == 10 and
            smoke.get("command_count") == 252),
      check("required_stage_family_coverage",
            smoke.get("selected_ffn_command_count") == 40 and
            smoke.get("linear_preconv_command_count") == 30 and
            smoke.get("attention_front_command_count") == 10),
      check("single_whole_token_host_submission_contract",
            smoke.get("compile_count") == 1 and
            smoke.get("token_submission_count") == 1 and
            smoke.get("host_input_boundary_count") == 1 and
            smoke.get("host_output_boundary_count") == 1 and
            smoke.get("intermediate_host_read_count") == 0),
      check("native_runtime_link_map",
            link_map["returncode"] == 0 and
            "openvino" not in lower_link_map and
            "libdnnl" not in lower_link_map and
            smoke.get("maps_native_only") is True),
      check("accepted_consensus_decode_anchor",
            consensus.get("required_checks_passed") is True and
            len(consensus.get("rows", [])) == 3 and
            all(row.get("candidate_exact_reference_match") is True
                for row in consensus.get("rows", []))),
      check("corrected_state_budget_admission_bound",
            route.get("required_checks_passed") is True and
            state_budget.get("required_checks_passed") is True and
            state_budget.get("disposition") ==
                "revise_schedule_census_and_backend_admission" and
            selected_route.get("id") ==
                "resident_packed_full_token_schedule_v5" and
            math.isclose(
                float(smoke.get("wall_ms_per_token_max", math.nan)),
                float(admission.get("full_token_wall_ms_max", math.inf)),
                abs_tol=1e-9) and
            math.isclose(
                float(smoke.get(
                    "kernel_schedule_ms_per_token_max", math.nan)),
                kernel_cap_ms, abs_tol=1e-9) and
            math.isclose(
                float(smoke.get(
                    "host_submit_ms_per_token_max", math.nan)),
                float(admission.get("host_boundary_ms_max", math.inf)),
                abs_tol=1e-9) and
            math.isclose(
                float(smoke.get(
                    "strict_stream_bandwidth_gb_s_min", math.nan)),
                strict_target_gb_s, abs_tol=1e-9)),
      check("measured_carriers_clear_kernel_stream_rate",
            finite(kernel_required_gb_s) and
            q4_gb_s >= kernel_required_gb_s and
            q6_gb_s >= kernel_required_gb_s,
            kernel_required_gb_s=kernel_required_gb_s,
            q4_measured_gb_s=q4_gb_s, q6_measured_gb_s=q6_gb_s),
      check("packed_carrier_envelope_fits_kernel_budget",
            finite(carrier_envelope_ms) and carrier_margin_ms > 0.0,
            carrier_envelope_ms=carrier_envelope_ms,
            carrier_margin_ms=carrier_margin_ms,
            q4_carrier_ms=q4_carrier_ms, q6_carrier_ms=q6_carrier_ms,
            f32_kv_carrier_ms=other_carrier_ms),
  ]
  passed = all(row["pass"] for row in checks)
  created_at = iso_now()
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "git": state,
      "source": {
          "aggregate_sha256": source_sha256,
          "files": source_files,
      },
      "inputs": {
          "model": str(args.model),
          "route_gate": str(ROUTE_GATE.relative_to(ROOT)),
          "state_budget": str(STATE_BUDGET.relative_to(ROOT)),
          "consensus_gate": str(CONSENSUS_GATE.relative_to(ROOT)),
      },
      "smoke": smoke,
      "budget": {
          "q4_carrier_ms": q4_carrier_ms,
          "q6_carrier_ms": q6_carrier_ms,
          "f32_state_carrier_ms": other_carrier_ms,
          "carrier_envelope_ms": carrier_envelope_ms,
          "kernel_schedule_ms_max": kernel_cap_ms,
          "carrier_margin_ms": carrier_margin_ms,
          "kernel_stream_bandwidth_gb_s_min": kernel_required_gb_s,
          "q4_measured_gb_s": q4_gb_s,
          "q6_measured_gb_s": q6_gb_s,
      },
      "checks": checks,
      "required_checks_passed": passed,
      "disposition": (
          "admit_packed_token_backend_implementation"
          if passed else "reject_packed_token_source_design"),
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "checks": checks,
      "required_checks_passed": passed,
      "product_promotion_ready": False,
      "speedup_claims_allowed": False,
  })
  write_json(out / "build.json", {
      "configure": configure,
      "build": build,
      "smoke_run": smoke_run,
      "link_map": link_map,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "artifact": str(out),
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "git": state,
      "source_sha256": source_sha256,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  metrics = [
      ("strict_stream_bytes_per_token",
       smoke.get("strict_stream_bytes_per_token")),
      ("kernel_stream_bandwidth_gb_s_min", kernel_required_gb_s),
      ("carrier_envelope_ms", carrier_envelope_ms),
      ("carrier_margin_ms", carrier_margin_ms),
      ("host_submission_count_per_token",
       smoke.get("token_submission_count")),
  ]
  with (out / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in metrics:
      fh.write(json.dumps({
          "metric": metric, "phase": "source_design", "value": value,
      }, sort_keys=True) + "\n")
  (out / "summary.md").write_text("\n".join([
      "# Packed whole-token schedule source gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- locked tensors / commands: `"
      f"{smoke.get('covered_tensor_count')} / {smoke.get('command_count')}`",
      f"- active weights / state read / state write bytes: `"
      f"{smoke.get('active_weight_bytes_per_token')} / "
      f"{smoke.get('resident_state_read_bytes_per_token')} / "
      f"{smoke.get('resident_state_write_bytes_per_token')}`",
      f"- strict stream bytes/token: `"
      f"{smoke.get('strict_stream_bytes_per_token')}`",
      f"- kernel stream rate required: `{kernel_required_gb_s:.3f} GB/s`",
      f"- measured Q4 / Q6 carriers: `{q4_gb_s:.3f} / {q6_gb_s:.3f} GB/s`",
      f"- carrier envelope / kernel cap: `"
      f"{carrier_envelope_ms:.3f} / {kernel_cap_ms:.3f} ms`",
      f"- arithmetic margin: `{carrier_margin_ms:.3f} ms`",
      "- host contract: `one token submission; no intermediate read`", "",
      "This admits backend implementation against a compiled whole-token "
      "command stream. It is not a timed product result or speedup claim.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out),
      "pass": passed,
      "carrier_envelope_ms": carrier_envelope_ms,
      "carrier_margin_ms": carrier_margin_ms,
      "kernel_stream_bandwidth_gb_s_min": kernel_required_gb_s,
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
