#!/usr/bin/env python3
"""Gate one O(1)-in-layer-count runtime loop over both accepted carriers."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-native-carrier-loop-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
PREFILL_ARTIFACT = (
    ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq664residentZ")
Q6_ARTIFACT = ROOT / "output/q6-rowstripe16-58gbps-gate-20260711Tseq658cleanZ"
CAPTURE = (
    ROOT / "output/onednn-q4k-routed-moe-component-gate-"
    "20260711Tseq646cleanZ/raw/capture/payloads")
PREFILL_CAP_US = 9526.177
Q6_FLOOR_GB_S = 58.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--prefill-artifact", type=Path,
                      default=PREFILL_ARTIFACT)
  parser.add_argument("--q6-artifact", type=Path, default=Q6_ARTIFACT)
  parser.add_argument("--capture", type=Path, default=CAPTURE)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.jobs <= 0 or args.timeout_s <= 0:
    parser.error("--jobs and --timeout-s must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/native-carrier-loop-gate-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected object")
  return value


def git_output(*args: str) -> str:
  run = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return run.stdout.strip() if run.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": False,
    }
  except subprocess.TimeoutExpired as error:
    return {
        "command": command,
        "returncode": 124,
        "stdout": error.stdout if isinstance(error.stdout, str) else "",
        "stderr": error.stderr if isinstance(error.stderr, str) else "",
        "timed_out": True,
    }


def run_env(command: list[str], env_script: Path,
            timeout_s: int) -> dict[str, Any]:
  shell = (
      f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
      f"export INTEL_FORCE_PROBE=b080 && {shlex.join(command)}")
  return run(["bash", "-lc", shell], timeout_s)


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw / f"{name}.command.json", {
      "command": result["command"],
      "returncode": result["returncode"],
      "timed_out": result["timed_out"],
  })
  (raw / f"{name}.stdout").write_text(
      str(result["stdout"]), encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      str(result["stderr"]), encoding="utf-8")


def parse_probe(result: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(result.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  prepacked = args.prefill_artifact / "raw/prepacked"
  gateup = args.prefill_artifact / "raw/gateup.0.bin"
  down = args.prefill_artifact / "raw/down.0.bin"
  grouped_kernel = ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"
  q6_kernel = ROOT / "engine/gpu/opencl/q4x8_matvec.cl"
  input_path = args.capture / "attn_post_norm-27__tok1023__ord0.bin"
  topk_path = args.capture / "ffn_moe_topk-27__tok1023__ord1.bin"
  router_path = args.capture / "ffn_moe_weights_norm-27__tok1023__ord2.bin"
  oracle_path = args.capture / "ffn_moe_out-27__tok1023__ord5.bin"
  required = [
      args.model, args.env_script, args.prefill_artifact / "result.json",
      args.q6_artifact / "gate.json", prepacked, gateup, down,
      grouped_kernel, q6_kernel, input_path, topk_path, router_path,
      oracle_path, ROOT / "engine/include/intel_qwen36/native_carrier_loop.hpp",
      ROOT / "engine/src/layer.cpp", ROOT / "engine/src/loop.cpp",
      ROOT / "engine/tools/native_carrier_loop_smoke.cpp",
      ROOT / "engine/boundaries.json",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  source_state = git_state()
  prefill_result = load_json(args.prefill_artifact / "result.json")
  q6_result = load_json(args.q6_artifact / "gate.json")
  build_dir = raw / "build"
  configure = run_env([
      "cmake", "-S", str(ROOT / "engine"), "-B", str(build_dir),
      "-DCMAKE_BUILD_TYPE=Release",
  ], args.env_script, args.timeout_s)
  write_run(raw, "configure", configure)
  build = run_env([
      "cmake", "--build", str(build_dir), f"-j{args.jobs}",
      "--target", "iq36-native-carrier-loop-smoke",
  ], args.env_script, args.timeout_s) if configure["returncode"] == 0 else {
      "command": [], "returncode": 125, "stdout": "",
      "stderr": "configure failed", "timed_out": False,
  }
  write_run(raw, "build", build)
  binary = build_dir / "iq36-native-carrier-loop-smoke"
  command = [
      str(binary), str(args.model), str(prepacked), str(gateup), str(down),
      str(grouped_kernel), str(q6_kernel), str(input_path), str(topk_path),
      "1024", str(router_path), str(oracle_path),
  ]
  smoke = run_env(command, args.env_script, args.timeout_s) \
      if build["returncode"] == 0 else {
          "command": command, "returncode": 125, "stdout": "",
          "stderr": "build failed", "timed_out": False,
      }
  write_run(raw, "smoke", smoke)
  probe = parse_probe(smoke)
  ldd = run(["ldd", str(binary)], args.timeout_s) \
      if binary.is_file() else {
          "command": ["ldd", str(binary)], "returncode": 125, "stdout": "",
          "stderr": "binary missing", "timed_out": False,
      }
  write_run(raw, "ldd", ldd)
  ldd_lower = str(ldd["stdout"]).lower()
  boundaries = load_json(ROOT / "engine/boundaries.json")
  target_registered = any(
      isinstance(row, dict) and
      row.get("target") == "iq36-native-carrier-loop-smoke" and
      row.get("source") == "tools/native_carrier_loop_smoke.cpp"
      for row in boundaries.get("infra_targets", []))

  checks = [
      check("repository_clean_at_gate", source_state["dirty"] is False),
      check("locked_model_path", args.model.resolve() == MODEL.resolve()),
      check("clean_seq658_q6_prerequisite",
            q6_result.get("required_checks_passed") is True and
            q6_result.get("git", {}).get("dirty") is False),
      check("clean_seq664_resident_prefill_prerequisite",
            prefill_result.get("required_checks_passed") is True and
            prefill_result.get("git", {}).get("dirty") is False),
      check("parameterized_loop_target_registered", target_registered),
      check("fresh_cmake_build_passed",
            configure["returncode"] == 0 and build["returncode"] == 0),
      check("dual_carrier_smoke_passed",
            smoke["returncode"] == 0 and
            probe.get("required_checks_passed") is True),
      check("arc_b390_selected_for_both_carriers",
            "B390" in str(probe.get("grouped_device_name", "")) and
            "B390" in str(probe.get("q6_device_name", ""))),
      check("one_parameterized_40_layer_implementation",
            probe.get("parameterized_layer_count") == 40 and
            probe.get("grouped_layer_count") == 1 and
            probe.get("q6_layer_count") == 1),
      check("resident_contexts_programs_and_weights_loaded_once",
            probe.get("grouped_context_create_count") == 1 and
            probe.get("grouped_program_load_count") == 3 and
            probe.get("grouped_run_count") == 1 and
            probe.get("q6_context_create_count") == 1 and
            probe.get("q6_run_count") == 1 and
            probe.get("q6_resident_weight_bytes") == 220200960),
      check("grouped_prefill_carrier_passed_in_loop",
            probe.get("prefill_mismatch_count") == 0 and
            float(probe.get("prefill_complete_minimum_us", 1e30)) <=
                PREFILL_CAP_US,
            required_cap_us=PREFILL_CAP_US),
      check("exact_q6_carrier_passed_in_loop",
            probe.get("q6_mismatch_count") == 0 and
            float(probe.get("q6_effective_packed_gb_s", 0.0)) >=
                Q6_FLOOR_GB_S,
            required_gb_s=Q6_FLOOR_GB_S),
      check("runtime_links_and_maps_no_onednn_or_openvino",
            ldd["returncode"] == 0 and "dnnl" not in ldd_lower and
            "openvino" not in ldd_lower and
            probe.get("maps_native_only") is True),
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "git": source_state,
      "prefill_prerequisite": str(args.prefill_artifact),
      "q6_prerequisite": str(args.q6_artifact),
      "probe": probe,
      "checks": checks,
      "required_checks_passed": passed,
      "disposition": (
          "accept_resident_parameterized_dual_carrier_loop"
          if passed else "reject_resident_parameterized_dual_carrier_loop"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "checks": [row for row in checks if
                 "carrier" in row["name"] or "runtime" in row["name"]],
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out),
      "git": source_state,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  (out / "summary.md").write_text("\n".join([
      "# Resident parameterized dual-carrier loop gate", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- commit: `{source_state['commit']}`",
      f"- prefill complete minimum: "
      f"`{probe.get('prefill_complete_minimum_us')} us` / "
      f"cap `{PREFILL_CAP_US} us`",
      f"- exact Q6: `{probe.get('q6_effective_packed_gb_s')} GB/s` / "
      f"floor `{Q6_FLOOR_GB_S} GB/s`",
      "", "This closes the resident carrier-boundary loop only. Teacher-forced",
      "distribution, deterministic tokens, context ladder, and product speed",
      "remain separate gates.", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out),
      "prefill_complete_minimum_us": probe.get("prefill_complete_minimum_us"),
      "q6_effective_packed_gb_s": probe.get("q6_effective_packed_gb_s"),
      "required_checks_passed": passed,
  }, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
