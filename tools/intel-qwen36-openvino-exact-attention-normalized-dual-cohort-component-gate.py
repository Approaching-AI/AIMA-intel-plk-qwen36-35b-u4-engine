#!/usr/bin/env python3
"""Gate the sole exact-attention normalized-F16 two-cohort component."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "normalized-dual-cohort-component-gate-v1")
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
RUNNER = ROOT / "engine/tools/exact_score_staging_component.cpp"
BOUND = ROOT / (
    "output/openvino-exact-attention-normalized-dual-cohort-bound-"
    "20260724Tseq2147-clean/result.json")
CODEGEN = ROOT / (
    "output/openvino-exact-attention-normalized-dual-cohort-codegen-"
    "20260724Tseq2148-clean/result.json")
CAPTURE_PROGRAM = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/raw/programs/capture/"
    "existing_shim.program.bin")
DUAL_PROGRAM = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/raw/programs/dual/"
    "existing_shim.program.bin")
NORMALIZED_PROGRAM = ROOT / (
    "output/openvino-exact-attention-normalized-dual-cohort-codegen-"
    "20260724Tseq2148-clean/raw/normalized-dual-cohort/"
    "existing_shim.program.bin")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-exact-score-staging-component"

MIN_SAMPLES = 20
DELTA_CAP_MS = -0.1175998
EXPECTED_PROGRAM_BYTES = 281_568
EXPECTED_PROGRAM_SHA256 = (
    "2c6588cf161ebdf94b2cd74a9a586758247156b7e0b0a7887d77bec186da922b")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=60)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s < 10:
    parser.error("--timeout-s must be at least 10")
  if args.memory_preflight_gib < 8.0:
    parser.error("--memory-preflight-gib must be at least 8")
  if args.memory_stop_gib < 4.0:
    parser.error("--memory-stop-gib must be at least 4")
  if args.memory_preflight_gib <= args.memory_stop_gib:
    parser.error("--memory-preflight-gib must exceed --memory-stop-gib")
  return args


def run(
    command: list[str], timeout: int,
) -> subprocess.CompletedProcess[str]:
  try:
    return subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)
  except subprocess.TimeoutExpired as error:
    stdout = (
        error.stdout.decode("utf-8", "replace")
        if isinstance(error.stdout, bytes) else error.stdout or "")
    stderr = (
        error.stderr.decode("utf-8", "replace")
        if isinstance(error.stderr, bytes) else error.stderr or "")
    return subprocess.CompletedProcess(
        command, 124, stdout, stderr + "\nworker timeout")


def activated_timed_worker(
    command: list[str], timeout: int, time_path: Path,
) -> subprocess.CompletedProcess[str]:
  shell_command = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      f"/usr/bin/timeout --signal=TERM --kill-after=5s {timeout}s "
      f"/usr/bin/time -v -o {shlex.quote(str(time_path))} " +
      " ".join(shlex.quote(part) for part in command))
  return run(["bash", "-lc", shell_command], timeout + 10)


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  patterns = {
      "maximum_resident_kib": r"Maximum resident set size \(kbytes\): (\d+)",
      "major_page_faults": r"Major \(requiring I/O\) page faults: (\d+)",
      "swaps": r"Swaps: (\d+)",
      "elapsed": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)",
      "exit_status": r"Exit status: (\d+)",
  }
  result: dict[str, Any] = {"raw": text}
  for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match:
      result[key] = (
          match.group(1) if key == "elapsed" else int(match.group(1)))
  return result


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.relative_to(ROOT))
  except ValueError:
    return str(path)


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, minimum_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  row = {
      "label": label,
      "available_bytes": available,
      "minimum_bytes": minimum_bytes,
      "pass": available >= minimum_bytes,
  }
  rows.append(row)
  if not row["pass"]:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {minimum_bytes} bytes")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [row for row in dirty if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def signed_delta_cap_inference(
    values: list[float], cap: float,
) -> dict[str, Any]:
  if len(values) != MIN_SAMPLES or any(
      not math.isfinite(value) for value in values):
    return {
        "error": "signed delta samples must be 20 finite values",
        "rate_pass": False,
        "cap_ms": cap,
    }
  translation = max(1.0, 1.0 - min(values), 1.0 - cap)
  translated = latency_cap_inference(
      [value + translation for value in values],
      cap=cap + translation, min_samples=MIN_SAMPLES,
      seed=214901)
  point = float(translated["point_estimate_ms"]) - translation
  upper = (
      float(translated["upper_confidence_bound_ms"]) - translation)
  median = statistics.median(values)
  mad = statistics.median(abs(value - median) for value in values)
  return {
      "method":
          "translation_invariant_one_sided_percentile_bootstrap_median",
      "confidence": translated["confidence"],
      "bootstrap_resamples": translated["bootstrap_resamples"],
      "bootstrap_seed": translated["bootstrap_seed"],
      "sample_count": translated["sample_count"],
      "minimum_sample_count": translated["minimum_sample_count"],
      "point_estimate_ms": point,
      "upper_confidence_bound_ms": upper,
      "cap_ms": cap,
      "sample_count_pass": translated["sample_count_pass"],
      "rate_pass":
          translated["sample_count_pass"] and upper <= cap,
      "translation_ms": translation,
      "dispersion": {
          "metric": "median_absolute_deviation_ms",
          "mad_ms": mad,
          "promotion_gate": False,
      },
  }


def fixed_result(result: dict[str, Any]) -> bool:
  return bool(
      result.get("schema_version") ==
          "intel-qwen36-exact-attention-normalized-dual-cohort-component-v1"
      and result.get("algorithm") ==
          "generated_m256_n16_onchip_kq_softmax_producer_vs_consumer"
      and result.get("context_tokens") == 131072
      and result.get("head_dim") == 256
      and result.get("query_heads") == 16
      and result.get("kv_heads") == 2
      and result.get("gqa_group") == 8
      and result.get("key_block") == 256
      and result.get("dual_subgroups") == 32
      and result.get("normalized_producer_subgroups") == 16
      and result.get("normalized_consumer_subgroups") == 16
      and result.get("normalized_total_subgroups") == 32
      and result.get("normalized_workgroup_items") == 512
      and result.get("mandatory_key_value_payload_bytes") ==
          268_435_456
      and result.get("output_compared_values") == 4096
      and result.get("sample_count") == MIN_SAMPLES
      and result.get("schedule") ==
          "interleaved_dual_normalized_normalized_dual")


def distribution_pass(result: dict[str, Any]) -> bool:
  rows = result.get("paired_samples", [])
  if not isinstance(rows, list) or len(rows) != MIN_SAMPLES:
    return False
  for index, row in enumerate(rows):
    if row.get("sample") != index:
      return False
    if row.get("order") != (
        "dual_normalized" if index % 2 == 0 else "normalized_dual"):
      return False
    try:
      dual = float(row["dual_ms"])
      normalized = float(row["normalized_ms"])
      delta = float(row["delta_ms"])
      ratio = float(row["speedup_ratio"])
    except (KeyError, TypeError, ValueError):
      return False
    if any(not math.isfinite(value) for value in (
        dual, normalized, delta, ratio)):
      return False
    if dual <= 0.0 or normalized <= 0.0 or ratio <= 0.0:
      return False
    if not math.isclose(
        delta, normalized - dual, rel_tol=0.0, abs_tol=3.0e-6):
      return False
    if not math.isclose(
        ratio, dual / normalized, rel_tol=0.0, abs_tol=3.0e-9):
      return False
  return True


def resource_pass(result: dict[str, Any]) -> bool:
  resources = result.get("resources", {})
  if set(resources) != {"dual", "normalized"}:
    return False
  expected = {
      "dual": {
          "register_count": 96,
          "spill_memory_bytes": 0,
          "local_memory_bytes": 59_424,
          "maximum_workgroup_items": 1024,
          "preferred_workgroup_multiple": 16,
      },
      "normalized": {
          "register_count": 96,
          "spill_memory_bytes": 0,
          "local_memory_bytes": 28_704,
          "maximum_workgroup_items": 1024,
          "preferred_workgroup_multiple": 16,
      },
  }
  return resources == expected


def summary(payload: dict[str, Any]) -> str:
  result = payload["result"]
  inference = payload["performance_inference"]
  rows = result.get("paired_samples", [])
  dual = [float(row["dual_ms"]) for row in rows] if rows else []
  normalized = [
      float(row["normalized_ms"]) for row in rows] if rows else []
  return "\n".join([
      "# Exact-attention normalized dual-cohort component gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- output mismatches: `{result.get('output_mismatch_count')}`",
      f"- dual / normalized median: "
      f"`{statistics.median(dual) if dual else None} / "
      f"{statistics.median(normalized) if normalized else None} ms/layer`",
      f"- normalized-minus-dual median / 95% UCB / cap: "
      f"`{inference.get('point_estimate_ms')} / "
      f"{inference.get('upper_confidence_bound_ms')} / "
      f"{inference.get('cap_ms')} ms/layer`",
      f"- normalized resources: "
      f"`{result.get('resources', {}).get('normalized')}`",
      f"- peak RSS / swaps: "
      f"`{payload['worker_resources'].get('maximum_resident_kib')} KiB / "
      f"{payload['worker_resources'].get('swaps')}`",
      "",
      "A pass admits one graph compile only. Plugin, model, and product work",
      "remain closed until later gates.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start-preflight", preflight_bytes, memory)

  required_paths = (
      SOURCE, RUNNER, BOUND, CODEGEN, CAPTURE_PROGRAM,
      DUAL_PROGRAM, NORMALIZED_PROGRAM, CMAKE, ENV_SCRIPT)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing normalized dual-cohort component inputs: " +
        ", ".join(missing))

  git = git_state(out_dir)
  bound = load_json(BOUND)
  codegen = load_json(CODEGEN)
  source = SOURCE.read_text(encoding="utf-8")
  runner = RUNNER.read_text(encoding="utf-8")
  bound_pass = bool(
      bound.get("verdict") ==
          "admit_one_exact_attention_normalized_dual_cohort_codegen_gate"
      and bound.get("required_checks_passed") is True
      and bound.get("compiler_resource_gate_admitted") is True
      and bound.get("component_admitted") is False
      and bound.get("kernel_enqueue_admitted") is False
      and bound.get("model_worker_admitted") is False
      and bound.get("recovery_contract", {}).get(
          "required_delta_cap_ms") == DELTA_CAP_MS)
  codegen_result = codegen.get("candidate_result", {})
  codegen_pass = bool(
      codegen.get("verdict") ==
          "admit_one_exact_attention_normalized_dual_cohort_component"
      and codegen.get("required_checks_passed") is True
      and codegen.get("component_admitted") is True
      and codegen.get("kernel_enqueue_admitted") is True
      and codegen.get("graph_integration_admitted") is False
      and codegen.get("plugin_build_admitted") is False
      and codegen.get("model_worker_admitted") is False
      and codegen.get("kernel_worker_launched") is False
      and codegen_result.get("kernel_register_count") == 96
      and codegen_result.get("kernel_spill_memory_bytes") == 0
      and codegen_result.get("kernel_local_memory_bytes") == 28_704
      and codegen_result.get("kernel_maximum_workgroup_size") == 1024)
  source_checks = {
      "fixed_normalized_dual_kernel":
          source.count(
              "__kernel void "
              "iq36_exact_score_normalized_dual_cohort(") == 1
          and "IQ36_COMPONENT_NORMALIZED_DUAL_COHORT 9" in source
          and "__attribute__((reqd_work_group_size(16, 32, 1)))"
              in source,
      "runner_has_one_fixed_component_mode":
          "RunNormalizedDualCohortComponent" in runner
          and "RunDualControl" in runner
          and "--normalized-dual-cohort" in runner
          and "interleaved_dual_normalized_normalized_dual" in runner
          and "constexpr int kSamples = 20;" in runner,
      "program_is_the_audited_sole_compile":
          NORMALIZED_PROGRAM.stat().st_size == EXPECTED_PROGRAM_BYTES
          and sha256(NORMALIZED_PROGRAM) == EXPECTED_PROGRAM_SHA256,
  }

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"]
  configure = run(configure_command, 300)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target", TARGET, "-j1"]
  build = run(build_command, args.timeout_s)
  write_json(raw_dir / "component-build.json", {
      "configure": {
          "command": configure_command,
          "returncode": configure.returncode,
          "stdout": configure.stdout,
          "stderr": configure.stderr,
      },
      "build": {
          "command": build_command,
          "returncode": build.returncode,
          "stdout": build.stdout,
          "stderr": build.stderr,
      },
  })
  sample_memory("after-serial-build", stop_bytes, memory)
  executable = BUILD_DIR / TARGET
  build_ok = bool(
      configure.returncode == 0 and build.returncode == 0
      and executable.is_file())

  worker_command = [
      str(executable), str(CAPTURE_PROGRAM), str(DUAL_PROGRAM),
      str(NORMALIZED_PROGRAM), "--normalized-dual-cohort"]
  sample_memory("before-sole-component-worker", preflight_bytes, memory)
  worker = (
      activated_timed_worker(
          worker_command, args.timeout_s,
          raw_dir / "component.time.txt")
      if build_ok and bound_pass and codegen_pass
      and all(source_checks.values()) else
      subprocess.CompletedProcess(
          worker_command, 1, "", "precondition failed"))
  sample_memory("after-sole-component-worker", stop_bytes, memory)
  (raw_dir / "component.stdout").write_text(
      worker.stdout, encoding="utf-8")
  (raw_dir / "component.stderr").write_text(
      worker.stderr, encoding="utf-8")
  write_json(raw_dir / "worker-command.json", {
      "command": worker_command,
      "returncode": worker.returncode,
  })
  result = parse_last_json(worker.stdout)
  worker_resources = parse_time(raw_dir / "component.time.txt")
  rows = result.get("paired_samples", [])
  deltas = [
      float(row.get("delta_ms", math.nan))
      for row in rows if isinstance(row, dict)]
  inference = signed_delta_cap_inference(deltas, DELTA_CAP_MS)
  numeric_pass = bool(
      result.get("numeric_pass") is True
      and result.get("output_mismatch_count") == 0)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("clean_seq2147_bound_admits_only_codegen", bound_pass),
      check("clean_seq2148_codegen_admits_only_component", codegen_pass),
      check("fixed_source_runner_and_program_contract",
            all(source_checks.values()), source_checks=source_checks),
      check("component_build_is_serial_j1", build_ok,
            build_command=build_command),
      check("sole_standalone_worker_executes_without_timeout",
            worker.returncode == 0, returncode=worker.returncode),
      check("fixed_128k_normalized_dual_cohort_shape",
            fixed_result(result)),
      check("twenty_pair_interleaved_distribution",
            distribution_pass(result)),
      check("normalized_output_is_bitwise_equal_to_accepted_dual",
            numeric_pass),
      check("runtime_resources_match_the_compile_gate",
            resource_pass(result), resources=result.get("resources", {})),
      check("one_sided_95pct_delta_ucb_clears_layer_kill_number",
            inference.get("rate_pass") is True,
            inference=inference),
      check("worker_rss_and_swap_are_bounded",
            int(worker_resources.get("maximum_resident_kib", 1 << 62))
                < 4 * 1024 * 1024
            and int(worker_resources.get("swaps", -1)) == 0,
            worker_resources=worker_resources),
      check("memory_guards_never_tripped",
            all(row["pass"] for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
      check("no_graph_plugin_or_model_worker_launched", True),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "promote_exact_attention_normalized_dual_cohort_component"
      if required else
      "reject_exact_attention_normalized_dual_cohort_component")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in required_paths
  ]
  payload = {
      "schema_version": SCHEMA,
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_promoted": required,
      "component_rejected": not required,
      "graph_compile_admitted": required,
      "graph_integration_admitted": False,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "standalone_component_worker_launched": worker.returncode in {0, 2},
      "model_worker_launched": False,
      "checks": checks,
      "result": result,
      "performance_inference": inference,
      "delta_cap_ms_per_layer": DELTA_CAP_MS,
      "worker_resources": worker_resources,
      "worker_command": worker_command,
      "memory_preflight_bytes": preflight_bytes,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "correctness.json", {
      "numeric_pass": numeric_pass,
      "output_compared_values": result.get("output_compared_values"),
      "output_mismatch_count": result.get("output_mismatch_count"),
  })
  write_json(out_dir / "performance.json", {
      "paired_samples": rows,
      "inference": inference,
  })
  write_json(out_dir / "resources.json", {
      "runtime_kernel_resources": result.get("resources", {}),
      "worker_resources": worker_resources,
  })
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": [
          "result.json", "correctness.json", "performance.json",
          "resources.json", "summary.md", "raw/component-build.json",
          "raw/component.stdout", "raw/component.stderr",
          "raw/component.time.txt", "raw/worker-command.json",
      ],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  dual_values = [
      float(row["dual_ms"]) for row in rows
      if isinstance(row, dict) and "dual_ms" in row]
  normalized_values = [
      float(row["normalized_ms"]) for row in rows
      if isinstance(row, dict) and "normalized_ms" in row]
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "numeric_pass": numeric_pass,
      "dual_median_ms":
          statistics.median(dual_values) if dual_values else None,
      "normalized_median_ms":
          statistics.median(normalized_values)
          if normalized_values else None,
      "delta_median_ms": inference.get("point_estimate_ms"),
      "delta_ucb_ms": inference.get("upper_confidence_bound_ms"),
      "delta_cap_ms": DELTA_CAP_MS,
      "graph_compile_admitted": required,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
