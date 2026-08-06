#!/usr/bin/env python3
"""Gate the sole exact-attention 48-subgroup standalone component."""

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
    "triple-cohort-component-gate-v1")
OFFICIAL_PREFETCH_SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "triple-official-prefetch-component-gate-v1")
SOURCE = ROOT / "engine/gpu/opencl/exact_score_staging_component.cl"
RUNNER = ROOT / "engine/tools/exact_score_staging_component.cpp"
AUDIT = ROOT / (
    "output/openvino-exact-attention-triple-cohort-codegen-audit-"
    "20260724Tseq2145a-clean/result.json")
OFFICIAL_PREFETCH_AUDIT = ROOT / (
    "output/openvino-exact-attention-triple-official-prefetch-codegen-"
    "20260724Tseq2153-clean/result.json")
DECOMPOSITION = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/result.json")
CAPTURE_PROGRAM = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/raw/programs/capture/"
    "existing_shim.program.bin")
DUAL_PROGRAM = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/raw/programs/dual/"
    "existing_shim.program.bin")
TRIPLE_PROGRAM = ROOT / (
    "output/openvino-exact-attention-triple-cohort-codegen-"
    "20260724Tseq2145-clean/raw/triple-cohort/"
    "existing_shim.program.bin")
OFFICIAL_PREFETCH_PROGRAM = ROOT / (
    "output/openvino-exact-attention-triple-official-prefetch-codegen-"
    "20260724Tseq2153-clean/raw/triple-cohort/"
    "existing_shim.program.bin")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
BUILD_DIR = ROOT / "build/engine"
TARGET = "iq36-exact-score-staging-component"

MIN_SAMPLES = 20
DELTA_CAP_MS = -0.1175998
EXPECTED_PROGRAM_BYTES = 294_240
EXPECTED_PROGRAM_SHA256 = (
    "866b8e47aa48c8476aa44fbd85f6c5e63367d77558fa9cb33086df2d21122c91")
OFFICIAL_PREFETCH_PROGRAM_BYTES = 296_336
OFFICIAL_PREFETCH_PROGRAM_SHA256 = (
    "af8a8b5241ab2168ba1d3df2e3eec3e4818bcd3c1a8b312c1621d396ded669b8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=60)
  parser.add_argument("--memory-preflight-gib", type=float, default=8.0)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument(
      "--official-prefetch", action="store_true",
      help="run only the seq2153 official-address triple candidate")
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
    values: list[float], cap: float, seed: int = 214601,
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
      seed=seed)
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
          "intel-qwen36-exact-attention-triple-cohort-component-v1"
      and result.get("algorithm") ==
          "generated_m256_n16_onchip_kq_softmax_vs_pipeline"
      and result.get("context_tokens") == 131072
      and result.get("head_dim") == 256
      and result.get("query_heads") == 16
      and result.get("kv_heads") == 2
      and result.get("gqa_group") == 8
      and result.get("key_block") == 256
      and result.get("dual_subgroups") == 32
      and result.get("triple_kq_subgroups") == 16
      and result.get("triple_softmax_subgroups") == 16
      and result.get("triple_vs_subgroups") == 16
      and result.get("triple_total_subgroups") == 48
      and result.get("triple_workgroup_items") == 768
      and result.get("mandatory_key_value_payload_bytes") ==
          268_435_456
      and result.get("output_compared_values") == 4096
      and result.get("sample_count") == MIN_SAMPLES
      and result.get("schedule") ==
          "interleaved_dual_triple_triple_dual")


def distribution_pass(result: dict[str, Any]) -> bool:
  rows = result.get("paired_samples", [])
  if not isinstance(rows, list) or len(rows) != MIN_SAMPLES:
    return False
  for index, row in enumerate(rows):
    if row.get("sample") != index:
      return False
    if row.get("order") != (
        "dual_triple" if index % 2 == 0 else "triple_dual"):
      return False
    try:
      dual = float(row["dual_ms"])
      triple = float(row["triple_ms"])
      delta = float(row["delta_ms"])
      ratio = float(row["speedup_ratio"])
    except (KeyError, TypeError, ValueError):
      return False
    if any(not math.isfinite(value) for value in (
        dual, triple, delta, ratio)):
      return False
    if dual <= 0.0 or triple <= 0.0 or ratio <= 0.0:
      return False
    if not math.isclose(
        delta, triple - dual, rel_tol=0.0, abs_tol=3.0e-6):
      return False
    if not math.isclose(
        ratio, dual / triple, rel_tol=0.0, abs_tol=3.0e-9):
      return False
  return True


def resource_pass(result: dict[str, Any]) -> bool:
  resources = result.get("resources", {})
  if set(resources) != {"dual", "triple"}:
    return False
  expected = {
      "dual": {
          "register_count": 96,
          "spill_memory_bytes": 0,
          "local_memory_bytes": 59_424,
          "maximum_workgroup_items": 1024,
          "preferred_workgroup_multiple": 16,
      },
      "triple": {
          "register_count": 96,
          "spill_memory_bytes": 0,
          "local_memory_bytes": 61_472,
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
  triple = [float(row["triple_ms"]) for row in rows] if rows else []
  return "\n".join([
      "# Exact-attention triple-cohort component gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- output mismatches: `{result.get('output_mismatch_count')}`",
      f"- dual / triple median: "
      f"`{statistics.median(dual) if dual else None} / "
      f"{statistics.median(triple) if triple else None} ms/layer`",
      f"- triple-minus-dual median / 95% UCB / cap: "
      f"`{inference.get('point_estimate_ms')} / "
      f"{inference.get('upper_confidence_bound_ms')} / "
      f"{inference.get('cap_ms')} ms/layer`",
      f"- triple resources: `{result.get('resources', {}).get('triple')}`",
      f"- peak RSS / swaps: "
      f"`{payload['worker_resources'].get('maximum_resident_kib')} KiB / "
      f"{payload['worker_resources'].get('swaps')}`",
      "",
      "Graph, plugin, model, and product work remain closed unless this",
      "standalone bitwise and one-sided latency gate passes.",
      "",
  ])


def main() -> int:
  args = parse_args()
  official_prefetch = bool(args.official_prefetch)
  schema = OFFICIAL_PREFETCH_SCHEMA if official_prefetch else SCHEMA
  audit_path = (
      OFFICIAL_PREFETCH_AUDIT if official_prefetch else AUDIT)
  triple_program = (
      OFFICIAL_PREFETCH_PROGRAM if official_prefetch else TRIPLE_PROGRAM)
  expected_program_bytes = (
      OFFICIAL_PREFETCH_PROGRAM_BYTES
      if official_prefetch else EXPECTED_PROGRAM_BYTES)
  expected_program_sha256 = (
      OFFICIAL_PREFETCH_PROGRAM_SHA256
      if official_prefetch else EXPECTED_PROGRAM_SHA256)
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  preflight_bytes = int(args.memory_preflight_gib * 1024**3)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start-preflight", preflight_bytes, memory)

  required_paths = (
      SOURCE, RUNNER, audit_path, DECOMPOSITION, CAPTURE_PROGRAM,
      DUAL_PROGRAM, triple_program, CMAKE, ENV_SCRIPT)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing triple-cohort component inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  audit = load_json(audit_path)
  decomposition = load_json(DECOMPOSITION)
  source = SOURCE.read_text(encoding="utf-8")
  runner = RUNNER.read_text(encoding="utf-8")
  if official_prefetch:
    audit_pass = bool(
        audit.get("schema_version") ==
            "intel-qwen36-openvino-exact-attention-"
            "triple-official-prefetch-codegen-gate-v1"
        and audit.get("verdict") ==
            "admit_one_exact_attention_triple_official_prefetch_component"
        and audit.get("required_checks_passed") is True
        and audit.get("component_admitted") is True
        and audit.get("kernel_enqueue_admitted") is True
        and audit.get("graph_integration_admitted") is False
        and audit.get("model_worker_admitted") is False
        and audit.get("compiler_workers_launched") is True
        and audit.get("kernel_worker_launched") is False
        and audit.get("official_prefetch_mode") is True
        and audit.get("host_define") == "IQ36_COMPONENT_PROGRAM=11"
        and audit.get("disassembly", {}).get(
            "prefetch_send_count") == 16
        and audit.get("candidate_result", {}).get(
            "kernel_local_memory_bytes") == 61_472)
  else:
    audit_pass = bool(
        audit.get("verdict") ==
            "admit_one_exact_attention_triple_cohort_component"
        and audit.get("required_checks_passed") is True
        and audit.get("component_admitted") is True
        and audit.get("kernel_enqueue_admitted") is True
        and audit.get("graph_integration_admitted") is False
        and audit.get("model_worker_admitted") is False
        and audit.get("compiler_workers_launched") is False
        and audit.get("kernel_worker_launched") is False
        and audit.get("candidate_result", {}).get(
            "kernel_local_memory_bytes") == 61_472)
  decomposition_pass = bool(
      decomposition.get("required_checks_passed") is True
      and decomposition.get("compiler_resource_probe_admitted") is True
      and decomposition.get("result", {}).get("numeric_pass") is True
      and decomposition.get(
          "performance_inference", {}).get(
              "projected_onchip_saving", {}).get("rate_pass") is True)
  source_checks = {
      "fixed_triple_kernel":
          source.count(
              "__kernel void iq36_exact_score_triple_cohort(") == 1
          and "__attribute__((reqd_work_group_size(16, 48, 1)))"
              in source,
      "runner_has_one_fixed_component_mode":
          "RunTripleCohortComponent" in runner
          and "RunTripleCohort" in runner
          and "--triple-cohort" in runner
          and "interleaved_dual_triple_triple_dual" in runner
          and "constexpr int kSamples = 20;" in runner,
      "program_is_the_audited_sole_compile":
          triple_program.stat().st_size == expected_program_bytes
          and sha256(triple_program) == expected_program_sha256,
  }
  if official_prefetch:
    source_checks["official_prefetch_address_mode_is_fixed"] = bool(
        "IQ36_COMPONENT_TRIPLE_OFFICIAL_PREFETCH 11" in source
        and "const __global half* next_key =" in source
        and "(ulong)next_block * ugemm_kq_wg_tile_m * IQ36_D;"
            in source
        and "IQ36_CONTEXT - next_block * ugemm_kq_wg_tile_m,"
            in source
        and "ugemm_kq_wg_tile_m, IQ36_D, IQ36_D," in source)

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
      str(triple_program), "--triple-cohort"]
  sample_memory("before-sole-component-worker", preflight_bytes, memory)
  worker = (
      activated_timed_worker(
          worker_command, args.timeout_s,
          raw_dir / "component.time.txt")
      if build_ok and audit_pass and decomposition_pass
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
  inference = signed_delta_cap_inference(
      deltas, DELTA_CAP_MS,
      seed=215401 if official_prefetch else 214601)
  numeric_pass = bool(
      result.get("numeric_pass") is True
      and result.get("output_mismatch_count") == 0)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("clean_codegen_audit_admits_only_component", audit_pass),
      check("clean_decomposition_bound_is_retained",
            decomposition_pass),
      check("fixed_source_runner_and_program_contract",
            all(source_checks.values()), source_checks=source_checks),
      check("component_build_is_serial_j1", build_ok,
            build_command=build_command),
      check("sole_standalone_worker_executes_without_timeout",
            worker.returncode == 0, returncode=worker.returncode),
      check("fixed_128k_triple_cohort_shape", fixed_result(result)),
      check("twenty_pair_interleaved_distribution",
            distribution_pass(result)),
      check("triple_output_is_bitwise_equal_to_accepted_dual",
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
  if official_prefetch:
    verdict = (
        "promote_exact_attention_triple_official_prefetch_component"
        if required else
        "reject_exact_attention_triple_official_prefetch_component")
  else:
    verdict = (
        "promote_exact_attention_triple_cohort_component"
        if required else
        "reject_exact_attention_triple_cohort_component")
  sources = [
      {"path": display(path), "sha256": sha256(path)}
      for path in required_paths
  ]
  payload = {
      "schema_version": schema,
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
      "official_prefetch_mode": official_prefetch,
      "candidate_program": display(triple_program),
      "candidate_program_sha256": sha256(triple_program),
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
  triple_values = [
      float(row["triple_ms"]) for row in rows
      if isinstance(row, dict) and "triple_ms" in row]
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "numeric_pass": numeric_pass,
      "dual_median_ms":
          statistics.median(dual_values) if dual_values else None,
      "triple_median_ms":
          statistics.median(triple_values) if triple_values else None,
      "delta_median_ms": inference.get("point_estimate_ms"),
      "delta_ucb_ms": inference.get("upper_confidence_bound_ms"),
      "delta_cap_ms": DELTA_CAP_MS,
      "official_prefetch_mode": official_prefetch,
      "graph_compile_admitted": required,
      "model_worker_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
