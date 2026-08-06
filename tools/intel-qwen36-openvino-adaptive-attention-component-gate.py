#!/usr/bin/env python3
"""Gate the matched 32k/64k adaptive block32-I8 attention component."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from iq36_perf_inference import latency_cap_inference


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SOURCE = ROOT / "engine/gpu/opencl/direct_i8_hotcold_gqa_decode.cl"
RUNNER = ROOT / "engine/tools/adaptive_i8_hotcold_gqa_decode.cpp"
BOUNDARIES = ROOT / "engine/boundaries.json"
STATUS = ROOT / "doc/active" / WS / "STATUS.md"
ROUTES = ROOT / "doc/active" / WS / "routes-ledger.json"
SOURCE_BOUND = ROOT / (
    "output/openvino-adaptive-attention-component-bound-"
    "20260720Tseq1672-clean/bound.json")
NUMERIC_BOUND = ROOT / (
    "output/openvino-adaptive-attention-deterministic-bound-"
    "20260720Tseq1671-all10-step178/bound.json")
BUILD_DIR = ROOT / "build/engine"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
TARGETS = {
    512: "iq36-adaptive-i8-hotcold-gqa-decode-top512",
    256: "iq36-adaptive-i8-hotcold-gqa-decode-top256",
}
CAPS_MS = {512: 0.7923170134317261, 256: 0.7247750157917522}
WEIGHTS = {512: 2, 256: 8}
WEIGHTED_CAP_MS = 7.38283415319747
MIN_SAMPLES = 20


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.timeout_s <= 0:
    parser.error("--timeout-s must be positive")
  if args.memory_stop_gib <= 0.0:
    parser.error("--memory-stop-gib must be positive")
  return args


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      command, cwd=ROOT, check=False, capture_output=True, text=True,
      encoding="utf-8", errors="replace", timeout=timeout)


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_last_json(stdout: str) -> dict[str, Any]:
  for line in reversed(stdout.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
  dirty = run(["git", "status", "--porcelain"], 30).stdout.splitlines()
  try:
    out_rel = str(out_dir.relative_to(ROOT))
  except ValueError:
    out_rel = ""
  dirty = [row for row in dirty if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def parse_time(path: Path) -> dict[str, Any]:
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  patterns = {
      "maximum_resident_kib": r"Maximum resident set size \(kbytes\): (\d+)",
      "major_page_faults": r"Major \(requiring I/O\) page faults: (\d+)",
      "swaps": r"Swaps: (\d+)",
      "elapsed": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)",
  }
  result: dict[str, Any] = {"raw": text}
  for key, pattern in patterns.items():
    match = re.search(pattern, text)
    if match:
      result[key] = match.group(1) if key == "elapsed" else int(match.group(1))
  return result


def environment() -> dict[str, Any]:
  commands = {
      "hostname": ["hostname"],
      "kernel": ["uname", "-a"],
      "bios_version": [
          "bash", "-lc", "head -n 1 /sys/class/dmi/id/bios_version"],
      "opencl": [
          "bash", "-lc",
          f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && clinfo -l"],
  }
  result: dict[str, Any] = {}
  for name, command in commands.items():
    completed = run(command, 60)
    result[name] = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
  return result


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def close(left: Any, right: float, tolerance: float = 1.0e-9) -> bool:
  try:
    return math.isclose(float(left), right, rel_tol=0.0, abs_tol=tolerance)
  except (TypeError, ValueError):
    return False


def numeric_and_selection_pass(result: dict[str, Any]) -> bool:
  validations = [result.get("base_validation", {}),
                 result.get("target_validation", {})]
  return bool(
      result.get("numeric_pass") is True
      and result.get("selection_pass") is True
      and result.get("required_checks_passed") is True
      and all(
          row.get("candidate_shape_pass") is True
          and row.get("union_exact") is True
          and row.get("union_deterministic") is True
          and row.get("implementation_vs_adaptive", {}).get("finite") is True
          and float(row.get("implementation_vs_adaptive", {}).get(
              "relative_l2", 1.0)) <= 1.0e-5
          and float(row.get("adaptive_vs_exact", {}).get(
              "relative_l2", 1.0)) <= 1.0e-4
          for row in validations))


def fixed_shape(result: dict[str, Any], topk: int) -> bool:
  return bool(
      result.get("algorithm") == "adaptive_block32_i8_exact_f16_correction"
      and result.get("topk_per_query") == topk
      and result.get("local_topk_per_chunk") == 64
      and result.get("chunk_tokens") == 512
      and result.get("hot_tokens") == 16384
      and result.get("base_context_tokens") == 32768
      and result.get("target_context_tokens") == 65536
      and result.get("sample_count") == MIN_SAMPLES
      and result.get("schedule") == "interleaved_base_target_target_base"
      and result.get("dispatches_per_context") == 4)


def distribution_pass(result: dict[str, Any]) -> bool:
  samples = result.get("paired_samples", [])
  timing_keys = {"scan_ms", "select_ms", "correct_ms", "update_ms", "total_ms"}
  if len(samples) != MIN_SAMPLES:
    return False
  for index, row in enumerate(samples):
    if row.get("order") != ("base_target" if index % 2 == 0 else "target_base"):
      return False
    if set(row.get("base", {})) != timing_keys or set(row.get("target", {})) != timing_keys:
      return False
    differential = float(row.get("differential_ms", math.nan))
    expected = float(row["target"]["total_ms"]) - float(row["base"]["total_ms"])
    if not math.isfinite(differential) or differential <= 0.0:
      return False
    if not math.isclose(differential, expected, rel_tol=0.0, abs_tol=2.0e-6):
      return False
  return True


def summary(payload: dict[str, Any]) -> str:
  rows = payload["performance_inference"]
  return "\n".join([
      "# Adaptive block32-I8 attention component gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- top512 median / UCB / cap: "
      f"`{rows['512']['point_estimate_ms']} / "
      f"{rows['512']['upper_confidence_bound_ms']} / "
      f"{rows['512']['cap_ms']} ms`",
      f"- top256 median / UCB / cap: "
      f"`{rows['256']['point_estimate_ms']} / "
      f"{rows['256']['upper_confidence_bound_ms']} / "
      f"{rows['256']['cap_ms']} ms`",
      f"- weighted 2/8 UCB / cap: "
      f"`{payload['weighted_ucb_ms']} / {payload['weighted_cap_ms']} ms/token`",
      f"- peak RSS top512 / top256: "
      f"`{payload['worker_resources']['512'].get('maximum_resident_kib')} / "
      f"{payload['worker_resources']['256'].get('maximum_resident_kib')} KiB`",
      "",
      "This gate admits only the standalone component. Graph integration,",
      "product workers, and performance claims remain separately gated.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required_paths = [
      SOURCE, RUNNER, BOUNDARIES, STATUS, ROUTES, SOURCE_BOUND,
      NUMERIC_BOUND, CMAKE, ENV_SCRIPT]
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit("missing component inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  source_text = SOURCE.read_text(encoding="utf-8")
  runner_text = RUNNER.read_text(encoding="utf-8")
  boundaries = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
  source_bound = json.loads(SOURCE_BOUND.read_text(encoding="utf-8"))
  numeric_bound = json.loads(NUMERIC_BOUND.read_text(encoding="utf-8"))
  registered = {
      row.get("target"): row for row in boundaries.get("infra_targets", [])
      if row.get("target") in TARGETS.values()}
  registration_pass = all(
      registered.get(target, {}).get("source") ==
          "tools/adaptive_i8_hotcold_gqa_decode.cpp"
      and registered[target].get("compile_definitions") ==
          [f"IQ36_ADAPTIVE_COMPONENT_TOPK={topk}"]
      for topk, target in TARGETS.items())
  bound_budget = source_bound.get("budget", {})
  bound_pass = bool(
      source_bound.get("schema") ==
          "intel-qwen36-openvino-adaptive-attention-component-bound-v1"
      and source_bound.get("all_required_checks_pass") is True
      and source_bound.get("adaptive_component_implementation_admitted") is True
      and source_bound.get("git", {}).get("dirty") is False
      and close(bound_budget.get("high_layer", {}).get(
          "matched_64k_minus_32k_ucb_cap_ms"), CAPS_MS[512])
      and close(bound_budget.get("low_layer", {}).get(
          "matched_64k_minus_32k_ucb_cap_ms"), CAPS_MS[256])
      and close(bound_budget.get(
          "weighted_matched_64k_minus_32k_ucb_cap_ms"), WEIGHTED_CAP_MS))
  numeric_bound_pass = bool(
      numeric_bound.get("all_required_checks_pass") is True
      and numeric_bound.get("git", {}).get("dirty") is False
      and numeric_bound.get("checks", {}).get("numeric_bound") is True
      and numeric_bound.get("checks", {}).get("traffic_bound") is True)
  source_checks = {
      "adaptive_macro_and_locked_geometry":
          "#define IQ36_ADAPTIVE_LOCAL_TOPK 64U" in source_text
          and "IQ36_CONTEXT_TOKENS != 32768U" in source_text
          and "IQ36_CONTEXT_TOKENS != 65536U" in source_text
          and "IQ36_HOT_TOKENS != 16384U" in source_text
          and "IQ36_ADAPTIVE_TOPK != 256U" in source_text
          and "IQ36_ADAPTIVE_TOPK != 512U" in source_text,
      "deterministic_f16_u16_selection":
          "iq36_adaptive_ordered_half" in source_text
          and "iq36_adaptive_record_better" in source_text
          and "(ushort)left < (ushort)right" in source_text,
      "four_dispatch_entrypoints": all(name in source_text for name in (
          "iq36_direct_i8_hotcold_partial",
          "iq36_adaptive_select_reduce_union",
          "iq36_adaptive_correct_normalize",
          "iq36_direct_i8_update_state")),
      "scan_score_sidecar_and_partitioned_exact_correction":
          "__global half* approximate_cold_score" in source_text
          and "IQ36_ADAPTIVE_CORRECTION_PARTITIONS IQ36_COLD_CHUNK_COUNT"
              in source_text
          and "correction_completion" in source_text,
      "exact_sidecar_has_one_ordered_writer":
          "__global half* exact_cold_k" in source_text
          and "__global half* exact_cold_v" in source_text
          and "exact[((ulong)kv_head * IQ36_COLD_TOKENS + cold_token)"
              in source_text,
      "runner_owns_twenty_interleaved_pairs":
          "constexpr int kSamples = 20;" in runner_text
          and "interleaved_base_target_target_base" in runner_text
          and "differential_ms" in runner_text,
      "boundary_targets_registered": registration_pass,
  }
  sample_memory("after-source-audit", stop_bytes, memory)

  configure_command = [
      str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
      "-DCMAKE_BUILD_TYPE=Release"]
  configure = run(configure_command, 300)
  sample_memory("after-configure", stop_bytes, memory)
  build_command = [
      str(CMAKE), "--build", str(BUILD_DIR), "--target",
      TARGETS[512], TARGETS[256], "-j1"]
  build = run(build_command, 600)
  sample_memory("after-build", stop_bytes, memory)
  build_ok = configure.returncode == 0 and build.returncode == 0 and all(
      (BUILD_DIR / target).is_file() for target in TARGETS.values())
  write_json(raw_dir / "build.json", {
      "configure": {
          "command": configure_command, "returncode": configure.returncode,
          "stdout": configure.stdout, "stderr": configure.stderr},
      "build": {
          "command": build_command, "returncode": build.returncode,
          "stdout": build.stdout, "stderr": build.stderr},
  })

  results: dict[str, dict[str, Any]] = {}
  resources: dict[str, dict[str, Any]] = {}
  worker_codes: dict[str, int] = {}
  worker_commands: dict[str, list[str]] = {}
  for topk in (512, 256):
    label = str(topk)
    target = TARGETS[topk]
    executable = BUILD_DIR / target
    time_path = raw_dir / f"top{topk}.time.txt"
    shell_command = (
        f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
        f"/usr/bin/time -v -o {shlex.quote(str(time_path))} "
        f"{shlex.quote(str(executable))} {shlex.quote(str(SOURCE))}")
    command = ["bash", "-lc", shell_command]
    worker_commands[label] = command
    sample_memory(f"before-top{topk}-worker", stop_bytes, memory)
    completed = (
        run(command, args.timeout_s) if build_ok else
        subprocess.CompletedProcess(command, 1, "", "build failed"))
    sample_memory(f"after-top{topk}-worker", stop_bytes, memory)
    worker_codes[label] = completed.returncode
    results[label] = parse_last_json(completed.stdout)
    resources[label] = parse_time(time_path)
    (raw_dir / f"top{topk}.stdout").write_text(
        completed.stdout, encoding="utf-8")
    (raw_dir / f"top{topk}.stderr").write_text(
        completed.stderr, encoding="utf-8")
  write_json(raw_dir / "worker-commands.json", {
      topk: {"command": worker_commands[topk],
             "returncode": worker_codes[topk]}
      for topk in worker_commands})
  write_json(raw_dir / "environment.json", environment())

  inference: dict[str, dict[str, Any]] = {}
  for topk in (512, 256):
    label = str(topk)
    samples = [
        float(row.get("differential_ms", math.nan))
        for row in results[label].get("paired_samples", [])]
    try:
      inference[label] = latency_cap_inference(
          samples, cap=CAPS_MS[topk], min_samples=MIN_SAMPLES)
    except ValueError as error:
      inference[label] = {
          "error": str(error), "rate_pass": False, "cap_ms": CAPS_MS[topk]}
  weighted_ucb = sum(
      WEIGHTS[topk] * float(inference[str(topk)].get(
          "upper_confidence_bound_ms", math.inf))
      for topk in (512, 256))

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("clean_numeric_and_source_bounds", bound_pass and numeric_bound_pass,
            source_bound_pass=bound_pass, numeric_bound_pass=numeric_bound_pass),
      check("fixed_source_contract", all(source_checks.values()),
            source_checks=source_checks),
      check("component_build_serial_j1", build_ok, build_command=build_command),
      check("both_component_workers_execute",
            all(code == 0 for code in worker_codes.values()),
            returncodes=worker_codes),
      check("both_fixed_shapes",
            all(fixed_shape(results[str(topk)], topk)
                for topk in (512, 256))),
      check("both_twenty_pair_distributions",
            all(distribution_pass(results[str(topk)])
                for topk in (512, 256))),
      check("both_numeric_and_selection_oracles",
            all(numeric_and_selection_pass(results[str(topk)])
                for topk in (512, 256))),
      check("both_one_sided_95pct_ucbs_clear_layer_caps",
            all(inference[str(topk)].get("rate_pass") is True
                for topk in (512, 256)), inference=inference),
      check("weighted_two_high_eight_low_ucb_clears_cap",
            weighted_ucb <= WEIGHTED_CAP_MS,
            weighted_ucb_ms=weighted_ucb, cap_ms=WEIGHTED_CAP_MS),
      check("worker_rss_and_swap_are_bounded", all(
          int(resources[str(topk)].get("maximum_resident_kib", 1 << 62)) <
              4 * 1024 * 1024
          and int(resources[str(topk)].get("swaps", -1)) == 0
          for topk in (512, 256)), resources=resources),
      check("memory_guard_never_tripped",
            all(row["available_bytes"] >= stop_bytes for row in memory),
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory)),
  ]
  required = all(row["pass"] for row in checks)
  verdict = (
      "promote_adaptive_attention_standalone_component"
      if required else "reject_adaptive_attention_standalone_component")
  sources = [
      {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
      for path in (SOURCE, RUNNER, BOUNDARIES, SOURCE_BOUND, NUMERIC_BOUND)]
  payload = {
      "schema_version":
          "intel-qwen36-openvino-adaptive-attention-component-gate-v1",
      "workstream": WS,
      "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required,
      "component_promoted": required,
      "graph_integration_admitted": False,
      "product_worker_admitted": False,
      "long_worker_admitted": False,
      "product_claim_allowed": False,
      "checks": checks,
      "results": results,
      "performance_inference": inference,
      "weighted_ucb_ms": weighted_ucb,
      "weighted_cap_ms": WEIGHTED_CAP_MS,
      "worker_resources": resources,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "artifact": str(out_dir.relative_to(ROOT)),
      "created_at": payload["created_at"],
      "git": git,
      "required_checks_passed": required,
      "schema_version": payload["schema_version"],
      "sources": sources,
      "tool": str(Path(__file__).relative_to(ROOT)),
      "verdict": verdict,
      "workstream": WS,
  })
  write_json(out_dir / "correctness.json", {
      "checks": checks,
      "required_checks_passed": required,
      "graph_integration_admitted": False,
      "product_claim_allowed": False,
  })
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
    for topk in (512, 256):
      for index, row in enumerate(results[str(topk)].get("paired_samples", [])):
        stream.write(json.dumps({
            "topk_per_query": topk, "sample": index, **row,
            "verdict": verdict}, sort_keys=True) + "\n")
  write_json(out_dir / "smoothness.json", {
      "applicable": True,
      "dispersion": {
          topk: inference[topk].get("dispersion") for topk in inference},
      "role": "component_environment_telemetry_only",
      "required_checks_passed": required,
  })
  (out_dir / "summary.md").write_text(summary(payload), encoding="utf-8")
  print(json.dumps({
      "out_dir": str(out_dir.relative_to(ROOT)),
      "verdict": verdict,
      "required_checks_passed": required,
      "top512_ucb_ms": inference["512"].get("upper_confidence_bound_ms"),
      "top256_ucb_ms": inference["256"].get("upper_confidence_bound_ms"),
      "weighted_ucb_ms": weighted_ucb,
      "weighted_cap_ms": WEIGHTED_CAP_MS,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, sort_keys=True))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
