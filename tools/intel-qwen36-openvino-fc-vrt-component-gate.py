#!/usr/bin/env python3
"""Gate the single Xe3 160-GRF FC candidate against the complete FC cut."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import re
import shlex
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-fc-vrt-component-gate-v0"
BASE_GATE = ROOT / "tools/intel-qwen36-openvino-fc-microkernel-component-gate.py"
BASELINE = ROOT / (
    "output/openvino-fc-micro-component-20260715Tseq1233-"
    "max-native-fused-nonzero-warm512-cleanZ")
COMPUTE_RUNTIME = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "compute-runtime-82aab87fc932edc0558a0302d545a5bcc22edf41")
RELEASE_3004 = COMPUTE_RUNTIME / (
    "shared/source/release_helper/release_helper_3004.cpp")
XE3_CAPABILITY = COMPUTE_RUNTIME / "shared/source/xe3_core/hw_cmds_base.h"
XE3_PRODUCT = COMPUTE_RUNTIME / (
    "shared/source/xe3_core/os_agnostic_product_helper_xe3_core.inl")
XE3_OCCUPANCY = COMPUTE_RUNTIME / (
    "shared/source/helpers/gfx_core_helper_xe3_and_later.inl")
CODEGEN = ROOT / "engine/tools/openvino_moe_micro_codegen.cpp"
HOST_SOURCE = ROOT / "engine/gpu/opencl/openvino_fc_micro_host.cl"
BUILD_DIR = ROOT / "build/engine"
RUNTIME = BUILD_DIR / "iq36-openvino-fc-micro-runtime"
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
TIME = Path("/usr/bin/time")

CONTROL_GRFS = 256
CANDIDATE_GRFS = 160
CONTROL_THREADS_PER_EU = 4
CANDIDATE_THREADS_PER_EU = 6
EU_COUNT = 96
SIMD = 16
PHYSICAL_CARRIER_GBPS = 106.525
TARGET_MS = 8.183
T_ONE_SIDED_95_DF5 = 2.015048373333
ABBA = ("control", "candidate", "candidate", "control")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--warmup", type=int, default=512)
  parser.add_argument("--repeat", type=int, default=31)
  parser.add_argument("--abba-blocks", type=int, default=3)
  parser.add_argument("--timeout-s", type=int, default=600)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if (args.warmup < 1 or args.repeat < 5 or args.abba_blocks != 3 or
      args.timeout_s <= 0 or args.memory_stop_gib <= 0.0):
    parser.error(
        "warmup/repeat/timeout/memory must be positive and ABBA blocks fixed at 3")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/openvino-fc-vrt-component-{stamp}"
  return args


def load_base_gate() -> ModuleType:
  spec = importlib.util.spec_from_file_location("iq36_fc_component_base", BASE_GATE)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {BASE_GATE}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def git_output(*parts: str, cwd: Path = ROOT) -> str:
  result = subprocess.run(
      ["git", *parts], cwd=cwd, text=True, capture_output=True, check=True)
  return result.stdout.strip()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def memory_guard(label: str, stop_bytes: int,
                 samples: list[dict[str, Any]]) -> None:
  available = available_memory_bytes()
  samples.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory guard at {label}: {available} bytes < {stop_bytes} bytes")


def parse_time_report(text: str) -> dict[str, int | None]:
  def value(pattern: str) -> int | None:
    match = re.search(pattern, text, re.MULTILINE)
    return int(match.group(1)) if match else None
  return {
      "maximum_resident_set_kib": value(
          r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$"),
      "swaps": value(r"^\s*Swaps:\s*(\d+)\s*$"),
  }


def run_timed_intel(command: list[str], timeout_s: int,
                    time_path: Path) -> dict[str, Any]:
  timed = [str(TIME), "-v", "-o", str(time_path), *command]
  shell = (
      f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 && "
      "export INTEL_FORCE_PROBE=b080 && export DNNL_VERBOSE=0 && exec " +
      shlex.join(timed))
  try:
    result = subprocess.run(
        ["bash", "-lc", shell], cwd=ROOT, text=True, capture_output=True,
        timeout=timeout_s, check=False, encoding="utf-8", errors="replace")
    time_text = time_path.read_text(encoding="utf-8") if time_path.exists() else ""
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False, "resource_usage": parse_time_report(time_text)}
  except subprocess.TimeoutExpired as error:
    time_text = time_path.read_text(encoding="utf-8") if time_path.exists() else ""
    return {"command": command, "returncode": 124,
            "stdout": error.stdout or "", "stderr": error.stderr or "",
            "timed_out": True, "resource_usage": parse_time_report(time_text)}


def write_run(raw: Path, label: str, row: dict[str, Any]) -> None:
  (raw / f"{label}.json").write_text(
      json.dumps(row, indent=2) + "\n", encoding="utf-8")


def parse_json_line(row: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(row.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def text_has(path: Path, *needles: str) -> bool:
  text = path.read_text(encoding="utf-8")
  return all(needle in text for needle in needles)


def load_baseline() -> dict[str, Any]:
  return json.loads((BASELINE / "metrics.json").read_text(encoding="utf-8"))


def workgroups(row: dict[str, Any]) -> int:
  runtime = row["runtime"]
  return math.prod(
      int(global_size) // int(local_size)
      for global_size, local_size in zip(runtime["global"], runtime["local"]))


def build_preflight(baseline: dict[str, Any]) -> dict[str, Any]:
  nonzero = [row for row in baseline["cohorts"] if int(row["count"]) > 0]
  max_grf_min = max(int(row["package"]["grf_min"]) for row in nonzero)
  supported = [32, 64, 96, 128, 160, 192, 256]
  selected = min(value for value in supported if value >= max_grf_min)
  control_available_threads = CONTROL_THREADS_PER_EU * EU_COUNT
  candidate_available_threads = CANDIDATE_THREADS_PER_EU * EU_COUNT
  occupancy_ratio = candidate_available_threads / control_available_threads
  projected_rows = []
  for row in nonzero:
    threads_per_workgroup = math.prod(row["runtime"]["local"]) // SIMD
    resident_control_workgroups = control_available_threads // threads_per_workgroup
    total_workgroups = workgroups(row)
    occupancy_exposed = total_workgroups > resident_control_workgroups
    baseline_rate = float(row["runtime"]["parameter_gbps"])
    projected_rate = baseline_rate
    if occupancy_exposed:
      projected_rate = min(
          PHYSICAL_CARRIER_GBPS, baseline_rate * occupancy_ratio)
    projected_ms = int(row["cohort_bytes"]) / projected_rate / 1_000_000
    projected_rows.append({
        "name": row["name"], "count": row["count"],
        "cohort_bytes": row["cohort_bytes"],
        "grf_min": row["package"]["grf_min"],
        "threads_per_workgroup": threads_per_workgroup,
        "workgroups": total_workgroups,
        "control_resident_workgroups": resident_control_workgroups,
        "occupancy_exposed": occupancy_exposed,
        "baseline_gbps": baseline_rate,
        "projected_gbps": projected_rate,
        "projected_ms": projected_ms,
    })
  projected_ms = sum(float(row["projected_ms"]) for row in projected_rows)
  source_checks = [
      check("release_3004_supports_exact_vrt_ladder",
            text_has(RELEASE_3004,
                     "return {32u, 64u, 96u, 128u, 160u, 192u, 256u};")),
      check("xe3_vrt_enabled_by_default",
            text_has(XE3_CAPABILITY,
                     "enableVariableRegisterSizeAllocation = true") and
            text_has(XE3_PRODUCT,
                     "propertiesSupport.enableVariableRegisterSizeAllocation")),
      check("xe3_occupancy_maps_160_to_6_and_256_to_4",
            text_has(XE3_OCCUPANCY,
                     "grfCount <= 160u", "maxThreadsPerEuCount = 6",
                     "grfCount <= 256u", "maxThreadsPerEuCount = 4")),
      check("all_complete_fc_packages_need_146_grfs",
            len(nonzero) == 5 and
            all(int(row["package"]["grf_min"]) == 146 for row in nonzero)),
      check("smallest_supported_grf_above_package_min_is_160",
            selected == CANDIDATE_GRFS,
            maximum_package_grf_min=max_grf_min, selected=selected),
      check("complete_projected_schedule_clears_cut",
            projected_ms < TARGET_MS,
            projected_ms=projected_ms, target_ms=TARGET_MS),
  ]
  return {
      "source_checks": source_checks,
      "source_checks_passed": all(row["pass"] for row in source_checks),
      "control_grfs": CONTROL_GRFS,
      "candidate_grfs": CANDIDATE_GRFS,
      "control_threads_per_eu": CONTROL_THREADS_PER_EU,
      "candidate_threads_per_eu": CANDIDATE_THREADS_PER_EU,
      "occupancy_ratio": occupancy_ratio,
      "physical_carrier_gbps": PHYSICAL_CARRIER_GBPS,
      "projected_rows": projected_rows,
      "projected_schedule_ms": projected_ms,
      "target_ms": TARGET_MS,
      "decision": "admit_exactly_one_fixed_160_grf_component_candidate"
          if all(row["pass"] for row in source_checks)
          else "reject_before_compiler_or_gpu",
  }


def input_paths(baseline: dict[str, Any], row: dict[str, Any]) -> dict[str, Path]:
  if row["source"] == "real_layer0_qkv_capture":
    captured = baseline["capture"]["paths"]
    return {
        "input": ROOT / captured["input"],
        "weights": ROOT / captured["weights"],
        "scales": ROOT / captured["scales"],
        "zps": BASELINE / "raw/layer0-qkv-zps-group-major.u4",
        "oracle": ROOT / captured["oracle"],
    }
  shape = BASELINE / "raw" / row["name"]
  return {name: shape / filename for name, filename in {
      "input": "input.f16", "weights": "weights.u4", "scales": "scales.f16",
      "zps": "zps.u4", "oracle": "oracle.f16"}.items()}


def codegen_command(binary: Path, row: dict[str, Any], variant_dir: Path,
                    register_file_size: int) -> list[str]:
  return [
      str(binary), "--decode-fc", "--shape-name", str(row["name"]),
      "--m", str(row["m"]), "--k", str(row["k"]),
      "--register-file-size", str(register_file_size),
      "--dump-dir", str(variant_dir), "--host-source", str(HOST_SOURCE),
  ]


def runtime_command(row: dict[str, Any], inputs: dict[str, Path],
                    variant_dir: Path, actual: Path,
                    warmup: int, repeat: int) -> list[str]:
  settings = row["package"]["settings"]
  name = str(row["name"])
  return [
      str(RUNTIME), "--binary", str(variant_dir / f"{name}.program.bin"),
      "--kernel", f"iq36_moe_micro_{name}",
      "--input", str(inputs["input"]), "--weights", str(inputs["weights"]),
      "--scales", str(inputs["scales"]), "--zps", str(inputs["zps"]),
      "--oracle", str(inputs["oracle"]), "--actual", str(actual),
      "--m", str(row["m"]), "--k", str(row["k"]), "--quant-group", "64",
      "--sg-per-wg-m", str(settings["sg_per_wg_m"]),
      "--sg-per-wg-n", str(settings["sg_per_wg_n"]),
      "--sg-per-wg-k", str(settings["sg_per_wg_k"]),
      "--wg-tile-m", str(settings["wg_tile_m"]),
      "--wg-tile-n", str(settings["wg_tile_n"]),
      "--warmup", str(warmup), "--repeat", str(repeat),
      "--minimum-gbps", "1",
  ]


def one_sided_ucb95(values: list[float]) -> float:
  if len(values) != 6:
    raise ValueError("the fixed three-block ABBA gate must produce six pairs")
  return statistics.mean(values) + (
      T_ONE_SIDED_95_DF5 * statistics.stdev(values) / math.sqrt(len(values)))


def one_sided_lcb95(values: list[float]) -> float:
  if len(values) != 6:
    raise ValueError("the fixed three-block ABBA gate must produce six pairs")
  return statistics.mean(values) - (
      T_ONE_SIDED_95_DF5 * statistics.stdev(values) / math.sqrt(len(values)))


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory_samples: list[dict[str, Any]] = []
  resources: list[dict[str, Any]] = []
  baseline = load_baseline()
  preflight = build_preflight(baseline)
  (out / "preflight.json").write_text(
      json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
  if not preflight["source_checks_passed"]:
    print(json.dumps({"artifact": str(out.relative_to(ROOT)),
                      "verdict": preflight["decision"]}, separators=(",", ":")))
    return 2

  base = load_base_gate()
  commit = git_output("rev-parse", "HEAD")
  dirty = git_output("status", "--porcelain")
  runtime_commit = git_output("rev-parse", "HEAD", cwd=COMPUTE_RUNTIME)
  runtime_dirty = git_output("status", "--porcelain", cwd=COMPUTE_RUNTIME)
  memory_guard("before-configure", stop_bytes, memory_samples)
  configure = run_timed_intel(
      [str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR)],
      args.timeout_s, raw / "configure.time")
  write_run(raw, "configure", configure)
  resources.append(configure["resource_usage"])
  memory_guard("after-configure", stop_bytes, memory_samples)
  build = run_timed_intel(
      [str(CMAKE), "--build", str(BUILD_DIR), "--target",
       "iq36-openvino-fc-micro-runtime", "-j1"],
      args.timeout_s, raw / "runtime-build.time")
  write_run(raw, "runtime-build", build)
  resources.append(build["resource_usage"])
  memory_guard("after-runtime-build", stop_bytes, memory_samples)

  codegen_binary = raw / "openvino-fc-micro-codegen"
  codegen_build = run_timed_intel(
      base.codegen_build_command(codegen_binary), args.timeout_s,
      raw / "codegen-build.time")
  write_run(raw, "codegen-build", codegen_build)
  resources.append(codegen_build["resource_usage"])
  memory_guard("after-codegen-build", stop_bytes, memory_samples)

  cohort_rows: list[dict[str, Any]] = []
  programs: dict[tuple[str, str], Path] = {}
  for baseline_row in baseline["cohorts"]:
    name = str(baseline_row["name"])
    generated: dict[str, Any] = {}
    for variant, grfs in (("control", CONTROL_GRFS),
                          ("candidate", CANDIDATE_GRFS)):
      variant_dir = raw / name / variant / "codegen"
      command = codegen_command(codegen_binary, baseline_row, variant_dir, grfs)
      memory_guard(f"before-{name}-{variant}-codegen", stop_bytes, memory_samples)
      run = run_timed_intel(
          command, args.timeout_s, raw / f"{name}-{variant}-codegen.time")
      write_run(raw, f"{name}-{variant}-codegen", run)
      resources.append(run["resource_usage"])
      memory_guard(f"after-{name}-{variant}-codegen", stop_bytes, memory_samples)
      parsed = parse_json_line(run)
      package = (parsed.get("packages") or [{}])[0]
      generated[variant] = {
          "returncode": run["returncode"], "package": package,
          "top_level_register_file_size": parsed.get("register_file_size"),
      }
      programs[(name, variant)] = variant_dir
    cohort_rows.append({
        "name": name, "m": baseline_row["m"], "k": baseline_row["k"],
        "count": baseline_row["count"], "source": baseline_row["source"],
        "cohort_bytes": baseline_row["cohort_bytes"],
        "baseline_package": baseline_row["package"],
        "generated": generated, "runs": [],
    })

  by_name = {row["name"]: row for row in cohort_rows}
  for baseline_row in baseline["cohorts"]:
    name = str(baseline_row["name"])
    inputs = input_paths(baseline, baseline_row)
    for block in range(args.abba_blocks):
      for position, variant in enumerate(ABBA):
        actual = raw / name / variant / f"actual-b{block}-p{position}.f16"
        command = runtime_command(
            baseline_row, inputs, programs[(name, variant)], actual,
            args.warmup, args.repeat)
        label = f"{name}-b{block}-p{position}-{variant}"
        memory_guard(f"before-{label}", stop_bytes, memory_samples)
        run = run_timed_intel(
            command, args.timeout_s, raw / f"{label}.time")
        write_run(raw, label, run)
        resources.append(run["resource_usage"])
        memory_guard(f"after-{label}", stop_bytes, memory_samples)
        by_name[name]["runs"].append({
            "block": block, "position": position, "variant": variant,
            "returncode": run["returncode"],
            "resource_usage": run["resource_usage"],
            "result": parse_json_line(run),
        })

  schedule_candidate_ms: list[float] = []
  schedule_control_ms: list[float] = []
  for block in range(args.abba_blocks):
    for pair in range(2):
      candidate_position, control_position = ((1, 0) if pair == 0 else (2, 3))
      candidate_ms = 0.0
      control_ms = 0.0
      for row in cohort_rows:
        if int(row["count"]) == 0:
          continue
        indexed = {(run["block"], run["position"]): run
                   for run in row["runs"]}
        candidate_ms += (
            float(indexed[(block, candidate_position)]["result"]["kernel_median_us"])
            * int(row["count"]) / 1000.0)
        control_ms += (
            float(indexed[(block, control_position)]["result"]["kernel_median_us"])
            * int(row["count"]) / 1000.0)
      schedule_candidate_ms.append(candidate_ms)
      schedule_control_ms.append(control_ms)

  savings_ms = [control - candidate for control, candidate in zip(
      schedule_control_ms, schedule_candidate_ms)]
  candidate_ucb_ms = one_sided_ucb95(schedule_candidate_ms)
  saving_lcb_ms = one_sided_lcb95(savings_ms)
  all_runs = [run for row in cohort_rows for run in row["runs"]]
  all_candidate = [run for run in all_runs if run["variant"] == "candidate"]
  all_control = [run for run in all_runs if run["variant"] == "control"]
  all_resources = [row for row in resources if row]
  max_rss_kib = max(
      (int(row["maximum_resident_set_kib"])
       for row in all_resources if row.get("maximum_resident_set_kib") is not None),
      default=0)
  total_swaps = sum(
      int(row["swaps"]) for row in all_resources if row.get("swaps") is not None)
  checks = [
      check("repository_clean_at_gate", not dirty, dirty=dirty),
      check("pinned_compute_runtime_source_clean",
            runtime_commit == "82aab87fc932edc0558a0302d545a5bcc22edf41" and
            not runtime_dirty, commit=runtime_commit, dirty=runtime_dirty),
      check("source_preflight_admitted_fixed_160_only",
            preflight["source_checks_passed"] and
            preflight["candidate_grfs"] == CANDIDATE_GRFS),
      check("configure_and_serial_build_passed",
            configure["returncode"] == 0 and build["returncode"] == 0 and
            codegen_build["returncode"] == 0),
      check("all_twelve_codegen_packages_preserve_fixed_kernel",
            all(
                generated["returncode"] == 0 and
                generated["package"].get("grf_min") == 146 and
                generated["package"].get("settings") == row["baseline_package"].get("settings")
                for row in cohort_rows for generated in row["generated"].values())),
      check("candidate_programs_report_exact_160_grfs",
            all(run["returncode"] == 0 and
                run["result"].get("register_count") == CANDIDATE_GRFS
                for run in all_candidate)),
      check("control_programs_report_exact_256_grfs",
            all(run["returncode"] == 0 and
                run["result"].get("register_count") == CONTROL_GRFS
                for run in all_control)),
      check("all_programs_have_zero_spill",
            all(run["result"].get("spill_memory_bytes") == 0
                for run in all_runs)),
      check("all_real_and_synthetic_component_runs_correct",
            all(run["result"].get("correctness_pass") is True
                for run in all_runs)),
      check("candidate_complete_schedule_ucb_clears_cut",
            candidate_ucb_ms < TARGET_MS,
            candidate_ucb_ms=candidate_ucb_ms, target_ms=TARGET_MS),
      check("memory_guard_never_tripped_and_zero_swap",
            min(row["available_bytes"] for row in memory_samples) >= stop_bytes and
            total_swaps == 0,
            minimum_available_bytes=min(
                row["available_bytes"] for row in memory_samples),
            maximum_child_rss_kib=max_rss_kib, total_swaps=total_swaps),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  admitted = required_checks_passed and candidate_ucb_ms < TARGET_MS
  verdict = ("admit_fixed_160_grf_to_graph_integration"
             if admitted else "reject_fixed_160_grf_component_route")
  metrics = {
      "schema_version": SCHEMA,
      "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WORKSTREAM,
      "git": {"commit": commit, "dirty": bool(dirty), "status": dirty},
      "host": {"target_alias": "intel-ptl-local", "kernel": platform.release()},
      "preflight": preflight,
      "runtime": {"warmup": args.warmup, "repeat": args.repeat,
                  "abba_blocks": args.abba_blocks,
                  "abba_order": list(ABBA)},
      "cohorts": cohort_rows,
      "inference": {
          "candidate_schedule_samples_ms": schedule_candidate_ms,
          "control_schedule_samples_ms": schedule_control_ms,
          "paired_saving_samples_ms": savings_ms,
          "candidate_schedule_mean_ms": statistics.mean(schedule_candidate_ms),
          "candidate_schedule_median_ms": statistics.median(schedule_candidate_ms),
          "candidate_schedule_one_sided_95_ucb_ms": candidate_ucb_ms,
          "control_schedule_mean_ms": statistics.mean(schedule_control_ms),
          "paired_saving_mean_ms": statistics.mean(savings_ms),
          "paired_saving_one_sided_95_lcb_ms": saving_lcb_ms,
          "target_ms": TARGET_MS,
      },
      "memory": {
          "stop_bytes": stop_bytes, "samples": memory_samples,
          "maximum_child_rss_kib": max_rss_kib, "total_swaps": total_swaps,
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "verdict": verdict,
      "claim_boundary": (
          "fixed-shape component admission only; no graph, token, product, or "
          "speedup claim"),
  }
  (out / "metrics.json").write_text(
      json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  rows = []
  for row in cohort_rows:
    candidate = [float(run["result"].get("kernel_median_us", math.inf))
                 for run in row["runs"] if run["variant"] == "candidate"]
    control = [float(run["result"].get("kernel_median_us", math.inf))
               for run in row["runs"] if run["variant"] == "control"]
    rows.append(
        f"| `{row['name']}` | {row['count']} | {statistics.median(control):.3f} | "
        f"{statistics.median(candidate):.3f} |")
  report = f"""# OpenVINO Xe3 variable-register FC component gate

Verdict: **{verdict}**. Required checks: `{str(required_checks_passed).lower()}`.

The source-selected candidate is exactly `160` GRFs: every package needs `146`,
and PTL release 3004 supports `...128, 160, 192, 256`. Xe3 maps 160/256 GRFs
to 6/4 resident threads per EU. No other register size was compiled or run.

| cohort | calls/token | control 256 median us | candidate 160 median us |
|---|---:|---:|---:|
{chr(10).join(rows)}

The six paired ABBA complete-schedule samples give candidate mean/median
`{statistics.mean(schedule_candidate_ms):.6f}/{statistics.median(schedule_candidate_ms):.6f} ms`,
one-sided 95% latency UCB `{candidate_ucb_ms:.6f} ms`, and paired-saving LCB
`{saving_lcb_ms:.6f} ms`. The component cut is `{TARGET_MS:.3f} ms`.

Peak child RSS was `{max_rss_kib:,} KiB`, aggregate process swaps were
`{total_swaps}`, and minimum available memory was
`{min(row['available_bytes'] for row in memory_samples):,} B`. Workers were
serialized. This is a component decision only, not a graph or product speedup.
"""
  (out / "report.md").write_text(report, encoding="utf-8")
  manifest = {
      "schema_version": f"{SCHEMA}-manifest-v0",
      "artifact": str(out.relative_to(ROOT)), "git": metrics["git"],
      "inputs": {"baseline": str(BASELINE.relative_to(ROOT)),
                 "compute_runtime_commit": runtime_commit,
                 "codegen": str(CODEGEN.relative_to(ROOT))},
      "verdict": verdict, "required_checks_passed": required_checks_passed,
  }
  (out / "manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(json.dumps({
      "artifact": str(out.relative_to(ROOT)), "verdict": verdict,
      "candidate_schedule_ucb_ms": candidate_ucb_ms,
      "paired_saving_lcb_ms": saving_lcb_ms,
      "maximum_child_rss_kib": max_rss_kib,
      "required_checks_passed": required_checks_passed,
  }, separators=(",", ":")))
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
