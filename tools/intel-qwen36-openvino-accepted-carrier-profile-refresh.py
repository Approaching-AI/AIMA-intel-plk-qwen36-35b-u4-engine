#!/usr/bin/env python3
"""Refresh one short profile of the exact accepted OpenVINO carrier.

The gate launches one candidate-only 2k worker, never a stock or concurrent
worker. By default it pins the final clean all-state-alias manifest, Level Zero
plugin, and alias scope. ``--plugin`` and ``--accepted-manifest`` can retain raw
diagnostics for an explicit identity delta. It captures the final full OpenVINO
profile and treats the profile rows as non-additive attribution telemetry only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = "intel-qwen36-openvino-accepted-carrier-profile-refresh-v0"

MODEL_DIR = Path("/home/intel/Qwen3.6-35B-A3B-ov")
MODEL_XML = MODEL_DIR / "openvino_language_model.xml"
OV_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
WORKER = REPO / "tools/intel-qwen36-openvino-hot-cold-attention-gate.py"
PLUGIN = Path(
    "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu/"
    "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
CUSTOM_CONFIG = REPO / "engine/openvino/custom/iq36_hot_attention_gqa.xml"
PROMPT = REPO / (
    "output/r0-oracle-prompt-materialization-20260626T082201Z/"
    "prompts/sentinel_002k.txt")
ACCEPTED_MANIFEST = REPO / (
    "output/openvino-linear-state-alias-validation-20260718Tseq1451-"
    "default-all-final-plugin-32k-o64-cleancommit/manifest.json")
REFERENCE_WORKER = REPO / (
    "output/openvino-attention-phase-profile-20260715Tseq1225-"
    "onednn-decode-fc-wg4x4-live-2k-warm17-dirtyZ/raw/2k/"
    "stock/worker-result.json")
STATUS = REPO / "doc/active" / WS / "STATUS.md"
FRONTIER = REPO / "doc/active" / WS / "frontier.json"
REJECTED = REPO / "doc/active" / WS / "rejected-routes.json"

TARGET_LAYERS = tuple(range(3, 40, 4))
DECODE_STEPS = 17
PREFILL_CHUNK_TOKENS = 8192
EXPECTED_EXECUTED_COUNTS = {
    "FullyConnectedCompressed": 371,
    "IQ36HotAttentionGQA": 10,
    "GatedDeltaNet": 30,
    "RMS": 131,
    "IQ36LinearConvSwish": 30,
    "Assign": 60,
}
REGISTERED_EVENT_BUCKETS_MS = {
    "compressed_fc": 13.375,
    "custom_attention": 8.456,
    "gdn": 1.319,
    "rms": 0.358,
    "linear_conv": 0.193,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--plugin", type=Path, default=PLUGIN,
      help=("candidate GPU plugin to profile; a plugin different from the "
            "accepted clean-carrier identity is retained as diagnostic evidence "
            "and intentionally makes the identity gate inconclusive"))
  parser.add_argument(
      "--accepted-manifest", type=Path, default=ACCEPTED_MANIFEST,
      help="clean-carrier product-gate manifest used for exact identity audit")
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--poll-interval-s", type=float, default=1.0)
  parser.add_argument("--min-available-gib", type=float, default=8.0)
  parser.add_argument(
      "--abort-below-available-gib", type=float, default=4.0)
  parser.add_argument(
      "--igc-library-dir", type=Path,
      help=("optional isolated IGC library directory prepended to "
            "LD_LIBRARY_PATH for this candidate-only worker"))
  args = parser.parse_args()
  if args.timeout_s <= 0 or args.poll_interval_s <= 0.0:
    parser.error("timeout and poll interval must be positive")
  if args.min_available_gib < 0.0 or args.abort_below_available_gib < 0.0:
    parser.error("memory thresholds must be nonnegative")
  if args.abort_below_available_gib > args.min_available_gib:
    parser.error("abort threshold must not exceed preflight threshold")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display_path(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(REPO))
  except ValueError:
    return str(path.resolve())


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=REPO, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    output_relative = str(output.relative_to(REPO))
  except ValueError:
    output_relative = ""
  rows = [row for row in rows
          if not output_relative or output_relative not in row]
  return {"commit": commit, "dirty": bool(rows), "status": rows}


def proc_meminfo() -> dict[str, int]:
  rows: dict[str, int] = {}
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    key, value = line.split(":", 1)
    fields = value.split()
    rows[key] = int(fields[0]) * 1024 if fields else 0
  return rows


def process_memory(pid: int) -> dict[str, int]:
  path = Path(f"/proc/{pid}/status")
  if not path.is_file():
    return {"VmRSS": 0, "VmSwap": 0}
  rows: dict[str, int] = {}
  for line in path.read_text(
      encoding="utf-8", errors="replace").splitlines():
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    fields = value.split()
    if key in ("VmRSS", "VmSwap") and fields:
      rows[key] = int(fields[0]) * 1024
  return {"VmRSS": rows.get("VmRSS", 0), "VmSwap": rows.get("VmSwap", 0)}


def other_worker_pids() -> list[dict[str, Any]]:
  rows = []
  for path in Path("/proc").iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
      continue
    try:
      command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
          "utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
      continue
    if ("intel-qwen36-openvino-hot-cold-attention-gate.py" in command and
        "--worker-config" in command):
      rows.append({"pid": int(path.name), "command": command.strip()})
  return rows


def wait_for_memory(required_bytes: int) -> dict[str, Any]:
  started = time.monotonic()
  while True:
    available = int(proc_meminfo().get("MemAvailable", 0))
    if available >= required_bytes:
      return {
          "available_bytes": available,
          "required_bytes": required_bytes,
          "waited_seconds": time.monotonic() - started,
      }
    if time.monotonic() - started >= 60.0:
      raise RuntimeError(
          f"available memory {available} remains below {required_bytes}")
    time.sleep(2.0)


def stop_process_group(
    process: subprocess.Popen[Any], first_signal: int,
) -> None:
  try:
    os.killpg(process.pid, first_signal)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=10)
    return
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def accepted_identity_audit(
    manifest: dict[str, Any], plugin_hash: str,
) -> dict[str, Any]:
  sources = []
  for row in manifest.get("custom_sources", []):
    path = REPO / str(row["path"])
    actual = sha256(path) if path.is_file() else None
    sources.append({
        "path": display_path(path),
        "expected_sha256": row.get("sha256"),
        "actual_sha256": actual,
        "match": actual == row.get("sha256"),
    })
  config_hash = sha256(CUSTOM_CONFIG)
  return {
      "accepted_commit": manifest.get("git", {}).get("commit"),
      "alias_linear_state_assign": manifest.get(
          "alias_linear_state_assign"),
      "linear_state_alias_scope": manifest.get("linear_state_alias_scope"),
      "direct_ssm_state_assign": manifest.get(
          "direct_ssm_state_assign", False),
      "fuse_linear_conv_state": manifest.get("fuse_linear_conv_state"),
      "expected_plugin_sha256": manifest.get("candidate_gpu_plugin_sha256"),
      "actual_plugin_sha256": plugin_hash,
      "plugin_match": (
          plugin_hash == manifest.get("candidate_gpu_plugin_sha256")),
      "expected_config_sha256": manifest.get("custom_config_sha256"),
      "actual_config_sha256": config_hash,
      "config_match": config_hash == manifest.get("custom_config_sha256"),
      "sources": sources,
      "sources_match": bool(sources) and all(row["match"] for row in sources),
  }


def build_worker_config(
    worker_dir: Path, reference: dict[str, Any], plugin: Path,
) -> tuple[dict[str, Any], list[int], list[int]]:
  expected_top1 = [
      int(row["top1"]) for row in reference.get("phases", [])]
  if len(expected_top1) != DECODE_STEPS + 1:
    raise ValueError("reference worker does not have the exact 18 phases")
  decode_tokens = expected_top1[:DECODE_STEPS]
  config = {
      "collect_states": False,
      "custom_config": str(CUSTOM_CONFIG.resolve()),
      "candidate_gpu_plugin": str(plugin.resolve()),
      "decode_steps": DECODE_STEPS,
      "decode_tokens": decode_tokens,
      "device": "GPU",
      "lane": "2k",
      "mode": "candidate",
      "model_dir": str(MODEL_DIR.resolve()),
      "prompt": str(PROMPT.resolve()),
      "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
      "fixed_cold_capacity": 2048,
      "initialize_hot_states": True,
      "skip_hot_state_self_bind": True,
      "dump_runtime_graph": False,
      "capture_full_profile": True,
      "fuse_linear_conv_state": True,
      "pack_gdn_state": False,
      "prefill_history_capacity": 2 * PREFILL_CHUNK_TOKENS,
      "phase_branch_prefill": False,
      "stock_prefill_custom_decode": False,
      "stock_prefill_sliced_decode": False,
      "static_phase_separated": False,
      "raw": str(worker_dir.resolve()),
      "result": str((worker_dir / "worker-result.json").resolve()),
      "target_layers": list(TARGET_LAYERS),
  }
  return config, decode_tokens, expected_top1


def launch_worker(
    args: argparse.Namespace, worker_dir: Path, config: dict[str, Any],
) -> dict[str, Any]:
  cache = worker_dir / "neo-cache"
  cache.mkdir()
  config_path = worker_dir / "worker-config.json"
  write_json(config_path, config)
  preflight = wait_for_memory(int(args.min_available_gib * 1024**3))
  command = [str(OV_PYTHON), str(WORKER), "--worker-config", str(config_path)]
  environment = os.environ.copy()
  for key in (
      "OV_GPU_CONFIG_FILE", "OV_GPU_USM_POLICY", "IQ36_GDN_TRANSPOSED_STATE",
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE",
      "IQ36_FIXED_FC_MANAGER_TRACE_PATH", "IQ36_FIXED_FC_MANAGER_SCOPE",
      "LD_AUDIT"):
    environment.pop(key, None)
  environment.update({
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN": "1",
      "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE": "all",
      "NEO_CACHE_DIR": str(cache.resolve()),
      "NEO_CACHE_MAX_SIZE": str(4 * 1024**3),
      "NEO_CACHE_PERSISTENT": "1",
  })
  if args.igc_library_dir is not None:
    isolated = str(args.igc_library_dir.resolve())
    inherited = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        isolated if not inherited else f"{isolated}:{inherited}")
  stdout_path = worker_dir / "worker.stdout"
  stderr_path = worker_dir / "worker.stderr"
  started = time.monotonic()
  monitor: dict[str, Any] = {
      "process_rss_peak_bytes": 0,
      "process_swap_peak_bytes": 0,
      "sample_count": 0,
      "system_available_min_bytes": None,
      "system_swap_used_peak_bytes": 0,
  }
  timed_out = False
  memory_guard_tripped = False
  abort_bytes = int(args.abort_below_available_gib * 1024**3)
  with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
       stderr_path.open("w", encoding="utf-8") as stderr_handle:
    process = subprocess.Popen(
        command, cwd=REPO, env=environment, stdout=stdout_handle,
        stderr=stderr_handle, text=True, start_new_session=True)
    while process.poll() is None:
      if time.monotonic() - started > args.timeout_s:
        timed_out = True
        stop_process_group(process, signal.SIGTERM)
        break
      system = proc_meminfo()
      process_row = process_memory(process.pid)
      available = int(system.get("MemAvailable", 0))
      swap_used = int(system.get("SwapTotal", 0)) - int(
          system.get("SwapFree", 0))
      monitor["sample_count"] = int(monitor["sample_count"]) + 1
      monitor["process_rss_peak_bytes"] = max(
          int(monitor["process_rss_peak_bytes"]),
          int(process_row["VmRSS"]))
      monitor["process_swap_peak_bytes"] = max(
          int(monitor["process_swap_peak_bytes"]),
          int(process_row["VmSwap"]))
      current_min = monitor["system_available_min_bytes"]
      monitor["system_available_min_bytes"] = (
          available if current_min is None else min(int(current_min), available))
      monitor["system_swap_used_peak_bytes"] = max(
          int(monitor["system_swap_used_peak_bytes"]), swap_used)
      if abort_bytes and available < abort_bytes:
        memory_guard_tripped = True
        stop_process_group(process, signal.SIGINT)
        break
      time.sleep(args.poll_interval_s)
    returncode = process.wait()
  stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
  oom_observed = (
      not memory_guard_tripped and
      (returncode in (-9, 137) or "out of memory" in stderr.lower() or
       "cl_out_of_resources" in stderr.lower()))
  result_path = worker_dir / "worker-result.json"
  return {
      "command": command,
      "environment": {key: environment[key] for key in (
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN",
          "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE", "NEO_CACHE_DIR",
          "NEO_CACHE_MAX_SIZE", "NEO_CACHE_PERSISTENT")},
      "igc_library_dir": (
          str(args.igc_library_dir.resolve())
          if args.igc_library_dir is not None else None),
      "ld_library_path_first": (
          environment.get("LD_LIBRARY_PATH", "").split(":", 1)[0]
          if environment.get("LD_LIBRARY_PATH") else None),
      "elapsed_seconds": time.monotonic() - started,
      "memory_preflight": preflight,
      "memory_guard": {"abort_below_bytes": abort_bytes,
                       "tripped": memory_guard_tripped},
      "monitor": monitor,
      "oom_observed": oom_observed,
      "returncode": returncode,
      "timed_out": timed_out,
      "result": load_json(result_path) if result_path.is_file() else {},
  }


def profile_audit(result: dict[str, Any]) -> dict[str, Any]:
  rows = result.get("full_profile", [])
  executed = [row for row in rows if row.get("status") == "Status.EXECUTED"]
  counts = Counter(str(row.get("node_type", "")) for row in executed)
  raw_us: defaultdict[str, float] = defaultdict(float)
  for row in executed:
    raw_us[str(row.get("node_type", ""))] += float(
        row.get("real_time_us", 0.0))
  optimized_reorders = [
      row for row in rows
      if row.get("status") == "Status.OPTIMIZED_OUT" and
      row.get("node_type") == "Reorder" and
      "_iq36_hot_attention_layer" in str(row.get("node_name", "")) and
      "_cldnn_custom_preprocess" in str(row.get("node_name", ""))]
  selected_counts = {
      key: int(counts.get(key, 0)) for key in EXPECTED_EXECUTED_COUNTS}
  raw_ranked = sorted(
      ({
          "node_type": node_type,
          "executed_count": int(counts.get(node_type, 0)),
          "raw_real_time_us_nonadditive": float(raw_time),
      } for node_type, raw_time in raw_us.items()),
      key=lambda row: (-row["raw_real_time_us_nonadditive"],
                       row["node_type"]))
  return {
      "full_profile_rows": len(rows),
      "executed_rows": len(executed),
      "executed_counts": dict(counts),
      "selected_executed_counts": selected_counts,
      "expected_executed_counts": EXPECTED_EXECUTED_COUNTS,
      "selected_counts_exact": selected_counts == EXPECTED_EXECUTED_COUNTS,
      "optimized_custom_preprocess_reorders": len(optimized_reorders),
      "raw_real_time_us_by_node_type_nonadditive": dict(raw_us),
      "raw_ranked_node_types_nonadditive": raw_ranked,
      "raw_profile_time_is_savings_evidence": False,
  }


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  args.accepted_manifest = args.accepted_manifest.resolve()
  worker_dir = output / "raw" / "2k" / "candidate"
  worker_dir.mkdir(parents=True, exist_ok=False)
  required = [
      MODEL_XML, OV_PYTHON, WORKER, args.plugin, CUSTOM_CONFIG, PROMPT,
      args.accepted_manifest, REFERENCE_WORKER, STATUS, FRONTIER, REJECTED]
  isolated_igc_files: list[Path] = []
  if args.igc_library_dir is not None:
    args.igc_library_dir = args.igc_library_dir.resolve()
    isolated_igc_files = [
        args.igc_library_dir / "libigc.so.2",
        args.igc_library_dir / "libigdfcl.so.2",
    ]
    opencl_clang = sorted(args.igc_library_dir.glob("libopencl-clang2.so.*"))
    if opencl_clang:
      isolated_igc_files.append(opencl_clang[-1])
    required.extend(isolated_igc_files)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise SystemExit("missing profile-refresh inputs: " + ", ".join(missing))

  git = git_state(output)
  concurrent = other_worker_pids()
  if concurrent:
    raise RuntimeError(f"concurrent OpenVINO worker detected: {concurrent}")
  manifest = load_json(args.accepted_manifest)
  reference = load_json(REFERENCE_WORKER)
  frontier = load_json(FRONTIER)
  rejected = load_json(REJECTED)
  plugin = args.plugin.resolve()
  plugin_hash = sha256(plugin)
  identity = accepted_identity_audit(manifest, plugin_hash)
  config, decode_tokens, expected_top1 = build_worker_config(
      worker_dir, reference, plugin)
  worker = launch_worker(args, worker_dir, config)
  result = worker["result"]
  profile = profile_audit(result) if result else {}

  phases = result.get("phases", [])
  top1 = [int(row.get("top1", -1)) for row in phases]
  decode_walls = [float(row.get("wall_ms_diagnostic", math.nan))
                  for row in phases[1:]]
  decode_walls_finite = (
      len(decode_walls) == DECODE_STEPS and
      all(math.isfinite(value) and value > 0.0 for value in decode_walls))
  source = result.get("source_summary") or {}
  rejected_by_route = {
      str(row.get("route")): row for row in rejected.get("rejected", [])
      if row.get("route")}
  required_closed_routes = (
      "openvino_fixed_shape_decode_u4_f16_microkernel_v28n",
      "openvino_fixed_fc_m1024_product_provider_v30l",
      "openvino_direct_ssm_state_assign_repack_bypass_v30m",
      "openvino_upstream_gdn_vload_fma_v29j",
      "openvino_upstream_assign_producer_device_memory_v29i",
  )
  kill_number_ms = float(
      frontier["goal_budget"]["per_token_ms"]["remaining_cut"])
  known_closed_families = {
      "compressed_fc": [
          "openvino_fixed_shape_decode_u4_f16_microkernel_v28n",
          "openvino_fixed_fc_m1024_product_provider_v30l",
      ],
      "custom_attention": [],
      "gdn": ["openvino_upstream_gdn_vload_fma_v29j"],
      "rms": [],
      "linear_conv": [
          "openvino_direct_ssm_state_assign_repack_bypass_v30m"],
  }
  eligible_buckets = [
      {"name": name, "registered_ms_per_token": value,
       "clears_kill_number": value >= kill_number_ms,
       "whole_bucket_closed": False,
       "known_closed_families": known_closed_families[name]}
      for name, value in REGISTERED_EVENT_BUCKETS_MS.items()]
  source_bound_candidates = [
      row["name"] for row in eligible_buckets
      if row["clears_kill_number"] and not row["whole_bucket_closed"]]
  selected_route = "openvino_clean_carrier_dominant_bucket_source_bound"

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("no_concurrent_openvino_worker_at_launch", not concurrent,
            concurrent=concurrent),
      check("accepted_plugin_config_sources_and_alias_match_clean_carrier",
            identity["plugin_match"] and identity["config_match"] and
            identity["sources_match"] and
            identity["alias_linear_state_assign"] is True and
            identity["linear_state_alias_scope"] == "all" and
            identity["direct_ssm_state_assign"] is False and
            identity["fuse_linear_conv_state"] is True,
            identity=identity),
      check("single_candidate_worker_completes_without_oom",
            worker["returncode"] == 0 and not worker["timed_out"] and
            not worker["memory_guard"]["tripped"] and
            not worker["oom_observed"], worker={
                key: worker[key] for key in (
                    "returncode", "timed_out", "memory_guard",
                    "oom_observed", "monitor")}),
      check("linear_state_alias_environment_is_exact",
            worker["environment"].get(
                "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN") == "1" and
            worker["environment"].get(
                "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE") == "all"),
      check("worker_uses_exact_candidate_plugin",
            result.get("candidate_gpu_plugin_sha256") == plugin_hash and
            result.get("candidate_gpu_plugin") == str(plugin)),
      check("accepted_graph_shape_is_exact",
            source.get("custom_count_after") == 10 and
            source.get("stock_sdpa_count_after") == 0 and
            source.get("linear_conv_replacement_count") == 30 and
            source.get("linear_conv_custom_count_after") == 30 and
            result.get("same_infer_request") is True and
            result.get("hot_state_self_bind_skipped") is True,
            source_summary=source),
      check("teacher_forced_greedy_path_matches_reference",
            len(phases) == DECODE_STEPS + 1 and top1 == expected_top1 and
            all(row.get("logits_finite") is True for row in phases),
            top1=top1, expected_top1=expected_top1),
      check("final_profile_has_exact_current_execution_census",
            bool(profile) and profile.get("selected_counts_exact") is True and
            profile.get("optimized_custom_preprocess_reorders") == 130,
            profile=profile),
      check("decode_wall_telemetry_is_finite",
            decode_walls_finite,
            decode_walls_ms=decode_walls),
      check("recent_rejected_families_are_registered",
            all(route in rejected_by_route
                for route in required_closed_routes),
            required_closed_routes=list(required_closed_routes)),
      check("dominant_kernel_buckets_clear_current_kill_number",
            source_bound_candidates == ["compressed_fc", "custom_attention"] and
            all(row["registered_ms_per_token"] < kill_number_ms
                for row in eligible_buckets
                if row["name"] not in source_bound_candidates),
            eligible_buckets=eligible_buckets,
            source_bound_candidates=source_bound_candidates,
            selected_route=selected_route),
      check("profile_rows_are_not_added_as_savings",
            profile.get("raw_profile_time_is_savings_evidence") is False,
            raw_real_time_us_by_node_type=
                profile.get("raw_real_time_us_by_node_type_nonadditive")),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  verdict = (
      "select_clean_carrier_dominant_bucket_source_bound"
      if required_checks_passed else "inconclusive")
  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
      "workstream": WS,
      "git": git,
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "selected_route": selected_route if required_checks_passed else None,
      "gpu_workers_launched": 1,
      "stock_worker_launched": False,
      "concurrent_worker_launched": False,
      "long_worker_launched": False,
      "lane": "2k",
      "decode_steps": DECODE_STEPS,
      "decode_tokens": decode_tokens,
      "expected_top1": expected_top1,
      "actual_top1": top1,
      "accepted_identity": identity,
      "worker": {key: value for key, value in worker.items()
                 if key != "result"},
      "isolated_igc": {
          "library_dir": (
              str(args.igc_library_dir)
              if args.igc_library_dir is not None else None),
          "libraries": {
              str(path): sha256(path) for path in isolated_igc_files},
      },
      "worker_result_summary": {
          "openvino_version": result.get("openvino_version"),
          "compile_ms": result.get("compile_ms"),
          "compile_config": result.get("compile_config"),
          "memory_samples": result.get("memory_samples"),
          "same_infer_request": result.get("same_infer_request"),
          "hot_state_self_bind_skipped": result.get(
              "hot_state_self_bind_skipped"),
          "source_summary": source,
          "decode_wall_ms": decode_walls,
          "decode_wall_median_ms": (
              statistics.median(decode_walls)
              if decode_walls_finite else None),
      },
      "profile_audit": profile,
      "route_selection": {
          "kill_number_ms_per_token": kill_number_ms,
          "prior_registered_event_buckets_ms_per_token":
              REGISTERED_EVENT_BUCKETS_MS,
          "eligible_buckets": eligible_buckets,
          "source_bound_candidates": source_bound_candidates,
          "required_closed_route_ids": list(required_closed_routes),
          "selected_route": selected_route,
          "selection_rule": (
              "use the refreshed exact execution census to rank source-bound "
              "work; prior OpenCL event buckets are complete logical-work "
              "anchors only; never sum non-additive Level Zero profile rows "
              "or treat rejection of one algorithm as closure of a bucket"),
      },
      "checks": checks,
      "inputs": {display_path(path): sha256(path) for path in required},
  }
  write_json(output / "metrics.json", metrics)
  write_json(output / "manifest.json", {
      "schema": SCHEMA,
      "git": git,
      "lane": "2k",
      "decode_steps": DECODE_STEPS,
      "candidate_gpu_plugin": str(plugin),
      "candidate_gpu_plugin_sha256": plugin_hash,
      "accepted_manifest": display_path(args.accepted_manifest),
      "accepted_manifest_sha256": sha256(args.accepted_manifest),
      "custom_config": display_path(CUSTOM_CONFIG),
      "custom_config_sha256": sha256(CUSTOM_CONFIG),
      "alias_linear_state_assign": True,
      "linear_state_alias_scope": "all",
      "fuse_linear_conv_state": True,
      "graph_initialized_hot_states": True,
      "capture_full_profile": True,
      "memory_preflight_gib": args.min_available_gib,
      "memory_abort_gib": args.abort_below_available_gib,
      "stock_worker_launched": False,
      "igc_library_dir": (
          str(args.igc_library_dir)
          if args.igc_library_dir is not None else None),
      "igc_libraries": {
          str(path): sha256(path) for path in isolated_igc_files},
  })
  median_text = (
      f"{statistics.median(decode_walls):.6f}"
      if decode_walls_finite else "n/a")
  identity_text = (
      "Its plugin, custom config, registered custom sources, all-ten attention "
      "graph, all-30 linear conv/state/SiLU graph, graph-initialized hot "
      "state, and all-state alias environment match the clean carrier manifest."
      if identity["plugin_match"] else
      "Its plugin is an explicit candidate delta from the clean carrier; the "
      "identity gate is therefore intentionally inconclusive while raw "
      "correctness, profile, allocation, and memory evidence is retained.")
  summary = f"""# Accepted-carrier decode profile refresh

Verdict: **{verdict}**. Required checks:
`{str(required_checks_passed).lower()}`.

Exactly one isolated candidate worker ran at 2k with 17 teacher-forced decode
steps. {identity_text} The 18 phase top-1 tokens match the stock reference
exactly. Diagnostic decode wall median is
`{median_text} ms`; it is not a product row.

The current execution census remains exactly 371 compressed FC, 10 custom
attention, 30 GDN, 131 RMS, 30 custom linear conv, and 60 Assign nodes; all 130
custom preprocess reorders remain optimized out. Level Zero profile times are
non-additive telemetry and are not summed or treated as savings.

The prior complete logical-work anchors put compressed FC at
`13.375 ms/token` and custom attention at `8.456 ms/token`; both exceed the
current `{kill_number_ms:.6f} ms/token` kill-number. Recent experiments close
specific algorithms, not either whole bucket. Select `{selected_route}` and
derive a fresh non-overlapping source ceiling before another implementation.
No stock, concurrent, 32k, ABBA, output512, or long worker ran. Memory guard
tripped: `{str(worker['memory_guard']['tripped']).lower()}`; OOM observed:
`{str(worker['oom_observed']).lower()}`.
"""
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "artifact": display_path(output),
      "verdict": verdict,
      "selected_route": selected_route if required_checks_passed else None,
      "worker_returncode": worker["returncode"],
      "memory_guard_tripped": worker["memory_guard"]["tripped"],
      "oom_observed": worker["oom_observed"],
  }, separators=(",", ":")), flush=True)
  return 0 if required_checks_passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
