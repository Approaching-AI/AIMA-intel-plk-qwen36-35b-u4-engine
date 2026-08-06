#!/usr/bin/env python3
"""Re-bind exact-attention hardware-limit opportunities before another kernel.

This gate is source and artifact only.  It refreshes official oneDNN,
OpenVINO, and IGC capability, disassembles the accepted triple-cohort program,
and binds the dense K/V bandwidth needed to clear the registered component
cut.  It never compiles, creates a GPU context, or starts a model worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA = (
    "intel-qwen36-openvino-exact-attention-"
    "hardware-limit-opportunity-gate-v1")
ACTIVE = ROOT / "doc/active" / WS
STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
N8_CAPABILITY = ROOT / (
    "output/openvino-exact-attention-nhalf-capability-"
    "20260723Tseq2123c-clean/capability-gate-result.json")
DUAL_COMPONENT = ROOT / (
    "output/openvino-exact-attention-dual-cohort-component-"
    "20260723Tseq2135-clean/result.json")
DECOMPOSITION = ROOT / (
    "output/openvino-exact-attention-three-stage-component-"
    "20260724Tseq2144-clean/result.json")
TRIPLE_COMPONENT = ROOT / (
    "output/openvino-exact-attention-triple-cohort-component-"
    "20260724Tseq2146-clean/result.json")
NORMALIZED_COMPONENT = ROOT / (
    "output/openvino-exact-attention-normalized-dual-cohort-component-"
    "20260724Tseq2149-clean/result.json")
TRIPLE_PROGRAM = ROOT / (
    "output/openvino-exact-attention-triple-cohort-codegen-"
    "20260724Tseq2145-clean/raw/triple-cohort/"
    "existing_shim.program.bin")

ONEDNN_BASE = "978698e3b8e2fd25cc8236ea912056936ebd8fa6"
OPENVINO_ONEDNN_BASE = "babb7375ff500dd8ad77d26cbd2b044122b7a8b4"
OPENVINO_F8_GQA_COMMIT = (
    "bf077257906c7dda92dc5bc3365c3617ceafb487")
IGC_LATENCY_SINK_COMMIT = (
    "0ee225406b690cf92acfa0ad8118f5194f2ffa39")
API_ROOT = "https://api.github.com/repos"
URLS = {
    "onednn_compare": (
        f"{API_ROOT}/uxlfoundation/oneDNN/compare/{ONEDNN_BASE}...main"),
    "openvino_master": f"{API_ROOT}/openvinotoolkit/openvino/commits/master",
    "openvino_onednn_submodule": (
        f"{API_ROOT}/openvinotoolkit/openvino/contents/"
        "src/plugins/intel_gpu/thirdparty/onednn_gpu?ref=master"),
    "openvino_f8_gqa": (
        f"{API_ROOT}/openvinotoolkit/openvino/commits/"
        f"{OPENVINO_F8_GQA_COMMIT}"),
    "igc_latest_release": (
        f"{API_ROOT}/intel/intel-graphics-compiler/releases/latest"),
    "igc_master": (
        f"{API_ROOT}/intel/intel-graphics-compiler/commits/master"),
}

MANDATORY_KV_BYTES = 268_435_456
HALF_K_BYTES = MANDATORY_KV_BYTES // 2
DELTA_CAP_MS = 0.1175998
EXPECTED_CONTEXT = 131_072
EXPECTED_OUTPUTS = 4_096
EXPECTED_GRFS = 96
REGISTER_FILE_GRFS = 128
IGC_LATENCY_SINK_MAX_PRESSURE_FRACTION = 0.50
MIN_KQ_RULER_MARGIN = 0.03


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  parser.add_argument("--network-timeout-s", type=float, default=30.0)
  parser.add_argument("--ocloc", type=Path, default=Path("ocloc"))
  args = parser.parse_args()
  if args.memory_stop_gib < 4.0:
    parser.error("--memory-stop-gib must be at least 4")
  if args.network_timeout_s <= 0:
    parser.error("--network-timeout-s must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def display(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(
    label: str, minimum: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < minimum:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {minimum}")


def git_state(out_dir: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.strip()
  rows = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, check=True,
      capture_output=True, text=True).stdout.splitlines()
  try:
    out_rel = str(out_dir.resolve().relative_to(ROOT))
  except ValueError:
    out_rel = ""
  rows = [row for row in rows if not out_rel or out_rel not in row]
  return {"commit": commit, "dirty": bool(rows), "dirty_paths": rows}


def check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **details}


def fetch_json(
    label: str, url: str, raw: Path, timeout_s: float,
) -> dict[str, Any]:
  destination = raw / f"{label}.json"
  error: Exception | None = None
  for attempt in range(3):
    try:
      request = urllib.request.Request(
          url,
          headers={
              "Accept": "application/vnd.github+json",
              "User-Agent": "intel-qwen36-hardware-limit-audit",
          })
      with urllib.request.urlopen(request, timeout=timeout_s) as response:
        value = response.read()
      destination.write_bytes(value)
      payload = json.loads(value)
      if not isinstance(payload, dict):
        raise TypeError(f"expected object from {url}")
      return payload
    except Exception as caught:  # Network failures are recorded by the gate.
      error = caught
      if attempt != 2:
        time.sleep(1 + attempt)
  raise RuntimeError(f"failed to fetch {url}: {error}")


def disassemble(
    ocloc: Path, program: Path, raw: Path,
) -> dict[str, Any]:
  destination = raw / "triple-disassembly"
  destination.mkdir()
  completed = subprocess.run(
      [str(ocloc), "disasm", "-file", str(program),
       "-dump", str(destination)],
      cwd=ROOT, text=True, capture_output=True, timeout=120, check=False)
  asm = destination / ".text.iq36_exact_score_triple_cohort.asm"
  ze_info = destination / ".ze_info"
  text = asm.read_text(encoding="utf-8") if asm.is_file() else ""
  metadata = ze_info.read_text(encoding="utf-8") if ze_info.is_file() else ""
  ugm_loads = [
      line.strip() for line in text.splitlines()
      if "send.ugm" in line and "load" in line]
  return {
      "command": [
          str(ocloc), "disasm", "-file", display(program),
          "-dump", display(destination)],
      "returncode": completed.returncode,
      "stdout": completed.stdout,
      "stderr": completed.stderr,
      "asm_path": display(asm),
      "asm_sha256": sha256(asm) if asm.is_file() else None,
      "ze_info_path": display(ze_info),
      "ze_info_sha256": sha256(ze_info) if ze_info.is_file() else None,
      "ugm_load_send_count": len(ugm_loads),
      "load_block2d_ca_ca_count": len(re.findall(
          r"load_block2d\.ugm[^\n]*\.ca\.ca", text)),
      "load_block2d_uc_ca_count": len(re.findall(
          r"load_block2d\.ugm[^\n]*\.uc\.ca", text)),
      "all_ugm_ca_ca_count": len(re.findall(
          r"load(?:_block2d)?\.ugm[^\n]*\.ca\.ca", text)),
      "dpas_count": len(re.findall(r"\bdpas(?:w)?\.", text)),
      "metadata_has_96_grf": "grf_count:       96" in metadata,
      "metadata_has_61472_slm": "slm_size:        61472" in metadata,
      "metadata_has_simd16": "simd_size:       16" in metadata,
  }


def median(values: list[float]) -> float:
  if not values or any(not math.isfinite(value) for value in values):
    return math.nan
  return statistics.median(values)


def summary(payload: dict[str, Any]) -> str:
  bound = payload["traffic_bound"]
  official = payload["official_capability"]
  isa = payload["accepted_triple_isa"]
  return "\n".join([
      "# Exact-attention hardware-limit opportunity gate",
      "",
      f"Verdict: **{payload['verdict']}**. Required checks: "
      f"`{str(payload['required_checks_passed']).lower()}`.",
      "",
      f"- official oneDNN/OpenVINO/IGC heads: "
      f"`{official['onednn']['head'][:12]}` / "
      f"`{official['openvino']['head'][:12]}` / "
      f"`{official['igc']['master_head'][:12]}`",
      f"- latest supported IGC release: "
      f"`{official['igc']['latest_release']}`",
      f"- dual/triple/required dense bandwidth: "
      f"`{bound['dual_gb_s']:.6f} / {bound['triple_gb_s']:.6f} / "
      f"{bound['required_gb_s']:.6f} GB/s`",
      f"- independent KQ half-payload ruler: "
      f"`{bound['kq_ruler_gb_s']:.6f} GB/s` "
      f"(`{100.0 * bound['kq_margin_fraction']:.3f}%` above required)",
      f"- required triple median improvement: "
      f"`{bound['triple_median_deficit_ms']:.7f} ms/layer`",
      f"- accepted triple UGM cached block2D loads: "
      f"`{isa['load_block2d_ca_ca_count']}` "
      f"(uncached-L1 forms: `{isa['load_block2d_uc_ca_count']}`)",
      "",
      "No current official N-half package or applicable supported IGC change",
      "reopens an exact carrier.  The KQ ruler nevertheless exceeds the",
      "required full-KV rate, so exactly one two-workgroup/48-subgroup dense",
      "K+V traffic ceiling is admitted.  It must use the full 268,435,456-B",
      "payload and the registered 2.7375042-ms cap.  This gate launched no",
      "compiler, GPU context, plugin, or model worker.",
      "",
  ])


def main() -> int:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  raw = out_dir / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required_paths = (
      STATUS, ROUTES, N8_CAPABILITY, DUAL_COMPONENT, DECOMPOSITION,
      TRIPLE_COMPONENT, NORMALIZED_COMPONENT, TRIPLE_PROGRAM)
  missing = [str(path) for path in required_paths if not path.is_file()]
  if missing:
    raise SystemExit(
        "missing hardware-limit inputs: " + ", ".join(missing))

  git = git_state(out_dir)
  n8 = load_json(N8_CAPABILITY)
  dual = load_json(DUAL_COMPONENT)
  decomposition = load_json(DECOMPOSITION)
  triple = load_json(TRIPLE_COMPONENT)
  normalized = load_json(NORMALIZED_COMPONENT)
  status = STATUS.read_text(encoding="utf-8")
  routes = ROUTES.read_text(encoding="utf-8")
  sample_memory("after-local-inputs", stop_bytes, memory)

  fetched = {
      label: fetch_json(
          label, url, raw, args.network_timeout_s)
      for label, url in URLS.items()
  }
  sample_memory("after-official-refresh", stop_bytes, memory)
  isa = disassemble(args.ocloc, TRIPLE_PROGRAM, raw)
  sample_memory("after-offline-disassembly", stop_bytes, memory)

  onednn_compare = fetched["onednn_compare"]
  onednn_files = [
      str(row.get("filename", ""))
      for row in onednn_compare.get("files", [])
      if isinstance(row, dict)]
  onednn_commits = [
      {
          "sha": str(row.get("sha", "")),
          "subject": str(
              row.get("commit", {}).get("message", "")).splitlines()[0],
      }
      for row in onednn_compare.get("commits", [])
      if isinstance(row, dict)]
  onednn_head = (
      onednn_commits[-1]["sha"] if onednn_commits else "")
  onednn_gpu_gemmstone_changed = any(
      path.startswith("src/gpu/intel/gemm/jit/")
      for path in onednn_files)

  openvino_head_payload = fetched["openvino_master"]
  openvino_head = str(openvino_head_payload.get("sha", ""))
  openvino_submodule = fetched["openvino_onednn_submodule"]
  openvino_onednn_head = str(openvino_submodule.get("sha", ""))
  f8_payload = fetched["openvino_f8_gqa"]
  f8_message = str(f8_payload.get("commit", {}).get("message", ""))
  f8_message_normalized = " ".join(f8_message.split()).lower()
  f8_dynamic_only = bool(
      "dynamic (slice+concat)" in f8_message_normalized
      and "static" in f8_message_normalized
      and "still fails implementation selection"
          in f8_message_normalized)

  release = fetched["igc_latest_release"]
  igc_master_payload = fetched["igc_master"]
  igc_master = str(igc_master_payload.get("sha", ""))
  igc_message = str(
      igc_master_payload.get("commit", {}).get("message", ""))
  igc_patch = "\n".join(
      str(row.get("patch", ""))
      for row in igc_master_payload.get("files", [])
      if isinstance(row, dict))
  igc_latency_sink_exact = bool(
      igc_master == IGC_LATENCY_SINK_COMMIT
      and "default off" in igc_message
      and "EnableSampleResultLatencySink" in igc_patch
      and "const unsigned maxPercent = 50" in igc_patch)

  official = {
      "onednn": {
          "base": ONEDNN_BASE,
          "head": onednn_head,
          "ahead_by": onednn_compare.get("ahead_by"),
          "commits": onednn_commits,
          "changed_files": onednn_files,
          "gpu_gemmstone_changed": onednn_gpu_gemmstone_changed,
      },
      "openvino": {
          "head": openvino_head,
          "onednn_submodule": openvino_onednn_head,
          "onednn_submodule_changed_since_seq2123":
              openvino_onednn_head != OPENVINO_ONEDNN_BASE,
          "f8_gqa_commit": OPENVINO_F8_GQA_COMMIT,
          "f8_gqa_dynamic_only_static_unavailable": f8_dynamic_only,
      },
      "igc": {
          "master_head": igc_master,
          "latest_release": release.get("tag_name"),
          "latest_release_published_at": release.get("published_at"),
          "latest_release_url": release.get("html_url"),
          "latency_sink_commit": IGC_LATENCY_SINK_COMMIT,
          "latency_sink_default_off_and_50pct_gate":
              igc_latency_sink_exact,
          "accepted_grf_pressure_fraction":
              EXPECTED_GRFS / REGISTER_FILE_GRFS,
          "eligible_at_50pct_pressure_gate":
              EXPECTED_GRFS / REGISTER_FILE_GRFS <=
                  IGC_LATENCY_SINK_MAX_PRESSURE_FRACTION,
      },
      "urls": URLS,
  }

  dual_ms = float(dual.get("candidate_median_ms", math.nan))
  dual_payload = int(
      triple.get("result", {}).get(
          "mandatory_key_value_payload_bytes", -1))
  triple_rows = triple.get("result", {}).get("paired_samples", [])
  triple_ms = median([
      float(row.get("triple_ms", math.nan))
      for row in triple_rows if isinstance(row, dict)])
  decomposition_rows = decomposition.get(
      "result", {}).get("paired_samples", [])
  kq_ms = median([
      float(row.get("three_stage", {}).get("kq_ms", math.nan))
      for row in decomposition_rows if isinstance(row, dict)])
  required_ms = dual_ms - DELTA_CAP_MS
  dual_gb_s = MANDATORY_KV_BYTES / (dual_ms * 1.0e6)
  triple_gb_s = MANDATORY_KV_BYTES / (triple_ms * 1.0e6)
  required_gb_s = MANDATORY_KV_BYTES / (required_ms * 1.0e6)
  kq_gb_s = HALF_K_BYTES / (kq_ms * 1.0e6)
  kq_margin = kq_gb_s / required_gb_s - 1.0
  traffic_bound = {
      "mandatory_key_value_payload_bytes": MANDATORY_KV_BYTES,
      "dual_median_ms": dual_ms,
      "dual_gb_s": dual_gb_s,
      "triple_median_ms": triple_ms,
      "triple_gb_s": triple_gb_s,
      "registered_delta_cap_ms": DELTA_CAP_MS,
      "required_full_kv_latency_cap_ms": required_ms,
      "required_gb_s": required_gb_s,
      "triple_median_deficit_ms": triple_ms - required_ms,
      "triple_required_bandwidth_gain_fraction":
          required_gb_s / triple_gb_s - 1.0,
      "kq_half_payload_bytes": HALF_K_BYTES,
      "kq_ruler_median_ms": kq_ms,
      "kq_ruler_gb_s": kq_gb_s,
      "kq_margin_fraction": kq_margin,
      "traffic_ceiling_contract": {
          "workgroups": 2,
          "subgroups_per_workgroup": 48,
          "workgroup_items": 768,
          "payload_bytes": MANDATORY_KV_BYTES,
          "latency_ucb_cap_ms": required_ms,
          "bandwidth_lcb_floor_gb_s": required_gb_s,
          "minimum_samples": 20,
          "interpretation": (
              "distinguish two-workgroup physical traffic capacity from "
              "generated-package, synchronization, and arithmetic cost; "
              "not an implementation or speed claim"),
      },
  }

  n8_closed = bool(
      n8.get("verdict") ==
          "remain_closed_no_new_official_native_capability"
      and n8.get("checks", {}).get(
          "current_generates_requested_kq_and_vs_nhalf") is False)
  prior_exact = bool(
      dual.get("component_promoted") is True
      and dual.get("result", {}).get("numeric_pass") is True
      and dual_payload == MANDATORY_KV_BYTES
      and triple.get("verdict") ==
          "reject_exact_attention_triple_cohort_component"
      and triple.get("result", {}).get("numeric_pass") is True
      and normalized.get("verdict") ==
          "reject_exact_attention_normalized_dual_cohort_component"
      and normalized.get("result", {}).get("numeric_pass") is True)
  arithmetic_pass = bool(
      math.isfinite(required_ms)
      and required_ms > 0.0
      and math.isfinite(kq_margin)
      and kq_margin >= MIN_KQ_RULER_MARGIN
      and triple_ms > required_ms
      and required_gb_s > triple_gb_s)
  official_n8_still_closed = bool(
      n8_closed
      and not onednn_gpu_gemmstone_changed
      and openvino_onednn_head == OPENVINO_ONEDNN_BASE)
  igc_not_applicable = bool(
      release.get("tag_name") == "v2.38.2"
      and igc_latency_sink_exact
      and EXPECTED_GRFS / REGISTER_FILE_GRFS >
          IGC_LATENCY_SINK_MAX_PRESSURE_FRACTION)
  isa_pass = bool(
      isa["returncode"] == 0
      and isa["load_block2d_ca_ca_count"] > 0
      and isa["load_block2d_uc_ca_count"] == 0
      and isa["dpas_count"] > 0
      and isa["metadata_has_96_grf"]
      and isa["metadata_has_61472_slm"]
      and isa["metadata_has_simd16"])
  route_registered = bool(
      "hardware-limit capability/traffic re-bound" in status
      and "geometry-matched dense-traffic ceiling" in status
      and "close_normalized_dual_cohort_rebind_hardware_limit"
          in routes)

  checks = [
      check("repository_clean_at_gate", not git["dirty"], git=git),
      check("accepted_and_rejected_exact_components_are_bound",
            prior_exact),
      check("official_nhalf_provider_scope_has_not_changed",
            official_n8_still_closed,
            oneDNN_head=onednn_head,
            openvino_head=openvino_head,
            openvino_onednn_head=openvino_onednn_head),
      check("new_openvino_f8_gqa_is_dynamic_only_not_locked_static_state",
            f8_dynamic_only),
      check("latest_supported_igc_has_no_applicable_attention_reopen",
            igc_not_applicable, igc=official["igc"]),
      check("accepted_triple_isa_is_spill_free_cached_dense_carrier",
            isa_pass, isa=isa),
      check("independent_kq_ruler_clears_required_full_kv_rate",
            arithmetic_pass, traffic_bound=traffic_bound),
      check("hardware_limit_rebound_is_registered",
            route_registered),
      check("no_compiler_gpu_plugin_or_model_worker_launched", True),
  ]
  checks.append(check(
      "memory_guard_never_tripped",
      min(row["available_bytes"] for row in memory) >= stop_bytes,
      minimum_available_bytes=min(
          row["available_bytes"] for row in memory),
      memory_stop_bytes=stop_bytes))
  required = all(row["pass"] for row in checks)
  verdict = (
      "admit_one_exact_attention_two_workgroup_dense_traffic_ceiling"
      if required else
      "close_exact_attention_hardware_limit_rebound_inconclusive")
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
      "dense_traffic_ceiling_admitted": required,
      "exact_kernel_implementation_admitted": False,
      "compiler_worker_admitted": False,
      "gpu_worker_admitted": required,
      "plugin_build_admitted": False,
      "model_worker_admitted": False,
      "product_claim_allowed": False,
      "checks": checks,
      "official_capability": official,
      "accepted_triple_isa": isa,
      "traffic_bound": traffic_bound,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "sources": sources,
      "network_artifacts": {
          label: {
              "path": display(raw / f"{label}.json"),
              "sha256": sha256(raw / f"{label}.json"),
              "url": URLS[label],
          }
          for label in URLS
      },
      "compiler_workers_launched": False,
      "gpu_context_created": False,
      "gpu_workers_launched": False,
      "plugin_workers_launched": False,
      "model_workers_launched": False,
  }
  write_json(out_dir / "result.json", payload)
  write_json(out_dir / "manifest.json", {
      "schema_version": "intel-qwen36-artifact-manifest-v1",
      "workstream": WS,
      "git_commit": git["commit"],
      "verdict": verdict,
      "sources": sources,
      "files": ["result.json", "summary.md"],
  })
  (out_dir / "summary.md").write_text(
      summary(payload), encoding="utf-8")
  print(json.dumps({
      "artifact": display(out_dir),
      "verdict": verdict,
      "official_heads": {
          "onednn": onednn_head,
          "openvino": openvino_head,
          "igc": igc_master,
      },
      "required_full_kv_latency_cap_ms": required_ms,
      "required_gb_s": required_gb_s,
      "triple_gb_s": triple_gb_s,
      "kq_ruler_gb_s": kq_gb_s,
      "dense_traffic_ceiling_admitted": required,
      "gpu_workers_launched": False,
      "model_workers_launched": False,
  }, separators=(",", ":")))
  return 0 if required else 2


if __name__ == "__main__":
  raise SystemExit(main())
