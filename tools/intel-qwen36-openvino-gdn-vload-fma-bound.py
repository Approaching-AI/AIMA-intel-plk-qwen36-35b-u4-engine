#!/usr/bin/env python3
"""Gate the upstream GDN vload/FMA successor before a candidate build.

OpenVINO commit 9e4ea4f9 replaces scalar Q/K/V loads and vector-product
materialization in the reference GatedDeltaNet kernel with native vector loads
and scalar FMA chains.  This source-only gate proves that the optimized branch
is exactly the locked Qwen3.6 decode branch and that the complete GDN bucket is
large enough to fund the residual left by the fixed-FC ceiling.  It never
invokes a compiler, creates a GPU context, or loads the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WS = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WS
SCHEMA = "intel-qwen36-openvino-gdn-vload-fma-bound-v0"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
FIXED_FC = ROOT / (
    "output/openvino-fc-micro-component-"
    "20260715Tseq1233-max-native-fused-nonzero-warm512-cleanZ/metrics.json")
CURRENT_PROFILE = ROOT / (
    "output/openvino-assign-device-memory-component-"
    "20260717Tseq1284-control-2k-warm17-cleanZ/metrics.json")
RUNTIME_PROGRAM = ROOT / (
    "output/openvino-attention-phase-profile-"
    "20260715Tseq1136-dq-subgroup-32k-warm17-cleanZ/raw/32k/candidate/"
    "programs/stock-gdn-program009.cl")
TRANSPOSED_PATCH = ROOT / "engine/openvino/iq36-gdn-transposed-state.patch"
PINNED_OPENVINO = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")

PINNED_SHA = "90214e5be052438cec5617ed3ea7e37df1538f68"
UPSTREAM_COMMIT = "9e4ea4f9316d7755bfbf36faa68171cd6269c1b1"
UPSTREAM_PATH = (
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/"
    "gated_delta_net_ref.cl")
CURRENT_TPOT_MS = 29.748
TPOT_CAP_MS = 26.911
KILL_NUMBER_MS = CURRENT_TPOT_MS - TPOT_CAP_MS
REGISTERED_GDN_MS = 1.319


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0.0:
    parser.error("memory limit must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def display_path(path: Path) -> str:
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
    label: str, stop_bytes: int, rows: list[dict[str, Any]],
) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop_bytes:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop_bytes} bytes")


def git_output(args: list[str]) -> str:
  return subprocess.run(
      ["git", *args], cwd=PINNED_OPENVINO, text=True,
      capture_output=True, check=True).stdout


def git_state(output: Path) -> dict[str, Any]:
  commit = subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.strip()
  status = subprocess.run(
      ["git", "status", "--porcelain"], cwd=ROOT, text=True,
      capture_output=True, check=True).stdout.splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  status = [row for row in status if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


def macro(program: str, name: str) -> str | None:
  match = re.search(rf"^#define\s+{re.escape(name)}\s+(.+?)\s*$", program,
                    flags=re.MULTILINE)
  return match.group(1) if match else None


def main() -> int:
  args = parse_args()
  output = args.output.resolve()
  output.mkdir(parents=True, exist_ok=False)
  raw = output / "raw"
  raw.mkdir()
  stop_bytes = int(args.memory_stop_gib * 1024**3)
  memory: list[dict[str, Any]] = []
  sample_memory("start", stop_bytes, memory)

  required = (
      STATUS, ROUTES, REJECTED, FIXED_FC, CURRENT_PROFILE,
      RUNTIME_PROGRAM, TRANSPOSED_PATCH, PINNED_OPENVINO / ".git",
  )
  missing = [display_path(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing GDN-bound inputs: " + ", ".join(missing))

  git = git_state(output)
  status_text = STATUS.read_text(encoding="utf-8")
  routes = load_json(ROUTES)
  rejected = load_json(REJECTED)
  fixed = load_json(FIXED_FC)
  profile = load_json(CURRENT_PROFILE)
  runtime_program = RUNTIME_PROGRAM.read_text(encoding="utf-8")
  transposed_patch = TRANSPOSED_PATCH.read_text(encoding="utf-8")
  sample_memory("after-local-evidence", stop_bytes, memory)

  pinned_source = git_output(["show", f"{PINNED_SHA}:{UPSTREAM_PATH}"])
  upstream_source = git_output(
      ["show", f"{UPSTREAM_COMMIT}:{UPSTREAM_PATH}"])
  upstream_patch = git_output(
      ["diff", f"{UPSTREAM_COMMIT}^", UPSTREAM_COMMIT, "--", UPSTREAM_PATH])
  upstream_meta = git_output(
      ["show", "-s", "--format=%H%n%P%n%s%n%ci", UPSTREAM_COMMIT])
  changed_paths = [row for row in git_output(
      ["diff-tree", "--no-commit-id", "--name-only", "-r",
       UPSTREAM_COMMIT]).splitlines() if row]
  (raw / f"openvino-{UPSTREAM_COMMIT}.patch").write_text(
      upstream_patch, encoding="utf-8")
  (raw / f"openvino-{UPSTREAM_COMMIT}.meta.txt").write_text(
      upstream_meta, encoding="utf-8")
  sample_memory("after-upstream-evidence", stop_bytes, memory)

  aggregate = fixed["aggregate"]
  fixed_fc_saving_ms = float(aggregate["optimistic_saving_ms"])
  residual_ms = KILL_NUMBER_MS - fixed_fc_saving_ms
  complete_gdn_ceiling_ms = REGISTERED_GDN_MS
  union_ceiling_ms = fixed_fc_saving_ms + complete_gdn_ceiling_ms
  projected_tpot_floor_ms = CURRENT_TPOT_MS - union_ceiling_ms
  ceiling_margin_ms = complete_gdn_ceiling_ms - residual_ms

  executed = profile["profile_audit"]["executed_counts"]
  selected_exact = profile["profile_audit"]["selected_counts_exact"]
  registered = profile["route_selection"]["eligible_buckets"]
  gdn_route = next(row for row in registered if row.get("name") == "gdn")
  closed_routes = {
      row.get("route") for row in rejected.get("rejected", [])
      if isinstance(row, dict)}

  locked_macros = {
      name: macro(runtime_program, name) for name in (
          "K_HEAD_DIM", "V_HEAD_DIM", "SUBGROUP_SIZE", "V_BLOCK_SIZE",
          "FUSE_QK_L2NORM", "OUTPUT_STATE", "INPUT0_TYPE",
          "INPUT1_TYPE", "INPUT2_TYPE", "INPUT0_TYPE_SIZE",
          "INPUT1_TYPE_SIZE", "INPUT2_TYPE_SIZE")}
  expected_macros = {
      "K_HEAD_DIM": "128", "V_HEAD_DIM": "128", "SUBGROUP_SIZE": "16",
      "V_BLOCK_SIZE": "4", "FUSE_QK_L2NORM": "1",
      "OUTPUT_STATE": "1", "INPUT0_TYPE": "half", "INPUT1_TYPE": "half",
      "INPUT2_TYPE": "half", "INPUT0_TYPE_SIZE": "2",
      "INPUT1_TYPE_SIZE": "2", "INPUT2_TYPE_SIZE": "2"}

  source_markers = (
      "inline float FUNC(dot8_fma)(float8 a, float8 b)",
      "convert_float8(vload8(0, (const __global half*)p))",
      "convert_float4(vload4(0, (const __global half*)p))",
      "float8 s = h_state[v_idx][c] * (float8)(b_g);",
      "float8 s = fma(b_k[c], (float8)(update_val), h_state[v_idx][c]);",
  )
  pinned_markers_absent = all(value not in pinned_source for value in source_markers)
  upstream_markers_present = all(value in upstream_source for value in source_markers)
  runtime_is_predecessor = (
      "inline float FUNC(sum8)(float8 v)" in runtime_program
      and "FUNC(dot8_fma)" not in runtime_program
      and "load_q8_as_float8" not in runtime_program)
  transposed_extension_default_off = all(value in transposed_patch for value in (
      "IQ36_GDN_TRANSPOSED_STATE", "TRANSPOSED_STATE_LAYOUT",
      "transposed_state_layout ? 1 : 0"))

  checks = [
      check("repository_clean_at_gate", not git["dirty"],
            dirty_paths=git["dirty_paths"]),
      check("active_owner_gate_accepts_independent_new_capability",
            routes.get("active_route", {}).get("id")
            == "openvino_locked_target_owner_contract_decision"
            and "independently verified new capability" in re.sub(
                r"\s+", " ", status_text)),
      check("upstream_commit_is_exact_single_file_gdn_change",
            changed_paths == [UPSTREAM_PATH], changed_paths=changed_paths,
            commit=UPSTREAM_COMMIT),
      check("capability_is_absent_from_pinned_runtime",
            pinned_markers_absent, pinned_commit=PINNED_SHA),
      check("upstream_vload_fma_contract_is_exact",
            upstream_markers_present,
            patch_sha256=hashlib.sha256(
                upstream_patch.encode()).hexdigest()),
      check("locked_runtime_program_is_predecessor_kernel",
            runtime_is_predecessor),
      check("locked_runtime_hits_optimized_upstream_branch",
            locked_macros == expected_macros,
            observed=locked_macros, expected=expected_macros),
      check("current_runtime_gdn_census_is_exact",
            selected_exact and int(executed.get("GatedDeltaNet", -1)) == 30,
            executed_gdn=executed.get("GatedDeltaNet")),
      check("registered_complete_gdn_bucket_is_exact",
            abs(float(gdn_route["registered_ms_per_token"])
                - REGISTERED_GDN_MS) < 1e-12,
            registered_ms=gdn_route["registered_ms_per_token"]),
      check("fixed_fc_ceiling_and_residual_are_exact",
            fixed.get("required_checks_passed") is True
            and fixed.get("verdict") == "reject_before_graph_integration"
            and abs(fixed_fc_saving_ms - 2.152360000002597) < 1e-12
            and abs(residual_ms - 0.6846399999974031) < 1e-9),
      check("complete_gdn_ceiling_clears_fixed_fc_residual",
            complete_gdn_ceiling_ms > residual_ms,
            complete_gdn_ceiling_ms=complete_gdn_ceiling_ms,
            residual_ms=residual_ms, margin_ms=ceiling_margin_ms),
      check("nonoverlapping_fc_gdn_union_clears_kill_number",
            union_ceiling_ms > KILL_NUMBER_MS,
            union_ceiling_ms=union_ceiling_ms,
            kill_number_ms=KILL_NUMBER_MS,
            margin_ms=union_ceiling_ms - KILL_NUMBER_MS),
      check("rejected_transposed_state_route_is_not_reopened",
            "openvino_gdn_transposed_state_block_io_v28c" in closed_routes
            and transposed_extension_default_off),
      check("no_compiler_gpu_or_model_worker_ran", True,
            compiler_workers=0, gpu_workers=0, model_workers=0),
  ]
  required_checks_passed = all(row["pass"] for row in checks)
  admitted = required_checks_passed
  verdict = (
      "admit_upstream_gdn_vload_fma_short_component"
      if admitted else "reject_upstream_gdn_vload_fma_before_build")

  metrics = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": git,
      "upstream": {
          "commit": UPSTREAM_COMMIT,
          "path": UPSTREAM_PATH,
          "metadata": upstream_meta.splitlines(),
          "changed_paths": changed_paths,
          "patch_sha256": hashlib.sha256(
              upstream_patch.encode()).hexdigest(),
          "source_markers_present": upstream_markers_present,
          "absent_from_pinned": pinned_markers_absent,
      },
      "locked_runtime": {
          "program": display_path(RUNTIME_PROGRAM),
          "macros": locked_macros,
          "expected_macros": expected_macros,
          "predecessor_kernel": runtime_is_predecessor,
          "executed_gdn_count": executed.get("GatedDeltaNet"),
          "registered_gdn_ms_per_token": gdn_route[
              "registered_ms_per_token"],
          "transposed_state_extension_default_off": (
              transposed_extension_default_off),
      },
      "bound": {
          "current_tpot_ms": CURRENT_TPOT_MS,
          "tpot_cap_ms": TPOT_CAP_MS,
          "kill_number_ms": KILL_NUMBER_MS,
          "seq1233_fixed_fc_optimistic_saving_ms": fixed_fc_saving_ms,
          "fixed_fc_residual_ms": residual_ms,
          "complete_gdn_ceiling_ms": complete_gdn_ceiling_ms,
          "complete_gdn_ceiling_margin_ms": ceiling_margin_ms,
          "nonoverlapping_fc_gdn_union_ceiling_ms": union_ceiling_ms,
          "projected_tpot_floor_ms": projected_tpot_floor_ms,
          "interpretation": (
              "stress admission only: grant the upstream patch the entire "
              "registered GDN bucket and add it to the non-overlapping fixed-"
              "FC ceiling; the short component must itself save at least the "
              "fixed-FC residual, and no product speed claim follows"),
      },
      "checks": checks,
      "required_checks_passed": required_checks_passed,
      "component_build_admitted": admitted,
      "graph_integration_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "verdict": verdict,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
  }
  (output / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (output / "manifest.json").write_text(
      json.dumps({
          "schema": SCHEMA,
          "tool": display_path(Path(__file__)),
          "git": git,
          "inputs": [display_path(path) for path in required],
          "upstream_commit": UPSTREAM_COMMIT,
          "memory_stop_bytes": stop_bytes,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  summary = "\n".join((
      "# OpenVINO GDN vload/FMA capability bound",
      "",
      f"Verdict: **{verdict}**. Required checks: "
      f"`{str(required_checks_passed).lower()}`. No compiler, GPU context, or "
      "model worker ran.",
      "",
      f"The locked program is exactly FP16 `K=128`, `V=128`, subgroup 16, "
      f"V-block 4, fused Q/K normalization, and output-state enabled. All "
      f"`{executed.get('GatedDeltaNet')}` runtime GDN nodes therefore hit the "
      "branch changed by the upstream vector-load/FMA commit.",
      "",
      f"Seq1233 leaves `{residual_ms:.6f} ms/token` after its fixed-FC "
      f"ceiling. The complete registered GDN bucket is "
      f"`{complete_gdn_ceiling_ms:.6f} ms/token`, so the source-only stress "
      f"ceiling clears that residual by `{ceiling_margin_ms:.6f} ms/token`. "
      "This admits one serial short component only.",
      "",
      "The rejected transposed-state route remains default-off and is not "
      "reopened. The component must preserve the current state layout, exact "
      "execution census, and teacher-forced tokens, and must measure at least "
      f"`{residual_ms:.6f} ms/token` GDN saving before any longer row.",
      "",
  ))
  (output / "summary.md").write_text(summary, encoding="utf-8")
  print(json.dumps({
      "output": display_path(output),
      "verdict": verdict,
      "required_checks_passed": required_checks_passed,
      "fixed_fc_residual_ms": residual_ms,
      "complete_gdn_ceiling_ms": complete_gdn_ceiling_ms,
      "ceiling_margin_ms": ceiling_margin_ms,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, sort_keys=True))
  return 0 if required_checks_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
