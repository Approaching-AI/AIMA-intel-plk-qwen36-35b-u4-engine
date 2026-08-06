#!/usr/bin/env python3
"""Gate the upstream CM/XMX PagedAttention head-256 backport.

This source-only gate proves that OpenVINO commit cccbdbc4 supplies a provider
program that did not exist for the locked 16/2/256 attention ABI in the pinned
runtime.  The new program is CM/DPAS rather than the previously rejected OCL
PagedAttention binary.  Dense F16 state traffic still owns the decision: only
one standalone 32k layer component is admitted, and it must demonstrate both
the complete 0.5618915-ms UCB and the corresponding 119.434-GB/s K+V rate.

The gate never invokes a compiler, creates a GPU context, or loads the model.
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
ACTIVE = ROOT / "doc" / "active" / WS
SCHEMA = "intel-qwen36-openvino-cm-paged-attention-head256-bound-v0"

STATUS = ACTIVE / "STATUS.md"
ROUTES = ACTIVE / "routes-ledger.json"
REJECTED = ACTIVE / "rejected-routes.json"
MODEL_CONTRACT = ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json"
TARGET_CONTRACT = ROOT / "contracts/intel-qwen36-target-contract.json"
DENSE_BOUND = ROOT / (
    "output/openvino-attention-dense-state-traffic-bound-"
    "20260715Tseq1241-cleanZ/metrics.json")
PAGED_PROVIDER = ROOT / (
    "output/openvino-paged-gqa-provider-20260713Tseq777cleanZ/result.json")
BACKPORT_PATCH = ROOT / "engine/openvino/iq36-cm-paged-attention-head256.patch"

OPENVINO = Path("/home/intel/intel-qwen36-r0/source/openvino-90214e5be05")
BACKPORT_PROBE = Path(
    "/home/intel/intel-qwen36-r0/source/openvino-cm-pa-head256-probe")
PINNED_SHA = "90214e5be052438cec5617ed3ea7e37df1538f68"
UPSTREAM_SHA = "cccbdbc479564aca03cd8769904ec25cc1602947"
PATCH_SHA256 = "ad85f4ebc1e2eb2ce161476ed01ae287e75abe915d58b4015ae39fdc72c58fbc"

BACKPORT_COMMITS = (
    "7876e430ee8815b421504b177b5fd7e35f68e1b2",
    "eae8f5807a09c380a74d43ab0ddfd514bdaa3004",
    "a6efe1146c846c15fe9b22ff5f8757fdb7654b5d",
    "f144078a0c3481eec78d4c43390680975a5b8825",
    "ed4cfb1cf47bb02332fd9f2ff3ba01be83d9d39e",
    UPSTREAM_SHA,
)
BACKPORT_SUBJECTS = (
    "[GPU] Multi-subsequence enablement of PA CM path (#34806)",
    "[GPU] Remove unused variable from CM PA generator (#35984)",
    "Fix XAttention for mixed multi-sequence prefill/decode test cases (#36082)",
    "[GPU][CM] Fix unused lambda capture in paged_attention_gen (#36185)",
    "Fix xattention find_block (#35697)",
    "[GPU] Implement the feature of head size 256 for cm xattention supported page attention (#36422)",
)

CM_DIR = "src/plugins/intel_gpu/src/graph/impls/cm"
PRODUCTION_PATHS = (
    "src/plugins/intel_gpu/include/intel_gpu/runtime/internal_properties.hpp",
    "src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl",
    f"{CM_DIR}/include/cm_attention_common.hpp",
    f"{CM_DIR}/include/cm_pa_xe1.hpp",
    f"{CM_DIR}/include/cm_pa_xe2.hpp",
    f"{CM_DIR}/include/estimate.hpp",
    f"{CM_DIR}/include/find_block.hpp",
    f"{CM_DIR}/include/xattn_subseq_meta.hpp",
    f"{CM_DIR}/pa_kv_cache_update_ref.cm",
    f"{CM_DIR}/pa_multi_token.cm",
    f"{CM_DIR}/pa_single_token.cm",
    f"{CM_DIR}/pa_single_token_finalization.cm",
    f"{CM_DIR}/paged_attention.cpp",
    f"{CM_DIR}/paged_attention_gen.cpp",
    f"{CM_DIR}/paged_attention_gen.hpp",
    f"{CM_DIR}/xattn_find_block.cm",
    f"{CM_DIR}/xattn_gemm_qk.cm",
    f"{CM_DIR}/xattn_post_proc.cm",
    "src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp",
    "src/plugins/intel_gpu/src/graph/paged_attention.cpp",
)
UPSTREAM_CHANGED_PATHS = (
    f"{CM_DIR}/include/cm_pa_xe1.hpp",
    f"{CM_DIR}/include/cm_pa_xe2.hpp",
    f"{CM_DIR}/pa_kv_cache_update_ref.cm",
    f"{CM_DIR}/pa_multi_token.cm",
    f"{CM_DIR}/paged_attention.cpp",
    f"{CM_DIR}/paged_attention_gen.cpp",
    f"{CM_DIR}/paged_attention_gen.hpp",
    "src/plugins/intel_gpu/tests/unit/test_cases/paged_attention_gpu_test.cpp",
)

REGISTERED_ATTENTION_MS = 8.456
KILL_NUMBER_MS = 2.837085
TARGET_ATTENTION_MS = REGISTERED_ATTENTION_MS - KILL_NUMBER_MS
FULL_ATTENTION_LAYERS = 10
COMPONENT_UCB_CAP_MS = TARGET_ATTENTION_MS / FULL_ATTENTION_LAYERS
REQUIRED_DENSE_KV_GB_S = 119.43384799378529
RAW_LPDDR_GB_S = 136.5


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--memory-stop-gib", type=float, default=4.0)
  args = parser.parse_args()
  if args.memory_stop_gib <= 0:
    parser.error("memory stop must be positive")
  return args


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def available_memory_bytes() -> int:
  for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
    if line.startswith("MemAvailable:"):
      return int(line.split()[1]) * 1024
  raise RuntimeError("MemAvailable is missing")


def sample_memory(label: str, stop: int,
                  rows: list[dict[str, Any]]) -> None:
  available = available_memory_bytes()
  rows.append({"label": label, "available_bytes": available})
  if available < stop:
    raise RuntimeError(
        f"memory stop at {label}: {available} < {stop} bytes")


def git(repo: Path, *args: str, input_bytes: bytes | None = None,
        check: bool = True) -> subprocess.CompletedProcess[bytes]:
  return subprocess.run(
      ["git", *args], cwd=repo, input=input_bytes, capture_output=True,
      check=check)


def git_text(repo: Path, *args: str) -> str:
  return git(repo, *args).stdout.decode("utf-8", errors="replace")


def repo_state(output: Path) -> dict[str, Any]:
  commit = git_text(ROOT, "rev-parse", "HEAD").strip()
  status = git_text(ROOT, "status", "--porcelain").splitlines()
  try:
    relative = str(output.resolve().relative_to(ROOT))
  except ValueError:
    relative = ""
  status = [row for row in status if not relative or relative not in row]
  return {"commit": commit, "dirty": bool(status), "dirty_paths": status}


def check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **detail}


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
      STATUS, ROUTES, REJECTED, MODEL_CONTRACT, TARGET_CONTRACT,
      DENSE_BOUND, PAGED_PROVIDER, BACKPORT_PATCH, OPENVINO / ".git",
      BACKPORT_PROBE / ".git",
  )
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing CM head-256 bound inputs: " + ", ".join(missing))

  state = repo_state(output)
  status_text = STATUS.read_text(encoding="utf-8")
  routes = load_json(ROUTES)
  rejected = load_json(REJECTED)
  model = load_json(MODEL_CONTRACT)
  target = load_json(TARGET_CONTRACT)
  dense = load_json(DENSE_BOUND)
  paged = load_json(PAGED_PROVIDER)
  patch_bytes = BACKPORT_PATCH.read_bytes()
  sample_memory("after-local-evidence", stop_bytes, memory)

  pinned_head = git_text(OPENVINO, "rev-parse", "HEAD").strip()
  upstream_meta = git_text(
      OPENVINO, "show", "-s", "--format=%H%n%P%n%ci%n%s", UPSTREAM_SHA)
  upstream_changed = tuple(sorted(row for row in git_text(
      OPENVINO, "diff-tree", "--no-commit-id", "--name-only", "-r",
      UPSTREAM_SHA).splitlines() if row))

  manager_path = f"{CM_DIR}/paged_attention.hpp"
  gen_path = f"{CM_DIR}/paged_attention_gen.hpp"
  impl_path = f"{CM_DIR}/paged_attention.cpp"
  single_path = f"{CM_DIR}/pa_single_token.cm"
  test_path = "src/plugins/intel_gpu/tests/unit/test_cases/paged_attention_gpu_test.cpp"
  pinned_gen = git_text(OPENVINO, "show", f"{PINNED_SHA}:{gen_path}")
  pinned_impl = git_text(OPENVINO, "show", f"{PINNED_SHA}:{impl_path}")
  pinned_manager = git_text(OPENVINO, "show", f"{PINNED_SHA}:{manager_path}")
  upstream_gen = git_text(OPENVINO, "show", f"{UPSTREAM_SHA}:{gen_path}")
  upstream_impl = git_text(OPENVINO, "show", f"{UPSTREAM_SHA}:{impl_path}")
  upstream_test = git_text(OPENVINO, "show", f"{UPSTREAM_SHA}:{test_path}")
  upstream_patch = git_text(
      OPENVINO, "diff", f"{UPSTREAM_SHA}^", UPSTREAM_SHA, "--",
      *UPSTREAM_CHANGED_PATHS)
  (raw / f"openvino-{UPSTREAM_SHA}.meta.txt").write_text(
      upstream_meta, encoding="utf-8")
  (raw / f"openvino-{UPSTREAM_SHA}.patch").write_text(
      upstream_patch, encoding="utf-8")

  probe_head = git_text(BACKPORT_PROBE, "rev-parse", "HEAD").strip()
  probe_status = git_text(BACKPORT_PROBE, "status", "--porcelain").splitlines()
  probe_subjects = tuple(reversed(git_text(
      BACKPORT_PROBE, "log", "-6", "--format=%s").splitlines()))
  probe_diff = git(
      BACKPORT_PROBE, "diff", "--binary", f"{PINNED_SHA}..{probe_head}",
      "--",
      "src/plugins/intel_gpu/include/intel_gpu/runtime/internal_properties.hpp",
      "src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl",
      f"{CM_DIR}",
      "src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp",
      "src/plugins/intel_gpu/src/graph/paged_attention.cpp",
  ).stdout
  probe_single = git_text(BACKPORT_PROBE, "show", f"{probe_head}:{single_path}")
  probe_gen = git_text(BACKPORT_PROBE, "show", f"{probe_head}:{gen_path}")

  apply_check = git(
      BACKPORT_PROBE, "apply", "--check", "--reverse", "-",
      input_bytes=patch_bytes,
      check=False)
  patch_paths = tuple(sorted(set(re.findall(
      rb"^diff --git a/(.+?) b/", patch_bytes, flags=re.MULTILINE))))
  patch_paths_text = tuple(value.decode("utf-8") for value in patch_paths)
  sample_memory("after-upstream-and-backport-evidence", stop_bytes, memory)

  arch = model["product_model"]["architecture"]
  q_heads = int(arch["attention_heads"])
  kv_heads = int(arch["kv_heads"])
  head_dim = int(arch["head_dim"])
  gqa_ratio = q_heads // kv_heads
  target_device = target["runtime"]["opencl_device"]
  target_label = target["target"]["machine_label"]

  dense_budget = dense["budget"]
  raw_state_ms = float(dense_budget["mandatory_dense_kv_ms_at_raw_peak"])
  nonstate_margin_ms = TARGET_ATTENTION_MS - raw_state_ms
  dense_route = next(
      row for row in rejected["rejected"]
      if row.get("route") == "openvino_dense_f16_attention_algorithm_v28s")
  paged_route = next(
      row for row in rejected["rejected"]
      if row.get("route") == "native_product_paged_gqa_provider_v11")
  paged_result = paged["result"]
  trace_names = tuple(paged_result["trace_kernel_names"])

  head256_markers = (
      "if (desc->k_head_size == 256 && xe_arch >= 2)",
      "constexpr size_t num_team = 8;",
      "return num_team * get_q_step(params);",
  )
  impl_markers = (
      "if (desc->k_head_size == 256)",
      "return block_size_128;",
  )
  manager_markers = (
      'OV_GPU_PRIMITIVE_IMPL("cm::paged_attention::opt")',
      "if (!desc->has_xattention)",
      "ov::element::f16",
      "ov::element::i8",
      "!info.supports_immad",
      "!config.get_use_cm()",
  )
  single_markers = (
      "q_heads_per_kv_head = num_heads / num_kv_heads",
      "Q_head_chunks_per_kv_head",
      "Q_head_chunk_size",
      "cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF",
  )
  test_markers = (
      "smoke_cm_xattention_head_size",
      "{{1, 31}},   2, 2, 256, 256, 256",
      "DISABLE_CACHE_COMPRESSION",
      "std::vector<float>{100.0f}",
      "ENABLE_CACHE_COMPRESSION",
      "std::vector<float>{0.9f}",
  )

  checks = [
      check("repository_clean_at_gate", not state["dirty"],
            dirty_paths=state["dirty_paths"]),
      check("owner_gate_allows_independently_verified_new_capability",
            routes["active_route"]["id"]
            == "openvino_locked_target_owner_contract_decision"
            and "independently verified new capability" in re.sub(
                r"\s+", " ", status_text)),
      check("pinned_openvino_commit_is_exact", pinned_head == PINNED_SHA,
            pinned_head=pinned_head),
      check("upstream_head256_commit_and_scope_are_exact",
            upstream_meta.splitlines()[0] == UPSTREAM_SHA
            and upstream_changed == tuple(sorted(UPSTREAM_CHANGED_PATHS)),
            upstream_commit=UPSTREAM_SHA,
            changed_paths=list(upstream_changed)),
      check("head256_cm_capability_is_absent_pinned_present_upstream",
            head256_markers[0] not in pinned_gen
            and impl_markers[0] not in pinned_impl
            and all(marker in upstream_gen for marker in head256_markers)
            and all(marker in upstream_impl for marker in impl_markers)),
      check("upstream_tests_cover_head256_generate_f16_and_i8",
            all(marker in upstream_test for marker in test_markers)),
      check("cm_provider_requires_xmx_capability_and_f16_io",
            all(marker in pinned_manager for marker in manager_markers)),
      check("backported_single_token_program_is_dpas_and_exact_gqa_chunked",
            all(marker in probe_single for marker in single_markers)
            and "MaxRepeatCount = 8" in probe_gen
            and "q_heads_per_kv_head % q_head_chunk_size" in probe_gen),
      check("locked_attention_abi_maps_exactly_to_cm_program",
            (q_heads, kv_heads, head_dim, gqa_ratio) == (16, 2, 256, 8),
            q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
            gqa_ratio=gqa_ratio),
      check("target_is_locked_ptl_b390_xe3_class",
            "B390" in target_device and "PTL" in target_label,
            target_device=target_device, target_label=target_label),
      check("six_commit_production_backport_is_exact_and_clean",
            not probe_status and probe_subjects == BACKPORT_SUBJECTS
            and probe_diff == patch_bytes
            and sha256_bytes(patch_bytes) == PATCH_SHA256
            and patch_paths_text == tuple(sorted(PRODUCTION_PATHS)),
            commits=list(BACKPORT_COMMITS), probe_head=probe_head,
            patch_sha256=sha256_bytes(patch_bytes),
            patch_paths=list(patch_paths_text)),
      check("durable_backport_reverses_exact_probe_to_pinned_source",
            apply_check.returncode == 0,
            stderr=apply_check.stderr.decode("utf-8", errors="replace")),
      check("prior_paged_provider_is_ocl_and_materially_distinct",
            len(trace_names) == 4
            and all(name.startswith("paged_attention_opt__")
                    for name in trace_names)
            and all("_cm" not in name for name in trace_names)
            and paged_route["class"]
            == "exact_product_provider_terminal_repeat_and_noise_fail",
            trace_kernel_names=list(trace_names)),
      check("prior_paged_provider_timing_is_exact",
            paged_result["token_rows"][0]["attention_sum_ms"] == 33.109161
            and paged_result["repeat_attention_ms"] == 31.229054
            and paged_result["confirm_attention_ms"] == 27.951553
            and paged_result["attention_cap_ms"] == 28.25),
      check("dense_f16_reopen_contract_and_budget_are_exact",
            "119.434 GB/s" in dense_route["reopen_condition"]
            and float(dense_budget["registered_attention_ms_per_token"])
            == REGISTERED_ATTENTION_MS
            and float(dense_budget["kill_number_ms_per_token"])
            == KILL_NUMBER_MS
            and abs(float(dense_budget["target_attention_ms_per_token"])
                    - TARGET_ATTENTION_MS) < 1e-12
            and abs(float(dense_budget["required_dense_kv_gb_s_to_fund_kill"])
                    - REQUIRED_DENSE_KV_GB_S) < 1e-12
            and float(dense_budget["raw_lpddr_gb_s"]) == RAW_LPDDR_GB_S),
      check("raw_hardware_roof_leaves_a_bounded_component_window",
            RAW_LPDDR_GB_S > REQUIRED_DENSE_KV_GB_S
            and nonstate_margin_ms > 0,
            raw_lpddr_gb_s=RAW_LPDDR_GB_S,
            required_dense_kv_gb_s=REQUIRED_DENSE_KV_GB_S,
            raw_state_ms=raw_state_ms,
            nonstate_margin_ms=nonstate_margin_ms),
      check("gate_is_source_only",
            True, compiler_invocations=0, gpu_contexts=0, model_workers=0,
            product_workers=0),
  ]
  required_passed = all(row["pass"] for row in checks)
  verdict = (
      "admit_cm_paged_attention_head256_compile_and_one_component"
      if required_passed else
      "reject_cm_paged_attention_head256_before_compile")

  result = {
      "schema": SCHEMA,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "workstream": WS,
      "git": state,
      "verdict": verdict,
      "required_checks_passed": required_passed,
      "compile_admitted": required_passed,
      "component_admitted": required_passed,
      "graph_source_admitted": False,
      "model_worker_admitted": False,
      "long_worker_admitted": False,
      "product_worker_admitted": False,
      "budget": {
          "registered_attention_ms_per_token": REGISTERED_ATTENTION_MS,
          "kill_number_ms_per_token": KILL_NUMBER_MS,
          "target_attention_ms_per_token": TARGET_ATTENTION_MS,
          "full_attention_layers": FULL_ATTENTION_LAYERS,
          "per_layer_complete_ucb_cap_ms": COMPONENT_UCB_CAP_MS,
          "required_complete_dense_kv_gb_s": REQUIRED_DENSE_KV_GB_S,
          "raw_lpddr_gb_s": RAW_LPDDR_GB_S,
          "raw_state_ms_per_token": raw_state_ms,
          "raw_nonstate_margin_ms_per_token": nonstate_margin_ms,
          "interpretation": (
              "the raw hardware roof does not prove performance; it funds one "
              "standalone component whose measured UCB must independently "
              "satisfy the dense-F16 reopen contract"),
      },
      "capability": {
          "pinned_openvino": PINNED_SHA,
          "upstream_openvino": UPSTREAM_SHA,
          "provider_identity": "cm::paged_attention::opt",
          "kernel_language": "CM",
          "score_and_value_core": "F16 DPAS",
          "locked_abi": {
              "batch": 1, "q_heads": q_heads, "kv_heads": kv_heads,
              "head_dim": head_dim, "gqa_ratio": gqa_ratio,
          },
          "backport_patch": str(BACKPORT_PATCH.relative_to(ROOT)),
          "backport_patch_sha256": sha256_file(BACKPORT_PATCH),
          "backport_commits": list(BACKPORT_COMMITS),
      },
      "component_contract": {
          "context_tokens": 32768,
          "component_layers": 1,
          "cache_precision": "F16",
          "has_xattention": True,
          "xattention_threshold": 100.0,
          "xattention_behavior": (
              "bypass sparse estimation while retaining the CM provider; no "
              "threshold, precision, block, tile, subgroup, or property sweep"),
          "minimum_complete_samples": 20,
          "one_sided_95pct_ucb_cap_ms": COMPONENT_UCB_CAP_MS,
          "minimum_effective_dense_kv_gb_s": REQUIRED_DENSE_KV_GB_S,
          "numeric_cosine_min": 0.999,
          "numeric_relative_l2_max": 0.002,
          "required_provider_identity": "cm::paged_attention::opt",
          "timed_scope": (
              "KV cache update, complete single-token CM attention, all "
              "partitions, and finalization; zero timed host transfer"),
          "build_jobs_max": 4,
          "memory_stop_bytes": stop_bytes,
          "failure_action": (
              "close before graph integration, model workers, 32k product, "
              "ABBA, output512, long rows, or variants"),
      },
      "checks": checks,
      "memory_stop_bytes": stop_bytes,
      "memory_samples": memory,
      "inputs": {
          str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
          sha256_file(path)
          for path in required if path.is_file()
      },
  }
  write_json(output / "metrics.json", result)
  (output / "summary.md").write_text(
      "# CM/XMX PagedAttention head-256 source bound\n\n"
      f"Verdict: **{verdict}**. Required checks: `{required_passed}`.\n\n"
      f"The locked 16/2/256 ABI now has an upstream `cm::paged_attention::opt` "
      "F16-DPAS program that is absent from pinned OpenVINO and distinct from "
      "seq777's rejected OCL PagedAttention binaries. The six-commit production "
      "backport applies cleanly.\n\n"
      f"Dense F16 remains bounded: one 32k layer must measure complete one-sided "
      f"95% UCB `<= {COMPONENT_UCB_CAP_MS:.7f} ms` and effective K+V "
      f"`>= {REQUIRED_DENSE_KV_GB_S:.6f} GB/s`. Only a `-j4` build and that "
      "standalone component are admitted; no model, graph integration, long, "
      "ABBA, output512, or product worker is admitted.\n",
      encoding="utf-8")
  print(json.dumps({
      "verdict": verdict,
      "required_checks_passed": required_passed,
      "component_ucb_cap_ms": COMPONENT_UCB_CAP_MS,
      "required_dense_kv_gb_s": REQUIRED_DENSE_KV_GB_S,
      "minimum_available_bytes": min(
          row["available_bytes"] for row in memory),
  }, sort_keys=True))
  return 0 if required_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
