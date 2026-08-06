#!/usr/bin/env python3
"""Run the one paired captured-state native Level Zero component probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
ACTIVE = ROOT / "doc/active" / WORKSTREAM
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
  sys.path.insert(0, str(TOOLS))

import iq36_local  # noqa: E402


HARNESS_SOURCE = (
    ROOT / "tools/intel-qwen36-gpu-level-zero-postconv-recurrent-component-harness.py"
)
SCHEMA_VERSION = (
    "intel-qwen36-gpu-level-zero-postconv-recurrent-component-probe-gate-v0")
CURRENT_ROUTE = "gpu_level_zero_postconv_recurrent_component_probe_gate"
PASS_NEXT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_integration_contract_gate")
FAIL_NEXT_ROUTE = (
    "gpu_level_zero_postconv_recurrent_component_route_close_gate")


def _load_module(path: Path, name: str) -> Any:
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


HARNESS = _load_module(HARNESS_SOURCE, "iq36_level_zero_probe_harness")


def _load(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise TypeError(f"{path} does not contain a JSON object")
  return payload


def _rel(path: Path) -> str:
  try:
    return str(path.resolve().relative_to(ROOT))
  except ValueError:
    return str(path)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _has_candidate(routes: dict[str, Any], seq: int,
                   next_route: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq") == seq
      and row.get("selected_next_route") == next_route
      for row in routes.get("candidate_history", []))


def _has_switch(routes: dict[str, Any], seq: int, decision: str) -> bool:
  return any(
      isinstance(row, dict)
      and row.get("seq_covered") == seq
      and row.get("decision") == decision
      and row.get("resolved") is True
      for row in routes.get("switch_decisions", []))


def _parse_json_line(stdout: str) -> dict[str, Any] | None:
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line.startswith("{"):
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return None


def _comparison_exact(row: dict[str, Any], name: str) -> bool:
  comparison = row.get("comparisons", {}).get(name, {})
  return (
      comparison.get("same_size") is True
      and comparison.get("finite") is True
      and comparison.get("mismatch_count") == 0)


def _row_summary(label: str, command: dict[str, Any],
                 parsed: dict[str, Any] | None) -> dict[str, Any]:
  comparisons = [
      "q_conv_predelta_vs_cpu", "k_conv_predelta_vs_cpu",
      "v_conv_predelta_vs_cpu", "attention_vs_cpu", "state_vs_cpu",
      "final_vs_cpu",
  ]
  exact = (
      isinstance(parsed, dict)
      and all(_comparison_exact(parsed, name) for name in comparisons))
  timing = parsed.get("timings", {}) if isinstance(parsed, dict) else {}
  budget = (
      isinstance(parsed, dict)
      and parsed.get("checks", {}).get("paired_wall_budget_passed") is True
      and isinstance(timing.get("candidate_added_min_us"), (int, float))
      and timing["candidate_added_min_us"] <= 6.841858993929781)
  return {
      "label": label,
      "returncode": command.get("returncode"),
      "json_observed": isinstance(parsed, dict),
      "device_opencl": parsed.get("opencl_device") if parsed else None,
      "device_level_zero": parsed.get("level_zero_device") if parsed else None,
      "samples": parsed.get("samples") if parsed else None,
      "candidate_bit_exact": exact,
      "budget_passed": budget,
      "candidate_added_min_us": timing.get("candidate_added_min_us"),
      "candidate_added_mean_us": timing.get("candidate_added_mean_us"),
      "current_wall_min_us": timing.get("current_wall_min_us"),
      "candidate_wall_min_us": timing.get("candidate_wall_min_us"),
      "passed": exact and budget,
      "comparisons": parsed.get("comparisons", {}) if parsed else {},
      "stderr": command.get("stderr"),
  }


def compute(args: argparse.Namespace) -> dict[str, Any]:
  routes = _load(args.routes)
  target_compile = _load(args.target_compile)
  source_gate = _load(args.source_gate)
  payloads = HARNESS.payload_manifest(args.layer)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  binary = str(target_compile.get("compile", {}).get("binary", ""))
  expected_binary_sha = target_compile.get(
      "binary_identity", {}).get("sha256")
  expected_module_sha = source_gate.get("native_module", {}).get("sha256")
  remote_dir = (
      f"{args.remote_root.rstrip('/')}/seq{args.sequence}-level-zero-component")
  remote_payload = remote_dir + "/payload"
  remote_module = remote_dir + "/iq36_postconv_recurrent.bin"
  target_selects = (
      target_compile.get("required_checks_passed") is True
      and target_compile.get("target_compile_passed") is True
      and target_compile.get("component_probe_allowed") is True
      and target_compile.get("token_row_allowed") is False
      and target_compile.get("selected_next_route") == args.current_route
      and target_compile.get("disposition") == args.target_disposition
      and _has_candidate(routes, args.target_sequence, args.current_route)
      and _has_switch(
          routes, args.target_sequence, args.target_decision))
  setup = iq36_local.run_target(
      args.host,
      " && ".join([
          "rm -rf " + shlex.quote(remote_dir),
          "mkdir -p " + shlex.quote(remote_payload),
      ]),
      args.timeout_s)
  transfers: dict[str, Any] = {}
  if setup.get("returncode") == 0:
    for key, row in payloads.items():
      transfers[f"payload:{key}"] = iq36_local.copy_to(
          args.host, ROOT / str(row["path"]),
          remote_payload + "/" + str(row["stage_name"]), args.timeout_s)
    transfers["native_module"] = iq36_local.copy_to(
        args.host, args.native_module, remote_module, args.timeout_s)
  transfer_ok = (
      len(transfers) == len(payloads) + 1
      and all(row.get("returncode") == 0 for row in transfers.values()))
  identity = (
      iq36_local.run_target(
          args.host,
          " && ".join([
              f"sha256sum {shlex.quote(binary)}",
              f"sha256sum {shlex.quote(remote_module)}",
          ]),
          args.timeout_s)
      if transfer_ok else {})
  identity_hashes = [
      line.split(maxsplit=1)[0]
      for line in str(identity.get("stdout", "")).splitlines()
      if line.strip()
  ]
  identity_ok = (
      identity.get("returncode") == 0
      and identity_hashes == [expected_binary_sha, expected_module_sha])
  run_command = " && ".join([
      f"source {shlex.quote(args.env_script)} >/dev/null 2>&1",
      " ".join([
          shlex.quote(binary),
          "--model", shlex.quote(args.model),
          "--payload-dir", shlex.quote(remote_payload),
          "--native-module", shlex.quote(remote_module),
          "--opencl-device", shlex.quote(args.opencl_device),
          "--layer", str(args.layer),
          "--samples", str(args.samples),
      ]),
  ])
  repeat_command = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(run_command)}", args.timeout_s)
      if identity_ok else {})
  confirm_command = (
      iq36_local.run_target(
          args.host, f"bash -lc {shlex.quote(run_command)}", args.timeout_s)
      if identity_ok else {})
  repeat_json = _parse_json_line(str(repeat_command.get("stdout", "")))
  confirm_json = _parse_json_line(str(confirm_command.get("stdout", "")))
  repeat = _row_summary("repeat", repeat_command, repeat_json)
  confirm = _row_summary("confirm", confirm_command, confirm_json)
  process_check = iq36_local.run_target(
      args.host, "pgrep -af iq36-level-zero-component-probe || true",
      args.timeout_s)
  lingering = [
      line for line in str(process_check.get("stdout", "")).splitlines()
      if "pgrep -af" not in line and "bash -lc" not in line
  ]
  cleanup = iq36_local.run_target(
      args.host, "rm -rf " + shlex.quote(remote_dir), args.timeout_s)
  iq36_local.write_json(raw_dir / "setup.json", setup)
  iq36_local.write_json(raw_dir / "transfers.json", transfers)
  iq36_local.write_json(raw_dir / "identity.json", identity)
  iq36_local.write_json(raw_dir / "repeat-run.json", repeat_command)
  iq36_local.write_json(raw_dir / "confirm-run.json", confirm_command)
  iq36_local.write_json(raw_dir / "process-check.json", process_check)
  iq36_local.write_json(raw_dir / "cleanup.json", cleanup)
  rows_attempted = (
      repeat_command.get("returncode") is not None
      and confirm_command.get("returncode") is not None)
  rows_parsed = isinstance(repeat_json, dict) and isinstance(confirm_json, dict)
  candidate_passed = repeat["passed"] and confirm["passed"]
  evidence_complete = (
      target_selects and transfer_ok and identity_ok and rows_attempted
      and rows_parsed and not lingering and cleanup.get("returncode") == 0)
  selected_next = (
      args.pass_next_route if candidate_passed else args.fail_next_route)
  checks = [
      {"name": "seq616_selected_one_paired_level_zero_component_probe",
       "pass": target_selects},
      {"name": "binary_module_and_captured_payloads_staged_with_identity",
       "pass": transfer_ok and identity_ok,
       "detail": {
           "binary_sha256": expected_binary_sha,
           "native_module_sha256": expected_module_sha,
           "payloads": payloads,
       }},
      {"name": "repeat_and_confirm_both_completed_with_result_json",
       "pass": rows_attempted and rows_parsed},
      {"name": "both_rows_bit_exact_and_inside_paired_wall_budget",
       "pass": candidate_passed, "detail": [repeat, confirm]},
      {"name": "probe_process_payload_and_module_staging_cleaned",
       "pass": not lingering and cleanup.get("returncode") == 0,
       "detail": {"lingering": lingering}},
  ]
  return {
      "schema_version": args.schema_version,
      "workstream": WORKSTREAM,
      "sequence": args.sequence,
      "tool": args.tool_path,
      "inputs": {
          "routes": _rel(args.routes),
          "target_compile": _rel(args.target_compile),
          "source_gate": _rel(args.source_gate),
          "binary": binary,
          "binary_sha256": expected_binary_sha,
          "native_module": _rel(args.native_module),
          "native_module_sha256": _sha256(args.native_module),
          "payloads": payloads,
          "host": args.host,
          "model": args.model,
          "layer": args.layer,
          "samples": args.samples,
      },
      "rows": [repeat, confirm],
      "checks": checks,
      "required_checks_passed": evidence_complete,
      "measurement_complete": evidence_complete,
      "component_passed": candidate_passed,
      "component_rejected": evidence_complete and not candidate_passed,
      "decode_integration_allowed": False,
      "integration_contract_allowed": candidate_passed,
      "component_route_close_allowed": evidence_complete and not candidate_passed,
      "token_row_allowed": False,
      "validation_or_test_allowed": False,
      "speedup_claims_allowed": False,
      "promotion_allowed": False,
      "disposition": (
          args.accept_disposition
          if candidate_passed else
          args.reject_disposition),
      "selected_next_route": selected_next,
      "next_route_reason": (
          args.pass_reason
          if candidate_passed else
          args.fail_reason),
  }


def write_outputs(metrics: dict[str, Any], out_dir: Path) -> None:
  (out_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (out_dir / "manifest.json").write_text(
      json.dumps({
          "schema_version": metrics["schema_version"],
          "workstream": metrics["workstream"],
          "tool": metrics["tool"],
          "inputs": metrics["inputs"],
          "measurement_complete": metrics["measurement_complete"],
          "component_passed": metrics["component_passed"],
          "selected_next_route": metrics["selected_next_route"],
          "decode_integration_allowed": False,
          "token_row_allowed": False,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  failed = [row["name"] for row in metrics["checks"] if not row["pass"]]
  lines = [
      f"# Seq{metrics['sequence']} Native Level Zero Component Probe",
      "",
      f"- measurement_complete: `{str(metrics['measurement_complete']).lower()}`",
      f"- component_passed: `{str(metrics['component_passed']).lower()}`",
      f"- disposition: `{metrics['disposition']}`",
      f"- selected_next_route: `{metrics['selected_next_route']}`",
      f"- failed_checks: `{failed}`",
  ]
  for row in metrics["rows"]:
    lines.append(
        f"- {row['label']}: exact `{row['candidate_bit_exact']}`, "
        f"added `{row['candidate_added_min_us']}` us, "
        f"budget `{row['budget_passed']}`")
  lines += [
      "", metrics["next_route_reason"], "",
      "This is component evidence only, not a decode or speed claim.", "",
  ]
  (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=617)
  parser.add_argument("--target-sequence", type=int, default=616)
  parser.add_argument("--schema-version", default=SCHEMA_VERSION)
  parser.add_argument("--current-route", default=CURRENT_ROUTE)
  parser.add_argument("--pass-next-route", default=PASS_NEXT_ROUTE)
  parser.add_argument("--fail-next-route", default=FAIL_NEXT_ROUTE)
  parser.add_argument(
      "--target-decision",
      default="select_gpu_level_zero_postconv_recurrent_component_probe_gate")
  parser.add_argument(
      "--target-disposition",
      default="accept_level_zero_ocloc_fused_postconv_recurrent_v1_target_compile")
  parser.add_argument(
      "--accept-disposition",
      default="accept_level_zero_ocloc_fused_postconv_recurrent_v1_component")
  parser.add_argument(
      "--reject-disposition",
      default="reject_level_zero_ocloc_fused_postconv_recurrent_v1_component")
  parser.add_argument(
      "--pass-reason",
      default=(
          "The paired component passes exactness and the floor kill-number. "
          "Audit only a contiguous Level Zero island or external-memory "
          "contract next; a host OpenCL/Level Zero bridge remains forbidden."))
  parser.add_argument(
      "--fail-reason",
      default=(
          "The one locked Level Zero attempt failed exactness, budget, or "
          "runtime compatibility. Close the component class under the seq614 "
          "stop condition; do not sweep flags, workgroups, or arithmetic order."))
  parser.add_argument("--tool-path", default=_rel(Path(__file__)))
  parser.add_argument("--routes", type=Path,
                      default=ACTIVE / "routes-ledger.json")
  parser.add_argument(
      "--target-compile", type=Path,
      default=ROOT / (
          "output/seq616-gpu-level-zero-postconv-recurrent-component-target-"
          "compile-gate-20260710Tseq616Z/metrics.json"))
  parser.add_argument(
      "--source-gate", type=Path,
      default=ROOT / (
          "output/seq615-gpu-level-zero-postconv-recurrent-component-source-"
          "gate-20260710Tseq615Z/metrics.json"))
  parser.add_argument(
      "--native-module", type=Path,
      default=ROOT / (
          "output/seq615-gpu-level-zero-postconv-recurrent-component-source-"
          "gate-20260710Tseq615Z/generated/iq36_postconv_recurrent.bin"))
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=HARNESS.DEFAULT_HOST)
  parser.add_argument("--model", default=HARNESS.DEFAULT_MODEL)
  parser.add_argument("--env-script", default=HARNESS.DEFAULT_ENV_SCRIPT)
  parser.add_argument("--staging-root", "--remote-root", dest="remote_root", metavar="PATH", default=HARNESS.DEFAULT_REMOTE_ROOT)
  parser.add_argument("--opencl-device", default="B390")
  parser.add_argument("--layer", type=int, default=0)
  parser.add_argument("--samples", type=int, default=11)
  parser.add_argument("--timeout-s", type=int, default=900)
  parser.add_argument(
      "--out-dir", type=Path,
      default=ROOT / (
          "output/seq617-gpu-level-zero-postconv-recurrent-component-probe-"
          "gate-20260710Tseq617Z"))
  args = parser.parse_args()
  metrics = compute(args)
  write_outputs(metrics, args.out_dir)
  print(json.dumps({
      "required_checks_passed": metrics["required_checks_passed"],
      "measurement_complete": metrics["measurement_complete"],
      "component_passed": metrics["component_passed"],
      "disposition": metrics["disposition"],
      "selected_next_route": metrics["selected_next_route"],
      "out_dir": _rel(args.out_dir),
  }, sort_keys=True))
  return 0 if metrics["required_checks_passed"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
