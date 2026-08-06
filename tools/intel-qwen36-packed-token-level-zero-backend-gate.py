#!/usr/bin/env python3
"""Gate the real 40-layer packed-token Level Zero backend on PTL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
TOKEN_DIR = ROOT / "output/seq571-state-conditioned-head-correction-token-input-20260710Tseq571Z/token-input"
BUILD_DIR = ROOT / "build/engine"
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CASES = (
    ("fresh_arithmetic_01", "fit", "arithmetic"),
    ("fresh_code_03", "validation", "code"),
    ("fresh_instruction_04", "test", "instruction"),
)


def run(
    command: list[str], timeout: int = 7200,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
        timeout=timeout,
    )


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
    raise RuntimeError("smoke output has no JSON object")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--int8-block32-kv-gqa", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    raw_dir = out_dir / "raw"
    generated_dir = out_dir / "generated"
    raw_dir.mkdir(parents=True, exist_ok=False)
    generated_dir.mkdir(parents=True, exist_ok=True)

    git_commit = run(["git", "rev-parse", "HEAD"], 30).stdout.strip()
    dirty_paths = [
        line for line in run(["git", "status", "--porcelain"], 30).stdout.splitlines()
        if line and str(out_dir.relative_to(ROOT)) not in line
    ]
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    module = generated_dir / "iq36_q4x8_all.bin"
    compile_command = [
        "ocloc", "compile",
        "-file", str(ROOT / "engine/gpu/opencl/q4x8_matvec.cl"),
        "-device", "0xb080",
        "-output", "iq36_q4x8_all",
        "-out_dir", str(generated_dir),
        "-output_no_suffix",
        "--format", "zebin",
        "-options", "-cl-std=CL3.0 -D IQ36_USE_INTEGER_DOT=1",
        "-q",
    ]
    compile_run = run(compile_command, 300)
    write_json(raw_dir / "compile.json", {
        "command": compile_command,
        "returncode": compile_run.returncode,
        "stderr": compile_run.stderr,
        "stdout": compile_run.stdout,
    })

    configure_command = [
        str(CMAKE), "-S", str(ROOT / "engine"), "-B", str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    configure_run = run(configure_command, 300)
    build_command = [
        str(CMAKE), "--build", str(BUILD_DIR), "--target",
        "iq36-packed-token-level-zero-backend-smoke", "-j8",
    ]
    build_run = run(build_command, 600)
    write_json(raw_dir / "build.json", {
        "build": {
            "command": build_command,
            "returncode": build_run.returncode,
            "stderr": build_run.stderr,
            "stdout": build_run.stdout,
        },
        "configure": {
            "command": configure_command,
            "returncode": configure_run.returncode,
            "stderr": configure_run.stderr,
            "stdout": configure_run.stdout,
        },
    })

    executable = BUILD_DIR / "iq36-packed-token-level-zero-backend-smoke"
    rows: list[dict[str, Any]] = []
    distribution_diagnostic: dict[str, Any] | None = None
    build_ok = (
        compile_run.returncode == 0 and module.is_file() and
        configure_run.returncode == 0 and build_run.returncode == 0 and
        executable.is_file()
    )
    if build_ok:
        candidate_env = os.environ.copy()
        if args.int8_block32_kv_gqa:
            candidate_env["IQ36_INT8_BLOCK32_KV_GQA"] = "1"
        for case_id, split, domain in CASES:
            command = [
                str(executable), str(MODEL), str(module),
                str(TOKEN_DIR / f"{case_id}.tokens.u32"),
            ]
            completed = run(command, environment=candidate_env)
            (raw_dir / f"{case_id}.stdout").write_text(completed.stdout)
            (raw_dir / f"{case_id}.stderr").write_text(completed.stderr)
            write_json(raw_dir / f"{case_id}.command.json", {
                "command": command,
                "environment": {"IQ36_INT8_BLOCK32_KV_GQA": (
                    "1" if args.int8_block32_kv_gqa else "unset")},
                "returncode": completed.returncode,
            })
            try:
                result = parse_last_json(completed.stdout)
            except RuntimeError as error:
                result = {"parse_error": str(error)}
            rows.append({
                "case_id": case_id,
                "domain": domain,
                "returncode": completed.returncode,
                "split": split,
                **result,
            })
        distribution_command = [
            str(executable), str(MODEL), str(module),
            str(TOKEN_DIR / "fresh_code_03.tokens.u32"),
        ]
        distribution_env = os.environ.copy()
        distribution_env["IQ36_DISTRIBUTION_CHECK"] = "1"
        if args.int8_block32_kv_gqa:
            distribution_env["IQ36_INT8_BLOCK32_KV_GQA"] = "1"
        completed = run(distribution_command, environment=distribution_env)
        (raw_dir / "fresh_code_03.distribution.stdout").write_text(
            completed.stdout)
        (raw_dir / "fresh_code_03.distribution.stderr").write_text(
            completed.stderr)
        write_json(raw_dir / "fresh_code_03.distribution.command.json", {
            "command": distribution_command,
            "environment": {
                "IQ36_DISTRIBUTION_CHECK": "1",
                "IQ36_INT8_BLOCK32_KV_GQA": (
                    "1" if args.int8_block32_kv_gqa else "unset"),
            },
            "returncode": completed.returncode,
        })
        try:
            distribution_diagnostic = parse_last_json(completed.stdout)
        except RuntimeError as error:
            distribution_diagnostic = {"parse_error": str(error)}

    all_correct = bool(rows) and len(rows) == len(CASES) and all(
        row.get("returncode") == 0 and
        row.get("required_checks_passed") is True and
        row.get("exact_generated_ids") is True and
        row.get("same_top1") is True
        for row in rows
    )
    expected_kv_dtype = (
        "int8_block32_fp16_scale_f32_hot8192"
        if args.int8_block32_kv_gqa else "f32")
    kv_dtype_passed = bool(rows) and all(
        row.get("full_kv_dtype") == expected_kv_dtype for row in rows) and bool(
            isinstance(distribution_diagnostic, dict) and
            distribution_diagnostic.get("full_kv_dtype") == expected_kv_dtype)
    wall_rows = [float(row["wall_ms_median"]) for row in rows
                 if "wall_ms_median" in row]
    device_rows = [float(row["device_ms_median"]) for row in rows
                   if "device_ms_median" in row]
    wall_median = statistics.median(wall_rows) if wall_rows else None
    device_median = statistics.median(device_rows) if device_rows else None
    decode_tps = 1000.0 / wall_median if wall_median else None
    short_decode_speed_passed = bool(
        all_correct and wall_median is not None and device_median is not None and
        wall_median <= 20.080 and device_median <= 19.980
    )
    distribution_diagnostic_passed = bool(
        isinstance(distribution_diagnostic, dict) and
        isinstance(distribution_diagnostic.get("distribution_ladder"), dict) and
        distribution_diagnostic["distribution_ladder"].get(
            "required_checks_passed") is True
    )
    checks = [
        {"name": "repository_clean_at_gate", "pass": not dirty_paths,
         "dirty_paths": dirty_paths},
        {"name": "target_module_and_smoke_build", "pass": build_ok},
        {"name": "fit_validation_test_exact_eight_token_consensus",
         "pass": all_correct},
        {"name": "teacher_forced_full_vocab_distribution",
         "pass": distribution_diagnostic_passed},
        {"name": "full_kv_dtype_selected", "pass": kv_dtype_passed,
         "expected": expected_kv_dtype},
        {"name": "device_time_at_most_19_980_ms", "pass": bool(
            device_median is not None and device_median <= 19.980)},
        {"name": "wall_time_at_most_20_080_ms", "pass": bool(
            wall_median is not None and wall_median <= 20.080)},
    ]
    correctness_passed = (
        not dirty_paths and build_ok and all_correct and kv_dtype_passed and
        distribution_diagnostic_passed
    )
    short_decode_slice_ready = correctness_passed and short_decode_speed_passed
    if short_decode_slice_ready:
        disposition = "accept_short_decode_performance_and_distribution_slice"
    elif dirty_paths:
        disposition = "record_short_decode_candidate_reject_dirty_tree"
    elif not build_ok or not all_correct:
        disposition = "reject_real_backend_correctness"
    else:
        disposition = "accept_real_backend_correctness_reject_short_decode_speed"
    result = {
        "checks": checks,
        "correctness_checks_passed": correctness_passed,
        "created_at": created_at,
        "decode_tokens_s_median": decode_tps,
        "device_ms_median": device_median,
        "disposition": disposition,
        "distribution_diagnostic": distribution_diagnostic,
        "distribution_diagnostic_passed": distribution_diagnostic_passed,
        "git": {"commit": git_commit, "dirty": bool(dirty_paths),
                "dirty_paths": dirty_paths},
        "full_kv_dtype": expected_kv_dtype,
        "product_promotion_ready": False,
        "required_checks_passed": correctness_passed,
        "rows": rows,
        "schema_version": "intel-qwen36-packed-token-level-zero-backend-gate-v1",
        "scoped_short_decode_claim_allowed": short_decode_slice_ready,
        "short_decode_speed_passed": short_decode_speed_passed,
        "speedup_claims_allowed": short_decode_slice_ready,
        "wall_ms_median": wall_median,
    }
    write_json(out_dir / "result.json", result)
    write_json(out_dir / "correctness.json", {
        "checks": checks,
        "correctness_checks_passed": correctness_passed,
        "product_promotion_ready": False,
        "required_checks_passed": correctness_passed,
        "schema_version": result["schema_version"],
        "scoped_short_decode_claim_allowed": short_decode_slice_ready,
        "short_decode_speed_passed": short_decode_speed_passed,
        "speedup_claims_allowed": short_decode_slice_ready,
    })
    write_json(out_dir / "manifest.json", {
        "artifact": str(out_dir.relative_to(ROOT)),
        "created_at": created_at,
        "git": result["git"],
        "required_checks_passed": correctness_passed,
        "schema_version": result["schema_version"],
        "scoped_short_decode_claim_allowed": short_decode_slice_ready,
        "speedup_claims_allowed": short_decode_slice_ready,
        "tool": str(Path(__file__).relative_to(ROOT)),
        "workstream": "intel-qwen36-35b-a3b-gguf-q4km",
    })
    with (out_dir / "case-results.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    summary = [
        "# Packed-token Level Zero real-backend gate",
        "",
        f"- correctness checks passed: `{str(correctness_passed).lower()}`",
        f"- short decode speed passed: `{str(short_decode_speed_passed).lower()}`",
        f"- full KV dtype: `{expected_kv_dtype}`",
        "- product promotion ready: `false`",
        f"- median device / wall: `{device_median:.3f} / {wall_median:.3f} ms/token`"
        if device_median is not None and wall_median is not None else
        "- median device / wall: unavailable",
        f"- median decode: `{decode_tps:.3f} tok/s`" if decode_tps else
        "- median decode: unavailable",
        "",
        "| case | split | exact 8-token consensus | wall ms/token |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        summary.append(
            f"| {row['case_id']} | {row['split']} | "
            f"{str(bool(row.get('exact_generated_ids'))).lower()} | "
            f"{float(row.get('wall_ms_median', 0.0)):.3f} |"
        )
    summary.extend([
        "",
        "This gate promotes only the scoped 1k packed-token decode slice when "
        "its timing, exact-token, and teacher-forced distribution checks pass. "
        "Product promotion remains blocked on native prefill, long-context "
        "sentinel, smoothness, and the complete acceptance matrix.",
    ])
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    return 0 if correctness_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
