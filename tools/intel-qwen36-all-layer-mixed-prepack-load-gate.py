#!/usr/bin/env python3
"""Generate every real mixed-codec payload and load all 40 layers once."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-all-layer-mixed-prepack-load-gate-v2"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/"
    "tensor-index.jsonl")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CMAKE = Path("/home/intel/intel-box-env/conda/bin/cmake")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "oneDNN-01b479323f794da1a7a41a6fc084c7e11ccc2c3b")
ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")
Q4_ARTIFACT = (
    ROOT / "output/grouped-s8-u4-prefill-gate-"
    "20260711Tseq673cleanZ")
Q4_LAYERS = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18,
             20, 21, 23, 24, 26, 27, 29, 30, 32, 33]
Q6_LAYERS = [0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
             22, 25, 28, 31, 34, 35, 36, 37, 38, 39]
GATEUP_SIZES = {
    "gateup-weights.bin": 268_435_456,
    "gateup-scale-codes.bin": 16_777_216,
    "gateup-min-codes.bin": 16_777_216,
    "gateup-block-ds.bin": 8_388_608,
    "gateup-dmins.bin": 8_388_608,
}
Q4_DOWN_SIZES = {
    "down-weights.bin": 134_217_728,
    "down-scale-codes.bin": 8_388_608,
    "down-min-codes.bin": 8_388_608,
    "down-block-ds.bin": 4_194_304,
    "down-dmins.bin": 4_194_304,
}
Q6_DOWN_SIZES = {
    "q6-down-exact-per16-values-u8.bin": 268_435_456,
    "q6-down-exact-block-scales-i8.bin": 16_777_216,
    "q6-down-exact-block-d-f32.bin": 4_194_304,
}
EXPECTED_RESIDENT_BYTES = 21_726_494_720


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--tensor-index", type=Path, default=TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cmake", type=Path, default=CMAKE)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--onednn-source", type=Path, default=ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path, default=ONEDNN_BUILD)
  parser.add_argument("--q4-artifact", type=Path, default=Q4_ARTIFACT)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.jobs, args.timeout_s) <= 0:
    parser.error("jobs and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/all-layer-mixed-prepack-load-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {"commit": git_output("rev-parse", "HEAD"),
          "dirty": bool(dirty), "dirty_paths": dirty.splitlines()}


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": False}
  except subprocess.TimeoutExpired as error:
    return {"command": command, "returncode": 124,
            "stdout": error.stdout if isinstance(error.stdout, str) else "",
            "stderr": error.stderr if isinstance(error.stderr, str) else "",
            "timed_out": True}


def run_env(command: list[str], args: argparse.Namespace,
            extra_env: dict[str, str] | None = None) -> dict[str, Any]:
  exports = {"INTEL_FORCE_PROBE": "b080", "DNNL_VERBOSE": "0",
             **(extra_env or {})}
  export_text = " ".join(
      f"{key}={shlex.quote(value)}" for key, value in exports.items())
  shell = (
      f"source {shlex.quote(str(args.env_script))} >/dev/null 2>&1 && "
      f"export {export_text} && {shlex.join(command)}")
  return run(["bash", "-lc", shell], args.timeout_s)


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  write_json(raw / f"{name}.command.json", {
      "command": result["command"], "returncode": result["returncode"],
      "timed_out": result["timed_out"]})
  (raw / f"{name}.stdout").write_text(
      str(result["stdout"]), encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      str(result["stderr"]), encoding="utf-8")


def parse_probe(result: dict[str, Any]) -> dict[str, Any]:
  for line in reversed(str(result.get("stdout", "")).splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def tensor_rows(path: Path) -> dict[str, dict[str, Any]]:
  rows: dict[str, dict[str, Any]] = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
      row = json.loads(line)
      rows[str(row["name"])] = row
  return rows


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  layers_root = raw / "layers"
  layers_root.mkdir()
  build_dir = raw / "build"
  q4_raw = args.q4_artifact / "raw"
  required = [
      args.model, args.tensor_index, args.env_script, args.cmake, args.cxx,
      q4_raw / "gateup.0.bin", q4_raw / "down.0.bin",
      args.onednn_source / "include/oneapi/dnnl/dnnl.hpp",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      args.onednn_build / "src/libdnnl.so",
      ROOT / "engine/tools/onednn_grouped_q4k_moe_component.cpp",
      ROOT / "engine/tools/grouped_s8_u8_q6_surrogate_down.cpp",
      ROOT / "engine/tools/grouped_mixed_prefill_all_layer_load_smoke.cpp",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  state = git_state()
  rows = tensor_rows(args.tensor_index)
  observed_q4 = [
      layer for layer in range(40)
      if rows[f"blk.{layer}.ffn_down_exps.weight"].get("type") == 12]
  observed_q6 = [
      layer for layer in range(40)
      if rows[f"blk.{layer}.ffn_down_exps.weight"].get("type") == 14]
  dummy_q4 = rows["blk.27.ffn_down_exps.weight"]

  configure = run_env([
      str(args.cmake), "-S", str(ROOT / "engine"), "-B", str(build_dir),
      "-DCMAKE_BUILD_TYPE=Release"], args)
  write_run(raw, "configure", configure)
  build = run_env([
      str(args.cmake), "--build", str(build_dir), f"-j{args.jobs}",
      "--target", "iq36-grouped-mixed-prefill-all-layer-load-smoke"], args) \
      if configure["returncode"] == 0 else {
          "command": [], "returncode": 125, "stdout": "",
          "stderr": "configure failed", "timed_out": False}
  write_run(raw, "build", build)
  q6_tool = raw / "grouped-s8-u8-q6-surrogate-down"
  q6_build = run_env([
      str(args.cxx), "-std=c++17", "-O3", "-fopenmp",
      str(ROOT / "engine/tools/grouped_s8_u8_q6_surrogate_down.cpp"),
      str(ROOT / "engine/src/gguf_loader.cpp"),
      f"-I{ROOT / 'engine/include'}", "-lOpenCL", "-ldl", "-lpthread",
      "-o", str(q6_tool)], args)
  write_run(raw, "q6-tool-build", q6_build)
  q4_tool = raw / "offline-prepack-generator"
  q4_build = run_env([
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}",
      str(ROOT / "engine/tools/onednn_grouped_q4k_moe_component.cpp"),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(q4_tool),
  ], args)
  write_run(raw, "q4-tool-build", q4_build)

  generation_rows: list[dict[str, Any]] = []
  generation_pass = (build["returncode"] == 0 and
                     q4_build["returncode"] == 0 and
                     q6_build["returncode"] == 0)
  for layer in range(40):
    directory = layers_root / f"layer-{layer:02d}"
    directory.mkdir()
    gateup = rows[f"blk.{layer}.ffn_gate_up_exps.weight"]
    down = rows[f"blk.{layer}.ffn_down_exps.weight"]
    q4_down = down if layer in Q4_LAYERS else dummy_q4
    q4_command = [
        str(q4_tool), "--model", str(args.model),
        "--weight-offset", str(gateup["absolute_offset"]),
        "--weight-bytes", str(gateup["nbytes"]),
        "--down-weight-offset", str(q4_down["absolute_offset"]),
        "--down-weight-bytes", str(q4_down["nbytes"]),
        "--grouped-gateup-binary", str(q4_raw / "gateup.0.bin"),
        "--grouped-down-binary", str(q4_raw / "down.0.bin"),
        "--dump-prepacked-dir", str(directory), "--prepack-only",
    ]
    q4_run = run_env(q4_command, args, {
        "IQ36_GENERATE_S8_GROUPED": "1",
        "DNNL_PRIMITIVE_CACHE_CAPACITY": "0"})
    write_run(raw, f"layer-{layer:02d}-q4-prepack", q4_run)
    (directory / "gateup-scales.bin").unlink(missing_ok=True)
    q6_run: dict[str, Any] | None = None
    if layer in Q6_LAYERS and q4_run["returncode"] == 0:
      q6_command = [
          str(q6_tool), "--model", str(args.model), "--tensor",
          f"blk.{layer}.ffn_down_exps.weight", "--dump-prepacked-dir",
          str(directory), "--prepack-only", "--exact-block-accum",
      ]
      q6_run = run_env(q6_command, args)
      write_run(raw, f"layer-{layer:02d}-q6-prepack", q6_run)
      if q6_run["returncode"] == 0:
        for name in ["down-weights.bin", "down-scales.bin",
                     "down-scale-codes.bin", "down-min-codes.bin",
                     "down-block-ds.bin", "down-dmins.bin"]:
          (directory / name).unlink(missing_ok=True)
    elif q4_run["returncode"] == 0:
      (directory / "down-scales.bin").unlink(missing_ok=True)
    row_pass = q4_run["returncode"] == 0 and (
        layer in Q4_LAYERS or
        (q6_run is not None and q6_run["returncode"] == 0 and
         parse_probe(q6_run).get("prepack_only") is True and
         parse_probe(q6_run).get("exact_per16") is True and
         parse_probe(q6_run).get("exact_block_accum") is True))
    generation_pass = generation_pass and row_pass
    generation_rows.append({
        "layer": layer,
        "codec": ("Q4_K_EXACT_BLOCK" if layer in Q4_LAYERS
                  else "Q6_K_EXACT_BLOCK"),
        "gateup_tensor": gateup["name"], "down_tensor": down["name"],
        "q4_returncode": q4_run["returncode"],
        "q6_returncode": None if q6_run is None else q6_run["returncode"],
        "pass": row_pass,
    })

  payload_manifest: dict[str, Any] = {}
  sizes_pass = True
  unique_gateup: set[str] = set()
  unique_down: set[str] = set()
  total_payload_bytes = 0
  for layer in range(40):
    directory = layers_root / f"layer-{layer:02d}"
    expected = {**GATEUP_SIZES,
                **(Q4_DOWN_SIZES if layer in Q4_LAYERS else Q6_DOWN_SIZES)}
    files: dict[str, Any] = {}
    for name, expected_bytes in expected.items():
      path = directory / name
      size = path.stat().st_size if path.is_file() else -1
      digest = sha256(path) if size == expected_bytes else None
      files[name] = {"bytes": size, "sha256": digest}
      sizes_pass = sizes_pass and size == expected_bytes
      if size == expected_bytes:
        total_payload_bytes += size
    unique_gateup.add(str(files["gateup-weights.bin"]["sha256"]))
    down_name = "down-weights.bin" if layer in Q4_LAYERS \
        else "q6-down-exact-per16-values-u8.bin"
    unique_down.add(str(files[down_name]["sha256"]))
    payload_manifest[str(layer)] = {
        "codec": (
            "Q4_K_EXACT_BLOCK" if layer in Q4_LAYERS
            else "Q6_K_EXACT_BLOCK"),
        "files": files,
    }
  write_json(raw / "payload-manifest.json", payload_manifest)

  smoke_binary = (
      build_dir / "iq36-grouped-mixed-prefill-all-layer-load-smoke")
  smoke_command = [
      str(smoke_binary), str(q4_raw / "gateup.0.bin"),
      str(q4_raw / "down.0.bin"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u8_q6_surrogate_down.cl"),
      str(layers_root),
  ]
  smoke = run_env(smoke_command, args) \
      if generation_pass and sizes_pass else {
          "command": smoke_command, "returncode": 125, "stdout": "",
          "stderr": "payload generation or size check failed",
          "timed_out": False}
  write_run(raw, "all-layer-load", smoke)
  smoke_probe = parse_probe(smoke)

  checks = [
      check("repository_clean_at_gate", state["dirty"] is False),
      check("locked_model_path", args.model.resolve() == MODEL.resolve()),
      check("locked_20_q4_20_q6_census",
            observed_q4 == Q4_LAYERS and observed_q6 == Q6_LAYERS),
      check("all_40_payload_generations_passed", generation_pass,
            failed_layers=[row["layer"] for row in generation_rows
                           if not row["pass"]]),
      check("all_payload_sizes_and_hashes_recorded", sizes_pass,
            total_payload_bytes=total_payload_bytes),
      check("payloads_are_layer_specific",
            len(unique_gateup) == 40 and len(unique_down) == 40,
            unique_gateup_weights=len(unique_gateup),
            unique_down_weights=len(unique_down)),
      check("all_40_layers_load_in_one_native_context",
            smoke["returncode"] == 0 and
            smoke_probe.get("all_layer_load_pass") is True and
            smoke_probe.get("context_create_count") == 1 and
            smoke_probe.get("program_load_count") == 4 and
            smoke_probe.get("layer_count") == 40 and
            smoke_probe.get("resident_weight_bytes") ==
                EXPECTED_RESIDENT_BYTES and
            smoke_probe.get("maps_native_only") is True,
            probe=smoke_probe),
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state,
      "model": str(args.model),
      "q4_generator": str(q4_tool),
      "codec_census": {"q4_layers": observed_q4, "q6_layers": observed_q6},
      "generation": generation_rows,
      "payload_manifest": str(raw / "payload-manifest.json"),
      "total_payload_bytes": total_payload_bytes,
      "resident_probe": smoke_probe, "checks": checks,
      "required_checks_passed": passed,
      "all_40_layer_prepack_ready": passed,
      "disposition": (
          "accept_all_40_mixed_payload_generation_and_resident_load"
          if passed else "reject_all_40_mixed_payload_load"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False})
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out), "git": state,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False})
  (out / "summary.md").write_text("\n".join([
      "# All-layer mixed prepack and resident load", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- generated layers: `{sum(row['pass'] for row in generation_rows)}/40`",
      f"- payload bytes: `{total_payload_bytes}`",
      f"- resident bytes: `{smoke_probe.get('resident_weight_bytes')}`",
      f"- native-only maps: `{smoke_probe.get('maps_native_only')}`", "",
      "This closes payload generation and one-context loading only. Live",
      "preceding-layer state, teacher-forced correctness, tokens, and product",
      "performance remain open.", "",
  ]), encoding="utf-8")
  print(json.dumps({"artifact": str(out), "pass": passed,
                    "resident_probe": smoke_probe,
                    "total_payload_bytes": total_payload_bytes},
                   sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
