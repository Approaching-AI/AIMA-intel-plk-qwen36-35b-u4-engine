#!/usr/bin/env python3
"""Census the 40-layer down codecs and prove capture-free Q4 prepack reuse."""

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
SCHEMA = "intel-qwen36-all-layer-prepack-feasibility-gate-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
TENSOR_INDEX = (
    ROOT / "output/r1-native-gguf-load-map-20260705T071855Z/tensor-index.jsonl")
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CXX = Path("/home/intel/intel-box-env/conda/bin/g++")
ONEDNN_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "oneDNN-01b479323f794da1a7a41a6fc084c7e11ccc2c3b")
ONEDNN_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/onednn-01b479-ocl-grouped")
PREFILL_ARTIFACT = (
    ROOT / "output/grouped-s8-u4-prefill-gate-20260711Tseq665residentcapZ")
Q4_DOWN_LAYERS = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18,
                  20, 21, 23, 24, 26, 27, 29, 30, 32, 33]
Q6_DOWN_LAYERS = [0, 1, 2, 3, 4, 7, 10, 13, 16, 19,
                  22, 25, 28, 31, 34, 35, 36, 37, 38, 39]
PREPACK_SIZES = {
    "gateup-weights.bin": 268_435_456,
    "gateup-scales.bin": 67_108_864,
    "gateup-min-codes.bin": 16_777_216,
    "gateup-dmins.bin": 8_388_608,
    "down-weights.bin": 134_217_728,
    "down-scales.bin": 33_554_432,
    "down-min-codes.bin": 8_388_608,
    "down-dmins.bin": 4_194_304,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--tensor-index", type=Path, default=TENSOR_INDEX)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--onednn-source", type=Path, default=ONEDNN_SOURCE)
  parser.add_argument("--onednn-build", type=Path, default=ONEDNN_BUILD)
  parser.add_argument("--prefill-artifact", type=Path,
                      default=PREFILL_ARTIFACT)
  parser.add_argument("--jobs", type=int, default=16)
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if args.jobs <= 0 or args.timeout_s <= 0:
    parser.error("--jobs and --timeout-s must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/all-layer-prepack-feasibility-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected object")
  return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
          if line.strip()]


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


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


def run_env(command: list[str], env_script: Path, timeout_s: int,
            extra_env: dict[str, str] | None = None) -> dict[str, Any]:
  exports = {"INTEL_FORCE_PROBE": "b080", **(extra_env or {})}
  export_text = " ".join(
      f"{key}={shlex.quote(value)}" for key, value in exports.items())
  shell = (f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
           f"export {export_text} && {shlex.join(command)}")
  return run(["bash", "-lc", shell], timeout_s)


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


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  source = ROOT / "engine/tools/onednn_grouped_q4k_moe_component.cpp"
  base_source = ROOT / "engine/tools/onednn_q4k_bucket_component.cpp"
  gateup_binary = args.prefill_artifact / "raw/gateup.0.bin"
  down_binary = args.prefill_artifact / "raw/down.0.bin"
  accepted_prepack = args.prefill_artifact / "raw/prepacked"
  required = [
      args.model, args.tensor_index, args.env_script, args.cxx,
      args.onednn_source / "include/oneapi/dnnl/dnnl.hpp",
      args.onednn_build / "include/oneapi/dnnl/dnnl_config.h",
      args.onednn_build / "src/libdnnl.so", source, base_source,
      gateup_binary, down_binary, accepted_prepack,
      ROOT / "engine/tools/grouped_s8_u4_prefill_multilayer_load_smoke.cpp",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  created_at = iso_now()
  state = git_state()
  rows = {str(row["name"]): row for row in load_jsonl(args.tensor_index)}
  gateup_rows = [rows[f"blk.{layer}.ffn_gate_up_exps.weight"]
                 for layer in range(40)]
  down_rows = [rows[f"blk.{layer}.ffn_down_exps.weight"]
               for layer in range(40)]
  observed_q4 = [layer for layer, row in enumerate(down_rows)
                 if row.get("type") == 12 and row.get("nbytes") == 150_994_944]
  observed_q6 = [layer for layer, row in enumerate(down_rows)
                 if row.get("type") == 14 and row.get("nbytes") == 220_200_960]

  generator = raw / "offline-prepack-generator"
  build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DCL_TARGET_OPENCL_VERSION=300",
      f"-I{args.onednn_build / 'include'}",
      f"-I{args.onednn_source / 'include'}", str(source),
      f"-L{args.onednn_build / 'src'}",
      f"-Wl,-rpath,{args.onednn_build / 'src'}", "-ldnnl", "-lOpenCL",
      "-o", str(generator),
  ]
  build = run_env(build_command, args.env_script, args.timeout_s)
  write_run(raw, "generator-build", build)
  prepack_results: dict[int, dict[str, Any]] = {}
  prepack_dirs: dict[int, Path] = {}
  for layer in (5, 27):
    directory = raw / f"layer-{layer:02d}"
    prepack_dirs[layer] = directory
    gateup = gateup_rows[layer]
    down = down_rows[layer]
    command = [
        str(generator), "--model", str(args.model),
        "--weight-offset", str(gateup["absolute_offset"]),
        "--weight-bytes", str(gateup["nbytes"]),
        "--down-weight-offset", str(down["absolute_offset"]),
        "--down-weight-bytes", str(down["nbytes"]),
        "--grouped-gateup-binary", str(gateup_binary),
        "--grouped-down-binary", str(down_binary),
        "--dump-prepacked-dir", str(directory), "--prepack-only",
    ]
    result = run_env(command, args.env_script, args.timeout_s, {
        "DNNL_VERBOSE": "0", "DNNL_PRIMITIVE_CACHE_CAPACITY": "0",
        "IQ36_GENERATE_S8_GROUPED": "1",
    }) if build["returncode"] == 0 else {
        "command": command, "returncode": 125, "stdout": "",
        "stderr": "generator build failed", "timed_out": False,
    }
    prepack_results[layer] = result
    write_run(raw, f"prepack-layer-{layer:02d}", result)

  manifests: dict[int, dict[str, Any]] = {}
  all_sizes_pass = True
  for layer, directory in prepack_dirs.items():
    files: dict[str, Any] = {}
    for name, expected_size in PREPACK_SIZES.items():
      path = directory / name
      size = path.stat().st_size if path.is_file() else -1
      files[name] = {"bytes": size,
                     "sha256": sha256(path) if size == expected_size else None}
      all_sizes_pass = all_sizes_pass and size == expected_size
    manifests[layer] = {"files": files,
                        "probe": parse_probe(prepack_results[layer])}
  write_json(raw / "prepack-manifests.json", manifests)
  parity = all(
      manifests[27]["files"][name]["sha256"] == sha256(accepted_prepack / name)
      for name in PREPACK_SIZES)
  layer_specific = (
      manifests[5]["files"]["gateup-weights.bin"]["sha256"] !=
          manifests[27]["files"]["gateup-weights.bin"]["sha256"] and
      manifests[5]["files"]["down-weights.bin"]["sha256"] !=
          manifests[27]["files"]["down-weights.bin"]["sha256"])

  cmake_dir = raw / "build"
  configure = run_env([
      "cmake", "-S", str(ROOT / "engine"), "-B", str(cmake_dir),
      "-DCMAKE_BUILD_TYPE=Release",
  ], args.env_script, args.timeout_s)
  write_run(raw, "configure", configure)
  cmake_build = run_env([
      "cmake", "--build", str(cmake_dir), f"-j{args.jobs}", "--target",
      "iq36-grouped-s8-u4-prefill-multilayer-load-smoke",
  ], args.env_script, args.timeout_s) if configure["returncode"] == 0 else {
      "command": [], "returncode": 125, "stdout": "",
      "stderr": "configure failed", "timed_out": False,
  }
  write_run(raw, "multilayer-build", cmake_build)
  smoke_binary = (
      cmake_dir / "iq36-grouped-s8-u4-prefill-multilayer-load-smoke")
  smoke_command = [
      str(smoke_binary), str(gateup_binary), str(down_binary),
      str(ROOT / "engine/gpu/opencl/grouped_s8_u4_f16_contribution_moe.cl"),
      str(prepack_dirs[5]), str(prepack_dirs[27]),
  ]
  smoke = run_env(smoke_command, args.env_script, args.timeout_s) \
      if cmake_build["returncode"] == 0 else {
          "command": smoke_command, "returncode": 125, "stdout": "",
          "stderr": "multilayer build failed", "timed_out": False,
      }
  write_run(raw, "multilayer-smoke", smoke)
  smoke_probe = parse_probe(smoke)

  gateup_census_pass = all(
      row.get("type") == 12 and row.get("nbytes") == 301_989_888 and
      row.get("dims") == [2048, 1024, 256] for row in gateup_rows)
  prepack_probes_pass = all(
      result["returncode"] == 0 and
      manifests[layer]["probe"].get("prepack_only") is True and
      manifests[layer]["probe"].get("active_experts") == 256 and
      manifests[layer]["probe"].get("max_group_size") == 32
      for layer, result in prepack_results.items())
  checks = [
      check("repository_clean_at_gate", state["dirty"] is False),
      check("locked_model_path", args.model.resolve() == MODEL.resolve()),
      check("all_40_gateup_tensors_are_locked_q4k", gateup_census_pass),
      check("down_codec_census_is_20_q4k_20_q6k",
            observed_q4 == Q4_DOWN_LAYERS and observed_q6 == Q6_DOWN_LAYERS,
            q4_layers=observed_q4, q6_layers=observed_q6),
      check("capture_free_prepack_generator_build_passed",
            build["returncode"] == 0),
      check("two_real_q4_layers_prepacked_without_capture_or_oracle",
            prepack_probes_pass and all_sizes_pass),
      check("prepack_only_layer27_is_byte_identical_to_accepted_payload",
            parity),
      check("prepack_payloads_are_layer_specific", layer_specific),
      check("two_layers_load_in_one_resident_runtime",
            configure["returncode"] == 0 and cmake_build["returncode"] == 0 and
            smoke["returncode"] == 0 and
            smoke_probe.get("multilayer_load_pass") is True and
            smoke_probe.get("context_create_count") == 1 and
            smoke_probe.get("program_load_count") == 3 and
            smoke_probe.get("layer_count") == 2 and
            smoke_probe.get("resident_weight_bytes") == 1_082_130_432 and
            smoke_probe.get("maps_native_only") is True),
      check("q6_down_requires_distinct_grouped_prefill_carrier",
            len(observed_q6) == 20 and all(
                down_rows[layer]["nbytes"] != 150_994_944
                for layer in observed_q6)),
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at, "git": state,
      "down_codec_census": {"q4_k_layers": observed_q4,
                             "q6_k_layers": observed_q6},
      "q4_prepack": {"layer_count": 20,
                     "projected_resident_bytes": 20 * 541_065_216,
                     "sampled_layers": [5, 27], "manifests": manifests},
      "q6_prefill_gap": {"layer_count": 20,
                         "raw_weight_bytes": 20 * 220_200_960,
                         "required_route": "grouped_exact_q6_prefill_carrier"},
      "multilayer_probe": smoke_probe, "checks": checks,
      "required_checks_passed": passed,
      "all_40_layer_prepack_ready": False,
      "disposition": (
          "accept_q4_multilayer_prepack_select_grouped_exact_q6_prefill"
          if passed else "reject_all_layer_prepack_feasibility"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA, "checks": checks,
      "required_checks_passed": passed, "speedup_claims_allowed": False})
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out), "git": state,
      "required_checks_passed": passed, "speedup_claims_allowed": False})
  (out / "summary.md").write_text("\n".join([
      "# All-layer prepack feasibility", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- Q4_K down layers: `{len(observed_q4)}`",
      f"- Q6_K down layers: `{len(observed_q6)}`",
      "- capture-free Q4 prepack parity: " + f"`{str(parity).lower()}`",
      "- two-layer resident load: " +
      f"`{str(smoke_probe.get('multilayer_load_pass')).lower()}`", "",
      "The Q4 half is ready for all-layer generation. The Q6 half requires a",
      "distinct grouped exact-Q6 prefill carrier before live 40-layer work.", "",
  ]), encoding="utf-8")
  print(json.dumps({"out_dir": str(out), "q4_layer_count": len(observed_q4),
                    "q6_layer_count": len(observed_q6),
                    "required_checks_passed": passed}, sort_keys=True))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
