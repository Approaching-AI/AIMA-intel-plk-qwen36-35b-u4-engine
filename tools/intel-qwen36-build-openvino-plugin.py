#!/usr/bin/env python3
"""Reconstruct and build the exact promoted IQ36 OpenVINO GPU source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


OPENVINO_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
ONEDNN_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
OPENVINO_SUBMODULE = "src/plugins/intel_gpu/thirdparty/onednn_gpu"
OPENVINO_DIR_NAME = "openvino-90214e5be05"
BUILD_DIR_NAME = "openvino-90214e-l0-gpu"
OUTPUT_DIR_NAME = "openvino-90214e-l0-gpu-seq2109"
CI_BUILD_NUMBER = "2026.2.0-106-90214e5be05"
PROMOTED_RPATH = (
    "/home/intel/intel-qwen36-r0/output/"
    "openvino-90214e-l0-gpu-seq2067/bin/intel64/Release;"
    "/home/intel/intel-box-env/conda/lib")
SHORT_PLUGIN_SHA256 = (
    "b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12")
LONG_PLUGIN_SHA256 = (
    "c0515a401f579620c2fb440031e87e848ceaefab572715d4ace2b76ff2956121")
OPENVINO_PATCH_SHA256 = (
    "e722ba5f225273c8090c8610cd3fb80d2ffb0d2472d5e88511c4ad18ed046e9f")
ONEDNN_PATCH_SHA256 = (
    "090a385c0fc4f4384d34f79ee187bce7b8933df36ae03dc735f963f5ba716fd9")
LONG_PROFILE_PATCH_SHA256 = (
    "017f5eb4925c993db3e084ffa1bd420a44ea8dc66db8765a9ffa47ad3722b2ed")
SHORT_SOURCE_STATE_SHA256 = (
    "0776ca91cd9359a200f1e9a51afaeca83c2e9a9c5952dc2e552839ef12085743")
LONG_SOURCE_STATE_SHA256 = (
    "77153ecf9ed7fde3ae32efc78f08b0b0afeb9fe8802d83e34885ca9a292bd067")
LM_HEAD_PATH = (
    "src/plugins/intel_gpu/src/graph/impls/ocl/iq36_lm_head_i8q4.cpp")
FC_HORIZONTAL_PATH = (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
TRANSFORMATIONS_PIPELINE_PATH = (
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
LONG_HISTORY_COMMIT = "4b80f65e822f809d55a2d414a90fb7f483a7397c"

PROFILE_DELTA_FILES = {
    LM_HEAD_PATH: {
        "short": (133461,
                  "48795e95fc25f9a74941b6994ff545a6a8d45aaa9811682b8edc7ae3b3b6fcdd"),
        "long": (106546,
                 "81be0135a12f6d94b87d5ef3ad9e72bf2dca243f98e4ab9c376a51b3a28d51a4"),
    },
    FC_HORIZONTAL_PATH: {
        "short": (23774,
                  "1944c1af859c2ccd416a481da8d0bd336bbe39ad9a4bca0aed9ea56182b7996f"),
        "long": (22889,
                 "4a32d9c17d84390aef343bd60c992859fc75bc72d2f8ddff3a355c5276ba6020"),
    },
    TRANSFORMATIONS_PIPELINE_PATH: {
        "short": (92247,
                  "dd9d4c2eec7b9ba5d9bf889ac916f2b4c90e6922401524657d09b3f81892ff38"),
        "long": (91812,
                 "abbe70c6ed19abce6e6ae7ee586072436b9e3efdd8aaed3bdd3adeec09d73055"),
    },
}
PROFILE_DELTA_STATUS = {
    LM_HEAD_PATH: {"short": "??", "long": "??"},
    FC_HORIZONTAL_PATH: {"short": " M", "long": " M"},
    TRANSFORMATIONS_PIPELINE_PATH: {"short": " M", "long": None},
}

PROFILE_PLUGIN_SHA256 = {
    "short": SHORT_PLUGIN_SHA256,
    "long": LONG_PLUGIN_SHA256,
}
PROFILE_SOURCE_STATE_SHA256 = {
    "short": SHORT_SOURCE_STATE_SHA256,
    "long": LONG_SOURCE_STATE_SHA256,
}

CMAKE_DEFINITIONS = (
    "CMAKE_BUILD_TYPE=Release",
    "CMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    f"CMAKE_INSTALL_RPATH={PROMOTED_RPATH}",
    "CMAKE_COMPILE_WARNING_AS_ERROR=OFF",
    "ENABLE_LTO=OFF",
    "USE_BUILD_TYPE_SUBFOLDER=ON",
    "BUILD_SHARED_LIBS=ON",
    "ENABLE_LIBRARY_VERSIONING=ON",
    "ENABLE_FASTER_BUILD=OFF",
    "ENABLE_CLANG_TIDY=OFF",
    "ENABLE_UNSAFE_LOCATIONS=OFF",
    "ENABLE_PROXY=ON",
    "ENABLE_INTEL_CPU=OFF",
    "ENABLE_INTEL_GPU=ON",
    "GPU_RT_TYPE=L0",
    "ENABLE_ONEDNN_FOR_GPU=ON",
    "ENABLE_INTEL_NPU=OFF",
    "ENABLE_DEBUG_CAPS=OFF",
    "ENABLE_TESTS=OFF",
    "ENABLE_PROFILING_ITT=OFF",
    "ENABLE_PROFILING_FILTER=ALL",
    "ENABLE_PROFILING_FIRST_INFERENCE=ON",
    "SELECTIVE_BUILD=OFF",
    "ENABLE_DOCS=OFF",
    "ENABLE_PKGCONFIG_GEN=ON",
    "THREADING=TBB_ADAPTIVE",
    "ENABLE_MULTI=OFF",
    "ENABLE_AUTO=OFF",
    "ENABLE_AUTO_BATCH=OFF",
    "ENABLE_HETERO=OFF",
    "ENABLE_TEMPLATE=OFF",
    "ENABLE_PLUGINS_XML=OFF",
    "ENABLE_SAMPLES=OFF",
    "ENABLE_OV_ONNX_FRONTEND=OFF",
    "ENABLE_OV_PADDLE_FRONTEND=OFF",
    "ENABLE_OV_IR_FRONTEND=ON",
    "ENABLE_OV_PYTORCH_FRONTEND=OFF",
    "ENABLE_OV_JAX_FRONTEND=OFF",
    "ENABLE_OV_TF_FRONTEND=OFF",
    "ENABLE_OV_TF_LITE_FRONTEND=OFF",
    "ENABLE_SYSTEM_TBB=OFF",
    "ENABLE_SYSTEM_PUGIXML=OFF",
    "ENABLE_SYSTEM_OPENCL=OFF",
    "ENABLE_SYSTEM_LEVEL_ZERO=ON",
    "ENABLE_JS=OFF",
    "ENABLE_OPENVINO_DEBUG=OFF",
    "ENABLE_PYTHON=OFF",
    "ENABLE_PYTHON_API=OFF",
)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def run(
    command: list[str], *, cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    expected: tuple[int, ...] = (0,), capture: bool = False,
) -> subprocess.CompletedProcess[str]:
  result = subprocess.run(
      command, cwd=cwd, env=environment, check=False, text=True,
      stdout=subprocess.PIPE if capture else None,
      stderr=subprocess.PIPE if capture else None)
  if result.returncode not in expected:
    detail = ""
    if capture:
      detail = (result.stderr or result.stdout or "").strip()
    raise ValueError(
        f"command failed ({result.returncode}): {' '.join(command)}"
        + (f": {detail}" if detail else ""))
  return result


def git_bytes(
    root: Path, command: list[str], expected: tuple[int, ...] = (0,),
) -> bytes:
  result = subprocess.run(
      ["git", *command], cwd=root, check=False,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if result.returncode not in expected:
    raise ValueError(
        f"git {' '.join(command)} failed in {root}: "
        f"{result.stderr.decode('utf-8', errors='replace').strip()}")
  return result.stdout


def source_rows(
    root: Path, *, excluded: tuple[str, ...] = (),
) -> list[dict[str, object]]:
  raw = git_bytes(
      root, ["status", "--porcelain=v1", "-z", "--untracked-files=all"])
  rows: list[dict[str, object]] = []
  for item in raw.split(b"\0"):
    if not item:
      continue
    status = item[:2].decode("ascii")
    relative = item[3:].decode("utf-8", errors="surrogateescape")
    if relative in excluded:
      continue
    path = root / relative
    rows.append({
        "path": relative,
        "status": status,
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    })
  return sorted(rows, key=lambda row: str(row["path"]))


def profile_metadata() -> dict[str, object]:
  return {
      "short": {
          "source_state_sha256": SHORT_SOURCE_STATE_SHA256,
      },
      "long": {
          "source_state_sha256": LONG_SOURCE_STATE_SHA256,
          "base_profile": "short",
          "delta_path": "source/long-profile-from-short.patch",
          "delta_sha256": LONG_PROFILE_PATCH_SHA256,
          "historical_route_checkpoint": LONG_HISTORY_COMMIT,
          "changed_files": [
              {"path": path,
               "status": PROFILE_DELTA_STATUS[path]["long"] or "clean",
               "bytes": states["long"][0], "sha256": states["long"][1]}
              for path, states in PROFILE_DELTA_FILES.items()
          ],
      },
  }


def verify_bundle(
    bundle: Path,
) -> tuple[dict[str, object], Path, Path, Path]:
  source_dir = bundle / "source"
  state_path = source_dir / "source-state.json"
  openvino_patch = source_dir / "openvino-promoted-working-tree.patch"
  onednn_patch = source_dir / "onednn-promoted-working-tree.patch"
  long_patch = source_dir / "long-profile-from-short.patch"
  for path in (state_path, openvino_patch, onednn_patch, long_patch):
    if not path.is_file():
      raise ValueError(f"runtime bundle source asset is missing: {path}")
  if sha256_file(openvino_patch) != OPENVINO_PATCH_SHA256:
    raise ValueError("OpenVINO patch SHA-256 does not match the promoted source")
  if sha256_file(onednn_patch) != ONEDNN_PATCH_SHA256:
    raise ValueError("oneDNN patch SHA-256 does not match the promoted source")
  if sha256_file(long_patch) != LONG_PROFILE_PATCH_SHA256:
    raise ValueError(
        "long-profile patch SHA-256 does not match the promoted source")
  state = json.loads(state_path.read_text(encoding="utf-8"))
  if state.get("openvino_commit") != OPENVINO_COMMIT:
    raise ValueError("source-state OpenVINO commit is not the promoted commit")
  if state.get("onednn_commit") != ONEDNN_COMMIT:
    raise ValueError("source-state oneDNN commit is not the promoted commit")
  if state.get("openvino_patch_sha256") != OPENVINO_PATCH_SHA256:
    raise ValueError("source-state OpenVINO patch identity is inconsistent")
  if state.get("onednn_patch_sha256") != ONEDNN_PATCH_SHA256:
    raise ValueError("source-state oneDNN patch identity is inconsistent")
  canonical = {
      "openvino_commit": state["openvino_commit"],
      "onednn_commit": state["onednn_commit"],
      "openvino_files": state["openvino_files"],
      "onednn_files": state["onednn_files"],
  }
  encoded = (
      json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  actual_state = hashlib.sha256(encoded).hexdigest()
  if actual_state != SHORT_SOURCE_STATE_SHA256:
    raise ValueError(
        f"source-state fingerprint mismatch: {actual_state}")
  if state.get("source_state_sha256") != SHORT_SOURCE_STATE_SHA256:
    raise ValueError("source-state self-declared fingerprint is inconsistent")
  if state.get("profiles") != profile_metadata():
    raise ValueError("source-state profile metadata is inconsistent")
  return state, openvino_patch, onednn_patch, long_patch


def verify_commits(source: Path) -> Path:
  onednn = source / OPENVINO_SUBMODULE
  if not (source / ".git").exists():
    raise ValueError(f"OpenVINO checkout is not a Git worktree: {source}")
  if not onednn.is_dir():
    raise ValueError(
        "oneDNN GPU submodule is missing; run git submodule update "
        "--init --recursive")
  openvino_head = git_bytes(source, ["rev-parse", "HEAD"]).decode().strip()
  onednn_head = git_bytes(onednn, ["rev-parse", "HEAD"]).decode().strip()
  if openvino_head != OPENVINO_COMMIT or onednn_head != ONEDNN_COMMIT:
    raise ValueError(
        "source commit mismatch: expected "
        f"{OPENVINO_COMMIT}/{ONEDNN_COMMIT}, got "
        f"{openvino_head}/{onednn_head}")
  return onednn


def expected_openvino_rows(
    state: dict[str, object], profile: str,
) -> list[dict[str, object]]:
  rows = [dict(row) for row in state["openvino_files"]]
  if profile == "long":
    matches: set[str] = set()
    selected: list[dict[str, object]] = []
    for row in rows:
      path = str(row["path"])
      if path in PROFILE_DELTA_FILES:
        status = PROFILE_DELTA_STATUS[path]["long"]
        if status is None:
          matches.add(path)
          continue
        row["bytes"], row["sha256"] = PROFILE_DELTA_FILES[path]["long"]
        row["status"] = status
        matches.add(path)
      selected.append(row)
    if matches != set(PROFILE_DELTA_FILES):
      raise ValueError(
          "short source-state does not contain every long-profile delta row")
    rows = selected
  canonical = {
      "openvino_commit": state["openvino_commit"],
      "onednn_commit": state["onednn_commit"],
      "openvino_files": rows,
      "onednn_files": state["onednn_files"],
  }
  encoded = (
      json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  actual = hashlib.sha256(encoded).hexdigest()
  expected = PROFILE_SOURCE_STATE_SHA256[profile]
  if actual != expected:
    raise ValueError(
        f"derived {profile} source-state mismatch: expected {expected}, "
        f"got {actual}")
  return rows


def profile_delta_files_match(source: Path, profile: str) -> bool:
  for path, states in PROFILE_DELTA_FILES.items():
    candidate = source / path
    expected_bytes, expected_sha256 = states[profile]
    if (not candidate.is_file()
        or candidate.stat().st_size != expected_bytes
        or sha256_file(candidate) != expected_sha256):
      return False
  return True


def postimage_matches(
    source: Path, onednn: Path, state: dict[str, object], profile: str,
) -> bool:
  return (
      source_rows(source, excluded=(OPENVINO_SUBMODULE,))
      == expected_openvino_rows(state, profile)
      and source_rows(onednn) == state["onednn_files"]
      and profile_delta_files_match(source, profile))


def require_clean(source: Path, onednn: Path) -> None:
  outer = git_bytes(source, [
      "status", "--porcelain=v1", "-z", "--untracked-files=all",
      "--ignore-submodules=dirty",
  ])
  inner = git_bytes(onednn, [
      "status", "--porcelain=v1", "-z", "--untracked-files=all",
  ])
  if outer or inner:
    raise ValueError(
        "source checkout has unrelated changes and does not match the "
        "promoted postimage; refusing to apply patches")


def prepare_source(
    source: Path, state: dict[str, object],
    openvino_patch: Path, onednn_patch: Path, long_patch: Path,
    profile: str,
) -> None:
  onednn = verify_commits(source)
  if postimage_matches(source, onednn, state, profile):
    return
  if profile == "long" and postimage_matches(source, onednn, state, "short"):
    git_bytes(source, [
        "apply", "--check", "--whitespace=nowarn", str(long_patch)])
    git_bytes(source, [
        "apply", "--whitespace=nowarn", str(long_patch)])
    if not postimage_matches(source, onednn, state, profile):
      raise ValueError(
          "long-profile delta did not produce the promoted postimage")
    return
  require_clean(source, onednn)
  git_bytes(source, [
      "apply", "--check", "--whitespace=nowarn", str(openvino_patch)])
  git_bytes(onednn, [
      "apply", "--check", "--whitespace=nowarn", str(onednn_patch)])
  git_bytes(source, [
      "apply", "--whitespace=nowarn", str(openvino_patch)])
  git_bytes(onednn, [
      "apply", "--whitespace=nowarn", str(onednn_patch)])
  if not postimage_matches(source, onednn, state, "short"):
    raise ValueError(
        "patched checkout does not match the promoted short postimage")
  if profile == "long":
    git_bytes(source, [
        "apply", "--check", "--whitespace=nowarn", str(long_patch)])
    git_bytes(source, [
        "apply", "--whitespace=nowarn", str(long_patch)])
  if not postimage_matches(source, onednn, state, profile):
    raise ValueError(
        f"patched checkout does not match the promoted {profile} postimage")


def first_line(command: list[str], environment: dict[str, str]) -> str:
  result = run(command, environment=environment, capture=True)
  return result.stdout.splitlines()[0].strip()


def verify_toolchain(prefix: Path) -> tuple[dict[str, str], dict[str, str]]:
  binary = prefix / "bin"
  tools = {
      "cc": binary / "gcc",
      "cxx": binary / "g++",
      "cmake": binary / "cmake",
      "ninja": binary / "ninja",
      "ld": binary / "ld",
      "ar": binary / "ar",
      "python": binary / "python3.12",
      "pkg_config": Path("/usr/bin/pkg-config"),
  }
  for name, path in tools.items():
    if not path.is_file():
      raise ValueError(f"required {name} executable is missing: {path}")
  if not Path("/usr/include/python3.12").is_dir():
    raise ValueError("required system Python 3.12 headers are missing")
  environment = dict(os.environ)
  environment["PATH"] = str(binary) + os.pathsep + environment.get("PATH", "")
  environment["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
  observed = {
      "gcc": first_line([str(tools["cc"]), "--version"], environment),
      "g++": first_line([str(tools["cxx"]), "--version"], environment),
      "cmake": first_line([str(tools["cmake"]), "--version"], environment),
      "ninja": first_line([str(tools["ninja"]), "--version"], environment),
      "ld": first_line([str(tools["ld"]), "--version"], environment),
      "ar": first_line([str(tools["ar"]), "--version"], environment),
      "python": first_line([str(tools["python"]), "--version"], environment),
      "pkg_config": first_line(
          [str(tools["pkg_config"]), "--version"], environment),
      "level_zero": first_line(
          [str(tools["pkg_config"]), "--modversion", "level-zero"],
          environment),
      "libc": " ".join(platform.libc_ver()),
  }
  expected = {
      "gcc": "gcc (conda-forge gcc 14.3.0-19) 14.3.0",
      "g++": "g++ (conda-forge gcc 14.3.0-19) 14.3.0",
      "cmake": "cmake version 4.3.3",
      "ninja": "1.13.2",
      "ld": "GNU ld (GNU Binutils) 2.45.1",
      "ar": "GNU ar (GNU Binutils) 2.45.1",
      "python": "Python 3.12.13",
      "pkg_config": "1.8.1",
      "level_zero": "1.29.0",
      "libc": "glibc 2.39",
  }
  drift = {
      name: {"expected": expected[name], "actual": actual}
      for name, actual in observed.items() if actual != expected[name]
  }
  if drift:
    raise ValueError(
        "build toolchain identity mismatch: "
        + json.dumps(drift, sort_keys=True))
  return ({name: str(path) for name, path in tools.items()}, observed)


def logical_source(source: Path, work_root: Path) -> Path:
  target = work_root / "source" / OPENVINO_DIR_NAME
  target.parent.mkdir(parents=True, exist_ok=True)
  if target.exists() or target.is_symlink():
    if target.resolve() != source.resolve():
      raise ValueError(f"logical source path points elsewhere: {target}")
  elif source.resolve() == target.resolve():
    pass
  else:
    target.symlink_to(source.resolve(), target_is_directory=True)
  return target


def configure_and_build(
    source: Path, work_root: Path, prefix: Path,
    tools: dict[str, str], environment: dict[str, str],
    *, parallel: int, resume: bool,
) -> Path:
  build_dir = work_root / "build" / BUILD_DIR_NAME
  output_root = work_root / "output" / OUTPUT_DIR_NAME
  if build_dir.exists() and not resume:
    raise ValueError(
        f"build directory already exists; use --resume or a new root: {build_dir}")
  build_dir.parent.mkdir(parents=True, exist_ok=True)
  output_root.parent.mkdir(parents=True, exist_ok=True)
  command = [
      tools["cmake"], "-S", str(source), "-B", str(build_dir), "-G", "Ninja",
      f"-DCMAKE_C_COMPILER={tools['cc']}",
      f"-DCMAKE_CXX_COMPILER={tools['cxx']}",
      f"-DCMAKE_MAKE_PROGRAM={tools['ninja']}",
      "-DCMAKE_RANLIB:INTERNAL=:",
      f"-DCMAKE_PREFIX_PATH={prefix}",
      f"-DPKG_CONFIG_EXECUTABLE={tools['pkg_config']}",
      "-DPython3_INCLUDE_DIR=/usr/include/python3.12",
      f"-DOUTPUT_ROOT={output_root}",
      *(f"-D{value}" for value in CMAKE_DEFINITIONS),
  ]
  build_environment = dict(environment)
  build_environment["CI_BUILD_NUMBER"] = CI_BUILD_NUMBER
  run(command, environment=build_environment)
  run([
      tools["cmake"], "--build", str(build_dir),
      "--target", "openvino_intel_gpu_plugin",
      "--parallel", str(parallel),
  ], environment=build_environment)
  plugin = output_root / "bin/intel64/Release/libopenvino_intel_gpu_plugin.so"
  if not plugin.is_file():
    raise ValueError(f"build completed without the expected plugin: {plugin}")
  return plugin


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--bundle", type=Path, required=True)
  parser.add_argument("--work-root", type=Path)
  parser.add_argument("--toolchain-prefix", type=Path)
  parser.add_argument("--parallel", type=int, default=1)
  parser.add_argument(
      "--profile", choices=tuple(PROFILE_PLUGIN_SHA256), default="short",
      help="reconstruct the promoted short or long plugin source profile")
  parser.add_argument(
      "--resume", action="store_true",
      help="reuse an interrupted build directory after revalidating inputs")
  parser.add_argument(
      "--prepare-only", action="store_true",
      help="apply and verify the exact source postimage without compiling")
  parser.add_argument(
      "--result", type=Path,
      help="optional JSON result path (written only after all checks)")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if args.parallel < 1:
    raise SystemExit("error: --parallel must be at least 1")
  try:
    if not args.prepare_only and (
        args.work_root is None or args.toolchain_prefix is None):
      raise ValueError(
          "--work-root and --toolchain-prefix are required unless "
          "--prepare-only is used")
    source = args.source.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    state, openvino_patch, onednn_patch, long_patch = verify_bundle(bundle)
    prepare_source(
        source, state, openvino_patch, onednn_patch, long_patch, args.profile)
    result: dict[str, object] = {
        "profile": args.profile,
        "source_postimage_verified": True,
        "source_state_sha256": PROFILE_SOURCE_STATE_SHA256[args.profile],
        "base_source_state_sha256": SHORT_SOURCE_STATE_SHA256,
        "openvino_commit": OPENVINO_COMMIT,
        "onednn_commit": ONEDNN_COMMIT,
        "plugin_ci_build_identity": CI_BUILD_NUMBER,
    }
    mismatch = False
    if not args.prepare_only:
      assert args.work_root is not None
      assert args.toolchain_prefix is not None
      work_root = args.work_root.expanduser().resolve()
      prefix = args.toolchain_prefix.expanduser().resolve()
      tools, versions = verify_toolchain(prefix)
      environment = dict(os.environ)
      environment["PATH"] = (
          str(prefix / "bin") + os.pathsep + environment.get("PATH", ""))
      environment["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
      checkout = logical_source(source, work_root)
      plugin = configure_and_build(
          checkout, work_root, prefix, tools, environment,
          parallel=args.parallel, resume=args.resume)
      actual = sha256_file(plugin)
      expected_plugin = PROFILE_PLUGIN_SHA256[args.profile]
      result.update({
          "toolchain": versions,
          "build_parallel": args.parallel,
          "plugin": str(plugin),
          "plugin_bytes": plugin.stat().st_size,
          "plugin_sha256": actual,
          "promoted_plugin_sha256": expected_plugin,
          "bit_identical_to_promoted": actual == expected_plugin,
      })
      if actual != expected_plugin:
        mismatch = True
        result["validation_error"] = (
            "rebuilt plugin is not bit-identical to the promoted carrier; "
            f"expected {expected_plugin}, got {actual}. Do not deploy "
            "it without rerunning the full correctness/performance gate")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.result is not None:
      target = args.result.expanduser().resolve()
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
  except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(f"error: {error}") from error
  return 2 if mismatch else 0


if __name__ == "__main__":
  raise SystemExit(main())
