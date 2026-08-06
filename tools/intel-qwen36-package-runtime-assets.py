#!/usr/bin/env python3
"""Build a checksum-verified runtime asset bundle for the IQ36 service."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
R0_ROOT = ROOT.parent / "intel-qwen36-r0"
SHORT_PLUGIN_SHA256 = (
    "b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12")
LONG_PLUGIN_SHA256 = (
    "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269")
CUSTOM_CONFIG_SHA256 = (
    "bd7a679031bbde2fa2626f2138bf79a5626469ccbc041faadef3b12e811200ad")
OPENVINO_COMMIT = "90214e5be052438cec5617ed3ea7e37df1538f68"
ONEDNN_COMMIT = "20db47e2d3c4df1b66e93bed2e97d30da175512d"
OPENVINO_SUBMODULE = "src/plugins/intel_gpu/thirdparty/onednn_gpu"
SOURCE_STATE_SHA256 = (
    "0776ca91cd9359a200f1e9a51afaeca83c2e9a9c5952dc2e552839ef12085743")
LONG_SOURCE_STATE_SHA256 = (
    "b947c32eede6bb7f11429722503771706be6847cecf9905d58e9b0463f32817a")
OPENVINO_PATCH_SHA256 = (
    "e722ba5f225273c8090c8610cd3fb80d2ffb0d2472d5e88511c4ad18ed046e9f")
ONEDNN_PATCH_SHA256 = (
    "090a385c0fc4f4384d34f79ee187bce7b8933df36ae03dc735f963f5ba716fd9")
LONG_PROFILE_PATCH_SHA256 = (
    "a38003733c79aafd062b59771dba57a784eb569bc25683206a46ac38048bef3e")
LM_HEAD_PATH = (
    "src/plugins/intel_gpu/src/graph/impls/ocl/iq36_lm_head_i8q4.cpp")
FC_HORIZONTAL_PATH = (
    "src/plugins/intel_gpu/src/plugin/transformations/"
    "fc_horizontal_fusion.cpp")
TRANSFORMATIONS_PIPELINE_PATH = (
    "src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp")
LONG_HISTORY_COMMIT = "4b80f65e822f809d55a2d414a90fb7f483a7397c"
CI_BUILD_NUMBER = "2026.2.0-106-90214e5be05"

PROFILE_DELTA_FILES = {
    LM_HEAD_PATH: {
        "short": (133461,
                  "48795e95fc25f9a74941b6994ff545a6a8d45aaa9811682b8edc7ae3b3b6fcdd"),
        "long": (105805,
                 "8373143a711ee75ff8eb913a1e04b89a270d5b419a5196b863911626de8e45e9"),
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


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def verify_file(label: str, path: Path, expected: str) -> None:
  if not path.is_file():
    raise ValueError(f"{label} is not a regular file: {path}")
  actual = sha256_file(path)
  if actual != expected:
    raise ValueError(
        f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


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


def profile_metadata(state: dict[str, object]) -> dict[str, object]:
  short_rows = state["openvino_files"]
  matching = {
      str(row["path"]): row for row in short_rows
      if str(row["path"]) in PROFILE_DELTA_FILES
  }
  expected_matching = {
      path: {
          "path": path,
          "status": PROFILE_DELTA_STATUS[path]["short"],
          "bytes": states["short"][0],
          "sha256": states["short"][1],
      }
      for path, states in PROFILE_DELTA_FILES.items()
  }
  if matching != expected_matching:
    raise ValueError("promoted short profile-delta source identity drift")
  long_rows = [dict(row) for row in short_rows]
  selected = []
  for row in long_rows:
    path = str(row["path"])
    if path in PROFILE_DELTA_FILES:
      status = PROFILE_DELTA_STATUS[path]["long"]
      if status is None:
        continue
      row["bytes"], row["sha256"] = PROFILE_DELTA_FILES[path]["long"]
      row["status"] = status
    selected.append(row)
  long_rows = selected
  canonical = {
      "openvino_commit": state["openvino_commit"],
      "onednn_commit": state["onednn_commit"],
      "openvino_files": long_rows,
      "onednn_files": state["onednn_files"],
  }
  encoded = (
      json.dumps(canonical, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  actual = hashlib.sha256(encoded).hexdigest()
  if actual != LONG_SOURCE_STATE_SHA256:
    raise ValueError(
        "derived long source-state drift: expected "
        f"{LONG_SOURCE_STATE_SHA256}, got {actual}")
  return {
      "short": {
          "source_state_sha256": SOURCE_STATE_SHA256,
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


def source_snapshot(openvino: Path) -> tuple[dict[str, object], bytes, bytes]:
  onednn = openvino / OPENVINO_SUBMODULE
  if not onednn.is_dir():
    raise ValueError(f"OpenVINO oneDNN GPU submodule is missing: {onednn}")
  openvino_head = git_bytes(openvino, ["rev-parse", "HEAD"]).decode().strip()
  onednn_head = git_bytes(onednn, ["rev-parse", "HEAD"]).decode().strip()
  state = {
      "openvino_commit": openvino_head,
      "onednn_commit": onednn_head,
      "openvino_files": source_rows(
          openvino, excluded=(OPENVINO_SUBMODULE,)),
      "onednn_files": source_rows(onednn),
  }
  state_bytes = (
      json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  state_sha256 = hashlib.sha256(state_bytes).hexdigest()
  if openvino_head != OPENVINO_COMMIT or onednn_head != ONEDNN_COMMIT:
    raise ValueError(
        "OpenVINO source commit drift: expected "
        f"{OPENVINO_COMMIT}/{ONEDNN_COMMIT}, got "
        f"{openvino_head}/{onednn_head}")
  if state_sha256 != SOURCE_STATE_SHA256:
    raise ValueError(
        "promoted OpenVINO source-state drift: expected "
        f"{SOURCE_STATE_SHA256}, got {state_sha256}")

  tracked = git_bytes(openvino, [
      "diff", "--binary", "--no-ext-diff", "--", ".",
      f":(exclude){OPENVINO_SUBMODULE}",
  ])
  untracked = [
      str(row["path"]) for row in state["openvino_files"]
      if row["status"] == "??"
  ]
  parts = [tracked]
  for relative in sorted(untracked):
    parts.append(git_bytes(
        openvino,
        ["diff", "--no-index", "--binary", "--", "/dev/null", relative],
        expected=(0, 1)))
  openvino_patch = b"".join(parts)
  onednn_patch = git_bytes(onednn, ["diff", "--binary", "--no-ext-diff"])
  actual_openvino_patch = hashlib.sha256(openvino_patch).hexdigest()
  actual_onednn_patch = hashlib.sha256(onednn_patch).hexdigest()
  if actual_openvino_patch != OPENVINO_PATCH_SHA256:
    raise ValueError(
        "promoted OpenVINO patch drift: expected "
        f"{OPENVINO_PATCH_SHA256}, got {actual_openvino_patch}")
  if actual_onednn_patch != ONEDNN_PATCH_SHA256:
    raise ValueError(
        "promoted oneDNN patch drift: expected "
        f"{ONEDNN_PATCH_SHA256}, got {actual_onednn_patch}")
  state["source_state_sha256"] = state_sha256
  state["openvino_patch_sha256"] = actual_openvino_patch
  state["onednn_patch_sha256"] = actual_onednn_patch
  state["profiles"] = profile_metadata(state)
  return state, openvino_patch, onednn_patch


def custom_sources(config: Path) -> tuple[Path, ...]:
  try:
    # OpenVINO CONFIG_FILE is an XML fragment containing multiple sibling
    # CustomLayer elements, not a conventional single-root XML document.
    text = config.read_text(encoding="utf-8")
    root = ElementTree.fromstring(
        "<IQ36CustomLayers>" + text + "</IQ36CustomLayers>")
  except ElementTree.ParseError as error:
    raise ValueError(f"invalid custom CONFIG_FILE XML: {error}") from error
  raw_names = {node.get("filename") for node in root.iter("Source")}
  if not raw_names or None in raw_names:
    raise ValueError("custom CONFIG_FILE has no complete Source inventory")
  names = sorted(str(name) for name in raw_names)
  sources: list[Path] = []
  config_root = config.parent.resolve()
  for name in names:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError(f"unsafe Source filename in CONFIG_FILE: {name}")
    source = (config_root / relative).resolve()
    if source.parent != config_root or not source.is_file():
      raise ValueError(f"referenced custom source is missing or unsafe: {name}")
    sources.append(source)
  return tuple(sources)


def copy_file(source: Path, target: Path) -> dict[str, object]:
  target.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, target)
  return {
      "path": target.as_posix(),
      "sha256": sha256_file(target),
      "bytes": target.stat().st_size,
  }


def dynamic_dependencies(path: Path) -> list[dict[str, object]]:
  executable = shutil.which("ldd")
  if executable is None:
    raise ValueError("ldd is required to preflight plugin dependencies")
  result = subprocess.run(
      [executable, str(path)], check=False, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if result.returncode != 0:
    raise ValueError(
        f"cannot inspect dynamic dependencies for {path.name}: "
        f"{result.stdout.strip()}")
  rows = []
  missing = []
  for raw_line in result.stdout.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if "=>" in line:
      name, target = (part.strip() for part in line.split("=>", 1))
      target = target.split(" (", 1)[0].strip()
      resolved = target != "not found"
      provider = Path(target).name if resolved else None
    else:
      token = line.split(" (", 1)[0].strip()
      name = Path(token).name
      provider = name
      resolved = True
    rows.append({
        "name": name, "resolved": resolved, "provider": provider})
    if not resolved:
      missing.append(name)
  if missing:
    raise ValueError(
        f"unresolved dynamic dependencies for {path.name}: " +
        ", ".join(missing))
  return sorted(rows, key=lambda row: str(row["name"]))


def parser() -> argparse.ArgumentParser:
  sibling_output = R0_ROOT / "output"
  value = argparse.ArgumentParser(
      description=(
          "Copy the exact promoted GPU plugins, custom OpenCL sources, and "
          "OpenVINO notices into a self-verifying native deployment bundle."))
  value.add_argument("--output", type=Path, required=True)
  value.add_argument(
      "--short-plugin", type=Path,
      default=sibling_output / "openvino-90214e-l0-gpu-seq2291/bin/intel64/"
      "Release/libopenvino_intel_gpu_plugin.so")
  value.add_argument(
      "--long-plugin", type=Path,
      default=sibling_output / "openvino-90214e-l0-gpu-seq2119/bin/intel64/"
      "Release/libopenvino_intel_gpu_plugin.so")
  value.add_argument(
      "--custom-config", type=Path,
      default=ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml")
  value.add_argument(
      "--openvino-source", type=Path,
      default=R0_ROOT / "source/openvino-90214e5be05")
  value.add_argument(
      "--long-profile-delta", type=Path,
      default=ROOT / "engine/openvino/"
      "iq36-long-profile-from-short.patch")
  return value


def build(args: argparse.Namespace) -> dict[str, object]:
  output = args.output.expanduser().resolve()
  if output.exists():
    raise ValueError(f"output already exists; refusing to overwrite: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)

  verify_file("short plugin", args.short_plugin, SHORT_PLUGIN_SHA256)
  verify_file("long plugin", args.long_plugin, LONG_PLUGIN_SHA256)
  verify_file("custom CONFIG_FILE", args.custom_config, CUSTOM_CONFIG_SHA256)
  verify_file(
      "long-profile source delta", args.long_profile_delta,
      LONG_PROFILE_PATCH_SHA256)
  sources = custom_sources(args.custom_config)
  source_state, openvino_patch, onednn_patch = source_snapshot(
      args.openvino_source)
  dependency_inventory = {
      "short_plugin": dynamic_dependencies(args.short_plugin),
      "long_plugin": dynamic_dependencies(args.long_plugin),
  }

  license_file = args.openvino_source / "LICENSE"
  licensing_dir = args.openvino_source / "licensing"
  notice_files = tuple(sorted(licensing_dir.glob("*third-party-programs.txt")))
  if not license_file.is_file() or not notice_files:
    raise ValueError(
        "OpenVINO LICENSE or third-party program notices are missing from "
        f"{args.openvino_source}")

  with tempfile.TemporaryDirectory(
      prefix=f".{output.name}-", dir=output.parent) as temporary:
    staging = Path(temporary)
    files: list[dict[str, object]] = []

    def add(source: Path, relative: str, role: str) -> None:
      target = staging / relative
      record = copy_file(source, target)
      record["path"] = relative
      record["role"] = role
      files.append(record)

    def add_bytes(data: bytes, relative: str, role: str) -> None:
      target = staging / relative
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_bytes(data)
      files.append({
          "path": relative,
          "sha256": sha256_file(target),
          "bytes": target.stat().st_size,
          "role": role,
      })

    add(
        args.short_plugin,
        "plugins/short/libopenvino_intel_gpu_plugin.so", "short_plugin")
    add(
        args.long_plugin,
        "plugins/long/libopenvino_intel_gpu_plugin.so", "long_plugin")
    add(
        args.custom_config,
        "openvino/custom/iq36_hot_attention_gqa.xml", "custom_config")
    for source in sources:
      add(source, f"openvino/custom/{source.name}", "custom_opencl_source")
    for name in (
        "intel_qwen36_openvino_hot_cold_attention.py",
        "intel_qwen36_openvino_fixed_fc.py",
    ):
      add(ROOT / "tools" / name, f"tools/{name}", "graph_helper")
    add(
        ROOT / "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json",
        "contracts/qwen36-35b-a3b-openvino-u4-model-contract.json",
        "model_contract")
    add(
        ROOT / "contracts/qwen36-openai-http-publication-policy.json",
        "contracts/qwen36-openai-http-publication-policy.json",
        "publication_policy")
    add(ROOT / "LICENSE", "licenses/intel-qwen36/LICENSE", "license")
    add(license_file, "licenses/openvino/LICENSE", "license")
    for notice in notice_files:
      add(
          notice, f"licenses/openvino/{notice.name}",
          "third_party_notice")
    add_bytes(
        openvino_patch, "source/openvino-promoted-working-tree.patch",
        "promoted_openvino_source_patch")
    add_bytes(
        onednn_patch, "source/onednn-promoted-working-tree.patch",
        "promoted_onednn_source_patch")
    add(
        args.long_profile_delta,
        "source/long-profile-from-short.patch",
        "promoted_long_profile_source_delta")
    add_bytes(
        (json.dumps(source_state, indent=2, sort_keys=True) + "\n").encode(),
        "source/source-state.json", "promoted_source_state")
    add(
        ROOT / "tools/intel-qwen36-build-openvino-plugin.py",
        "source/build-openvino-plugin.py", "source_rebuild_helper")
    add(
        ROOT / "doc/reference/intel-qwen36-35b-a3b-gguf-q4km/"
        "openvino-plugin-rebuild.md",
        "source/README.md", "source_rebuild_documentation")
    add(
        ROOT / "doc/reference/intel-qwen36-35b-a3b-gguf-q4km/"
        "locked-model-provenance-boundary.md",
        "docs/locked-model-provenance-boundary.md",
        "model_provenance_documentation")

    manifest = {
        "format_version": "1.3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "openvino_source_commit": OPENVINO_COMMIT,
        "onednn_source_commit": ONEDNN_COMMIT,
        "promoted_plugins": {
            "embedded_build_identity": CI_BUILD_NUMBER,
            "short_sha256": SHORT_PLUGIN_SHA256,
            "long_sha256": LONG_PLUGIN_SHA256,
        },
        "source_snapshot": {
            "source_state_sha256": source_state["source_state_sha256"],
            "profiles": source_state["profiles"],
            "openvino_patch_sha256": source_state["openvino_patch_sha256"],
            "onednn_patch_sha256": source_state["onednn_patch_sha256"],
            "openvino_changed_files": len(source_state["openvino_files"]),
            "onednn_changed_files": len(source_state["onednn_files"]),
            "scope": (
                "exact short source postimage plus a verified three-file delta "
                "for the exact promoted long source postimage"),
            "rebuild_contract": {
                "ci_build_number": CI_BUILD_NUMBER,
                "expected_plugin_sha256": {
                    "short": SHORT_PLUGIN_SHA256,
                    "long": LONG_PLUGIN_SHA256,
                },
                "logical_source_directory": "source/openvino-90214e5be05",
                "logical_build_directory": "build/openvino-90214e-l0-gpu",
                "toolchain": {
                    "gcc_gxx": "conda-forge 14.3.0-19",
                    "binutils": "2.45.1",
                    "cmake": "4.3.3",
                    "ninja": "1.13.2",
                    "python": "3.12.13",
                    "system_pkg_config": "1.8.1",
                    "level_zero": "1.29.0",
                    "libc": "glibc 2.39",
                },
            },
        },
        "dynamic_dependencies_on_build_target": dependency_inventory,
        "service_environment": {
            "IQ36_SHORT_PLUGIN": (
                "plugins/short/libopenvino_intel_gpu_plugin.so"),
            "IQ36_LONG_PLUGIN": (
                "plugins/long/libopenvino_intel_gpu_plugin.so"),
            "IQ36_CUSTOM_CONFIG": (
                "openvino/custom/iq36_hot_attention_gqa.xml"),
        },
        "files": sorted(files, key=lambda row: str(row["path"])),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    staging.rename(output)

  return {
      "output": str(output),
      "manifest": str(output / "manifest.json"),
      "files": len(manifest["files"]),
      "custom_sources": len(sources),
  }


def main() -> int:
  args = parser().parse_args()
  try:
    result = build(args)
  except (OSError, ValueError) as error:
    raise SystemExit(f"error: {error}") from error
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
