#!/usr/bin/env python3
"""Build a verified, offline wheelhouse for the promoted HTTP carrier."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/service"))

from iq36_server.runtime_identity import validate_runtime_identity  # noqa: E402


EXPECTED_DISTRIBUTIONS = {
    "openvino": "2026.2.0rc2",
    "openvino-genai": "2026.2.0.0rc2",
    "openvino-telemetry": "2025.2.0",
    "openvino-tokenizers": "2026.2.0.0rc2",
}
PUBLIC_RUNTIME_DISTRIBUTIONS = {
    "attrs": "26.1.0",
    "h11": "0.16.0",
    "jsonschema": "4.26.0",
    "jsonschema-specifications": "2025.9.1",
    "numpy": "2.4.4",
    "pip": "26.2.1",
    "referencing": "0.37.0",
    "rpds-py": "2026.6.3",
    "typing-extensions": "4.15.0",
}
INDEX_AUDITABLE_RECONSTRUCTED_DISTRIBUTIONS = {"openvino-telemetry"}
NORMALIZED_MTIME = 315532800  # 1980-01-01, the earliest portable ZIP time.


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _probe(python: Path) -> dict:
  code = r'''
import importlib.metadata as md
import json
import platform
import sys
import sysconfig
import openvino as ov
import openvino_genai as ov_genai
import openvino_tokenizers

names = ("openvino", "openvino-genai", "openvino-telemetry",
         "openvino-tokenizers")
print(json.dumps({
    "python": platform.python_version(),
    "implementation": platform.python_implementation(),
    "machine": platform.machine(),
    "site_packages": sysconfig.get_path("purelib"),
    "scripts": sysconfig.get_path("scripts"),
    "openvino": ov.get_version(),
    "openvino_genai": ov_genai.__version__,
    "openvino_tokenizers": openvino_tokenizers.__version__,
    "distributions": {
        name: {"version": md.version(name),
               "dist_info": str(md.distribution(name)._path)}
        for name in names
    },
}, sort_keys=True))
'''
  result = subprocess.run(
      [str(python), "-c", code], check=True, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  return json.loads(result.stdout)


def _record_digest(path: Path) -> str:
  return base64.urlsafe_b64encode(
      hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode("ascii")


def _copy_distribution(
    site_packages: Path, scripts: Path, dist_info: Path, stage: Path,
) -> dict:
  record = dist_info / "RECORD"
  if not record.is_file():
    raise RuntimeError(f"distribution has no RECORD: {dist_info.name}")
  verified = 0
  copied = 0
  skipped_generated = 0
  skipped_scripts = 0
  with record.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.reader(handle))
  for row in rows:
    if len(row) != 3:
      raise RuntimeError(f"malformed RECORD row in {dist_info.name}: {row!r}")
    raw_name, encoded_hash, raw_size = row
    relative = PurePosixPath(raw_name)
    if relative.is_absolute():
      raise RuntimeError(f"absolute RECORD path: {raw_name}")
    parts = relative.parts
    external_script = (
        len(parts) >= 4 and parts[:3] == ("..", "..", "..") and
        parts[3] == "bin")
    if ".." in parts and not external_script:
      raise RuntimeError(f"unsafe RECORD path: {raw_name}")
    source = (
        scripts.joinpath(*parts[4:]) if external_script
        else site_packages.joinpath(*parts))
    if encoded_hash:
      algorithm, separator, expected_hash = encoded_hash.partition("=")
      if separator != "=" or algorithm != "sha256":
        raise RuntimeError(
            f"unsupported RECORD hash for {raw_name}: {encoded_hash}")
      if not source.is_file():
        raise RuntimeError(f"RECORD file is missing: {source}")
      if raw_size and source.stat().st_size != int(raw_size):
        raise RuntimeError(f"RECORD size mismatch: {raw_name}")
      if _record_digest(source) != expected_hash:
        raise RuntimeError(f"RECORD SHA-256 mismatch: {raw_name}")
      verified += 1
    elif raw_name.endswith(".pyc") or raw_name.endswith("/RECORD"):
      skipped_generated += 1
      continue
    else:
      raise RuntimeError(f"unhashed non-generated RECORD file: {raw_name}")
    if external_script:
      # entry_points.txt makes pip regenerate these scripts on installation.
      skipped_scripts += 1
      continue
    if raw_name.endswith(("/INSTALLER", "/REQUESTED")):
      skipped_generated += 1
      continue
    destination = stage.joinpath(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)
    os.utime(destination, (NORMALIZED_MTIME, NORMALIZED_MTIME))
    copied += 1
  return {
      "record_rows": len(rows), "record_files_verified": verified,
      "files_copied": copied, "generated_files_skipped": skipped_generated,
      "installed_scripts_skipped": skipped_scripts,
  }


def _pack(stage: Path, destination: Path) -> Path:
  before = set(destination.glob("*.whl"))
  environment = os.environ.copy()
  environment["SOURCE_DATE_EPOCH"] = str(NORMALIZED_MTIME)
  subprocess.run(
      [sys.executable, "-m", "wheel", "pack", str(stage),
       "--dest-dir", str(destination)], check=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
      env=environment)
  created = set(destination.glob("*.whl")) - before
  if len(created) != 1:
    raise RuntimeError(
        f"wheel pack produced {len(created)} new wheels for {stage.name}")
  return created.pop()


def _wheel_identity(path: Path) -> tuple[str, str]:
  with zipfile.ZipFile(path) as archive:
    names = [
        name for name in archive.namelist()
        if name.endswith(".dist-info/METADATA")]
    if len(names) != 1:
      raise RuntimeError(
          f"wheel must contain exactly one METADATA file: {path.name}")
    metadata = Parser().parsestr(
        archive.read(names[0]).decode("utf-8", errors="strict"))
  return str(metadata["Name"]), str(metadata["Version"])


def _download_public_wheels(python: Path, destination: Path) -> list[dict]:
  wheels = []
  for name, version in PUBLIC_RUNTIME_DISTRIBUTIONS.items():
    before = set(destination.glob("*.whl"))
    subprocess.run([
        str(python), "-m", "pip", "download", "--only-binary=:all:",
        "--no-deps", "--dest", str(destination), f"{name}=={version}",
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    created = set(destination.glob("*.whl")) - before
    if len(created) != 1:
      raise RuntimeError(
          f"pip download produced {len(created)} wheels for {name}=={version}")
    wheel = created.pop()
    observed_name, observed_version = _wheel_identity(wheel)
    if observed_name.lower().replace("_", "-") != name or \
        observed_version != version:
      raise RuntimeError(
          f"downloaded wheel identity mismatch for {name}=={version}: "
          f"{observed_name}=={observed_version}")
    wheels.append({
        "distribution": name, "version": version,
        "filename": wheel.name, "bytes": wheel.stat().st_size,
        "sha256": _sha256(wheel), "source": "package_index",
    })
  return wheels


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(
      description="Package the exact promoted OpenVINO Python runtime wheels")
  parser.add_argument(
      "--python", type=Path,
      default=Path("/home/intel/ov/openvino_env/bin/python"))
  parser.add_argument(
      "--service-wheel", type=Path, required=True,
      help="already-built intel-qwen36-server wheel to include")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)
  output = args.output.resolve()
  if output.exists():
    parser.error(f"output already exists: {output}")
  if not args.python.is_file():
    parser.error(f"Python interpreter does not exist: {args.python}")
  if not args.service_wheel.is_file():
    parser.error(f"service wheel does not exist: {args.service_wheel}")

  # Keep the venv path itself. Resolving its interpreter symlink would execute
  # the base interpreter without the selected environment's site-packages.
  probe = _probe(args.python.absolute())
  identity = validate_runtime_identity(
      probe["openvino"], probe["openvino_genai"],
      probe["openvino_tokenizers"])
  if probe["implementation"] != "CPython" or not probe["python"].startswith(
      "3.12."):
    raise RuntimeError(
        "the promoted Python runtime requires CPython 3.12; observed "
        f"{probe['implementation']} {probe['python']}")
  site_packages = Path(probe["site_packages"]).resolve()
  scripts = Path(probe["scripts"]).resolve()
  for name, expected in EXPECTED_DISTRIBUTIONS.items():
    observed = probe["distributions"][name]["version"]
    if observed != expected:
      raise RuntimeError(
          f"distribution mismatch for {name}: expected {expected}, "
          f"observed {observed}")

  output.mkdir(parents=True)
  wheels = []
  try:
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
      dist_info = Path(probe["distributions"][name]["dist_info"]).resolve()
      try:
        dist_info.relative_to(site_packages)
      except ValueError as error:
        raise RuntimeError(
            f"distribution metadata is outside site-packages: {name}") from error
      with tempfile.TemporaryDirectory(prefix="iq36-wheel-stage-") as temporary:
        stage = Path(temporary) / f"{name}-{expected}"
        stage.mkdir()
        record_summary = _copy_distribution(
            site_packages, scripts, dist_info, stage)
        wheel = _pack(stage, output)
      wheels.append({
          "distribution": name, "version": expected,
          "filename": wheel.name, "bytes": wheel.stat().st_size,
          "sha256": _sha256(wheel),
          "source": "verified_installed_record", **record_summary,
      })
    wheels.extend(_download_public_wheels(args.python.absolute(), output))

    service_name, service_version = _wheel_identity(args.service_wheel)
    if service_name != "intel-qwen36-server" or service_version != "0.1.0":
      raise RuntimeError(
          "service wheel identity mismatch: expected "
          f"intel-qwen36-server==0.1.0, observed "
          f"{service_name}=={service_version}")
    service_destination = output / args.service_wheel.name
    shutil.copy2(args.service_wheel, service_destination)
    wheels.append({
        "distribution": service_name, "version": service_version,
        "filename": service_destination.name,
        "bytes": service_destination.stat().st_size,
        "sha256": _sha256(service_destination), "source": "local_build",
    })

    constraints_source = ROOT / "deploy/bound-runtime-constraints.txt"
    constraints_destination = output / "bound-runtime-constraints.txt"
    shutil.copy2(constraints_source, constraints_destination)
    requirements_destination = output / "bound-runtime-requirements.txt"
    requirements_destination.write_text(
        "".join(
            f"{row['distribution']}=={row['version']} "
            f"--hash=sha256:{row['sha256']}\n"
            for row in sorted(
                wheels, key=lambda item: str(item["distribution"]))),
        encoding="utf-8")
    audit_requirements_destination = output / "bound-index-requirements.txt"
    audit_requirements_destination.write_text(
        "".join(
            f"{row['distribution']}=={row['version']} "
            f"--hash=sha256:{row['sha256']}\n"
            for row in sorted(
                (item for item in wheels
                 if item["source"] == "package_index" or
                 item["distribution"] in
                 INDEX_AUDITABLE_RECONSTRUCTED_DISTRIBUTIONS),
                key=lambda item: str(item["distribution"]))),
        encoding="utf-8")
    manifest = {
        "schema": "iq36-promoted-python-runtime-v1",
        "runtime_identity": identity,
        "python": {
            "implementation": probe["implementation"],
            "version": probe["python"], "machine": probe["machine"],
        },
        "wheels": wheels,
        "constraints": {
            "filename": constraints_destination.name,
            "sha256": _sha256(constraints_destination),
        },
        "hashed_requirements": {
            "filename": requirements_destination.name,
            "sha256": _sha256(requirements_destination),
            "require_hashes": True,
        },
        "index_audit_requirements": {
            "filename": audit_requirements_destination.name,
            "sha256": _sha256(audit_requirements_destination),
            "require_hashes": True,
            "scope": "package-index-resolvable dependencies only",
            "non_index_distributions": [
                str(row["distribution"]) for row in wheels
                if row["source"] != "package_index" and
                row["distribution"] not in
                INDEX_AUDITABLE_RECONSTRUCTED_DISTRIBUTIONS
            ],
        },
        "bootstrap_installer": (
            "python3.12 -m pip install --no-index --find-links=. "
            "pip==26.2.1"),
        "install": (
            "python3.12 -m pip install --no-index --find-links=. "
            "--require-hashes -r bound-runtime-requirements.txt"),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output": str(output), "manifest": str(manifest_path),
        "wheel_count": len(wheels), "wheels": wheels,
    }, indent=2))
  except Exception:
    shutil.rmtree(output)
    raise
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
