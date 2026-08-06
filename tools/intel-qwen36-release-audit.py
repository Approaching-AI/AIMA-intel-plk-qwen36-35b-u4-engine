#!/usr/bin/env python3
"""Run high-confidence secret and private-host checks on release files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 16 * 1024 * 1024


def release_files(
    extra_roots: list[Path], *, include_repository: bool,
) -> tuple[Path, ...]:
  files: set[Path] = set()
  if include_repository:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others",
         "--exclude-standard"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE)
    files.update(
        ROOT / item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0") if item)
  for root in extra_roots:
    if not root.is_dir():
      raise ValueError(f"extra release root is not a directory: {root}")
    files.update(
        path for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts)
  return tuple(sorted(files))


def patterns() -> tuple[tuple[str, re.Pattern[bytes]], ...]:
  # Split signature literals so this scanner does not report its own source.
  begin = b"-----" + b"BEGIN "
  private = b"PRIVATE KEY" + b"-----"
  return (
      ("private_key", re.compile(
          begin + rb"(?:RSA |EC |OPENSSH |DSA )?" + private)),
      ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
      ("github_token", re.compile(
          rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
      ("openai_api_key", re.compile(
          rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
      ("slack_token", re.compile(
          rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
      ("google_api_key", re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b")),
      ("stripe_live_key", re.compile(
          rb"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b")),
      ("credential_in_url", re.compile(
          rb"https?://[^\s/:@]+:[^\s/@]+@[^\s/]+")),
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path)
  parser.add_argument(
      "--forbid-host", action="append", default=[],
      help="additional literal private hostname to reject")
  parser.add_argument(
      "--extra-root", action="append", type=Path, default=[],
      help="also scan every file under an ignored release artifact directory")
  parser.add_argument(
      "--only-extra-roots", action="store_true",
      help=("scan only --extra-root trees; use this for the exact exported "
            "public snapshot and unpacked release payloads"))
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  signatures = patterns()
  forbidden_hosts = {
      ("intel-" + "Default-string").encode(),
      *(value.encode() for value in args.forbid_host if value),
  }
  findings: list[dict[str, object]] = []
  checked = 0
  binary_skipped = 0
  oversized = []
  checked_bytes = 0
  extra_roots = [path.expanduser().resolve() for path in args.extra_root]

  if args.only_extra_roots and not extra_roots:
    raise ValueError("--only-extra-roots requires at least one --extra-root")

  for path in release_files(
      extra_roots, include_repository=not args.only_extra_roots):
    try:
      relative = path.relative_to(ROOT).as_posix()
    except ValueError:
      relative = path.as_posix()
    if not path.is_file():
      continue
    size = path.stat().st_size
    with path.open("rb") as handle:
      prefix = handle.read(8192)
    if b"\0" in prefix:
      binary_skipped += 1
      continue
    if size > MAX_TEXT_BYTES:
      oversized.append({"path": relative, "bytes": size})
      continue
    data = path.read_bytes()
    checked += 1
    checked_bytes += len(data)
    for name, signature in signatures:
      for match in signature.finditer(data):
        findings.append({
            "kind": name,
            "path": relative,
            "line": data.count(b"\n", 0, match.start()) + 1,
        })
    for hostname in forbidden_hosts:
      start = 0
      while (index := data.find(hostname, start)) >= 0:
        findings.append({
            "kind": "private_hostname",
            "path": relative,
            "line": data.count(b"\n", 0, index) + 1,
        })
        start = index + len(hostname)

  passed = not findings and not oversized
  result = {
      "schema": "iq36-release-audit-v1",
      "created_at": datetime.now(timezone.utc).isoformat(),
      "pass": passed,
      "scope": (
          ("explicit release artifact roots only" if args.only_extra_roots
           else "git tracked and non-ignored untracked files plus explicit "
                "release artifact roots")),
      "extra_roots": [str(path) for path in extra_roots],
      "text_files_checked": checked,
      "text_bytes_checked": checked_bytes,
      "binary_files_skipped": binary_skipped,
      "oversized_unscanned_files": oversized,
      "findings": findings,
  }
  rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
  print(json.dumps({
      "pass": passed, "text_files_checked": checked,
      "binary_files_skipped": binary_skipped,
      "oversized": len(oversized), "findings": len(findings),
      "output": str(args.output) if args.output is not None else None,
  }, separators=(",", ":")))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
