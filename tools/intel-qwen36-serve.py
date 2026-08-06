#!/usr/bin/env python3
"""Launch the resident IQ36 OpenAI-compatible service in the bound OV env."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OV_PYTHON = Path(os.environ.get(
    "IQ36_OV_PYTHON", "/home/intel/ov/openvino_env/bin/python"))


def main() -> int:
  # The venv's python is commonly a symlink to /usr/bin/python.  Resolve the
  # environment directory, not the executable itself, or this launcher will
  # exec the same interpreter forever while comparing sys.prefix with /usr.
  expected_prefix = OV_PYTHON.parent.parent.resolve()
  if Path(sys.prefix).resolve() != expected_prefix:
    os.execv(str(OV_PYTHON), [str(OV_PYTHON), str(Path(__file__).resolve()),
                              *sys.argv[1:]])
  sys.path.insert(0, str(ROOT / "engine/service"))
  from iq36_server.cli import main as service_main
  return service_main(sys.argv[1:])


if __name__ == "__main__":
  raise SystemExit(main())
