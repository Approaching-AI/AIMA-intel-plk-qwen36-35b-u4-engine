#!/usr/bin/env python3
"""Extract generated KQ/VS microkernel shims from a captured stock SDPA.

The pinned GPU plugin emits the host-side inline-assembly placeholders and
their machine-code payloads before the per-program JIT definitions.  The
locked PTL candidate reuses those two generated packages inside SimpleGPU;
the companion plugin patch recognizes the explicit phase markers below and
invokes the existing oneDNN microkernel fuser for prefill specializations.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUT_MARKER = "#define FUNC(name)"
SPECS = {
    "prefill": {
        "output": ROOT / "engine/openvino/custom/iq36_prefill_microkernel_shims.cl",
        "begin": "// IQ36_EMBEDDED_MICROKERNEL_PREFILL_BEGIN",
        "end": "// IQ36_EMBEDDED_MICROKERNEL_PREFILL_END",
        "required": (
            "#define ugemm_kq_wg_tile_m 128",
            "#define ugemm_kq_wg_tile_n 32",
            "#define ugemm_vs_wg_tile_m 256",
            "#define ugemm_vs_wg_tile_n 32",
        ),
        "geometry": "PTL, F16 K/Q/V, D=256, K-tile=128, Q-tile=32",
        "trim_before": None,
        "expected_packages": 2,
    },
    "prefill64": {
        "output": ROOT / "output/iq36-prefill64-microkernel-shims.cl",
        "begin": "// IQ36_EMBEDDED_MICROKERNEL_PREFILL64_BEGIN",
        "end": "// IQ36_EMBEDDED_MICROKERNEL_PREFILL64_END",
        "required": (
            "#define ugemm_kq_wg_tile_m 128",
            "#define ugemm_kq_wg_tile_n 64",
            "#define ugemm_vs_wg_tile_m 128",
            "#define ugemm_vs_wg_tile_n 64",
        ),
        "geometry": "PTL, F16 K/Q/V, D=256, K-tile=128, Q-tile=64",
        "trim_before": None,
        "expected_packages": 2,
    },
    "prefill32split": {
        "output": ROOT / "output/iq36-prefill32split-microkernel-shims.cl",
        # Reuse the prefill phase marker because the pinned plugin fuser
        # dispatches generated packages by phase rather than tile geometry.
        "begin": "// IQ36_EMBEDDED_MICROKERNEL_PREFILL_BEGIN",
        "end": "// IQ36_EMBEDDED_MICROKERNEL_PREFILL_END",
        "required": (
            "#define ugemm_kq_sg_tile_n 16",
            "#define ugemm_kq_wg_tile_m 128",
            "#define ugemm_kq_wg_tile_n 32",
            "#define ugemm_kq_sg_per_wg_m 8",
            "#define ugemm_kq_sg_per_wg_n 2",
            "#define ugemm_vs_sg_tile_n 16",
            "#define ugemm_vs_wg_tile_m 256",
            "#define ugemm_vs_wg_tile_n 32",
            "#define ugemm_vs_sg_per_wg_m 8",
            "#define ugemm_vs_sg_per_wg_n 2",
        ),
        "geometry": (
            "PTL, F16 K/Q/V, D=256, K-tile=128, Q-tile=32, "
            "split-N=2"
        ),
        "trim_before": None,
        "expected_packages": 2,
    },
    "decode": {
        "output": ROOT / "engine/openvino/custom/iq36_decode_microkernel_shims.cl",
        "begin": "// IQ36_EMBEDDED_MICROKERNEL_DECODE_BEGIN",
        "end": "// IQ36_EMBEDDED_MICROKERNEL_DECODE_END",
        "required": (
            "#define ugemm_kq_wg_tile_m 256",
            "#define ugemm_kq_wg_tile_n 16",
            "#define ugemm_vs_wg_tile_m 256",
            "#define ugemm_vs_wg_tile_n 16",
        ),
        "geometry": "PTL, F16 K/Q/V, D=256, K-tile=256, Q-tile=16",
        # The stock source also emits paged-cache KcQ/VcS packages even
        # though this non-paged specialization compiles those calls out.
        "trim_before": "#define ugemm_kcq_sg_tile_m",
        "expected_packages": 2,
    },
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("--kind", choices=tuple(SPECS), default="prefill")
  parser.add_argument("--output", type=Path)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  spec = SPECS[args.kind]
  source = args.source.resolve()
  output = (args.output or spec["output"]).resolve()
  data = source.read_bytes()
  text = data.decode("utf-8")
  if CUT_MARKER not in text:
    raise ValueError(f"{source}: missing {CUT_MARKER!r}")
  shims = text.split(CUT_MARKER, 1)[0].rstrip()
  trim_before = spec["trim_before"]
  if trim_before is not None:
    if trim_before not in shims:
      raise ValueError(f"{source}: missing trim marker {trim_before!r}")
    shims = shims.split(trim_before, 1)[0].rstrip()
  # Generated provider comments can carry incidental trailing blanks.  Keep
  # tracked shims reproducible and clean without altering the package bytes.
  shims = "\n".join(line.rstrip() for line in shims.splitlines())
  expected_packages = spec["expected_packages"]
  if shims.count("@_u_@") != expected_packages:
    raise ValueError(
        f"{source}: expected exactly {expected_packages} embedded "
        "microkernels, observed "
        f"{shims.count('@_u_@')}")
  required = (
      "ugemm_kq_c_type ugemm_kq(",
      "ugemm_vs_c_type ugemm_vs(",
      *spec["required"],
  )
  missing = [needle for needle in required if needle not in shims]
  if missing:
    raise ValueError(f"{source}: incompatible shim geometry: {missing}")
  digest = hashlib.sha256(data).hexdigest()
  relative_source = source
  try:
    relative_source = source.relative_to(ROOT)
  except ValueError:
    pass
  generated = (
      "// Generated by tools/intel-qwen36-openvino-prefill-"
      "microkernel-shim-extract.py.\n"
      f"// Source: {relative_source}\n"
      f"// Source SHA256: {digest}\n"
      f"// Locked geometry: {spec['geometry']}.\n"
      f"{spec['begin']}\n"
      f"{shims}\n"
      f"{spec['end']}\n"
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(generated, encoding="utf-8")
  print(output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
