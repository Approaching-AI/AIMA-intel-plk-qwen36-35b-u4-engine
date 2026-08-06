from __future__ import annotations

from typing import Any


EXPECTED_OPENVINO_VERSION = (
    "2026.2.0-21902-90214e5be05-releases/2026/2")
EXPECTED_OPENVINO_GENAI_VERSION = "2026.2.0.0-3121-adf73e80e66"
EXPECTED_OPENVINO_TOKENIZERS_VERSION = "2026.2.0.0-681-f43dbd55981"


def validate_runtime_identity(
    openvino_version: str, openvino_genai_version: str,
    openvino_tokenizers_version: str,
) -> dict[str, Any]:
  """Fail closed when the imported runtime is not the promoted carrier.

  The custom GPU plugins are built against one exact OpenVINO source commit.
  Distribution versions such as ``2026.2.0`` are not precise enough: the
  public final wheel and the promoted release-candidate wheel have different
  OpenVINO commits while sharing the same release family.
  """
  observed = {
      "openvino": str(openvino_version),
      "openvino_genai": str(openvino_genai_version),
      "openvino_tokenizers": str(openvino_tokenizers_version),
  }
  expected = {
      "openvino": EXPECTED_OPENVINO_VERSION,
      "openvino_genai": EXPECTED_OPENVINO_GENAI_VERSION,
      "openvino_tokenizers": EXPECTED_OPENVINO_TOKENIZERS_VERSION,
  }
  mismatches = [
      f"{name}: expected {expected[name]}, observed {observed[name]}"
      for name in expected if observed[name] != expected[name]
  ]
  if mismatches:
    raise RuntimeError(
        "OpenVINO runtime identity mismatch; this build cannot inherit the "
        "promoted correctness/performance evidence (" + "; ".join(mismatches) +
        ")")
  return {
      "verified": True,
      "openvino": observed["openvino"],
      "openvino_genai": observed["openvino_genai"],
      "openvino_tokenizers": observed["openvino_tokenizers"],
  }


def verify_imported_runtime() -> dict[str, Any]:
  import openvino as ov
  import openvino_genai as ov_genai
  import openvino_tokenizers

  return validate_runtime_identity(
      ov.get_version(), ov_genai.__version__, openvino_tokenizers.__version__)
