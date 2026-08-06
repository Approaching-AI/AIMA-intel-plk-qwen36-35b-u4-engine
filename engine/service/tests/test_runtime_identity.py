import unittest

from iq36_server.runtime_identity import (
    EXPECTED_OPENVINO_GENAI_VERSION, EXPECTED_OPENVINO_VERSION,
    EXPECTED_OPENVINO_TOKENIZERS_VERSION,
    validate_runtime_identity)


class RuntimeIdentityTest(unittest.TestCase):
  def test_exact_promoted_runtime_passes(self):
    identity = validate_runtime_identity(
        EXPECTED_OPENVINO_VERSION, EXPECTED_OPENVINO_GENAI_VERSION,
        EXPECTED_OPENVINO_TOKENIZERS_VERSION)
    self.assertTrue(identity["verified"])
    self.assertEqual(identity["openvino"], EXPECTED_OPENVINO_VERSION)

  def test_public_final_openvino_build_is_rejected(self):
    with self.assertRaisesRegex(RuntimeError, "90214e5be05"):
      validate_runtime_identity(
          "2026.2.0-21903-52ddc073857-releases/2026/2",
          EXPECTED_OPENVINO_GENAI_VERSION,
          EXPECTED_OPENVINO_TOKENIZERS_VERSION)

  def test_genai_drift_is_rejected(self):
    with self.assertRaisesRegex(RuntimeError, "openvino_genai"):
      validate_runtime_identity(
          EXPECTED_OPENVINO_VERSION, "2026.2.1.0-other",
          EXPECTED_OPENVINO_TOKENIZERS_VERSION)

  def test_tokenizers_drift_is_rejected(self):
    with self.assertRaisesRegex(RuntimeError, "openvino_tokenizers"):
      validate_runtime_identity(
          EXPECTED_OPENVINO_VERSION, EXPECTED_OPENVINO_GENAI_VERSION,
          "2026.2.1.0-other")


if __name__ == "__main__":
  unittest.main()
