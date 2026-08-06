import hashlib
import tempfile
import unittest
from pathlib import Path

from iq36_server.model_identity import _verify_files


class ModelIdentityTest(unittest.TestCase):
  def test_full_and_metadata_verification(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      content = b"locked-model-test"
      (root / "model.bin").write_bytes(content)
      manifest = {"model.bin": {
          "bytes": len(content),
          "sha256": hashlib.sha256(content).hexdigest(),
      }}
      rows, fingerprint = _verify_files(root, manifest, full_hash=True)
      self.assertTrue(rows[0]["sha256_verified"])
      self.assertIsNotNone(fingerprint)
      rows, fingerprint = _verify_files(root, manifest, full_hash=False)
      self.assertFalse(rows[0]["sha256_verified"])
      self.assertIsNone(fingerprint)

  def test_drift_is_rejected(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      (root / "model.bin").write_bytes(b"drifted")
      with self.assertRaises(RuntimeError):
        _verify_files(root, {"model.bin": {
            "bytes": 7, "sha256": "0" * 64,
        }}, full_hash=True)


if __name__ == "__main__":
  unittest.main()
