import unittest
from dataclasses import replace

from iq36_server.config import ServerConfig


class ConfigTest(unittest.TestCase):
  def setUp(self):
    self.config = ServerConfig(backend="mock", preload_bucket=0)

  def test_context_and_memory_guard_validation(self):
    replace(
        self.config, max_context_length=8192, max_new_tokens=512,
        min_available_gib=8, abort_below_available_gib=4).validate()
    with self.assertRaises(ValueError):
      replace(
          self.config, max_context_length=512,
          max_new_tokens=512).validate()
    with self.assertRaises(ValueError):
      replace(
          self.config, min_available_gib=4,
          abort_below_available_gib=8).validate()
    with self.assertRaises(ValueError):
      replace(self.config, model_verification="trust-me").validate()

  def test_non_loopback_requires_auth(self):
    with self.assertRaises(ValueError):
      replace(self.config, host="0.0.0.0").validate()
    replace(self.config, host="0.0.0.0", api_key="secret").validate()

  def test_header_and_timeout_values_are_safe(self):
    with self.assertRaises(ValueError):
      replace(self.config, cors_origin="https://ok\r\nX-Bad: 1").validate()
    with self.assertRaises(ValueError):
      replace(self.config, request_timeout_s=0).validate()
    with self.assertRaises(ValueError):
      replace(self.config, cancel_grace_s=0).validate()
    with self.assertRaises(ValueError):
      replace(self.config, response_store_bytes=-1).validate()


if __name__ == "__main__":
  unittest.main()
