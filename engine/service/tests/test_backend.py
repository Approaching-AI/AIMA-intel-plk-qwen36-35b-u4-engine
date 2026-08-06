import unittest

from iq36_server.backend import WorkerError, stable_text_delta


class BackendTest(unittest.TestCase):
  def test_stream_delta_holds_partial_utf8_until_stable(self):
    committed, delta = stable_text_delta("hello", "hello\ufffd")
    self.assertEqual((committed, delta), ("hello", ""))
    committed, delta = stable_text_delta(committed, "hello中")
    self.assertEqual((committed, delta), ("hello中", "中"))

  def test_final_delta_flushes_literal_replacement_character(self):
    committed, delta = stable_text_delta("x", "x\ufffd", final=True)
    self.assertEqual((committed, delta), ("x\ufffd", "\ufffd"))

  def test_stream_delta_rejects_committed_prefix_rewrite(self):
    with self.assertRaises(WorkerError):
      stable_text_delta("hello", "hullo")


if __name__ == "__main__":
  unittest.main()
