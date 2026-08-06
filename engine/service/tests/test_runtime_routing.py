import unittest

from iq36_server.runtime import OpenVinoRuntime, PREFILL_ALIGNED_SHAPES


class RuntimeRoutingTest(unittest.TestCase):
  def test_prefill_ranges_cover_arbitrary_lengths_without_padding(self):
    for start, end in (
        (0, 1), (0, 16), (0, 32), (0, 33), (0, 81),
        (0, 2049), (0, 8207), (17, 8260), (8192, 16399),
    ):
      ranges = list(OpenVinoRuntime._prefill_ranges(start, end))
      self.assertTrue(ranges)
      self.assertEqual(ranges[0][0], start)
      self.assertEqual(ranges[-1][1], end)
      self.assertEqual(
          [left for left, _ in ranges[1:]],
          [right for _, right in ranges[:-1]])
      for left, right in ranges:
        self.assertIn(right - left, (*PREFILL_ALIGNED_SHAPES, 1))

  def test_aligned_chunks_then_query_one_tail(self):
    ranges = list(OpenVinoRuntime._prefill_ranges(0, 8207))
    self.assertEqual(ranges[0], (0, 8192))
    self.assertEqual(len(ranges), 16)
    self.assertTrue(all(right - left == 1 for left, right in ranges[1:]))

  def test_empty_suffix_has_no_work(self):
    self.assertEqual(list(OpenVinoRuntime._prefill_ranges(10, 10)), [])


if __name__ == "__main__":
  unittest.main()
