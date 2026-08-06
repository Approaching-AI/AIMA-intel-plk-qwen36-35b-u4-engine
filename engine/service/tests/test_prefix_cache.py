import unittest

from iq36_server.prefix_cache import PrefixCache


class Clock:
  def __init__(self):
    self.value = 0.0

  def __call__(self):
    return self.value


class PrefixCacheTest(unittest.TestCase):
  def test_longest_exact_prefix_and_lru(self):
    clock = Clock()
    cache = PrefixCache(max_bytes=10, max_entries=2, ttl_s=10, clock=clock)
    self.assertTrue(cache.put((1, 2), "a", 4))
    self.assertTrue(cache.put((1, 2, 3), "b", 4))
    entry = cache.find_longest((1, 2, 3, 4))
    self.assertIsNotNone(entry)
    self.assertEqual(entry.value, "b")
    self.assertTrue(cache.put((9,), "c", 4))
    self.assertIsNone(cache.find_longest((1, 2, 8)))
    self.assertEqual(cache.stats().evictions, 1)

  def test_ttl_and_oversize_rejection(self):
    clock = Clock()
    cache = PrefixCache(max_bytes=5, max_entries=2, ttl_s=3, clock=clock)
    self.assertFalse(cache.put((1,), "too large", 6))
    self.assertTrue(cache.put((2,), "ok", 5))
    clock.value = 3.0
    self.assertIsNone(cache.find_longest((2, 3)))
    stats = cache.stats()
    self.assertEqual(stats.expired, 1)
    self.assertEqual(stats.rejected, 1)

  def test_disabled_cache(self):
    cache = PrefixCache(max_bytes=0, max_entries=0, ttl_s=0)
    self.assertFalse(cache.put((1,), object(), 0))
    self.assertIsNone(cache.find_longest((1,)))


if __name__ == "__main__":
  unittest.main()
