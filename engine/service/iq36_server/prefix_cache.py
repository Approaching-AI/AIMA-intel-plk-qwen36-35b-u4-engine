from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class PrefixEntry(Generic[T]):
  tokens: tuple[int, ...]
  value: T
  byte_count: int
  created_at: float
  last_used_at: float


@dataclass(frozen=True)
class PrefixCacheStats:
  entries: int
  bytes: int
  hits: int
  misses: int
  inserts: int
  evictions: int
  expired: int
  rejected: int


class PrefixCache(Generic[T]):
  """Bounded process-local LRU for exact token-prefix state snapshots."""

  def __init__(
      self, *, max_bytes: int, max_entries: int, ttl_s: float,
      clock=time.monotonic,
  ) -> None:
    if max_bytes < 0 or max_entries < 0 or ttl_s < 0:
      raise ValueError("prefix cache bounds must be non-negative")
    self.max_bytes = max_bytes
    self.max_entries = max_entries
    self.ttl_s = ttl_s
    self._clock = clock
    self._entries: OrderedDict[tuple[int, ...], PrefixEntry[T]] = OrderedDict()
    self._bytes = 0
    self._hits = 0
    self._misses = 0
    self._inserts = 0
    self._evictions = 0
    self._expired = 0
    self._rejected = 0
    self._lock = threading.Lock()

  @property
  def enabled(self) -> bool:
    return self.max_bytes > 0 and self.max_entries > 0 and self.ttl_s > 0

  def _expire_locked(self, now: float) -> None:
    if not self._entries:
      return
    stale = [
        key for key, entry in self._entries.items()
        if now - entry.last_used_at >= self.ttl_s
    ]
    for key in stale:
      entry = self._entries.pop(key)
      self._bytes -= entry.byte_count
      self._expired += 1

  def find_longest(self, tokens: tuple[int, ...]) -> PrefixEntry[T] | None:
    now = self._clock()
    with self._lock:
      self._expire_locked(now)
      best_key: tuple[int, ...] | None = None
      for key in self._entries:
        if len(key) <= len(tokens) and len(key) > len(best_key or ()):
          if tokens[:len(key)] == key:
            best_key = key
      if best_key is None:
        self._misses += 1
        return None
      entry = self._entries.pop(best_key)
      entry.last_used_at = now
      self._entries[best_key] = entry
      self._hits += 1
      return entry

  def put(self, tokens: tuple[int, ...], value: T, byte_count: int) -> bool:
    if byte_count < 0:
      raise ValueError("byte count must be non-negative")
    now = self._clock()
    with self._lock:
      self._expire_locked(now)
      if (
          not self.enabled or not tokens or byte_count > self.max_bytes):
        self._rejected += 1
        return False
      previous = self._entries.pop(tokens, None)
      if previous is not None:
        self._bytes -= previous.byte_count
      entry = PrefixEntry(tokens, value, byte_count, now, now)
      self._entries[tokens] = entry
      self._bytes += byte_count
      self._inserts += 1
      while (
          len(self._entries) > self.max_entries or
          self._bytes > self.max_bytes):
        _, evicted = self._entries.popitem(last=False)
        self._bytes -= evicted.byte_count
        self._evictions += 1
      return tokens in self._entries

  def clear(self) -> None:
    with self._lock:
      self._entries.clear()
      self._bytes = 0

  def stats(self) -> PrefixCacheStats:
    now = self._clock()
    with self._lock:
      self._expire_locked(now)
      return PrefixCacheStats(
          entries=len(self._entries), bytes=self._bytes, hits=self._hits,
          misses=self._misses, inserts=self._inserts,
          evictions=self._evictions, expired=self._expired,
          rejected=self._rejected)

