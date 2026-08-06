from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StoredResponse:
  response_id: str
  messages: tuple[dict[str, Any], ...]
  assistant: dict[str, Any]
  response: dict[str, Any]
  created_at: float
  byte_count: int


class ResponseStore:
  """Byte-, entry-, and TTL-bounded local state for Responses."""

  def __init__(
      self, max_entries: int, ttl_s: float,
      *, max_bytes: int = 256 * 1024 * 1024,
      clock: Callable[[], float] = time.monotonic,
  ) -> None:
    if max_entries < 0 or max_bytes < 0 or ttl_s < 0:
      raise ValueError("response store bounds must be non-negative")
    self.max_entries = max_entries
    self.max_bytes = max_bytes
    self.ttl_s = ttl_s
    self._clock = clock
    self._values: OrderedDict[str, StoredResponse] = OrderedDict()
    self._bytes = 0
    self._lock = threading.Lock()

  def _expire(self, now: float) -> None:
    stale = [
        key for key, value in self._values.items()
        if now - value.created_at >= self.ttl_s]
    for key in stale:
      value = self._values.pop(key, None)
      if value is not None:
        self._bytes -= value.byte_count

  def put(
      self, response_id: str, messages: list[dict[str, Any]],
      assistant: dict[str, Any], response: dict[str, Any],
  ) -> bool:
    if self.max_entries == 0 or self.max_bytes == 0 or self.ttl_s == 0:
      return False
    now = self._clock()
    copied_messages = tuple(deepcopy(messages))
    copied_assistant = deepcopy(assistant)
    copied_response = deepcopy(response)
    byte_count = len(json.dumps(
        {
            "messages": copied_messages,
            "assistant": copied_assistant,
            "response": copied_response,
        },
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))
    if byte_count > self.max_bytes:
      return False
    value = StoredResponse(
        response_id=response_id, messages=copied_messages,
        assistant=copied_assistant, response=copied_response,
        created_at=now, byte_count=byte_count)
    with self._lock:
      self._expire(now)
      replaced = self._values.pop(response_id, None)
      if replaced is not None:
        self._bytes -= replaced.byte_count
      self._values[response_id] = value
      self._bytes += byte_count
      while (
          len(self._values) > self.max_entries or
          self._bytes > self.max_bytes
      ):
        _, evicted = self._values.popitem(last=False)
        self._bytes -= evicted.byte_count
    return True

  def get(self, response_id: str) -> StoredResponse | None:
    now = self._clock()
    with self._lock:
      self._expire(now)
      value = self._values.pop(response_id, None)
      if value is None:
        return None
      self._values[response_id] = value
      return deepcopy(value)

  def delete(self, response_id: str) -> bool:
    now = self._clock()
    with self._lock:
      self._expire(now)
      value = self._values.pop(response_id, None)
      if value is None:
        return False
      self._bytes -= value.byte_count
      return True

  def stats(self) -> dict[str, int]:
    now = self._clock()
    with self._lock:
      self._expire(now)
      return {"entries": len(self._values), "bytes": self._bytes}

  def clear(self) -> None:
    with self._lock:
      self._values.clear()
      self._bytes = 0
