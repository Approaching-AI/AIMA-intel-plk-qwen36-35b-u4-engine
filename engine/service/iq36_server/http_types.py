from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HTTPRequest:
  method: str
  target: str
  path: str
  query: str
  headers: dict[str, str]
  body: bytes
  client: str
  disconnect_event: asyncio.Event | None = None


@dataclass
class HTTPResponse:
  status: int
  headers: list[tuple[str, str]] = field(default_factory=list)
  body: bytes | None = None
  stream: AsyncIterator[bytes] | None = None
  on_disconnect: Callable[[], None] | None = None
