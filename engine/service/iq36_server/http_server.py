from __future__ import annotations

import asyncio
import json
import logging
from http import HTTPStatus
from urllib.parse import urlsplit

import h11

from .config import ServerConfig
from .http_types import HTTPRequest, HTTPResponse


LOGGER = logging.getLogger("iq36.http")


class HTTPServer:
  def __init__(self, config: ServerConfig, application) -> None:
    self.config = config
    self.application = application
    self.server: asyncio.AbstractServer | None = None
    self._connections: set[asyncio.Task] = set()

  async def start(self) -> None:
    self.server = await asyncio.start_server(
        self._connection, self.config.host, self.config.port,
        limit=min(self.config.max_request_bytes + 65536, 64 * 1024 * 1024),
        start_serving=True)
    sockets = self.server.sockets or []
    addresses = ", ".join(str(sock.getsockname()) for sock in sockets)
    LOGGER.info("HTTP service listening addresses=%s", addresses)

  async def close(self, timeout_s: float | None = None) -> None:
    listener = self.server
    if self.server is not None:
      self.server.close()
      self.server = None
    tasks = [task for task in self._connections if not task.done()]
    if tasks:
      if timeout_s is not None and timeout_s > 0:
        _, pending = await asyncio.wait(tasks, timeout=timeout_s)
      else:
        pending = set(tasks)
      for task in pending:
        task.cancel()
      if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    if listener is not None:
      await listener.wait_closed()

  @staticmethod
  def _headers(event) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_name, raw_value in event.headers:
      name = raw_name.decode("ascii").lower()
      value = raw_value.decode("latin-1")
      values[name] = value if name not in values else values[name] + ", " + value
    return values

  async def _read_request(
      self, connection: h11.Connection, reader: asyncio.StreamReader,
      client: str,
  ) -> HTTPRequest | None:
    request_event = None
    body = bytearray()
    while True:
      event = connection.next_event()
      if event is h11.NEED_DATA:
        try:
          data = await asyncio.wait_for(
              reader.read(65536), self.config.keepalive_timeout_s)
        except asyncio.TimeoutError:
          return None
        if not data:
          connection.receive_data(b"")
        else:
          connection.receive_data(data)
        continue
      if event is h11.PAUSED:
        return None
      if isinstance(event, h11.ConnectionClosed):
        return None
      if isinstance(event, h11.Request):
        request_event = event
        headers = self._headers(event)
        content_length = headers.get("content-length")
        if content_length is not None:
          try:
            declared = int(content_length)
          except ValueError as error:
            raise h11.RemoteProtocolError("invalid Content-Length") from error
          if declared > self.config.max_request_bytes:
            raise BodyTooLarge()
      elif isinstance(event, h11.Data):
        body.extend(event.data)
        if len(body) > self.config.max_request_bytes:
          raise BodyTooLarge()
      elif isinstance(event, h11.EndOfMessage):
        if request_event is None:
          raise h11.RemoteProtocolError("body received before request")
        target = request_event.target.decode("ascii", errors="surrogateescape")
        split = urlsplit(target)
        return HTTPRequest(
            method=request_event.method.decode("ascii").upper(),
            target=target, path=split.path, query=split.query,
            headers=self._headers(request_event), body=bytes(body),
            client=client, disconnect_event=asyncio.Event())

  @staticmethod
  async def _watch_disconnect(
      reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
      event: asyncio.Event,
  ) -> None:
    while not event.is_set():
      if reader.at_eof() or writer.is_closing():
        event.set()
        return
      await asyncio.sleep(0.05)

  def _base_headers(self, response: HTTPResponse) -> list[tuple[bytes, bytes]]:
    headers = list(response.headers)
    headers.extend([
        ("server", "iq36"),
        ("x-content-type-options", "nosniff"),
    ])
    if self.config.cors_origin:
      headers.extend([
          ("access-control-allow-origin", self.config.cors_origin),
          ("access-control-allow-headers", "authorization,content-type"),
          ("access-control-allow-methods", "GET,POST,DELETE,OPTIONS"),
      ])
    if response.body is not None:
      headers.append(("content-length", str(len(response.body))))
    return [
        (name.encode("ascii"), value.encode("latin-1"))
        for name, value in headers]

  async def _send(
      self, connection: h11.Connection, writer: asyncio.StreamWriter,
      response: HTTPResponse,
  ) -> None:
    reason = HTTPStatus(response.status).phrase.encode("ascii")
    event = h11.Response(
        status_code=response.status, reason=reason,
        headers=self._base_headers(response))
    writer.write(connection.send(event))
    await writer.drain()
    try:
      if response.body is not None:
        if response.body:
          writer.write(connection.send(h11.Data(data=response.body)))
          await writer.drain()
      elif response.stream is not None:
        async for chunk in response.stream:
          if not chunk:
            continue
          writer.write(connection.send(h11.Data(data=chunk)))
          await writer.drain()
      writer.write(connection.send(h11.EndOfMessage()))
      await writer.drain()
    except (BrokenPipeError, ConnectionError, asyncio.CancelledError):
      if response.on_disconnect is not None:
        response.on_disconnect()
      if response.stream is not None:
        await response.stream.aclose()
      raise

  async def _simple_error(
      self, connection, writer, status: int, message: str,
  ) -> None:
    body = json.dumps({"error": {
        "message": message, "type": "invalid_request_error",
        "param": None, "code": None,
    }}, separators=(",", ":")).encode("utf-8")
    response = HTTPResponse(
        status=status,
        headers=[("content-type", "application/json; charset=utf-8"),
                 ("connection", "close")], body=body)
    try:
      await self._send(connection, writer, response)
    except Exception:
      pass

  async def _connection(
      self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
  ) -> None:
    task = asyncio.current_task()
    if task is not None:
      self._connections.add(task)
    peer = writer.get_extra_info("peername")
    client = str(peer[0]) if isinstance(peer, tuple) and peer else str(peer)
    connection = h11.Connection(
        h11.SERVER, max_incomplete_event_size=64 * 1024)
    try:
      while True:
        try:
          request = await asyncio.wait_for(
              self._read_request(connection, reader, client),
              timeout=self.config.keepalive_timeout_s)
        except asyncio.TimeoutError:
          break
        except BodyTooLarge:
          await self._simple_error(
              connection, writer, 413,
              f"Request body exceeds {self.config.max_request_bytes} bytes.")
          break
        except h11.RemoteProtocolError as error:
          await self._simple_error(connection, writer, 400, str(error))
          break
        if request is None:
          break
        started = time_monotonic()
        disconnect_task = asyncio.create_task(self._watch_disconnect(
            reader, writer, request.disconnect_event))
        try:
          response = await self.application.dispatch(request)
          await self._send(connection, writer, response)
        except (BrokenPipeError, ConnectionError, asyncio.CancelledError):
          request.disconnect_event.set()
          break
        finally:
          disconnect_task.cancel()
          await asyncio.gather(disconnect_task, return_exceptions=True)
        self.application.metrics.http(
            request.method, request.path, response.status)
        request_id = next(
            (value for name, value in response.headers
             if name.lower() == "x-request-id"), "-")
        LOGGER.info(
            "request request_id=%s method=%s path=%s status=%s client=%s "
            "elapsed_ms=%.3f",
            request_id, request.method, request.path, response.status, client,
            (time_monotonic() - started) * 1000.0)
        if connection.our_state is h11.MUST_CLOSE \
            or connection.their_state is h11.MUST_CLOSE:
          break
        connection.start_next_cycle()
    except asyncio.CancelledError:
      raise
    except (BrokenPipeError, ConnectionError):
      # A client may reset an idle keep-alive connection after receiving its
      # response. This is a normal disconnect, not a server failure.
      pass
    except Exception:
      LOGGER.exception("connection failure client=%s", client)
    finally:
      writer.close()
      try:
        await writer.wait_closed()
      except (BrokenPipeError, ConnectionError):
        pass
      if task is not None:
        self._connections.discard(task)


class BodyTooLarge(Exception):
  pass


def time_monotonic() -> float:
  return asyncio.get_running_loop().time()
