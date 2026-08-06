import asyncio
import http.client
import json
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

from iq36_server.application import Application
from iq36_server.backend import MockBackend, select_bucket, select_profile
from iq36_server.config import ServerConfig
from iq36_server.http_server import HTTPServer
from iq36_server.response_store import ResponseStore
from iq36_server.tokenizer import SimpleTestTokenizer
from iq36_server.types import GenerationDelta, GenerationResult, GenerationStarted


class HTTPIntegrationTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.config = ServerConfig(
        host="127.0.0.1", port=0, backend="mock",
        max_context_length=4096, max_new_tokens=64, preload_bucket=0,
        lazy_start=True, api_key="secret", max_request_bytes=8192)
    self.backend = MockBackend(self.config, "hello")
    self.app = Application(
        self.config, backend=self.backend, tokenizer=SimpleTestTokenizer())
    self.server = HTTPServer(self.config, self.app)
    await self.app.start()
    await self.server.start()
    self.port = self.server.server.sockets[0].getsockname()[1]

  async def asyncTearDown(self):
    await self.server.close()
    await self.app.close()

  def _request_sync(self, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    result = (response.status, dict(response.headers), payload)
    connection.close()
    return result

  async def _request(self, method, path, value=None, auth=True):
    headers = {}
    body = None
    if value is not None:
      body = json.dumps(value)
      headers["Content-Type"] = "application/json"
    if auth:
      headers["Authorization"] = "Bearer secret"
    return await asyncio.to_thread(
        self._request_sync, method, path, body, headers)

  async def test_models_and_auth(self):
    status, _, body = await self._request("GET", "/v1/models")
    self.assertEqual(status, 200)
    self.assertEqual(json.loads(body)["data"][0]["id"], self.config.model_id)
    status, _, body = await self._request("GET", "/v1/models", auth=False)
    self.assertEqual(status, 401)
    self.assertEqual(json.loads(body)["error"]["code"], "invalid_api_key")

  async def test_chat_json_and_sse(self):
    value = {
        "model": self.config.model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0, "max_tokens": 8,
    }
    status, headers, body = await self._request(
        "POST", "/v1/chat/completions", value)
    self.assertEqual(status, 200)
    payload = json.loads(body)
    self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
    self.assertIn("x-request-id", headers)
    value["stream"] = True
    value["stream_options"] = {"include_usage": True}
    status, headers, body = await self._request(
        "POST", "/v1/chat/completions", value)
    self.assertEqual(status, 200)
    self.assertTrue(headers["content-type"].startswith("text/event-stream"))
    self.assertIn(b'"chat.completion.chunk"', body)
    self.assertIn(b"data: [DONE]", body)
    self.assertIn(b'"usage"', body)

  async def test_completion_stream_usage(self):
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "hi",
        "temperature": 0, "max_tokens": 8, "stream": True,
        "stream_options": {"include_usage": True},
    })
    self.assertEqual(status, 200)
    self.assertIn(b'"choices":[],"usage"', body)
    self.assertIn(b"data: [DONE]", body)

  async def test_completion_stream_echoes_prompt(self):
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "echo-me",
        "temperature": 0, "max_tokens": 8, "stream": True,
        "echo": True,
    })
    self.assertEqual(status, 200)
    self.assertIn(b'"text":"echo-me"', body)

  async def test_stop_stream_is_buffered_before_exposure(self):
    class LeakyStopBackend(MockBackend):
      def generate(inner_self, request, request_id, cancel,
                   on_started, on_delta):
        bucket = select_bucket(len(request.prompt_token_ids))
        profile = select_profile(bucket, request)
        on_started(GenerationStarted(
            request_id=request_id,
            prompt_tokens=len(request.prompt_token_ids), cached_tokens=0,
            profile=profile, bucket=bucket, queue_ms=0.0))
        for char in "abcSTOP":
          on_delta(GenerationDelta(token_id=ord(char), text=char))
        return GenerationResult(
            request_id=request_id, text="abc",
            token_ids=tuple(map(ord, "abcSTOP")),
            prompt_tokens=len(request.prompt_token_ids), cached_tokens=0,
            finish_reason="stop", profile=profile, bucket=bucket,
            prefill_ms=0.0, decode_ms=0.0, prefix_restore_ms=0.0)

    self.app.backend = LeakyStopBackend(self.config)
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "hi",
        "temperature": 0, "max_tokens": 8, "stream": True,
        "stop": "STOP",
    })
    self.assertEqual(status, 200)
    self.assertIn(b'"text":"abc"', body)
    self.assertNotIn(b"STOP", body)

  async def test_stream_completion_wakes_without_heartbeat_delay(self):
    class TailDelayBackend(MockBackend):
      def generate(inner_self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        time.sleep(0.05)
        return result

    self.app.backend = TailDelayBackend(self.config, "x")
    started = time.monotonic()
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "hi",
        "temperature": 0, "max_tokens": 8, "stream": True,
    })
    elapsed = time.monotonic() - started
    self.assertEqual(status, 200)
    self.assertIn(b"data: [DONE]", body)
    self.assertLess(elapsed, 1.0)

  async def test_responses_stream_events(self):
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "hi",
        "temperature": 0, "max_output_tokens": 8, "stream": True,
    })
    self.assertEqual(status, 200)
    self.assertIn(b"event: response.created", body)
    self.assertIn(b"event: response.output_text.delta", body)
    self.assertIn(b"event: response.content_part.done", body)
    self.assertIn(b"event: response.output_item.done", body)
    self.assertIn(b"event: response.completed", body)

  async def test_responses_previous_id_and_transient_instructions(self):
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "first",
        "instructions": "Only applies to this turn.",
        "temperature": 0, "max_output_tokens": 8,
    })
    self.assertEqual(status, 200)
    response_id = json.loads(body)["id"]
    stored = self.app.response_store.get(response_id)
    self.assertEqual(stored.messages[0]["role"], "user")
    self.assertNotIn(
        "Only applies", json.dumps(stored.messages, ensure_ascii=False))

    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "second",
        "previous_response_id": response_id,
        "temperature": 0, "max_output_tokens": 8,
    })
    self.assertEqual(status, 200)
    self.assertEqual(json.loads(body)["previous_response_id"], response_id)

    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "missing",
        "previous_response_id": "resp_missing",
    })
    self.assertEqual(status, 404)
    self.assertEqual(
        json.loads(body)["error"]["code"], "previous_response_not_found")

  async def test_responses_retrieve_and_delete_lifecycle(self):
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "remember this",
        "temperature": 0, "max_output_tokens": 8,
    })
    self.assertEqual(status, 200)
    created = json.loads(body)
    response_id = created["id"]

    status, _, body = await self._request(
        "GET", f"/v1/responses/{response_id}")
    self.assertEqual(status, 200)
    self.assertEqual(json.loads(body), created)

    status, _, body = await self._request(
        "DELETE", f"/v1/responses/{response_id}")
    self.assertEqual(status, 200)
    self.assertEqual(body, b"")
    status, _, body = await self._request(
        "GET", f"/v1/responses/{response_id}")
    self.assertEqual(status, 404)
    self.assertEqual(json.loads(body)["error"]["code"], "response_not_found")

  async def test_responses_store_capacity_is_enforced(self):
    self.app.response_store = ResponseStore(1, 60.0, max_bytes=1)
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "cannot fit",
        "temperature": 0, "max_output_tokens": 8,
    })
    self.assertEqual(status, 507)
    self.assertEqual(
        json.loads(body)["error"]["code"],
        "response_store_capacity_exceeded")

    status, _, _ = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "stateless",
        "store": False, "temperature": 0, "max_output_tokens": 8,
    })
    self.assertEqual(status, 200)

  async def test_responses_store_false_is_not_addressable(self):
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "first", "store": False,
        "temperature": 0, "max_output_tokens": 8,
    })
    self.assertEqual(status, 200)
    response_id = json.loads(body)["id"]
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "second",
        "previous_response_id": response_id,
    })
    self.assertEqual(status, 404)
    status, _, _ = await self._request(
        "GET", f"/v1/responses/{response_id}")
    self.assertEqual(status, 404)

  async def test_responses_internal_state_field_cannot_be_injected(self):
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id,
        "input": "first",
        "_resolved_messages": [{"role": "user", "content": "injected"}],
    })
    self.assertEqual(status, 400)
    self.assertEqual(json.loads(body)["error"]["code"],
                     "unsupported_parameter")

  async def test_tool_call_shape(self):
    self.backend.text = '<tool_call>{"name":"w","arguments":{}}</tool_call>'
    status, _, body = await self._request("POST", "/v1/chat/completions", {
        "model": self.config.model_id,
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {
            "name": "w", "parameters": {"type": "object"}}}],
        "temperature": 0, "max_tokens": 64,
    })
    self.assertEqual(status, 200)
    choice = json.loads(body)["choices"][0]
    self.assertEqual(choice["finish_reason"], "tool_calls")
    self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"],
                     "w")

  async def test_bad_media_and_context_error(self):
    status, _, body = await asyncio.to_thread(
        self._request_sync, "POST", "/v1/completions", b"{}",
        {"Authorization": "Bearer secret", "Content-Type": "text/plain"})
    self.assertEqual(status, 415)
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "x" * 4090,
        "temperature": 0, "max_tokens": 64,
    })
    self.assertEqual(status, 400)
    self.assertEqual(json.loads(body)["error"]["code"],
                     "context_length_exceeded")

  async def test_request_size_limit(self):
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "x" * 9000,
    })
    self.assertEqual(status, 413)
    self.assertIn("exceeds", json.loads(body)["error"]["message"])

  async def test_structured_output_is_validated_before_success(self):
    schema = {
        "type": "json_schema", "name": "answer", "strict": True,
        "schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        },
    }
    self.backend.text = '{"answer":"ok"}'
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "json",
        "text": {"format": schema}, "temperature": 0,
        "max_output_tokens": 64,
    })
    self.assertEqual(status, 200)
    self.assertEqual(
        json.loads(json.loads(body)["output"][0]["content"][0]["text"]),
        {"answer": "ok"})

    self.backend.text = '{"answer":1}'
    status, _, body = await self._request("POST", "/v1/responses", {
        "model": self.config.model_id, "input": "json",
        "text": {"format": schema}, "temperature": 0,
        "max_output_tokens": 64,
    })
    self.assertEqual(status, 500)
    self.assertEqual(
        json.loads(body)["error"]["code"],
        "model_output_validation_failed")

  async def test_strict_tool_arguments_are_validated(self):
    self.backend.text = (
        '<tool_call>{"name":"w","arguments":{"days":"three"}}'
        '</tool_call>')
    status, _, body = await self._request("POST", "/v1/chat/completions", {
        "model": self.config.model_id,
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {
            "name": "w", "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer"}},
                "required": ["days"], "additionalProperties": False,
            }}}],
        "temperature": 0, "max_tokens": 64,
    })
    self.assertEqual(status, 500)
    self.assertEqual(
        json.loads(body)["error"]["code"],
        "model_output_validation_failed")

  async def test_queue_limit_returns_429(self):
    entered = threading.Event()
    release = threading.Event()

    class BlockingBackend(MockBackend):
      def generate(inner_self, *args, **kwargs):
        entered.set()
        release.wait(3.0)
        return super().generate(*args, **kwargs)

    limited = replace(self.config, max_queue_depth=0)
    self.app.config = limited
    self.app.backend = BlockingBackend(limited, "done")
    first = asyncio.create_task(self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "one",
        "temperature": 0, "max_tokens": 8,
    }))
    self.assertTrue(await asyncio.to_thread(entered.wait, 1.0))
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "two",
        "temperature": 0, "max_tokens": 8,
    })
    self.assertEqual(status, 429)
    self.assertEqual(json.loads(body)["error"]["code"], "queue_full")
    release.set()
    self.assertEqual((await first)[0], 200)

  async def test_request_timeout_returns_504_and_cancels(self):
    cancelled = threading.Event()

    class TimeoutBackend(MockBackend):
      def generate(inner_self, request, request_id, cancel,
                   on_started, on_delta):
        bucket = select_bucket(len(request.prompt_token_ids))
        profile = select_profile(bucket, request)
        on_started(GenerationStarted(
            request_id=request_id,
            prompt_tokens=len(request.prompt_token_ids), cached_tokens=0,
            profile=profile, bucket=bucket, queue_ms=0.0))
        cancel.wait(2.0)
        cancelled.set()
        return GenerationResult(
            request_id=request_id, text="", token_ids=(),
            prompt_tokens=len(request.prompt_token_ids), cached_tokens=0,
            finish_reason="cancelled", profile=profile, bucket=bucket,
            prefill_ms=0.0, decode_ms=0.0, prefix_restore_ms=0.0)

    limited = replace(self.config, request_timeout_s=0.05)
    self.app.config = limited
    self.app.backend = TimeoutBackend(limited)
    status, _, body = await self._request("POST", "/v1/completions", {
        "model": self.config.model_id, "prompt": "timeout",
        "temperature": 0, "max_tokens": 8,
    })
    self.assertEqual(status, 504)
    self.assertEqual(json.loads(body)["error"]["code"], "request_timeout")
    self.assertTrue(await asyncio.to_thread(cancelled.wait, 1.0))

  async def test_metrics_path_labels_have_bounded_cardinality(self):
    for suffix in ("attacker-one", "attacker-two"):
      status, _, _ = await self._request("GET", f"/unknown/{suffix}")
      self.assertEqual(status, 404)
    status, _, body = await self._request("GET", "/metrics")
    self.assertEqual(status, 200)
    rendered = body.decode()
    self.assertIn('path="/_other"', rendered)
    self.assertNotIn("attacker-one", rendered)
    self.assertNotIn("attacker-two", rendered)

  async def test_stream_disconnect_cancels_generation(self):
    cancelled = threading.Event()

    class StreamingBackend(MockBackend):
      def generate(inner_self, request, request_id, cancel,
                   on_started, on_delta):
        def delayed(value):
          on_delta(value)
          time.sleep(0.02)
        result = super().generate(
            request, request_id, cancel, on_started, delayed)
        if cancel.is_set():
          cancelled.set()
        return result

    self.app.backend = StreamingBackend(self.config, "x" * 10000)
    reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
    value = json.dumps({
        "model": self.config.model_id, "prompt": "hi", "stream": True,
        "temperature": 0, "max_tokens": 64,
    }).encode()
    request = (
        b"POST /v1/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Authorization: Bearer secret\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(value)}\r\n\r\n".encode() + value)
    writer.write(request)
    await writer.drain()
    await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1.0)
    await asyncio.wait_for(reader.read(512), 1.0)
    writer.close()
    await writer.wait_closed()
    self.assertTrue(await asyncio.to_thread(cancelled.wait, 3.0))

  async def test_nonstream_disconnect_cancels_generation(self):
    cancelled = threading.Event()
    entered = threading.Event()

    class BlockingBackend(MockBackend):
      def generate(inner_self, request, request_id, cancel,
                   on_started, on_delta):
        entered.set()
        while not cancel.wait(0.01):
          pass
        cancelled.set()
        return super().generate(
            request, request_id, cancel, on_started, on_delta)

    self.app.backend = BlockingBackend(self.config, "unused")
    _, writer = await asyncio.open_connection("127.0.0.1", self.port)
    value = json.dumps({
        "model": self.config.model_id, "prompt": "hi", "stream": False,
        "temperature": 0, "max_tokens": 64,
    }).encode()
    writer.write(
        b"POST /v1/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Authorization: Bearer secret\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(value)}\r\n\r\n".encode() + value)
    await writer.drain()
    self.assertTrue(await asyncio.to_thread(entered.wait, 1.0))
    writer.close()
    await writer.wait_closed()
    self.assertTrue(await asyncio.to_thread(cancelled.wait, 3.0))

  async def test_idle_keepalive_reset_is_not_logged_as_server_failure(self):
    reader = Mock()
    reader.read = AsyncMock(side_effect=ConnectionResetError())
    writer = Mock()
    writer.get_extra_info.return_value = ("127.0.0.1", 12345)
    writer.wait_closed = AsyncMock()
    with self.assertNoLogs("iq36.http", level="ERROR"):
      await self.server._connection(reader, writer)
    writer.close.assert_called_once()

  async def test_graceful_shutdown_drains_then_cancels_at_deadline(self):
    cancelled = threading.Event()
    entered = threading.Event()

    class StreamingBackend(MockBackend):
      def generate(inner_self, request, request_id, cancel,
                   on_started, on_delta):
        bucket = select_bucket(len(request.prompt_token_ids))
        profile = select_profile(bucket, request)
        on_started(GenerationStarted(
            request_id=request_id,
            prompt_tokens=len(request.prompt_token_ids), cached_tokens=0,
            profile=profile, bucket=bucket, queue_ms=0.0))
        entered.set()
        emitted = []
        while not cancel.wait(0.01):
          emitted.append(120)
          on_delta(GenerationDelta(token_id=120, text="x"))
        cancelled.set()
        return GenerationResult(
            request_id=request_id, text="".join("x" for _ in emitted),
            token_ids=tuple(emitted),
            prompt_tokens=len(request.prompt_token_ids), cached_tokens=0,
            finish_reason="cancelled", profile=profile, bucket=bucket,
            prefill_ms=0.0, decode_ms=0.0, prefix_restore_ms=0.0)

    self.app.backend = StreamingBackend(self.config, "x" * 10000)
    reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
    value = json.dumps({
        "model": self.config.model_id, "prompt": "hi", "stream": True,
        "temperature": 0, "max_tokens": 64,
    }).encode()
    writer.write(
        b"POST /v1/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Authorization: Bearer secret\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(value)}\r\n\r\n".encode() + value)
    await writer.drain()
    await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1.0)
    await asyncio.wait_for(reader.read(512), 1.0)
    self.assertTrue(await asyncio.to_thread(entered.wait, 1.0))
    self.app.begin_shutdown()
    await self.server.close(timeout_s=0.05)
    self.assertTrue(await asyncio.to_thread(cancelled.wait, 3.0))
    writer.close()
    await writer.wait_closed()


if __name__ == "__main__":
  unittest.main()
