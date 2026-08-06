from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .backend import MockBackend, ResidentBackend, WorkerError
from .config import ServerConfig
from .http_types import HTTPRequest, HTTPResponse
from .json_utils import strict_json_loads
from .metrics import Metrics
from .model_identity import MODEL_CONTRACT_RELATIVE, verify_model_identity
from .protocol import (
    APIError, decode_json_object, prepare_chat_completion, prepare_completion,
    prepare_response, responses_input_messages)
from .response_store import ResponseStore
from .runtime_identity import verify_imported_runtime
from .tokenizer import SimpleTestTokenizer, TokenizerAdapter
from .types import (
    GenerationDelta, GenerationResult, GenerationStarted, PreparedRequest,
    ToolCall)


LOGGER = logging.getLogger("iq36.app")


def _json_bytes(value: Any) -> bytes:
  return json.dumps(
      value, ensure_ascii=False, separators=(",", ":"),
      allow_nan=False).encode("utf-8")


def _sse(value: Any, event: str | None = None) -> bytes:
  data = value if isinstance(value, str) else _json_bytes(value).decode("utf-8")
  prefix = f"event: {event}\n" if event else ""
  return (prefix + "data: " + data + "\n\n").encode("utf-8")


def _usage(result: GenerationResult) -> dict[str, Any]:
  completion = len(result.token_ids)
  return {
      "prompt_tokens": result.prompt_tokens,
      "completion_tokens": completion,
      "total_tokens": result.prompt_tokens + completion,
      "prompt_tokens_details": {"cached_tokens": result.cached_tokens},
      "completion_tokens_details": {"reasoning_tokens": 0},
  }


def _responses_usage(result: GenerationResult) -> dict[str, Any]:
  output = len(result.token_ids)
  return {
      "input_tokens": result.prompt_tokens,
      "input_tokens_details": {
          "cached_tokens": result.cached_tokens, "cache_write_tokens": 0},
      "output_tokens": output,
      "output_tokens_details": {"reasoning_tokens": 0},
      "total_tokens": result.prompt_tokens + output,
  }


def _response_logprobs(result: GenerationResult) -> list[dict[str, Any]]:
  return [
      {
          "token": item.token, "logprob": item.logprob,
          "bytes": list(item.bytes),
          "top_logprobs": list(item.top_logprobs),
      } for item in result.logprobs
  ]


def _tool_json(call: ToolCall, index: int | None = None) -> dict[str, Any]:
  value = {
      "id": call.id,
      "type": "function",
      "function": {"name": call.name, "arguments": call.arguments},
  }
  if index is not None:
    value["index"] = index
  return value


def _chat_finish(result: GenerationResult) -> str:
  if result.tool_calls:
    return "tool_calls"
  return "length" if result.finish_reason == "length" else "stop"


def _chat_logprobs(result: GenerationResult) -> dict[str, Any] | None:
  if not result.logprobs:
    return None
  return {"content": [
      {
          "token": item.token, "logprob": item.logprob,
          "bytes": list(item.bytes), "top_logprobs": list(item.top_logprobs),
      } for item in result.logprobs
  ], "refusal": None}


def _completion_logprobs(result: GenerationResult) -> dict[str, Any] | None:
  if not result.logprobs:
    return None
  offset = 0
  text_offsets = []
  tokens = []
  token_logprobs = []
  top_logprobs = []
  for item in result.logprobs:
    text_offsets.append(offset)
    tokens.append(item.token)
    token_logprobs.append(item.logprob)
    top_logprobs.append({
        str(row.get("token", "")): float(row.get("logprob", 0.0))
        for row in item.top_logprobs})
    offset += len(item.token)
  return {
      "text_offset": text_offsets, "token_logprobs": token_logprobs,
      "tokens": tokens, "top_logprobs": top_logprobs,
  }


def _response_object(
    response_id: str, created: int, request: PreparedRequest,
    result: GenerationResult,
) -> dict[str, Any]:
  output: list[dict[str, Any]] = []
  if result.text or not result.tool_calls:
    output.append({
        "id": f"msg_{response_id[5:]}", "type": "message",
        "status": "completed", "role": "assistant",
        "content": [{
            "type": "output_text", "text": result.text,
            "annotations": [], "logprobs": _response_logprobs(result),
        }],
    })
  for index, call in enumerate(result.tool_calls):
    output.append({
        "id": f"fc_{response_id[5:]}_{index}",
        "type": "function_call", "status": "completed",
        "call_id": call.id, "name": call.name,
        "arguments": call.arguments,
    })
  incomplete = result.finish_reason == "length"
  return {
      "id": response_id, "object": "response", "created_at": created,
      "status": "incomplete" if incomplete else "completed",
      "background": False,
      "completed_at": None if incomplete else int(time.time()),
      "error": None,
      "incomplete_details": (
          {"reason": "max_output_tokens"} if incomplete else None),
      "instructions": request.request_metadata.get("instructions"),
      "max_output_tokens": request.params.max_new_tokens,
      "model": request.model, "output": output,
      "parallel_tool_calls": request.parallel_tool_calls,
      "previous_response_id": request.previous_response_id,
      "reasoning": {"effort": None, "summary": None},
      "store": request.store, "temperature": request.params.temperature,
      "text": {"format": request.response_format or {"type": "text"}},
      "tool_choice": request.tool_choice or "auto",
      "tools": [
          {
              "type": "function", **tool["function"],
          } for tool in request.tools
      ],
      "top_p": request.params.top_p,
      "truncation": "disabled", "usage": _responses_usage(result),
      "user": request.user,
      "metadata": request.request_metadata.get("response_metadata", {}),
  }


def _response_shell(
    response_id: str, created: int, request: PreparedRequest,
) -> dict[str, Any]:
  return {
      "id": response_id, "object": "response", "created_at": created,
      "status": "in_progress", "background": False,
      "completed_at": None, "error": None, "incomplete_details": None,
      "instructions": request.request_metadata.get("instructions"),
      "max_output_tokens": request.params.max_new_tokens,
      "model": request.model, "output": [],
      "parallel_tool_calls": request.parallel_tool_calls,
      "previous_response_id": request.previous_response_id,
      "reasoning": {"effort": None, "summary": None},
      "store": request.store, "temperature": request.params.temperature,
      "text": {"format": request.response_format or {"type": "text"}},
      "tool_choice": request.tool_choice or "auto",
      "tools": [
          {"type": "function", **tool["function"]}
          for tool in request.tools
      ],
      "top_p": request.params.top_p, "truncation": "disabled",
      "usage": None, "user": request.user,
      "metadata": request.request_metadata.get("response_metadata", {}),
  }


@dataclass
class _Session:
  request_id: str
  created: int
  request: PreparedRequest
  cancel: threading.Event
  events: asyncio.Queue
  future: asyncio.Future
  deadline: float
  started: GenerationStarted | None = None
  finalized: bool = False
  error_reported: bool = False
  terminal_error: APIError | None = None
  response_object: dict[str, Any] | None = None
  disconnect_watcher: asyncio.Task | None = None


class Application:
  def __init__(self, config: ServerConfig, *, backend=None, tokenizer=None) -> None:
    self.config = config
    self.tokenizer = tokenizer or (
        SimpleTestTokenizer() if config.backend == "mock" else None)
    self.backend = backend or (
        MockBackend(config) if config.backend == "mock"
        else ResidentBackend(config))
    self.metrics = Metrics()
    self.response_store = ResponseStore(
        config.response_store_entries, config.response_store_ttl_s,
        max_bytes=config.response_store_bytes)
    self._admission_lock = asyncio.Lock()
    self._admitted = 0
    self._closing = False
    self._created_model = int(time.time())
    self.model_identity: dict[str, Any] | None = None
    self.runtime_identity: dict[str, Any] | None = None

  async def start(self) -> None:
    if self.config.backend == "openvino":
      self.runtime_identity = await asyncio.to_thread(verify_imported_runtime)
      self.model_identity = await asyncio.to_thread(
          verify_model_identity, self.config.model_dir,
          self.config.repo_root / MODEL_CONTRACT_RELATIVE,
          self.config.model_verification)
      LOGGER.info(
          "model identity ready mode=%s sha256_verified=%s "
          "fingerprint=%s files=%s bytes=%s elapsed_ms=%.3f",
          self.model_identity["mode"],
          self.model_identity["sha256_verified"],
          self.model_identity["model_fingerprint"],
          self.model_identity["files_verified"],
          self.model_identity["bytes_verified"],
          self.model_identity["elapsed_ms"])
      if self.tokenizer is None:
        self.tokenizer = await asyncio.to_thread(
            TokenizerAdapter, self.config.model_dir)
    await asyncio.to_thread(self.backend.start)

  def begin_shutdown(self) -> None:
    self._closing = True

  async def close(self) -> None:
    self.begin_shutdown()
    self.response_store.clear()
    await asyncio.to_thread(self.backend.close)

  def _authenticated(self, request: HTTPRequest) -> bool:
    if self.config.api_key is None:
      return True
    value = request.headers.get("authorization", "")
    if not value.lower().startswith("bearer "):
      return False
    return hmac.compare_digest(value[7:].strip(), self.config.api_key)

  def _json_response(
      self, value: Any, status: int = 200,
      headers: list[tuple[str, str]] | None = None,
  ) -> HTTPResponse:
    return HTTPResponse(
        status=status,
        headers=[("content-type", "application/json; charset=utf-8"),
                 *(headers or [])],
        body=_json_bytes(value))

  def _error_response(self, error: APIError) -> HTTPResponse:
    if not error.metric_reported:
      self.metrics.error(error.code or error.error_type)
      error.metric_reported = True
    return self._json_response(error.payload(), error.status)

  async def _admit(self) -> None:
    async with self._admission_lock:
      capacity = 1 + self.config.max_queue_depth
      if self._closing:
        raise APIError(
            "The service is shutting down.", status=503,
            error_type="server_error", code="server_shutting_down")
      if self._admitted >= capacity:
        raise APIError(
            "The batch-1 inference queue is full. Retry later.", status=429,
            error_type="rate_limit_error", code="queue_full")
      self._admitted += 1
      self.metrics.admitted()

  async def _release(self) -> None:
    async with self._admission_lock:
      self._admitted = max(0, self._admitted - 1)

  def _report_session_error(
      self, session: _Session, kind: str, error: APIError | None = None,
  ) -> None:
    if not session.error_reported:
      session.error_reported = True
      self.metrics.error(kind)
    if error is not None:
      error.metric_reported = True

  def _finalize_session(
      self, session: _Session, result: GenerationResult,
  ) -> None:
    if session.finalized:
      if session.terminal_error is not None:
        raise session.terminal_error
      return
    try:
      self._validate_result(session.request, result)
      if session.request.endpoint == "responses":
        session.response_object = _response_object(
            session.request_id, session.created, session.request, result)
        if (
            session.request.store and
            result.finish_reason not in ("cancelled", "error")
        ):
          assistant: dict[str, Any] = {
              "role": "assistant", "content": result.text,
              "reasoning_content": "",
          }
          if result.tool_calls:
            assistant["tool_calls"] = [
                _tool_json(call) for call in result.tool_calls]
          if not self.response_store.put(
              session.request_id,
              list(session.request.request_metadata.get("messages", [])),
              assistant, session.response_object,
          ):
            raise APIError(
                "The response could not be retained within the configured "
                "response-store capacity. Retry with store=false or increase "
                "the response-store bounds.",
                status=507, error_type="server_error",
                code="response_store_capacity_exceeded")
    except APIError as error:
      session.finalized = True
      session.terminal_error = error
      self.metrics.failed(
          session.request.endpoint, started=True,
          kind=error.code or "response_finalization_error")
      self._report_session_error(session, error.code or "finalization_error", error)
      raise
    session.finalized = True
    self.metrics.completed(session.request.endpoint, result)

  @staticmethod
  def _validate_result(
      request: PreparedRequest, result: GenerationResult,
  ) -> None:
    def model_error(message: str) -> APIError:
      return APIError(
          message, status=500, error_type="server_error",
          code="model_output_validation_failed")

    calls = result.tool_calls
    tools = {tool["function"]["name"]: tool["function"]
             for tool in request.tools}
    if len(calls) > 1 and not request.parallel_tool_calls:
      raise model_error(
          "The model emitted multiple function calls while "
          "parallel_tool_calls was false.")
    for call in calls:
      tool = tools.get(call.name)
      if tool is None:
        raise model_error(
            f"The model emitted an unavailable function '{call.name}'.")
      try:
        arguments = strict_json_loads(call.arguments)
      except (json.JSONDecodeError, ValueError) as error:
        raise model_error(
            f"The model emitted invalid JSON arguments for '{call.name}'.") \
            from error
      if tool.get("strict") is True:
        try:
          Draft202012Validator(tool["parameters"]).validate(arguments)
        except ValidationError as error:
          raise model_error(
              f"The model emitted arguments for '{call.name}' that do not "
              f"satisfy its strict schema: {error.message}") from error
    choice = request.tool_choice
    if choice == "none" and calls:
      raise model_error(
          "The model emitted a function call while tool_choice was 'none'.")
    if choice == "required" and not calls:
      raise model_error(
          "The model did not emit a function call required by tool_choice.")
    if isinstance(choice, dict):
      function = choice.get("function")
      expected = function.get("name") if isinstance(function, dict) else None
      if not calls or any(call.name != expected for call in calls):
        raise model_error(
            f"The model did not follow the named tool choice '{expected}'.")

    response_format = request.response_format
    if response_format is None or calls:
      return
    kind = response_format.get("type")
    if kind not in ("json_object", "json_schema"):
      return
    try:
      value = strict_json_loads(result.text)
    except (json.JSONDecodeError, ValueError) as error:
      raise model_error(
          "The model did not produce valid JSON for the requested response "
          "format.") from error
    if kind == "json_object" and not isinstance(value, dict):
      raise model_error(
          "The model produced valid JSON, but not a JSON object as requested.")
    if kind == "json_schema":
      descriptor = (
          response_format if "schema" in response_format
          else response_format.get("json_schema", {}))
      try:
        Draft202012Validator(descriptor["schema"]).validate(value)
      except ValidationError as error:
        raise model_error(
            "The model output does not satisfy the requested JSON Schema: "
            f"{error.message}") from error

  async def _future_done(
      self, session: _Session, done: asyncio.Future, started: bool,
  ) -> None:
    try:
      result = done.result()
    except asyncio.CancelledError:
      if not session.finalized:
        session.finalized = True
        self.metrics.failed(
            session.request.endpoint, started=started, kind="cancelled")
    except Exception:
      if not session.finalized:
        session.finalized = True
        self.metrics.failed(
            session.request.endpoint, started=started, kind="error")
      self._report_session_error(session, "backend_error")
    else:
      try:
        self._finalize_session(session, result)
      except APIError:
        pass
    finally:
      if session.disconnect_watcher is not None:
        session.disconnect_watcher.cancel()
        await asyncio.gather(
            session.disconnect_watcher, return_exceptions=True)
      await self._release()

  def _new_id(self, endpoint: str) -> str:
    prefix = {
        "chat.completions": "chatcmpl-",
        "completions": "cmpl-",
        "responses": "resp_",
    }[endpoint]
    return prefix + secrets.token_hex(12)

  async def _session(
      self, request: PreparedRequest,
      disconnect_event: asyncio.Event | None = None,
  ) -> _Session:
    await self._admit()
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    request_id = self._new_id(request.endpoint)
    cancel = threading.Event()
    holder: dict[str, Any] = {"started": False}

    def on_started(value: GenerationStarted) -> None:
      holder["started"] = True
      self.metrics.started(value)
      loop.call_soon_threadsafe(events.put_nowait, ("started", value))

    def on_delta(value: GenerationDelta) -> None:
      loop.call_soon_threadsafe(events.put_nowait, ("delta", value))

    try:
      future = loop.run_in_executor(
          None, self.backend.generate, request, request_id, cancel,
          on_started, on_delta)
    except Exception:
      self.metrics.failed(
          request.endpoint, started=False, kind="submission_error")
      await self._release()
      raise
    session = _Session(
        request_id=request_id, created=int(time.time()), request=request,
        cancel=cancel, events=events, future=future,
        deadline=time.monotonic() + self.config.request_timeout_s)

    if disconnect_event is not None:
      async def watch_disconnect() -> None:
        await disconnect_event.wait()
        cancel.set()
      session.disconnect_watcher = asyncio.create_task(watch_disconnect())

    def released(done: asyncio.Future) -> None:
      events.put_nowait(("future_done", None))
      asyncio.create_task(
          self._future_done(session, done, bool(holder["started"])))

    future.add_done_callback(released)
    return session

  async def _result(self, session: _Session) -> GenerationResult:
    try:
      remaining = max(0.0, session.deadline - time.monotonic())
      result = await asyncio.wait_for(
          asyncio.shield(session.future), remaining)
    except asyncio.TimeoutError as error:
      session.cancel.set()
      api_error = APIError(
          "Inference exceeded the configured request timeout.", status=504,
          error_type="server_error", code="request_timeout")
      self._report_session_error(session, "request_timeout", api_error)
      raise api_error from error
    except WorkerError as error:
      api_error = APIError(
          "The inference backend failed. See the server log with this request "
          f"ID: {session.request_id}.", status=500,
          error_type="server_error", code="backend_error")
      self._report_session_error(session, "backend_error", api_error)
      raise api_error from error
    except Exception as error:
      api_error = APIError(
          "The inference backend failed. See the server log with this request "
          f"ID: {session.request_id}.", status=500,
          error_type="server_error", code="backend_error")
      self._report_session_error(session, "backend_error", api_error)
      raise api_error from error
    self._finalize_session(session, result)
    return result

  def _chat_json(
      self, session: _Session, result: GenerationResult,
  ) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": result.text if result.text else None,
        "refusal": None,
    }
    if result.tool_calls:
      message["tool_calls"] = [
          _tool_json(call) for call in result.tool_calls]
    return {
        "id": session.request_id, "object": "chat.completion",
        "created": session.created, "model": session.request.model,
        "choices": [{
            "index": 0, "message": message,
            "logprobs": _chat_logprobs(result),
            "finish_reason": _chat_finish(result),
        }],
        "usage": _usage(result),
        "system_fingerprint": (
            f"iq36-{result.profile}-{result.bucket}"),
    }

  def _completion_json(
      self, session: _Session, result: GenerationResult,
  ) -> dict[str, Any]:
    text = result.text
    if session.request.request_metadata.get("echo"):
      text = session.request.prompt + text
    return {
        "id": session.request_id, "object": "text_completion",
        "created": session.created, "model": session.request.model,
        "choices": [{
            "text": text, "index": 0,
            "logprobs": _completion_logprobs(result),
            "finish_reason": (
                "length" if result.finish_reason == "length" else "stop"),
        }],
        "usage": _usage(result),
        "system_fingerprint": f"iq36-{result.profile}-{result.bucket}",
    }

  async def _nonstream(self, session: _Session) -> HTTPResponse:
    result = await self._result(session)
    if session.request.endpoint == "chat.completions":
      value = self._chat_json(session, result)
    elif session.request.endpoint == "completions":
      value = self._completion_json(session, result)
    else:
      value = session.response_object
      if value is None:
        raise APIError(
            "The response was not finalized.", status=500,
            error_type="server_error", code="internal_error")
    return self._json_response(
        value, headers=[("x-request-id", session.request_id)])

  async def _next_event(self, session: _Session):
    while True:
      if session.future.done() and session.events.empty():
        return None
      remaining = session.deadline - time.monotonic()
      if remaining <= 0:
        session.cancel.set()
        error = APIError(
            "Inference exceeded the configured request timeout.", status=504,
            error_type="server_error", code="request_timeout")
        self._report_session_error(session, "request_timeout", error)
        raise error
      try:
        return await asyncio.wait_for(
            session.events.get(), min(15.0, remaining))
      except asyncio.TimeoutError:
        if time.monotonic() >= session.deadline:
          session.cancel.set()
          error = APIError(
              "Inference exceeded the configured request timeout.",
              status=504, error_type="server_error", code="request_timeout")
          self._report_session_error(session, "request_timeout", error)
          raise error
        return ("keepalive", None)

  async def _chat_stream(self, session: _Session):
    request = session.request
    base = {
        "id": session.request_id, "object": "chat.completion.chunk",
        "created": session.created, "model": request.model,
        "system_fingerprint": None,
    }
    yield _sse({
        **base, "choices": [{
            "index": 0, "delta": {"role": "assistant", "content": ""},
            "logprobs": None, "finish_reason": None,
        }]})
    buffer = bool(
        request.tools or request.request_metadata.get("enable_thinking") or
        request.params.stop or request.params.logprobs or
        request.response_format and
        request.response_format.get("type") != "text")
    try:
      while True:
        event = await self._next_event(session)
        if event is None:
          break
        kind, value = event
        if kind == "keepalive":
          yield b": keep-alive\n\n"
        elif kind == "started":
          session.started = value
        elif kind == "delta" and not buffer and value.text:
          logprobs = None
          if value.logprob is not None:
            logprobs = {"content": [{
                "token": value.logprob.token,
                "logprob": value.logprob.logprob,
                "bytes": list(value.logprob.bytes),
                "top_logprobs": list(value.logprob.top_logprobs),
            }]}
          yield _sse({
              **base, "choices": [{
                  "index": 0, "delta": {"content": value.text},
                  "logprobs": logprobs, "finish_reason": None,
              }]})
      result = await self._result(session)
      base["system_fingerprint"] = f"iq36-{result.profile}-{result.bucket}"
      if buffer and result.text:
        yield _sse({
            **base, "choices": [{
                "index": 0, "delta": {"content": result.text},
                "logprobs": _chat_logprobs(result), "finish_reason": None,
            }]})
      if result.tool_calls:
        yield _sse({
            **base, "choices": [{
                "index": 0,
                "delta": {"tool_calls": [
                    _tool_json(call, index)
                    for index, call in enumerate(result.tool_calls)]},
                "logprobs": None, "finish_reason": None,
            }]})
      yield _sse({
          **base, "choices": [{
              "index": 0, "delta": {}, "logprobs": None,
              "finish_reason": _chat_finish(result),
          }]})
      if request.stream_include_usage:
        yield _sse({**base, "choices": [], "usage": _usage(result)})
      yield _sse("[DONE]")
    except APIError as error:
      session.cancel.set()
      yield _sse(error.payload())
      yield _sse("[DONE]")
    except (asyncio.CancelledError, GeneratorExit):
      session.cancel.set()
      raise

  async def _completion_stream(self, session: _Session):
    request = session.request
    base = {
        "id": session.request_id, "object": "text_completion",
        "created": session.created, "model": request.model,
    }
    buffer = bool(request.params.stop or request.params.logprobs)
    text_offset = 0
    try:
      if request.request_metadata.get("echo") and request.prompt:
        yield _sse({
            **base, "choices": [{
                "text": request.prompt, "index": 0,
                "logprobs": None, "finish_reason": None,
            }]})
        text_offset = len(request.prompt)
      while True:
        event = await self._next_event(session)
        if event is None:
          break
        kind, value = event
        if kind == "keepalive":
          yield b": keep-alive\n\n"
        elif kind == "started":
          session.started = value
        elif kind == "delta" and not buffer and value.text:
          logprobs = None
          if value.logprob is not None:
            logprobs = {
                "text_offset": [text_offset],
                "token_logprobs": [value.logprob.logprob],
                "tokens": [value.logprob.token],
                "top_logprobs": [{
                    str(row.get("token", "")): float(row.get("logprob", 0.0))
                    for row in value.logprob.top_logprobs}],
            }
          yield _sse({
              **base, "choices": [{
                  "text": value.text, "index": 0,
                  "logprobs": logprobs, "finish_reason": None,
              }]})
          text_offset += len(value.text)
      result = await self._result(session)
      if buffer and result.text:
        yield _sse({
            **base, "choices": [{
                "text": result.text, "index": 0,
                "logprobs": _completion_logprobs(result),
                "finish_reason": None,
            }]})
      yield _sse({
          **base, "choices": [{
              "text": "", "index": 0, "logprobs": None,
              "finish_reason": (
                  "length" if result.finish_reason == "length" else "stop"),
          }]})
      if request.stream_include_usage:
        yield _sse({**base, "choices": [], "usage": _usage(result)})
      yield _sse("[DONE]")
    except APIError as error:
      session.cancel.set()
      yield _sse(error.payload())
      yield _sse("[DONE]")
    except (asyncio.CancelledError, GeneratorExit):
      session.cancel.set()
      raise

  async def _responses_stream(self, session: _Session):
    request = session.request
    shell = _response_shell(session.request_id, session.created, request)
    sequence = 0
    message_id = f"msg_{session.request_id[5:]}"
    yield _sse({
        "type": "response.created", "sequence_number": sequence,
        "response": shell}, "response.created")
    sequence += 1
    yield _sse({
        "type": "response.in_progress", "sequence_number": sequence,
        "response": shell}, "response.in_progress")
    sequence += 1
    buffer = bool(
        request.tools or request.request_metadata.get("enable_thinking") or
        request.params.stop or request.params.logprobs or
        request.response_format and
        request.response_format.get("type") != "text")
    item_added = False
    try:
      while True:
        event = await self._next_event(session)
        if event is None:
          break
        kind, value = event
        if kind == "keepalive":
          yield b": keep-alive\n\n"
        elif kind == "started":
          session.started = value
        elif kind == "delta" and not buffer and value.text:
          if not item_added:
            item_added = True
            yield _sse({
                "type": "response.output_item.added",
                "sequence_number": sequence, "output_index": 0,
                "item": {"id": message_id, "type": "message",
                         "status": "in_progress", "role": "assistant",
                         "content": []}}, "response.output_item.added")
            sequence += 1
            yield _sse({
                "type": "response.content_part.added",
                "sequence_number": sequence, "item_id": message_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": "",
                         "annotations": [], "logprobs": []}},
                "response.content_part.added")
            sequence += 1
          yield _sse({
              "type": "response.output_text.delta",
              "sequence_number": sequence, "item_id": message_id,
              "output_index": 0, "content_index": 0,
              "delta": value.text,
              "logprobs": ([{
                  "token": value.logprob.token,
                  "logprob": value.logprob.logprob,
                  "bytes": list(value.logprob.bytes),
                  "top_logprobs": list(value.logprob.top_logprobs),
              }] if value.logprob is not None else []),
              "obfuscation": None},
              "response.output_text.delta")
          sequence += 1
      result = await self._result(session)
      response = session.response_object
      if response is None:
        raise APIError(
            "The response was not finalized.", status=500,
            error_type="server_error", code="internal_error")
      has_message = bool(result.text or not result.tool_calls)
      if has_message:
        if not item_added:
          item_added = True
          yield _sse({
              "type": "response.output_item.added",
              "sequence_number": sequence, "output_index": 0,
              "item": {"id": message_id, "type": "message",
                       "status": "in_progress", "role": "assistant",
                       "content": []}}, "response.output_item.added")
          sequence += 1
          yield _sse({
              "type": "response.content_part.added",
              "sequence_number": sequence, "item_id": message_id,
              "output_index": 0, "content_index": 0,
              "part": {"type": "output_text", "text": "",
                       "annotations": [], "logprobs": []}},
              "response.content_part.added")
          sequence += 1
          if buffer:
            yield _sse({
                "type": "response.output_text.delta",
                "sequence_number": sequence, "item_id": message_id,
                "output_index": 0, "content_index": 0,
                "delta": result.text,
                "logprobs": _response_logprobs(result),
                "obfuscation": None},
                "response.output_text.delta")
            sequence += 1
        yield _sse({
            "type": "response.output_text.done",
            "sequence_number": sequence, "item_id": message_id,
            "output_index": 0, "content_index": 0,
            "text": result.text, "logprobs": _response_logprobs(result)},
            "response.output_text.done")
        sequence += 1
        part = {
            "type": "output_text", "text": result.text,
            "annotations": [], "logprobs": _response_logprobs(result),
        }
        yield _sse({
            "type": "response.content_part.done",
            "sequence_number": sequence, "item_id": message_id,
            "output_index": 0, "content_index": 0, "part": part},
            "response.content_part.done")
        sequence += 1
        yield _sse({
            "type": "response.output_item.done",
            "sequence_number": sequence, "output_index": 0,
            "item": {
                "id": message_id, "type": "message",
                "status": "completed", "role": "assistant",
                "content": [part],
            }}, "response.output_item.done")
        sequence += 1
      output_index = 1 if has_message else 0
      for call in result.tool_calls:
        item = {
            "id": f"fc_{session.request_id[5:]}_{output_index}",
            "type": "function_call", "status": "in_progress",
            "call_id": call.id, "name": call.name, "arguments": "",
        }
        yield _sse({
            "type": "response.output_item.added",
            "sequence_number": sequence, "output_index": output_index,
            "item": item}, "response.output_item.added")
        sequence += 1
        yield _sse({
            "type": "response.function_call_arguments.delta",
            "sequence_number": sequence, "item_id": item["id"],
            "output_index": output_index, "delta": call.arguments,
            "obfuscation": None}, "response.function_call_arguments.delta")
        sequence += 1
        yield _sse({
            "type": "response.function_call_arguments.done",
            "sequence_number": sequence, "item_id": item["id"],
            "output_index": output_index, "arguments": call.arguments},
            "response.function_call_arguments.done")
        sequence += 1
        yield _sse({
            "type": "response.output_item.done",
            "sequence_number": sequence, "output_index": output_index,
            "item": {
                **item, "status": "completed", "arguments": call.arguments,
            }}, "response.output_item.done")
        sequence += 1
        output_index += 1
      terminal = (
          "response.incomplete"
          if response["status"] == "incomplete" else "response.completed")
      yield _sse({
          "type": terminal, "sequence_number": sequence,
          "response": response}, terminal)
    except APIError as error:
      session.cancel.set()
      failed = {
          **shell, "status": "failed",
          "error": {"code": error.code or "server_error",
                    "message": error.message},
      }
      yield _sse({
          "type": "response.failed", "sequence_number": sequence,
          "response": failed}, "response.failed")
    except (asyncio.CancelledError, GeneratorExit):
      session.cancel.set()
      raise

  def _stream_response(self, session: _Session) -> HTTPResponse:
    if session.request.endpoint == "chat.completions":
      stream = self._chat_stream(session)
    elif session.request.endpoint == "completions":
      stream = self._completion_stream(session)
    else:
      stream = self._responses_stream(session)
    return HTTPResponse(
        status=200,
        headers=[
            ("content-type", "text/event-stream; charset=utf-8"),
            ("cache-control", "no-cache, no-transform"),
            ("x-accel-buffering", "no"),
            ("x-request-id", session.request_id),
        ],
        stream=stream, on_disconnect=session.cancel.set)

  def _prepare(self, path: str, payload: dict[str, Any]) -> PreparedRequest:
    if path == "/v1/chat/completions":
      return prepare_chat_completion(payload, self.config, self.tokenizer)
    if path == "/v1/completions":
      return prepare_completion(payload, self.config, self.tokenizer)
    if path == "/v1/responses":
      return prepare_response(payload, self.config, self.tokenizer)
    raise APIError(
        f"Unknown endpoint: {path}", status=404,
        error_type="invalid_request_error", code="not_found")

  def _resolve_previous_response(
      self, payload: dict[str, Any],
  ) -> dict[str, Any]:
    previous_id = payload.get("previous_response_id")
    if previous_id is None:
      return payload
    if not isinstance(previous_id, str):
      return payload
    previous = self.response_store.get(previous_id)
    if previous is None:
      raise APIError(
          f"Previous response '{previous_id}' was not found or has expired.",
          status=404, param="previous_response_id",
          code="previous_response_not_found")
    current = responses_input_messages(payload.get("input"))
    resolved = dict(payload)
    resolved["_resolved_messages"] = [
        *previous.messages, previous.assistant, *current]
    return resolved

  async def dispatch(self, request: HTTPRequest) -> HTTPResponse:
    try:
      if request.method == "OPTIONS":
        return HTTPResponse(status=204, body=b"")
      if request.query:
        raise APIError(
            "Query parameters are not supported on this endpoint.",
            status=400, code="unsupported_query_parameters")
      if request.path == "/healthz":
        return self._json_response({"status": "ok"})
      if request.path == "/readyz":
        status = self.backend.status()
        code = 200 if status.ready else 503
        return self._json_response({
            "status": "ready" if status.ready else "not_ready",
            "active": status.active,
            "loaded_workers": list(status.loaded_workers),
            "runtime_identity": self.runtime_identity,
            "model_identity": self.model_identity,
            "last_error": status.last_error,
        }, code)
      if not self._authenticated(request):
        raise APIError(
            "Incorrect API key provided.", status=401,
            error_type="invalid_request_error", code="invalid_api_key")
      if request.path == "/metrics":
        status = self.backend.status()
        return HTTPResponse(
            status=200,
            headers=[("content-type", "text/plain; version=0.0.4")],
            body=self.metrics.render(status.ready))
      if request.method == "GET" and request.path == "/v1/models":
        return self._json_response({
            "object": "list", "data": [{
                "id": self.config.model_id, "object": "model",
                "created": self._created_model, "owned_by": "intel-qwen36",
            }]})
      if request.method == "GET" and request.path.startswith("/v1/models/"):
        model = request.path[len("/v1/models/"):]
        if model != self.config.model_id:
          raise APIError(
              f"The model '{model}' does not exist.", status=404,
              param="model", code="model_not_found")
        return self._json_response({
            "id": model, "object": "model", "created": self._created_model,
            "owned_by": "intel-qwen36"})
      response_prefix = "/v1/responses/"
      if request.path.startswith(response_prefix):
        response_id = request.path[len(response_prefix):]
        if not response_id or "/" in response_id:
          raise APIError(
              f"Unknown endpoint: {request.method} {request.path}",
              status=404, code="not_found")
        if request.method == "GET":
          stored = self.response_store.get(response_id)
          if stored is None:
            raise APIError(
                f"Response '{response_id}' was not found or has expired.",
                status=404, code="response_not_found")
          return self._json_response(stored.response)
        if request.method == "DELETE":
          if not self.response_store.delete(response_id):
            raise APIError(
                f"Response '{response_id}' was not found or has expired.",
                status=404, code="response_not_found")
          return HTTPResponse(status=200, body=b"")
      if request.method != "POST" or request.path not in (
          "/v1/completions", "/v1/chat/completions", "/v1/responses"):
        raise APIError(
            f"Unknown endpoint: {request.method} {request.path}", status=404,
            code="not_found")
      content_type = request.headers.get("content-type", "")
      if not content_type.lower().startswith("application/json"):
        raise APIError(
            "Content-Type must be application/json.", status=415,
            code="unsupported_media_type")
      payload = decode_json_object(request.body)
      if request.path == "/v1/responses":
        if "_resolved_messages" in payload:
          raise APIError(
              "Unsupported parameter for responses: '_resolved_messages'.",
              param="_resolved_messages", code="unsupported_parameter")
        payload = self._resolve_previous_response(payload)
      prepared = self._prepare(request.path, payload)
      session = await self._session(prepared, request.disconnect_event)
      if prepared.stream:
        return self._stream_response(session)
      return await self._nonstream(session)
    except APIError as error:
      return self._error_response(error)
    except Exception:
      LOGGER.exception("unhandled request failure path=%s", request.path)
      self.metrics.error("internal_error")
      return self._error_response(APIError(
          "Internal server error.", status=500, error_type="server_error",
          code="internal_error"))
