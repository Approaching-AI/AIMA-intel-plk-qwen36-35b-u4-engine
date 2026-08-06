#!/usr/bin/env python3
"""Run a bounded real-service acceptance smoke against one IQ36 endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "qwen3.6-35b-a3b-u4"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/service"))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-url", default="http://127.0.0.1:8000")
  parser.add_argument("--model", default=MODEL)
  parser.add_argument(
      "--model-dir", type=Path,
      default=Path("/home/intel/Qwen3.6-35B-A3B-ov"))
  parser.add_argument("--api-key-file", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--timeout", type=float, default=180.0)
  parser.add_argument(
      "--long", action="store_true",
      help="also switch through 16k compact-greedy and full-logit profiles")
  parser.add_argument(
      "--max-context", action="store_true",
      help="also exercise an exact 131072-token prompt through HTTP")
  parser.add_argument(
      "--openai-sdk", action="store_true",
      help="also exercise JSON, streaming, tools, and state via OpenAI Python")
  args = parser.parse_args()
  if args.timeout <= 0:
    parser.error("timeout must be positive")
  return args


class Client:
  def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
    self.base_url = base_url.rstrip("/")
    self.api_key = api_key
    self.timeout = timeout

  def request(
      self, method: str, path: str, payload: dict[str, Any] | None = None,
  ) -> tuple[int, dict[str, str], bytes, float]:
    headers = {}
    body = None
    if self.api_key:
      headers["Authorization"] = "Bearer " + self.api_key
    if payload is not None:
      body = json.dumps(
          payload, ensure_ascii=False, separators=(",", ":")).encode()
      headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        self.base_url + path, data=body, headers=headers, method=method)
    started = time.monotonic()
    try:
      with urllib.request.urlopen(request, timeout=self.timeout) as response:
        value = response.read()
        return (
            response.status,
            {name.lower(): header_value
             for name, header_value in response.headers.items()},
            value,
            time.monotonic() - started)
    except urllib.error.HTTPError as error:
      return (
          error.code,
          {name.lower(): header_value
           for name, header_value in error.headers.items()},
          error.read(),
          time.monotonic() - started)


def decode_json(body: bytes) -> dict[str, Any]:
  value = json.loads(body.decode("utf-8"))
  if not isinstance(value, dict):
    raise TypeError("expected JSON object")
  return value


def parse_sse(body: bytes) -> list[dict[str, Any]]:
  events = []
  for block in body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
    if not block or block.startswith(":"):
      continue
    event_name = None
    data = []
    for line in block.splitlines():
      if line.startswith("event:"):
        event_name = line[6:].strip()
      elif line.startswith("data:"):
        data.append(line[5:].lstrip())
    if not data:
      continue
    raw = "\n".join(data)
    if raw == "[DONE]":
      events.append({"event": event_name, "data": "[DONE]"})
    else:
      events.append({"event": event_name, "data": json.loads(raw)})
  return events


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  api_key = (
      args.api_key_file.read_text(encoding="utf-8").strip()
      if args.api_key_file else os.environ.get("IQ36_API_KEY"))
  client = Client(args.base_url, api_key, args.timeout)
  checks: list[dict[str, Any]] = []

  status, _, body, elapsed = client.request("GET", "/healthz")
  health = decode_json(body)
  checks.append(check(
      "health", status == 200 and health.get("status") == "ok",
      status=status, elapsed_ms=elapsed * 1000.0, response=health))

  status, _, body, elapsed = client.request("GET", "/readyz")
  ready = decode_json(body)
  workers = ready.get("loaded_workers", [])
  model_identity = ready.get("model_identity") or {}
  checks.append(check(
      "ready_after_warmup",
      status == 200 and ready.get("status") == "ready" and
      bool(workers) and
      all(float(row.get("warmup_ms", 0)) > 0 for row in workers) and
      model_identity.get("mode") == "full" and
      model_identity.get("sha256_verified") is True and
      model_identity.get("model_fingerprint") ==
          "eb05132e47fe0fd1dc42fa3082e7241696ed1449dec246a3cc14bef4af21d7ec",
      status=status, elapsed_ms=elapsed * 1000.0, response=ready))

  status, _, body, elapsed = client.request("GET", "/v1/models")
  models = decode_json(body)
  ids = [row.get("id") for row in models.get("data", [])]
  checks.append(check(
      "models", status == 200 and args.model in ids,
      status=status, elapsed_ms=elapsed * 1000.0, model_ids=ids))

  chat_request = {
      "model": args.model,
      "messages": [{"role": "user", "content": "只回答：你好"}],
      "temperature": 0, "max_tokens": 8,
  }
  status, _, body, elapsed = client.request(
      "POST", "/v1/chat/completions", chat_request)
  chat = decode_json(body)
  chat_text = (
      chat.get("choices", [{}])[0].get("message", {}).get("content"))
  checks.append(check(
      "chat_json", status == 200 and chat_text == "你好",
      status=status, elapsed_ms=elapsed * 1000.0,
      text=chat_text, usage=chat.get("usage")))

  stream_request = {
      **chat_request,
      "messages": [{"role": "user", "content": "只回答：再见"}],
      "stream": True, "stream_options": {"include_usage": True},
  }
  status, headers, body, elapsed = client.request(
      "POST", "/v1/chat/completions", stream_request)
  chat_events = parse_sse(body)
  streamed_text = "".join(
      str(event["data"].get("choices", [{}])[0].get("delta", {}).get(
          "content", ""))
      for event in chat_events
      if isinstance(event["data"], dict) and event["data"].get("choices"))
  checks.append(check(
      "chat_sse",
      status == 200 and
      headers.get("content-type", "").startswith("text/event-stream") and
      streamed_text == "再见" and
      chat_events[-1]["data"] == "[DONE]" and elapsed < 5.0,
      status=status, elapsed_ms=elapsed * 1000.0, text=streamed_text,
      event_count=len(chat_events)))

  response_request = {
      "model": args.model, "input": "只回答：你好",
      "temperature": 0, "max_output_tokens": 8, "stream": True,
  }
  status, _, body, elapsed = client.request(
      "POST", "/v1/responses", response_request)
  response_events = parse_sse(body)
  event_types = [
      event["data"].get("type") for event in response_events
      if isinstance(event["data"], dict)]
  lifecycle = [
      event_type for index, event_type in enumerate(event_types)
      if index == 0 or event_type != event_types[index - 1]]
  required_events = [
      "response.created", "response.in_progress",
      "response.output_item.added", "response.content_part.added",
      "response.output_text.delta", "response.output_text.done",
      "response.content_part.done", "response.output_item.done",
      "response.completed",
  ]
  checks.append(check(
      "responses_sse_lifecycle",
      status == 200 and lifecycle == required_events and elapsed < 5.0,
      status=status, elapsed_ms=elapsed * 1000.0,
      event_types=event_types, lifecycle=lifecycle))

  first_payload = {
      "model": args.model, "input": "只回答：你好",
      "instructions": "只在这一轮生效。",
      "temperature": 0, "max_output_tokens": 8,
  }
  status1, _, body1, elapsed1 = client.request(
      "POST", "/v1/responses", first_payload)
  first = decode_json(body1)
  status2, _, body2, elapsed2 = client.request(
      "POST", "/v1/responses", {
          "model": args.model, "input": "只回答：好的",
          "previous_response_id": first.get("id"),
          "temperature": 0, "max_output_tokens": 8,
      })
  second = decode_json(body2)
  checks.append(check(
      "responses_previous_response_id",
      status1 == 200 and status2 == 200 and
      second.get("previous_response_id") == first.get("id"),
      first_status=status1, second_status=status2,
      elapsed_ms=(elapsed1 + elapsed2) * 1000.0,
      previous_response_id=second.get("previous_response_id")))

  first_id = first.get("id")
  retrieve_status, _, retrieve_body, retrieve_elapsed = client.request(
      "GET", f"/v1/responses/{first_id}")
  retrieved = decode_json(retrieve_body)
  delete_status, _, delete_body, delete_elapsed = client.request(
      "DELETE", f"/v1/responses/{first_id}")
  missing_status, _, missing_body, missing_elapsed = client.request(
      "GET", f"/v1/responses/{first_id}")
  missing = decode_json(missing_body)
  checks.append(check(
      "responses_retrieve_delete_lifecycle",
      retrieve_status == 200 and retrieved == first and
      delete_status == 200 and delete_body == b"" and
      missing_status == 404 and
      missing.get("error", {}).get("code") == "response_not_found",
      retrieve_status=retrieve_status, delete_status=delete_status,
      post_delete_status=missing_status,
      elapsed_ms=(
          retrieve_elapsed + delete_elapsed + missing_elapsed) * 1000.0))

  tool_request = {
      "model": args.model,
      "messages": [{
          "role": "user", "content": "上海天气怎么样？请调用工具。"}],
      "tools": [{"type": "function", "function": {
          "name": "get_weather", "description": "查询指定城市天气",
          "parameters": {
              "type": "object",
              "properties": {"city": {"type": "string"}},
              "required": ["city"],
          }}}],
      "tool_choice": "required", "temperature": 0, "max_tokens": 64,
  }
  status, _, body, elapsed = client.request(
      "POST", "/v1/chat/completions", tool_request)
  tool_response = decode_json(body)
  choice = tool_response.get("choices", [{}])[0]
  calls = choice.get("message", {}).get("tool_calls") or []
  tool_ok = bool(
      status == 200 and choice.get("finish_reason") == "tool_calls" and
      calls and calls[0].get("function", {}).get("name") == "get_weather")
  if tool_ok:
    try:
      tool_ok = json.loads(calls[0]["function"]["arguments"]).get("city") == "上海"
    except (json.JSONDecodeError, KeyError, AttributeError):
      tool_ok = False
  checks.append(check(
      "function_tool_call", tool_ok, status=status,
      elapsed_ms=elapsed * 1000.0,
      finish_reason=choice.get("finish_reason"), call_count=len(calls)))
  if calls:
    tool_call_id = calls[0].get("id")
    followup_messages = [
        *tool_request["messages"],
        choice.get("message", {}),
        {"role": "tool", "tool_call_id": tool_call_id,
         "content": "上海天气晴，气温25℃。"},
        {"role": "user", "content": "根据工具结果简短回答。"},
    ]
    follow_status, _, follow_body, follow_elapsed = client.request(
        "POST", "/v1/chat/completions", {
            "model": args.model, "messages": followup_messages,
            "tools": tool_request["tools"],
            "temperature": 0, "max_tokens": 64,
        })
    follow_value = decode_json(follow_body)
    follow_text = follow_value.get("choices", [{}])[0].get(
        "message", {}).get("content")
  else:
    follow_status = 0
    follow_elapsed = 0.0
    follow_text = None
  checks.append(check(
      "multi_turn_tool_result",
      tool_ok and follow_status == 200 and isinstance(follow_text, str) and
      "上海" in follow_text and ("25" in follow_text or "晴" in follow_text),
      status=follow_status, elapsed_ms=follow_elapsed * 1000.0,
      text=follow_text))

  cache_request = {
      "model": args.model,
      "messages": [{
          "role": "user", "content": "只回答：缓存验证通过"}],
      "temperature": 0, "max_tokens": 16,
      "logprobs": True, "top_logprobs": 0,
  }
  cache_rows = []
  for _ in range(2):
    status, _, body, elapsed = client.request(
        "POST", "/v1/chat/completions", cache_request)
    value = decode_json(body)
    cache_rows.append({
        "status": status, "elapsed_ms": elapsed * 1000.0,
        "text": value.get("choices", [{}])[0].get("message", {}).get(
            "content"),
        "tokens": [
            row.get("token") for row in
            (value.get("choices", [{}])[0].get("logprobs", {}) or {}).get(
                "content", [])],
        "cached_tokens": value.get("usage", {}).get(
            "prompt_tokens_details", {}).get("cached_tokens"),
    })
  bypass = {**cache_request, "prefix_cache": False}
  status, _, body, elapsed = client.request(
      "POST", "/v1/chat/completions", bypass)
  bypass_value = decode_json(body)
  bypass_cached = bypass_value.get("usage", {}).get(
      "prompt_tokens_details", {}).get("cached_tokens")
  checks.append(check(
      "exact_prefix_cache_and_bypass",
      all(row["status"] == 200 for row in cache_rows) and
      cache_rows[0]["text"] == cache_rows[1]["text"] and
      cache_rows[0]["tokens"] == cache_rows[1]["tokens"] and
      int(cache_rows[1]["cached_tokens"] or 0) > 0 and
      bypass_cached == 0,
      cache_rows=cache_rows,
      bypass={"status": status, "elapsed_ms": elapsed * 1000.0,
              "cached_tokens": bypass_cached}))

  # Materialize a non-tile-aligned raw completion prompt. This proves the HTTP
  # caller is not required to know or hit a 32-token/power-of-two boundary.
  from iq36_server.tokenizer import TokenizerAdapter
  tokenizer = TokenizerAdapter(args.model_dir)
  repeat = 1
  while True:
    arbitrary_prompt = "hello " * repeat
    token_count = len(tokenizer.encode(arbitrary_prompt))
    if token_count >= 33 and token_count % 32:
      break
    repeat += 1
    if repeat > 10000:
      raise RuntimeError("could not materialize a non-aligned prompt")
  status, _, body, elapsed = client.request(
      "POST", "/v1/completions", {
          "model": args.model, "prompt": arbitrary_prompt,
          "temperature": 0, "max_tokens": 4,
      })
  arbitrary = decode_json(body)
  observed_tokens = arbitrary.get("usage", {}).get("prompt_tokens")
  checks.append(check(
      "arbitrary_non_aligned_context_length",
      status == 200 and observed_tokens == token_count,
      status=status, elapsed_ms=elapsed * 1000.0,
      expected_prompt_tokens=token_count,
      observed_prompt_tokens=observed_tokens))

  status, _, body, elapsed = client.request("GET", "/readyz")
  resident = decode_json(body)
  resident_workers = resident.get("loaded_workers", [])
  initial_pid = workers[0].get("worker_pid") if workers else None
  checks.append(check(
      "short_worker_remains_resident",
      status == 200 and len(resident_workers) == 1 and
      initial_pid is not None and
      resident_workers[0].get("worker_pid") == initial_pid and
      resident_workers[0].get("profile") == "short_full" and
      resident_workers[0].get("bucket") == 2048,
      status=status, elapsed_ms=elapsed * 1000.0,
      initial_worker_pid=initial_pid,
      current_workers=resident_workers))

  if args.long:
    repeat = 8206
    for _ in range(8):
      long_prompt = "hello " * repeat
      long_tokens = len(tokenizer.encode(long_prompt))
      if long_tokens == 8207:
        break
      repeat += 8207 - long_tokens
      if repeat < 1:
        raise RuntimeError("failed to materialize the long prompt")
    else:
      raise RuntimeError(
          f"could not materialize 8207 tokens; last count was {long_tokens}")

    status, _, body, elapsed = client.request(
        "POST", "/v1/completions", {
            "model": args.model, "prompt": long_prompt,
            "temperature": 0, "max_tokens": 4, "prefix_cache": False,
        })
    compact = decode_json(body)
    compact_fingerprint = compact.get("system_fingerprint")
    ready_status, _, ready_body, _ = client.request("GET", "/readyz")
    compact_ready = decode_json(ready_body)
    compact_workers = compact_ready.get("loaded_workers", [])
    checks.append(check(
        "long_compact_arbitrary_context",
        status == 200 and compact.get("usage", {}).get("prompt_tokens") == 8207 and
        compact_fingerprint == "iq36-long_compact-16384" and
        ready_status == 200 and len(compact_workers) == 1 and
        compact_workers[0].get("profile") == "long_compact" and
        compact_workers[0].get("plugin_sha256") ==
            "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269",
        status=status, elapsed_ms=elapsed * 1000.0,
        prompt_tokens=compact.get("usage", {}).get("prompt_tokens"),
        system_fingerprint=compact_fingerprint,
        loaded_workers=compact_workers))

    status, _, body, elapsed = client.request(
        "POST", "/v1/completions", {
            "model": args.model, "prompt": long_prompt,
            "temperature": 1, "top_p": 0.9, "seed": 42,
            "max_tokens": 4, "prefix_cache": False,
        })
    full = decode_json(body)
    full_fingerprint = full.get("system_fingerprint")
    ready_status, _, ready_body, _ = client.request("GET", "/readyz")
    full_ready = decode_json(ready_body)
    full_workers = full_ready.get("loaded_workers", [])
    checks.append(check(
        "long_full_logits_sampling_isolated",
        status == 200 and full.get("usage", {}).get("prompt_tokens") == 8207 and
        full_fingerprint == "iq36-long_full-16384" and
        ready_status == 200 and len(full_workers) == 1 and
        full_workers[0].get("profile") == "long_full" and
        full_workers[0].get("plugin_sha256") ==
            "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269",
        status=status, elapsed_ms=elapsed * 1000.0,
        prompt_tokens=full.get("usage", {}).get("prompt_tokens"),
        system_fingerprint=full_fingerprint,
        loaded_workers=full_workers))

  if args.max_context:
    repeat = 131071
    for _ in range(8):
      maximum_prompt = "hello " * repeat
      maximum_tokens = len(tokenizer.encode(maximum_prompt))
      if maximum_tokens == 131072:
        break
      repeat += 131072 - maximum_tokens
      if repeat < 1:
        raise RuntimeError("failed to materialize the maximum prompt")
    else:
      raise RuntimeError(
          "could not materialize 131072 tokens; last count was "
          f"{maximum_tokens}")
    status, _, body, elapsed = client.request(
        "POST", "/v1/completions", {
            "model": args.model, "prompt": maximum_prompt,
            "temperature": 0, "max_tokens": 4, "prefix_cache": False,
        })
    maximum = decode_json(body)
    fingerprint = maximum.get("system_fingerprint")
    ready_status, _, ready_body, _ = client.request("GET", "/readyz")
    maximum_ready = decode_json(ready_body)
    maximum_workers = maximum_ready.get("loaded_workers", [])
    checks.append(check(
        "maximum_131072_token_context",
        status == 200 and
        maximum.get("usage", {}).get("prompt_tokens") == 131072 and
        fingerprint == "iq36-long_compact-131072" and
        ready_status == 200 and len(maximum_workers) == 1 and
        maximum_workers[0].get("profile") == "long_compact" and
        maximum_workers[0].get("bucket") == 131072,
        status=status, elapsed_ms=elapsed * 1000.0,
        prompt_tokens=maximum.get("usage", {}).get("prompt_tokens"),
        system_fingerprint=fingerprint, loaded_workers=maximum_workers))

  status, _, body, elapsed = client.request("GET", "/metrics")
  metrics = body.decode("utf-8", errors="replace")
  checks.append(check(
      "prometheus_metrics",
      status == 200 and "iq36_backend_ready 1" in metrics and
      "iq36_generations_total" in metrics,
      status=status, elapsed_ms=elapsed * 1000.0))

  sdk_version = None
  if args.openai_sdk:
    try:
      import openai
      from openai import NotFoundError, OpenAI
    except ImportError as error:
      raise RuntimeError(
          "--openai-sdk requires the openai Python package") from error
    sdk_version = openai.__version__
    sdk = OpenAI(
        api_key=api_key or "local-sdk-smoke",
        base_url=args.base_url.rstrip("/") + "/v1",
        timeout=args.timeout, max_retries=0)
    try:
      sdk_models = sdk.models.list()
      sdk_model_ids = [row.id for row in sdk_models.data]
      sdk_retrieved_model = sdk.models.retrieve(args.model)
      checks.append(check(
          "openai_sdk_models",
          args.model in sdk_model_ids and sdk_retrieved_model.id == args.model,
          model_ids=sdk_model_ids,
          retrieved_model=sdk_retrieved_model.id))

      sdk_completion_prompt = "Continue briefly: OpenVINO"
      sdk_completion = sdk.completions.create(
          model=args.model, prompt=sdk_completion_prompt, temperature=0,
          max_tokens=16, extra_body={"prefix_cache": False})
      sdk_completion_text = sdk_completion.choices[0].text
      sdk_completion_stream = sdk.completions.create(
          model=args.model, prompt=sdk_completion_prompt, temperature=0,
          max_tokens=16, stream=True,
          stream_options={"include_usage": True},
          extra_body={"prefix_cache": False})
      sdk_completion_deltas = []
      sdk_completion_usage = None
      for event in sdk_completion_stream:
        if event.choices:
          sdk_completion_deltas.append(event.choices[0].text)
        if event.usage is not None:
          sdk_completion_usage = event.usage.total_tokens
      sdk_completion_stream_text = "".join(sdk_completion_deltas)
      checks.append(check(
          "openai_sdk_completions_json_and_stream",
          bool(sdk_completion_text) and
          sdk_completion_stream_text == sdk_completion_text and
          isinstance(sdk_completion_usage, int),
          json_text=sdk_completion_text,
          stream_text=sdk_completion_stream_text,
          stream_total_tokens=sdk_completion_usage))

      sdk_chat = sdk.chat.completions.create(
          model=args.model,
          messages=[{"role": "user", "content": "只回答：你好"}],
          temperature=0, max_tokens=8)
      sdk_chat_text = sdk_chat.choices[0].message.content
      sdk_chat_stream = sdk.chat.completions.create(
          model=args.model,
          messages=[{"role": "user", "content": "只回答：再见"}],
          temperature=0, max_tokens=8, stream=True,
          stream_options={"include_usage": True})
      sdk_chat_deltas = []
      sdk_chat_usage = None
      for event in sdk_chat_stream:
        if event.choices and event.choices[0].delta.content:
          sdk_chat_deltas.append(event.choices[0].delta.content)
        if event.usage is not None:
          sdk_chat_usage = event.usage.total_tokens
      sdk_chat_stream_text = "".join(sdk_chat_deltas)
      checks.append(check(
          "openai_sdk_chat_json_and_stream",
          sdk_chat_text == "你好" and sdk_chat_stream_text == "再见" and
          isinstance(sdk_chat_usage, int),
          json_text=sdk_chat_text, stream_text=sdk_chat_stream_text,
          stream_total_tokens=sdk_chat_usage))

      sdk_response = sdk.responses.create(
          model=args.model, input="只回答：你好", temperature=0,
          max_output_tokens=8)
      sdk_retrieved_response = sdk.responses.retrieve(sdk_response.id)
      with sdk.responses.stream(
          model=args.model, input="只回答：再见", temperature=0,
          max_output_tokens=8,
      ) as sdk_response_stream:
        sdk_response_event_types = [
            event.type for event in sdk_response_stream]
        sdk_final_response = sdk_response_stream.get_final_response()
      sdk.responses.delete(sdk_response.id)
      sdk_deleted_missing = False
      try:
        sdk.responses.retrieve(sdk_response.id)
      except NotFoundError:
        sdk_deleted_missing = True
      checks.append(check(
          "openai_sdk_responses_json_stream_retrieve_delete",
          sdk_response.output_text == "你好" and
          sdk_retrieved_response.id == sdk_response.id and
          sdk_final_response.output_text == "再见" and
          "response.output_text.delta" in sdk_response_event_types and
          sdk_response_event_types[-1] == "response.completed" and
          sdk_deleted_missing,
          json_text=sdk_response.output_text,
          stream_text=sdk_final_response.output_text,
          stream_event_types=sdk_response_event_types,
          post_delete_not_found=sdk_deleted_missing))

      sdk_tool = {
          "type": "function",
          "function": {
              "name": "get_weather", "description": "查询指定城市天气",
              "parameters": {
                  "type": "object",
                  "properties": {"city": {"type": "string"}},
                  "required": ["city"],
              },
          },
      }
      sdk_tool_chat = sdk.chat.completions.create(
          model=args.model,
          messages=[{"role": "user",
                     "content": "上海天气怎么样？请调用工具。"}],
          tools=[sdk_tool], tool_choice="required",
          temperature=0, max_tokens=64)
      sdk_chat_calls = sdk_tool_chat.choices[0].message.tool_calls or []
      sdk_tool_chat_ok = bool(
          sdk_chat_calls and
          sdk_chat_calls[0].function.name == "get_weather" and
          json.loads(sdk_chat_calls[0].function.arguments).get("city") ==
              "上海")
      sdk_tool_response = sdk.responses.create(
          model=args.model, input="上海天气怎么样？请调用工具。",
          tools=[sdk_tool["function"] | {"type": "function"}],
          tool_choice="required", temperature=0, max_output_tokens=64)
      sdk_response_calls = [
          item for item in sdk_tool_response.output
          if item.type == "function_call"]
      sdk_tool_response_ok = bool(
          sdk_response_calls and
          sdk_response_calls[0].name == "get_weather" and
          json.loads(sdk_response_calls[0].arguments).get("city") == "上海")
      checks.append(check(
          "openai_sdk_chat_and_responses_function_tools",
          sdk_tool_chat_ok and sdk_tool_response_ok,
          chat_call_count=len(sdk_chat_calls),
          responses_call_count=len(sdk_response_calls)))
    finally:
      sdk.close()

  passed = all(row["pass"] for row in checks)
  result = {
      "schema": "iq36-http-service-smoke-v1",
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "base_url": args.base_url, "model": args.model,
      "openai_sdk_version": sdk_version,
      "required_checks_passed": passed,
      "checks": checks,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, ensure_ascii=False, indent=2) + "\n",
      encoding="utf-8")
  print(json.dumps({
      "output": str(args.output), "required_checks_passed": passed,
      "check_count": len(checks),
  }, separators=(",", ":")))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
