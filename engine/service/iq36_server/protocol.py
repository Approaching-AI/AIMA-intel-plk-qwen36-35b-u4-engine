from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .config import PROMOTED_MAX_NEW_TOKENS, ServerConfig
from .json_utils import strict_json_loads
from .types import PreparedRequest, SamplingParams


_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_MAX_TOOLS = 128
_MAX_STOP_STRINGS = 8

_COMMON_GENERATION_FIELDS = {
    "model", "temperature", "top_p", "top_k", "presence_penalty",
    "frequency_penalty", "repetition_penalty", "seed", "stop",
    "ignore_eos", "stream", "prefix_cache", "user",
}
_CHAT_FIELDS = _COMMON_GENERATION_FIELDS | {
    "messages", "max_completion_tokens", "max_tokens", "n", "logprobs",
    "top_logprobs", "stream_options", "tools", "tool_choice",
    "parallel_tool_calls", "response_format", "enable_thinking",
    "chat_template_kwargs",
}
_COMPLETION_FIELDS = _COMMON_GENERATION_FIELDS | {
    "prompt", "max_tokens", "n", "best_of", "suffix", "echo", "logprobs",
    "stream_options",
}
_RESPONSES_FIELDS = _COMMON_GENERATION_FIELDS | {
    "input", "instructions", "max_output_tokens", "logprobs",
    "top_logprobs", "tools", "tool_choice", "parallel_tool_calls", "text",
    "store", "previous_response_id", "metadata", "truncation",
    "enable_thinking", "chat_template_kwargs", "_resolved_messages",
}


class APIError(Exception):
  def __init__(
      self, message: str, *, status: int = 400,
      error_type: str = "invalid_request_error", param: str | None = None,
      code: str | None = None,
  ) -> None:
    super().__init__(message)
    self.message = message
    self.status = status
    self.error_type = error_type
    self.param = param
    self.code = code
    self.metric_reported = False

  def payload(self) -> dict[str, Any]:
    return {
        "error": {
            "message": self.message,
            "type": self.error_type,
            "param": self.param,
            "code": self.code,
        }
    }


def _field_error(message: str, param: str, code: str | None = None) -> APIError:
  return APIError(message, param=param, code=code)


def _reject_unknown_fields(
    payload: Mapping[str, Any], allowed: set[str], endpoint: str,
) -> None:
  unknown = sorted(str(name) for name in payload if name not in allowed)
  if unknown:
    name = unknown[0]
    raise _field_error(
        f"Unsupported parameter for {endpoint}: '{name}'.",
        name, "unsupported_parameter")


def decode_json_object(body: bytes) -> dict[str, Any]:
  try:
    value = strict_json_loads(body.decode("utf-8"))
  except UnicodeDecodeError as error:
    raise APIError("Request body must be valid UTF-8 JSON.") from error
  except json.JSONDecodeError as error:
    raise APIError(
        f"Invalid JSON at line {error.lineno}, column {error.colno}.") from error
  except ValueError as error:
    raise APIError(f"Invalid JSON: {error}.") from error
  if not isinstance(value, dict):
    raise APIError("Request body must be a JSON object.")
  return value


def _model(payload: Mapping[str, Any], config: ServerConfig) -> str:
  value = payload.get("model")
  if not isinstance(value, str) or not value:
    raise _field_error("'model' is required and must be a string.", "model")
  if value != config.model_id:
    raise APIError(
        f"The model '{value}' does not exist.", status=404,
        error_type="invalid_request_error", param="model",
        code="model_not_found")
  return value


def _finite_number(
    payload: Mapping[str, Any], name: str, default: float,
    minimum: float, maximum: float,
) -> float:
  value = payload.get(name, default)
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise _field_error(f"'{name}' must be a number.", name)
  value = float(value)
  if not math.isfinite(value) or not minimum <= value <= maximum:
    raise _field_error(
        f"'{name}' must be between {minimum} and {maximum}.", name)
  return value


def _integer(
    payload: Mapping[str, Any], name: str, default: int,
    minimum: int, maximum: int,
) -> int:
  value = payload.get(name, default)
  if isinstance(value, bool) or not isinstance(value, int):
    raise _field_error(f"'{name}' must be an integer.", name)
  if not minimum <= value <= maximum:
    raise _field_error(
        f"'{name}' must be between {minimum} and {maximum}.", name)
  return value


def _boolean(payload: Mapping[str, Any], name: str, default: bool) -> bool:
  value = payload.get(name, default)
  if not isinstance(value, bool):
    raise _field_error(f"'{name}' must be a boolean.", name)
  return value


def _stop(payload: Mapping[str, Any]) -> tuple[str, ...]:
  value = payload.get("stop")
  if value is None:
    return ()
  if isinstance(value, str):
    values = [value]
  elif isinstance(value, list) and all(isinstance(item, str) for item in value):
    values = value
  else:
    raise _field_error("'stop' must be a string or an array of strings.", "stop")
  if len(values) > _MAX_STOP_STRINGS:
    raise _field_error(
        f"At most {_MAX_STOP_STRINGS} stop strings are supported.", "stop")
  if any(not item for item in values):
    raise _field_error("Stop strings must not be empty.", "stop")
  return tuple(values)


def _seed(payload: Mapping[str, Any]) -> int | None:
  value = payload.get("seed")
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, int):
    raise _field_error("'seed' must be an integer.", "seed")
  if not -(2 ** 63) <= value < 2 ** 63:
    raise _field_error("'seed' must be a signed 64-bit integer.", "seed")
  return value


def _max_tokens(
    payload: Mapping[str, Any], config: ServerConfig, *, responses: bool = False,
) -> int:
  names = (
      ("max_output_tokens",) if responses else
      ("max_completion_tokens", "max_tokens"))
  present = [name for name in names if payload.get(name) is not None]
  if len(present) > 1:
    first = payload[present[0]]
    if any(payload[name] != first for name in present[1:]):
      raise _field_error(
          f"{', '.join(present)} must not specify different values.",
          present[-1])
  name = present[0] if present else names[0]
  value = payload.get(name, config.max_new_tokens)
  if isinstance(value, bool) or not isinstance(value, int):
    raise _field_error(f"'{name}' must be an integer.", name)
  if not 1 <= value <= config.max_new_tokens:
    raise _field_error(
        f"'{name}' must be between 1 and {config.max_new_tokens}; the "
        "promoted carrier is locked to at most 512 generated tokens.", name)
  return value


def _sampling(
    payload: Mapping[str, Any], config: ServerConfig, *,
    max_tokens: int, completion_logprobs: bool = False,
    responses_logprobs: bool = False,
) -> SamplingParams:
  temperature = _finite_number(payload, "temperature", 1.0, 0.0, 2.0)
  top_p = _finite_number(payload, "top_p", 1.0, 0.0, 1.0)
  if top_p == 0.0:
    raise _field_error("'top_p' must be greater than 0.", "top_p")
  top_k = _integer(payload, "top_k", 0, 0, 248320)
  presence = _finite_number(payload, "presence_penalty", 0.0, -2.0, 2.0)
  frequency = _finite_number(payload, "frequency_penalty", 0.0, -2.0, 2.0)
  repetition = _finite_number(
      payload, "repetition_penalty", 1.0, 0.01, 100.0)
  if completion_logprobs:
    raw_logprobs = payload.get("logprobs")
    if raw_logprobs is None:
      logprobs = False
      top_logprobs = 0
    elif isinstance(raw_logprobs, int) and not isinstance(raw_logprobs, bool):
      if not 0 <= raw_logprobs <= 20:
        raise _field_error("'logprobs' must be between 0 and 20.", "logprobs")
      logprobs = True
      top_logprobs = raw_logprobs
    else:
      raise _field_error("'logprobs' must be an integer or null.", "logprobs")
  elif responses_logprobs:
    top_logprobs = _integer(payload, "top_logprobs", 0, 0, 20)
    logprobs = _boolean(payload, "logprobs", top_logprobs > 0)
    if top_logprobs and not logprobs:
      raise _field_error(
          "'top_logprobs' requires 'logprobs' to be true.", "top_logprobs")
  else:
    logprobs = _boolean(payload, "logprobs", False)
    top_logprobs = _integer(payload, "top_logprobs", 0, 0, 20)
    if top_logprobs and not logprobs:
      raise _field_error(
          "'top_logprobs' requires 'logprobs' to be true.", "top_logprobs")
  return SamplingParams(
      max_new_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
      seed=_seed(payload),
      presence_penalty=presence,
      frequency_penalty=frequency,
      repetition_penalty=repetition,
      stop=_stop(payload),
      logprobs=logprobs,
      top_logprobs=top_logprobs,
      ignore_eos=_boolean(payload, "ignore_eos", False),
  )


def _text_content(value: Any, param: str, *, response_input: bool = False) -> str:
  if isinstance(value, str):
    return value
  if value is None:
    return ""
  if not isinstance(value, list):
    raise _field_error(f"'{param}' must be text or an array of text parts.", param)
  pieces = []
  accepted_types = {"text", "input_text", "output_text"}
  for index, part in enumerate(value):
    item_param = f"{param}.{index}"
    if not isinstance(part, dict):
      raise _field_error("Content parts must be objects.", item_param)
    _reject_unknown_fields(
        part, {"type", "text"}, f"content part '{item_param}'")
    part_type = part.get("type")
    if part_type not in accepted_types:
      qualifier = "Responses input" if response_input else "Chat content"
      raise _field_error(
          f"{qualifier} supports text parts only; got {part_type!r}.",
          f"{item_param}.type", "unsupported_content_type")
    text = part.get("text")
    if not isinstance(text, str):
      raise _field_error("Text content part requires a string 'text'.", item_param)
    pieces.append(text)
  return "".join(pieces)


def _assistant_tool_calls(value: Any, param: str) -> list[dict[str, Any]]:
  if value is None:
    return []
  if not isinstance(value, list):
    raise _field_error("'tool_calls' must be an array.", param)
  normalized = []
  for index, item in enumerate(value):
    item_param = f"{param}.{index}"
    if not isinstance(item, dict) or item.get("type", "function") != "function":
      raise _field_error("Only function tool calls are supported.", item_param)
    _reject_unknown_fields(
        item, {"id", "type", "function"}, f"tool call '{item_param}'")
    function = item.get("function")
    if not isinstance(function, dict):
      raise _field_error("Tool call requires a function object.", item_param)
    _reject_unknown_fields(
        function, {"name", "arguments"},
        f"tool-call function '{item_param}.function'")
    call_id = item.get("id")
    if not isinstance(call_id, str) or not call_id:
      raise _field_error(
          "Tool call requires a non-empty 'id'.", f"{item_param}.id")
    name = function.get("name")
    if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
      raise _field_error("Invalid function name.", f"{item_param}.function.name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
      try:
        arguments = strict_json_loads(arguments)
      except (json.JSONDecodeError, ValueError) as error:
        raise _field_error(
            "Tool-call arguments must contain a JSON object.",
            f"{item_param}.function.arguments") from error
    if not isinstance(arguments, dict):
      raise _field_error(
          "Tool-call arguments must be a JSON object.",
          f"{item_param}.function.arguments")
    normalized.append({
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": arguments},
    })
  return normalized


def _normalize_messages(
    value: Any, *, instructions: str | None = None,
) -> list[dict[str, Any]]:
  if not isinstance(value, list) or not value:
    raise _field_error("'messages' must be a non-empty array.", "messages")
  rows = []
  leading_instructions: list[str] = [instructions] if instructions else []
  saw_dialogue = False
  tool_call_ids: set[str] = set()
  for index, item in enumerate(value):
    param = f"messages.{index}"
    if not isinstance(item, dict):
      raise _field_error("Each message must be an object.", param)
    role = item.get("role")
    if role not in ("developer", "system", "user", "assistant", "tool"):
      raise _field_error(f"Unsupported message role {role!r}.", f"{param}.role")
    allowed = {"role", "content"}
    if role == "assistant":
      allowed.update({"reasoning_content", "tool_calls", "refusal"})
    elif role == "tool":
      allowed.add("tool_call_id")
    _reject_unknown_fields(item, allowed, f"message '{param}'")
    if role == "assistant" and item.get("refusal") is not None:
      raise _field_error(
          "Assistant refusal content is not supported by this text model.",
          f"{param}.refusal", "unsupported_content_type")
    if role in ("developer", "system"):
      if saw_dialogue:
        raise _field_error(
            "Developer and system messages must precede dialogue messages.",
            f"{param}.role")
      content = _text_content(item.get("content"), f"{param}.content")
      if content:
        leading_instructions.append(content)
      continue
    saw_dialogue = True
    content = _text_content(item.get("content"), f"{param}.content")
    row: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
      # Preserve the empty thinking envelope used by the default generation
      # prompt.  This makes the next rendered turn begin with the exact token
      # sequence that produced the prior answer, enabling real state-prefix
      # reuse instead of an identical-request-only cache.
      row["reasoning_content"] = (
          item["reasoning_content"]
          if isinstance(item.get("reasoning_content"), str) else "")
      calls = _assistant_tool_calls(item.get("tool_calls"), f"{param}.tool_calls")
      if calls:
        row["tool_calls"] = calls
        tool_call_ids.update(
            str(call["id"]) for call in calls if call.get("id"))
    elif role == "tool":
      call_id = item.get("tool_call_id")
      if not isinstance(call_id, str) or not call_id:
        raise _field_error(
            "Tool messages require 'tool_call_id'.", f"{param}.tool_call_id")
      if tool_call_ids and call_id not in tool_call_ids:
        raise _field_error(
            f"Unknown tool_call_id '{call_id}'.", f"{param}.tool_call_id")
      row["tool_call_id"] = call_id
    rows.append(row)
  if not rows or not any(row["role"] == "user" for row in rows):
    raise _field_error("At least one user message is required.", "messages")
  if leading_instructions:
    rows.insert(0, {"role": "system", "content": "\n\n".join(
        item for item in leading_instructions if item)})
  return rows


def _normalize_tools(value: Any, *, responses: bool = False) -> tuple[dict[str, Any], ...]:
  if value is None:
    return ()
  if not isinstance(value, list):
    raise _field_error("'tools' must be an array.", "tools")
  if len(value) > _MAX_TOOLS:
    raise _field_error(f"At most {_MAX_TOOLS} tools are supported.", "tools")
  normalized = []
  names = set()
  for index, item in enumerate(value):
    param = f"tools.{index}"
    if not isinstance(item, dict) or item.get("type") != "function":
      raise _field_error(
          "This local service supports function tools only.", param,
          "unsupported_tool_type")
    _reject_unknown_fields(
        item,
        {"type", "name", "description", "parameters", "strict"}
        if responses else {"type", "function"},
        f"tool '{param}'")
    function = item if responses else item.get("function")
    if not isinstance(function, dict):
      raise _field_error("Function tool definition must be an object.", param)
    _reject_unknown_fields(
        function,
        {"type", "name", "description", "parameters", "strict"}
        if responses else {"name", "description", "parameters", "strict"},
        f"function tool '{param}'")
    name = function.get("name")
    if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
      raise _field_error("Invalid function tool name.", f"{param}.name")
    if name in names:
      raise _field_error(f"Duplicate function tool '{name}'.", f"{param}.name")
    names.add(name)
    description = function.get("description", "")
    if not isinstance(description, str):
      raise _field_error("Tool description must be a string.", f"{param}.description")
    parameters = function.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
      raise _field_error("Tool parameters must be a JSON Schema object.", f"{param}.parameters")
    _check_json_schema(parameters, f"{param}.parameters")
    strict = function.get("strict")
    if strict is not None and not isinstance(strict, bool):
      raise _field_error("Tool 'strict' must be a boolean.", f"{param}.strict")
    normalized.append({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
            **({"strict": strict} if strict is not None else {}),
        },
    })
  return tuple(normalized)


@dataclass(frozen=True)
class _ToolPolicy:
  rendered_tools: tuple[dict[str, Any], ...]
  choice: Any
  instruction: str | None


def _tool_policy(
    tools: tuple[dict[str, Any], ...], choice: Any, *, parallel: bool,
) -> _ToolPolicy:
  names = {tool["function"]["name"] for tool in tools}
  serial = (
      "Emit at most one function call in this response."
      if tools and not parallel else None)
  if choice is None:
    choice = "auto" if tools else "none"
  if isinstance(choice, str):
    if choice not in ("none", "auto", "required"):
      raise _field_error("Invalid 'tool_choice'.", "tool_choice")
    if choice == "none":
      return _ToolPolicy((), choice, None)
    if not tools:
      raise _field_error("'tool_choice' requires at least one tool.", "tool_choice")
    required = (
        "You must call one or more of the supplied functions. Do not answer "
        "without a function call." if choice == "required" else None)
    instruction = " ".join(
        item for item in (required, serial) if item) or None
    return _ToolPolicy(tools, choice, instruction)
  if not isinstance(choice, dict) or choice.get("type") != "function":
    raise _field_error("Invalid 'tool_choice'.", "tool_choice")
  function = choice.get("function")
  name = function.get("name") if isinstance(function, dict) else choice.get("name")
  if not isinstance(name, str) or name not in names:
    raise _field_error("Named 'tool_choice' must reference a supplied tool.", "tool_choice")
  selected = tuple(tool for tool in tools if tool["function"]["name"] == name)
  return _ToolPolicy(
      selected, choice,
      " ".join(item for item in (
          f"You must call the function '{name}'. Do not call another function "
          "or answer without that function call.", serial) if item))


def _response_format(
    value: Any, *, responses: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
  if value is None:
    return None, None
  if not isinstance(value, dict):
    raise _field_error("'response_format' must be an object.", "response_format")
  kind = value.get("type")
  if kind == "text":
    _reject_unknown_fields(value, {"type"}, "response format")
    return value, None
  if kind == "json_object":
    _reject_unknown_fields(value, {"type"}, "response format")
    return value, "Return exactly one valid JSON value and no surrounding prose."
  if kind == "json_schema":
    descriptor = value if responses else value.get("json_schema")
    _reject_unknown_fields(
        value,
        {"type", "name", "description", "schema", "strict"}
        if responses else {"type", "json_schema"},
        "response format")
    if (
        not isinstance(descriptor, dict) or
        not isinstance(descriptor.get("schema"), dict)
    ):
      raise _field_error(
          "JSON-schema response format requires a JSON object in 'schema'.",
          "response_format")
    _reject_unknown_fields(
        descriptor,
        {"type", "name", "description", "schema", "strict"}
        if responses else {"name", "description", "schema", "strict"},
        "JSON-schema response format")
    name = descriptor.get("name")
    if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
      raise _field_error(
          "JSON-schema response format requires a valid 'name'.",
          "response_format")
    strict = descriptor.get("strict")
    if strict is not None and not isinstance(strict, bool):
      raise _field_error(
          "JSON-schema response format 'strict' must be a boolean.",
          "response_format")
    description = descriptor.get("description")
    if description is not None and not isinstance(description, str):
      raise _field_error(
          "JSON-schema response format 'description' must be a string.",
          "response_format")
    _check_json_schema(descriptor["schema"], "response_format")
    instruction = (
        ((description + "\n") if description else "") +
        "Return exactly one JSON value that satisfies this schema and no "
        "surrounding prose: " + json.dumps(
            descriptor["schema"], ensure_ascii=False, separators=(",", ":")))
    return value, instruction
  raise _field_error(
      "Supported response formats are text, json_object, and json_schema.",
      "response_format", "unsupported_response_format")


def _thinking(payload: Mapping[str, Any]) -> bool:
  value = payload.get("enable_thinking")
  kwargs = payload.get("chat_template_kwargs")
  if kwargs is not None:
    if not isinstance(kwargs, dict):
      raise _field_error(
          "'chat_template_kwargs' must be an object.",
          "chat_template_kwargs")
    _reject_unknown_fields(
        kwargs, {"enable_thinking"}, "chat_template_kwargs")
    nested = kwargs.get("enable_thinking")
    if nested is not None and not isinstance(nested, bool):
      raise _field_error(
          "'chat_template_kwargs.enable_thinking' must be a boolean.",
          "chat_template_kwargs.enable_thinking")
    if value is None:
      value = nested
    elif nested is not None and nested != value:
      raise _field_error(
          "'enable_thinking' and 'chat_template_kwargs.enable_thinking' "
          "must not conflict.", "chat_template_kwargs.enable_thinking")
  if value is None:
    return False
  if not isinstance(value, bool):
    raise _field_error("'enable_thinking' must be a boolean.", "enable_thinking")
  return value


def _user(payload: Mapping[str, Any]) -> str | None:
  value = payload.get("user")
  if value is None:
    return None
  if not isinstance(value, str) or not value or len(value) > 256:
    raise _field_error(
        "'user' must be a non-empty string no longer than 256 characters.",
        "user")
  return value


def _prefix_cache(payload: Mapping[str, Any]) -> bool:
  value = payload.get("prefix_cache", True)
  if not isinstance(value, bool):
    raise _field_error("'prefix_cache' must be a boolean.", "prefix_cache")
  return value


def _stream_include_usage(payload: Mapping[str, Any]) -> bool:
  value = payload.get("stream_options")
  if value is None:
    return False
  if not isinstance(value, dict):
    raise _field_error("'stream_options' must be an object.", "stream_options")
  unknown = sorted(str(name) for name in value if name != "include_usage")
  if unknown:
    raise _field_error(
        f"Unsupported stream option: '{unknown[0]}'.",
        f"stream_options.{unknown[0]}", "unsupported_parameter")
  include = value.get("include_usage", False)
  if not isinstance(include, bool):
    raise _field_error(
        "'stream_options.include_usage' must be a boolean.",
        "stream_options.include_usage")
  return include


def _metadata(value: Any) -> dict[str, str]:
  if value is None:
    return {}
  if not isinstance(value, dict) or len(value) > 16:
    raise _field_error(
        "'metadata' must be an object with at most 16 entries.", "metadata")
  normalized: dict[str, str] = {}
  for key, item in value.items():
    if not isinstance(key, str) or len(key) > 64:
      raise _field_error(
          "Metadata keys must be strings no longer than 64 characters.",
          "metadata")
    if not isinstance(item, str) or len(item) > 512:
      raise _field_error(
          "Metadata values must be strings no longer than 512 characters.",
          f"metadata.{key}")
    normalized[key] = item
  return normalized


def _check_json_schema(schema: dict[str, Any], param: str) -> None:
  if len(json.dumps(schema, separators=(",", ":"))) > 256 * 1024:
    raise _field_error(
        "JSON Schema must not exceed 256 KiB.", param,
        "schema_too_large")
  stack: list[tuple[Any, int]] = [(schema, 0)]
  nodes = 0
  while stack:
    value, depth = stack.pop()
    nodes += 1
    if nodes > 10000 or depth > 64:
      raise _field_error(
          "JSON Schema is too complex.", param, "schema_too_complex")
    if isinstance(value, dict):
      unsafe = sorted(
          name for name in ("pattern", "patternProperties") if name in value)
      if unsafe:
        raise _field_error(
            f"JSON Schema keyword '{unsafe[0]}' is not supported because "
            "Python regular-expression validation is not time-bounded.",
            param, "unsupported_schema_keyword")
      reference = value.get("$ref")
      if isinstance(reference, str) and not reference.startswith("#"):
        raise _field_error(
            "External JSON Schema references are not supported.",
            param, "unsupported_schema_reference")
      stack.extend((item, depth + 1) for item in value.values())
    elif isinstance(value, list):
      stack.extend((item, depth + 1) for item in value)
  try:
    Draft202012Validator.check_schema(schema)
  except SchemaError as error:
    raise _field_error(
        f"Invalid JSON Schema: {error.message}", param) from error


def _render_and_check(
    *, messages: list[dict[str, Any]], tools: tuple[dict[str, Any], ...],
    policy_instruction: str | None, format_instruction: str | None,
    tokenizer, enable_thinking: bool, params: SamplingParams,
    config: ServerConfig,
) -> tuple[str, tuple[int, ...]]:
  instructions = [item for item in (policy_instruction, format_instruction) if item]
  if instructions:
    if messages and messages[0]["role"] == "system":
      messages = [dict(messages[0]), *messages[1:]]
      messages[0]["content"] = (
          str(messages[0].get("content", "")) + "\n\n" +
          "\n".join(instructions)).strip()
    else:
      messages = [{"role": "system", "content": "\n".join(instructions)}, *messages]
  try:
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, enable_thinking=enable_thinking)
    token_ids = tokenizer.encode(prompt)
  except Exception as error:
    raise APIError(
        f"The message history cannot be rendered by the locked chat template: "
        f"{error}", param="messages", code="invalid_message_history") from error
  _check_context(token_ids, params, config)
  return prompt, token_ids


def _check_context(
    token_ids: Sequence[int], params: SamplingParams, config: ServerConfig,
) -> None:
  prompt_tokens = len(token_ids)
  total = prompt_tokens + params.max_new_tokens
  if prompt_tokens > 131072 or total > config.max_context_length:
    raise APIError(
        "This request has "
        f"{prompt_tokens} prompt tokens and requests {params.max_new_tokens} "
        f"new tokens ({total} total), exceeding the configured context "
        f"capacity {config.max_context_length}. The service never silently "
        "truncates input.", param="messages", code="context_length_exceeded")


def prepare_chat_completion(
    payload: Mapping[str, Any], config: ServerConfig, tokenizer,
) -> PreparedRequest:
  _reject_unknown_fields(payload, _CHAT_FIELDS, "chat completions")
  model = _model(payload, config)
  if payload.get("n", 1) != 1:
    raise _field_error("Only n=1 is supported by the locked batch-1 engine.", "n")
  tools = _normalize_tools(payload.get("tools"))
  parallel = _boolean(payload, "parallel_tool_calls", True)
  policy = _tool_policy(
      tools, payload.get("tool_choice"), parallel=parallel)
  messages = _normalize_messages(payload.get("messages"))
  response_format, format_instruction = _response_format(payload.get("response_format"))
  max_tokens = _max_tokens(payload, config)
  params = _sampling(payload, config, max_tokens=max_tokens)
  prompt, token_ids = _render_and_check(
      messages=messages, tools=policy.rendered_tools,
      policy_instruction=policy.instruction,
      format_instruction=format_instruction, tokenizer=tokenizer,
      enable_thinking=_thinking(payload), params=params, config=config)
  stream = _boolean(payload, "stream", False)
  include_usage = _stream_include_usage(payload)
  return PreparedRequest(
      endpoint="chat.completions", model=model, prompt=prompt,
      prompt_token_ids=token_ids, params=params, stream=stream,
      stream_include_usage=include_usage, tools=policy.rendered_tools,
      tool_choice=policy.choice,
      parallel_tool_calls=parallel,
      response_format=response_format,
      prefix_cache=_prefix_cache(payload),
      user=_user(payload),
      request_metadata={"messages": messages,
                        "enable_thinking": _thinking(payload)},
  )


def prepare_completion(
    payload: Mapping[str, Any], config: ServerConfig, tokenizer,
) -> PreparedRequest:
  _reject_unknown_fields(payload, _COMPLETION_FIELDS, "completions")
  model = _model(payload, config)
  prompt = payload.get("prompt")
  if not isinstance(prompt, str):
    raise _field_error(
        "The batch-1 service requires 'prompt' to be one string.", "prompt")
  if payload.get("n", 1) != 1:
    raise _field_error("Only n=1 is supported by the locked batch-1 engine.", "n")
  if payload.get("best_of", 1) != 1:
    raise _field_error("Only best_of=1 is supported.", "best_of")
  if payload.get("suffix") is not None:
    raise _field_error("'suffix' is not supported by this causal model.", "suffix")
  max_tokens = _max_tokens(payload, config)
  params = _sampling(
      payload, config, max_tokens=max_tokens, completion_logprobs=True)
  echo = _boolean(payload, "echo", False)
  if echo and params.logprobs:
    raise _field_error(
        "This service cannot return prompt-token log probabilities; 'echo' "
        "and 'logprobs' cannot be combined.", "echo",
        "unsupported_parameter_combination")
  token_ids = tokenizer.encode(prompt)
  _check_context(token_ids, params, config)
  include_usage = _stream_include_usage(payload)
  return PreparedRequest(
      endpoint="completions", model=model, prompt=prompt,
      prompt_token_ids=token_ids, params=params,
      stream=_boolean(payload, "stream", False),
      stream_include_usage=include_usage,
      user=_user(payload),
      request_metadata={"echo": echo},
      prefix_cache=_prefix_cache(payload),
  )


def responses_input_messages(value: Any) -> list[dict[str, Any]]:
  if isinstance(value, str):
    return [{"role": "user", "content": value}]
  if not isinstance(value, list) or not value:
    raise _field_error("'input' must be text or a non-empty input array.", "input")
  messages = []
  for index, item in enumerate(value):
    param = f"input.{index}"
    if not isinstance(item, dict):
      raise _field_error("Response input items must be objects.", param)
    item_type = item.get("type")
    if item_type == "function_call":
      call_id = item.get("call_id")
      name = item.get("name")
      arguments = item.get("arguments", "{}")
      if not isinstance(call_id, str) or not call_id:
        raise _field_error(
            "Function call requires 'call_id'.", f"{param}.call_id")
      if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
        raise _field_error(
            "Function call requires a valid 'name'.", f"{param}.name")
      if not isinstance(arguments, str):
        arguments = json.dumps(
            arguments, ensure_ascii=False, separators=(",", ":"))
      try:
        parsed_arguments = strict_json_loads(arguments)
      except (json.JSONDecodeError, ValueError) as error:
        raise _field_error(
            "Function-call arguments must contain a JSON object.",
            f"{param}.arguments") from error
      if not isinstance(parsed_arguments, dict):
        raise _field_error(
            "Function-call arguments must contain a JSON object.",
            f"{param}.arguments")
      call = {
          "id": call_id, "type": "function",
          "function": {"name": name, "arguments": arguments},
      }
      if messages and messages[-1].get("role") == "assistant" \
          and not messages[-1].get("content"):
        messages[-1].setdefault("tool_calls", []).append(call)
      else:
        messages.append({
            "role": "assistant", "content": "", "tool_calls": [call]})
      continue
    if item_type == "function_call_output":
      call_id = item.get("call_id")
      if not isinstance(call_id, str) or not call_id:
        raise _field_error("Function output requires 'call_id'.", f"{param}.call_id")
      output = item.get("output")
      if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
      messages.append({
          "role": "tool", "tool_call_id": call_id, "content": output})
      continue
    role = item.get("role")
    if role not in ("developer", "system", "user", "assistant", "tool"):
      raise _field_error("Unsupported Responses input item.", param)
    messages.append({
        "role": role,
        "content": _text_content(
            item.get("content"), f"{param}.content", response_input=True),
        **({"tool_call_id": item["tool_call_id"]}
           if isinstance(item.get("tool_call_id"), str) else {}),
        **({"tool_calls": item["tool_calls"]}
           if item.get("tool_calls") is not None else {}),
    })
  return messages


def prepare_response(
    payload: Mapping[str, Any], config: ServerConfig, tokenizer,
) -> PreparedRequest:
  _reject_unknown_fields(payload, _RESPONSES_FIELDS, "responses")
  model = _model(payload, config)
  instructions = payload.get("instructions")
  if instructions is not None and not isinstance(instructions, str):
    raise _field_error("'instructions' must be a string.", "instructions")
  resolved = payload.get("_resolved_messages")
  raw_messages = (
      resolved if isinstance(resolved, list)
      else responses_input_messages(payload.get("input")))
  stored_messages = _normalize_messages(raw_messages)
  messages = _normalize_messages(raw_messages, instructions=instructions)
  tools = _normalize_tools(payload.get("tools"), responses=True)
  parallel = _boolean(payload, "parallel_tool_calls", True)
  raw_choice = payload.get("tool_choice")
  if isinstance(raw_choice, dict) and raw_choice.get("type") == "function" \
      and "function" not in raw_choice:
    raw_choice = {
        "type": "function", "function": {"name": raw_choice.get("name")}}
  policy = _tool_policy(tools, raw_choice, parallel=parallel)
  text_config = payload.get("text")
  if text_config is not None and not isinstance(text_config, dict):
    raise _field_error("'text' must be an object.", "text")
  if isinstance(text_config, dict):
    _reject_unknown_fields(text_config, {"format"}, "responses text")
  response_format, format_instruction = _response_format(
      text_config.get("format") if isinstance(text_config, dict) else None,
      responses=True)
  truncation = payload.get("truncation", "disabled")
  if truncation != "disabled":
    raise _field_error(
        "This service does not silently truncate context; 'truncation' must "
        "be 'disabled'.", "truncation", "unsupported_truncation")
  max_tokens = _max_tokens(payload, config, responses=True)
  params = _sampling(
      payload, config, max_tokens=max_tokens, responses_logprobs=True)
  prompt, token_ids = _render_and_check(
      messages=messages, tools=policy.rendered_tools,
      policy_instruction=policy.instruction,
      format_instruction=format_instruction, tokenizer=tokenizer,
      enable_thinking=_thinking(payload), params=params, config=config)
  previous = payload.get("previous_response_id")
  if previous is not None and not isinstance(previous, str):
    raise _field_error(
        "'previous_response_id' must be a string.", "previous_response_id")
  return PreparedRequest(
      endpoint="responses", model=model, prompt=prompt,
      prompt_token_ids=token_ids, params=params,
      stream=_boolean(payload, "stream", False),
      tools=policy.rendered_tools, tool_choice=policy.choice,
      parallel_tool_calls=parallel,
      response_format=response_format,
      user=_user(payload),
      store=_boolean(payload, "store", True),
      previous_response_id=previous,
      prefix_cache=_prefix_cache(payload),
      request_metadata={
          "messages": stored_messages,
          "instructions": instructions,
          "response_metadata": _metadata(payload.get("metadata")),
          "enable_thinking": _thinking(payload),
      },
  )
