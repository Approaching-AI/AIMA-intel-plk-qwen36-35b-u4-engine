from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .json_utils import strict_json_loads
from .types import ToolCall


_TOOL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_BLOCK = re.compile(
    r"<function=([^>\n]+)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE)
_PARAMETER_BLOCK = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE)
_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class ParsedAssistantText:
  content: str
  reasoning: str | None
  tool_calls: tuple[ToolCall, ...]


def _call_id(request_id: str, index: int, name: str) -> str:
  digest = hashlib.sha256(
      f"{request_id}:{index}:{name}".encode("utf-8")).hexdigest()[:24]
  return f"call_{digest}"


def _argument_value(text: str):
  value = text.strip()
  try:
    return strict_json_loads(value)
  except (json.JSONDecodeError, TypeError, ValueError):
    return value


def _parse_xml_call(body: str) -> tuple[str, dict] | None:
  match = _FUNCTION_BLOCK.fullmatch(body.strip())
  if match is None:
    return None
  name = match.group(1).strip()
  if not _FUNCTION_NAME.fullmatch(name):
    return None
  arguments = {}
  parameter_text = match.group(2)
  matches = list(_PARAMETER_BLOCK.finditer(parameter_text))
  if parameter_text.strip() and not matches:
    return None
  cursor = 0
  for parameter in matches:
    if parameter_text[cursor:parameter.start()].strip():
      return None
    key = parameter.group(1).strip()
    if not key or key in arguments:
      return None
    arguments[key] = _argument_value(parameter.group(2))
    cursor = parameter.end()
  if parameter_text[cursor:].strip():
    return None
  return name, arguments


def _parse_json_call(body: str) -> tuple[str, dict] | None:
  try:
    value = strict_json_loads(body)
  except (json.JSONDecodeError, ValueError):
    return None
  if not isinstance(value, dict):
    return None
  function = value.get("function") if isinstance(value.get("function"), dict) else value
  name = function.get("name")
  arguments = function.get("arguments", {})
  if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
    return None
  if isinstance(arguments, str):
    try:
      arguments = strict_json_loads(arguments)
    except (json.JSONDecodeError, ValueError):
      return None
  if not isinstance(arguments, dict):
    return None
  return name, arguments


def _split_reasoning(text: str) -> tuple[str | None, str]:
  stripped = text.lstrip()
  if not stripped.startswith("<think>"):
    return None, text
  end = stripped.find("</think>")
  if end < 0:
    return stripped[len("<think>"):].strip() or None, ""
  reasoning = stripped[len("<think>"):end].strip() or None
  return reasoning, stripped[end + len("</think>"):].lstrip("\n")


def parse_assistant_text(
    text: str, request_id: str, *, allow_tool_calls: bool = True,
) -> ParsedAssistantText:
  reasoning, visible = _split_reasoning(text)
  if not allow_tool_calls:
    return ParsedAssistantText(
        content=visible.strip(), reasoning=reasoning, tool_calls=())
  calls: list[ToolCall] = []
  content_parts: list[str] = []
  cursor = 0
  for block in _TOOL_BLOCK.finditer(visible):
    content_parts.append(visible[cursor:block.start()])
    body = block.group(1).strip()
    parsed = _parse_xml_call(body) or _parse_json_call(body)
    if parsed is None:
      content_parts.append(block.group(0))
    else:
      name, arguments = parsed
      calls.append(ToolCall(
          id=_call_id(request_id, len(calls), name),
          name=name,
          arguments=json.dumps(
              arguments, ensure_ascii=False, separators=(",", ":"))))
    cursor = block.end()
  content_parts.append(visible[cursor:])
  content = "".join(content_parts).strip()
  return ParsedAssistantText(
      content=content, reasoning=reasoning, tool_calls=tuple(calls))
