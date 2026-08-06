from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence

from .runtime_identity import verify_imported_runtime


class TokenizerAdapter:
  """Thread-safe facade over the model's locked OpenVINO tokenizer IR."""

  def __init__(self, model_dir: Path) -> None:
    import openvino_genai as ov_genai

    self.runtime_identity = verify_imported_runtime()
    self._tokenizer = ov_genai.Tokenizer(str(model_dir))
    self._lock = threading.Lock()

  @property
  def eos_token_id(self) -> int:
    with self._lock:
      return int(self._tokenizer.get_eos_token_id())

  @property
  def chat_template(self) -> str:
    with self._lock:
      return str(self._tokenizer.get_original_chat_template())

  def apply_chat_template(
      self,
      messages: Sequence[dict[str, Any]],
      *,
      tools: Sequence[dict[str, Any]] = (),
      enable_thinking: bool = False,
  ) -> str:
    with self._lock:
      return str(self._tokenizer.apply_chat_template(
          list(messages), True, "", list(tools) if tools else None,
          {"enable_thinking": bool(enable_thinking),
           "preserve_thinking": True}))

  def encode(self, text: str) -> tuple[int, ...]:
    with self._lock:
      encoded = self._tokenizer.encode(text)
      return tuple(
          int(value) for value in encoded.input_ids.data.reshape(-1))

  def decode(
      self, token_ids: Sequence[int], *, skip_special_tokens: bool = False,
  ) -> str:
    with self._lock:
      return str(self._tokenizer.decode(
          [int(value) for value in token_ids],
          skip_special_tokens=skip_special_tokens))


class SimpleTestTokenizer:
  """Deterministic dependency-free tokenizer used only by protocol tests."""

  eos_token_id = 0

  def apply_chat_template(
      self, messages: Sequence[dict[str, Any]], *,
      tools: Sequence[dict[str, Any]] = (), enable_thinking: bool = False,
  ) -> str:
    tool_text = f"<tools>{tools!r}</tools>\n" if tools else ""
    rows = [tool_text]
    rows.extend(
        f"<{message['role']}>{message.get('content', '')}</{message['role']}>"
        for message in messages)
    rows.append("<assistant>")
    if enable_thinking:
      rows.append("<think>")
    return "\n".join(rows)

  def encode(self, text: str) -> tuple[int, ...]:
    return tuple(text.encode("utf-8"))

  def decode(
      self, token_ids: Sequence[int], *, skip_special_tokens: bool = False,
  ) -> str:
    del skip_special_tokens
    return bytes(int(value) for value in token_ids).decode(
        "utf-8", errors="replace")
