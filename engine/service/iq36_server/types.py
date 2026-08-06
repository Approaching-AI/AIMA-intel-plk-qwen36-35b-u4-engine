from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Endpoint = Literal["completions", "chat.completions", "responses"]


@dataclass(frozen=True)
class ToolCall:
  id: str
  name: str
  arguments: str


@dataclass(frozen=True)
class SamplingParams:
  max_new_tokens: int = 512
  temperature: float = 1.0
  top_p: float = 1.0
  top_k: int = 0
  seed: int | None = None
  presence_penalty: float = 0.0
  frequency_penalty: float = 0.0
  repetition_penalty: float = 1.0
  stop: tuple[str, ...] = ()
  logprobs: bool = False
  top_logprobs: int = 0
  ignore_eos: bool = False

  @property
  def greedy(self) -> bool:
    return (
        self.temperature == 0.0
        and self.presence_penalty == 0.0
        and self.frequency_penalty == 0.0
        and self.repetition_penalty == 1.0
        and not self.logprobs)

  @property
  def requires_full_logits(self) -> bool:
    return not self.greedy


@dataclass(frozen=True)
class PreparedRequest:
  endpoint: Endpoint
  model: str
  prompt: str
  prompt_token_ids: tuple[int, ...]
  params: SamplingParams
  stream: bool
  stream_include_usage: bool = False
  request_metadata: dict[str, Any] = field(default_factory=dict)
  tools: tuple[dict[str, Any], ...] = ()
  tool_choice: Any = None
  parallel_tool_calls: bool = True
  response_format: dict[str, Any] | None = None
  user: str | None = None
  store: bool = False
  previous_response_id: str | None = None
  prefix_cache: bool = True


@dataclass(frozen=True)
class TokenLogprob:
  token: str
  logprob: float
  bytes: tuple[int, ...]
  top_logprobs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GenerationStarted:
  request_id: str
  prompt_tokens: int
  cached_tokens: int
  profile: str
  bucket: int
  queue_ms: float
  prefix_restore_ms: float = 0.0


@dataclass(frozen=True)
class GenerationDelta:
  token_id: int
  text: str
  logprob: TokenLogprob | None = None


@dataclass(frozen=True)
class GenerationResult:
  request_id: str
  text: str
  token_ids: tuple[int, ...]
  prompt_tokens: int
  cached_tokens: int
  finish_reason: Literal["stop", "length", "cancelled", "error"]
  profile: str
  bucket: int
  prefill_ms: float
  decode_ms: float
  prefix_restore_ms: float
  logprobs: tuple[TokenLogprob, ...] = ()
  tool_calls: tuple[ToolCall, ...] = ()
  reasoning: str | None = None


@dataclass(frozen=True)
class BackendStatus:
  ready: bool
  active: bool
  loaded_workers: tuple[dict[str, Any], ...]
  last_error: str | None = None
