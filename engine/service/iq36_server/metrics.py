from __future__ import annotations

import threading
import time
from collections import Counter

from .types import GenerationResult, GenerationStarted


class Metrics:
  _FIXED_PATHS = frozenset({
      "/healthz", "/readyz", "/metrics", "/v1/models",
      "/v1/completions", "/v1/chat/completions", "/v1/responses",
  })

  def __init__(self) -> None:
    self.started_at = time.time()
    self._lock = threading.Lock()
    self._http = Counter()
    self._generation = Counter()
    self._errors = Counter()
    self._active = 0
    self._queued = 0
    self._input_tokens = 0
    self._output_tokens = 0
    self._cached_tokens = 0
    self._queue_ms_sum = 0.0
    self._prefill_ms_sum = 0.0
    self._decode_ms_sum = 0.0
    self._restore_ms_sum = 0.0

  def http(self, method: str, path: str, status: int) -> None:
    path = self._normalized_path(path)
    with self._lock:
      self._http[(method, path, status)] += 1

  @classmethod
  def _normalized_path(cls, path: str) -> str:
    if path in cls._FIXED_PATHS:
      return path
    if path.startswith("/v1/models/"):
      return "/v1/models/{model}"
    if path.startswith("/v1/responses/"):
      return "/v1/responses/{response_id}"
    return "/_other"

  def error(self, kind: str) -> None:
    with self._lock:
      self._errors[kind] += 1

  def admitted(self) -> None:
    with self._lock:
      self._queued += 1

  def started(self, value: GenerationStarted) -> None:
    with self._lock:
      self._queued = max(0, self._queued - 1)
      self._active += 1
      self._queue_ms_sum += value.queue_ms

  def completed(self, endpoint: str, result: GenerationResult) -> None:
    with self._lock:
      self._active = max(0, self._active - 1)
      self._generation[(endpoint, result.profile, result.bucket,
                        result.finish_reason)] += 1
      self._input_tokens += result.prompt_tokens
      self._output_tokens += len(result.token_ids)
      self._cached_tokens += result.cached_tokens
      self._prefill_ms_sum += result.prefill_ms
      self._decode_ms_sum += result.decode_ms
      self._restore_ms_sum += result.prefix_restore_ms

  def failed(self, endpoint: str, *, started: bool, kind: str) -> None:
    with self._lock:
      if started:
        self._active = max(0, self._active - 1)
      else:
        self._queued = max(0, self._queued - 1)
      self._generation[(endpoint, "unknown", 0, kind)] += 1

  @staticmethod
  def _labels(**values) -> str:
    if not values:
      return ""
    def escaped(value) -> str:
      return (str(value).replace("\\", "\\\\").replace("\n", "\\n")
              .replace('"', '\\"'))
    body = ",".join(
        f'{name}="{escaped(value)}"' for name, value in values.items())
    return "{" + body + "}"

  def render(self, backend_ready: bool) -> bytes:
    with self._lock:
      lines = [
          "# HELP iq36_process_start_time_seconds Process start time.",
          "# TYPE iq36_process_start_time_seconds gauge",
          f"iq36_process_start_time_seconds {self.started_at:.3f}",
          "# HELP iq36_backend_ready Whether a resident backend is ready.",
          "# TYPE iq36_backend_ready gauge",
          f"iq36_backend_ready {1 if backend_ready else 0}",
          "# HELP iq36_generation_active Active batch-1 generation count.",
          "# TYPE iq36_generation_active gauge",
          f"iq36_generation_active {self._active}",
          "# HELP iq36_generation_queued Admitted requests waiting or loading.",
          "# TYPE iq36_generation_queued gauge",
          f"iq36_generation_queued {self._queued}",
          "# TYPE iq36_input_tokens_total counter",
          f"iq36_input_tokens_total {self._input_tokens}",
          "# TYPE iq36_output_tokens_total counter",
          f"iq36_output_tokens_total {self._output_tokens}",
          "# TYPE iq36_cached_input_tokens_total counter",
          f"iq36_cached_input_tokens_total {self._cached_tokens}",
          "# TYPE iq36_queue_milliseconds_sum counter",
          f"iq36_queue_milliseconds_sum {self._queue_ms_sum:.6f}",
          "# TYPE iq36_prefill_milliseconds_sum counter",
          f"iq36_prefill_milliseconds_sum {self._prefill_ms_sum:.6f}",
          "# TYPE iq36_decode_milliseconds_sum counter",
          f"iq36_decode_milliseconds_sum {self._decode_ms_sum:.6f}",
          "# TYPE iq36_prefix_restore_milliseconds_sum counter",
          f"iq36_prefix_restore_milliseconds_sum {self._restore_ms_sum:.6f}",
          "# TYPE iq36_http_requests_total counter",
      ]
      for (method, path, status), count in sorted(self._http.items()):
        lines.append(
            "iq36_http_requests_total" + self._labels(
                method=method, path=path, status=status) + f" {count}")
      lines.append("# TYPE iq36_generations_total counter")
      for (endpoint, profile, bucket, finish), count in sorted(
          self._generation.items()):
        lines.append(
            "iq36_generations_total" + self._labels(
                endpoint=endpoint, profile=profile, bucket=bucket,
                finish_reason=finish) + f" {count}")
      lines.append("# TYPE iq36_errors_total counter")
      for kind, count in sorted(self._errors.items()):
        lines.append(
            "iq36_errors_total" + self._labels(kind=kind) + f" {count}")
    return ("\n".join(lines) + "\n").encode("utf-8")
