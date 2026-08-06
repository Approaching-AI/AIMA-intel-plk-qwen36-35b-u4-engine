from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .config import ServerConfig
from .tool_calls import parse_assistant_text
from .types import (
    BackendStatus, GenerationDelta, GenerationResult, GenerationStarted,
    PreparedRequest, TokenLogprob)


LOGGER = logging.getLogger("iq36.backend")
BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)


def available_memory_bytes() -> int:
  with open("/proc/meminfo", encoding="ascii") as handle:
    for line in handle:
      if line.startswith("MemAvailable:"):
        return int(line.split()[1]) * 1024
  raise RuntimeError("/proc/meminfo does not expose MemAvailable")


def select_bucket(prompt_tokens: int) -> int:
  if prompt_tokens < 1:
    raise ValueError("prompt token count must be positive")
  for bucket in BUCKETS:
    if prompt_tokens <= bucket:
      return bucket
  raise ValueError("prompt exceeds the promoted 128k carrier")


def select_profile(bucket: int, request: PreparedRequest) -> str:
  if bucket <= 8192:
    return "short_full"
  return "long_full" if request.params.requires_full_logits else "long_compact"


def _token_logprob(value) -> TokenLogprob | None:
  if not isinstance(value, dict):
    return None
  return TokenLogprob(
      token=str(value.get("token", "")),
      logprob=float(value.get("logprob", 0.0)),
      bytes=tuple(int(item) for item in value.get("bytes", [])),
      top_logprobs=tuple(value.get("top_logprobs", [])))


class WorkerError(RuntimeError):
  pass


def stable_text_delta(
    committed: str, cumulative: str, *, final: bool = False,
) -> tuple[str, str]:
  """Return a monotonic detokenized delta without exposing partial UTF-8."""
  stable = cumulative if final else cumulative.rstrip("\ufffd")
  if not stable.startswith(committed):
    raise WorkerError("detokenizer rewrote an already streamed text prefix")
  return stable, stable[len(committed):]


class WorkerClient:
  def __init__(
      self, config: ServerConfig, profile: str, bucket: int,
      compile_cache_dir: Path,
  ) -> None:
    self.config = config
    self.profile = profile
    self.bucket = bucket
    self.compile_cache_dir = compile_cache_dir
    self._temporary = tempfile.TemporaryDirectory(
        prefix=f"iq36-{profile}-{bucket}-")
    self._temp_path = Path(self._temporary.name)
    self._stderr_tail: deque[str] = deque(maxlen=200)
    self.process: subprocess.Popen | None = None
    self.ready: dict | None = None
    self.last_used = time.monotonic()
    self._io_lock = threading.Lock()
    self._stdin_lock = threading.Lock()

  def start(self, timeout_s: float) -> dict:
    plugin = (
        self.config.short_plugin if self.profile == "short_full"
        else self.config.long_plugin)
    payload = {
        "repo_root": str(self.config.repo_root),
        "model_dir": str(self.config.model_dir),
        "device": self.config.device,
        "plugin": str(plugin),
        "custom_config": str(self.config.custom_config),
        "profile": self.profile,
        "bucket": self.bucket,
        "compile_cache_dir": str(self.compile_cache_dir),
        "prefix_cache_bytes": self.config.prefix_cache_bytes,
        "prefix_cache_entries": self.config.prefix_cache_entries,
        "prefix_cache_ttl_s": self.config.prefix_cache_ttl_s,
        "prewarm": self.config.prewarm,
    }
    path = self._temp_path / "worker-config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    environment = os.environ.copy()
    python_path = [
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[3] / "tools"),
    ]
    if environment.get("PYTHONPATH"):
      python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    self.process = subprocess.Popen(
        [str(self.config.ov_python), "-m", "iq36_server.worker",
         "--config", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=environment, start_new_session=True)
    threading.Thread(target=self._drain_stderr, daemon=True).start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
      event = self._read_event(timeout_s=max(0.0, deadline - time.monotonic()))
      if event is None:
        break
      if event.get("event") == "ready":
        self.ready = event
        return event
      if event.get("event") == "fatal":
        raise WorkerError(event.get("message", "worker failed during startup"))
    self.close(force=True)
    raise WorkerError(
        "worker did not become ready; stderr tail: " +
        " | ".join(self._stderr_tail))

  def _drain_stderr(self) -> None:
    process = self.process
    if process is None or process.stderr is None:
      return
    for line in process.stderr:
      line = line.rstrip()
      self._stderr_tail.append(line)
      LOGGER.debug("worker[%s/%s] %s", self.profile, self.bucket, line)

  def _read_event(self, timeout_s: float | None = None) -> dict | None:
    process = self.process
    if process is None or process.stdout is None:
      return None
    if timeout_s is not None:
      ready, _, _ = select.select([process.stdout], [], [], timeout_s)
      if not ready:
        return None
    while True:
      line = process.stdout.readline()
      if not line:
        return None
      try:
        value = json.loads(line)
      except json.JSONDecodeError:
        self._stderr_tail.append("stdout: " + line.rstrip())
        continue
      if isinstance(value, dict):
        return value

  def _send(self, value: dict) -> None:
    with self._stdin_lock:
      process = self.process
      if process is None or process.stdin is None or process.poll() is not None:
        raise WorkerError("worker process is not running")
      process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
      process.stdin.flush()

  def cancel(self, request_id: str) -> None:
    try:
      self._send({"command": "cancel", "request_id": request_id})
    except (BrokenPipeError, WorkerError):
      pass

  def generate(
      self, request: PreparedRequest, request_id: str,
      cancel: threading.Event,
      on_started: Callable[[GenerationStarted], None],
      on_delta: Callable[[GenerationDelta], None],
      queue_ms: float,
  ) -> GenerationResult:
    with self._io_lock:
      self.last_used = time.monotonic()
      self._send({
          "command": "generate", "request_id": request_id,
          "prompt_token_ids": list(request.prompt_token_ids),
          "params": asdict(request.params),
          "prefix_cache": request.prefix_cache,
      })
      watcher_done = threading.Event()
      def watch_cancel() -> None:
        abort_bytes = int(
            self.config.abort_below_available_gib * 1024 ** 3)
        while not watcher_done.wait(0.25):
          if cancel.is_set():
            self.cancel(request_id)
            if not watcher_done.wait(self.config.cancel_grace_s):
              process = self.process
              if process is not None and process.poll() is None:
                LOGGER.error(
                    "worker ignored cancellation request=%s grace_s=%.3f; "
                    "terminating isolated worker",
                    request_id, self.config.cancel_grace_s)
                process.terminate()
            return
          try:
            available = available_memory_bytes()
          except OSError:
            continue
          if available < abort_bytes:
            LOGGER.error(
                "memory abort request=%s available_bytes=%s threshold_bytes=%s",
                request_id, available, abort_bytes)
            cancel.set()
            self.cancel(request_id)
            if not watcher_done.wait(self.config.cancel_grace_s):
              process = self.process
              if process is not None and process.poll() is None:
                process.terminate()
            return
      threading.Thread(target=watch_cancel, daemon=True).start()
      committed = ""
      try:
        while True:
          event = self._read_event()
          if event is None:
            raise WorkerError(
                "worker exited during generation: " +
                " | ".join(self._stderr_tail))
          kind = event.get("event")
          if event.get("request_id") not in (None, request_id):
            continue
          if kind == "started":
            on_started(GenerationStarted(
                request_id=request_id,
                prompt_tokens=int(event["prompt_tokens"]),
                cached_tokens=int(event["cached_tokens"]),
                profile=str(event["profile"]), bucket=int(event["bucket"]),
                queue_ms=queue_ms,
                prefix_restore_ms=float(event.get("prefix_restore_ms", 0.0))))
          elif kind == "token":
            if request.params.stop:
              continue
            cumulative = str(event.get("text", ""))
            committed, delta = stable_text_delta(committed, cumulative)
            on_delta(GenerationDelta(
                token_id=int(event["token_id"]), text=delta,
                logprob=_token_logprob(event.get("logprob"))))
          elif kind == "done":
            final_text = str(event.get("text", ""))
            committed, final_delta = stable_text_delta(
                committed, final_text, final=True)
            token_ids = tuple(
                int(item) for item in event.get("token_ids", []))
            if final_delta:
              on_delta(GenerationDelta(
                  token_id=token_ids[-1] if token_ids else -1,
                  text=final_delta))
            parsed = parse_assistant_text(
                final_text, request_id,
                allow_tool_calls=bool(request.tools))
            return GenerationResult(
                request_id=request_id, text=parsed.content,
                token_ids=token_ids,
                prompt_tokens=int(event["prompt_tokens"]),
                cached_tokens=int(event["cached_tokens"]),
                finish_reason=str(event["finish_reason"]),
                profile=str(event["profile"]), bucket=int(event["bucket"]),
                prefill_ms=float(event.get("prefill_ms", 0.0)),
                decode_ms=float(event.get("decode_ms", 0.0)),
                prefix_restore_ms=float(event.get("prefix_restore_ms", 0.0)),
                logprobs=tuple(
                    item for value in event.get("logprobs", [])
                    if (item := _token_logprob(value)) is not None),
                tool_calls=parsed.tool_calls, reasoning=parsed.reasoning)
          elif kind in ("error", "fatal"):
            raise WorkerError(str(event.get("message", "worker error")))
      finally:
        watcher_done.set()

  def close(self, *, force: bool = False) -> None:
    process = self.process
    if process is not None and process.poll() is None:
      if not force:
        try:
          self._send({"command": "shutdown"})
          process.wait(timeout=10)
        except (BrokenPipeError, WorkerError, subprocess.TimeoutExpired):
          force = True
      if force and process.poll() is None:
        process.terminate()
        try:
          process.wait(timeout=5)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait(timeout=5)
    for stream in (
        process.stdin if process else None,
        process.stdout if process else None,
        process.stderr if process else None):
      if stream is not None:
        stream.close()
    self.process = None
    self._temporary.cleanup()


class ResidentBackend:
  def __init__(self, config: ServerConfig) -> None:
    self.config = config
    self._workers: OrderedDict[tuple[str, int], WorkerClient] = OrderedDict()
    self._generation_lock = threading.Lock()
    self._pool_lock = threading.Lock()
    self._active_worker: WorkerClient | None = None
    self._active_request: str | None = None
    self._last_error: str | None = None
    self._closed = False
    self.compile_cache_dir = Path(os.environ.get(
        "IQ36_COMPILE_CACHE_DIR", str(Path.home() / ".cache/iq36/openvino")))

  def start(self) -> None:
    if self.config.lazy_start or self.config.preload_bucket == 0:
      return
    self._get_worker("short_full", self.config.preload_bucket)

  def _get_worker(self, profile: str, bucket: int) -> WorkerClient:
    key = (profile, bucket)
    with self._pool_lock:
      worker = self._workers.pop(key, None)
      if worker is not None and worker.process is not None \
          and worker.process.poll() is None:
        self._workers[key] = worker
        return worker
      if worker is not None:
        worker.close(force=True)
      while len(self._workers) >= self.config.max_resident_workers:
        _, evicted = self._workers.popitem(last=False)
        evicted.close()
      threshold = int(self.config.min_available_gib * 1024 ** 3)
      deadline = time.monotonic() + 60.0
      available = available_memory_bytes()
      while available < threshold and time.monotonic() < deadline:
        time.sleep(1.0)
        available = available_memory_bytes()
      if available < threshold:
        raise WorkerError(
            "resident worker admission rejected: MemAvailable "
            f"{available / 1024 ** 3:.3f} GiB is below the configured "
            f"{self.config.min_available_gib:.3f} GiB floor")
      worker = WorkerClient(
          self.config, profile, bucket, self.compile_cache_dir)
      try:
        ready = worker.start(self.config.request_timeout_s)
        LOGGER.info(
            "resident worker ready profile=%s bucket=%s compile_ms=%.3f",
            profile, bucket, float(ready.get("compile_ms", 0.0)))
      except Exception as error:
        self._last_error = str(error)
        worker.close(force=True)
        raise
      self._workers[key] = worker
      return worker

  def generate(
      self, request: PreparedRequest, request_id: str,
      cancel: threading.Event,
      on_started: Callable[[GenerationStarted], None],
      on_delta: Callable[[GenerationDelta], None],
  ) -> GenerationResult:
    queued_at = time.monotonic()
    with self._generation_lock:
      queue_ms = (time.monotonic() - queued_at) * 1000.0
      if self._closed:
        raise WorkerError("backend is shutting down")
      bucket = select_bucket(len(request.prompt_token_ids))
      profile = select_profile(bucket, request)
      worker = self._get_worker(profile, bucket)
      self._active_worker = worker
      self._active_request = request_id
      try:
        return worker.generate(
            request, request_id, cancel, on_started, on_delta, queue_ms)
      except Exception as error:
        self._last_error = str(error)
        raise
      finally:
        self._active_worker = None
        self._active_request = None

  def status(self) -> BackendStatus:
    with self._pool_lock:
      workers = tuple(
          worker.ready or {"profile": key[0], "bucket": key[1]}
          for key, worker in self._workers.items()
          if worker.process is not None and worker.process.poll() is None)
    return BackendStatus(
        ready=bool(workers) or self.config.lazy_start,
        active=self._active_request is not None,
        loaded_workers=workers, last_error=self._last_error)

  def close(self) -> None:
    self._closed = True
    worker = self._active_worker
    request_id = self._active_request
    if worker is not None and request_id is not None:
      worker.cancel(request_id)
    with self._generation_lock:
      with self._pool_lock:
        workers = list(self._workers.values())
        self._workers.clear()
      for item in workers:
        item.close()


class MockBackend:
  def __init__(self, config: ServerConfig, text: str = "mock response") -> None:
    self.config = config
    self.text = text
    self._closed = False
    self._active = False

  def start(self) -> None:
    pass

  def generate(
      self, request: PreparedRequest, request_id: str,
      cancel: threading.Event,
      on_started: Callable[[GenerationStarted], None],
      on_delta: Callable[[GenerationDelta], None],
  ) -> GenerationResult:
    self._active = True
    bucket = select_bucket(len(request.prompt_token_ids))
    profile = select_profile(bucket, request)
    on_started(GenerationStarted(
        request_id=request_id, prompt_tokens=len(request.prompt_token_ids),
        cached_tokens=0, profile=profile, bucket=bucket, queue_ms=0.0))
    emitted = []
    for index, char in enumerate(self.text):
      if cancel.is_set() or index >= request.params.max_new_tokens:
        break
      emitted.append(char)
      on_delta(GenerationDelta(token_id=ord(char), text=char))
    finish = "cancelled" if cancel.is_set() else (
        "length" if len(emitted) < len(self.text) else "stop")
    text = "".join(emitted)
    parsed = parse_assistant_text(
        text, request_id, allow_tool_calls=bool(request.tools))
    self._active = False
    return GenerationResult(
        request_id=request_id, text=parsed.content,
        token_ids=tuple(ord(char) for char in emitted),
        prompt_tokens=len(request.prompt_token_ids), cached_tokens=0,
        finish_reason=finish, profile=profile, bucket=bucket,
        prefill_ms=0.1, decode_ms=0.1, prefix_restore_ms=0.0,
        tool_calls=parsed.tool_calls, reasoning=parsed.reasoning)

  def status(self) -> BackendStatus:
    return BackendStatus(
        ready=not self._closed, active=self._active,
        loaded_workers=({"profile": "mock", "bucket": 8192},))

  def close(self) -> None:
    self._closed = True
