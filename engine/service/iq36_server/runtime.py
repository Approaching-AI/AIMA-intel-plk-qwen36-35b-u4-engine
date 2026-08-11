from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable
from xml.sax.saxutils import quoteattr

from .prefix_cache import PrefixCache
from .runtime_identity import validate_runtime_identity


FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
PREFILL_CHUNK_TOKENS = 8192
PREFILL_ALIGNED_SHAPES = (8192, 4096, 2048, 32)
VOCABULARY = 248320
SHORT_PLUGIN_SHA256 = (
    "b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12")
LONG_PLUGIN_SHA256 = (
    "c0515a401f579620c2fb440031e87e848ceaefab572715d4ace2b76ff2956121")
CUSTOM_CONFIG_SHA256 = (
    "bd7a679031bbde2fa2626f2138bf79a5626469ccbc041faadef3b12e811200ad")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _load_graph(root: Path):
  tools_dir = root / "tools"
  if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))
  path = tools_dir / "intel_qwen36_openvino_hot_cold_attention.py"
  spec = importlib.util.spec_from_file_location("iq36_service_graph", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load graph transformer: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _token_or_logits_custom_class(ov):
  class IQ36GreedyTokenOrLogits(ov.Op):
    def __init__(self, inputs=None):
      super().__init__(self, inputs)
      if inputs is not None:
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
      self.set_output_type(0, ov.Type.i32, ov.PartialShape([1, 1, 1, 1]))

    def clone_with_new_inputs(self, new_inputs):
      return IQ36GreedyTokenOrLogits(new_inputs)

    def visit_attributes(self, visitor):
      del visitor
      return True

  return IQ36GreedyTokenOrLogits


@dataclass(frozen=True)
class RuntimeConfig:
  repo_root: Path
  model_dir: Path
  device: str
  plugin: Path
  custom_config: Path
  profile: str
  bucket: int
  compile_cache_dir: Path
  prefix_cache_bytes: int
  prefix_cache_entries: int
  prefix_cache_ttl_s: float
  prewarm: bool


@dataclass
class StateSnapshot:
  states: dict[str, Any]
  next_output: Any
  byte_count: int


class OpenVinoRuntime:
  """One compiled profile/bucket and one strictly serial InferRequest."""

  def __init__(self, config: RuntimeConfig) -> None:
    self.config = config
    self.compact = config.profile == "long_compact"
    if config.profile not in ("short_full", "long_compact", "long_full"):
      raise ValueError(f"unsupported worker profile: {config.profile}")
    if config.bucket not in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
      raise ValueError(f"unsupported bucket: {config.bucket}")
    if config.profile == "short_full" and config.bucket > 8192:
      raise ValueError("short_full is limited to 2k/4k/8k")
    if config.profile != "short_full" and config.bucket <= 8192:
      raise ValueError("long profiles are limited to 16k+")

    expected_plugin = (
        SHORT_PLUGIN_SHA256 if config.profile == "short_full"
        else LONG_PLUGIN_SHA256)
    observed_plugin = sha256_file(config.plugin)
    if observed_plugin != expected_plugin:
      raise RuntimeError(
          f"GPU plugin fingerprint mismatch: expected {expected_plugin}, "
          f"observed {observed_plugin}")
    observed_config = sha256_file(config.custom_config)
    if observed_config != CUSTOM_CONFIG_SHA256:
      raise RuntimeError(
          f"CONFIG_FILE fingerprint mismatch: expected {CUSTOM_CONFIG_SHA256}, "
          f"observed {observed_config}")

    self._configure_environment()
    import numpy as np
    import openvino as ov
    import openvino_genai as ov_genai
    import openvino_tokenizers

    self.runtime_identity = validate_runtime_identity(
        ov.get_version(), ov_genai.__version__,
        openvino_tokenizers.__version__)

    self.np = np
    self.ov = ov
    self.tokenizer = ov_genai.Tokenizer(str(config.model_dir))
    self.eos_token_ids = self._load_eos_ids()
    self.graph = _load_graph(config.repo_root)
    self._temporary = tempfile.TemporaryDirectory(prefix="iq36-worker-")
    self._temp_path = Path(self._temporary.name)
    self.compile_started_at = time.monotonic()
    self.core, self.compiled, self.source_summary = self._compile()
    self.compile_ms = (time.monotonic() - self.compile_started_at) * 1000.0
    self.request = self.compiled.create_infer_request()
    self.warmup_ms = 0.0
    if config.prewarm:
      started = time.perf_counter_ns()
      self._warmup()
      self.warmup_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    self.prefix_cache: PrefixCache[StateSnapshot] = PrefixCache(
        max_bytes=config.prefix_cache_bytes,
        max_entries=config.prefix_cache_entries,
        ttl_s=config.prefix_cache_ttl_s)

  def _configure_environment(self) -> None:
    keys = (
        "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN",
        "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE",
        "IQ36_GPU_DQ_REALLOC_FASTPATH",
        "IQ36_GPU_FC_STABLE_PREP_FASTPATH",
        "IQ36_LM_HEAD_I8Q1",
        "IQ36_LM_HEAD_I8Q1_GATED_EXACT",
        "IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4",
        "IQ36_LM_HEAD_I8Q1_GATED_Q4",
        "IQ36_LM_HEAD_I8Q1_GREEDY_LOCAL2",
        "IQ36_LM_HEAD_I8Q1_TOKEN_ONLY",
    )
    for key in keys:
      os.environ.pop(key, None)
    os.environ.update({
        "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN": "1",
        "IQ36_GPU_ALIAS_LINEAR_STATE_ASSIGN_SCOPE": "all",
        "IQ36_GPU_DQ_REALLOC_FASTPATH": "1",
        "IQ36_GPU_FC_STABLE_PREP_FASTPATH": "1",
        "IQ36_LM_HEAD_I8Q1": "1",
    })
    if self.config.profile == "short_full":
      os.environ["IQ36_LM_HEAD_I8Q1_GATED_EXACT"] = "1"
      os.environ["IQ36_LM_HEAD_I8Q1_GATED_EXACT_AFFINE_Q4"] = "1"
    elif self.config.profile == "long_full":
      os.environ["IQ36_LM_HEAD_I8Q1_GATED_EXACT"] = "1"
    else:
      os.environ["IQ36_LM_HEAD_I8Q1_GREEDY_LOCAL2"] = "1"
      os.environ["IQ36_LM_HEAD_I8Q1_TOKEN_ONLY"] = "1"
    cache = self.config.compile_cache_dir / self.config.profile / str(
        self.config.bucket)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NEO_CACHE_DIR"] = str(cache)
    os.environ["NEO_CACHE_PERSISTENT"] = "1"
    os.environ.setdefault("NEO_CACHE_MAX_SIZE", str(8 * 1024 * 1024 * 1024))

  def _load_eos_ids(self) -> frozenset[int]:
    path = self.config.model_dir / "generation_config.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = value.get("eos_token_id", self.tokenizer.get_eos_token_id())
    if isinstance(raw, int):
      raw = [raw]
    return frozenset(int(item) for item in raw)

  def _compile(self):
    np = self.np
    ov = self.ov
    registry = self._temp_path / "plugins.xml"
    registry.write_text(
        "<ie><plugins><plugin name=\"GPU\" location="
        f"{quoteattr(str(self.config.plugin.resolve()))}/></plugins></ie>\n",
        encoding="utf-8")
    core = ov.Core(str(registry))
    core.set_property(
        self.config.device, {"CONFIG_FILE": str(self.config.custom_config)})
    prefill_capacity = max(2 * PREFILL_CHUNK_TOKENS, self.config.bucket)
    source, summary = self.graph.make_candidate_model(
        core, self.config.model_dir, ov, np, FULL_ATTENTION_LAYERS,
        exact_phase_decode=True,
        exact_phase_dual_cohort=True,
        initialize_hot_states=True,
        fixed_cold_capacity=self.config.bucket,
        prefill_history_capacity=prefill_capacity,
        exact_history_layers=FULL_ATTENTION_LAYERS,
        exact_history_capacity=prefill_capacity + 1024,
        fuse_linear_conv_state=True,
        decode_stock_micro_layers=FULL_ATTENTION_LAYERS)

    embedding = core.read_model(
        str(self.config.model_dir / "openvino_text_embeddings_model.xml"))
    embedding_parameter = embedding.get_parameters()[0]
    embedding_value = embedding.get_results()[0].input_value(0)
    inputs_embeds = next(
        parameter for parameter in source.get_parameters()
        if "inputs_embeds" in parameter.output(0).get_names())
    inputs_embeds.output(0).replace(embedding_value)
    source.remove_parameter(inputs_embeds)
    source.add_parameters([embedding_parameter])
    source.validate_nodes_and_infer_types()

    original_result = source.get_results()[0]
    logits_output = original_result.input_value(0)
    logits_shape = ov.opset13.shape_of(logits_output, "i64")
    sequence_length = ov.opset13.gather(
        logits_shape, ov.opset13.constant(np.array(1, dtype=np.int64)),
        ov.opset13.constant(np.array(0, dtype=np.int64)))
    last_index = ov.opset13.subtract(
        sequence_length, ov.opset13.constant(np.array(1, dtype=np.int64)))
    last_logits = ov.opset13.gather(
        logits_output, last_index,
        ov.opset13.constant(np.array(1, dtype=np.int64)))
    last_logits.set_friendly_name("iq36_service_last_query_logits")
    source.remove_result(original_result)
    if self.compact:
      token_input = ov.opset13.reshape(
          last_logits,
          ov.opset13.constant(np.array([1, 1, 1, VOCABULARY], dtype=np.int64)),
          False)
      token_input.set_friendly_name("iq36_service_greedy_token_input")
      token_class = _token_or_logits_custom_class(ov)
      token = token_class([token_input.output(0)])
      token.set_friendly_name("iq36_service_greedy_token")
      source.add_results([ov.opset13.result(token.output(0))])
    else:
      source.add_results([ov.opset13.result(last_logits.output(0))])
    source.validate_nodes_and_infer_types()
    compile_config = {
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": 256,
        "PERFORMANCE_HINT": "LATENCY",
        "ACTIVATIONS_SCALE_FACTOR": 0.0,
    }
    compiled = core.compile_model(source, self.config.device, compile_config)
    return core, compiled, summary

  def ready_info(self) -> dict[str, Any]:
    return {
        "worker_pid": os.getpid(),
        "profile": self.config.profile,
        "bucket": self.config.bucket,
        "compile_ms": self.compile_ms,
        "warmup_ms": self.warmup_ms,
        "plugin_sha256": sha256_file(self.config.plugin),
        "custom_config_sha256": sha256_file(self.config.custom_config),
        "openvino_version": self.ov.get_version(),
        "openvino_genai_version": self.runtime_identity["openvino_genai"],
        "openvino_tokenizers_version": self.runtime_identity[
            "openvino_tokenizers"],
        "runtime_identity_verified": self.runtime_identity["verified"],
        "state_count": len(self.request.query_state()),
        "compact_token_only": self.compact,
    }

  def _make_inputs(
      self, token_ids: list[int], start: int, total: int,
      attention_mask, beam_idx,
  ) -> dict[str, Any]:
    np = self.np
    ids = np.asarray([token_ids], dtype=np.int64)
    count = len(token_ids)
    positions = np.arange(start, start + count, dtype=np.int64)
    return {
        "attention_mask": attention_mask[:, :total],
        "beam_idx": beam_idx,
        "input": ids,
        "position_ids": np.tile(positions, (4, 1)).reshape(4, 1, count),
    }

  def _infer(self, tokens: list[int], start: int, total: int,
             attention_mask, beam_idx):
    outputs = self.request.infer(self._make_inputs(
        tokens, start, total, attention_mask, beam_idx))
    output = self.np.asarray(outputs[self.compiled.output(0)])
    if self.compact:
      return int(output.reshape(-1)[-1])
    return self.np.asarray(output, dtype=self.np.float32).reshape(-1)

  def _warmup(self) -> None:
    np = self.np
    shapes = [min(self.config.bucket, PREFILL_CHUNK_TOKENS)]
    if shapes[0] != 32:
      shapes.append(32)
    for shape in shapes:
      self.request.reset_state()
      attention_mask = np.ones((1, shape + 2), dtype=np.int64)
      beam_idx = np.zeros((1,), dtype=np.int32)
      output = self._infer(
          [1] * shape, 0, shape, attention_mask, beam_idx)
      token_id = int(output) if self.compact else int(np.argmax(output))
      self._infer(
          [token_id], shape, shape + 1, attention_mask, beam_idx)
    self.request.reset_state()

  @staticmethod
  def _prefill_ranges(start: int, end: int):
    """Use tile-aligned prefill shapes, then exact query-one decode tails.

    The promoted prefill operation is numerically valid for query counts that
    are multiples of its 32-token tile.  A non-aligned multi-token query is not
    equivalent to stock OpenVINO.  Decomposing only the final 0..31 tokens as
    query-one calls preserves arbitrary user lengths without padding or token
    changes and keeps the set of compiled shapes finite.
    """
    cursor = start
    remaining = end - start
    for shape in PREFILL_ALIGNED_SHAPES:
      while remaining >= shape:
        yield cursor, cursor + shape
        cursor += shape
        remaining -= shape
    while remaining:
      yield cursor, cursor + 1
      cursor += 1
      remaining -= 1

  def _snapshot(self, next_output) -> StateSnapshot:
    np = self.np
    rows = {}
    byte_count = 0
    for state in self.request.query_state():
      tensor = state.state
      array = np.array(tensor.data, copy=True)
      rows[state.name] = array
      byte_count += int(array.nbytes)
    output = (
        int(next_output) if self.compact
        else np.array(next_output, dtype=np.float32, copy=True))
    if not self.compact:
      byte_count += int(output.nbytes)
    return StateSnapshot(rows, output, byte_count)

  def _restore(self, snapshot: StateSnapshot) -> None:
    np = self.np
    ov = self.ov
    states = {state.name: state for state in self.request.query_state()}
    if set(states) != set(snapshot.states):
      raise RuntimeError("prefix state schema differs from the active worker")
    for name, array in snapshot.states.items():
      states[name].state = ov.Tensor(np.array(array, copy=True))

  def _sample(
      self, output, params: dict[str, Any], counts: Counter[int], rng,
  ) -> tuple[int, dict[str, Any] | None]:
    np = self.np
    if self.compact:
      return int(output), None
    logits = np.array(output, dtype=np.float64, copy=True)
    repetition = float(params.get("repetition_penalty", 1.0))
    presence = float(params.get("presence_penalty", 0.0))
    frequency = float(params.get("frequency_penalty", 0.0))
    for token_id, count in counts.items():
      if repetition != 1.0:
        logits[token_id] = (
            logits[token_id] * repetition if logits[token_id] < 0
            else logits[token_id] / repetition)
      logits[token_id] -= presence + frequency * count

    temperature = float(params.get("temperature", 1.0))
    if temperature == 0.0:
      token = int(np.argmax(logits))
      distribution_logits = logits
    else:
      logits /= temperature
      distribution_logits = logits.copy()
      top_k = int(params.get("top_k", 0))
      if top_k and top_k < logits.size:
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
      top_p = float(params.get("top_p", 1.0))
      if top_p < 1.0:
        order = np.argsort(logits)[::-1]
        finite = np.isfinite(logits[order])
        order = order[finite]
        shifted = logits[order] - logits[order[0]]
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        cumulative = np.cumsum(probabilities)
        keep_count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        keep = order[:max(1, keep_count)]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
      maximum = np.max(logits)
      probabilities = np.exp(logits - maximum)
      probabilities /= probabilities.sum()
      token = int(rng.choice(logits.size, p=probabilities))

    if not bool(params.get("logprobs", False)):
      return token, None
    maximum = np.max(distribution_logits)
    log_norm = maximum + math.log(
        float(np.exp(distribution_logits - maximum).sum()))
    top_count = int(params.get("top_logprobs", 0))
    if top_count:
      top_ids = np.argpartition(distribution_logits, -top_count)[-top_count:]
      top_ids = sorted(
          (int(item) for item in top_ids),
          key=lambda item: float(distribution_logits[item]), reverse=True)
    else:
      top_ids = []
    def token_row(token_id: int) -> dict[str, Any]:
      token_text = str(self.tokenizer.decode(
          [token_id], skip_special_tokens=False))
      return {
          "token": token_text,
          "logprob": float(distribution_logits[token_id] - log_norm),
          "bytes": list(token_text.encode("utf-8")),
      }
    selected = token_row(token)
    selected["top_logprobs"] = [token_row(item) for item in top_ids]
    return token, selected

  @staticmethod
  def _truncate_stop(text: str, stops: tuple[str, ...]) -> tuple[str, bool]:
    positions = [position for stop in stops if (position := text.find(stop)) >= 0]
    if not positions:
      return text, False
    return text[:min(positions)], True

  def generate(
      self, request_id: str, prompt_ids: tuple[int, ...],
      params: dict[str, Any], cancel: Event,
      emit: Callable[[dict[str, Any]], None],
      *, use_prefix_cache: bool = True,
  ) -> dict[str, Any]:
    np = self.np
    if not prompt_ids:
      raise ValueError("empty prompt token sequence")
    if len(prompt_ids) > self.config.bucket:
      raise ValueError(
          f"prompt has {len(prompt_ids)} tokens, exceeding worker bucket "
          f"{self.config.bucket}")
    max_new = int(params["max_new_tokens"])
    attention_mask = np.ones(
        (1, len(prompt_ids) + max_new), dtype=np.int64)
    beam_idx = np.zeros((1,), dtype=np.int32)
    self.request.reset_state()
    prefix_started = time.perf_counter_ns()
    entry = self.prefix_cache.find_longest(prompt_ids) \
        if use_prefix_cache else None
    cached_tokens = 0
    next_output = None
    if entry is not None:
      self._restore(entry.value)
      cached_tokens = len(entry.tokens)
      if cached_tokens == len(prompt_ids):
        next_output = entry.value.next_output
    prefix_restore_ms = (
        (time.perf_counter_ns() - prefix_started) / 1_000_000.0
        if entry is not None else 0.0)
    emit({
        "event": "started", "request_id": request_id,
        "prompt_tokens": len(prompt_ids), "cached_tokens": cached_tokens,
        "profile": self.config.profile, "bucket": self.config.bucket,
        "prefix_restore_ms": prefix_restore_ms,
    })

    prefill_started = time.perf_counter_ns()
    if next_output is None:
      for start, end in self._prefill_ranges(cached_tokens, len(prompt_ids)):
        if cancel.is_set():
          break
        next_output = self._infer(
            [int(item) for item in prompt_ids[start:end]], start, end,
            attention_mask, beam_idx)
    prefill_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000.0
    if cancel.is_set() or next_output is None:
      return {
          "event": "done", "request_id": request_id, "text": "",
          "token_ids": [], "prompt_tokens": len(prompt_ids),
          "cached_tokens": cached_tokens, "finish_reason": "cancelled",
          "profile": self.config.profile, "bucket": self.config.bucket,
          "prefill_ms": prefill_ms, "decode_ms": 0.0,
          "prefix_restore_ms": prefix_restore_ms, "logprobs": [],
          "cache": self.prefix_cache.stats().__dict__,
      }

    if use_prefix_cache and self.prefix_cache.enabled:
      snapshot = self._snapshot(next_output)
      self.prefix_cache.put(prompt_ids, snapshot, snapshot.byte_count)

    generated: list[int] = []
    logprobs: list[dict[str, Any]] = []
    # OpenAI-style presence/frequency penalties apply to text already present,
    # including prompt tokens, and then accumulate generated tokens.
    counts: Counter[int] = Counter(prompt_ids)
    rng = np.random.default_rng(params.get("seed"))
    stops = tuple(str(item) for item in params.get("stop", []))
    finish_reason = "length"
    visible_text = ""
    decode_started = time.perf_counter_ns()
    for step in range(max_new):
      if cancel.is_set():
        finish_reason = "cancelled"
        break
      token_id, token_logprob = self._sample(next_output, params, counts, rng)
      generated.append(token_id)
      counts[token_id] += 1
      if token_logprob is not None:
        logprobs.append(token_logprob)
      decoded = str(self.tokenizer.decode(
          generated, skip_special_tokens=True))
      visible_text, stopped = self._truncate_stop(decoded, stops)
      emit({
          "event": "token", "request_id": request_id,
          "token_id": token_id, "text": visible_text,
          "logprob": token_logprob,
      })
      if stopped or (
          token_id in self.eos_token_ids and
          not bool(params.get("ignore_eos", False))):
        finish_reason = "stop"
        break
      if step + 1 == max_new:
        finish_reason = "length"
        break
      start = len(prompt_ids) + step
      total = start + 1
      next_output = self._infer(
          [token_id], start, total, attention_mask, beam_idx)
    decode_ms = (time.perf_counter_ns() - decode_started) / 1_000_000.0

    state_tokens = prompt_ids + tuple(generated[:-1])
    if (use_prefix_cache and self.prefix_cache.enabled and generated and
        state_tokens != prompt_ids):
      snapshot = self._snapshot(next_output)
      self.prefix_cache.put(state_tokens, snapshot, snapshot.byte_count)
    return {
        "event": "done", "request_id": request_id, "text": visible_text,
        "token_ids": generated, "prompt_tokens": len(prompt_ids),
        "cached_tokens": cached_tokens, "finish_reason": finish_reason,
        "profile": self.config.profile, "bucket": self.config.bucket,
        "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "prefix_restore_ms": prefix_restore_ms, "logprobs": logprobs,
        "cache": self.prefix_cache.stats().__dict__,
    }

  def close(self) -> None:
    self.prefix_cache.clear()
    self.request = None
    self.compiled = None
    self.core = None
    gc.collect()
    self._temporary.cleanup()
