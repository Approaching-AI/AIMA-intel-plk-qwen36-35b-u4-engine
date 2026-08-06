from __future__ import annotations

import argparse
import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

from .model_identity import MODEL_CONTRACT_RELATIVE


SHORT_PLUGIN_SHA256 = (
    "b63eede5177f4f9e05d02e97d9f24f52b4289504c2a7c7b4e06c580d1d880e12")
LONG_PLUGIN_SHA256 = (
    "01c04ced415a7b7a5e5bda77a995b2b97b68eb3d9f2c5f3396844d042ddda269")
CUSTOM_CONFIG_SHA256 = (
    "bd7a679031bbde2fa2626f2138bf79a5626469ccbc041faadef3b12e811200ad")
PROMOTED_CONTEXT_CEILING = 131584
PROMOTED_MAX_NEW_TOKENS = 512


def _repo_root() -> Path:
  source_root = Path(__file__).resolve().parents[3]
  marker = Path("tools/intel_qwen36_openvino_hot_cold_attention.py")
  if (source_root / marker).is_file():
    return source_root
  working_root = Path.cwd().resolve()
  if (working_root / marker).is_file():
    return working_root
  return source_root


DEFAULT_REPO_ROOT = _repo_root()


def _env_int(name: str, default: int) -> int:
  raw = os.environ.get(name)
  return default if raw in (None, "") else int(raw)


def _env_float(name: str, default: float) -> float:
  raw = os.environ.get(name)
  return default if raw in (None, "") else float(raw)


def _read_secret(path: Path | None, environment_name: str) -> str | None:
  if path is not None:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
      raise ValueError(f"empty API key file: {path}")
    return value
  value = os.environ.get(environment_name)
  return value if value else None


@dataclass(frozen=True)
class ServerConfig:
  host: str = "127.0.0.1"
  port: int = 8000
  model_id: str = "qwen3.6-35b-a3b-u4"
  repo_root: Path = DEFAULT_REPO_ROOT
  model_dir: Path = Path("/home/intel/Qwen3.6-35B-A3B-ov")
  model_verification: str = "full"
  device: str = "GPU"
  backend: str = "openvino"
  api_key: str | None = None
  allow_unauthenticated_non_loopback: bool = False
  max_request_bytes: int = 4 * 1024 * 1024
  max_context_length: int = PROMOTED_CONTEXT_CEILING
  max_new_tokens: int = PROMOTED_MAX_NEW_TOKENS
  max_queue_depth: int = 8
  max_resident_workers: int = 1
  prefix_cache_bytes: int = 2 * 1024 * 1024 * 1024
  prefix_cache_entries: int = 4
  prefix_cache_ttl_s: float = 900.0
  response_store_entries: int = 256
  response_store_bytes: int = 256 * 1024 * 1024
  response_store_ttl_s: float = 3600.0
  request_timeout_s: float = 900.0
  cancel_grace_s: float = 2.0
  shutdown_timeout_s: float = 30.0
  keepalive_timeout_s: float = 30.0
  preload_bucket: int = 2048
  lazy_start: bool = False
  prewarm: bool = True
  min_available_gib: float = 8.0
  abort_below_available_gib: float = 4.0
  log_level: str = "INFO"
  cors_origin: str | None = None
  short_plugin: Path = Path(
      "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu-seq2291/"
      "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
  long_plugin: Path = Path(
      "/home/intel/intel-qwen36-r0/output/openvino-90214e-l0-gpu-seq2119/"
      "bin/intel64/Release/libopenvino_intel_gpu_plugin.so")
  custom_config: Path = (
      DEFAULT_REPO_ROOT / "engine/openvino/custom/iq36_hot_attention_gqa.xml")
  ov_python: Path = Path("/home/intel/ov/openvino_env/bin/python")

  def validate(self, *, check_files: bool = True) -> None:
    if not (1 <= self.port <= 65535):
      raise ValueError("port must be in [1, 65535]")
    if self.backend not in ("openvino", "mock"):
      raise ValueError("backend must be openvino or mock")
    if self.model_verification not in ("full", "metadata", "off"):
      raise ValueError("model verification must be full, metadata, or off")
    if not self.model_id or any(char.isspace() for char in self.model_id):
      raise ValueError("model id must be non-empty and contain no whitespace")
    if any(ord(char) < 32 or ord(char) == 127 for char in self.host):
      raise ValueError("host must not contain control characters")
    if self.api_key == "":
      raise ValueError("API key must not be empty")
    if self.cors_origin is not None and (
        not self.cors_origin or
        any(char in self.cors_origin for char in "\r\n")
    ):
      raise ValueError("CORS origin must be non-empty and single-line")
    if not (1 <= self.max_request_bytes <= 1024 * 1024 * 1024):
      raise ValueError("max request bytes must be in [1, 1 GiB]")
    if not (1 <= self.max_new_tokens <= PROMOTED_MAX_NEW_TOKENS):
      raise ValueError(
          f"max new tokens must be in [1, {PROMOTED_MAX_NEW_TOKENS}]")
    if not (
        self.max_new_tokens < self.max_context_length <=
        PROMOTED_CONTEXT_CEILING):
      raise ValueError(
          "max context length must exceed max new tokens and be no greater "
          f"than the promoted carrier ceiling {PROMOTED_CONTEXT_CEILING}")
    if self.max_queue_depth < 0:
      raise ValueError("max queue depth must be non-negative")
    if self.max_resident_workers < 1:
      raise ValueError("max resident workers must be positive")
    if self.prefix_cache_bytes < 0 or self.prefix_cache_entries < 0:
      raise ValueError("prefix cache bounds must be non-negative")
    if self.prefix_cache_ttl_s < 0:
      raise ValueError("prefix cache TTL must be non-negative")
    if (
        self.response_store_entries < 0 or self.response_store_bytes < 0 or
        self.response_store_ttl_s < 0
    ):
      raise ValueError("response store bounds must be non-negative")
    if (
        self.request_timeout_s <= 0 or self.shutdown_timeout_s <= 0 or
        self.keepalive_timeout_s <= 0 or self.cancel_grace_s <= 0
    ):
      raise ValueError(
          "request, cancel grace, shutdown, and keepalive timeouts must be "
          "positive")
    if self.preload_bucket not in (0, 2048, 4096, 8192):
      raise ValueError("preload bucket must be 0, 2048, 4096, or 8192")
    if not (
        self.min_available_gib > 0 and
        0 < self.abort_below_available_gib <= self.min_available_gib):
      raise ValueError(
          "memory guards require 0 < abort threshold <= admission threshold")
    try:
      address = ipaddress.ip_address(self.host)
      loopback = address.is_loopback
    except ValueError:
      loopback = self.host.lower() == "localhost"
    if (
        not loopback and self.api_key is None and
        not self.allow_unauthenticated_non_loopback):
      raise ValueError(
          "non-loopback bind requires IQ36_API_KEY or --api-key-file; use "
          "--allow-unauthenticated only behind an authenticated proxy")
    if check_files and self.backend == "openvino":
      for label, path in (
          ("model directory", self.model_dir),
          ("repository graph helper", self.repo_root /
           "tools/intel_qwen36_openvino_hot_cold_attention.py"),
          ("repository fixed-FC helper", self.repo_root /
           "tools/intel_qwen36_openvino_fixed_fc.py"),
          ("locked model contract", self.repo_root / MODEL_CONTRACT_RELATIVE),
          ("short GPU plugin", self.short_plugin),
          ("long GPU plugin", self.long_plugin),
          ("custom CONFIG_FILE", self.custom_config),
          ("OpenVINO Python", self.ov_python),
      ):
        if not path.exists():
          raise ValueError(f"{label} does not exist: {path}")


def build_argument_parser() -> argparse.ArgumentParser:
  repo = Path(os.environ.get("IQ36_REPO_ROOT", str(_repo_root())))
  sibling = repo.parent / "intel-qwen36-r0" / "output"
  parser = argparse.ArgumentParser(
      prog="iq36-serve",
      description="Resident OpenAI-compatible IQ36 OpenVINO service")
  parser.add_argument("--host", default=os.environ.get("IQ36_HOST", "127.0.0.1"))
  parser.add_argument("--port", type=int, default=_env_int("IQ36_PORT", 8000))
  parser.add_argument(
      "--model-id", default=os.environ.get(
          "IQ36_MODEL_ID", "qwen3.6-35b-a3b-u4"))
  parser.add_argument("--repo-root", type=Path, default=repo)
  parser.add_argument(
      "--model-dir", type=Path,
      default=Path(os.environ.get(
          "IQ36_MODEL_DIR", "/home/intel/Qwen3.6-35B-A3B-ov")))
  parser.add_argument(
      "--model-verification", choices=("full", "metadata", "off"),
      default=os.environ.get("IQ36_MODEL_VERIFICATION", "full"),
      help="model identity gate; full is required for production")
  parser.add_argument("--device", default=os.environ.get("IQ36_DEVICE", "GPU"))
  parser.add_argument(
      "--backend", choices=("openvino", "mock"),
      default=os.environ.get("IQ36_BACKEND", "openvino"))
  parser.add_argument(
      "--api-key-file", type=Path,
      default=Path(os.environ["IQ36_API_KEY_FILE"])
      if os.environ.get("IQ36_API_KEY_FILE") else None,
      help="read bearer key from a file; IQ36_API_KEY is also supported")
  parser.add_argument("--allow-unauthenticated", action="store_true")
  parser.add_argument(
      "--max-request-bytes", type=int,
      default=_env_int("IQ36_MAX_REQUEST_BYTES", 4 * 1024 * 1024))
  parser.add_argument(
      "--max-context-length", type=int,
      default=_env_int("IQ36_MAX_CONTEXT_LENGTH", PROMOTED_CONTEXT_CEILING))
  parser.add_argument(
      "--max-new-tokens", type=int,
      default=_env_int("IQ36_MAX_NEW_TOKENS", PROMOTED_MAX_NEW_TOKENS))
  parser.add_argument(
      "--max-queue-depth", type=int,
      default=_env_int("IQ36_MAX_QUEUE_DEPTH", 8))
  parser.add_argument(
      "--max-resident-workers", type=int,
      default=_env_int("IQ36_MAX_RESIDENT_WORKERS", 1))
  parser.add_argument(
      "--prefix-cache-bytes", type=int,
      default=_env_int("IQ36_PREFIX_CACHE_BYTES", 2 * 1024 * 1024 * 1024))
  parser.add_argument(
      "--prefix-cache-entries", type=int,
      default=_env_int("IQ36_PREFIX_CACHE_ENTRIES", 4))
  parser.add_argument(
      "--prefix-cache-ttl", type=float,
      default=_env_float("IQ36_PREFIX_CACHE_TTL", 900.0))
  parser.add_argument(
      "--response-store-entries", type=int,
      default=_env_int("IQ36_RESPONSE_STORE_ENTRIES", 256))
  parser.add_argument(
      "--response-store-bytes", type=int,
      default=_env_int("IQ36_RESPONSE_STORE_BYTES", 256 * 1024 * 1024))
  parser.add_argument(
      "--response-store-ttl", type=float,
      default=_env_float("IQ36_RESPONSE_STORE_TTL", 3600.0))
  parser.add_argument(
      "--request-timeout", type=float,
      default=_env_float("IQ36_REQUEST_TIMEOUT", 900.0))
  parser.add_argument(
      "--cancel-grace", type=float,
      default=_env_float("IQ36_CANCEL_GRACE", 2.0),
      help="seconds before terminating a worker that ignores cancellation")
  parser.add_argument(
      "--shutdown-timeout", type=float,
      default=_env_float("IQ36_SHUTDOWN_TIMEOUT", 30.0))
  parser.add_argument(
      "--keepalive-timeout", type=float,
      default=_env_float("IQ36_KEEPALIVE_TIMEOUT", 30.0))
  parser.add_argument(
      "--preload-bucket", type=int,
      default=_env_int("IQ36_PRELOAD_BUCKET", 2048))
  parser.add_argument("--lazy", action="store_true")
  parser.add_argument(
      "--no-prewarm", action="store_true",
      help="mark a compiled worker ready before shape warmup (diagnostic only)")
  parser.add_argument(
      "--min-available-gib", type=float,
      default=_env_float("IQ36_MIN_AVAILABLE_GIB", 8.0))
  parser.add_argument(
      "--abort-below-available-gib", type=float,
      default=_env_float("IQ36_ABORT_BELOW_AVAILABLE_GIB", 4.0))
  parser.add_argument(
      "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
      default=os.environ.get("IQ36_LOG_LEVEL", "INFO").upper())
  parser.add_argument("--cors-origin", default=os.environ.get("IQ36_CORS_ORIGIN"))
  parser.add_argument(
      "--short-plugin", type=Path,
      default=Path(os.environ.get(
          "IQ36_SHORT_PLUGIN",
          str(sibling / "openvino-90214e-l0-gpu-seq2291/bin/intel64/"
              "Release/libopenvino_intel_gpu_plugin.so"))))
  parser.add_argument(
      "--long-plugin", type=Path,
      default=Path(os.environ.get(
          "IQ36_LONG_PLUGIN",
          str(sibling / "openvino-90214e-l0-gpu-seq2119/bin/intel64/"
              "Release/libopenvino_intel_gpu_plugin.so"))))
  parser.add_argument(
      "--custom-config", type=Path,
      default=Path(os.environ.get(
          "IQ36_CUSTOM_CONFIG",
          str(repo / "engine/openvino/custom/iq36_hot_attention_gqa.xml"))))
  parser.add_argument(
      "--ov-python", type=Path,
      default=Path(os.environ.get(
          "IQ36_OV_PYTHON", "/home/intel/ov/openvino_env/bin/python")))
  return parser


def config_from_args(args: argparse.Namespace) -> ServerConfig:
  config = ServerConfig(
      host=args.host,
      port=args.port,
      model_id=args.model_id,
      repo_root=args.repo_root,
      model_dir=args.model_dir,
      model_verification=args.model_verification,
      device=args.device,
      backend=args.backend,
      api_key=_read_secret(args.api_key_file, "IQ36_API_KEY"),
      allow_unauthenticated_non_loopback=args.allow_unauthenticated,
      max_request_bytes=args.max_request_bytes,
      max_context_length=args.max_context_length,
      max_new_tokens=args.max_new_tokens,
      max_queue_depth=args.max_queue_depth,
      max_resident_workers=args.max_resident_workers,
      prefix_cache_bytes=args.prefix_cache_bytes,
      prefix_cache_entries=args.prefix_cache_entries,
      prefix_cache_ttl_s=args.prefix_cache_ttl,
      response_store_entries=args.response_store_entries,
      response_store_bytes=args.response_store_bytes,
      response_store_ttl_s=args.response_store_ttl,
      request_timeout_s=args.request_timeout,
      cancel_grace_s=args.cancel_grace,
      shutdown_timeout_s=args.shutdown_timeout,
      keepalive_timeout_s=args.keepalive_timeout,
      preload_bucket=args.preload_bucket,
      lazy_start=args.lazy,
      prewarm=not args.no_prewarm,
      min_available_gib=args.min_available_gib,
      abort_below_available_gib=args.abort_below_available_gib,
      log_level=args.log_level,
      cors_origin=args.cors_origin,
      short_plugin=args.short_plugin,
      long_plugin=args.long_plugin,
      custom_config=args.custom_config,
      ov_python=args.ov_python,
  )
  config.validate()
  return config
