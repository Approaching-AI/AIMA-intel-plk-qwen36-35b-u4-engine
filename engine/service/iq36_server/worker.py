from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import traceback
from pathlib import Path

from .runtime import OpenVinoRuntime, RuntimeConfig


_write_lock = threading.Lock()


def _emit(value: dict) -> None:
  with _write_lock:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


def _read_commands(commands: queue.Queue, active: dict) -> None:
  for line in sys.stdin:
    try:
      value = json.loads(line)
      if not isinstance(value, dict):
        continue
    except json.JSONDecodeError:
      continue
    command = value.get("command")
    if command == "cancel":
      if value.get("request_id") == active.get("request_id"):
        event = active.get("cancel")
        if event is not None:
          event.set()
    elif command == "shutdown":
      event = active.get("cancel")
      if event is not None:
        event.set()
      commands.put(value)
      return
    else:
      commands.put(value)


def _load_config(path: Path) -> RuntimeConfig:
  value = json.loads(path.read_text(encoding="utf-8"))
  return RuntimeConfig(
      repo_root=Path(value["repo_root"]),
      model_dir=Path(value["model_dir"]),
      device=str(value["device"]),
      plugin=Path(value["plugin"]),
      custom_config=Path(value["custom_config"]),
      profile=str(value["profile"]),
      bucket=int(value["bucket"]),
      compile_cache_dir=Path(value["compile_cache_dir"]),
      prefix_cache_bytes=int(value["prefix_cache_bytes"]),
      prefix_cache_entries=int(value["prefix_cache_entries"]),
      prefix_cache_ttl_s=float(value["prefix_cache_ttl_s"]),
      prewarm=bool(value.get("prewarm", True)),
  )


def main(argv=None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=Path, required=True)
  args = parser.parse_args(argv)
  try:
    runtime = OpenVinoRuntime(_load_config(args.config))
  except Exception as error:
    _emit({
        "event": "fatal", "message": str(error),
        "traceback": traceback.format_exc()})
    return 1
  _emit({"event": "ready", **runtime.ready_info()})
  commands: queue.Queue = queue.Queue()
  active: dict = {}
  reader = threading.Thread(
      target=_read_commands, args=(commands, active), daemon=True)
  reader.start()
  while True:
    command = commands.get()
    if command.get("command") == "shutdown":
      break
    if command.get("command") != "generate":
      _emit({"event": "error", "message": "unknown worker command"})
      continue
    request_id = str(command.get("request_id"))
    cancel = threading.Event()
    active.update(request_id=request_id, cancel=cancel)
    try:
      result = runtime.generate(
          request_id,
          tuple(int(item) for item in command["prompt_token_ids"]),
          dict(command["params"]), cancel, _emit,
          use_prefix_cache=bool(command.get("prefix_cache", True)))
      _emit(result)
    except Exception as error:
      _emit({
          "event": "error", "request_id": request_id,
          "message": str(error), "traceback": traceback.format_exc()})
    finally:
      active.clear()
  runtime.close()
  _emit({"event": "stopped"})
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
