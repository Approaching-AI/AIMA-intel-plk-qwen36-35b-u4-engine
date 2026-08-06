from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone

from . import __version__
from .application import Application
from .config import build_argument_parser, config_from_args
from .http_server import HTTPServer


class JsonFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:
    value = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    if record.exc_info:
      value["exception"] = self.formatException(record.exc_info)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _configure_logging(level: str) -> None:
  handler = logging.StreamHandler(sys.stderr)
  handler.setFormatter(JsonFormatter())
  root = logging.getLogger()
  root.handlers.clear()
  root.addHandler(handler)
  root.setLevel(level)


async def _serve(config) -> None:
  application = Application(config)
  server = HTTPServer(config, application)
  stop = asyncio.Event()
  loop = asyncio.get_running_loop()
  for name in (signal.SIGINT, signal.SIGTERM):
    try:
      loop.add_signal_handler(name, stop.set)
    except NotImplementedError:
      pass
  logging.getLogger("iq36").info(
      "starting iq36 service version=%s model=%s backend=%s",
      __version__, config.model_id, config.backend)
  try:
    await application.start()
    await server.start()
    await stop.wait()
    logging.getLogger("iq36").info("shutdown requested")
  finally:
    application.begin_shutdown()
    await server.close(timeout_s=config.shutdown_timeout_s)
    try:
      await asyncio.wait_for(
          application.close(), timeout=config.shutdown_timeout_s)
    except asyncio.TimeoutError:
      logging.getLogger("iq36").error(
          "backend shutdown exceeded %.3f seconds",
          config.shutdown_timeout_s)


def main(argv=None) -> int:
  parser = build_argument_parser()
  parser.add_argument("--version", action="version", version=__version__)
  args = parser.parse_args(argv)
  try:
    config = config_from_args(args)
  except (OSError, ValueError) as error:
    parser.error(str(error))
  _configure_logging(config.log_level)
  try:
    asyncio.run(_serve(config))
  except KeyboardInterrupt:
    pass
  except Exception:
    logging.getLogger("iq36").exception("service failed")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
