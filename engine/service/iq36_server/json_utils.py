from __future__ import annotations

import json
from typing import Any


def strict_json_loads(value: str) -> Any:
  def reject_constant(name: str):
    raise ValueError(f"non-finite JSON number: {name}")

  def unique_object(pairs):
    result = {}
    for name, item in pairs:
      if name in result:
        raise ValueError(f"duplicate JSON object key: {name!r}")
      result[name] = item
    return result

  return json.loads(
      value, parse_constant=reject_constant,
      object_pairs_hook=unique_object)
