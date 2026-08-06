#!/usr/bin/env python3
"""Record exact greedy-token agreement across llama, native, and OpenVINO."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-cross-reference-token-consensus-v0"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
TOKEN_FILE = (
    ROOT / "output/r2-native-matrix-20260629T011942Z/token-input/"
    "prefill_shape_008k.tokens.u32")
OPENVINO_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OPENVINO_MODEL = Path("/home/intel/Qwen3.6-35B-A3B-ov")
LIVE_ARTIFACT = (
    ROOT / "output/all-layer-greedy-exact-q4q6-crdiv-"
    "20260711Tseq722cleanZ")

WORKER = r'''#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai


class TokenStreamer(ov_genai.StreamerBase):
  def __init__(self):
    super().__init__()
    self.ids = []

  def write(self, token):
    if hasattr(token, "__iter__"):
      self.ids.extend(int(value) for value in token)
    else:
      self.ids.append(int(token))
    return ov_genai.StreamingStatus.RUNNING

  def end(self):
    pass


def main():
  cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  ids = np.fromfile(
      cfg["token_file"], dtype=np.uint32, count=cfg["token_count"])
  tokenizer = ov_genai.Tokenizer(cfg["model"])
  prompt = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
  roundtrip = np.asarray(tokenizer.encode(prompt).input_ids.data).reshape(-1)
  scheduler = ov_genai.SchedulerConfig()
  scheduler.enable_prefix_caching = False
  scheduler.max_num_batched_tokens = sys.maxsize
  pipeline = ov_genai.VLMPipeline(
      cfg["model"], "GPU", scheduler_config=scheduler,
      DYNAMIC_QUANTIZATION_GROUP_SIZE=256)
  generation = ov_genai.GenerationConfig()
  generation.max_new_tokens = cfg["generated_tokens"]
  generation.ignore_eos = True
  generation.apply_chat_template = False
  streamer = TokenStreamer()
  result = pipeline.generate(
      prompt, generation_config=generation, streamer=streamer)
  payload = {
      "apply_chat_template": False,
      "decoded_text": result.texts[0],
      "generated_token_ids": streamer.ids,
      "input_roundtrip_exact": bool(
          len(roundtrip) == len(ids) and
          np.array_equal(roundtrip.astype(np.uint32), ids)),
      "input_token_count": int(len(ids)),
      "input_token_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
      "openvino_genai_version": ov_genai.__version__,
      "openvino_runtime_version": ov.get_version(),
      "prefix_caching": False,
  }
  print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
  main()
'''


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--token-file", type=Path, default=TOKEN_FILE)
  parser.add_argument("--token-count", type=int, default=1024)
  parser.add_argument("--generated-tokens", type=int, default=8)
  parser.add_argument("--openvino-python", type=Path,
                      default=OPENVINO_PYTHON)
  parser.add_argument("--openvino-model", type=Path, default=OPENVINO_MODEL)
  parser.add_argument("--live-artifact", type=Path, default=LIVE_ARTIFACT)
  parser.add_argument("--timeout-s", type=int, default=300)
  parser.add_argument("--out-dir", type=Path)
  args = parser.parse_args()
  if min(args.token_count, args.generated_tokens, args.timeout_s) <= 0:
    parser.error("counts and timeout must be positive")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/cross-reference-token-consensus-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {"commit": git_output("rev-parse", "HEAD"),
          "dirty": bool(dirty), "dirty_paths": dirty.splitlines()}


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected object")
  return value


def parse_last_json(text: str) -> dict[str, Any]:
  for line in reversed(text.splitlines()):
    try:
      value = json.loads(line)
    except json.JSONDecodeError:
      continue
    if isinstance(value, dict):
      return value
  return {}


def first_divergence(left: list[int], right: list[int]) -> int | None:
  for index, (lhs, rhs) in enumerate(zip(left, right)):
    if lhs != rhs:
      return index
  return None if len(left) == len(right) else min(len(left), len(right))


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  out = args.out_dir.resolve()
  raw = out / "raw"
  raw.mkdir(parents=True, exist_ok=False)
  required = [
      args.token_file, args.openvino_python,
      args.openvino_model / "openvino_language_model.xml",
      args.live_artifact / "result.json",
      args.live_artifact / "raw/baseline/generated-tokens.json",
      args.live_artifact / "raw/injected/generated-tokens.json",
      args.live_artifact / "raw/injected/live-injection-summary.json",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  all_ids = array("I")
  all_ids.frombytes(args.token_file.read_bytes())
  selected = all_ids[:args.token_count]
  selected_bytes = selected.tobytes()
  token_sha256 = hashlib.sha256(selected_bytes).hexdigest()
  baseline = load_json(
      args.live_artifact / "raw/baseline/generated-tokens.json")
  injected = load_json(
      args.live_artifact / "raw/injected/generated-tokens.json")
  injection = load_json(
      args.live_artifact / "raw/injected/live-injection-summary.json")
  llama_ids = [int(value) for value in baseline.get("token_ids", [])]
  native_ids = [int(value) for value in injected.get("token_ids", [])]

  worker = raw / "openvino-token-worker.py"
  config = raw / "openvino-token-config.json"
  worker.write_text(WORKER, encoding="utf-8")
  write_json(config, {
      "generated_tokens": args.generated_tokens,
      "model": str(args.openvino_model.resolve()),
      "token_count": args.token_count,
      "token_file": str(args.token_file.resolve()),
  })
  run = subprocess.run(
      [str(args.openvino_python), str(worker), str(config)], cwd=ROOT,
      check=False, capture_output=True, text=True, timeout=args.timeout_s)
  (raw / "openvino.stdout").write_text(run.stdout, encoding="utf-8")
  (raw / "openvino.stderr").write_text(run.stderr, encoding="utf-8")
  write_json(raw / "openvino.command.json", {
      "command": [str(args.openvino_python), str(worker), str(config)],
      "returncode": run.returncode,
  })
  openvino = parse_last_json(run.stdout)
  openvino_ids = [
      int(value) for value in openvino.get("generated_token_ids", [])]

  llama_openvino_divergence = first_divergence(llama_ids, openvino_ids)
  native_llama_divergence = first_divergence(native_ids, llama_ids)
  native_openvino_divergence = first_divergence(native_ids, openvino_ids)
  reference_consensus = llama_ids == openvino_ids
  candidate_all_reference_match = (
      native_ids == llama_ids and native_ids == openvino_ids)
  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("locked_1024_token_payload_materialized",
            len(all_ids) >= args.token_count and len(selected) == 1024,
            selected_sha256=token_sha256),
      check("clean_native_live_prerequisite_complete",
            injection.get("context_create_count") == 1 and
            injection.get("injection_count") == 40 and
            injection.get("maps_exclude_onednn_openvino") is True),
      check("llama_and_native_eight_token_rows_recorded",
            len(llama_ids) == args.generated_tokens and
            len(native_ids) == args.generated_tokens),
      check("openvino_raw_prompt_greedy_row_recorded",
            run.returncode == 0 and
            openvino.get("apply_chat_template") is False and
            openvino.get("prefix_caching") is False and
            openvino.get("input_roundtrip_exact") is True and
            openvino.get("input_token_count") == args.token_count and
            openvino.get("input_token_sha256") == token_sha256 and
            len(openvino_ids) == args.generated_tokens),
      check("reference_consensus_status_explicitly_recorded", True,
            reference_consensus=reference_consensus,
            first_divergence=llama_openvino_divergence),
  ]
  passed = all(row["pass"] for row in checks)
  result = {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": iso_now(), "git": state,
      "token_file": str(args.token_file),
      "selected_token_count": args.token_count,
      "selected_token_sha256": token_sha256,
      "generated_token_count": args.generated_tokens,
      "live_artifact": str(args.live_artifact),
      "llama_cpp_token_ids": llama_ids,
      "native_injection_token_ids": native_ids,
      "openvino_token_ids": openvino_ids,
      "openvino": openvino,
      "reference_consensus": reference_consensus,
      "candidate_exact_match_to_all_references":
          candidate_all_reference_match,
      "first_divergence": {
          "llama_cpp_vs_openvino": llama_openvino_divergence,
          "native_vs_llama_cpp": native_llama_divergence,
          "native_vs_openvino": native_openvino_divergence,
      },
      "checks": checks, "required_checks_passed": passed,
      "disposition": (
          "accept_reference_consensus_measurement_and_require_consensus_cases"
          if passed and not reference_consensus else
          "accept_reference_consensus_measurement" if passed else
          "reject_incomplete_reference_consensus_measurement"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "reference_consensus": reference_consensus,
      "candidate_exact_match_to_all_references":
          candidate_all_reference_match,
      "checks": checks, "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA, "workstream": WORKSTREAM,
      "created_at": result["created_at"],
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out), "git": state,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  (out / "summary.md").write_text("\n".join([
      "# Cross-reference greedy-token consensus", "",
      f"- measurement complete: `{str(passed).lower()}`",
      f"- llama.cpp: `{llama_ids}`",
      f"- OpenVINO: `{openvino_ids}`",
      f"- native injection: `{native_ids}`",
      f"- reference consensus: `{str(reference_consensus).lower()}`",
      "- speedup claims allowed: `false`", "",
  ]), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out), "pass": passed,
      "llama_cpp": llama_ids, "openvino": openvino_ids,
      "native": native_ids,
      "reference_consensus": reference_consensus,
      "first_divergence": result["first_divergence"],
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
