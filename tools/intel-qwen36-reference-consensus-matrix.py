#!/usr/bin/env python3
"""Measure llama.cpp/OpenVINO greedy-token consensus on canonical raw tokens."""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "intel-qwen36-reference-consensus-matrix-v1"
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
MODEL = Path("/home/intel/models/gguf/qwen3.6-35b-a3b-q4_k_m.gguf")
MODEL_BYTES = 21_166_755_168
TOKEN_INPUT_DIR = (
    ROOT / "output/r1-engine-seed-prompt-input-check-20260627T155328Z/"
    "token-input")
FRESH_CORPUS = (
    ROOT / "doc/active/intel-qwen36-35b-a3b-gguf-q4km/"
    "intel-qwen36-35b-a3b-gguf-q4km-state-conditioned-head-correction-"
    "corpus-2026-07-10.json")
FRESH_TOKEN_INPUT_DIR = (
    ROOT / "output/seq571-state-conditioned-head-correction-token-input-"
    "20260710Tseq571Z/token-input")
DEFAULT_CASES = (
    "short_math_001",
    "short_factual_002",
    "short_transform_003",
    "router_math_reason_001",
    "router_code_reason_002",
    "router_instruction_003",
)
PROMPT_FILES = (
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/"
    "deterministic-greedy.jsonl",
    ROOT / "benchmarks/intel-qwen36-35b-a3b-gguf-q4km/prompts/"
    "router-stability.jsonl",
)
ENV_SCRIPT = Path(
    "/home/intel/intel-box-run/current/tools/intel/activate-intel-box-env.sh")
CXX = Path("/home/intel/intel-box-env/conda/bin/c++")
LLAMA_SOURCE = Path(
    "/home/intel/intel-qwen36-r0/source/"
    "llama.cpp-7c158fbb4aec1bdc9c81d6ca0e785139f4826fae")
LLAMA_BUILD = Path(
    "/home/intel/intel-qwen36-r0/build/"
    "llama-qwen36-boundary-capture-noflash-20260629T234151Z")
OPENVINO_PYTHON = Path("/home/intel/ov/openvino_env/bin/python")
OPENVINO_MODEL = Path("/home/intel/Qwen3.6-35B-A3B-ov")


OPENVINO_WORKER = r'''#!/usr/bin/env python3
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
  config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
  scheduler = ov_genai.SchedulerConfig()
  scheduler.enable_prefix_caching = False
  scheduler.max_num_batched_tokens = sys.maxsize
  pipeline = ov_genai.VLMPipeline(
      config["model"], "GPU", scheduler_config=scheduler,
      DYNAMIC_QUANTIZATION_GROUP_SIZE=256)
  tokenizer = pipeline.get_tokenizer()
  generation = ov_genai.GenerationConfig()
  generation.max_new_tokens = config["generated_tokens"]
  generation.ignore_eos = True
  generation.apply_chat_template = False
  rows = []
  failures = []
  for case in config["cases"]:
    try:
      ids = np.fromfile(case["token_file"], dtype=np.uint32)
      prompt = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
      roundtrip = np.asarray(
          tokenizer.encode(prompt).input_ids.data).reshape(-1)
      streamer = TokenStreamer()
      result = pipeline.generate(
          prompt, generation_config=generation, streamer=streamer)
      perf = result.perf_metrics
      rows.append({
          "case_id": case["case_id"],
          "decoded_prompt_sha256": hashlib.sha256(
              prompt.encode("utf-8")).hexdigest(),
          "generated_token_ids": streamer.ids,
          "input_roundtrip_exact": bool(
              len(roundtrip) == len(ids) and
              np.array_equal(roundtrip.astype(np.uint32), ids)),
          "input_token_count": int(len(ids)),
          "input_token_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
          "runtime_input_token_count": int(perf.get_num_input_tokens()),
      })
    except Exception as exc:
      failures.append({"case_id": case["case_id"], "error": repr(exc)})
  payload = {
      "apply_chat_template": False,
      "failures": failures,
      "openvino_genai_version": ov_genai.__version__,
      "openvino_runtime_version": ov.get_version(),
      "prefix_caching": False,
      "rows": rows,
  }
  Path(config["result_path"]).write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return 0 if not failures else 2


if __name__ == "__main__":
  raise SystemExit(main())
'''


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=MODEL)
  parser.add_argument("--corpus", choices=("canonical", "fresh"),
                      default="canonical")
  parser.add_argument("--token-input-dir", type=Path)
  parser.add_argument("--case-id", action="append", default=[])
  parser.add_argument("--generated-tokens", type=int, default=8)
  parser.add_argument("--threads", type=int, default=16)
  parser.add_argument("--env-script", type=Path, default=ENV_SCRIPT)
  parser.add_argument("--cxx", type=Path, default=CXX)
  parser.add_argument("--llama-source", type=Path, default=LLAMA_SOURCE)
  parser.add_argument("--llama-build", type=Path, default=LLAMA_BUILD)
  parser.add_argument("--openvino-python", type=Path,
                      default=OPENVINO_PYTHON)
  parser.add_argument("--openvino-model", type=Path,
                      default=OPENVINO_MODEL)
  parser.add_argument("--timeout-s", type=int, default=1200)
  parser.add_argument("--out-dir", type=Path)
  parser.add_argument("--plan-only", action="store_true")
  args = parser.parse_args()
  if min(args.generated_tokens, args.threads, args.timeout_s) <= 0:
    parser.error("generated tokens, threads, and timeout must be positive")
  if not args.case_id:
    if args.corpus == "canonical":
      args.case_id = list(DEFAULT_CASES)
    else:
      corpus = json.loads(FRESH_CORPUS.read_text(encoding="utf-8"))
      args.case_id = [str(row["id"]) for row in corpus.get("prompts", [])]
  if args.token_input_dir is None:
    args.token_input_dir = (
        TOKEN_INPUT_DIR if args.corpus == "canonical" else
        FRESH_TOKEN_INPUT_DIR)
  if len(args.case_id) != len(set(args.case_id)):
    parser.error("case ids must be unique")
  if args.out_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir = ROOT / f"output/reference-consensus-matrix-{stamp}"
  return args


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.write_text(
      "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
      encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git_output(*args: str) -> str:
  result = subprocess.run(
      ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
  return result.stdout.strip() if result.returncode == 0 else ""


def git_state() -> dict[str, Any]:
  dirty = git_output("status", "--porcelain")
  return {
      "commit": git_output("rev-parse", "HEAD"),
      "dirty": bool(dirty),
      "dirty_paths": dirty.splitlines(),
  }


def run(command: list[str], timeout_s: int) -> dict[str, Any]:
  try:
    process = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s)
    timed_out = False
  except subprocess.TimeoutExpired as exc:
    process = subprocess.CompletedProcess(
        command, 124,
        exc.stdout if isinstance(exc.stdout, str) else "",
        exc.stderr if isinstance(exc.stderr, str) else "")
    timed_out = True
  return {
      "command": command,
      "returncode": process.returncode,
      "stdout": process.stdout,
      "stderr": process.stderr,
      "timed_out": timed_out,
  }


def run_env(command: list[str], env_script: Path,
            timeout_s: int) -> dict[str, Any]:
  shell = " && ".join([
      f"source {shlex.quote(str(env_script))} >/dev/null 2>&1",
      "export INTEL_FORCE_PROBE=b080 DNNL_VERBOSE=0",
      shlex.join(command),
  ])
  result = run(["bash", "-lc", shell], timeout_s)
  result["logical_command"] = command
  return result


def write_run(raw: Path, name: str, result: dict[str, Any]) -> None:
  (raw / f"{name}.stdout").write_text(
      str(result.get("stdout", "")), encoding="utf-8")
  (raw / f"{name}.stderr").write_text(
      str(result.get("stderr", "")), encoding="utf-8")
  write_json(raw / f"{name}.command.json", {
      key: result.get(key) for key in (
          "command", "logical_command", "returncode", "timed_out")
      if key in result
  })


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def declared_token_cases(corpus: str) -> dict[str, dict[str, Any]]:
  if corpus == "fresh":
    contract = load_json(FRESH_CORPUS)
    return {
        str(row["id"]): {
            **row,
            "kind": "token_exact",
            "prompt_set": "fresh-" + str(row.get("split")),
        }
        for row in contract.get("prompts", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
  cases: dict[str, dict[str, Any]] = {}
  for prompt_file in PROMPT_FILES:
    for line in prompt_file.read_text(encoding="utf-8").splitlines():
      row = json.loads(line)
      if isinstance(row, dict) and isinstance(row.get("id"), str):
        cases[row["id"]] = row
  return cases


def case_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
  declared = declared_token_cases(args.corpus)
  cases = []
  for case_id in args.case_id:
    if declared.get(case_id, {}).get("kind") != "token_exact":
      raise SystemExit(f"case is not a declared token_exact prompt: {case_id}")
    token_file = args.token_input_dir / f"{case_id}.tokens.u32"
    if not token_file.is_file() or token_file.stat().st_size % 4:
      raise SystemExit(f"missing or invalid token file: {token_file}")
    payload = token_file.read_bytes()
    cases.append({
        "case_id": case_id,
        "domain": declared[case_id].get("domain"),
        "prompt_set": declared[case_id].get("prompt_set"),
        "split": declared[case_id].get("split"),
        "token_count": len(payload) // 4,
        "token_file": str(token_file.resolve()),
        "token_sha256": sha256_bytes(payload),
    })
  return cases


def first_divergence(left: list[int], right: list[int]) -> int | None:
  for index, (lhs, rhs) in enumerate(zip(left, right)):
    if lhs != rhs:
      return index
  return None if len(left) == len(right) else min(len(left), len(right))


def check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
  return {"name": name, "pass": bool(passed), **evidence}


def main() -> int:
  args = parse_args()
  cases = case_plan(args)
  plan = {
      "apply_chat_template": False,
      "cases": cases,
      "corpus": args.corpus,
      "generated_tokens": args.generated_tokens,
      "prefix_caching": False,
  }
  if args.plan_only:
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0

  out = args.out_dir.resolve()
  raw = out / "raw"
  build_dir = raw / "build"
  llama_root = raw / "llama"
  raw.mkdir(parents=True, exist_ok=False)
  build_dir.mkdir()
  llama_root.mkdir()
  required = [
      args.model, args.env_script, args.cxx,
      args.llama_source / "include/llama.h",
      args.llama_source / "ggml/include/ggml.h",
      args.llama_build / "bin/libllama.so.0.0.1",
      ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp",
      ROOT / "engine/src/grouped_s8_u4_prefill_runtime.cpp",
      args.openvino_python,
      args.openvino_model / "openvino_language_model.xml",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))

  state = git_state()
  binary = build_dir / "reference-capture"
  build_command = [
      str(args.cxx), "-std=gnu++17", "-O3", "-DNDEBUG",
      "-DIQ36_GROUPED_LIVE_INJECTION", "-DGGML_BACKEND_SHARED",
      "-DGGML_SHARED", "-DGGML_USE_CPU", "-DLLAMA_SHARED",
      f"-I{args.llama_source / 'include'}",
      f"-I{args.llama_source / 'ggml/include'}",
      f"-I{ROOT / 'engine/include'}",
      str(ROOT / "engine/tools/q5_teacher_forced_boundary_capture.cpp"),
      str(ROOT / "engine/src/grouped_s8_u4_prefill_runtime.cpp"),
      f"-L{args.llama_build / 'bin'}",
      f"-Wl,-rpath,{args.llama_build / 'bin'}",
      "-Wl,-l:libllama.so.0.0.1", "-Wl,-l:libggml.so.0.13.1",
      "-Wl,-l:libggml-cpu.so.0.13.1",
      "-Wl,-l:libggml-base.so.0.13.1", "-fopenmp", "-pthread",
      "-lOpenCL", "-o", str(binary),
  ]
  build = run_env(build_command, args.env_script, args.timeout_s)
  write_run(raw, "build", build)

  llama_rows: dict[str, dict[str, Any]] = {}
  for case in cases:
    case_id = str(case["case_id"])
    case_out = llama_root / case_id
    command = [
        str(binary), "--model", str(args.model.resolve()),
        "--token-ids-file", str(case["token_file"]),
        "--binary-u32-token-file", "--token-count",
        str(case["token_count"]), "--batch-all", "--threads",
        str(args.threads), "--n-ctx",
        str(max(128, int(case["token_count"]) + args.generated_tokens + 16)),
        "--ngl", "0", "--top-k", "16",
        "--predicts-generated-position", "0",
        "--live-injection-boundaries", "--no-tensor-dumps",
        "--generate-count", str(args.generated_tokens),
        "--out-dir", str(case_out), "--case-id", case_id,
    ]
    result = (
        run_env(command, args.env_script, args.timeout_s)
        if build["returncode"] == 0 else {
            "command": command, "logical_command": command,
            "returncode": 125, "stdout": "", "stderr": "build failed",
            "timed_out": False,
        })
    write_run(raw, f"llama-{case_id}", result)
    generated_path = case_out / "generated-tokens.json"
    generated = load_json(generated_path) if generated_path.is_file() else {}
    llama_rows[case_id] = {
        "case_id": case_id,
        "generated_token_ids": [
            int(value) for value in generated.get("token_ids", [])],
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
    }

  ldd = run(["ldd", str(binary)], args.timeout_s) if binary.is_file() else {
      "command": ["ldd", str(binary)], "returncode": 125,
      "stdout": "", "stderr": "build failed", "timed_out": False,
  }
  write_run(raw, "ldd", ldd)

  worker_path = raw / "openvino-worker.py"
  worker_config_path = raw / "openvino-config.json"
  worker_result_path = raw / "openvino-result.json"
  worker_path.write_text(OPENVINO_WORKER, encoding="utf-8")
  write_json(worker_config_path, {
      **plan,
      "model": str(args.openvino_model.resolve()),
      "result_path": str(worker_result_path),
  })
  openvino_run = run([
      str(args.openvino_python), str(worker_path), str(worker_config_path)
  ], args.timeout_s)
  write_run(raw, "openvino", openvino_run)
  openvino = (
      load_json(worker_result_path) if worker_result_path.is_file() else {})
  openvino_rows = {
      str(row.get("case_id")): row for row in openvino.get("rows", [])
      if isinstance(row, dict)
  }

  rows = []
  for case in cases:
    case_id = str(case["case_id"])
    llama = llama_rows.get(case_id, {})
    ov = openvino_rows.get(case_id, {})
    llama_ids = [int(value) for value in llama.get(
        "generated_token_ids", [])]
    ov_ids = [int(value) for value in ov.get("generated_token_ids", [])]
    complete = (
        llama.get("returncode") == 0 and
        len(llama_ids) == args.generated_tokens and
        len(ov_ids) == args.generated_tokens and
        ov.get("input_roundtrip_exact") is True and
        ov.get("input_token_count") == case["token_count"] and
        ov.get("runtime_input_token_count") == case["token_count"] and
        ov.get("input_token_sha256") == case["token_sha256"])
    rows.append({
        **case,
        "complete": complete,
        "first_divergence": first_divergence(llama_ids, ov_ids),
        "llama_cpp_token_ids": llama_ids,
        "openvino_token_ids": ov_ids,
        "reference_consensus": complete and llama_ids == ov_ids,
    })

  consensus_rows = [row for row in rows if row["reference_consensus"]]
  complete_rows = [row for row in rows if row["complete"]]
  expected_case_ids = (
      list(DEFAULT_CASES) if args.corpus == "canonical" else
      [str(row["id"]) for row in load_json(FRESH_CORPUS).get("prompts", [])]
  )
  ldd_lower = str(ldd.get("stdout", "")).lower()
  checks = [
      check("repository_clean_at_gate", state["dirty"] is False,
            dirty_paths=state["dirty_paths"]),
      check("locked_model_bound",
            args.model.resolve() == MODEL.resolve() and
            args.model.stat().st_size == MODEL_BYTES,
            model=str(args.model.resolve()),
            model_bytes=args.model.stat().st_size),
      check("full_preregistered_corpus_selected",
            args.case_id == expected_case_ids and
            len(cases) == len(expected_case_ids) and
            (args.corpus != "fresh" or
             {row.get("split") for row in cases} == {
                 "fit", "validation", "test"}),
            corpus=args.corpus,
            case_ids=args.case_id),
      check("llama_reference_runner_builds", build["returncode"] == 0),
      check("llama_reference_runner_native_links",
            ldd.get("returncode") == 0 and
            "openvino" not in ldd_lower and "dnnl" not in ldd_lower),
      check("all_llama_rows_complete",
            len(complete_rows) == len(cases) and all(
                llama_rows[row["case_id"]].get("returncode") == 0
                for row in rows)),
      check("openvino_raw_protocol_complete",
            openvino_run["returncode"] == 0 and
            openvino.get("apply_chat_template") is False and
            openvino.get("prefix_caching") is False and
            not openvino.get("failures") and
            len(openvino_rows) == len(cases)),
      check("minimum_three_reference_consensus_cases",
            len(consensus_rows) >= 3,
            consensus_case_ids=[row["case_id"] for row in consensus_rows],
            consensus_count=len(consensus_rows)),
  ]
  passed = all(row["pass"] for row in checks)
  created_at = iso_now()
  result = {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "git": state,
      "config": plan,
      "model": {
          "path": str(args.model.resolve()),
          "bytes": args.model.stat().st_size,
      },
      "llama_cpp": {
          "source": str(args.llama_source),
          "source_commit": args.llama_source.name.rsplit("-", 1)[-1],
          "runner_sha256": sha256_file(binary) if binary.is_file() else None,
      },
      "openvino": {
          "model": str(args.openvino_model.resolve()),
          "openvino_genai_version": openvino.get("openvino_genai_version"),
          "openvino_runtime_version": openvino.get(
              "openvino_runtime_version"),
          "apply_chat_template": openvino.get("apply_chat_template"),
          "prefix_caching": openvino.get("prefix_caching"),
      },
      "case_count": len(rows),
      "complete_case_count": len(complete_rows),
      "reference_consensus_count": len(consensus_rows),
      "reference_consensus_case_ids": [
          row["case_id"] for row in consensus_rows],
      "rows": rows,
      "checks": checks,
      "required_checks_passed": passed,
      "disposition": (
          "accept_full_corpus_census_and_three_case_reference_consensus_gate"
          if passed else "reject_incomplete_reference_consensus_gate"),
      "speedup_claims_allowed": False,
  }
  write_json(out / "result.json", result)
  write_jsonl(out / "case-results.jsonl", rows)
  write_json(out / "correctness.json", {
      "schema_version": SCHEMA,
      "checks": checks,
      "reference_consensus_count": len(consensus_rows),
      "reference_consensus_case_ids": [
          row["case_id"] for row in consensus_rows],
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  write_json(out / "manifest.json", {
      "schema_version": SCHEMA,
      "workstream": WORKSTREAM,
      "created_at": created_at,
      "tool": str(Path(__file__).resolve().relative_to(ROOT)),
      "artifact": str(out),
      "git": state,
      "required_checks_passed": passed,
      "speedup_claims_allowed": False,
  })
  summary = [
      "# Reference token-consensus matrix", "",
      f"- required checks passed: `{str(passed).lower()}`",
      f"- complete cases: `{len(complete_rows)} / {len(rows)}`",
      f"- consensus cases: `{len(consensus_rows)} / {len(rows)}`",
      "", "| case | llama.cpp | OpenVINO | consensus | first divergence |",
      "|---|---|---|---:|---:|",
  ]
  for row in rows:
    summary.append(
        f"| {row['case_id']} | `{row['llama_cpp_token_ids']}` | "
        f"`{row['openvino_token_ids']}` | "
        f"{str(row['reference_consensus']).lower()} | "
        f"{row['first_divergence']} |")
  summary.extend(["", "Speedup claims allowed: `false`.", ""])
  (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
  print(json.dumps({
      "artifact": str(out),
      "pass": passed,
      "complete_cases": len(complete_rows),
      "consensus_cases": len(consensus_rows),
      "consensus_case_ids": [row["case_id"] for row in consensus_rows],
  }, sort_keys=True))
  return 0 if passed else 2


if __name__ == "__main__":
  raise SystemExit(main())
