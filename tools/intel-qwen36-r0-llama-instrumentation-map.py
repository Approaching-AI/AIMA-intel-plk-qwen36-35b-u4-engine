#!/usr/bin/env python3
"""Map R0 oracle boundary tensor types to llama.cpp hook points."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess

import iq36_local
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSTREAM = "intel-qwen36-35b-a3b-gguf-q4km"
SCHEMA_VERSION = "intel-qwen36-r0-llama-instrumentation-map-v0"
DEFAULT_HOST = "local"
EXPECTED_SHA = "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
EXPECTED_SOURCE_DIR = f"/home/intel/intel-qwen36-r0/source/llama.cpp-{EXPECTED_SHA}"


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-llama-instrumentation-map-<UTC>.",
  )
  return parser.parse_args()


def rel(path: Path | None) -> str | None:
  if path is None:
    return None
  return str(path.resolve().relative_to(ROOT))


def latest(pattern: str, filename: str) -> Path | None:
  paths = sorted((ROOT / "output").glob(f"{pattern}/{filename}"))
  return paths[-1] if paths else None


def load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise SystemExit(f"{path}: expected JSON object")
  return value


def write_json(path: Path, value: Any) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      encoding="utf-8",
  )


def run(cmd: list[str], *, timeout_s: int) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
  except subprocess.TimeoutExpired as exc:
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    return {
        "command": cmd,
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr + f"\nlocal timeout after {timeout_s}s",
        "timed_out": True,
    }
  return {
      "command": cmd,
      "returncode": result.returncode,
      "stdout": result.stdout,
      "stderr": result.stderr,
      "timed_out": False,
  }


def run_target(host: str, remote_script: str, *, timeout_s: int) -> dict[str, Any]:
  return iq36_local.run_target(host, remote_script, timeout_s)


def parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key:
      values[key.strip()] = value.strip()
  return values


def bool_text(value: Any) -> bool:
  return value == "true"


def int_or_none(value: Any) -> int | None:
  if not isinstance(value, str) or not value.strip():
    return None
  try:
    return int(value)
  except ValueError:
    return None


def line_location(
    values: dict[str, str],
    key: str,
    file_path: str,
    symbol: str,
    note: str,
    *,
    line_end_key: str | None = None,
) -> dict[str, Any]:
  line_start = int_or_none(values.get(key))
  line_end = int_or_none(values.get(line_end_key)) if line_end_key else line_start
  return {
      "file": file_path,
      "line_end": line_end,
      "line_start": line_start,
      "note": note,
      "symbol": symbol,
  }


def load_boundary_types(capture_spec_path: Path) -> list[str]:
  capture_spec = load_json(capture_spec_path)
  specs = capture_spec.get("boundary_specs", [])
  if not isinstance(specs, list):
    raise SystemExit(f"{capture_spec_path}: boundary_specs must be a list")
  boundary_types = []
  for spec in specs:
    if not isinstance(spec, dict) or not isinstance(spec.get("boundary_type"), str):
      raise SystemExit(f"{capture_spec_path}: malformed boundary spec")
    boundary_types.append(spec["boundary_type"])
  return boundary_types


def source_stage_from_route(source_route_path: Path) -> dict[str, Any]:
  route = load_json(source_route_path)
  stage = route.get("target_source_stage", {})
  if not isinstance(stage, dict):
    raise SystemExit(f"{source_route_path}: target_source_stage missing")
  return {
      "expected_sha": EXPECTED_SHA,
      "source_route_path": rel(source_route_path.parent),
      "source_stage_dir": stage.get("stage_dir") or EXPECTED_SOURCE_DIR,
      "source_stage_ready": stage.get("source_ready_for_instrumentation") is True,
      "source_stage_sha": stage.get("source_rev_parse"),
  }


def write_remote_output(raw_dir: Path, name: str, result: dict[str, Any]) -> None:
  (raw_dir / f"{name}.stdout").write_text(result["stdout"], encoding="utf-8")
  (raw_dir / f"{name}.stderr").write_text(result["stderr"], encoding="utf-8")


def capture_source_snippets(host: str, stage_dir: str, raw_dir: Path) -> dict[str, Any]:
  probe_script = f"""
set -u
cd {shlex.quote(stage_dir)}
line() {{
  grep -nE "$2" "$1" | head -1 | cut -d: -f1 || true
}}
printf 'source_rev_parse='; git rev-parse HEAD
printf 'source_status_short_count='; git status --short | wc -l
printf 'qwen35moe_cpp_present='; test -f src/models/qwen35moe.cpp && echo true || echo false
printf 'llama_graph_cpp_present='; test -f src/llama-graph.cpp && echo true || echo false
printf 'llama_sampler_cpp_present='; test -f src/llama-sampler.cpp && echo true || echo false
printf 'llama_model_cpp_present='; test -f src/llama-model.cpp && echo true || echo false
printf 'llama_graph_h_present='; test -f src/llama-graph.h && echo true || echo false
printf 'model_dispatch_qwen35moe_line='; line src/llama-model.cpp 'case LLM_ARCH_QWEN35MOE'
printf 'qwen35moe_graph_ctor_line='; line src/models/qwen35moe.cpp 'llama_model_qwen35moe::graph::graph'
printf 'build_inp_embd_line='; line src/models/qwen35moe.cpp 'inpL = build_inp_embd'
printf 'model_input_embed_cb_line='; line src/models/qwen35moe.cpp 'model\\.input_embed'
printf 'attn_norm_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "attn_norm"'
printf 'attn_residual_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "attn_residual"'
printf 'attn_post_norm_line='; line src/models/qwen35moe.cpp 'cb\\(attn_post_norm, "attn_post_norm"'
printf 'post_moe_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "post_moe"'
printf 'result_norm_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "result_norm"'
printf 'result_output_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "result_output"'
printf 'build_layer_attn_line='; line src/models/qwen35moe.cpp 'build_layer_attn\\('
printf 'qcur_full_line='; line src/models/qwen35moe.cpp 'Qcur_full = build_lora_mm'
printf 'qcur_normed_line='; line src/models/qwen35moe.cpp 'cb\\(Qcur, "Qcur_normed"'
printf 'kcur_line='; line src/models/qwen35moe.cpp 'Kcur = build_lora_mm'
printf 'vcur_line='; line src/models/qwen35moe.cpp 'Vcur = build_lora_mm'
printf 'kcur_normed_line='; line src/models/qwen35moe.cpp 'cb\\(Kcur, "Kcur_normed"'
printf 'rope_q_line='; line src/models/qwen35moe.cpp 'Qcur = ggml_rope_multi'
printf 'rope_k_line='; line src/models/qwen35moe.cpp 'Kcur = ggml_rope_multi'
printf 'attn_call_line='; line src/models/qwen35moe.cpp 'cur = build_attn\\(inp,'
printf 'attn_pregate_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "attn_pregate"'
printf 'attn_gated_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "attn_gated"'
printf 'attn_output_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "attn_output"'
printf 'build_layer_ffn_line='; line src/models/qwen35moe.cpp 'build_layer_ffn\\(ggml_tensor'
printf 'build_moe_ffn_call_line='; line src/models/qwen35moe.cpp 'build_moe_ffn\\(cur,'
printf 'ffn_moe_out_qwen_line='; line src/models/qwen35moe.cpp 'cb\\(moe_out, "ffn_moe_out"'
printf 'shared_expert_start_line='; line src/models/qwen35moe.cpp 'ffn_up_shexp != nullptr'
printf 'shared_expert_gate_line='; line src/models/qwen35moe.cpp 'cb\\(shared_gate, "shared_expert_gate"'
printf 'shared_expert_gated_line='; line src/models/qwen35moe.cpp 'cb\\(ffn_shexp, "ffn_shexp_gated"'
printf 'ffn_out_shared_line='; line src/models/qwen35moe.cpp 'cb\\(cur, "ffn_out"'
printf 'ffn_out_else_line='; line src/models/qwen35moe.cpp 'cur = moe_out;'
printf 'llama_graph_cb_line='; line src/llama-graph.cpp 'void llm_graph_context::cb'
printf 'context_graph_get_cb_line='; line src/llama-context.cpp 'llm_graph_cb llama_context::graph_get_cb'
printf 'graph_cb_type_line='; line src/llama-graph.h 'using llm_graph_cb'
printf 'inp_tokens_line='; line src/llama-graph.cpp 'cb\\(inp->tokens, "inp_tokens"'
printf 'inp_embd_line='; line src/llama-graph.cpp 'cb\\(inp->embd, "inp_embd"'
printf 'embd_line='; line src/llama-graph.cpp 'cb\\(cur, "embd"'
printf 'moe_helper_line='; line src/llama-graph.cpp 'ggml_tensor \\* llm_graph_context::build_moe_ffn'
printf 'moe_logits_line='; line src/llama-graph.cpp 'cb\\(logits, "ffn_moe_logits"'
printf 'moe_probs_line='; line src/llama-graph.cpp 'cb\\(probs, "ffn_moe_probs"'
printf 'moe_topk_line='; line src/llama-graph.cpp 'cb\\(selected_experts, "ffn_moe_topk"'
printf 'moe_weights_line='; line src/llama-graph.cpp 'cb\\(weights, "ffn_moe_weights_norm"'
printf 'moe_weights_fallback_line='; line src/llama-graph.cpp 'cb\\(weights, "ffn_moe_weights"'
printf 'moe_gate_up_line='; line src/llama-graph.cpp 'cb\\(gate_up, "ffn_moe_gate_up"'
printf 'moe_gate_line='; line src/llama-graph.cpp 'cb\\(cur, "ffn_moe_gate"'
printf 'moe_up_line='; line src/llama-graph.cpp 'cb\\(up, "ffn_moe_up"'
printf 'moe_swiglu_line='; line src/llama-graph.cpp 'cb\\(cur, "ffn_moe_swiglu"'
printf 'moe_down_line='; line src/llama-graph.cpp 'cb\\(experts, "ffn_moe_down"'
printf 'moe_weighted_line='; line src/llama-graph.cpp 'cb\\(experts, "ffn_moe_weighted"'
printf 'moe_out_line='; line src/llama-graph.cpp 'cb\\(moe_out, "ffn_moe_out"'
printf 'kq_softmax_line='; line src/llama-graph.cpp 'cb\\(kq, "kq_soft_max"'
printf 'kqv_line='; line src/llama-graph.cpp 'cb\\(kqv, "kqv"'
printf 'kqv_out_line='; line src/llama-graph.cpp 'cb\\(cur, "kqv_out"'
printf 'sampler_sample_line='; line src/llama-sampler.cpp 'llama_token llama_sampler_sample'
printf 'sampler_logits_line='; line src/llama-sampler.cpp 'llama_get_logits_ith'
printf 'sampler_apply_line='; line src/llama-sampler.cpp 'llama_sampler_apply\\(smpl, &cur_p\\)'
printf 'sampler_token_line='; line src/llama-sampler.cpp 'auto token = cur_p.data'
printf 'sampler_top_k_backend_line='; line src/llama-sampler.cpp 'llama_sampler_top_k_backend_apply'
printf 'sampler_top_k_line='; line src/llama-sampler.cpp 'ggml_top_k\\(ctx, data->logits'
printf 'sampler_top_k_rows_line='; line src/llama-sampler.cpp 'ggml_set_name\\(top_k_rows'
"""
  probe = run_target(host, probe_script, timeout_s=30)
  write_remote_output(raw_dir, "line_probes", probe)
  values = parse_key_values(probe["stdout"])

  snippets = {
      "llama_model_dispatch": "nl -ba src/llama-model.cpp | sed -n '276,288p'",
      "qwen35moe_main_graph": "nl -ba src/models/qwen35moe.cpp | sed -n '156,255p'",
      "qwen35moe_attention": "nl -ba src/models/qwen35moe.cpp | sed -n '284,365p'",
      "qwen35moe_ffn": "nl -ba src/models/qwen35moe.cpp | sed -n '499,553p'",
      "llama_graph_cb_inputs": (
          "nl -ba src/llama-graph.cpp | sed -n '1040,1050p'; "
          "nl -ba src/llama-context.cpp | sed -n '2304,2311p'; "
          "nl -ba src/llama-context.cpp | sed -n '2344,2350p'; "
          "nl -ba src/llama-graph.h | sed -n '584,584p'; "
          "nl -ba src/llama-graph.cpp | sed -n '1800,1808p'; "
          "nl -ba src/llama-graph.cpp | sed -n '1858,1868p'"
      ),
      "llama_graph_moe": "nl -ba src/llama-graph.cpp | sed -n '1436,1790p'",
      "llama_graph_attention_core": (
          "nl -ba src/llama-graph.cpp | sed -n '2118,2138p'; "
          "nl -ba src/llama-graph.cpp | sed -n '2208,2224p'"
      ),
      "llama_sampler": (
          "nl -ba src/llama-sampler.cpp | sed -n '806,870p'; "
          "nl -ba src/llama-sampler.cpp | sed -n '1246,1304p'"
      ),
  }
  raw_status = {"line_probes": {
      "returncode": probe["returncode"],
      "timed_out": probe["timed_out"],
  }}
  snippet_files: list[str] = []
  for name, script in snippets.items():
    result = run_target(
        host,
        f"set -u\ncd {shlex.quote(stage_dir)}\n{script}",
        timeout_s=30,
    )
    write_remote_output(raw_dir, name, result)
    raw_status[name] = {
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
    }
    snippet_files.append(rel(raw_dir / f"{name}.stdout") or str(raw_dir / f"{name}.stdout"))
  return {
      "raw_command_status": raw_status,
      "raw_snippet_files": snippet_files,
      "values": values,
  }


def boundary_mapping(values: dict[str, str]) -> list[dict[str, Any]]:
  loc = line_location
  return [
      {
          "boundary_type": "embedding",
          "capture_scope": "global",
          "hook_kind": "existing_cb_plus_input_tensor",
          "input_tensor_cues": ["inp_tokens", "token_id"],
          "output_tensor_cues": ["embd", "model.input_embed"],
          "source_locations": [
              loc(values, "inp_tokens_line", "src/llama-graph.cpp", "llm_graph_context::build_inp_embd", "token id input tensor"),
              loc(values, "build_inp_embd_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "main graph embedding call"),
              loc(values, "model_input_embed_cb_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "model input embedding callback"),
              loc(values, "embd_line", "src/llama-graph.cpp", "llm_graph_context::build_inp_embd", "materialized embedding callback"),
          ],
      },
      {
          "boundary_type": "layer_input_rmsnorm",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_pre_input_hook",
          "input_tensor_cues": ["inpL"],
          "output_tensor_cues": ["attn_norm"],
          "source_locations": [
              loc(values, "attn_norm_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "RMSNorm output before attention"),
          ],
      },
      {
          "boundary_type": "qkv_projection",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_projection_input_hook",
          "input_tensor_cues": ["attn_norm"],
          "output_tensor_cues": ["Qcur_full", "Qcur_normed", "Kcur", "Kcur_normed", "Vcur"],
          "source_locations": [
              loc(values, "qcur_full_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "Q projection and gate projection"),
              loc(values, "qcur_normed_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "normalized Q tensor"),
              loc(values, "kcur_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "K projection"),
              loc(values, "vcur_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "V projection"),
              loc(values, "kcur_normed_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "normalized K tensor"),
          ],
      },
      {
          "boundary_type": "rope",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_position_input_hook",
          "input_tensor_cues": ["Qcur_normed", "Kcur_normed", "inp_pos"],
          "output_tensor_cues": ["Qcur", "Kcur"],
          "source_locations": [
              loc(values, "rope_q_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "RoPE applied to Q"),
              loc(values, "rope_k_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "RoPE applied to K"),
          ],
      },
      {
          "boundary_type": "attention",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_kv_cache_context",
          "input_tensor_cues": ["Qcur", "Kcur", "Vcur", "k_cache", "v_cache"],
          "output_tensor_cues": ["kq_soft_max", "kqv", "kqv_out", "attn_pregate", "attn_gated"],
          "source_locations": [
              loc(values, "attn_call_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "attention call from Qwen35MoE layer"),
              loc(values, "kq_softmax_line", "src/llama-graph.cpp", "llm_graph_context::build_attn_mha", "attention probabilities"),
              loc(values, "kqv_line", "src/llama-graph.cpp", "llm_graph_context::build_attn_mha", "attention value matmul"),
              loc(values, "kqv_out_line", "src/llama-graph.cpp", "llm_graph_context::build_attn", "attention context output"),
              loc(values, "attn_gated_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "Qwen attention gate applied before output projection"),
          ],
      },
      {
          "boundary_type": "attention_output_projection",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_projection_input_hook",
          "input_tensor_cues": ["attn_gated"],
          "output_tensor_cues": ["attn_output"],
          "source_locations": [
              loc(values, "attn_gated_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "input to output projection"),
              loc(values, "attn_output_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_attn", "attention output projection callback"),
          ],
      },
      {
          "boundary_type": "post_attention_residual",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_residual_input_hook",
          "input_tensor_cues": ["attn_output", "inpSA"],
          "output_tensor_cues": ["attn_residual"],
          "source_locations": [
              loc(values, "attn_residual_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "attention residual add output"),
          ],
      },
      {
          "boundary_type": "ffn_rmsnorm",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_pre_input_hook",
          "input_tensor_cues": ["attn_residual"],
          "output_tensor_cues": ["attn_post_norm"],
          "source_locations": [
              loc(values, "attn_post_norm_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "post-attention RMSNorm feeding MoE FFN"),
          ],
      },
      {
          "boundary_type": "router_topk",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_multiple_outputs",
          "input_tensor_cues": ["attn_post_norm", "ffn_gate_inp"],
          "output_tensor_cues": ["ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights", "ffn_moe_weights_norm"],
          "source_locations": [
              loc(values, "moe_logits_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "router logits"),
              loc(values, "moe_probs_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "router probabilities"),
              loc(values, "moe_topk_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "selected expert ids"),
              loc(values, "moe_weights_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "normalized selected expert weights"),
          ],
      },
      {
          "boundary_type": "selected_expert_gate_up",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_fused_or_separate_expert_projection",
          "input_tensor_cues": ["attn_post_norm", "selected_experts"],
          "output_tensor_cues": ["ffn_moe_gate_up", "ffn_moe_gate", "ffn_moe_up"],
          "source_locations": [
              loc(values, "build_moe_ffn_call_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_ffn", "Qwen35MoE routed expert call"),
              loc(values, "moe_gate_up_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "fused gate/up expert matmul path"),
              loc(values, "moe_gate_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "separate gate expert path"),
              loc(values, "moe_up_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "separate up expert path"),
          ],
      },
      {
          "boundary_type": "swiglu",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb",
          "input_tensor_cues": ["ffn_moe_gate", "ffn_moe_up"],
          "output_tensor_cues": ["ffn_moe_swiglu"],
          "source_locations": [
              loc(values, "moe_swiglu_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "routed expert SwiGLU"),
          ],
      },
      {
          "boundary_type": "selected_expert_down",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_selected_expert_context",
          "input_tensor_cues": ["ffn_moe_swiglu", "selected_experts", "ffn_moe_weights"],
          "output_tensor_cues": ["ffn_moe_down", "ffn_moe_weighted"],
          "source_locations": [
              loc(values, "moe_down_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "selected expert down projection"),
              loc(values, "moe_weighted_line", "src/llama-graph.cpp", "llm_graph_context::build_moe_ffn", "expert output after router weight"),
          ],
      },
      {
          "boundary_type": "shared_expert",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_if_present",
          "input_tensor_cues": ["attn_post_norm", "ffn_up_shexp", "ffn_gate_shexp", "ffn_gate_inp_shexp"],
          "output_tensor_cues": ["ffn_shexp", "shared_expert_gate", "shared_expert_gate_sigmoid", "ffn_shexp_gated"],
          "source_locations": [
              loc(values, "shared_expert_start_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_ffn", "shared expert branch"),
              loc(values, "shared_expert_gate_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_ffn", "shared expert gate"),
              loc(values, "shared_expert_gated_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_ffn", "gated shared expert output"),
          ],
      },
      {
          "boundary_type": "moe_residual",
          "capture_scope": "per_layer",
          "hook_kind": "existing_cb_with_residual_input_hook",
          "input_tensor_cues": ["ffn_out", "ffn_residual"],
          "output_tensor_cues": ["post_moe"],
          "source_locations": [
              loc(values, "ffn_moe_out_qwen_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_ffn", "routed MoE output before shared expert combine"),
              loc(values, "ffn_out_shared_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::build_layer_ffn", "MoE/shared expert combined output"),
              loc(values, "post_moe_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "residual add after MoE"),
          ],
      },
      {
          "boundary_type": "final_norm",
          "capture_scope": "global",
          "hook_kind": "existing_cb_with_pre_input_hook",
          "input_tensor_cues": ["l_out", "inpL"],
          "output_tensor_cues": ["h_nextn", "result_norm"],
          "source_locations": [
              loc(values, "result_norm_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "final RMSNorm output"),
          ],
      },
      {
          "boundary_type": "lm_head",
          "capture_scope": "global",
          "hook_kind": "existing_cb_with_projection_input_hook",
          "input_tensor_cues": ["result_norm"],
          "output_tensor_cues": ["result_output", "t_logits"],
          "source_locations": [
              loc(values, "result_output_line", "src/models/qwen35moe.cpp", "llama_model_qwen35moe::graph::graph", "LM head logits output"),
          ],
      },
      {
          "boundary_type": "sampler",
          "capture_scope": "global",
          "hook_kind": "sampler_output_hook_required",
          "input_tensor_cues": ["result_output", "llama_get_logits_ith", "top_k_rows"],
          "output_tensor_cues": ["sampled token id", "top_k_candidates", "top_k_rows"],
          "source_locations": [
              loc(values, "sampler_sample_line", "src/llama-sampler.cpp", "llama_sampler_sample", "CPU sampler entry"),
              loc(values, "sampler_logits_line", "src/llama-sampler.cpp", "llama_sampler_sample", "logits read for sampling"),
              loc(values, "sampler_apply_line", "src/llama-sampler.cpp", "llama_sampler_sample", "sampler chain application"),
              loc(values, "sampler_token_line", "src/llama-sampler.cpp", "llama_sampler_sample", "selected token id"),
              loc(values, "sampler_top_k_line", "src/llama-sampler.cpp", "llama_sampler_top_k_backend_apply", "backend top-k logits path"),
              loc(values, "sampler_top_k_rows_line", "src/llama-sampler.cpp", "llama_sampler_top_k_backend_apply", "backend top-k logits rows"),
          ],
      },
  ]


def build_summary(payload: dict[str, Any]) -> str:
  source = payload["source_stage"]
  coverage = payload["coverage"]
  lines = [
      "# R0 llama.cpp Instrumentation Map",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- source dir: `{source['source_stage_dir']}`",
      f"- source SHA: `{source['source_rev_parse']}`",
      f"- required boundary types: {coverage['required_boundary_type_count']}",
      f"- mapped boundary types: {coverage['mapped_boundary_type_count']}",
      f"- route status: `{payload['route_status']}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This artifact maps hook points only. It does not patch source, build a",
      "runtime, dump tensors, create an oracle bundle, or close R0.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-llama-instrumentation-map-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  source_route_path = latest("r0-llama-source-build-route-*", "source-route.json")
  boundary_preflight_path = latest("r0-boundary-capture-route-preflight-*", "preflight.json")
  capture_spec_path = latest("r0-oracle-capture-spec-*", "capture-spec.json")
  capture_queue_path = latest("r0-oracle-capture-queue-*", "capture-queue.json")
  oracle_contract_path = ROOT / "oracle/oracle-bundle-contract.json"
  model_contract_path = ROOT / "contracts/qwen36-35b-a3b-gguf-q4km-model-contract.json"
  for name, path in (
      ("source route", source_route_path),
      ("boundary preflight", boundary_preflight_path),
      ("capture spec", capture_spec_path),
      ("capture queue", capture_queue_path),
  ):
    if path is None:
      raise SystemExit(f"no latest {name} artifact found under output/")

  assert source_route_path is not None
  assert boundary_preflight_path is not None
  assert capture_spec_path is not None
  assert capture_queue_path is not None
  stage = source_stage_from_route(source_route_path)
  stage_dir = stage["source_stage_dir"]
  if not isinstance(stage_dir, str) or not stage_dir:
    stage_dir = EXPECTED_SOURCE_DIR
  source_probe = capture_source_snippets(args.host, stage_dir, raw_dir)
  values = source_probe["values"]
  boundary_types = load_boundary_types(capture_spec_path)
  mappings = boundary_mapping(values)
  mapped_types = [item["boundary_type"] for item in mappings]
  missing_types = sorted(set(boundary_types) - set(mapped_types))
  extra_types = sorted(set(mapped_types) - set(boundary_types))
  unmapped_locations = [
      item["boundary_type"]
      for item in mappings
      if not item.get("source_locations")
      or any(loc.get("line_start") is None for loc in item["source_locations"])
  ]

  graph_callback_surface = {
      "callback_type_location": line_location(
          values,
          "graph_cb_type_line",
          "src/llama-graph.h",
          "llm_graph_cb",
          "graph callback function type",
      ),
      "context_callback_location": line_location(
          values,
          "context_graph_get_cb_line",
          "src/llama-context.cpp",
          "llama_context::graph_get_cb",
          "default tensor naming callback",
      ),
      "graph_context_callback_location": line_location(
          values,
          "llama_graph_cb_line",
          "src/llama-graph.cpp",
          "llm_graph_context::cb",
          "central callback forwarding point",
      ),
      "recommended_patch_shape": [
          "add a disabled-by-default R0 boundary dump mode that filters workstream/model/source token before tensor materialization",
          "reuse existing cb tensor names where sufficient, but add explicit pre-input hooks for residual, projection, and norm inputs",
          "write JSONL rows matching boundary-references/inputs.jsonl and boundary-references/outputs.jsonl with external tensor payload paths",
      ],
  }

  payload = {
      "architecture_route": {
          "architecture": "qwen35moe",
          "dispatch_location": line_location(
              values,
              "model_dispatch_qwen35moe_line",
              "src/llama-model.cpp",
              "LLM_ARCH_QWEN35MOE",
              "model factory dispatches locked GGUF architecture to qwen35moe implementation",
          ),
          "main_graph_location": line_location(
              values,
              "qwen35moe_graph_ctor_line",
              "src/models/qwen35moe.cpp",
              "llama_model_qwen35moe::graph::graph",
              "main decoder graph for locked model architecture",
          ),
          "moe_helper_location": line_location(
              values,
              "moe_helper_line",
              "src/llama-graph.cpp",
              "llm_graph_context::build_moe_ffn",
              "generic MoE router and selected expert implementation used by Qwen35MoE",
          ),
      },
      "boundary_mappings": mappings,
      "capture_scope": {
          "capture_phase": "prefill_last_prompt_token",
          "source_prompt_case_id": "short_math_001",
          "source_token_position": 15,
          "task_inputs": "boundary-references/inputs.jsonl",
          "task_outputs": "boundary-references/outputs.jsonl",
      },
      "coverage": {
          "extra_boundary_types": extra_types,
          "mapped_boundary_type_count": len(set(mapped_types)),
          "missing_boundary_types": missing_types,
          "required_boundary_type_count": len(boundary_types),
          "unmapped_location_boundary_types": unmapped_locations,
      },
      "created_at": created_at,
      "evidence": {
          "boundary_capture_route_preflight": rel(boundary_preflight_path.parent),
          "capture_queue": rel(capture_queue_path.parent),
          "capture_spec": rel(capture_spec_path.parent),
          "model_contract": rel(model_contract_path),
          "oracle_contract": rel(oracle_contract_path),
          "raw_dir": rel(raw_dir),
          "source_build_route": rel(source_route_path.parent),
      },
      "graph_callback_surface": graph_callback_surface,
      "host": args.host,
      "raw_command_status": source_probe["raw_command_status"],
      "raw_snippet_files": source_probe["raw_snippet_files"],
      "r0_oracle_gate_closed": False,
      "route_status": "source_mapped_ready_for_instrumentation_patch",
      "schema_version": SCHEMA_VERSION,
      "source_stage": {
          "expected_sha": EXPECTED_SHA,
          "llama_graph_cpp_present": bool_text(values.get("llama_graph_cpp_present")),
          "llama_graph_h_present": bool_text(values.get("llama_graph_h_present")),
          "llama_model_cpp_present": bool_text(values.get("llama_model_cpp_present")),
          "llama_sampler_cpp_present": bool_text(values.get("llama_sampler_cpp_present")),
          "qwen35moe_cpp_present": bool_text(values.get("qwen35moe_cpp_present")),
          "source_matches_expected_sha": values.get("source_rev_parse") == EXPECTED_SHA,
          "source_matches_source_route": values.get("source_rev_parse") == stage.get("source_stage_sha"),
          "source_rev_parse": values.get("source_rev_parse"),
          "source_route_stage_ready": stage.get("source_stage_ready") is True,
          "source_stage_dir": stage_dir,
          "source_status_short_count": int_or_none(values.get("source_status_short_count")),
      },
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "source_stage_matches_exact_llama_build_commit",
          "pass": payload["source_stage"]["source_matches_expected_sha"] is True
          and payload["source_stage"]["source_matches_source_route"] is True
          and payload["source_stage"]["source_route_stage_ready"] is True,
      },
      {
          "name": "required_source_files_present",
          "pass": all(
              payload["source_stage"][key] is True
              for key in (
                  "qwen35moe_cpp_present",
                  "llama_graph_cpp_present",
                  "llama_sampler_cpp_present",
                  "llama_model_cpp_present",
                  "llama_graph_h_present",
              )
          ),
      },
      {
          "name": "locked_architecture_route_identified",
          "pass": payload["architecture_route"]["dispatch_location"]["line_start"] is not None
          and payload["architecture_route"]["main_graph_location"]["line_start"] is not None,
      },
      {
          "name": "all_required_boundary_types_mapped",
          "pass": len(boundary_types) == 17
          and payload["coverage"]["mapped_boundary_type_count"] == 17
          and missing_types == []
          and extra_types == [],
          "missing_boundary_types": missing_types,
          "extra_boundary_types": extra_types,
      },
      {
          "name": "mapped_boundaries_have_line_level_locations",
          "pass": unmapped_locations == [],
          "unmapped_location_boundary_types": unmapped_locations,
      },
      {
          "name": "graph_callback_surface_identified",
          "pass": graph_callback_surface["callback_type_location"]["line_start"] is not None
          and graph_callback_surface["context_callback_location"]["line_start"] is not None
          and graph_callback_surface["graph_context_callback_location"]["line_start"] is not None,
      },
      {
          "name": "instrumentation_map_does_not_close_oracle_gate",
          "pass": payload["r0_oracle_gate_closed"] is False
          and payload["route_status"] == "source_mapped_ready_for_instrumentation_patch",
      },
  ]
  correctness = {
      "checks": checks,
      "gate": "r0_llama_instrumentation_map",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-llama-instrumentation-map.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "instrumentation-map.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("required_boundary_type_count", len(boundary_types)),
        ("mapped_boundary_type_count", payload["coverage"]["mapped_boundary_type_count"]),
        ("source_matches_expected_sha", payload["source_stage"]["source_matches_expected_sha"]),
        ("source_status_short_count", payload["source_stage"]["source_status_short_count"]),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_llama_instrumentation_map",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"llama instrumentation map output: {out_dir}")
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
