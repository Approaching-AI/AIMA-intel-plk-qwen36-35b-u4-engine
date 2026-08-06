#!/usr/bin/env python3
"""Create and optionally apply the R0 llama.cpp boundary capture tool patch."""

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
SCHEMA_VERSION = "intel-qwen36-r0-boundary-capture-instrumentation-patch-v0"
DEFAULT_HOST = "local"
EXPECTED_SHA = "7c158fbb4aec1bdc9c81d6ca0e785139f4826fae"
REMOTE_SOURCE_DIR = f"/home/intel/intel-qwen36-r0/source/llama.cpp-{EXPECTED_SHA}"


CAPTURE_CMAKE = """set(TARGET llama-qwen36-boundary-capture)
add_executable(${TARGET} qwen36-boundary-capture.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
"""


CAPTURE_CPP = r'''#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model MODEL --prompt-file FILE --out-dir DIR [options]\n"
        "\n"
        "options:\n"
        "  --case-id ID                    case id for JSONL metadata (default: short_math_001)\n"
        "  --source-token-position N       zero-based prompt token position to capture (default: last token)\n"
        "  --threads N                     decode threads (default: 1)\n"
        "  --n-ctx N                       context length (default: prompt tokens + 1)\n"
        "  --ngl N                         GPU layers (default: 0)\n"
        "  --max-tensors N                 stop after N matched tensor dumps (default: unlimited)\n"
        "  --filter REGEX                  additional tensor-name regex, can repeat\n",
        argv0);
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path + ": " + std::strerror(errno));
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    return out;
}

static std::string safe_name(const std::string & name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "unnamed" : out;
}

struct args_t {
    std::string model;
    std::string prompt_file;
    std::string out_dir;
    std::string case_id = "short_math_001";
    int source_token_position = -1;
    int threads = 1;
    int n_ctx = 0;
    int ngl = 0;
    int max_tensors = 0;
    std::vector<std::string> filters;
};

static args_t parse_args(int argc, char ** argv) {
    args_t args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "-m" || a == "--model") {
            args.model = require_value(a.c_str());
        } else if (a == "--prompt-file") {
            args.prompt_file = require_value(a.c_str());
        } else if (a == "--out-dir") {
            args.out_dir = require_value(a.c_str());
        } else if (a == "--case-id") {
            args.case_id = require_value(a.c_str());
        } else if (a == "--source-token-position") {
            args.source_token_position = std::stoi(require_value(a.c_str()));
        } else if (a == "--threads") {
            args.threads = std::stoi(require_value(a.c_str()));
        } else if (a == "--n-ctx") {
            args.n_ctx = std::stoi(require_value(a.c_str()));
        } else if (a == "--ngl") {
            args.ngl = std::stoi(require_value(a.c_str()));
        } else if (a == "--max-tensors") {
            args.max_tensors = std::stoi(require_value(a.c_str()));
        } else if (a == "--filter") {
            args.filters.push_back(require_value(a.c_str()));
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (args.model.empty() || args.prompt_file.empty() || args.out_dir.empty()) {
        usage(argv[0]);
        throw std::runtime_error("missing required --model, --prompt-file, or --out-dir");
    }
    return args;
}

struct capture_state {
    std::filesystem::path out_dir;
    std::ofstream tensor_jsonl;
    std::vector<std::regex> filters;
    std::string case_id;
    int source_token_position = -1;
    int current_token_position = -1;
    int max_tensors = 0;
    int captured = 0;
    bool enabled = false;
};

static bool matches_filters(const capture_state & state, const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    for (const auto & filter : state.filters) {
        if (std::regex_search(name, filter)) {
            return true;
        }
    }
    return false;
}

static void write_tensor_record(capture_state & state, const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr) {
        return;
    }
    if (state.max_tensors > 0 && state.captured >= state.max_tensors) {
        return;
    }

    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> bytes(nbytes);
    ggml_backend_tensor_get(t, bytes.data(), 0, nbytes);

    const std::string name = t->name && t->name[0] ? t->name : "unnamed";
    const int ordinal = state.captured;
    const std::string file_name =
        safe_name(name) + "__tok" + std::to_string(state.current_token_position) +
        "__ord" + std::to_string(ordinal) + ".bin";
    const std::filesystem::path payload_rel = std::filesystem::path("payloads") / file_name;
    const std::filesystem::path payload_path = state.out_dir / payload_rel;

    {
        std::ofstream out(payload_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("failed to open tensor payload for write: " + payload_path.string());
        }
        out.write(reinterpret_cast<const char *>(bytes.data()), (std::streamsize) bytes.size());
    }

    state.tensor_jsonl
        << "{"
        << "\"case_id\":\"" << json_escape(state.case_id) << "\","
        << "\"source_token_position\":" << state.source_token_position << ","
        << "\"observed_token_position\":" << state.current_token_position << ","
        << "\"tensor_name\":\"" << json_escape(name) << "\","
        << "\"tensor_type\":\"" << ggml_type_name(t->type) << "\","
        << "\"tensor_op\":\"" << ggml_op_desc(t) << "\","
        << "\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "],"
        << "\"nb\":[" << t->nb[0] << "," << t->nb[1] << "," << t->nb[2] << "," << t->nb[3] << "],"
        << "\"nbytes\":" << nbytes << ","
        << "\"payload_path\":\"" << json_escape(payload_rel.generic_string()) << "\""
        << "}\n";
    state.captured++;
}

static bool boundary_eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);
    if (state == nullptr || !state->enabled) {
        return ask ? false : true;
    }
    const bool match = matches_filters(*state, t ? t->name : nullptr);
    if (ask) {
        if (state->max_tensors > 0 && state->captured >= state->max_tensors) {
            return false;
        }
        return match;
    }
    if (match) {
        write_tensor_record(*state, t);
    }
    return true;
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        throw std::runtime_error("failed to determine prompt token count");
    }
    std::vector<llama_token> tokens(n_prompt);
    const int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true);
    if (n < 0 || n != n_prompt) {
        throw std::runtime_error("failed to tokenize prompt");
    }
    return tokens;
}

static std::vector<std::pair<int, float>> topk_logits(const float * logits, int n_vocab, int k) {
    std::vector<std::pair<int, float>> rows;
    rows.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        rows.emplace_back(i, logits[i]);
    }
    const int kk = std::min(k, n_vocab);
    std::partial_sort(rows.begin(), rows.begin() + kk, rows.end(),
        [](const auto & a, const auto & b) {
            return a.second > b.second;
        });
    rows.resize(kk);
    return rows;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
        args_t args = parse_args(argc, argv);
        std::filesystem::create_directories(std::filesystem::path(args.out_dir) / "payloads");

        const std::vector<std::string> default_filters = {
            "^inp_tokens$",
            "^embd$",
            "^model\\.input_embed$",
            "^attn_norm-[0-9]+$",
            "^Qcur_full-[0-9]+$",
            "^Qcur_normed-[0-9]+$",
            "^Kcur-[0-9]+$",
            "^Kcur_normed-[0-9]+$",
            "^Vcur-[0-9]+$",
            "^Qcur-[0-9]+$",
            "^kq_soft_max-[0-9]+$",
            "^kqv-[0-9]+$",
            "^kqv_out-[0-9]+$",
            "^attn_pregate-[0-9]+$",
            "^attn_gated-[0-9]+$",
            "^attn_output-[0-9]+$",
            "^attn_residual-[0-9]+$",
            "^attn_post_norm-[0-9]+$",
            "^ffn_moe_logits-[0-9]+$",
            "^ffn_moe_probs-[0-9]+$",
            "^ffn_moe_topk-[0-9]+$",
            "^ffn_moe_weights(_norm)?-[0-9]+$",
            "^ffn_moe_gate_up-[0-9]+$",
            "^ffn_moe_gate-[0-9]+$",
            "^ffn_moe_up-[0-9]+$",
            "^ffn_moe_swiglu-[0-9]+$",
            "^ffn_moe_down-[0-9]+$",
            "^ffn_moe_weighted-[0-9]+$",
            "^ffn_moe_out-[0-9]+$",
            "^shared_expert_gate.*-[0-9]+$",
            "^ffn_shexp.*-[0-9]+$",
            "^ffn_out-[0-9]+$",
            "^post_moe-[0-9]+$",
            "^result_norm$",
            "^result_output$",
        };

        capture_state state;
        state.out_dir = args.out_dir;
        state.case_id = args.case_id;
        state.max_tensors = args.max_tensors;
        state.tensor_jsonl.open(state.out_dir / "tensor-dumps.jsonl");
        if (!state.tensor_jsonl) {
            throw std::runtime_error("failed to open tensor-dumps.jsonl");
        }
        for (const auto & filter : default_filters) {
            state.filters.emplace_back(filter, std::regex::optimize);
        }
        for (const auto & filter : args.filters) {
            state.filters.emplace_back(filter, std::regex::optimize);
        }

        ggml_backend_load_all();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = args.ngl;
        llama_model * model = llama_model_load_from_file(args.model.c_str(), mparams);
        if (!model) {
            throw std::runtime_error("failed to load model");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string prompt = read_file(args.prompt_file);
        const auto tokens = tokenize_prompt(vocab, prompt);
        const int source_pos = args.source_token_position >= 0
            ? args.source_token_position
            : (int) tokens.size() - 1;
        if (source_pos < 0 || source_pos >= (int) tokens.size()) {
            throw std::runtime_error("source token position outside tokenized prompt");
        }
        state.source_token_position = source_pos;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = args.n_ctx > 0 ? (uint32_t) args.n_ctx : (uint32_t) tokens.size() + 1;
        cparams.n_batch = 1;
        cparams.n_ubatch = 1;
        cparams.n_seq_max = 1;
        cparams.n_threads = args.threads;
        cparams.n_threads_batch = args.threads;
        cparams.no_perf = true;
        cparams.cb_eval = boundary_eval_cb;
        cparams.cb_eval_user_data = &state;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            llama_model_free(model);
            throw std::runtime_error("failed to create context");
        }

        llama_batch batch = llama_batch_init(1, 0, 1);
        for (int i = 0; i < (int) tokens.size(); ++i) {
            state.current_token_position = i;
            state.enabled = i == source_pos;
            batch.n_tokens = 1;
            batch.token[0] = tokens[i];
            batch.pos[0] = i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = i == source_pos ? 1 : 0;
            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                throw std::runtime_error("llama_decode failed at token " + std::to_string(i) + " rc=" + std::to_string(rc));
            }
        }

        llama_synchronize(ctx);
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        {
            std::ofstream topk(state.out_dir / "sampler-topk.json");
            if (!topk) {
                throw std::runtime_error("failed to open sampler-topk.json");
            }
            topk << "{"
                 << "\"case_id\":\"" << json_escape(args.case_id) << "\","
                 << "\"source_token_position\":" << source_pos << ","
                 << "\"prompt_token_count\":" << tokens.size() << ","
                 << "\"top_k\":[";
            if (logits) {
                const auto rows = topk_logits(logits, n_vocab, 8);
                for (size_t i = 0; i < rows.size(); ++i) {
                    if (i) {
                        topk << ",";
                    }
                    topk << "{\"token_id\":" << rows[i].first << ",\"logit\":" << rows[i].second << "}";
                }
            }
            topk << "],\"logits_present\":" << (logits ? "true" : "false") << "}\n";
        }
        {
            std::ofstream summary(state.out_dir / "capture-summary.json");
            if (!summary) {
                throw std::runtime_error("failed to open capture-summary.json");
            }
            summary << "{"
                    << "\"case_id\":\"" << json_escape(args.case_id) << "\","
                    << "\"prompt_token_count\":" << tokens.size() << ","
                    << "\"source_token_position\":" << source_pos << ","
                    << "\"captured_tensor_count\":" << state.captured << ","
                    << "\"batch_size_policy\":\"one_token_decode_steps\","
                    << "\"logits_present\":" << (logits ? "true" : "false")
                    << "}\n";
        }

        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);

        fprintf(stdout, "captured_tensor_count=%d\n", state.captured);
        fprintf(stdout, "prompt_token_count=%zu\n", tokens.size());
        fprintf(stdout, "source_token_position=%d\n", source_pos);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
'''


def iso_now() -> str:
  return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--target", "--host", dest="host", metavar="TARGET", default=DEFAULT_HOST)
  parser.add_argument(
      "--apply-target-patch",
      action="store_true",
      help="Apply the generated patch to the staged target source.",
  )
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=None,
      help="Output directory. Defaults to output/r0-boundary-capture-instrumentation-patch-<UTC>.",
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


def run(cmd: list[str], *, timeout_s: int, input_text: str | None = None) -> dict[str, Any]:
  try:
    result = subprocess.run(
        cmd,
        input=input_text,
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


def apply_patch_local(host: str, patch_text: str, *, check_only: bool,
                      timeout_s: int) -> dict[str, Any]:
  mode = "--check" if check_only else ""
  remote_script = f"set -u\ncd {shlex.quote(REMOTE_SOURCE_DIR)}\ngit apply {mode} -"
  return iq36_local.run_target(
      host,
      remote_script,
      timeout_s,
      input_text=patch_text,
  )


def parse_key_values(stdout: str) -> dict[str, str]:
  values: dict[str, str] = {}
  for line in stdout.splitlines():
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key:
      values[key.strip()] = value.strip()
  return values


def generate_patch() -> str:
  def add_file(path: str, text: str) -> str:
    lines = [
        "diff --git a/" + path + " b/" + path,
        "new file mode 100644",
        "index 0000000000..1111111111",
        "--- /dev/null",
        "+++ b/" + path,
        "@@ -0,0 +1," + str(len(text.splitlines())) + " @@",
    ]
    lines.extend("+" + line for line in text.splitlines())
    return "\n".join(lines) + "\n"

  cmake_hunk = """diff --git a/tools/CMakeLists.txt b/tools/CMakeLists.txt
index e4dc8bff51..b4f0f50c77 100644
--- a/tools/CMakeLists.txt
+++ b/tools/CMakeLists.txt
@@ -20,6 +20,7 @@ else()
     add_subdirectory(llama-bench)
     add_subdirectory(completion)
     add_subdirectory(perplexity)
+    add_subdirectory(qwen36-boundary-capture)
     add_subdirectory(quantize)
     if (LLAMA_BUILD_SERVER)
         add_subdirectory(ui)
"""
  return (
      cmake_hunk
      + add_file("tools/qwen36-boundary-capture/CMakeLists.txt", CAPTURE_CMAKE)
      + add_file("tools/qwen36-boundary-capture/qwen36-boundary-capture.cpp", CAPTURE_CPP)
  )


def target_source_status(host: str, raw_dir: Path, label: str) -> dict[str, Any]:
  script = "\n".join([
      "set -u",
      f"cd {shlex.quote(REMOTE_SOURCE_DIR)}",
      "printf 'source_rev_parse='; git rev-parse HEAD",
      "printf 'source_status_short_count='; git status --short | wc -l",
      "printf 'source_status_short='; git status --short | tr '\\n' ';'; printf '\\n'",
      "printf 'capture_cmake_present='; test -f tools/qwen36-boundary-capture/CMakeLists.txt && echo true || echo false",
      "printf 'capture_cpp_present='; test -f tools/qwen36-boundary-capture/qwen36-boundary-capture.cpp && echo true || echo false",
      "printf 'tools_cmake_registered='; grep -q 'qwen36-boundary-capture' tools/CMakeLists.txt && echo true || echo false",
  ])
  result = run_target(host, script, timeout_s=30)
  (raw_dir / f"{label}.stdout").write_text(result["stdout"], encoding="utf-8")
  (raw_dir / f"{label}.stderr").write_text(result["stderr"], encoding="utf-8")
  values = parse_key_values(result["stdout"])
  return {
      "capture_cmake_present": values.get("capture_cmake_present") == "true",
      "capture_cpp_present": values.get("capture_cpp_present") == "true",
      "returncode": result["returncode"],
      "source_rev_parse": values.get("source_rev_parse"),
      "source_status_short": values.get("source_status_short", ""),
      "source_status_short_count": int(values.get("source_status_short_count", "-1")),
      "timed_out": result["timed_out"],
      "tools_cmake_registered": values.get("tools_cmake_registered") == "true",
  }


def build_summary(payload: dict[str, Any]) -> str:
  patch = payload["patch_route"]
  after = payload["target_source_after"]
  lines = [
      "# R0 Boundary Capture Instrumentation Patch",
      "",
      f"- workstream: `{WORKSTREAM}`",
      f"- host: `{payload['host']}`",
      f"- source dir: `{REMOTE_SOURCE_DIR}`",
      f"- apply attempted: `{str(patch['apply_target_patch']).lower()}`",
      f"- patch check passed: `{str(patch['git_apply_check_passed']).lower()}`",
      f"- apply passed: `{str(patch['git_apply_passed']).lower()}`",
      f"- capture tool present: `{str(after['capture_cpp_present']).lower()}`",
      f"- route status: `{payload['route_status']}`",
      f"- R0 oracle gate closed: `{str(payload['r0_oracle_gate_closed']).lower()}`",
      "",
      "This artifact adds a dedicated llama.cpp boundary capture executable to",
      "the staged source when applied. It does not build the runtime, run the",
      "model, dump oracle tensors, create an oracle bundle, or close R0.",
      "",
  ]
  return "\n".join(lines)


def main() -> int:
  args = parse_args()
  created_at = iso_now()
  stamp = created_at.replace("-", "").replace(":", "")
  out_dir = args.out_dir or ROOT / f"output/r0-boundary-capture-instrumentation-patch-{stamp}"
  out_dir = out_dir.resolve()
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  instrumentation_map_path = latest(
      "r0-llama-instrumentation-map-*",
      "instrumentation-map.json",
  )
  if instrumentation_map_path is None:
    raise SystemExit("no latest llama instrumentation map artifact found under output/")
  instrumentation_map = load_json(instrumentation_map_path)
  correctness_path = instrumentation_map_path.parent / "correctness.json"
  instrumentation_correctness = load_json(correctness_path)
  patch_text = generate_patch()
  (out_dir / "boundary-capture-tool.patch").write_text(patch_text, encoding="utf-8")

  before = target_source_status(args.host, raw_dir, "target_source_before")
  apply_check = apply_patch_local(args.host, patch_text, check_only=True, timeout_s=30)
  (raw_dir / "git_apply_check.stdout").write_text(apply_check["stdout"], encoding="utf-8")
  (raw_dir / "git_apply_check.stderr").write_text(apply_check["stderr"], encoding="utf-8")

  apply_result: dict[str, Any] = {
      "returncode": None,
      "stdout": "",
      "stderr": "",
      "timed_out": False,
  }
  if args.apply_target_patch:
    apply_result = apply_patch_local(args.host, patch_text, check_only=False, timeout_s=30)
    (raw_dir / "git_apply.stdout").write_text(apply_result["stdout"], encoding="utf-8")
    (raw_dir / "git_apply.stderr").write_text(apply_result["stderr"], encoding="utf-8")
  after = target_source_status(args.host, raw_dir, "target_source_after")

  patch_route = {
      "apply_target_patch": args.apply_target_patch,
      "generated_patch_path": rel(out_dir / "boundary-capture-tool.patch"),
      "git_apply_check_passed": apply_check["returncode"] == 0,
      "git_apply_check_returncode": apply_check["returncode"],
      "git_apply_passed": (
          apply_result["returncode"] == 0 if args.apply_target_patch else False
      ),
      "git_apply_returncode": apply_result["returncode"],
      "patch_intent": [
          "register tools/qwen36-boundary-capture in tools/CMakeLists.txt",
          "add llama-qwen36-boundary-capture executable",
          "decode the source prompt one token at a time with batch size 1",
          "enable cb_eval tensor dumping only at source_token_position",
          "write tensor-dumps.jsonl, payload binaries, sampler-topk.json, and capture-summary.json",
      ],
  }
  route_status = (
      "target_source_patched_ready_for_build"
      if args.apply_target_patch
      and patch_route["git_apply_passed"]
      and after["capture_cpp_present"]
      and after["tools_cmake_registered"]
      else "patch_generated_ready_to_apply"
      if patch_route["git_apply_check_passed"]
      else "patch_check_failed"
  )
  payload = {
      "created_at": created_at,
      "evidence": {
          "instrumentation_map": rel(instrumentation_map_path.parent),
          "raw_dir": rel(raw_dir),
      },
      "host": args.host,
      "patch_route": patch_route,
      "r0_oracle_gate_closed": False,
      "route_status": route_status,
      "schema_version": SCHEMA_VERSION,
      "target_source_after": after,
      "target_source_before": before,
      "workstream": WORKSTREAM,
  }
  checks = [
      {
          "name": "latest_instrumentation_map_available",
          "pass": instrumentation_map.get("schema_version")
          == "intel-qwen36-r0-llama-instrumentation-map-v0"
          and instrumentation_correctness.get("required_checks_passed") is True
          and instrumentation_map.get("coverage", {}).get("mapped_boundary_type_count") == 17,
      },
      {
          "name": "target_source_matches_exact_commit_before_patch",
          "pass": before["source_rev_parse"] == EXPECTED_SHA,
          "target_source_before": before,
      },
      {
          "name": "target_source_clean_before_patch",
          "pass": before["source_status_short_count"] == 0,
          "target_source_before": before,
      },
      {
          "name": "generated_patch_applies_to_clean_source",
          "pass": patch_route["git_apply_check_passed"] is True,
          "git_apply_check_returncode": patch_route["git_apply_check_returncode"],
      },
      {
          "name": "target_patch_applied_when_requested",
          "pass": (
              patch_route["git_apply_passed"] is True
              and after["capture_cpp_present"] is True
              and after["capture_cmake_present"] is True
              and after["tools_cmake_registered"] is True
              if args.apply_target_patch
              else True
          ),
          "target_source_after": after,
      },
      {
          "name": "patch_route_does_not_close_oracle_gate",
          "pass": payload["r0_oracle_gate_closed"] is False,
      },
  ]
  correctness = {
      "checks": checks,
      "gate": "r0_boundary_capture_instrumentation_patch",
      "required_checks_passed": all(check["pass"] for check in checks),
      "schema_version": SCHEMA_VERSION,
      "workstream": WORKSTREAM,
  }
  write_json(out_dir / "manifest.json", {
      "captured_at": created_at,
      "schema_version": SCHEMA_VERSION,
      "tool": "tools/intel-qwen36-r0-boundary-capture-instrumentation-patch.py",
      "workstream": WORKSTREAM,
  })
  write_json(out_dir / "patch-route.json", payload)
  write_json(out_dir / "correctness.json", correctness)
  with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
    for metric, value in (
        ("git_apply_check_passed", patch_route["git_apply_check_passed"]),
        ("git_apply_passed", patch_route["git_apply_passed"]),
        ("apply_target_patch", args.apply_target_patch),
        ("capture_cpp_present", after["capture_cpp_present"]),
        ("r0_oracle_gate_closed", False),
    ):
      fh.write(json.dumps({
          "metric": metric,
          "phase": "r0_boundary_capture_instrumentation_patch",
          "value": value,
      }, sort_keys=True) + "\n")
  (out_dir / "summary.md").write_text(build_summary(payload), encoding="utf-8")
  print(f"boundary capture instrumentation patch output: {out_dir}")
  return 0 if correctness["required_checks_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
