import unittest

from iq36_server.backend import select_bucket, select_profile
from iq36_server.config import ServerConfig
from iq36_server.protocol import (
    APIError, decode_json_object, prepare_chat_completion, prepare_completion,
    prepare_response)
from iq36_server.tokenizer import SimpleTestTokenizer


class ProtocolTest(unittest.TestCase):
  def setUp(self):
    self.config = ServerConfig(
        backend="mock", max_context_length=4096, max_new_tokens=64,
        preload_bucket=0)
    self.tokenizer = SimpleTestTokenizer()

  def test_arbitrary_lengths_route_up_without_padding(self):
    self.assertEqual(select_bucket(1), 2048)
    self.assertEqual(select_bucket(2048), 2048)
    self.assertEqual(select_bucket(2049), 4096)
    self.assertEqual(select_bucket(8179), 8192)
    self.assertEqual(select_bucket(8193), 16384)

  def test_total_context_limit_has_standard_error(self):
    payload = {
        "model": self.config.model_id,
        "prompt": "x" * 4050,
        "max_tokens": 64,
    }
    with self.assertRaises(APIError) as caught:
      prepare_completion(payload, self.config, self.tokenizer)
    self.assertEqual(caught.exception.code, "context_length_exceeded")
    self.assertIn("never silently truncates", caught.exception.message)

  def test_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
    for body in (b'{"model":"a","model":"b"}', b'{"value":NaN}'):
      with self.subTest(body=body):
        with self.assertRaises(APIError):
          decode_json_object(body)

  def test_prefix_cache_can_be_bypassed_per_request(self):
    request = prepare_completion({
        "model": self.config.model_id, "prompt": "x",
        "temperature": 0, "max_tokens": 8, "prefix_cache": False,
    }, self.config, self.tokenizer)
    self.assertFalse(request.prefix_cache)
    with self.assertRaises(APIError):
      prepare_completion({
          "model": self.config.model_id, "prompt": "x",
          "prefix_cache": "false",
      }, self.config, self.tokenizer)

  def test_completion_echo_rejects_unavailable_prompt_logprobs(self):
    with self.assertRaises(APIError) as caught:
      prepare_completion({
          "model": self.config.model_id, "prompt": "x",
          "echo": True, "logprobs": 1,
      }, self.config, self.tokenizer)
    self.assertEqual(
        caught.exception.code, "unsupported_parameter_combination")

  def test_tool_template_and_named_choice(self):
    request = prepare_chat_completion({
        "model": self.config.model_id,
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{
            "type": "function", "function": {
                "name": "weather", "description": "weather",
                "parameters": {"type": "object"},
            }}],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
        "temperature": 0, "max_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertIn("weather", request.prompt)
    self.assertEqual(len(request.tools), 1)
    self.assertTrue(request.params.greedy)

  def test_message_history_converts_tool_arguments(self):
    request = prepare_chat_completion({
        "model": self.config.model_id,
        "messages": [
            {"role": "user", "content": "weather"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "weather", "arguments": "{\"city\":\"x\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "summarize"},
        ],
        "temperature": 0, "max_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertIn("sunny", request.prompt)

  def test_profile_capability_routing(self):
    greedy = prepare_completion({
        "model": self.config.model_id, "prompt": "x",
        "temperature": 0, "max_tokens": 8,
    }, self.config, self.tokenizer)
    sampled = prepare_completion({
        "model": self.config.model_id, "prompt": "x",
        "temperature": 1, "max_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertEqual(select_profile(16384, greedy), "long_compact")
    self.assertEqual(select_profile(16384, sampled), "long_full")
    self.assertEqual(select_profile(8192, sampled), "short_full")

  def test_responses_flat_function_tool(self):
    request = prepare_response({
        "model": self.config.model_id,
        "input": "hello",
        "tools": [{"type": "function", "name": "f",
                   "description": "d", "parameters": {"type": "object"}}],
        "temperature": 0, "max_output_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertEqual(request.tools[0]["function"]["name"], "f")

  def test_responses_stateless_function_call_round_trip(self):
    request = prepare_response({
        "model": self.config.model_id,
        "input": [
            {"role": "user", "content": "weather"},
            {"type": "function_call", "call_id": "call_1",
             "name": "weather", "arguments": '{"city":"Shanghai"}'},
            {"type": "function_call_output", "call_id": "call_1",
             "output": '{"temperature":25}'},
        ],
        "tools": [{"type": "function", "name": "weather",
                   "parameters": {"type": "object"}}],
        "temperature": 0, "max_output_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertIn("temperature", request.prompt)

  def test_response_format_shapes_and_schema_validation(self):
    schema = {
        "name": "answer", "strict": True,
        "schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        },
    }
    responses = prepare_response({
        "model": self.config.model_id, "input": "hello",
        "text": {"format": {"type": "json_schema", **schema}},
        "temperature": 0, "max_output_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertEqual(responses.response_format["schema"]["type"], "object")
    chat = prepare_chat_completion({
        "model": self.config.model_id,
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_schema", "json_schema": schema},
        "temperature": 0, "max_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertEqual(
        chat.response_format["json_schema"]["schema"]["type"], "object")
    with self.assertRaises(APIError):
      prepare_response({
          "model": self.config.model_id, "input": "hello",
          "text": {"format": {
              "type": "json_schema", "name": "bad",
              "schema": {"type": "not-a-real-json-type"},
          }},
      }, self.config, self.tokenizer)
    with self.assertRaises(APIError) as caught:
      prepare_response({
          "model": self.config.model_id, "input": "hello",
          "text": {"format": {
              "type": "json_schema", "name": "unsafe_regex",
              "schema": {"type": "string", "pattern": "(a+)+$"},
          }},
      }, self.config, self.tokenizer)
    self.assertEqual(caught.exception.code, "unsupported_schema_keyword")

  def test_responses_rejects_auto_truncation(self):
    with self.assertRaises(APIError) as caught:
      prepare_response({
          "model": self.config.model_id, "input": "hello",
          "truncation": "auto",
      }, self.config, self.tokenizer)
    self.assertEqual(caught.exception.code, "unsupported_truncation")

  def test_batch_and_unknown_model_rejected(self):
    with self.assertRaises(APIError):
      prepare_chat_completion({
          "model": self.config.model_id,
          "messages": [{"role": "user", "content": "x"}], "n": 2,
      }, self.config, self.tokenizer)
    with self.assertRaises(APIError) as caught:
      prepare_completion({"model": "missing", "prompt": "x"},
                         self.config, self.tokenizer)
    self.assertEqual(caught.exception.status, 404)

  def test_unsupported_semantic_fields_are_not_silently_ignored(self):
    with self.assertRaises(APIError) as caught:
      prepare_chat_completion({
          "model": self.config.model_id,
          "messages": [{"role": "user", "content": "x"}],
          "logit_bias": {"1": 10},
      }, self.config, self.tokenizer)
    self.assertEqual(caught.exception.code, "unsupported_parameter")
    self.assertEqual(caught.exception.param, "logit_bias")

  def test_nested_semantic_fields_and_invalid_user_are_rejected(self):
    base = {
        "model": self.config.model_id,
        "messages": [{"role": "user", "content": "x"}],
    }
    for payload in (
        {**base, "user": 123},
        {**base, "chat_template_kwargs": {"unknown": True}},
        {**base, "messages": [{
            "role": "user", "content": "x", "audio": {}}]},
        {**base, "tools": [{"type": "function", "function": {
            "name": "f", "parameters": {"type": "object"},
            "server_only_field": True,
        }}]},
    ):
      with self.subTest(payload=payload):
        with self.assertRaises(APIError):
          prepare_chat_completion(payload, self.config, self.tokenizer)

    with self.assertRaises(APIError):
      prepare_response({
          "model": self.config.model_id, "input": "x",
          "text": {"format": {"type": "text"}, "verbosity": "low"},
      }, self.config, self.tokenizer)

  def test_responses_top_logprobs_requests_full_logits(self):
    request = prepare_response({
        "model": self.config.model_id, "input": "hello",
        "top_logprobs": 3, "max_output_tokens": 8,
    }, self.config, self.tokenizer)
    self.assertTrue(request.params.logprobs)
    self.assertEqual(request.params.top_logprobs, 3)


if __name__ == "__main__":
  unittest.main()
