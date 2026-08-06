import json
import unittest

from iq36_server.tool_calls import parse_assistant_text


class ToolCallParserTest(unittest.TestCase):
  def test_qwen_xml_call(self):
    text = """<think>private</think>
Checking now.
<tool_call>
<function=get_weather>
<parameter=city>
Shanghai
</parameter>
<parameter=days>
3
</parameter>
</function>
</tool_call>"""
    parsed = parse_assistant_text(text, "chatcmpl-test")
    self.assertEqual(parsed.reasoning, "private")
    self.assertEqual(parsed.content, "Checking now.")
    self.assertEqual(len(parsed.tool_calls), 1)
    self.assertEqual(parsed.tool_calls[0].name, "get_weather")
    self.assertEqual(
        json.loads(parsed.tool_calls[0].arguments),
        {"city": "Shanghai", "days": 3})
    self.assertTrue(parsed.tool_calls[0].id.startswith("call_"))

  def test_json_call_and_multiple_calls(self):
    text = (
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>\n'
        '<tool_call>{"function":{"name":"b","arguments":"{\\"y\\":2}"}}'
        '</tool_call>')
    parsed = parse_assistant_text(text, "resp_test")
    self.assertEqual([call.name for call in parsed.tool_calls], ["a", "b"])
    self.assertNotEqual(parsed.tool_calls[0].id, parsed.tool_calls[1].id)

  def test_malformed_call_remains_content(self):
    text = "before <tool_call><function=bad name></function></tool_call> after"
    parsed = parse_assistant_text(text, "x")
    self.assertFalse(parsed.tool_calls)
    self.assertIn("<tool_call>", parsed.content)

  def test_xml_call_rejects_unparsed_parameter_gaps(self):
    text = (
        "<tool_call><function=f>garbage"
        "<parameter=x>1</parameter>tail</function></tool_call>")
    parsed = parse_assistant_text(text, "x")
    self.assertFalse(parsed.tool_calls)
    self.assertEqual(parsed.content, text)


if __name__ == "__main__":
  unittest.main()
