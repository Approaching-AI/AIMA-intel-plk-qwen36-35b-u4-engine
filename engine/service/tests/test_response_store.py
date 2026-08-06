import unittest

from iq36_server.response_store import ResponseStore


class Clock:
  def __init__(self):
    self.value = 0.0

  def __call__(self):
    return self.value


class ResponseStoreTest(unittest.TestCase):
  @staticmethod
  def _response(response_id):
    return {"id": response_id, "object": "response", "output": []}

  def test_lru_ttl_and_defensive_copy(self):
    clock = Clock()
    store = ResponseStore(2, 10.0, clock=clock)
    messages = [{"role": "user", "content": "one"}]
    assistant = {"role": "assistant", "content": "answer"}
    self.assertTrue(store.put(
        "resp_1", messages, assistant, self._response("resp_1")))
    messages[0]["content"] = "mutated"
    first = store.get("resp_1")
    self.assertEqual(first.messages[0]["content"], "one")
    first.assistant["content"] = "also mutated"
    self.assertEqual(store.get("resp_1").assistant["content"], "answer")

    store.put("resp_2", [], assistant, self._response("resp_2"))
    store.get("resp_1")
    store.put("resp_3", [], assistant, self._response("resp_3"))
    self.assertIsNone(store.get("resp_2"))
    self.assertIsNotNone(store.get("resp_1"))

    clock.value = 10.0
    self.assertIsNone(store.get("resp_1"))
    self.assertIsNone(store.get("resp_3"))

  def test_disabled_store(self):
    store = ResponseStore(0, 0.0)
    self.assertFalse(store.put(
        "resp_1", [{"role": "user", "content": "x"}],
        {"role": "assistant", "content": "y"},
        self._response("resp_1")))
    self.assertIsNone(store.get("resp_1"))

  def test_byte_bound_delete_and_stats(self):
    assistant = {"role": "assistant", "content": "answer"}
    probe = ResponseStore(3, 10.0, max_bytes=10000)
    probe.put("resp_1", [], assistant, self._response("resp_1"))
    one_entry_bytes = probe.stats()["bytes"]

    store = ResponseStore(3, 10.0, max_bytes=one_entry_bytes * 2 - 1)
    self.assertTrue(store.put(
        "resp_1", [], assistant, self._response("resp_1")))
    self.assertTrue(store.put(
        "resp_2", [], assistant, self._response("resp_2")))
    self.assertIsNone(store.get("resp_1"))
    self.assertEqual(store.stats()["entries"], 1)
    self.assertTrue(store.delete("resp_2"))
    self.assertFalse(store.delete("resp_2"))
    self.assertEqual(store.stats(), {"entries": 0, "bytes": 0})

    too_small = ResponseStore(1, 10.0, max_bytes=1)
    self.assertFalse(too_small.put(
        "resp_1", [], assistant, self._response("resp_1")))


if __name__ == "__main__":
  unittest.main()
