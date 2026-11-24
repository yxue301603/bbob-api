import io
import json
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from selection_bench import api


def completion(content, **message_fields):
    message = {"content": content, **message_fields}
    return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())


class SelectionApiTests(unittest.TestCase):
    def setUp(self):
        env = {"DEEPSEEK_BASE_URL": "https://model.example/v1/", "DEEPSEEK_MODEL": "test-id",
               "DEEPSEEK_API_KEY": "local-test-marker"}
        self.enterContext(patch.dict(os.environ, env))
        self.client = self.enterContext(patch.object(api, "build_opener")).return_value
        self.enterContext(patch.object(api.time, "sleep"))

    def test_settings_and_recovery(self):
        self.client.open.side_effect = [TimeoutError(), completion('{"pick":true}'), completion('{"pick":7}')]
        self.assertEqual(api.choose("trial", 0, "candidates"), 7)
        req = self.client.open.call_args.args[0]
        self.assertEqual(req.full_url, "https://model.example/v1/chat/completions")
        body = json.loads(req.data)
        self.assertEqual(body["messages"], [{"role": "system", "content": api.SYSTEM},
                                            {"role": "user", "content": "candidates"}])
        self.assertEqual((body["model"], body["temperature"], body["thinking"], body["response_format"]),
                         ("test-id", 0, {"type": "disabled"}, {"type": "json_object"}))

    def test_exhausted_invalid_responses_raise_without_logging_contents(self):
        self.client.open.side_effect = [completion(content) for content in
                                        ('{"pick":8}', '{"pick":-1}', '{"pick":2.5}', 'local-test-marker')]
        with self.assertRaisesRegex(RuntimeError, "after 4 attempts") as error:
            api.choose("trial", 1, "candidates")
        self.assertNotIn("local-test-marker", str(error.exception))

    def test_redirect_and_reasoning_are_terminal(self):
        self.assertIsNone(api.NoRedirect().redirect_request(None, None, 302, None, None, "https://other.example"))
        self.client.open.side_effect = [HTTPError("https://model.example/v1", 302, "secret", {}, io.BytesIO())]
        with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
            api.choose("trial", 1, "candidates")
        self.client.open.side_effect = [completion('{"pick":0}', reasoning_content="unexpected")]
        with self.assertRaisesRegex(RuntimeError, "reasoning"):
            api.choose("trial", 1, "candidates")


if __name__ == "__main__":
    unittest.main()
