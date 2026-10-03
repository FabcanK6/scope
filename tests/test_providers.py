"""Bring-your-own-key providers: request formats, schema conversion, errors and fallbacks (no network)."""
import io
import json
import unittest
import urllib.error
from unittest import mock

from scope.engine import EXTRACT_SCHEMA, LLMParser
from scope.llm import GeminiClient, LLMError
from scope.protocol import PROTOCOL_SCHEMA
from scope.providers import (
    AnthropicClient,
    CompatibleClient,
    OpenAIClient,
    make_client,
    to_json_schema,
)

NOTE = "IMV Site 5, 03/02/2026. Enrollment is 2 behind target."
ANSWER = {"findings": [{"issue": "ENROLLMENT_LAG", "status": "active", "severity": "major",
                        "evidence": "Enrollment is 2 behind target.", "explanation": "", "repeat": False,
                        "subjects_affected": None, "site_wide": False, "escalation_evidence": ""}],
          "actions": [], "summary": "Enrollment is behind.", "visit": {"visit_type": "IMV", "visit_date": "03/02/2026",
                                                                       "site": "Site 5", "monitor": None, "pi": None,
                                                                       "screened": None, "enrolled": None}}


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(json.dumps(body).encode()))


class Recorder:
    """Stands in for urllib.request.urlopen: records requests, plays back responses or errors."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode()) if req.data else None
        self.requests.append({"url": req.full_url, "headers": {k.lower(): v for k, v in req.header_items()},
                              "body": body})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Resp(json.dumps(reply).encode())


def openai_reply(content):
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}]}


def anthropic_reply(content):
    return {"content": [{"type": "text", "text": json.dumps(content)}], "stop_reason": "end_turn"}


class TestSchema(unittest.TestCase):
    def test_strict_json_schema(self):
        js = to_json_schema(EXTRACT_SCHEMA)
        self.assertEqual(js["type"], "object")
        self.assertFalse(js["additionalProperties"])
        self.assertEqual(js["required"], ["findings", "actions", "summary", "visit"])  # follows propertyOrdering
        visit = js["properties"]["visit"]
        self.assertEqual(visit["properties"]["site"]["type"], ["string", "null"])
        self.assertEqual(set(visit["required"]), set(visit["properties"]))  # strict: all present, nullable instead
        item = js["properties"]["findings"]["items"]
        self.assertEqual(item["properties"]["subjects_affected"]["type"], ["integer", "null"])
        self.assertIn("ENROLLMENT_LAG", item["properties"]["issue"]["enum"])
        self.assertNotIn("propertyOrdering", json.dumps(js))
        self.assertNotIn("nullable", json.dumps(js))
        self.assertEqual(to_json_schema(PROTOCOL_SCHEMA)["properties"]["rules"]["type"], "array")


class TestOpenAI(unittest.TestCase):
    def test_request_and_engine(self):
        rec = Recorder(openai_reply(ANSWER))
        with mock.patch("urllib.request.urlopen", rec):
            client = OpenAIClient("sk-test", model="gpt-test")
            result = LLMParser(client).analyze(NOTE)
        req = rec.requests[0]
        self.assertEqual(req["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(req["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(req["body"]["model"], "gpt-test")
        self.assertEqual(req["body"]["messages"][0]["role"], "system")
        fmt = req["body"]["response_format"]
        self.assertEqual((fmt["type"], fmt["json_schema"]["strict"]), ("json_schema", True))
        self.assertEqual(result["risk"]["level"], "medium")
        self.assertEqual(result["model"], "gpt-test")
        self.assertEqual(client.label, "OpenAI gpt-test")

    def test_compatible_service_without_schema_support(self):
        rec = Recorder(http_error(400, {"error": {"message": "response_format json_schema is not supported"}}),
                       openai_reply(ANSWER))
        with mock.patch("urllib.request.urlopen", rec):
            client = CompatibleClient("", model="mistral-test", base_url="https://api.example.com/v1/")
            LLMParser(client).analyze(NOTE)
        self.assertEqual(rec.requests[1]["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(rec.requests[1]["body"]["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", rec.requests[1]["body"]["messages"][0]["content"])
        self.assertNotIn("authorization", rec.requests[1]["headers"])  # local server, no key

    def test_errors(self):
        client = OpenAIClient("sk-bad", model="gpt-test")
        client._sleep = lambda s: None
        with mock.patch("urllib.request.urlopen", Recorder(http_error(401, {"error": {"message": "bad key"}}))):
            with self.assertRaisesRegex(LLMError, "key was rejected"):
                client.generate("s", "p")
        no_credit = http_error(429, {"error": {"code": "insufficient_quota", "message": "You exceeded your quota"}})
        with mock.patch("urllib.request.urlopen", Recorder(no_credit)):
            with self.assertRaisesRegex(LLMError, "no credit"):
                client.generate("s", "p")
        busy = [http_error(503, {"error": {"message": "overloaded"}}) for _ in range(3)]
        with mock.patch("urllib.request.urlopen", Recorder(*busy)):
            with self.assertRaisesRegex(LLMError, "OpenAI is busy"):
                client.generate("s", "p")

    def test_model_list(self):
        models = {"data": [{"id": "text-embedding-3", "created": 3}, {"id": "gpt-old", "created": 1},
                           {"id": "gpt-new", "created": 5}, {"id": "whisper-1", "created": 4}]}
        with mock.patch("urllib.request.urlopen", Recorder(models, openai_reply({"ok": 1}))):
            client = OpenAIClient("sk-test")
            self.assertEqual(client.list_models(), ["gpt-new", "gpt-old"])
        with mock.patch("urllib.request.urlopen", Recorder(models, openai_reply({"ok": 1}))) as rec:
            client = OpenAIClient("sk-test")
            client.generate("s", "p")  # no model chosen: newest chat model
            self.assertEqual(rec.requests[1]["body"]["model"], "gpt-new")


class TestAnthropic(unittest.TestCase):
    def test_request_and_engine(self):
        rec = Recorder(anthropic_reply(ANSWER))
        with mock.patch("urllib.request.urlopen", rec):
            client = AnthropicClient("ak-test", model="claude-test")
            result = LLMParser(client).analyze(NOTE)
        req = rec.requests[0]
        self.assertEqual(req["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(req["headers"]["x-api-key"], "ak-test")
        self.assertEqual(req["headers"]["anthropic-version"], "2023-06-01")
        self.assertIn("system", req["body"])
        self.assertEqual(req["body"]["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(result["risk"]["level"], "medium")

    def test_older_model_falls_back_to_instructions(self):
        rec = Recorder(http_error(400, {"error": {"message": "output_config: structured outputs not supported"}}),
                       anthropic_reply(ANSWER))
        with mock.patch("urllib.request.urlopen", rec):
            AnthropicClient("ak-test", model="claude-old").generate_json("s", "p", EXTRACT_SCHEMA)
        self.assertNotIn("output_config", rec.requests[1]["body"])
        self.assertIn("JSON Schema", rec.requests[1]["body"]["system"])


class TestFactory(unittest.TestCase):
    def test_make_client(self):
        self.assertIsInstance(make_client("Google Gemini", "g"), GeminiClient)
        self.assertIsInstance(make_client("OpenAI", "o"), OpenAIClient)
        self.assertIsInstance(make_client("Anthropic Claude", "a"), AnthropicClient)
        self.assertIsInstance(make_client("Other (OpenAI-compatible)", "", base_url="http://localhost:11434/v1"),
                              CompatibleClient)
        with self.assertRaises(LLMError):
            make_client("OpenAI", "")
        with self.assertRaises(LLMError):
            make_client("Other (OpenAI-compatible)", "k")  # no base URL


if __name__ == "__main__":
    unittest.main()
