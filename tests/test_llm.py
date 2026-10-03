"""LLM assistant tests with a fake Gemini transport (no network, no key needed)."""
import json
import unittest

from scope.llm import BUSY_MESSAGE, GeminiClient, LLMError, ModelBusy, ModelNotFound, draft_followup, verify_quote

NOTE = ("Visit Type: Directed/For-Cause Monitoring Visit\nDate: September 18, 2026\n"
        "The subject was hospitalized on August 30, 2026, but the site did not notify the Sponsor until "
        "September 12, 2026, violating the mandatory 24-hour protocol reporting window. "
        "No temperature excursions were noted on the digital data logger.")


def fake_response(payload) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class FakeClient(GeminiClient):
    def __init__(self, replies, models=("gemini-9-flash",)):
        super().__init__("test-key")
        self.replies = list(replies)
        self.models = list(models)
        self.calls = []
        self.waits = []
        self._sleep = self.waits.append  # no real waiting in tests

    def _request(self, method, path, body=None):
        self.calls.append((method, path))
        if method == "GET":
            return {"models": [{"name": f"models/{m}", "supportedGenerationMethods": ["generateContent"]}
                               for m in self.models]}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class TestLLM(unittest.TestCase):
    def test_verify_quote(self):
        self.assertTrue(verify_quote("violating the mandatory 24-hour protocol reporting window", NOTE))
        self.assertTrue(verify_quote("NO TEMPERATURE excursions were noted on the digital data logger.", NOTE))
        self.assertFalse(verify_quote("The PI refused to sign the delegation log.", NOTE))

    def test_engine_verifies_and_scores(self):
        from scope.engine import LLMParser

        answer = {
            "visit": {"visit_type": "Directed/For-Cause Monitoring Visit", "visit_date": "September 18, 2026",
                      "site": None, "monitor": "Somebody Else", "pi": None, "screened": None, "enrolled": None},
            "findings": [
                {"issue": "SAE_REPORTING", "status": "active", "severity": "critical",
                 "evidence": "the site did not notify the Sponsor until September 12, 2026", "explanation": "late"},
                {"issue": "TEMP_EXCURSION", "status": "no_issue", "severity": "minor",
                 "evidence": "No temperature excursions were noted on the digital data logger.", "explanation": ""},
                {"issue": "CONSENT", "status": "active", "severity": "critical",
                 "evidence": "Consent was signed after dosing.", "explanation": "invented"},
            ],
            "actions": [{"owner": "PI", "action": "invent an action", "due": None}],
            "summary": "Late SAE.",
        }
        parser = LLMParser(FakeClient([fake_response(answer)]))
        rec = parser.analyze(NOTE)
        self.assertEqual(rec["risk"]["level"], "high")
        self.assertEqual([i["code"] for i in rec["issues"]], ["SAE_REPORTING"])  # invented consent ignored
        self.assertEqual(rec["issues"][0]["severity"], "critical")
        self.assertEqual(rec["visit"]["visit_date"]["iso"], "2026-09-18")
        self.assertIsNone(rec["visit"]["monitor"])  # name not in the note -> dropped
        self.assertEqual(rec["actions"], [])  # action not in the note -> dropped
        self.assertTrue(rec["review"]["needed"])
        self.assertEqual(rec["model"], "gemini-9-flash")

    def test_model_fallback_and_errors(self):
        from scope.engine import LLMParser

        empty = {"visit": {}, "findings": [], "actions": [], "summary": "Nothing to report."}
        client = FakeClient([ModelNotFound("gone"), fake_response(empty)], models=("gemini-9-flash", "gemini-8-flash"))
        rec = LLMParser(client).analyze(NOTE)
        self.assertEqual(rec["risk"]["level"], "low")
        self.assertEqual(client.model, "gemini-8-flash")
        with self.assertRaises(LLMError):
            LLMParser(FakeClient([LLMError("The free Gemini quota is used up for now.")])).analyze(NOTE)

    def test_busy_retry_then_next_model(self):
        from scope.engine import LLMParser

        empty = {"visit": {}, "findings": [], "actions": [], "summary": "Nothing to report."}
        # first model busy 3 times (1 try + 2 retries), second model answers
        client = FakeClient([ModelBusy("HTTP 503")] * 3 + [fake_response(empty)],
                            models=("gemini-9-flash", "gemini-8-flash"))
        LLMParser(client).analyze(NOTE)
        self.assertEqual(client.model, "gemini-8-flash")
        self.assertEqual(client.waits, [2.0, 5.0])
        # busy once, then fine on the same model
        client = FakeClient([ModelBusy("HTTP 503"), fake_response(empty)])
        LLMParser(client).analyze(NOTE)
        self.assertEqual(client.model, "gemini-9-flash")
        # everything busy -> one friendly message, no raw JSON
        client = FakeClient([ModelBusy("HTTP 503")] * 20, models=("gemini-9-flash", "gemini-8-flash"))
        with self.assertRaises(LLMError) as ctx:
            LLMParser(client).analyze(NOTE)
        self.assertEqual(str(ctx.exception), BUSY_MESSAGE)

    def test_pinned_model_falls_back_when_busy(self):
        client = FakeClient([ModelBusy("HTTP 503")] * 3 + [fake_response("ok")], models=("gemini-8-flash",))
        client.preferred = client.model = "gemini-9-flash"
        self.assertEqual(client.candidates()[:2], ["gemini-9-flash", "gemini-8-flash"])
        self.assertEqual(client.generate("s", "p"), "ok")
        self.assertEqual(client.model, "gemini-8-flash")

    def test_http_error_mapping(self):
        import io
        import urllib.error
        from unittest import mock

        def raise_http(code, body):
            err = urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body.encode()))
            return mock.patch("urllib.request.urlopen", side_effect=err)

        client = GeminiClient("k", model="gemini-9-flash")
        body = '{"error": {"code": 503, "message": "This model is currently experiencing high demand."}}'
        with raise_http(503, body), self.assertRaises(ModelBusy):
            client._request("POST", "models/x:generateContent", {})
        with raise_http(400, '{"error": {"message": "Bad field."}}'), self.assertRaises(LLMError) as ctx:
            client._request("POST", "models/x:generateContent", {})
        self.assertEqual(str(ctx.exception), "Gemini API error 400: Bad field.")

    def test_messy_visit_values(self):
        from scope.engine import LLMParser

        note = ("Monitor: Hannah Price, RN\nVisit Type: Interim Monitoring Visit (IMV)\nDate: October 14, 2025\n"
                "Site 231: 5 screened, 2 randomized. All consents verified.")
        answer = {"visit": {"visit_type": "Interim Monitoring Visit (IMV) - from the header",
                            "visit_date": None, "screened": "5", "enrolled": "2"},
                  "findings": [], "actions": [], "summary": "All consents verified."}
        rec = LLMParser(FakeClient([fake_response(answer)])).analyze(note)
        self.assertEqual(rec["visit"]["visit_type"]["code"], "IMV")  # clean piece of a messy value
        self.assertEqual(rec["visit"]["visit_date"]["iso"], "2025-10-14")  # from the labelled header line
        self.assertEqual((rec["visit"]["screened"], rec["visit"]["enrolled"]), (5, 2))  # not the "2" in "231"
        self.assertFalse(rec["review"]["needed"])
        self.assertEqual(rec["llm_output"]["visit"]["screened"], "5")

    def test_prompt_examples_are_not_test_notes(self):
        from scope.data.handwritten import load_handwritten, load_realistic
        from scope.engine import SYSTEM

        for r in load_handwritten() + load_realistic():
            first = " ".join(r["text"].split())[:80]
            self.assertNotIn(first, " ".join(SYSTEM.split()), r["id"])

    def test_garbled_answer_is_retried_never_scored(self):
        from scope.data.handwritten import load_handwritten
        from scope.engine import UNREADABLE, LLMParser

        note = next(r["text"] for r in load_handwritten() if r["id"] == "hw-04")  # the missed-SAE note
        # what gemini-3.6-flash actually returned in the live app
        garbled = {"visit": {"visit_type": "IMV\u6b63\u9762 Monitoring Visit (IMV) or IMV? Note says 'IMV', so IMV. "
                                           "Wait, let's check: 'IMV'"},
                   "findings": [], "actions": [], "summary": ""}
        good = {"visit": {"visit_type": "IMV"}, "summary": "SAE never reported.", "actions": [],
                "findings": [{"issue": "SAE_REPORTING", "status": "active", "severity": "critical",
                              "evidence": "never made it into EDC as an SAE", "explanation": "unreported SAE"}]}
        client = FakeClient([fake_response(garbled), fake_response(garbled), fake_response(good)])
        rec = LLMParser(client).analyze(note)
        self.assertEqual(rec["risk"]["level"], "high")
        self.assertEqual(len([c for c in client.calls if c[0] == "POST"]), 3)
        empty = {"visit": {}, "findings": [], "actions": [], "summary": ""}
        with self.assertRaises(LLMError) as ctx:
            LLMParser(FakeClient([fake_response(garbled), fake_response(empty), fake_response(garbled)])).analyze(note)
        self.assertEqual(str(ctx.exception), UNREADABLE)

    def test_safety_tripwire(self):
        from scope.data.handwritten import load_handwritten
        from scope.engine import LLMParser

        note = next(r["text"] for r in load_handwritten() if r["id"] == "hw-04")
        missed = {"visit": {}, "actions": [], "summary": "PI has not signed labs.",
                  "findings": [{"issue": "PI_OVERSIGHT", "status": "active", "severity": "major",
                                "evidence": "He has not signed any lab reports since November either.",
                                "explanation": "unsigned labs"}]}
        rec = LLMParser(FakeClient([fake_response(missed)])).analyze(note)
        self.assertEqual(len(rec["alerts"]), 1)
        self.assertIn("serious adverse event", rec["alerts"][0])
        self.assertTrue(rec["review"]["needed"])
        # a note that only says there were no SAEs, read correctly, raises no alert
        fine = {"visit": {}, "actions": [], "summary": "Nothing open.",
                "findings": [{"issue": "TEMP_EXCURSION", "status": "no_issue", "severity": "minor",
                              "evidence": "No temperature excursions were noted on the digital data logger.",
                              "explanation": ""},
                             {"issue": "SAE_REPORTING", "status": "active", "severity": "critical",
                              "evidence": "violating the mandatory 24-hour protocol reporting window",
                              "explanation": "late"}]}
        self.assertEqual(LLMParser(FakeClient([fake_response(fine)])).analyze(NOTE)["alerts"], [])

    def test_no_low_temperature_sent(self):
        client = FakeClient([fake_response("ok")])
        sent = []
        original = client._request
        client._request = lambda m, p, b=None: (sent.append(b), original(m, p, b))[1]
        client.generate("s", "p")
        self.assertNotIn("temperature", sent[-1]["generationConfig"])

    def test_letter(self):
        from scope.predict import RuleBasedParser

        rec = RuleBasedParser().analyze(NOTE)
        client = FakeClient([fake_response("Dear Dr. [Name],\n\nThank you for hosting the visit.")])
        self.assertTrue(draft_followup(client, rec).startswith("Dear"))

    def test_teacher_check(self):
        import importlib.util
        import random

        from scope.data.generate import sample_note

        spec = importlib.util.spec_from_file_location("teacher", "scripts/generate_teacher_notes.py")
        teacher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(teacher)
        row = sample_note(random.Random(3), style="formal")
        text = row["text"]
        marked, last = "", 0
        for sp in sorted(row["spans"], key=lambda x: x["char_start"]):
            marked += text[last:sp["char_start"]] + f"[[{sp['text']}|{sp['label']}]]"
            last = sp["char_end"]
        marked += text[last:]
        first_sentence = text.split("\n")[0]
        findings = [{"issue": c, "status": "active", "severity": s, "evidence": first_sentence}
                    for c, s in row["severities"].items()]
        good = teacher.check(row, {"note": marked, "findings": findings})
        self.assertIsNotNone(good)
        self.assertEqual(good["tags"], row["tags"])
        self.assertEqual(good["actions"], row["actions"])
        # a re-graded finding is rejected
        if findings:
            findings[0] = {**findings[0], "severity": "critical" if findings[0]["severity"] != "critical" else "minor"}
            self.assertIsNone(teacher.check(row, {"note": marked, "findings": findings}))
        self.assertTrue(teacher.scenario_prompt(row, random.Random(0)).startswith("Style:"))

    def test_no_key(self):
        with self.assertRaises(LLMError):
            GeminiClient("")


if __name__ == "__main__":
    unittest.main()
