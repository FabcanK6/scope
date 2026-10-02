"""LLM assistant tests with a fake Gemini transport (no network, no key needed)."""
import json
import unittest

from scope.llm import GeminiClient, LLMError, ModelNotFound, draft_followup, verify_quote

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

        empty = {"visit": {}, "findings": [], "actions": [], "summary": ""}
        client = FakeClient([ModelNotFound("gone"), fake_response(empty)], models=("gemini-9-flash", "gemini-8-flash"))
        rec = LLMParser(client).analyze(NOTE)
        self.assertEqual(rec["risk"]["level"], "low")
        self.assertEqual(client.model, "gemini-8-flash")
        with self.assertRaises(LLMError):
            LLMParser(FakeClient([LLMError("The free Gemini quota is used up for now.")])).analyze(NOTE)

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
