"""LLM assistant tests with a fake Gemini transport (no network, no key needed)."""
import json
import unittest

from scope.llm import (
    BUSY_MESSAGE,
    GeminiClient,
    LLMError,
    ModelBusy,
    ModelNotFound,
    ModelSlow,
    QuotaExceeded,
    draft_followup,
    verify_quote,
)

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
        GeminiClient._exhausted.clear()  # shared across sessions in the app; fresh for every test
        self.replies = list(replies)
        self.models = list(models)
        self.calls = []
        self.waits = []
        self._sleep = self.waits.append  # no real waiting in tests

    def _request(self, method, path, body=None):
        self.calls.append((method, path))
        self.bodies = getattr(self, 'bodies', []) + [body]
        if method == "GET":
            return {"models": [{"name": f"models/{m}", "supportedGenerationMethods": ["generateContent"]}
                               for m in self.models]}
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class TestSpeed(unittest.TestCase):
    """A reading should take seconds, not minutes (live: Gemini 3.8 Flash at its default thinking timed out)."""

    def test_gemini_3_thinks_low_older_models_unchanged(self):
        client = FakeClient([fake_response("ok"), fake_response("ok")], models=("gemini-3.8-flash",))
        client.generate("s", "p")
        self.assertEqual(client.bodies[-1]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "low"})
        client = FakeClient([fake_response("ok")], models=("gemini-2.5-flash",))
        client.generate("s", "p")
        self.assertNotIn("thinkingConfig", client.bodies[-1]["generationConfig"])

    def test_model_that_refuses_a_thinking_level_is_asked_without(self):
        client = FakeClient([LLMError("Gemini API error 400: Thinking level is not supported for this model."),
                             fake_response("ok")], models=("gemini-3-flash",))
        self.assertEqual(client.generate("s", "p"), "ok")
        self.assertNotIn("thinkingConfig", client.bodies[-1]["generationConfig"])
        self.assertIn("gemini-3-flash", client._no_thinking)

    def test_timeout_moves_on_at_once_and_slow_model_goes_last(self):
        client = FakeClient([ModelSlow("timeout"), fake_response("ok"), fake_response("ok")],
                            models=("gemini-9-flash", "gemini-8-flash"))
        self.assertEqual(client.generate("s", "p"), "ok")
        self.assertEqual(client.waits, [])  # no waiting and retrying a model that timed out
        self.assertEqual(client.model, "gemini-8-flash")
        order = client.candidates()
        self.assertLess(order.index("gemini-8-flash"), order.index("gemini-9-flash"))  # the slow one goes last
        client.generate("s", "p")
        self.assertEqual(client.calls[-1], ("POST", "models/gemini-8-flash:generateContent"))

    def test_engine_log_says_what_was_tried(self):
        from scope.engine import LLMParser

        empty = {"visit": {}, "findings": [], "actions": [], "summary": "Nothing to report."}
        client = FakeClient([ModelSlow("timeout"), QuotaExceeded('"PerDay"'), fake_response(empty)],
                            models=("gemini-9-flash", "gemini-8-flash", "gemini-7-flash"))
        rec = LLMParser(client).analyze(NOTE)
        log = rec["engine_log"]
        self.assertTrue(log[0].startswith("gemini-9-flash: timed out"), log)
        self.assertTrue(log[1].startswith("gemini-8-flash: daily free quota used up"), log)
        self.assertTrue(log[2].startswith("gemini-7-flash: answered"), log)
        client = FakeClient([ModelBusy("HTTP 503")] * 20, models=("gemini-9-flash",))
        with self.assertRaises(LLMError) as ctx:
            LLMParser(client).analyze(NOTE)
        self.assertIn("busy", ctx.exception.engine_log[-1])

    def test_model_out_of_daily_quota_is_skipped_until_midnight_pacific(self):
        from scope.llm import next_quota_reset

        client = FakeClient([QuotaExceeded('"quotaId": "GenerateRequestsPerDayPerProjectPerModel"'),
                             fake_response("ok"), fake_response("ok")], models=("gemini-9-flash", "gemini-8-flash"))
        client.generate("s", "p")
        other_session = GeminiClient("test-key")  # same key, new visitor: the spent model goes last
        other_session._listed = ["gemini-9-flash", "gemini-8-flash"]
        self.assertEqual(other_session.candidates()[:2], ["gemini-8-flash", "gemini-flash-latest"])
        self.assertEqual(other_session.candidates()[-1], "gemini-9-flash")
        # 23:30 Pacific (daylight time) on 3 Oct 2026 -> resets 30 minutes later
        self.assertEqual(next_quota_reset(1791095400.0) - 1791095400.0, 1800.0)

    def test_time_budget(self):
        client = FakeClient([ModelSlow("timeout")] * 6, models=("gemini-9-flash", "gemini-8-flash", "gemini-7-flash"))
        t = iter([0, 0, 0, 70, 70, 140, 140, 220, 220, 300, 300])
        client._clock = lambda: next(t)
        with self.assertRaises(LLMError) as ctx:
            client.generate("s", "p")
        self.assertEqual(str(ctx.exception), BUSY_MESSAGE)
        posts = [c for c in client.calls if c[0] == "POST"]
        self.assertLess(len(posts), 4)  # gave up once the budget was spent, not after every model


class TestLLM(unittest.TestCase):
    def test_verify_quote(self):
        self.assertTrue(verify_quote("violating the mandatory 24-hour protocol reporting window", NOTE))
        self.assertTrue(verify_quote("NO TEMPERATURE excursions were noted on the digital data logger.", NOTE))
        self.assertFalse(verify_quote("The PI refused to sign the delegation log.", NOTE))
        # parts joined with an ellipsis (live: PED-44 status epilepticus finding was dropped over this)
        self.assertTrue(verify_quote("The subject was hospitalized on August 30, 2026... violating the mandatory "
                                     "24-hour protocol reporting window.", NOTE))
        self.assertTrue(verify_quote("The subject was hospitalized on August 30, 2026 \u2026 No temperature "
                                     "excursions were noted", NOTE))
        self.assertFalse(verify_quote("No temperature excursions were noted... The subject was hospitalized",
                                      NOTE))  # wrong order
        self.assertFalse(verify_quote("The subject was hospitalized on August 30, 2026... and then died.", NOTE))
        self.assertFalse(verify_quote("The... Sponsor", NOTE))  # pieces too short to mean anything
        self.assertTrue(verify_quote("hospitalized ... violating the mandatory 24-hour", NOTE))
        far = "The subject was hospitalized." + " Filler sentence." * 30 + " The site did not notify the Sponsor."
        self.assertFalse(verify_quote("The subject was hospitalized... The site did not notify the Sponsor", far))

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
        from scope.data.handwritten import load_handwritten, load_realistic, load_stress
        from scope.engine import SYSTEM

        for r in load_handwritten() + load_realistic() + load_stress():
            first = " ".join(r["text"].split())[:80]
            self.assertNotIn(first, " ".join(SYSTEM.split()), r["id"])

    def test_looping_visit_detail_does_not_sink_good_findings(self):
        from scope.engine import LLMParser

        note = ("IMV, Site 44, 12 October 2026. The pharmacy fridge temperature log is blank for Saturday and Sunday "
                "on the last two weekends; staff said no one checks the log at weekends. Drug accountability "
                "reconciled for all kits.")
        # live (gemini-3.5-flash): correct findings, then visit_type looped into thousands of digits
        answer = {"findings": [{"issue": "TEMP_EXCURSION", "status": "active", "severity": "major",
                                "evidence": "The pharmacy fridge temperature log is blank for Saturday and Sunday on "
                                            "the last two weekends", "explanation": "logs not kept"}],
                  "actions": [], "summary": "Weekend log gaps.",
                  "visit": {"visit_type": "IMV" + "2689357293572935" * 200, "visit_date": "12 October 2026",
                            "site": "Site 44"}}
        client = FakeClient([fake_response(answer)])
        rec = LLMParser(client).analyze(note)
        self.assertEqual(len([c for c in client.calls if c[0] == "POST"]), 1)  # accepted first time
        self.assertEqual(rec["risk"]["level"], "medium")
        self.assertEqual(rec["visit"]["visit_type"]["code"], "IMV")  # the looping tail is not kept

    def test_garbled_visit_type_does_not_swallow_site_and_date(self):
        from scope.engine import LLMParser

        # live (v8.6, fallback model): the visit type came back garbled, a piece of it matched a chunk of the
        # first line, and that chunk pushed out the real site and visit date
        note = ("IMV site 77 - 7 Oct 2026. Reviewed ICFs for 6 subjects; subj 77-004 signed an outdated ICF v2.0. "
                "PI to re-consent subj 77-004 by 14 Oct 2026.")
        answer = {"visit": {"visit_type": "imv礼 site 77 - 7 oct 2026… site 77 - 7 oct 2026…",
                            "visit_date": None, "site": "site 77"},
                  "summary": "Outdated consent.", "actions": [],
                  "findings": [{"issue": "CONSENT", "status": "active", "severity": "major",
                                "evidence": "subj 77-004 signed an outdated ICF v2.0", "explanation": "old ICF"}]}
        rec = LLMParser(FakeClient([fake_response(answer)])).analyze(note)
        v = rec["visit"]
        self.assertEqual(v["visit_type"]["code"], "IMV")
        self.assertEqual(v["visit_type"]["text"], "IMV")
        self.assertEqual(v["site"]["text"], "site 77")
        self.assertEqual(v["visit_date"]["iso"], "2026-10-07")
        self.assertEqual(rec["risk"]["level"], "medium")
        self.assertEqual(rec["review"]["reasons"], [])  # recovered from the header line, nothing to review
        self.assertTrue(any(t.startswith("garbled visit type") for t in rec["engine_log"]))
        # the same garbled value through _locate alone would have matched a long chunk; a clean value still works
        rec = LLMParser(FakeClient([fake_response({**answer, "visit": {"visit_type": "IMV",
                                                                       "visit_date": "7 Oct 2026"}})])).analyze(note)
        self.assertEqual(rec["visit"]["visit_date"]["iso"], "2026-10-07")
        self.assertFalse(any(t.startswith("garbled") for t in rec["engine_log"]))

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
        from scope.engine import Unreadable

        with self.assertRaises(Unreadable) as ctx:
            LLMParser(FakeClient([fake_response(garbled), fake_response(empty), fake_response("not json")])
                      ).analyze(note)
        self.assertTrue(str(ctx.exception).startswith(UNREADABLE))
        self.assertEqual(ctx.exception.reason, "not valid JSON")
        self.assertEqual(ctx.exception.raw, "not json")  # kept for diagnosis
        # a bad answer from one model -> the next attempt goes to another model
        client = FakeClient([fake_response(empty), fake_response(good)], models=("gemini-9-flash", "gemini-8-flash"))
        rec = LLMParser(client).analyze(note)
        self.assertEqual(rec["model"], "gemini-8-flash")
        self.assertEqual(client.avoid, set())
        self.assertNotIn("verified", rec["llm_output"]["findings"][0])  # raw answer is untouched

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

    def test_quota_falls_back_to_next_model(self):
        day = '{"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}'
        minute = '{"error": {"details": [{"quotaId": "GenerateRequestsPerMinute"}, {"retryDelay": "27s"}]}}'
        client = FakeClient([QuotaExceeded(day), fake_response("ok")], models=("gemini-9-flash", "gemini-8-flash"))
        self.assertEqual(client.generate("s", "p"), "ok")
        self.assertEqual(client.model, "gemini-8-flash")
        client = FakeClient([QuotaExceeded(day)] * 6, models=("gemini-9-flash", "gemini-8-flash"))
        with self.assertRaises(LLMError) as ctx:
            client.generate("s", "p")
        self.assertIn("midnight Pacific", str(ctx.exception))
        client = FakeClient([QuotaExceeded(minute)] * 6, models=("gemini-9-flash",))
        with self.assertRaises(LLMError) as ctx:
            client.generate("s", "p")
        self.assertIn("27 seconds", str(ctx.exception))

    def test_rubric_v3_escalation(self):
        from scope.llm import final_severity

        def f(sev, **kw):
            return final_severity({"severity": sev, "escalation_ok": True, **kw})

        self.assertEqual(f("minor"), ("minor", []))
        self.assertEqual(f("minor", subjects_affected=3), ("major", ["3 subjects affected"]))
        self.assertEqual(f("minor", subjects_affected=2), ("minor", []))
        self.assertEqual(f("major", site_wide=True), ("major", []))  # spread raises up to major only
        self.assertEqual(f("major", repeat=True), ("critical", ["repeat finding"]))
        self.assertEqual(f("minor", subjects_affected=5, repeat=True),
                         ("critical", ["5 subjects affected", "repeat finding"]))  # both rules stack
        self.assertEqual(f("critical", repeat=True), ("critical", []))
        self.assertEqual(final_severity({"severity": "minor", "repeat": True, "escalation_ok": False}),
                         ("minor", []))

    def test_escalation_needs_evidence_in_note(self):
        from scope.engine import LLMParser

        note = ("IMV 03/02/2026. Con-meds not entered for Subjects 101, 102 and 103. "
                "Two queries still open from the last visit.")
        answer = {"visit": {}, "actions": [], "summary": "Data gaps.",
                  "findings": [
                      {"issue": "DATA_ENTRY_BACKLOG", "status": "active", "severity": "minor",
                       "evidence": "Con-meds not entered for Subjects 101, 102 and 103.", "explanation": "",
                       "repeat": False, "subjects_affected": 3, "site_wide": False,
                       "escalation_evidence": "for Subjects 101, 102 and 103"},
                      {"issue": "QUERY_AGING", "status": "active", "severity": "minor",
                       "evidence": "Two queries still open from the last visit.", "explanation": "",
                       "repeat": True, "subjects_affected": None, "site_wide": False,
                       "escalation_evidence": "cited in the previous three reports"}]}  # not in the note
        rec = LLMParser(FakeClient([fake_response(answer)])).analyze(note)
        sev = {f["issue"]: f["final_severity"] for f in rec["findings"]}
        self.assertEqual(sev, {"DATA_ENTRY_BACKLOG": "major", "QUERY_AGING": "minor"})
        self.assertEqual(rec["risk"]["level"], "medium")  # 3 + 1 points
        self.assertTrue(any("not raised" in r for r in rec["review"]["reasons"]))

    def test_v1_baseline_keeps_its_12_issue_types(self):
        from scope.schema import ISSUE_CODES, V1_ISSUE_CODES

        self.assertEqual(len(V1_ISSUE_CODES), 12)
        self.assertEqual(len(ISSUE_CODES), 22)
        self.assertTrue(set(V1_ISSUE_CODES) <= set(ISSUE_CODES))

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
