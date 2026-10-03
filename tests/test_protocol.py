"""Protocol intake: reading documents, drafting rules with verified quotes, building a study profile."""
import io
import unittest
from pathlib import Path

from scope import profile as P
from scope import protocol as PR
from scope.engine import LLMParser, build_system
from tests.test_llm import FakeClient, fake_response

EXAMPLE_PDF = Path(__file__).resolve().parents[1] / "profiles" / "example_protocol_ZLV-301.pdf"

PAGES = [
    "Protocol ZLV-301, Amendment 2 (Version 3.0). A Phase III study in plaque psoriasis.",
    "10.2 Hospitalizations: a hospitalization for an elective procedure that was planned before the subject signed\n"
    "the informed consent form is not a serious adverse event.\n10.3 Reporting: all serious adverse events must be "
    "reported to the sponsor within 2 business days of the site becoming aware of the event.",
]

DRAFT_ANSWER = {
    "study": {"title": "Zelvatinib in psoriasis", "protocol_number": "ZLV-301", "version": "Amendment 2",
              "phase": "III", "therapeutic_area": "Dermatology"},
    "rules": [
        {"topic": "SAE_REPORTING", "rule": "SAEs must be reported within 2 business days of site awareness.",
         "severity": "critical", "page": 2,
         "quote": "all serious adverse events must be reported to the sponsor within 2 business days of the site "
                  "becoming aware of the event"},
        {"topic": "SAE_REPORTING", "rule": "An elective hospitalization planned before consent is not an SAE.",
         "severity": "definition", "page": None,
         "quote": "a hospitalization for an elective procedure that was planned before the subject signed the "
                  "informed consent form is not a serious adverse event"},
        {"topic": "TEMP_EXCURSION", "rule": "Invented rule.", "severity": "major", "page": 1,
         "quote": "Excursions above 30 C must be reported within 1 hour."},  # not in the protocol -> dropped
        {"topic": "MADE_UP_TOPIC", "rule": "Study is in psoriasis.", "severity": "definition", "page": 1,
         "quote": "A Phase III study in plaque psoriasis."},
    ],
}


class TestProtocol(unittest.TestCase):
    def test_read_text_and_pages(self):
        pages = PR.read_document("p.txt", "\f".join(PAGES).encode())
        self.assertEqual(len(pages), 2)
        long = PR.read_document("p.txt", ("word " * 3000).encode())
        self.assertGreater(len(long), 3)
        with self.assertRaises(PR.LLMError):
            PR.read_document("p.xlsx", b"")

    def test_read_example_pdf(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf not installed")
        pages = PR.read_document(EXAMPLE_PDF.name, EXAMPLE_PDF.read_bytes())
        self.assertEqual(len(pages), 6)
        self.assertTrue(PR.verify_quote("within 2 business days of the site becoming aware of the event",
                                        pages[4]))

    def test_read_docx(self):
        try:
            import docx
        except ImportError:
            self.skipTest("python-docx not installed")
        d = docx.Document()
        d.add_paragraph("10.3 All SAEs must be reported within 3 calendar days.")
        buf = io.BytesIO()
        d.save(buf)
        self.assertIn("within 3 calendar days", PR.read_document("p.docx", buf.getvalue())[0])

    def test_long_protocol_keeps_relevant_pages(self):
        filler = ["Statistical methods and sample size. " * 120 for _ in range(60)]
        pages = filler[:30] + [PAGES[1]] + filler[30:]
        keep = PR.select_pages(pages, max_chars=20_000)
        self.assertIn(30, keep)
        self.assertLess(sum(len(pages[i]) for i in keep), 20_001)

    def test_draft_verifies_quotes(self):
        draft = PR.draft_rules(FakeClient([fake_response(DRAFT_ANSWER)]), PAGES)
        self.assertEqual(len(draft["rules"]), 3)
        self.assertEqual(draft["dropped"], 1)  # invented quote
        self.assertEqual(draft["rules"][1]["page"], 2)  # page found from the quote
        self.assertEqual(draft["rules"][2]["topic"], PR.GENERAL)  # unknown topic -> general
        self.assertEqual(draft["study"]["protocol_number"], "ZLV-301")

    def test_apply_rules_builds_a_cited_profile(self):
        draft = PR.draft_rules(FakeClient([fake_response(DRAFT_ANSWER)]), PAGES)
        prof = PR.apply_rules(P.default_profile(), draft, [0, 1], "zlv301.pdf", who="Lead CRA")
        self.assertEqual(prof["name"], "ZLV-301 study profile")
        self.assertEqual(prof["version"], "3.2")
        self.assertEqual(len(prof["study_rules"]), 2)
        self.assertIn("[SAE_REPORTING, critical if broken] (protocol ZLV-301 Amendment 2, p. 2)",
                      prof["study_rules"][0])
        self.assertNotIn("if broken", prof["study_rules"][1])  # a definition, not a severity
        self.assertEqual(prof["protocol"]["reference"], "ZLV-301 Amendment 2")
        self.assertIn("2 rules added from protocol", prof["changes"][-1]["change"])
        system = build_system(prof)
        self.assertIn("within 2 business days of site awareness", system)
        self.assertIn("business days are Monday to Friday", system)
        self.assertEqual(P.loads(P.dumps(prof))["protocol"]["file"], "zlv301.pdf")  # survives save/load

    def test_profile_from_protocol_scores_notes(self):
        draft = PR.draft_rules(FakeClient([fake_response(DRAFT_ANSWER)]), PAGES)
        prof = PR.apply_rules(P.default_profile(), draft, [0, 1], "zlv301.pdf")
        note = "IMV 05/04/2026. Subject 7 SAE (MI) on Friday 05/01/2026, reported to sponsor Wednesday 05/06/2026."
        ans = {"visit": {}, "actions": [], "summary": "Late SAE under this protocol.",
               "findings": [{"issue": "SAE_REPORTING", "status": "active", "severity": "critical",
                             "evidence": "reported to sponsor Wednesday 05/06/2026", "explanation": "3 business days",
                             "repeat": False, "subjects_affected": None, "site_wide": False,
                             "escalation_evidence": ""}]}
        client = FakeClient([fake_response(ans)])
        parser = LLMParser(client, profile=prof)
        self.assertIn("2 business days", parser.system_for(note))
        rec = parser.analyze(note)
        self.assertEqual(rec["risk"]["level"], "high")
        self.assertEqual(rec["profile"]["name"], "ZLV-301 study profile")


if __name__ == "__main__":
    unittest.main()
