"""Study profiles: editable rubric, study rules, escalation settings and CRA corrections."""
import json
import unittest

from scope import profile as P
from scope.engine import LLMParser, build_system, extract_schema
from tests.test_llm import FakeClient, fake_response

NOTE = ("IMV Site 12, 03/02/2026. Week 4 PK sample for Subject 12-004 was not collected. "
        "Enrollment is 2 behind target. Con-meds not entered for Subjects 12-001, 12-002, 12-003 and 12-005.")


def finding(issue, severity="minor", status="active", evidence="Enrollment is 2 behind target.", **kw):
    return {"issue": issue, "status": status, "severity": severity, "evidence": evidence, "explanation": "",
            "repeat": False, "subjects_affected": None, "site_wide": False, "escalation_evidence": "", **kw}


def answer(*findings):
    return fake_response({"visit": {}, "actions": [], "summary": "Visit summary.", "findings": list(findings)})


class TestProfile(unittest.TestCase):
    def test_default_profile_is_rubric_v31(self):
        p = P.default_profile()
        self.assertEqual(len(p["topics"]), 22)
        text = P.rubric_text(p)
        for t in p["topics"]:
            self.assertIn(t["code"], text)
        self.assertIn("reported outside 24 hours", text)
        self.assertEqual(P.validate(p)["thresholds"], {"medium": 3, "high": 6})

    def test_hide_topic_and_add_study_topic(self):
        p = P.default_profile()
        for t in p["topics"]:
            if t["code"] == "ENROLLMENT_LAG":
                t["enabled"] = False
        p["topics"].append({"display": "Missed PK samples", "group": "Study-specific", "minor": "",
                            "major": "", "critical": "a scheduled PK sample not collected"})
        p = P.validate(p)
        codes = extract_schema(p)["properties"]["findings"]["items"]["properties"]["issue"]["enum"]
        self.assertNotIn("ENROLLMENT_LAG", codes)
        self.assertIn("STUDY_MISSED_PK_SAMPLES", codes)
        self.assertIn("a scheduled PK sample not collected", build_system(p))
        client = FakeClient([answer(
            finding("ENROLLMENT_LAG", "major"),  # hidden in this study -> ignored
            finding("STUDY_MISSED_PK_SAMPLES", "critical",
                    evidence="Week 4 PK sample for Subject 12-004 was not collected."))])
        rec = LLMParser(client, profile=p).analyze(NOTE)
        self.assertEqual([i["code"] for i in rec["issues"]], ["STUDY_MISSED_PK_SAMPLES"])
        self.assertEqual(rec["issues"][0]["display"], "Missed PK samples")
        self.assertEqual(rec["risk"]["level"], "high")
        self.assertEqual(rec["profile"]["name"], p["name"])

    def test_study_rules_and_thresholds(self):
        p = P.default_profile()
        p["study_rules"] = ["A missed Week 4 PK sample is critical (PROTOCOL_DEVIATION)."]
        p["thresholds"] = {"medium": 3, "high": 9}
        p = P.validate(p)
        self.assertIn("override the rubric", build_system(p))
        self.assertIn("missed Week 4 PK sample is critical", build_system(p))
        rec = LLMParser(FakeClient([answer(finding("ENROLLMENT_LAG", "critical"))]), profile=p).analyze(NOTE)
        self.assertEqual(rec["risk"]["level"], "medium")  # 6 points: high needs 9 in this study

    def test_escalation_settings(self):
        many = {"severity": "minor", "escalation_ok": True, "subjects_affected": 4, "repeat": True}
        p = P.default_profile()
        self.assertEqual(P.final_severity(many, p)[0], "critical")
        p["escalation"] = {"subjects_threshold": 5, "subjects_max": "major", "repeat": False}
        p = P.validate(p)
        self.assertEqual(P.final_severity(many, p), ("minor", []))
        self.assertIn('"repeat" = always false', P.rubric_text(p))

    def test_corrections_need_approval_then_teach(self):
        p = P.default_profile()
        c = P.add_correction(p, note=NOTE, issue="ENROLLMENT_LAG", status="active", severity="minor",
                             quote="Enrollment is 2 behind target", reason="2 behind is normal for this study",
                             scope_said="major", cra_risk="low")
        parser = LLMParser(FakeClient([]), profile=p)
        self.assertNotIn("2 behind is normal", parser.system_for(NOTE))  # not approved yet
        c["approved"] = True
        self.assertIn("2 behind is normal", parser.system_for(NOTE))
        self.assertIn("ENROLLMENT_LAG (Enrollment) is active, minor", parser.system_for(NOTE))

    def test_most_relevant_corrections_are_used(self):
        p = P.default_profile()
        for i in range(10):
            c = P.add_correction(p, note=f"Fridge log gap number {i}", issue="TEMP_EXCURSION", status="active",
                                 severity="minor", quote=f"fridge log gap {i}", reason="")
            c["approved"] = True
        c = P.add_correction(p, note="Enrollment is 2 behind target at site 12", issue="ENROLLMENT_LAG",
                             status="active", severity="minor", quote="Enrollment is 2 behind target", reason="")
        c["approved"] = True
        picked = P.relevant_corrections(p, NOTE, k=3)
        self.assertEqual(picked[0]["issue"], "ENROLLMENT_LAG")
        self.assertEqual(len(picked), 3)

    def test_file_round_trip_and_errors(self):
        p = P.default_profile()
        p["name"] = "Study ABC-123"
        P.log_change(p, "Enrollment lag made minor", who="Lead CRA")
        again = P.loads(P.dumps(p))
        self.assertEqual(again["name"], "Study ABC-123")
        self.assertEqual(again["changes"][0]["change"], "Enrollment lag made minor")
        self.assertEqual(P.fingerprint(again), P.fingerprint(p))
        with self.assertRaises(P.ProfileError):
            P.loads("{not json")
        with self.assertRaises(P.ProfileError):
            P.loads(json.dumps({"name": "x"}))
        bad = P.default_profile()
        bad["thresholds"] = {"medium": 7, "high": 6}
        with self.assertRaises(P.ProfileError):
            P.validate(bad)
        self.assertEqual(P.bump_version("3.1"), "3.2")
        self.assertEqual(P.bump_version("1"), "2")


if __name__ == "__main__":
    unittest.main()
