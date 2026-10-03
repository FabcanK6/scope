"""SCOPE's own model: trains on expert notes and approved cases, measures itself fairly, switches on at a bar."""
import unittest

from scope import learning as L
from scope import ownmodel as OM
from scope import profile as P

HIGH = ["Subject {i} was hospitalized and the SAE was never reported to the sponsor.",
        "Consent was signed after dosing for subject {i}; procedures done before consent.",
        "Subject {i} was dosed despite a hold; wrong dose given and not reported."]
MEDIUM = ["Data entry is {i} weeks behind and queries are aging past 30 days.",
          "The delegation log is missing a signature for the new coordinator {i}.",
          "Enrollment is {i} subjects behind target; recovery plan requested."]
LOW = ["All consents in order for {i} subjects; no deviations, no SAEs, storage fine.",
       "Routine visit {i}: data current, SDV up to date, nothing open.",
       "Everything reviewed at visit {i} was compliant; no findings."]


def toy_experts(n=8):
    rows = []
    for risk, templates in (("high", HIGH), ("medium", MEDIUM), ("low", LOW)):
        for i in range(n):
            rows.append({"text": templates[i % 3].format(i=i), "risk": risk, "source": "expert", "id": f"{risk}-{i}"})
    return rows


class TestOwnModel(unittest.TestCase):
    def test_learns_and_switches_on_when_it_clears_the_bar(self):
        model = OM.train(experts=toy_experts())
        self.assertGreaterEqual(model.report["accuracy"], 0.8)
        self.assertTrue(model.active)
        self.assertEqual(model.predict("The SAE for subject 9 was never reported; hospitalized.")["risk"], "high")
        strict = OM.train(experts=toy_experts(), bar=1.01)
        self.assertFalse(strict.active)  # same model, bar not cleared: stays in the background
        self.assertIsNone(OM.second_opinion(strict, "anything", "low"))

    def test_second_look_only_for_a_possible_missed_high(self):
        model = OM.train(experts=toy_experts())
        note = "Subject 4 was hospitalized and the SAE was never reported to the sponsor."
        self.assertTrue(OM.second_opinion(model, note, "low")["second_look"])
        self.assertFalse(OM.second_opinion(model, note, "high")["second_look"])

    def test_approved_cases_join_training_but_never_the_test(self):
        experts = toy_experts()
        case = L.make_case("confirmation", experts[0]["text"], {"risk": {"level": "high"}, "findings": []},
                           P.default_profile())
        library = {"approved": [{**case, "status": "approved", "applies_to": L.ALL_STUDIES}]}
        rows = OM.dataset(library, experts)
        self.assertEqual(sum(r["source"] == "shared" for r in rows), 1)
        model = OM.train(library, experts)
        self.assertEqual((model.report["expert_notes"], model.report["shared_cases"]), (24, 1))
        self.assertEqual(model.report["tested"], 24)  # only expert notes are test notes

    def test_real_expert_notes(self):
        model = OM.train()
        r = model.report
        self.assertEqual(r["expert_notes"], 56)
        self.assertIsNotNone(r["accuracy"])
        self.assertFalse(model.active)  # about 0.6 today: in the background until it clears 0.8

    def test_too_little_data_is_not_measured(self):
        r = OM.train(experts=toy_experts(1)).report
        self.assertIsNone(r["accuracy"])
        self.assertFalse(r["active"])


if __name__ == "__main__":
    unittest.main()
