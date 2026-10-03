"""Shared learning: cases, review, storage (local and Hugging Face, faked), and use in the engine."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from scope import learning as L
from scope import profile as P
from scope.engine import LLMParser
from tests.test_llm import FakeClient, fake_response

NOTE = ("IMV Site 7, 2 October 2026. The temperature log for the pharmacy fridge has no entries for the weekend of "
        "26-27 September; the coordinator says nobody checks it at weekends. No excursion was recorded.")
SIMILAR = ("IMV Site 9, 9 October 2026. Fridge temperature log is blank for Saturday and Sunday again, nobody "
           "checks it at weekends. Drug accountability fine.")
UNRELATED = "IMV Site 3, 1 October 2026. Enrollment is 4 behind target; the PI will send a recovery plan."
EMPTY = {"visit": {}, "findings": [], "actions": [], "summary": "Nothing to report."}


def reading(text=NOTE):
    return LLMParser(FakeClient([fake_response(EMPTY)])).analyze(text)


def correction(**over):
    c = {"issue": "TEMP_EXCURSION", "display": "IP storage and temperature", "status": "active",
         "severity": "major", "quote": "nobody checks it at weekends", "reason": "Unmonitored storage is major.",
         "risk": "medium"}
    return {**c, **over}


class TestCases(unittest.TestCase):
    def test_make_and_validate(self):
        case = L.make_case("correction", NOTE, reading(), P.default_profile(), correction(), by="FM")
        self.assertEqual(case["status"], "pending")
        self.assertEqual(case["scope_said"]["risk"], "low")
        self.assertEqual(case["note_hash"], L.note_hash("  " + NOTE.upper()))
        L.validate_case(case)
        with self.assertRaises(ValueError):
            L.make_case("correction", NOTE, reading(), P.default_profile())
        with self.assertRaises(L.LearningError):
            L.validate_case({**case, "id": "../../etc"})
        with self.assertRaises(L.LearningError):
            L.validate_case({**case, "correction": {**correction(), "quote": ""}})


class TestLocalStore(unittest.TestCase):
    def test_review_flow(self):
        with tempfile.TemporaryDirectory() as d:
            store = L.LocalStore(d)
            case = L.make_case("correction", NOTE, reading(), P.default_profile(), correction())
            conf = L.make_case("confirmation", UNRELATED, reading(UNRELATED), P.default_profile())
            store.add(case)
            store.add(conf)
            lib = store.load()
            self.assertEqual(len(lib["pending"]), 2)
            self.assertEqual(L.digest(lib), L.digest({"approved": []}))
            store.decide(lib["pending"][0], approve=True, applies_to=L.ALL_STUDIES, note="agreed")
            store.decide(lib["pending"][1], approve=False)
            lib = store.load()
            self.assertEqual([len(lib[f]) for f in L.FOLDERS], [0, 1, 1])
            self.assertEqual(lib["approved"][0]["applies_to"], L.ALL_STUDIES)
            self.assertFalse(list(Path(d, "pending").glob("*.json")))
            (Path(d) / "approved" / "junk.json").write_text("{not json")
            self.assertEqual(len(store.load()["approved"]), 1)  # bad files are skipped, not fatal


class TestUsingRulings(unittest.TestCase):
    def library(self, applies_to=L.ALL_STUDIES):
        case = L.make_case("correction", NOTE, reading(), P.default_profile(), correction())
        return {"approved": [{**case, "status": "approved", "applies_to": applies_to}], "pending": [],
                "rejected": []}

    def test_relevant_only_when_similar(self):
        prof = P.default_profile()
        topics = P.topic_map(prof)
        cases = L.rulings(self.library(), prof["name"], topics)
        self.assertEqual(len(cases), 1)
        self.assertTrue(L.relevant(cases, SIMILAR))
        self.assertFalse(L.relevant(cases, UNRELATED))
        self.assertFalse(L.relevant(cases, NOTE, exclude_same_note=True))
        # a ruling for one study does not reach another; a switched-off topic is not used
        self.assertFalse(L.rulings(self.library("Other study"), prof["name"], topics))
        topics["TEMP_EXCURSION"]["enabled"] = False
        self.assertFalse(L.rulings(self.library(), prof["name"], topics))

    def test_engine_uses_and_reports_rulings(self):
        parser = LLMParser(FakeClient([fake_response(EMPTY), fake_response(EMPTY)]))
        parser.library = self.library()
        system = parser.system_for(SIMILAR)
        self.assertIn("Rulings from expert reviewers", system)
        self.assertIn('"nobody checks it at weekends": TEMP_EXCURSION', system)
        self.assertNotIn("Rulings from expert reviewers", parser.system_for(UNRELATED))
        rec = parser.analyze(SIMILAR)
        self.assertEqual(len(rec["learned_from"]), 1)
        self.assertIn("an active problem, major", rec["learned_from"][0]["ruling"])
        parser.holdout = True
        self.assertNotIn("Rulings from expert reviewers", parser.system_for(NOTE))

    def test_training_rows(self):
        lib = self.library()
        conf = L.make_case("confirmation", UNRELATED, {"risk": {"level": "medium"}, "findings": [
            {"issue": "ENROLLMENT_LAG", "status": "active", "severity": "major", "verified": True}]},
            P.default_profile())
        lib["approved"].append({**conf, "status": "approved", "applies_to": L.ALL_STUDIES})
        rows = L.training_rows(lib)
        self.assertEqual(rows[0]["risk"], "medium")
        self.assertEqual(rows[0]["severities"], {"TEMP_EXCURSION": "major"})
        self.assertEqual((rows[1]["risk"], rows[1]["issues"]), ("medium", ["ENROLLMENT_LAG"]))


class FakeHub(types.ModuleType):
    """Stands in for huggingface_hub: a dict of files, commits recorded."""

    def __init__(self):
        super().__init__("huggingface_hub")
        self.files, self.commits = {}, []
        hub = self

        class Op:
            def __init__(self, path_in_repo, path_or_fileobj=None):
                self.path_in_repo, self.data = path_in_repo, path_or_fileobj

        class CommitOperationAdd(Op):
            pass

        class CommitOperationDelete(Op):
            pass

        class HfApi:
            def __init__(self, token=None):
                self.token = token

            def create_commit(self, repo_id, repo_type, operations, commit_message):
                assert repo_type == "dataset"
                hub.commits.append((repo_id, commit_message, [type(o).__name__ for o in operations]))
                for o in operations:
                    if isinstance(o, CommitOperationAdd):
                        hub.files[o.path_in_repo] = o.data
                    else:
                        hub.files.pop(o.path_in_repo)

        def snapshot_download(repo_id, repo_type, token, allow_patterns):
            d = tempfile.mkdtemp()
            for path, data in hub.files.items():
                Path(d, path).parent.mkdir(parents=True, exist_ok=True)
                Path(d, path).write_bytes(data)
            return d

        self.CommitOperationAdd, self.CommitOperationDelete = CommitOperationAdd, CommitOperationDelete
        self.HfApi, self.snapshot_download = HfApi, snapshot_download


class TestHFStore(unittest.TestCase):
    def setUp(self):
        self.hub = FakeHub()
        sys.modules["huggingface_hub"] = self.hub

    def tearDown(self):
        sys.modules.pop("huggingface_hub", None)

    def test_add_review_load(self):
        store = L.HFStore("someone/scope-learning", "hf_test")
        case = L.make_case("correction", NOTE, reading(), P.default_profile(), correction())
        store.add(case)
        self.assertIn(f"pending/{case['id']}.json", self.hub.files)
        pending = store.load()["pending"][0]
        store.decide(pending, approve=True, applies_to="Demo study")
        self.assertEqual(self.hub.commits[-1][2], ["CommitOperationAdd", "CommitOperationDelete"])
        self.assertEqual(list(self.hub.files), [f"approved/{case['id']}.json"])
        approved = store.load()["approved"][0]
        self.assertEqual(approved["applies_to"], "Demo study")
        store.add_vote(L.make_vote(f"case:{case['id']}", L.voter_hash("x"), "agree"))
        self.assertIn(f"votes/case-{case['id']}/{L.voter_hash('x')}.json", self.hub.files)
        self.assertEqual(len(store.load()["votes"]), 1)
        self.assertEqual(json.loads(self.hub.files[f"approved/{case['id']}.json"])["status"], "approved")

    def test_needs_settings_and_reports_errors(self):
        with self.assertRaises(L.LearningError):
            L.HFStore("", "hf_test")

        def broken(**kw):
            raise OSError("network down")

        self.hub.snapshot_download = broken
        with self.assertRaisesRegex(L.LearningError, "Could not read"):
            L.HFStore("someone/scope-learning", "hf_test").load()



class TestCommunity(unittest.TestCase):
    def test_note_consensus(self):
        self.assertEqual(L.note_consensus(["high"] * 3), "high")
        self.assertIsNone(L.note_consensus(["high"] * 2))  # not enough people yet
        self.assertEqual(L.note_consensus(["high", "high", "high", "medium"]), "high")  # 75%
        self.assertIsNone(L.note_consensus(["high", "high", "high", "low"]))  # someone two levels away
        self.assertIsNone(L.note_consensus(["medium"] * 3 + ["high"] * 2))  # only 60%

    def test_shared_case_goes_live_when_three_agree(self):
        with tempfile.TemporaryDirectory() as d:
            store = L.LocalStore(d)
            alice, bob, cara, dan = (L.voter_hash(x) for x in ("a", "b", "c", "d"))
            case = L.make_case("correction", NOTE, reading(), P.default_profile(), correction(), voter=alice)
            store.add(case)
            store.add_vote(L.make_vote(f"case:{case['id']}", alice, "agree"))  # the sharer counts once only
            store.add_vote(L.make_vote(f"case:{case['id']}", bob, "agree"))
            self.assertEqual(L.case_tally(case, store.load()), (2, 0))
            self.assertEqual(L.apply_consensus(store, store.load()), [])
            store.add_vote(L.make_vote(f"case:{case['id']}", bob, "disagree"))  # changes mind: replaces
            store.add_vote(L.make_vote(f"case:{case['id']}", bob, "agree"))
            store.add_vote(L.make_vote(f"case:{case['id']}", cara, "agree"))
            decided = L.apply_consensus(store, store.load())
            self.assertEqual([c["status"] for c in decided], ["approved"])
            lib = store.load()
            self.assertEqual(lib["approved"][0]["applies_to"], L.ALL_STUDIES)
            self.assertIn("3 agreed", lib["approved"][0]["curator_note"])
            self.assertEqual(len(lib["votes"]), 3)  # alice, bob (once), cara
            # dropped when three disagree
            other = L.make_case("confirmation", UNRELATED, reading(UNRELATED), P.default_profile(), voter=alice)
            store.add(other)
            for v in (bob, cara, dan):
                store.add_vote(L.make_vote(f"case:{other['id']}", v, "disagree"))
            self.assertEqual([c["status"] for c in L.apply_consensus(store, store.load())], ["rejected"])

    def test_study_rulings_stay_with_their_study(self):
        prof = {**P.default_profile(), "name": "Demo · Psoriasis · ZLV-301 (Phase III)"}
        case = L.make_case("correction", NOTE, reading(), prof, correction())
        self.assertEqual(L.default_scope(case), prof["name"])

    def test_practice_notes_become_labelled_rows(self):
        from scope.data.handwritten import load_practice

        practice = load_practice()
        self.assertEqual(len(practice), 25)
        votes = [L.make_vote(f"note:{practice[0]['id']}", L.voter_hash(x), "high") for x in "abc"]
        votes.append(L.make_vote(f"note:{practice[1]['id']}", L.voter_hash("a"), "low"))
        rows = L.community_rows({"votes": votes}, practice)
        self.assertEqual([(r["id"], r["risk"], r["votes"]) for r in rows], [(practice[0]["id"], "high", 3)])
        with self.assertRaises(L.LearningError):
            L.make_vote("note:../x", L.voter_hash("a"), "high")
        with self.assertRaises(L.LearningError):
            L.make_vote(f"note:{practice[0]['id']}", L.voter_hash("a"), "agree")

    def test_own_model_learns_from_agreed_practice_notes(self):
        from scope import ownmodel as OM
        from scope.data.handwritten import load_practice

        practice = load_practice()
        votes = [L.make_vote(f"note:{n['id']}", L.voter_hash(x), n["risk"]) for n in practice for x in "abc"]
        rows = OM.dataset({"votes": votes, "approved": []}, experts=[], practice=practice)
        self.assertEqual({r["source"] for r in rows}, {"community"})
        self.assertEqual(len(rows), 25)


if __name__ == "__main__":
    unittest.main()
