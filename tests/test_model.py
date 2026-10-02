"""Model tests. Skipped automatically when torch/transformers are not installed.

Builds a tiny randomly initialised BERT (no download needed) and checks the
forward pass, label alignment, save/load round trip, the parser and calibration.
"""
import tempfile
import unittest

try:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestScopeModel(unittest.TestCase):
    def setUp(self):
        from scope.data.generate import generate
        from scope.model import ScopeModel

        self.rows = generate(8, seed=21)
        vocab = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
        for r in self.rows:
            vocab.update(w.lower() for w in r["tokens"])
        self.tmp = tempfile.TemporaryDirectory()
        vocab_file = f"{self.tmp.name}/vocab.txt"
        with open(vocab_file, "w") as f:
            f.write("\n".join(sorted(vocab)))
        self.tokenizer = BertTokenizerFast(vocab_file=vocab_file)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=512)
        self.model = ScopeModel(BertModel(cfg))

    def tearDown(self):
        self.tmp.cleanup()

    def test_forward_and_alignment(self):
        from scope.train import make_collate

        batch = make_collate(self.tokenizer, 512)(self.rows[:2])
        labelled = (batch["tag_labels"][0] != -100).sum().item()
        self.assertEqual(labelled, len(self.rows[0]["tokens"]))  # one label per word (first sub-token)
        out = self.model(**batch)
        self.assertEqual(out["risk_logits"].shape, (2, 3))
        self.assertEqual(out["issue_logits"].shape, (2, 12))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

    def test_save_load_and_analyze(self):
        from scope.model import ScopeModel
        from scope.predict import BertParser, load_parser

        d = f"{self.tmp.name}/ckpt"
        self.model.save(d, self.tokenizer, base_model="tiny-test", max_length=512)
        _model, _tok, cfg = ScopeModel.load(d)
        self.assertEqual(cfg["base_model"], "tiny-test")
        parser = load_parser(d, device="cpu")
        self.assertIsInstance(parser, BertParser)
        rec = parser.analyze(self.rows[0]["text"])
        self.assertIn(rec["risk"]["level"], ["low", "medium", "high"])
        self.assertAlmostEqual(sum(rec["risk"]["probabilities"].values()), 1.0, places=3)
        preds = parser.predict_batch([r["text"] for r in self.rows])
        self.assertEqual(len(preds), len(self.rows))
        self.assertEqual(len(preds[0]["tags"]), len(self.rows[0]["tokens"]))

    def test_fit_temperature(self):
        from scope.model import fit_temperature

        torch.manual_seed(0)
        labels = torch.randint(0, 3, (200,))
        logits = torch.nn.functional.one_hot(labels, 3).float() * 8.0
        flip = torch.rand(200) < 0.3          # 30% wrong but very confident -> T should grow
        logits[flip] = logits[flip].roll(1, dims=1)
        self.assertGreater(fit_temperature(logits, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
