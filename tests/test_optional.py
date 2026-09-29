"""Small tensor/serialization checks; no pretrained downloads or training runs."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_imports import load

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSLATION = all(importlib.util.find_spec(name) for name in ("torch", "sentencepiece", "sacrebleu", "nltk"))


@unittest.skipUnless(HAS_TORCH, "Install the neural or translation extra")
class TensorTests(unittest.TestCase):
    def test_padding_does_not_change_mean_embedding_score(self):
        import torch
        module = load(ROOT / "chapter_8/72.py")
        embeddings = torch.tensor([[0.,0.], [1.,2.], [3.,4.]])
        model = module.BoWLogisticRegression(embeddings).eval()
        first = model(torch.tensor([1,2]))
        padded = model(torch.tensor([1,2,0,0]))
        self.assertTrue(torch.allclose(first, padded))
        self.assertFalse(model.embedding.weight.requires_grad)

    def test_collate_preserves_label_alignment(self):
        import torch
        module = load(ROOT / "chapter_8/75.py")
        batch = module.collate([
            {"input_ids":torch.tensor([2]), "label":torch.tensor([0.])},
            {"input_ids":torch.tensor([1,2]), "label":torch.tensor([1.])},
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[1,2], [2,0]])
        self.assertEqual(batch["label"].tolist(), [[1.], [0.]])


@unittest.skipUnless(HAS_TRANSLATION, "Install the translation extra")
class TranslationTests(unittest.TestCase):
    def test_beam_selection_and_invalid_width(self):
        module = load(ROOT / "chapter_10_2020/94_2020ver.py")
        hypotheses = [module.BeamHypothesis([2,4,3], -3., 2), module.BeamHypothesis([2,3], -1., 1)]
        self.assertEqual(module.keep_best(hypotheses, 1)[0], hypotheses[1])
        self.assertEqual(module.parse_beam_sizes(["5,1", "5"]), [1,5])
        with self.assertRaises(ValueError):
            module.parse_beam_sizes(["0"])

    def test_checkpoint_and_tokenizer_paths_survive_relocation(self):
        import torch
        module = load(ROOT / "chapter_10_2020/95_2020ver.py")
        config = dict(source_vocab_size=8, target_vocab_size=8, d_model=8, nhead=2,
                      num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16,
                      dropout=0., max_length=16)
        model = module.TransformerNMT(**config).eval()
        state = SimpleNamespace(state_dict=lambda: {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            (original / "spm").mkdir(parents=True)
            source, target = original / "spm/source.model", original / "spm/target.model"
            source.write_bytes(b"tokenizer fixture")
            target.write_bytes(b"tokenizer fixture")
            module.save_checkpoint(original / "best.pt", model, state, state, state,
                                   1, 0., 0., config, source, target, argparse.Namespace())
            moved = root / "relocated"
            shutil.move(str(original), str(moved))
            with patch.object(module.spm, "SentencePieceProcessor", side_effect=lambda **kw: Path(kw["model_file"])):
                restored, source_path, target_path, saved = module.load_model_bundle(moved / "best.pt", torch.device("cpu"))
                self.assertTrue(source_path.is_file())
                self.assertTrue(target_path.is_file())
                self.assertEqual(saved["source_spm"], "spm/source.model")
                # Legacy absolute-path checkpoints and explicit overrides remain supported.
                saved.pop("spm_path_base")
                saved["source_spm"] = str(source_path)
                saved["target_spm"] = str(target_path)
                torch.save(saved, moved / "legacy.pt")
                _, legacy_source, _, _ = module.load_model_bundle(moved / "legacy.pt", torch.device("cpu"))
                self.assertEqual(legacy_source, source_path)
                _, override, _, _ = module.load_model_bundle(moved / "best.pt", torch.device("cpu"), source_spm_path=target_path)
                self.assertEqual(override, target_path)
            src = torch.tensor([[2,4,3]])
            tgt = torch.tensor([[2,5]])
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(src,tgt), restored(src,tgt)))


if __name__ == "__main__":
    unittest.main()
