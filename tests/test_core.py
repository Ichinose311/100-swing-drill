"""Regression checks using synthetic, local data only."""
import contextlib
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_imports import load


class CoreTests(unittest.TestCase):
    def test_alternate_characters_match_task_01(self):
        module = load(ROOT / "chapter_1/01.py")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            module.main()
        self.assertEqual(output.getvalue().strip(), "タクシー")

    def test_character_and_word_ngrams(self):
        module = load(ROOT / "chapter_1/ngrams.py")
        self.assertEqual(module.generate_n_gram(3, "I am"), ["I a", " am"])
        self.assertEqual(module.generate_n_gram(2, ["I", "am", "an", "NLPer"]),
                         [["I", "am"], ["am", "an"], ["an", "NLPer"]])
        self.assertEqual(module.generate_n_gram(3, []), [])
        with self.assertRaises(ValueError):
            module.generate_n_gram(0, "text")

    def test_membership_is_reported_for_both_sets(self):
        module = load(ROOT / "chapter_1/06.py")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            module.main()
        self.assertIn("X contains se: True", output.getvalue())
        self.assertIn("Y contains se: False", output.getvalue())

    def test_cipher_leaves_non_ascii_letters_unchanged(self):
        module = load(ROOT / "chapter_1/08.py")
        original = "Hello, World! café 日本語"
        self.assertEqual(module.cipher("azAé"), "zaAé")
        self.assertEqual(module.cipher(module.cipher(original)), original)

    def test_all_string_exercises_run_outside_repo(self):
        with tempfile.TemporaryDirectory() as cwd:
            for path in sorted((ROOT / "chapter_1").glob("[0-9][0-9].py")):
                result = subprocess.run([sys.executable, str(path)], cwd=cwd,
                                        capture_output=True, env={**os.environ, "PYTHONUTF8": "1"})
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))

    def test_import_has_no_output_or_file_access(self):
        with tempfile.TemporaryDirectory() as cwd, contextlib.redirect_stdout(io.StringIO()) as output:
            old = Path.cwd()
            try:
                os.chdir(cwd)
                for chapter in (1, 2, 3):
                    for path in sorted((ROOT / f"chapter_{chapter}").glob("*.py")):
                        load(path)
                self.assertEqual(list(Path(cwd).iterdir()), [])
            finally:
                os.chdir(old)
            self.assertEqual(output.getvalue(), "")

    def test_wikipedia_stream_and_missing_title(self):
        module = load(ROOT / "chapter_3/wiki_article.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wiki.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                for title in ["日本", "イギリス"]:
                    output.write(json.dumps({"title": title, "text": "本文:" + title}) + "\n")
            self.assertEqual(module.load_article(path=path), "本文:イギリス")
            with self.assertRaises(ValueError):
                module.load_article("missing", path)

    def test_markup_cleaning(self):
        module = load(ROOT / "chapter_4/wiki_corpus.py")
        self.assertEqual(module.remove_markup("'''bold''' [[target|label]]<ref>x</ref>"), "bold label")

    def test_country_extraction_deduplicates_and_filters(self):
        module = load(ROOT / "chapter_6/country_vectors.py")
        class Vectors(dict):
            @property
            def key_to_index(self):
                return self
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.txt"
            path.write_text(": capital-world\nTokyo Japan Paris France\nParis France Nowhere Missing\n: gram1\na b c d\n", encoding="utf-8")
            names, vectors = module.load_country_vectors(path, Vectors(Japan=[1,0], France=[0,1]))
        self.assertEqual(names, ["France", "Japan"])
        self.assertEqual(vectors, [[0,1], [1,0]])

    def test_bow_counts_and_training_only_vocabulary(self):
        from sklearn.feature_extraction import DictVectorizer
        module = load(ROOT / "chapter_7/sentiment_data.py")
        self.assertEqual(module.text_to_feature("too loud too"), {"too":2, "loud":1})
        vectorizer = DictVectorizer()
        vectorizer.fit_transform([module.text_to_feature("good movie")])
        dev = vectorizer.transform([module.text_to_feature("unseen good")])
        self.assertNotIn("unseen", vectorizer.vocabulary_)
        self.assertEqual(dev.sum(), 1)

    def test_sst2_zip_loader(self):
        module = load(ROOT / "chapter_7/sentiment_data.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sst.zip"
            with zipfile.ZipFile(path, "w") as output:
                output.writestr("SST-2/train.tsv", "sentence\tlabel\ngood film\t1\n")
                output.writestr("SST-2/dev.tsv", "sentence\tlabel\nbad film\t0\n")
            train, dev = module.load_splits(path)
            self.assertEqual(module.df_to_examples(train)[0]["label"], 1)
            self.assertEqual(dev.iloc[0]["sentence"], "bad film")
            with zipfile.ZipFile(path) as archive, self.assertRaises(FileNotFoundError):
                module.find_tsv_in_zip(archive, "test.tsv")

    def test_metrics_and_no_positive_prediction(self):
        module = load(ROOT / "chapter_7/67.py")
        scores = module.evaluate([0,0,1,1], [0,1,0,1])
        self.assertEqual(set(scores.values()), {0.5})
        self.assertEqual(module.evaluate([0,1], [0,0])["F1スコア"], 0)

    def test_demo_command_outside_repo(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, str(ROOT / "chapter_7/67.py"), "--demo"],
                                    cwd=cwd, capture_output=True, env={**os.environ, "PYTHONUTF8": "1"})
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertIn(b"Synthetic demo", result.stdout)

    def test_dataset_hash_and_overwrite_protection(self):
        module = load(ROOT / "scripts/prepare_data.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "download"
            source.write_bytes(b"fixture\n")
            module.MANIFEST["fixture"] = {"sha256":module.digest(source), "destinations":["chapter/data.txt"]}
            with contextlib.redirect_stdout(io.StringIO()):
                module.prepare("fixture", source, root)
            destination = root / "chapter/data.txt"
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            destination.write_bytes(b"different")
            with self.assertRaises(FileExistsError):
                module.prepare("fixture", source, root)
            source.write_bytes(b"wrong hash")
            with self.assertRaises(ValueError):
                module.prepare("fixture", source, root)

    def test_api_client_requires_key_only_on_use(self):
        from unittest.mock import patch
        module = load(ROOT / "chapter_5/gemini_client.py")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                module.get_client()


if __name__ == "__main__":
    unittest.main()
