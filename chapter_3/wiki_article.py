"""Stream a named article from the official Wikipedia JSONL archive."""
import gzip
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "jawiki-country.json.gz"

def load_article(title="イギリス", path=DATA_PATH):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            article = json.loads(line)
            if article["title"] == title:
                return article["text"]
    raise ValueError(f"Article not found: {title}")
