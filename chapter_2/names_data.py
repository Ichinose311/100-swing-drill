"""Shared location and loader for the chapter 2 input."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "popular-names.txt"

def load_names():
    if not DATA_PATH.is_file():
        raise FileNotFoundError("Run: uv run python scripts/prepare_data.py popular-names")
    return pd.read_csv(DATA_PATH, sep="\t", header=None)
