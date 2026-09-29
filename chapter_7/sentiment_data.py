"""Chapter 7: shared SST-2 loading and count-based features."""
from collections import Counter
import zipfile
import pandas as pd

def find_tsv_in_zip(zip_file, target_name):
    """
    zip内から train.tsv / dev.tsv を探す
    """
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name

    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")

def text_to_feature(text):
    """
    テキストをBoW特徴量に変換する
    """
    tokens = text.split()
    return dict(Counter(tokens))

def df_to_examples(df):
    """
    DataFrameを辞書オブジェクトのリストに変換する
    """
    examples = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = int(row["label"])

        example = {
            "text": text,
            "label": label,
            "feature": text_to_feature(text),
        }

        examples.append(example)

    return examples

def load_splits(zip_path):
    """Load training/development splits without learning a vocabulary on dev."""
    with zipfile.ZipFile(zip_path) as archive:
        frames = []
        for split in ("train.tsv", "dev.tsv"):
            with archive.open(find_tsv_in_zip(archive, split)) as stream:
                frames.append(pd.read_csv(stream, sep="\t"))
    return tuple(frames)
