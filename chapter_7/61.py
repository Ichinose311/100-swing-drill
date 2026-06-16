from pathlib import Path
import zipfile
import pandas as pd
from collections import Counter


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
    スペース区切りで単語を分割し、出現頻度を辞書にする
    例: "too loud , too goofy"
    -> {'too': 2, 'loud': 1, ',': 1, 'goofy': 1}
    """
    tokens = text.split()
    return dict(Counter(tokens))


def df_to_examples(df):
    """
    DataFrameを、各事例の辞書オブジェクトのリストに変換する
    """
    examples = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = str(row["label"])

        example = {
            "text": text,
            "label": label,
            "feature": text_to_feature(text),
        }

        examples.append(example)

    return examples


def main():
    base_dir = Path(__file__).parent
    zip_path = base_dir / "SST-2.zip"

    with zipfile.ZipFile(zip_path, "r") as z:
        train_path = find_tsv_in_zip(z, "train.tsv")
        dev_path = find_tsv_in_zip(z, "dev.tsv")

        with z.open(train_path) as f:
            train_df = pd.read_csv(f, sep="\t")

        with z.open(dev_path) as f:
            dev_df = pd.read_csv(f, sep="\t")

    train_data = df_to_examples(train_df)
    dev_data = df_to_examples(dev_df)

    print("学習データの事例数:", len(train_data))
    print("検証データの事例数:", len(dev_data))

    print("\n===== 学習データの最初の事例 =====")
    print(train_data[0])


if __name__ == "__main__":
    main()

#出力例
'''
学習データの事例数: 67349
検証データの事例数: 872

===== 学習データの最初の事例 =====
{'text': 'hide new secretions from the parental units ', 'label': '0', 'feature': {'hide': 1, 'new': 1, 'secretions': 1, 'from': 1, 'the': 1, 'parental': 1, 'units': 1}}
'''