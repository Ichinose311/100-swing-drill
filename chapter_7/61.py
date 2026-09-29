from sentiment_data import load_splits, text_to_feature
from pathlib import Path
import pandas as pd


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

    train_df, dev_df = load_splits(zip_path)

    train_data = df_to_examples(train_df)
    dev_data = df_to_examples(dev_df)

    print("学習データの事例数:", len(train_data))
    print("検証データの事例数:", len(dev_data))

    print("\n===== 学習データの最初の事例 =====")
    print(train_data[0])


if __name__ == "__main__":
    main()
