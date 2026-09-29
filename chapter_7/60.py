from sentiment_data import load_splits
from pathlib import Path


def count_labels(zip_path):
    train_df, dev_df = load_splits(zip_path)

    print("===== train.tsv =====")
    print(train_df["label"].value_counts().sort_index())

    print("\n===== dev.tsv =====")
    print(dev_df["label"].value_counts().sort_index())


def main():
    base_dir = Path(__file__).parent

    # SST-2.zip がこの Python ファイルと同じディレクトリにある想定
    zip_path = base_dir / "SST-2.zip"

    count_labels(zip_path)


if __name__ == "__main__":
    main()
