from pathlib import Path
import zipfile
import pandas as pd


def find_tsv_in_zip(zip_file, target_name):
    """
    zip内から train.tsv / dev.tsv を探して読み込む
    """
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name
    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")


def count_labels(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        train_path = find_tsv_in_zip(z, "train.tsv")
        dev_path = find_tsv_in_zip(z, "dev.tsv")

        with z.open(train_path) as f:
            train_df = pd.read_csv(f, sep="\t")

        with z.open(dev_path) as f:
            dev_df = pd.read_csv(f, sep="\t")

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

#出力例
