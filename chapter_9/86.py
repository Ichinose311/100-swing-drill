from pathlib import Path
import csv
import zipfile

import torch
from transformers import AutoTokenizer


def find_file_in_zip(zip_file, target_name):
    """
    zip内から train.tsv / dev.tsv を探す
    """
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name

    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")


def load_sst2_tsv_from_zip(zip_path, target_name):
    """
    SST-2.zip から train.tsv / dev.tsv を読み込み、
    text, label を持つ辞書のリストに変換する
    """
    examples = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_path = find_file_in_zip(zf, target_name)

        with zf.open(tsv_path, "r") as f:
            lines = (line.decode("utf-8") for line in f)
            reader = csv.DictReader(lines, delimiter="\t")

            for row in reader:
                example = {
                    "text": row["sentence"],
                    "label": int(row["label"]),
                }
                examples.append(example)

    return examples


def main():
    base_dir = Path(__file__).parent
    sst_zip_path = base_dir / "SST-2.zip"

    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased"
    )

    # 訓練データを読み込む
    train_examples = load_sst2_tsv_from_zip(
        sst_zip_path,
        "train.tsv"
    )

    # 冒頭4事例をミニバッチにする
    mini_batch_examples = train_examples[:4]

    texts = [example["text"] for example in mini_batch_examples]
    labels = [example["label"] for example in mini_batch_examples]

    # パディングして長さを揃える
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # ラベルもTensorにする
    label_tensor = torch.tensor(labels)

    print("===== 入力文 =====")
    for i, text in enumerate(texts, start=1):
        print(f"{i}: {text}")
    print()

    print("===== ラベル =====")
    print(label_tensor)
    print()

    print("===== input_ids =====")
    print(encoded["input_ids"])
    print("shape:", encoded["input_ids"].shape)
    print()

    print("===== attention_mask =====")
    print(encoded["attention_mask"])
    print("shape:", encoded["attention_mask"].shape)
    print()

    print("===== tokens =====")
    for i, input_ids in enumerate(encoded["input_ids"], start=1):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(f"{i}:", tokens)


if __name__ == "__main__":
    main()

#出力結果
