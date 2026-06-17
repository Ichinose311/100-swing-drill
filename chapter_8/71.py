from pathlib import Path
import zipfile
import csv
import pickle
import io

import torch


def find_tsv_in_zip(zip_path, target_name):
    """
    zipファイル内から train.tsv / dev.tsv を探す
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        candidates = [
            name for name in names
            if name.endswith(target_name)
            and "__MACOSX" not in name
        ]

        if not candidates:
            return None

        # SST-2 を含むパスを優先する
        sst_candidates = [
            name for name in candidates
            if "SST-2" in name or "SST" in name
        ]

        if sst_candidates:
            return sst_candidates[0]

        return candidates[0]


def find_dataset_files(base_dir):
    """
    GLUE_baselines-master.zip / SST-2.zip の中から
    train.tsv と dev.tsv を探す
    """
    zip_paths = [
        base_dir / "GLUE_baselines-master.zip",
        base_dir / "SST-2.zip",
    ]

    train_info = None
    dev_info = None

    for zip_path in zip_paths:
        if not zip_path.exists():
            continue

        train_name = find_tsv_in_zip(zip_path, "train.tsv")
        dev_name = find_tsv_in_zip(zip_path, "dev.tsv")

        if train_name is not None and train_info is None:
            train_info = (zip_path, train_name)

        if dev_name is not None and dev_info is None:
            dev_info = (zip_path, dev_name)

    if train_info is None:
        raise FileNotFoundError("train.tsv が zip 内に見つかりません")

    if dev_info is None:
        raise FileNotFoundError("dev.tsv が zip 内に見つかりません")

    return train_info, dev_info


def text_to_input_ids(text, token_to_id):
    """
    テキストをトークンID列に変換する

    単語埋め込みの語彙に含まれていない単語は無視する
    """
    input_ids = []

    for token in text.split():
        if token in token_to_id:
            input_ids.append(token_to_id[token])

    return input_ids


def load_sst_tsv(zip_path, tsv_name, token_to_id):
    """
    SST-2のtsvを読み込み、各事例を辞書オブジェクトに変換する
    """
    examples = []
    total_count = 0
    removed_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(tsv_name, "r") as f:
            text_file = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.DictReader(text_file, delimiter="\t")

            for row in reader:
                total_count += 1

                text = row["sentence"]
                label = int(row["label"])

                input_ids = text_to_input_ids(text, token_to_id)

                # 空のトークン列になる事例は削除
                if len(input_ids) == 0:
                    removed_count += 1
                    continue

                example = {
                    "text": text,
                    "label": torch.tensor([float(label)], dtype=torch.float32),
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                }

                examples.append(example)

    return examples, total_count, removed_count


def main():
    base_dir = Path(__file__).parent

    token_to_id_path = base_dir / "token_to_id.pkl"

    train_output_path = base_dir / "sst_train_examples.pt"
    dev_output_path = base_dir / "sst_dev_examples.pt"

    # 70.pyで作成した単語ID辞書を読み込む
    with open(token_to_id_path, "rb") as f:
        token_to_id = pickle.load(f)

    # train.tsv / dev.tsv を探す
    train_info, dev_info = find_dataset_files(base_dir)

    train_zip_path, train_tsv_name = train_info
    dev_zip_path, dev_tsv_name = dev_info

    print("使用するファイル")
    print(f"train: {train_zip_path.name} / {train_tsv_name}")
    print(f"dev  : {dev_zip_path.name} / {dev_tsv_name}")
    print()

    # データセットを読み込む
    train_examples, train_total, train_removed = load_sst_tsv(
        train_zip_path,
        train_tsv_name,
        token_to_id
    )

    dev_examples, dev_total, dev_removed = load_sst_tsv(
        dev_zip_path,
        dev_tsv_name,
        token_to_id
    )

    # 保存
    torch.save(train_examples, train_output_path)
    torch.save(dev_examples, dev_output_path)

    print("データセットを読み込みました")
    print()
    print("訓練セット")
    print(f"元の事例数: {train_total}")
    print(f"削除した事例数: {train_removed}")
    print(f"残った事例数: {len(train_examples)}")

    print()
    print("開発セット")
    print(f"元の事例数: {dev_total}")
    print(f"削除した事例数: {dev_removed}")
    print(f"残った事例数: {len(dev_examples)}")

    print()
    print("事例の確認")
    print(train_examples[0])

    print()
    print(f"保存しました: {train_output_path}")
    print(f"保存しました: {dev_output_path}")


if __name__ == "__main__":
    main()

#出力結果
'''
使用するファイル
train: SST-2.zip / SST-2/train.tsv
dev  : SST-2.zip / SST-2/dev.tsv

データセットを読み込みました

訓練セット
元の事例数: 67349
削除した事例数: 1512
残った事例数: 65837

開発セット
元の事例数: 872
削除した事例数: 0
残った事例数: 872

事例の確認
{'text': 'hide new secretions from the parental units ', 'label': tensor([0.]), 'input_ids': tensor([ 5785,    66,    18,    12, 15095,  1594])}

保存しました: /home/ichinose/projects/100-swing-drill/chapter_8/sst_train_examples.pt
保存しました: /home/ichinose/projects/100-swing-drill/chapter_8/sst_dev_examples.pt
'''