from pathlib import Path
import zipfile
from collections import Counter

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def find_tsv_in_zip(zip_file, target_name):
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name
    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")


def text_to_feature(text):
    tokens = text.split()
    return dict(Counter(tokens))


def df_to_examples(df):
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


def main():
    base_dir = Path(__file__).parent
    zip_path = base_dir / "SST-2.zip"

    # train.tsv / dev.tsv を読み込む
    with zipfile.ZipFile(zip_path, "r") as z:
        train_path = find_tsv_in_zip(z, "train.tsv")
        dev_path = find_tsv_in_zip(z, "dev.tsv")

        with z.open(train_path) as f:
            train_df = pd.read_csv(f, sep="\t")

        with z.open(dev_path) as f:
            dev_df = pd.read_csv(f, sep="\t")

    # 61番と同じ形式のデータに変換
    train_data = df_to_examples(train_df)
    dev_data = df_to_examples(dev_df)

    # 特徴ベクトルとラベルを取り出す
    train_features = [example["feature"] for example in train_data]
    train_labels = [example["label"] for example in train_data]

    dev_features = [example["feature"] for example in dev_data]
    dev_labels = [example["label"] for example in dev_data]

    # 辞書形式のBoWを、機械学習モデルに入力できる行列に変換
    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform(train_features)
    X_dev = vectorizer.transform(dev_features)

    y_train = train_labels
    y_dev = dev_labels

    # ロジスティック回帰モデルを学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("学習完了")

    print("学習データ数:", X_train.shape[0])
    print("特徴数:", X_train.shape[1])

    # 確認用：検証データで正解率を見る
    y_pred = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred)

    print("検証データの正解率:", accuracy)


if __name__ == "__main__":
    main()

#出力例
'''
学習完了
学習データ数: 67349
特徴数: 14816
検証データの正解率: 0.8107798165137615
'''