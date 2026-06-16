from pathlib import Path
import zipfile
from collections import Counter

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


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
    例: "too loud , too goofy"
    -> {'too': 2, 'loud': 1, ',': 1, 'goofy': 1}
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

    # 辞書形式のBoWを、ロジスティック回帰に入力できる行列に変換
    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform(train_features)
    X_dev = vectorizer.transform(dev_features)

    # ロジスティック回帰モデルを学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    print("学習完了")
    print("学習データ数:", X_train.shape[0])
    print("検証データ数:", X_dev.shape[0])
    print("特徴数:", X_train.shape[1])

    # 検証データ全体を予測
    dev_predictions = model.predict(X_dev)

    # 混同行列を作成
    # labels=[0, 1] と指定することで、行・列の順番を 0, 1 に固定する
    cm = confusion_matrix(dev_labels, dev_predictions, labels=[0, 1])

    print("\n===== 混同行列 =====")
    print(cm)

    print("\n行: 正解ラベル, 列: 予測ラベル")
    print("        予測0   予測1")
    print(f"正解0   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"正解1   {cm[1][0]:5d}  {cm[1][1]:5d}")

    # pandasで見やすく表示
    cm_df = pd.DataFrame(
        cm,
        index=["正解: ネガティブ(0)", "正解: ポジティブ(1)"],
        columns=["予測: ネガティブ(0)", "予測: ポジティブ(1)"],
    )

    print("\n===== 混同行列（表形式） =====")
    print(cm_df)


if __name__ == "__main__":
    main()

#出力例
