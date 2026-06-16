from pathlib import Path
import zipfile
from collections import Counter

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


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

    # 辞書形式のBoWを、ロジスティック回帰に入力できる行列に変換
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_features)

    # ロジスティック回帰モデルを学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    print("学習完了")
    print("学習データ数:", X_train.shape[0])
    print("特徴数:", X_train.shape[1])

    # 特徴量名を取得
    feature_names = vectorizer.get_feature_names_out()

    # ロジスティック回帰の重みを取得
    # 2値分類なので coef_[0] に各特徴量の重みが入っている
    weights = model.coef_[0]

    # 特徴量名と重みをDataFrameにまとめる
    weight_df = pd.DataFrame({
        "feature": feature_names,
        "weight": weights,
    })

    # 重みが高い特徴量トップ20
    top_positive = weight_df.sort_values("weight", ascending=False).head(20)

    # 重みが低い特徴量トップ20
    top_negative = weight_df.sort_values("weight", ascending=True).head(20)

    print("\n===== 重みの高い特徴量トップ20 =====")
    print(top_positive.to_string(index=False))

    print("\n===== 重みの低い特徴量トップ20 =====")
    print(top_negative.to_string(index=False))


if __name__ == "__main__":
    main()

#出力例
