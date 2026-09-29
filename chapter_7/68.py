from sentiment_data import df_to_examples, load_splits
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def main():
    base_dir = Path(__file__).parent
    zip_path = base_dir / "SST-2.zip"

    # train.tsv / dev.tsv を読み込む
    train_df, dev_df = load_splits(zip_path)

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
