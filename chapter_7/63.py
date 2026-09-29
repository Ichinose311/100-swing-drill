from sentiment_data import df_to_examples, load_splits
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

    dev_features = [example["feature"] for example in dev_data]
    dev_labels = [example["label"] for example in dev_data]

    # 辞書形式のBoWを、機械学習モデルに入力できる行列に変換
    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform(train_features)
    X_dev = vectorizer.transform(dev_features)

    #y_train = train_labels
    #y_dev = dev_labels

    # ロジスティック回帰モデルを学習する
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    print("学習完了")
    print("学習データ数:", X_train.shape[0])
    print("特徴数:", X_train.shape[1])

    # 検証データの先頭の事例を予測
    first_dev_example = dev_data[0]
    first_dev_vector = X_dev[0]

    predicted_label = model.predict(first_dev_vector)[0]
    true_label = first_dev_example["label"]

    print("\n===== 検証データの先頭の事例 =====")
    print("text:", first_dev_example["text"])
    print("正解ラベル:", true_label)
    print("予測ラベル:", predicted_label)

    if predicted_label == true_label:
        print("結果: 一致しています")
    else:
        print("結果: 一致していません")

if __name__ == "__main__":
    main()
