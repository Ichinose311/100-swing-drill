from pathlib import Path
import zipfile
from collections import Counter

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def evaluate(y_true, y_pred):
    """
    正解率・適合率・再現率・F1スコアを計算する
    ここでは label=1、つまりポジティブを正例として評価する
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    return {
        "正解率": accuracy,
        "適合率": precision,
        "再現率": recall,
        "F1スコア": f1,
    }


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

    # 学習データ・検証データで予測
    train_predictions = model.predict(X_train)
    dev_predictions = model.predict(X_dev)

    # 評価指標を計算
    train_scores = evaluate(train_labels, train_predictions)
    dev_scores = evaluate(dev_labels, dev_predictions)

    # 表形式で表示
    scores_df = pd.DataFrame(
        [train_scores, dev_scores],
        index=["学習データ", "検証データ"]
    )

    print("\n===== 評価結果 =====")
    print(scores_df)

    # 小数第4位まで見やすく表示
    print("\n===== 評価結果（小数第4位まで） =====")
    print(scores_df.round(4))


if __name__ == "__main__":
    main()

#出力例
'''
学習完了
学習データ数: 67349
検証データ数: 872
特徴数: 14816

===== 評価結果 =====
            正解率       適合率       再現率     F1スコア
学習データ  0.942508  0.943160  0.954457  0.948775
検証データ  0.810780  0.798715  0.840090  0.818880

===== 評価結果（小数第4位まで） =====
          正解率     適合率     再現率   F1スコア
学習データ  0.9425  0.9432  0.9545  0.9488
検証データ  0.8108  0.7987  0.8401  0.8189
'''