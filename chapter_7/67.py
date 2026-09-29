import argparse
from sentiment_data import df_to_examples, load_splits
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(y_true, y_pred):
    """
    正解率・適合率・再現率・F1スコアを計算する
    ここでは label=1、つまりポジティブを正例として評価する
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "正解率": accuracy,
        "適合率": precision,
        "再現率": recall,
        "F1スコア": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="BoW sentiment classification and evaluation")
    parser.add_argument("--demo", action="store_true", help="Use tiny synthetic data; no download")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("SST-2.zip"))
    args = parser.parse_args()
    zip_path = args.dataset

    # train.tsv / dev.tsv を読み込む
    if args.demo:
        from demo_data import demo_frames
        train_df, dev_df = demo_frames()
        print("Synthetic demo: these scores are not SST-2 benchmark results.")
    else:
        train_df, dev_df = load_splits(zip_path)

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
