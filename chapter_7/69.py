from pathlib import Path
import zipfile
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

    with zipfile.ZipFile(zip_path, "r") as z:
        train_path = find_tsv_in_zip(z, "train.tsv")
        dev_path = find_tsv_in_zip(z, "dev.tsv")

        with z.open(train_path) as f:
            train_df = pd.read_csv(f, sep="\t")

        with z.open(dev_path) as f:
            dev_df = pd.read_csv(f, sep="\t")

    train_data = df_to_examples(train_df)
    dev_data = df_to_examples(dev_df)

    train_features = [example["feature"] for example in train_data]
    train_labels = [example["label"] for example in train_data]

    dev_features = [example["feature"] for example in dev_data]
    dev_labels = [example["label"] for example in dev_data]

    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform(train_features)
    X_dev = vectorizer.transform(dev_features)

    print("データ読み込み完了")
    print("学習データ数:", X_train.shape[0])
    print("検証データ数:", X_dev.shape[0])
    print("特徴数:", X_train.shape[1])

    # scikit-learn の LogisticRegression では、
    # C が小さいほど正則化が強く、C が大きいほど正則化が弱い
    C_values = [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    ]

    results = []

    for C in C_values:
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear",
        )

        model.fit(X_train, train_labels)

        dev_predictions = model.predict(X_dev)
        dev_accuracy = accuracy_score(dev_labels, dev_predictions)

        train_predictions = model.predict(X_train)
        train_accuracy = accuracy_score(train_labels, train_predictions)

        results.append({
            "C": C,
            "train_accuracy": train_accuracy,
            "dev_accuracy": dev_accuracy,
        })

        print(
            f"C={C:7.3f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"dev_accuracy={dev_accuracy:.4f}"
        )

    results_df = pd.DataFrame(results)

    print("\n===== 実験結果 =====")
    print(results_df)

    best_row = results_df.loc[results_df["dev_accuracy"].idxmax()]

    print("\n===== 検証データで最も正解率が高かった設定 =====")
    print(f"C: {best_row['C']}")
    print(f"検証データ正解率: {best_row['dev_accuracy']:.4f}")

    # グラフを作成
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["C"],
        results_df["dev_accuracy"],
        marker="o",
        label="Dev accuracy",
    )

    plt.xscale("log")
    plt.xlabel("Regularization parameter C")
    plt.ylabel("Accuracy")
    plt.title("Effect of regularization parameter C on dev accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = base_dir / "regularization_accuracy.png"
    plt.savefig(output_path)

    print(f"\nグラフを保存しました: {output_path}")


if __name__ == "__main__":
    main()

#出力例
'''
データ読み込み完了
学習データ数: 67349
検証データ数: 872
特徴数: 14816
C=  0.001 train_accuracy=0.6493 dev_accuracy=0.6846
C=  0.003 train_accuracy=0.6963 dev_accuracy=0.7110
C=  0.010 train_accuracy=0.7673 dev_accuracy=0.7557
C=  0.030 train_accuracy=0.8295 dev_accuracy=0.7947
C=  0.100 train_accuracy=0.8834 dev_accuracy=0.8005
C=  0.300 train_accuracy=0.9195 dev_accuracy=0.8062
C=  1.000 train_accuracy=0.9428 dev_accuracy=0.8108
C=  3.000 train_accuracy=0.9552 dev_accuracy=0.8211
C= 10.000 train_accuracy=0.9636 dev_accuracy=0.8257
C= 30.000 train_accuracy=0.9672 dev_accuracy=0.8096
C=100.000 train_accuracy=0.9693 dev_accuracy=0.7947

===== 実験結果 =====
          C  train_accuracy  dev_accuracy
0     0.001        0.649290      0.684633
1     0.003        0.696328      0.711009
2     0.010        0.767302      0.755734
3     0.030        0.829485      0.794725
4     0.100        0.883443      0.800459
5     0.300        0.919524      0.806193
6     1.000        0.942820      0.810780
7     3.000        0.955203      0.821101
8    10.000        0.963622      0.825688
9    30.000        0.967216      0.809633
10  100.000        0.969294      0.794725

===== 検証データで最も正解率が高かった設定 =====
C: 10.0
検証データ正解率: 0.8257

グラフを保存しました: /home/ichinose/projects/100-swing-drill/chapter_7/regularization_accuracy.png
'''