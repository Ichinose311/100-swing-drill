from pathlib import Path
import zipfile
from collections import Counter

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


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


def predict_sentiment(text, model, vectorizer):
    feature = text_to_feature(text)
    X = vectorizer.transform([feature])

    predicted_label = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return predicted_label, probabilities


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

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    print("学習完了")
    print("学習データ数:", X_train.shape[0])
    print("特徴数:", X_train.shape[1])

    input_text = "the worst movie I 've ever seen"

    predicted_label, probabilities = predict_sentiment(
        input_text,
        model,
        vectorizer
    )

    label_name = {
        0: "ネガティブ",
        1: "ポジティブ",
    }

    print("\n===== 入力テキストの予測 =====")
    print("text:", input_text)
    print("予測ラベル:", predicted_label, label_name[predicted_label])

    print("\n===== 条件付き確率 =====")
    for label, prob in zip(model.classes_, probabilities):
        print(f"P(label={label} | text) = {prob:.6f}  ({label_name[label]})")


if __name__ == "__main__":
    main()

# 出力例
