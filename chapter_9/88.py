from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    base_dir = Path(__file__).parent
    model_dir = base_dir / "bert_sst2_finetuned_best"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} が見つかりません。先に 87.py を実行してモデルを保存してください。"
        )

    sentences = [
        "The movie was full of incomprehensibilities.",
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish.",
    ]

    label_names = {
        0: "negative",
        1: "positive",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用デバイス:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    print()
    print("===== 極性分析結果 =====")

    for sentence, pred, probs in zip(sentences, predictions, probabilities):
        pred_label = pred.item()
        negative_prob = probs[0].item()
        positive_prob = probs[1].item()

        print("文:", sentence)
        print("予測ラベル:", pred_label, label_names[pred_label])
        print(f"negative確率: {negative_prob:.6f}")
        print(f"positive確率: {positive_prob:.6f}")
        print("-" * 50)


if __name__ == "__main__":
    main()

#出力結果
'''
===== 極性分析結果 =====
文: The movie was full of incomprehensibilities.
予測ラベル: 0 negative
negative確率: 0.995988
positive確率: 0.004012
--------------------------------------------------
文: The movie was full of fun.
予測ラベル: 1 positive
negative確率: 0.000249
positive確率: 0.999751
--------------------------------------------------
文: The movie was full of excitement.
予測ラベル: 1 positive
negative確率: 0.000693
positive確率: 0.999307
--------------------------------------------------
文: The movie was full of crap.
予測ラベル: 0 negative
negative確率: 0.999416
positive確率: 0.000584
--------------------------------------------------
文: The movie was full of rubbish.
予測ラベル: 0 negative
negative確率: 0.999371
positive確率: 0.000629
--------------------------------------------------
'''