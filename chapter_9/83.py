import itertools

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def main():
    model_name = "google-bert/bert-base-uncased"

    sentences = [
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()

    # 文をまとめてトークンIDに変換
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # BERTで文をエンコード
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state の形:
    # [文数, トークン数, 隠れ層次元]
    #
    # [CLS] は各文の先頭トークンなので、[:, 0, :] を取り出す
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    print("[CLS] ベクトルの形:", cls_embeddings.shape)
    print()

    # 全ての2文組み合わせについてコサイン類似度を計算
    for i, j in itertools.combinations(range(len(sentences)), 2):
        vec_i = cls_embeddings[i]
        vec_j = cls_embeddings[j]

        similarity = F.cosine_similarity(
            vec_i.unsqueeze(0),
            vec_j.unsqueeze(0)
        ).item()

        print(f"文{i + 1} と 文{j + 1}")
        print(f"文{i + 1}: {sentences[i]}")
        print(f"文{j + 1}: {sentences[j]}")
        print(f"コサイン類似度: {similarity:.6f}")
        print("-" * 50)


if __name__ == "__main__":
    main()

#出力結果
'''
[CLS] ベクトルの形: torch.Size([4, 768])

文1 と 文2
文1: The movie was full of fun.
文2: The movie was full of excitement.
コサイン類似度: 0.988061
--------------------------------------------------
文1 と 文3
文1: The movie was full of fun.
文3: The movie was full of crap.
コサイン類似度: 0.955766
--------------------------------------------------
文1 と 文4
文1: The movie was full of fun.
文4: The movie was full of rubbish.
コサイン類似度: 0.947533
--------------------------------------------------
文2 と 文3
文2: The movie was full of excitement.
文3: The movie was full of crap.
コサイン類似度: 0.954127
--------------------------------------------------
文2 と 文4
文2: The movie was full of excitement.
文4: The movie was full of rubbish.
コサイン類似度: 0.948663
--------------------------------------------------
文3 と 文4
文3: The movie was full of crap.
文4: The movie was full of rubbish.
コサイン類似度: 0.980693
--------------------------------------------------
'''