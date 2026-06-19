import itertools

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(last_hidden_state, attention_mask, special_tokens_mask):
    """
    最終層の各トークンベクトルの平均を取る。
    ただし、[CLS], [SEP], [PAD] は平均に含めない。
    """

    # attention_mask:
    #   通常トークンと特殊トークンは1、[PAD]は0
    #
    # special_tokens_mask:
    #   [CLS], [SEP], [PAD]などの特殊トークンは1、それ以外は0
    #
    # よって、通常トークンだけを1にする
    valid_mask = attention_mask * (1 - special_tokens_mask)

    # [バッチサイズ, トークン数] -> [バッチサイズ, トークン数, 1]
    valid_mask = valid_mask.unsqueeze(-1)

    # 特殊トークン以外のベクトルだけ残す
    masked_embeddings = last_hidden_state * valid_mask

    # トークン方向に足し合わせる
    summed_embeddings = masked_embeddings.sum(dim=1)

    # 各文の有効トークン数
    token_counts = valid_mask.sum(dim=1)

    # 0除算を避ける
    token_counts = token_counts.clamp(min=1e-9)

    # 平均ベクトル
    mean_embeddings = summed_embeddings / token_counts

    return mean_embeddings


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

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )

    # 最終層の全トークン埋め込み
    last_hidden_state = outputs.last_hidden_state

    # 平均による文ベクトル
    sentence_embeddings = mean_pooling(
        last_hidden_state,
        inputs["attention_mask"],
        inputs["special_tokens_mask"],
    )

    print("平均文ベクトルの形:", sentence_embeddings.shape)
    print()

    # 全ての2文組み合わせについてコサイン類似度を計算
    for i, j in itertools.combinations(range(len(sentences)), 2):
        similarity = F.cosine_similarity(
            sentence_embeddings[i].unsqueeze(0),
            sentence_embeddings[j].unsqueeze(0),
        ).item()

        print(f"文{i + 1} と 文{j + 1}")
        print(f"文{i + 1}: {sentences[i]}")
        print(f"文{j + 1}: {sentences[j]}")
        print(f"コサイン類似度: {similarity:.6f}")
        print("-" * 50)


if __name__ == "__main__":
    main()

#出力結果
