from pathlib import Path

import torch
import torch.nn as nn


class BoWLogisticRegression(nn.Module):
    """
    単語埋め込みの平均ベクトルを使うBag of Wordsモデル

    入力:
        input_ids: 単語ID列
            shape: (seq_len,) または (batch_size, seq_len)

    出力:
        logit: ポジティブである度合いを表すスコア
            shape: (1,) または (batch_size, 1)
    """

    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # 事前学習済み単語埋め込み
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0
        )

        # 平均ベクトルを1次元のスコアに変換する線形層
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids):
        # 1文だけの場合: (seq_len,) -> (1, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # input_ids: (batch_size, seq_len)
        # embeddings: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)

        # <PAD>を平均に含めないためのマスク
        # mask: (batch_size, seq_len, 1)
        mask = (input_ids != 0).unsqueeze(-1)

        # <PAD>部分のベクトルを0にする
        embeddings = embeddings * mask

        # 文ごとに単語ベクトルを合計
        # summed: (batch_size, embedding_dim)
        summed = embeddings.sum(dim=1)

        # 文ごとの有効トークン数
        # lengths: (batch_size, 1)
        lengths = mask.sum(dim=1).clamp(min=1)

        # 平均ベクトル
        # bow_vector: (batch_size, embedding_dim)
        bow_vector = summed / lengths

        # ロジスティック回帰の線形部分
        # logit: (batch_size, 1)
        logit = self.linear(bow_vector)

        return logit


def main():
    base_dir = Path(__file__).parent

    embedding_path = base_dir / "embedding_matrix.pt"
    train_path = base_dir / "sst_train_examples.pt"

    # 70.pyで作成した単語埋め込み行列
    embedding_matrix = torch.load(embedding_path)

    # 71.pyで作成した訓練データ
    train_examples = torch.load(train_path)

    # モデルを作成
    model = BoWLogisticRegression(embedding_matrix)

    print(model)
    print()

    # 1事例で動作確認
    example = train_examples[0]

    text = example["text"]
    label = example["label"]
    input_ids = example["input_ids"]

    logit = model(input_ids)
    probability = torch.sigmoid(logit)

    print("入力テキスト")
    print(text)

    print()
    print("正解ラベル")
    print(label)

    print()
    print("input_ids")
    print(input_ids)

    print()
    print("モデル出力 logit")
    print(logit)

    print()
    print("ポジティブである確率")
    print(probability)


if __name__ == "__main__":
    main()

#出力結果
