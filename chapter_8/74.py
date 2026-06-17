from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class BoWLogisticRegression(nn.Module):
    """
    単語埋め込みの平均ベクトルを使うBag of Wordsモデル
    """

    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # 事前学習済み単語埋め込み
        # 評価時も単語埋め込みは固定
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0
        )

        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        # <PAD>を平均に含めないためのマスク
        mask = (input_ids != 0).unsqueeze(-1)

        embeddings = embeddings * mask

        summed = embeddings.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)

        bow_vector = summed / lengths

        logit = self.linear(bow_vector)

        return logit


def collate_fn(batch):
    """
    長さの異なるinput_idsを<PAD>でpaddingしてミニバッチ化する
    """
    texts = [example["text"] for example in batch]

    labels = torch.stack([
        example["label"] for example in batch
    ])

    input_ids = [example["input_ids"] for example in batch]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    return {
        "text": texts,
        "label": labels,
        "input_ids": input_ids,
    }


def evaluate(model, dataloader, loss_fn, device):
    """
    開発セットの損失値と正解率を計算する
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = input_ids.size(0)

            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total_examples += batch_size

    avg_loss = total_loss / total_examples
    accuracy = correct / total_examples

    return avg_loss, accuracy, correct, total_examples


def main():
    base_dir = Path(__file__).parent

    embedding_path = base_dir / "embedding_matrix.pt"
    dev_path = base_dir / "sst_dev_examples.pt"
    model_path = base_dir / "bow_logistic_regression.pt"

    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用デバイス: {device}")

    # 単語埋め込み行列を読み込む
    embedding_matrix = torch.load(
        embedding_path,
        map_location="cpu"
    )

    # 開発セットを読み込む
    dev_examples = torch.load(
        dev_path,
        map_location="cpu"
    )

    dev_loader = DataLoader(
        dev_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # モデルを作成
    model = BoWLogisticRegression(embedding_matrix)

    # 問題73で保存した学習済み重みを読み込む
    state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    dev_loss, dev_accuracy, correct, total = evaluate(
        model,
        dev_loader,
        loss_fn,
        device
    )

    print()
    print("開発セットでの評価結果")
    print(f"事例数: {total}")
    print(f"正解数: {correct}")
    print(f"dev_loss: {dev_loss:.4f}")
    print(f"dev_accuracy: {dev_accuracy:.4f}")
    print(f"dev_accuracy(%): {dev_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

#出力結果
'''
使用デバイス: cpu

開発セットでの評価結果
事例数: 872
正解数: 687
dev_loss: 0.4582
dev_accuracy: 0.7878
dev_accuracy(%): 78.78%
'''