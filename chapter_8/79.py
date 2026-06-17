from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class BoWMLPClassifier(nn.Module):
    """
    平均単語ベクトルにMLPを通す分類モデル
    """

    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # 79番では単語埋め込みも更新する
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0
        )

        # 平均単語ベクトルを多層ニューラルネットワークに通す
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        # PADでない部分だけ1
        mask = (input_ids != 0).unsqueeze(-1)

        # PAD部分のベクトルを0にする
        embeddings = embeddings * mask

        # 単語ベクトルの和
        summed = embeddings.sum(dim=1)

        # PADを除いたトークン数
        lengths = mask.sum(dim=1).clamp(min=1)

        # 平均単語ベクトル
        bow_vector = summed / lengths

        # MLPで分類
        logits = self.classifier(bow_vector)

        return logits


def collate(batch):
    """
    75番のcollate関数
    長い系列順に並び替えて、0でpaddingする
    """

    batch = sorted(
        batch,
        key=lambda example: len(example["input_ids"]),
        reverse=True
    )

    input_ids_list = [
        example["input_ids"]
        for example in batch
    ]

    labels = torch.stack([
        example["label"]
        for example in batch
    ])

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids,
        "label": labels,
    }


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_size = input_ids.size(0)

        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total_examples += batch_size

    avg_loss = total_loss / total_examples
    accuracy = correct / total_examples

    return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, device):
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

    return avg_loss, accuracy


def main():
    base_dir = Path(__file__).parent

    embedding_path = base_dir / "embedding_matrix.pt"
    train_path = base_dir / "sst_train_examples.pt"
    dev_path = base_dir / "sst_dev_examples.pt"

    model_output_path = base_dir / "bow_mlp_classifier.pt"

    batch_size = 64
    num_epochs = 20

    # 単語埋め込みも更新するので、少し小さめの学習率にする
    learning_rate = 1e-4

    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが使えません。GPUサーバ上で実行してください。")

    device = torch.device("cuda")

    print(f"使用デバイス: {device}")
    print(f"GPU名: {torch.cuda.get_device_name(0)}")

    embedding_matrix = torch.load(
        str(embedding_path),
        map_location="cpu"
    )

    train_examples = torch.load(
        str(train_path),
        map_location="cpu"
    )

    dev_examples = torch.load(
        str(dev_path),
        map_location="cpu"
    )

    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True
    )

    model = BoWMLPClassifier(embedding_matrix)
    model.to(device)

    print()
    print("モデル構造")
    print(model)

    print()
    print("学習対象パラメータ")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    print()
    print("MLPモデルの学習開始")
    print(f"batch_size: {batch_size}")
    print(f"learning_rate: {learning_rate}")

    best_dev_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        dev_loss, dev_accuracy = evaluate(
            model,
            dev_loader,
            loss_fn,
            device
        )

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            torch.save(model.state_dict(), str(model_output_path))

        print(
            f"epoch: {epoch:02d} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_accuracy:.4f} | "
            f"dev_loss: {dev_loss:.4f} | "
            f"dev_acc: {dev_accuracy:.4f}"
        )

    print()
    print("学習完了")
    print(f"開発セットでの最高正解率: {best_dev_accuracy:.4f}")
    print(f"開発セットでの最高正解率(%): {best_dev_accuracy * 100:.2f}%")
    print(f"保存しました: {model_output_path}")


if __name__ == "__main__":
    main()

#出力結果
