from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class BoWLogisticRegression(nn.Module):
    """
    単語埋め込みの平均ベクトルを使うロジスティック回帰モデル
    """

    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # 単語埋め込み行列は固定する
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0
        )

        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)

        embeddings = self.embedding(input_ids)
        # embeddings: (batch_size, seq_len, embedding_dim)

        # PADでない部分だけ1にする
        mask = (input_ids != 0).unsqueeze(-1)
        # mask: (batch_size, seq_len, 1)

        # PAD部分のベクトルを0にする
        embeddings = embeddings * mask

        # 単語ベクトルの和
        summed = embeddings.sum(dim=1)
        # summed: (batch_size, embedding_dim)

        # PADを除いたトークン数
        lengths = mask.sum(dim=1).clamp(min=1)
        # lengths: (batch_size, 1)

        # 平均単語ベクトル
        bow_vector = summed / lengths

        # ロジットを出力
        logits = self.linear(bow_vector)

        return logits


def collate(batch):
    """
    75番のcollate関数

    1. トークン列の長い順に並び替える
    2. 最大系列長に合わせて0でpaddingする
    3. input_idsとlabelをまとめて返す
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


def evaluate(model, dataloader, loss_fn, device):
    """
    開発セットで損失値と正解率を計算する
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

    return avg_loss, accuracy


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    1エポック分学習する
    """
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


def main():
    base_dir = Path(__file__).parent

    embedding_path = base_dir / "embedding_matrix.pt"
    train_path = base_dir / "sst_train_examples.pt"
    dev_path = base_dir / "sst_dev_examples.pt"

    model_output_path = base_dir / "bow_logistic_regression_finetune.pt"

    batch_size = 64
    num_epochs = 20
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
        collate_fn=collate
    )

    dev_loader = DataLoader(
        dev_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate
    )

    model = BoWLogisticRegression(embedding_matrix)
    model.to(device)

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
    print("ミニバッチ学習開始")
    print(f"batch_size: {batch_size}")

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
'''
使用デバイス: cuda
GPU名: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

学習対象パラメータ
embedding.weight: requires_grad=True
linear.weight: requires_grad=True
linear.bias: requires_grad=True

ミニバッチ学習開始
batch_size: 64
epoch: 01 | train_loss: 0.6366 | train_acc: 0.6795 | dev_loss: 0.6243 | dev_acc: 0.6628
epoch: 02 | train_loss: 0.5118 | train_acc: 0.8171 | dev_loss: 0.5388 | dev_acc: 0.7729
epoch: 03 | train_loss: 0.4103 | train_acc: 0.8586 | dev_loss: 0.4824 | dev_acc: 0.8096
epoch: 04 | train_loss: 0.3495 | train_acc: 0.8755 | dev_loss: 0.4509 | dev_acc: 0.8211
epoch: 05 | train_loss: 0.3119 | train_acc: 0.8862 | dev_loss: 0.4325 | dev_acc: 0.8245
epoch: 06 | train_loss: 0.2864 | train_acc: 0.8948 | dev_loss: 0.4237 | dev_acc: 0.8257
epoch: 07 | train_loss: 0.2678 | train_acc: 0.9010 | dev_loss: 0.4166 | dev_acc: 0.8222
epoch: 08 | train_loss: 0.2534 | train_acc: 0.9055 | dev_loss: 0.4155 | dev_acc: 0.8211
epoch: 09 | train_loss: 0.2419 | train_acc: 0.9092 | dev_loss: 0.4152 | dev_acc: 0.8200
epoch: 10 | train_loss: 0.2324 | train_acc: 0.9126 | dev_loss: 0.4178 | dev_acc: 0.8142
epoch: 11 | train_loss: 0.2244 | train_acc: 0.9161 | dev_loss: 0.4202 | dev_acc: 0.8142
epoch: 12 | train_loss: 0.2175 | train_acc: 0.9187 | dev_loss: 0.4236 | dev_acc: 0.8154
epoch: 13 | train_loss: 0.2116 | train_acc: 0.9206 | dev_loss: 0.4291 | dev_acc: 0.8154
epoch: 14 | train_loss: 0.2064 | train_acc: 0.9219 | dev_loss: 0.4334 | dev_acc: 0.8108
epoch: 15 | train_loss: 0.2018 | train_acc: 0.9235 | dev_loss: 0.4391 | dev_acc: 0.8028
epoch: 16 | train_loss: 0.1977 | train_acc: 0.9247 | dev_loss: 0.4452 | dev_acc: 0.7982
epoch: 17 | train_loss: 0.1940 | train_acc: 0.9257 | dev_loss: 0.4516 | dev_acc: 0.7970
epoch: 18 | train_loss: 0.1906 | train_acc: 0.9269 | dev_loss: 0.4589 | dev_acc: 0.7936
epoch: 19 | train_loss: 0.1877 | train_acc: 0.9274 | dev_loss: 0.4646 | dev_acc: 0.7947
epoch: 20 | train_loss: 0.1850 | train_acc: 0.9287 | dev_loss: 0.4693 | dev_acc: 0.7982

学習完了
開発セットでの最高正解率: 0.8257
開発セットでの最高正解率(%): 82.57%
保存しました: bow_logistic_regression_finetune.pt
'''