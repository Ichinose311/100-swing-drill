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
            freeze=True,
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

    model_output_path = base_dir / "bow_logistic_regression_minibatch.pt"

    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-3

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
embedding.weight: requires_grad=False
linear.weight: requires_grad=True
linear.bias: requires_grad=True

ミニバッチ学習開始
batch_size: 64
epoch: 01 | train_loss: 0.5361 | train_acc: 0.7719 | dev_loss: 0.5417 | dev_acc: 0.7764
epoch: 02 | train_loss: 0.4425 | train_acc: 0.8161 | dev_loss: 0.5073 | dev_acc: 0.7787
epoch: 03 | train_loss: 0.4171 | train_acc: 0.8222 | dev_loss: 0.4888 | dev_acc: 0.7764
epoch: 04 | train_loss: 0.4055 | train_acc: 0.8257 | dev_loss: 0.4793 | dev_acc: 0.7798
epoch: 05 | train_loss: 0.3990 | train_acc: 0.8271 | dev_loss: 0.4741 | dev_acc: 0.7798
epoch: 06 | train_loss: 0.3948 | train_acc: 0.8299 | dev_loss: 0.4711 | dev_acc: 0.7821
epoch: 07 | train_loss: 0.3921 | train_acc: 0.8307 | dev_loss: 0.4669 | dev_acc: 0.7821
epoch: 08 | train_loss: 0.3901 | train_acc: 0.8319 | dev_loss: 0.4652 | dev_acc: 0.7856
epoch: 09 | train_loss: 0.3886 | train_acc: 0.8321 | dev_loss: 0.4647 | dev_acc: 0.7856
epoch: 10 | train_loss: 0.3875 | train_acc: 0.8320 | dev_loss: 0.4626 | dev_acc: 0.7867
epoch: 11 | train_loss: 0.3866 | train_acc: 0.8330 | dev_loss: 0.4609 | dev_acc: 0.7856
epoch: 12 | train_loss: 0.3859 | train_acc: 0.8335 | dev_loss: 0.4572 | dev_acc: 0.7844
epoch: 13 | train_loss: 0.3853 | train_acc: 0.8340 | dev_loss: 0.4551 | dev_acc: 0.7913
epoch: 14 | train_loss: 0.3849 | train_acc: 0.8337 | dev_loss: 0.4572 | dev_acc: 0.7867
epoch: 15 | train_loss: 0.3844 | train_acc: 0.8338 | dev_loss: 0.4555 | dev_acc: 0.7867
epoch: 16 | train_loss: 0.3841 | train_acc: 0.8342 | dev_loss: 0.4560 | dev_acc: 0.7844
epoch: 17 | train_loss: 0.3838 | train_acc: 0.8346 | dev_loss: 0.4543 | dev_acc: 0.7878
epoch: 18 | train_loss: 0.3836 | train_acc: 0.8349 | dev_loss: 0.4564 | dev_acc: 0.7878
epoch: 19 | train_loss: 0.3833 | train_acc: 0.8342 | dev_loss: 0.4554 | dev_acc: 0.7867
epoch: 20 | train_loss: 0.3832 | train_acc: 0.8347 | dev_loss: 0.4548 | dev_acc: 0.7856

学習完了
開発セットでの最高正解率: 0.7913
開発セットでの最高正解率(%): 79.13%
保存しました: bow_logistic_regression_minibatch.pt
'''