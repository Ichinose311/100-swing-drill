from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import wandb


class BoWLogisticRegression(nn.Module):
    """
    単語埋め込みの平均ベクトルを使うBag of Wordsモデル
    """

    def __init__(self, embedding_matrix):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        # 事前学習済み単語埋め込み
        # freeze=True により、埋め込み行列は学習中に更新されない
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0
        )

        # 平均ベクトルを1次元のスコアに変換する線形層
        # ここだけ学習する
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)

        # embeddings: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)

        # <PAD>を平均に含めないためのマスク
        # mask: (batch_size, seq_len, 1)
        mask = (input_ids != 0).unsqueeze(-1)

        # <PAD>部分のベクトルを0にする
        embeddings = embeddings * mask

        # 単語ベクトルの合計
        # summed: (batch_size, embedding_dim)
        summed = embeddings.sum(dim=1)

        # 有効トークン数
        # lengths: (batch_size, 1)
        lengths = mask.sum(dim=1).clamp(min=1)

        # 平均ベクトル
        # bow_vector: (batch_size, embedding_dim)
        bow_vector = summed / lengths

        # logit: (batch_size, 1)
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
    損失値と正解率を計算する
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


def main():
    base_dir = Path(__file__).parent

    embedding_path = base_dir / "embedding_matrix.pt"
    train_path = base_dir / "sst_train_examples.pt"
    dev_path = base_dir / "sst_dev_examples.pt"

    model_output_path = base_dir / "bow_logistic_regression.pt"

    config = {
        "batch_size": 64,
        "num_epochs": 20,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "embedding_finetuning": False,
        "model": "BoWLogisticRegression",
    }

    # wandb初期化
    run = wandb.init(
        entity="account",
        project="100-swing-drill-chapter-8",
        name="73-bow-logistic-regression",
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用デバイス: {device}")

    # 70.pyで作成した単語埋め込み行列
    embedding_matrix = torch.load(embedding_path, map_location="cpu")

    # 71.pyで作成したデータセット
    train_examples = torch.load(train_path, map_location="cpu")
    dev_examples = torch.load(dev_path, map_location="cpu")

    train_loader = DataLoader(
        train_examples,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_examples,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BoWLogisticRegression(embedding_matrix)
    model.to(device)

    print()
    print("学習対象パラメータ")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # wandbにモデルの勾配やパラメータを記録
    # embeddingはfreezeされているので、主にlinear層が記録される
    wandb.watch(
        model,
        log="all",
        log_freq=100
    )

    loss_fn = nn.BCEWithLogitsLoss()

    # requires_grad=True のパラメータだけ最適化する
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"]
    )

    print()
    print("学習開始")

    global_step = 0

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()

        total_train_loss = 0.0
        train_correct = 0
        total_train_examples = 0

        for batch in train_loader:
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

            total_train_loss += loss.item() * batch_size
            train_correct += (preds == labels).sum().item()
            total_train_examples += batch_size

            global_step += 1

            # ミニバッチごとの損失をwandbに記録
            wandb.log(
                {
                    "batch/train_loss": loss.item(),
                    "step": global_step,
                },
                step=global_step
            )

        train_loss = total_train_loss / total_train_examples
        train_accuracy = train_correct / total_train_examples

        dev_loss, dev_accuracy = evaluate(
            model,
            dev_loader,
            loss_fn,
            device
        )

        print(
            f"epoch: {epoch:02d} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_accuracy:.4f} | "
            f"dev_loss: {dev_loss:.4f} | "
            f"dev_acc: {dev_accuracy:.4f}"
        )

        # epochごとの指標をwandbに記録
        wandb.log(
            {
                "epoch": epoch,
                "epoch/train_loss": train_loss,
                "epoch/train_accuracy": train_accuracy,
                "epoch/dev_loss": dev_loss,
                "epoch/dev_accuracy": dev_accuracy,
            },
            step=global_step
        )

    # モデル保存
    torch.save(model.state_dict(), model_output_path)

    # 保存したモデルをwandbにも登録
    wandb.save(str(model_output_path))

    print()
    print("学習完了")
    print(f"保存しました: {model_output_path}")

    wandb.finish()

if __name__ == "__main__":
    main()

#出力結果
