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
        entity="72hjk8hyrz-",
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
'''
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/ichinose/.netrc.
wandb: Currently logged in as: 72hjk8hyrz (72hjk8hyrz-) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.27.2
wandb: Run data is saved locally in /home/ichinose/projects/100-swing-drill/chapter_8/wandb/run-20260617_203548-6q5nkrqm
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run 73-bow-logistic-regression
wandb: ⭐️ View project at https://wandb.ai/72hjk8hyrz-/100-swing-drill-chapter-8
wandb: 🚀 View run at https://wandb.ai/72hjk8hyrz-/100-swing-drill-chapter-8/runs/6q5nkrqm
使用デバイス: cpu

学習対象パラメータ
embedding.weight: requires_grad=False
linear.weight: requires_grad=True
linear.bias: requires_grad=True

学習開始
epoch: 01 | train_loss: 0.5382 | train_acc: 0.7615 | dev_loss: 0.5422 | dev_acc: 0.7718
grep -n "wandb.init" -A 6 73.pyepoch: 02 | train_loss: 0.4433 | train_acc: 0.8155 | dev_loss: 0.5070 | dev_acc: 0.7752
epoch: 03 | train_loss: 0.4176 | train_acc: 0.8221 | dev_loss: 0.4930 | dev_acc: 0.7798
epoch: 04 | train_loss: 0.4058 | train_acc: 0.8261 | dev_loss: 0.4802 | dev_acc: 0.7798
epoch: 05 | train_loss: 0.3991 | train_acc: 0.8268 | dev_loss: 0.4725 | dev_acc: 0.7810
epoch: 06 | train_loss: 0.3950 | train_acc: 0.8288 | dev_loss: 0.4699 | dev_acc: 0.7833
epoch: 07 | train_loss: 0.3921 | train_acc: 0.8304 | dev_loss: 0.4689 | dev_acc: 0.7833
epoch: 08 | train_loss: 0.3902 | train_acc: 0.8314 | dev_loss: 0.4656 | dev_acc: 0.7856
epoch: 09 | train_loss: 0.3887 | train_acc: 0.8318 | dev_loss: 0.4600 | dev_acc: 0.7833
epoch: 10 | train_loss: 0.3876 | train_acc: 0.8325 | dev_loss: 0.4632 | dev_acc: 0.7856
epoch: 11 | train_loss: 0.3866 | train_acc: 0.8328 | dev_loss: 0.4610 | dev_acc: 0.7856
epoch: 12 | train_loss: 0.3859 | train_acc: 0.8332 | dev_loss: 0.4640 | dev_acc: 0.7798
epoch: 13 | train_loss: 0.3853 | train_acc: 0.8331 | dev_loss: 0.4600 | dev_acc: 0.7867
epoch: 14 | train_loss: 0.3849 | train_acc: 0.8342 | dev_loss: 0.4549 | dev_acc: 0.7901
epoch: 15 | train_loss: 0.3844 | train_acc: 0.8342 | dev_loss: 0.4588 | dev_acc: 0.7878
epoch: 16 | train_loss: 0.3841 | train_acc: 0.8339 | dev_loss: 0.4574 | dev_acc: 0.7867
epoch: 17 | train_loss: 0.3838 | train_acc: 0.8346 | dev_loss: 0.4562 | dev_acc: 0.7867
epoch: 18 | train_loss: 0.3836 | train_acc: 0.8347 | dev_loss: 0.4569 | dev_acc: 0.7878
epoch: 19 | train_loss: 0.3833 | train_acc: 0.8344 | dev_loss: 0.4576 | dev_acc: 0.7844
epoch: 20 | train_loss: 0.3831 | train_acc: 0.8346 | dev_loss: 0.4582 | dev_acc: 0.7878
wandb: WARNING Saving files without folders. If you want to preserve subdirectories pass base_path to wandb.save, i.e. wandb.save("/mnt/folder/file.h5", base_path="/mnt")
wandb: WARNING Symlinked 1 file into the W&B run directory; call wandb.save again to sync new files.

学習完了
保存しました: /home/ichinose/projects/100-swing-drill/chapter_8/bow_logistic_regression.pt
wandb: 
wandb: Run history:
wandb:     batch/train_loss ██▆▆▆▅▅▄▄▂▁▃▄▆▃▅▃▂▆▅▄▂▄▃▃▅▄▄▁▄▅▃▂▆▄▄▂▁█▂
wandb:                epoch ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb:   epoch/dev_accuracy ▁▂▄▄▅▅▅▆▅▆▆▄▇█▇▇▇▇▆▇
wandb:       epoch/dev_loss █▅▄▃▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁
wandb: epoch/train_accuracy ▁▆▇▇▇▇██████████████
wandb:     epoch/train_loss █▄▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                 step ▁▁▁▂▂▂▂▂▃▃▄▄▄▄▄▄▄▄▄▄▅▅▅▆▆▆▆▆▆▆▆▆▆▆▇▇▇███
wandb: 
wandb: Run summary:
wandb:     batch/train_loss 0.41334
wandb:                epoch 20
wandb:   epoch/dev_accuracy 0.78784
wandb:       epoch/dev_loss 0.45821
wandb: epoch/train_accuracy 0.83462
wandb:     epoch/train_loss 0.38314
wandb:                 step 20580
wandb: 
wandb: 🚀 View run 73-bow-logistic-regression at: https://wandb.ai/72hjk8hyrz-/100-swing-drill-chapter-8/runs/6q5nkrqm
wandb: ⭐️ View project at: https://wandb.ai/72hjk8hyrz-/100-swing-drill-chapter-8
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20260617_203548-6q5nkrqm/logs
'''