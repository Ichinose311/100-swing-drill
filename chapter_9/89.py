from pathlib import Path
import csv
import zipfile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "google-bert/bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
NUM_LABELS = 2


def find_file_in_zip(zip_file, target_name):
    """
    zip内から train.tsv / dev.tsv を探す
    """
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name

    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")


def load_sst2_tsv_from_zip(zip_path, target_name):
    """
    SST-2.zip から train.tsv / dev.tsv を読み込む
    """
    texts = []
    labels = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_path = find_file_in_zip(zf, target_name)

        with zf.open(tsv_path, "r") as f:
            lines = (line.decode("utf-8") for line in f)
            reader = csv.DictReader(lines, delimiter="\t")

            for row in reader:
                texts.append(row["sentence"])
                labels.append(int(row["label"]))

    return texts, labels


class SST2Dataset(Dataset):
    """
    SST-2用Dataset
    """

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


def create_collate_fn(tokenizer):
    """
    DataLoader内でミニバッチごとにパディングする関数
    """

    def collate_fn(batch):
        texts = [example["text"] for example in batch]
        labels = [example["label"] for example in batch]

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        encodings["labels"] = torch.tensor(labels, dtype=torch.long)

        return encodings

    return collate_fn


class BertMaxPoolingClassifier(nn.Module):
    """
    BERT + 最大値プーリングによる分類モデル

    87番の AutoModelForSequenceClassification とは異なり、
    [CLS]だけではなく、各トークンの最終層ベクトルにmax poolingを行う。
    """

    def __init__(self, model_name, num_labels):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        token_type_ids=None,
        special_tokens_mask=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 最終層の各トークンベクトル
        # shape: [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.last_hidden_state

        # PADや特殊トークンをmax poolingの対象から除外する
        if special_tokens_mask is not None:
            valid_mask = attention_mask.bool() & (~special_tokens_mask.bool())
        else:
            valid_mask = attention_mask.bool()

        # shape: [batch_size, seq_len, 1]
        valid_mask = valid_mask.unsqueeze(-1)

        # pooling対象外の位置を非常に小さい値にする
        masked_hidden_state = last_hidden_state.masked_fill(
            ~valid_mask,
            -1e9
        )

        # トークン方向に最大値プーリング
        # shape: [batch_size, hidden_size]
        pooled_output, _ = masked_hidden_state.max(dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


def train_one_epoch(model, dataloader, optimizer, device):
    """
    1 epoch分学習する
    """
    model.train()

    total_loss = 0.0
    total_examples = 0

    for step, batch in enumerate(dataloader, start=1):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs["loss"]

        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if step % 100 == 0:
            print(f"  step {step:4d} / {len(dataloader)}  loss: {loss.item():.4f}")

    avg_loss = total_loss / total_examples

    return avg_loss


def evaluate(model, dataloader, device):
    """
    devデータでlossとaccuracyを計算する
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = outputs["loss"]
            logits = outputs["logits"]

            predictions = torch.argmax(logits, dim=1)
            labels = batch["labels"]

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


def main():
    base_dir = Path(__file__).parent
    sst_zip_path = base_dir / "SST-2.zip"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用デバイス:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_texts, train_labels = load_sst2_tsv_from_zip(
        sst_zip_path,
        "train.tsv"
    )

    dev_texts, dev_labels = load_sst2_tsv_from_zip(
        sst_zip_path,
        "dev.tsv"
    )

    print("訓練データ数:", len(train_texts))
    print("開発データ数:", len(dev_texts))

    train_dataset = SST2Dataset(train_texts, train_labels)
    dev_dataset = SST2Dataset(dev_texts, dev_labels)

    collate_fn = create_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BertMaxPoolingClassifier(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_dev_accuracy = 0.0
    best_dev_loss = float("inf")
    best_epoch = 0

    output_path = base_dir / "bert_sst2_maxpooling_best.pt"

    for epoch in range(1, EPOCHS + 1):
        print()
        print(f"===== epoch {epoch} / {EPOCHS} =====")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        dev_loss, dev_accuracy = evaluate(
            model,
            dev_loader,
            device,
        )

        print(f"train loss: {train_loss:.4f}")
        print(f"dev loss  : {dev_loss:.4f}")
        print(f"dev acc   : {dev_accuracy:.4f}")

        # 最後のepochではなく、全epoch中で最も高いdev accuracyを採用する
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_dev_loss = dev_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "best_epoch": best_epoch,
                    "best_dev_loss": best_dev_loss,
                    "best_dev_accuracy": best_dev_accuracy,
                },
                output_path,
            )

            print("best model を保存しました")

    print()
    print("===== 結果 =====")
    print("アーキテクチャ: BERT + max pooling + linear classifier")
    print(f"best epoch       : {best_epoch}")
    print(f"best dev loss    : {best_dev_loss:.4f}")
    print(f"best dev accuracy: {best_dev_accuracy:.4f}")
    print(f"best model path  : {output_path}")


if __name__ == "__main__":
    main()

#出力結果
'''
===== epoch 1 / 3 =====
  step  100 / 4210  loss: 0.3516
  step  200 / 4210  loss: 0.4204
  step  300 / 4210  loss: 0.2467
  step  400 / 4210  loss: 0.4062
  step  500 / 4210  loss: 0.2123
  step  600 / 4210  loss: 0.3105
  step  700 / 4210  loss: 0.1532
  step  800 / 4210  loss: 0.1039
  step  900 / 4210  loss: 0.2763
  step 1000 / 4210  loss: 0.2324
  step 1100 / 4210  loss: 0.0516
  step 1200 / 4210  loss: 0.0765
  step 1300 / 4210  loss: 0.0447
  step 1400 / 4210  loss: 0.1571
  step 1500 / 4210  loss: 0.0658
  step 1600 / 4210  loss: 0.2683
  step 1700 / 4210  loss: 0.0285
  step 1800 / 4210  loss: 0.2248
  step 1900 / 4210  loss: 0.2003
  step 2000 / 4210  loss: 0.1900
  step 2100 / 4210  loss: 0.3199
  step 2200 / 4210  loss: 0.0772
  step 2300 / 4210  loss: 0.0232
  step 2400 / 4210  loss: 0.1604
  step 2500 / 4210  loss: 0.2113
  step 2600 / 4210  loss: 0.1274
  step 2700 / 4210  loss: 0.0420
  step 2800 / 4210  loss: 0.1052
  step 2900 / 4210  loss: 0.0951
  step 3000 / 4210  loss: 0.1353
  step 3100 / 4210  loss: 0.0141
  step 3200 / 4210  loss: 0.0328
  step 3300 / 4210  loss: 0.2190
  step 3400 / 4210  loss: 0.1371
  step 3500 / 4210  loss: 0.1118
  step 3600 / 4210  loss: 0.2137
  step 3700 / 4210  loss: 0.1485
  step 3800 / 4210  loss: 0.2742
  step 3900 / 4210  loss: 0.3491
  step 4000 / 4210  loss: 0.0622
  step 4100 / 4210  loss: 0.4119
  step 4200 / 4210  loss: 0.1529
train loss: 0.1993
dev loss  : 0.2074
dev acc   : 0.9197
best model を保存しました

===== epoch 2 / 3 =====
  step  100 / 4210  loss: 0.2003
  step  200 / 4210  loss: 0.1445
  step  300 / 4210  loss: 0.0579
  step  400 / 4210  loss: 0.0964
  step  500 / 4210  loss: 0.0420
  step  600 / 4210  loss: 0.0747
  step  700 / 4210  loss: 0.1542
  step  800 / 4210  loss: 0.0560
  step  900 / 4210  loss: 0.1968
  step 1000 / 4210  loss: 0.1991
  step 1100 / 4210  loss: 0.0425
  step 1200 / 4210  loss: 0.0556
  step 1300 / 4210  loss: 0.0184
  step 1400 / 4210  loss: 0.1114
  step 1500 / 4210  loss: 0.2896
  step 1600 / 4210  loss: 0.0661
  step 1700 / 4210  loss: 0.3262
  step 1800 / 4210  loss: 0.0612
  step 1900 / 4210  loss: 0.5071
  step 2000 / 4210  loss: 0.0060
  step 2100 / 4210  loss: 0.0148
  step 2200 / 4210  loss: 0.4073
  step 2300 / 4210  loss: 0.0091
  step 2400 / 4210  loss: 0.1762
  step 2500 / 4210  loss: 0.0139
  step 2600 / 4210  loss: 0.0136
  step 2700 / 4210  loss: 0.0840
  step 2800 / 4210  loss: 0.0895
  step 2900 / 4210  loss: 0.3067
  step 3000 / 4210  loss: 0.0794
  step 3100 / 4210  loss: 0.0961
  step 3200 / 4210  loss: 0.1013
  step 3300 / 4210  loss: 0.2176
  step 3400 / 4210  loss: 0.0925
  step 3500 / 4210  loss: 0.0354
  step 3600 / 4210  loss: 0.0680
  step 3700 / 4210  loss: 0.1225
  step 3800 / 4210  loss: 0.1566
  step 3900 / 4210  loss: 0.1447
  step 4000 / 4210  loss: 0.0899
  step 4100 / 4210  loss: 0.0530
  step 4200 / 4210  loss: 0.0617
train loss: 0.1059
dev loss  : 0.2399
dev acc   : 0.9163

===== epoch 3 / 3 =====
  step  100 / 4210  loss: 0.1149
  step  200 / 4210  loss: 0.0052
  step  300 / 4210  loss: 0.0099
  step  400 / 4210  loss: 0.0059
  step  500 / 4210  loss: 0.0232
  step  600 / 4210  loss: 0.3523
  step  700 / 4210  loss: 0.1442
  step  800 / 4210  loss: 0.0051
  step  900 / 4210  loss: 0.0209
  step 1000 / 4210  loss: 0.0064
  step 1100 / 4210  loss: 0.0291
  step 1200 / 4210  loss: 0.3020
  step 1300 / 4210  loss: 0.0198
  step 1400 / 4210  loss: 0.0070
  step 1500 / 4210  loss: 0.1936
  step 1600 / 4210  loss: 0.0565
  step 1700 / 4210  loss: 0.0999
  step 1800 / 4210  loss: 0.0538
  step 1900 / 4210  loss: 0.0331
  step 2000 / 4210  loss: 0.0746
  step 2100 / 4210  loss: 0.0058
  step 2200 / 4210  loss: 0.0043
  step 2300 / 4210  loss: 0.0137
  step 2400 / 4210  loss: 0.0122
  step 2500 / 4210  loss: 0.0502
  step 2600 / 4210  loss: 0.0022
  step 2700 / 4210  loss: 0.0876
  step 2800 / 4210  loss: 0.0481
  step 2900 / 4210  loss: 0.0099
  step 3000 / 4210  loss: 0.0216
  step 3100 / 4210  loss: 0.2049
  step 3200 / 4210  loss: 0.0519
  step 3300 / 4210  loss: 0.0493
  step 3400 / 4210  loss: 0.0260
  step 3500 / 4210  loss: 0.0081
  step 3600 / 4210  loss: 0.0342
  step 3700 / 4210  loss: 0.0971
  step 3800 / 4210  loss: 0.0104
  step 3900 / 4210  loss: 0.0503
  step 4000 / 4210  loss: 0.0202
  step 4100 / 4210  loss: 0.1390
  step 4200 / 4210  loss: 0.0791
train loss: 0.0698
dev loss  : 0.2266
dev acc   : 0.9197

===== 結果 =====
アーキテクチャ: BERT + max pooling + linear classifier
best epoch       : 1
best dev loss    : 0.2074
best dev accuracy: 0.9197
best model path  : /home/ichinose/projects/100-swing-drill/chapter_9/bert_sst2_maxpooling_best.pt
'''