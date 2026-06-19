from pathlib import Path
import csv
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "google-bert/bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128


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

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }

        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]

        return item


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

            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            labels = batch["labels"]

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


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
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if step % 100 == 0:
            print(f"  step {step:4d} / {len(dataloader)}  loss: {loss.item():.4f}")

    avg_loss = total_loss / total_examples

    return avg_loss


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

    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer)
    dev_dataset = SST2Dataset(dev_texts, dev_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_dev_accuracy = 0.0
    best_dev_loss = float("inf")
    best_epoch = 0

    output_dir = base_dir / "bert_sst2_finetuned_best"
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print()
        print(f"===== epoch {epoch} / {EPOCHS} =====")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        dev_loss, dev_accuracy = evaluate(
            model,
            dev_loader,
            device
        )

        print(f"train loss: {train_loss:.4f}")
        print(f"dev loss  : {dev_loss:.4f}")
        print(f"dev acc   : {dev_accuracy:.4f}")

        # 最後のepochではなく、全epoch中で最良のdev accuracyを記録する
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_dev_loss = dev_loss
            best_epoch = epoch

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            print("best model を保存しました")

    print()
    print("===== 結果 =====")
    print(f"best epoch       : {best_epoch}")
    print(f"best dev loss    : {best_dev_loss:.4f}")
    print(f"best dev accuracy: {best_dev_accuracy:.4f}")


if __name__ == "__main__":
    main()

#出力結果
'''
===== epoch 1 / 3 =====
  step  100 / 4210  loss: 0.5060
  step  200 / 4210  loss: 0.2537
  step  300 / 4210  loss: 0.3640
  step  400 / 4210  loss: 0.2412
  step  500 / 4210  loss: 0.3858
  step  600 / 4210  loss: 0.3115
  step  700 / 4210  loss: 0.0693
  step  800 / 4210  loss: 0.0554
  step  900 / 4210  loss: 0.3977
  step 1000 / 4210  loss: 0.2303
  step 1100 / 4210  loss: 0.1264
  step 1200 / 4210  loss: 0.2146
  step 1300 / 4210  loss: 0.4335
  step 1400 / 4210  loss: 0.1073
  step 1500 / 4210  loss: 0.1744
  step 1600 / 4210  loss: 0.6003
  step 1700 / 4210  loss: 0.5228
  step 1800 / 4210  loss: 0.1257
  step 1900 / 4210  loss: 0.1485
  step 2000 / 4210  loss: 0.0589
  step 2100 / 4210  loss: 0.1077
  step 2200 / 4210  loss: 0.1513
  step 2300 / 4210  loss: 0.2568
  step 2400 / 4210  loss: 0.3121
  step 2500 / 4210  loss: 0.0355
  step 2600 / 4210  loss: 0.1058
  step 2700 / 4210  loss: 0.0292
  step 2800 / 4210  loss: 0.0228
  step 2900 / 4210  loss: 0.1686
  step 3000 / 4210  loss: 0.1662
  step 3100 / 4210  loss: 0.0533
  step 3200 / 4210  loss: 0.2154
  step 3300 / 4210  loss: 0.2412
  step 3400 / 4210  loss: 0.5445
  step 3500 / 4210  loss: 0.2049
  step 3600 / 4210  loss: 0.0161
  step 3700 / 4210  loss: 0.2816
  step 3800 / 4210  loss: 0.0582
  step 3900 / 4210  loss: 0.1554
  step 4000 / 4210  loss: 0.1669
  step 4100 / 4210  loss: 0.0533
  step 4200 / 4210  loss: 0.0279
train loss: 0.1997
dev loss  : 0.2083
dev acc   : 0.9255
Writing model shards: 100%|█| 1/1 [00:00<00:00,  2.32
best model を保存しました

===== epoch 2 / 3 =====
  step  100 / 4210  loss: 0.0113
  step  200 / 4210  loss: 0.2006
  step  300 / 4210  loss: 0.1259
  step  400 / 4210  loss: 0.1505
  step  500 / 4210  loss: 0.1088
  step  600 / 4210  loss: 0.0198
  step  700 / 4210  loss: 0.1533
  step  800 / 4210  loss: 0.0545
  step  900 / 4210  loss: 0.0091
  step 1000 / 4210  loss: 0.0118
  step 1100 / 4210  loss: 0.0847
  step 1200 / 4210  loss: 0.0451
  step 1300 / 4210  loss: 0.0373
  step 1400 / 4210  loss: 0.0079
  step 1500 / 4210  loss: 0.0081
  step 1600 / 4210  loss: 0.1416
  step 1700 / 4210  loss: 0.0224
  step 1800 / 4210  loss: 0.4164
  step 1900 / 4210  loss: 0.1655
  step 2000 / 4210  loss: 0.0295
  step 2100 / 4210  loss: 0.0110
  step 2200 / 4210  loss: 0.0596
  step 2300 / 4210  loss: 0.2041
  step 2400 / 4210  loss: 0.1724
  step 2500 / 4210  loss: 0.0447
  step 2600 / 4210  loss: 0.3573
  step 2700 / 4210  loss: 0.0078
  step 2800 / 4210  loss: 0.0417
  step 2900 / 4210  loss: 0.0179
  step 3000 / 4210  loss: 0.0347
  step 3100 / 4210  loss: 0.1078
  step 3200 / 4210  loss: 0.0283
  step 3300 / 4210  loss: 0.0348
  step 3400 / 4210  loss: 0.0428
  step 3500 / 4210  loss: 0.0772
  step 3600 / 4210  loss: 0.3082
  step 3700 / 4210  loss: 0.0883
  step 3800 / 4210  loss: 0.0877
  step 3900 / 4210  loss: 0.0114
  step 4000 / 4210  loss: 0.2314
  step 4100 / 4210  loss: 0.1479
  step 4200 / 4210  loss: 0.0305
train loss: 0.1083
dev loss  : 0.2257
dev acc   : 0.9243

===== epoch 3 / 3 =====
  step  100 / 4210  loss: 0.0783
  step  200 / 4210  loss: 0.0580
  step  300 / 4210  loss: 0.3358
  step  400 / 4210  loss: 0.0071
  step  500 / 4210  loss: 0.0565
  step  600 / 4210  loss: 0.0012
  step  700 / 4210  loss: 0.0321
  step  800 / 4210  loss: 0.0115
  step  900 / 4210  loss: 0.0191
  step 1000 / 4210  loss: 0.0126
  step 1100 / 4210  loss: 0.0090
  step 1200 / 4210  loss: 0.0512
  step 1300 / 4210  loss: 0.1856
  step 1400 / 4210  loss: 0.0443
  step 1500 / 4210  loss: 0.0418
  step 1600 / 4210  loss: 0.1006
  step 1700 / 4210  loss: 0.0098
  step 1800 / 4210  loss: 0.0172
  step 1900 / 4210  loss: 0.2445
  step 2000 / 4210  loss: 0.1910
  step 2100 / 4210  loss: 0.0556
  step 2200 / 4210  loss: 0.0045
  step 2300 / 4210  loss: 0.2477
  step 2400 / 4210  loss: 0.0051
  step 2500 / 4210  loss: 0.0537
  step 2600 / 4210  loss: 0.0822
  step 2700 / 4210  loss: 0.0198
  step 2800 / 4210  loss: 0.1996
  step 2900 / 4210  loss: 0.0073
  step 3000 / 4210  loss: 0.1086
  step 3100 / 4210  loss: 0.0008
  step 3200 / 4210  loss: 0.2411
  step 3300 / 4210  loss: 0.0071
  step 3400 / 4210  loss: 0.0695
  step 3500 / 4210  loss: 0.0523
  step 3600 / 4210  loss: 0.1428
  step 3700 / 4210  loss: 0.0312
  step 3800 / 4210  loss: 0.0042
  step 3900 / 4210  loss: 0.0556
  step 4000 / 4210  loss: 0.0680
  step 4100 / 4210  loss: 0.0363
  step 4200 / 4210  loss: 0.1793
train loss: 0.0719
dev loss  : 0.2417
dev acc   : 0.9335
Writing model shards: 100%|█| 1/1 [00:00<00:00,  2.09
best model を保存しました

===== 結果 =====
best epoch       : 3
best dev loss    : 0.2417
best dev accuracy: 0.9335
'''