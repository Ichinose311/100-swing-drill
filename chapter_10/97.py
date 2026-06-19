from pathlib import Path
import zipfile

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, AutoModel, set_seed


MODEL_NAME = "openai-community/gpt2-medium"

BASE_DIR = Path(__file__).resolve().parents[1]
SST2_ZIP_PATH = BASE_DIR / "chapter_8" / "SST-2.zip"

MAX_LENGTH = 128
EMBED_BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3

TRAIN_EMB_PATH = Path(__file__).parent / "97_train_embeddings.pt"
DEV_EMB_PATH = Path(__file__).parent / "97_dev_embeddings.pt"
BEST_MODEL_PATH = Path(__file__).parent / "97_mlp_classifier.pt"


def find_file_in_zip(zip_file, target_name):
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name
    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")


def load_sst2(zip_path, split):
    target = f"{split}.tsv"

    with zipfile.ZipFile(zip_path) as zf:
        file_name = find_file_in_zip(zf, target)
        with zf.open(file_name) as f:
            df = pd.read_csv(f, sep="\t")

    df = df[["sentence", "label"]].copy()
    df["label"] = df["label"].astype(int)

    return df


def mean_pooling(last_hidden_state, attention_mask):
    """
    last_hidden_state: [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]

    padding部分を除いて平均を取る。
    """
    mask = attention_mask.unsqueeze(-1).float()
    masked_hidden = last_hidden_state * mask

    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


def encode_texts(model, tokenizer, device, texts):
    """
    GPT2-mediumで文をベクトル化する。
    """
    model.eval()
    embeddings = []

    for start in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="encoding"):
        batch_texts = texts[start:start + EMBED_BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        sentence_embeddings = mean_pooling(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )

        embeddings.append(sentence_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_labels=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        return self.classifier(x)


def evaluate(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * len(y)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    print(f"SST-2 path: {SST2_ZIP_PATH}")

    train_df = load_sst2(SST2_ZIP_PATH, "train")
    dev_df = load_sst2(SST2_ZIP_PATH, "dev")

    print(f"train size: {len(train_df)}")
    print(f"dev size: {len(dev_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2系はpad_tokenを持たないのでEOSをpad_tokenとして使う
    tokenizer.pad_token = tokenizer.eos_token

    # すでに埋め込みを作成済みなら再利用
    if TRAIN_EMB_PATH.exists() and DEV_EMB_PATH.exists():
        print("\n保存済み埋め込みを読み込みます")
        train_data = torch.load(TRAIN_EMB_PATH)
        dev_data = torch.load(DEV_EMB_PATH)

        train_embeddings = train_data["embeddings"]
        train_labels = train_data["labels"]
        dev_embeddings = dev_data["embeddings"]
        dev_labels = dev_data["labels"]

    else:
        print("\nGPT2-mediumで文ベクトルを作成します")

        gpt2 = AutoModel.from_pretrained(MODEL_NAME)
        gpt2.to(device)
        gpt2.eval()

        # GPT2-medium本体は特徴抽出器として使うため、勾配計算しない
        for param in gpt2.parameters():
            param.requires_grad = False

        train_embeddings = encode_texts(
            model=gpt2,
            tokenizer=tokenizer,
            device=device,
            texts=train_df["sentence"].tolist(),
        )
        dev_embeddings = encode_texts(
            model=gpt2,
            tokenizer=tokenizer,
            device=device,
            texts=dev_df["sentence"].tolist(),
        )

        train_labels = torch.tensor(train_df["label"].tolist(), dtype=torch.long)
        dev_labels = torch.tensor(dev_df["label"].tolist(), dtype=torch.long)

        torch.save(
            {
                "embeddings": train_embeddings,
                "labels": train_labels,
            },
            TRAIN_EMB_PATH,
        )
        torch.save(
            {
                "embeddings": dev_embeddings,
                "labels": dev_labels,
            },
            DEV_EMB_PATH,
        )

        print(f"train embeddings saved to: {TRAIN_EMB_PATH}")
        print(f"dev embeddings saved to: {DEV_EMB_PATH}")

        # GPUメモリを空ける
        del gpt2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    input_dim = train_embeddings.shape[1]
    print(f"\n文ベクトルの次元数: {input_dim}")

    train_dataset = TensorDataset(train_embeddings, train_labels)
    dev_dataset = TensorDataset(dev_embeddings, dev_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    clf_device = device
    classifier = MLPClassifier(input_dim=input_dim)
    classifier.to(clf_device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0

    print("\n分類器を学習します")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        classifier.train()

        total_loss = 0.0
        total_examples = 0

        for x, y in train_loader:
            x = x.to(clf_device)
            y = y.to(clf_device)

            optimizer.zero_grad()

            logits = classifier(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_examples += len(y)

        train_loss = total_loss / total_examples
        dev_loss, dev_acc = evaluate(classifier, dev_loader, clf_device)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss: {train_loss:.4f} | "
            f"dev_loss: {dev_loss:.4f} | "
            f"dev_acc: {dev_acc:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(classifier.state_dict(), BEST_MODEL_PATH)

    print("=" * 80)
    print(f"best dev accuracy: {best_dev_acc:.4f}")
    print(f"best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()

#出力結果
'''
使用デバイス: cuda
SST-2 path: /data/student/i2311021/projects/100-swing-drill/chapter_8/SST-2.zip
train size: 67349
dev size: 872
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

GPT2-mediumで文ベクトルを作成します
Loading weights: 100%|█| 292/292 [00:00<00:00
encoding: 100%|█| 4210/4210 [01:28<00:00, 47.
encoding: 100%|█| 55/55 [00:01<00:00, 43.93it
train embeddings saved to: /data/student/i2311021/projects/100-swing-drill/chapter_10/97_train_embeddings.pt
dev embeddings saved to: /data/student/i2311021/projects/100-swing-drill/chapter_10/97_dev_embeddings.pt

文ベクトルの次元数: 1024

分類器を学習します
================================================================================
epoch 01 | train_loss: 0.3817 | dev_loss: 0.3550 | dev_acc: 0.8544
epoch 02 | train_loss: 0.3127 | dev_loss: 0.3423 | dev_acc: 0.8612
epoch 03 | train_loss: 0.2963 | dev_loss: 0.3686 | dev_acc: 0.8521
epoch 04 | train_loss: 0.2869 | dev_loss: 0.3466 | dev_acc: 0.8567
epoch 05 | train_loss: 0.2833 | dev_loss: 0.3841 | dev_acc: 0.8486
epoch 06 | train_loss: 0.2752 | dev_loss: 0.3414 | dev_acc: 0.8693
epoch 07 | train_loss: 0.2705 | dev_loss: 0.3266 | dev_acc: 0.8612
epoch 08 | train_loss: 0.2670 | dev_loss: 0.3230 | dev_acc: 0.8693
epoch 09 | train_loss: 0.2590 | dev_loss: 0.3802 | dev_acc: 0.8624
epoch 10 | train_loss: 0.2563 | dev_loss: 0.3654 | dev_acc: 0.8498
epoch 11 | train_loss: 0.2528 | dev_loss: 0.3804 | dev_acc: 0.8440
epoch 12 | train_loss: 0.2474 | dev_loss: 0.3861 | dev_acc: 0.8567
epoch 13 | train_loss: 0.2437 | dev_loss: 0.3220 | dev_acc: 0.8739
epoch 14 | train_loss: 0.2384 | dev_loss: 0.3457 | dev_acc: 0.8612
epoch 15 | train_loss: 0.2345 | dev_loss: 0.3819 | dev_acc: 0.8567
epoch 16 | train_loss: 0.2322 | dev_loss: 0.3355 | dev_acc: 0.8750
epoch 17 | train_loss: 0.2281 | dev_loss: 0.3825 | dev_acc: 0.8589
epoch 18 | train_loss: 0.2229 | dev_loss: 0.3436 | dev_acc: 0.8612
epoch 19 | train_loss: 0.2218 | dev_loss: 0.3335 | dev_acc: 0.8635
epoch 20 | train_loss: 0.2153 | dev_loss: 0.3341 | dev_acc: 0.8750
================================================================================
best dev accuracy: 0.8750
best model saved to: /data/student/i2311021/projects/100-swing-drill/chapter_10/97_mlp_classifier.pt
'''