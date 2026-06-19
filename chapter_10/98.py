from pathlib import Path
import zipfile

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


MODEL_NAME = "openai-community/gpt2-medium"

BASE_DIR = Path(__file__).resolve().parents[1]
SST2_ZIP_PATH = BASE_DIR / "chapter_8" / "SST-2.zip"

OUTPUT_DIR = Path(__file__).parent / "98_gpt2_medium_sst2_sft"

MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
EPOCHS = 1
LR = 5e-5

# 動作確認だけしたい場合は 1000 などにする
MAX_TRAIN_EXAMPLES = None
MAX_DEV_EXAMPLES = None


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


def make_prompt(sentence):
    return (
        "Review: " + sentence + "\n"
        "Question: Is the sentiment of this review positive or negative?\n"
        "Answer:"
    )


def label_to_text(label):
    # SST-2: 0 = negative, 1 = positive
    if int(label) == 1:
        return " positive"
    return " negative"


class SST2SFTDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        prompt = make_prompt(row["sentence"])
        answer = label_to_text(row["label"]) + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + answer_ids

        # 長すぎる場合は後ろを切る
        input_ids = input_ids[:MAX_LENGTH]

        # prompt部分はlossを計算しない
        labels = [-100] * len(prompt_ids) + answer_ids
        labels = labels[:MAX_LENGTH]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for x in batch:
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        labels = x["labels"]

        pad_len = max_len - len(input_ids)

        input_ids = torch.cat([
            input_ids,
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
        ])

        attention_mask = torch.cat([
            attention_mask,
            torch.zeros(pad_len, dtype=torch.long),
        ])

        labels = torch.cat([
            labels,
            torch.full((pad_len,), -100, dtype=torch.long),
        ])

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


def parse_generated_label(text):
    text = text.strip().lower()

    if text.startswith("positive"):
        return 1
    if text.startswith("negative"):
        return 0

    # 応答が崩れた場合の保険
    if "positive" in text and "negative" not in text:
        return 1
    if "negative" in text and "positive" not in text:
        return 0

    return -1


def evaluate_by_generation(model, tokenizer, device, df):
    model.eval()

    gold_labels = []
    pred_labels = []
    examples = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="evaluating"):
        sentence = row["sentence"]
        gold = int(row["label"])

        prompt = make_prompt(sentence)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred = parse_generated_label(generated_text)

        gold_labels.append(gold)
        pred_labels.append(pred)

        if len(examples) < 10:
            examples.append({
                "sentence": sentence,
                "gold": gold,
                "generated": generated_text,
                "pred": pred,
            })

    # pred=-1 は不正解扱い
    acc = accuracy_score(gold_labels, pred_labels)

    return acc, examples


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    train_df = load_sst2(SST2_ZIP_PATH, "train")
    dev_df = load_sst2(SST2_ZIP_PATH, "dev")

    if MAX_TRAIN_EXAMPLES is not None:
        train_df = train_df.head(MAX_TRAIN_EXAMPLES)
    if MAX_DEV_EXAMPLES is not None:
        dev_df = dev_df.head(MAX_DEV_EXAMPLES)

    print(f"train size: {len(train_df)}")
    print(f"dev size: {len(dev_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.train()

    train_dataset = SST2SFTDataset(train_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    global_step = 0

    print("\nファインチューニング開始")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_steps = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}")

        optimizer.zero_grad()

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss / GRAD_ACCUM_STEPS
            else:
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            loss.backward()

            if step % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            total_steps += 1

            progress.set_postfix({
                "loss": total_loss / total_steps,
            })

        avg_loss = total_loss / total_steps
        print(f"epoch {epoch} train_loss: {avg_loss:.4f}")

        print("\n開発データで生成評価")
        dev_acc, examples = evaluate_by_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            df=dev_df,
        )

        print(f"dev generation accuracy: {dev_acc:.4f}")

        print("\n生成例")
        print("-" * 80)
        for ex in examples:
            print(f"sentence: {ex['sentence']}")
            print(f"gold: {ex['gold']}")
            print(f"generated: {ex['generated']!r}")
            print(f"pred: {ex['pred']}")
            print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 80)
    print(f"ファインチューニング済みモデルを保存しました: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#出力結果
'''
使用デバイス: cuda
train size: 67349
dev size: 872
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█| 292/292 [00:00<00:00
/data/student/i2311021/projects/100-swing-drill/chapter_10/98.py:247: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=False)

ファインチューニング開始
================================================================================
epoch 1:   0%|     | 0/33675 [00:00<?, ?it/s][transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
epoch 1: 100%|█| 33675/33675 [25:20<00:00, 22
epoch 1 train_loss: 0.0992

開発データで生成評価
evaluating: 100%|█| 872/872 [00:14<00:00, 61.
dev generation accuracy: 0.9346

生成例
--------------------------------------------------------------------------------
sentence: it 's a charming and often affecting journey . 
gold: 1
generated: ' positive'
pred: 1

sentence: unflinchingly bleak and desperate 
gold: 0
generated: ' negative'
pred: 0

sentence: allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . 
gold: 1
generated: ' positive'
pred: 1

sentence: the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales . 
gold: 1
generated: ' positive'
pred: 1

sentence: it 's slow -- very , very slow . 
gold: 0
generated: ' negative'
pred: 0

sentence: although laced with humor and a few fanciful touches , the film is a refreshingly serious look at young women . 
gold: 1
generated: ' positive'
pred: 1

sentence: a sometimes tedious film . 
gold: 0
generated: ' negative'
pred: 0

sentence: or doing last year 's taxes with your ex-wife . 
gold: 0
generated: ' negative'
pred: 0

sentence: you do n't have to know about music to appreciate the film 's easygoing blend of comedy and romance . 
gold: 1
generated: ' positive'
pred: 1

sentence: in exactly 89 minutes , most of which passed as slowly as if i 'd been sitting naked on an igloo , formula 51 sank from quirky to jerky to utter turkey . 
gold: 0
generated: ' negative'
pred: 0

Writing model shards: 100%|█| 1/1 [00:13<00:0
================================================================================
ファインチューニング済みモデルを保存しました: /data/student/i2311021/projects/100-swing-drill/chapter_10/98_gpt2_medium_sst2_sft
'''