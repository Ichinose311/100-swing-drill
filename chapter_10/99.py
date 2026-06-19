from pathlib import Path
import zipfile

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import DPOConfig, DPOTrainer


MODEL_NAME = "openai-community/gpt2-medium"

BASE_DIR = Path(__file__).resolve().parents[1]
SST2_ZIP_PATH = BASE_DIR / "chapter_8" / "SST-2.zip"

OUTPUT_DIR = Path(__file__).parent / "99_gpt2_medium_sst2_dpo"

MAX_LENGTH = 128

# まず動作確認するなら 1000 / 200 などにする
MAX_TRAIN_EXAMPLES = None
MAX_DEV_EXAMPLES = None

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 16
EPOCHS = 1
LR = 1e-6
BETA = 0.1


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
    if int(label) == 1:
        return " positive"
    return " negative"


def opposite_label_text(label):
    if int(label) == 1:
        return " negative"
    return " positive"


def make_dpo_dataset(df):
    """
    SST-2をDPO用の preference dataset に変換する。

    prompt:
      問題96と同じプロンプト

    chosen:
      正解ラベル

    rejected:
      不正解ラベル
    """
    rows = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        label = int(row["label"])

        rows.append({
            "prompt": make_prompt(sentence),
            "chosen": label_to_text(label),
            "rejected": opposite_label_text(label),
        })

    return Dataset.from_list(rows)


def parse_generated_label(text):
    text = text.strip().lower()

    if text.startswith("positive"):
        return 1
    if text.startswith("negative"):
        return 0

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

    train_dataset = make_dpo_dataset(train_df)

    print("\nDPOデータ例")
    print("=" * 80)
    print(train_dataset[0])
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # DPOTrainerでは padding side は left が必要
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    # GPT-2系でpad_tokenを追加した場合に備える
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # DPOでは policy model と reference model を比較する
    # ref_model=None の場合、DPOTrainerが参照モデルを用意する
    training_args = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        beta=BETA,
        max_length=MAX_LENGTH,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
        bf16=(device == "cuda"),
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nDPO学習開始")
    print("=" * 80)
    trainer.train()

    print("\nモデル保存")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"saved to: {OUTPUT_DIR}")

    # 学習済みモデルを評価
    trained_model = trainer.model
    trained_model.to(device)

    print("\n開発データで生成評価")
    dev_acc, examples = evaluate_by_generation(
        model=trained_model,
        tokenizer=tokenizer,
        device=device,
        df=dev_df,
    )

    print("=" * 80)
    print(f"dev generation accuracy: {dev_acc:.4f}")
    print("=" * 80)

    print("\n生成例")
    print("-" * 80)
    for ex in examples:
        print(f"sentence: {ex['sentence']}")
        print(f"gold: {ex['gold']}")
        print(f"generated: {ex['generated']!r}")
        print(f"pred: {ex['pred']}")
        print()


if __name__ == "__main__":
    main()

#出力例
'''
モデル保存
Writing model shards: 100%|█| 1/1 [00:06<00:0
saved to: /data/student/i2311021/projects/100-swing-drill/chapter_10/99_gpt2_medium_sst2_dpo

開発データで生成評価
evaluating: 100%|█| 872/872 [05:10<00:00,  2.
================================================================================
dev generation accuracy: 0.0883
================================================================================

生成例
--------------------------------------------------------------------------------
sentence: it 's a charming and often affecting journey . 
gold: 1
generated: ' I think it'
pred: -1

sentence: unflinchingly bleak and desperate 
gold: 0
generated: ' Positive.\n'
pred: 1

sentence: allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . 
gold: 1
generated: ' I think it'
pred: -1

sentence: the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales . 
gold: 1
generated: ' I think it'
pred: -1

sentence: it 's slow -- very , very slow . 
gold: 0
generated: ' I think it'
pred: -1

sentence: although laced with humor and a few fanciful touches , the film is a refreshingly serious look at young women . 
gold: 1
generated: ' I think it'
pred: -1

sentence: a sometimes tedious film . 
gold: 0
generated: ' Positive. '
pred: 1

sentence: or doing last year 's taxes with your ex-wife . 
gold: 0
generated: ' I think it'
pred: -1

sentence: you do n't have to know about music to appreciate the film 's easygoing blend of comedy and romance . 
gold: 1
generated: ' I think it'
pred: -1

sentence: in exactly 89 minutes , most of which passed as slowly as if i 'd been sitting naked on an igloo , formula 51 sank from quirky to jerky to utter turkey . 
gold: 0
generated: ' I think it'
pred: -1
'''