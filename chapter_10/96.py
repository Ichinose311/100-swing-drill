from pathlib import Path
import zipfile

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from sklearn.metrics import accuracy_score


MODEL_NAME = "openai-community/gpt2-medium"

# chapter_10/96.py から見て ../chapter_8/SST-2.zip
SST2_ZIP_PATH = Path(__file__).resolve().parents[1] / "chapter_8" / "SST-2.zip"

BATCH_SIZE = 16
MAX_EXAMPLES = None  # 動作確認だけなら 100 などに変更してよい


def find_file_in_zip(zip_file, target_name):
    """
    zip内から dev.tsv を探す
    """
    for name in zip_file.namelist():
        if name.endswith(target_name):
            return name
    raise FileNotFoundError(f"{target_name} が zip 内に見つかりません")


def load_sst2_dev(zip_path):
    """
    SST-2.zip から dev.tsv を読み込む
    """
    with zipfile.ZipFile(zip_path) as zf:
        dev_name = find_file_in_zip(zf, "dev.tsv")
        with zf.open(dev_name) as f:
            df = pd.read_csv(f, sep="\t")

    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"dev.tsv の列名が想定と違います: {df.columns.tolist()}")

    df = df[["sentence", "label"]].copy()
    df["label"] = df["label"].astype(int)

    if MAX_EXAMPLES is not None:
        df = df.head(MAX_EXAMPLES)

    return df


def make_prompt(sentence):
    """
    GPT2-medium に与える感情分析用プロンプト
    """
    return (
        "Review: " + sentence + "\n"
        "Question: Is the sentiment of this review positive or negative?\n"
        "Answer:"
    )


def compute_label_logprobs(model, tokenizer, device, prompts, label_texts):
    """
    各 prompt について、各 label_text の対数尤度を計算する。

    label_texts 例:
      [" negative", " positive"]

    戻り値:
      logprob_matrix: shape = [len(prompts), len(label_texts)]
    """
    candidates = []

    for prompt_id, prompt in enumerate(prompts):
        # prompt 部分のトークン長
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        for label_id, label_text in enumerate(label_texts):
            full_text = prompt + label_text
            candidates.append(
                {
                    "prompt_id": prompt_id,
                    "label_id": label_id,
                    "text": full_text,
                    "prompt_len": prompt_len,
                }
            )

    all_logprobs = torch.empty(
        len(prompts),
        len(label_texts),
        dtype=torch.float32,
    )

    for start in tqdm(range(0, len(candidates), BATCH_SIZE), desc="scoring"):
        batch_candidates = candidates[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch_candidates]

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)

        for row, candidate in enumerate(batch_candidates):
            prompt_id = candidate["prompt_id"]
            label_id = candidate["label_id"]
            prompt_len = candidate["prompt_len"]

            seq_len = int(attention_mask[row].sum().item())

            # ラベル部分の log probability を足す
            # token[t] の確率は logits[t-1] から得られる
            label_logprob = 0.0
            for token_pos in range(prompt_len, seq_len):
                token_id = input_ids[row, token_pos]
                token_logprob = log_probs[row, token_pos - 1, token_id]
                label_logprob += float(token_logprob.item())

            all_logprobs[prompt_id, label_id] = label_logprob

    return all_logprobs


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    print(f"SST-2 path: {SST2_ZIP_PATH}")
    df = load_sst2_dev(SST2_ZIP_PATH)

    print(f"評価データ数: {len(df)}")
    print(df.head())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 系には pad_token がないため、EOS を pad_token として使う
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    prompts = [make_prompt(s) for s in df["sentence"].tolist()]

    # SST-2: 0 = negative, 1 = positive
    label_texts = [" negative", " positive"]

    logprob_matrix = compute_label_logprobs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        label_texts=label_texts,
    )

    # 対数尤度が大きい方を予測
    pred_labels = torch.argmax(logprob_matrix, dim=1).cpu().numpy()

    gold_labels = df["label"].to_numpy()
    acc = accuracy_score(gold_labels, pred_labels)

    print("\n評価結果")
    print("=" * 80)
    print(f"accuracy: {acc:.4f}")
    print("=" * 80)

    # 結果を保存
    result_df = df.copy()
    result_df["prompt"] = prompts
    result_df["logprob_negative"] = logprob_matrix[:, 0].cpu().numpy()
    result_df["logprob_positive"] = logprob_matrix[:, 1].cpu().numpy()
    result_df["pred_label"] = pred_labels
    result_df["correct"] = result_df["label"] == result_df["pred_label"]

    output_path = Path(__file__).parent / "96_prompt_sentiment_results.csv"
    result_df.to_csv(output_path, index=False)

    print(f"\n詳細結果を保存しました: {output_path}")

    print("\n予測例")
    print("=" * 80)
    for i in range(min(10, len(result_df))):
        row = result_df.iloc[i]
        print(f"[{i}]")
        print(f"sentence: {row['sentence']}")
        print(f"gold: {row['label']}  pred: {row['pred_label']}  correct: {row['correct']}")
        print(f"logprob_negative: {row['logprob_negative']:.4f}")
        print(f"logprob_positive: {row['logprob_positive']:.4f}")
        print()


if __name__ == "__main__":
    main()

#出力結果
'''
使用デバイス: cuda
SST-2 path: /data/student/i2311021/projects/100-swing-drill/chapter_8/SST-2.zip
評価データ数: 872
                                            sentence  label
0    it 's a charming and often affecting journey .       1
1                 unflinchingly bleak and desperate       0
2  allows us to hope that nolan is poised to emba...      1
3  the acting , costumes , music , cinematography...      1
4                  it 's slow -- very , very slow .       0
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█| 292/292 [00:00<00:00
scoring: 100%|█| 109/109 [00:03<00:00, 27.97i

評価結果
================================================================================
accuracy: 0.5894
================================================================================

詳細結果を保存しました: /data/student/i2311021/projects/100-swing-drill/chapter_10/96_prompt_sentiment_results.csv

予測例
================================================================================
[0]
sentence: it 's a charming and often affecting journey . 
gold: 1  pred: 1  correct: True
logprob_negative: -4.4805
logprob_positive: -3.9080

[1]
sentence: unflinchingly bleak and desperate 
gold: 0  pred: 1  correct: False
logprob_negative: -5.6442
logprob_positive: -5.5828

[2]
sentence: allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . 
gold: 1  pred: 1  correct: True
logprob_negative: -4.1188
logprob_positive: -3.7022

[3]
sentence: the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales . 
gold: 1  pred: 1  correct: True
logprob_negative: -5.0834
logprob_positive: -4.6872

[4]
sentence: it 's slow -- very , very slow . 
gold: 0  pred: 1  correct: False
logprob_negative: -3.8393
logprob_positive: -3.6058

[5]
sentence: although laced with humor and a few fanciful touches , the film is a refreshingly serious look at young women . 
gold: 1  pred: 1  correct: True
logprob_negative: -4.8942
logprob_positive: -4.3142

[6]
sentence: a sometimes tedious film . 
gold: 0  pred: 1  correct: False
logprob_negative: -5.0475
logprob_positive: -4.7635

[7]
sentence: or doing last year 's taxes with your ex-wife . 
gold: 0  pred: 1  correct: False
logprob_negative: -5.4830
logprob_positive: -5.2432

[8]
sentence: you do n't have to know about music to appreciate the film 's easygoing blend of comedy and romance . 
gold: 1  pred: 1  correct: True
logprob_negative: -3.7535
logprob_positive: -3.5503

[9]
sentence: in exactly 89 minutes , most of which passed as slowly as if i 'd been sitting naked on an igloo , formula 51 sank from quirky to jerky to utter turkey . 
gold: 0  pred: 1  correct: False
logprob_negative: -3.3390
logprob_positive: -3.1500
'''