import csv
import os
import re
import time
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types


SUBJECT = "high_school_computer_science"
MODEL = "gemini-2.5-flash-lite"

# 最初は20問で試す。全問なら None にする。
MAX_QUESTIONS = 20

# 無料枠対策
REQUEST_INTERVAL = 7

# 実験設定
TEMPERATURE = 0.0

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def find_csv_path(subject):
    candidates = [
        Path("data/JMMLU/JMMLU") / f"{subject}.csv",
        Path("data/JMMLU/JMMLU_NC_ND") / f"{subject}.csv",
        Path("../data/JMMLU/JMMLU") / f"{subject}.csv",
        Path("../data/JMMLU/JMMLU_NC_ND") / f"{subject}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"{subject}.csv が見つかりません。")


def normalize_answer(text):
    text = text.strip().upper()
    match = re.search(r"[ABCD]", text)
    if match:
        return match.group(0)
    return ""


def move_correct_answer_to_d(a, b, c, d, gold):
    """
    正解の選択肢を必ずDに移動する。
    例:
      gold == "B" の場合
      D = 元のB
      A, B, C = 元のA, C, D
    """
    labels = ["A", "B", "C", "D"]
    options = {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
    }

    correct_text = options[gold]
    wrong_texts = [options[label] for label in labels if label != gold]

    new_a = wrong_texts[0]
    new_b = wrong_texts[1]
    new_c = wrong_texts[2]
    new_d = correct_text

    new_gold = "D"

    return new_a, new_b, new_c, new_d, new_gold


def ask_llm(question, a, b, c, d):
    prompt = f"""
以下の4択問題に答えてください。
正解だと思う選択肢を A, B, C, D のいずれか1文字だけで出力してください。
理由や説明は出力しないでください。

問題:
{question}

A. {a}
B. {b}
C. {c}
D. {d}

答え:
"""

    for retry in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=5,
                ),
            )
            return normalize_answer(response.text)

        except Exception as e:
            error_message = str(e)

            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                print("レート制限に達しました。60秒待って再試行します。")
                time.sleep(60)
            elif "503" in error_message or "UNAVAILABLE" in error_message:
                print("サーバー混雑です。30秒待って再試行します。")
                time.sleep(30)
            else:
                raise e

    return ""


def main():
    csv_path = find_csv_path(SUBJECT)

    total = 0
    correct = 0
    pred_counter = Counter()

    print("==== 実験設定 ====")
    print(f"科目: {SUBJECT}")
    print(f"モデル: {MODEL}")
    print(f"温度: {TEMPERATURE}")
    print("設定: 正解選択肢をすべてDに移動")
    print(f"CSV: {csv_path}")
    print()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader, start=1):
            if MAX_QUESTIONS is not None and total >= MAX_QUESTIONS:
                break

            if len(row) < 6:
                continue

            question, a, b, c, d, gold = row[:6]
            gold = gold.strip().upper()

            # ヘッダー行対策
            if gold == "正解":
                continue

            # 正解をDに移動
            new_a, new_b, new_c, new_d, new_gold = move_correct_answer_to_d(
                a, b, c, d, gold
            )

            pred = ask_llm(question, new_a, new_b, new_c, new_d)

            total += 1
            pred_counter[pred] += 1

            is_correct = pred == new_gold

            if is_correct:
                correct += 1

            print(
                f"{total}: pred={pred}, gold={new_gold}, "
                f"original_gold={gold}, {'OK' if is_correct else 'NG'}"
            )

            time.sleep(REQUEST_INTERVAL)

    accuracy = correct / total if total > 0 else 0

    print()
    print("==== 結果 ====")
    print(f"科目: {SUBJECT}")
    print(f"問題数: {total}")
    print(f"正解数: {correct}")
    print(f"正解率: {accuracy:.3f}")
    print(f"正解率(%): {accuracy * 100:.1f}%")

    print()
    print("==== 予測ラベルの分布 ====")
    for label in ["A", "B", "C", "D", ""]:
        if label in pred_counter:
            shown_label = label if label != "" else "抽出失敗"
            print(f"{shown_label}: {pred_counter[label]}")


if __name__ == "__main__":
    main()
