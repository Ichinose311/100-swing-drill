import csv
import os
import re
import time
from pathlib import Path

from google import genai


SUBJECT = "high_school_computer_science"
MODEL = "gemini-2.5-flash-lite"

# 無料枠や503対策のため、最初は20問だけで試す
# 全問で評価したい場合は None にする
MAX_QUESTIONS = None

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
    """
    LLMの出力から A/B/C/D のどれかを取り出す。
    """
    text = text.strip().upper()

    # 例: "解答: A", "答えはBです", "C"
    match = re.search(r"[ABCD]", text)
    if match:
        return match.group(0)

    return ""


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

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    return normalize_answer(response.text)


def main():
    csv_path = find_csv_path(SUBJECT)

    total = 0
    correct = 0

    print(f"科目: {SUBJECT}")
    print(f"モデル: {MODEL}")
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

            # ヘッダー行がある場合に備える
            if gold == "正解":
                continue

            try:
                pred = ask_llm(question, a, b, c, d)
            except Exception as e:
                print(f"{i}問目でエラー: {e}")
                time.sleep(5)
                continue

            total += 1
            is_correct = pred == gold

            if is_correct:
                correct += 1

            print(f"{total}: pred={pred}, gold={gold}, {'OK' if is_correct else 'NG'}")

            # 無料枠・レート制限対策
            time.sleep(1)

    accuracy = correct / total if total > 0 else 0

    print()
    print("==== 結果 ====")
    print(f"科目: {SUBJECT}")
    print(f"問題数: {total}")
    print(f"正解数: {correct}")
    print(f"正解率: {accuracy:.3f}")
    print(f"正解率(%): {accuracy * 100:.1f}%")


if __name__ == "__main__":
    main()

# 出力結果
