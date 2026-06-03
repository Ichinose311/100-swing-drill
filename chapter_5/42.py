import csv  # CSVファイルを読み込むための標準ライブラリ
import os  # 環境変数からAPIキーを取得するために使用
import re  # LLMの出力からA/B/C/Dを抽出するために使用
import time  # API呼び出し間隔を空けるために使用
from pathlib import Path  # ファイルパスを扱いやすくするために使用

from google import genai  # Gemini APIを利用するためのライブラリ


# 使用するJMMLUの科目名
SUBJECT = "high_school_computer_science"

# 使用する大規模言語モデル
MODEL = "gemini-2.5-flash-lite"

# 評価する問題数
# 無料枠や503エラー対策のため、最初は20問だけで試す
# 全問で評価したい場合は None にする
MAX_QUESTIONS = 20

# 環境変数 GEMINI_API_KEY からAPIキーを読み込み、Gemini APIのクライアントを作成する
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def find_csv_path(subject):
    """
    指定した科目のCSVファイルを探す関数。
    実行場所によって相対パスが変わる可能性があるため、
    複数の候補パスを順番に確認する。
    """

    candidates = [
        Path("data/JMMLU/JMMLU") / f"{subject}.csv",
        Path("data/JMMLU/JMMLU_NC_ND") / f"{subject}.csv",
        Path("../data/JMMLU/JMMLU") / f"{subject}.csv",
        Path("../data/JMMLU/JMMLU_NC_ND") / f"{subject}.csv",
    ]

    # 候補パスの中から実際に存在するファイルを返す
    for path in candidates:
        if path.exists():
            return path

    # どの候補にもファイルが存在しなかった場合はエラーにする
    raise FileNotFoundError(f"{subject}.csv が見つかりません。")


def normalize_answer(text):
    """
    LLMの出力から A/B/C/D のどれかを取り出す関数。
    例:
      "A"
      "答えはBです"
      "解答: C"
    のような出力から、最初に出てきたA/B/C/Dを抽出する。
    """

    # 前後の空白を削除し、大文字に統一する
    text = text.strip().upper()

    # 出力中に含まれる A/B/C/D のいずれかを探す
    match = re.search(r"[ABCD]", text)

    # 見つかった場合は、その文字を返す
    if match:
        return match.group(0)

    # A/B/C/D が見つからなかった場合は空文字を返す
    return ""


def ask_llm(question, a, b, c, d):
    """
    1問分の4択問題をGeminiに解答させる関数。
    問題文と選択肢A〜Dをプロンプトに入れ、
    正解だと思う選択肢を1文字だけ出力するように指示する。
    """

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

    # Gemini APIを呼び出して、モデルに解答を生成させる
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    # モデルの出力からA/B/C/Dのいずれかを抽出して返す
    return normalize_answer(response.text)


def main():
    """
    メイン処理。
    JMMLUのCSVファイルを読み込み、各問題をLLMに解答させる。
    予測ラベルと正解ラベルを比較し、正解率を計算する。
    """

    # 対象科目のCSVファイルを探す
    csv_path = find_csv_path(SUBJECT)

    # 解答した問題数
    total = 0

    # 正解した問題数
    correct = 0

    # 実験設定を表示する
    print(f"科目: {SUBJECT}")
    print(f"モデル: {MODEL}")
    print(f"CSV: {csv_path}")
    print()

    # CSVファイルを開いて、1行ずつ問題を読み込む
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        # iはCSV上の行番号、rowは1行分のデータ
        for i, row in enumerate(reader, start=1):

            # 指定した問題数に達したら終了する
            if MAX_QUESTIONS is not None and total >= MAX_QUESTIONS:
                break

            # 1行に「問題文、A、B、C、D、正解」の6要素がない場合はスキップする
            if len(row) < 6:
                continue

            # CSVの先頭6列を取り出す
            question, a, b, c, d, gold = row[:6]

            # 正解ラベルを大文字に統一する
            gold = gold.strip().upper()

            # ヘッダー行がある場合に備えてスキップする
            if gold == "正解":
                continue

            try:
                # LLMに問題を解かせ、予測ラベルを取得する
                pred = ask_llm(question, a, b, c, d)

            except Exception as e:
                # APIエラーなどが発生した場合は、エラー内容を表示して次の問題へ進む
                print(f"{i}問目でエラー: {e}")
                time.sleep(5)
                continue

            # 解答に成功した問題数を1増やす
            total += 1

            # 予測ラベルと正解ラベルが一致しているか判定する
            is_correct = pred == gold

            # 正解していた場合は正解数を1増やす
            if is_correct:
                correct += 1

            # 各問題の予測、正解、正誤を表示する
            print(f"{total}: pred={pred}, gold={gold}, {'OK' if is_correct else 'NG'}")

            # 無料枠・レート制限対策として、API呼び出し間隔を空ける
            time.sleep(1)

    # 正解率を計算する
    accuracy = correct / total if total > 0 else 0

    # 最終結果を表示する
    print()
    print("==== 結果 ====")
    print(f"科目: {SUBJECT}")
    print(f"問題数: {total}")
    print(f"正解数: {correct}")
    print(f"正解率: {accuracy:.3f}")
    print(f"正解率(%): {accuracy * 100:.1f}%")


# このファイルを直接実行したときだけ main() を実行する
if __name__ == "__main__":
    main()

# 出力結果
