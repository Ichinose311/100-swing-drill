import csv  # CSVファイルを読み込むための標準ライブラリ
import os  # 環境変数からAPIキーを取得するために使用
import re  # LLMの出力からA/B/C/Dを抽出するために使用
import time  # API呼び出し間隔を空けるために使用
from collections import Counter  # 予測ラベルの出現回数を数えるために使用
from pathlib import Path  # ファイルパスを扱いやすくするために使用

from google import genai  # Gemini APIを利用するためのライブラリ
from google.genai import types  # Gemini APIの生成設定を指定するために使用


# 使用するJMMLUの科目名
SUBJECT = "high_school_computer_science"

# 使用する大規模言語モデル
MODEL = "gemini-2.5-flash-lite"

# 評価する問題数
# 最初は20問で試す。全問で評価したい場合は None にする。
MAX_QUESTIONS = 20

# API呼び出しの間隔
# 無料枠やレート制限に引っかかりにくくするために使用する
REQUEST_INTERVAL = 7

# 生成のランダム性を制御する温度パラメータ
# 0.0にすると、できるだけ決定的な出力になりやすい
TEMPERATURE = 0.0

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
      "D"
      "答えはDです"
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


def move_correct_answer_to_d(a, b, c, d, gold):
    """
    正解の選択肢を必ずDに移動する関数。

    目的:
      元の問題では正解がA/B/C/Dのいずれかに分散している。
      ここでは、実験設定として「正解が必ずDになる」ように選択肢を並べ替える。
      これにより、選択肢の位置や記号によって正解率が変わるかを調べる。

    例:
      元の選択肢:
        A = 選択肢1
        B = 選択肢2  ← 正解
        C = 選択肢3
        D = 選択肢4

      変換後:
        A = 元のA
        B = 元のC
        C = 元のD
        D = 元のB  ← 正解
    """

    # 選択肢のラベル一覧
    labels = ["A", "B", "C", "D"]

    # ラベルと選択肢本文を対応させる
    options = {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
    }

    # 元の正解ラベルに対応する選択肢本文を取り出す
    correct_text = options[gold]

    # 正解以外の選択肢本文を取り出す
    wrong_texts = [options[label] for label in labels if label != gold]

    # 不正解選択肢をA〜Cに配置する
    new_a = wrong_texts[0]
    new_b = wrong_texts[1]
    new_c = wrong_texts[2]

    # 正解選択肢をDに配置する
    new_d = correct_text

    # 並べ替え後の正解ラベルは必ずDになる
    new_gold = "D"

    # 並べ替え後の選択肢と正解ラベルを返す
    return new_a, new_b, new_c, new_d, new_gold


def ask_llm(question, a, b, c, d):
    """
    1問分の4択問題をGeminiに解答させる関数。
    問題文と並べ替え後の選択肢A〜Dをプロンプトに入れ、
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

    # APIエラーに備えて最大3回まで再試行する
    for retry in range(3):
        try:
            # Gemini APIを呼び出して、モデルに解答を生成させる
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,  # 出力のランダム性を抑える
                    max_output_tokens=5,  # A/B/C/Dだけを出してほしいので出力を短く制限する
                ),
            )

            # モデルの出力からA/B/C/Dのいずれかを抽出して返す
            return normalize_answer(response.text)

        except Exception as e:
            # エラーメッセージを文字列として取得する
            error_message = str(e)

            # レート制限に達した場合は60秒待って再試行する
            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                print("レート制限に達しました。60秒待って再試行します。")
                time.sleep(60)

            # サーバー混雑などの一時的なエラーの場合は30秒待って再試行する
            elif "503" in error_message or "UNAVAILABLE" in error_message:
                print("サーバー混雑です。30秒待って再試行します。")
                time.sleep(30)

            # それ以外のエラーは想定外なので、そのままエラーとして投げる
            else:
                raise e

    # 3回試しても成功しなかった場合は空文字を返す
    return ""


def main():
    """
    メイン処理。
    JMMLUのCSVファイルを読み込み、各問題について正解選択肢をDに移動する。
    そのうえでLLMに解答させ、正解率と予測ラベルの分布を計算する。
    """

    # 対象科目のCSVファイルを探す
    csv_path = find_csv_path(SUBJECT)

    # 解答した問題数
    total = 0

    # 正解した問題数
    correct = 0

    # LLMがA/B/C/Dをそれぞれ何回選んだかを数える
    pred_counter = Counter()

    # 実験設定を表示する
    print("==== 実験設定 ====")
    print(f"科目: {SUBJECT}")
    print(f"モデル: {MODEL}")
    print(f"温度: {TEMPERATURE}")
    print("設定: 正解選択肢をすべてDに移動")
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

            # 元の正解ラベルを大文字に統一する
            gold = gold.strip().upper()

            # ヘッダー行がある場合に備えてスキップする
            if gold == "正解":
                continue

            # 元の正解選択肢をDに移動する
            # これにより、並べ替え後の正解ラベルは必ずDになる
            new_a, new_b, new_c, new_d, new_gold = move_correct_answer_to_d(
                a, b, c, d, gold
            )

            # 並べ替え後の問題をLLMに解かせる
            pred = ask_llm(question, new_a, new_b, new_c, new_d)

            # 解答した問題数を1増やす
            total += 1

            # 予測ラベルの出現回数を記録する
            pred_counter[pred] += 1

            # 予測ラベルと並べ替え後の正解ラベルDが一致しているか判定する
            is_correct = pred == new_gold

            # 正解していた場合は正解数を1増やす
            if is_correct:
                correct += 1

            # 各問題の予測、並べ替え後の正解、元の正解、正誤を表示する
            print(
                f"{total}: pred={pred}, gold={new_gold}, "
                f"original_gold={gold}, {'OK' if is_correct else 'NG'}"
            )

            # 無料枠・レート制限対策として、API呼び出し間隔を空ける
            time.sleep(REQUEST_INTERVAL)

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

    # 予測ラベルの分布を表示する
    # これを見ることで、LLMが特定の選択肢を選びやすいかを確認できる
    print()
    print("==== 予測ラベルの分布 ====")
    for label in ["A", "B", "C", "D", ""]:
        if label in pred_counter:
            shown_label = label if label != "" else "抽出失敗"
            print(f"{shown_label}: {pred_counter[label]}")


# このファイルを直接実行したときだけ main() を実行する
if __name__ == "__main__":
    main()
    
#出力結果
'''
==== 実験設定 ====
科目: high_school_computer_science
モデル: gemini-2.5-flash-lite
温度: 0.0
設定: 正解選択肢をすべてDに移動
CSV: ../data/JMMLU/JMMLU/high_school_computer_science.csv

1: pred=D, gold=D, original_gold=A, OK
2: pred=D, gold=D, original_gold=A, OK
3: pred=C, gold=D, original_gold=C, NG
4: pred=D, gold=D, original_gold=B, OK
5: pred=D, gold=D, original_gold=C, OK
6: pred=D, gold=D, original_gold=B, OK
7: pred=D, gold=D, original_gold=B, OK
8: pred=D, gold=D, original_gold=A, OK
9: pred=D, gold=D, original_gold=B, OK
10: pred=D, gold=D, original_gold=B, OK
11: pred=D, gold=D, original_gold=C, OK
12: pred=D, gold=D, original_gold=C, OK
13: pred=D, gold=D, original_gold=B, OK
14: pred=D, gold=D, original_gold=C, OK
15: pred=D, gold=D, original_gold=A, OK
16: pred=D, gold=D, original_gold=D, OK
17: pred=D, gold=D, original_gold=D, OK
18: pred=D, gold=D, original_gold=D, OK
19: pred=D, gold=D, original_gold=A, OK
20: pred=D, gold=D, original_gold=C, OK

==== 結果 ====
科目: high_school_computer_science
問題数: 20
正解数: 19
正解率: 0.950
正解率(%): 95.0%

==== 予測ラベルの分布 ====
C: 1
D: 19
'''
