import json
import os
import re
import time

from google import genai
from google.genai import types


MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# 問題46で生成した川柳をここに貼る
SENRYU_TEXT = """
1. 一限目　目覚まし鳴るも　夢の中
2. レポートを　出した直後に　ミス気づく
3. 学食で　財布と相談　今日も水
4. 出席を　取る日だけ来る　友がいる
5. 締切日　なぜか掃除が　はかどる日
6. 教科書を　買っただけでも　勉強感
7. 空きコマで　寝るつもりが　日が暮れる
8. 単位より　睡眠時間　欲しくなる
9. Zoom授業　顔は出さずに　飯を食う
10. 試験前　過去問だけが　神になる
"""


def extract_json(text):
    """
    Geminiの出力からJSON部分を取り出す。
    response_mime_typeを指定しても、念のため保険として用意する。
    """
    text = text.strip()

    # ```json ... ``` のような形式に備える
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    return json.loads(text)


def judge_senryu(senryu_text):
    prompt = f"""
あなたは川柳コンテストの審査員です。
以下の川柳10個について、「面白さ」を10段階で評価してください。

評価基準:
- 1点: ほとんど面白くない
- 5点: 普通
- 10点: 非常に面白い
- 大学生の日常として共感できるか
- オチやユーモアがあるか
- 五・七・五の厳密さよりも、面白さを重視する

出力形式:
必ずJSONのみで出力してください。
説明文や前置きは不要です。

JSON形式:
[
  {{
    "number": 1,
    "senryu": "川柳本文",
    "score": 8,
    "reason": "評価理由"
  }}
]

評価対象:
{senryu_text}
"""

    for retry in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                ),
            )

            return extract_json(response.text)

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

    raise RuntimeError("3回再試行しましたが、API呼び出しに失敗しました。")


def main():
    print("==== 実験設定 ====")
    print(f"モデル: {MODEL}")
    print(f"温度: {TEMPERATURE}")
    print("評価対象: 問題46で生成した川柳10個")
    print()

    results = judge_senryu(SENRYU_TEXT)

    print("==== 評価結果 ====")

    total_score = 0

    for item in results:
        number = item["number"]
        senryu = item["senryu"]
        score = item["score"]
        reason = item["reason"]

        total_score += score

        print(f"{number}. {senryu}")
        print(f"   面白さ: {score}/10")
        print(f"   理由: {reason}")
        print()

    average = total_score / len(results)

    print("==== 集計 ====")
    print(f"平均点: {average:.2f}/10")


if __name__ == "__main__":
    main()
