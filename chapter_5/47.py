import json  # Geminiの出力をJSONとして読み込むために使用
import os  # 環境変数からAPIキーを取得するために使用
import re  # Geminiの出力から余計な記号を取り除くために使用
import time  # APIエラー時に一定時間待つために使用

from google import genai  # Gemini APIを利用するためのライブラリ
from google.genai import types  # Gemini APIの生成設定を指定するために使用


# 使用するGeminiモデル
MODEL = "gemini-2.5-flash-lite"

# 評価タスクでは出力のばらつきを抑えたいので、温度は0.0にする
TEMPERATURE = 0.0

# 環境変数 GEMINI_API_KEY からAPIキーを読み込み、Gemini APIのクライアントを作成する
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# 問題46で生成した川柳をここに貼る
# この10個の川柳をLLMに評価させる
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
    Geminiの出力からJSON部分を取り出す関数。

    response_mime_type="application/json" を指定していても、
    念のため ```json ... ``` のようなコードブロック形式で返ってきた場合に備えて、
    余計な記号を取り除いてから json.loads() で読み込む。
    """

    # 前後の空白や改行を取り除く
    text = text.strip()

    # ```json から始まる場合に備えて削除する
    text = re.sub(r"^```json", "", text)

    # ``` から始まる場合に備えて削除する
    text = re.sub(r"^```", "", text)

    # 末尾の ``` を削除する
    text = re.sub(r"```$", "", text)

    # 再度、前後の空白を取り除く
    text = text.strip()

    # JSON文字列をPythonのリストや辞書に変換する
    return json.loads(text)


def judge_senryu(senryu_text):
    """
    Geminiを川柳コンテストの審査員として使い、
    川柳10個の面白さを10段階で評価させる関数。

    引数:
        senryu_text: 評価対象の川柳10個

    戻り値:
        各川柳の番号、本文、点数、評価理由を含むリスト
    """

    # Geminiに入力するプロンプト
    # 評価基準とJSON形式を明示して、10個の川柳を評価させる
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

    # APIエラーに備えて最大3回まで再試行する
    for retry in range(3):
        try:
            # Gemini APIを呼び出して、川柳の評価を生成させる
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,  # 評価のばらつきを抑えるため0.0に設定
                    max_output_tokens=2000,  # 10個分の評価理由を出力できるように大きめに設定
                    response_mime_type="application/json",  # JSON形式で返すように指定
                ),
            )

            # Geminiの出力をJSONとして読み込み、Pythonのデータに変換して返す
            return extract_json(response.text)

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

    # 3回再試行しても成功しなかった場合はエラーにする
    raise RuntimeError("3回再試行しましたが、API呼び出しに失敗しました。")


def main():
    """
    メイン処理。
    問題46で生成した川柳10個をLLMに評価させ、
    各川柳の点数と評価理由、平均点を表示する。
    """

    # 実験設定を表示する
    print("==== 実験設定 ====")
    print(f"モデル: {MODEL}")
    print(f"温度: {TEMPERATURE}")
    print("評価対象: 問題46で生成した川柳10個")
    print()

    # LLMに川柳を評価させる
    results = judge_senryu(SENRYU_TEXT)

    # 評価結果を表示する
    print("==== 評価結果 ====")

    # 平均点を計算するための合計点
    total_score = 0

    # 各川柳について、番号・本文・点数・理由を取り出して表示する
    for item in results:
        number = item["number"]
        senryu = item["senryu"]
        score = item["score"]
        reason = item["reason"]

        # 合計点に加算する
        total_score += score

        # 1つの川柳に対する評価結果を表示する
        print(f"{number}. {senryu}")
        print(f"   面白さ: {score}/10")
        print(f"   理由: {reason}")
        print()

    # 10個の川柳の平均点を計算する
    average = total_score / len(results)

    # 集計結果を表示する
    print("==== 集計 ====")
    print(f"平均点: {average:.2f}/10")


# このファイルを直接実行したときだけ main() を実行する
if __name__ == "__main__":
    main()
#出力結果
