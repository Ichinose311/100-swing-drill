import os
import time

from google import genai
from google.genai import types


MODEL = "gemini-2.5-flash-lite"

# 川柳は少し創造性が必要なので、温度は0.7くらいにする
TEMPERATURE = 0.7

# お題
THEME = "電気通信大学稲葉研"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def generate_senryu(theme):
    prompt = f"""
次のお題に沿って、川柳の案を10個作成してください。

お題: {theme}

条件:
- 川柳らしく、五・七・五に近い形にしてください
- 電気通信大学稲葉研が伝わる内容にしてください
- ユーモアがあるものにしてください
- 1から10まで番号を付けて出力してください
- 余計な説明は書かず、川柳だけを出力してください
"""

    for retry in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=500,
                ),
            )

            return response.text.strip()

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
    print(f"お題: {THEME}")
    print()

    result = generate_senryu(THEME)

    print("==== 生成結果 ====")
    print(result)


if __name__ == "__main__":
    main()