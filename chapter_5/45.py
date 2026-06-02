import os
import re
import time

from google import genai
from google.genai import types


MODEL = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def normalize_answer(text):
    """
    LLMの出力から駅名を取り出す。
    今回は「緑が丘駅」または「緑が丘」が含まれていれば、それを答えとして扱う。
    """
    text = text.strip()

    if "緑が丘" in text or "緑ケ丘" in text:
        return "緑が丘駅"

    return text


def ask_llm():
    prompt = """
以下の問いかけに対する応答を生成してください。
答えを明確に述べ、簡単に理由も説明してください。

前提:
先ほどの問いでは、つばめちゃんの目的地は「緑が丘駅」である。

追加の問い:
さらに、つばめちゃんが自由が丘駅で乗り換えたとき、
先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。
目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、
反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
"""

    for retry in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=200,
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
    answer = ask_llm()
    normalized = normalize_answer(answer)

    print("==== LLMの応答 ====")
    print(answer)

    print()
    print("==== 抽出した答え ====")
    print(normalized)


if __name__ == "__main__":
    main()