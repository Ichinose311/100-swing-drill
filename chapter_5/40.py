import os
from google import genai


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = """
以下の日本史の問題に答えてください。
ただし、zero-shot推論として、例題や解答例は与えず、この問題文だけから解答を生成してください。

問題:
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。

出力形式:
解答: ○→○→○
理由: 簡潔に説明
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

print(response.text)

# 出力結果
