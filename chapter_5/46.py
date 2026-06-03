import os  # 環境変数からAPIキーを取得するために使用
import time  # APIエラー時に一定時間待つために使用

from google import genai  # Gemini APIを利用するためのライブラリ
from google.genai import types  # Gemini APIの生成設定を指定するために使用


# 使用するGeminiモデル
MODEL = "gemini-2.5-flash-lite"

# 川柳は創作的なタスクなので、少しランダム性を持たせる
# 0.0に近いほど安定した出力、1.0に近いほど多様な出力になりやすい
TEMPERATURE = 0.7

# 川柳を作成するお題
THEME = "大学生の日常"

# 環境変数 GEMINI_API_KEY からAPIキーを読み込み、Gemini APIのクライアントを作成する
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def generate_senryu(theme):
    """
    指定したお題に沿って、Geminiに川柳を10個生成させる関数。

    引数:
        theme: 川柳のお題

    戻り値:
        Geminiが生成した川柳の文字列
    """

    # Geminiに入力するプロンプト
    # お題、川柳の形式、出力個数、出力形式を指定している
    prompt = f"""
次のお題に沿って、川柳の案を10個作成してください。

お題: {theme}

条件:
- 川柳らしく、五・七・五に近い形にしてください
- 大学生の日常が伝わる内容にしてください
- ユーモアがあるものにしてください
- 1から10まで番号を付けて出力してください
- 余計な説明は書かず、川柳だけを出力してください
"""

    # APIエラーに備えて最大3回まで再試行する
    for retry in range(3):
        try:
            # Gemini APIを呼び出して、川柳を生成させる
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,  # 創作の多様性を出すために0.7に設定
                    max_output_tokens=500,  # 川柳10個を出力できる程度の上限を指定
                ),
            )

            # 生成結果の前後の空白を取り除いて返す
            return response.text.strip()

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
    実験設定を表示した後、指定したお題で川柳を10個生成し、結果を表示する。
    """

    # 実験設定を表示する
    print("==== 実験設定 ====")
    print(f"モデル: {MODEL}")
    print(f"温度: {TEMPERATURE}")
    print(f"お題: {THEME}")
    print()

    # 指定したお題に沿って川柳を生成する
    result = generate_senryu(THEME)

    # 生成された川柳を表示する
    print("==== 生成結果 ====")
    print(result)


# このファイルを直接実行したときだけ main() を実行する
if __name__ == "__main__":
    main()
#出力結果
'''
==== 実験設定 ====
モデル: gemini-2.5-flash-lite
温度: 0.7
お題: 大学生の日常

==== 生成結果 ====
1.  単位取得 夢と消える 夏の夜
2.  バイト代 消える 飲み会代
3.  課題山 積みで 眠れない
4.  教授の話 どこ吹く風
5.  遅刻魔 単位危うし 春の夢
6.  コンビニ飯 栄養偏り 成長期
7.  サークル活動 恋も仕事も？
8.  ゼミ発表 緊張で声 裏返る
9.  テスト前 徹夜で臨む 運任せ
10. 卒業旅行 貯金足りない 現実味
'''