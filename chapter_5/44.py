import os  # 環境変数からAPIキーを取得するために使用
import re  # 文字列検索に使用。今回はほぼ使っていないが、出力整形などに利用可能
import time  # APIエラー時に一定時間待つために使用

from google import genai  # Gemini APIを利用するためのライブラリ
from google.genai import types  # Gemini APIの生成設定を指定するために使用


# 使用するGeminiモデル
MODEL = "gemini-2.5-flash-lite"

# 生成のランダム性を制御する温度パラメータ
# 0.0にすると、できるだけ決定的な応答になりやすい
TEMPERATURE = 0.0

# 環境変数 GEMINI_API_KEY からAPIキーを読み込み、Gemini APIのクライアントを作成する
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def normalize_answer(text):
    """
    LLMの出力から駅名を取り出す関数。

    今回の問題では正解が「緑が丘駅」なので、
    LLMの応答に「緑が丘」または「緑ケ丘」が含まれていれば、
    正規化した答えとして「緑が丘駅」を返す。

    それ以外の場合は、LLMの出力をそのまま返す。
    """

    # 前後の空白や改行を取り除く
    text = text.strip()

    # 「緑が丘」または表記ゆれの「緑ケ丘」が含まれているか確認する
    if "緑が丘" in text or "緑ケ丘" in text:
        return "緑が丘駅"

    # 該当しない場合は、元の応答をそのまま返す
    return text


def ask_llm():
    """
    Geminiに鉄道路線に関する推論問題を解かせる関数。

    問題文をプロンプトとして与え、
    目的地の駅名と簡単な理由を答えるように指示する。
    """

    # Geminiに入力するプロンプト
    # 自由が丘から大井町方面の急行に間違えて乗った場合の目的地を推論させる
    prompt = """
以下の問いかけに対する応答を生成してください。
答えの駅名を明確に述べ、簡単に理由も説明してください。

問い:
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
つばめちゃんの目的地でした。
目的地の駅の名前を答えてください。

参考: 東急線・みなとみらい線路線案内
"""

    # APIエラーに備えて最大3回まで再試行する
    for retry in range(3):
        try:
            # Gemini APIを呼び出して応答を生成する
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,  # 出力のランダム性を抑える
                    max_output_tokens=200,  # 駅名と理由を出すため、出力上限を200トークンにする
                ),
            )

            # 生成された応答の前後の空白を取り除いて返す
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
    Geminiに問題を解かせ、応答全文と抽出した答えを表示する。
    """

    # Geminiに問題を解かせる
    answer = ask_llm()

    # LLMの応答から駅名を抽出・正規化する
    normalized = normalize_answer(answer)

    # LLMが生成した応答全文を表示する
    print("==== LLMの応答 ====")
    print(answer)

    print()

    # 抽出した答えだけを表示する
    print("==== 抽出した答え ====")
    print(normalized)


# このファイルを直接実行したときだけ main() を実行する
if __name__ == "__main__":
    main()
    
#出力結果
