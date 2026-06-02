import gzip
import json
import re
from collections import Counter
from pathlib import Path

import MeCab


def remove_markup(text):
    # 強調マークアップ除去
    text = re.sub(r"'{2,5}", "", text)

    # refタグ除去
    text = re.sub(r"<ref[^>/]*?/>", "", text)
    text = re.sub(r"<ref[^>]*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<references\s*/>", "", text)

    # ファイル・画像リンク除去
    text = re.sub(r"\[\[(?:File|ファイル|Image|画像):.*?\]\]", "", text, flags=re.IGNORECASE)

    # カテゴリリンク除去
    text = re.sub(r"\[\[(?:Category|カテゴリ):.*?\]\]", "", text, flags=re.IGNORECASE)

    # 外部リンク除去
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", "", text)

    # 内部リンク除去: [[記事名|表示文字]] → 表示文字
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)

    # 内部リンク除去: [[記事名]] → 記事名
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # テンプレート除去・簡易変換
    # {{lang|en|United Kingdom}} → United Kingdom のように最後の要素を残す
    old = None
    while old != text:
        old = text
        text = re.sub(r"\{\{[^{}|]*\|(?:[^{}|]*\|)*([^{}|]*)\}\}", r"\1", text)
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # HTMLタグ除去
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)

    # 見出しマークアップ除去
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1", text)

    return text


def find_data_file():
    candidates = [
        Path("jawiki-country.json.gz"),
        Path(__file__).resolve().parent / "jawiki-country.json.gz",
        Path(__file__).resolve().parent.parent / "jawiki-country.json.gz",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("jawiki-country.json.gz が見つかりません。")


def iter_articles():
    data_path = find_data_file()

    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        # JSONL形式で1行ずつ記事が格納されているので、1行ずつ読み込んでJSONとしてパースする
        '''
        {"title": "イギリス", "text": "イギリス（英: United Kingdom、正式名称: グレートブリテン及び北アイルランド連合王国）は、ヨーロッパの島国である。 ..."}'''
        for line in f:
            yield json.loads(line)

# 記事本文をMeCabで形態素解析して、単語と品詞を1つずつ返す関数
def iter_morphs():
    tagger = MeCab.Tagger()

    for article in iter_articles():
        text = remove_markup(article["text"]) #マークアップを除去したテキスト
        node = tagger.parseToNode(text) #形態素解析して、単語がnodeとして連結されている

        while node:
            surface = node.surface
            features = node.feature.split(",")
            pos = features[0]

            if surface != "":
                yield surface, pos

            node = node.next

# 単語の出現回数を数える関数
def count_words(pos=None, exclude_symbols=True):
    counter = Counter() #単語の出現回数を数えるためのCounterオブジェクトを作る
    # iter_morphs()を呼び出して、単語と品詞を1つずつ取得する
    for surface, part_of_speech in iter_morphs():
        # 品詞が記号の場合は数えないようにする
        if exclude_symbols and part_of_speech in ["記号", "補助記号"]:
            continue
        # 品詞に限らず全部数える
        if pos is None or part_of_speech == pos:
            counter[surface] += 1

    return counter
