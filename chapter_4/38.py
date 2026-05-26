import math
from collections import Counter

import MeCab

from wiki_corpus import iter_articles, remove_markup

def extract_nouns(text, tagger):
    nouns = []

    node = tagger.parseToNode(text)

    while node:
        surface = node.surface
        features = node.feature.split(",")

        if surface != "" and features[0] == "名詞":
            nouns.append(surface)

        node = node.next

    return nouns


def main():
    tagger = MeCab.Tagger()

    # コーパス全体の記事数
    total_articles = 0

    # 各名詞が何記事に出現したか
    document_frequency = Counter()

    # 日本記事の名詞リスト
    japan_nouns = None

    for article in iter_articles():
        total_articles += 1

        title = article["title"]
        text = remove_markup(article["text"])

        nouns = extract_nouns(text, tagger)

        # 同じ記事内で同じ名詞が複数回出ても、DFでは1回だけ数える
        unique_nouns = set(nouns)
        document_frequency.update(unique_nouns)

        if title == "日本":
            japan_nouns = nouns

    if japan_nouns is None:
        raise ValueError("title が '日本' の記事が見つかりませんでした。")

    # 日本記事中の名詞出現回数
    term_frequency_count = Counter(japan_nouns)

    # 日本記事中の名詞総数
    total_nouns_in_japan = sum(term_frequency_count.values())

    results = []

    '''
    TF = 日本記事中のその名詞の出現回数 / 日本記事中の名詞総数
    IDF = log(全記事数 / その名詞が出現する記事数)
    TF-IDF = TF * IDF
    '''
    for noun, count in term_frequency_count.items():
        tf = count / total_nouns_in_japan
        idf = math.log(total_articles / document_frequency[noun])
        tf_idf = tf * idf

        results.append((noun, tf, idf, tf_idf))

    # TF-IDFスコアの高い順に並べる
    results.sort(key=lambda x: x[3], reverse=True)

    print("名詞\tTF\tIDF\tTF-IDF")
    for noun, tf, idf, tf_idf in results[:20]:
        print(f"{noun}\t{tf:.6f}\t{idf:.6f}\t{tf_idf:.6f}")


if __name__ == "__main__":
    main()

# 出力結果
