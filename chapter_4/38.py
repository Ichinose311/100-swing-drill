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
'''
名詞    TF      IDF     TF-IDF
天皇    0.002770        2.805379        0.007770
倭国    0.001296        4.820282        0.006245
朝鮮    0.002725        2.255332        0.006146
列島    0.001876        3.028522        0.005682
書紀    0.000983        5.513429        0.005419
倭      0.001072        4.820282        0.005168
琉球    0.001161        4.414816        0.005128
明治    0.002189        2.335375        0.005112
沖縄    0.001653        2.805379        0.004637
県      0.007371        0.608154        0.004483
北海道  0.001519        2.623057        0.003984
千島    0.000804        4.820282        0.003876
樺太    0.000938        4.127134        0.003872
韓国    0.002323        1.642228        0.003815
昭和    0.001430        2.568990        0.003672
政令    0.001161        3.115533        0.003619
台湾    0.002278        1.543137        0.003516
日本    0.039267        0.088479        0.003474
秒      0.000625        5.513429        0.003448
アイヌ  0.000715        4.820282        0.003445
'''