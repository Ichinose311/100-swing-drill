from wiki_corpus import count_words

counter = count_words()

for word, count in counter.most_common(20):
    print(f"{word}\t{count}")

# 出力結果
