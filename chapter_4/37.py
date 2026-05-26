from wiki_corpus import count_words

counter = count_words(pos="名詞")

for word, count in counter.most_common(20):
    print(f"{word}\t{count}")

# 出力結果
'''
年      23659
月      9443
語      5574
1       5461
日本    4768
こと    4477
2       4260
世界    3604
3       3520
政府    3331
大統領  3144
共和    3117
5       3060
ため    3051
4       2930
経済    2703
州      2625
万      2567
アメリカ        2503
独立    2424
'''