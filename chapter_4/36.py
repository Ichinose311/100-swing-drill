from wiki_corpus import count_words

counter = count_words()

for word, count in counter.most_common(20):
    print(f"{word}\t{count}")

# 出力結果
'''
の      86559
に      60962
は      50525
が      42895
を      37340
て      35234
た      34875
で      33511
と      33189
し      29381
年      23659
れ      13568
いる    13117
さ      12280
ある    11478
も      10803
月      9443
する    9217
人      8987
から    8381
'''