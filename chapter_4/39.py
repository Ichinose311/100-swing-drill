import matplotlib.pyplot as plt

from wiki_corpus import count_words

def main():
    # コーパス全体における単語の出現頻度
    counter = count_words()

    # 出現頻度の高い順に並べる
    frequencies = [count for word, count in counter.most_common()]

    # 順位: 1, 2, 3, ...
    ranks = range(1, len(frequencies) + 1)

    # 両対数グラフを描画
    plt.figure()
    plt.scatter(ranks, frequencies, s=5)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf's law")

    plt.savefig("39_zipf.png")
    plt.show()


if __name__ == "__main__":
    main()