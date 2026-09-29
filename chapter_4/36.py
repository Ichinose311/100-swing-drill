from wiki_corpus import count_words


def main():
    counter = count_words()

    for word, count in counter.most_common(20):
        print(f"{word}\t{count}")



if __name__ == "__main__":
    main()
