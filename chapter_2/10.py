from names_data import DATA_PATH


def main():
    with DATA_PATH.open(encoding="utf-8") as source:
        line_count = sum(1 for _ in source)
    print(line_count)


if __name__ == "__main__":
    main()
