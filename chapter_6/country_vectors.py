"""Shared country selection for exercises 57–59."""

def load_country_vectors(questions_path, model):
    # 国名を抽出する対象セクション
    target_sections = {
        "capital-common-countries",
        "capital-world"
    }

    current_section = None
    countries = set()

    # questions-words.txt から国名を抽出
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith(":"):
                current_section = line[2:]
                continue

            if current_section not in target_sections:
                continue

            words = line.split()

            if len(words) != 4:
                continue

            word1, word2, word3, word4 = words

            # capital系のセクションでは、
            # 2列目と4列目が国名
            countries.add(word2)
            countries.add(word4)

    # モデルに存在する国名だけを使う
    country_names = []
    country_vectors = []

    for country in sorted(countries):
        if country in model.key_to_index:
            country_names.append(country)
            country_vectors.append(model[country])

    return country_names, country_vectors
