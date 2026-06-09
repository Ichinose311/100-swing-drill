from pathlib import Path

from gensim.models import KeyedVectors
from sklearn.cluster import KMeans


def main():
    base_dir = Path(__file__).parent

    vector_path = base_dir / "GoogleNews-vectors-negative300.bin.gz"
    questions_path = base_dir / "questions-words.txt"
    output_path = base_dir / "country_kmeans_result.txt"

    # 学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        vector_path,
        binary=True
    )

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

    print(f"抽出した国名数: {len(country_names)}")

    # k-meansクラスタリング
    kmeans = KMeans(
        n_clusters=5,
        random_state=0,
        n_init=10
    )

    labels = kmeans.fit_predict(country_vectors)

    # 結果をファイルに保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("country\tcluster\n")

        for country, label in zip(country_names, labels):
            f.write(f"{country}\t{label}\n")

    # クラスタごとに表示
    for cluster_id in range(5):
        print()
        print(f"クラスタ {cluster_id}")

        for country, label in zip(country_names, labels):
            if label == cluster_id:
                print(country)

    print(f"\n結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()

#出力結果
'''
抽出した国名数: 116

クラスタ 0
Afghanistan
Bahrain
Bangladesh
Bhutan
Egypt
Indonesia
Iran
Iraq
Jordan
Lebanon
Libya
Morocco
Nepal
Oman
Pakistan
Qatar
Syria
Thailand
Tunisia

クラスタ 1
Australia
Bahamas
Belize
Canada
Chile
China
Cuba
Dominica
Ecuador
Fiji
Greenland
Guyana
Honduras
Jamaica
Japan
Laos
Nicaragua
Peru
Philippines
Samoa
Suriname
Taiwan
Tuvalu
Uruguay
Venezuela
Vietnam

クラスタ 2
Armenia
Azerbaijan
Belarus
Kazakhstan
Kyrgyzstan
Moldova
Russia
Tajikistan
Turkmenistan
Ukraine
Uzbekistan

クラスタ 3
Albania
Austria
Belgium
Bulgaria
Croatia
Cyprus
Denmark
England
Estonia
Finland
France
Georgia
Germany
Greece
Hungary
Ireland
Italy
Latvia
Liechtenstein
Lithuania
Macedonia
Malta
Montenegro
Norway
Poland
Portugal
Romania
Serbia
Slovakia
Slovenia
Spain
Sweden
Switzerland
Turkey

クラスタ 4
Algeria
Angola
Botswana
Burundi
Eritrea
Gabon
Gambia
Ghana
Guinea
Kenya
Liberia
Madagascar
Malawi
Mali
Mauritania
Mozambique
Namibia
Niger
Nigeria
Rwanda
Senegal
Somalia
Sudan
Uganda
Zambia
Zimbabwe

結果を保存しました: /home/ichinose/projects/100-swing-drill/chapter_6/country_kmeans_result.txt
'''