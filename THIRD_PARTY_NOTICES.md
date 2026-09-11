# Third-party sources and notices

このファイルは、実装・例題・データ・事前学習モデルの主な出典をまとめたものです。
リンク先のライセンスと利用条件が優先されます。データやモデル本体はこのリポジトリでは再配布しません。

## 教材と書籍

- [言語処理100本ノック2025](https://nlp100.github.io/2025/ja/index.html)、岡崎直観氏ほか、CC BY 4.0。
- [言語処理100本ノック2020 第10章](https://nlp100.github.io/2020/ja/ch10.html)、CC BY 4.0。
- Dustin Boswell、Trevor Foucher著、角征典訳『[リーダブルコード](https://www.oreilly.co.jp/books/9784873115658/)』。ソース内のコメントで項目名とページを参照しています。
- 第5章40・41の問題文: 文部科学省「高等学校卒業程度認定試験」[令和5年度第1回](https://www.mext.go.jp/a_menu/koutou/shiken/kakomon/1411255_00010.htm)、[令和4年度第1回](https://www.mext.go.jp/a_menu/koutou/shiken/kakomon/1411255_00007.htm)。問題番号の対応は[教材第5章](https://nlp100.github.io/2025/ja/ch05.html)を参照してください。
- 夏目漱石『吾輩は猫である』冒頭: [教材第5章 問49](https://nlp100.github.io/2025/ja/ch05.html#id11)掲載文。

## データセット

- `popular-names.txt`: [言語処理100本ノック第2章](https://nlp100.github.io/2025/ja/ch02.html)配布データ。米国で生まれた子どもの名前・性別・人数・年の集計。
- `jawiki-country.json.gz`: [言語処理100本ノック第3章](https://nlp100.github.io/2025/ja/ch03.html)配布データ。Wikipedia記事を含むため、利用時は[Wikimediaの利用条件](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)と帰属表示を確認してください。
- JMMLU: Yin et al. (2024), [official repository](https://github.com/nlp-waseda/JMMLU)。通常のJMMLUはCC BY-SA 4.0、`JMMLU_NC_ND`はCC BY-NC-ND 4.0です。日本史など一部タスクには追加条件があります。
- Google News word2vec / questions-words: Mikolov et al. (2013)。取得方法は[教材第6章](https://nlp100.github.io/2025/ja/ch06.html)を参照してください。
- WordSimilarity-353: Finkelstein et al. (2002), “Placing Search in Context: The Concept Revisited,” [official distribution](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html), CC BY 4.0。
- Stanford Sentiment Treebank: Socher et al. (2013), “Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank,” [official site](https://nlp.stanford.edu/sentiment/)。
- SST-2 / GLUE: Wang et al. (2018), “GLUE: A Multi-Task Benchmark and Analysis Platform.” 取得方法は[教材第7章](https://nlp100.github.io/2025/ja/ch07.html)を参照してください。
- Kyoto Free Translation Task (KFTT): Graham Neubig, [official site](https://www.phontron.com/kftt/index-ja.html), CC BY-SA 3.0。
- Japanese-English Subtitle Corpus (JESC): Pryzant et al. (2018), “[JESC: Japanese-English Subtitle Corpus](https://aclanthology.org/L18-1182/),” [official data site](https://nlp.stanford.edu/projects/jesc/)。

## 事前学習モデルと生成API

- [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)。
- [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)。
- Google Gemini API。第5章の実験結果は使用時点のモデル出力であり、再実行時に同一になるとは限りません。

派生物を公開・配布するときは、元データ・モデル・APIの最新の利用条件を改めて確認してください。
