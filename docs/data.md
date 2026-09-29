# データの準備

コマンドはリポジトリルートから実行します。元データやモデル重みは再配布しません。
出典・利用条件は[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)を参照してください。
自動取得は[manifest](../scripts/datasets.json)に記録された配布元・サイズ上限・SHA-256を使用します。

| 名前 | 配置先 | 取得方法 |
|---|---|---|
| popular-names | `chapter_2/popular-names.txt` | `uv run --locked python scripts/prepare_data.py popular-names` |
| Wikipedia | 第3・4章の`jawiki-country.json.gz` | `uv run --locked python scripts/prepare_data.py wiki` |
| SST-2 | 第7・8・9章の`SST-2.zip` | `uv run --locked python scripts/prepare_data.py sst2` |
| WordSim-353 | `chapter_6/wordsim353.zip` | `uv run --locked python scripts/prepare_data.py wordsim353` |
| questions-words | `chapter_6/questions-words.txt` | 公式配布ファイルを取得し`--from-file`を使用 |
| KFTT | `chapter_10_2020/kftt-data-1.0.tar.gz` | 公式配布ファイルを取得し`--from-file`を使用 |

第10章のSST-2コードは`chapter_8/SST-2.zip`を参照します。準備スクリプトが必要箇所に配置します。
Wikipediaの非圧縮版も準備スクリプトが作りますが、現在の第3・4章はgzip版を読みます。

```bash
uv run --locked python scripts/prepare_data.py --help
uv run --locked python scripts/prepare_data.py questions-words --from-file downloads/questions-words.txt
uv run --locked python scripts/prepare_data.py kftt --from-file downloads/kftt-data-1.0.tar.gz
```

上の`downloads/`は自分で取得したファイルの保存先の例です。手元の相対パスに置き換えてください。
ハッシュ不一致のファイルは配置しません。

- **Google News word2vec:** [教材第6章](https://nlp100.github.io/2025/ja/ch06.html)の案内から取得し、`chapter_6/GoogleNews-vectors-negative300.bin.gz`に置きます。第8章70も同じファイルを使用します。
- **questions-words:** [TensorFlowの配布ファイル](https://download.tensorflow.org/data/questions-words.txt)。自動取得対象外です。TLS検証を無効にせず、取得元とファイルを確認してください。
- **JMMLU:** [公式リポジトリ](https://github.com/nlp-waseda/JMMLU)を`data/JMMLU/`へ配置します。第5章42・43はその下の`JMMLU/high_school_computer_science.csv`または`JMMLU_NC_ND/high_school_computer_science.csv`を探します。サブセット間で利用条件が異なります。
- **KFTT:** [公式サイト](https://www.phontron.com/kftt/index-ja.html)から`kftt-data-1.0.tar.gz`を取得します。
- **JESC:** [公式サイト](https://nlp.stanford.edu/projects/jesc/)から訓練データを取得し、98の`prepare --jesc-train`で指定します。訓練・開発・評価の分割を混ぜないでください。
- **BERT / GPT-2:** 各実行ファイルの`from_pretrained`で初回に取得します。モデル名はコードと[出典一覧](../THIRD_PARTY_NOTICES.md)を参照してください。

## ファイルの扱い

`data/`、アーカイブ、モデル重み、語彙ファイル、cache、学習ログは`.gitignore`で除外します。
再実行の出力は各コードの保存先に生成されます。確認して公開する図や指標のみ`results/`へコピーし、条件と出典を添えます。
APIキーは環境変数から読み込みます。`.env.example`をコピーしただけでは読み込まれません。
