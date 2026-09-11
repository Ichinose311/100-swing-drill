# 言語処理100本ノック

「[言語処理100本ノック2025](https://nlp100.github.io/2025/ja/index.html)」の問題00〜99と、
[2020年版の第10章](https://nlp100.github.io/2020/ja/ch10.html)（機械翻訳）に取り組んだ学習用実装です。
各 `XX.py` は原則として独立した演習スクリプトです。`main.py` はありません。

## 対応環境

- macOS / Linux
- Python 3.10〜3.12
- [uv](https://docs.astral.sh/uv/)

形態素解析を使う章ではMeCab、係り受け解析ではCaboChaを別途インストールしてください。

```bash
# macOS
brew install mecab mecab-ipadic cabocha swig

# Ubuntuの例
sudo apt update
sudo apt install mecab mecab-ipadic-utf8 libmecab-dev swig
```

## セットアップ

```bash
git clone https://github.com/Ichinose311/100-swing-drill.git
cd 100-swing-drill
uv sync
```

章に応じて追加依存を選びます。

```bash
uv sync --extra llm          # 第5章: Gemini API
uv sync --extra vectors      # 第6章: word2vec
uv sync --extra neural       # 第8〜10章: PyTorch / Transformers
uv sync --extra translation  # 2020年版第10章
```

複数を同時に指定できます。すべて必要な場合は `uv sync --all-extras` を使います。

## データの準備

第三者データ、学習済みモデル、学習結果、キャッシュ、生ログはGitに含めていません。
公式配布物の既知のSHA-256は `scripts/datasets.json` に記録しています。
次のコマンドは公式配布物を取得してハッシュを検証し、各章が参照する場所へ配置します。

```bash
uv run python scripts/prepare_data.py popular-names
uv run python scripts/prepare_data.py wiki
uv run python scripts/prepare_data.py wordsim353
uv run python scripts/prepare_data.py sst2
```

その他のデータは利用条件を確認して、提供元から取得してください。

- `GoogleNews-vectors-negative300.bin.gz`: [言語処理100本ノック第6章](https://nlp100.github.io/2025/ja/ch06.html)から案内されているGoogle News学習済みword2vecを `chapter_6/` に配置します。
- `questions-words.txt`: [TensorFlowの配布ページ](https://download.tensorflow.org/data/questions-words.txt)から取得し、次のように検証・配置します。配布元のTLS証明書を利用環境で検証できない場合があるため、準備スクリプトからの自動取得は行いません。

  ```bash
  uv run python scripts/prepare_data.py questions-words --from-file /path/to/questions-words.txt
  ```

- JMMLU: [公式リポジトリ](https://github.com/nlp-waseda/JMMLU)を `data/JMMLU/` に配置します。`JMMLU` と `JMMLU_NC_ND` はライセンスが異なります。
- KFTT: [Kyoto Free Translation Task](https://www.phontron.com/kftt/index-ja.html)から `kftt-data-1.0.tar.gz` を取得し、次のように検証・配置します。

  ```bash
  uv run python scripts/prepare_data.py kftt --from-file /path/to/kftt-data-1.0.tar.gz
  ```

- JESC: [公式サイト](https://nlp.stanford.edu/projects/jesc/)のofficial splitsを取得し、`chapter_10_2020/98_2020ver.py` の `--jesc-train` に渡します。

配置スクリプトがハッシュ不一致を報告した場合は、ファイルを使用せず、提供元や版の変更を確認してください。
ZIP、pickle、PyTorchチェックポイントなどは、信頼できる提供元から取得したものだけを読み込んでください。

## 実行

データファイルを相対パスで読む演習があるため、対象の章へ移動して実行します。

```bash
cd chapter_1
uv run python 00.py

cd ../chapter_3
uv run python 20.py

cd ../chapter_7
uv run python 60.py
```

第5章はGemini APIを呼び出すため、料金・レート制限・送信する内容を確認してから環境変数を設定します。
キーをソースコードや `.env` に書いてcommitしないでください。

```bash
export GEMINI_API_KEY="your-key"
cd chapter_5
uv run python 40.py
```

W&Bを利用するスクリプトは実験情報を外部サービスへ送信します。送信しない場合は実行前に
`WANDB_MODE=disabled` を設定してください。2020年版の翻訳Webアプリは既定で
`127.0.0.1` のみに公開されます。ネットワークへ公開する場合は認証やアクセス制御を別途用意してください。

## 出典と第三者コンテンツ

詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にまとめています。主な教材・データは次の通りです。

- 言語処理100本ノック2025 / 2020: 岡崎直観氏ほか。教材はCC BY 4.0。
- `popular-names.txt`、Wikipedia国別記事データ: 言語処理100本ノックの配布データ。
- 第5章の歴史問題: 文部科学省「高等学校卒業程度認定試験」令和4・5年度第1回。
- WordSim-353: Finkelstein et al. (2002)、CC BY 4.0。
- SST / SST-2 / GLUE: Socher et al. (2013)、Wang et al. (2018)。
- KFTT: Graham Neubig、CC BY-SA 3.0。
- JESC: Pryzant et al. (2018)。
- BERT / GPT-2: Hugging Face上の各モデルカードを参照。
- コメントで参照する『リーダブルコード』: Dustin Boswell、Trevor Foucher著、角征典訳。

このリポジトリには第三者の配布アーカイブやモデル重みを再配布していません。
過去の実験結果や図を利用する場合も、元データ・モデルの条件に従ってください。

## ライセンス

第三者コンテンツには各提供元のライセンスが適用されます。リポジトリ独自のコードには、
現時点で明示的なオープンソースライセンスを設定していません。利用許諾を追加する場合は、
第三者コンテンツと分けて `LICENSE` を追加してください。
