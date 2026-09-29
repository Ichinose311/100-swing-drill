# NLP 100 Exercises

自然言語処理100本ノックに取り組んだ、Pythonによる学習・実装の記録です。
**2025年版の00–99と、2020年版第10章「機械翻訳」の90–99**を収録しています。
文字列処理から特徴量設計、分類器の学習・評価、事前学習モデルの利用までを、問題番号に沿って読めます。

**Chapterへ移動:** [1](#chapter-1) · [2](#chapter-2) · [3](#chapter-3) · [4](#chapter-4) · [5](#chapter-5) · [6](#chapter-6) · [7](#chapter-7) · [8](#chapter-8) · [9](#chapter-9) · [10](#chapter-10) · [機械翻訳・2020](#translation-2020)

初めて見る方は、[実装の見どころ](#highlighted-implementations)と、外部データ不要の[実行例](#how-to-run)から確認できます。
図・指標は[results/](results/README.md)、データの入手方法は[データガイド](docs/data.md)にまとめています。
全110ファイルの存在と、全問の正答・実験の再現を区別し、[今回確認した範囲](docs/verification.md)を記載しています。

## What I Learned

- **Pythonでのデータ処理:** 文字列・リスト・辞書・集合、ファイル入出力、正規表現、CSV/JSONL、集計と可視化。
- **日本語テキストの分析:** 形態素・文節・係り受け、単語頻度、文書頻度からのTF-IDF計算。
- **特徴量と評価の関係:** BoW、単語ベクトル、文ベクトルを使った分類、混同行列、適合率・再現率・F1、正則化。
- **ニューラルモデルの実装:** 可変長入力のpadding、maskを考慮したpooling、ミニバッチ学習、埋め込みの固定と更新。
- **言語モデルの利用と調整:** BERTの文表現・分類、GPT-2の生成確率・perplexity、SFT・DPOのデータ構成と学習処理。
- **実験条件の設計:** few-shot、選択肢の位置による偏り、LLMによる評価の揺れ、翻訳のbeam幅・ハイパーパラメータの比較。

教材に沿った実装です。モデルやアルゴリズム自体の新規提案を主張するものではありません。
問題ごとの処理を残しつつ、共通のデータ読み込みを切り出し、学習処理と評価処理を追える構成にしています。

## Topics

| Chapter | Topic | 主な技術 |
|---|---|---|
| [1 / 00–09](#chapter-1) | 準備運動 | Python、スライス、n-gram、集合 |
| [2 / 10–19](#chapter-2) | UNIXコマンド相当の処理 | pandas、TSV、分割・並べ替え・集計 |
| [3 / 20–29](#chapter-3) | 正規表現 | JSONL、gzip、re、MediaWiki API |
| [4 / 30–39](#chapter-4) | 言語解析 | MeCab、CaboCha、Counter、TF-IDF、Zipf則 |
| [5 / 40–49](#chapter-5) | 大規模言語モデル | Gemini API、JMMLU、プロンプト、評価の頑健性 |
| [6 / 50–59](#chapter-6) | 単語ベクトル | gensim、cosine類似度、Spearman相関、k-means、Ward法、t-SNE |
| [7 / 60–69](#chapter-7) | 機械学習 | SST-2、BoW、DictVectorizer、ロジスティック回帰 |
| [8 / 70–79](#chapter-8) | ニューラルネット | PyTorch、Embedding、DataLoader、MLP、GPU |
| [9 / 80–89](#chapter-9) | BERT型モデル | Transformers、tokenization、pooling、fine-tuning |
| [10 / 90–99](#chapter-10) | GPT型モデル | GPT-2、decoding、perplexity、SFT、TRL/DPO |
| [2020 第10章](#translation-2020) | 機械翻訳 | Transformer、SentencePiece、SacreBLEU、TensorBoard、Flask |

## Repository Structure

```text
100-swing-drill/
├── chapter_1/ … chapter_10/   # 2025年版: 各章10問、00.py … 99.py
│   ├── NN.py                 # 問題番号に対応する実行ファイル
│   └── <purpose>.py          # 必要な章だけ共通処理を配置
├── chapter_10_2020/           # 90_2020ver.py … 99_2020ver.py
├── scripts/
│   ├── prepare_data.py       # データ取得・SHA-256確認・配置
│   ├── datasets.json         # 配布元・ハッシュ・保存先
│   ├── check_repository.py   # 構文、回答の欠落、ローカルリンクを確認
│   └── check_imports.py      # 通信を遮断してimportを確認
├── tests/                    # 合成データを使う回帰テスト
├── results/                  # 既存の図・評価表・実験記録
├── docs/                     # データ、実行順、検証範囲
├── pyproject.toml            # Python範囲・基本依存・章別extras
├── uv.lock                   # 依存バージョンの固定
├── .env.example              # 環境変数名のみ
└── THIRD_PARTY_NOTICES.md     # 教材・データ・モデルの出典
```

問題番号のファイル名は教材との対応を優先して維持しています。
共通処理の例は[Wikipedia読込](chapter_3/wiki_article.py)、[国名ベクトル抽出](chapter_6/country_vectors.py)、[SST-2前処理](chapter_7/sentiment_data.py)です。
ダウンロードデータ・学習済み重み・cache・実行ログはGit管理対象外です。
公開済み結果は`results/`に保存し、再実行で直接上書きしません。

## Environment

- Python **3.10–3.12**（`pyproject.toml`の対応範囲）。`.python-version`の既定は3.10。今回のローカル確認はWindows / Python 3.10・3.12。
- 依存管理は[uv](https://docs.astral.sh/uv/getting-started/installation/)と`uv.lock`。`requirements.txt`との二重管理はしていません。
- 基本: NumPy、pandas、SciPy、scikit-learn、Matplotlib、requests、MeCab / UniDic-lite。

| 追加環境 | 対象 | 主なlibrary |
|---|---|---|
| `llm` | 第5章 | google-genai |
| `vectors` | 第6章 | gensim |
| `neural` | 第8–10章 | torch、transformers、datasets、accelerate、trl、gensim、wandb |
| `translation` | 2020年第10章 | torch、sentencepiece、sacrebleu、janome、nltk、tensorboard、flask |

第4章33–35は**CaboCha本体・辞書・Pythonバインディングを別途準備**します。
`uv sync`だけでは入りません。[CaboCha公式](https://taku910.github.io/cabocha/)を参照してください。
第8章77・78および機械翻訳の訓練コードはCUDA GPUを要求します。他の大規模モデルもCPUでは時間とメモリを要します。
GPU環境は[PyTorch公式](https://pytorch.org/get-started/locally/)に従って構成してください。
Windowsで追加ライブラリの`DLL load failed`が出る場合は、Pythonのbit数・パッケージの対応と
[Microsoft Visual C++ Runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)を確認してください。

## Setup

Gitとuvをインストールした後、次を実行します。以降のコマンドはリポジトリのルートが基準です。

```bash
git clone https://github.com/Ichinose311/100-swing-drill.git
cd 100-swing-drill
uv sync --locked
uv run --locked python chapter_1/02.py
uv run --locked python chapter_7/67.py --demo
```

第1章は標準ライブラリのみなので、環境を作る前でも`python chapter_1/02.py`で実行できます。
最後のコマンドは手書きの小さな合成データを使用します。実データの性能指標ではありません。
追加環境は必要なものを選び、**実行時にも同じextraを指定**します。

```bash
uv sync --locked --extra neural
uv run --locked --extra neural python scripts/check_imports.py --help
```

全環境を入れる場合は`uv sync --locked --all-extras`です。
APIや大きなモデルを使わずに確認したい場合は、基本環境の実行例から始めてください。

## Chapters

<a id="chapter-1"></a>
### Chapter 1 — 準備運動

文字列のスライス、辞書・集合、関数化を学習しました。Python標準機能で処理を組み立てています。
代表例: [02: 文字列の逆順](chapter_1/02.py)、[05: n-gram生成](chapter_1/05.py)、[06: 集合演算](chapter_1/06.py)。[章の全コード](chapter_1/)

<a id="chapter-2"></a>
### Chapter 2 — UNIXコマンド

行数・先頭末尾・列抽出・分割・頻度集計をPython / pandasで実装し、テキスト処理の基本操作を学習しました。
代表例: [15: ファイル分割](chapter_2/15.py)、[18: 出現頻度](chapter_2/18.py)、[19: 数値ソート](chapter_2/19.py)。[章の全コード](chapter_2/)

<a id="chapter-3"></a>
### Chapter 3 — 正規表現

WikipediaのJSONLから記事を取り出し、正規表現で見出し・項目・マークアップを扱いました。29ではrequestsによるAPIアクセスを使用します。
代表例: [23: セクション抽出](chapter_3/23.py)、[28: マークアップ除去](chapter_3/28.py)、[29: 国旗画像のURL取得](chapter_3/29.py)。[章の全コード](chapter_3/)

<a id="chapter-4"></a>
### Chapter 4 — 言語解析

MeCabの形態素情報、CaboChaの係り受け、Counterを使った頻度分析を扱いました。TFとDFの違いをコードで確認しています。
代表例: [32: 「名詞の名詞」](chapter_4/32.py)、[38: 名詞のTF-IDF](chapter_4/38.py)、[39: Zipf則の可視化](chapter_4/39.py)。[章の全コード](chapter_4/)

<a id="chapter-5"></a>
### Chapter 5 — 大規模言語モデル

Gemini APIでzero/few-shot、選択肢位置の変更、構造化出力、LLMによる評価のばらつきを扱いました。
代表例: [43: 正解選択肢の位置を変える評価](chapter_5/43.py)、[48: 誘導文に対する評価の頑健性](chapter_5/48.py)。[章の全コード](chapter_5/)

<a id="chapter-6"></a>
### Chapter 6 — 単語ベクトル

gensimで学習済みベクトルを読み込み、cosine類似度・アナロジー・人手評価との順位相関を計算しました。クラスタリングと可視化も実装しています。
代表例: [56: WordSim-353とSpearman相関](chapter_6/56.py)、[57: k-means](chapter_6/57.py)、[59: t-SNE](chapter_6/59.py)。[章の全コード](chapter_6/)

<a id="chapter-7"></a>
### Chapter 7 — 機械学習

SST-2をBoWへ変換し、scikit-learnで分類器を学習しました。訓練データで語彙を作り、開発データで評価する流れ、重みの解釈、正則化を学習しています。
代表例: [67: 複数指標で評価](chapter_7/67.py)、[68: 特徴量の重み](chapter_7/68.py)、[69: 正則化の比較](chapter_7/69.py)。[章の全コード](chapter_7/)

<a id="chapter-8"></a>
### Chapter 8 — ニューラルネット

PyTorchで単語埋め込みの平均を入力とする分類器を実装しました。paddingを除いた平均、DataLoader、固定埋め込みとfine-tuning、MLPを扱います。
代表例: [72: maskを使うBoW分類器](chapter_8/72.py)、[75: 可変長ミニバッチ](chapter_8/75.py)、[79: MLP分類器](chapter_8/79.py)。[章の全コード](chapter_8/)

<a id="chapter-9"></a>
### Chapter 9 — BERT型モデル

Transformersでトークン化・文表現・類似度・分類のfine-tuningを実装しました。特殊トークンとpaddingをpoolingから除く処理も扱っています。
代表例: [84: 平均pooling](chapter_9/84.py)、[87: 感情分類の学習](chapter_9/87.py)、[89: max pooling分類器](chapter_9/89.py)。[章の全コード](chapter_9/)

<a id="chapter-10"></a>
### Chapter 10 — GPT型モデル

GPT-2を用い、生成方法による出力の違い、確率とperplexity、文表現による分類、教師あり学習と選好学習の入力形式を学習しました。
代表例: [91: decoding比較](chapter_10/91.py)、[93: perplexity](chapter_10/93.py)、[98: SFT](chapter_10/98.py)、[99: DPO](chapter_10/99.py)。[章の全コード](chapter_10/)

<a id="translation-2020"></a>
### Chapter 10 (2020) — 機械翻訳

KFTTの日英対訳を使うTransformerをPyTorchで組み、語彙・mask・自己回帰生成からBLEU評価まで実装しました。SentencePiece、探索・調整、JESCを使う追加学習、Flaskの推論画面へ展開しています。
代表例: [91: Transformer訓練](chapter_10_2020/91_2020ver.py)、[94: beam search](chapter_10_2020/94_2020ver.py)、[95: サブワード翻訳](chapter_10_2020/95_2020ver.py)、[99: 翻訳サーバ](chapter_10_2020/99_2020ver.py)。[章の全コード](chapter_10_2020/)

## Highlighted Implementations

| 実装 | 何を実装したか | 何を学んだか |
|---|---|---|
| [TF-IDF](chapter_4/38.py) | 記事を順に読み、名詞のTFと記事単位のDFを集計 | 単純な出現回数と文書に特徴的な語の違い |
| [BoW感情分類・評価](chapter_7/67.py) | 疎な特徴行列、ロジスティック回帰、4種類の評価指標 | 訓練と開発データの分離、指標ごとの意味 |
| [平均埋め込み分類器](chapter_8/72.py) | paddingをmaskで除き、文ベクトルを線形層へ入力 | バッチ内の長さの違いを考慮したTensor演算 |
| [LLM評価の頑健性](chapter_5/48.py) | 通常・高得点誘導・低得点誘導の条件を反復し集計 | 評価条件ごとの変化とばらつきを比較すること |
| [翻訳のbeam search](chapter_10_2020/94_2020ver.py) | encoder出力の再利用、候補選択、長さで正規化したスコア、BLEU比較 | 探索の計算量と生成品質を別々に評価すること |

これらはコードを読む入口です。実験精度の優劣を示すランキングではありません。
既存の[正則化の比較図](results/chapter_7/regularization_accuracy.png)や[翻訳の評価記録](results/README.md)も参照できます。

## How to Run

### 1. 外部データなしで実行

```bash
uv run --locked python chapter_1/02.py
uv run --locked python chapter_4/32.py
uv run --locked python chapter_7/67.py --demo
```

順に`desserts`、同梱文中の「名詞の名詞」、合成データ8件で学習・4件で評価した表を出力します。
第4章の結果は辞書に依存します。デモのスコアはSST-2での精度を示しません。

### 2. 実データを準備して実行

```bash
uv run --locked python scripts/prepare_data.py popular-names
uv run --locked python chapter_2/18.py
uv run --locked python scripts/prepare_data.py sst2
uv run --locked python chapter_7/67.py
```

データ取得はネットワークを使用します。[scripts/datasets.json](scripts/datasets.json)のSHA-256を検証して所定の章へ配置します。
既存ファイルと内容が異なる場合は上書きを拒否します。取得済みデータは`--from-file`でも指定できます。

### 3. 検証と追加の章

```bash
uv run --locked python scripts/check_repository.py
uv run --locked python scripts/check_imports.py
uv run --locked python -m unittest discover -s tests -v
```

学習済みファイルが必要な順序、APIキーの設定、extraを指定するコマンドは[章別実行ガイド](docs/running.md)に記載しています。
有料API呼び出しや大規模学習は上の検証コマンドに含みません。
W&Bは第8章73では既定で無効です。使用する場合は環境変数で明示的に有効にします。

## References

- [言語処理100本ノック2025](https://nlp100.github.io/2025/ja/index.html) — 00–99の教材、CC BY 4.0。
- [言語処理100本ノック2020・機械翻訳](https://nlp100.github.io/2020/ja/ch10.html) — 追加実装の教材、CC BY 4.0。
- Boswell / Foucher著、角征典訳『[リーダブルコード](https://www.oreilly.co.jp/books/9784873115658/)』— コード内の学習コメントで参照。
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — Wikipedia、JMMLU、Google News、WordSim-353、SST-2、KFTT、JESC、モデル等の出典。

教材のライセンスと、データ・モデル・本リポジトリのコードの扱いは別です。本コード全体へのOSSライセンスは未設定です。
