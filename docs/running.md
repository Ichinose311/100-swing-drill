# 章別の実行順

ルートディレクトリを基準にしたコマンドです。データは[データガイド](data.md)で用意します。
第1–4章の入力パスもスクリプトから解決するため、章へ`cd`する必要はありません。
CLIがない回答では、実験条件はファイル上部の定数を変更します。

| 章 | 前提と実行例 |
|---|---|
| 1 | 標準ライブラリのみ。`uv run --locked python chapter_1/06.py` |
| 2 | popular-namesを準備。`uv run --locked python chapter_2/19.py`。15・16は章内に分割・shuffleファイルを作成 |
| 3 | wikiを準備。`uv run --locked python chapter_3/23.py`。29のみMediaWikiへHTTPアクセス |
| 4 | 30–35は同梱`text.txt`。36–39はwikiが必要。`uv run --locked python chapter_4/38.py`。33–35はCaboChaが別途必要 |
| 5 | `llm` extraとAPIキー。42・43はJMMLUも必要。モデル・回数等を確認して実行 |
| 6 | `vectors` extraとGoogle News。54–55、57–59にはquestions-words、56にはWordSim-353も必要 |
| 7 | SST-2を準備。60–69はそれぞれ実行可能。67は`--demo`または`--dataset`も利用可能 |
| 8 | `neural` extra。Google News + SST-2 → 70 → 71 → 72以降。74は73の保存モデルが必要。77・78はCUDA必須 |
| 9 | `neural` extra。80–84は事前学習モデル、85–89はSST-2。88の前に87で学習済みモデルを保存 |
| 10 | `neural` extra。90–95は事前学習モデル、96–99はSST-2も必要。98・99は別々にGPT-2から学習する実装 |
| 2020 機械翻訳 | `translation` extra。KFTT → 90の前処理 → 91の学習 → 92–94の推論・評価。95–99はサブワード版の別系統 |

## 第5章: Gemini

API利用料金・レート制限・利用可能なモデルは自分の契約で確認してください。
現在のコード内のモデル名が使えない場合は`MODEL`等を変更します。

macOS / Linux:

```bash
export GEMINI_API_KEY="your-key"
uv run --locked --extra llm python chapter_5/40.py
```

PowerShell:

```powershell
$env:GEMINI_API_KEY="your-key"
uv run --locked --extra llm python chapter_5/40.py
```

`your-key`は説明用です。実際の値をソースやGitへ書かないでください。
48は`chapter_5/senryu_judge_robustness.csv`を作成します。既存公開結果は[results内](../results/chapter_5/senryu_judge_robustness.csv)に保存しています。

## 第6–9章

```bash
uv run --locked --extra vectors python chapter_6/50.py
uv run --locked --extra vectors python chapter_6/54.py
uv run --locked python chapter_6/55.py
uv run --locked --extra neural python chapter_8/70.py
uv run --locked --extra neural python chapter_8/71.py
uv run --locked --extra neural python chapter_8/72.py
uv run --locked --extra neural python chapter_9/80.py
```

54の結果を55が読みます。70・71は後続問題が使う埋め込み・語彙・データファイルを作成します。
第8章73のW&B送信は既定で無効です。使う場合にのみ`WANDB_MODE=online`と必要に応じて`WANDB_ENTITY`を設定します。

## 2020年版: 機械翻訳

KFTTを配置した後の流れです。訓練はCUDA GPUが必要です。

```bash
uv run --locked --extra translation python chapter_10_2020/90_2020ver.py
uv run --locked --extra translation python chapter_10_2020/91_2020ver.py
uv run --locked --extra translation python chapter_10_2020/92_2020ver.py "私は学生です。"
uv run --locked --extra translation python chapter_10_2020/93_2020ver.py
uv run --locked --extra translation python chapter_10_2020/94_2020ver.py --beam-sizes 1 5 --max-dev-examples 20
```

95はSentencePieceを使う別のモデルを学習します。`train`は必要ならトークナイザも作成します。

```bash
uv run --locked --extra translation python chapter_10_2020/95_2020ver.py train --epochs 10
uv run --locked --extra translation python chapter_10_2020/95_2020ver.py eval
uv run --locked --extra translation python chapter_10_2020/95_2020ver.py translate "私は学生です。"
uv run --locked --extra translation python chapter_10_2020/99_2020ver.py --host 127.0.0.1 --port 5000
```

最後のコマンド後にブラウザで`http://127.0.0.1:5000`を開きます。
公開の`results/`には重みを含まないため、評価記録だけをcloneして推論することはできません。
新規checkpointはSentencePieceファイルをcheckpointからの相対パスで記録します。移動時はその相対配置も維持してください。
既存の絶対パス形式のcheckpointも読み込みますが、そのパスがない環境では`load_model_bundle`のパス指定で補う必要があります。

96はTensorBoard付き学習、97は探索、98はJESC混合データで追加学習します。各条件は`--help`で確認できます。

```bash
uv run --locked --extra translation python chapter_10_2020/95_2020ver.py --help
uv run --locked --extra translation python chapter_10_2020/97_2020ver.py --help
uv run --locked --extra translation python chapter_10_2020/98_2020ver.py --help
```

## 通信なしの検証

```bash
uv run --locked python scripts/check_repository.py
uv run --locked python scripts/check_imports.py
uv run --locked python -m unittest discover -s tests -v
uv run --locked --all-extras python scripts/check_imports.py --all-extras
```

最後のコマンドはすべての追加依存が必要です。import確認中のAPIアクセスやモデル取得は禁止しています。
CaboCha未導入なら33–35を明示的にスキップします。import成功は学習・推論の完走を意味しません。
