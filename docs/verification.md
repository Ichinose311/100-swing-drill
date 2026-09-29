# 検証記録

公開用整理時の検証範囲です。既存の全実験を再実行したという意味ではありません。

## 確認したこと

| 項目 | 確認方法・結果 |
|---|---|
| 回答の保持 | 2025年版100問 + 2020年版機械翻訳10問、全110エントリを確認 |
| Python構文 | `scripts/check_repository.py`で全ソースをcompile / AST解析 |
| READMEのローカルリンク | README・docs・resultsのファイルとアンカーを同スクリプトで確認 |
| import | Python 3.12 + 全extrasで115モジュール成功。通信を遮断して実施 |
| 第1章 | 全10問をリポジトリ外の作業ディレクトリから実行 |
| 第4章 | 同梱テキストによるMeCab 31・32を実行。31をUniDic-liteの語彙素列に修正 |
| 第7章デモ | `67.py --demo`の前処理・学習・評価を実行。合成8件 / 4件の動作例 |
| データ準備 | popular-names / SST-2の公式配布ファイルを取得し、manifestのSHA-256を確認 |
| 実データの実行 | 第2章18の頻度集計、第7章67のSST-2学習・評価が完了 |
| 回帰テスト | 基本11件 + 追加依存で4件、計15件成功（Python 3.12）。基本環境では追加4件を明示的にskip |
| 結果の保存 | 59ファイルを元commitと照合。テキストは改行をLFへ正規化して比較、PNGはbyte比較 |
| 重複の削除 | metrics.tsv 18件の全列・全行をmetrics.jsonと比較して一致を確認 |

## SST-2の実行例

Windows / Python 3.10.20、uv.lockの基本依存、コマンド`uv run --locked python chapter_7/67.py`。
訓練67,349件、開発872件、訓練語彙14,816特徴。開発データを語彙構築には使用していません。

| 分割 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| train | 0.942211 | 0.942804 | 0.954297 | 0.948516 |
| dev | 0.808486 | 0.797849 | 0.835586 | 0.816282 |

正例はpositive（label=1）。この単一実行の結果を他方式への優位性やtest精度として扱いません。

## テストの内容

[基本テスト](../tests/test_core.py)は、import時の副作用、異なる作業ディレクトリ、Wikipediaの読み込み、
国名の抽出、BoW頻度、未知語の扱い、ZIP読込、評価指標、データ検証・上書き防止を確認します。
[追加テスト](../tests/test_optional.py)は、paddingを増やしても分類スコアが変わらないこと、バッチ内のlabelの対応、
beam候補選択、checkpointとSentencePieceの相対配置の移動・旧形式との互換性を確認します。
checkpointテストのトークナイザ部分はmockであり、実データからのSentencePiece学習は行いません。

## 未確認の範囲と改善候補

- 全extrasの成功環境は既存ランタイムのPython 3.12.14です。uvが取得したWindows用Python 3.12.13では、95の直接起動時にSentencePieceのDLL読込失敗を確認しました。ネイティブライブラリの実行環境が必要であり、全Windows環境での動作を保証していません。
- CaboChaが未導入のため、第4章33–35のimportと実行は未確認です。
- Geminiの有料API呼び出し、Google News全体の分析、BERT / GPT-2のモデル取得・訓練、KFTT / JESCのGPU実験は未実行です。
- `--help`、構文、importが成功しても、モデル取得や長時間学習の互換性までは保証しません。
- 既存実験のseed、データ件数、ライブラリ版、再実行コマンドを結果ごとに残すと比較しやすくなります。
- 第8章等の段階的なモデル実装は学習上の違いを読みやすくするため保持しています。今後は差分を設定値へ寄せる整理も可能です。
- 長文入力、空入力、API出力の形式揺れ、学習の端数バッチなどは、全章横断の追加検証が必要です。
- 第10章98の長文切り詰めで教師ラベルが残るか、勾配累積の端数が更新されるかは、今後の学習コード改善対象です。

## 公開ファイルの確認

APIキー・password・tokenの埋め込み、個人メール、ユーザー固有の絶対パスを対象に、公開対象のテキストを点検しています。
キーは環境変数で取得し、cache、生成物、重み、ローカル設定は`.gitignore`で除外します。
自動パターン検査はすべての秘密情報を検出する保証ではありません。
取得したGit履歴のテキストblobでも、検査したプロバイダキー・秘密鍵の形式には一致がありませんでした。
データ中の人名・地名は教材・サンプルとして必要なものを残しています。

## 再実行

```bash
uv run --locked python scripts/check_repository.py
uv run --locked python scripts/check_imports.py
uv run --locked python -m unittest discover -s tests -v
uv run --locked --all-extras python scripts/check_imports.py --all-extras
uv run --locked --all-extras python -m unittest discover -s tests -v
```

GitHub ActionsではLinux / Python 3.10・3.12とWindows / Python 3.12の基本環境を検証します。
CIにも有料API呼び出し・モデル取得・大規模データの学習を含めていません。
