# 実験記録

整理前から公開されていた図・指標59ファイルを保存しています。今回の整理で実験を再実行した結果ではありません。
元の配置、元commit、SHA-256は[manifest.json](manifest.json)に記録しています。本文・数値・画像は元のGit blobと一致することを確認しました。
同じ値を重複保存していた`metrics.tsv`18ファイルは削除し、`metrics.json`に情報を保持しています。

## 読み始める場所

| テーマ | 記録 | 読み取り方 |
|---|---|---|
| 単語の頻度 | [Zipf則](chapter_4/39_zipf.png) | 順位と頻度の分布 |
| LLM評価 | [誘導文と採点結果](chapter_5/senryu_judge_robustness.csv) | normal / positive_attack / negative_attackの比較 |
| 単語ベクトル | [t-SNE](chapter_6/country_tsne.png)、[Ward法](chapter_6/country_ward_dendrogram.png) | 可視化とクラスタリングは評価指標とは区別 |
| 正則化 | [開発正解率とC](chapter_7/regularization_accuracy.png) | パラメータによる変化を観察 |
| サブワード翻訳 | [summary](chapter_10_2020/models/95_sentencepiece/summary.json)、[各epoch](chapter_10_2020/models/95_sentencepiece/metrics.json) | 開発BLEUと損失の推移 |
| 翻訳の探索 | [beam幅別BLEU](chapter_10_2020/outputs/94_beam_search/beam_bleu_scores.tsv) | 同じ設定・分割の範囲で比較 |
| ハイパーパラメータ | [探索一覧](chapter_10_2020/outputs/97_tuning/tuning_results.tsv) | 学習率・batch size・optimizerと開発BLEU |
| ドメイン適応 | [比較JSON](chapter_10_2020/outputs/98_domain_adapt/domain_adaptation_result.json) | 記録上のbaseline / adaptedの比較 |

## 結果の範囲

- `smoke`は小規模な動作確認、`benchmark`は実行時間等を確認するための記録です。通常の実験とまとめて性能比較しません。
- 元の実験にはデータ数・seed・ライブラリ版・実行コマンドの記録が十分でないものがあります。ファイルにない条件を推測して補っていません。
- 例えばドメイン適応JSONにはBLEU 19.6064 → 20.5523という記録がありますが、今回の環境で再現確認した数値ではありません。
- JSON内の`best_model`等は当時の重みの保存先を示します。重みは含まれず、このディレクトリ内のリンク先ではありません。
- 再実行の出力は各章の`models/`、`outputs/`等に作られ、Gitから除外されます。公開する際は内容と条件を確認してここへ追加します。

各図の見た目や記録値から、未記録の実験条件や一般的な優位性は判断できません。
出典は[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)、現在の検証範囲は[検証記録](../docs/verification.md)を参照してください。
