# 言語処理100本ノック

## 概要
本プロジェクトは、「言語処理100本ノック2025」に取り組んだ際の実装をまとめたリポジトリである 

---

## 動作環境
- OS: macOS / Linux
- Python: 3.12以上
- パッケージ管理: uv

---

## セットアップ方法

### 1. リポジトリをクローン
bash\
`git clone https://github.com/Ichinose311/100-swing-drill.git`\
`cd 100-swing-drill`

### 2. 外部依存のインストール
macOS(Homebrew)\
`brew install mecab mecab-ipadic cabocha swig`\
Linux(Ubuntu)\
`sudo apt update\
sudo apt install mecab mecab-ipadic-utf8 libmecab-dev swig`\
*CaboCha/CRF++が必要な場合は別途インストールする

### 3. Python環境の構築
bash\
`uv sync`

### 4. PyTorchのイントール
bash\
`uv pip install torch`

### 実行方法
bash \
`uv run python main.py`
