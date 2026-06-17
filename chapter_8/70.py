from pathlib import Path
import pickle

import numpy as np
import torch
from gensim.models import KeyedVectors


def main():
    base_dir = Path(__file__).parent

    vector_path = base_dir.parent / "chapter_6" / "GoogleNews-vectors-negative300.bin.gz"

    embedding_path = base_dir / "embedding_matrix.pt"
    token_to_id_path = base_dir / "token_to_id.pkl"
    id_to_token_path = base_dir / "id_to_token.pkl"

    # 語彙数を制限する場合
    # GoogleNews全体を読むなら None にする
    # ただし全体を読むとメモリを大量に使う
    max_vocab = 100000
    # max_vocab = None

    # 事前学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(
        vector_path,
        binary=True,
        limit=max_vocab
    )

    # 単語リスト
    tokens = model.index_to_key

    # 単語埋め込みの語彙数
    # 先頭に <PAD> を追加するので +1
    vocab_size = len(tokens) + 1

    # 単語埋め込みの次元数
    embedding_dim = model.vector_size

    # 単語埋め込み行列を作成
    # 0行目は <PAD> 用のゼロベクトル
    embedding_matrix = np.zeros(
        (vocab_size, embedding_dim),
        dtype=np.float32
    )

    # 1行目以降に事前学習済み単語ベクトルを格納
    embedding_matrix[1:] = model.vectors

    # トークンIDから単語への対応
    id_to_token = ["<PAD>"] + tokens

    # 単語からトークンIDへの対応
    token_to_id = {}

    for token_id, token in enumerate(id_to_token):
        token_to_id[token] = token_id

    # PyTorchで扱いやすいようにTensorに変換
    embedding_tensor = torch.tensor(embedding_matrix)

    # 保存
    torch.save(embedding_tensor, embedding_path)

    with open(token_to_id_path, "wb") as f:
        pickle.dump(token_to_id, f)

    with open(id_to_token_path, "wb") as f:
        pickle.dump(id_to_token, f)

    print("単語埋め込み行列を作成しました")
    print(f"読み込んだ単語数: {len(tokens)}")
    print(f"語彙数 V: {vocab_size}")
    print(f"埋め込み次元数 d: {embedding_dim}")
    print(f"埋め込み行列の形状: {embedding_tensor.shape}")

    print()
    print("確認")
    print(f"<PAD> のID: {token_to_id['<PAD>']}")
    print(f"<PAD> のベクトル先頭5次元: {embedding_tensor[0][:5]}")

    if "United_States" in token_to_id:
        united_states_id = token_to_id["United_States"]
        print(f"United_States のID: {united_states_id}")
        print(f"United_States のベクトル先頭5次元: {embedding_tensor[united_states_id][:5]}")

    print()
    print(f"保存しました: {embedding_path}")
    print(f"保存しました: {token_to_id_path}")
    print(f"保存しました: {id_to_token_path}")


if __name__ == "__main__":
    main()

#出力結果
