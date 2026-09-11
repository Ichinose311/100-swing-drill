from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from janome.tokenizer import Tokenizer as JanomeTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


# ============================================================
# パス
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "91_transformer"
    / "best_model.pt"
)


# ============================================================
# 特殊トークン
# ============================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ============================================================
# トークナイザ
# ============================================================

# 90と同じ日本語形態素解析器を使用する
japanese_tokenizer = JanomeTokenizer()

# 英語トークンを自然な文に戻すために使用する
english_detokenizer = TreebankWordDetokenizer()


# ============================================================
# Positional Encoding
# 91で使用したものと同じ構造
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        frequencies = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(
            max_length,
            d_model,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(
            positions * frequencies
        )

        encoding[:, 1::2] = torch.cos(
            positions * frequencies
        )

        encoding = encoding.unsqueeze(0)

        self.register_buffer(
            "encoding",
            encoding,
            persistent=False,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = embeddings.size(1)

        embeddings = (
            embeddings
            + self.encoding[:, :sequence_length]
        )

        return self.dropout(embeddings)


# ============================================================
# Transformerモデル
# 91で使用したものと同じ構造
# ============================================================

class TransformerNMT(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_length: int,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        self.source_embedding = nn.Embedding(
            num_embeddings=source_vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        self.target_embedding = nn.Embedding(
            num_embeddings=target_vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        self.source_position = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_length=max_length,
        )

        self.target_position = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_length=max_length,
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        self.output_layer = nn.Linear(
            d_model,
            target_vocab_size,
        )

    @staticmethod
    def create_causal_mask(
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Decoderが未来の単語を参照しないためのマスク。
        """

        return torch.triu(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        source:
            [バッチサイズ, 日本語文長]

        target:
            [バッチサイズ, 生成済み英語文長]

        戻り値:
            [バッチサイズ, 英語文長, 英語語彙数]
        """

        source_padding_mask = source.eq(self.pad_id)
        target_padding_mask = target.eq(self.pad_id)

        target_causal_mask = self.create_causal_mask(
            target_length=target.size(1),
            device=target.device,
        )

        source_embeddings = self.source_embedding(source)
        source_embeddings *= math.sqrt(self.d_model)
        source_embeddings = self.source_position(
            source_embeddings
        )

        target_embeddings = self.target_embedding(target)
        target_embeddings *= math.sqrt(self.d_model)
        target_embeddings = self.target_position(
            target_embeddings
        )

        transformer_output = self.transformer(
            src=source_embeddings,
            tgt=target_embeddings,
            tgt_mask=target_causal_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )

        return self.output_layer(transformer_output)


# ============================================================
# 語彙
# ============================================================

class Vocabulary:
    def __init__(
        self,
        id_to_token: list[str],
    ) -> None:
        self.id_to_token = id_to_token

        self.token_to_id = {
            token: index
            for index, token in enumerate(id_to_token)
        }

    def __len__(self) -> int:
        return len(self.id_to_token)

    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def get_token(self, token_id: int) -> str:
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]

        return UNK_TOKEN


# ============================================================
# モデルの読み込み
# ============================================================

def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[
    TransformerNMT,
    Vocabulary,
    Vocabulary,
    dict,
]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません。\n"
            f"確認した場所: {model_path}\n"
            "先に91の学習プログラムを実行してください。"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    source_vocab = Vocabulary(
        checkpoint["source_vocab"]
    )

    target_vocab = Vocabulary(
        checkpoint["target_vocab"]
    )

    model_config = checkpoint["model_config"]
    special_tokens = checkpoint.get(
        "special_tokens",
        {},
    )

    pad_id = special_tokens.get(
        "pad_id",
        PAD_ID,
    )

    model = TransformerNMT(
        source_vocab_size=model_config[
            "source_vocab_size"
        ],
        target_vocab_size=model_config[
            "target_vocab_size"
        ],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_encoder_layers=model_config[
            "num_encoder_layers"
        ],
        num_decoder_layers=model_config[
            "num_decoder_layers"
        ],
        dim_feedforward=model_config[
            "dim_feedforward"
        ],
        dropout=model_config["dropout"],
        max_length=model_config["max_length"],
        pad_id=pad_id,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return (
        model,
        source_vocab,
        target_vocab,
        checkpoint,
    )


# ============================================================
# 日本語の前処理
# ============================================================

def tokenize_japanese(
    sentence: str,
) -> list[str]:
    """
    日本語を90と同じJanomeで形態素分割する。

    例:
        京都は美しいです。
        ↓
        ["京都", "は", "美しい", "です", "。"]
    """

    return [
        token.surface
        for token in japanese_tokenizer.tokenize(
            sentence
        )
    ]


def encode_japanese(
    sentence: str,
    source_vocab: Vocabulary,
    max_length: int,
) -> tuple[list[str], list[int]]:
    tokens = tokenize_japanese(sentence)

    # BOSとEOSの2トークン分を空ける
    maximum_token_count = max_length - 2

    if len(tokens) > maximum_token_count:
        print(
            f"注意: 入力が長いため、先頭"
            f"{maximum_token_count}形態素だけを使用します。"
        )

        tokens = tokens[:maximum_token_count]

    token_ids = [
        source_vocab.get_id(token)
        for token in tokens
    ]

    input_ids = [
        BOS_ID,
        *token_ids,
        EOS_ID,
    ]

    return tokens, input_ids


# ============================================================
# 翻訳
# ============================================================

@torch.inference_mode()
def translate(
    sentence: str,
    model: TransformerNMT,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    device: torch.device,
    max_length: int,
) -> tuple[list[str], list[str], str]:
    """
    貪欲法により、最も確率の高い英語トークンを
    1語ずつ生成する。
    """

    source_tokens, source_ids = encode_japanese(
        sentence=sentence,
        source_vocab=source_vocab,
        max_length=max_length,
    )

    source_tensor = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device,
    )

    # 最初はBOSのみ
    generated_ids = [BOS_ID]

    for _ in range(max_length - 1):
        target_tensor = torch.tensor(
            [generated_ids],
            dtype=torch.long,
            device=device,
        )

        logits = model(
            source=source_tensor,
            target=target_tensor,
        )

        # 最後の位置における次単語の予測値
        next_token_logits = logits[0, -1].clone()

        # PADとBOSは翻訳結果として生成しない
        next_token_logits[PAD_ID] = float("-inf")
        next_token_logits[BOS_ID] = float("-inf")

        next_token_id = int(
            torch.argmax(next_token_logits).item()
        )

        # EOSが生成されたら終了
        if next_token_id == EOS_ID:
            break

        generated_ids.append(next_token_id)

    # 先頭のBOSを除く
    translation_ids = generated_ids[1:]

    target_tokens = [
        target_vocab.get_token(token_id)
        for token_id in translation_ids
    ]

    # Treebank形式の英語トークンを自然な文に戻す
    translated_sentence = (
        english_detokenizer.detokenize(
            target_tokens
        )
    )

    return (
        source_tokens,
        target_tokens,
        translated_sentence,
    )


# ============================================================
# 結果表示
# ============================================================

def print_translation(
    sentence: str,
    model: TransformerNMT,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    device: torch.device,
    max_length: int,
) -> None:
    (
        source_tokens,
        target_tokens,
        translated_sentence,
    ) = translate(
        sentence=sentence,
        model=model,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        device=device,
        max_length=max_length,
    )

    print("\n" + "=" * 60)
    print(f"入力文       : {sentence}")
    print(
        "日本語形態素 : "
        + " ".join(source_tokens)
    )
    print(
        "英語トークン : "
        + " ".join(target_tokens)
    )
    print(f"翻訳結果     : {translated_sentence}")
    print("=" * 60)


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "91で学習したTransformerモデルを用いて"
            "日本語を英語に翻訳します。"
        )
    )

    parser.add_argument(
        "sentence",
        nargs="?",
        default=None,
        help="翻訳する日本語文",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="学習済みモデルのパス",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"使用デバイス: {device}")

    if device.type == "cuda":
        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )

    print(f"モデルを読み込んでいます: {args.model}")

    (
        model,
        source_vocab,
        target_vocab,
        checkpoint,
    ) = load_model(
        model_path=args.model,
        device=device,
    )

    model_config = checkpoint["model_config"]
    max_length = model_config["max_length"]

    print(
        f"学習済みEpoch: "
        f"{checkpoint.get('epoch', '不明')}"
    )

    print(
        f"日本語語彙数: {len(source_vocab):,}"
    )

    print(
        f"英語語彙数  : {len(target_vocab):,}"
    )

    # コマンドライン引数で文が与えられた場合
    if args.sentence is not None:
        print_translation(
            sentence=args.sentence,
            model=model,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            device=device,
            max_length=max_length,
        )

        return

    # 対話形式
    print("\n日本語文を入力してください。")
    print("終了する場合は exit または quit と入力してください。")

    while True:
        try:
            sentence = input("\n日本語 > ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if sentence.lower() in {
            "exit",
            "quit",
        }:
            print("終了します。")
            break

        if not sentence:
            continue

        print_translation(
            sentence=sentence,
            model=model,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            device=device,
            max_length=max_length,
        )


if __name__ == "__main__":
    main()

#出力
