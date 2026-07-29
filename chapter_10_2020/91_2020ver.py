from __future__ import annotations

import math
import random
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 設定
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent

# 90で作成したデータ
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_JA_PATH = DATA_DIR / "train.ja"
TRAIN_EN_PATH = DATA_DIR / "train.en"
DEV_JA_PATH = DATA_DIR / "dev.ja"
DEV_EN_PATH = DATA_DIR / "dev.en"

# モデル保存先
MODEL_DIR = BASE_DIR / "models" / "91_transformer"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"
LAST_MODEL_PATH = MODEL_DIR / "last_model.pt"

# 語彙
MIN_FREQUENCY = 2
MAX_JA_VOCAB_SIZE = 30000
MAX_EN_VOCAB_SIZE = 30000

# 長すぎる文を除外
# BOSとEOSを含めた最大長
MAX_LENGTH = 100

# モデル
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# 学習
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 4000
CLIP_GRAD_NORM = 1.0

# 半精度学習
USE_AMP = True

# DataLoader
# 大学のサーバーでエラーになる場合は0にする
NUM_WORKERS = 2

LOG_INTERVAL = 100

# Noneなら全データを使用
# 動作確認だけなら、例えば10000と1000に変更する
MAX_TRAIN_EXAMPLES = None
MAX_DEV_EXAMPLES = None


# ============================================================
# 特殊トークン
# ============================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ============================================================
# 乱数固定
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 対訳データの読み込み
# ============================================================

def load_parallel_data(
    source_path: Path,
    target_path: Path,
    max_length: int,
    max_examples: int | None = None,
) -> tuple[list[tuple[str, str]], int]:
    """
    日英の対訳データを読み込む。

    各ファイルの同じ行番号が対訳関係になる。

    Returns
    -------
    pairs:
        日本語文と英語文の組
    skipped:
        空文または長すぎるため除外した文数
    """

    if not source_path.exists():
        raise FileNotFoundError(
            f"日本語ファイルが見つかりません: {source_path}"
        )

    if not target_path.exists():
        raise FileNotFoundError(
            f"英語ファイルが見つかりません: {target_path}"
        )

    pairs: list[tuple[str, str]] = []
    skipped = 0

    with (
        source_path.open("r", encoding="utf-8") as source_file,
        target_path.open("r", encoding="utf-8") as target_file,
    ):
        lines = zip_longest(
            source_file,
            target_file,
            fillvalue=None,
        )

        for line_number, (source_line, target_line) in enumerate(
            lines,
            start=1,
        ):
            if source_line is None or target_line is None:
                raise ValueError(
                    "日本語ファイルと英語ファイルの行数が一致しません。"
                    f"確認した行: {line_number}"
                )

            source_text = source_line.strip()
            target_text = target_line.strip()

            if not source_text or not target_text:
                skipped += 1
                continue

            source_length = len(source_text.split()) + 2
            target_length = len(target_text.split()) + 2

            if (
                source_length > max_length
                or target_length > max_length
            ):
                skipped += 1
                continue

            pairs.append((source_text, target_text))

            if (
                max_examples is not None
                and len(pairs) >= max_examples
            ):
                break

    return pairs, skipped


# ============================================================
# 語彙
# ============================================================

class Vocabulary:
    def __init__(
        self,
        sentences: list[str],
        min_frequency: int,
        max_size: int,
    ) -> None:
        counter: Counter[str] = Counter()

        for sentence in sentences:
            counter.update(sentence.split())

        # 出現回数が多い順に並べる
        # 出現回数が同じ場合は文字列順にする
        words = sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )

        vocabulary_words = []

        for word, frequency in words:
            if frequency < min_frequency:
                continue

            vocabulary_words.append(word)

            if len(vocabulary_words) >= max_size - len(SPECIAL_TOKENS):
                break

        self.id_to_token = SPECIAL_TOKENS + vocabulary_words
        self.token_to_id = {
            token: index
            for index, token in enumerate(self.id_to_token)
        }

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, sentence: str) -> list[int]:
        """
        文をID列に変換し、前後にBOSとEOSを付ける。
        """

        token_ids = [
            self.token_to_id.get(token, UNK_ID)
            for token in sentence.split()
        ]

        return [BOS_ID] + token_ids + [EOS_ID]


# ============================================================
# Dataset
# ============================================================

class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        self.pairs = pairs
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_text, target_text = self.pairs[index]

        source_ids = self.source_vocab.encode(source_text)
        target_ids = self.target_vocab.encode(target_text)

        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    バッチ内の長さをそろえるため、短い文をPADで埋める。
    """

    source_sequences = [
        source
        for source, _ in batch
    ]

    target_sequences = [
        target
        for _, target in batch
    ]

    source_batch = pad_sequence(
        source_sequences,
        batch_first=True,
        padding_value=PAD_ID,
    )

    target_batch = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=PAD_ID,
    )

    return source_batch, target_batch


# ============================================================
# Positional Encoding
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

        # [最大文長, hidden size]
        # から
        # [1, 最大文長, hidden size]
        # に変換する
        encoding = encoding.unsqueeze(0)

        self.register_buffer(
            "encoding",
            encoding,
            persistent=False,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequence_length = embeddings.size(1)

        embeddings = (
            embeddings
            + self.encoding[:, :sequence_length]
        )

        return self.dropout(embeddings)


# ============================================================
# Transformer翻訳モデル
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
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                "D_MODELはNHEADで割り切れる値にしてください。"
            )

        self.d_model = d_model

        self.source_embedding = nn.Embedding(
            num_embeddings=source_vocab_size,
            embedding_dim=d_model,
            padding_idx=PAD_ID,
        )

        self.target_embedding = nn.Embedding(
            num_embeddings=target_vocab_size,
            embedding_dim=d_model,
            padding_idx=PAD_ID,
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

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """
        行列パラメータをXavier初期化する。
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def create_causal_mask(
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Decoderが未来の単語を参照しないようにするマスク。

        Trueになっている場所は参照できない。
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
        Parameters
        ----------
        source:
            [バッチサイズ, 日本語文長]
        target:
            [バッチサイズ, 英語文長]

        Returns
        -------
        logits:
            [バッチサイズ, 英語文長, 英語語彙数]
        """

        source_padding_mask = source.eq(PAD_ID)
        target_padding_mask = target.eq(PAD_ID)

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

        logits = self.output_layer(transformer_output)

        return logits


# ============================================================
# 学習率スケジューラ
# ============================================================

def create_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    最初は学習率を徐々に上げ、
    warmup終了後は徐々に下げる。
    """

    def learning_rate_lambda(step: int) -> float:
        current_step = max(step, 1)

        warmup_rate = current_step / warmup_steps
        decay_rate = math.sqrt(
            warmup_steps / current_step
        )

        return min(warmup_rate, decay_rate)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_lambda,
    )


# ============================================================
# 1エポックの学習
# ============================================================

def train_one_epoch(
    model: TransformerNMT,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
) -> float:
    model.train()

    total_loss = 0.0
    total_tokens = 0

    epoch_start_time = time.time()

    for step, (source, target) in enumerate(
        data_loader,
        start=1,
    ):
        source = source.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        # Decoderへの入力
        #
        # <bos> I like Kyoto .
        decoder_input = target[:, :-1]

        # 正解
        #
        # I like Kyoto . <eos>
        expected_output = target[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            logits = model(
                source,
                decoder_input,
            )

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                expected_output.reshape(-1),
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "\nLossがNaNまたはinfになりました。\n"
                "USE_AMPをFalseにする、学習率を下げる、"
                "またはバッチサイズを変更してください。"
            )

        scaler.scale(loss).backward()

        # AMPで拡大された勾配を元に戻してから
        # 勾配クリッピングを行う
        scaler.unscale_(optimizer)

        clip_grad_norm_(
            model.parameters(),
            max_norm=CLIP_GRAD_NORM,
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        valid_token_count = (
            expected_output.ne(PAD_ID).sum().item()
        )

        total_loss += loss.item() * valid_token_count
        total_tokens += valid_token_count

        if step % LOG_INTERVAL == 0:
            average_loss = total_loss / max(total_tokens, 1)
            perplexity = math.exp(
                min(average_loss, 20.0)
            )

            elapsed = time.time() - epoch_start_time

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d} "
                f"| Step {step:05d}/{len(data_loader):05d} "
                f"| Loss {average_loss:.4f} "
                f"| PPL {perplexity:.2f} "
                f"| LR {current_lr:.7f} "
                f"| 経過 {elapsed:.1f}秒"
            )

    return total_loss / max(total_tokens, 1)


# ============================================================
# 開発データで評価
# ============================================================

@torch.no_grad()
def evaluate(
    model: TransformerNMT,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for source, target in data_loader:
        source = source.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        decoder_input = target[:, :-1]
        expected_output = target[:, 1:]

        with torch.cuda.amp.autocast(
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            logits = model(
                source,
                decoder_input,
            )

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                expected_output.reshape(-1),
            )

        valid_token_count = (
            expected_output.ne(PAD_ID).sum().item()
        )

        total_loss += loss.item() * valid_token_count
        total_tokens += valid_token_count

    return total_loss / max(total_tokens, 1)


# ============================================================
# モデルの保存
# ============================================================

def save_checkpoint(
    path: Path,
    model: TransformerNMT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    epoch: int,
    best_dev_loss: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_dev_loss": best_dev_loss,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),

        # 語彙はID順のリストとして保存
        "source_vocab": source_vocab.id_to_token,
        "target_vocab": target_vocab.id_to_token,

        "model_config": {
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "max_length": MAX_LENGTH,
        },

        "special_tokens": {
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
        },
    }

    torch.save(checkpoint, path)


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA対応GPUを利用できません。\n"
            "GPUが割り当てられているか、"
            "CUDA対応版PyTorchが入っているか確認してください。"
        )

    device = torch.device("cuda")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("=" * 60)
    print("使用デバイス")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    amp_enabled = USE_AMP

    print(f"AMP: {amp_enabled}")

    # --------------------------------------------------------
    # データ読み込み
    # --------------------------------------------------------

    print("\n訓練データを読み込んでいます。")

    train_pairs, train_skipped = load_parallel_data(
        source_path=TRAIN_JA_PATH,
        target_path=TRAIN_EN_PATH,
        max_length=MAX_LENGTH,
        max_examples=MAX_TRAIN_EXAMPLES,
    )

    print("開発データを読み込んでいます。")

    dev_pairs, dev_skipped = load_parallel_data(
        source_path=DEV_JA_PATH,
        target_path=DEV_EN_PATH,
        max_length=MAX_LENGTH,
        max_examples=MAX_DEV_EXAMPLES,
    )

    if not train_pairs:
        raise ValueError("訓練データが0件です。")

    if not dev_pairs:
        raise ValueError("開発データが0件です。")

    print("\nデータ数")
    print(f"Train: {len(train_pairs):,}")
    print(f"Dev  : {len(dev_pairs):,}")
    print(f"Train除外: {train_skipped:,}")
    print(f"Dev除外  : {dev_skipped:,}")

    # --------------------------------------------------------
    # 語彙作成
    # --------------------------------------------------------

    print("\n日本語語彙を作成しています。")

    source_vocab = Vocabulary(
        sentences=[
            source_text
            for source_text, _ in train_pairs
        ],
        min_frequency=MIN_FREQUENCY,
        max_size=MAX_JA_VOCAB_SIZE,
    )

    print("英語語彙を作成しています。")

    target_vocab = Vocabulary(
        sentences=[
            target_text
            for _, target_text in train_pairs
        ],
        min_frequency=MIN_FREQUENCY,
        max_size=MAX_EN_VOCAB_SIZE,
    )

    print(f"日本語語彙数: {len(source_vocab):,}")
    print(f"英語語彙数  : {len(target_vocab):,}")

    # --------------------------------------------------------
    # DatasetとDataLoader
    # --------------------------------------------------------

    train_dataset = TranslationDataset(
        pairs=train_pairs,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )

    dev_dataset = TranslationDataset(
        pairs=dev_pairs,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    # --------------------------------------------------------
    # モデル
    # --------------------------------------------------------

    model = TransformerNMT(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\n学習対象パラメータ数: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = create_learning_rate_scheduler(
        optimizer=optimizer,
        warmup_steps=WARMUP_STEPS,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
    )

    # --------------------------------------------------------
    # 学習
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_dev_loss = float("inf")

    print("\n学習を開始します。")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
        )

        dev_loss = evaluate(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        train_perplexity = math.exp(
            min(train_loss, 20.0)
        )

        dev_perplexity = math.exp(
            min(dev_loss, 20.0)
        )

        elapsed = time.time() - epoch_start

        print("\n" + "-" * 60)
        print(f"Epoch {epoch:02d} 完了")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train PPL : {train_perplexity:.2f}")
        print(f"Dev Loss  : {dev_loss:.4f}")
        print(f"Dev PPL   : {dev_perplexity:.2f}")
        print(f"時間      : {elapsed:.1f}秒")
        print("-" * 60)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                source_vocab=source_vocab,
                target_vocab=target_vocab,
                epoch=epoch,
                best_dev_loss=best_dev_loss,
            )

            print(
                "開発データのLossが改善しました。\n"
                f"保存先: {BEST_MODEL_PATH}"
            )

        save_checkpoint(
            path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            epoch=epoch,
            best_dev_loss=best_dev_loss,
        )

    print("\n学習が完了しました。")
    print(f"最良モデル: {BEST_MODEL_PATH}")
    print(f"最終モデル: {LAST_MODEL_PATH}")


if __name__ == "__main__":
    main()

#出力
