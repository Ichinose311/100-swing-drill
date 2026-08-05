from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import time
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Sequence

import matplotlib
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
ORIG_DIR = BASE_DIR / "kftt-data-1.0" / "data" / "orig"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "95_sentencepiece"
DEFAULT_SPM_DIR = DEFAULT_MODEL_DIR / "spm"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "95_sentencepiece"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

DETOKENIZER = TreebankWordDetokenizer()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_parallel(
    source_path: Path,
    target_path: Path,
    max_examples: int | None = None,
) -> list[tuple[str, str]]:
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            f"Parallel data was not found: {source_path}, {target_path}"
        )

    pairs: list[tuple[str, str]] = []
    with (
        source_path.open(encoding="utf-8") as source_file,
        target_path.open(encoding="utf-8") as target_file,
    ):
        for line_number, lines in enumerate(
            zip_longest(source_file, target_file), start=1
        ):
            source_line, target_line = lines
            if source_line is None or target_line is None:
                raise ValueError(
                    f"The line counts differ near line {line_number}: "
                    f"{source_path}, {target_path}"
                )
            source = source_line.strip()
            target = target_line.strip()
            if not source or not target:
                continue
            pairs.append((source, target))
            if max_examples is not None and len(pairs) >= max_examples:
                break

    if not pairs:
        raise ValueError("No usable parallel sentences were found.")
    return pairs


def detokenize_english(text: str) -> str:
    return DETOKENIZER.detokenize(text.split())


def train_sentencepiece(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    force: bool = False,
) -> Path:
    model_path = model_prefix.with_suffix(".model")
    vocab_path = model_prefix.with_suffix(".vocab")
    if model_path.exists() and vocab_path.exists() and not force:
        print(f"SentencePiece already exists: {model_path}")
        return model_path

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    if force:
        model_path.unlink(missing_ok=True)
        vocab_path.unlink(missing_ok=True)

    print(f"Training SentencePiece: {model_path}")
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        hard_vocab_limit=False,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )
    return model_path


def ensure_sentencepiece_models(args: argparse.Namespace) -> tuple[Path, Path]:
    source_model = Path(args.source_spm)
    target_model = Path(args.target_spm)
    source_prefix = source_model.with_suffix("")
    target_prefix = target_model.with_suffix("")

    if args.force_spm or not source_model.exists():
        train_sentencepiece(
            Path(args.train_ja),
            source_prefix,
            args.source_vocab_size,
            args.spm_model_type,
            0.9995,
            args.force_spm,
        )
    if args.force_spm or not target_model.exists():
        train_sentencepiece(
            Path(args.train_en),
            target_prefix,
            args.target_vocab_size,
            args.spm_model_type,
            1.0,
            args.force_spm,
        )
    return source_model, target_model


class SubwordTranslationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        source_sp: spm.SentencePieceProcessor,
        target_sp: spm.SentencePieceProcessor,
        max_length: int,
    ) -> None:
        self.pairs = pairs
        self.source_sp = source_sp
        self.target_sp = target_sp
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.pairs[index]
        source_ids = self.source_sp.encode(source, out_type=int)
        target_ids = self.target_sp.encode(target, out_type=int)
        source_ids = source_ids[: self.max_length - 2]
        target_ids = target_ids[: self.max_length - 2]
        return (
            torch.tensor([BOS_ID, *source_ids, EOS_ID], dtype=torch.long),
            torch.tensor([BOS_ID, *target_ids, EOS_ID], dtype=torch.long),
        )


def collate_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    sources, targets = zip(*batch)
    return (
        pad_sequence(sources, batch_first=True, padding_value=PAD_ID),
        pad_sequence(targets, batch_first=True, padding_value=PAD_ID),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.dropout(embeddings + self.encoding[:, : embeddings.size(1)])


class TransformerNMT(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.source_embedding = nn.Embedding(
            source_vocab_size, d_model, padding_idx=PAD_ID
        )
        self.target_embedding = nn.Embedding(
            target_vocab_size, d_model, padding_idx=PAD_ID
        )
        self.source_position = PositionalEncoding(d_model, dropout, max_length)
        self.target_position = PositionalEncoding(d_model, dropout, max_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.output_layer = nn.Linear(d_model, target_vocab_size)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_padding_mask = source.eq(PAD_ID)
        embeddings = self.source_embedding(source) * math.sqrt(self.d_model)
        embeddings = self.source_position(embeddings)
        memory = self.transformer.encoder(
            embeddings, src_key_padding_mask=source_padding_mask
        )
        return memory, source_padding_mask

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_padding_mask = target.eq(PAD_ID)
        embeddings = self.target_embedding(target) * math.sqrt(self.d_model)
        embeddings = self.target_position(embeddings)
        output = self.transformer.decoder(
            embeddings,
            memory,
            tgt_mask=self.causal_mask(target.size(1), target.device),
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_layer(output)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory, source_padding_mask = self.encode(source)
        return self.decode(target, memory, source_padding_mask)


def encode_source_batch(
    texts: Sequence[str],
    source_sp: spm.SentencePieceProcessor,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    sequences = []
    for text in texts:
        ids = source_sp.encode(text, out_type=int)[: max_length - 2]
        sequences.append(torch.tensor([BOS_ID, *ids, EOS_ID]))
    return pad_sequence(
        sequences, batch_first=True, padding_value=PAD_ID
    ).to(device)


def target_ids_to_text(
    token_ids: Sequence[int], target_sp: spm.SentencePieceProcessor
) -> str:
    content = []
    for token_id in token_ids:
        if token_id == EOS_ID:
            break
        if token_id not in (PAD_ID, BOS_ID):
            content.append(int(token_id))
    tokenized_text = target_sp.decode(content)
    return detokenize_english(tokenized_text)


@torch.inference_mode()
def greedy_decode_batch(
    model: TransformerNMT,
    source: torch.Tensor,
    max_length: int,
) -> list[list[int]]:
    model.eval()
    memory, memory_padding_mask = model.encode(source)
    generated = torch.full(
        (source.size(0), 1), BOS_ID, dtype=torch.long, device=source.device
    )
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)

    for _ in range(max_length - 1):
        logits = model.decode(generated, memory, memory_padding_mask)[:, -1]
        logits[:, PAD_ID] = -torch.inf
        logits[:, BOS_ID] = -torch.inf
        next_ids = logits.argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, EOS_ID), next_ids)
        generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
        finished |= next_ids.eq(EOS_ID)
        if bool(finished.all()):
            break
    return generated.tolist()


@dataclass(frozen=True)
class BeamHypothesis:
    token_ids: tuple[int, ...]
    log_probability: float

    @property
    def score(self) -> float:
        # BOS is not generated. EOS is included when it is generated.
        return self.log_probability / max(len(self.token_ids) - 1, 1)


def top_hypotheses(
    hypotheses: Sequence[BeamHypothesis], beam_size: int
) -> list[BeamHypothesis]:
    return heapq.nlargest(beam_size, hypotheses, key=lambda item: item.score)


@torch.inference_mode()
def beam_decode_one(
    model: TransformerNMT,
    source: torch.Tensor,
    beam_size: int,
    max_length: int,
) -> list[int]:
    model.eval()
    memory, memory_padding_mask = model.encode(source)
    active = [BeamHypothesis((BOS_ID,), 0.0)]
    finished: list[BeamHypothesis] = []

    for _ in range(max_length - 1):
        target = torch.tensor(
            [hypothesis.token_ids for hypothesis in active],
            dtype=torch.long,
            device=source.device,
        )
        expanded_memory = memory.expand(len(active), -1, -1)
        expanded_mask = memory_padding_mask.expand(len(active), -1)
        log_probs = F.log_softmax(
            model.decode(target, expanded_memory, expanded_mask)[:, -1], dim=-1
        )
        log_probs[:, PAD_ID] = -torch.inf
        log_probs[:, BOS_ID] = -torch.inf
        k = min(beam_size, log_probs.size(1))
        values, indices = torch.topk(log_probs, k=k, dim=-1)

        next_active: list[BeamHypothesis] = []
        for row, hypothesis in enumerate(active):
            for value, token_id in zip(values[row].tolist(), indices[row].tolist()):
                candidate = BeamHypothesis(
                    hypothesis.token_ids + (token_id,),
                    hypothesis.log_probability + value,
                )
                if token_id == EOS_ID:
                    finished.append(candidate)
                else:
                    next_active.append(candidate)

        finished = top_hypotheses(finished, beam_size)
        active = top_hypotheses(next_active, beam_size)
        if not active:
            break

    candidates = finished if finished else active
    return list(max(candidates, key=lambda item: item.score).token_ids)


@torch.inference_mode()
def translate_texts(
    model: TransformerNMT,
    texts: Sequence[str],
    source_sp: spm.SentencePieceProcessor,
    target_sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_length: int,
    batch_size: int,
    beam_size: int = 1,
    log_interval: int = 0,
) -> list[str]:
    translations: list[str] = []
    start = time.time()
    if beam_size == 1:
        for start_index in range(0, len(texts), batch_size):
            batch_texts = texts[start_index : start_index + batch_size]
            source = encode_source_batch(batch_texts, source_sp, max_length, device)
            generated = greedy_decode_batch(model, source, max_length)
            translations.extend(
                target_ids_to_text(ids, target_sp) for ids in generated
            )
            done = min(start_index + batch_size, len(texts))
            if log_interval and done % log_interval < batch_size:
                print(f"Decoded {done:,}/{len(texts):,} ({time.time()-start:.1f}s)")
    else:
        for index, text in enumerate(texts, start=1):
            source = encode_source_batch([text], source_sp, max_length, device)
            generated = beam_decode_one(model, source, beam_size, max_length)
            translations.append(target_ids_to_text(generated, target_sp))
            if log_interval and index % log_interval == 0:
                print(f"Decoded {index:,}/{len(texts):,} ({time.time()-start:.1f}s)")
    return translations


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    return sacrebleu.corpus_bleu(
        list(hypotheses), [list(references)], tokenize="13a"
    ).score


def evaluate_bleu(
    model: TransformerNMT,
    pairs: Sequence[tuple[str, str]],
    source_sp: spm.SentencePieceProcessor,
    target_sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_length: int,
    batch_size: int,
    beam_size: int = 1,
) -> tuple[float, list[str]]:
    hypotheses = translate_texts(
        model,
        [source for source, _ in pairs],
        source_sp,
        target_sp,
        device,
        max_length,
        batch_size,
        beam_size,
    )
    return corpus_bleu(hypotheses, [target for _, target in pairs]), hypotheses


def build_optimizer(
    name: str,
    parameters,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    common = {
        "params": parameters,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.98),
        "eps": 1e-9,
    }
    choices = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "radam": torch.optim.RAdam,
    }
    return choices[name](**common)


def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        step = max(step, 1)
        if warmup_steps <= 0:
            return 1.0
        return min(step / warmup_steps, math.sqrt(warmup_steps / step))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_epoch(
    model: TransformerNMT,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    clip_norm: float,
    epoch: int,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()
    for step, (source, target) in enumerate(loader, start=1):
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        decoder_input = target[:, :-1]
        expected = target[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(source, decoder_input)
            loss = criterion(logits.flatten(0, 1), expected.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss became NaN or infinity.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        valid_tokens = int(expected.ne(PAD_ID).sum())
        total_loss += loss.detach().item() * valid_tokens
        total_tokens += valid_tokens
        if log_interval and step % log_interval == 0:
            average = total_loss / max(total_tokens, 1)
            print(
                f"Epoch {epoch:02d} | {step:05d}/{len(loader):05d} | "
                f"loss {average:.4f} | ppl {math.exp(min(average, 20)):.2f} | "
                f"lr {optimizer.param_groups[0]['lr']:.7g} | {time.time()-start:.1f}s"
            )
    return total_loss / max(total_tokens, 1)


@torch.inference_mode()
def evaluate_loss(
    model: TransformerNMT,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for source, target in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        decoder_input = target[:, :-1]
        expected = target[:, 1:]
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(source, decoder_input)
            loss = criterion(logits.flatten(0, 1), expected.flatten())
        valid_tokens = int(expected.ne(PAD_ID).sum())
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    return total_loss / max(total_tokens, 1)


def model_config_from_args(
    args: argparse.Namespace,
    source_vocab_size: int,
    target_vocab_size: int,
) -> dict:
    return {
        "source_vocab_size": source_vocab_size,
        "target_vocab_size": target_vocab_size,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "dim_feedforward": args.feedforward,
        "dropout": args.dropout,
        "max_length": args.max_length,
    }


def save_checkpoint(
    path: Path,
    model: TransformerNMT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_dev_bleu: float,
    best_dev_loss: float,
    model_config: dict,
    source_spm: Path,
    target_spm: Path,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_config = {}
    for key, value in vars(args).items():
        if key == "function" or callable(value):
            continue
        training_config[key] = str(value) if isinstance(value, Path) else value
    torch.save(
        {
            "epoch": epoch,
            "best_dev_bleu": best_dev_bleu,
            "best_dev_loss": best_dev_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": model_config,
            "source_spm": str(source_spm.resolve()),
            "target_spm": str(target_spm.resolve()),
            "special_tokens": {
                "pad_id": PAD_ID,
                "unk_id": UNK_ID,
                "bos_id": BOS_ID,
                "eos_id": EOS_ID,
            },
            "training_config": training_config,
        },
        path,
    )


def load_model_bundle(
    checkpoint_path: Path,
    device: torch.device,
    source_spm_path: Path | None = None,
    target_spm_path: Path | None = None,
) -> tuple[
    TransformerNMT,
    spm.SentencePieceProcessor,
    spm.SentencePieceProcessor,
    dict,
]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TransformerNMT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    source_path = source_spm_path or Path(checkpoint["source_spm"])
    target_path = target_spm_path or Path(checkpoint["target_spm"])
    source_sp = spm.SentencePieceProcessor(model_file=str(source_path))
    target_sp = spm.SentencePieceProcessor(model_file=str(target_path))
    return model, source_sp, target_sp, checkpoint


def sample_pairs(
    pairs: Sequence[tuple[str, str]], count: int, seed: int
) -> list[tuple[str, str]]:
    if count <= 0 or count >= len(pairs):
        return list(pairs)
    return random.Random(seed).sample(list(pairs), count)


def save_history(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "metrics.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(history)


def command_spm(args: argparse.Namespace) -> None:
    source_model, target_model = ensure_sentencepiece_models(args)
    source_sp = spm.SentencePieceProcessor(model_file=str(source_model))
    target_sp = spm.SentencePieceProcessor(model_file=str(target_model))
    print(f"Japanese pieces: {source_sp.vocab_size():,} ({source_model})")
    print(f"English pieces : {target_sp.vocab_size():,} ({target_model})")


def command_train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training.")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    source_model, target_model = ensure_sentencepiece_models(args)
    source_sp = spm.SentencePieceProcessor(model_file=str(source_model))
    target_sp = spm.SentencePieceProcessor(model_file=str(target_model))
    train_pairs = read_parallel(
        Path(args.train_ja), Path(args.train_en), args.max_train_examples
    )
    dev_pairs = read_parallel(
        Path(args.dev_ja), Path(args.dev_en), args.max_dev_examples
    )
    print(f"Train: {len(train_pairs):,}, Dev: {len(dev_pairs):,}")

    train_dataset = SubwordTranslationDataset(
        train_pairs, source_sp, target_sp, args.max_length
    )
    dev_dataset = SubwordTranslationDataset(
        dev_pairs, source_sp, target_sp, args.max_length
    )
    loader_options = {
        "collate_fn": collate_batch,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **loader_options
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, **loader_options
    )

    model_config = model_config_from_args(
        args, source_sp.vocab_size(), target_sp.vocab_size()
    )
    model = TransformerNMT(**model_config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial["model_config"] != model_config:
            raise ValueError("The initial checkpoint model configuration does not match.")
        model.load_state_dict(initial["model_state_dict"])
        print(f"Initialized from: {args.init_checkpoint}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID, label_smoothing=args.label_smoothing
    )
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), args.learning_rate, args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args.warmup_steps)
    amp_enabled = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard is missing. Install it with: python -m pip install tensorboard"
            ) from error
        writer = SummaryWriter(log_dir=str(Path(args.log_dir) / args.run_name))

    train_bleu_pairs = sample_pairs(train_pairs, args.bleu_train_samples, args.seed)
    dev_bleu_pairs = sample_pairs(dev_pairs, args.bleu_dev_samples, args.seed)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pt"
    last_path = model_dir / "last_model.pt"
    history: list[dict] = []
    best_dev_bleu = -math.inf
    best_dev_loss = math.inf
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            amp_enabled,
            args.clip_norm,
            epoch,
            args.log_interval,
        )
        dev_loss = evaluate_loss(model, dev_loader, criterion, device, amp_enabled)
        train_bleu, _ = evaluate_bleu(
            model,
            train_bleu_pairs,
            source_sp,
            target_sp,
            device,
            args.max_length,
            args.eval_batch_size,
        )
        dev_bleu, _ = evaluate_bleu(
            model,
            dev_bleu_pairs,
            source_sp,
            target_sp,
            device,
            args.max_length,
            args.eval_batch_size,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "train_bleu": train_bleu,
            "dev_bleu": dev_bleu,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        save_history(history, model_dir)
        print(
            f"Epoch {epoch:02d} complete | train loss {train_loss:.4f} | "
            f"dev loss {dev_loss:.4f} | train BLEU {train_bleu:.2f} | "
            f"dev BLEU {dev_bleu:.2f} | {row['seconds']:.1f}s"
        )
        if writer is not None:
            writer.add_scalars("loss", {"train": train_loss, "dev": dev_loss}, epoch)
            writer.add_scalars("BLEU", {"train": train_bleu, "dev": dev_bleu}, epoch)
            writer.add_scalar("learning_rate", row["learning_rate"], epoch)

        if dev_bleu > best_dev_bleu:
            best_dev_bleu = dev_bleu
            best_dev_loss = dev_loss
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_dev_bleu,
                best_dev_loss,
                model_config,
                source_model,
                target_model,
                args,
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_dev_bleu,
            best_dev_loss,
            model_config,
            source_model,
            target_model,
            args,
        )

    if writer is not None:
        writer.close()
    summary = {
        "best_epoch": best_epoch,
        "best_dev_bleu": best_dev_bleu,
        "best_dev_loss": best_dev_loss,
        "best_model": str(best_path.resolve()),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, source_sp, target_sp, checkpoint = load_model_bundle(
        Path(args.checkpoint), device
    )
    pairs = read_parallel(Path(args.source), Path(args.reference), args.max_examples)
    bleu, hypotheses = evaluate_bleu(
        model,
        pairs,
        source_sp,
        target_sp,
        device,
        checkpoint["model_config"]["max_length"],
        args.batch_size,
        args.beam_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    print(f"BLEU = {bleu:.4f}")
    print(f"Hypotheses: {output}")


def command_translate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, source_sp, target_sp, checkpoint = load_model_bundle(
        Path(args.checkpoint), device
    )
    translations = translate_texts(
        model,
        [args.text],
        source_sp,
        target_sp,
        device,
        checkpoint["model_config"]["max_length"],
        1,
        args.beam_size,
    )
    print(translations[0])


def command_beam(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, source_sp, target_sp, checkpoint = load_model_bundle(
        Path(args.checkpoint), device
    )
    pairs = read_parallel(Path(args.dev_ja), Path(args.dev_en), args.max_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for beam_size in sorted(set(args.beam_sizes)):
        start = time.time()
        bleu, hypotheses = evaluate_bleu(
            model,
            pairs,
            source_sp,
            target_sp,
            device,
            checkpoint["model_config"]["max_length"],
            args.batch_size,
            beam_size,
        )
        seconds = time.time() - start
        rows.append({"beam_size": beam_size, "bleu": bleu, "seconds": seconds})
        (output_dir / f"dev.beam{beam_size}.en").write_text(
            "\n".join(hypotheses) + "\n", encoding="utf-8"
        )
        print(f"beam={beam_size:3d} | BLEU={bleu:.4f} | {seconds:.1f}s")

    with (output_dir / "beam_bleu_scores.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    plt.figure(figsize=(8, 5))
    plt.plot(
        [row["beam_size"] for row in rows],
        [row["bleu"] for row in rows],
        marker="o",
    )
    plt.xlabel("Beam size")
    plt.ylabel("BLEU")
    plt.title("SentencePiece NMT: beam size vs. dev BLEU")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "beam_bleu_plot.png", dpi=160)
    plt.close()


def add_spm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-ja", type=Path, default=ORIG_DIR / "kyoto-train.ja"
    )
    parser.add_argument(
        "--train-en", type=Path, default=ORIG_DIR / "kyoto-train.en"
    )
    parser.add_argument(
        "--source-spm", type=Path, default=DEFAULT_SPM_DIR / "source.model"
    )
    parser.add_argument(
        "--target-spm", type=Path, default=DEFAULT_SPM_DIR / "target.model"
    )
    parser.add_argument("--source-vocab-size", type=int, default=16000)
    parser.add_argument("--target-vocab-size", type=int, default=16000)
    parser.add_argument("--spm-model-type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--force-spm", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="95: SentencePiece NMT")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spm_parser = subparsers.add_parser("spm", help="Train SentencePiece models")
    add_spm_arguments(spm_parser)
    spm_parser.set_defaults(function=command_spm)

    train_parser = subparsers.add_parser("train", help="Train the Transformer")
    add_spm_arguments(train_parser)
    train_parser.add_argument("--dev-ja", type=Path, default=ORIG_DIR / "kyoto-dev.ja")
    train_parser.add_argument("--dev-en", type=Path, default=ORIG_DIR / "kyoto-dev.en")
    train_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train_parser.add_argument(
        "--log-dir", type=Path, default=BASE_DIR / "outputs" / "96_tensorboard"
    )
    train_parser.add_argument("--run-name", default="sentencepiece_transformer")
    train_parser.add_argument("--tensorboard", action="store_true")
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--eval-batch-size", type=int, default=128)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "radam"], default="adamw"
    )
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=4000)
    train_parser.add_argument("--clip-norm", type=float, default=1.0)
    train_parser.add_argument("--label-smoothing", type=float, default=0.1)
    train_parser.add_argument("--max-length", type=int, default=128)
    train_parser.add_argument("--d-model", type=int, default=256)
    train_parser.add_argument("--nhead", type=int, default=8)
    train_parser.add_argument("--encoder-layers", type=int, default=4)
    train_parser.add_argument("--decoder-layers", type=int, default=4)
    train_parser.add_argument("--feedforward", type=int, default=1024)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--log-interval", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no-amp", action="store_true")
    train_parser.add_argument("--max-train-examples", type=int)
    train_parser.add_argument("--max-dev-examples", type=int)
    train_parser.add_argument(
        "--bleu-train-samples",
        type=int,
        default=2000,
        help="0 evaluates BLEU on the entire training set",
    )
    train_parser.add_argument(
        "--bleu-dev-samples",
        type=int,
        default=0,
        help="0 evaluates BLEU on the entire development set",
    )
    train_parser.add_argument("--init-checkpoint", type=Path)
    train_parser.set_defaults(function=command_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate BLEU")
    eval_parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "best_model.pt"
    )
    eval_parser.add_argument(
        "--source", type=Path, default=ORIG_DIR / "kyoto-test.ja"
    )
    eval_parser.add_argument(
        "--reference", type=Path, default=ORIG_DIR / "kyoto-test.en"
    )
    eval_parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "test.hypothesis.en"
    )
    eval_parser.add_argument("--beam-size", type=int, default=1)
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--max-examples", type=int)
    eval_parser.set_defaults(function=command_eval)

    translate_parser = subparsers.add_parser("translate", help="Translate one sentence")
    translate_parser.add_argument("text")
    translate_parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "best_model.pt"
    )
    translate_parser.add_argument("--beam-size", type=int, default=5)
    translate_parser.set_defaults(function=command_translate)

    beam_parser = subparsers.add_parser("beam", help="Run task 94 with subwords")
    beam_parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "best_model.pt"
    )
    beam_parser.add_argument("--dev-ja", type=Path, default=ORIG_DIR / "kyoto-dev.ja")
    beam_parser.add_argument("--dev-en", type=Path, default=ORIG_DIR / "kyoto-dev.en")
    beam_parser.add_argument(
        "--beam-sizes", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100]
    )
    beam_parser.add_argument("--batch-size", type=int, default=128)
    beam_parser.add_argument("--max-examples", type=int)
    beam_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "beam_search"
    )
    beam_parser.set_defaults(function=command_beam)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()


# ===== EXECUTION RESULT =====
# Log: outputs/95_sentencepiece/full_run.log
# Epoch 10 | 03400/06880 | loss 3.4962 | ppl 32.99 | lr 7.423832e-05 | 113.7s
# Epoch 10 | 03500/06880 | loss 3.4967 | ppl 33.01 | lr 7.418156e-05 | 117.2s
# Epoch 10 | 03600/06880 | loss 3.4970 | ppl 33.02 | lr 7.412493e-05 | 120.9s
# Epoch 10 | 03700/06880 | loss 3.4973 | ppl 33.03 | lr 7.406843e-05 | 124.5s
# Epoch 10 | 03800/06880 | loss 3.4977 | ppl 33.04 | lr 7.401206e-05 | 128.0s
# Epoch 10 | 03900/06880 | loss 3.4977 | ppl 33.04 | lr 7.395581e-05 | 131.5s
# Epoch 10 | 04000/06880 | loss 3.4977 | ppl 33.04 | lr 7.38997e-05 | 134.8s
# Epoch 10 | 04100/06880 | loss 3.4981 | ppl 33.05 | lr 7.384371e-05 | 138.3s
# Epoch 10 | 04200/06880 | loss 3.4981 | ppl 33.05 | lr 7.378785e-05 | 141.9s
# Epoch 10 | 04300/06880 | loss 3.4985 | ppl 33.07 | lr 7.373211e-05 | 145.6s
# Epoch 10 | 04400/06880 | loss 3.4989 | ppl 33.08 | lr 7.36765e-05 | 149.1s
# Epoch 10 | 04500/06880 | loss 3.4993 | ppl 33.09 | lr 7.362102e-05 | 152.6s
# Epoch 10 | 04600/06880 | loss 3.4995 | ppl 33.10 | lr 7.356566e-05 | 156.0s
# Epoch 10 | 04700/06880 | loss 3.4997 | ppl 33.11 | lr 7.351043e-05 | 159.3s
# Epoch 10 | 04800/06880 | loss 3.4997 | ppl 33.11 | lr 7.345532e-05 | 162.6s
# Epoch 10 | 04900/06880 | loss 3.5002 | ppl 33.12 | lr 7.340033e-05 | 165.9s
# Epoch 10 | 05000/06880 | loss 3.5003 | ppl 33.13 | lr 7.334547e-05 | 169.2s
# Epoch 10 | 05100/06880 | loss 3.5010 | ppl 33.15 | lr 7.329073e-05 | 172.4s
# Epoch 10 | 05200/06880 | loss 3.5008 | ppl 33.14 | lr 7.323611e-05 | 175.8s
# Epoch 10 | 05300/06880 | loss 3.5009 | ppl 33.15 | lr 7.318162e-05 | 179.2s
# Epoch 10 | 05400/06880 | loss 3.5011 | ppl 33.15 | lr 7.312724e-05 | 182.6s
# Epoch 10 | 05500/06880 | loss 3.5015 | ppl 33.17 | lr 7.307299e-05 | 185.9s
# Epoch 10 | 05600/06880 | loss 3.5015 | ppl 33.17 | lr 7.301886e-05 | 189.2s
# Epoch 10 | 05700/06880 | loss 3.5015 | ppl 33.16 | lr 7.296485e-05 | 192.5s
# Epoch 10 | 05800/06880 | loss 3.5014 | ppl 33.16 | lr 7.291095e-05 | 195.9s
# Epoch 10 | 05900/06880 | loss 3.5013 | ppl 33.16 | lr 7.285718e-05 | 199.3s
# Epoch 10 | 06000/06880 | loss 3.5014 | ppl 33.16 | lr 7.280353e-05 | 202.7s
# Epoch 10 | 06100/06880 | loss 3.5017 | ppl 33.17 | lr 7.274999e-05 | 206.1s
# Epoch 10 | 06200/06880 | loss 3.5017 | ppl 33.17 | lr 7.269657e-05 | 209.4s
# Epoch 10 | 06300/06880 | loss 3.5017 | ppl 33.17 | lr 7.264327e-05 | 212.7s
# Epoch 10 | 06400/06880 | loss 3.5019 | ppl 33.18 | lr 7.259009e-05 | 216.1s
# Epoch 10 | 06500/06880 | loss 3.5022 | ppl 33.19 | lr 7.253702e-05 | 219.8s
# Epoch 10 | 06600/06880 | loss 3.5022 | ppl 33.19 | lr 7.248407e-05 | 223.4s
# Epoch 10 | 06700/06880 | loss 3.5025 | ppl 33.20 | lr 7.243124e-05 | 226.9s
# Epoch 10 | 06800/06880 | loss 3.5027 | ppl 33.20 | lr 7.237852e-05 | 230.1s
# Epoch 10 complete | train loss 3.5026 | dev loss 3.5672 | train BLEU 20.27 | dev BLEU 16.13 | 242.5s
# {
#   "best_epoch": 10,
#   "best_dev_bleu": 16.132226569783963,
#   "best_dev_loss": 3.5671728337093986,
#   "best_model": "/data/student/i2311021/projects/100-swing-drill/chapter_10_2020/models/95_sentencepiece/best_model.pt",
#   "batch_size": 64,
#   "learning_rate": 0.0003,
#   "optimizer": "adamw"
# }
