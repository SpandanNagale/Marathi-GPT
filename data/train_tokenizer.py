"""
Train a SentencePiece BPE tokenizer on a stratified sample of data/clean/.

Sampling strategy: a single global acceptance probability
p = sample_gb / total_corpus_size is applied independently to every document
across every source. This is mathematically equivalent to stratified-by-source
sampling (each source contributes in proportion to its true size in the
corpus) without needing a separate per-source quota pass.

A small, disjoint slice of documents is reserved as a held-out set (never
eligible for training) via a stable hash of (source, doc_id) - this is what
eval/fertility.py and eval/perplexity.py read by default, guaranteeing zero
overlap with the tokenizer's training sample without any cross-script
coordination.

Text is NFC-normalized before being written to the training corpus, and the
SentencePiece trainer is run with normalization_rule_name="identity" so it
does not additionally apply its default NFKC normalization on top (which can
alter rare Devanagari conjuncts/matras).
"""

import argparse
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import xxhash

from shard_io import iter_shards, estimate_tokens

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in-dir", default="data/clean", help="Directory of cleaned jsonl.zst shards.")
    parser.add_argument("--out-dir", default="data/tokenizer", help="Directory to write tokenizer model.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--sample-gb", type=float, default=3.0, help="Approx GB of stratified sample to train on.")
    parser.add_argument(
        "--holdout-permille",
        type=float,
        default=10.0,
        help="Per-mille (parts per 1000) of docs reserved as held-out, never used for training.",
    )
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only consider this many docs from the first shard of each source (fast test mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute sampling stats without writing the sample corpus or training a model.",
    )
    return parser.parse_args()


def _iter_source_docs(in_dir: Path, source: str, limit):
    shard_dir = in_dir / source
    shard_paths = sorted(shard_dir.glob("shard_*.jsonl.zst"))
    if limit is not None:
        shard_paths = shard_paths[:1]
    n = 0
    for path in shard_paths:
        for rec in iter_shards(path):
            if limit is not None and n >= limit:
                return
            yield rec
            n += 1


def discover_sources(in_dir: Path) -> list:
    return sorted(p.name for p in in_dir.iterdir() if p.is_dir() and p.name != "_stage1")


def is_holdout(source: str, doc_id: str, holdout_permille: float) -> bool:
    h = xxhash.xxh64(f"{source}:{doc_id}".encode("utf-8")).intdigest()
    return (h % 1000) < holdout_permille


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    sources = discover_sources(in_dir)
    if not sources:
        raise SystemExit(f"No source subdirectories found under {in_dir}")

    print(f"=== pass 1/2: measuring corpus size across {len(sources)} source(s) ===")
    source_bytes = Counter()
    source_docs = Counter()
    start = time.time()
    for source in sources:
        for rec in _iter_source_docs(in_dir, source, args.limit):
            n_bytes = len(rec["text"].encode("utf-8"))
            source_bytes[source] += n_bytes
            source_docs[source] += 1
        print(f"[{source}] {source_docs[source]:,} docs, {source_bytes[source] / 1e9:.3f} GB")

    grand_total_bytes = sum(source_bytes.values())
    if grand_total_bytes == 0:
        raise SystemExit("No documents found - did you run data/clean.py first?")

    target_bytes = args.sample_gb * (1024**3)
    accept_prob = min(1.0, target_bytes / grand_total_bytes)
    print(
        f"\ncorpus total: {grand_total_bytes / 1e9:.3f} GB across "
        f"{sum(source_docs.values()):,} docs | target sample: {args.sample_gb:.2f} GB "
        f"| global accept probability: {accept_prob:.4f}"
    )

    print(f"\n=== pass 2/2: stratified sampling + NFC normalization ===")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_sample.txt"
    holdout_path = out_dir / "holdout.txt"

    sampled_bytes = Counter()
    sampled_docs = Counter()
    holdout_docs = Counter()

    train_fh = None if args.dry_run else open(train_path, "w", encoding="utf-8")
    holdout_fh = None if args.dry_run else open(holdout_path, "w", encoding="utf-8")
    try:
        for source in sources:
            for rec in _iter_source_docs(in_dir, source, args.limit):
                text = unicodedata.normalize("NFC", rec["text"])
                if not text.strip():
                    continue
                doc_id = str(rec.get("id", ""))

                if is_holdout(source, doc_id, args.holdout_permille):
                    holdout_docs[source] += 1
                    if holdout_fh:
                        holdout_fh.write(text + "\n")
                    continue

                if rng.random() < accept_prob:
                    sampled_docs[source] += 1
                    sampled_bytes[source] += len(text.encode("utf-8"))
                    if train_fh:
                        train_fh.write(text + "\n")
    finally:
        if train_fh:
            train_fh.close()
        if holdout_fh:
            holdout_fh.close()

    total_sampled_docs = sum(sampled_docs.values())
    total_sampled_bytes = sum(sampled_bytes.values())
    total_holdout_docs = sum(holdout_docs.values())
    elapsed = time.time() - start

    print(f"\n{'source':<12} {'docs':>10} {'sample_docs':>12} {'holdout_docs':>13} {'sample_GB':>10}")
    for source in sources:
        print(
            f"{source:<12} {source_docs[source]:>10,} {sampled_docs[source]:>12,} "
            f"{holdout_docs[source]:>13,} {sampled_bytes[source] / 1e9:>10.3f}"
        )
    print(
        f"\nsampled {total_sampled_docs:,} docs ({total_sampled_bytes / 1e9:.3f} GB) for training, "
        f"{total_holdout_docs:,} docs held out, in {elapsed:.1f}s"
    )

    if args.dry_run:
        print("\n[train_tokenizer] --dry-run: stopping before SentencePiece training")
        return

    if total_sampled_bytes < args.vocab_size * 100:
        print(
            f"\n[train_tokenizer] WARNING: only {total_sampled_bytes:,} bytes sampled for a "
            f"{args.vocab_size}-piece vocab. SentencePiece will still hit exactly that vocab "
            f"size, but will pad it out with meaningless low/zero-frequency pieces once real "
            f"merges run out. Fine for a quick pipeline smoke test; lower --vocab-size or "
            f"increase --sample-gb / your input data for a real tokenizer."
        )

    print(f"\n=== training SentencePiece BPE (vocab_size={args.vocab_size}) ===")
    import sentencepiece as spm

    model_prefix = str(out_dir / "marathi_bpe")
    spm.SentencePieceTrainer.train(
        input=str(train_path),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=args.character_coverage,
        normalization_rule_name="identity",  # we already NFC-normalized; avoid SentencePiece's default NFKC
        user_defined_symbols=SPECIAL_TOKENS,
        input_sentence_size=20_000_000,
        shuffle_input_sentence=True,
        seed_sentencepiece_size=1_000_000,
        max_sentence_length=65536,  # default (4192 bytes) silently *drops* longer lines/paragraphs
        num_threads=16,
    )
    print(f"Wrote {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    main()
