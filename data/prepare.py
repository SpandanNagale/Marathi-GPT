"""
Tokenize the full cleaned corpus (data/clean/) into uint16 memmap .bin shards
+ .idx offset metadata (nanoGPT pattern), split into:

  train.bin      the training set. Wikipedia documents are written twice
                 (--wiki-upweight, default 2x) since it's small but high
                 quality - the model sees ~2 epochs of it per 1 epoch of
                 everything else.
  val.bin        a general validation split (--val-fraction, default 0.5%),
                 stratified by source.
  wiki_val.bin   a held-out Wikipedia-only slice (--wiki-eval-fraction,
                 default 2%) for reporting Wikipedia perplexity separately
                 from the general val split (see eval/perplexity.py).

Split assignment is a single deterministic hash bucket per (source, doc_id):
bucket 0..val_fraction -> val; if wikipedia, next wiki_eval_fraction -> wiki_val;
everything else -> train. Applying the same thresholds regardless of source
means val is naturally stratified (proportional to each source's share of
the corpus) without a separate per-source quota pass - the same trick used
in train_tokenizer.py's sampling and clean.py has nothing to do with this,
it's specific to this split.

Each {split}.bin is a flat array of uint16 token ids (dtype chosen because
our 32,000-piece vocab fits comfortably under 65,536), with the tokenizer's
<|endoftext|> id appended after every document. {split}.idx is a flat int64
array of each document's starting token offset within the .bin file (last
document's end is the file length). int64, not uint64: numpy silently
upcasts uint64-minus-python-int to float64 (a real gotcha we hit while
testing this), which then breaks any downstream indexing like idx[i+1]-1 -
int64 has no such trap and comfortably covers any realistic token count.
A meta.json records vocab size, dtype, the eos id, and per-split doc/token
counts.

Single-process: SentencePiece's encode() is C++-fast, and splitting/upweighting
requires writing to shared output files in a fixed order, so multiprocessing
here would need real synchronization for uncertain benefit - not worth it
against "boring, clear code" for a step that runs once per corpus version.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xxhash

from shard_io import iter_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in-dir", default="data/clean", help="Directory of cleaned jsonl.zst shards.")
    parser.add_argument("--out-dir", default="data/bin", help="Directory to write memmap .bin/.idx shards.")
    parser.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer",
        help="Directory containing the trained SentencePiece model.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument(
        "--wiki-eval-fraction",
        type=float,
        default=0.02,
        help="Fraction of Wikipedia docs reserved as a separate held-out eval slice.",
    )
    parser.add_argument(
        "--wiki-upweight",
        type=int,
        default=2,
        help="Number of times Wikipedia documents are repeated in the train split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many docs from the first shard of each source (fast test mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute split/token counts without writing any .bin/.idx/meta files.",
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


def assign_split(source: str, doc_id: str, val_fraction: float, wiki_eval_fraction: float) -> str:
    h = xxhash.xxh64(f"{source}:{doc_id}".encode("utf-8")).intdigest()
    frac = (h % 1_000_000) / 1_000_000
    if frac < val_fraction:
        return "val"
    if source == "wikipedia" and frac < val_fraction + wiki_eval_fraction:
        return "wiki_val"
    return "train"


class SplitWriter:
    """Appends uint16 tokens to a flat .bin file and tracks doc start offsets for .idx."""

    def __init__(self, out_dir: Path, name: str, dry_run: bool):
        self.dry_run = dry_run
        self.docs = 0
        self.tokens = 0
        self.offsets = [0]
        self._fh = None if dry_run else open(out_dir / f"{name}.bin", "wb")

    def write_doc(self, ids: list) -> None:
        self.docs += 1
        self.tokens += len(ids)
        self.offsets.append(self.tokens)
        if not self.dry_run:
            np.array(ids, dtype=np.uint16).tofile(self._fh)

    def close(self, out_dir: Path, name: str) -> None:
        if self._fh:
            self._fh.close()
        if not self.dry_run:
            np.array(self.offsets, dtype=np.int64).tofile(out_dir / f"{name}.idx")


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    tokenizer_dir = Path(args.tokenizer_dir)

    import sentencepiece as spm

    model_path = tokenizer_dir / "marathi_bpe.model"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found - run data/train_tokenizer.py first.")
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    if sp.vocab_size() > 65536:
        raise SystemExit(
            f"vocab_size={sp.vocab_size()} does not fit in uint16 (max 65536) - "
            "retrain the tokenizer with a smaller --vocab-size or widen the dtype here."
        )
    eos_id = sp.piece_to_id("<|endoftext|>")

    sources = discover_sources(in_dir)
    if not sources:
        raise SystemExit(f"No source subdirectories found under {in_dir}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    writers = {name: SplitWriter(out_dir, name, args.dry_run) for name in ("train", "val", "wiki_val")}

    print(f"=== tokenizing {len(sources)} source(s) into train/val/wiki_val ===")
    start = time.time()
    docs_seen = 0
    for source in sources:
        source_docs = 0
        for rec in _iter_source_docs(in_dir, source, args.limit):
            doc_id = str(rec.get("id", ""))
            text = rec["text"]
            if not text.strip():
                continue

            split = assign_split(source, doc_id, args.val_fraction, args.wiki_eval_fraction)
            ids = sp.encode(text, out_type=int) + [eos_id]

            repeats = args.wiki_upweight if (split == "train" and source == "wikipedia") else 1
            for _ in range(repeats):
                writers[split].write_doc(ids)

            source_docs += 1
            docs_seen += 1
            if docs_seen % 2000 == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(f"\r{docs_seen:,} docs tokenized | {docs_seen / elapsed:.1f} docs/sec", end="", flush=True)
        print(f"\n[{source}] {source_docs:,} docs processed")

    print()
    meta = {
        "vocab_size": sp.vocab_size(),
        "dtype": "uint16",
        "eos_token_id": eos_id,
        "wiki_upweight": args.wiki_upweight,
        "splits": {},
    }
    print(f"{'split':<10} {'docs':>12} {'tokens':>14}")
    for name, w in writers.items():
        w.close(out_dir, name)
        meta["splits"][name] = {"docs": w.docs, "tokens": w.tokens}
        print(f"{name:<10} {w.docs:>12,} {w.tokens:>14,}")

    if args.dry_run:
        print("\n[prepare] --dry-run: no .bin/.idx/meta.json files written")
        return

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {meta_path} and .bin/.idx files to {out_dir}")


if __name__ == "__main__":
    main()
