"""
Train a SentencePiece BPE tokenizer on a stratified sample of the cleaned corpus.

vocab_size=32000, byte-fallback on, NFC-normalized, with reserved special tokens
<|endoftext|>, <|user|>, <|assistant|>. See marathi-gpt-project-plan.md §3.
Implemented in Phase 4.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir", default="data/clean", help="Directory of cleaned jsonl.zst shards."
    )
    parser.add_argument(
        "--out-dir", default="data/tokenizer", help="Directory to write tokenizer model."
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--sample-gb",
        type=float,
        default=3.0,
        help="Approx GB of stratified sample to train on.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use this many documents (fast test mode).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "data/train_tokenizer.py is a Phase 1 stub. Implementation lands in Phase 4."
    )


if __name__ == "__main__":
    main()
