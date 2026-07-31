"""
Tokenize the full cleaned corpus into uint16 memmap .bin shards + .idx metadata
(nanoGPT pattern), with a train/val split (val=0.5%, stratified by source) plus
a held-out Wikipedia slice for eval. Wikipedia upweighted 2x in the train mix.
See marathi-gpt-project-plan.md §5. Implemented in Phase 5.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir", default="data/clean", help="Directory of cleaned jsonl.zst shards."
    )
    parser.add_argument(
        "--out-dir", default="data/bin", help="Directory to write memmap .bin shards."
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer",
        help="Directory containing the trained SentencePiece model.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many documents (fast test mode).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "data/prepare.py is a Phase 1 stub. Implementation lands in Phase 5."
    )


if __name__ == "__main__":
    main()
