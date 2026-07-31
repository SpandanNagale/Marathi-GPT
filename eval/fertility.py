"""
Compare tokens-per-word fertility on held-out Marathi text between our tokenizer,
GPT-4's (tiktoken cl100k/o200k) and Llama-3's. Outputs a markdown table and a
matplotlib chart saved to assets/. Implemented in Phase 4.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer",
        help="Directory containing our trained SentencePiece model.",
    )
    parser.add_argument("--eval-text", default=None, help="Held-out Marathi text file.")
    parser.add_argument("--out-dir", default="assets")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "eval/fertility.py is a Phase 1 stub. Implementation lands in Phase 4."
    )


if __name__ == "__main__":
    main()
