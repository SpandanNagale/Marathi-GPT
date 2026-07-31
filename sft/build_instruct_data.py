"""
Assemble Marathi instruction data: ai4bharat indic-instruct Marathi split +
translated Alpaca-cleaned, plus a hook for hand-written examples in
sft/manual_examples.jsonl. Formats with the chat special tokens from Phase 4.
Implemented in Phase 9.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument(
        "--manual-examples",
        default="sft/manual_examples.jsonl",
        help="Hand-written examples to fold into the instruction set.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "sft/build_instruct_data.py is a Phase 1 stub. Implementation lands in Phase 9."
    )


if __name__ == "__main__":
    main()
