"""
Compute held-out perplexity on the Wikipedia-mr slice and the general val split,
reported separately. Implemented in Phase 8.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/bin")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "eval/perplexity.py is a Phase 1 stub. Implementation lands in Phase 8."
    )


if __name__ == "__main__":
    main()
