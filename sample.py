"""
Load a trained checkpoint and generate text from a prompts file.

See prompts/gallery_mr.txt for the curated Marathi prompt gallery.
Implemented in Phase 8.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument(
        "--prompts-file",
        default="prompts/gallery_mr.txt",
        help="File with one prompt per line.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "sample.py is a Phase 1 stub. Implementation lands in Phase 8."
    )


if __name__ == "__main__":
    main()
