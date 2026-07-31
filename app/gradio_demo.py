"""
Simple Gradio chat UI over the SFT model, CPU-capable. Implemented in Phase 9.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SFT model checkpoint.")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "app/gradio_demo.py is a Phase 1 stub. Implementation lands in Phase 9."
    )


if __name__ == "__main__":
    main()
