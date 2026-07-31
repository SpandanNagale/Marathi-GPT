"""
Full fine-tune of the 124M base model on instruction data, lower LR, same infra
as train.py. Implemented in Phase 9.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint.")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "sft/finetune.py is a Phase 1 stub. Implementation lands in Phase 9."
    )


if __name__ == "__main__":
    main()
