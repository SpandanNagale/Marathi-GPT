"""
Single-file raw PyTorch training loop for MarathiGPT.

bf16 autocast, torch.compile, AdamW, cosine LR with warmup, grad accumulation,
memmap dataloader, wandb logging, checkpointing with resume support.
See marathi-gpt-project-plan.md §5. Implemented in Phase 7.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to a configs/*.yaml file."
    )
    parser.add_argument(
        "--resume", default=None, help="Path to a checkpoint to resume from."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 50 steps and report tokens/sec, then exit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "train.py is a Phase 1 stub. Implementation lands in Phase 7."
    )


if __name__ == "__main__":
    main()
