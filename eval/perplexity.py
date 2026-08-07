"""
Compute held-out perplexity, reported separately for every split named in
data/bin/meta.json's "splits" (currently "val" and "wiki_val" - see
data/prepare.py). Walks each split's memmap exhaustively in non-overlapping
block_size chunks (not a sampled estimate like train.py's estimate_val_loss)
so the number is a real, reproducible measurement.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from model import ...`
from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--data-dir", default="data/bin", help="Directory with <split>.bin + meta.json.")
    parser.add_argument(
        "--splits", default=None, help="Comma-separated split names to evaluate. Default: all splits in meta.json."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Cap chunks evaluated per split, for a fast check.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --limit 4.")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def split_perplexity(model: GPT, data_path: Path, block_size: int, batch_size: int, device: str, limit=None):
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    n_chunks = (len(data) - 1) // block_size
    if limit is not None:
        n_chunks = min(n_chunks, limit)

    total_loss, total_tokens = 0.0, 0
    for start in range(0, n_chunks, batch_size):
        chunk_ids = range(start, min(start + batch_size, n_chunks))
        xs, ys = [], []
        for c in chunk_ids:
            lo = c * block_size
            xs.append(data[lo : lo + block_size].astype(np.int64))
            ys.append(data[lo + 1 : lo + 1 + block_size].astype(np.int64))
        x = torch.from_numpy(np.stack(xs)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        _, loss = model(x, y)  # mean cross-entropy over this batch
        n_tok = x.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    mean_loss = total_loss / total_tokens
    del data
    return mean_loss, total_tokens


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.limit = args.limit or 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    block_size = model.config.block_size

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    available_splits = list(meta["splits"].keys())
    splits = args.splits.split(",") if args.splits else available_splits

    print(f"checkpoint={args.checkpoint} block_size={block_size} device={device}")
    print(f"{'split':<12} {'tokens':>10} {'loss':>8} {'perplexity':>12}")
    for split in splits:
        path = data_dir / f"{split}.bin"
        if not path.exists():
            print(f"{split:<12} (missing: {path})", file=sys.stderr)
            continue
        loss, n_tokens = split_perplexity(model, path, block_size, args.batch_size, device, args.limit)
        ppl = float(np.exp(loss))
        print(f"{split:<12} {n_tokens:>10,} {loss:>8.4f} {ppl:>12.2f}")


if __name__ == "__main__":
    main()
