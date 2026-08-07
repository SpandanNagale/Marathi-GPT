"""
Full fine-tune of a pretrained checkpoint on the instruction data built by
build_instruct_data.py. Reuses train.py's infra (bf16 autocast, torch.compile,
AdamW, cosine LR, checkpointing) but with an in-memory, padded, per-example
dataloader instead of train.py's continuous memmap stream - the SFT set is
small enough (thousands, not billions of tokens) to just tokenize once and
keep in RAM, and each example needs its own loss mask (only the assistant's
response contributes to the loss, not the `<|user|>` prompt).

No LoRA: at these model sizes (tens of millions of params) a full fine-tune
is cheap enough to just do directly.
"""

import argparse
import functools
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from model import ...`
from model import GPT, GPTConfig

print = functools.partial(print, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Pretrained checkpoint to initialize from.")
    parser.add_argument("--data-dir", default="data/sft", help="Directory with train.jsonl / val.jsonl.")
    parser.add_argument("--tokenizer", default="data/tokenizer/marathi_bpe.model")
    parser.add_argument("--checkpoint-dir", default="checkpoints/sft")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Full fine-tune: much lower than pretraining.")
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true", help="Run 2 real steps on real data, then exit.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data: tokenize the whole (small) instruction set once, keep it in RAM.
# ---------------------------------------------------------------------------


def tokenize_example(sp, eos_id: int, block_size: int, instruction: str, response: str):
    prompt_ids = sp.encode(f"<|user|>\n{instruction}\n<|assistant|>\n", out_type=int)
    response_ids = sp.encode(response, out_type=int) + [eos_id]
    full_ids = prompt_ids + response_ids
    if len(full_ids) > block_size:
        return None
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return full_ids, loss_mask


def load_examples(path: Path, sp, eos_id: int, block_size: int):
    examples, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out = tokenize_example(sp, eos_id, block_size, row["instruction"], row["response"])
            if out is None:
                skipped += 1
            else:
                examples.append(out)
    if skipped:
        print(f"  skipped {skipped}/{skipped + len(examples)} examples longer than block_size={block_size}")
    return examples


def make_batch(examples: list, pad_id: int, device: str):
    """Right-pad a list of (full_ids, loss_mask) into an (x, y) batch. Padded
    and prompt positions get target -1 (model.py's cross_entropy ignores it)."""
    max_len = max(len(ids) for ids, _ in examples)
    B, T = len(examples), max_len - 1
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), -1, dtype=torch.long)
    for i, (ids, mask) in enumerate(examples):
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.long)
        L = len(ids) - 1
        x[i, :L] = ids_t[:-1]
        y_i = ids_t[1:].clone()
        y_i[mask_t[1:] == 0] = -1
        y[i, :L] = y_i
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ---------------------------------------------------------------------------
# LR schedule, optimizer, checkpointing - same shape as train.py
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def configure_optimizer(model: GPT, weight_decay: float, lr: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())


def raw_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def estimate_val_loss(model, val_examples, pad_id, batch_size, device, ctx) -> float:
    if not val_examples:
        return float("nan")
    model.eval()
    losses = []
    for start in range(0, len(val_examples), batch_size):
        batch = val_examples[start : start + batch_size]
        x, y = make_batch(batch, pad_id, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()
    print(f"device={device} dtype={dtype}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = GPTConfig(**ckpt["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"initialized from {args.checkpoint} (pretrain step={ckpt.get('step', '?')})")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos_id = sp.piece_to_id("<|endoftext|>")

    data_dir = Path(args.data_dir)
    train_examples = load_examples(data_dir / "train.jsonl", sp, eos_id, model_config.block_size)
    val_path = data_dir / "val.jsonl"
    val_examples = load_examples(val_path, sp, eos_id, model_config.block_size) if val_path.exists() else []
    print(f"train examples: {len(train_examples)}, val examples: {len(val_examples)}")

    optimizer = configure_optimizer(model, args.weight_decay, args.learning_rate)
    step = 0
    if args.resume:
        rckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model(model).load_state_dict(rckpt["model"])
        optimizer.load_state_dict(rckpt["optimizer"])
        step = rckpt["step"]
        print(f"resumed from {args.resume} at step {step}")

    if args.compile:
        model = torch.compile(model)

    steps_per_epoch = max(1, len(train_examples) // args.micro_batch_size)
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_ratio * max_steps))

    if args.dry_run:
        print("[dry-run] running 2 real steps on real data, then exiting (no checkpoints)")
        max_steps = min(max_steps, step + 2)

    print(f"max_steps={max_steps} ({steps_per_epoch} steps/epoch x {args.epochs} epochs)")

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(project="marathi-gpt-sft", config=vars(args))
        except Exception as e:
            print(f"[sft] --wandb requested but init failed ({e}); continuing without it")

    ckpt_dir = Path(args.checkpoint_dir)
    best_val_loss = float("inf")
    order = list(range(len(train_examples)))
    epoch_pos = len(order)  # force a reshuffle on the first step
    start_time = time.time()

    while step < max_steps:
        if epoch_pos >= len(order):
            rng.shuffle(order)
            epoch_pos = 0
        idxs = order[epoch_pos : epoch_pos + args.micro_batch_size]
        epoch_pos += args.micro_batch_size
        batch = [train_examples[i] for i in idxs]

        lr = get_lr(step, warmup_steps, max_steps, args.learning_rate, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.time()
        x, y = make_batch(batch, eos_id, device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        step += 1
        if step % 10 == 0 or step == max_steps:
            elapsed_h = (time.time() - start_time) / 3600
            print(f"step {step}/{max_steps} loss={loss.item():.4f} lr={lr:.2e} {dt * 1000:.0f}ms/step {elapsed_h:.2f}h elapsed")
            if wandb_run:
                wandb_run.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

        if step % args.eval_interval == 0 or step == max_steps:
            val_loss = estimate_val_loss(model, val_examples, eos_id, args.micro_batch_size, device, ctx)
            print(f"  eval: val_loss={val_loss:.4f} (best={min(best_val_loss, val_loss):.4f})")
            if wandb_run:
                wandb_run.log({"val/loss": val_loss}, step=step)
            if val_loss < best_val_loss and not args.dry_run:
                best_val_loss = val_loss
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": raw_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "model_config": vars(model_config),
                    },
                    ckpt_dir / "best.pt",
                )

        if not args.dry_run and (step % args.checkpoint_interval == 0 or step == max_steps):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": raw_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "model_config": vars(model_config),
                },
                ckpt_dir / "latest.pt",
            )
            print(f"  saved checkpoint at step {step}")

        if args.dry_run and step >= max_steps:
            break

    print(f"\ndone: {step} steps, best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
