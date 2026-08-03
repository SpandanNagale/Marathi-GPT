"""
Single-file raw PyTorch training loop for MarathiGPT (nanoGPT-style: no
Lightning/Accelerate). bf16 autocast, torch.compile, AdamW, cosine LR with
warmup, gradient accumulation to a fixed effective batch size, a memmap
dataloader over data/prepare.py's uint16 .bin shards, checkpointing with
resume support, and optional wandb logging.

Windows note: no multiprocessing/DataLoader workers are used (the memmap
dataloader just slices numpy arrays on the main thread), so there's no
spawn-related hazard here - the `if __name__ == "__main__":` guard below is
just standard hygiene.
"""

import argparse
import functools
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from model import GPT, GPTConfig

# Runs can take hours and are often redirected to a log file / background
# process; unbuffered output means `tail`-ing that file actually shows
# progress instead of sitting empty until the process exits.
print = functools.partial(print, flush=True)

# RTX 5070 dense (non-sparse) BF16 tensor throughput, from NVIDIA's Blackwell
# whitepaper - used only to report Model FLOPs Utilization (MFU), a diagnostic
# ratio of achieved-vs-peak compute, not a hard requirement.
GPU_PEAK_BF16_TFLOPS = 61.73


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file.")
    parser.add_argument("--data-dir", default="data/bin", help="Directory with train.bin/val.bin/meta.json.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoints. Defaults to checkpoints/<config name>/.",
    )
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override the config's train.max_steps."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 50 steps on random data and report tokens/sec + MFU, then exit (no checkpoints).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a couple of real steps on real data to sanity-check the loop, then exit.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (opt-in).")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def load_train_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["train"]


# ---------------------------------------------------------------------------
# Data loading - memmap over data/prepare.py's uint16 .bin shards
# ---------------------------------------------------------------------------


def get_batch(data_dir: Path, split: str, block_size: int, batch_size: int, device: str):
    # Re-opened every call (not cached) - a small, well-known nanoGPT pattern
    # that sidesteps a memmap memory-leak footgun in some numpy versions.
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ---------------------------------------------------------------------------
# LR schedule: cosine with linear warmup
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Optimizer: weight decay on matrices (ndim>=2), not on RMSNorm's ndim==1 weights
# ---------------------------------------------------------------------------


def configure_optimizer(model: GPT, weight_decay: float, lr: float, betas: tuple) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())


@torch.no_grad()
def estimate_val_loss(model, data_dir, block_size, batch_size, device, ctx, eval_iters) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data_dir, "val", block_size, batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def raw_model(model):
    """Unwrap torch.compile's OptimizedModule to get the plain GPT for state_dict."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(ckpt_dir: Path, model, optimizer, step, best_val_loss, model_config, keep_last=3):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": vars(model_config),
    }
    path = ckpt_dir / f"ckpt_{step:07d}.pt"
    torch.save(payload, path)
    torch.save(payload, ckpt_dir / "latest.pt")

    rolling = sorted(ckpt_dir.glob("ckpt_*.pt"))
    for old in rolling[:-keep_last]:
        old.unlink()
    return path


def load_checkpoint(path: Path, device: str) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Benchmark mode: throughput/MFU on random data, no dataset required
# ---------------------------------------------------------------------------


def run_benchmark(model_config, train_cfg, device, ctx, n_steps=50):
    model = GPT(model_config).to(device)
    if train_cfg["compile"]:
        model = torch.compile(model)
    optimizer = configure_optimizer(model, train_cfg["weight_decay"], train_cfg["learning_rate"], (train_cfg["beta1"], train_cfg["beta2"]))

    micro_bs, block_size, grad_accum = train_cfg["micro_batch_size"], model_config.block_size, train_cfg["grad_accum_steps"]
    idx = torch.randint(0, model_config.vocab_size, (micro_bs, block_size), device=device)
    targets = torch.randint(0, model_config.vocab_size, (micro_bs, block_size), device=device)

    def step_fn():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with ctx:
                _, loss = model(idx, targets)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()

    step_fn()  # warmup / compile
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_steps):
        step_fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    tokens_per_step = micro_bs * block_size * grad_accum
    tokens_per_sec = tokens_per_step * n_steps / elapsed
    print(f"\n[benchmark] {n_steps} steps in {elapsed:.1f}s -> {tokens_per_sec:,.0f} tokens/sec")
    if device == "cuda":
        flops_per_token = 6 * raw_model(model).get_num_params() + 12 * model_config.n_layer * model_config.n_head * (
            model_config.n_embd // model_config.n_head
        ) * block_size
        flops_achieved = flops_per_token * tokens_per_sec
        mfu = flops_achieved / (GPU_PEAK_BF16_TFLOPS * 1e12)
        print(f"[benchmark] estimated MFU: {mfu:.1%} of {GPU_PEAK_BF16_TFLOPS} TFLOPS peak (bf16, dense)")
    print(f"[benchmark] peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if device == "cuda" else "")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_config = GPTConfig.from_yaml(args.config)
    train_cfg = load_train_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if train_cfg["dtype"] == "bfloat16" and device == "cuda" else torch.float32
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()
    print(f"device={device} dtype={dtype} params(config)={args.config}")

    if args.benchmark:
        run_benchmark(model_config, train_cfg, device, ctx)
        return

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["vocab_size"] != model_config.vocab_size:
        raise SystemExit(
            f"vocab_size mismatch: tokenizer/meta.json says {meta['vocab_size']}, "
            f"{args.config} says {model_config.vocab_size}"
        )

    config_name = Path(args.config).stem
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints") / config_name

    model = GPT(model_config).to(device)
    optimizer = configure_optimizer(
        model, train_cfg["weight_decay"], train_cfg["learning_rate"], (train_cfg["beta1"], train_cfg["beta2"])
    )

    step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), device)
        raw_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {args.resume} at step {step}, best_val_loss={best_val_loss:.4f}")

    if train_cfg["compile"]:
        model = torch.compile(model)

    max_steps = args.max_steps or train_cfg["max_steps"]
    warmup_steps = max(1, int(train_cfg["warmup_ratio"] * max_steps))
    block_size = model_config.block_size
    micro_bs = train_cfg["micro_batch_size"]
    grad_accum = train_cfg["grad_accum_steps"]
    tokens_per_step = micro_bs * block_size * grad_accum

    if args.dry_run:
        print("[dry-run] running 2 real steps on real data, then exiting (no checkpoints)")
        max_steps = min(max_steps, step + 2)

    print(
        f"training {config_name}: max_steps={max_steps} tokens/step={tokens_per_step:,} "
        f"({tokens_per_step * max_steps / 1e9:.2f}B tokens total)"
    )

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(project="marathi-gpt", name=config_name, config=train_cfg)
        except Exception as e:
            print(f"[train] --wandb requested but wandb init failed ({e}); continuing without it")

    start_time = time.time()
    tokens_seen = 0
    while step < max_steps:
        lr = get_lr(step, warmup_steps, max_steps, train_cfg["learning_rate"], train_cfg["min_lr"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(data_dir, "train", block_size, micro_bs, device)
            with ctx:
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_sum += loss.item()
        train_loss = loss_sum / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_seen += tokens_per_step

        step += 1
        if step % 10 == 0 or step == max_steps:
            toks_per_sec = tokens_per_step / dt
            elapsed_h = (time.time() - start_time) / 3600
            print(
                f"step {step}/{max_steps} loss={train_loss:.4f} lr={lr:.2e} "
                f"{toks_per_sec:,.0f} tok/s {elapsed_h:.2f}h elapsed"
            )
            if wandb_run:
                wandb_run.log(
                    {"train/loss": train_loss, "train/lr": lr, "train/tokens_per_sec": toks_per_sec},
                    step=step,
                )

        if step % train_cfg["eval_interval"] == 0 or step == max_steps:
            val_loss = estimate_val_loss(model, data_dir, block_size, micro_bs, device, ctx, train_cfg["eval_iters"])
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

        if not args.dry_run and (step % train_cfg["checkpoint_interval"] == 0 or step == max_steps):
            path = save_checkpoint(ckpt_dir, model, optimizer, step, best_val_loss, model_config, train_cfg["keep_last_n"])
            print(f"  saved checkpoint: {path}")

        if args.dry_run and step >= max_steps:
            break

    print(f"\ndone: {step} steps, {tokens_seen / 1e6:.1f}M tokens, best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
