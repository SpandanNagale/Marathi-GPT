"""
Model forward/backward + loss sanity check.

Builds a tiny GPT (a handful of layers/heads, small enough to run on CPU in
well under a second), runs a forward pass on random token ids, and checks:
  - output logits have the expected (B, T, vocab_size) shape
  - the loss is finite and close to ln(vocab_size) - the loss a freshly
    initialized model should produce, since its output distribution over an
    untrained vocab should be close to uniform (cross-entropy of a uniform
    distribution over V classes is exactly ln(V))
  - gradients flow: loss.backward() produces finite, non-zero gradients
  - generate() produces the right output shape with valid token ids
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GPT, GPTConfig  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, vocab_size=50, dropout=0.0, bias=False)
    model = GPT(config)
    model.eval()

    B, T = 4, 12
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, config.vocab_size), f"unexpected logits shape {logits.shape}"
    print(f"PASS: logits shape {tuple(logits.shape)} matches (B={B}, T={T}, vocab_size={config.vocab_size})")

    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    expected = math.log(config.vocab_size)
    assert abs(loss.item() - expected) < 0.5, (
        f"loss {loss.item():.3f} too far from ln(vocab_size)={expected:.3f} for a freshly-initialized model"
    )
    print(f"PASS: loss {loss.item():.3f} is close to ln(vocab_size)={expected:.3f}")

    model.train()
    logits, loss = model(idx, targets)
    loss.backward()
    grad = model.wte.weight.grad
    assert grad is not None, "no gradient reached the embedding weight"
    assert torch.isfinite(grad).all(), "embedding weight gradient contains NaN/Inf"
    assert grad.abs().sum() > 0, "embedding weight gradient is all zero"
    print("PASS: backward() produces finite, non-zero gradients")

    model.eval()
    prompt = torch.randint(0, config.vocab_size, (2, 3))
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (2, 8), f"unexpected generate() shape {out.shape}"
    assert out.min() >= 0 and out.max() < config.vocab_size, "generated ids out of vocab range"
    print(f"PASS: generate() produced shape {tuple(out.shape)} with valid token ids")

    out_p = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_p=0.9)
    assert out_p.shape == (2, 8)
    print("PASS: generate() with top_p sampling also produces the expected shape")

    print(f"\nmodel params: {model.get_num_params():,}")
    print("test_model: ALL PASS")


if __name__ == "__main__":
    main()
