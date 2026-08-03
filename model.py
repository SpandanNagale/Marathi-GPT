"""
MarathiGPT model: a single-file decoder-only transformer, nanoGPT-shaped but
with the 2020s-era upgrades: RMSNorm instead of LayerNorm, RoPE instead of
learned position embeddings, SwiGLU instead of a plain GELU MLP, no bias
terms anywhere, and weight tying between the token embedding and the output
head. See marathi-gpt-project-plan.md §4 for the full rationale.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0
    bias: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "GPTConfig":
        """Load the `model:` section of a configs/*.yaml file (nano/small/base)."""
        import yaml

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m = cfg["model"]
        return cls(
            n_layer=m["n_layer"],
            n_head=m["n_head"],
            n_embd=m["n_embd"],
            block_size=m["block_size"],
            vocab_size=m["vocab_size"],
            dropout=m.get("dropout", 0.0),
            bias=m.get("bias", False),
        )


class RMSNorm(nn.Module):
    """RMSNorm: like LayerNorm but without mean-centering or a bias term -
    just rescale by the root-mean-square of the activations. Computed in
    float32 regardless of input dtype, since the mean-of-squares reduction
    loses precision fast under bf16 - a well-known gotcha with this norm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute the cos/sin tables for rotary position embeddings.

    Standard RoPE: pair up each head's dimensions, rotate each pair by an
    angle proportional to its position and inversely proportional to its
    frequency band. Returns (cos, sin) of shape (max_seq_len, head_dim),
    ready to broadcast against a (B, n_head, T, head_dim) tensor.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim). cos/sin: (T, head_dim), broadcast over B and n_head."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Flash attention via PyTorch's fused kernel; is_causal handles the mask.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


def _swiglu_hidden_dim(n_embd: int) -> int:
    """SwiGLU has 3 weight matrices instead of an MLP's 2, so its hidden dim
    is shrunk to 8/3 * n_embd (rounded up to a multiple of 64 for GPU
    efficiency) to keep total MLP parameter count roughly comparable."""
    hidden = int(8 * n_embd / 3)
    return ((hidden + 63) // 64) * 64


class SwiGLUMLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = _swiglu_hidden_dim(config.n_embd)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """Pre-norm transformer block: norm -> attn -> residual, norm -> mlp -> residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying (embed == unembed)

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 residual scaling: the last linear in each residual branch
        # (attention's out_proj, MLP's down_proj) gets an extra 1/sqrt(2*n_layer)
        # shrink, since its output adds directly into the residual stream and
        # there are 2*n_layer such additions accumulating variance. gate_proj/
        # up_proj/qkv_proj do NOT get this - only the branch's final projection.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} exceeds block_size {self.config.block_size}"

        x = self.drop(self.wte(idx))
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. idx: (B, T) prompt token ids. Returns (B, T+max_new_tokens)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # drop tokens once cumulative probability exceeds top_p, but
                # always keep at least the single highest-probability token
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(sorted_remove, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
