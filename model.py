"""
MarathiGPT model: single-file decoder-only transformer.

Pre-norm, RMSNorm, RoPE, SwiGLU MLP, no biases, weight tying, GPT-2 init,
attention via F.scaled_dot_product_attention. See marathi-gpt-project-plan.md §4.
Implemented in Phase 6.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0
    bias: bool = False


# GPTConfig.from_yaml(path) and the GPT nn.Module (with .generate()) land in Phase 6.
