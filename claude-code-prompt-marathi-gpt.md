# Claude Code Prompt — MarathiGPT

Paste everything below the line into Claude Code from an empty `marathi-gpt/` directory. Recommended: also drop the project plan file (`marathi-gpt-project-plan.md`) into the repo root first and mention it — Claude Code will read it for full context.

---

I'm building **MarathiGPT** — a GPT-style language model pretrained from scratch on Marathi text, in the spirit of Karpathy's nanoGPT, as a portfolio project targeting Indic LLM companies (Sarvam AI, AI4Bharat). Read `marathi-gpt-project-plan.md` in the repo root for the full blueprint before writing any code.

## My environment
- Windows 11, RTX 5070 (12GB VRAM), Ryzen 7 9800X3D, 64GB RAM
- Python 3.11+, PyTorch 2.x with CUDA. Assume `torch.compile` and bf16 both work.
- Datasets will come from HuggingFace Hub (`datasets` library) — I have disk space for ~100GB of raw data.

## What to build

Build the project in the exact order below. **After each phase, stop, show me how to run it, and wait for my confirmation before continuing.** Every script must be runnable standalone with sensible CLI args (`argparse`), have a `--limit` / `--dry-run` mode for fast testing on tiny data, and print clear progress/stats.

### Phase 1 — Repo scaffold
Create this structure with stub files, a `.gitignore` (ignore `data/raw`, `data/clean`, `data/bin`, checkpoints, wandb), `requirements.txt`, and a README skeleton:

```
marathi-gpt/
├── configs/ (nano.yaml, small.yaml, base.yaml)
├── data/ (download.py, clean.py, train_tokenizer.py, prepare.py)
├── model.py
├── train.py
├── sample.py
├── eval/ (perplexity.py, mahasent_fewshot.py, fertility.py)
├── sft/ (build_instruct_data.py, finetune.py)
└── app/ (gradio_demo.py)
```

### Phase 2 — Data download (`data/download.py`)
Stream-download Marathi splits to `data/raw/` as zstd-compressed jsonl shards (~500MB each), one subdir per source:
1. `ai4bharat/sangraha` — Marathi, **verified split only**
2. IndicCorp v2 (AI4Bharat) — Marathi
3. L3Cube MahaCorpus
4. Marathi Wikipedia (latest dump via HF `wikimedia/wikipedia`, `20xx.mr` config)

Use streaming mode (don't load whole datasets in RAM), resumable per-source, `--source` flag to download one at a time, and print token-count estimates as it goes. If a dataset ID doesn't exist or has changed, search for the current correct HF dataset ID rather than guessing.

### Phase 3 — Cleaning pipeline (`data/clean.py`)
Multiprocessing pipeline: raw shards → `data/clean/`:
1. fastText langID (`lid.176.bin`), keep documents with mr confidence ≥ 0.65
2. Devanagari character ratio ≥ 0.70
3. Gopher-style quality heuristics (min/max word count, symbol-to-word ratio, repeated-line fraction, boilerplate lines)
4. Exact dedup via xxhash on NFC-normalized text
5. Fuzzy dedup: MinHash LSH, Jaccard ≥ 0.8 (use `datasketch`; shard-friendly, bounded memory)
6. PII scrub: Indian phone numbers, emails, Aadhaar-like 12-digit patterns
7. Write a `stats.json` + printed table: docs and tokens surviving each stage, per source

### Phase 4 — Tokenizer (`data/train_tokenizer.py` + `eval/fertility.py`)
- SentencePiece BPE, vocab 32,000, byte-fallback, NFC normalization, trained on a stratified ~3GB sample of cleaned data
- Special tokens: `<|endoftext|>`, `<|user|>`, `<|assistant|>`
- `fertility.py`: compare tokens-per-word on held-out Marathi text between my tokenizer, GPT-4's (`tiktoken` cl100k/o200k), and Llama-3's — output a markdown table and a matplotlib chart saved to `assets/`

### Phase 5 — Pre-tokenization (`data/prepare.py`)
Tokenize the full clean corpus into uint16 memmap `.bin` shards + `.idx` metadata (nanoGPT pattern), with a train/val split (val = 0.5%, stratified by source, plus a separate held-out Wikipedia slice for eval). Upweight Wikipedia 2x in the train mix.

### Phase 6 — Model (`model.py`)
Single-file decoder-only transformer, target ~300 readable lines, heavily commented since this is a portfolio piece:
- Pre-norm, **RMSNorm**, **RoPE**, **SwiGLU** MLP (hidden = 8/3·d_model rounded to /64), no biases, weight tying
- Attention via `F.scaled_dot_product_attention` (Flash)
- GPT-2 init (normal 0.02, residual scaling 1/√(2·n_layer))
- `GPTConfig` dataclass loaded from the yaml configs:
  - nano: 6L/6H/384d, ctx 512 (~15M)
  - small: 8L/8H/512d, ctx 1024 (~50M)
  - base: 12L/12H/768d, ctx 1024 (~124M)
- `generate()` method with temperature, top-k, top-p

### Phase 7 — Training loop (`train.py`)
Single-file raw PyTorch (no Lightning/Accelerate):
- bf16 autocast, `torch.compile`, AdamW (0.9/0.95, wd 0.1), cosine LR with 2% warmup, grad clip 1.0
- Gradient accumulation to reach ~0.5M effective tokens/step; micro-batch size auto-suggested per config for 12GB
- Memmap dataloader with random offsets, pinned memory
- wandb logging (loss, LR, tokens/sec, MFU estimate), checkpoint every 1000 steps with resume support, keep last 3 + best-val
- Windows-friendly: guard multiprocessing with `if __name__ == "__main__"`, no fork-only assumptions
- A `--benchmark` flag that runs 50 steps and reports tokens/sec so I can estimate wall-clock time per tier

### Phase 8 — Sampling & eval
- `sample.py`: load checkpoint, generate from a prompts file; include `prompts/gallery_mr.txt` with ~15 Marathi prompts (news continuation, story opening, proverb completion, formal letter)
- `eval/perplexity.py`: PPL on the held-out Wikipedia slice and the general val split, reported separately
- `eval/mahasent_fewshot.py`: few-shot sentiment classification on L3Cube MahaSent via log-likelihood scoring of label verbalizers

### Phase 9 — SFT + demo (build last, after I confirm pretraining works)
- `sft/build_instruct_data.py`: assemble Marathi instruction data (ai4bharat indic-instruct Marathi split + translated Alpaca-cleaned; leave a hook for my hand-written examples in `sft/manual_examples.jsonl`), format with the chat special tokens
- `sft/finetune.py`: full fine-tune of the 124M base model, lower LR, same infra as train.py
- `app/gradio_demo.py`: simple chat UI over the SFT model, CPU-capable

## Hard rules
- **Never** fabricate dataset IDs, URLs, or benchmark numbers — verify or ask.
- Every phase must first be proven on tiny data (`--limit 1000` docs) before I run it at scale.
- Prefer clear, boring, well-commented code over cleverness — this repo will be read by hiring teams.
- No GPU-required code paths in data scripts; only `model.py`/`train.py`/eval touch CUDA.
- Write tests only where cheap and valuable: tokenizer round-trip, memmap read/write, model forward shape + loss sanity (random data loss ≈ ln(32000)).

Start with Phase 1 now.
