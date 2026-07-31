# MarathiGPT — Project Blueprint
*A GPT trained from scratch on Marathi, nanoGPT-style (Karpathy's "Let's build GPT" → "Let's reproduce GPT-2" lineage), sized for an RTX 5070 (12GB) + 64GB RAM.*

**Positioning:** Portfolio centerpiece for Indic LLM roles (Sarvam AI, AI4Bharat, Krutrim, Ola, CoRover). The story you're selling: *"I built the full pretraining stack — corpus, tokenizer, model, training loop, eval — for a low-resource Indian language, on consumer hardware."* That end-to-end narrative matters more than the model being good.

---

## 1. Scope & Model Tiers

Train in three stages so you always have something working:

| Tier | Params | Layers / Heads / d_model | Context | Purpose |
|------|--------|--------------------------|---------|---------|
| `marathi-nano` | ~15M | 6 / 6 / 384 | 512 | Debug the pipeline end-to-end in ~1 day |
| `marathi-small` | ~50M | 8 / 8 / 512 | 1024 | First "real" model, coherent Marathi |
| `marathi-base` | ~124M (GPT-2 small shape) | 12 / 12 / 768 | 1024 | Final portfolio model |

124M with bf16 + FlashAttention + gradient accumulation fits comfortably in 12GB. Don't go bigger — data, not VRAM, is your bottleneck (see §5).

## 2. Dataset

Target: **8–12B Marathi tokens raw → ~4–6B after cleaning/dedup.** That's realistically achievable and enough for a well-trained 124M model (Chinchilla-optimal is ~2.5B tokens; you'll do 1–2 epochs over more).

### Sources (in priority order)
1. **AI4Bharat Sangraha** (HuggingFace: `ai4bharat/sangraha`) — the big one. Verified + synthetic splits; Marathi verified portion alone is several billion tokens. Use `verified` split as your core.
2. **IndicCorp v2** (AI4Bharat) — large crawled Marathi corpus, complements Sangraha.
3. **L3Cube MahaCorpus** — ~25M sentences, Pune-based lab, clean news-heavy Marathi. Also gives you a talking point (L3Cube is literally in your city).
4. **Marathi Wikipedia** — small (~100M tokens) but highest quality; upweight it (2–3 epochs over it).
5. **CulturaX / mC4 (mr split)** — bulk filler if you need volume; noisiest, filter hardest.
6. Optional flavor: Marathi literature from archive.org (public domain — Sane Guruji, historical texts), news sites via Sangraha already.

### Cleaning pipeline (this is a portfolio section in itself)
```
raw shards → language ID filter (fastText lid.176, keep mr ≥ 0.65)
           → Devanagari ratio filter (≥ 70% Devanagari chars)
           → length/quality heuristics (Gopher rules: min words, symbol ratio,
             repeated-line ratio, boilerplate strip)
           → exact dedup (xxhash on normalized text)
           → fuzzy dedup (MinHash LSH, Jaccard ≥ 0.8, via `datasketch` or
             `text-dedup`)
           → PII scrub (regex: phone, email, Aadhaar-like patterns)
           → shard to .jsonl.zst
```
Log before/after stats at every stage — that table goes in your README and blog post.

## 3. Tokenizer

Train your own — this is a key differentiator vs "I fine-tuned Llama."

- **SentencePiece BPE** (or HF `tokenizers` BPE), **vocab 32,000**, byte-fallback on.
- Train on a 2–5GB stratified sample of the cleaned corpus.
- NFC-normalize Devanagari first; do **not** split conjuncts/matras mid-grapheme (byte fallback handles rare cases).
- Reserve special tokens: `<|endoftext|>`, `<|user|>`, `<|assistant|>` (for later instruction tuning).
- **Benchmark fertility** (tokens per word) against GPT-4's tokenizer and Llama-3's on Marathi text — you'll show ~3–5× better compression on Marathi. Killer chart for the blog post.

## 4. Architecture

Stay faithful to nanoGPT but with the modern upgrades everyone expects in 2026:

- Decoder-only transformer, pre-norm
- **RMSNorm** instead of LayerNorm
- **RoPE** instead of learned positional embeddings
- **SwiGLU** MLP (hidden = 8/3 · d_model, rounded to multiple of 64)
- No bias terms; weight tying (embed = unembed)
- FlashAttention via `torch.nn.functional.scaled_dot_product_attention`
- Init: GPT-2 scheme (normal 0.02, residual scaling 1/√(2·n_layer))

Keep it in **one readable `model.py` (~300 lines)** — reviewers at Sarvam/AI4Bharat will actually read it.

## 5. Training

- **Framework:** raw PyTorch, single file `train.py` (nanoGPT style). No Lightning/Accelerate — the point is showing you understand the loop.
- **Precision:** bf16 autocast + grad scaler unnecessary on bf16; `torch.compile` on.
- **Optimizer:** AdamW (β=0.9/0.95, wd=0.1), cosine schedule with warmup (~2% of steps), grad clip 1.0.
- **Batch:** effective batch ~0.5M tokens/step via gradient accumulation. On 12GB at 1024 ctx: micro-batch ~8–12 for the 124M model, accumulate to taste.
- **Throughput estimate (5070, bf16, compiled):** roughly 45–70k tokens/sec for 124M → **2.5B tokens ≈ 12–16 GPU-days.** Plan for the small model first (~2–3 days) while base trains overnight over ~2 weeks.
- **Data loading:** pre-tokenize entire corpus to uint16 memmap `.bin` shards (nanoGPT's `prepare.py` pattern). 64GB RAM makes this trivial.
- Checkpoint every ~1k steps, keep loss curves in **wandb** (public project link = credibility).

## 6. Evaluation

- Held-out perplexity (Wikipedia-mr and Sangraha-verified test slices, reported separately)
- **IndicSentiment / MahaSent** (L3Cube) few-shot classification accuracy
- Generation gallery: 15–20 curated prompts (news continuation, story, proverbs/म्हणी, letter writing) with side-by-side outputs at each checkpoint — shows learning progression
- Tokenizer fertility comparison table (§3)
- Optional: FLORES-200 mr perplexity vs mBERT-era baselines

## 7. Phase 2 — Instruction Tuning (after base pretraining)

- SFT on ~50–100k Marathi instructions: translate Alpaca-cleaned + Dolly via IndicTrans2 (AI4Bharat — another nice tie-in), plus `ai4bharat/indic-instruct` Marathi split, plus 200–500 hand-written examples (proverbs, Marathi-specific culture — small effort, big quality)
- Chat template using the reserved tokens from §3
- Full fine-tune (124M is small enough; no LoRA needed)

## 8. Deliverables & Distribution

1. **GitHub repo** `marathi-gpt` — clean structure below, heavy README with loss curves, eval tables, sample generations in Devanagari
2. **HF Hub:** model weights + tokenizer + dataset card for your cleaned mix
3. **Gradio demo** on HF Spaces (CPU inference is fine at 124M)
4. **Blog post / LinkedIn series:** "Pretraining a Marathi GPT from scratch on a single RTX 5070" — 3-part series (data, training, results). Tag Sarvam/AI4Bharat people. This is honestly worth as much as the model.

## 9. Repo Structure

```
marathi-gpt/
├── README.md
├── configs/
│   ├── nano.yaml, small.yaml, base.yaml
├── data/
│   ├── download.py        # pull Sangraha, IndicCorp, MahaCorpus, wiki
│   ├── clean.py           # filters + dedup pipeline (§2)
│   ├── train_tokenizer.py
│   └── prepare.py         # tokenize → uint16 memmap shards
├── model.py               # ~300 lines, the whole architecture
├── train.py               # single-file training loop
├── sample.py              # generation / prompt gallery
├── eval/
│   ├── perplexity.py
│   ├── mahasent_fewshot.py
│   └── fertility.py       # tokenizer comparison
├── sft/
│   ├── build_instruct_data.py
│   └── finetune.py
└── app/
    └── gradio_demo.py
```

## 10. Timeline (aggressive but doable)

| Week | Milestone |
|------|-----------|
| 1 | Download + clean corpus, train tokenizer, fertility benchmark |
| 2 | `model.py` + `train.py`, train nano tier, verify pipeline |
| 2–3 | Train small (50M), build eval harness while it runs |
| 3–5 | Train base (124M) overnight runs; write blog part 1–2 |
| 5–6 | Instruction tuning, Gradio demo, HF release, blog part 3, LinkedIn push |

## Key risks
- **Data quality > everything.** If generations are garbage, it's almost always the corpus, not the architecture. Budget real time on §2.
- **Don't scale past 124M.** A crisp 124M beats a half-trained 350M for the portfolio story.
- **Thermal/power:** multi-day runs on a desktop — cap power limit (~90%) with `nvidia-smi -pl`, negligible speed loss, much safer.
