# MarathiGPT

A GPT-style language model pretrained from scratch on Marathi text, nanoGPT-style
(architecture + training loop written from first principles, no fine-tuned base model).

Portfolio project: full pretraining stack — corpus collection, cleaning, a custom
tokenizer, model, training loop, and evaluation — for a low-resource Indian language,
trained on a single consumer GPU (RTX 5070, 12GB).

See [marathi-gpt-project-plan.md](marathi-gpt-project-plan.md) for the full blueprint.

## Status

Work in progress. Built phase by phase:

- [x] Phase 1 — Repo scaffold
- [x] Phase 2 — Data download
- [x] Phase 3 — Cleaning pipeline
- [x] Phase 4 — Tokenizer
- [x] Phase 5 — Pre-tokenization
- [x] Phase 6 — Model
- [x] Phase 7 — Training loop
- [ ] Phase 8 — Sampling & eval
- [ ] Phase 9 — SFT + demo

## Repo structure

```
marathi-gpt/
├── configs/            # nano / small / base model + training configs
├── data/               # download, clean, tokenizer training, pre-tokenization
├── model.py            # ~300-line decoder-only transformer
├── train.py            # single-file training loop
├── sample.py           # generation from checkpoints
├── eval/               # perplexity, few-shot sentiment, tokenizer fertility
├── sft/                # instruction-tuning data + fine-tuning
└── app/                # Gradio chat demo
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA build; verified on RTX 5070
pip install -r requirements.txt
```

## Quickstart

```bash
python data/download.py --source wikipedia --limit 1000   # drop --limit for the full source
python data/clean.py --limit 1000
python data/train_tokenizer.py --limit 1000 --vocab-size 1000  # drop both for a real 32k tokenizer
python data/prepare.py --limit 1000
python train.py --config configs/nano.yaml --benchmark     # pure throughput/MFU check, no data needed
python train.py --config configs/nano.yaml --max-steps 500 # a real (if small) training run
```

### First real training run

Ran the full pipeline end-to-end on the complete Marathi Wikipedia (94,133
articles) rather than a toy sample: `download.py --source wikipedia` (no
limit) → `clean.py` (54,895 docs survived langID/quality/dedup) →
`train_tokenizer.py --vocab-size 32000` (real 32k-vocab tokenizer, see
`assets/fertility_chart.png` — 4.8x fewer tokens than GPT-4's cl100k_base on
held-out Marathi) → `prepare.py` (33.1M train tokens after Wikipedia's 2x
upweight) → `train.py --config configs/nano.yaml --max-steps 500`.

Nano (22.9M params) trained in ~28 minutes on the RTX 5070: loss went
8.63 → 4.39 (train) / 4.59 (val) over 500 steps (262M tokens). Sample
generation at that checkpoint produces grammatical Marathi with correctly-
learned Wikipedia article structure (section headers like "इतिहास"/History,
"बाह्य दुवे"/External links) but falls into repetition loops on longer
continuations - expected for a 22.9M-param model this early in training, not
a bug.

Small (42.1M params) trained on the identical data/token budget (500 steps,
262M tokens) in ~89 minutes: loss went 8.69 → 4.25 (train) / 4.56 (val) -
modestly better than nano at the same budget, as expected for more params.
Its generations are noticeably more fluent: full, grammatically clean
sentences ("पुणे शहर हे येथील ऐतिहासिक शहर होते" - "Pune city was a historic
city here") and a correctly-reproduced Wikipedia city-article template
(इतिहास/History, भूगोल/Geography, हवामान/Climate, लोकसंख्या/Population
sections with plausible census-style demographic prose). One prompt still
degenerated into a nonsense numeral sequence - a real, honest failure mode at
this scale/budget, not cherry-picked away.

## Model tiers

| Tier | Params (measured) | Layers / Heads / d_model | Context |
|------|--------------------|---------------------------|---------|
| nano  | 22.9M (53.6% tied embedding) | 6 / 6 / 384  | 512  |
| small | 42.1M (38.9% tied embedding) | 8 / 8 / 512  | 1024 |
| base  | 109.5M (22.4% tied embedding) | 12 / 12 / 768 | 1024 |

Original targets were ~15M/~50M/~124M; actual counts run a bit higher for the
smaller tiers because the 32,000-word tokenizer's tied embedding table
(vocab_size × n_embd) is a fixed cost that dominates a narrow model
disproportionately - it shrinks from 54% of nano's params to 22% of base's
as d_model grows. Not a bug, just an artifact of picking the vocab size
before the model width.
