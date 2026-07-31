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
- [ ] Phase 4 — Tokenizer
- [ ] Phase 5 — Pre-tokenization
- [ ] Phase 6 — Model
- [ ] Phase 7 — Training loop
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
pip install -r requirements.txt
```

## Quickstart (once later phases land)

```bash
python data/download.py --source wikipedia --limit 1000
python data/clean.py --limit 1000
python data/train_tokenizer.py --limit 1000
python data/prepare.py --limit 1000
python train.py --config configs/nano.yaml --benchmark
```

## Model tiers

| Tier | Params | Layers / Heads / d_model | Context |
|------|--------|---------------------------|---------|
| nano  | ~15M  | 6 / 6 / 384  | 512  |
| small | ~50M  | 8 / 8 / 512  | 1024 |
| base  | ~124M | 12 / 12 / 768 | 1024 |
