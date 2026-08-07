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
- [x] Phase 8 — Sampling & eval
- [x] Phase 9 — SFT + demo

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
python sample.py --checkpoint checkpoints/nano/best.pt      # generate from prompts/gallery_mr.txt
python eval/perplexity.py --checkpoint checkpoints/nano/best.pt
python eval/mahasent_fewshot.py --checkpoint checkpoints/nano/best.pt --limit 300
python sft/build_instruct_data.py                           # pulls Marathi Dolly from ai4bharat/indic-align
python sft/finetune.py --checkpoint checkpoints/nano/best.pt --checkpoint-dir checkpoints/sft-nano
python app/gradio_demo.py --checkpoint checkpoints/sft-nano/best.pt
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

### Evaluation (Phase 8)

`eval/perplexity.py` walks the held-out splits exhaustively (non-overlapping
block_size chunks, not a sampled estimate) and reports perplexity separately
per split. On the small model:

| Split | Tokens | Loss | Perplexity |
|-------|--------|------|------------|
| val | 72,704 | 4.5596 | 95.55 |
| wiki_val | 388,096 | 4.6273 | 102.23 |

`eval/mahasent_fewshot.py` runs 3-shot sentiment classification on
[L3Cube MahaSent](https://github.com/l3cube-pune/MarathiNLP) (Marathi tweets,
3-class) by scoring label-word log-likelihoods - no fine-tuning, the model
just picks whichever label continuation it finds most probable. On 300 held-out
test tweets: **33.7% accuracy** (101/300) vs a 33.3% random baseline -
chance level. Expected and honest: the model has only ever seen Wikipedia
prose, never tweets or social-register text, and 42M params / 262M
pretraining tokens isn't enough to generalize zero-shot to an unseen domain
and label-verbalizer task. A real result, not swept under the rug.

### Instruction tuning & demo (Phase 9)

`sft/build_instruct_data.py` builds the SFT set from
[ai4bharat/indic-align](https://huggingface.co/datasets/ai4bharat/indic-align)'s
Marathi-translated Dolly split (IndicTrans2-translated by AI4Bharat) plus 18
hand-written examples in `sft/manual_examples.jsonl` (greetings, Marathi
proverbs, basic facts) - **14,950 train / 79 val** examples after holding out
examples longer than the model's context. (Anudesh, a second candidate
source, turned out to have an undocumented, non-parallel schema on inspection
- English-only `interactions` with no per-language columns - so it was left
out rather than risk silently training on English text; see the comment in
`build_instruct_data.py`.)

`sft/finetune.py` does a full fine-tune (42M params is small enough that
LoRA isn't necessary) of the small pretrained checkpoint, with the loss
masked to the assistant's response tokens only (not the `<|user|>` prompt).
2,796 steps (3 epochs, micro-batch 16, lr 2e-5 cosine) took **~8 minutes** on
the RTX 5070: val loss went 5.37 → **4.90**. Sample chat turns after SFT:

```
USER: नमस्कार, तू कोण आहेस?
ASSISTANT: मला मला सर्वात चांगले वाटते. मला वाटते की मला एक वाईट मत आहे...
  (grammatical Marathi, but repetitive and doesn't actually answer - typical
  small-model/small-SFT-budget failure mode)

USER: महाराष्ट्राची राजधानी कोणती आहे?
ASSISTANT: महाराष्ट्र
  (on-topic but incomplete/wrong - should be "मुंबई")

USER: एक छोटी मराठी कविता लिही.
ASSISTANT: सुरुवातीला, 'अ' हा सर्व कविता सादर करण्यासाठी, 'धाक्या' हा एक उत्तम
कथा दिग्दर्शन आणि दिग्दर्शन आहे.
  (fluent but not a coherent poem)
```

`app/gradio_demo.py` serves this checkpoint as a chat UI (smoke-tested: server
starts, serves the chat API, returns real generations end-to-end). Honest
summary: the model reliably speaks fluent, grammatical Marathi and follows
the chat format after SFT, but factual accuracy and coherence at this
scale (42M params, 262M pretraining tokens, ~15k SFT examples) are limited -
exactly what the plan's "portfolio scale, not production quality" framing
predicts, not a bug to fix.

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
