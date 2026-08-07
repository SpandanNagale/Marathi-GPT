"""
Few-shot sentiment classification on L3Cube MahaSent, scored by log-likelihood
of label verbalizers rather than fine-tuning: the model is given a few
labeled exemplars, then for each test tweet we score how likely each label
word is as a continuation and pick the highest.

Data: github.com/l3cube-pune/MarathiNLP (L3CubeMahaSent Dataset), a 3-class
Marathi tweet sentiment corpus - Positive(1) / Negative(-1) / Neutral(0).
Not on the HF Hub (verified: only distributed via that GitHub repo), so this
downloads the raw CSVs directly and caches them locally.
"""

import argparse
import csv
import random
import sys
import urllib.request
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from model import ...`
from model import GPT, GPTConfig

DATA_BASE_URL = "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3CubeMahaSent%20Dataset/"
LABEL_NAMES = {1: "सकारात्मक", -1: "नकारात्मक", 0: "तटस्थ"}  # positive / negative / neutral


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--tokenizer", default="data/tokenizer/marathi_bpe.model")
    parser.add_argument("--cache-dir", default="data/eval/mahasent", help="Where to cache the downloaded CSVs.")
    parser.add_argument("--n-shots", type=int, default=3, help="Exemplars per class in the few-shot prompt.")
    parser.add_argument("--limit", type=int, default=300, help="Number of test tweets to evaluate.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true", help="Alias for --limit 10.")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def download_csv(cache_dir: Path, split: str) -> list[tuple[str, int]]:
    path = cache_dir / f"tweets-{split}.csv"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = DATA_BASE_URL + f"tweets-{split}.csv"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tweet = row["tweet"].strip()
            if tweet:
                rows.append((tweet, int(row["label"])))
    return rows


@torch.no_grad()
def score_completion(model: GPT, sp, prompt: str, completion: str, device: str) -> float:
    """Sum of log P(completion token | prompt + preceding completion tokens).

    Assumes tokenizing `prompt` gives a token-level prefix of tokenizing
    `prompt + completion` - true here since completions start with a space
    before a whole word, so BPE doesn't re-merge across the boundary.
    """
    prompt_ids = sp.encode(prompt, out_type=int)
    full_ids = sp.encode(prompt + completion, out_type=int)
    idx = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(idx[:, :-1])
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = idx[:, 1:]
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    start = len(prompt_ids) - 1  # completion tokens begin at this position in token_logp
    return token_logp[start:].sum().item()


def build_prefix(train_rows: list[tuple[str, int]], n_shots: int, rng: random.Random) -> str:
    by_label = {label: [t for t, l in train_rows if l == label] for label in LABEL_NAMES}
    exemplars = []
    for label, tweets in by_label.items():
        exemplars += [(t, label) for t in rng.sample(tweets, min(n_shots, len(tweets)))]
    rng.shuffle(exemplars)
    return "\n\n".join(f"वाक्य: {t}\nभावना: {LABEL_NAMES[l]}" for t, l in exemplars)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.limit = min(args.limit, 10)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    cache_dir = Path(args.cache_dir)
    train_rows = download_csv(cache_dir, "train")
    test_rows = download_csv(cache_dir, "test")
    print(f"loaded {len(train_rows)} train / {len(test_rows)} test tweets from L3Cube MahaSent")

    prefix = build_prefix(train_rows, args.n_shots, rng)

    rng.shuffle(test_rows)
    eval_rows = test_rows[: args.limit]

    correct = 0
    for tweet, gold in eval_rows:
        prompt = f"{prefix}\n\nवाक्य: {tweet}\nभावना:"
        scores = {label: score_completion(model, sp, prompt, " " + name, device) for label, name in LABEL_NAMES.items()}
        pred = max(scores, key=scores.get)
        correct += int(pred == gold)

    acc = correct / len(eval_rows)
    print(
        f"\nMahaSent {args.n_shots}-shot accuracy: {acc:.1%} ({correct}/{len(eval_rows)}) "
        f"[random baseline: {1 / len(LABEL_NAMES):.1%}]"
    )


if __name__ == "__main__":
    main()
