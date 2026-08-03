"""
Compare tokenizer fertility (tokens per word) on held-out Marathi text between
our SentencePiece tokenizer, GPT-4's tokenizers (tiktoken cl100k_base and
o200k_base), and Llama-3's tokenizer. Lower fertility = better compression =
fewer tokens needed to represent the same text = cheaper training/inference.

Llama-3's tokenizer is loaded from `Xenova/llama3-tokenizer` on the HF Hub -
an ungated mirror of the tokenizer files (the official meta-llama repos
require accepting a license and an HF token; this mirror doesn't).

Outputs a markdown table and a matplotlib bar chart to --out-dir (assets/ by
default), plus prints the table to the console.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        default="data/tokenizer",
        help="Directory containing our trained marathi_bpe.model.",
    )
    parser.add_argument(
        "--eval-text",
        default=None,
        help="Held-out Marathi text file, one doc/line per line. "
        "Defaults to <tokenizer-dir>/holdout.txt (produced by train_tokenizer.py).",
    )
    parser.add_argument("--out-dir", default="assets")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only use this many lines of held-out text (fast test mode)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing table/chart files.")
    return parser.parse_args()


def load_eval_lines(path: Path, limit) -> list:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def fertility_for_encoder(lines: list, encode_fn) -> dict:
    total_tokens = 0
    total_words = 0
    for line in lines:
        total_tokens += len(encode_fn(line))
        total_words += len(line.split())
    return {
        "tokens": total_tokens,
        "words": total_words,
        "fertility": total_tokens / max(1, total_words),
    }


def build_tokenizers(tokenizer_dir: Path) -> dict:
    tokenizers = {}

    import sentencepiece as spm

    our_model_path = tokenizer_dir / "marathi_bpe.model"
    if not our_model_path.exists():
        raise SystemExit(f"{our_model_path} not found - run data/train_tokenizer.py first.")
    sp = spm.SentencePieceProcessor(model_file=str(our_model_path))
    tokenizers["MarathiGPT (ours)"] = {
        "encode": sp.encode,
        "vocab_size": sp.vocab_size(),
    }

    import tiktoken

    for name in ["cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        tokenizers[f"GPT-4 ({name})"] = {
            "encode": enc.encode,
            "vocab_size": enc.n_vocab,
        }

    from transformers import AutoTokenizer

    llama_tok = AutoTokenizer.from_pretrained("Xenova/llama3-tokenizer")
    tokenizers["Llama-3"] = {
        "encode": lambda text: llama_tok.encode(text, add_special_tokens=False),
        "vocab_size": llama_tok.vocab_size,
    }

    return tokenizers


def render_markdown_table(results: dict) -> str:
    lines = [
        "| Tokenizer | Vocab size | Tokens | Words | Fertility (tokens/word) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['vocab_size']:,} | {r['tokens']:,} | {r['words']:,} | {r['fertility']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def render_chart(results: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = [results[n]["fertility"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color="#4C72B0")
    ax.set_ylabel("Tokens per word (lower is better)")
    ax.set_title("Tokenizer fertility on held-out Marathi text")
    ax.bar_label(bars, fmt="%.2f")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tokenizer_dir = Path(args.tokenizer_dir)
    eval_text_path = Path(args.eval_text) if args.eval_text else tokenizer_dir / "holdout.txt"
    if not eval_text_path.exists():
        raise SystemExit(f"{eval_text_path} not found - pass --eval-text or run data/train_tokenizer.py first.")

    lines = load_eval_lines(eval_text_path, args.limit)
    if not lines:
        raise SystemExit(f"No usable lines found in {eval_text_path}")
    print(f"Evaluating fertility on {len(lines)} held-out lines from {eval_text_path}")

    tokenizers = build_tokenizers(tokenizer_dir)

    results = {}
    for name, t in tokenizers.items():
        print(f"[{name}] encoding...")
        results[name] = {"vocab_size": t["vocab_size"], **fertility_for_encoder(lines, t["encode"])}

    table = render_markdown_table(results)
    print("\n" + table)

    if args.dry_run:
        print("[fertility] --dry-run: not writing table/chart files")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "fertility_table.md"
    chart_path = out_dir / "fertility_chart.png"
    table_path.write_text(table, encoding="utf-8")
    render_chart(results, chart_path)
    print(f"Wrote {table_path} and {chart_path}")


if __name__ == "__main__":
    main()
