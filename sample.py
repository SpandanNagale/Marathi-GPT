"""
Load a trained checkpoint and generate text from prompts.

Three ways to supply prompts: --prompt "..." for one-off generation,
--interactive for a REPL, or (default) --prompts-file for a batch run over
prompts/gallery_mr.txt.
"""

import sys

if sys.platform == "win32" and not sys.flags.utf8_mode:
    # stdout defaults to the system codepage (cp1252) on Windows, which can't
    # encode Devanagari - re-exec once under Python's UTF-8 mode to fix this
    # at the interpreter level. See data/download.py for the same pattern.
    import subprocess

    sys.exit(subprocess.run([sys.executable, "-X", "utf8", *sys.argv]).returncode)

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint, e.g. checkpoints/small/best.pt.")
    parser.add_argument("--tokenizer", default="data/tokenizer/marathi_bpe.model")
    parser.add_argument(
        "--prompts-file",
        default="prompts/gallery_mr.txt",
        help="File with one prompt per line. Used when --prompt/--interactive aren't given.",
    )
    parser.add_argument("--prompt", default=None, help="A single prompt (overrides --prompts-file).")
    parser.add_argument("--interactive", action="store_true", help="Read prompts from stdin in a loop.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples generated per prompt.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--dry-run", action="store_true", help="Load the checkpoint and generate one short sample, then exit."
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"loaded {checkpoint_path} (step={ckpt.get('step', '?')}, "
        f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f})"
    )
    return model


@torch.no_grad()
def generate_one(model, sp, prompt, device, max_new_tokens, temperature, top_k, top_p) -> str:
    ids = sp.encode(prompt, out_type=int)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return sp.decode(out[0].tolist())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    if args.dry_run:
        prompt = "नमस्कार"
        text = generate_one(model, sp, prompt, device, 20, args.temperature, args.top_k, args.top_p)
        print(f"[dry-run] PROMPT: {prompt}\nOUTPUT: {text}")
        return

    if args.interactive:
        print("Interactive mode - type a Marathi prompt, empty line to quit.")
        while True:
            try:
                prompt = input("\n> ").strip()
            except EOFError:
                break
            if not prompt:
                break
            for i in range(args.num_samples):
                text = generate_one(
                    model, sp, prompt, device, args.max_new_tokens, args.temperature, args.top_k, args.top_p
                )
                tag = f"[{i + 1}] " if args.num_samples > 1 else ""
                print(f"{tag}{text}")
        return

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            raise SystemExit(f"{prompts_path} not found. Pass --prompt \"...\" or --interactive instead.")
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for prompt in prompts:
        print(f"\nPROMPT: {prompt}")
        for i in range(args.num_samples):
            text = generate_one(
                model, sp, prompt, device, args.max_new_tokens, args.temperature, args.top_k, args.top_p
            )
            tag = f"[{i + 1}] " if args.num_samples > 1 else ""
            print(f"{tag}OUTPUT: {text}")


if __name__ == "__main__":
    main()
