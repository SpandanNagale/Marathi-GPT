"""
Simple Gradio chat UI over a checkpoint (base or SFT), CPU-capable at these
model sizes. Each turn is generated independently from a fresh
`<|user|>...<|assistant|>` prompt (no multi-turn context window) - matches
how sft/build_instruct_data.py's training examples are single-turn.
"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `from model import ...`
from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (SFT'd, for best chat behavior).")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {checkpoint_path} (step={ckpt.get('step', '?')})")
    return model


def main() -> None:
    args = parse_args()
    import gradio as gr

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=str(Path(args.tokenizer_dir) / "marathi_bpe.model"))
    eos_text = "<|endoftext|>"

    @torch.no_grad()
    def respond(message: str, history) -> str:
        prompt_ids = sp.encode(f"<|user|>\n{message}\n<|assistant|>\n", out_type=int)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        new_ids = out[0, len(prompt_ids) :].tolist()
        text = sp.decode(new_ids)
        return text.split(eos_text)[0].strip()

    demo = gr.ChatInterface(
        respond,
        title="MarathiGPT",
        description=(
            "A GPT-style language model pretrained from scratch on Marathi text "
            "(see github.com/SpandanNagale/Marathi-GPT). Small model, small training "
            "budget so far - expect a portfolio-scale demo, not production quality."
        ),
        examples=["महाराष्ट्राची राजधानी कोणती आहे?", "एक छोटी मराठी कविता लिही.", "नमस्कार, तू कोण आहेस?"],
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
