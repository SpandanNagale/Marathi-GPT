"""
Tokenizer round-trip sanity check.

Trains a throwaway tiny SentencePiece model (same settings train_tokenizer.py
uses: BPE, byte-fallback, identity normalization, our reserved special
tokens) on a few lines of inline Marathi text, then asserts that
encode -> decode reproduces the original text exactly, that the special
tokens got real (non-<unk>) ids, and that byte-fallback correctly round-trips
a character that can't appear in such a tiny training set (an emoji).

Self-contained: no dependency on data/clean/ or a pre-trained tokenizer, so
this runs in a couple of seconds from a clean checkout.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from train_tokenizer import SPECIAL_TOKENS  # noqa: E402

SAMPLE_TEXT = [
    "भारत हा माझा देश आहे. महाराष्ट्र राज्य हे भारताच्या पश्चिम भागात आहे.",
    "मराठी भाषा ही महाराष्ट्राची अधिकृत भाषा आहे.",
    "पुणे आणि मुंबई ही महाराष्ट्रातील मोठी शहरे आहेत.",
    "छत्रपती शिवाजी महाराज हे मराठी साम्राज्याचे संस्थापक होते.",
    "गणेश चतुर्थी हा महाराष्ट्रातील एक महत्त्वाचा सण आहे.",
] * 20  # repeat so SentencePiece has enough frequency signal for a tiny vocab


def main() -> None:
    import sentencepiece as spm

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus_path = tmp / "corpus.txt"
        corpus_path.write_text("\n".join(SAMPLE_TEXT), encoding="utf-8")

        model_prefix = str(tmp / "test_tokenizer")
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=model_prefix,
            vocab_size=500,  # byte_fallback reserves all 256 byte values as required pieces
            model_type="bpe",
            byte_fallback=True,
            normalization_rule_name="identity",
            user_defined_symbols=SPECIAL_TOKENS,
        )

        sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")

        # 1. plain round-trip
        text = SAMPLE_TEXT[0]
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        assert decoded == text, f"round-trip mismatch:\n  in:  {text!r}\n  out: {decoded!r}"
        print(f"PASS: round-trip ({len(ids)} tokens) matches exactly")

        # 2. special tokens got real ids, not <unk>
        for tok in SPECIAL_TOKENS:
            tid = sp.piece_to_id(tok)
            assert tid != sp.unk_id(), f"special token {tok!r} fell back to <unk>"
        print(f"PASS: all {len(SPECIAL_TOKENS)} special tokens have real ids")

        # 3. byte-fallback on a character absent from training data
        rare_text = "भारत 🙂 चाचणी"
        rare_ids = sp.encode(rare_text)
        rare_decoded = sp.decode(rare_ids)
        assert rare_decoded == rare_text, f"byte-fallback round-trip mismatch: {rare_decoded!r}"
        print("PASS: byte-fallback round-trips an out-of-vocab character (emoji)")

    print("\ntest_tokenizer: ALL PASS")


if __name__ == "__main__":
    main()
