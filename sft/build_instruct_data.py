"""
Assemble Marathi instruction data for SFT, formatted as plain-text JSONL
{"instruction": ..., "response": ...} pairs (finetune.py tokenizes and applies
the chat template at load time).

Source: ai4bharat/indic-align (github/HF: verified real dataset, IndicTrans2-
translated into 14 Indic languages incl. Marathi - see
https://huggingface.co/datasets/ai4bharat/indic-align). We use the
`indicalign-instruct` subsets, each a parquet file with one column per
language variant; we read the `mar_Deva` column, which holds
[[turn1_instruction, turn1_response], [turn2_instruction, turn2_response], ...]
per row. Multi-turn rows are flattened into independent single-turn examples
(each turn becomes its own training example, discarding prior-turn context) -
a deliberate simplification to keep the format single-turn and boring.

Plus a hook for hand-written examples in sft/manual_examples.jsonl (same
{"instruction", "response"} format), folded into the train split.
"""

import argparse
import json
import sys
from pathlib import Path

import xxhash
from datasets import load_dataset

SOURCES = {
    # name -> parquet URL under ai4bharat/indic-align's indicalign-instruct/.
    # Only "dolly" is confirmed to have the expected per-language-column
    # schema (verified by hand: each row has a `mar_Deva` column holding
    # [[instruction, response], ...]). "anudesh" was checked and has a
    # different, undocumented structure (English-only `interactions` field,
    # no per-language columns) - left here as a placeholder, NOT in the
    # default --sources, since using it would silently mix in English text.
    # wikihow/oasst are unverified; add them only after checking their schema
    # the same way (see the schema-mismatch SystemExit below).
    "dolly": "https://huggingface.co/datasets/ai4bharat/indic-align/resolve/main/indicalign-instruct/dolly/Dolly.parquet",
    "anudesh": "https://huggingface.co/datasets/ai4bharat/indic-align/resolve/main/indicalign-instruct/anudesh/anudesh1.parquet",
    "wikihow": "https://huggingface.co/datasets/ai4bharat/indic-align/resolve/main/indicalign-instruct/wikihow/wiki_how.parquet",
    "oasst": "https://huggingface.co/datasets/ai4bharat/indic-align/resolve/main/indicalign-instruct/oasst/oasst.parquet",
}
MARATHI_COLUMN = "mar_Deva"
VAL_PERMILLE = 5  # 0.5% held out for val, same hashing pattern as data/train_tokenizer.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument(
        "--sources", default=None, help=f"Comma-separated subset of {sorted(SOURCES)} to include. Default: dolly."
    )
    parser.add_argument(
        "--manual-examples",
        default="sft/manual_examples.jsonl",
        help="Hand-written examples to fold into the train split.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap examples read per source, for a fast test.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true", help="Alias for --limit 200 --sources dolly, no writes.")
    return parser.parse_args()


def iter_marathi_turns(source_name: str, url: str, limit: int | None):
    ds = load_dataset("parquet", data_files=url, streaming=True, split="train")
    n_seen, n_yielded = 0, 0
    for row in ds:
        n_seen += 1
        if n_seen == 1 and MARATHI_COLUMN not in row:
            raise SystemExit(
                f"[{source_name}] expected column '{MARATHI_COLUMN}' not found; got {list(row.keys())}. "
                "The ai4bharat/indic-align schema may have changed - check the dataset page before re-running."
            )
        for turn in row.get(MARATHI_COLUMN) or []:
            if len(turn) != 2:
                continue
            instruction, response = turn
            instruction, response = instruction.strip(), response.strip()
            if instruction and response:
                yield instruction, response
                n_yielded += 1
        if limit is not None and n_yielded >= limit:
            break
    print(f"  [{source_name}] {n_yielded} instruction/response pairs from {n_seen} rows")


def load_manual_examples(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append((row["instruction"].strip(), row["response"].strip()))
    return examples


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.limit = args.limit or 200
    if args.sources is None:
        args.sources = "dolly"

    source_names = [s.strip() for s in args.sources.split(",") if s.strip()]
    for name in source_names:
        if name not in SOURCES:
            raise SystemExit(f"unknown source '{name}', choose from {sorted(SOURCES)}")

    print(f"=== collecting Marathi instructions from: {', '.join(source_names)} ===")
    examples: list[tuple[str, str, str]] = []  # (instruction, response, source)
    for name in source_names:
        for instruction, response in iter_marathi_turns(name, SOURCES[name], args.limit):
            examples.append((instruction, response, name))

    manual_path = Path(args.manual_examples)
    manual = load_manual_examples(manual_path)
    print(f"  [manual] {len(manual)} hand-written examples from {manual_path}")
    examples += [(i, r, "manual") for i, r in manual]

    train, val = [], []
    for instruction, response, source in examples:
        h = xxhash.xxh64(f"{source}:{instruction}".encode("utf-8")).intdigest()
        (val if source != "manual" and (h % 1000) < VAL_PERMILLE else train).append(
            {"instruction": instruction, "response": response, "source": source}
        )

    print(f"\ntotal: {len(examples)} examples -> train={len(train)} val={len(val)}")
    for name in source_names + ["manual"]:
        n = sum(1 for e in train + val if e["source"] == name)
        print(f"  {name:<10} {n}")

    if args.dry_run:
        print("\n[dry-run] not writing output files")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in [("train", train), ("val", val)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
