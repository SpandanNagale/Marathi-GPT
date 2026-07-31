"""
Stream-download Marathi corpora into data/raw/<source>/ as zstd-compressed
jsonl shards (~500MB each, one subdirectory per source). Safe to re-run:
each source resumes from where it left off.

Sources
-------
sangraha    ai4bharat/sangraha, `verified/mar` split (HF streaming, parquet).
indiccorp   ai4bharat/IndicCorpV2, `mar_Deva` split (HF streaming, plain text).
wikipedia   wikimedia/wikipedia, `20231101.mr` config (HF streaming, parquet) -
            the only Marathi dump config currently published for this dataset.
mahacorpus  L3Cube-MahaCorpus. NOT on the HF Hub - distributed as a Google
            Drive zip (see github.com/l3cube-pune/MarathiNLP). This source
            does not publish a formal per-document schema (it's a sentence
            corpus), so documents here are recovered heuristically from
            blank-line-delimited paragraphs, falling back to fixed-size
            chunks of consecutive lines.

Token counts printed while downloading are a cheap whitespace-split proxy,
not final BPE token counts - our tokenizer doesn't exist until Phase 4.
"""

import os
import sys

if sys.platform == "win32" and not sys.flags.utf8_mode:
    # HF `datasets`' streaming text reader ignores the `encoding="utf-8"` kwarg
    # on Windows and falls back to the system codepage (cp1252), which crashes
    # on Devanagari text. Re-exec once under Python's UTF-8 mode to fix this
    # at the interpreter level, before any HF imports happen. subprocess (not
    # os.execv) because execv's argument quoting breaks on Windows paths that
    # contain spaces.
    import subprocess

    sys.exit(subprocess.run([sys.executable, "-X", "utf8", *sys.argv]).returncode)

import argparse
import time
import zipfile
from pathlib import Path

from shard_io import ShardWriter, estimate_tokens

SOURCES = ("sangraha", "indiccorp", "mahacorpus", "wikipedia")

# L3Cube-MahaCorpus "All sources" combined zip (752M tokens / 57.2M sentences),
# per github.com/l3cube-pune/MarathiNLP. Not resolvable as a normal HF dataset id.
MAHACORPUS_GDRIVE_ID = "1UjZ-X2S77AQyCkHqw2mFXRWYf9WOZS0m"


def _progress(source: str, n_docs: int, n_tokens: int, start: float) -> None:
    elapsed = max(time.time() - start, 1e-6)
    print(
        f"[{source}] {n_docs:>10,} docs | ~{n_tokens / 1e6:>8.1f}M words (proxy tokens) "
        f"| {n_docs / elapsed:>7.1f} docs/sec",
        end="\r",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Per-source downloaders — each returns {"docs": int, "tokens_est": int}
# ---------------------------------------------------------------------------


def download_sangraha(out_dir: Path, limit, dry_run: bool, shard_mb: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset(
        "ai4bharat/sangraha", data_dir="verified/mar", split="train", streaming=True
    )
    writer = ShardWriter(out_dir / "sangraha", shard_mb, dry_run)
    skip = writer.skip_to_resume_point()
    if skip:
        print(f"[sangraha] resuming: skipping {skip:,} already-downloaded docs")

    n_docs, n_tokens = skip, 0
    start = time.time()
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        if limit and n_docs - skip >= limit:
            break
        text = ex["text"]
        writer.write({"id": ex.get("doc_id", str(i)), "source": "sangraha", "text": text})
        n_docs += 1
        n_tokens += estimate_tokens(text)
        if n_docs % 500 == 0:
            _progress("sangraha", n_docs, n_tokens, start)
    writer.close()
    print()
    return {"docs": n_docs, "tokens_est": n_tokens}


def download_indiccorp(out_dir: Path, limit, dry_run: bool, shard_mb: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset(
        "ai4bharat/IndicCorpV2", "indiccorp_v2", split="mar_Deva", streaming=True
    )
    writer = ShardWriter(out_dir / "indiccorp", shard_mb, dry_run)
    skip = writer.skip_to_resume_point()
    if skip:
        print(f"[indiccorp] resuming: skipping {skip:,} already-downloaded docs")

    n_docs, n_tokens = skip, 0
    start = time.time()
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        if limit and n_docs - skip >= limit:
            break
        text = ex["text"]
        writer.write({"id": str(i), "source": "indiccorp", "text": text})
        n_docs += 1
        n_tokens += estimate_tokens(text)
        if n_docs % 500 == 0:
            _progress("indiccorp", n_docs, n_tokens, start)
    writer.close()
    print()
    return {"docs": n_docs, "tokens_est": n_tokens}


def download_wikipedia(out_dir: Path, limit, dry_run: bool, shard_mb: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.mr", split="train", streaming=True)
    writer = ShardWriter(out_dir / "wikipedia", shard_mb, dry_run)
    skip = writer.skip_to_resume_point()
    if skip:
        print(f"[wikipedia] resuming: skipping {skip:,} already-downloaded docs")

    n_docs, n_tokens = skip, 0
    start = time.time()
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        if limit and n_docs - skip >= limit:
            break
        text = ex["text"]
        writer.write(
            {
                "id": ex.get("id", str(i)),
                "source": "wikipedia",
                "title": ex.get("title", ""),
                "text": text,
            }
        )
        n_docs += 1
        n_tokens += estimate_tokens(text)
        if n_docs % 500 == 0:
            _progress("wikipedia", n_docs, n_tokens, start)
    writer.close()
    print()
    return {"docs": n_docs, "tokens_est": n_tokens}


def download_mahacorpus(
    out_dir: Path, limit, dry_run: bool, shard_mb: int, lines_per_doc: int = 50
) -> dict:
    try:
        import gdown
    except ImportError:
        raise SystemExit(
            "MahaCorpus is distributed via Google Drive. Run `pip install gdown` first."
        )

    cache_dir = out_dir / "mahacorpus" / "_download_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "mahacorpus.zip"
    extract_dir = cache_dir / "extracted"

    if zip_path.exists():
        print(f"[mahacorpus] found cached zip at {zip_path}, skipping download")
    else:
        print(f"[mahacorpus] downloading zip from Google Drive (id={MAHACORPUS_GDRIVE_ID})...")
        if not dry_run:
            gdown.download(id=MAHACORPUS_GDRIVE_ID, output=str(zip_path), quiet=False)

    if not dry_run and not extract_dir.exists():
        print(f"[mahacorpus] extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    txt_files = sorted(extract_dir.rglob("*.txt")) if extract_dir.exists() else []
    if not txt_files and not dry_run:
        raise SystemExit(f"No .txt files found under {extract_dir} — inspect the archive manually.")

    writer = ShardWriter(out_dir / "mahacorpus", shard_mb, dry_run)
    skip = writer.skip_to_resume_point()
    if skip:
        print(f"[mahacorpus] resuming: skipping {skip:,} already-downloaded docs")

    n_docs, n_tokens, seen = skip, 0, 0
    start = time.time()
    for fpath in txt_files:
        raw = fpath.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            paragraphs = [
                "\n".join(lines[i : i + lines_per_doc])
                for i in range(0, len(lines), lines_per_doc)
            ]
        for para in paragraphs:
            if seen < skip:
                seen += 1
                continue
            if limit and n_docs - skip >= limit:
                break
            writer.write({"id": f"{fpath.stem}-{seen}", "source": "mahacorpus", "text": para})
            n_docs += 1
            n_tokens += estimate_tokens(para)
            seen += 1
            if n_docs % 500 == 0:
                _progress("mahacorpus", n_docs, n_tokens, start)
        if limit and n_docs - skip >= limit:
            break
    writer.close()
    print()
    return {"docs": n_docs, "tokens_est": n_tokens}


DOWNLOADERS = {
    "sangraha": download_sangraha,
    "indiccorp": download_indiccorp,
    "mahacorpus": download_mahacorpus,
    "wikipedia": download_wikipedia,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        choices=SOURCES + ("all",),
        default="all",
        help="Which single source to download (default: all).",
    )
    parser.add_argument(
        "--out-dir", default="data/raw", help="Output directory for downloaded shards."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only download this many documents per source (fast test mode).",
    )
    parser.add_argument(
        "--shard-mb", type=int, default=500, help="Target size of each output shard, in MB."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the download loop without writing any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sources = SOURCES if args.source == "all" else (args.source,)

    summary = {}
    for source in sources:
        print(f"=== {source} ===")
        try:
            stats = DOWNLOADERS[source](out_dir, args.limit, args.dry_run, args.shard_mb)
        except Exception as e:
            print(f"[{source}] FAILED: {e}", file=sys.stderr)
            continue
        summary[source] = stats

    print("\n=== Summary ===")
    print(f"{'source':<12} {'docs':>12} {'~tokens (M)':>14}")
    for source, stats in summary.items():
        print(f"{source:<12} {stats['docs']:>12,} {stats['tokens_est'] / 1e6:>14.1f}")


if __name__ == "__main__":
    main()
