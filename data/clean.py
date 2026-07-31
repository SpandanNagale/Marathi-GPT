"""
Clean raw jsonl shards (data/raw/<source>/) into data/clean/<source>/.

Pipeline, applied in order:
  1. fastText langID (lid.176.bin) - keep docs with mr confidence >= --lid-threshold
  2. Devanagari character ratio >= --devanagari-ratio
  3. Gopher-style quality heuristics: boilerplate line stripping, word-count
     bounds, symbol-to-word ratio, repeated-line fraction
  4. Exact dedup: xxhash64 of NFC-normalized text
  5. Fuzzy dedup: MinHash LSH (datasketch), Jaccard >= --jaccard-threshold
  6. PII scrub: Indian phone numbers, emails, Aadhaar-like 12-digit numbers
     (redacted in place with [PHONE]/[EMAIL]/[AADHAAR], doc is not dropped)
  7. stats.json + a printed survivor table, per source and per stage

Stages 1-3 are per-document and run in a multiprocessing Pool, one worker per
input shard file, and are resumable at shard granularity (a shard already
present in the stage-1 output is not reprocessed). Stages 4-6 need a global
view to catch duplicates across shards *and across sources* (e.g. the same
article scraped by two sources), so they run afterward in a single sequential
pass over all stage-1 survivors - this pass is NOT incrementally resumable
(the dedup hash set/LSH index only lives in memory for one run), so every
invocation rebuilds each requested source's final output from scratch off of
stage-1's (already-resumable) output.

Fuzzy dedup keeps its MinHash/LSH index in memory, bounded by the number of
*surviving* documents (not the full corpus) - see --fuzzy-dedup-max-docs for
the safety cap once that index gets too large to keep growing unboundedly.
"""

import argparse
import json
import multiprocessing as mp
import re
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import xxhash
import zstandard as zstd
from datasketch import MinHash, MinHashLSH

from shard_io import ShardWriter, iter_shards, estimate_tokens

LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

DEVANAGARI_RE = re.compile("[ऀ-ॿ]")
NON_WHITESPACE_RE = re.compile(r"\S")
SYMBOL_RE = re.compile(r"[#*•~|>%$^&=+_<>\[\]{}\\/]")

# Longer patterns first so a 12-digit Aadhaar-like run isn't partially eaten
# by the 10-digit phone regex before it gets a chance to match.
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\-\s]?\d{4}[\-\s]?\d{4}(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(?:(?:\+91|91|0)[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Common junk lines found on scraped news/web pages (nav, cookie banners,
# calls-to-action). Not exhaustive - a heuristic denylist, not a classifier.
BOILERPLATE_SUBSTRINGS = [
    "click here", "read more", "subscribe", "sign up", "log in", "loading...",
    "all rights reserved", "terms of service", "privacy policy", "cookie policy",
    "advertisement", "please enable javascript", "follow us on", "©",
    "अधिक वाचा", "येथे क्लिक करा", "सर्व हक्क राखीव", "सदस्यता घ्या",
]


def ensure_lid_model(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[clean] downloading fastText langID model to {path} ...")

    def _report(block_num, block_size, total_size):
        done = block_num * block_size
        pct = min(100, done * 100 // total_size) if total_size > 0 else 0
        print(f"\r[clean]   {pct}%", end="", flush=True)

    urllib.request.urlretrieve(LID_MODEL_URL, str(path), reporthook=_report)
    print()
    return path


# ---------------------------------------------------------------------------
# Per-document filters (stages 1-3)
# ---------------------------------------------------------------------------


def devanagari_ratio(text: str) -> float:
    non_ws = NON_WHITESPACE_RE.findall(text)
    if not non_ws:
        return 0.0
    deva = DEVANAGARI_RE.findall(text)
    return len(deva) / len(non_ws)


def strip_boilerplate_lines(text: str, max_frac: float):
    lines = text.splitlines()
    if not lines:
        return text, False
    kept, removed = [], 0
    for line in lines:
        low = line.strip().lower()
        if low and any(p in low for p in BOILERPLATE_SUBSTRINGS):
            removed += 1
        else:
            kept.append(line)
    frac_removed = removed / len(lines)
    if frac_removed > max_frac:
        return "", True  # rejected: mostly boilerplate
    return "\n".join(kept), False


def gopher_quality(text: str, params: dict):
    """Returns (passed, filtered_text)."""
    filtered, rejected = strip_boilerplate_lines(text, params["boilerplate_max_frac"])
    if rejected:
        return False, filtered

    words = filtered.split()
    if len(words) < params["min_words"] or len(words) > params["max_words"]:
        return False, filtered

    symbols = len(SYMBOL_RE.findall(filtered))
    if symbols / max(1, len(words)) > params["symbol_ratio_max"]:
        return False, filtered

    lines = [l.strip() for l in filtered.splitlines() if l.strip()]
    if len(lines) >= 2:
        counts = Counter(lines)
        repeated = sum(c for c in counts.values() if c > 1)
        if repeated / len(lines) > params["repeated_line_max"]:
            return False, filtered

    return True, filtered


def scrub_pii(text: str):
    counts = {"phone": 0, "email": 0, "aadhaar": 0}

    def _sub(pattern, token, key, s):
        def repl(m):
            counts[key] += 1
            return token

        return pattern.sub(repl, s)

    text = _sub(AADHAAR_RE, "[AADHAAR]", "aadhaar", text)
    text = _sub(PHONE_RE, "[PHONE]", "phone", text)
    text = _sub(EMAIL_RE, "[EMAIL]", "email", text)
    return text, counts


# ---------------------------------------------------------------------------
# Stage 1-3 worker (multiprocessing, one call per input shard file)
# ---------------------------------------------------------------------------

_WORKER_MODEL = None


def _init_worker(lid_model_path: str) -> None:
    global _WORKER_MODEL
    import fasttext

    fasttext.FastText.eprint = lambda *a, **kw: None  # silence load_model's warning banner
    _WORKER_MODEL = fasttext.load_model(lid_model_path)


def _passes_langid(text: str, threshold: float) -> bool:
    single_line = text.replace("\n", " ").strip()
    if not single_line:
        return False
    labels, probs = _WORKER_MODEL.predict(single_line, k=1)
    return labels[0] == "__label__mr" and probs[0] >= threshold


def _process_shard_stage123(task) -> dict:
    in_path, out_path, limit, params, dry_run = task
    counts = Counter()
    tokens = Counter()

    fh = None
    compressor = None
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(out_path, "wb")
        compressor = zstd.ZstdCompressor(level=9).stream_writer(fh)
    try:
        for i, rec in enumerate(iter_shards(in_path)):
            if limit is not None and i >= limit:
                break
            counts["input"] += 1
            text = rec.get("text", "")
            tokens["input"] += estimate_tokens(text)

            if not _passes_langid(text, params["lid_threshold"]):
                continue
            counts["after_langid"] += 1

            if devanagari_ratio(text) < params["devanagari_ratio"]:
                continue
            counts["after_devanagari"] += 1

            passed, filtered_text = gopher_quality(text, params)
            if not passed:
                continue
            counts["after_quality"] += 1
            tokens["after_quality"] += estimate_tokens(filtered_text)

            if dry_run:
                continue
            rec = dict(rec)
            rec["text"] = filtered_text
            line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
            compressor.write(line)
    finally:
        if not dry_run:
            compressor.flush(zstd.FLUSH_FRAME)
            fh.close()

    return {"counts": dict(counts), "tokens": dict(tokens)}


# ---------------------------------------------------------------------------
# Stage 4-6: sequential global dedup + PII scrub
# ---------------------------------------------------------------------------


def dedup_and_scrub(stage1_dir: Path, out_dir: Path, sources, args) -> dict:
    """Exact dedup + fuzzy dedup + PII scrub, in one sequential pass.

    Not incrementally resumable: the exact-hash set and MinHash/LSH index
    only exist in memory for the duration of this call, so a partial resume
    would silently fail to catch duplicates against documents written by a
    previous, already-exited run. To keep dedup correct, every invocation
    does a full rebuild from stage-1 survivors for the requested sources
    (stage 1-3 is the expensive part and stays resumable at shard granularity;
    this stage is comparatively cheap to redo in full).
    """
    for source in sources:
        source_out_dir = out_dir / source
        if source_out_dir.exists():
            for f in source_out_dir.glob("shard_*.jsonl.zst"):
                f.unlink()
            state_path = source_out_dir / "_state.json"
            if state_path.exists():
                state_path.unlink()

    exact_hashes = set()
    lsh = MinHashLSH(threshold=args.jaccard_threshold, num_perm=args.num_perm)
    fuzzy_dedup_active = True
    next_minhash_id = 0

    writers = {source: ShardWriter(out_dir / source, args.shard_mb) for source in sources}

    per_source_counts = {source: Counter() for source in sources}
    totals = Counter()
    pii_counts = Counter()
    start = time.time()

    for source in sources:
        counts = per_source_counts[source]
        for rec in iter_shards(stage1_dir / source):
            counts["stage1_survivors"] += 1
            totals["stage1_survivors"] += 1
            text = rec["text"]

            normalized = unicodedata.normalize("NFC", text)
            h = xxhash.xxh64(normalized.encode("utf-8")).intdigest()
            if h in exact_hashes:
                counts["dropped_exact_dup"] += 1
                continue
            exact_hashes.add(h)
            counts["after_exact_dedup"] += 1

            if not args.skip_fuzzy_dedup and fuzzy_dedup_active:
                words = normalized.split()
                shingles = {
                    " ".join(words[i : i + args.ngram_size])
                    for i in range(max(1, len(words) - args.ngram_size + 1))
                }
                mh = MinHash(num_perm=args.num_perm)
                for s in shingles:
                    mh.update(s.encode("utf-8"))

                if lsh.query(mh):
                    counts["dropped_fuzzy_dup"] += 1
                    continue

                lsh.insert(f"doc-{next_minhash_id}", mh)
                next_minhash_id += 1
                if next_minhash_id >= args.fuzzy_dedup_max_docs:
                    fuzzy_dedup_active = False
                    print(
                        f"\n[clean] fuzzy-dedup index hit --fuzzy-dedup-max-docs="
                        f"{args.fuzzy_dedup_max_docs}, disabling it for the remainder "
                        f"of this run (docs after this point are exact-deduped only)"
                    )
            counts["after_fuzzy_dedup"] += 1

            scrubbed, doc_pii = scrub_pii(text)
            for k, v in doc_pii.items():
                pii_counts[k] += v

            rec = dict(rec)
            rec["text"] = scrubbed
            writers[source].write(rec)
            counts["final"] += 1
            totals["final"] += 1

            if totals["stage1_survivors"] % 2000 == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(
                    f"\r[dedup+scrub] {totals['stage1_survivors']:>10,} seen "
                    f"| {totals['final']:>10,} kept "
                    f"| {totals['stage1_survivors'] / elapsed:>7.1f} docs/sec",
                    end="",
                    flush=True,
                )

    for w in writers.values():
        w.close()
    print()
    return {
        "per_source": {s: dict(c) for s, c in per_source_counts.items()},
        "totals": dict(totals),
        "pii": dict(pii_counts),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in-dir", default="data/raw", help="Directory of raw jsonl.zst shards.")
    parser.add_argument("--out-dir", default="data/clean", help="Directory to write cleaned shards.")
    parser.add_argument(
        "--source",
        default="all",
        help="Which single source subdirectory of --in-dir to clean (default: all found).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only read this many docs from the first shard of each source (fast test mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run stage 1-3 filters and print stats without writing any output files.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for stage 1-3.")
    parser.add_argument(
        "--lid-model",
        default="data/models/lid.176.bin",
        help="Path to fastText lid.176.bin (auto-downloaded if missing).",
    )
    parser.add_argument("--lid-threshold", type=float, default=0.65)
    parser.add_argument("--devanagari-ratio", type=float, default=0.70)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=100_000)
    parser.add_argument("--symbol-ratio-max", type=float, default=0.10)
    parser.add_argument("--repeated-line-max", type=float, default=0.30)
    parser.add_argument("--boilerplate-max-frac", type=float, default=0.50)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--num-perm", type=int, default=128, help="MinHash permutations.")
    parser.add_argument("--ngram-size", type=int, default=5, help="Word-shingle size for MinHash.")
    parser.add_argument(
        "--skip-fuzzy-dedup", action="store_true", help="Skip MinHash LSH fuzzy dedup (stage 5)."
    )
    parser.add_argument(
        "--fuzzy-dedup-max-docs",
        type=int,
        default=2_000_000,
        help="Cap on the in-memory MinHash LSH index size; fuzzy dedup is "
        "disabled for the rest of the run past this many surviving docs.",
    )
    parser.add_argument("--shard-mb", type=int, default=500, help="Target size of each output shard, in MB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if args.source == "all":
        sources = sorted(p.name for p in in_dir.iterdir() if p.is_dir())
    else:
        sources = [args.source]
    if not sources:
        raise SystemExit(f"No source subdirectories found under {in_dir}")

    lid_model_path = ensure_lid_model(Path(args.lid_model))

    quality_params = {
        "lid_threshold": args.lid_threshold,
        "devanagari_ratio": args.devanagari_ratio,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "symbol_ratio_max": args.symbol_ratio_max,
        "repeated_line_max": args.repeated_line_max,
        "boilerplate_max_frac": args.boilerplate_max_frac,
    }

    stage1_dir = out_dir / "_stage1"
    stage1_stats = {}

    print("=== stage 1-3: langID + Devanagari ratio + quality heuristics ===")
    for source in sources:
        shard_paths = sorted((in_dir / source).glob("shard_*.jsonl.zst"))
        if not shard_paths:
            print(f"[{source}] no input shards found, skipping")
            continue
        if args.limit is not None:
            shard_paths = shard_paths[:1]

        tasks = []
        for p in shard_paths:
            out_path = stage1_dir / source / p.name
            if out_path.exists() and not args.dry_run:
                continue  # already processed in a previous run
            tasks.append((p, out_path, args.limit, quality_params, args.dry_run))

        source_counts = Counter()
        source_tokens = Counter()
        if tasks:
            with mp.Pool(
                args.workers, initializer=_init_worker, initargs=(str(lid_model_path),)
            ) as pool:
                for result in pool.imap_unordered(_process_shard_stage123, tasks):
                    source_counts.update(result["counts"])
                    source_tokens.update(result["tokens"])

        stage1_stats[source] = {"counts": dict(source_counts), "tokens": dict(source_tokens)}
        print(
            f"[{source}] input={source_counts.get('input', 0):,} "
            f"-> after_langid={source_counts.get('after_langid', 0):,} "
            f"-> after_devanagari={source_counts.get('after_devanagari', 0):,} "
            f"-> after_quality={source_counts.get('after_quality', 0):,}"
        )

    if args.dry_run:
        print("\n[clean] --dry-run: stopping before dedup/PII/write stages")
        _print_and_save_stats(stage1_stats, {}, out_dir, dry_run=True)
        return

    print("\n=== stage 4-6: exact dedup + fuzzy dedup + PII scrub (sequential) ===")
    dedup_stats = dedup_and_scrub(stage1_dir, out_dir, sources, args)

    _print_and_save_stats(stage1_stats, dedup_stats, out_dir, dry_run=False)


def _print_and_save_stats(stage1_stats, dedup_stats, out_dir, dry_run):
    total_input = sum(s["counts"].get("input", 0) for s in stage1_stats.values())
    total_after_quality = sum(s["counts"].get("after_quality", 0) for s in stage1_stats.values())
    per_source_final = dedup_stats.get("per_source", {})

    print("\n=== Summary ===")
    print(f"{'source':<12} {'input':>10} {'after_quality':>14} {'final':>10}")
    for source, s in stage1_stats.items():
        final = "-" if dry_run else per_source_final.get(source, {}).get("final", 0)
        print(
            f"{source:<12} {s['counts'].get('input', 0):>10,} "
            f"{s['counts'].get('after_quality', 0):>14,} {final:>10}"
        )

    stats = {
        "per_source_stage123": stage1_stats,
        "dedup_and_pii": dedup_stats,
        "totals": {"input": total_input, "after_quality": total_after_quality},
    }
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
