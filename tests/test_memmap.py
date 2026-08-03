"""
Memmap .bin/.idx read/write sanity check for data/prepare.py's SplitWriter.

Writes a handful of synthetic token sequences (with the doc-separator token
included, exactly like prepare.py does), then re-opens the result the same
way training/eval code will - np.memmap for .bin, np.fromfile for .idx - and
asserts every document's tokens and the eos placement survive exactly.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from prepare import SplitWriter  # noqa: E402

EOS_ID = 3
DOCS = [
    [10, 20, 30, EOS_ID],
    [40, 50, 60, 70, 80, EOS_ID],
    [5, EOS_ID],
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)

        writer = SplitWriter(out_dir, "test_split", dry_run=False)
        for doc in DOCS:
            writer.write_doc(doc)
        writer.close(out_dir, "test_split")

        assert writer.docs == len(DOCS)
        assert writer.tokens == sum(len(d) for d in DOCS)
        print(f"PASS: writer counted {writer.docs} docs, {writer.tokens} tokens")

        bin_path = out_dir / "test_split.bin"
        idx_path = out_dir / "test_split.idx"
        assert bin_path.exists() and idx_path.exists()

        # exact file size check: .bin is uint16 (2 bytes/token), .idx is int64 (8 bytes/entry)
        expected_bin_bytes = writer.tokens * 2
        expected_idx_bytes = (len(DOCS) + 1) * 8
        assert bin_path.stat().st_size == expected_bin_bytes, "bin file size mismatch"
        assert idx_path.stat().st_size == expected_idx_bytes, "idx file size mismatch"
        print("PASS: .bin/.idx file sizes match token/doc counts exactly")

        tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        offsets = np.fromfile(idx_path, dtype=np.int64)
        assert len(offsets) == len(DOCS) + 1
        assert offsets[-1] == len(tokens) == writer.tokens

        for i, doc in enumerate(DOCS):
            start, end = offsets[i], offsets[i + 1]
            recovered = tokens[start:end].tolist()
            assert recovered == doc, f"doc {i} mismatch: {recovered} != {doc}"
            assert recovered[-1] == EOS_ID, f"doc {i} does not end in eos"
        print(f"PASS: all {len(DOCS)} documents recovered exactly via memmap, each ending in eos")

        # Windows won't let TemporaryDirectory clean up while the memmap's
        # file handle is still open - release it explicitly before exiting.
        del tokens

    print("\ntest_memmap: ALL PASS")


if __name__ == "__main__":
    main()
