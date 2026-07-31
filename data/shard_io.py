"""
Shared read/write helpers for the zstd-compressed jsonl shards used across
download.py, clean.py, and prepare.py.

Shards are written with a streaming zstd compressor that has no content-size
in its frame header, so they must be read back with a streaming decompressor
(`ZstdDecompressor().stream_reader`), not the one-shot `.decompress()` API.
"""

import json
from pathlib import Path
from typing import Iterator

import zstandard as zstd


class ShardWriter:
    """Writes jsonl records into zstd-compressed shards of ~target_mb each.

    Resumable via a `_state.json` progress file dropped alongside the shards:
    on restart, the caller skips `docs_done` records from its input and
    writing continues from a fresh shard (a partially-filled shard from the
    previous run is left as-is rather than being reopened/appended to).
    """

    def __init__(self, out_dir: Path, target_mb: int = 500, dry_run: bool = False):
        self.out_dir = out_dir
        self.target_bytes = target_mb * 1024 * 1024
        self.dry_run = dry_run
        self.state_path = out_dir / "_state.json"
        self.docs_done = 0
        self.shard_idx = 0
        self._fh = None
        self._compressor = None
        self._bytes_in_shard = 0
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.state_path.exists():
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.docs_done = state["docs_done"]
                self.shard_idx = state["next_shard_idx"]

    def skip_to_resume_point(self) -> int:
        return self.docs_done

    def _open_new_shard(self) -> None:
        path = self.out_dir / f"shard_{self.shard_idx:04d}.jsonl.zst"
        self._fh = open(path, "wb")
        self._compressor = zstd.ZstdCompressor(level=9).stream_writer(self._fh)
        self._bytes_in_shard = 0

    def write(self, record: dict) -> None:
        self.docs_done += 1
        if self.dry_run:
            return
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if self._compressor is None:
            self._open_new_shard()
        self._compressor.write(line)
        self._bytes_in_shard += len(line)
        if self._bytes_in_shard >= self.target_bytes:
            self._close_shard()
            self.shard_idx += 1

    def _close_shard(self) -> None:
        if self._compressor is not None:
            self._compressor.flush(zstd.FLUSH_FRAME)
            self._fh.close()
            self._compressor = None
            self._fh = None

    def close(self) -> None:
        self._close_shard()
        if not self.dry_run:
            self.state_path.write_text(
                json.dumps(
                    {"docs_done": self.docs_done, "next_shard_idx": self.shard_idx}
                ),
                encoding="utf-8",
            )


def _iter_one_shard_file(path: Path) -> Iterator[dict]:
    with open(path, "rb") as fh:
        with zstd.ZstdDecompressor().stream_reader(fh) as reader:
            buffer = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if line:
                        yield json.loads(line)
            if buffer.strip():
                yield json.loads(buffer)


def iter_shards(path: Path) -> Iterator[dict]:
    """Yield jsonl records from shard_*.jsonl.zst file(s), in order.

    `path` may be a directory (all shard_*.jsonl.zst files inside, sorted) or
    a single shard file.
    """
    path = Path(path)
    files = sorted(path.glob("shard_*.jsonl.zst")) if path.is_dir() else [path]
    for f in files:
        yield from _iter_one_shard_file(f)


def estimate_tokens(text: str) -> int:
    """Cheap whitespace-split proxy for token count (no tokenizer until Phase 4)."""
    return len(text.split())
