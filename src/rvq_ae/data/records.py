import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from rvq_ae.audio.timeline import Chunk
from rvq_ae.constants import DATASET_REPO, SAMPLE_RATE, VOCABS

UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    return UNSAFE.sub("_", value).strip("_") or "sample"


@dataclass(frozen=True, slots=True)
class Record:
    shard_id: int
    shard_path: str
    sample_id: str
    audio_file: str
    tensor_file: str
    split: str
    emitted_frames: int
    sample_rate: int
    vocab_sizes: tuple[int, ...]
    offset: int
    chunks: tuple[Chunk, ...] | None
    prompt: str = ""
    lyrics: str = ""
    seed: int = 0
    genre: str = ""

    @property
    def stem(self) -> str:
        return f"shard-{self.shard_id:06d}-{safe_name(self.sample_id)}"

    @property
    def exact(self) -> bool:
        """True when the generator recorded the chunk stitching table."""
        return bool(self.chunks)


def parse_entry(entry: Mapping[str, Any]) -> Record | None:
    """One record per index entry; None for entries without a succeeded job."""
    jobs = entry.get("manifest", {}).get("jobs", [])
    if len(jobs) != 1:
        return None
    job = jobs[0]
    if job.get("status") not in (None, "succeeded"):
        return None
    alignment = job.get("alignment") or entry.get("alignment") or {}
    stitching = alignment.get("chunk_stitching")
    chunks = tuple(Chunk.parse(chunk) for chunk in stitching) if stitching else None
    split = (
        job.get("dataset_split")
        or entry.get("dataset_split")
        or job.get("source_metadata", {}).get("dataset_split", "train")
    )
    vocabs = tuple(int(v) for v in job.get("codebook_vocab_sizes", VOCABS))
    if len(vocabs) != len(VOCABS):
        raise ValueError(
            f"shard {entry['shard_id']} declares {len(vocabs)} codebooks, expected {len(VOCABS)}"
        )
    return Record(
        shard_id=int(entry["shard_id"]),
        shard_path=str(entry["path"]),
        sample_id=str(job["id"]),
        audio_file=str(job["audio_file"]),
        tensor_file=str(job["tensor_file"]),
        split=str(split).lower(),
        emitted_frames=int(job["emitted_frames"]),
        sample_rate=int(job.get("sampling_rate", SAMPLE_RATE)),
        vocab_sizes=vocabs,
        offset=int(alignment.get("emitted_code_row_offset", 1)),
        chunks=chunks,
        prompt=str(job.get("prompt") or ""),
        lyrics=str(job.get("lyrics") or ""),
        seed=int(job.get("seed") or 0),
        genre=str(job.get("source_metadata", {}).get("source_detail") or "").lower(),
    )


def iter_entries(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_index(path: Path) -> list[Record]:
    return [record for record in map(parse_entry, iter_entries(path)) if record is not None]


def index_files(source: str | Path, *, revision: str | None, limit: int) -> list[Path]:
    """JSONL files from a local file, a local folder, or the indexes/ folder of a Hub dataset."""
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl")) or sorted((path / "indexes").glob("*.jsonl"))
    else:
        root = snapshot_download(
            str(source), repo_type="dataset", revision=revision, allow_patterns=["indexes/*.jsonl"]
        )
        files = sorted((Path(root) / "indexes").glob("*.jsonl"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no index files found under {source}")
    return files


def load_records(
    source: str | Path = DATASET_REPO,
    *,
    revision: str | None = None,
    limit: int = 0,
) -> dict[str, list[Record]]:
    """Records grouped by split, sorted by shard id."""
    grouped: dict[str, list[Record]] = {}
    for file in index_files(source, revision=revision, limit=limit):
        for record in read_index(file):
            grouped.setdefault(record.split, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda record: record.shard_id)
    return grouped
