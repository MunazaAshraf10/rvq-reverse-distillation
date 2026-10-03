"""Shared plumbing of the experiments: sharding, resumable result files, encoders, provenance.

Every experiment is a module with a main(argv) that processes its share of the items (index modulo
world) and appends one JSON row per item and condition to results/<experiment>/part-<rank>.jsonl.
Rows carry a key; a rerun skips keys already written, so an interrupted job resumes where it
stopped. rvq_ae.experiments.launch starts one process per GPU.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypeVar

import torch
from huggingface_hub import hf_hub_download

from rvq_ae.constants import COLLECTION, DATASET_REPO
from rvq_ae.data.records import Record, load_records
from rvq_ae.data.splits import make_split
from rvq_ae.hub import VARIANTS, load_encoder
from rvq_ae.minimax.replay import Track
from rvq_ae.models.encoder import RvqEncoder

DATASET_REVISION = "5029b1e7f1bbfbf028b76b38564fecccda94a111"
"""Corpus revision behind every number in this repository."""

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Shard:
    rank: int
    world: int

    @classmethod
    def from_env(cls) -> Self:
        return cls(int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")))

    def take(self, items: Sequence[T]) -> list[T]:
        return [item for index, item in enumerate(items) if index % self.world == self.rank]


class Sink:
    """Append only JSONL result file of one rank; keys already present are reported as done."""

    def __init__(self, folder: Path, shard: Shard) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"part-{shard.rank}.jsonl"
        self.done = {row["key"] for row in read_rows(self.path)} if self.path.is_file() else set()

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        self.done.add(row["key"])


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_results(folder: Path) -> list[dict[str, Any]]:
    """Every row of an experiment, all ranks merged."""
    return [row for path in sorted(folder.glob("part-*.jsonl")) for row in read_rows(path)]


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_provenance(folder: Path, args: argparse.Namespace) -> None:
    """Arguments, code revision and library versions of a run, next to its results."""
    folder.mkdir(parents=True, exist_ok=True)
    values = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "git": git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "dataset_revision": DATASET_REVISION,
    }
    (folder / "provenance.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def corpus_part(
    rule: str, part: str, *, limit: int = 0, exact_only: bool = True, dataset: str = DATASET_REPO
) -> list[Record]:
    """Records of one part of a split, sorted by shard id; limit keeps a hashed, stable subset.

    dataset is the Hub corpus or a local folder in the same layout (MM3-OOD); a local folder is not
    re-split, its records are grouped by their own split field.
    """
    local = Path(dataset).is_dir()
    grouped = load_records(dataset, revision=None if local else DATASET_REVISION)
    parts = grouped if local else make_split(grouped, rule).parts
    records = [record for record in parts[part] if record.exact or not exact_only]
    if 0 < limit < len(records):
        records = sorted(records, key=lambda record: hash_key(record.stem))[:limit]
    return sorted(records, key=lambda record: record.shard_id)


def corpus_track(record: Record, dataset: str = DATASET_REPO) -> Track:
    """Audio, codes and stored conditioning of one corpus record, from a local folder or the Hub cache."""
    if Path(dataset).is_dir():
        return Track.read(Path(dataset) / record.shard_path, record)
    path = hf_hub_download(DATASET_REPO, record.shard_path, repo_type="dataset", revision=DATASET_REVISION)
    return Track.read(Path(path), record)


def hash_key(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def encoder_name(spec: str) -> str:
    """Short label of an encoder spec: a release variant (v1 to v4), or <rule>/<config>/<seed> of a
    run folder such as runs/genre-ood/v4_169m/seed-1/final. The split rule is part of the label
    because both rules train the same configurations under the same seeds."""
    if spec in VARIANTS:
        return spec
    path = Path(spec)
    run = path.parent if path.name == "final" else path
    return f"{run.parent.parent.name}/{run.parent.name}/{run.name}"


def load_spec(spec: str, device: torch.device) -> RvqEncoder:
    """A released variant by name or a trained checkpoint folder by path."""
    if spec in VARIANTS:
        return load_encoder(COLLECTION, variant=spec, device=device)
    return load_encoder(spec, device=device)


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True, help="result folder of the experiment")
    parser.add_argument("--device", default="cuda")
