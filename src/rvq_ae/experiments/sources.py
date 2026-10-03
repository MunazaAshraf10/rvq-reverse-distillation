"""Evaluation clips from recorded and generated music, behind one interface.

    song-describer  Song Describer Dataset (Manco et al., 2023), valid subset: 547 Creative Commons
                    MTG-Jamendo recordings (2 min excerpts) with human captions and genre tags.
    musdb           MUSDB18-HQ test set (Rafii et al., 2019): 50 professionally mixed songs, as the
                    full mixture and as the accompaniment (drums + bass + other) without vocals.
    corpus          generated tracks of the reverse distillation corpus, any split part.

Every clip is a fixed length excerpt from the middle of its track, stereo at its native rate.
Excerpt selection is deterministic, so every condition of an experiment sees the same audio.
"""

import csv
import glob
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from torch import Tensor

from rvq_ae.audio.io import load_audio
from rvq_ae.audio.timeline import frame_bounds
from rvq_ae.constants import FRAME_RATE, HOP, SAMPLE_RATE
from rvq_ae.experiments.assets import jamendo_tags
from rvq_ae.experiments.common import corpus_part, corpus_track, hash_key

SONG_DESCRIBER = ("renumics/song-describer-dataset", "dc39062efec7515add304b98a54da2948709a808")
MUSDB = ("roro128/musdb18-hq-flac", "32fd4e850b45ea0b635be067b921661984b53029")
INSTRUMENTAL = "[Instrumental]"


@dataclass(slots=True)
class Clip:
    key: str
    audio: Tensor
    """[2, samples] float waveform."""
    rate: int
    caption: str
    lyrics: str
    group: str
    """Genre or stem, for per group breakdowns."""
    source: str
    codes: Tensor | None = None
    """[frames, 8] true emitted codes of the excerpt, for generated clips."""


def excerpt(audio: Tensor, rate: int, seconds: float) -> Tensor:
    """The centred seconds long excerpt of a [channels, samples] waveform, stereo."""
    if audio.shape[0] == 1:
        audio = audio.expand(2, -1)
    length = int(seconds * rate)
    start = max(0, (audio.shape[-1] - length) // 2)
    return audio[:2, start : start + length].contiguous()


def dataset_files(spec: tuple[str, str], pattern: str) -> list[str]:
    root = snapshot_download(spec[0], repo_type="dataset", revision=spec[1], allow_patterns=[pattern])
    return sorted(glob.glob(f"{root}/{pattern}"))


def jamendo_genres() -> dict[int, str]:
    """First genre tag of every MTG-Jamendo track in the Song Describer metadata."""
    genres: dict[int, str] = {}
    with jamendo_tags().open(encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            tags = [tag.split("---", 1)[1] for tag in row[5:] if tag.startswith("genre---")]
            if row and row[0].startswith("track_"):
                genres[int(row[0].removeprefix("track_"))] = tags[0] if tags else "unknown"
    return genres


def song_describer(seconds: float = 30.0, limit: int = 0) -> Iterator[Clip]:
    """One clip per valid track, captioned with its first valid caption; limit keeps a hashed subset."""
    genres = jamendo_genres()
    seen: set[int] = set()
    rows: list[dict[str, object]] = []
    for file in dataset_files(SONG_DESCRIBER, "data/*.parquet"):
        for row in pq.read_table(file).to_pylist():
            track = int(row["track_id"])
            if row["is_valid_subset"] and track not in seen:
                seen.add(track)
                rows.append(row)
    rows.sort(key=lambda row: hash_key(f"sdd:{row['track_id']}"))
    for row in rows[: limit or None]:
        track = int(row["track_id"])  # type: ignore[call-overload]
        audio, rate = load_audio(row["path"]["bytes"])  # type: ignore[index]
        yield Clip(
            key=f"sdd-{track}",
            audio=excerpt(audio, rate, seconds),
            rate=rate,
            caption=str(row["caption"]),
            lyrics=INSTRUMENTAL,
            group=genres.get(track, "unknown"),
            source="song-describer",
        )


def musdb(seconds: float = 30.0, limit: int = 0) -> Iterator[Clip]:
    """Each test song twice: mixture, and accompaniment (drums + bass + other) without vocals."""
    stems: dict[str, dict[str, tuple[Tensor, int]]] = {}
    for file in dataset_files(MUSDB, "data/test-*.parquet"):
        for row in pq.read_table(file).to_pylist():
            song = Path(row["path"]).parent.name
            stems.setdefault(song, {})[row["instrument"]] = load_audio(row["audio"]["bytes"])
    songs = sorted(stems, key=lambda song: hash_key(f"musdb:{song}"))[: limit or None]
    for song in songs:
        parts = stems[song]
        rate = parts["mixture"][1]
        accompaniment = sum(parts[name][0] for name in ("drums", "bass", "other"))
        for group, audio in (("mixture", parts["mixture"][0]), ("accompaniment", accompaniment)):
            yield Clip(
                key=f"musdb-{hash_key(song)}-{group}",
                audio=excerpt(torch.as_tensor(audio), rate, seconds),
                rate=rate,
                caption="",
                lyrics=INSTRUMENTAL,
                group=group,
                source="musdb",
            )


def corpus(rule: str, part: str, seconds: float = 30.0, limit: int = 0) -> Iterator[Clip]:
    """Generated tracks, captioned and with lyrics exactly as they were generated.

    The excerpt starts on a frame of the stitched timeline, so it carries its true codes; frame f
    starts at latent s_f, which is sample 512 s_f.
    """
    for record in corpus_part(rule, part, limit=limit):
        track = corpus_track(record)
        frames = int(seconds * FRAME_RATE)
        first = max(0, (track.frames - frames) // 2)
        bounds = frame_bounds(track.frames, record.chunks)
        start = bounds[first] * HOP * track.sample_rate // SAMPLE_RATE
        audio = track.audio[:2, start : start + int(seconds * track.sample_rate)]
        yield Clip(
            key=record.stem,
            audio=audio.contiguous(),
            rate=track.sample_rate,
            caption=record.prompt,
            lyrics=record.lyrics,
            group=record.genre,
            source=f"corpus-{part}",
            codes=track.codes[1 + first : 1 + first + frames],
        )


def clips(source: str, *, seconds: float = 30.0, limit: int = 0) -> Iterator[Clip]:
    """Clips of a source name: song-describer, musdb, or corpus:<rule>:<part>."""
    if source == "song-describer":
        return song_describer(seconds, limit)
    if source == "musdb":
        return musdb(seconds, limit)
    if source.startswith("corpus:"):
        _, rule, part = source.split(":")
        return corpus(rule, part, seconds, limit)
    raise ValueError(f"unknown clip source {source!r}")
