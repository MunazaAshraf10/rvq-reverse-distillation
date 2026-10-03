"""Condition replay: how much of the generator's conditioning a code sequence recovers.

A code sequence is teacher forced through the official language model and depth decoder with the
track's own caption and lyrics, the per frame hidden states pass through the condition encoder with
the official chunking, and the stitched conditioning is compared with the one the generator stored
while it rendered the track. The score of a track is the mean cosine over stitched latent frames;
the true sampled codes score 0.9999 (float reassociation between the cached rollout and one
parallel pass), which is the ceiling of the metric.
"""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_bytes
from torch import Tensor

from rvq_ae.audio.io import load_audio
from rvq_ae.data.records import Record
from rvq_ae.minimax.official import MiniMax


@dataclass(slots=True)
class Track:
    """Everything replay needs from one generated track."""

    record: Record
    audio: Tensor
    """[channels, samples] float waveform."""
    sample_rate: int
    codes: Tensor
    """[frames + 1, 8] sampled codes including the priming row 0."""
    condition: Tensor
    """[latents, 2048] stitched conditioning stored by the generator."""

    @property
    def frames(self) -> int:
        return int(self.codes.shape[0]) - 1

    @classmethod
    def read(cls, shard: Path, record: Record) -> Self:
        with zipfile.ZipFile(shard) as archive:
            audio, rate = load_audio(archive.read(record.audio_file))
            tensors = load_bytes(archive.read(record.tensor_file))
        frames = min(record.emitted_frames, int(tensors["codes"].shape[0]) - record.offset)
        codes = tensors["codes"][: frames + 1].long()
        return cls(record, audio, rate, codes, tensors["condition_embeddings"])


@dataclass(slots=True)
class Replay:
    cosine: float
    """Mean cosine over stitched latent frames."""
    semantic_nll: float
    """Mean NLL of the semantic codes under the conditional language model."""
    acoustic_nll: float
    """Mean NLL of the acoustic codes under the depth decoder."""

    def to_dict(self) -> dict[str, float]:
        return {"cosine": self.cosine, "semantic_nll": self.semantic_nll, "acoustic_nll": self.acoustic_nll}


def with_priming(emitted: Tensor, track: Track) -> Tensor:
    """Prepend the track's priming row to emitted codes [frames, 8], padding or trimming to its length.

    The priming row is sampled before the first emitted frame and never reaches the audio, so no
    encoder can recover it; every code sequence is replayed with the true one. A sequence shorter
    than the track (an encoder sees only complete latents) is extended with its own last frame.
    """
    emitted = emitted.to(torch.long)
    if emitted.shape[0] < track.frames:
        emitted = torch.cat([emitted, emitted[-1:].expand(track.frames - emitted.shape[0], -1)])
    return torch.cat([track.codes[:1], emitted[: track.frames]])


@torch.no_grad()
def replay(minimax: MiniMax, track: Track, codes: Tensor) -> Replay:
    """Replay a full code sequence [frames + 1, 8] (priming row included) against the stored conditioning."""
    analysis = minimax.analyse(codes, track.record.prompt, track.record.lyrics)
    condition = minimax.stitched_condition(analysis.hiddens, track.record.chunks)
    stored = track.condition.to(condition.device)
    if condition.shape != stored.shape:
        raise ValueError(
            f"{track.record.stem}: replayed {tuple(condition.shape)} != stored {tuple(stored.shape)}"
        )
    cosine = F.cosine_similarity(condition.float(), stored.float(), dim=-1)
    return Replay(
        cosine=cosine.mean().item(),
        semantic_nll=analysis.semantic_nll.mean().item(),
        acoustic_nll=analysis.acoustic_nll.mean().item(),
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
