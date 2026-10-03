"""Batching of render jobs around the one-stage-per-GPU constraint, and the result writer.

The 8B language model and the 2.4B flow transformer do not fit a 24 GB card together. Experiments
therefore build the conditioning of several clips with the language model on the device, queue it
here, and flush: the queue swaps the renderer in, renders every job, swaps the language model back,
and hands each clip its waveforms. Rendering noise is seeded by the clip, so the conditions of one
clip differ only in their conditioning.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import Tensor

from rvq_ae.constants import SAMPLE_RATE
from rvq_ae.experiments.common import Shard, Sink, hash_key
from rvq_ae.experiments.metrics import Analysis
from rvq_ae.experiments.sources import Clip
from rvq_ae.minimax.official import MiniMax
from rvq_ae.minimax.render import load_scheduler, render

Waveforms = dict[str, tuple[Tensor, int]]


def decode(minimax: MiniMax, latents: Tensor) -> Tensor:
    """Waveform [2, samples] of DAV latents [latents, 128] through the MiniMax vocoder."""
    if minimax.vocoder is None:
        raise ValueError("load MiniMax with render=True")
    with torch.no_grad():
        decoded = minimax.vocoder(latents.T[None].float().to(minimax.device))[0]
    return decoded.clamp(-1, 1).cpu()


def seed_of(key: str) -> int:
    return int(hash_key(key), 16) % (2**31)


@dataclass(slots=True)
class Job[Item]:
    item: Item
    key: str
    conditions: dict[str, list[Tensor]] = field(default_factory=dict)
    """Window conditioning per condition name, rendered by MiniMax."""
    latents: dict[str, Tensor] = field(default_factory=dict)
    """DAV latents [latents, 128] per condition name, decoded by the MiniMax vocoder."""
    waveforms: Waveforms = field(default_factory=dict)
    """Waveforms that need no rendering (codecs, external models), passed through unchanged."""


class RenderQueue[Item]:
    def __init__(self, minimax: MiniMax, size: int, finish: Callable[[Item, Waveforms], None]) -> None:
        self.minimax = minimax
        self.size = size
        self.finish = finish
        self.scheduler: FlowMatchEulerDiscreteScheduler = load_scheduler()
        self.jobs: list[Job[Item]] = []
        minimax.place("analysis")

    def add(self, job: Job[Item]) -> None:
        self.jobs.append(job)
        if len(self.jobs) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.jobs:
            return
        self.minimax.place("render")
        for job in self.jobs:
            for name, chunks in job.conditions.items():
                waveform = render(self.minimax, chunks, seed=seed_of(job.key), scheduler=self.scheduler)
                job.waveforms[name] = (waveform, SAMPLE_RATE)
            for name, latents in job.latents.items():
                job.waveforms[name] = (decode(self.minimax, latents), SAMPLE_RATE)
            job.conditions.clear()
            job.latents.clear()
        self.minimax.place("analysis")
        jobs, self.jobs = self.jobs, []
        for job in jobs:
            self.finish(job.item, job.waveforms)


class Writer:
    """Metric rows to the sink, embeddings to a side file, audio of the first clips to disk."""

    def __init__(self, out: Path, sink: Sink, shard: Shard, save_audio: int) -> None:
        self.sink = sink
        self.embeddings = out / "embeddings" / f"part-{shard.rank}.jsonl"
        self.embeddings.parent.mkdir(parents=True, exist_ok=True)
        self.audio = out / "audio"
        self.save_audio = save_audio
        self.saved = 0

    def write(self, clip: Clip, condition: str, analysis: Analysis, metrics: dict[str, object]) -> None:
        base = {
            "key": f"{clip.key}/{condition}",
            "clip": clip.key,
            "group": clip.group,
            "source": clip.source,
        }
        self.sink.write(metrics | base | {"condition": condition})
        with self.embeddings.open("a", encoding="utf-8") as handle:
            embeddings = {"clap": analysis.clap.half().tolist(), "mert": analysis.mert.half().tolist()}
            handle.write(json.dumps({"key": base["key"], "condition": condition, **embeddings}) + "\n")

    def keep_audio(self, clip: Clip, waveforms: Waveforms) -> None:
        if self.saved >= self.save_audio:
            return
        folder = self.audio / clip.key
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "reference.flac", clip.audio.T.numpy(), clip.rate)
        for condition, (waveform, rate) in waveforms.items():
            name = re.sub(r"[^A-Za-z0-9@._-]+", "-", condition)
            sf.write(folder / f"{name}.flac", waveform.T.numpy(), rate)
        self.saved += 1
