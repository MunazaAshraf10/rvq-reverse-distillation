"""What a reference recording can do to MiniMax Music 3 through the recovered encoder.

reference    reference-guided generation: MiniMax writes music for a *different* caption (one from
             a clip of another genre) while every Nth semantic code is restricted to the encoder's
             top 5 candidates for the reference (the released integration). Sweeping N trades
             adherence to the reference against adherence to the new caption.
                 ref:<spec>@<N>     constraint every N frames
                 text               the target caption alone (N = infinity)
                 musicgen-melody    MusicGen melody conditioned on the reference chroma and the caption
continuation audio-prompted continuation: the first 10 s of a recording are encoded and become the
             language model's history, MiniMax continues for 20 s, and the continuation is compared
             with what the recording actually does next.
                 cont:<spec>        the encoder's codes of the prompt as history
                 text               the caption alone, no history
                 musicgen           MusicGen's native audio continuation with the caption
                 other              the true continuation of another recording (floor)

Clips come from the Song Describer Dataset, so every reference is a real recording.
"""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MusicgenForConditionalGeneration,
    MusicgenMelodyForConditionalGeneration,
)
from transformers.audio_utils import chroma_filter_bank

from rvq_ae.audio.dav import load_dav
from rvq_ae.audio.io import resample
from rvq_ae.constants import DAV_REPO, FRAME_RATE
from rvq_ae.experiments.common import Shard, Sink, common_args, load_spec, write_provenance
from rvq_ae.experiments.metrics import Analysis, Embedder, compare
from rvq_ae.experiments.pipeline import Job, RenderQueue, Waveforms, Writer, seed_of
from rvq_ae.experiments.sources import Clip, clips
from rvq_ae.inference import CodeEncoder
from rvq_ae.minimax.official import MiniMax
from rvq_ae.minimax.rollout import Stream, rollout_streams

log = logging.getLogger("rvq_ae.experiments.usecases")

MUSICGEN = {
    "musicgen-melody": ("facebook/musicgen-melody", "68d653a95788ec0d2b0abccab22c0b3a200c2d90"),
    "musicgen": ("facebook/musicgen-medium", "d3bd7b00761b78ad7a8a05145ee31e7832e9916c"),
}
MUSICGEN_RATE = 32_000
MUSICGEN_FRAME_RATE = 50
PROMPT_SECONDS = 10.0


@dataclass(slots=True)
class Task:
    """One clip of one use case: the reference, the caption to generate for, and what to compare with."""

    clip: Clip
    caption: str
    truth: Clip
    """The audio a generation is scored against: the reference itself, or the true continuation."""
    floor: Clip | None = None
    """For continuation, the true continuation of another recording."""


def cut(clip: Clip, start: float, end: float | None = None) -> Clip:
    first = int(start * clip.rate)
    last = None if end is None else int(end * clip.rate)
    return replace(clip, audio=clip.audio[:, first:last].contiguous())


def pair_tasks(every: list[Clip], use: str) -> list[Task]:
    """Deterministic partners: the next clip (cyclically) of another genre."""
    tasks: list[Task] = []
    for index, clip in enumerate(every):
        others = [every[(index + step) % len(every)] for step in range(1, len(every))]
        partner = next((other for other in others if other.group != clip.group), others[0])
        if use == "reference":
            tasks.append(Task(clip, partner.caption, clip))
        else:
            truth = cut(clip, PROMPT_SECONDS)
            tasks.append(
                Task(cut(clip, 0.0, PROMPT_SECONDS), clip.caption, truth, cut(partner, PROMPT_SECONDS))
            )
    return tasks


MELODY_FFT = 16_384
MELODY_HOP = 4_096


def melody_chroma(wave: Tensor) -> Tensor:
    """MusicGen melody conditioning [1, frames, 12] of a mono 32 kHz waveform [samples].

    The same computation as transformers' MusicgenMelodyFeatureExtractor (which needs torchaudio,
    unavailable for this torch build): a power spectrogram normalised by the window energy, the
    librosa chroma filter bank, max normalisation, then a one hot of the strongest chroma per frame.
    """
    window = torch.hann_window(MELODY_FFT)
    stft = torch.stft(
        wave, MELODY_FFT, MELODY_HOP, window=window, center=True, pad_mode="reflect", return_complex=True
    )
    spectrum = stft.abs().pow(2) / window.pow(2).sum()
    filters = torch.from_numpy(
        chroma_filter_bank(
            sampling_rate=MUSICGEN_RATE, num_frequency_bins=MELODY_FFT, tuning=0, num_chroma=12
        )
    ).float()
    chroma = F.normalize(filters @ spectrum, p=float("inf"), dim=0, eps=1e-6).T
    return F.one_hot(chroma.argmax(-1), 12).float()[None]


class MusicGen:
    """MusicGen melody (reference chroma) or medium (audio continuation), sampled as in its paper."""

    def __init__(self, name: str, device: torch.device) -> None:
        repo, revision = MUSICGEN[name]
        model_class = (
            MusicgenMelodyForConditionalGeneration
            if name == "musicgen-melody"
            else MusicgenForConditionalGeneration
        )
        self.model = (
            model_class.from_pretrained(repo, revision=revision, dtype=torch.float16).to(device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.processor = (
            None if name == "musicgen-melody" else AutoProcessor.from_pretrained(repo, revision=revision)
        )
        self.name = name
        self.device = device

    @torch.no_grad()
    def __call__(
        self, audio: Tensor, rate: int, caption: str, seconds: float, seed: int
    ) -> tuple[Tensor, int]:
        torch.manual_seed(seed)
        wave = resample(audio.float().mean(0, keepdim=True), rate, MUSICGEN_RATE)[0].numpy()
        if self.processor is None:
            inputs = dict(self.tokenizer([caption], padding=True, return_tensors="pt"))
            inputs["input_features"] = melody_chroma(torch.from_numpy(wave))
        else:
            inputs = dict(
                self.processor(
                    audio=wave, sampling_rate=MUSICGEN_RATE, text=[caption], padding=True, return_tensors="pt"
                )
            )
        inputs = {
            key: value.to(self.device, torch.float16 if value.is_floating_point() else None)
            for key, value in inputs.items()
        }
        tokens = int(seconds * MUSICGEN_FRAME_RATE)
        output = self.model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=tokens)
        generated = output[0, 0].float().cpu()
        return generated[-int(seconds * MUSICGEN_RATE) :][None], MUSICGEN_RATE


def stream_of(encoders: dict[str, CodeEncoder], task: Task, condition: str, device: torch.device) -> Stream:
    """How one MiniMax condition steers the rollout: constraint, prefix, or nothing (text)."""
    kind, _, rest = condition.partition(":")
    if kind == "text":
        return Stream()
    name, _, step = rest.partition("@")
    encoder = encoders[name]
    encoder.model.to(device)
    encoded = encoder.encode(task.clip.audio, task.clip.rate, topk=5)
    encoder.model.to("cpu")
    if kind == "ref":
        return Stream(candidates=encoded.semantic_topk, interval=int(step))
    return Stream(prefix=encoded.codes)


def conditioning(
    minimax: MiniMax, encoders: dict[str, CodeEncoder], task: Task, conditions: list[str], seconds: float
) -> dict[str, list[Tensor]]:
    """Window conditioning of every MiniMax condition of a task, decoded as one batched rollout."""
    generator = torch.Generator(device=minimax.device).manual_seed(seed_of(task.clip.key))
    streams = [stream_of(encoders, task, condition, minimax.device) for condition in conditions]
    results = rollout_streams(
        minimax,
        task.caption,
        "[Instrumental]",
        streams,
        frames=int(seconds * FRAME_RATE),
        generator=generator,
    )
    return {
        condition: [part.cpu() for part in minimax.chunk_conditions(result.hiddens.to(minimax.device))]
        for condition, result in zip(conditions, results, strict=True)
    }


def score(embedder: Embedder, writer: Writer, task: Task, waveforms: Waveforms) -> None:
    """Every waveform against the truth (and, for continuations, against the prompt), plus CLAP text."""
    truth = Analysis.of(embedder, task.truth.audio, task.truth.rate)
    prompt = None if task.floor is None else Analysis.of(embedder, task.clip.audio, task.clip.rate)
    target = embedder.clap_text([task.caption])[0]
    source = embedder.clap_text([task.clip.caption])[0]
    for condition, (waveform, rate) in waveforms.items():
        candidate = Analysis.of(embedder, waveform, rate)
        metrics: dict[str, object] = dict(compare(candidate, truth))
        metrics["clap_text_target"] = float(candidate.clap @ target)
        metrics["clap_text_source"] = float(candidate.clap @ source)
        if prompt is not None:
            metrics |= {f"prompt_{key}": value for key, value in compare(candidate, prompt).items()}
        writer.write(task.clip, condition, candidate, metrics)
    writer.keep_audio(task.clip, waveforms)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="usecases")
    common_args(parser)
    parser.add_argument("use", choices=("reference", "continuation"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--save-audio", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shard = Shard.from_env()
    device = torch.device(args.device)
    sink = Sink(args.out, shard)
    if shard.rank == 0:
        write_provenance(args.out, args)
    writer = Writer(args.out, sink, shard, args.save_audio)
    seconds = 20.0 if args.use == "reference" else 30.0
    tasks = shard.take(pair_tasks(list(clips("song-describer", seconds=seconds, limit=args.limit)), args.use))
    generative = [c for c in args.conditions if c.split(":")[0] in ("ref", "cont", "text")]
    if generative and any(c in MUSICGEN for c in args.conditions):
        parser.error("MusicGen and MiniMax conditions do not fit one 24 GB GPU together; run them separately")
    embedder = Embedder(device)
    external = {c: MusicGen(c, device) for c in args.conditions if c in MUSICGEN}
    minimax = MiniMax.load(device=device, render=True) if generative else None
    encoders: dict[str, CodeEncoder] = {}
    if minimax is not None:
        dav = load_dav(DAV_REPO, device=device)
        specs = {c.partition(":")[2].partition("@")[0] for c in generative if ":" in c}
        encoders = {
            spec: CodeEncoder(dav, load_spec(spec, torch.device("cpu")), device=device) for spec in specs
        }
        for encoder in encoders.values():
            encoder.model.to("cpu")

    def finish(task: Task, waveforms: Waveforms) -> None:
        if args.use == "continuation":
            for condition in generative:
                if condition in waveforms:
                    waveform, rate = waveforms[condition]
                    waveforms[condition] = (waveform[:, int(PROMPT_SECONDS * rate) :], rate)
        score(embedder, writer, task, waveforms)

    queue: RenderQueue[Task] | None = (
        RenderQueue(minimax, args.group, finish) if minimax is not None else None
    )
    for index, task in enumerate(tasks):
        wanted = [c for c in args.conditions if f"{task.clip.key}/{c}" not in sink.done]
        if not wanted:
            continue
        job: Job[Task] = Job(task, task.clip.key)
        for condition in wanted:
            if condition in external:
                length = task.truth.audio.shape[-1] / task.truth.rate
                generate = external[condition]
                job.waveforms[condition] = generate(
                    task.clip.audio, task.clip.rate, task.caption, length, seed_of(task.clip.key)
                )
            elif condition == "other" and task.floor is not None:
                job.waveforms[condition] = (task.floor.audio, task.floor.rate)
        mine = [condition for condition in wanted if condition in generative]
        if minimax is not None and mine:
            job.conditions = conditioning(minimax, encoders, task, mine, seconds)
        if queue is None:
            finish(task, job.waveforms)
        else:
            queue.add(job)
        log.info("rank %d: %d/%d tasks", shard.rank, index + 1, len(tasks))
    if queue is not None:
        queue.flush()


if __name__ == "__main__":
    main()
