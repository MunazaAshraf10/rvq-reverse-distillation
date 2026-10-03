"""Analysis by synthesis: encode a clip, render the codes with MiniMax Music 3, compare with the clip.

This is the evaluation that applies to recorded music, where no true codes exist. The same protocol
runs on generated clips, which places the real-audio results on the scale of the generator's own
distribution. Conditions, each producing one waveform per clip:

    enc:<spec>       the encoder's greedy codes, teacher forced and rendered (2.1 kbps)
    sem:<spec>       the encoder's semantic codes with acoustic books sampled by MiniMax (0.35 kbps)
    true             the generator's sampled codes of the excerpt, rendered again (generated clips)
    text             MiniMax from the caption alone, no reference (floor)
    dav              the DAV autoencoder: posterior mean decoded by the MiniMax vocoder (latent ceiling)
    codec:<name>     a neural codec reconstruction (rvq_ae.experiments.codecs)

Every code sequence is preceded by the same priming row, sampled from the language model with the
clip's caption, and rendered with noise seeded by the clip, so the conditions of a clip differ only
in their codes. Rendering is batched by rvq_ae.experiments.pipeline.RenderQueue.

Rows hold the metrics; the CLAP and MERT embeddings behind the Frechet distances are written to
embeddings/part-<rank>.jsonl beside them.
"""

import argparse
import logging
from collections.abc import Sequence

import torch
from torch import Tensor

from rvq_ae.audio.dav import DavEncoder, load_dav
from rvq_ae.audio.io import resample
from rvq_ae.constants import DAV_REPO, FRAME_RATE
from rvq_ae.experiments.codecs import Codec
from rvq_ae.experiments.common import Shard, Sink, common_args, load_spec, write_provenance
from rvq_ae.experiments.metrics import Analysis, Embedder, compare
from rvq_ae.experiments.pipeline import Job, RenderQueue, Waveforms, Writer, seed_of
from rvq_ae.experiments.sources import Clip, clips
from rvq_ae.inference import CodeEncoder
from rvq_ae.minimax.official import MiniMax
from rvq_ae.minimax.rollout import rollout

log = logging.getLogger("rvq_ae.experiments.resynth")

GENERATIVE = ("enc", "sem", "text", "true")


def dav_latents(dav: DavEncoder, audio: Tensor, rate: int) -> Tensor:
    """DAV posterior means [latents, 128] of a waveform."""
    with torch.no_grad():
        wave = resample(audio.float(), rate, dav.sample_rate).to(next(dav.parameters()).device)
        return dav.encode(wave)[0].transpose(0, 1).float().cpu()


def code_sequence(
    minimax: MiniMax,
    encoders: dict[str, CodeEncoder],
    clip: Clip,
    condition: str,
    priming: Tensor,
    generator: torch.Generator,
) -> Tensor | None:
    """Emitted codes [frames, 8] of one MiniMax condition, or None when it does not apply to the clip."""
    frames = int(clip.audio.shape[-1] / clip.rate * FRAME_RATE)
    kind, _, name = condition.partition(":")
    if kind in ("enc", "sem"):
        encoder = encoders[name]
        encoder.model.to(minimax.device)
        codes = encoder.encode(clip.audio, clip.rate).codes[:frames]
        encoder.model.to("cpu")
        if kind == "enc":
            return codes
        states = minimax.language_states(torch.cat([priming, codes]), clip.caption, clip.lyrics)
        return torch.cat([codes[:, :1], minimax.sample_acoustic(states, codes[:, 0], generator=generator)], 1)
    if kind == "true":
        return clip.codes
    if kind == "text":
        result = rollout(
            minimax, clip.caption, clip.lyrics, frames=frames, generator=generator, stop_at_end=False
        )
        return result.codes[1:]
    raise ValueError(f"unknown condition {condition!r}")


def conditioning(
    minimax: MiniMax, encoders: dict[str, CodeEncoder], clip: Clip, wanted: list[str]
) -> dict[str, list[Tensor]]:
    """Window conditioning of every MiniMax condition of one clip (language model stage)."""
    generator = torch.Generator(device=minimax.device).manual_seed(seed_of(clip.key))
    priming = rollout(minimax, clip.caption, clip.lyrics, frames=0, generator=generator).codes
    out: dict[str, list[Tensor]] = {}
    for condition in wanted:
        codes = code_sequence(minimax, encoders, clip, condition, priming, generator)
        if codes is None:
            continue
        hiddens = minimax.analyse(torch.cat([priming, codes]), clip.caption, clip.lyrics).hiddens
        out[condition] = [part.cpu() for part in minimax.chunk_conditions(hiddens)]
    return out


def score(embedder: Embedder, writer: Writer, clip: Clip, waveforms: Waveforms) -> None:
    """Every waveform of a clip against the clip itself, plus CLAP text agreement with its caption."""
    reference = Analysis.of(embedder, clip.audio, clip.rate)
    text = embedder.clap_text([clip.caption])[0] if clip.caption else None
    if f"{clip.key}/reference" not in writer.sink.done:
        writer.write(
            clip,
            "reference",
            reference,
            {"clap_text": None if text is None else float(reference.clap @ text)},
        )
    for condition, (waveform, rate) in waveforms.items():
        candidate = Analysis.of(embedder, waveform, rate)
        metrics: dict[str, object] = dict(compare(candidate, reference))
        metrics["clap_text"] = None if text is None else float(candidate.clap @ text)
        writer.write(clip, condition, candidate, metrics)
    writer.keep_audio(clip, waveforms)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="resynth")
    common_args(parser)
    parser.add_argument("--source", required=True, help="song-describer, musdb or corpus:<rule>:<part>")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--group", type=int, default=8, help="clips per language model / renderer swap")
    parser.add_argument(
        "--save-audio", type=int, default=0, help="keep the audio of the first N clips per rank"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shard = Shard.from_env()
    device = torch.device(args.device)
    sink = Sink(args.out, shard)
    if shard.rank == 0:
        write_provenance(args.out, args)
    writer = Writer(args.out, sink, shard, args.save_audio)
    minimax = MiniMax.load(device=device, render=True)
    dav = load_dav(DAV_REPO, device=device)
    specs = {c.partition(":")[2] for c in args.conditions if c.partition(":")[0] in ("enc", "sem")}
    encoders = {spec: CodeEncoder(dav, load_spec(spec, torch.device("cpu")), device=device) for spec in specs}
    for encoder in encoders.values():
        encoder.model.to("cpu")
    codecs = {c: Codec(c.partition(":")[2], device) for c in args.conditions if c.startswith("codec:")}
    embedder = Embedder(device)
    queue: RenderQueue[Clip] = RenderQueue(
        minimax, args.group, lambda clip, waveforms: score(embedder, writer, clip, waveforms)
    )

    mine = shard.take(list(clips(args.source, seconds=args.seconds, limit=args.limit)))
    for index, clip in enumerate(mine):
        wanted = [c for c in args.conditions if f"{clip.key}/{c}" not in sink.done]
        if clip.codes is None:
            wanted = [c for c in wanted if c != "true"]
        if not wanted:
            continue
        job: Job[Clip] = Job(clip, clip.key)
        job.conditions = conditioning(
            minimax, encoders, clip, [c for c in wanted if c.split(":")[0] in GENERATIVE]
        )
        if "dav" in wanted:
            job.latents["dav"] = dav_latents(dav, clip.audio, clip.rate)
        for condition in (c for c in wanted if c.startswith("codec:")):
            job.waveforms[condition] = codecs[condition](clip.audio, clip.rate)
        queue.add(job)
        log.info("rank %d: %d/%d clips queued", shard.rank, index + 1, len(mine))
    queue.flush()


if __name__ == "__main__":
    main()
