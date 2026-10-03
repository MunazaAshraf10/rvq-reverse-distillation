"""Condition replay of encoders on generated tracks with known codes.

One row per (track, encoder, timeline): replay cosine and code likelihoods of the encoder's greedy
codes, plus token agreement with the sampled codes; one control row per track replays the sampled
codes themselves.

    python -m rvq_ae.experiments.launch --gpus 0,1,2,3 -- replay \\
        --out results/replay/published --rule published --part holdout \\
        --encoders v1 v2 v3 v4 --timelines exact nominal stride
"""

import argparse
import logging
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import replace

import torch

from rvq_ae.audio.dav import DavEncoder, load_dav
from rvq_ae.audio.perturb import STRENGTHS, perturb
from rvq_ae.audio.timeline import TIMELINES, timeline_bounds
from rvq_ae.constants import DATASET_REPO, DAV_REPO
from rvq_ae.experiments.common import (
    Shard,
    Sink,
    common_args,
    corpus_part,
    corpus_track,
    encoder_name,
    load_spec,
    write_provenance,
)
from rvq_ae.inference import CodeEncoder
from rvq_ae.minimax.official import MiniMax
from rvq_ae.minimax.replay import Track, replay, with_priming
from rvq_ae.minimax.rollout import Stream, rollout_streams
from rvq_ae.models.encoder import RvqEncoder

log = logging.getLogger("rvq_ae.experiments.replay")

DECODES = ("encoder", "complete-greedy", "complete-sampled")


def perturbations(specs: list[str]) -> list[tuple[str, float]]:
    """Parsed name:strength pairs; 'grid' expands to the whole benchmark grid of rvq_ae.audio.perturb."""
    if specs == ["grid"]:
        grid = [(name, strength) for name, strengths in STRENGTHS.items() for strength in strengths]
        return [("clean", 0.0), *grid]
    return [(spec.split(":")[0], float(spec.split(":")[1]) if ":" in spec else 0.0) for spec in specs]


def row_key(stem: str, name: str, timeline: str, decode: str) -> str:
    """Row key; the encoder decode keeps the key layout of the published reproduction."""
    return f"{stem}/{name}/{timeline}" + ("" if decode == "encoder" else f"/{decode}")


def complete(minimax: MiniMax, track: Track, jobs: list[tuple[torch.Tensor, str]]) -> list[torch.Tensor]:
    """Generator completed decoding, batched: each job forces an encoder's semantic codes at every
    frame and lets the official depth decoder choose the acoustic books (argmax for complete-greedy,
    a top 50 sample for complete-sampled), with the track's recorded priming row and prompt."""
    if not jobs:
        return []
    streams = [
        Stream(
            candidates=with_priming(codes, track)[1:, :1],
            interval=1,
            greedy_depth=decode == "complete-greedy",
            priming=track.codes[0],
        )
        for codes, decode in jobs
    ]
    generator = torch.Generator(device=minimax.device).manual_seed(track.record.shard_id)
    codes: list[torch.Tensor] = []
    # Two streams (four cache rows) per rollout: the key value cache of a six minute track for more
    # does not fit beside the 8B language model on a 24 GB card.
    for start in range(0, len(streams), 2):
        results = rollout_streams(
            minimax,
            track.record.prompt,
            track.record.lyrics,
            streams[start : start + 2],
            frames=track.frames,
            generator=generator,
        )
        codes += [result.codes[1:] for result in results]
    return codes


def agreement(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Exact top 1 agreement of greedy codes with the sampled ones, semantic and pooled acoustic."""
    frames = min(predicted.shape[0], target.shape[0])
    hits = (predicted[:frames] == target[:frames]).float()
    return {"semantic_top1": hits[:, 0].mean().item(), "acoustic_top1": hits[:, 1:].mean().item()}


def encode_all(
    models: dict[str, RvqEncoder],
    todo: list[tuple[str, str]],
    dav: DavEncoder,
    track: Track,
    device: torch.device,
) -> list[tuple[str, str, torch.Tensor]]:
    """Greedy codes of every (encoder, timeline) pair; each encoder is on the device only while encoding."""
    if not todo:
        return []
    latents: torch.Tensor | None = None
    results: list[tuple[str, str, torch.Tensor]] = []
    for name in dict.fromkeys(name for name, _ in todo):
        encoder = CodeEncoder(dav, models[name], device=device)
        if latents is None:
            latents = encoder.latents(track.audio, track.sample_rate)
        for timeline in (timeline for model, timeline in todo if model == name):
            bounds = timeline_bounds(track.frames, track.record.chunks, timeline)
            bounds = bounds[: bisect_right(bounds, latents.shape[0])]
            results.append((name, timeline, encoder.encode_latents(latents, bounds=bounds).codes))
        models[name].to("cpu")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="replay")
    common_args(parser)
    parser.add_argument("--rule", default="published")
    parser.add_argument("--dataset", default=DATASET_REPO, help="the Hub corpus or a local folder (MM3-OOD)")
    parser.add_argument("--part", default="holdout")
    parser.add_argument("--limit", type=int, default=0, help="stable hashed subset of the part")
    parser.add_argument("--encoders", nargs="+", required=True, help="release variants or run folders")
    parser.add_argument("--timelines", nargs="+", default=["exact"], choices=TIMELINES)
    parser.add_argument(
        "--decodes",
        nargs="+",
        default=["encoder"],
        choices=DECODES,
        help="encoder: the encoder's own codes; complete-*: its semantic codes, acoustic books by MiniMax",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=[],
        help="name:strength pairs (rvq_ae.audio.perturb) or 'grid' for the benchmark grid; codes are encoded "
        "from the degraded audio and replayed against the clean track's conditioning",
    )
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shard = Shard.from_env()
    device = torch.device(args.device)
    sink = Sink(args.out, shard)
    if shard.rank == 0:
        write_provenance(args.out, args)
    records = shard.take(corpus_part(args.rule, args.part, limit=args.limit, dataset=args.dataset))
    minimax = MiniMax.load(device=device)
    dav = load_dav(DAV_REPO, device=device)
    models = {encoder_name(spec): load_spec(spec, torch.device("cpu")) for spec in args.encoders}

    for index, record in enumerate(records):
        track = corpus_track(record, args.dataset)
        base = {"shard": record.shard_id, "genre": record.genre, "frames": track.frames}
        if not args.no_control and f"{record.stem}/control" not in sink.done:
            result = replay(minimax, track, track.codes)
            sink.write({"key": f"{record.stem}/control", **base, "encoder": "control", **result.to_dict()})

        todo = [
            (name, timeline)
            for name in models
            for timeline in args.timelines
            if any(row_key(record.stem, name, timeline, decode) not in sink.done for decode in args.decodes)
        ]
        for degradation, strength in perturbations(args.perturbations):
            tag = f"{degradation}:{strength:g}"
            pending = [
                (name, timeline)
                for name in models
                for timeline in args.timelines
                if f"{row_key(record.stem, name, timeline, 'encoder')}/{tag}" not in sink.done
            ]
            if not pending:
                continue
            audio = track.audio
            if degradation != "clean":
                audio = perturb(audio, track.sample_rate, degradation, strength, seed=record.shard_id)
            for name, timeline, encoded in encode_all(
                models, pending, dav, replace(track, audio=audio), device
            ):
                result = replay(minimax, track, with_priming(encoded, track))
                row = {"key": f"{row_key(record.stem, name, timeline, 'encoder')}/{tag}", **base}
                row |= {
                    "encoder": name,
                    "timeline": timeline,
                    "perturbation": degradation,
                    "strength": strength,
                }
                sink.write(row | result.to_dict() | agreement(encoded, track.codes[1:]))
        if args.perturbations:
            continue
        clean = encode_all(models, todo, dav, track, device)
        requested = [
            (name, timeline, codes, decode)
            for name, timeline, codes in clean
            for decode in args.decodes
            if row_key(record.stem, name, timeline, decode) not in sink.done
        ]
        to_complete = [(codes, decode) for _, _, codes, decode in requested if decode != "encoder"]
        completed = iter(complete(minimax, track, to_complete))
        for name, timeline, codes, decode in requested:
            final = codes if decode == "encoder" else next(completed)
            result = replay(minimax, track, with_priming(final, track))
            row = {"key": row_key(record.stem, name, timeline, decode), **base, "encoder": name}
            row |= {"timeline": timeline, "decode": decode}
            sink.write(row | result.to_dict() | agreement(final, track.codes[1:]))
        log.info("rank %d: %d/%d %s", shard.rank, index + 1, len(records), record.stem)


if __name__ == "__main__":
    main()
