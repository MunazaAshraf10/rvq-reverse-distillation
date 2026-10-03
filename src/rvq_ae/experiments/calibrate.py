"""Calibration of condition replay: what scores do known corruptions of the true codes get?

Every condition replaces part of the sampled code tuple of a held-out track and replays the result:

    random            uniform random codes in every book (the floor)
    other-track       the codes of another held-out track, tiled to length
    shift-k           the true codes delayed by k frames (first frame repeated)
    semantic-random   true c_0, uniform random acoustic books
    acoustic-random   uniform random c_0, true acoustic books
    semantic-greedy   true c_0, acoustic books from the official depth decoder by argmax
    semantic-sampled  true c_0, acoustic books sampled from the official depth decoder (top 50);
                      an alternative valid completion, the redundancy reference
    noise-sem-p       a fraction p of the frames get a uniform random c_0
    noise-ac-p        a fraction p of the acoustic codes are replaced uniformly at random
    <encoder>-c0      the encoder's c_0 with acoustic books sampled from the official depth decoder,
                      the decoding path of the released reference integration
    <encoder>-c0-greedy  the encoder's c_0 with the official depth decoder's argmax acoustic books

The depth decoder reads language model states computed from the true history in one parallel pass;
sampling it frame by frame with its own history would cost a sequential rollout per track.
"""

import argparse
import logging
from collections.abc import Sequence

import torch

from rvq_ae.audio.dav import load_dav
from rvq_ae.audio.timeline import timeline_bounds
from rvq_ae.constants import ACOUSTIC_VOCAB, DAV_REPO, SEMANTIC_VOCAB
from rvq_ae.experiments.common import (
    Shard,
    Sink,
    common_args,
    corpus_part,
    corpus_track,
    load_spec,
    write_provenance,
)
from rvq_ae.inference import CodeEncoder
from rvq_ae.minimax.official import MiniMax
from rvq_ae.minimax.replay import Track, replay, with_priming

log = logging.getLogger("rvq_ae.experiments.calibrate")

NOISE_LEVELS = (0.1, 0.25, 0.5)
SHIFTS = (1, 5)


def random_codes(frames: int, generator: torch.Generator) -> torch.Tensor:
    semantic = torch.randint(SEMANTIC_VOCAB, (frames, 1), generator=generator)
    return torch.cat([semantic, torch.randint(ACOUSTIC_VOCAB, (frames, 7), generator=generator)], dim=1)


def conditions(
    minimax: MiniMax, track: Track, other: torch.Tensor, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Emitted code sequences [frames, 8] of every corruption of one track."""
    true = track.codes[1:]
    frames = true.shape[0]
    noise = random_codes(frames, generator)
    states = minimax.language_states(track.codes, track.record.prompt, track.record.lyrics)
    out = {
        "random": noise,
        "other-track": other.repeat((frames + other.shape[0] - 1) // other.shape[0], 1)[:frames],
        "semantic-random": torch.cat([true[:, :1], noise[:, 1:]], dim=1),
        "acoustic-random": torch.cat([noise[:, :1], true[:, 1:]], dim=1),
        "semantic-greedy": torch.cat(
            [true[:, :1], minimax.sample_acoustic(states, true[:, 0], greedy=True)], dim=1
        ),
        "semantic-sampled": torch.cat(
            [true[:, :1], minimax.sample_acoustic(states, true[:, 0], generator=generator)], dim=1
        ),
    }
    for shift in SHIFTS:
        out[f"shift-{shift}"] = torch.cat([true[:1].expand(shift, -1), true[:-shift]])
    for level in NOISE_LEVELS:
        mask = torch.rand(frames, generator=generator) < level
        semantic = true.clone()
        semantic[mask, 0] = noise[mask, 0]
        out[f"noise-sem-{level}"] = semantic
        mask = torch.rand(frames, 7, generator=generator) < level
        acoustic = true.clone()
        acoustic[:, 1:][mask] = noise[:, 1:][mask]
        out[f"noise-ac-{level}"] = acoustic
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="calibrate")
    common_args(parser)
    parser.add_argument("--rule", default="published")
    parser.add_argument("--part", default="holdout")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--encoders", nargs="*", default=["v4"], help="encoders scored as <name>-c0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shard = Shard.from_env()
    device = torch.device(args.device)
    sink = Sink(args.out, shard)
    if shard.rank == 0:
        write_provenance(args.out, args)
    every = corpus_part(args.rule, args.part, limit=args.limit)
    half = len(every) // 2
    partner = {record.stem: every[(index + half) % len(every)] for index, record in enumerate(every)}
    minimax = MiniMax.load(device=device)
    dav = load_dav(DAV_REPO, device=device)
    models = {name: load_spec(name, torch.device("cpu")) for name in args.encoders}

    records = shard.take(every)
    for index, record in enumerate(records):
        expected = ["random", *(f"{name}-c0{tail}" for name in models for tail in ("", "-greedy"))]
        if all(f"{record.stem}/{condition}" in sink.done for condition in expected):
            continue
        generator = torch.Generator().manual_seed(args.seed * 1_000_003 + record.shard_id)
        track = corpus_track(record)
        other = corpus_track(partner[record.stem]).codes[1:]
        candidates = conditions(minimax, track, other, generator)
        states = minimax.language_states(track.codes, record.prompt, record.lyrics)
        for name, model in models.items():
            encoder = CodeEncoder(dav, model, device=device)
            bounds = timeline_bounds(track.frames, record.chunks, "exact")
            latents = encoder.latents(track.audio, track.sample_rate)
            codes = encoder.encode_latents(latents, bounds=bounds).codes
            semantic = with_priming(codes, track)[1:, 0]
            model.to("cpu")
            acoustic = minimax.sample_acoustic(states, semantic, generator=generator)
            candidates[f"{name}-c0"] = torch.cat([semantic[:, None], acoustic], dim=1)
            greedy = minimax.sample_acoustic(states, semantic, greedy=True)
            candidates[f"{name}-c0-greedy"] = torch.cat([semantic[:, None], greedy], dim=1)
        base = {"shard": record.shard_id, "genre": record.genre, "frames": track.frames}
        for condition, codes in candidates.items():
            key = f"{record.stem}/{condition}"
            if key in sink.done:
                continue
            result = replay(minimax, track, with_priming(codes, track))
            sink.write({"key": key, **base, "condition": condition, **result.to_dict()})
        log.info("rank %d: %d/%d %s", shard.rank, index + 1, len(records), record.stem)


if __name__ == "__main__":
    main()
