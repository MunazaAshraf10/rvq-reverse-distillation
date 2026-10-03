"""MM3-OOD: a generated test set with exact codes, outside the corpus' caption distribution.

Captions come from MusicCaps (Agostinelli et al., 2023; CC BY-SA 4.0), restricted to captions that
mention none of the corpus genres (rock, pop, blues, jazz) and stratified over twelve genre
buckets that the corpus does not contain. Half of the tracks are instrumental; the other half sing
the opening lines of a public domain poem (CC0), split into tagged sections. Each track is sampled
with the official recipe (rollout + render) and written in the corpus shard layout, so every loader
and experiment of this repository reads it unchanged:

    <out>/data/<id>.zip      manifest.json, <job>/audio.flac, <job>/prediction.safetensors
    <out>/indexes/ood.jsonl  one index entry per shard

prediction.safetensors holds codes (priming row first), teacher_topk_ids and teacher_topk_logits
(post guidance top 50), and condition_embeddings on the stitched latent timeline.
"""

import argparse
import csv
import io
import json
import logging
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save as save_bytes

from rvq_ae.constants import FRAME_RATE, HOP, SAMPLE_RATE, VOCABS
from rvq_ae.experiments.assets import musiccaps, poems
from rvq_ae.experiments.common import Shard, hash_key, write_provenance
from rvq_ae.minimax.official import CHUNK_FRAMES, MiniMax, chunk_starts, kept_spans
from rvq_ae.minimax.render import load_scheduler, render
from rvq_ae.minimax.rollout import rollout

log = logging.getLogger("rvq_ae.experiments.generate")

CORPUS_GENRES = re.compile(r"\b(rock|pop|blues|jazz)\b", re.IGNORECASE)
BUCKETS = {
    "electronic": r"\b(electronic|techno|house|edm|trance|dubstep)\b",
    "classical": r"\b(classical|orchestra|orchestral|symphon\w*|string quartet)\b",
    "folk": r"\bfolk\b",
    "ambient": r"\b(ambient|drone|meditat\w*)\b",
    "latin": r"\b(latin|salsa|bossa|samba|flamenco|reggaeton)\b",
    "country": r"\b(country|bluegrass)\b",
    "metal": r"\b(metal|heavy metal|death metal)\b",
    "hip-hop": r"\b(hip hop|hip-hop|rap|trap)\b",
    "reggae": r"\b(reggae|dub|ska)\b",
    "soul": r"\b(r&b|soul|funk|gospel)\b",
    "world": r"\b(indian|arabic|african|world|traditional|sitar|tabla)\b",
    "choral": r"\b(choir|choral|a cappella|acapella)\b",
}
PER_BUCKET = 20
SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    caption: str
    lyrics: str
    genre: str
    seed: int


def poem_lyrics(text: str) -> str:
    """Up to 12 lines of a poem as verse, chorus, verse sections of four lines."""
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 3][:12]
    sections = ["[verse]", "[chorus]", "[verse]"]
    out: list[str] = []
    for index in range(0, len(lines), 4):
        out.append(sections[min(index // 4, 2)])
        out.extend(lines[index : index + 4])
    return "\n".join(out)


def make_jobs(captions: Path, poems: Path) -> list[Job]:
    """The stratified jobs (20 per genre bucket); deterministic in the two input files."""
    rows = [
        row
        for row in csv.DictReader(captions.open(encoding="utf-8"))
        if not CORPUS_GENRES.search(row["caption"] + row["aspect_list"])
    ]
    rows.sort(key=lambda row: hash_key(f"ood:{row['ytid']}"))
    texts = [
        poem["text"]
        for poem in json.loads(poems.read_text(encoding="utf-8"))
        if len(str(poem["text"]).splitlines()) >= 8
    ]
    texts.sort(key=lambda text: hash_key(f"poem:{text[:200]}"))
    taken: set[str] = set()
    jobs: list[Job] = []
    for genre, pattern in BUCKETS.items():
        matches = [
            row
            for row in rows
            if row["ytid"] not in taken and re.search(pattern, row["caption"], re.IGNORECASE)
        ]
        for row in matches[:PER_BUCKET]:
            taken.add(row["ytid"])
            sung = len(jobs) % 2 == 1
            lyrics = poem_lyrics(texts[len(jobs)]) if sung else "[Instrumental]"
            identifier = hash_key(f"job:{row['ytid']}")
            jobs.append(Job(identifier, row["caption"], lyrics, genre, int(identifier, 16) % (2**31)))
    return jobs


def stitching(lengths: list[int], frames: int) -> list[dict[str, int]]:
    """The chunk_stitching table of a render, in the corpus' field names."""
    table: list[dict[str, int]] = []
    position = 0
    for index, ((start, end), first) in enumerate(
        zip(kept_spans(lengths), chunk_starts(frames), strict=True)
    ):
        table.append(
            {
                "chunk_index": index,
                "semantic_frame_start": first,
                "semantic_frame_end_exclusive": min(first + CHUNK_FRAMES, frames),
                "raw_flow_latent_length": lengths[index],
                "kept_flow_latent_start": start,
                "kept_flow_latent_end_exclusive": end,
                "stitched_flow_latent_start": position,
                "stitched_flow_latent_end_exclusive": position + end - start,
            }
        )
        position += end - start
    return table


def generate(
    minimax: MiniMax, job: Job, scheduler: FlowMatchEulerDiscreteScheduler
) -> tuple[bytes, dict[str, object]]:
    """One shard (zip bytes) and its index entry."""
    generator = torch.Generator(device=minimax.device).manual_seed(job.seed)
    minimax.place("analysis")
    result = rollout(minimax, job.caption, job.lyrics, frames=int(SECONDS * FRAME_RATE), generator=generator)
    frames = result.hiddens.shape[0]
    chunks = minimax.chunk_conditions(result.hiddens.to(minimax.device))
    spans = kept_spans([chunk.shape[0] for chunk in chunks])
    condition = torch.cat([chunk[start:end] for chunk, (start, end) in zip(chunks, spans, strict=True)]).cpu()
    minimax.place("render")
    audio = render(minimax, [chunk.cpu() for chunk in chunks], seed=job.seed, scheduler=scheduler)
    tensors = save_bytes(
        {
            "codes": result.codes.to(torch.int16),
            "teacher_topk_ids": result.topk_ids.to(torch.int32),
            "teacher_topk_logits": result.topk_logits.to(torch.float16),
            "condition_embeddings": condition.to(torch.bfloat16).contiguous(),
        }
    )
    flac = io.BytesIO()
    sf.write(flac, audio.T.numpy(), SAMPLE_RATE, format="FLAC")
    folder = f"0000-{job.id}"
    audio_file, tensor_file = f"{folder}/audio.flac", f"{folder}/prediction.safetensors"
    entry_job = {
        "status": "succeeded",
        "id": job.id,
        "prompt": job.caption,
        "lyrics": job.lyrics,
        "seed": job.seed,
        "dataset_split": "ood",
        "audio_file": audio_file,
        "tensor_file": tensor_file,
        "emitted_frames": frames,
        "priming_frames": 1,
        "ended_by_eos": result.ended,
        "sampling_rate": SAMPLE_RATE,
        "frame_rate": FRAME_RATE,
        "latent_hop_length": HOP,
        "flow_latent_length": int(condition.shape[0]),
        "condition_length": int(condition.shape[0]),
        "codebook_vocab_sizes": list(VOCABS),
        "teacher_distribution": "post_cfg_top50_logits",
        "source_metadata": {"source": "musiccaps", "source_detail": job.genre, "source_category": "genre"},
        "alignment": {
            "codes_include_priming_row": True,
            "emitted_code_row_offset": 1,
            "chunk_stitching": stitching([chunk.shape[0] for chunk in chunks], frames),
        },
    }
    manifest = {"schema_version": 1, "model": "MiniMaxAI/MiniMax-Music3", "jobs": [entry_job]}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", json.dumps(manifest, indent=1))
        handle.writestr(audio_file, flac.getvalue())
        handle.writestr(tensor_file, tensors)
    return archive.getvalue(), {"manifest": manifest, "dataset_split": "ood"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="generate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--captions", type=Path, default=None, help="default: pinned MusicCaps release")
    parser.add_argument("--poems", type=Path, default=None, help="default: pinned public domain poems")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    shard = Shard.from_env()
    if shard.rank == 0:
        write_provenance(args.out, args)
    jobs = make_jobs(args.captions or musiccaps(), args.poems or poems())[: args.limit or None]
    minimax = MiniMax.load(device=args.device, render=True)
    scheduler = load_scheduler()
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    (args.out / "indexes").mkdir(parents=True, exist_ok=True)
    index = args.out / "indexes" / f"ood-{shard.rank}.jsonl"
    for number, job in enumerate(shard.take(jobs)):
        path = args.out / "data" / f"{job.id}.zip"
        if path.is_file():
            continue
        payload, entry = generate(minimax, job, scheduler)
        entry |= {"shard_id": int(job.id[:8], 16), "path": f"data/{job.id}.zip"}
        path.with_suffix(".tmp").write_bytes(payload)
        path.with_suffix(".tmp").rename(path)
        with index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        log.info("rank %d: %d %s %s", shard.rank, number + 1, job.genre, job.id)


if __name__ == "__main__":
    main()
