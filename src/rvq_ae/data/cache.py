import json
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load as load_bytes
from safetensors.torch import save_file
from torch import Tensor
from tqdm.auto import tqdm

from rvq_ae.audio.dav import DavEncoder
from rvq_ae.audio.io import load_audio, resample
from rvq_ae.audio.perturb import random_chain
from rvq_ae.audio.timeline import frame_bounds
from rvq_ae.constants import CACHE_FORMAT, DATASET_REPO, dtype_name
from rvq_ae.data.records import Record


def cache_paths(root: Path, record: Record) -> tuple[Path, Path]:
    folder = root / f"{record.shard_id // 1000:05d}"
    return folder / f"{record.stem}.safetensors", folder / f"{record.stem}.json"


def views_path(root: Path, record: Record) -> Path:
    """Augmented latent views of a track, beside its cache entry: tensors view1 .. viewN."""
    return cache_paths(root, record)[0].with_suffix(".views.safetensors")


def read_meta(root: Path, record: Record) -> dict[str, Any]:
    return dict(json.loads(cache_paths(root, record)[1].read_text(encoding="utf-8")))


def cached(root: Path, record: Record, *, need_topk: bool) -> bool:
    tensors, meta = cache_paths(root, record)
    if not (tensors.is_file() and meta.is_file()):
        return False
    values = read_meta(root, record)
    return values.get("format") == CACHE_FORMAT and (values.get("has_topk", False) or not need_topk)


def shard_file(
    record: Record, *, corpus: Path | None, repo: str = DATASET_REPO, revision: str | None = None
) -> Path:
    if corpus is not None:
        path = corpus / record.shard_path
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        return path
    return Path(hf_hub_download(repo, record.shard_path, repo_type="dataset", revision=revision))


def read_shard(shard: Path, record: Record) -> tuple[Tensor, int, dict[str, Tensor]]:
    """Waveform [channels, samples], its sample rate and the prediction tensors of one shard."""
    with zipfile.ZipFile(shard) as archive:
        audio, rate = load_audio(archive.read(record.audio_file))
        tensors = load_bytes(archive.read(record.tensor_file))
    return audio, rate, tensors


def atomic_write(path: Path, write: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    write(temp)
    os.replace(temp, path)


@torch.no_grad()
def encode_record(
    record: Record,
    *,
    dav: DavEncoder,
    shard: Path,
    root: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Encode one track into the cache: a safetensors file plus a JSON sidecar.

    Audio is decoded from the shard and passed once through the frozen DAV encoder, then stored as
    [latent_frames, 128] beside the sampled codes and the teacher top k tensors, so that training
    never touches audio again. Both files are written atomically.
    """
    audio, rate, tensors = read_shard(shard, record)
    audio = resample(audio, rate, dav.sample_rate)
    latents = dav.encode(audio.to(device))[0].transpose(0, 1).to("cpu", dtype).contiguous()
    codes = tensors["codes"].to(torch.int16)
    if codes.ndim != 2 or codes.shape[1] != len(record.vocab_sizes):
        raise ValueError(
            f"{record.stem}: codes must be [rows, {len(record.vocab_sizes)}], got {tuple(codes.shape)}"
        )
    payload: dict[str, Tensor] = {"latents": latents, "codes": codes}
    has_topk = "teacher_topk_ids" in tensors
    if has_topk != ("teacher_topk_logits" in tensors):
        raise ValueError(f"{record.stem}: teacher_topk_ids and teacher_topk_logits must both be present")
    if has_topk:
        payload["teacher_topk_ids"] = tensors["teacher_topk_ids"].to(torch.int32)
        payload["teacher_topk_logits"] = tensors["teacher_topk_logits"].to(dtype)
    frames = min(record.emitted_frames, int(codes.shape[0]) - record.offset)
    meta = {
        "format": CACHE_FORMAT,
        "record": record.stem,
        "latent_frames": int(latents.shape[0]),
        "latent_channels": int(latents.shape[1]),
        "code_frames": int(codes.shape[0]),
        "has_topk": has_topk,
        "topk": int(payload["teacher_topk_ids"].shape[-1]) if has_topk else 0,
        "exact": record.exact,
        "mapped_latent_frames": frame_bounds(max(frames, 1), record.chunks)[-1],
        "dtype": dtype_name(dtype),
    }
    tensors_path, meta_path = cache_paths(root, record)
    tensors_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(tensors_path, lambda temp: save_file(payload, temp, metadata={"format": CACHE_FORMAT}))
    atomic_write(meta_path, lambda temp: temp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8"))
    return meta


@torch.no_grad()
def encode_views(
    record: Record,
    *,
    dav: DavEncoder,
    shard: Path,
    root: Path,
    views: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Encode views randomly degraded copies of a track (perturb.random_chain, seeded by the track).

    Each view has exactly the latent length of the clean track, since every degradation preserves
    the timeline, so a training window reads its view with the clean track's boundaries and codes.
    """
    audio, rate, _ = read_shard(shard, record)
    audio = resample(audio, rate, dav.sample_rate)
    clean_frames = int(read_meta(root, record)["latent_frames"])
    payload: dict[str, Tensor] = {}
    for view in range(1, views + 1):
        rng = np.random.default_rng(int.from_bytes(f"{record.stem}:{view}".encode()[-8:], "little"))
        degraded = random_chain(audio, dav.sample_rate, rng)
        latents = dav.encode(degraded.to(device))[0].transpose(0, 1).to("cpu", dtype).contiguous()
        if latents.shape[0] != clean_frames:
            raise ValueError(
                f"{record.stem}: view {view} has {latents.shape[0]} latents, expected {clean_frames}"
            )
        payload[f"view{view}"] = latents
    path = views_path(root, record)
    atomic_write(
        path, lambda temp: save_file(payload, temp, metadata={"format": CACHE_FORMAT, "views": str(views)})
    )


def build_cache(
    records: Sequence[Record],
    *,
    root: Path,
    dav: DavEncoder,
    device: torch.device,
    corpus: Path | None = None,
    repo: str = DATASET_REPO,
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    need_topk: bool = False,
    rebuild: bool = False,
    views: int = 0,
    rank: int = 0,
    world: int = 1,
    progress: bool = True,
) -> int:
    """Encode this rank's share (index modulo world) of the records; returns how many were written.

    views > 0 also writes that many augmented latent views per record (encode_views).
    """
    todo = [record for index, record in enumerate(records) if index % world == rank]
    written = 0
    for record in tqdm(todo, desc=f"latent cache rank {rank}", disable=not progress):
        missing = rebuild or not cached(root, record, need_topk=need_topk)
        missing_views = views > 0 and (rebuild or not views_path(root, record).is_file())
        if not (missing or missing_views):
            continue
        shard = shard_file(record, corpus=corpus, repo=repo, revision=revision)
        if missing:
            encode_record(record, dav=dav, shard=shard, root=root, device=device, dtype=dtype)
        if missing_views:
            encode_views(record, dav=dav, shard=shard, root=root, views=views, device=device, dtype=dtype)
        written += 1
    return written
