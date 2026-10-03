from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Self

import torch
from torch import Tensor

from rvq_ae.constants import (
    CHUNK_FRAMES,
    CHUNK_HOP,
    OWNED_FROM,
    RATE_DEN,
    RATE_NUM,
    STITCH_HOP,
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True, slots=True)
class Chunk:
    """One rollout chunk: its frame span and the stitched latent span it produced."""

    index: int
    frame_start: int
    frame_end: int
    latent_start: int
    latent_end: int

    @classmethod
    def parse(cls, entry: Mapping[str, Any]) -> Self:
        """Build from a chunk_stitching entry of the dataset index."""
        return cls(
            index=int(entry["chunk_index"]),
            frame_start=int(entry["semantic_frame_start"]),
            frame_end=int(entry["semantic_frame_end_exclusive"]),
            latent_start=int(entry["stitched_flow_latent_start"]),
            latent_end=int(entry["stitched_flow_latent_end_exclusive"]),
        )


def nominal_bounds(frames: int) -> list[int]:
    """Reconstruct the stitched timeline for a track without chunk metadata.

    Chunk c starts at latent 345 c. Frame f belongs to chunk c = clamp((f - 25) // 100, 0, W - 1)
    with W = max(1, (n - 1) // 100) chunks. Inside chunk c with cf frames and cl = floor(cf 441 / 128)
    latents, local frame lf maps to latent ceil(lf cl / cf). The formula matches the generator for
    full chunks (100 frames -> 345 latents) and interpolates the shorter last chunk.
    """
    windows = max(1, (frames - 1) // CHUNK_HOP)
    bounds: list[int] = []
    for frame in range(frames + 1):
        chunk = min(max((frame - OWNED_FROM) // CHUNK_HOP, 0), windows - 1)
        local = frame - chunk * CHUNK_HOP
        chunk_frames = min(CHUNK_FRAMES, frames - chunk * CHUNK_HOP)
        chunk_latents = chunk_frames * RATE_NUM // RATE_DEN
        bounds.append(chunk * STITCH_HOP + ceil_div(local * chunk_latents, chunk_frames))
    return bounds


def stitched_bounds(frames: int, chunks: Sequence[Chunk]) -> list[int]:
    """Exact timeline from the generator's chunk metadata.

    Chunk 0 owns frames [f_0, f_1 + 25); chunk i > 0 owns [f_i + 25, f_(i+1) + 25); the last chunk
    owns through its own end, clamped to n. Owned frames map linearly (integer ceil) onto the kept
    latents of that chunk. A later chunk rewrites the boundary it shares with the previous one.
    """
    ordered = sorted(chunks, key=lambda chunk: chunk.index)
    bounds = [0] * (frames + 1)
    for position, chunk in enumerate(ordered):
        owner_start = chunk.frame_start if position == 0 else chunk.frame_start + OWNED_FROM
        if position + 1 < len(ordered):
            owner_end = ordered[position + 1].frame_start + OWNED_FROM
        else:
            owner_end = chunk.frame_end
        owner_start = min(owner_start, frames)
        owner_end = min(owner_end, frames)
        owned = owner_end - owner_start
        kept = chunk.latent_end - chunk.latent_start
        if owned <= 0 or kept <= 0:
            continue
        for frame in range(owner_start, owner_end + 1):
            bounds[frame] = chunk.latent_start + ceil_div((frame - owner_start) * kept, owned)
    for frame in range(frames):
        if bounds[frame + 1] < bounds[frame]:
            raise ValueError(f"chunk metadata is not monotonic at frame {frame}")
    return bounds


def stride_bounds(frames: int) -> list[int]:
    """Fixed stride timeline s_t = floor(t 441 / 128), the naive alternative to the stitched one."""
    return [frame * RATE_NUM // RATE_DEN for frame in range(frames + 1)]


TIMELINES = ("exact", "nominal", "stride")
"""Timelines an encoder can be run on: the recorded stitching, its reconstruction, a fixed stride."""


def timeline_bounds(frames: int, chunks: Sequence[Chunk] | None, timeline: str) -> list[int]:
    """Boundaries of a timeline by name; exact requires the recorded stitching table."""
    if timeline == "exact":
        if not chunks:
            raise ValueError("the exact timeline needs a chunk stitching table")
        return stitched_bounds(frames, chunks)
    if timeline == "nominal":
        return nominal_bounds(frames)
    if timeline == "stride":
        return stride_bounds(frames)
    raise ValueError(f"unknown timeline {timeline!r}; expected one of {TIMELINES}")


def frame_bounds(frames: int, chunks: Sequence[Chunk] | None) -> list[int]:
    """Boundaries s_0 <= s_1 <= ... <= s_n of the frame to latent timeline.

    MiniMax Music 3 renders a track in 200 frame chunks with a 100 frame hop and stitches the
    resulting DAV latents with a 345 latent hop, so frame t covers a variable span [s_t, s_(t+1))
    of three or four latents rather than a fixed 441 / 128 = 3.4453125 stride. The exact timeline
    is used when the dataset records the chunk stitching table, the nominal reconstruction otherwise.
    """
    if chunks:
        return stitched_bounds(frames, chunks)
    return nominal_bounds(frames)


def usable_frames(frames: int, latent_frames: int, chunks: Sequence[Chunk] | None) -> int:
    """Largest frame count whose final boundary fits inside the available latents."""
    count = frames
    while count > 1 and frame_bounds(count, chunks)[-1] > latent_frames:
        count -= 1
    return count


def pool_matrix(bounds: Sequence[int], dtype: torch.dtype = torch.float32) -> Tensor:
    """Mean pooling operator P with P[t, l] = 1 / (s_(t+1) - s_t) for l in [s_t, s_(t+1)).

    Every row sums to one, so the pooled feature of a frame is the mean of the latent features it
    was rendered from. This is the published definition of the operator; Pool.of computes the same
    result in linear time and is what the model actually runs. Latent indices are local to
    bounds[0], so the caller slices latents [bounds[0], bounds[-1]).
    """
    if len(bounds) < 2:
        raise ValueError("at least two boundaries are required")
    origin = bounds[0]
    length = bounds[-1] - origin
    pool = torch.zeros(len(bounds) - 1, length, dtype=dtype)
    for frame in range(len(bounds) - 1):
        start = bounds[frame] - origin
        end = bounds[frame + 1] - origin
        if end <= start:
            raise ValueError(f"frame {frame} has an empty latent span [{start}, {end})")
        pool[frame, start:end] = 1.0 / (end - start)
    return pool


class Pool(NamedTuple):
    """Segment form of the pooling operator P: which frame owns each latent, and how many it owns.

    Applying P as a dense [T, L] matrix product costs O(T L d); because every row of P is a
    contiguous constant block, the same result is a segment mean that costs O(L d). frame holds the
    owning frame of every latent and span holds s_(t+1) - s_t, so

        sum[b, frame[b, l]] += hidden[b, l];   pooled = sum / span.

    Right padded latents carry frame T, a sink row that is discarded, which is what lets collate
    pad a batch with zeros without disturbing any frame. A NamedTuple of tensors survives DataLoader
    collation, pin_memory and torch.compile unchanged.
    """

    frame: Tensor
    span: Tensor

    @classmethod
    def of(cls, bounds: Sequence[int], length: int | None = None) -> Self:
        """Build from the n + 1 boundaries of one window, padded out to length latents."""
        if len(bounds) < 2:
            raise ValueError("at least two boundaries are required")
        frames = len(bounds) - 1
        origin = bounds[0]
        span = torch.tensor(bounds[1:], dtype=torch.float32) - torch.tensor(bounds[:-1], dtype=torch.float32)
        if bool((span <= 0).any()):
            empty = int((span <= 0).nonzero()[0])
            raise ValueError(f"frame {empty} has an empty latent span")
        covered = bounds[-1] - origin
        total = covered if length is None else length
        if total < covered:
            raise ValueError(f"length {total} is shorter than the {covered} latents the frames cover")
        frame = torch.full((total,), frames, dtype=torch.int64)
        frame[:covered] = torch.repeat_interleave(
            torch.arange(frames, dtype=torch.int64), span.to(torch.int64)
        )
        return cls(frame=frame, span=span)

    def batched(self) -> Self:
        """Add a leading batch dimension so a single window can be applied to a [1, L, d] tensor."""
        return type(self)(frame=self.frame.unsqueeze(0), span=self.span.unsqueeze(0))

    def to(self, device: torch.device) -> Self:
        return type(self)(
            frame=self.frame.to(device, non_blocking=True), span=self.span.to(device, non_blocking=True)
        )

    def apply(self, hidden: Tensor) -> Tensor:
        """Pooled frame features [B, T, d] from latent features [B, L, d]."""
        batch, dim = hidden.shape[0], hidden.shape[-1]
        frames = self.span.shape[-1]
        sums = hidden.new_zeros(batch, frames + 1, dim)
        sums.scatter_add_(1, self.frame.unsqueeze(-1).expand(-1, -1, dim), hidden)
        return sums[:, :frames] / self.span.unsqueeze(-1)
