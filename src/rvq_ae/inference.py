import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from rvq_ae.audio.dav import DavEncoder, load_dav
from rvq_ae.audio.io import resample
from rvq_ae.audio.timeline import Pool, nominal_bounds, usable_frames
from rvq_ae.constants import DAV_REPO, FRAME_RATE, MAX_FRAMES, MODEL_REPO, RATE_DEN, RATE_NUM, WINDOW
from rvq_ae.hub import load_encoder
from rvq_ae.models.encoder import RvqEncoder


@dataclass(slots=True)
class EncodeResult:
    codes: Tensor
    """[frames, codebooks] int64 greedy codes."""
    confidence: Tensor
    """[frames, codebooks] softmax probability of the chosen code."""
    candidates: Tensor | None
    """[frames, codebooks, k] most likely codes per codebook, when requested."""
    frames: int
    latent_frames: int

    @property
    def semantic_topk(self) -> Tensor | None:
        return None if self.candidates is None else self.candidates[:, 0]

    def to_dict(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "frame_rate": FRAME_RATE,
            "duration": self.frames / FRAME_RATE,
            "latent_frames": self.latent_frames,
            "codes": self.codes.tolist(),
            "confidence": self.confidence.tolist(),
            "semantic_topk": None if self.semantic_topk is None else self.semantic_topk.tolist(),
        }


def windows(frames: int, size: int = WINDOW) -> list[tuple[int, int]]:
    """Non overlapping [start, end) frame windows; the last one may be shorter."""
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]


def frame_count(samples: int, sample_rate: int, latent_frames: int) -> int:
    """Frames of a track: round(duration * 25), capped, then shrunk to the available latents."""
    frames = min(max(1, round(samples / sample_rate * FRAME_RATE)), MAX_FRAMES)
    return usable_frames(frames, latent_frames, None)


def frames_for_latents(latent_frames: int) -> int:
    """Largest frame count whose nominal timeline fits inside latent_frames latents."""
    frames = min(max(1, math.ceil(latent_frames * RATE_DEN / RATE_NUM)), MAX_FRAMES)
    return usable_frames(frames, latent_frames, None)


class CodeEncoder:
    """Waveform to RVQ codes: DAV encoder, windowed pooling and greedy decoding."""

    def __init__(self, dav: DavEncoder, model: RvqEncoder, *, device: torch.device) -> None:
        self.dav = dav.to(device).eval()
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(
        cls,
        model: str = MODEL_REPO,
        *,
        variant: str | None = None,
        subfolder: str | None = None,
        dav: str = DAV_REPO,
        device: str | torch.device = "cuda",
        revision: str | None = None,
    ) -> Self:
        target = torch.device(device)
        return cls(
            load_dav(dav, device=target),
            load_encoder(model, variant=variant, subfolder=subfolder, revision=revision, device=target),
            device=target,
        )

    @property
    def window(self) -> int:
        return int(self.model.position.shape[1])

    @torch.no_grad()
    def latents(self, audio: Tensor, sample_rate: int) -> Tensor:
        """DAV posterior means [latent_frames, 128] for audio [channels, samples]."""
        audio = resample(audio.float(), sample_rate, self.dav.sample_rate)
        return self.dav.encode(audio.to(self.device))[0].transpose(0, 1).contiguous()

    @torch.no_grad()
    def encode_latents(
        self,
        latents: Tensor,
        frames: int | None = None,
        *,
        topk: int = 0,
        bounds: Sequence[int] | None = None,
    ) -> EncodeResult:
        """Greedy codes for latents [latent_frames, 128].

        frames defaults to what the latents cover on the nominal timeline. bounds overrides the
        timeline with n + 1 explicit boundaries, for example the recorded stitching of a generated
        track; it must fit inside the latents.
        """
        latents = latents.to(self.device, torch.float32)
        if bounds is None:
            frames = (
                frames_for_latents(latents.shape[0])
                if frames is None
                else usable_frames(frames, latents.shape[0], None)
            )
            bounds = nominal_bounds(frames)
        else:
            frames = len(bounds) - 1
            if bounds[-1] > latents.shape[0]:
                raise ValueError(f"the timeline needs {bounds[-1]} latents, {latents.shape[0]} available")
        codes: list[Tensor] = []
        confidence: list[Tensor] = []
        candidates: list[Tensor] = []
        for start, end in windows(frames, self.window):
            pool = Pool.of(bounds[start : end + 1]).batched().to(self.device)
            window = latents[bounds[start] : bounds[end]]
            logits = self.model(window[None], pool)
            probs = [torch.softmax(scores[0].float(), dim=-1) for scores in logits]
            best = [prob.max(dim=-1) for prob in probs]
            codes.append(torch.stack([item.indices for item in best], dim=-1))
            confidence.append(torch.stack([item.values for item in best], dim=-1))
            if topk > 0:
                candidates.append(torch.stack([prob.topk(topk, dim=-1).indices for prob in probs], dim=1))
        return EncodeResult(
            codes=torch.cat(codes).cpu(),
            confidence=torch.cat(confidence).cpu(),
            candidates=torch.cat(candidates).cpu() if topk > 0 else None,
            frames=frames,
            latent_frames=int(latents.shape[0]),
        )

    @torch.no_grad()
    def encode(self, audio: Tensor, sample_rate: int, *, topk: int = 0) -> EncodeResult:
        """Codes for a waveform [channels, samples] at any sample rate."""
        latents = self.latents(audio, sample_rate)
        samples = round(audio.shape[-1] / sample_rate * self.dav.sample_rate)
        frames = frame_count(samples, self.dav.sample_rate, latents.shape[0])
        return self.encode_latents(latents, frames, topk=topk)
