import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn

from rvq_ae.constants import DAV_REPO, HOP, LATENT_CHANNELS, SAMPLE_RATE

WEIGHTS_FILE = "audio_vae/diffusion_pytorch_model.safetensors"
CONFIG_FILE = "audio_vae/config.json"


class Snake(nn.Module):
    """x + sin^2(alpha x) / alpha with a learned per channel frequency alpha.

    Ziyin et al., Neural Networks Fail to Learn Periodic Functions and How to Fix It (2020).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class Stack(nn.Module):
    """Sequential container named block so state dict keys match the released checkpoint."""

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.block = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResUnit(Stack):
    """x + conv1(snake(conv7(snake(x)))) with a dilated 7 tap conv; length is preserved."""

    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__(
            Snake(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=3 * dilation),
            Snake(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


def enc_block(dim: int, stride: int) -> Stack:
    """Three residual units at dim / 2, then a strided conv doubling the channels."""
    return Stack(
        ResUnit(dim // 2, 1),
        ResUnit(dim // 2, 3),
        ResUnit(dim // 2, 9),
        Snake(dim // 2),
        nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
    )


class DavEncoder(nn.Module):
    """Encoder half of the MiniMax Music 3 DAV autoencoder, frozen input stage of this work.

    The block layout follows the Descript Audio Codec (Kumar et al., High Fidelity Audio
    Compression with Improved RVQGAN, 2023). Only the encoder and the posterior mean projection are
    kept: mono audio at 44.1 kHz maps to 1024 channels at hop 512, the mean projection reduces those
    to 64, and stereo is encoded channel by channel and concatenated into the 128 channel latent,
    left channels first.
    """

    def __init__(
        self,
        dim: int = 64,
        rates: tuple[int, ...] = (2, 4, 8, 8),
        latent_dim: int = 1024,
        out: int = LATENT_CHANNELS // 2,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self.hop = math.prod(rates)
        self.sample_rate = sample_rate
        layers: list[nn.Module] = [nn.Conv1d(1, dim, kernel_size=7, padding=3)]
        width = dim
        for stride in rates:
            width *= 2
            layers.append(enc_block(width, stride))
        layers.extend((Snake(width), nn.Conv1d(width, latent_dim, kernel_size=3, padding=1)))
        self.encoder = Stack(*layers)
        self.mean_proj = nn.Conv1d(latent_dim, out, kernel_size=1)

    @property
    def latent_channels(self) -> int:
        return 2 * self.mean_proj.out_channels

    def prepare(self, audio: Tensor) -> Tensor:
        """Any of [S], [C, S], [B, C, S] to stereo [B, 2, S] with S padded to a multiple of the hop."""
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[None]
        if audio.ndim != 3:
            raise ValueError("audio must be [batch, channels, samples]")
        if audio.shape[1] == 1:
            audio = audio.expand(-1, 2, -1)
        elif audio.shape[1] != 2:
            raise ValueError("audio must be mono or stereo")
        remainder = audio.shape[-1] % self.hop
        if remainder:
            audio = F.pad(audio, (0, self.hop - remainder))
        return audio

    def forward(self, audio: Tensor) -> Tensor:
        """Posterior means [batch, 128, samples / hop]."""
        audio = self.prepare(audio)
        batch = audio.shape[0]
        hidden = self.encoder(audio.reshape(batch * 2, 1, -1))
        return self.mean_proj(hidden).reshape(batch, self.latent_channels, -1)

    @torch.no_grad()
    def encode(self, audio: Tensor, *, segment_seconds: float = 30.0, overlap_seconds: float = 1.0) -> Tensor:
        """Posterior means of a track of any length, computed in overlapping segments.

        Memory of a single pass grows with track length (about 1 GB per 6 s of stereo at 44.1 kHz),
        so the waveform is cut into segments whose boundaries fall on latent frames; each segment is
        encoded with a margin on both sides and only its centre is kept. The receptive field of the
        released encoder is about 25 thousand samples (0.56 s), so the one second margin reproduces
        the single pass output up to float rounding.
        """
        audio = self.prepare(audio)
        samples = audio.shape[-1]
        segment = max(self.hop, int(segment_seconds * self.sample_rate) // self.hop * self.hop)
        margin = max(self.hop, int(overlap_seconds * self.sample_rate) // self.hop * self.hop)
        if samples <= segment + 2 * margin:
            return self.forward(audio)
        pieces: list[Tensor] = []
        for start in range(0, samples, segment):
            end = min(start + segment, samples)
            lo = max(start - margin, 0)
            hi = min(end + margin, samples)
            latents = self.forward(audio[:, :, lo:hi])
            keep = slice(
                (start - lo) // self.hop, (start - lo) // self.hop + (end - start + self.hop - 1) // self.hop
            )
            pieces.append(latents[:, :, keep])
        return torch.cat(pieces, dim=-1)


def fold_weight_norm(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Replace legacy weight_g / weight_v pairs by the effective weight g * v / ||v||.

    The norm runs over every dimension except the output channel (weight_norm dim 0).
    """
    folded: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.endswith(".weight_v"):
            stem = key.removesuffix(".weight_v")
            gain = state[stem + ".weight_g"]
            norm = value.norm(dim=tuple(range(1, value.ndim)), keepdim=True)
            folded[stem + ".weight"] = value * (gain / norm)
        elif not key.endswith(".weight_g"):
            folded[key] = value
    return folded


def encoder_keys(keys: list[str]) -> list[str]:
    return [key for key in keys if key.startswith(("encoder.", "mean_proj."))]


def read_weights(path: Path) -> dict[str, Tensor]:
    """Read the encoder tensors of a safetensors file or a legacy dav.pth checkpoint."""
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key) for key in encoder_keys(list(handle.keys()))}
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key: state[key] for key in encoder_keys(list(state))}


def locate(source: str | Path, *, revision: str | None) -> tuple[Path, Path | None]:
    """Weights and optional config for a local file, a local diffusers folder or a Hub repository."""
    path = Path(source)
    if path.is_file():
        return path, None
    if path.is_dir():
        weights = path / WEIGHTS_FILE
        if not weights.is_file():
            raise FileNotFoundError(f"{weights} not found")
        config = path / CONFIG_FILE
        return weights, config if config.is_file() else None
    weights = Path(hf_hub_download(str(source), WEIGHTS_FILE, revision=revision))
    config = Path(hf_hub_download(str(source), CONFIG_FILE, revision=revision))
    return weights, config


def from_config(values: Mapping[str, Any]) -> DavEncoder:
    """Build from the audio_vae/config.json of the diffusers layout."""
    return DavEncoder(
        dim=int(values["encoder_dim"]),
        rates=tuple(int(rate) for rate in values["encoder_rates"]),
        latent_dim=int(values["encoder_latent_dim"]),
        out=int(values["channel_latent_channels"]),
        sample_rate=int(values["sampling_rate"]),
    )


def load_dav(
    source: str | Path = DAV_REPO,
    *,
    revision: str | None = None,
    device: str | torch.device = "cpu",
) -> DavEncoder:
    """Frozen encoder in float32 from the Hub, a diffusers folder or a dav.pth file."""
    weights, config = locate(source, revision=revision)
    model = DavEncoder() if config is None else from_config(json.loads(config.read_text(encoding="utf-8")))
    model.load_state_dict(fold_weight_norm(read_weights(weights)), strict=True)
    model.requires_grad_(False)
    return model.eval().to(device)


def latent_frames(samples: int) -> int:
    return math.ceil(samples / HOP)
