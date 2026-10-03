"""Reference based and reference free audio metrics for music.

Embedding similarity (CLAP music, MERT), harmony (CQT chroma, key), rhythm (tempo), spectral
distance (multi resolution log mel) and set level Frechet distance. Every function takes a waveform
[channels, samples] and its sample rate; models resample internally.

Model revisions are pinned so the scores are reproducible:
    CLAP  laion/larger_clap_music_and_speech (music_speech_audioset_epoch_15_esc_89.98). The music
          only conversion (laion/larger_clap_music) maps every caption to the same text embedding
          (pairwise cosine 0.999 under transformers 4.57 and 5.18), so it cannot score text.
    MERT  m-a-p/MERT-v1-95M, revision 12af15fe, all 12 layers mean pooled over time then averaged
"""

from dataclasses import dataclass
from typing import Self

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from librosa.beat import beat_track
from scipy import linalg
from torch import Tensor
from transformers import AutoModel, ClapModel, ClapProcessor

from rvq_ae.audio.io import resample

CLAP_REPO = "laion/larger_clap_music_and_speech"
CLAP_REVISION = "195c3a3e68faebb3e2088b9a79e79b43ddbda76b"
MERT_REPO = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
CLAP_RATE = 48_000
MERT_RATE = 24_000
ANALYSIS_RATE = 22_050
CLAP_WINDOW = 10
"""CLAP sees 10 s windows; a clip embedding is the normalised mean of its window embeddings."""

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
"""Krumhansl Kessler key profiles."""


def mono(audio: Tensor) -> Tensor:
    return audio.float().mean(dim=0) if audio.ndim == 2 else audio.float()


def to_numpy(audio: Tensor, rate: int, target: int) -> np.ndarray:
    return resample(mono(audio)[None], rate, target)[0].numpy()


class Embedder:
    """CLAP and MERT on one device, with every embedding unit normalised."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.clap = ClapModel.from_pretrained(CLAP_REPO, revision=CLAP_REVISION).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(CLAP_REPO, revision=CLAP_REVISION)
        self.mert = AutoModel.from_pretrained(MERT_REPO, revision=MERT_REVISION, trust_remote_code=True)
        self.mert = self.mert.to(device).eval()
        self.layers: list[Tensor] = []
        for layer in self.mert.encoder.layers:
            layer.register_forward_hook(self.keep)

    def keep(self, module: torch.nn.Module, inputs: object, output: Tensor | tuple[Tensor, ...]) -> None:
        self.layers.append(output[0] if isinstance(output, tuple) else output)

    @torch.no_grad()
    def clap_audio(self, audio: Tensor, rate: int) -> Tensor:
        """[512] mean of the CLAP embeddings of consecutive 10 s windows."""
        wave = to_numpy(audio, rate, CLAP_RATE)
        step = CLAP_WINDOW * CLAP_RATE
        windows = [wave[start : start + step] for start in range(0, max(len(wave) - step // 2, 1), step)]
        inputs = self.processor(audio=windows, sampling_rate=CLAP_RATE, return_tensors="pt").to(self.device)
        features = self.clap.get_audio_features(**inputs)
        features = getattr(features, "pooler_output", features)
        return F.normalize(F.normalize(features, dim=-1).mean(0), dim=0).cpu()

    @torch.no_grad()
    def clap_text(self, texts: list[str]) -> Tensor:
        """[N, 512] CLAP text embeddings."""
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        features = self.clap.get_text_features(**inputs)
        return F.normalize(getattr(features, "pooler_output", features), dim=-1).cpu()

    @torch.no_grad()
    def mert_embedding(self, audio: Tensor, rate: int) -> Tensor:
        """[768] MERT embedding: every layer mean pooled over time, then averaged over layers."""
        wave = torch.from_numpy(to_numpy(audio, rate, MERT_RATE)).to(self.device)
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        pooled: list[Tensor] = []
        for start in range(0, wave.shape[0], 30 * MERT_RATE):
            piece = wave[start : start + 30 * MERT_RATE]
            if piece.shape[0] < MERT_RATE:
                continue
            self.layers.clear()
            self.mert(piece[None])
            pooled.append(torch.stack([layer[0].mean(0) for layer in self.layers]).mean(0) * piece.shape[0])
        return F.normalize(torch.stack(pooled).sum(0), dim=0).cpu()


@dataclass(slots=True)
class Analysis:
    """Embeddings and music descriptors of one clip, computed once and compared many times."""

    clap: Tensor
    mert: Tensor
    chroma: np.ndarray
    """[12, frames] CQT chroma at 22.05 kHz, hop 512."""
    tempo: float
    key: tuple[int, str]
    mel: Tensor
    """Mono waveform at 22.05 kHz for the spectral distance."""

    @classmethod
    def of(cls, embedder: Embedder, audio: Tensor, rate: int) -> Self:
        wave = to_numpy(audio, rate, ANALYSIS_RATE)
        chroma = librosa.feature.chroma_cqt(y=wave, sr=ANALYSIS_RATE, hop_length=512)
        tempo = float(np.atleast_1d(beat_track(y=wave, sr=ANALYSIS_RATE)[0])[0])
        return cls(
            clap=embedder.clap_audio(audio, rate),
            mert=embedder.mert_embedding(audio, rate),
            chroma=chroma,
            tempo=tempo,
            key=estimate_key(chroma),
            mel=torch.from_numpy(wave),
        )


def estimate_key(chroma: np.ndarray) -> tuple[int, str]:
    """Krumhansl Schmuckler key: the tonic and mode whose profile correlates best with mean chroma."""
    profile = chroma.mean(axis=1)
    best = (-np.inf, 0, "major")
    for tonic in range(12):
        for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = float(np.corrcoef(profile, np.roll(template, tonic))[0, 1])
            if score > best[0]:
                best = (score, tonic, mode)
    return best[1], best[2]


def key_score(estimate: tuple[int, str], reference: tuple[int, str]) -> float:
    """MIREX weighted key score: same 1, perfect fifth 0.5, relative 0.3, parallel 0.2, else 0."""
    (tonic, mode), (ref_tonic, ref_mode) = estimate, reference
    if estimate == reference:
        return 1.0
    if mode == ref_mode and (tonic - ref_tonic) % 12 in (5, 7):
        return 0.5
    relative = (ref_tonic + 9) % 12 if ref_mode == "major" else (ref_tonic + 3) % 12
    if mode != ref_mode and tonic == relative:
        return 0.3
    if mode != ref_mode and tonic == ref_tonic:
        return 0.2
    return 0.0


def tempo_accuracy(estimate: float, reference: float, tolerance: float = 0.04) -> tuple[float, float]:
    """Acc1 (within 4 percent) and Acc2 (within 4 percent of the reference times 1/3, 1/2, 1, 2 or 3)."""
    if reference <= 0:
        return 0.0, 0.0
    acc1 = float(abs(estimate - reference) <= tolerance * reference)
    acc2 = float(
        any(
            abs(estimate - factor * reference) <= tolerance * factor * reference
            for factor in (1 / 3, 0.5, 1, 2, 3)
        )
    )
    return acc1, acc2


def chroma_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean framewise cosine of time aligned chroma, over the shorter clip."""
    frames = min(a.shape[1], b.shape[1])
    x, y = a[:, :frames], b[:, :frames]
    dots = (x * y).sum(0) / (np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0) + 1e-8)
    return float(dots.mean())


def mel_distance(a: Tensor, b: Tensor, rate: int = ANALYSIS_RATE) -> float:
    """Multi resolution log mel L1 distance (windows 512, 1024, 2048; 80 bands), loudness matched."""
    length = min(a.shape[0], b.shape[0])
    x, y = a[:length].numpy(), b[:length].numpy()
    y = y * (np.sqrt((x**2).mean()) / (np.sqrt((y**2).mean()) + 1e-8))
    total = 0.0
    for window in (512, 1024, 2048):
        mel_x = librosa.feature.melspectrogram(y=x, sr=rate, n_fft=window, hop_length=window // 4, n_mels=80)
        mel_y = librosa.feature.melspectrogram(y=y, sr=rate, n_fft=window, hop_length=window // 4, n_mels=80)
        total += float(np.abs(np.log10(mel_x + 1e-5) - np.log10(mel_y + 1e-5)).mean())
    return total / 3


def compare(candidate: Analysis, reference: Analysis) -> dict[str, float]:
    """Every reference based metric of a candidate clip against its reference."""
    acc1, acc2 = tempo_accuracy(candidate.tempo, reference.tempo)
    return {
        "clap": float(candidate.clap @ reference.clap),
        "mert": float(candidate.mert @ reference.mert),
        "chroma": chroma_similarity(candidate.chroma, reference.chroma),
        "key_score": key_score(candidate.key, reference.key),
        "tempo_acc1": acc1,
        "tempo_acc2": acc2,
        "mel": mel_distance(candidate.mel, reference.mel),
    }


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two embedding sets [N, d]."""
    mu_x, mu_y = x.mean(0), y.mean(0)
    sigma_x, sigma_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    root = np.real(linalg.sqrtm(sigma_x @ sigma_y))
    return float(((mu_x - mu_y) ** 2).sum() + np.trace(sigma_x + sigma_y - 2 * root))
