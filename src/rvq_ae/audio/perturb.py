"""Label preserving degradations of a waveform: the recording and distribution channel, not the music.

Each perturbation keeps the musical content (and therefore the generator's true codes) fixed while
changing what a recording of it would sound like: additive noise, lossy coding, room reverberation,
band limiting, resampling and downmixing. Pitch shifting and time stretching are excluded on
purpose; they change the content, so the true codes would no longer be the target.

perturb(audio, rate, name, strength) is deterministic in (name, strength, seed); random_chain draws
a random combination for training augmentation.
"""

import subprocess
from collections.abc import Callable

import numpy as np
import torch
from scipy import fft, signal
from torch import Tensor

from rvq_ae.audio.io import resample

Perturbation = Callable[[Tensor, int, float, np.random.Generator], Tensor]


def add_noise(audio: Tensor, rate: int, snr_db: float, rng: np.random.Generator) -> Tensor:
    """Pink noise at the given signal to noise ratio in dB."""
    length = fft.next_fast_len(audio.shape[-1], real=True)
    spectrum = fft.rfft(rng.standard_normal((audio.shape[0], length)), axis=-1, workers=4)
    spectrum /= np.sqrt(np.maximum(np.arange(spectrum.shape[-1]), 1))
    pink = fft.irfft(spectrum, n=length, axis=-1, workers=4)[:, : audio.shape[-1]]
    pink = torch.from_numpy(pink).float()
    power = audio.pow(2).mean()
    scale = torch.sqrt(power / (pink.pow(2).mean() * 10 ** (snr_db / 10) + 1e-12))
    return audio + scale * pink


def mp3(audio: Tensor, rate: int, kbps: float, rng: np.random.Generator) -> Tensor:
    """MP3 encode and decode with ffmpeg (LAME, constant bitrate)."""
    channels = audio.shape[0]
    pcm = (audio.clamp(-1, 1).T.numpy() * 32767).astype("<i2").tobytes()
    fmt = ["-f", "s16le", "-ar", str(rate), "-ac", str(channels)]
    encoded = subprocess.run(
        ["ffmpeg", "-v", "error", *fmt, "-i", "pipe:0", "-b:a", f"{int(kbps)}k", "-f", "mp3", "pipe:1"],
        input=pcm,
        capture_output=True,
        check=True,
    ).stdout
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", *fmt, "pipe:1"],
        input=encoded,
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(decoded, dtype="<i2").reshape(-1, channels).T.astype(np.float32) / 32767
    return torch.from_numpy(align(samples, audio.numpy()).copy())


def align(degraded: np.ndarray, reference: np.ndarray, max_lag: int = 4096) -> np.ndarray:
    """Remove the codec delay: shift degraded left by the lag that maximises correlation with reference."""
    probe = min(reference.shape[-1], 1 << 18)
    x, y = reference.mean(0)[:probe], degraded.mean(0)[: probe + max_lag]
    scores = signal.correlate(y, x, mode="valid", method="fft")[: max_lag + 1]
    return degraded[:, int(np.argmax(scores)) :]


def reverb(audio: Tensor, rate: int, rt60: float, rng: np.random.Generator) -> Tensor:
    """Convolution with a synthetic stereo room response of the given RT60, half wet."""
    length = int(rt60 * rate)
    time = np.arange(length) / rate
    decay = np.exp(-6.9 * time / rt60)
    responses = rng.standard_normal((audio.shape[0], length)) * decay
    responses[:, 0] = 0.0
    responses /= np.sqrt((responses**2).sum(axis=-1, keepdims=True)) + 1e-12
    wet = signal.fftconvolve(audio.numpy(), responses, axes=-1)[:, : audio.shape[-1]]
    mixed = 0.5 * audio.numpy() + 0.5 * wet * (np.abs(audio.numpy()).max() / (np.abs(wet).max() + 1e-12))
    return torch.from_numpy(mixed.astype(np.float32))


def lowpass(audio: Tensor, rate: int, cutoff: float, rng: np.random.Generator) -> Tensor:
    sos = signal.butter(8, cutoff, btype="lowpass", fs=rate, output="sos")
    return torch.from_numpy(signal.sosfiltfilt(sos, audio.numpy(), axis=-1).astype(np.float32))


def highpass(audio: Tensor, rate: int, cutoff: float, rng: np.random.Generator) -> Tensor:
    sos = signal.butter(4, cutoff, btype="highpass", fs=rate, output="sos")
    return torch.from_numpy(signal.sosfiltfilt(sos, audio.numpy(), axis=-1).astype(np.float32))


def resampled(audio: Tensor, rate: int, target: float, rng: np.random.Generator) -> Tensor:
    """Down to target Hz and back: the band limit of a telephone, broadcast or low rate file."""
    return resample(resample(audio, rate, int(target)), int(target), rate)


def downmix(audio: Tensor, rate: int, _: float, rng: np.random.Generator) -> Tensor:
    return audio.mean(dim=0, keepdim=True).expand_as(audio).clone()


PERTURBATIONS: dict[str, Perturbation] = {
    "noise": add_noise,
    "mp3": mp3,
    "reverb": reverb,
    "lowpass": lowpass,
    "highpass": highpass,
    "resample": resampled,
    "mono": downmix,
}

STRENGTHS: dict[str, tuple[float, ...]] = {
    "noise": (30.0, 20.0, 10.0),
    "mp3": (128.0, 64.0, 32.0),
    "reverb": (0.3, 0.8, 1.5),
    "lowpass": (8000.0, 4000.0, 2000.0),
    "highpass": (100.0, 300.0, 800.0),
    "resample": (22050.0, 16000.0, 8000.0),
    "mono": (0.0,),
}
"""Benchmark grid, mild to severe for every perturbation."""


def same_length(audio: Tensor, length: int) -> Tensor:
    """Trim or zero pad the last axis, so a perturbed track keeps the timeline of the original."""
    if audio.shape[-1] >= length:
        return audio[..., :length]
    return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))


def perturb(audio: Tensor, rate: int, name: str, strength: float, *, seed: int = 0) -> Tensor:
    """One named perturbation at one strength, deterministic in seed."""
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; expected one of {sorted(PERTURBATIONS)}")
    out = PERTURBATIONS[name](audio.float(), rate, strength, np.random.default_rng(seed))
    return same_length(out, audio.shape[-1]).clamp(-1, 1)


def random_chain(audio: Tensor, rate: int, rng: np.random.Generator) -> Tensor:
    """Training augmentation: one to three perturbations with strengths drawn from wide ranges."""
    draws: dict[str, Callable[[], float]] = {
        "noise": lambda: float(rng.uniform(10, 40)),
        "mp3": lambda: float(rng.choice([32, 48, 64, 96, 128, 192])),
        "reverb": lambda: float(rng.uniform(0.2, 1.5)),
        "lowpass": lambda: float(rng.uniform(3000, 16000)),
        "highpass": lambda: float(rng.uniform(40, 400)),
        "resample": lambda: float(rng.choice([16000, 22050, 32000])),
        "mono": lambda: 0.0,
    }
    names = rng.choice(sorted(draws), size=int(rng.integers(1, 4)), replace=False)
    length = audio.shape[-1]
    for name in names:
        audio = same_length(PERTURBATIONS[str(name)](audio.float(), rate, draws[str(name)](), rng), length)
    gain = 10 ** (rng.uniform(-6, 3) / 20)
    return (audio * gain).clamp(-1, 1)
