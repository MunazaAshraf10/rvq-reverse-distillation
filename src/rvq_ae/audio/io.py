import io
from pathlib import Path

import numpy as np
import soundfile
import soxr
import torch
from torch import Tensor


def load_audio(source: Path | bytes) -> tuple[Tensor, int]:
    """Decode a file or in memory bytes to a float32 tensor [channels, samples] and its sample rate."""
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    data, rate = soundfile.read(handle, dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(rate)


def resample(audio: Tensor, source_rate: int, target_rate: int) -> Tensor:
    """Band limited resampling (soxr, very high quality preset) of [channels, samples]."""
    if source_rate == target_rate:
        return audio
    converted = soxr.resample(audio.numpy(force=True).T, source_rate, target_rate, quality="VHQ")
    return torch.from_numpy(np.ascontiguousarray(converted.T)).to(audio.dtype)
