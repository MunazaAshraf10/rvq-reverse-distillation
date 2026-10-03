import torch

SAMPLE_RATE = 44_100
"""DAV input sample rate in Hz."""

HOP = 512
"""Samples per DAV latent frame, so the latent rate is 44100 / 512 = 86.1328125 Hz."""

LATENT_CHANNELS = 128
"""Posterior mean channels: 64 per stereo channel, left first."""

FRAME_RATE = 25
"""RVQ frames per second (40 ms per frame)."""

RATE_NUM = 441
RATE_DEN = 128
"""Latents per RVQ frame as a fraction: 441 / 128 = 3.4453125 = (44100 / 512) / 25."""

CHUNK_FRAMES = 200
"""RVQ frames produced by one language model rollout chunk (8 s)."""

CHUNK_HOP = 100
"""RVQ frames between consecutive rollout chunks (4 s)."""

STITCH_HOP = 345
"""Latents between consecutive stitched chunks: floor(100 * 441 / 128)."""

OWNED_FROM = 25
"""Local frame index from which a non first chunk owns its frames (1 s warm up)."""

WINDOW = 128
"""Encoder context in RVQ frames (5.12 s)."""

MAX_FRAMES = 9_000
"""Longest track the language model handles (6 min at 25 Hz)."""

SEMANTIC_VOCAB = 16_384
ACOUSTIC_VOCAB = 1_024
VOCABS = (SEMANTIC_VOCAB,) + (ACOUSTIC_VOCAB,) * 7
"""Codebook sizes: one semantic book followed by seven acoustic books."""

SEMANTIC_EOS = SEMANTIC_VOCAB
"""End of track id emitted by the language model; outside the semantic vocabulary."""

IGNORE = -100
"""Target value that masks a frame out of every loss and metric."""

WEIGHTS_FORMAT = "rvq-ae-minimax-m3-encoder-v1"
"""Metadata tag written into exported safetensors files."""

LEGACY_WEIGHTS_FORMAT = "simpletuner-minimaxmusic-rvq-encoder-v1"
"""Metadata tag of the published SimpleTuner checkpoints, accepted on load."""

CACHE_FORMAT = "rvq-ae-minimax-m3-latent-cache-v1"
"""Metadata tag of the DAV latent cache files."""

MODEL_REPO = "Munaza10/rvq-reverse-distillation"
"""Hub repository of the encoder (at the root) and the degraded-view encoder (under augmented/)."""

COLLECTION = "SimpleTuner/open-rvq-encoder-minimax-music3"
"""Hub repository of the preliminary checkpoints v1 to v4 under encoders/, used to validate replay."""

DAV_REPO = "SimpleTuner/MiniMax-Music-3-Encoder"
"""Hub repository holding the DAV autoencoder in the audio_vae/ diffusers layout."""

DATASET_REPO = "bghira/minimax-music3-rvq-reverse-distillation"
"""Hub dataset of generated tracks with sampled codes and teacher top 50 logits."""

PRECISIONS: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
"""Autocast dtype by name; None runs the forward pass in full float32."""

CACHE_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
"""Storage dtype of the latent cache, by name."""


def precision_dtype(name: str) -> torch.dtype | None:
    """Autocast dtype for a precision name, the only place a precision arrives as text."""
    if name not in PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; expected one of {sorted(PRECISIONS)}")
    return PRECISIONS[name]


def cache_dtype(name: str) -> torch.dtype:
    """Latent cache dtype for a name, used as the argparse converter."""
    if name not in CACHE_DTYPES:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(CACHE_DTYPES)}")
    return CACHE_DTYPES[name]


def dtype_name(dtype: torch.dtype | None) -> str:
    """Inverse of precision_dtype and cache_dtype, for JSON and metadata."""
    for table in (PRECISIONS, CACHE_DTYPES):
        for name, value in table.items():
            if value is dtype:
                return name
    raise ValueError(f"no name registered for {dtype}")
