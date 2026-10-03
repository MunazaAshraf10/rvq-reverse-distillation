import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from rvq_ae.constants import LATENT_CHANNELS, VOCABS, WINDOW

DEFAULT_BASE_WIDTH = 128


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Encoder hyperparameters, serialised in the same JSON layout as the released checkpoints."""

    latent_channels: int = LATENT_CHANNELS
    codebook_vocab_sizes: tuple[int, ...] = VOCABS
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_position_embeddings: int = WINDOW
    conv_dilations: tuple[int, ...] = (1, 3, 9)
    mup: bool = False
    mup_base_width: int = DEFAULT_BASE_WIDTH
    mup_output_mult: float = 1.0
    mup_readout_zero_init: bool = False
    mup_attention_multiplier: float = 8.0
    depth_decoder: bool = False
    depth_decoder_dim: int = 512
    depth_decoder_layers: int = 2
    depth_decoder_heads: int = 8
    depth_decoder_ff_mult: int = 4
    depth_decoder_dropout: float = 0.1
    depth_decoder_feedback: bool = True
    """False ablates conditioning on lower codes: the decoder attends across depth positions that
    all carry only the frame context, without a causal mask or code embeddings."""

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.depth_decoder and len(self.codebook_vocab_sizes) < 2:
            raise ValueError("the depth decoder needs at least two codebooks")
        if self.mup and self.mup_base_width <= 0:
            raise ValueError("mup_base_width must be positive")

    @property
    def width_mult(self) -> float:
        """muP width multiplier m = d_model / base width; 1 for standard parametrisation."""
        return self.d_model / self.mup_base_width if self.mup else 1.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def attention_scale(self) -> float:
        """muP scales attention logits by 1 / d_head (times a constant) instead of 1 / sqrt(d_head)."""
        if self.mup:
            return self.mup_attention_multiplier / self.head_dim
        return self.head_dim**-0.5

    @property
    def num_codebooks(self) -> int:
        return len(self.codebook_vocab_sizes)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        names = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        kwargs: dict[str, Any] = dict(values)
        for name in ("codebook_vocab_sizes", "conv_dilations"):
            if name in kwargs:
                kwargs[name] = tuple(int(v) for v in kwargs[name])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        """Published field names; mup_base_width is written only when it differs from the default."""
        values = dataclasses.asdict(self)
        values["codebook_vocab_sizes"] = list(self.codebook_vocab_sizes)
        values["conv_dilations"] = list(self.conv_dilations)
        if self.mup_base_width == DEFAULT_BASE_WIDTH:
            del values["mup_base_width"]
        if not self.depth_decoder:
            for name in list(values):
                if name.startswith("depth_decoder"):
                    del values[name]
        elif self.depth_decoder_feedback:
            del values["depth_decoder_feedback"]
        return values

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
