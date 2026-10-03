import re
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

WIDE = re.compile(
    r"^(transformer\.\d+\.(q_proj|k_proj|v_proj|out_proj|linear1|linear2)|blocks\.\d+\.conv[12])\.weight$"
)
WIDE_FAN_IN = re.compile(
    r"^(transformer\.\d+\.|blocks\.\d+\.|heads\.\d+\.|depth_decoder\.context_projection\.)"
)
NO_DECAY = re.compile(r"bias|norm|embed")


def wide(name: str) -> bool:
    """Matrix like weights: input and output dimensions both scale with d_model."""
    return WIDE.match(name) is not None


def wide_fan_in(name: str) -> bool:
    """Parameters of modules whose input dimension scales with d_model."""
    return WIDE_FAN_IN.match(name) is not None


def no_decay(name: str) -> bool:
    """Biases, normalisation gains and embeddings train without weight decay."""
    return NO_DECAY.search(name) is not None


class Readout(nn.Linear):
    """Linear map from the width dimension to a fixed size with the muP 1 / m input scale.

    Rule one of the maximal update parametrisation (Yang et al., Tensor Programs V: Tuning Large
    Neural Networks via Zero-Shot Hyperparameter Transfer, 2022): a readout computes
    y = output_mult * W (x / m) + b with width multiplier m = d_model / base width. Rule two lives
    in EncoderConfig.attention_scale and rule three in param_groups. The multiplier is a plain
    attribute rather than a buffer, so the state dict matches a standard nn.Linear exactly.
    """

    def __init__(
        self,
        dim: int,
        out: int,
        width_mult: float,
        *,
        bias: bool = True,
        zero_init: bool = False,
        output_mult: float = 1.0,
    ) -> None:
        super().__init__(dim, out, bias=bias)
        self.width_mult = float(width_mult)
        self.output_mult = float(output_mult)
        self.zero_init = zero_init
        if zero_init:
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(self.output_mult * x / self.width_mult, self.weight, self.bias)


@torch.no_grad()
def rescale_init(model: nn.Module, width_mult: float) -> None:
    """Convert a freshly initialised standard parametrisation model to muP initialisation.

    Following the reference implementation, biases fed by a width sized fan in and non zero readout
    weights and biases are multiplied by sqrt(m). Call exactly once on a new model, never after
    loading weights.
    """
    if width_mult == 1.0:
        return
    scale = width_mult**0.5
    readouts = {name for name, module in model.named_modules() if isinstance(module, Readout)}
    for name, param in model.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        if module_name in readouts:
            module = model.get_submodule(module_name)
            if not module.zero_init:
                param.mul_(scale)
        elif name.endswith(".bias") and wide_fan_in(name):
            param.mul_(scale)


def param_groups(
    named: Iterable[tuple[str, nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    width_mult: float,
) -> list[dict[str, object]]:
    """AdamW parameter groups: the decay split first, then the muP scaling of matrix like weights.

    Rule three of muP: weights whose input and output dimensions both grow with width take
    lr / m, and their weight decay is multiplied by m so that the decay applied per step is
    invariant to width. Everything else keeps the base learning rate. The rules are matched on
    parameter names, so fixed width components (the latent stem, the depth decoder) are untouched.
    """
    buckets: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, param in named:
        if not param.requires_grad:
            continue
        buckets.setdefault((no_decay(name), wide(name)), []).append(param)
    groups: list[dict[str, object]] = []
    for (decay_free, matrix_like), params in sorted(buckets.items()):
        decay = 0.0 if decay_free else weight_decay
        group_lr = lr / width_mult if matrix_like else lr
        group_decay = decay * width_mult if matrix_like else decay
        groups.append({"params": params, "lr": group_lr, "weight_decay": group_decay})
    return groups
