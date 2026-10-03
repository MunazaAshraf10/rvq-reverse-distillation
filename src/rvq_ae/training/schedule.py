import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

SCHEDULES = ("linear", "cosine")


def lr_lambda(step: int, *, warmup: int, total: int, floor: float) -> float:
    """Multiplier of the base learning rate at a given optimizer step.

    step / warmup during warm up, then (1 - floor) (1 - t) + floor with
    t = (step - warmup) / (total - warmup), and floor once the schedule ends. This is polynomial
    decay with power one, the schedule of the v2, v3 and v4 releases.
    """
    if warmup > 0 and step < warmup:
        return step / warmup
    if total <= warmup:
        return floor
    progress = min((step - warmup) / (total - warmup), 1.0)
    return (1.0 - floor) * (1.0 - progress) + floor


def cosine_lambda(step: int, *, half_period: int, floor: float) -> float:
    """Multiplier of the cosine schedule the v1 release was trained with.

    floor + (1 - floor) (1 + cos(pi step / P)) / 2 with half period P: no warm up, a multiplier
    of one at step zero, the floor at P, 3P, 5P, ..., and full reheating in between. With P = 500
    this reproduces the published v1 trace, whose minima fall at steps 500, 1,500, ..., 17,500.
    The source trainer stepped its scheduler once per rank per micro batch with a period scaled
    by the rank count, which is the same curve expressed in optimizer steps; its truncation of the
    absolute learning rate to 1e-9 is below any effect and is not reproduced.
    """
    if half_period <= 0:
        raise ValueError("half_period must be positive")
    return floor + (1.0 - floor) * (1.0 + math.cos(math.pi * step / half_period)) / 2.0


def make_scheduler(optimizer: Optimizer, *, schedule: str, warmup: int, total: int, floor: float) -> LambdaLR:
    """Linear warm up and decay, or the cosine schedule whose half period is the warmup value."""
    if schedule == "linear":
        return LambdaLR(optimizer, lambda step: lr_lambda(step, warmup=warmup, total=total, floor=floor))
    if schedule == "cosine":
        return LambdaLR(optimizer, lambda step: cosine_lambda(step, half_period=warmup, floor=floor))
    raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
