from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from rvq_ae.data.dataset import Batch
from rvq_ae.models.encoder import RvqEncoder
from rvq_ae.models.losses import rvq_loss, topk_hits

TOPS = (1, 5)


@dataclass(slots=True)
class Counter:
    """Float64 accumulator that can be summed across processes with one all reduce."""

    values: Tensor

    @classmethod
    def zeros(cls, books: int) -> Self:
        # loss, ce, kl, frames, then per k: forced correct, forced total, free correct, free total (K each)
        return cls(torch.zeros(4 + len(TOPS) * 4 * books, dtype=torch.float64))

    def add_loss(self, total: Tensor, ce: Tensor, kl: Tensor, frames: int) -> None:
        self.values[0] += total.item() * frames
        self.values[1] += ce.item() * frames
        self.values[2] += kl.item() * frames
        self.values[3] += frames

    def add_hits(self, books: int, slot: int, correct: Tensor, total: Tensor, *, forced: bool) -> None:
        base = 4 + slot * 4 * books + (0 if forced else 2 * books)
        self.values[base : base + books] += correct.to(torch.float64).cpu()
        self.values[base + books : base + 2 * books] += total.to(torch.float64).cpu()

    def metrics(self, books: int, *, depth: bool) -> dict[str, float]:
        frames = max(self.values[3].item(), 1.0)
        out = {
            "loss": self.values[0].item() / frames,
            "ce": self.values[1].item() / frames,
            "kl": self.values[2].item() / frames,
        }
        for slot, k in enumerate(TOPS):
            for forced in (True, False) if depth else (True,):
                base = 4 + slot * 4 * books + (0 if forced else 2 * books)
                correct = self.values[base : base + books]
                total = self.values[base + books : base + 2 * books].clamp(min=1.0)
                prefix = "teacher_forced_" if (forced and depth) else ""
                out[f"{prefix}semantic_top{k}"] = (correct[0] / total[0]).item()
                out[f"{prefix}acoustic_top{k}"] = (correct[1:].sum() / total[1:].sum()).item()
                for book in range(books):
                    out[f"{prefix}book{book}_top{k}"] = (correct[book] / total[book]).item()
        return out


def autocast(device: torch.device, dtype: torch.dtype | None) -> torch.autocast | nullcontext[None]:
    """Autocast context for the given dtype; a dtype of None runs in full float32."""
    if dtype is None:
        return nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def unwrap(model: nn.Module) -> RvqEncoder:
    inner = getattr(model, "module", model)
    if not isinstance(inner, RvqEncoder):
        raise TypeError("expected an RvqEncoder or a wrapper exposing it as .module")
    return inner


def to_device(batch: Batch, device: torch.device) -> Batch:
    moved: Batch = {
        "latents": batch["latents"].to(device, non_blocking=True),
        "pool": batch["pool"].to(device),
        "target": batch["target"].to(device, non_blocking=True),
    }
    if "ids" in batch:
        moved["ids"] = batch["ids"].to(device, non_blocking=True)
        moved["teacher"] = batch["teacher"].to(device, non_blocking=True)
    return moved


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[Batch],
    *,
    device: torch.device,
    kl_weight: float = 0.0,
    tau: float = 1.0,
    precision: torch.dtype | None = None,
    max_batches: int = 0,
    reduce: Callable[[Tensor], Tensor] | None = None,
) -> dict[str, float]:
    """Validation metrics: loss terms and per codebook top k accuracy.

    Models with a depth decoder are scored twice: teacher forced, which is the training condition,
    and free running, which is the inference condition where each acoustic codebook sees greedy
    predictions of the ones below it. reduce(tensor) sums the counter across processes when given.
    """
    encoder = unwrap(model)
    books = encoder.cfg.num_codebooks
    depth = encoder.depth_decoder is not None
    counter = Counter.zeros(books)
    was_training = model.training
    model.eval()
    for index, batch in enumerate(batches):
        if 0 < max_batches <= index:
            break
        batch = to_device(batch, device)
        with autocast(device, precision):
            forced = model(batch["latents"], batch["pool"], batch["target"])
            loss = rvq_loss(
                forced,
                batch["target"],
                ids=batch.get("ids"),
                teacher=batch.get("teacher"),
                kl_weight=kl_weight,
                tau=tau,
            )
            free = model(batch["latents"], batch["pool"]) if depth else forced
        frames = batch["target"].shape[0] * batch["target"].shape[1]
        counter.add_loss(loss.total, loss.ce, loss.kl, frames)
        for slot, k in enumerate(TOPS):
            counter.add_hits(books, slot, *topk_hits(forced, batch["target"], k), forced=True)
            if depth:
                counter.add_hits(books, slot, *topk_hits(free, batch["target"], k), forced=False)
    model.train(was_training)
    if reduce is not None:
        counter.values = reduce(counter.values)
    return counter.metrics(books, depth=depth)
