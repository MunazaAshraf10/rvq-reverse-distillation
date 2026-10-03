import dataclasses
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from rvq_ae.constants import DATASET_REPO, WINDOW, dtype_name, precision_dtype
from rvq_ae.data.dataset import Batch, WindowDataset, collate
from rvq_ae.data.records import Record, load_records
from rvq_ae.data.splits import make_split
from rvq_ae.hub import STATE_NAME, WEIGHTS_NAME, save_encoder
from rvq_ae.models.config import EncoderConfig
from rvq_ae.models.encoder import RvqEncoder
from rvq_ae.models.losses import rvq_loss, topk_hits
from rvq_ae.models.mup import param_groups
from rvq_ae.training.evaluate import autocast, evaluate, to_device
from rvq_ae.training.schedule import SCHEDULES, make_scheduler

log = logging.getLogger("rvq_ae.training.trainer")

TRAINING_STATE = "training_state.pt"
METRICS_FILE = "metrics.jsonl"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training hyperparameters; precision is a torch dtype, named only in the JSON layout."""

    output: str = "output/rvq-encoder"
    model: EncoderConfig = field(default_factory=EncoderConfig)
    dataset: str = DATASET_REPO
    corpus: str | None = None
    revision: str | None = None
    index_limit: int = 0
    cache: str = "cache/latents"
    split_rule: str = "published"
    train_split: str = "train"
    validation_split: str = "holdout"
    max_train_records: int = 0
    max_validation_records: int = 0
    exact_only: bool = True
    augment_views: int = 0
    """Augmented latent views per training track to sample from (rvq-ae cache --views); 0 disables."""
    window: int = WINDOW
    stride: int = WINDOW
    random_crop: bool = True
    batch_size: int = 16
    validation_batch_size: int = 0
    accumulation: int = 1
    workers: int = 4
    epochs: int = 20
    max_steps: int = 0
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    schedule: str = "linear"
    warmup: int = 500
    lr_end: float = 1e-7
    grad_clip: float = 1.0
    kl_weight: float = 0.25
    tau: float = 1.0
    precision: torch.dtype | None = torch.bfloat16
    device: str = "cuda"
    tf32: bool = True
    compile: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42
    log_every: int = 20
    validate_every: int = 500
    checkpoint_every: int = 500
    keep_checkpoints: int = 0
    """Keep only the newest N periodic checkpoints (and the newest best one); 0 keeps everything."""
    max_validation_batches: int = 0
    resume: str | None = None
    wandb_project: str | None = None
    run_name: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        names = {item.name for item in dataclasses.fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown training keys: {unknown}")
        kwargs = dict(values)
        if "model" in kwargs:
            kwargs["model"] = EncoderConfig.from_dict(kwargs["model"])
        if "betas" in kwargs:
            kwargs["betas"] = tuple(float(v) for v in kwargs["betas"])
        if "precision" in kwargs:
            kwargs["precision"] = precision_dtype(str(kwargs["precision"]))
        if kwargs.get("schedule", "linear") not in SCHEDULES:
            raise ValueError(f"unknown schedule {kwargs['schedule']!r}; expected one of {SCHEDULES}")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path, **overrides: Any) -> Self:
        values = json.loads(path.read_text(encoding="utf-8"))
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        values = dataclasses.asdict(self)
        values["model"] = self.model.to_dict()
        values["betas"] = list(self.betas)
        values["precision"] = dtype_name(self.precision)
        return values


@dataclass(slots=True)
class TrainState:
    step: int = 0
    epoch: int = 0
    batch: int = 0
    best: float = math.inf


@dataclass(frozen=True, slots=True)
class Dist:
    rank: int
    world: int
    local_rank: int
    device: torch.device

    @classmethod
    def init(cls, device: str) -> Self:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        target = torch.device(device)
        if target.type == "cuda":
            target = torch.device("cuda", local_rank)
            torch.cuda.set_device(target)
        if world > 1 and not dist.is_initialized():
            backend = "nccl" if target.type == "cuda" else "gloo"
            dist.init_process_group(backend, device_id=target if target.type == "cuda" else None)
        return cls(rank=rank, world=world, local_rank=local_rank, device=target)

    @property
    def main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier()

    def reduce(self, values: Tensor) -> Tensor:
        """Sum across processes; float64 counters travel over the device tensor path."""
        if self.world <= 1:
            return values
        moved = values.to(self.device if self.device.type == "cuda" else "cpu")
        dist.all_reduce(moved, op=dist.ReduceOp.SUM)
        return moved.cpu()

    def gather(self, item: object) -> list[object]:
        if self.world <= 1:
            return [item]
        items: list[object] = [None] * self.world
        dist.all_gather_object(items, item)
        return items

    def finish(self) -> None:
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device: torch.device) -> dict[str, Tensor]:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def set_rng_state(state: dict[str, Tensor], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def select_records(records: list[Record], limit: int) -> list[Record]:
    return records[:limit] if limit > 0 else records


def build_datasets(cfg: TrainConfig) -> tuple[WindowDataset, WindowDataset | None]:
    grouped = make_split(
        load_records(cfg.dataset, revision=cfg.revision, limit=cfg.index_limit), cfg.split_rule
    ).parts
    if cfg.train_split not in grouped:
        raise ValueError(f"split {cfg.train_split!r} not found; available: {sorted(grouped)}")
    root = Path(cfg.cache)
    need_topk = cfg.kl_weight > 0
    train = WindowDataset(
        select_records(grouped[cfg.train_split], cfg.max_train_records),
        root,
        size=cfg.window,
        stride=cfg.stride,
        random_crop=cfg.random_crop,
        exact_only=cfg.exact_only,
        need_topk=need_topk,
        seed=cfg.seed,
        views=cfg.augment_views,
    )
    validation = None
    if cfg.validation_split in grouped:
        validation = WindowDataset(
            select_records(grouped[cfg.validation_split], cfg.max_validation_records),
            root,
            size=cfg.window,
            stride=cfg.window,
            random_crop=False,
            exact_only=cfg.exact_only,
            need_topk=need_topk,
            seed=cfg.seed,
        )
    if len(train) == 0:
        raise ValueError("the training split has no usable windows; run the cache command first")
    return train, validation


def epoch_indices(sampler: DistributedSampler[int], epoch: int, skip: int) -> list[int]:
    """This rank's sample order for the epoch with the first skip samples removed (resume)."""
    sampler.set_epoch(epoch)
    return list(sampler)[skip:]


def loader(
    dataset: WindowDataset, indices: list[int], cfg: TrainConfig, batch_size: int
) -> DataLoader[Batch]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices,
        collate_fn=collate,
        num_workers=cfg.workers,
        pin_memory=cfg.device == "cuda",
        persistent_workers=False,
        drop_last=False,
    )


def validation_loader(dataset: WindowDataset, cfg: TrainConfig, world: Dist) -> DataLoader[Batch]:
    indices = list(range(world.rank, len(dataset), world.world))
    batch_size = cfg.validation_batch_size or cfg.batch_size
    return loader(dataset, indices, cfg, batch_size)


def build_model(cfg: TrainConfig, world: Dist) -> tuple[RvqEncoder, nn.Module]:
    """The raw encoder (for saving) and the module that runs the forward pass."""
    seed_everything(cfg.seed)
    raw = RvqEncoder(cfg.model)
    raw.checkpointing = cfg.gradient_checkpointing
    raw.to(world.device)
    runner: nn.Module = cast(nn.Module, torch.compile(raw)) if cfg.compile else raw
    if world.world > 1:
        device_ids = [world.local_rank] if world.device.type == "cuda" else None
        runner = DistributedDataParallel(runner, device_ids=device_ids)
    return raw, runner


def build_optimizer(cfg: TrainConfig, raw: RvqEncoder) -> torch.optim.AdamW:
    groups = param_groups(
        raw.named_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, width_mult=cfg.model.width_mult
    )
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=raw.position.is_cuda)


def total_steps(cfg: TrainConfig, train: WindowDataset, world: Dist) -> int:
    per_rank = len(train) // world.world
    micro = per_rank // cfg.batch_size
    per_epoch = max(1, math.ceil(micro / cfg.accumulation))
    return cfg.max_steps if cfg.max_steps > 0 else per_epoch * cfg.epochs


def save_state(
    folder: Path,
    *,
    raw: RvqEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainState,
    cfg: TrainConfig,
    world: Dist,
) -> None:
    rng = world.gather(rng_state(world.device))
    if world.main:
        save_encoder(raw, folder)
        trainer_state = dataclasses.asdict(state) | {"config": cfg.to_dict()}
        (folder / STATE_NAME).write_text(json.dumps(trainer_state, indent=2) + "\n", encoding="utf-8")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "world": world.world,
                "rng": rng,
            },
            folder / TRAINING_STATE,
        )
    world.barrier()


def load_state(
    folder: Path,
    *,
    raw: RvqEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: TrainConfig,
    world: Dist,
) -> TrainState:

    raw.load_state_dict(load_file(folder / WEIGHTS_NAME, device="cpu"), strict=True)
    raw.to(world.device)
    values = json.loads((folder / STATE_NAME).read_text(encoding="utf-8"))
    state = TrainState(step=values["step"], epoch=values["epoch"], batch=values["batch"], best=values["best"])
    saved = torch.load(folder / TRAINING_STATE, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    if saved["world"] == world.world:
        set_rng_state(saved["rng"][world.rank], world.device)
    else:
        log.warning("world size changed from %d to %d; reseeding", saved["world"], world.world)
        seed_everything(cfg.seed + world.rank + state.step)
    return state


class Logger:
    """Rank zero metrics sink: JSONL on disk and optionally Weights and Biases."""

    def __init__(self, cfg: TrainConfig, world: Dist) -> None:
        self.enabled = world.main
        self.path = Path(cfg.output) / METRICS_FILE
        self.wandb: Any = None
        if self.enabled and cfg.wandb_project:
            import wandb

            self.wandb = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.to_dict())

    def log(self, step: int, values: dict[str, float], *, prefix: str) -> None:
        if not self.enabled:
            return
        row = {"step": step, **{f"{prefix}/{key}": value for key, value in values.items()}}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if self.wandb is not None:
            self.wandb.log(row, step=step)
        log.info("%s", " ".join(f"{key}={value:.4g}" for key, value in row.items()))

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


@dataclass(slots=True)
class Metrics:
    """Running sums over a logging interval: loss terms, frames, semantic and acoustic top 1 hits."""

    values: Tensor = field(default_factory=lambda: torch.zeros(8, dtype=torch.float64))

    def add(
        self, total: Tensor, ce: Tensor, kl: Tensor, target: Tensor, correct: Tensor, count: Tensor
    ) -> None:
        frames = target.shape[0] * target.shape[1]
        self.values[0] += total.item() * frames
        self.values[1] += ce.item() * frames
        self.values[2] += kl.item() * frames
        self.values[3] += frames
        self.values[4] += correct[0].item()
        self.values[5] += count[0].item()
        self.values[6] += correct[1:].sum().item()
        self.values[7] += count[1:].sum().item()

    def metrics(self, reduce: Callable[[Tensor], Tensor]) -> dict[str, float]:
        values = reduce(self.values)
        frames = max(values[3].item(), 1.0)
        return {
            "loss": values[0].item() / frames,
            "ce": values[1].item() / frames,
            "kl": values[2].item() / frames,
            "semantic_top1": values[4].item() / max(values[5].item(), 1.0),
            "acoustic_top1": values[6].item() / max(values[7].item(), 1.0),
        }


def micro_steps(
    cfg: TrainConfig, train: WindowDataset, sampler: DistributedSampler[int], state: TrainState
) -> Iterator[tuple[int, Batch]]:
    """Batches from the current epoch onward, resuming mid epoch, tagged with their epoch."""
    epoch = state.epoch
    while True:
        train.set_epoch(epoch)
        skip = state.batch * cfg.batch_size if epoch == state.epoch else 0
        batches = loader(train, epoch_indices(sampler, epoch, skip), cfg, cfg.batch_size)
        for batch in batches:
            yield epoch, batch
        epoch += 1


def enable_tf32(enabled: bool) -> None:
    """Select the float32 matmul and convolution backend.

    On Ampere, tf32 trades ten mantissa bits for roughly an order of magnitude more throughput on
    the float32 paths, which here means the pooling and the readouts; ieee keeps full precision.
    """
    mode = "tf32" if enabled else "ieee"
    torch.backends.cuda.matmul.fp32_precision = mode
    torch.backends.cudnn.fp32_precision = mode


class Trainer:
    """Distributed trainer: torchrun, DistributedDataParallel, autocast and exact resume.

    Checkpoints carry the weights, the optimizer, the scheduler and the per rank random state, so a
    resumed run reproduces an uninterrupted one bitwise as long as the world size is unchanged.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.world = Dist.init(cfg.device)
        self.output = Path(cfg.output)
        if self.world.main:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "train_config.json").write_text(
                json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        logging.basicConfig(
            level=logging.INFO if self.world.main else logging.WARNING, format="%(asctime)s %(message)s"
        )
        if self.world.device.type == "cuda":
            enable_tf32(cfg.tf32)

        self.train_set, self.validation_set = build_datasets(cfg)
        self.raw, self.runner = build_model(cfg, self.world)
        self.optimizer = build_optimizer(cfg, self.raw)
        self.steps = total_steps(cfg, self.train_set, self.world)
        self.scheduler = make_scheduler(
            self.optimizer,
            schedule=cfg.schedule,
            warmup=cfg.warmup,
            total=self.steps,
            floor=cfg.lr_end / cfg.lr,
        )
        self.state = TrainState()
        if cfg.resume:
            self.state = load_state(
                Path(cfg.resume),
                raw=self.raw,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                cfg=cfg,
                world=self.world,
            )
        else:
            seed_everything(cfg.seed + self.world.rank)
        self.sampler: DistributedSampler[int] = DistributedSampler(
            self.train_set,
            num_replicas=self.world.world,
            rank=self.world.rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=True,
        )
        self.logger = Logger(cfg, self.world)
        self.metrics = Metrics()
        self.started = 0.0

    def checkpoint(self, folder: Path) -> None:
        save_state(
            folder,
            raw=self.raw,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=self.state,
            cfg=self.cfg,
            world=self.world,
        )

    def prune(self, folder: Path, *, keep: int) -> None:
        """Delete all but the newest keep checkpoint-N folders under folder (rank zero, keep > 0)."""
        if keep <= 0 or not self.world.main:
            return
        found = sorted(folder.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[1]))
        for old in found[:-keep]:
            shutil.rmtree(old)

    def validate(self) -> None:
        """Score the validation split and keep the checkpoint if it is the best loss so far."""
        if self.validation_set is None:
            return
        metrics = evaluate(
            self.runner,
            validation_loader(self.validation_set, self.cfg, self.world),
            device=self.world.device,
            kl_weight=self.cfg.kl_weight,
            tau=self.cfg.tau,
            precision=self.cfg.precision,
            max_batches=self.cfg.max_validation_batches,
            reduce=self.world.reduce,
        )
        self.logger.log(self.state.step, metrics, prefix="validation")
        if metrics["loss"] < self.state.best:
            self.state.best = metrics["loss"]
            self.checkpoint(self.output / "best" / f"checkpoint-{self.state.step}")
            self.prune(self.output / "best", keep=1)

    def accumulate(self, batch: Batch, *, last: bool) -> None:
        """One micro batch: forward, backward and the running metric window.

        Gradient synchronisation is suppressed on every micro batch but the last of an
        accumulation group, so DistributedDataParallel all reduces once per optimizer step.
        """
        cfg = self.cfg
        sync = (
            self.runner.no_sync()
            if (isinstance(self.runner, DistributedDataParallel) and not last)
            else torch.autocast("cpu", enabled=False)
        )
        with sync, autocast(self.world.device, cfg.precision):
            logits = self.runner(batch["latents"], batch["pool"], batch["target"])
            loss = rvq_loss(
                logits,
                batch["target"],
                ids=batch.get("ids"),
                teacher=batch.get("teacher"),
                kl_weight=cfg.kl_weight,
                tau=cfg.tau,
            )
        torch.autograd.backward(loss.total / cfg.accumulation)
        self.metrics.add(
            loss.total.detach(),
            loss.ce.detach(),
            loss.kl.detach(),
            batch["target"],
            *topk_hits(logits, batch["target"], 1),
        )

    def step(self) -> None:
        """Clip, step the optimizer and the schedule, then log, validate and checkpoint on cadence."""
        cfg = self.cfg
        torch.nn.utils.clip_grad_norm_(self.raw.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        if self.state.step % cfg.log_every == 0:
            values = self.metrics.metrics(self.world.reduce)
            values["lr"] = float(self.scheduler.get_last_lr()[0])
            values["epoch"] = self.state.epoch
            values["seconds"] = time.time() - self.started
            self.logger.log(self.state.step, values, prefix="train")
            self.metrics = Metrics()
        if cfg.validate_every > 0 and self.state.step % cfg.validate_every == 0:
            self.validate()
        if cfg.checkpoint_every > 0 and self.state.step % cfg.checkpoint_every == 0:
            self.checkpoint(self.output / f"checkpoint-{self.state.step}")
            self.prune(self.output, keep=cfg.keep_checkpoints)

    def run(self) -> Path:
        cfg = self.cfg
        if self.world.main:
            log.info(
                "parameters %d, windows %d, steps %d, world %d",
                self.raw.parameter_count(),
                len(self.train_set),
                self.steps,
                self.world.world,
            )
        self.runner.train()
        self.started = time.time()
        micro = 0
        for epoch, batch in micro_steps(cfg, self.train_set, self.sampler, self.state):
            if self.state.step >= self.steps:
                break
            if epoch != self.state.epoch:
                self.state.epoch, self.state.batch = epoch, 0
            last = (micro + 1) % cfg.accumulation == 0
            self.accumulate(to_device(batch, self.world.device), last=last)
            micro += 1
            self.state.batch += 1
            if last:
                self.step()
        self.validate()
        self.checkpoint(self.output / "final")
        self.logger.close()
        self.world.finish()
        return self.output


def train(cfg: TrainConfig) -> Path:
    """Run a training job to completion and return its output folder.

    Launch with torchrun --nproc_per_node N -m rvq_ae train --config configs/v4_169m.json
    """
    return Trainer(cfg).run()
