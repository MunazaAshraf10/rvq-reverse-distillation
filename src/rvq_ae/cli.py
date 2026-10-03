import argparse
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from rvq_ae.audio.dav import load_dav
from rvq_ae.audio.io import load_audio
from rvq_ae.constants import (
    CACHE_DTYPES,
    DATASET_REPO,
    DAV_REPO,
    MODEL_REPO,
    PRECISIONS,
    cache_dtype,
    precision_dtype,
)
from rvq_ae.data.cache import build_cache
from rvq_ae.data.dataset import WindowDataset, collate
from rvq_ae.data.records import Record, load_records
from rvq_ae.data.splits import RULES, make_split
from rvq_ae.hub import STATE_NAME, load_encoder, push_folder
from rvq_ae.inference import CodeEncoder
from rvq_ae.training.evaluate import evaluate
from rvq_ae.training.trainer import TrainConfig, train

log = logging.getLogger("rvq_ae")


def names(table: Mapping[str, object]) -> str:
    """Argparse metavar listing the accepted names of a dtype table."""
    return "{" + ",".join(table) + "}"


def device_arg(parser: argparse.ArgumentParser, default: str = "cuda") -> None:
    parser.add_argument("--device", default=default, help="torch device, for example cuda, cuda:1 or cpu")


def dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", default=DATASET_REPO, help="Hub dataset id, local index folder or JSONL file"
    )
    parser.add_argument("--corpus", type=Path, default=None, help="local folder holding the data/ shard ZIPs")
    parser.add_argument("--revision", default=None, help="dataset revision")
    parser.add_argument("--index-limit", type=int, default=0, help="read only the first N index files")
    parser.add_argument("--cache", type=Path, default=Path("cache/latents"), help="latent cache folder")


def model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=MODEL_REPO, help="Hub repository or local folder with the encoder")
    parser.add_argument(
        "--variant", default=None, help="preliminary checkpoint v1, v2, v3 or v4 of the collection"
    )
    parser.add_argument("--subfolder", default=None, help="checkpoint subfolder such as augmented or final")
    parser.add_argument("--dav", default=DAV_REPO, help="DAV encoder repository, folder or dav.pth file")
    parser.add_argument("--model-revision", default=None, help="encoder repository revision")


def cmd_cache(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dav = load_dav(args.dav, device=device)
    grouped = load_records(args.dataset, revision=args.revision, limit=args.index_limit)
    splits = args.split or sorted(grouped)
    records = [record for split in splits for record in grouped.get(split, [])]
    if args.max_records > 0:
        records = records[: args.max_records]
    written = build_cache(
        records,
        root=args.cache,
        dav=dav,
        device=device,
        corpus=args.corpus,
        repo=args.dataset,
        revision=args.revision,
        dtype=args.dtype,
        need_topk=not args.no_topk,
        rebuild=args.rebuild,
        views=args.views,
        rank=args.rank,
        world=args.world,
    )
    log.info("wrote %d of %d records to %s", written, len(records), args.cache)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig.load(
        args.config,
        output=args.output,
        resume=args.resume,
        device=args.device,
        max_steps=args.max_steps,
        seed=args.seed,
        split_rule=args.split_rule,
        validation_split=args.validation_split,
    )
    train(cfg)


def checkpoints(run: Path) -> list[tuple[str, int, Path]]:
    """(name, step, folder) for checkpoint-N, best/checkpoint-N and final, ordered by step."""

    found: list[tuple[str, int, Path]] = []
    for folder in list(run.glob("checkpoint-*")) + list(run.glob("best/checkpoint-*")) + [run / "final"]:
        if not (folder / STATE_NAME).is_file():
            continue
        step = int(json.loads((folder / STATE_NAME).read_text(encoding="utf-8"))["step"])
        found.append((str(folder.relative_to(run)), step, folder))
    return sorted(found, key=lambda item: (item[1], item[0]))


def records_of(args: argparse.Namespace) -> list[Record]:
    """The records of --split under --split-rule, truncated to --max-records."""
    grouped = load_records(args.dataset, revision=args.revision, limit=args.index_limit)
    parts = make_split(grouped, args.split_rule).parts
    if args.split not in parts:
        raise SystemExit(f"split {args.split!r} not in rule {args.split_rule!r}: {sorted(parts)}")
    records = parts[args.split]
    return records[: args.max_records] if args.max_records > 0 else records


def cmd_evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    records = records_of(args)
    dataset = WindowDataset(
        records,
        args.cache,
        size=args.window,
        stride=args.window,
        exact_only=not args.all_records,
        need_topk=args.kl_weight > 0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers)
    targets: list[tuple[str, int, Path | None]] = (
        list(checkpoints(args.run)) if args.run else [(args.subfolder or args.variant or "model", 0, None)]
    )
    if args.checkpoint:
        targets = [target for target in targets if target[0] in args.checkpoint]
    if not targets:
        raise SystemExit(f"no checkpoint to evaluate under {args.run}")
    rows: list[dict[str, float | str]] = []
    for name, step, folder in targets:
        model = (
            load_encoder(folder, device=device)
            if folder is not None
            else load_encoder(
                args.model,
                variant=args.variant,
                subfolder=args.subfolder,
                revision=args.model_revision,
                device=device,
            )
        )
        metrics = evaluate(
            model,
            loader,
            device=device,
            kl_weight=args.kl_weight,
            tau=args.tau,
            precision=args.precision,
            max_batches=args.max_batches,
        )
        rows.append({"checkpoint": name, "step": step, **metrics})
        log.info("%s step %d: %s", name, step, json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    out = args.out or (args.run / "evaluation" if args.run else Path("evaluation"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s", out)


def cmd_encode(args: argparse.Namespace) -> None:
    codec = CodeEncoder.load(
        args.model,
        variant=args.variant,
        subfolder=args.subfolder,
        dav=args.dav,
        device=args.device,
        revision=args.model_revision,
    )
    audio, rate = load_audio(args.audio)
    result = codec.encode(audio, rate, topk=args.topk)
    if args.out is None:
        print(json.dumps(result.to_dict()))
    elif args.out.suffix == ".safetensors":
        tensors = {"codes": result.codes, "confidence": result.confidence}
        if result.candidates is not None:
            tensors["candidates"] = result.candidates
        save_file(tensors, args.out)
    else:
        args.out.write_text(json.dumps(result.to_dict()) + "\n", encoding="utf-8")
    log.info("%d frames (%.2f s) from %s", result.frames, result.frames / 25, args.audio)


def cmd_push(args: argparse.Namespace) -> None:
    url = push_folder(
        args.folder, args.repo, path_in_repo=args.path_in_repo, private=args.private, message=args.message
    )
    log.info("uploaded %s to %s", args.folder, url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rvq-ae", description="Open RVQ encoder for MiniMax Music 3")
    commands = parser.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("cache", help="encode dataset audio to DAV latents")
    dataset_args(cache)
    device_arg(cache)
    cache.add_argument("--dav", default=DAV_REPO)
    cache.add_argument("--split", action="append", help="split to cache (repeatable); default all")
    cache.add_argument("--max-records", type=int, default=0)
    cache.add_argument("--dtype", type=cache_dtype, default=torch.bfloat16, metavar=names(CACHE_DTYPES))
    cache.add_argument("--no-topk", action="store_true", help="accept caches without teacher top k tensors")
    cache.add_argument("--rebuild", action="store_true")
    cache.add_argument("--views", type=int, default=0, help="augmented latent views per record (training)")
    cache.add_argument(
        "--rank", type=int, default=0, help="this worker's index when caching on several machines"
    )
    cache.add_argument("--world", type=int, default=1, help="number of caching workers")
    cache.set_defaults(func=cmd_cache)

    train = commands.add_parser("train", help="train an encoder (launch with torchrun for several GPUs)")
    train.add_argument("--config", type=Path, required=True, help="training JSON, see configs/")
    train.add_argument("--output", default=None)
    train.add_argument("--resume", default=None, help="checkpoint folder to resume from")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--split-rule", default=None, help="published or genre-ood")
    train.add_argument("--validation-split", default=None, help="holdout, test-id or test-ood")
    device_arg(train)
    train.set_defaults(func=cmd_train)

    evaluate = commands.add_parser("evaluate", help="score checkpoints on a cached split")
    dataset_args(evaluate)
    model_args(evaluate)
    device_arg(evaluate)
    evaluate.add_argument(
        "--run", type=Path, default=None, help="training output folder; scores every checkpoint"
    )
    evaluate.add_argument("--split-rule", default="published", choices=RULES)
    evaluate.add_argument("--split", default="holdout", help="holdout (published) or test-id / test-ood")
    evaluate.add_argument("--checkpoint", action="append", help="only these checkpoint names, e.g. final")
    evaluate.add_argument("--max-records", type=int, default=0)
    evaluate.add_argument(
        "--all-records", action="store_true", help="include records without exact alignment"
    )
    evaluate.add_argument("--window", type=int, default=128)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--max-batches", type=int, default=0)
    evaluate.add_argument("--kl-weight", type=float, default=0.25)
    evaluate.add_argument("--tau", type=float, default=1.0)
    evaluate.add_argument("--precision", type=precision_dtype, default=None, metavar=names(PRECISIONS))
    evaluate.add_argument("--out", type=Path, default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    encode = commands.add_parser("encode", help="encode one audio file to codes")
    model_args(encode)
    device_arg(encode)
    encode.add_argument("--audio", type=Path, required=True)
    encode.add_argument(
        "--out", type=Path, default=None, help=".safetensors or .json; prints JSON when omitted"
    )
    encode.add_argument("--topk", type=int, default=0, help="also return the top k candidates per codebook")
    encode.set_defaults(func=cmd_encode)

    push = commands.add_parser("push", help="upload a checkpoint folder to the Hub")
    push.add_argument("--folder", type=Path, required=True)
    push.add_argument("--repo", required=True)
    push.add_argument("--path-in-repo", default="")
    push.add_argument("--private", action="store_true")
    push.add_argument("--message", default="Add RVQ encoder checkpoint")
    push.set_defaults(func=cmd_push)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)
