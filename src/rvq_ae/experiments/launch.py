"""Run one experiment module as one process per GPU and wait for all of them.

    python -m rvq_ae.experiments.launch --gpus 0,1,2,3 -- replay --out results/replay/x ...

Each process sees a single GPU (CUDA_VISIBLE_DEVICES) and its rank in RANK and WORLD_SIZE; the
experiment takes its share of the items from those. Logs go to <out>/log-<rank>.txt.
"""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="launch")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("experiment")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    importlib.import_module(f"rvq_ae.experiments.{args.experiment}")
    gpus = args.gpus.split(",")
    rest = [item for item in args.args if item != "--"]
    out = Path(rest[rest.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    processes = []
    for rank, gpu in enumerate(gpus):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": gpu, "RANK": str(rank), "WORLD_SIZE": str(len(gpus))}
        log = (out / f"log-{rank}.txt").open("a", encoding="utf-8")
        command = [sys.executable, "-m", f"rvq_ae.experiments.{args.experiment}", *rest]
        processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"ranks failed with exit codes {codes}")


if __name__ == "__main__":
    main()
