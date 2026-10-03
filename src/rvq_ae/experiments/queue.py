"""A minimal GPU job queue: one worker per GPU pulls the next shell command from a jobs file.

    python -m rvq_ae.experiments.queue jobs.txt --gpus 0,1,2,3

Each line of the jobs file is one command; blank lines and lines starting with # are skipped. Jobs
are identified by their position among the command lines, so a list may only grow at its end.

<log folder>/<index>.done marks a finished job. Before starting a job a worker atomically creates
<index>.claim holding its process id; a job whose claim belongs to a live process is skipped, and a
claim left by a dead process is taken over. A failed job keeps its claim while its queue process
lives. Several queue processes, started at different times on different GPUs, can therefore work
through one list together, and rerunning a queue after they exit retries exactly the failed jobs.
Every job runs with CUDA_VISIBLE_DEVICES set to its worker's GPU; its output is appended to
<log folder>/<index>.log.
"""

import argparse
import os
import subprocess
import threading
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(logs: Path, index: int) -> bool:
    """Atomically claim a job; a claim of a dead process is replaced."""
    path = logs / f"{index}.claim"
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(path.read_text().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and alive(owner):
                return False
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        return True
    return False


def run_queue(jobs: list[str], gpus: list[str], logs: Path) -> list[int]:
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    pending = [(index, job) for index, job in enumerate(jobs) if not (logs / f"{index}.done").exists()]
    failed: list[int] = []

    def worker(gpu: str) -> None:
        while True:
            with lock:
                while pending and not claim(logs, pending[0][0]):
                    pending.pop(0)
                if not pending:
                    return
                index, job = pending.pop(0)
            env = os.environ | {"CUDA_VISIBLE_DEVICES": gpu}
            with (logs / f"{index}.log").open("a", encoding="utf-8") as log:
                log.write(f"# gpu {gpu}: {job}\n")
                log.flush()
                done = subprocess.run(job, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            if done.returncode == 0:
                (logs / f"{index}.done").touch()
                (logs / f"{index}.claim").unlink(missing_ok=True)
            else:
                # The claim of a failed job stays while this process lives, so a queue running
                # beside it does not repeat the failure; a later rerun takes the stale claim over.
                with lock:
                    failed.append(index)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(prog="queue")
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--logs", type=Path, default=None, help="default: <jobs file stem>.logs/")
    args = parser.parse_args()
    lines = args.jobs.read_text(encoding="utf-8").splitlines()
    jobs = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    logs = args.logs or args.jobs.with_suffix(".logs")
    failed = run_queue(jobs, args.gpus.split(","), logs)
    if failed:
        raise SystemExit(f"failed jobs: {failed}")


if __name__ == "__main__":
    main()
