"""Write the job lists behind every result of the paper, for rvq_ae.experiments.queue.

    uv run python scripts/jobs.py                 # runs/{train,experiments,evaluate}.jobs
    uv run python -m rvq_ae.experiments.queue runs/train.jobs --gpus 0,1

train        the seed and split grid: 3 seeds of the encoder (v4_169m) and the independent-head
             baselines (v1_41m, v2_155m) on the published split, and of those, the two controls
             (v2_169m_deep, v4_157m_no_feedback) and the degraded-view encoder (v4_169m_augmented)
             on the genre held-out split. One GPU per run.
experiments  calibration, decoding, robustness, the generated test set, real music resynthesis and
             the use cases, each sharded over SHARDS jobs.
evaluate     token metrics and condition replay of every trained run (written once runs finish).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = f"cd {ROOT} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PY = "uv run python -m rvq_ae.experiments"
SHARDS = 4
SEEDS = (1, 2, 3)
GRID = {
    "genre-ood": ("test-id", ["v4_169m", "v2_155m", "v2_169m_deep", "v4_157m_no_feedback", "v1_41m"]),
    "published": ("holdout", ["v4_169m", "v2_155m", "v1_41m"]),
}
AUGMENTED = ("genre-ood", "test-id", "v4_169m_augmented")
"""Trained last: it needs the degraded latent views (rvq-ae cache --views 2)."""
TEST_PARTS = {"genre-ood": ("test-id", "test-ood"), "published": ("holdout",)}
OOD_REPLAY_LIMIT = 200
CODECS = "codec:encodec-2.2 codec:dac-1.7 codec:dac-2.6"


def sharded(command: str, shards: int = SHARDS) -> list[str]:
    return [f"{ENV} RANK={rank} WORLD_SIZE={shards} {PY}.{command}" for rank in range(shards)]


def train_command(rule: str, validation: str, config: str, seed: int) -> str:
    return (
        f"{ENV} uv run rvq-ae train --config configs/{config}.json --seed {seed} --split-rule {rule} "
        f"--validation-split {validation} --output runs/{rule}/{config}/seed-{seed}"
    )


def train_jobs() -> list[str]:
    lines = ["# Seed and split grid; one single GPU training run per line."]
    for seed in SEEDS:
        for rule, (validation, configs) in GRID.items():
            lines += [train_command(rule, validation, config, seed) for config in configs]
    lines.append("# Trained on degraded latent views; run after rvq-ae cache --split train --views 2.")
    lines += [train_command(*AUGMENTED, seed) for seed in SEEDS]
    return lines


def all_runs(rule: str) -> list[str]:
    configs = GRID[rule][1] + ([AUGMENTED[2]] if rule == AUGMENTED[0] else [])
    return [f"runs/{rule}/{config}/seed-{seed}" for config in configs for seed in SEEDS]


def experiment_jobs() -> list[str]:
    blocks = {
        "Condition replay of the released encoders on three timelines, and its calibration": [
            *sharded(
                "replay --out results/replay/published --encoders v1 v2 v3 v4 "
                "--timelines exact nominal stride"
            ),
            *sharded("calibrate --out results/calibration/published --encoders v1 v4"),
        ],
        "Generator completed decoding (sequential), 32 hashed held-out tracks": sharded(
            "replay --out results/replay/completed --limit 32 --encoders v1 v4 "
            "--decodes complete-greedy complete-sampled --no-control"
        ),
        "MM3-OOD generation: 240 tracks from MusicCaps captions outside the corpus genres": sharded(
            "generate --out data/mm3-ood"
        ),
        "Real music resynthesis: Song Describer, 200 clips of 20 s": sharded(
            f"resynth --out results/resynth/song-describer --source song-describer --limit 200 "
            f"--conditions enc:v1 enc:v4 sem:v4 text dav {CODECS} --save-audio 3"
        ),
        "Generated resynthesis: published held-out, 100 clips (same protocol, with the true codes)": sharded(
            f"resynth --out results/resynth/holdout --source corpus:published:holdout --limit 100 "
            f"--conditions enc:v1 enc:v4 sem:v4 true text dav {CODECS} --save-audio 2"
        ),
        "Robustness: label preserving degradations, 64 held-out tracks": sharded(
            "replay --out results/robustness/published --limit 64 --encoders v1 v4 "
            "--perturbations grid --no-control"
        ),
        "Real music resynthesis: MUSDB18-HQ test, mixtures and accompaniments": sharded(
            f"resynth --out results/resynth/musdb --source musdb "
            f"--conditions enc:v1 enc:v4 sem:v4 text dav {CODECS} --save-audio 2"
        ),
        "Use case: reference-guided generation, 150 Song Describer references": [
            *sharded(
                "usecases reference --out results/usecases/reference --limit 150 "
                "--conditions ref:v4@1 ref:v4@5 ref:v4@10 ref:v1@5 text --save-audio 3"
            ),
            *sharded(
                "usecases reference --out results/usecases/reference-musicgen --limit 150 "
                "--conditions musicgen-melody --save-audio 3",
                1,
            ),
        ],
        "Use case: audio-prompted continuation, 150 Song Describer recordings": [
            *sharded(
                "usecases continuation --out results/usecases/continuation --limit 150 "
                "--conditions cont:v4 cont:v1 text other --save-audio 3"
            ),
            *sharded(
                "usecases continuation --out results/usecases/continuation-musicgen --limit 150 "
                "--conditions musicgen --save-audio 3",
                1,
            ),
        ],
    }
    lines: list[str] = []
    for title, jobs in blocks.items():
        lines += [f"# {title}", *jobs]
    return lines


def evaluate_jobs() -> list[str]:
    """Token metrics and replay of every run, MM3-OOD for every encoder, and the augmentation study."""
    revision = "--revision 5029b1e7f1bbfbf028b76b38564fecccda94a111"
    lines = ["# Token metrics of every trained run, on every test part of its split."]
    for rule in GRID:
        for run in all_runs(rule):
            for part in TEST_PARTS[rule]:
                config, seed = run.split("/")[-2:]
                lines.append(
                    f"{ENV} uv run rvq-ae evaluate --run {run} --checkpoint final --split-rule {rule} "
                    f"--split {part} {revision} --out results/tokens/{rule}/{part}/{config}/{seed}"
                )
    lines.append("# Condition replay of every trained run (all runs of a split share one MiniMax load).")
    for rule in GRID:
        runs = " ".join(f"{run}/final" for run in all_runs(rule))
        for part in TEST_PARTS[rule]:
            limit = f" --limit {OOD_REPLAY_LIMIT}" if part == "test-ood" else ""
            lines += sharded(
                f"replay --out results/replay/{rule}/{part} --rule {rule} --part {part}{limit} "
                f"--encoders {runs}"
            )
    everything = " ".join(["v1 v2 v3 v4", *(f"{run}/final" for rule in GRID for run in all_runs(rule))])
    lines.append("# MM3-OOD: the latent cache, then token metrics and replay of every encoder.")
    lines.append(
        f"{ENV} uv run rvq-ae cache --dataset data/mm3-ood --corpus data/mm3-ood --split ood "
        "--cache cache/latents"
    )
    lines += sharded(
        f"replay --out results/replay/mm3-ood --dataset data/mm3-ood --part ood --encoders {everything}"
    )
    augmented = " ".join(
        f"runs/genre-ood/{config}/seed-{seed}/final" for config in ("v4_169m", AUGMENTED[2]) for seed in SEEDS
    )
    lines.append("# Augmentation study: robustness and real music, v4 against v4 trained on degraded views.")
    lines += sharded(
        f"replay --out results/robustness/genre-ood --rule genre-ood --part test-id --limit 32 "
        f"--encoders {augmented} --perturbations grid --no-control"
    )
    pair = " ".join(f"enc:runs/genre-ood/{config}/seed-1/final" for config in ("v4_169m", AUGMENTED[2]))
    lines += sharded(
        f"resynth --out results/resynth/song-describer-augmented --source song-describer --limit 200 "
        f"--conditions {pair}"
    )
    return lines


def main() -> None:
    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    for name, lines in (
        ("train", train_jobs()),
        ("experiments", experiment_jobs()),
        ("evaluate", evaluate_jobs()),
    ):
        path = out / f"{name}.jobs"
        jobs = [line for line in lines if not line.startswith("#")]
        if path.is_file():
            # The queue marks finished jobs by position, so a list may only grow at its end.
            text = path.read_text(encoding="utf-8")
            old = [line for line in text.splitlines() if line and not line.startswith("#")]
            tail = [
                (a.split("-m rvq_ae")[-1], b.split("-m rvq_ae")[-1]) for a, b in zip(old, jobs, strict=False)
            ]
            if len(old) > len(jobs) or any(a.split("rvq-ae")[-1] != b.split("rvq-ae")[-1] for a, b in tail):
                raise SystemExit(f"{path} would reorder its jobs; move it aside first")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(name, len(jobs), "jobs")


if __name__ == "__main__":
    main()
