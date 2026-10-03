"""Uncertainty for paired, per track results, optionally over several training seeds.

Every model is scored on the same tracks, so differences are paired by track. With one run per
model the interval is a percentile bootstrap over tracks; with several seeds it is a hierarchical
bootstrap that resamples seeds, then tracks, so the interval covers both training and test set
variation. p values are two sided paired sign flip permutation tests; families of comparisons are
corrected with Holm's step down procedure.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

DRAWS = 10_000


@dataclass(frozen=True, slots=True)
class Interval:
    mean: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "low": self.low, "high": self.high}


def bootstrap(values: Sequence[float], *, draws: int = DRAWS, seed: int = 0, level: float = 0.95) -> Interval:
    """Percentile bootstrap interval of a mean."""
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(draws, len(data)))].mean(axis=1)
    tail = (1 - level) / 2
    return Interval(float(data.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)))


def hierarchical(
    runs: Sequence[Mapping[str, float]], *, draws: int = DRAWS, seed: int = 0, level: float = 0.95
) -> Interval:
    """Interval of the mean over seeds and tracks; runs[i] maps track to value for seed i.

    Each draw resamples the seeds with replacement and, independently for each drawn seed, the
    tracks with replacement. Only tracks present in every run are used.
    """
    tracks = sorted(set.intersection(*(set(run) for run in runs)))
    table = np.array([[run[track] for track in tracks] for run in runs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    seeds, count = table.shape
    means = np.empty(draws)
    for draw in range(draws):
        drawn = table[rng.choice(seeds, size=seeds)]
        columns = rng.choice(count, size=(seeds, count))
        means[draw] = np.take_along_axis(drawn, columns, axis=1).mean()
    tail = (1 - level) / 2
    return Interval(float(table.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)))


def paired_difference(
    first: Mapping[str, float], second: Mapping[str, float], *, draws: int = DRAWS, seed: int = 0
) -> tuple[Interval, float, int, int]:
    """Interval of the mean per track difference first - second, its p value, wins and track count."""
    tracks = sorted(set(first) & set(second))
    diffs = np.array([first[track] - second[track] for track in tracks])
    interval = bootstrap(diffs.tolist(), draws=draws, seed=seed)
    return interval, sign_flip(diffs, draws=draws, seed=seed), int((diffs > 0).sum()), len(tracks)


def sign_flip(diffs: np.ndarray, *, draws: int = DRAWS, seed: int = 0) -> float:
    """Two sided p value of a zero mean paired difference by random sign flips."""
    rng = np.random.default_rng(seed)
    observed = abs(diffs.mean())
    flips = rng.choice((-1.0, 1.0), size=(draws, len(diffs)))
    null = np.abs((flips * diffs).mean(axis=1))
    return float((1 + (null >= observed).sum()) / (draws + 1))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm adjusted p values, in the input order."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()
