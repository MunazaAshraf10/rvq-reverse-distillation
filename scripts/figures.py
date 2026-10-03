"""Every figure of the paper, computed from the result files under results/.

    uv run python scripts/figures.py [--out DIR]

Figures whose inputs do not exist yet are skipped. Style: categorical hues in a fixed order
(validated for colour vision deficiency on the white page), thin marks, recessive axes, a direct
label on every series, reference levels in neutral grey.
"""

import argparse
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from rvq_ae.experiments.common import read_results
from rvq_ae.experiments.stats import bootstrap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WIDTH = 5.5
"""NeurIPS text width in inches."""

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


class MissingInputError(Exception):
    pass


def rows_of(folder: Path) -> list[dict[str, object]]:
    if not folder.is_dir() or not any(folder.glob("part-*.jsonl")):
        raise MissingInputError(str(folder.relative_to(ROOT)))
    return read_results(folder)


def grid(ax: Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def replay_scale() -> plt.Figure:
    """Where encoders sit on the calibrated replay scale (mean and 95% interval over tracks)."""
    calibration = defaultdict(list)
    for row in rows_of(RESULTS / "calibration" / "published"):
        calibration[str(row["condition"])].append(float(row["cosine"]))  # type: ignore[arg-type]
    released = defaultdict(list)
    for row in rows_of(RESULTS / "replay" / "published"):
        if row.get("timeline", "exact") in ("exact", None):
            released[str(row["encoder"])].append(float(row["cosine"]))  # type: ignore[arg-type]
    references = [
        ("random codes", calibration["random"]),
        ("another track's codes", calibration["other-track"]),
        ("true $c_0$, random acoustic", calibration["semantic-random"]),
        ("true codes, 1 frame late", calibration["shift-1"]),
        ("true $c_0$, MiniMax acoustic$^\\dagger$", calibration["semantic-sampled"]),
        ("true codes (control)", released["control"]),
    ]
    names = {"v1": "IH-41M", "v2": "IH-155M", "v4": "ours"}
    models = [(label, released[name]) for name, label in names.items()]
    rows = sorted(references + models, key=lambda item: statistics.mean(item[1]))
    figure, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for index, (label, values) in enumerate(rows):
        interval = bootstrap(values)
        model = label in names.values()
        color = BLUE if model else MUTED
        ax.errorbar(
            interval.mean,
            index,
            xerr=[[interval.mean - interval.low], [interval.high - interval.mean]],
            fmt="o",
            color=color,
            markersize=4,
            elinewidth=1,
            capsize=0,
        )
        ax.annotate(
            f"{interval.mean:.4f}",
            (interval.mean, index),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("condition replay cosine")
    grid(ax, "x")
    return figure


def robustness() -> plt.Figure:
    """Replay against degradation strength, one panel per degradation: ours against ours trained on
    degraded views (genre split, three seeds each, mean over seeds and tracks)."""
    rows = rows_of(RESULTS / "robustness" / "genre-ood")
    curves: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    clean: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        config = str(row["encoder"]).split("/")[-2]
        name = str(row["perturbation"])
        if name == "clean":
            clean[config].append(float(row["cosine"]))  # type: ignore[arg-type]
        else:
            curves[(name, config)][float(row["strength"])].append(float(row["cosine"]))  # type: ignore[arg-type]
    panels = [
        ("noise", "pink noise SNR (dB)", True),
        ("mp3", "MP3 bitrate (kbit/s)", True),
        ("reverb", "reverb RT60 (s)", False),
        ("lowpass", "low-pass cutoff (Hz)", True),
        ("resample", "resampled to (Hz)", True),
        ("highpass", "high-pass cutoff (Hz)", False),
    ]
    series = {"v4_169m": ("ours", BLUE), "v4_169m_augmented": ("ours + degraded views", ORANGE)}
    figure, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.0), sharey=True)
    for ax, (name, label, descending) in zip(axes.flat, panels, strict=True):
        for config, (short, color) in series.items():
            points = curves.get((name, config))
            if not points:
                continue
            xs = sorted(points, reverse=descending)
            ys = [statistics.mean(clean[config])] + [statistics.mean(points[x]) for x in xs]
            ax.plot(range(len(ys)), ys, color=color, linewidth=1.5, marker="o", markersize=3, label=short)
            ax.set_xticks(range(len(ys)), ["clean", *(f"{x:g}" for x in xs)], fontsize=6.5)
        ax.set_xlabel(label, fontsize=7)
        grid(ax)
    axes[0, 0].set_ylabel("replay cosine")
    axes[1, 0].set_ylabel("replay cosine")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.04))
    figure.tight_layout()
    return figure


GAP_CONDITIONS = (
    ("enc:v4", "our codes", BLUE),
    ("enc:v1", "IH-41M codes", ORANGE),
    ("text", "caption only", MUTED),
)
GAP_METRICS = (("mert", "MERT cosine"), ("chroma", "chroma cosine"), ("clap", "CLAP cosine"))


def domain_gap() -> plt.Figure:
    """The same synthesis protocol on generated and on recorded music."""
    sources = [
        ("holdout", "generated\n(held-out)"),
        ("song-describer", "Song\nDescriber"),
        ("musdb", "MUSDB18\n-HQ"),
    ]
    data: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    found = []
    for folder, label in sources:
        try:
            rows = rows_of(RESULTS / "resynth" / folder)
        except MissingInputError:
            continue
        found.append((folder, label))
        for row in rows:
            for metric, _ in GAP_METRICS:
                if metric in row and row[metric] is not None:
                    data[(folder, str(row["condition"]), metric)].append(float(row[metric]))  # type: ignore[arg-type]
    if not found:
        raise MissingInputError("results/resynth")
    figure, axes = plt.subplots(1, len(GAP_METRICS), figsize=(WIDTH, 2.0))
    for ax, (metric, title) in zip(axes, GAP_METRICS, strict=True):
        for offset, (condition, label, color) in enumerate(GAP_CONDITIONS):
            for position, (folder, _) in enumerate(found):
                values = data.get((folder, condition, metric))
                if not values:
                    continue
                interval = bootstrap(values)
                x = position + (offset - 1) * 0.22
                ax.errorbar(
                    x,
                    interval.mean,
                    yerr=[[interval.mean - interval.low], [interval.high - interval.mean]],
                    fmt="o",
                    color=color,
                    markersize=3.5,
                    elinewidth=1,
                    label=label if position == 0 else None,
                )
        ax.set_xticks(range(len(found)), [label for _, label in found], fontsize=6.5)
        ax.set_title(title, fontsize=8, color=INK)
        grid(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=len(labels), fontsize=7, bbox_to_anchor=(0.5, -0.06)
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return figure


def reference_tradeoff() -> plt.Figure:
    """Reference adherence against target caption adherence as the constraint interval varies."""
    rows = rows_of(RESULTS / "usecases" / "reference")
    try:
        rows += rows_of(RESULTS / "usecases" / "reference-musicgen")
    except MissingInputError:
        pass
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"])].append((float(row["clap_text_target"]), float(row["mert"])))  # type: ignore[arg-type]
    means = {
        name: (statistics.mean(x for x, _ in v), statistics.mean(y for _, y in v))
        for name, v in groups.items()
    }
    figure, ax = plt.subplots(figsize=(3.3, 2.4))
    sweep = [name for name in ("ref:v4@1", "ref:v4@5", "ref:v4@10", "text") if name in means]
    ax.plot(
        [means[n][0] for n in sweep],
        [means[n][1] for n in sweep],
        color=BLUE,
        linewidth=1.5,
        marker="o",
        markersize=4,
    )
    labels = {"ref:v4@1": "N=1", "ref:v4@5": "N=5", "ref:v4@10": "N=10", "text": "caption only"}
    for name in sweep:
        ax.annotate(
            labels[name], means[name], xytext=(4, 2), textcoords="offset points", fontsize=7, color=INK
        )
    offsets = {"ref:v1@5": (-46, 2), "musicgen-melody": (4, -8)}
    marked = (("ref:v1@5", ORANGE, "IH-41M, N=5"), ("musicgen-melody", AQUA, "MusicGen-melody"))
    for name, color, label in marked:
        if name in means:
            ax.plot(*means[name], "o", color=color, markersize=4)
            ax.annotate(
                label, means[name], xytext=offsets[name], textcoords="offset points", fontsize=7, color=INK
            )
    ax.set_xlabel("CLAP similarity to the target caption")
    ax.set_ylabel("MERT similarity to the reference")
    grid(ax, "both")
    return figure


FIGURES: dict[str, Callable[[], plt.Figure]] = {
    "replay_scale": replay_scale,
    "robustness": robustness,
    "domain_gap": domain_gap,
    "reference_tradeoff": reference_tradeoff,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="figures")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES.items():
        try:
            figure = build()
        except MissingInputError as error:
            print(f"skip {name}: missing {error}")
            continue
        figure.savefig(args.out / f"{name}.pdf")
        figure.savefig(args.out / f"{name}.png", dpi=200)
        plt.close(figure)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
