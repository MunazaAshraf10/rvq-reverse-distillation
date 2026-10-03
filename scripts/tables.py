"""Every table of the paper, computed from the result files under results/.

    uv run python scripts/tables.py [--tex DIR] [--markdown FILE]

A table whose inputs do not exist yet is skipped with a notice, so the paper builds at any stage of
the experiment queue. Numbers are never typed by hand: each function reads per track rows, computes
means and intervals with rvq_ae.experiments.stats, and returns a header and rows that are written as
LaTeX files into --tex and as one Markdown file, --markdown.
"""

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub.errors import HfHubHTTPError

from rvq_ae.constants import DATASET_REPO
from rvq_ae.data.records import load_records
from rvq_ae.data.splits import GENRE_OOD, ID_TEST, OOD_TEST, PUBLISHED, lyric_key, make_split, prompt_key
from rvq_ae.experiments.common import DATASET_REVISION, read_results
from rvq_ae.experiments.metrics import frechet_distance
from rvq_ae.experiments.stats import Interval, bootstrap, hierarchical, holm, paired_difference

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PUBLISHED_RESULTS = RESULTS / "published"

Row = list[str]
Table = tuple[Row, list[Row]]

RELEASES = ("v1", "v2", "v3", "v4")
RELEASE_LABELS = {
    "community": "community 41M",
    "v1": "IH-41M",
    "v2": "IH-155M",
    "v3": "IH-155M+MERT",
    "v4": "ours",
}
CONFIG_LABELS = {
    "v1_41m": "IH-41M",
    "v2_155m": "IH-155M",
    "v2_169m_deep": "IH-169M (control)",
    "v4_157m_no_feedback": "no feedback (control)",
    "v4_169m": "ours",
    "v4_169m_augmented": "ours + degraded views",
}


class MissingInputError(Exception):
    """Raised by a table whose inputs are not there yet."""


# Formatting ---------------------------------------------------------------------------------------


def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def ci(interval: Interval, digits: int = 4) -> str:
    return f"{interval.mean:.{digits}f} [{interval.low:.{digits}f}, {interval.high:.{digits}f}]"


def pm(values: Sequence[float], digits: int = 4) -> str:
    """Mean and standard deviation over seeds."""
    if len(values) < 2:
        return num(values[0], digits)
    return f"{statistics.mean(values):.{digits}f} $\\pm$ {statistics.stdev(values):.{digits}f}"


def rows_of(folder: Path) -> list[dict[str, Any]]:
    if not folder.is_dir() or not any(folder.glob("part-*.jsonl")):
        raise MissingInputError(str(folder.relative_to(ROOT)))
    return read_results(folder)


def by_track(rows: Iterable[dict[str, Any]], value: str = "cosine", track: str = "shard") -> dict[str, float]:
    return {str(row[track]): float(row[value]) for row in rows}


# Condition replay of the released encoders --------------------------------------------------------


def released_replay() -> dict[tuple[str, str], dict[str, float]]:
    """Per track replay cosine of every released encoder and timeline, plus the control."""
    rows = rows_of(RESULTS / "replay" / "published")
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        out[(row["encoder"], row.get("timeline", "-"))][str(row["shard"])] = float(row["cosine"])
    return out


def community_replay() -> dict[str, float]:
    """The community encoder's published per track replay (its checkpoint cannot be rerun here)."""
    data = json.loads((PUBLISHED_RESULTS / "replay" / "raw-metrics-serveurperso-v1.json").read_text())
    return {
        str(record["shard_id"]): float(record["condition_embedding_replay"]["predicted_codes"]["cosine_mean"])
        for record in data["records"]
    }


def published_replay(variant: str) -> dict[str, float]:
    data = json.loads((PUBLISHED_RESULTS / "replay" / f"raw-metrics-simpletuner-{variant}.json").read_text())
    return {
        str(record["shard_id"]): float(record["condition_embedding_replay"]["predicted_codes"]["cosine_mean"])
        for record in data["records"]
    }


def reproduction_table() -> Table:
    """Our reimplementation of condition replay against the published per track values."""
    ours = released_replay()
    header = ["Model", "Published", "Reproduced", "Mean difference", "Max $|$difference$|$", "Tracks"]
    rows: list[Row] = []
    for variant in RELEASES:
        published = published_replay(variant)
        mine = ours[(variant, "exact")]
        diffs = [mine[t] - published[t] for t in mine if t in published]
        rows.append(
            [
                RELEASE_LABELS[variant],
                num(statistics.mean(published.values())),
                num(statistics.mean(mine.values())),
                f"{statistics.mean(diffs):+.4f}",
                num(max(abs(d) for d in diffs)),
                str(len(diffs)),
            ]
        )
    control = ours[("control", "-")]
    rows.append(
        ["true codes (control)", "0.9999", num(statistics.mean(control.values())), "", "", str(len(control))]
    )
    return header, rows


def timeline_table() -> Table:
    """Replay of every released encoder on the exact, nominal and fixed stride timelines."""
    ours = released_replay()
    header = ["Model", "Exact (stitching table)", "Nominal (Eq. 1)", "Fixed stride 441/128"]
    rows = [
        [RELEASE_LABELS[v]]
        + [ci(bootstrap(list(ours[(v, t)].values()))) for t in ("exact", "nominal", "stride")]
        for v in RELEASES
    ]
    return header, rows


def released_comparisons() -> Table:
    """Paired differences between the released encoders on the published held-out split."""
    ours = released_replay()
    scores = {variant: ours[(variant, "exact")] for variant in RELEASES} | {"community": community_replay()}
    pairs = (("v1", "community"), ("v2", "v1"), ("v3", "v2"), ("v4", "v2"), ("v4", "v3"))
    results = [paired_difference(scores[a], scores[b]) for a, b in pairs]
    adjusted = holm([p for _, p, _, _ in results])
    header = ["Comparison", "Mean difference", "95\\% interval", "Tracks improved", "Holm $p$"]
    rows = [
        [
            f"{RELEASE_LABELS[a]} vs {RELEASE_LABELS[b]}",
            f"{interval.mean:+.4f}",
            f"[{interval.low:+.4f}, {interval.high:+.4f}]",
            f"{wins} / {count}",
            f"{p:.4f}",
        ]
        for (a, b), (interval, _, wins, count), p in zip(pairs, results, adjusted, strict=True)
    ]
    return header, rows


# Calibration of the metric ------------------------------------------------------------------------

CALIBRATION = (
    ("random", "uniformly random codes"),
    ("other-track", "codes of another held-out track"),
    ("semantic-random", "true $c_0$, random acoustic books"),
    ("acoustic-random", "random $c_0$, true acoustic books"),
    ("shift-5", "true codes delayed by 5 frames"),
    ("shift-1", "true codes delayed by 1 frame"),
    ("noise-sem-0.5", "50\\% of $c_0$ replaced"),
    ("noise-ac-0.5", "50\\% of acoustic codes replaced"),
    ("noise-sem-0.1", "10\\% of $c_0$ replaced"),
    ("noise-ac-0.1", "10\\% of acoustic codes replaced"),
    ("semantic-sampled", "true $c_0$, acoustic books sampled by MiniMax$^\\dagger$"),
    ("semantic-greedy", "true $c_0$, acoustic books by MiniMax argmax$^\\dagger$"),
    ("v1-c0", "IH-41M $c_0$, acoustic books sampled by MiniMax$^\\dagger$"),
    ("v4-c0", "ours $c_0$, acoustic books sampled by MiniMax$^\\dagger$"),
    ("v1-c0-greedy", "IH-41M $c_0$, acoustic books by MiniMax argmax$^\\dagger$"),
    ("v4-c0-greedy", "ours $c_0$, acoustic books by MiniMax argmax$^\\dagger$"),
)


def calibration_table() -> Table:
    rows_in = rows_of(RESULTS / "calibration" / "published")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows_in:
        groups[row["condition"]].append(float(row["cosine"]))
    released = released_replay()
    header = ["Code sequence", "Replay cosine", "95\\% interval"]
    rows: list[Row] = []
    for name, label in CALIBRATION:
        if name in groups:
            interval = bootstrap(groups[name])
            rows.append([label, num(interval.mean), f"[{num(interval.low)}, {num(interval.high)}]"])
    for variant in ("v1", "v4"):
        interval = bootstrap(list(released[(variant, "exact")].values()))
        rows.append(
            [
                f"{RELEASE_LABELS[variant]}, all eight books",
                num(interval.mean),
                f"[{num(interval.low)}, {num(interval.high)}]",
            ]
        )
    control = bootstrap(list(released[("control", "-")].values()))
    rows.append(["true codes (control)", num(control.mean), f"[{num(control.low)}, {num(control.high)}]"])
    return header, sorted(rows, key=lambda row: float(row[1]))


# Corpus and splits --------------------------------------------------------------------------------


def corpus_table() -> Table:
    """Content statistics of the corpus and of both split rules (needs the corpus index from the Hub)."""
    try:
        grouped = load_records(DATASET_REPO, revision=DATASET_REVISION)
    except (OSError, HfHubHTTPError) as error:
        raise MissingInputError(f"corpus index ({error.__class__.__name__})") from error
    records = [record for part in grouped.values() for record in part]
    published = make_split(grouped, PUBLISHED).parts
    strict = make_split(grouped, GENRE_OOD)

    def overlap(test: Sequence[Any], train: Sequence[Any]) -> str:
        prompts = {prompt_key(r.prompt) for r in train}
        lyrics = {lyric_key(r.lyrics) for r in train} - {""}
        shared_prompt = sum(prompt_key(r.prompt) in prompts for r in test)
        shared_lyric = sum(lyric_key(r.lyrics) in lyrics for r in test)
        return f"{shared_prompt} / {shared_lyric} of {len(test)}"

    genres = defaultdict(int)
    for record in records:
        genres[record.genre] += 1
    header = ["Statistic", "Value"]
    rows = [
        ["tracks (exact stitching table)", f"{len(records):,} ({sum(r.exact for r in records):,})"],
        ["genre tags", ", ".join(f"{g} {c}" for g, c in sorted(genres.items(), key=lambda kv: -kv[1]))],
        ["distinct normalised prompts", f"{len({prompt_key(r.prompt) for r in records})}"],
        ["instrumental tracks", f"{sum(lyric_key(r.lyrics) == '' for r in records)}"],
        ["published split: train / held-out", f"{len(published['train']):,} / {len(published['holdout'])}"],
        [
            "published held-out sharing a prompt / a lyric with train",
            overlap(published["holdout"], published["train"]),
        ],
        [
            "genre held-out split: train / ID test / OOD test (jazz)",
            f"{len(strict.parts['train']):,} / {len(strict.parts[ID_TEST])} / {len(strict.parts[OOD_TEST])}",
        ],
        ["training tracks dropped for a shared lyric", f"{strict.dropped}"],
        [
            "ID test sharing a prompt / a lyric with train",
            overlap(strict.parts[ID_TEST], strict.parts["train"]),
        ],
        [
            "OOD test sharing a prompt / a lyric with train",
            overlap(strict.parts[OOD_TEST], strict.parts["train"]),
        ],
    ]
    return header, rows


# Evaluation by synthesis ------------------------------------------------------------------------

SYNTHESIS_LABELS = {
    "dav": "DAV autoencoder (latent ceiling)",
    "true": "true codes (ceiling)",
    "codec:encodec-2.2": "EnCodec, 2.2 kbit/s",
    "codec:dac-2.6": "DAC, 2.6 kbit/s",
    "codec:dac-1.7": "DAC, 1.7 kbit/s",
    "enc:v4": "our codes, 2.1 kbit/s",
    "enc:v1": "IH-41M codes, 2.1 kbit/s",
    "sem:v4": "our $c_0$ + MiniMax acoustic, 0.35 kbit/s",
    "text": "caption only (floor)",
}
SYNTHESIS_METRICS = (
    ("mert", "MERT"),
    ("clap", "CLAP"),
    ("chroma", "Chroma"),
    ("key_score", "Key"),
    ("tempo_acc2", "Tempo"),
    ("mel", "Mel $\\downarrow$"),
)


def synthesis_rows(source: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_of(RESULTS / "resynth" / source):
        groups[str(row["condition"])].append(row)
    return groups


def embeddings(folder: Path, kind: str = "clap") -> dict[str, np.ndarray]:
    """Set level embeddings per condition (rows of embeddings/part-*.jsonl, committed gzipped)."""
    groups: dict[str, list[list[float]]] = defaultdict(list)
    parts = {path.name.removesuffix(".gz"): path for path in (folder / "embeddings").glob("part-*.jsonl.gz")}
    parts |= {path.name: path for path in (folder / "embeddings").glob("part-*.jsonl")}
    for _, path in sorted(parts.items()):
        data = path.read_bytes()
        text = gzip.decompress(data) if path.suffix == ".gz" else data
        for line in text.decode("utf-8").splitlines():
            if line:
                row = json.loads(line)
                groups[row["condition"]].append(row[kind])
    return {name: np.asarray(values, dtype=np.float64) for name, values in groups.items()}


def synthesis_table(source: str) -> Table:
    """Every condition of one source against its references: means over clips, FAD over the set."""
    groups = synthesis_rows(source)
    sets = embeddings(RESULTS / "resynth" / source)
    header = ["Condition"] + [label for _, label in SYNTHESIS_METRICS] + ["FAD $\\downarrow$", "Clips"]
    rows: list[Row] = []
    for condition, label in SYNTHESIS_LABELS.items():
        if condition not in groups:
            continue
        values = groups[condition]
        cells = [label]
        for metric, _ in SYNTHESIS_METRICS:
            numbers = [float(row[metric]) for row in values if row.get(metric) is not None]
            cells.append(num(statistics.mean(numbers), 3) if numbers else "")
        fad = frechet_distance(sets[condition], sets["reference"]) if condition in sets else float("nan")
        cells += [num(fad, 3), str(len(values))]
        rows.append(cells)
    return header, rows


GAP_SOURCES = (
    ("holdout", "generated, published held-out"),
    ("song-describer", "recorded, Song Describer"),
    ("musdb:mixture", "recorded, MUSDB18-HQ mixtures"),
    ("musdb:accompaniment", "recorded, MUSDB18-HQ accompaniments"),
)


def recovered(values: dict[str, list[dict[str, Any]]], condition: str, metric: str) -> Interval:
    """Share of the floor (caption only) to ceiling (DAV) gap a condition recovers, paired by clip."""
    by_clip = {
        name: {row["clip"]: float(row[metric]) for row in values[name]} for name in (condition, "text", "dav")
    }
    clips = sorted(set.intersection(*(set(v) for v in by_clip.values())))
    shares = [
        (by_clip[condition][c] - by_clip["text"][c]) / (by_clip["dav"][c] - by_clip["text"][c])
        for c in clips
        if abs(by_clip["dav"][c] - by_clip["text"][c]) > 1e-6
    ]
    return bootstrap(shares)


def domain_gap_table() -> Table:
    """How much of what the codec path can carry the recovered codes carry, generated against recorded."""
    header = ["Audio", "our codes", "IH-41M codes", "our $c_0$ only", "Clips"]
    rows: list[Row] = []
    for source, label in GAP_SOURCES:
        folder, _, group = source.partition(":")
        try:
            groups = synthesis_rows(folder)
        except MissingInputError:
            continue
        if group:
            groups = {name: [r for r in values if r.get("group") == group] for name, values in groups.items()}
        cells = [label]
        for condition in ("enc:v4", "enc:v1", "sem:v4"):
            cells.append(ci(recovered(groups, condition, "mert"), 3) if condition in groups else "")
        cells.append(str(len(groups.get("enc:v4", []))))
        rows.append(cells)
    if not rows:
        raise MissingInputError("results/resynth")
    return header, rows


# Seeds and splits ---------------------------------------------------------------------------------

GRID_PARTS = (
    ("published", "holdout", "published held-out"),
    ("genre-ood", "test-id", "genre split, ID"),
    ("genre-ood", "test-ood", "genre split, OOD (jazz)"),
)


def trained_replay(rule: str, part: str) -> dict[str, dict[str, dict[str, float]]]:
    """config -> seed -> track -> replay cosine."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows_of(RESULTS / "replay" / rule / part):
        if row["encoder"] == "control":
            continue
        config, seed = str(row["encoder"]).split("/")[-2:]
        out[config][seed][str(row["shard"])] = float(row["cosine"])
    return out


def seeds_table() -> Table:
    """Replay of every retrained configuration on every test part: mean over seeds with a hierarchical
    95% interval over seeds and tracks."""
    header = ["Model"] + [label for _, _, label in GRID_PARTS]
    found = {}
    for rule, part, _ in GRID_PARTS:
        try:
            found[(rule, part)] = trained_replay(rule, part)
        except MissingInputError:
            found[(rule, part)] = {}
    if not any(found.values()):
        raise MissingInputError("results/replay/<rule>/<part>")
    rows: list[Row] = []
    for config, label in CONFIG_LABELS.items():
        cells = [label]
        for rule, part, _ in GRID_PARTS:
            seeds = found[(rule, part)].get(config)
            if not seeds:
                cells.append("")
                continue
            interval = (
                hierarchical(list(seeds.values()))
                if len(seeds) > 1
                else bootstrap(list(next(iter(seeds.values())).values()))
            )
            cells.append(f"{ci(interval)} ({len(seeds)})")
        rows.append(cells)
    return header, rows


def tokens_table() -> Table:
    """Exact token agreement of every retrained configuration, mean $\\pm$ s.d. over seeds."""
    header = ["Model", "Split", "Sem. top-1", "Sem. top-5", "Ac. top-1", "Ac. top-1 (forced)"]
    rows: list[Row] = []
    for rule, part, label in GRID_PARTS:
        for config, name in CONFIG_LABELS.items():
            files = sorted((RESULTS / "tokens" / rule / part / config).glob("seed-*/metrics.json"))
            if not files:
                continue
            metrics = [rows[0] for rows in (json.loads(f.read_text()) for f in files) if rows]

            def cell(key: str, metrics: list[dict[str, float]] = metrics) -> str:
                values = [100 * m[key] for m in metrics if key in m]
                return pm(values, 2) if values else ""

            rows.append(
                [
                    name,
                    label,
                    cell("semantic_top1"),
                    cell("semantic_top5"),
                    cell("acoustic_top1"),
                    cell("teacher_forced_acoustic_top1"),
                ]
            )
    if not rows:
        raise MissingInputError("results/tokens")
    return header, rows


def mm3_table() -> Table:
    """Replay and exact semantic agreement on MM3-OOD, for every released and retrained encoder."""
    rows_in = rows_of(RESULTS / "replay" / "mm3-ood")
    cosine: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    semantic: dict[str, list[float]] = defaultdict(list)
    for row in rows_in:
        name = str(row["encoder"])
        parts = name.split("/")
        group, seed = (name, "-") if len(parts) == 1 else ("/".join(parts[:2]), parts[2])
        cosine[group][seed][str(row["shard"])] = float(row["cosine"])
        if "semantic_top1" in row:
            semantic[group].append(float(row["semantic_top1"]))
    header = ["Encoder", "Trained on", "Replay cosine", "Sem. top-1", "Seeds"]
    rows: list[Row] = []
    for group in [f"{rule}/{c}" for rule in ("published", "genre-ood") for c in CONFIG_LABELS]:
        if group not in cosine:
            continue
        seeds = cosine[group]
        interval = (
            hierarchical(list(seeds.values()))
            if len(seeds) > 1
            else bootstrap(list(next(iter(seeds.values())).values()))
        )
        label = CONFIG_LABELS[group.split("/")[1]]
        trained = group.split("/")[0]
        rows.append([label, trained, ci(interval), pct(statistics.mean(semantic[group])), str(len(seeds))])
    if "control" in cosine:
        control = bootstrap(list(cosine["control"]["-"].values()))
        rows.append(["true codes (control)", "", ci(control), "", ""])
    return header, rows


# Robustness, decoding and use cases ----------------------------------------------------------------


def robustness_table() -> Table:
    rows_in = rows_of(RESULTS / "robustness" / "published")
    groups: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in rows_in:
        groups[(str(row["perturbation"]), float(row["strength"]), str(row["encoder"]))].append(
            float(row["cosine"])
        )
    encoders = sorted({key[2] for key in groups})
    header = ["Degradation", "Strength"] + [f"{RELEASE_LABELS.get(e, e)}" for e in encoders]
    rows = [
        [name, "" if name in ("clean", "mono") else f"{strength:g}"]
        + [
            num(statistics.mean(groups[(name, strength, e)])) if (name, strength, e) in groups else ""
            for e in encoders
        ]
        for name, strength in dict.fromkeys((k[0], k[1]) for k in groups)
    ]
    return header, rows


def augmentation_table() -> Table:
    """v4 against v4 trained on degraded views, under every degradation (genre split ID test,
    32 tracks, mean over three seeds each, paired difference with 95% interval over tracks)."""
    rows_in = rows_of(RESULTS / "robustness" / "genre-ood")
    groups: dict[tuple[str, float, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows_in:
        config = str(row["encoder"]).split("/")[-2]
        groups[(str(row["perturbation"]), float(row["strength"]), config)][str(row["shard"])].append(
            float(row["cosine"])
        )
    header = ["Degradation", "Strength", "ours", "+ degraded views", "Difference"]
    rows: list[Row] = []
    for name, strength in dict.fromkeys((k[0], k[1]) for k in groups):
        plain = {t: statistics.mean(v) for t, v in groups[(name, strength, "v4_169m")].items()}
        augmented = {t: statistics.mean(v) for t, v in groups[(name, strength, "v4_169m_augmented")].items()}
        if not plain or not augmented:
            continue
        interval, _, _, _ = paired_difference(augmented, plain)
        rows.append(
            [
                name,
                "" if name in ("clean", "mono") else f"{strength:g}",
                num(statistics.mean(plain.values()), 3),
                num(statistics.mean(augmented.values()), 3),
                f"{interval.mean:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]",
            ]
        )
    return header, rows


def decoding_table() -> Table:
    rows_in = rows_of(RESULTS / "replay" / "completed")
    released = released_replay()
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows_in:
        groups[(str(row["encoder"]), str(row["decode"]))][str(row["shard"])] = float(row["cosine"])
    header = [
        "Encoder",
        "Own acoustic codes",
        "MiniMax completes, argmax",
        "MiniMax completes, sampled",
        "Tracks",
    ]
    rows: list[Row] = []
    for encoder in ("v1", "v4"):
        tracks = groups.get((encoder, "complete-greedy"), {})
        own = {t: released[(encoder, "exact")][t] for t in tracks if t in released[(encoder, "exact")]}
        cells = [RELEASE_LABELS[encoder], num(statistics.mean(own.values())) if own else ""]
        for decode in ("complete-greedy", "complete-sampled"):
            values = groups.get((encoder, decode))
            cells.append(ci(bootstrap(list(values.values()))) if values else "")
        cells.append(str(len(tracks)))
        rows.append(cells)
    return header, rows


USE_LABELS = {
    "ref:v4@1": "ours, every frame",
    "ref:v4@5": "ours, every 5th frame",
    "ref:v4@10": "ours, every 10th frame",
    "ref:v1@5": "IH-41M, every 5th frame",
    "text": "target caption only",
    "musicgen-melody": "MusicGen-melody (chroma)",
    "cont:v4": "our codes as history",
    "cont:v1": "IH-41M codes as history",
    "musicgen": "MusicGen-medium (audio prompt)",
    "other": "another recording (floor)",
}


def usecase_rows(name: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for folder in (RESULTS / "usecases" / name, RESULTS / "usecases" / f"{name}-musicgen"):
        if folder.is_dir():
            for row in read_results(folder):
                groups[str(row["condition"])].append(row)
    if not groups:
        raise MissingInputError(f"results/usecases/{name}")
    return groups


def reference_table() -> Table:
    groups = usecase_rows("reference")
    header = ["Generation", "MERT to ref.", "Chroma to ref.", "CLAP to ref.", "CLAP to target text", "Clips"]
    rows = []
    for condition in ("ref:v4@1", "ref:v4@5", "ref:v4@10", "ref:v1@5", "text", "musicgen-melody"):
        if condition in groups:
            values = groups[condition]
            rows.append(
                [USE_LABELS[condition]]
                + [
                    num(statistics.mean(float(r[m]) for r in values), 3)
                    for m in ("mert", "chroma", "clap", "clap_text_target")
                ]
                + [str(len(values))]
            )
    return header, rows


def continuation_table() -> Table:
    groups = usecase_rows("continuation")
    header = ["Continuation", "MERT to truth", "CLAP to truth", "Key", "Tempo", "MERT to prompt", "Clips"]
    rows = []
    for condition in ("cont:v4", "cont:v1", "text", "musicgen", "other"):
        if condition in groups:
            values = groups[condition]
            rows.append(
                [USE_LABELS[condition]]
                + [
                    num(statistics.mean(float(r[m]) for r in values), 3)
                    for m in ("mert", "clap", "key_score", "tempo_acc2", "prompt_mert")
                ]
                + [str(len(values))]
            )
    return header, rows


# Writers ------------------------------------------------------------------------------------------


def markdown(header: Row, rows: Sequence[Row]) -> str:
    def clean(cell: str) -> str:
        return re.sub(r"\$\\pm\$", "±", cell).replace("\\%", "%").replace("$", "").replace("\\", "")

    lines = ["| " + " | ".join(clean(cell) for cell in header) + " |"]
    lines.append("|" + "|".join("---" if index == 0 else "---:" for index in range(len(header))) + "|")
    lines += ["| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def escape(cell: str) -> str:
    """Escape underscores outside inline math."""
    parts = cell.split("$")
    return "$".join(part if index % 2 else part.replace("_", "\\_") for index, part in enumerate(parts))


def latex(header: Row, rows: Sequence[Row], *, label: str, caption: str, wide: bool = False) -> str:
    """A booktabs table for the NeurIPS single column layout."""
    spec = "l" + "r" * (len(header) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small" if not wide else "\\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\begin{adjustbox}{max width=\\linewidth}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        *(" & ".join(escape(cell) for cell in row) + " \\\\" for row in rows),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{adjustbox}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


TABLES: dict[str, tuple[Callable[[], Table], str]] = {
    "reproduction": (
        reproduction_table,
        "Validation of our reimplementation of condition replay against previously published per track "
        "values of four preliminary checkpoints (130 exact held-out tracks).",
    ),
    "timeline": (
        timeline_table,
        "Condition replay (mean and 95\\% bootstrap interval over 130 tracks) of each encoder "
        "run on three frame to latent timelines.",
    ),
    "released": (
        released_comparisons,
        "Paired differences of condition replay between encoders and an independent community encoder "
        "(130 tracks, 10,000 bootstrap draws, Holm corrected sign flip tests).",
    ),
    "calibration": (
        calibration_table,
        "Calibration of condition replay on the 130 held-out tracks. $\\dagger$: the depth decoder reads "
        "language model states computed from the true history, an upper bound for any decoder of the same"
        " semantic codes.",
    ),
    "corpus": (corpus_table, "The reverse distillation corpus and the two split rules."),
    "synthesis": (
        lambda: synthesis_table("song-describer"),
        "Evaluation by synthesis on 200 Song Describer recordings: renderings of each condition against "
        "the recording (means over clips; FAD between CLAP embedding sets).",
    ),
    "synthesis_holdout": (
        lambda: synthesis_table("holdout"),
        "The same protocol on 100 generated held-out clips.",
    ),
    "synthesis_musdb": (
        lambda: synthesis_table("musdb"),
        "The same protocol on MUSDB18-HQ test mixtures and accompaniments.",
    ),
    "gap": (
        domain_gap_table,
        "Share of the gap between the caption-only floor and the DAV ceiling (MERT cosine, per clip) "
        "recovered by each code stream, on generated and recorded audio (mean and 95\\% interval).",
    ),
    "seeds": (
        seeds_table,
        "Condition replay (mean, 95\\% hierarchical interval over seeds and tracks, number of seeds).",
    ),
    "tokens": (
        tokens_table,
        "Exact token agreement (percent, mean $\\pm$ s.d. over seeds).",
    ),
    "augmentation": (
        augmentation_table,
        "Condition replay under degradation: ours against ours trained on degraded latent views (genre "
        "split ID test, 32 tracks, three seeds each; difference paired by track).",
    ),
    "mm3": (
        mm3_table,
        "MM3-OOD: 240 tracks generated for this study from captions outside the corpus genres. Replay "
        "(mean and 95\\% interval over seeds and tracks) and exact semantic agreement.",
    ),
    "robustness": (
        robustness_table,
        "Condition replay of codes encoded from degraded held-out audio (64 tracks).",
    ),
    "decoding": (
        decoding_table,
        "Generator-completed decoding: the encoder's semantic codes with acoustic codes chosen by "
        "MiniMax's own depth decoder, sequentially.",
    ),
    "reference": (
        reference_table,
        "Reference-guided generation for a caption of another genre (Song Describer).",
    ),
    "continuation": (
        continuation_table,
        "Audio-prompted continuation of Song Describer recordings (20 s after a 10 s prompt).",
    ),
}


def main() -> None:
    """Write every table as LaTeX into --tex and all of them as one Markdown file, --markdown."""
    parser = argparse.ArgumentParser(prog="tables")
    parser.add_argument("--tex", type=Path, default=ROOT / "outputs" / "tables")
    parser.add_argument("--markdown", type=Path, default=ROOT / "outputs" / "results.md")
    args = parser.parse_args()
    args.tex.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Results",
        "",
        "Generated by tables.py from the per track result files under results/; do not edit by hand.",
    ]
    for name, (build, caption) in TABLES.items():
        try:
            header, rows = build()
        except MissingInputError as error:
            print(f"skip {name}: missing {error}")
            continue
        (args.tex / f"{name}.tex").write_text(
            latex(header, rows, label=name, caption=caption), encoding="utf-8"
        )
        plain = caption.replace("\\%", "%").replace("\\dagger", "+").replace("$", "")
        sections += ["", f"## {name}", "", plain, "", markdown(header, rows)]
        print(f"wrote {name} ({len(rows)} rows)")
    args.markdown.write_text("\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
