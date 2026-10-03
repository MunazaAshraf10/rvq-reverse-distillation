"""Train and evaluation splits of the reverse distillation corpus.

published   the corpus' own 5 percent hash split (salt holdout-v1). Tracks are disjoint, but the
            held-out prompts and lyrics largely recur in training, so it measures generalisation
            across generations of the same captions.
genre-ood   leave one genre out. The tracks of the held-out genre whose prompt never occurs in
            another genre form the out of distribution test set; the published held-out tracks of
            the other genres form an in-distribution test set for the same models. Training then
            drops every track that shares a lyric text with either test set, so the out of
            distribution set shares neither a prompt nor a lyric with training and the
            in-distribution set shares prompts (as intended) but no lyric.
"""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rvq_ae.data.records import Record

PUBLISHED = "published"
GENRE_OOD = "genre-ood"
RULES = (PUBLISHED, GENRE_OOD)

TRAIN = "train"
HOLDOUT = "holdout"
ID_TEST = "test-id"
OOD_TEST = "test-ood"

HELD_OUT_GENRE = "jazz"
"""The genre left out by genre-ood; jazz is the corpus genre furthest from rock, pop and blues."""

SPACE = re.compile(r"\s+")
PUNCTUATION = re.compile(r"[^\w\s]+")
TAG = re.compile(r"\[[^\]]*\]")


def prompt_key(prompt: str) -> str:
    """Prompt identity up to case, punctuation and whitespace ("Rock," and " rock" are one prompt)."""
    return SPACE.sub(" ", PUNCTUATION.sub(" ", prompt.lower())).strip()


def lyric_key(lyrics: str) -> str:
    """Lyric identity up to case, whitespace and section tags; empty for instrumentals."""
    text = SPACE.sub(" ", PUNCTUATION.sub(" ", TAG.sub(" ", lyrics.lower()))).strip()
    return hashlib.sha1(text.encode()).hexdigest() if text else ""


@dataclass(frozen=True, slots=True)
class Split:
    rule: str
    parts: dict[str, list[Record]]
    dropped: int
    """Training tracks removed because they share a lyric with a test track."""

    def summary(self) -> dict[str, int]:
        return {name: len(records) for name, records in self.parts.items()} | {"dropped": self.dropped}


def lyric_disjoint(train: Sequence[Record], tests: Sequence[Record]) -> tuple[list[Record], int]:
    """Training tracks sharing no lyric text with any test track, and how many were dropped."""
    lyrics = {key for key in (lyric_key(record.lyrics) for record in tests) if key}
    kept = [record for record in train if lyric_key(record.lyrics) not in lyrics]
    return kept, len(train) - len(kept)


def make_split(grouped: dict[str, list[Record]], rule: str, *, genre: str = HELD_OUT_GENRE) -> Split:
    """Apply a split rule to records grouped by their published split."""
    if rule == PUBLISHED:
        return Split(rule, {TRAIN: grouped[TRAIN], HOLDOUT: grouped[HOLDOUT]}, 0)
    if rule != GENRE_OOD:
        raise ValueError(f"unknown split rule {rule!r}; expected one of {RULES}")
    records = sorted((r for part in grouped.values() for r in part), key=lambda r: r.shard_id)
    if not any(record.genre == genre for record in records):
        raise ValueError(f"no track carries the genre {genre!r}")
    other_prompts = {prompt_key(r.prompt) for r in records if r.genre != genre}
    ood = [r for r in records if r.genre == genre and prompt_key(r.prompt) not in other_prompts]
    in_distribution = [r for r in grouped[HOLDOUT] if r.genre != genre]
    candidates = [r for r in grouped[TRAIN] if r.genre != genre]
    train, dropped = lyric_disjoint(candidates, ood + in_distribution)
    return Split(rule, {TRAIN: train, ID_TEST: in_distribution, OOD_TEST: ood}, dropped)
