"""Small metadata files the experiments need, fetched once from pinned sources and verified.

    mtg-jamendo genre tags   Song Describer release on Zenodo (record 10072001), md5 checked
    MusicCaps captions       google/MusicCaps on the Hub, pinned revision (CC BY-SA 4.0)
    public domain poems      DanFosing/public-domain-poetry on the Hub, pinned revision (CC0)

Downloads land in $RVQ_AE_CACHE (default ~/.cache/rvq_ae) or the Hugging Face cache.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

JAMENDO_URL = "https://zenodo.org/api/records/10072001/files/song_describer_14_04_23.mtg-jamendo.tsv/content"
JAMENDO_MD5 = "3532f2df8b4c21a7ea85d9121eae244a"
MUSICCAPS = ("google/MusicCaps", "musiccaps-public.csv", "0a51889b340037bb75a9a0858af2e4ece21f7f89")
POEMS = ("DanFosing/public-domain-poetry", "poems.json", "84a87909d09ff0c3ae040c4e0af25a6344d96531")


def cache_dir() -> Path:
    path = Path(os.environ.get("RVQ_AE_CACHE", Path.home() / ".cache" / "rvq_ae"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def jamendo_tags() -> Path:
    """The MTG-Jamendo metadata of the Song Describer release, downloaded once and md5 checked."""
    path = cache_dir() / "song_describer_14_04_23.mtg-jamendo.tsv"
    if not path.is_file():
        data = urllib.request.urlopen(JAMENDO_URL, timeout=120).read()
        if hashlib.md5(data).hexdigest() != JAMENDO_MD5:
            raise ValueError("the downloaded MTG-Jamendo metadata does not match the published md5")
        path.write_bytes(data)
    return path


def hub_file(spec: tuple[str, str, str]) -> Path:
    repo, filename, revision = spec
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def musiccaps() -> Path:
    return hub_file(MUSICCAPS)


def poems() -> Path:
    return hub_file(POEMS)
