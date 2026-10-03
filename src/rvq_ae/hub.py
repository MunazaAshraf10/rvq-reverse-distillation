import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from rvq_ae.constants import COLLECTION, LEGACY_WEIGHTS_FORMAT, MODEL_REPO, WEIGHTS_FORMAT
from rvq_ae.models.config import EncoderConfig
from rvq_ae.models.encoder import RvqEncoder

# Two checkpoint layouts are read: the per run layout the trainer writes and the model repository uses
# (rvq_encoder_config.json and rvq_encoder.safetensors), and the preliminary collection
# (encoders/<name>.json and .safetensors), addressed by variant.
CONFIG_NAME = "rvq_encoder_config.json"
WEIGHTS_NAME = "rvq_encoder.safetensors"
STATE_NAME = "trainer_state.json"

VARIANTS = {
    "v1": "encoders/minimax_music3_rvq_encoder_v1_41m_independent_heads",
    "v2": "encoders/minimax_music3_rvq_encoder_v2_155m_wide_independent_heads",
    "v3": "encoders/minimax_music3_rvq_encoder_v3_155m_mert_aligned_independent_heads",
    "v4": "encoders/minimax_music3_rvq_encoder_v4_169m_autoregressive_depth_recommended",
}


@dataclass(frozen=True, slots=True)
class EncoderFiles:
    config: Path
    weights: Path


def relative_files(variant: str | None, subfolder: str | None) -> tuple[str, str]:
    if variant is not None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        stem = VARIANTS[variant]
        return f"{stem}.json", f"{stem}.safetensors"
    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    return prefix + CONFIG_NAME, prefix + WEIGHTS_NAME


def resolve_encoder(
    source: str | Path = MODEL_REPO,
    *,
    variant: str | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
) -> EncoderFiles:
    """Locate config and weights in a local folder or a Hub repository.

    A variant names a preliminary checkpoint and reads from the collection unless a local folder
    is given; otherwise the files sit at the root of the source or of its subfolder.
    """
    root = Path(source)
    if root.is_dir():
        config, weights = relative_files(variant, subfolder)
        files = EncoderFiles(root / config, root / weights)
        for path in (files.config, files.weights):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found")
        return files
    if variant is not None and str(source) == MODEL_REPO:
        source = COLLECTION
    config, weights = relative_files(variant, subfolder)
    return EncoderFiles(
        Path(hf_hub_download(str(source), config, revision=revision)),
        Path(hf_hub_download(str(source), weights, revision=revision)),
    )


def check_format(path: Path) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    tag = metadata.get("format")
    if tag not in (WEIGHTS_FORMAT, LEGACY_WEIGHTS_FORMAT):
        raise ValueError(f"{path} carries format {tag!r}, expected {WEIGHTS_FORMAT!r}")


def load_encoder(
    source: str | Path = MODEL_REPO,
    *,
    variant: str | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
    device: str | torch.device = "cpu",
) -> RvqEncoder:
    """Frozen encoder in float32 with strict key matching."""
    files = resolve_encoder(source, variant=variant, subfolder=subfolder, revision=revision)
    check_format(files.weights)
    cfg = EncoderConfig.load(files.config)
    model = RvqEncoder(cfg)
    model.load_state_dict(load_file(files.weights, device="cpu"), strict=True)
    model.requires_grad_(False)
    return model.eval().to(device)


def export_state(model: nn.Module) -> dict[str, Tensor]:
    """Float32 contiguous CPU tensors with any DistributedDataParallel prefix removed."""
    return {
        key.removeprefix("module."): value.detach().to("cpu", torch.float32).contiguous()
        for key, value in model.state_dict().items()
    }


def save_encoder(model: RvqEncoder, folder: Path) -> EncoderFiles:
    folder.mkdir(parents=True, exist_ok=True)
    files = EncoderFiles(folder / CONFIG_NAME, folder / WEIGHTS_NAME)
    model.cfg.save(files.config)
    save_file(export_state(model), files.weights, metadata={"format": WEIGHTS_FORMAT})
    return files


def push_folder(
    folder: Path,
    repo_id: str,
    *,
    path_in_repo: str = "",
    private: bool = False,
    message: str = "Add RVQ encoder checkpoint",
) -> str:
    """Upload config, weights and trainer state of one checkpoint folder; returns the commit url."""
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[CONFIG_NAME, WEIGHTS_NAME, STATE_NAME],
        commit_message=message,
    )
    return str(commit.commit_url)


def read_state(folder: Path) -> dict[str, object]:
    return dict(json.loads((folder / STATE_NAME).read_text(encoding="utf-8")))
