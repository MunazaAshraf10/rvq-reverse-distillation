"""Render per window conditioning to a waveform with the official flow transformer and vocoder.

This is the official chunk loop of diffusers.modular_pipelines.minimax_music3 written out without
the modular pipeline machinery: every 200 frame window is flow matched for 30 Euler steps with
classifier free guidance 1.7 against an all zero condition, its first 172 latents are blended
toward the previous window's carry at every step, and the vocoded windows are cropped and joined.
The conditional and unconditional passes run as one batch of two, which is the same computation
as the guider's two passes.
"""

from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from torch import Tensor

from rvq_ae.minimax.official import (
    CROP_LEFT,
    CROP_RIGHT,
    MINIMAX_REPO,
    MINIMAX_REVISION,
    OVERLAP_LATENTS,
    MiniMax,
)

FLOW_STEPS = 30
FLOW_GUIDANCE = 1.7


def load_scheduler(revision: str = MINIMAX_REVISION) -> FlowMatchEulerDiscreteScheduler:
    root = Path(snapshot_download(MINIMAX_REPO, revision=revision, allow_patterns=["scheduler/*"]))
    scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        root / "scheduler"
    )
    return scheduler


@torch.no_grad()
def render(
    minimax: MiniMax,
    conditions: list[Tensor],
    *,
    seed: int,
    steps: int = FLOW_STEPS,
    guidance: float = FLOW_GUIDANCE,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
) -> Tensor:
    """Stereo waveform [2, samples] at 44.1 kHz from the window conditions of MiniMax.chunk_conditions.

    The noise of every window is drawn from one CPU generator seeded with seed, so a render is a
    deterministic function of (conditions, seed).
    """
    if minimax.transformer is None or minimax.vocoder is None:
        raise ValueError("load MiniMax with render=True")
    transformer, vocoder, device = minimax.transformer, minimax.vocoder, minimax.device
    scheduler = scheduler or load_scheduler()
    generator = torch.Generator("cpu").manual_seed(seed)
    channels = transformer.config.in_channels
    previous: Tensor | None = None
    chunks: list[Tensor] = []
    for condition in conditions:
        condition = condition.to(device, transformer.dtype)[None]
        length = condition.shape[1]
        noise = torch.randn((1, channels, length), generator=generator).to(device, transformer.dtype)
        overlap = 0 if previous is None else min(previous.shape[-1], length)
        prompt = noise[..., :overlap].clone()
        latents = noise
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), device=device)
        pair = torch.cat([condition, torch.zeros_like(condition)])
        for timestep in scheduler.timesteps:
            if previous is not None and overlap > 0:
                time = timestep.to(latents.dtype)
                latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time) * prompt + time * previous[..., :overlap]
            prediction = transformer(
                hidden_states=latents.expand(2, -1, -1),
                timestep=timestep.expand(2).to(latents.dtype),
                encoder_hidden_states=pair,
                return_dict=False,
            )[0]
            conditional, unconditional = prediction.chunk(2)
            velocity = unconditional + guidance * (conditional - unconditional)
            latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
        if previous is not None and overlap > 0:
            latents[..., :overlap] = previous[..., :overlap]
        low = max(0, length - 2 * OVERLAP_LATENTS)
        previous = latents[..., low : max(low, length - OVERLAP_LATENTS)]
        chunks.append(latents)
    hop = int(np.prod(vocoder.config.upsampling_ratios))
    pieces: list[Tensor] = []
    for index, latents in enumerate(chunks):
        waveform = vocoder(latents.float())[0]
        left = 0 if index == 0 else CROP_LEFT * hop
        right = 0 if index == len(chunks) - 1 else CROP_RIGHT * hop
        pieces.append(waveform[..., left : waveform.shape[-1] - right])
    return torch.cat(pieces, dim=-1).float().clamp(-1.0, 1.0).cpu()
