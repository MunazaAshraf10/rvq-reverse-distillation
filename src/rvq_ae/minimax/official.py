"""The released MiniMax Music 3 components, driven in teacher forced mode.

Generation samples one RVQ frame at a time: the language model predicts the semantic code from its
last hidden state, the depth decoder samples the seven acoustic codes, and the eight hidden states
of that frame (one language model state, seven depth states) are what the condition encoder turns
into the conditioning of the diffusion transformer. Given a complete code sequence the same hidden
states come out of one parallel forward pass of each model, which is what this module computes.

The prompt template, the code offsets and the frame embedding are taken from the diffusers
integration (diffusers.modular_pipelines.minimax_music3), pinned to the release in pyproject.toml,
so that every step matches the official sampling loop.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from diffusers import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)
from diffusers.modular_pipelines.minimax_music3 import encoders as official
from huggingface_hub import snapshot_download
from torch import Tensor
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

from rvq_ae.audio.timeline import Chunk

MINIMAX_REPO = "MiniMaxAI/MiniMax-Music3"
MINIMAX_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
"""Revision used for every replay, rendering and generation result in this repository."""

MINIMAX_FILES = [
    f"{folder}/*"
    for folder in (
        "tokenizer",
        "language_model",
        "rvq_depth_decoder",
        "condition_encoder",
        "transformer",
        "vocoder",
        "scheduler",
    )
]
"""The diffusers layout only; the repository also holds a duplicate Qwen folder and raw .pth files."""

CODE_OFFSET = official._AUDIO_CODE_OFFSET
END_TOKEN = official._AUDIO_END_TOKEN_ID
CFG_TOKEN = official._AUDIO_CFG_TOKEN_ID
SEMANTIC_VOCAB = official._SEMANTIC_VOCAB_SIZE
AR_CFG_SCALE = official._AR_CFG_SCALE
AR_TOP_K = official._AR_CFG_TOP_K

CHUNK_FRAMES = 200
CHUNK_HOP = 100
OVERLAP_LATENTS = 172
"""Rendering windows: 200 frames every 100 frames; neighbours share 172 latents of conditioning."""

CROP_LEFT = 86
CROP_RIGHT = 344 - 86
"""Latents dropped at the left of every window but the first and at the right of every window but the last."""


def prompt_text(caption: str, lyrics: str) -> str:
    """The checkpoint's special token prompt, byte for byte as the official tokenize step builds it."""
    return (
        f"{official._IM_START}{official._CAPTION_START}{official._clean_caption(caption)}{official._CAPTION_END}"
        f"{official._LYRICS_START}{official._normalize_lyrics(lyrics)}{official._LYRICS_END}"
        f"{official._IM_END}{official._AUDIO_START}"
    )


def chunk_starts(frames: int) -> list[int]:
    """First frame of every rendering window, as the official chunk preparation step computes it."""
    return [0] if frames <= CHUNK_FRAMES else list(range(0, frames - CHUNK_HOP, CHUNK_HOP))


def kept_spans(lengths: list[int]) -> list[tuple[int, int]]:
    """Kept [start, end) latents of each window, following the official waveform cropping."""
    last = len(lengths) - 1
    return [
        (0 if index == 0 else CROP_LEFT, length if index == last else length - CROP_RIGHT)
        for index, length in enumerate(lengths)
    ]


def check_stitching(spans: list[tuple[int, int]], chunks: tuple[Chunk, ...]) -> None:
    """Fail loudly when the recorded stitching table disagrees with the official crop rule."""
    ordered = sorted(chunks, key=lambda chunk: chunk.index)
    if len(ordered) != len(spans):
        raise ValueError(f"{len(ordered)} recorded chunks, {len(spans)} rendering windows")
    position = 0
    for (start, end), chunk in zip(spans, ordered, strict=True):
        if (chunk.latent_start, chunk.latent_end) != (position, position + end - start):
            raise ValueError(f"chunk {chunk.index}: recorded stitching disagrees with the crop rule")
        position += end - start


@dataclass(slots=True)
class Analysis:
    """Per frame outputs of one teacher forced pass."""

    hiddens: Tensor
    """[frames, 8 * 4096] condition encoder input, language model state first."""
    semantic_nll: Tensor
    """[frames] negative log likelihood of the semantic code under the conditional language model."""
    acoustic_nll: Tensor
    """[frames, 7] negative log likelihood of each acoustic code under the depth decoder."""


class MiniMax:
    """The language model, depth decoder and condition encoder, plus the renderer when requested."""

    def __init__(
        self,
        tokenizer: Qwen2Tokenizer,
        language_model: Qwen3ForCausalLM,
        depth: MiniMaxMusic3RVQDepthDecoder,
        condition: MiniMaxMusic3ConditionEncoder,
        *,
        transformer: MiniMaxMusic3Transformer1DModel | None = None,
        vocoder: MiniMaxMusic3Vocoder | None = None,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.language_model = language_model.eval()
        self.depth = depth.eval()
        self.condition = condition.eval()
        self.transformer = None if transformer is None else transformer.eval()
        self.vocoder = None if vocoder is None else vocoder.eval()
        self.device = device

    @classmethod
    def load(
        cls,
        *,
        device: str | torch.device = "cuda",
        render: bool = False,
        repo: str = MINIMAX_REPO,
        revision: str = MINIMAX_REVISION,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Self:
        """Load the analysis path, and the diffusion transformer and vocoder when render is true.

        The vocoder runs in float32 as in the official pipeline; everything else in bfloat16.
        """
        target = torch.device(device)
        root = Path(snapshot_download(repo, revision=revision, allow_patterns=MINIMAX_FILES))
        tokenizer = Qwen2Tokenizer.from_pretrained(root / "tokenizer")
        language_model = Qwen3ForCausalLM.from_pretrained(root / "language_model", dtype=dtype).to(target)
        depth = MiniMaxMusic3RVQDepthDecoder.from_pretrained(root / "rvq_depth_decoder", torch_dtype=dtype)
        condition = MiniMaxMusic3ConditionEncoder.from_pretrained(
            root / "condition_encoder", torch_dtype=dtype
        )
        transformer = vocoder = None
        if render:
            transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
                root / "transformer", torch_dtype=dtype
            )
            vocoder = MiniMaxMusic3Vocoder.from_pretrained(root / "vocoder", torch_dtype=torch.float32)
        return cls(
            tokenizer,
            language_model,
            depth.to(target),
            condition.to(target),
            transformer=transformer,
            vocoder=vocoder,
            device=target,
        )

    def place(self, stage: str) -> None:
        """Keep only one stage on the device: analysis (language model, depth decoder, condition
        encoder) or render (flow transformer, vocoder). The 8B language model and the 2.4B flow
        transformer do not fit a 24 GB card together, so batch experiments alternate stages.
        """
        analysis = (self.language_model, self.depth, self.condition)
        rendering = tuple(module for module in (self.transformer, self.vocoder) if module is not None)
        if stage not in ("analysis", "render"):
            raise ValueError(f"unknown stage {stage!r}")
        idle, active = (rendering, analysis) if stage == "analysis" else (analysis, rendering)
        for module in idle:
            module.to("cpu")
        torch.cuda.empty_cache()
        for module in active:
            module.to(self.device)

    def prompt_ids(self, caption: str, lyrics: str) -> Tensor:
        """Conditional prompt token ids [1, P]."""
        ids: Tensor = self.tokenizer(prompt_text(caption, lyrics), return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def frame_embeddings(self, rows: Tensor) -> Tensor:
        """Language model input embeddings [rows, H] of complete code frames [rows, 8]."""
        return official._embed_audio_frame(self.language_model, self.depth, rows)[:, 0]

    @torch.no_grad()
    def analyse(self, codes: Tensor, caption: str, lyrics: str, *, chunk: int = 1024) -> Analysis:
        """Teacher forced hidden states and code likelihoods of an emitted code sequence.

        codes is [n + 1, 8] including the priming row 0, so emitted frame i is row i + 1. The language
        model reads the prompt followed by rows 0 .. n - 1, and its state at prompt position P - 1 + r
        is the state that sampled row r; emitted frame i therefore takes position P + i. The depth
        decoder then reads [state, c_0, c_1, ..., c_6] for each frame with a causal mask, and its
        positions 1 .. 7 are the states that sampled c_1 .. c_7.
        """
        codes = codes.to(self.device, torch.long)
        frames = codes.shape[0] - 1
        states = self.language_states(codes, caption, lyrics)
        emitted = codes[1:]
        readout = self.language_model.lm_head.weight[CODE_OFFSET : CODE_OFFSET + SEMANTIC_VOCAB]
        hidden_parts: list[Tensor] = []
        semantic: list[Tensor] = []
        acoustic: list[Tensor] = []
        for start in range(0, frames, chunk):
            part = slice(start, min(start + chunk, frames))
            logits = (states[part] @ readout.T).float()
            semantic.append(F.cross_entropy(logits, emitted[part, 0], reduction="none"))
            depth_states, nll = self.depth_pass(states[part], emitted[part])
            hidden_parts.append(torch.cat([states[part], depth_states], dim=-1))
            acoustic.append(nll)
        return Analysis(torch.cat(hidden_parts), torch.cat(semantic), torch.cat(acoustic))

    def depth_pass(self, states: Tensor, codes: Tensor) -> tuple[Tensor, Tensor]:
        """Depth decoder states [frames, 7 * 4096] and acoustic code NLL [frames, 7], teacher forced."""
        depth = self.depth
        vocab = depth.config.audio_vocab_size
        steps = [states, self.language_model.model.embed_tokens(codes[:, 0] + CODE_OFFSET)]
        for book in range(1, codes.shape[1] - 1):
            steps.append(depth.audio_embeddings(codes[:, book] + (book - 1) * vocab))
        sequence = depth.projection(torch.stack(steps, dim=1))
        hidden = depth(sequence)[:, 1:]
        nll = torch.stack(
            [
                F.cross_entropy(head(hidden[:, index]).float(), codes[:, index + 1], reduction="none")
                for index, head in enumerate(depth.audio_heads)
            ],
            dim=-1,
        )
        return hidden.flatten(1), nll

    @torch.no_grad()
    def language_states(self, codes: Tensor, caption: str, lyrics: str) -> Tensor:
        """Teacher forced language model states [frames, 4096] that sampled the emitted frames."""
        codes = codes.to(self.device, torch.long)
        prompt = self.prompt_ids(caption, lyrics)
        model = self.language_model.model
        inputs = torch.cat([model.embed_tokens(prompt)[0], self.frame_embeddings(codes[:-1])])
        states = model(inputs_embeds=inputs[None], use_cache=False).last_hidden_state[0]
        return states[prompt.shape[1] : prompt.shape[1] + codes.shape[0] - 1]

    @torch.no_grad()
    def sample_acoustic(
        self,
        states: Tensor,
        semantic: Tensor,
        *,
        generator: torch.Generator | None = None,
        greedy: bool = False,
        top_k: int = AR_TOP_K,
        chunk: int = 1024,
    ) -> Tensor:
        """Acoustic codes [frames, 7] drawn from the official depth decoder given states and c_0.

        This is the generator's own conditional distribution over the acoustic books (conditional
        branch, top 50 sampling at temperature 1, or argmax when greedy), i.e. an alternative valid
        completion of the frame rather than the one that was sampled.
        """
        depth = self.depth
        vocab = depth.config.audio_vocab_size
        out: list[Tensor] = []
        for start in range(0, states.shape[0], chunk):
            part = slice(start, min(start + chunk, states.shape[0]))
            steps = [
                states[part],
                self.language_model.model.embed_tokens(semantic[part].to(self.device) + CODE_OFFSET),
            ]
            codes: list[Tensor] = []
            for book in range(1, depth.config.num_codebooks):
                hidden = depth(depth.projection(torch.stack(steps, dim=1)))[:, -1]
                logits = depth.audio_heads[book - 1](hidden).float()
                if greedy:
                    code = logits.argmax(dim=-1)
                else:
                    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                    probs = torch.softmax(logits.masked_fill(logits < threshold, -float("inf")), dim=-1)
                    where = generator.device if generator is not None else probs.device
                    code = torch.multinomial(probs.to(where), 1, generator=generator)[:, 0].to(self.device)
                codes.append(code)
                if book < depth.config.num_codebooks - 1:
                    steps.append(depth.audio_embeddings(code + (book - 1) * vocab))
            out.append(torch.stack(codes, dim=-1))
        return torch.cat(out).cpu()

    @torch.no_grad()
    def chunk_conditions(self, hiddens: Tensor) -> list[Tensor]:
        """Conditioning [latents, 2048] of each rendering window, with the official overlap splice."""
        conditions: list[Tensor] = []
        previous: Tensor | None = None
        frames = hiddens.shape[0]
        for start in chunk_starts(frames):
            window = hiddens[start : min(start + CHUNK_FRAMES, frames)]
            # A 3 tap convolution over at most 200 frames: cuDNN buys nothing here, and its workspace
            # allocation fails when a long rollout has left the 24 GB card nearly full.
            with torch.backends.cudnn.flags(enabled=False):
                condition = self.condition(window[None].to(self.device))[0].to(torch.bfloat16)
            if previous is not None:
                overlap = min(previous.shape[0], condition.shape[0])
                condition[:overlap] = previous[:overlap]
            length = condition.shape[0]
            low = max(0, length - 2 * OVERLAP_LATENTS)
            previous = condition[low : max(low, length - OVERLAP_LATENTS)]
            conditions.append(condition)
        return conditions

    def stitched_condition(self, hiddens: Tensor, chunks: tuple[Chunk, ...] | None = None) -> Tensor:
        """Conditioning on the stitched latent timeline [latents, 2048], as the generator stored it."""
        conditions = self.chunk_conditions(hiddens)
        spans = kept_spans([condition.shape[0] for condition in conditions])
        if chunks:
            check_stitching(spans, chunks)
        return torch.cat(
            [condition[start:end] for condition, (start, end) in zip(conditions, spans, strict=True)]
        )
