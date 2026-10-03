"""The official autoregressive sampling loop, with the two ways a reference recording can steer it.

The loop is the official one (diffusers.modular_pipelines.minimax_music3.encoders): a conditional
and an unconditional prompt run side by side with a key value cache; each step samples the
semantic code from classifier free guided logits (scale 1.5, top 50) and the seven acoustic codes
from the guided depth decoder, and the frame's conditional hidden states become the conditioning
of the renderer. Frame 0 is the priming frame and is not emitted.

A reference enters in one of two ways:
    constraint   emitted frame i with i % interval == 0 samples its semantic code only among the
                 reference encoder's candidates for frame i (top 5), as in the released integration.
    prefix       the first emitted frames are forced to the reference codes (all eight books) and
                 the language model continues from that history: audio prompted continuation.

Several streams that share one prompt are decoded as one batch (rows: the conditional branch of
every stream, then the unconditional ones). A decoding step is bound by kernel launches rather than
arithmetic at this batch size, so S streams cost little more than one. The rollout also records
what the reverse distillation corpus stores: the sampled codes including the priming row and the
post guidance top 50 ids and logits of every book.
"""

from dataclasses import dataclass

import torch
from diffusers.modular_pipelines.minimax_music3 import encoders as official
from torch import Tensor

from rvq_ae.minimax.official import (
    AR_CFG_SCALE,
    AR_TOP_K,
    CFG_TOKEN,
    CODE_OFFSET,
    END_TOKEN,
    SEMANTIC_VOCAB,
    MiniMax,
)

TEACHER_TOP_K = 50


@dataclass(frozen=True, slots=True)
class Stream:
    """How one sequence of a batched rollout is steered; the default samples freely."""

    candidates: Tensor | None = None
    """[frames, k] semantic candidates, used at every interval-th emitted frame."""
    interval: int = 0
    prefix: Tensor | None = None
    """[m, 8] codes forced at the first m emitted frames."""
    greedy_depth: bool = False
    """Argmax of the guided depth decoder instead of sampling it."""
    priming: Tensor | None = None
    """[8] replaces the sampled priming frame, for example by a track's recorded one."""

    def forced(self, emitted: int) -> Tensor | None:
        if emitted < 0:
            return self.priming
        if self.prefix is not None and emitted < self.prefix.shape[0]:
            return self.prefix[emitted]
        return None

    def constrained(self, emitted: int) -> bool:
        """Whether emitted frame emitted is restricted; frames past the reference are left free."""
        if self.candidates is None or self.interval <= 0 or emitted < 0:
            return False
        return emitted < self.candidates.shape[0] and emitted % self.interval == 0


@dataclass(slots=True)
class Rollout:
    codes: Tensor
    """[frames + 1, 8] sampled codes, priming row first."""
    hiddens: Tensor
    """[frames, 8 * 4096] conditioning input of every emitted frame."""
    topk_ids: Tensor
    """[frames + 1, 8, 50] post guidance top 50 ids per book (semantic ids without the code offset)."""
    topk_logits: Tensor
    """[frames + 1, 8, 50] the matching guided logits."""
    ended: bool
    """True when the language model emitted the end of audio token."""


def text_ids(minimax: MiniMax, caption: str, lyrics: str) -> Tensor:
    """[2, P] conditional prompt and its classifier free counterpart, as the official tokenize step."""
    ids = minimax.prompt_ids(caption, lyrics)
    unconditional = ids.clone()
    unconditional[:, 1:-2] = CFG_TOKEN
    return torch.cat([ids, unconditional])


def guided(logits: Tensor, streams: int) -> Tensor:
    """Classifier free guided logits [S, V] from rows [conditional x S, unconditional x S]."""
    conditional, unconditional = logits[:streams].float(), logits[streams:].float()
    return unconditional + (conditional - unconditional) * AR_CFG_SCALE


def semantic_code_ids(tokens: Tensor) -> Tensor:
    """Vocabulary ids to semantic code ids; the end of audio token becomes 16384, as in the corpus."""
    return torch.where(tokens == END_TOKEN, SEMANTIC_VOCAB, tokens - CODE_OFFSET)


def sample(scores: Tensor, generator: torch.Generator, greedy: Tensor) -> Tensor:
    """One code per row [S]: the official top 50 sample, or the argmax where greedy [S] is set."""
    drawn = official._sample_top_k(scores, generator)
    return torch.where(greedy, scores.argmax(dim=-1), drawn)


@torch.no_grad()
def rollout_streams(
    minimax: MiniMax,
    caption: str,
    lyrics: str,
    streams: list[Stream],
    *,
    frames: int,
    generator: torch.Generator,
    stop_at_end: bool = False,
) -> list[Rollout]:
    """Decode len(streams) sequences of up to frames emitted frames under one prompt.

    stop_at_end lets the end of audio token finish the rollout; it needs a single stream. The token
    is never sampled inside a forced frame.
    """
    if stop_at_end and len(streams) != 1:
        raise ValueError("stop_at_end needs exactly one stream")
    lm, depth, device = minimax.language_model, minimax.depth, minimax.device
    count = len(streams)
    ids = text_ids(minimax, caption, lyrics)
    ids = torch.cat([ids[:1].expand(count, -1), ids[1:].expand(count, -1)])
    output = lm.model(inputs_embeds=lm.model.embed_tokens(ids), use_cache=True)
    cache, state = output.past_key_values, output.last_hidden_state[:, -1]
    allowed = torch.zeros(lm.config.vocab_size, dtype=torch.bool, device=device)
    allowed[CODE_OFFSET : CODE_OFFSET + SEMANTIC_VOCAB] = True
    allowed[END_TOKEN] = True
    greedy = torch.tensor([stream.greedy_depth for stream in streams], device=device)
    never = torch.zeros(count, dtype=torch.bool, device=device)
    rows: list[Tensor] = []
    hiddens: list[Tensor] = []
    top_ids: list[Tensor] = []
    top_values: list[Tensor] = []
    ended = False
    for step in range(frames + 1):
        emitted = step - 1
        forced = [stream.forced(emitted) for stream in streams]
        logits = lm.lm_head(state).float().masked_fill(~allowed, -float("inf"))
        base = guided(logits, count)
        threshold = logits[:count].topk(AR_TOP_K, dim=-1).values[:, -1:]
        scores = base.masked_fill(logits[:count] < threshold, -float("inf")).masked_fill(
            ~allowed, -float("inf")
        )
        for index, stream in enumerate(streams):
            if stream.candidates is not None and stream.constrained(emitted):
                # The released integration applies guidance over the candidates alone, so a candidate
                # outside the conditional top 50 stays eligible.
                keep = torch.zeros_like(allowed)
                keep[stream.candidates[emitted].to(device) + CODE_OFFSET] = True
                scores[index] = base[index].masked_fill(~keep, -float("inf"))
            if forced[index] is not None or not stop_at_end:
                scores[index, END_TOKEN] = -float("inf")
        values, best = scores.topk(TEACHER_TOP_K, dim=-1)
        tokens = sample(scores, generator, never)
        for index, frame in enumerate(forced):
            if frame is not None:
                tokens[index] = frame[0].to(device) + CODE_OFFSET
        if stop_at_end and int(tokens[0].item()) == END_TOKEN:
            ended = True
            break
        semantic = tokens - CODE_OFFSET
        codes, depth_hidden, book_ids, book_values = depth_step(
            minimax, state, semantic, generator, forced, greedy
        )
        rows.append(codes.cpu())
        top_ids.append(torch.cat([semantic_code_ids(best)[:, None], book_ids], dim=1).cpu())
        top_values.append(torch.cat([values[:, None], book_values], dim=1).cpu())
        if step > 0:
            hiddens.append(torch.cat([state[:count], depth_hidden], dim=-1).cpu())
        feedback = official._embed_audio_frame(lm, depth, torch.cat([codes, codes]))
        output = lm.model(inputs_embeds=feedback, past_key_values=cache, use_cache=True)
        cache, state = output.past_key_values, output.last_hidden_state[:, -1]
    del cache, output
    torch.cuda.empty_cache()
    return [
        Rollout(
            codes=torch.stack([row[index] for row in rows]),
            hiddens=torch.stack([hidden[index] for hidden in hiddens]) if hiddens else torch.empty(0),
            topk_ids=torch.stack([row[index] for row in top_ids]),
            topk_logits=torch.stack([row[index] for row in top_values]),
            ended=ended,
        )
        for index in range(count)
    ]


def rollout(
    minimax: MiniMax,
    caption: str,
    lyrics: str,
    *,
    frames: int,
    generator: torch.Generator,
    candidates: Tensor | None = None,
    interval: int = 0,
    prefix: Tensor | None = None,
    stop_at_end: bool = True,
    greedy_depth: bool = False,
    priming: Tensor | None = None,
) -> Rollout:
    """A single stream rollout; see Stream for the steering arguments. Candidates of width one with
    interval one force the semantic stream and leave the acoustic books to the generator: generator
    completed decoding."""
    stream = Stream(candidates, interval, prefix, greedy_depth, priming)
    return rollout_streams(
        minimax, caption, lyrics, [stream], frames=frames, generator=generator, stop_at_end=stop_at_end
    )[0]


def depth_step(
    minimax: MiniMax,
    state: Tensor,
    semantic: Tensor,
    generator: torch.Generator,
    forced: list[Tensor | None],
    greedy: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One frame of the guided depth decoder for S streams.

    Returns codes [S, 8], conditional hidden states [S, 7 * 4096], and the guided top 50 ids and
    logits [S, 7, 50]. Mirrors official._generate_depth_codes; rows of forced frames keep their
    codes instead of sampling.
    """
    depth, lm = minimax.depth, minimax.language_model
    count = semantic.shape[0]
    vocab = depth.config.audio_vocab_size
    sequence = [depth.projection(state).unsqueeze(1)]
    both = torch.cat([semantic, semantic])
    sequence.append(depth.projection(lm.model.embed_tokens(both + CODE_OFFSET)).unsqueeze(1))
    codes = [semantic]
    hidden_parts: list[Tensor] = []
    book_ids: list[Tensor] = []
    book_values: list[Tensor] = []
    for book in range(1, depth.config.num_codebooks):
        hidden = depth(torch.cat(sequence, dim=1))[:, -1]
        hidden_parts.append(hidden[:count])
        scores = guided(depth.audio_heads[book - 1](hidden), count)
        values, ids = scores.topk(TEACHER_TOP_K, dim=-1)
        book_ids.append(ids)
        book_values.append(values)
        code = sample(scores, generator, greedy)
        for index, frame in enumerate(forced):
            if frame is not None:
                code[index] = frame[book].to(code.device)
        codes.append(code)
        if book < depth.config.num_codebooks - 1:
            embed = depth.audio_embeddings(torch.cat([code, code]) + (book - 1) * vocab)
            sequence.append(depth.projection(embed).unsqueeze(1))
    return (
        torch.stack(codes, dim=1),
        torch.cat(hidden_parts, dim=-1),
        torch.stack(book_ids, dim=1),
        torch.stack(book_values, dim=1),
    )
