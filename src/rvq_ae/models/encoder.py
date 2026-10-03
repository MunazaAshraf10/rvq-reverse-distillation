import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from rvq_ae.audio.timeline import Pool
from rvq_ae.constants import IGNORE
from rvq_ae.models.config import EncoderConfig
from rvq_ae.models.layers import Layer
from rvq_ae.models.mup import Readout, rescale_init

POSITION_STD = 0.02


class ResBlock(nn.Module):
    """x + conv2(gelu(conv1(gelu(norm(x))))) with a dilated 3 tap conv1 and a 1 tap conv2.

    GroupNorm with one group normalises over channels and time jointly, so zero padded
    latents in a batch take part in the statistics exactly as they did during training.
    """

    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.conv1(F.gelu(self.norm(x)))
        return x + self.conv2(F.gelu(hidden))


def head(
    dim: int, vocab: int, cfg: EncoderConfig, *, bias: bool = True, zero_init: bool = False
) -> nn.Linear:
    if cfg.mup:
        return Readout(
            dim, vocab, cfg.width_mult, bias=bias, zero_init=zero_init, output_mult=cfg.mup_output_mult
        )
    return nn.Linear(dim, vocab, bias=bias)


class DepthDecoder(nn.Module):
    """Autoregressive over codebook depth, independent per frame.

    Sequence for one frame: [context, e_0(c_0), e_1(c_1), ..., e_(K-2)(c_(K-2))] with learned depth
    positions and a causal mask. Head j reads position j + 1 and predicts acoustic codebook j + 1.
    Training teacher forces the true codes; inference feeds back greedy predictions.

    Without feedback (the ablation) every position carries the frame context, attention across
    depth is bidirectional, and no code is ever read: the same transformer over depth, minus the
    conditioning on lower codebooks.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        vocabs = cfg.codebook_vocab_sizes
        dim = cfg.depth_decoder_dim
        self.feedback = cfg.depth_decoder_feedback
        self.context_projection = head(cfg.d_model, dim, cfg, bias=False)
        self.prior_embeddings = nn.ModuleList(
            nn.Embedding(vocab, dim) for vocab in (vocabs[:-1] if self.feedback else ())
        )
        self.position = nn.Parameter(torch.empty(1, len(vocabs), dim).normal_(std=POSITION_STD))
        scale = (dim // cfg.depth_decoder_heads) ** -0.5
        self.layers = nn.ModuleList(
            Layer(
                dim,
                cfg.depth_decoder_heads,
                cfg.depth_decoder_ff_mult,
                cfg.depth_decoder_dropout,
                scale=scale,
                causal=self.feedback,
                fused=False,
            )
            for _ in range(cfg.depth_decoder_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList(nn.Linear(dim, vocab) for vocab in vocabs[1:])

    def embed(self, book: int, codes: Tensor) -> Tensor:
        """Prior token embedding; masked targets (IGNORE) borrow token 0 and are excluded by the loss."""
        return self.prior_embeddings[book](codes.masked_fill(codes == IGNORE, 0))

    def decode(self, sequence: Tensor) -> Tensor:
        hidden = sequence + self.position[:, : sequence.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)

    def forward(self, context: Tensor, targets: Tensor) -> list[Tensor]:
        """Teacher forced logits, one tensor [B, T, vocab_k] per acoustic codebook."""
        if not self.feedback:
            return self.contextual(context)
        batch, frames = context.shape[:2]
        count = batch * frames
        steps = [self.context_projection(context).reshape(count, 1, -1)]
        for book in range(len(self.prior_embeddings)):
            steps.append(self.embed(book, targets[:, :, book].reshape(count)).unsqueeze(1))
        hidden = self.decode(torch.cat(steps, dim=1))
        return [
            layer(hidden[:, index + 1]).reshape(batch, frames, -1) for index, layer in enumerate(self.heads)
        ]

    def contextual(self, context: Tensor) -> list[Tensor]:
        """Logits of the no feedback ablation: one bidirectional pass over K context positions."""
        batch, frames = context.shape[:2]
        projected = self.context_projection(context).reshape(batch * frames, 1, -1)
        hidden = self.decode(projected.expand(-1, len(self.heads) + 1, -1))
        return [
            layer(hidden[:, index + 1]).reshape(batch, frames, -1) for index, layer in enumerate(self.heads)
        ]

    def generate(self, context: Tensor, semantic: Tensor) -> list[Tensor]:
        """Greedy free running logits given the semantic codes [B, T]."""
        if not self.feedback:
            return self.contextual(context)
        batch, frames = context.shape[:2]
        count = batch * frames
        steps = [
            self.context_projection(context).reshape(count, 1, -1),
            self.embed(0, semantic.reshape(count)).unsqueeze(1),
        ]
        logits: list[Tensor] = []
        for index, layer in enumerate(self.heads):
            hidden = self.decode(torch.cat(steps, dim=1))[:, -1]
            scores = layer(hidden)
            logits.append(scores.reshape(batch, frames, -1))
            if index + 1 < len(self.heads):
                steps.append(self.embed(index + 1, scores.argmax(dim=-1)).unsqueeze(1))
        return logits


class RvqEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.d_model
        self.conv_in = nn.Conv1d(cfg.latent_channels, dim, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(ResBlock(dim, dilation) for dilation in cfg.conv_dilations)
        self.position = nn.Parameter(
            torch.empty(1, cfg.max_position_embeddings, dim).normal_(std=POSITION_STD)
        )
        self.transformer = nn.ModuleList(
            Layer(dim, cfg.num_heads, cfg.ff_mult, cfg.dropout, scale=cfg.attention_scale, causal=False)
            for _ in range(cfg.num_layers)
        )
        self.norm_out = nn.LayerNorm(dim)
        vocabs = cfg.codebook_vocab_sizes[:1] if cfg.depth_decoder else cfg.codebook_vocab_sizes
        self.heads = nn.ModuleList(
            head(dim, vocab, cfg, zero_init=cfg.mup_readout_zero_init) for vocab in vocabs
        )
        self.depth_decoder = DepthDecoder(cfg) if cfg.depth_decoder else None
        self.checkpointing = False
        rescale_init(self, cfg.width_mult)

    def features(self, latents: Tensor, pool: Pool) -> Tensor:
        """Frame features [B, T, d] from DAV latents [B, L, C].

        latents at 86.1328125 Hz
          -> conv stem and dilated residual stack, still at the latent rate
          -> mean pooling with the per sample operator P onto 25 Hz frames
          -> learned frame positions and N bidirectional pre norm layers.
        """
        frames = pool.span.shape[-1]
        if frames > self.position.shape[1]:
            raise ValueError(f"{frames} frames exceed the {self.position.shape[1]} frame context")
        hidden = self.conv_in(latents.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = pool.apply(hidden.transpose(1, 2)) + self.position[:, :frames]
        for layer in self.transformer:
            if self.checkpointing and self.training:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        return self.norm_out(hidden)

    def logits(self, features: Tensor, targets: Tensor | None = None) -> list[Tensor]:
        """One logits tensor per codebook; targets [B, T, K] enable teacher forcing of the depth decoder."""
        semantic = self.heads[0](features)
        if self.depth_decoder is None:
            return [semantic] + [layer(features) for layer in list(self.heads)[1:]]
        if targets is not None:
            return [semantic, *self.depth_decoder(features, targets)]
        return [semantic, *self.depth_decoder.generate(features, semantic.argmax(dim=-1))]

    def forward(self, latents: Tensor, pool: Pool, targets: Tensor | None = None) -> list[Tensor]:
        return self.logits(self.features(latents, pool), targets)

    @torch.no_grad()
    def codes(self, latents: Tensor, pool: Pool) -> Tensor:
        """Greedy codes [B, T, K]."""
        return torch.stack([scores.argmax(dim=-1) for scores in self.forward(latents, pool)], dim=-1)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
