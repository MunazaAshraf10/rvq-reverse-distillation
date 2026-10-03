from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from rvq_ae.constants import IGNORE


@dataclass(slots=True)
class Loss:
    total: Tensor
    ce: Tensor
    kl: Tensor


def codebook_ce(logits: Sequence[Tensor], target: Tensor) -> Tensor:
    """Per codebook mean cross entropy [K]; frames whose target is IGNORE are skipped."""
    losses = [
        F.cross_entropy(scores.flatten(0, 1).float(), target[:, :, book].flatten(), ignore_index=IGNORE)
        for book, scores in enumerate(logits)
    ]
    return torch.stack(losses)


def topk_kl(logits: Sequence[Tensor], ids: Tensor, teacher: Tensor, target: Tensor, *, tau: float) -> Tensor:
    """Per codebook KL(p_T || p_S) restricted to the teacher's stored top k support [K].

    p_T = softmax(t / tau) over the valid stored ids, log p_S = log_softmax(s / tau) gathered at the
    same ids, KL = tau^2 sum_i p_T(i) (log p_T(i) - log p_S(i)). The tau^2 factor keeps gradient
    magnitudes comparable across temperatures. Ids outside [0, vocab) (padding, the semantic end
    of track id) are dropped; a frame with no valid id or an IGNORE target is excluded from the mean.
    ids and teacher are [B, T, K, k].
    """
    losses: list[Tensor] = []
    for book, scores in enumerate(logits):
        vocab = scores.shape[-1]
        book_ids = ids[:, :, book]
        valid = (book_ids >= 0) & (book_ids < vocab)
        keep = valid.any(dim=-1) & (target[:, :, book] != IGNORE)
        if not bool(keep.any()):
            losses.append(scores.sum() * 0.0)
            continue
        teacher_scores = (teacher[:, :, book].float() / tau).masked_fill(~valid, float("-inf"))
        log_teacher = torch.log_softmax(teacher_scores, dim=-1)
        prob_teacher = log_teacher.exp()
        log_student = torch.log_softmax(scores.float() / tau, dim=-1)
        gathered = log_student.gather(-1, book_ids.masked_fill(~valid, 0))
        terms = torch.where(valid, prob_teacher * (log_teacher - gathered), torch.zeros_like(gathered))
        per_frame = terms.sum(dim=-1) * tau * tau
        losses.append(per_frame[keep].mean())
    return torch.stack(losses)


def rvq_loss(
    logits: Sequence[Tensor],
    target: Tensor,
    *,
    ids: Tensor | None = None,
    teacher: Tensor | None = None,
    kl_weight: float = 0.0,
    tau: float = 1.0,
) -> Loss:
    """Reverse distillation objective L = mean_k CE_k + w_KL mean_k KL_k over the K codebooks.

    CE_k is the hard cross entropy against the code the generator sampled; KL_k distils the
    generator's stored top k distribution (Hinton et al., Distilling the Knowledge in a Neural
    Network, 2015). The released models use w_KL = 0.25 and tau = 1.
    """
    ce = codebook_ce(logits, target).mean()
    if kl_weight <= 0.0:
        return Loss(total=ce, ce=ce, kl=torch.zeros_like(ce))
    if ids is None or teacher is None:
        raise ValueError("teacher ids and logits are required when kl_weight is positive")
    kl = topk_kl(logits, ids, teacher, target, tau=tau).mean()
    return Loss(total=ce + kl_weight * kl, ce=ce, kl=kl)


@torch.no_grad()
def topk_hits(logits: Sequence[Tensor], target: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Per codebook (correct, total) counts of targets found in the student's top k [K], [K]."""
    correct: list[Tensor] = []
    total: list[Tensor] = []
    for book, scores in enumerate(logits):
        labels = target[:, :, book]
        mask = labels != IGNORE
        top = scores.topk(k, dim=-1).indices
        hit = (top == labels.masked_fill(~mask, 0).unsqueeze(-1)).any(dim=-1) & mask
        correct.append(hit.sum())
        total.append(mask.sum())
    return torch.stack(correct), torch.stack(total)
