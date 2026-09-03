"""
Stage 3 -- model definitions.

Two families are implemented:

*   ``MLPHead`` -- a configurable feed-forward classifier used for every
    single-stream and naive-fusion baseline, so that baselines and the proposed
    model differ only in *how* modalities are combined, never in optimiser,
    capacity budget or regularisation. Any performance gap is therefore
    attributable to the fusion mechanism.

*   ``ConsistencyGatedFusion`` (CGF) -- the proposed artefact. Its design
    follows from an observation made in the literature review: most multimodal
    fake-news detectors concatenate modality vectors and let the classifier
    discover cross-modal relationships implicitly, which biases them towards
    whichever single modality is easiest to exploit. CGF instead makes the
    image/caption *relationship* a first-class input:

      1. Both streams are projected into a shared space.
      2. Explicit interaction terms (Hadamard product and absolute difference)
         are computed, following the sentence-pair matching literature.
      3. Three consistency scores (frozen CLIP cosine, in-batch retrieval rank,
         and a learned cosine in the projected space) are concatenated.
      4. A *consistency gate* -- a sigmoid function of both streams and the
         consistency scores -- rescales the visual stream element-wise, so the
         network can learn to discount vision when the image does not
         corroborate the claim, rather than being forced to trust it equally
         everywhere.

    The gate is deliberately low-dimensional and inspectable: its mean
    activation per example is written out at evaluation time and analysed in
    Chapter 6, which gives the model a degree of intrinsic interpretability
    that a plain concatenation network does not have.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
class _GradientReversal(torch.autograd.Function):
    """Identity forwards, sign-flipped gradient backwards (Ganin et al., 2016).

    Placed between a shared representation and an auxiliary *domain*
    classifier, this turns the domain classifier's success into a penalty on
    the representation, pushing it towards features that cannot identify which
    source community a post came from.
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


class DomainAdversary(nn.Module):
    """Auxiliary head that tries to recover the source community.

    Motivation (Chapter 4): Fakeddit labels are assigned by subreddit, so any
    classifier can reach near-perfect accuracy by learning community style
    instead of veracity. Training the shared representation to *defeat* this
    head is a direct, principled remedy for that confound, and follows the
    event-adversarial idea of EANN (Wang et al., 2018) and the domain-
    adversarial framework of Ganin et al. (2016).
    """

    def __init__(self, d_in: int, n_domains: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_domains))

    def forward(self, h: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(grad_reverse(h, lambd))


# --------------------------------------------------------------------------- #
class MLPHead(nn.Module):
    """Baseline classifier over an arbitrary pre-computed feature vector."""

    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.3,
                 n_classes: int = 2, n_domains: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 2, n_classes)
        self.adversary = (DomainAdversary(hidden // 2, n_domains)
                          if n_domains else None)
        self.last_gate = None

    def encode(self, batch: dict) -> torch.Tensor:
        return self.trunk(batch["x"])

    def forward(self, batch: dict) -> torch.Tensor:
        return self.out(self.encode(batch))


# --------------------------------------------------------------------------- #
class ConsistencyGatedFusion(nn.Module):
    """The proposed multimodal detector.

    Parameters
    ----------
    d_text, d_img, d_meta, d_llm
        Input dimensionalities. ``d_llm=0`` disables the LLM stream.
    use_gate, use_consistency, use_interaction, use_meta, use_llm
        Ablation switches. Each can be turned off independently so that the
        contribution of every component is measurable (Chapter 6, Table 5).
    """

    def __init__(self, d_text: int, d_img: int, d_meta: int, d_llm: int = 0,
                 hidden: int = 256, dropout: float = 0.3, n_classes: int = 2,
                 use_gate: bool = True, use_consistency: bool = True,
                 use_interaction: bool = True, use_meta: bool = True,
                 use_llm: bool = True, n_domains: int = 0):
        super().__init__()
        self.use_gate = use_gate
        self.use_consistency = use_consistency
        self.use_interaction = use_interaction
        self.use_meta = use_meta and d_meta > 0
        self.use_llm = use_llm and d_llm > 0

        def proj(d_in: int) -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(d_in), nn.Linear(d_in, hidden),
                nn.GELU(), nn.Dropout(dropout))

        self.text_proj = proj(d_text)
        self.img_proj = proj(d_img)
        self.llm_proj = proj(d_llm) if self.use_llm else None

        # two frozen consistency scores + one learned cosine
        self.n_cons = 3 if use_consistency else 0

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden + self.n_cons, hidden), nn.Sigmoid())

        d_fuse = 2 * hidden
        if use_interaction:
            d_fuse += 2 * hidden
        d_fuse += self.n_cons
        if self.use_meta:
            d_fuse += d_meta
        if self.use_llm:
            d_fuse += hidden

        self.trunk = nn.Sequential(
            nn.LayerNorm(d_fuse),
            nn.Linear(d_fuse, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden // 2, n_classes)
        self.adversary = (DomainAdversary(hidden // 2, n_domains)
                          if n_domains else None)
        self.last_gate = None          # exposed for interpretability analysis

    def encode(self, batch: dict) -> torch.Tensor:
        return self.trunk(self._fuse(batch))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.out(self.encode(batch))

    def _fuse(self, batch: dict) -> torch.Tensor:
        h_t = self.text_proj(batch["text"])
        h_v = self.img_proj(batch["image"])

        cons = None
        if self.use_consistency:
            learned = F.cosine_similarity(h_t, h_v, dim=-1, eps=1e-6)
            cons = torch.cat(
                [batch["cons"], learned.unsqueeze(1)], dim=1)  # (B, 3)

        if self.use_gate:
            gin = [h_t, h_v] + ([cons] if cons is not None else [])
            g = self.gate(torch.cat(gin, dim=1))
            self.last_gate = g.detach()
            h_v = g * h_v

        parts = [h_t, h_v]
        if self.use_interaction:
            parts += [h_t * h_v, (h_t - h_v).abs()]
        if cons is not None:
            parts.append(cons)
        if self.use_meta:
            parts.append(batch["meta"])
        if self.use_llm:
            parts.append(self.llm_proj(batch["llm"]))
        return torch.cat(parts, dim=1)


# --------------------------------------------------------------------------- #
class LateFusion(nn.Module):
    """Independent per-modality heads combined by a learned scalar weight.

    Included because late (decision-level) fusion is the other conventional
    baseline in the fake-news literature, and it behaves very differently from
    early fusion when one modality is uninformative.
    """

    def __init__(self, d_text: int, d_img: int, hidden: int = 256,
                 dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        self.text_head = MLPHead(d_text, hidden, dropout, n_classes)
        self.img_head = MLPHead(d_img, hidden, dropout, n_classes)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.adversary = None
        self.last_gate = None

    def forward(self, batch: dict) -> torch.Tensor:
        a = torch.sigmoid(self.alpha)
        lt = self.text_head({"x": batch["text"]})
        lv = self.img_head({"x": batch["image"]})
        return a * lt + (1 - a) * lv


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
