"""Torch-dependent auxiliary training losses.

Kept OUT of train_colab_lib.py on purpose — that module's contract is "no
torch/PIL imports" so its pure logic stays testable everywhere. This module
is imported by the training notebook (and by tests, which do have torch).
"""
import math

import torch
import torch.nn.functional as F


def bg_purity_loss(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    erosion_px: int = 11,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean BCE toward 0 over the ERODED true-background region.

    THE PROBLEM THIS TERM EXISTS FOR (v10): lucida leaves a faint gray haze
    on real-photo backgrounds (hair-category bg_mae ~0.0070 vs birefnet-hr
    0.0003 — a 20x gap stable across v5..v9; HF discussion #1, the cat
    masks). Two data-side attacks (bokeh hard-negatives, glow compacting)
    only held it flat: with a ~21% semi-transparent-GT share the standard
    all-pixel losses make mid-alpha hedging cheap. This term re-prices it —
    extra BCE pressure exactly where the GT says PURE background.

    Region: `gt == 0` eroded by `erosion_px` (via max-pool dilation of the
    foreground) — the soft-edge band is EXCLUDED, so legitimate fur/glass
    softness is untouched; the penalty covers the same region as the
    benchmark's bg_mae metric. Images with no measurable background
    contribute 0 (not NaN).

    `pred_logits`: raw model output at any scale, shape (N, 1, H, W) —
    sigmoid is applied HERE (BiRefNet's PixLoss also sigmoids internally,
    i.e. the training loop carries logits). `gt` in [0, 1], same shape.
    """
    if pred_logits.shape != gt.shape:
        raise ValueError(f"shape mismatch: {tuple(pred_logits.shape)} vs {tuple(gt.shape)}")
    bg = _eroded_bg_mask(gt, erosion_px)
    n_bg = bg.sum()
    if n_bg < 1:
        return pred_logits.sum() * 0.0  # keeps the graph/dtype, contributes nothing
    p = torch.sigmoid(pred_logits.float())
    # BCE with target 0: -log(1 - p); clamped for numerical safety.
    bce = -torch.log((1.0 - p).clamp(min=eps))
    return (bce * bg).sum() / n_bg


def _eroded_bg_mask(gt: torch.Tensor, erosion_px: int) -> torch.Tensor:
    fg = (gt > 0).float()
    pad = erosion_px // 2
    dilated_fg = F.max_pool2d(fg, kernel_size=erosion_px, stride=1, padding=pad)
    if dilated_fg.shape[-2:] != gt.shape[-2:]:  # even kernels pad asymmetrically
        dilated_fg = dilated_fg[..., : gt.shape[-2], : gt.shape[-1]]
    return 1.0 - dilated_fg


def bg_hinge_loss(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    tau_p: float = 0.002,
    erosion_px: int = 11,
    max_soft_ratio: float | None = None,
) -> torch.Tensor:
    """Hinge penalty over the ERODED true-background region:
    mean(relu(logit - logit(tau_p))).

    WHY A HINGE (the v10 lesson, measured): the BCE variant above has a
    gradient proportional to p — a p=0.5 blob gets pushed hard, but the FAINT
    haze that actually plagues real photos (p ~ 0.01-0.05) receives almost
    nothing. One epoch at lambda=60 halved the large-blob categories'
    residue (illustration bg_mae 0.0126 -> 0.0052, design -> 0.0001) while
    hair sat still (0.0070 -> 0.0071). In logit space the hinge gives every
    background pixel above the tau_p threshold a CONSTANT gradient, however
    faint it is — the only shape that can grind haze all the way down.

    tau_p is the tolerated background probability (default 0.002 ~ half an
    8-bit level); below it the pixel costs nothing, so the model is not asked
    for infinite logits. Same erosion contract as bg_purity_loss: the
    soft-edge band is exempt, legitimate softness is untouched.

    `max_soft_ratio` (v12, the epoch-11 lesson): the unmasked full epoch
    bought its background gains by crushing the glow categories (fx MAE
    0.0180 -> 0.0308 — the model truncates radiance just outside the GT
    glow). Samples whose GT soft-alpha ratio (0.05 < gt < 0.95) exceeds this
    threshold are EXEMPTED per-sample: at 0.03 that is exactly the synthetic
    semi-transparent categories (transparent ~19%, design ~15%, fx ~7-17%,
    text ~11% soft) while every photo category keeps the pressure (hair
    ~1.4%, camo/complex/thin ~0%). None disables the gating."""
    if pred_logits.shape != gt.shape:
        raise ValueError(f"shape mismatch: {tuple(pred_logits.shape)} vs {tuple(gt.shape)}")
    if not 0.0 < tau_p < 1.0:
        raise ValueError(f"tau_p must be in (0, 1), got {tau_p}")
    bg = _eroded_bg_mask(gt, erosion_px)
    if max_soft_ratio is not None:
        gt_f = gt.float()
        soft = ((gt_f > 0.05) & (gt_f < 0.95)).float()
        soft_ratio = soft.mean(dim=(-3, -2, -1))  # per sample
        keep = (soft_ratio <= max_soft_ratio).float().view(-1, 1, 1, 1)
        bg = bg * keep
    n_bg = bg.sum()
    if n_bg < 1:
        return pred_logits.sum() * 0.0
    tau_logit = math.log(tau_p / (1.0 - tau_p))
    hinge = F.relu(pred_logits.float() - tau_logit)
    return (hinge * bg).sum() / n_bg
