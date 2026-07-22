"""Torch-dependent auxiliary training losses.

Kept OUT of train_colab_lib.py on purpose — that module's contract is "no
torch/PIL imports" so its pure logic stays testable everywhere. This module
is imported by the training notebook (and by tests, which do have torch).
"""
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
    fg = (gt > 0).float()
    pad = erosion_px // 2
    dilated_fg = F.max_pool2d(fg, kernel_size=erosion_px, stride=1, padding=pad)
    if dilated_fg.shape[-2:] != gt.shape[-2:]:  # even kernels pad asymmetrically
        dilated_fg = dilated_fg[..., : gt.shape[-2], : gt.shape[-1]]
    bg = 1.0 - dilated_fg

    n_bg = bg.sum()
    if n_bg < 1:
        return pred_logits.sum() * 0.0  # keeps the graph/dtype, contributes nothing
    p = torch.sigmoid(pred_logits.float())
    # BCE with target 0: -log(1 - p); clamped for numerical safety.
    bce = -torch.log((1.0 - p).clamp(min=eps))
    return (bce * bg).sum() / n_bg
