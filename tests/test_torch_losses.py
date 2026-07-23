"""training/torch_losses.py — the v10 background-purity loss.

What matters:
- pressure lands ONLY on the eroded true background (soft edges and the
  subject interior contribute nothing),
- clean-background predictions cost ~0, residue costs monotonically more,
- gradients flow (it is a training loss), and edge cases (no background at
  all) return exactly 0 instead of NaN.
"""
import math

import torch

from training.torch_losses import bg_purity_loss


def _square_gt(n=1, size=64, lo=20, hi=44):
    gt = torch.zeros(n, 1, size, size)
    gt[..., lo:hi, lo:hi] = 1.0
    return gt


def test_clean_background_costs_nothing():
    gt = _square_gt()
    logits = torch.full_like(gt, -12.0)  # sigmoid ~ 6e-6 everywhere
    loss = bg_purity_loss(logits, gt)
    assert loss.item() < 1e-4


def test_residue_costs_and_scales_monotonically():
    gt = _square_gt()
    faint = torch.full_like(gt, -3.0)   # p ~ 0.047 haze
    strong = torch.full_like(gt, -1.0)  # p ~ 0.269 smear
    l_faint = bg_purity_loss(faint, gt).item()
    l_strong = bg_purity_loss(strong, gt).item()
    assert 0.01 < l_faint < l_strong
    # matches the analytic BCE at that probability (region is uniform)
    assert math.isclose(l_faint, -math.log(1 - torch.sigmoid(torch.tensor(-3.0)).item()), rel_tol=0.05)


def test_soft_edge_band_and_interior_are_exempt():
    """Residue ONLY inside the subject and in the 11px band around it must
    not be penalized — the term must not fight legitimate soft edges."""
    gt = _square_gt()
    logits = torch.full_like(gt, -12.0)
    logits[..., 15:49, 15:49] = 4.0  # covers subject + the 5px surrounding band
    loss = bg_purity_loss(logits, gt, erosion_px=11)
    assert loss.item() < 1e-4


def test_gradients_flow_to_background_pixels_only():
    gt = _square_gt()
    logits = torch.zeros_like(gt, requires_grad=True)
    bg_purity_loss(logits, gt).backward()
    grad = logits.grad
    assert grad is not None
    assert grad[..., 0, 0].abs().item() > 0        # far background: pressured
    assert grad[..., 32, 32].abs().item() == 0.0   # subject interior: untouched
    assert grad[..., 18, 32].abs().item() == 0.0   # edge band (within 5px): untouched


def test_no_background_returns_zero_not_nan():
    gt = torch.ones(1, 1, 32, 32)  # subject fills the frame
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss = bg_purity_loss(logits, gt)
    assert loss.item() == 0.0
    loss.backward()  # must not blow up


def test_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        bg_purity_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 9, 9))


# ============================================================================
# bg_hinge_loss — the v11 constant-gradient variant
# ============================================================================
from training.torch_losses import bg_hinge_loss  # noqa: E402


def test_hinge_faint_haze_gets_constant_gradient():
    """THE v10 LESSON: faint haze (p ~ 0.03, logit ~ -3.5) must receive the
    SAME gradient magnitude as a strong blob — that is the whole point."""
    gt = _square_gt()
    faint = torch.full_like(gt, -3.5, requires_grad=True)
    strong = torch.full_like(gt, -0.5, requires_grad=True)
    bg_hinge_loss(faint, gt).backward()
    bg_hinge_loss(strong, gt).backward()
    g_faint = faint.grad[..., 0, 0].abs().item()
    g_strong = strong.grad[..., 0, 0].abs().item()
    assert g_faint > 0
    assert abs(g_faint - g_strong) / g_strong < 1e-5  # constant, not p-proportional


def test_hinge_below_threshold_costs_nothing():
    gt = _square_gt()
    logits = torch.full_like(gt, -12.0)  # p ~ 6e-6, far below tau_p=0.002
    assert bg_hinge_loss(logits, gt).item() == 0.0


def test_hinge_soft_edge_band_and_interior_exempt():
    gt = _square_gt()
    logits = torch.full_like(gt, -12.0)
    logits[..., 15:49, 15:49] = 6.0  # subject + surrounding 5px band
    assert bg_hinge_loss(logits, gt, erosion_px=11).item() < 1e-6


def test_hinge_no_background_returns_zero():
    gt = torch.ones(1, 1, 32, 32)
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss = bg_hinge_loss(logits, gt)
    assert loss.item() == 0.0
    loss.backward()


def test_hinge_invalid_tau_raises():
    import pytest
    with pytest.raises(ValueError):
        bg_hinge_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8), tau_p=0.0)


def test_hinge_soft_gt_samples_are_exempted_per_sample():
    """v12: a batch mixing a photo-like GT (hard 0/1) and a glow-like GT
    (lots of semi-transparency) — the hinge must pressure ONLY the photo
    sample; the glow sample's residue costs nothing."""
    gt = torch.zeros(2, 1, 64, 64)
    gt[:, :, 20:44, 20:44] = 1.0
    gt[1, :, 10:54, 10:54] = 0.4  # glow-like: ~37% soft pixels
    gt[1, :, 20:44, 20:44] = 1.0
    logits = torch.full((2, 1, 64, 64), -1.0, requires_grad=True)  # residue everywhere

    loss = bg_hinge_loss(logits, gt, max_soft_ratio=0.03)
    loss.backward()
    assert loss.item() > 0
    assert logits.grad[0, ..., 0, 0].abs().item() > 0   # photo sample: pressured
    assert logits.grad[1].abs().sum().item() == 0.0     # glow sample: fully exempt


def test_hinge_all_samples_soft_returns_zero():
    gt = torch.full((1, 1, 32, 32), 0.5)  # entirely semi-transparent GT
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss = bg_hinge_loss(logits, gt, max_soft_ratio=0.03)
    assert loss.item() == 0.0
    loss.backward()
