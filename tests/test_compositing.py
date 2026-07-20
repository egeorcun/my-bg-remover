import numpy as np
import pytest

from bgr.compositing import augment, compose


def _solid(h, w, color) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = color
    return out


@pytest.fixture
def fg_alpha():
    """32x32 red square fg, fully opaque 16x16 center, semi-transparent ring at the edges."""
    h = w = 32
    fg = _solid(h, w, (220, 30, 30))
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[8:24, 8:24] = 1.0
    alpha[4:8, 4:28] = 0.5
    alpha[24:28, 4:28] = 0.5
    return fg, alpha


@pytest.fixture
def bg():
    return _solid(32, 32, (10, 200, 10))


# ---------------------------------------------------------------------------
# compose()
# ---------------------------------------------------------------------------


def test_compose_deterministic_same_seed(fg_alpha, bg):
    fg, alpha = fg_alpha
    rgb1, a1 = compose(fg, alpha, bg, np.random.default_rng(42))
    rgb2, a2 = compose(fg, alpha, bg, np.random.default_rng(42))
    assert np.array_equal(rgb1, rgb2)
    assert np.array_equal(a1, a2)


def test_compose_different_seed_differs(fg_alpha, bg):
    fg, alpha = fg_alpha
    rgb1, a1 = compose(fg, alpha, bg, np.random.default_rng(1))
    rgb2, a2 = compose(fg, alpha, bg, np.random.default_rng(2))
    assert not (np.array_equal(rgb1, rgb2) and np.array_equal(a1, a2))


def test_compose_alpha_matches_placed_fg_when_no_scaling(fg_alpha, bg):
    """With scale fixed at 1.0 and bg the same size as fg, x0=y0=0 is forced;
    the composite alpha must exactly equal the placed (unscaled) fg alpha."""
    fg, alpha = fg_alpha
    rng = np.random.default_rng(7)
    rgb, out_alpha = compose(fg, alpha, bg, rng, scale_range=(1.0, 1.0))
    assert np.array_equal(out_alpha, alpha)
    # at the fully opaque center pixel rgb == fg color
    assert tuple(rgb[16, 16]) == (220, 30, 30)
    # at the alpha=0 corner rgb == bg color
    assert tuple(rgb[0, 0]) == (10, 200, 10)


def test_compose_size_contract_bg_larger_than_fg(fg_alpha):
    fg, alpha = fg_alpha
    big_bg = _solid(128, 96, (5, 5, 5))
    rgb, out_alpha = compose(fg, alpha, big_bg, np.random.default_rng(0))
    assert rgb.shape[:2] == big_bg.shape[:2]
    assert out_alpha.shape == big_bg.shape[:2]


def test_compose_size_contract_bg_smaller_than_fg(fg_alpha):
    fg, alpha = fg_alpha
    small_bg = _solid(10, 12, (5, 5, 5))
    rgb, out_alpha = compose(fg, alpha, small_bg, np.random.default_rng(0))
    fh, fw = fg.shape[:2]
    assert rgb.shape[0] >= fh and rgb.shape[1] >= fw
    assert out_alpha.shape == rgb.shape[:2]


def test_compose_shape_mismatch_raises(fg_alpha, bg):
    fg, _ = fg_alpha
    bad_alpha = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compose(fg, bad_alpha, bg, np.random.default_rng(0))


def test_compose_output_dtype(fg_alpha, bg):
    fg, alpha = fg_alpha
    rgb, out_alpha = compose(fg, alpha, bg, np.random.default_rng(3))
    assert rgb.dtype == np.uint8
    assert out_alpha.dtype == np.float32
    assert out_alpha.min() >= 0.0 and out_alpha.max() <= 1.0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def test_compose_alpha_and_rgb_colocated_under_random_placement():
    """Real random path (default scale_range, bg > fg): the alpha>0 bounding box
    must be EXACTLY identical to the bounding box of pixels in the output rgb that
    deviate from the background color — i.e. the region where RGB was pasted and
    the alpha region must land in the same place."""
    fg = _solid(32, 32, (255, 0, 255))  # magenta fg
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[4:28, 2:26] = 1.0  # asymmetric inner rectangle
    bg_color = (128, 128, 128)
    for seed in range(5):
        bg = _solid(80, 96, bg_color)  # bg LARGER than fg, fresh copy each round
        rgb, out_alpha = compose(fg, alpha, bg, np.random.default_rng(seed))
        alpha_mask = out_alpha > 0
        rgb_diff_mask = np.any(rgb != np.array(bg_color, dtype=np.uint8), axis=-1)
        assert alpha_mask.any(), f"seed={seed}: alpha is completely empty"
        assert rgb_diff_mask.any(), f"seed={seed}: no trace of fg in rgb"
        assert _bbox(alpha_mask) == _bbox(rgb_diff_mask), (
            f"seed={seed}: alpha region {_bbox(alpha_mask)} != rgb paste region "
            f"{_bbox(rgb_diff_mask)}"
        )


# ---------------------------------------------------------------------------
# augment()
# ---------------------------------------------------------------------------


@pytest.fixture
def noisy_rgb_alpha():
    rng = np.random.default_rng(123)
    h = w = 40
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    # left-right ASYMMETRIC pattern: makes flip detection reliable via exact-equality comparison.
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[10:30, 4:20] = 1.0
    alpha[5:10, 4:12] = 0.5
    return rgb, alpha


def test_augment_deterministic_same_seed(noisy_rgb_alpha):
    rgb, alpha = noisy_rgb_alpha
    rgb1, a1 = augment(rgb, alpha, np.random.default_rng(9))
    rgb2, a2 = augment(rgb, alpha, np.random.default_rng(9))
    assert np.array_equal(rgb1, rgb2)
    assert np.array_equal(a1, a2)


def test_augment_preserves_shape(noisy_rgb_alpha):
    rgb, alpha = noisy_rgb_alpha
    out_rgb, out_alpha = augment(rgb, alpha, np.random.default_rng(5))
    assert out_rgb.shape == rgb.shape
    assert out_alpha.shape == alpha.shape


def test_augment_alpha_only_ever_exactly_unchanged_or_flipped(noisy_rgb_alpha):
    """Color jitter/blur/JPEG NEVER touch alpha: the alpha output must be exactly
    identical to either the original or its horizontal flip (no transform other
    than the geometric one)."""
    rgb, alpha = noisy_rgb_alpha
    flips_seen = {True: False, False: False}
    for seed in range(30):
        _, out_alpha = augment(rgb, alpha, np.random.default_rng(seed))
        is_flipped = np.array_equal(out_alpha, alpha[:, ::-1])
        is_unchanged = np.array_equal(out_alpha, alpha)
        assert is_flipped or is_unchanged, f"seed={seed}: alpha was affected by color/blur/jpeg"
        flips_seen[is_flipped and not is_unchanged] = True
    # both possible branches (flip / no-flip) must be seen at least once in 30 tries
    assert flips_seen[True], "no flip observed across 30 seeds (rng should be ~50%)"
    assert flips_seen[False], "no non-flip observed across 30 seeds"


def test_augment_flip_applies_to_rgb_content_too():
    """When a flip happens, rgb is mirrored horizontally too: the left/right brightness order changes."""
    h, w = 40, 40
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, : w // 2] = 230  # bright left
    rgb[:, w // 2 :] = 20  # dark right
    # asymmetric alpha: a constant (symmetric) pattern would make flip detection impossible
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[:, :5] = 1.0

    found_flip = found_no_flip = False
    for seed in range(30):
        out_rgb, out_alpha = augment(rgb, alpha, np.random.default_rng(seed))
        is_flipped = np.array_equal(out_alpha, alpha[:, ::-1])
        # use the means of interior regions to avoid edge effects
        left_mean = out_rgb[:, 5:15].mean()
        right_mean = out_rgb[:, -15:-5].mean()
        if is_flipped:
            found_flip = True
            assert left_mean < right_mean, f"seed={seed}: left/right brightness not inverted after flip"
        else:
            found_no_flip = True
            assert left_mean > right_mean, f"seed={seed}: left/right brightness not preserved without flip"
    assert found_flip and found_no_flip


def test_augment_jpeg_and_jitter_change_rgb_but_not_alpha(noisy_rgb_alpha):
    rgb, alpha = noisy_rgb_alpha
    # find a no-flip seed
    seed = next(
        s
        for s in range(30)
        if np.array_equal(augment(rgb, alpha, np.random.default_rng(s))[1], alpha)
    )
    out_rgb, out_alpha = augment(rgb, alpha, np.random.default_rng(seed))
    assert not np.array_equal(out_rgb, rgb), "jitter/blur/jpeg did not change rgb at all"
    assert np.array_equal(out_alpha, alpha)


def test_augment_output_dtype(noisy_rgb_alpha):
    rgb, alpha = noisy_rgb_alpha
    out_rgb, out_alpha = augment(rgb, alpha, np.random.default_rng(1))
    assert out_rgb.dtype == np.uint8
    assert out_alpha.dtype == np.float32
    assert out_alpha.min() >= 0.0 and out_alpha.max() <= 1.0


def test_augment_shape_mismatch_raises(noisy_rgb_alpha):
    rgb, _ = noisy_rgb_alpha
    bad_alpha = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        augment(rgb, bad_alpha, np.random.default_rng(0))
