import numpy as np
import pytest

from bgr.compositing import augment, compose


def _solid(h, w, color) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = color
    return out


@pytest.fixture
def fg_alpha():
    """32x32 kırmızı kare fg, merkezde 16x16 tam opak, kenarlarda yarı saydam halka."""
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
    """scale sabit 1.0 ve bg==fg boyutunda iken x0=y0=0 zorunlu olur;
    kompozit alpha tam olarak yerleştirilen (ölçeklenmemiş) fg alpha'sına eşit olmalı."""
    fg, alpha = fg_alpha
    rng = np.random.default_rng(7)
    rgb, out_alpha = compose(fg, alpha, bg, rng, scale_range=(1.0, 1.0))
    assert np.array_equal(out_alpha, alpha)
    # tam opak merkez pikselde rgb == fg rengi
    assert tuple(rgb[16, 16]) == (220, 30, 30)
    # alpha=0 köşede rgb == bg rengi
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


# ---------------------------------------------------------------------------
# augment()
# ---------------------------------------------------------------------------


@pytest.fixture
def noisy_rgb_alpha():
    rng = np.random.default_rng(123)
    h = w = 40
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    # sol-sağ ASİMETRİK desen: flip tespiti tam eşitlik karşılaştırmasıyla güvenilir olsun.
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
    """Renk jitter/blur/JPEG alpha'ya HİÇ dokunmaz: alpha çıktısı ya orijinaliyle
    ya da yatay flip'iyle birebir aynı olmalı (geometrik dışında hiçbir dönüşüm yok)."""
    rgb, alpha = noisy_rgb_alpha
    flips_seen = {True: False, False: False}
    for seed in range(30):
        _, out_alpha = augment(rgb, alpha, np.random.default_rng(seed))
        is_flipped = np.array_equal(out_alpha, alpha[:, ::-1])
        is_unchanged = np.array_equal(out_alpha, alpha)
        assert is_flipped or is_unchanged, f"seed={seed}: alpha renk/blur/jpeg'den etkilenmiş"
        flips_seen[is_flipped and not is_unchanged] = True
    # olası her iki dal da (flip / no-flip) 30 denemede en az bir kez görülmeli
    assert flips_seen[True], "30 seed'de hiç flip gözlenmedi (rng ~%50 olmalı)"
    assert flips_seen[False], "30 seed'de hiç flip-olmayan gözlenmedi"


def test_augment_flip_applies_to_rgb_content_too():
    """Flip olduğunda rgb de yatayda ters çevrilir: sol/sağ parlaklık sırası değişir."""
    h, w = 40, 40
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, : w // 2] = 230  # sol parlak
    rgb[:, w // 2 :] = 20  # sağ karanlık
    # asimetrik alpha: sabit (simetrik) desen flip tespitini imkansız kılar
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[:, :5] = 1.0

    found_flip = found_no_flip = False
    for seed in range(30):
        out_rgb, out_alpha = augment(rgb, alpha, np.random.default_rng(seed))
        is_flipped = np.array_equal(out_alpha, alpha[:, ::-1])
        # kenar etkilerinden kaçınmak için iç bölgelerin ortalamasını kullan
        left_mean = out_rgb[:, 5:15].mean()
        right_mean = out_rgb[:, -15:-5].mean()
        if is_flipped:
            found_flip = True
            assert left_mean < right_mean, f"seed={seed}: flip sonrası sol/sağ parlaklık ters değil"
        else:
            found_no_flip = True
            assert left_mean > right_mean, f"seed={seed}: flip olmadan sol/sağ parlaklık korunmamış"
    assert found_flip and found_no_flip


def test_augment_jpeg_and_jitter_change_rgb_but_not_alpha(noisy_rgb_alpha):
    rgb, alpha = noisy_rgb_alpha
    # no-flip seed'i bul
    seed = next(
        s
        for s in range(30)
        if np.array_equal(augment(rgb, alpha, np.random.default_rng(s))[1], alpha)
    )
    out_rgb, out_alpha = augment(rgb, alpha, np.random.default_rng(seed))
    assert not np.array_equal(out_rgb, rgb), "jitter/blur/jpeg rgb'yi hiç değiştirmedi"
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
