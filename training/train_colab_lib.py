"""Pure-Python, dependency-free helper logic for Phase 3
(`training/train_colab.ipynb`). Deliberately does NOT import `torch`/`PIL` — so
it can be tested both on Colab (inside the BiRefNet training loop) and in this
repo with `pytest`, without a GPU/torch (see `tests/test_train_colab_lib.py`).
The notebook adds this file to `sys.path` after cloning the repo and imports it
(the same "SINGLE SOURCE OF TRUTH" pattern used for `scripts/` — see the cell
(d) note in `training/prepare_data_colab.ipynb`) — the logic is never copied and
rewritten inside the notebook, which eliminates the drift risk.

Covers six independent concerns:
1. Category-weighted sampling (`compute_sample_weights` / `compute_expected_shares`)
   — computes the weights fed into `torch.utils.data.WeightedRandomSampler`.
2. Checkpoint discovery/pruning (`find_latest_checkpoint` / `prune_old_checkpoints`)
   — automatic resume after a Colab session drop + capping Drive disk quota usage.
3. Deterministic + PERSISTENT TRAIN/VAL split (`deterministic_val_split` /
   `load_or_create_val_split`) + fixed quick-evaluation subset
   (`fixed_eval_subset`).
4. Reproductions of two small pieces of the official BiRefNet
   `train.py`/`config.py` logic (`should_apply_finetune_reweight`,
   `effective_lr`) — see the line-level references in the per-function
   docstrings within this module.
5. BiRefNet `config.py` text patching (`apply_config_patches`) — IDEMPOTENT
   (re-running the notebook on the same VM must not blow up).
6. Drive -> local disk data copying (`copy_pairs`) — with file-size validation
   for both im and gt (half-finished/truncated copies get repaired).
7. v3 — VAL leak exclusion + composite manifest merge + empty-manifest
   guard (`strip_composite_copy_suffix` / `derive_val_excluded_source_ids` /
   `merge_composite_manifest` / `ensure_manifest_pairs`) — so that `training/
   v3_veri_guncelleme_hucresi.py` can exclude the VAL set before its `_o00`
   generation, update the composite manifest on Drive (without overwriting it),
   and stop an export from proceeding on an empty manifest early and loudly
   (see that file's module docstring).
8. Packing/consuming the TRAIN data as tar shards on Drive
   (`tar_shard_name` / `split_stems_to_shards` / `validate_tar_manifest`) —
   the SHARED contract between `training/veri_tar_paketleme_hucresi.py` (the
   packing side, a free CPU Colab cell) and `train_colab.ipynb` cell (c) (the
   side that downloads and extracts): instead of copying 52k+ small files one
   by one over the Drive FUSE mount (~75 min, with occasional transient
   'Errno 5'), ~8 large tar shards are copied and extracted locally (~10 min).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from benchmark.testset import append_entries, load_manifest

CKPT_FILENAME_RE = re.compile(r"^epoch_(\d+)\.pth$")


# ============================================================================
# 1) Category-weighted sampling
# ============================================================================
def load_stem_categories(manifest_path: str | Path) -> dict[str, str]:
    """Reads the composite manifest (`benchmark.testset` format:
    id/image/category/gt_alpha JSONL rows — see the `scripts/export_birefnet.py`
    docstring; during export, stem = row['id']) and returns a `{stem: category}`
    dict.

    The notebook reads this file from the `bg-remover-data/
    train_composites_manifest.jsonl` copy on Drive (see
    `training/colab_devam_hucresi.py` `stage7_drive_copy`)."""
    result: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result[row["id"]] = row["category"]
    return result


SAMPLER_PRESET_V1: dict[str, float] = {"transparent": 0.20, "camouflage": 0.20}
"""The target actually used in the v1 fine-tune run (`epoch_1.pth`) —
the project's internal phase report (removed from the repo) §5 item 3. Because it only pinned
transparent+camouflage, the remaining 60% share was distributed among
hair/complex/thin/general in proportion to their RAW counts; since hair's raw
volume (~9422) far exceeded complex (~2190) and thin (~810), hair took most of
that 60% and complex/thin got almost no share at all — the root cause of the
"catastrophic forgetting" in the v1 comparison report (complex MAE 0.156 vs
0.024 baseline, thin 0.090 vs 0.018, hair 0.013 vs 0.0045)."""

SAMPLER_PRESET_V2: dict[str, float] = {
    "camouflage": 0.18,
    "transparent": 0.20,
    "hair": 0.20,
    "complex": 0.20,
    "thin": 0.12,
    "general": 0.10,
}
"""v2 rebalancing target (sums to EXACTLY 100% — `compute_sample_weights` only
raises ValueError on `sum > 1.0`; when the sum is exactly 1.0, `_other` stems
whose category is missing from the manifest get ZERO weight, i.e. they are
never sampled — a deliberate choice: data of unknown category must not muddy
the training mix; notebook cell (e) already prints the `n_unknown` count to the
console). camouflage was slightly LOWERED from v1's 20% to 18% (its raw share
is already ~28-36% — even left out of the sampler in v1 it would naturally take
a large share; 18% is enough to protect the v1 gains). transparent was KEPT at
20%: the ideogram scoring made it concrete — transparent is the only category
bgr-v1 LOSES to ideogram (MAE 0.0437 vs 0.0343, the closest chase target), so
cutting its share would have been wrong. hair/complex/thin were given EXPLICIT
targets (they had none in v1) — hair 20% (its absolute error is already small,
0.013 MAE — a modest recovery target), complex 20%, thin 12% — to recover the
categories that collapsed in v1; general 10% curated general-purpose images.
See the internal review notes (not in the repo)."""

SAMPLER_PRESET_V3: dict[str, float] = {
    "camouflage": 0.16,
    "transparent": 0.24,
    "hair": 0.18,
    "complex": 0.20,
    "thin": 0.12,
    "general": 0.10,
}
"""v3 rebalancing target (sums to EXACTLY 100%) — the adjustment made after
v2's real benchmark results (see `results/baseline/metrics.json`,
the internal review notes (not in the repo)). It answers two concrete findings:

1. **Domain gap / persistence of over-deletion**: the root cause of
   over-deletion not improving from v1 to v2 on the real-photo benchmark was
   that ALL categories EXCEPT camouflage were trained only on SYNTHETIC
   composited backgrounds — changing sampler shares cannot fix that; the data
   needed original-background samples (`scripts/make_composites.py` `_o00`
   copies — see that file's v3 note). The ONLY thing the sampler side can do
   is make sure this new data is seen enough within the epoch.
2. **transparent got WORSE from v1 to v2** (MAE 0.0437 -> 0.0481) — we moved
   AWAY from ideogram's 0.0343 target (v2's cut to 18% may have moved in the
   wrong direction). v3 RAISES transparent to 24% (+6 points over v2's 18%),
   prioritizing the chase after ideogram — the single largest share in this
   preset.
   camouflage already leaves a strong margin in v2 (bgr-v2 MAE 0.0310, closest
   general-purpose competitor birefnet-hr 0.0752 — 59% better; ideogram is MUCH
   worse at camo: 0.1179) — thanks to that margin, the camo share could be
   trimmed a bit further from v2's 18% to 16%, and the 2 points gained were
   transferred to transparent. hair from 20% to 18% (its absolute error is
   already small, 0.0156 MAE), complex/thin/general KEPT at their v2 values
   (20%/12%/10%) (v1's collapsed categories — see the SAMPLER_PRESET_V2
   docstring — are still recovering; no evidence yet to justify cutting their
   share). See the internal review notes (not in the repo)."""

SAMPLER_PRESET_V4: dict[str, float] = {
    "camouflage": 0.12,
    "transparent": 0.18,
    "hair": 0.08,
    "complex": 0.19,
    "thin": 0.13,
    "general": 0.04,
    "text": 0.10,
    "fx": 0.08,
    "illustration": 0.08,
}
"""v4 rebalancing target (sums to EXACTLY 100%) — the adjustment made after
v3's real benchmark results. After the v3 benchmark the user shifted the focus
to two axes: keeping complex+thin recovering, and acquiring NEW capabilities —
logo/text preservation (`text`), around-object VFX glow (`fx`) and illustration
(`illustration`); the data for the three new categories is produced by
`training/v4_veri_guncelleme_hucresi.py` (`scripts/make_textfx.py` + ToonOut).

1. **transparent KEPT at 18%**: only 0.0043 away from Ideogram's 0.0343
   target — the chase continues; cutting the share would repeat the v2 lesson
   (MAE got worse when the share was cut, see SAMPLER_PRESET_V3 docstring
   item 2); but v3's 24% single-largest-share is no longer needed either, 18%
   is enough to protect it.
2. **camouflage DOWN to 12%**: v3 MAE 0.0304 vs Ideogram 0.1179 — the margin
   is ENORMOUS (roughly a quarter of Ideogram's). The share that went from
   18% to 16% in v2->v3 could safely be lowered to 12% thanks to this margin;
   the freed points were transferred to the new categories.
3. **hair DOWN to 8%**: at 0.0067 MAE it is already close to rmbg's 0.0045 —
   the share can be reduced (it was 18% in v3; the absolute error is small,
   8% suffices for protection).
4. **complex 19% / thin 13%**: kept close to v3's 20%/12% (the focus
   categories — v1's collapsed categories are still the priority; thin was
   slightly strengthened with +1 point). general went from 10% to 4% (curated
   general-purpose images; the least risky cut to make room for the new
   categories).
5. **text 10% / fx 8% / illustration 8%**: the new capabilities — a combined
   26% share, enough in-epoch visibility for the model to learn these three
   skills from scratch."""

SAMPLER_PRESET_V5: dict[str, float] = {
    "camouflage": 0.12,
    "transparent": 0.22,
    "hair": 0.12,
    "complex": 0.19,
    "thin": 0.12,
    "general": 0.04,
    "text": 0.07,
    "fx": 0.05,
    "illustration": 0.07,
}
"""v5 rebalancing target (sums to EXACTLY 100%) — the adjustment made after
visually reviewing the v4 benchmark (191 images). v4 findings: text 0.0119
(Ideogram BEATEN) and illustration 0.0129 targets were MET -> their shares can
be pulled back from 10%/8% to 7%/7% (gain-protection mode). But two categories
paid the price of the share cuts: transparent 0.0386->0.0405 (the Ideogram
0.0343 target slipped away) and hair 0.0067->0.0106 -> transparent 18%->22%
and hair 8%->12% were RESTORED. fx 8%->5%: in v4 the fx data contributed to
ghosting (mid-alpha on solid objects; complex mid-alpha ratio 4.5%->5.9%) —
together with the v5 fix of `make_textfx._render_fx_sample` (narrow halo band,
short streaks, particles concentrated on the bbox), its share was lowered too.
complex KEPT at 19% (InSPyReNet's 0.0110 showed how high the category ceiling
is; our realistic epoch-5 target is ~0.045-0.055). camo 12% (the margin is
enormous), general 4% unchanged."""

SAMPLER_PRESET_V7: dict[str, float] = {
    "camouflage": 0.10,
    "transparent": 0.22,
    "hair": 0.10,
    "complex": 0.19,
    "thin": 0.12,
    "general": 0.02,
    "text": 0.06,
    "fx": 0.05,
    "illustration": 0.06,
    "design": 0.08,
}
"""v7 target (sums to EXACTLY 100%) — new synthetic `design` category (8%) for
issue #2 (the print-design/sticker style gap): stylized subjects on paper-white
backgrounds (halftone/posterize/ink) + distressed display text + smoke/glow
effects. The share was carved out of categories whose targets were already
comfortably met: camo .12->.10 (0.0249, 2.3x ahead of the closest competitor),
hair .12->.10, text .07->.06 (0.0112, commercial reference beaten),
illustration .07->.06 (0.0089, ahead of everyone), general .04->.02.
transparent (.22 — the Ideogram 0.0343 target is still open) and
complex/thin/fx UNCHANGED."""

SAMPLER_PRESETS: dict[str, dict[str, float]] = {
    "v1": SAMPLER_PRESET_V1,
    "v2": SAMPLER_PRESET_V2,
    "v3": SAMPLER_PRESET_V3,
    "v4": SAMPLER_PRESET_V4,
    "v5": SAMPLER_PRESET_V5,
    "v7": SAMPLER_PRESET_V7,
}
"""The table the notebook's `SAMPLER_PRESET` parameter ("v1"/"v2"/"v3"/"v4")
is resolved against — see the `training/train_colab.ipynb` parameters cell and
cell (e)."""


def resolve_sampler_num_samples(dataset_len: int, num_samples: int | None = None) -> int:
    """Resolves the value passed to `WeightedRandomSampler(weights,
    num_samples=...)` (a pure function that computes only this NUMBER, not the
    sampler OBJECT, so it stays testable without depending on torch — see the
    module-level docstring's "does not import torch/PIL" principle).

    `num_samples=None` (default): IDENTICAL to the v1/v2 behavior —
    returns `dataset_len` (the current `len(train_dataset)`), i.e. the epoch
    length grows/shrinks with the dataset.

    If `num_samples` is given (v3): the epoch length is locked to this FIXED
    value, INDEPENDENT of the dataset's real size. In v3, when ~14k new pairs
    were added to the dataset via `scripts/make_composites.py`'s `_o00` copies
    (see that file's v3 note), leaving `num_samples=None` would have let the
    per-epoch iteration count (and hence the Colab unit cost) grow
    automatically; instead the notebook passes `EPOCH_NUM_SAMPLES=27715`
    (PARITY with v2's epoch size) — the epoch cost stays fixed at ~48 units.
    Since `WeightedRandomSampler` already runs with `replacement=True`,
    `num_samples < dataset_len` is NOT data loss — it only shortens how many
    samples the epoch draws; the newly added `_o00` samples remain
    probabilistically selectable through the sampler weights (according to the
    category shares).

    `num_samples <= 0` -> `ValueError` (WeightedRandomSampler itself rejects
    this too, but it is caught early with a clear message)."""
    if num_samples is None:
        return dataset_len
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0: {num_samples}")
    return num_samples


def compute_sample_weights(
    stems: list[str],
    stem_category: dict[str, str],
    target_share: dict[str, float] | None = None,
    default_category: str = "_other",
) -> list[float]:
    """For `stems` (which MUST be in the SAME ORDER as MyData.image_paths —
    WeightedRandomSampler weights have to align with dataset indices), produces
    weights that pin the EXPECTED in-epoch share of every category named in
    `target_share` to `target_share`, while preserving the relative proportions
    of the remaining categories AMONG THEMSELVES (proportional to their raw
    counts).

    If `target_share=None` (default), `SAMPLER_PRESET_V1` is used — IDENTICAL
    to the behavior of the v1 fine-tune run (epoch_1.pth) (backward
    compatibility: existing callers keep getting the same result without
    changing anything). For v2 rebalancing, `SAMPLER_PRESET_V2` (or
    `SAMPLER_PRESETS["v2"]`) must be passed explicitly — see the module-level
    `SAMPLER_PRESETS` and the v2 preparation report (in v1 the combined
    transparent+camouflage share climbed above 50% and caused "catastrophic
    forgetting" in complex/thin/hair).

    Algorithm: for a targeted category c, per-sample weight =
    target_share[c] / count(c) (the category as a whole takes exactly the
    target_share[c] share; samples within the category are weighted equally).
    For untargeted categories, per-sample weight =
    (1 - sum(target_share)) / total_untargeted_count — the SAME constant value
    for ALL untargeted samples, which keeps their share relative to each other
    proportional to raw counts (as in the unweighted case).

    This is the LEAST INVASIVE mechanism compared to physical oversampling
    (producing extra composite files): it works without touching
    `scripts/make_composites.py`'s `CATEGORY_MULTIPLIER` factors
    (transparent x10, camouflage x2 — see that file's docstring), changing only
    the `DataLoader`'s sampler; ONLY a `sampler=` argument is added on top of
    the official `train.py`'s `prepare_dataloader`
    (`shuffle=is_train, sampler=None`) (see the notebook training cell).
    """
    if target_share is None:
        target_share = SAMPLER_PRESET_V1
    categories = [stem_category.get(s, default_category) for s in stems]
    counts = Counter(categories)

    targeted = {c: share for c, share in target_share.items() if counts.get(c, 0) > 0}
    sum_targeted = sum(targeted.values())
    if sum_targeted > 1.0 + 1e-9:  # exactly 1.0 IS allowed (see SAMPLER_PRESET_V2); epsilon for fp summation noise
        raise ValueError(f"target_share sum cannot exceed 1.0 (over present categories): {targeted}")
    remaining_mass = max(0.0, 1.0 - sum_targeted)  # sum==1.0 -> untargeted (_other) samples get 0 weight (never sampled)

    other_categories = [c for c in counts if c not in targeted]
    n_other_total = sum(counts[c] for c in other_categories)

    per_category_weight: dict[str, float] = {}
    for c, share in targeted.items():
        per_category_weight[c] = share / counts[c]
    other_weight = (remaining_mass / n_other_total) if n_other_total > 0 else 0.0
    for c in other_categories:
        per_category_weight[c] = other_weight

    return [per_category_weight[c] for c in categories]


def compute_expected_shares(
    weights: list[float], stems: list[str], stem_category: dict[str, str], default_category: str = "_other"
) -> dict[str, float]:
    """Diagnostics: computes each category's EXPECTED in-epoch sampling share
    (`sum(weights in cat) / sum(all weights)`) with the given weights (even if
    unnormalized). The notebook calls this right after building the sampler to
    print to the console that the target (>=20%) was really met."""
    total = sum(weights)
    if total <= 0:
        return {}
    sums: dict[str, float] = {}
    for w, s in zip(weights, stems):
        c = stem_category.get(s, default_category)
        sums[c] = sums.get(c, 0.0) + w
    return {c: v / total for c, v in sums.items()}


# ============================================================================
# 2) Checkpoint discovery / pruning (resume + Drive disk quota)
# ============================================================================
def find_latest_checkpoint(ckpt_dir: str | Path, pattern: re.Pattern = CKPT_FILENAME_RE) -> tuple[str, int] | None:
    """Scans `ckpt_dir` for files matching the `epoch_<N>.pth` pattern and
    returns the one with the largest N as `(path, epoch)`; if there is none,
    returns `None` (first run — start from scratch with
    `BiRefNet.from_pretrained(HF_MODEL_ID)`)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    best: tuple[str, int] | None = None
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if best is None or epoch > best[1]:
            best = (str(p), epoch)
    return best


def prune_old_checkpoints(
    ckpt_dir: str | Path, keep_last_n: int, pattern: re.Pattern = CKPT_FILENAME_RE
) -> list[str]:
    """Keeps only the checkpoints of the last `keep_last_n` epochs in
    `ckpt_dir` and DELETES the rest; returns the deleted file paths. Called
    both on the local Colab disk and on Drive (100 epochs x ~the current
    BiRefNet checkpoint size fills the Drive quota quickly — see the notebook
    parameters cell `KEEP_LAST_N_CHECKPOINTS`)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    entries: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda t: t[0], reverse=True)
    removed = []
    for _, p in entries[keep_last_n:]:
        p.unlink()
        removed.append(str(p))
    return removed


# ============================================================================
# 3) Deterministic TRAIN/VAL split + fixed quick-evaluation subset
# ============================================================================
def deterministic_val_split(all_stems: list[str], seed: int, val_fraction: float) -> tuple[list[str], list[str]]:
    """Deterministically splits `all_stems` (input order IRRELEVANT — it is
    sorted first, then shuffled with a seed, so the same result is produced
    regardless of filesystem listing order) into a (train_stems, val_stems)
    pair. Produces the SAME val set on re-runs (idempotency, task item 6) —
    there is NO physical moving of files; the notebook merely uses this list to
    decide which files to copy into the TRAIN/ vs VAL/ subdirectory."""
    import random

    stems_sorted = sorted(all_stems)
    rng = random.Random(seed)
    shuffled = stems_sorted[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def load_or_create_val_split(
    all_stems: list[str], seed: int, val_fraction: float, persist_path: str | Path
) -> tuple[list[str], list[str]]:
    """The PERSISTENT version of `deterministic_val_split`: on the first run it
    performs the split and writes the val list to `persist_path` (JSON); on
    subsequent runs it reads it from the file.

    Why it is needed: the dataset on Drive can GROW later (the Phase 2 pipeline
    is idempotent — a new run may add new pairs). A purely deterministic split
    would produce a DIFFERENT val set once the input list changed — images seen
    in training during earlier epochs would leak into val. With the persisted
    list, the val set stays whatever was chosen on the FIRST run; ALL stems
    added LATER go to TRAIN (a deliberate, simple choice: cross-epoch
    comparability of the val set is worth more than growing val
    proportionally — the val share may drift slightly below 2% over time, with
    no practical impact since the quick evaluation already uses a fixed `n=24`
    subset).

    Stems recorded in the file that NO LONGER exist on disk are silently
    dropped from val (if data was deleted, the split still stays consistent)."""
    persist_path = Path(persist_path)
    if persist_path.exists():
        saved = json.loads(persist_path.read_text())
        saved_val = set(saved["val_stems"])
        all_set = set(all_stems)
        val = sorted(saved_val & all_set)
        train = sorted(all_set - saved_val)
        return train, val

    train, val = deterministic_val_split(all_stems, seed=seed, val_fraction=val_fraction)
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    persist_path.write_text(
        json.dumps({"seed": seed, "val_fraction": val_fraction, "val_stems": val}, ensure_ascii=False, indent=1)
    )
    return train, val


def fixed_eval_subset(val_stems: list[str], seed: int, n: int) -> list[str]:
    """Selects from the VAL set (2% — can be hundreds of images) a fixed subset
    of `n` (default 24) images that is the SAME every epoch — so that the
    periodic quick evaluation is comparable across epochs (MAE measured on
    different random images each time would let noise hide the epoch-to-epoch
    trend)."""
    import random

    stems_sorted = sorted(val_stems)
    rng = random.Random(seed)
    shuffled = stems_sorted[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[: min(n, len(shuffled))])


# ============================================================================
# 4) Small pieces of the official BiRefNet train.py/config.py logic
# ============================================================================
def should_apply_finetune_reweight(epoch: int, total_epochs: int, finetune_last_epochs: int) -> bool:
    """The condition inside the official BiRefNet `train.py::Trainer.train_epoch`:
    `if epoch > args.epochs + config.finetune_last_epochs:` (source:
    ZhengPeng7/BiRefNet `train.py`, GitHub `main` branch, function
    `train_epoch`, around line ~195 — fetched into this repo with `curl` and
    reviewed, see the Phase 3 report). `finetune_last_epochs` is a NEGATIVE
    number (`-10` for the `Matting` task in `config.py` — during the last 10
    epochs the pixel loss weights are changed gradually, the "documented
    fine-tune trick"). `total_epochs` is the FINAL TARGET epoch count of the
    training, NOT of that Colab SESSION (the `EPOCHS` parameter — on resumes
    the SAME value must be passed in EVERY SESSION, otherwise this threshold
    drifts from session to session).

    Two guards ADDED on top of the official condition (for short runs — the
    official code assumed EPOCHS>=150 and never handled this case):
    - `finetune_last_epochs == 0` -> always False (`config.py` comment:
      "choose 0 to skip").
    - `total_epochs <= |finetune_last_epochs|` (e.g. EPOCHS=6, ft=-10) ->
      always False: the window start (total+ft+1) would fall BEFORE epoch 1
      and the decay exponent would already be n>1 at the very first epoch
      (e.g. 0.9^5) — corrupting the loss weights before training even starts
      makes no sense in short exploratory runs, so the trick is skipped
      entirely. Thanks to this guard, whenever the trick IS applied the
      exponent always starts from n>=1
      (epoch > total+ft >= 1 -> n = epoch-(total+ft) >= 1)."""
    if finetune_last_epochs == 0:
        return False
    if finetune_last_epochs < 0 and total_epochs <= -finetune_last_epochs:
        return False
    return epoch > total_epochs + finetune_last_epochs


def effective_lr(task: str, batch_size: int, accum_steps: int, base_lr_override: float | None = None) -> float:
    """An ADAPTED version of the formula in the official BiRefNet `config.py`:
    `self.lr = (1e-4 if 'DIS5K' in self.task else 1e-5) * math.sqrt(self.batch_size / 4)`
    (source: `config.py`, GitHub `main`, `Config.__init__`). The official code
    has NO gradient accumulation (`train.py` HARD-CODES
    `accelerator.gradient_accumulation_steps=1`, and the
    `accelerator.accumulate(...)` context is left COMMENTED OUT — unused);
    since this notebook ADDS gradient accumulation, we widen the `batch_size`
    in the formula to the REAL (effective) batch per optimizer step
    (`batch_size * accum_steps`) — to correctly apply the official formula's
    "grow lr with the square root of the effective batch" logic to an
    effective batch that now grows along two axes (physical batch +
    accumulation). If `base_lr_override` is set (`LR` manually configured in
    the parameters cell), this computation is skipped entirely."""
    if base_lr_override is not None:
        return float(base_lr_override)
    base = 1e-4 if "DIS5K" in task else 1e-5
    effective_batch = batch_size * accum_steps
    return base * math.sqrt(effective_batch / 4)


# ============================================================================
# 5) BiRefNet config.py text patching (IDEMPOTENT)
# ============================================================================
_TASK_LIST = ["DIS5K", "COD", "HRSOD", "General", "General-2K", "Matting"]
_TASK_LINE_RE = re.compile(
    r"self\.task = (\['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'\])\[\d+\]"
)
_HOME_LINE_RE = re.compile(r"self\.sys_home_dir = \[os\.path\.expanduser\('~'\), '[^']*'\]\[1\]")
_BS_LINE_RE = re.compile(r"self\.batch_size = \d+")


def apply_config_patches(src: str, task: str, sys_home_dir: str, batch_size: int) -> str:
    """Applies three patches to the BiRefNet `config.py` source: (1) the
    selected `self.task` index, (2) the second element of `self.sys_home_dir`
    (the root of data_root_dir), (3) `self.batch_size`. The line patterns were
    verified against `Config.__init__` on the GitHub `main` branch (see the
    Phase 3 report).

    IDEMPOTENT and re-parameterizable: the regexes match the line BOTH in its
    original (unpatched) AND in its previously patched form — re-running the
    notebook on the same VM (with the same or DIFFERENT parameter values) does
    not fail, `apply(apply(src)) == apply(src)`. If a pattern does not match
    at all (the repo `main` branch has changed), a clear ValueError is raised
    instead of passing SILENTLY."""
    if task not in _TASK_LIST:
        raise ValueError(f"unknown task: {task!r} (valid: {_TASK_LIST})")
    idx = _TASK_LIST.index(task)

    patched, n = _TASK_LINE_RE.subn(rf"self.task = \1[{idx}]", src, count=1)
    if n == 0:
        raise ValueError(
            "expected `self.task = [...][N]` line not found in config.py — "
            "the BiRefNet main branch may have changed; inspect config.py manually."
        )
    patched, n = _HOME_LINE_RE.subn(
        f"self.sys_home_dir = [os.path.expanduser('~'), '{sys_home_dir}'][1]", patched, count=1
    )
    if n == 0:
        raise ValueError("expected `self.sys_home_dir = [...]` line not found in config.py.")
    patched, n = _BS_LINE_RE.subn(f"self.batch_size = {batch_size}", patched, count=1)
    if n == 0:
        raise ValueError("expected `self.batch_size = N` line not found in config.py.")
    return patched


# ============================================================================
# 6) Drive -> local disk data copying (size-validated, idempotent)
# ============================================================================
def copy_pairs(
    stems: list[str],
    src_im_dir: str | Path,
    src_gt_dir: str | Path,
    dst_im_dir: str | Path,
    dst_gt_dir: str | Path,
    im_ext: str = ".jpg",
    gt_ext: str = ".png",
    max_workers: int = 16,
) -> int:
    """Copies the (im, gt) pairs in `stems` from source to destination;
    returns the number of pairs copied. Idempotent: a pair is skipped only if
    BOTH im AND gt exist at the destination AND both file sizes exactly match
    the source — checking only the im size is not enough, since in a
    half-finished Colab copy the gt file may have been left truncated; in that
    case the pair is copied AGAIN (repair).

    Copying tens of thousands of small files over the Drive FUSE mount with a
    SINGLE THREAD takes hours (measured in a live Colab session); therefore
    each pair is an independent unit of work distributed over `max_workers`
    threads via `ThreadPoolExecutor` (I/O-bound copying — the GIL is not a
    bottleneck here). Since each pair's destination files are unique to it,
    there is NO shared state between threads (no race-condition risk); the
    result (copied count, skipped/repaired pairs) is ORDER-INDEPENDENT and
    identical to a serial run. Errors in individual pairs are NOT raised
    immediately — ALL of them are collected, all remaining pairs are processed
    (partial progress is not lost), and at the end the FIRST error is re-raised
    together with the total error count. Progress (rate + ETA) is printed to
    the console every 2000 completed pairs."""
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    src_im_dir, src_gt_dir = Path(src_im_dir), Path(src_gt_dir)
    dst_im_dir, dst_gt_dir = Path(dst_im_dir), Path(dst_gt_dir)

    def _copy_one(stem: str) -> bool:
        src_im, src_gt = src_im_dir / f"{stem}{im_ext}", src_gt_dir / f"{stem}{gt_ext}"
        dst_im, dst_gt = dst_im_dir / f"{stem}{im_ext}", dst_gt_dir / f"{stem}{gt_ext}"
        if (
            dst_im.exists()
            and dst_gt.exists()
            and dst_im.stat().st_size == src_im.stat().st_size
            and dst_gt.stat().st_size == src_gt.stat().st_size
        ):
            return False
        shutil.copy2(src_im, dst_im)
        shutil.copy2(src_gt, dst_gt)
        return True

    total = len(stems)
    copied = 0
    completed = 0
    errors: list[tuple[str, BaseException]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stem = {executor.submit(_copy_one, stem): stem for stem in stems}
        for future in as_completed(future_to_stem):
            stem = future_to_stem[future]
            completed += 1
            try:
                if future.result():
                    copied += 1
            except Exception as exc:  # per-item error: collect, keep processing
                errors.append((stem, exc))
            if completed % 2000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                print(
                    f"copy_pairs: {completed}/{total} done "
                    f"({rate:.1f} pairs/s, ETA {eta:.0f}s)"
                )

    if errors:
        first_stem, first_exc = errors[0]
        raise RuntimeError(
            f"copy_pairs: {len(errors)}/{total} pairs failed to copy "
            f"(first error, stem={first_stem!r}: {first_exc!r})"
        ) from first_exc

    return copied


# ============================================================================
# 7) v3 — VAL leak exclusion + composite manifest merge
# ============================================================================
_COMPOSITE_COPY_SUFFIX_RE = re.compile(r"_[vo]\d{2}$")


def strip_composite_copy_suffix(stem: str) -> str:
    """`<source_id>_v<NN>` or `<source_id>_o<NN>` -> `<source_id>` (see the
    `scripts/make_composites.py` naming contract: `_v<NN>` composited copies,
    `_o<NN>` original-background copies). If there is no match (an unexpected
    stem), `stem` is returned AS IS.

    CAUTION — a non-match is a LEAK RISK, not harmless: a non-matching VAL
    stem enters the exclusion set in its SUFFIXED (wrong) form; that string
    matches no `id` in the source manifest, so the REAL source id is NOT
    excluded and an `_o00` copy of that source IS generated into the training
    set — the guard is effectively BYPASSED for that source (the same image is
    seen both in TRAIN as `_o00` and in VAL through another copy). Callers
    must therefore ALWAYS diagnose non-matching stems —
    `derive_val_excluded_source_ids` returns them separately, and
    `training/v3_veri_guncelleme_hucresi.py` prints a loud warning on a
    non-empty mismatch list."""
    return _COMPOSITE_COPY_SUFFIX_RE.sub("", stem)


def derive_val_excluded_source_ids(val_stems: list[str]) -> tuple[set[str], list[str]]:
    """Derives the SOURCE row ids from the (composite) stems in the VAL set —
    these ids must be excluded from `scripts/make_composites.py`'s `_o00`
    generation (VAL leak guard): even though VAL_HOLDOUT already contains
    specific `_v<NN>`/`_o<NN>` copies, adding ANOTHER `_o00` copy of the SAME
    source image to the training set would mean that image is seen (even if
    via a different variant) in both TRAIN and VAL — the model could memorize
    that SOURCE image. The `"val_stems"` list in the `val_stems.json` written
    by `training.train_colab_lib.load_or_create_val_split` is fed directly
    into this function (see `training/v3_veri_guncelleme_hucresi.py`).

    Returns: `(excluded_source_ids, unmatched_stems)`. `unmatched_stems` are
    the val stems that do NOT match the suffix pattern (`_[vo]\\d{2}$`) —
    since they enter the exclusion set as-is (suffixed/wrong form), they match
    no id in the source manifest and the guard is effectively BYPASSED for
    those sources (details: the `strip_composite_copy_suffix` docstring). The
    caller must report a non-empty `unmatched_stems` LOUDLY (see the
    `stage_composites_o` warning in the v3 cell)."""
    excluded: set[str] = set()
    unmatched: list[str] = []
    for s in val_stems:
        stripped = strip_composite_copy_suffix(s)
        if stripped == s:
            unmatched.append(s)
        excluded.add(stripped)
    return excluded, unmatched


def merge_composite_manifest(local_manifest_path: str | Path, drive_manifest_path: str | Path) -> int:
    """APPENDs the rows from `local_manifest_path` to `drive_manifest_path` —
    only the `id`s not yet PRESENT in `drive_manifest_path` (dedupe;
    idempotent: if the same call is made twice, the second call appends 0
    rows). `drive_manifest_path` (the large Drive copy — ~28k+ rows — already
    containing ALL of v1/v2's `_v<NN>` rows) is NEVER read in full and
    REWRITTEN, only opened and appended to
    (`benchmark.testset.append_entries`) — this is the DELIBERATE DEPARTURE of
    `training/v3_veri_guncelleme_hucresi.py` from the `colab_devam_hucresi.py`
    pattern of fully overwriting with `shutil.copy2` (in that file the local
    composite manifest was already COMPLETE/current, so overwriting was safe;
    here the local manifest contains only the NEW `_o00` rows). If
    `local_manifest_path` does not exist (no `_o00` was ever generated),
    silently returns `0`.

    If `drive_manifest_path` exists, its rows are read ONE BY ONE and only the
    `id` fields are added to the set (the full `load_manifest` + `_validate`
    is not called — to avoid unnecessary validation/memory cost on the large
    file when only the id set is needed); `local_manifest_path` (small, ~14k
    rows) is read with the full `load_manifest` (including the duplicate-id
    guard)."""
    local_manifest_path = Path(local_manifest_path)
    drive_manifest_path = Path(drive_manifest_path)
    if not local_manifest_path.exists():
        return 0

    local_rows = load_manifest(str(local_manifest_path))
    existing_ids: set[str] = set()
    if drive_manifest_path.exists():
        with open(drive_manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    new_rows = [r for r in local_rows if r["id"] not in existing_ids]
    if new_rows:
        append_entries(str(drive_manifest_path), new_rows)
    return len(new_rows)


def ensure_manifest_pairs(manifest_path: str | Path, min_pairs: int = 1) -> int:
    """Returns the number of rows with GT (gt_alpha != null) in
    `manifest_path`; if the file is missing or the count is below `min_pairs`,
    raises a CLEAR `RuntimeError` — preventing the pipeline from continuing
    with an empty/deficient manifest and crashing much further down with a
    baffling error (e.g. the export's FileNotFoundError) (lesson from the live
    v3 run: the manifest was built with 0 pairs while the raw data had never
    downloaded, and the failure only surfaced at the export stage — as a
    SYMPTOM; this guard catches the CAUSE, right after manifest setup,
    loudly). See the end of the "manifest" stage in
    `training/v3_veri_guncelleme_hucresi.py`."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise RuntimeError(
            f"manifest file missing: {manifest_path} — raw data download/manifest setup "
            f"must have failed; inspect the logs of the earlier stages."
        )
    n = sum(1 for r in load_manifest(str(manifest_path)) if r.get("gt_alpha"))
    if n < min_pairs:
        raise RuntimeError(
            f"manifest contains only {n} pairs with GT (< {min_pairs}): {manifest_path} — "
            f"raw data sources may be missing/empty; NOT proceeding to export "
            f"(continuing with an empty manifest causes baffling errors downstream)."
        )
    return n


# ============================================================================
# 8) Packing the TRAIN data into tar shards (Drive FUSE speedup)
#    Packing side: training/veri_tar_paketleme_hucresi.py (free CPU Colab,
#    paste-run). Consuming side: training/train_colab.ipynb cell (c)
#    (validates the manifest with `tcl.validate_tar_manifest`, then downloads/
#    extracts the shards). The three functions below are copied VERBATIM into
#    the packing cell — by its paste-run design that cell must NOT require a
#    repo clone; the single source of truth is HERE, and drift between the
#    copy and this source is caught by the AST comparison test in
#    tests/test_train_colab_lib.py.
# ============================================================================
def tar_shard_name(index: int) -> str:
    """Shard tar file name for `index` (0-based): `TRAIN_shard_{index:02d}.tar`.
    The SINGLE source of the naming contract — the packing cell writes under
    this name, the notebook side reads via the `name` fields in the manifest."""
    if index < 0:
        raise ValueError(f"index must be >= 0: {index}")
    return f"TRAIN_shard_{index:02d}.tar"


def split_stems_to_shards(stems: list[str], shard_size: int) -> list[list[str]]:
    """Splits `stems` into ORDERED, DETERMINISTIC shards: first `sorted()`,
    then consecutive `shard_size`-sized slices — the result is INDEPENDENT of
    the input (filesystem listing) order and IDENTICAL across re-runs
    (idempotent shard skipping is only possible this way: the same stem set
    lands in the same shard on every run). The total is PRESERVED: the
    consecutive concatenation of the slices is `sorted(stems)` itself (no
    loss/duplication); the last slice may be shorter than `shard_size`.
    Empty list -> empty list. `shard_size <= 0` -> ValueError."""
    if shard_size <= 0:
        raise ValueError(f"shard_size must be > 0: {shard_size}")
    stems_sorted = sorted(stems)
    return [stems_sorted[i : i + shard_size] for i in range(0, len(stems_sorted), shard_size)]


def validate_tar_manifest(manifest: dict, expected_total: int | None = None) -> int:
    """Validates the internal consistency of the tar manifest
    (`bg-remover-data/tar/_manifest.json`) and returns `total_pairs`; raises a
    CLEAR RuntimeError on every inconsistency (continuing silently = the risk
    of training on missing/corrupt data):
    - `shards` must be a non-empty list, `total_pairs` a positive integer;
    - every shard entry must have `name`/`pairs`/`bytes`, with `pairs`/`bytes` > 0;
    - shard names must be unique (the same tar must not be counted twice);
    - the sum of shard `pairs` must equal `total_pairs`;
    - if `expected_total` is given, `total_pairs` must also equal it (the
      packing cell passes the source TRAIN listing length — guaranteeing the
      manifest describes the same dataset as the Drive listing)."""
    shards = manifest.get("shards")
    total = manifest.get("total_pairs")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(
            f"tar manifest has no non-empty 'shards' list (the packing cell may "
            f"never have run, or may have died halfway): {shards!r}"
        )
    if not isinstance(total, int) or total <= 0:
        raise RuntimeError(f"tar manifest has no positive 'total_pairs' field: {total!r}")
    names: list[str] = []
    total_from_shards = 0
    for entry in shards:
        name, pairs, n_bytes = entry.get("name"), entry.get("pairs"), entry.get("bytes")
        if not name or not isinstance(pairs, int) or pairs <= 0 or not isinstance(n_bytes, int) or n_bytes <= 0:
            raise RuntimeError(f"corrupt shard entry (name/pairs/bytes missing or <= 0): {entry!r}")
        names.append(name)
        total_from_shards += pairs
    if len(set(names)) != len(names):
        raise RuntimeError(f"tar manifest contains duplicate shard names: {names}")
    if total_from_shards != total:
        raise RuntimeError(
            f"sum of shard 'pairs' ({total_from_shards}) does not match the manifest's "
            f"'total_pairs' value ({total}) — manifest is corrupt, re-run the packing cell."
        )
    if expected_total is not None and total != expected_total:
        raise RuntimeError(
            f"manifest 'total_pairs' value ({total}) does not match the expected "
            f"source pair count ({expected_total})."
        )
    return total
