"""DATA TAR PACKING CELL — pasted and run as a SINGLE cell in a fresh (FREE,
CPU is enough — NO GPU NEEDED) Colab session; packs the
`bg-remover-data/TRAIN/{im,gt}` pairs on Drive (52,882+52,882 small files)
into a small number of LARGE tar shards and puts them under
`bg-remover-data/tar/`.

WHY: on every training run, `training/train_colab.ipynb` cell (c) copies
these small files to the VM ONE BY ONE over Drive FUSE (~75 min, with
occasional transient 'Errno 5' errors). Once this cell has run ONCE, the
training side sees the manifest and switches to the shard download+extract
path (~10 min, ~7x speedup).

SOURCE / ATTRIBUTION: the flow pattern (Drive mount BEFORE everything ->
`report()` stage tracking -> `_listdir_retry` Errno 5 protection ->
`drive.flush_and_unmount()` at the end of the job) was taken from
`training/v4_veri_guncelleme_hucresi.py`. The 2026-07-12 lesson applies
VERBATIM: Drive writes are buffered asynchronously — if the VM is shut down
without a flush, files are SILENTLY lost. The `tar_shard_name` /
`split_stems_to_shards` / `validate_tar_manifest` functions are EXACT COPIES
from `training/train_colab_lib.py` (so that this cell, by paste-run design,
does NOT REQUIRE a repo clone + `pip install -e .`) — that file is the
SINGLE SOURCE OF TRUTH; if the copy drifts, the AST comparison test in
`tests/test_train_colab_lib.py` turns red, so update from there if you see
drift.

FLOW:
1. Drive mount -> list `TRAIN/{im,gt}` (with retry) -> verify im/gt stem
   matching (RuntimeError if any are unmatched — half pairs do not enter the
   tar).
2. The stems are split into ORDERED and DETERMINISTIC `SHARD_SIZE`-sized
   slices (52,882 pairs, SHARD_SIZE=7000 -> 8 shards, ~6-7k pairs per shard).
   Each shard holds `im/<file>` + `gt/<file>` paths inside
   `TRAIN_shard_{k:02d}.tar`.
3. Each tar is first created on the VM's LOCAL disk (writing a tar directly
   to Drive FUSE is SLOW), the member count is verified, it is copied to
   Drive with its size verified, and the local tar is deleted IMMEDIATELY
   (VM disk safety: disk ~100GB, data ~30GB — still, shards are processed
   one at a time, at most 1 shard sits on disk at any moment).
4. IDEMPOTENT: a shard that already exists on Drive and matches the expected
   pair count + byte size in a previous manifest (the final `_manifest.json`
   or the intermediate `_manifest_partial.json`) is SKIPPED — a half-finished
   run resumes safely.
5. The final `bg-remover-data/tar/_manifest.json` is written only AFTER ALL
   shards are verified (RuntimeError if the total pair count does not match
   the TRAIN listing); `_manifest_partial.json` is updated after each shard.
   The training notebook looks ONLY at the final manifest — half-finished
   packing is recognized by the absence of the manifest and it falls back to
   the old copy_pairs path.
6. Report + `drive.flush_and_unmount()`.
"""

import io
import json
import shutil
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --- Constants (Drive layout SAME as v4_veri_guncelleme_hucresi.py) ---
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
TAR_SUBDIR = "tar"                      # shards + manifest go here: bg-remover-data/tar/
SHARD_SIZE = 7000                       # 52,882 pairs -> 8 shards (7x7000 + 1x3882), shard ~3-4GB
LOCAL_TAR_DIR = Path("/content/tar_build")  # tars are first created here (local disk)

MANIFEST_NAME = "_manifest.json"            # FINAL — written only once all shards are verified
MANIFEST_PARTIAL_NAME = "_manifest_partial.json"  # updated after each shard (safe resume)

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"


# ==========================================================================
# Status reporting — EXACTLY IDENTICAL to `v4_veri_guncelleme_hucresi.py::report`.
# ==========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now()
    line = f"[{ts}] stage={stage} status={status}"
    if extra:
        line += " " + json.dumps(extra, ensure_ascii=False, default=str)
    print(line)

    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

    history = []
    if STATUS_PATH.exists():
        try:
            history = json.loads(STATUS_PATH.read_text()).get("history", [])
        except Exception:
            history = []
    history.append({"stage": stage, "status": status, "time": ts, "detail": extra})
    payload = {"stage": stage, "status": status, "time": ts, "detail": extra, "history": history}
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


# ==========================================================================
# Drive FUSE Errno 5 protections — a copy of the _listdir_retry pattern inside
# `v4_veri_guncelleme_hucresi.py::stage_drive_copy_textfx` (listing) + the
# same pattern adapted to file READING (_read_with_retry): while adding to the
# tar, 52k files are read one by one, and a transient I/O error must not take
# down the whole shard.
# ==========================================================================
def _listdir_retry(d: Path, attempts: int = 5, wait_s: int = 30) -> list[Path]:
    """Drive FUSE occasionally throws a transient 'Errno 5 I/O error' on
    directories with 50k+ files (seen in the v3/v4 runs — retrying was
    enough); waits and retries, and on the last attempt re-raises the error
    as-is."""
    for i in range(attempts):
        try:
            return list(d.iterdir())
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"WARNING: {e} while listing {d} — waiting {wait_s}s and retrying "
                  f"({i + 1}/{attempts - 1}).")
            time.sleep(wait_s)
    raise AssertionError("unreachable")


def _read_with_retry(p: Path, attempts: int = 4, wait_s: int = 15) -> bytes:
    """Reads a single file from Drive FUSE; waits and retries on a transient
    OSError. Instead of adding to the tar directly with `tf.add(p)`, we read
    into memory first: if `tf.add` errored MID-read, the tar stream would be
    corrupted with a half member — here not a single byte is written to the
    tar before the read completes."""
    for i in range(attempts):
        try:
            return p.read_bytes()
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"WARNING: {e} while reading {p} — waiting {wait_s}s and retrying "
                  f"({i + 1}/{attempts - 1}).")
            time.sleep(wait_s)
    raise AssertionError("unreachable")


# ==========================================================================
# EXACT COPY from training/train_colab_lib.py (see the module docstring: so
# that the paste-run cell does not require a repo clone; drift is caught by
# the AST test).
# ==========================================================================
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


# ==========================================================================
# Stage "env" — Drive mount (BEFORE everything: STATUS_DIR lives on Drive!) +
# source directory check. Source pattern: v4_veri_guncelleme_hucresi.py::stage0_env_sanity.
# ==========================================================================
def stage0_env() -> tuple[Path, Path, Path]:
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Could not mount Drive: {DRIVE_ROOT} does not exist"

    report("env", "running")
    data_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    train_im = data_dir / "TRAIN" / "im"
    train_gt = data_dir / "TRAIN" / "gt"
    assert train_im.is_dir() and train_gt.is_dir(), (
        f"Expected data not found on Drive: {train_im} / {train_gt} — this cell is for "
        f"packing an EXISTING TRAIN dataset (the Phase 2 / v4 cells must have run first)."
    )
    tar_dir = data_dir / TAR_SUBDIR
    report("env", "done", train_im=str(train_im), tar_dir=str(tar_dir))
    return train_im, train_gt, tar_dir


# ==========================================================================
# Stage "list" — TRAIN/{im,gt} listing (with retry) + pair validation.
# ==========================================================================
def stage_list(train_im: Path, train_gt: Path) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    report("list", "running")

    def _by_stem(d: Path, label: str) -> dict[str, Path]:
        # macOS AppleDouble leftovers (`._*`) are not images — filter them out (v4 pattern).
        files = [p for p in _listdir_retry(d) if p.is_file() and not p.name.startswith("._")]
        by_stem: dict[str, Path] = {}
        dupes: list[str] = []
        for p in sorted(files):
            if p.stem in by_stem:
                dupes.append(f"{by_stem[p.stem].name} <-> {p.name}")
            by_stem[p.stem] = p
        if dupes:
            raise RuntimeError(
                f"The {label} directory has multiple files with the same stem (which one "
                f"belongs to the pair is ambiguous — which one would enter the tar would be "
                f"undefined): {dupes[:10]}{' ...' if len(dupes) > 10 else ''}"
            )
        return by_stem

    im_by_stem = _by_stem(train_im, "TRAIN/im")
    gt_by_stem = _by_stem(train_gt, "TRAIN/gt")

    im_only = sorted(set(im_by_stem) - set(gt_by_stem))
    gt_only = sorted(set(gt_by_stem) - set(im_by_stem))
    if im_only or gt_only:
        raise RuntimeError(
            f"TRAIN im/gt stem matching is BROKEN — half pairs cannot enter the tar (the "
            f"training side would crash on these files): im without gt={len(im_only)} "
            f"(e.g. {im_only[:5]}), gt without im={len(gt_only)} (e.g. {gt_only[:5]}). "
            f"Repair the dataset first (the 2026-07-12 lesson: a half-finished Drive flush "
            f"can leave broken pairs like this)."
        )

    stems = sorted(im_by_stem)
    assert stems, "TRAIN/im is empty — no data to pack."
    print(f"TRAIN: {len(stems)} pairs verified (im={len(im_by_stem)}, gt={len(gt_by_stem)}).")
    report("list", "done", pairs=len(stems))
    return im_by_stem, gt_by_stem, stems


# ==========================================================================
# Stage "pack" — create the shards locally, copy to Drive (with size
# verification), delete the local tar. IDEMPOTENT: a shard that matches a
# previous (final or partial) manifest and sits on Drive at the correct size
# is skipped.
# ==========================================================================
def _load_previous_entries(tar_dir: Path) -> dict[str, dict]:
    prev: dict[str, dict] = {}
    # final is read first, partial AFTER (the partial may belong to a more recent half-finished run).
    for name in (MANIFEST_NAME, MANIFEST_PARTIAL_NAME):
        p = tar_dir / name
        if not p.exists():
            continue
        try:
            for e in json.loads(p.read_text()).get("shards", []):
                if e.get("name"):
                    prev[e["name"]] = e
        except Exception as exc:
            print(f"WARNING: {p} could not be read ({exc}) — this manifest will NOT BE USED for skipping.")
    return prev


def _write_partial_manifest(tar_dir: Path, entries: list[dict], n_source_pairs: int) -> None:
    (tar_dir / MANIFEST_PARTIAL_NAME).write_text(json.dumps({
        "note": "Intermediate state of a HALF-FINISHED run — the training side looks ONLY at _manifest.json.",
        "updated_at": _now(),
        "shard_size": SHARD_SIZE,
        "source_pairs": n_source_pairs,
        "shards": entries,
    }, ensure_ascii=False, indent=2))


def stage_pack(im_by_stem: dict[str, Path], gt_by_stem: dict[str, Path],
               stems: list[str], tar_dir: Path) -> list[dict]:
    report("pack", "running")
    shards = split_stems_to_shards(stems, SHARD_SIZE)
    print(f"{len(stems)} pairs -> {len(shards)} shards (SHARD_SIZE={SHARD_SIZE}).")
    tar_dir.mkdir(parents=True, exist_ok=True)
    LOCAL_TAR_DIR.mkdir(parents=True, exist_ok=True)
    prev_entries = _load_previous_entries(tar_dir)

    entries: list[dict] = []
    for k, shard in enumerate(shards):
        name = tar_shard_name(k)
        drive_tar = tar_dir / name

        # Idempotent skip: the previous manifest entry matches this shard's
        # expected pair count AND the file on Drive is at that entry's byte size.
        prev = prev_entries.get(name)
        if (
            prev
            and prev.get("pairs") == len(shard)
            and isinstance(prev.get("bytes"), int) and prev["bytes"] > 0
            and drive_tar.exists() and drive_tar.stat().st_size == prev["bytes"]
        ):
            print(f"{name}: present on Drive and matching the manifest "
                  f"({prev['pairs']} pairs, {prev['bytes'] / 1e9:.2f} GB) — SKIPPED.")
            entries.append(prev)
            _write_partial_manifest(tar_dir, entries, len(stems))
            continue

        t0 = time.time()
        local_tar = LOCAL_TAR_DIR / name
        if local_tar.exists():
            local_tar.unlink()  # half-finished local tar from a previous run — recreate from scratch
        with tarfile.open(local_tar, "w") as tf:
            for i, stem in enumerate(shard, start=1):
                for src, arc_prefix in ((im_by_stem[stem], "im"), (gt_by_stem[stem], "gt")):
                    data = _read_with_retry(src)
                    info = tarfile.TarInfo(name=f"{arc_prefix}/{src.name}")
                    info.size = len(data)
                    info.mtime = int(time.time())
                    tf.addfile(info, io.BytesIO(data))
                if i % 1000 == 0:
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"  {name}: {i}/{len(shard)} pairs added "
                          f"({rate:.1f} pairs/s, ETA {(len(shard) - i) / rate:.0f} s)")

        # Member count verification (local disk — fast): each pair is 2 members (im+gt).
        with tarfile.open(local_tar) as tf:
            n_members = len(tf.getnames())
        if n_members != 2 * len(shard):
            raise RuntimeError(
                f"{name}: tar member count does not match the expectation: {n_members} != {2 * len(shard)} "
                f"— the local tar is corrupt, re-run the cell (the shard is recreated from scratch)."
            )

        n_bytes = local_tar.stat().st_size
        print(f"{name}: local tar ready ({len(shard)} pairs, {n_bytes / 1e9:.2f} GB, "
              f"{time.time() - t0:.0f} s) — copying to Drive...")
        shutil.copy2(local_tar, drive_tar)
        drive_size = drive_tar.stat().st_size
        if drive_size != n_bytes:
            raise RuntimeError(
                f"{name}: the size of the Drive copy ({drive_size}) does not match the local "
                f"tar ({n_bytes}) — the transfer may have been cut short, re-run the cell."
            )
        local_tar.unlink()  # VM disk safety: at most 1 shard sits on disk at any moment

        entry = {"name": name, "pairs": len(shard), "files": 2 * len(shard), "bytes": n_bytes}
        entries.append(entry)
        _write_partial_manifest(tar_dir, entries, len(stems))  # safe resume point
        print(f"{name}: written to Drive and verified ({time.time() - t0:.0f} s total).")

    report("pack", "done", shards=len(entries))
    return entries


# ==========================================================================
# Stage "manifest" — the final _manifest.json (only once ALL shards are verified).
# ==========================================================================
def stage_manifest(entries: list[dict], stems: list[str],
                   im_by_stem: dict[str, Path], gt_by_stem: dict[str, Path], tar_dir: Path) -> dict:
    report("manifest", "running")
    manifest = {
        "created_at": _now(),
        "shard_size": SHARD_SIZE,
        "total_pairs": sum(e["pairs"] for e in entries),
        "source_counts": {"im": len(im_by_stem), "gt": len(gt_by_stem)},
        "shards": entries,
    }
    # RuntimeError if the total pair count does not match the TRAIN listing (task requirement).
    validate_tar_manifest(manifest, expected_total=len(stems))
    (tar_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    partial = tar_dir / MANIFEST_PARTIAL_NAME
    if partial.exists():
        partial.unlink()  # the job is done — the intermediate-state file must not cause confusion
    print(f"{tar_dir / MANIFEST_NAME}: {len(entries)} shards, {manifest['total_pairs']} pairs total.")
    report("manifest", "done", total_pairs=manifest["total_pairs"], shards=len(entries))
    return manifest


# ==========================================================================
# Orchestration — runs at top level (when the cell is pasted and executed).
# ==========================================================================
LOCAL_SRC = Path("/content/pack_src")  # local bulk copy target (localize stage)


def stage_localize(
    im_by_stem: dict[str, Path],
    gt_by_stem: dict[str, Path],
    stems: list[str],
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Copies the pairs on Drive to the VM's local disk IN PARALLEL and
    rewrites the path maps to local. WHY: reading one by one directly from
    FUSE into the tar is ~2.3 pairs/s (52.9k pairs = about 6.5 hours!); the
    parallel copy is ~11-18 pairs/s (~75 min), and tarring from local takes
    minutes. Idempotent: an existing local file whose size matches is skipped
    (an interrupted run resumes where it left off)."""
    import concurrent.futures

    report("localize", "running")
    # Disk space: this VM may have large folders left over from a previous
    # data run — clean the known temporaries to make room for tar production.
    for junk in ("/content/birefnet_format_textfx", "/content/my-bg-remover/data/train_textfx",
                 "/content/downloads", "/content/my-bg-remover/data/raw_train"):
        if Path(junk).exists():
            shutil.rmtree(junk, ignore_errors=True)
            print(f"disk cleanup: {junk} deleted.")
    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (needed ~35 GB)")

    local_im, local_gt = LOCAL_SRC / "im", LOCAL_SRC / "gt"
    local_im.mkdir(parents=True, exist_ok=True)
    local_gt.mkdir(parents=True, exist_ok=True)

    def _copy_one(stem: str) -> bool:
        pairs = ((im_by_stem[stem], local_im), (gt_by_stem[stem], local_gt))
        for src, dst_dir in pairs:
            dst = dst_dir / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                continue
            for attempt in range(4):
                try:
                    shutil.copy2(src, dst)
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(10)
        return True

    t0 = time.time()
    done = 0
    errors: list[tuple[str, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_copy_one, s): s for s in stems}
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as e:  # noqa: BLE001 - collect single-file errors
                errors.append((futures[fut], e))
            if done % 2000 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  localize: {done}/{len(stems)} pairs "
                      f"({rate:.1f} pairs/s, ETA {(len(stems) - done) / rate:.0f} s)")
    if errors:
        raise RuntimeError(
            f"localize: {len(errors)}/{len(stems)} pairs could not be copied "
            f"(first error, stem={errors[0][0]!r}: {errors[0][1]!r})"
        )
    report("localize", "done", pairs=done, seconds=int(time.time() - t0))
    new_im = {s: local_im / im_by_stem[s].name for s in stems}
    new_gt = {s: local_gt / gt_by_stem[s].name for s in stems}
    return new_im, new_gt


def main() -> None:
    train_im, train_gt, tar_dir = stage0_env()  # Drive mount happens HERE — before everything else
    im_by_stem, gt_by_stem, stems = stage_list(train_im, train_gt)
    im_by_stem, gt_by_stem = stage_localize(im_by_stem, gt_by_stem, stems)
    entries = stage_pack(im_by_stem, gt_by_stem, stems, tar_dir)
    stage_manifest(entries, stems, im_by_stem, gt_by_stem, tar_dir)
    report("ALL", "done")
    # CRITICAL (the 2026-07-12 lesson): Drive writes are buffered
    # ASYNCHRONOUSLY — if the VM is shut down before this flush completes, the
    # files (including the tar shards!) are SILENTLY lost. flush_and_unmount()
    # FORCES the buffer to drain and blocks until it finishes. It is called
    # AFTER everything that writes to Drive (including report).
    print("Flushing Drive (waiting for async writes to land in the cloud)...")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("Drive flush COMPLETE — the VM can now be safely shut down. In subsequent training "
          "runs, train_colab.ipynb cell (c) will use the tar path automatically.")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
