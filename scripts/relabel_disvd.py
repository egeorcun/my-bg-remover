"""Relabel DIS-VD manifest rows from their filename tokens (thin/complex).

Usage:
    uv run python scripts/relabel_disvd.py

Reason: sample_disvd_multi() initially distributed the DIS-VD pool RANDOMLY across
thin/complex/general. The real DIS5K class is encoded inside the id (e.g.
disvd_thin_20_Sports_8_Racket_4827171149_3140bffe12_o -> class 'Racket'). This
script ONLY fixes the 'category' field by recomputing it with classify_disvd();
ids/filenames DO NOT CHANGE (cached model outputs remain valid).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_testset import classify_disvd, parse_disvd_class  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/testset/manifest.jsonl"


def main() -> None:
    lines = MANIFEST.read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    changed = 0
    for row in rows:
        if not row["id"].startswith("disvd_"):
            continue
        old_cat = row["category"]
        cls = parse_disvd_class(row["id"])
        new_cat = classify_disvd(row["id"])
        print(f"{row['id']}: class={cls!r} category {old_cat!r} -> {new_cat!r}")
        if new_cat != old_cat:
            changed += 1
        row["category"] = new_cat

    MANIFEST.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"\n{changed} DIS-VD rows relabeled.")
    print("Final manifest category distribution:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
