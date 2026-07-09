"""Manifest tabanlı test seti. JSONL: id, image, category, gt_alpha (nullable)."""
import json

CATEGORIES = {"hair", "transparent", "thin", "product", "complex", "illustration", "general"}
_KEYS = {"id", "image", "category", "gt_alpha"}


def _validate(row: dict) -> None:
    missing = _KEYS - set(row)
    if missing:
        raise ValueError(f"eksik anahtar(lar): {sorted(missing)}")
    if row["category"] not in CATEGORIES:
        raise ValueError(f"bilinmeyen kategori: {row['category']}")


def load_manifest(path: str) -> list[dict]:
    rows = []
    seen_ids: set[str] = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                _validate(row)
                if row["id"] in seen_ids:
                    raise ValueError(f"tekrarlanan id: {row['id']}")
                seen_ids.add(row["id"])
                rows.append(row)
    return rows


def append_entries(path: str, entries: list[dict]) -> None:
    for row in entries:
        _validate(row)
    with open(path, "a") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
