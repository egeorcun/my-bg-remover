"""`training/train_colab.ipynb` (Faz 3, BiRefNet fine-tune) için statik
doğrulama — GPU/Colab/Drive olmadan çalışabilecek TEK doğrulama katmanı:
JSON/nbformat yapısı + her kod hücresinin `ast.parse`'ı (satır büyüsü — `!`/`%`
ile başlayan satırlar — çıkarılarak). Aynı yöntem `training/prepare_data_colab.ipynb`
için de kullanıldı (bkz. `.superpowers/sdd/colab-devam-report.md`)."""
import ast
from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "training" / "train_colab.ipynb"


def _load_notebook():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(nb)
    return nb


def _strip_magics(source: str) -> str:
    lines = [ln for ln in source.splitlines() if not ln.strip().startswith(("!", "%"))]
    return "\n".join(lines)


def test_notebook_exists():
    assert NOTEBOOK_PATH.is_file()


def test_notebook_is_valid_nbformat():
    _load_notebook()  # nbformat.validate içeride çağrılıyor, hata fırlatmazsa OK


def test_notebook_has_markdown_and_code_cells():
    nb = _load_notebook()
    cell_types = {c.cell_type for c in nb.cells}
    assert "markdown" in cell_types
    assert "code" in cell_types
    assert len(nb.cells) > 10


def test_every_code_cell_parses_as_valid_python():
    nb = _load_notebook()
    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        cleaned = _strip_magics(cell.source)
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            errors.append((i, str(e)))
    assert not errors, f"ast.parse hataları: {errors}"


def test_parameters_cell_defines_required_names():
    """Görev madde 5: EPOCHS, BATCH, ACCUM, LR, RESUME, DATA_DIR, N_EVAL_EVERY
    parametre hücresinde tanımlı olmalı."""
    nb = _load_notebook()
    code_sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for name in ("EPOCHS", "BATCH", "ACCUM", "LR", "RESUME", "DATA_DIR", "N_EVAL_EVERY"):
        assert f"{name} = " in code_sources or f"{name}=" in code_sources, f"parametre eksik: {name}"


def test_notebook_documents_key_mechanism_choices():
    """Rapor gereksinimi: init-weights, sampler ve resume mekanizmalarının
    gerekçesi notebook içinde (yorum/markdown) belgelenmiş olmalı."""
    nb = _load_notebook()
    all_text = "\n".join(c.source for c in nb.cells)
    assert "WeightedRandomSampler" in all_text
    assert "from_pretrained" in all_text
    assert "find_latest_checkpoint" in all_text
    assert "BiRefNet_HR-matting" in all_text
