"""Static validation for `training/train_colab.ipynb` (Phase 3, BiRefNet
fine-tune) — the ONLY validation layer that can run without GPU/Colab/Drive:
JSON/nbformat structure + `ast.parse` of every code cell (with line magics —
lines starting with `!`/`%` — stripped). The same method was used for
`training/prepare_data_colab.ipynb` (see the internal review notes (not in the repo))."""
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
    _load_notebook()  # nbformat.validate is called inside; OK if it doesn't raise


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
    assert not errors, f"ast.parse errors: {errors}"


def test_parameters_cell_defines_required_names():
    """Task item 5: EPOCHS, BATCH, ACCUM, LR, RESUME, DATA_DIR, N_EVAL_EVERY
    must be defined in the parameters cell."""
    nb = _load_notebook()
    code_sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for name in ("EPOCHS", "BATCH", "ACCUM", "LR", "RESUME", "DATA_DIR", "N_EVAL_EVERY"):
        assert f"{name} = " in code_sources or f"{name}=" in code_sources, f"missing parameter: {name}"


def test_notebook_documents_key_mechanism_choices():
    """Report requirement: the rationale for the init-weights, sampler and
    resume mechanisms must be documented inside the notebook
    (comments/markdown)."""
    nb = _load_notebook()
    all_text = "\n".join(c.source for c in nb.cells)
    assert "WeightedRandomSampler" in all_text
    assert "from_pretrained" in all_text
    assert "find_latest_checkpoint" in all_text
    assert "BiRefNet_HR-matting" in all_text


def test_data_copy_cell_has_tar_fast_path_with_copy_pairs_fallback():
    """The data copy cell (c): if a tar manifest EXISTS, the shard
    download+extract path (size-validated copy, extractall(filter="data"),
    manifest validation, VAL moving); if NOT, the old copy_pairs path
    (backward compatibility) — see training/veri_tar_paketleme_hucresi.py."""
    nb = _load_notebook()
    cells = [c for c in nb.cells if c.cell_type == "code" and "tcl.copy_pairs(train_stems" in c.source]
    assert len(cells) == 1, "data copy cell (with the copy_pairs fallback) not found"
    src = cells[0].source
    # the tar fast path
    assert "_manifest.json" in src
    assert "tcl.validate_tar_manifest" in src
    assert 'extractall' in src and 'filter="data"' in src
    # the persistent VAL split + the Errno 5 guard remain intact
    assert "tcl.load_or_create_val_split" in src
    assert "_iterdir_retry" in src
    # on the tar path the val stems are MOVED (replace) from TRAIN to val_holdout,
    # and the pairs added after the tars (the delta) still go to TRAIN via copy_pairs.
    assert ".replace(" in src
    assert "delta_train" in src and "delta_val" in src
