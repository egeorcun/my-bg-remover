import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compare_v1 as cv1  # noqa: E402


def _metrics():
    return {
        "per_category": {
            "birefnet-hr": {"camo": {"mae": 0.10, "sad": 10.0}},
            "bgr-v1": {"camo": {"mae": 0.05, "sad": 20.0}},
        },
        "overall": {
            "birefnet-hr": {"mae": 0.10, "sad": 10.0},
            "bgr-v1": {"mae": 0.05, "sad": 20.0},
        },
    }


def test_build_table_reports_improvement_and_regression():
    table = cv1.build_table(_metrics(), v1_models=["bgr-v1"], baseline_models=["birefnet-hr"])
    assert "bgr-v1 vs baselines" in table
    # mae improved (0.05 < 0.10) -> must be marked "better"
    assert "better" in table
    # sad regressed (20.0 > 10.0) -> must be marked "worse"
    assert "worse" in table


def test_missing_v1_model_warns_without_crashing():
    table = cv1.build_table(_metrics(), v1_models=["bgr-v1", "bgr-v1+refine"], baseline_models=["birefnet-hr"])
    assert "bgr-v1+refine" in table
    assert "WARNING" in table
    assert "not found" in table


def test_absent_baseline_skipped_gracefully():
    table = cv1.build_table(_metrics(), v1_models=["bgr-v1"], baseline_models=["ideogram"])
    # ideogram is not in metrics.json -> the table is still generated, just without that column
    assert "ideogram" not in table
    assert "bgr-v1" in table


def test_delta_cell_shows_baseline_value_and_direction():
    assert "better" in cv1._delta_cell(0.05, 0.10)
    assert "worse" in cv1._delta_cell(0.10, 0.05)
    assert cv1._delta_cell(0.10, 0.10).count("=") == 1
