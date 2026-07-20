"""Markdown table comparing bgr-v1 (fine-tune) fine-tune results with baseline models.

Reads the `metrics.json` produced by `benchmark/run.py` (per_category + overall),
compares `bgr-v1`/`bgr-v1+refine` with the default baselines (`rmbg-2.0`,
`birefnet-hr`, `rmbg-2.0+refine`). Ideogram is not in metrics.json because it is
not a segmenter run through `bgr/registry.py` (no GT-comparison metric is computed
for it; it is only a visual reference in the gallery); this script includes
ideogram in the list only if it is actually present in metrics.json.

Usage:
    uv run python scripts/compare_v1.py --metrics results/baseline/metrics.json
    uv run python scripts/compare_v1.py --metrics results/baseline/metrics.json \
        --v1 bgr-v1,bgr-v1+refine --baselines rmbg-2.0,birefnet-hr,rmbg-2.0+refine
"""
import argparse
import json
from pathlib import Path

METRIC_ORDER = ["mae", "sad", "mse", "grad", "conn"]  # all: lower = better


def _delta_cell(v1_value: float, baseline_value: float) -> str:
    """The baseline's own value + v1's arrow/percent delta relative to it (lower=better).

    The arrow shows how v1 fares against this baseline: if v1 < baseline, v1 is
    better (↓ good); if v1 > baseline, v1 is worse (↑ bad).
    """
    if baseline_value == 0:
        return f"{baseline_value:.4f}"
    delta_pct = (v1_value - baseline_value) / abs(baseline_value) * 100
    if v1_value < baseline_value:
        arrow = "↓ v1 better"
    elif v1_value > baseline_value:
        arrow = "↑ v1 worse"
    else:
        arrow = "="
    return f"{baseline_value:.4f} ({arrow} {delta_pct:+.1f}%)"


def build_table(metrics: dict, v1_models: list[str], baseline_models: list[str]) -> str:
    per_category = metrics.get("per_category", {})
    overall = metrics.get("overall", {})

    # only baselines actually present in metrics.json (e.g. ideogram is not included if it ran without GT)
    present_baselines = [b for b in baseline_models if b in overall]
    present_v1 = [v for v in v1_models if v in overall]
    missing_v1 = [v for v in v1_models if v not in overall]

    lines: list[str] = []
    lines.append("# bgr-v1 comparison report")
    lines.append("")
    if missing_v1:
        lines.append(
            f"> WARNING: v1 model(s) not found in metrics.json: {', '.join(missing_v1)} "
            "— a `benchmark.run --models " + ",".join(missing_v1) + "` run is needed first."
        )
        lines.append("")
    if not present_v1:
        lines.append("No v1 model to compare was found, no table generated.")
        return "\n".join(lines)

    all_categories = sorted({c for m in present_v1 + present_baselines for c in per_category.get(m, {})})

    for v1_name in present_v1:
        lines.append(f"## {v1_name} vs baselines")
        lines.append("")
        header = ["category", "metric", v1_name] + present_baselines
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for cat in all_categories:
            cat_metrics = per_category.get(v1_name, {}).get(cat)
            if cat_metrics is None:
                continue
            for metric in METRIC_ORDER:
                if metric not in cat_metrics:
                    continue
                v1_value = cat_metrics[metric]
                row = [cat, metric, f"{v1_value:.4f}"]
                for b in present_baselines:
                    b_value = per_category.get(b, {}).get(cat, {}).get(metric)
                    row.append(_delta_cell(v1_value, b_value) if b_value is not None else "n/a")
                lines.append("| " + " | ".join(row) + " |")

        # overall
        lines.append("")
        lines.append(f"**Overall — {v1_name}**")
        lines.append("")
        header = ["metric", v1_name] + present_baselines
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for metric in METRIC_ORDER:
            if metric not in overall.get(v1_name, {}):
                continue
            v1_value = overall[v1_name][metric]
            row = [metric, f"{v1_value:.4f}"]
            for b in present_baselines:
                b_value = overall.get(b, {}).get(metric)
                row.append(_delta_cell(v1_value, b_value) if b_value is not None else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="results/baseline/metrics.json")
    ap.add_argument("--v1", default="bgr-v1,bgr-v1+refine")
    ap.add_argument("--baselines", default="rmbg-2.0,birefnet-hr,rmbg-2.0+refine,ideogram")
    a = ap.parse_args()

    metrics = json.loads(Path(a.metrics).read_text())
    table = build_table(
        metrics,
        v1_models=a.v1.split(","),
        baseline_models=a.baselines.split(","),
    )
    print(table)


if __name__ == "__main__":
    main()
