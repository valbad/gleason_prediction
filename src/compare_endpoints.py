"""
Compare two binary classification endpoints across dataset statistics and
model results.

Endpoints
---------
binary_label_int         — GG2+ / csPCa  (Gleason ≥ 3+4=7)
binary_label_gg3plus_int — GG3+ / high-grade  (Gleason ≥ 4+3=7)

Reads
-----
data/share/needle_features_v1.csv
reports/shareable_tabular_results_binary_label_int.csv          (optional)
reports/shareable_tabular_results_binary_label_gg3plus_int.csv  (optional)

Writes
------
reports/endpoint_comparison.csv   — model results augmented with lift columns
reports/endpoint_comparison.md    — full comparison report

Usage
-----
    python reports/compare_endpoints.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

ENDPOINTS: dict[str, str] = {
    "binary_label_int":         "GG2+ / csPCa (Gleason ≥ 3+4=7)",
    "binary_label_gg3plus_int": "GG3+ / high-grade (Gleason ≥ 4+3=7)",
}

PATIENT_COL  = "patient_number"
SPLIT_COL    = "split"
VALID_SPLITS = ("train", "val", "test")


# ── Dataset statistics ────────────────────────────────────────────────────────

def dataset_stats(df_raw: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-split row / patient / positive counts for one label column."""
    mask = (
        (df_raw["label_join_status"] == "coord_match")
        & df_raw[label_col].notna()
        & df_raw[SPLIT_COL].isin(VALID_SPLITS)
    )
    df = df_raw[mask].copy()
    df[label_col] = df[label_col].astype(int)

    rows = []
    for split in list(VALID_SPLITS) + ["ALL"]:
        sub = df if split == "ALL" else df[df[SPLIT_COL] == split]
        n   = len(sub)
        rows.append({
            "label_column": label_col,
            "split":        split,
            "n_rows":       n,
            "n_patients":   int(sub[PATIENT_COL].nunique()),
            "n_positives":  int(sub[label_col].sum()),
            "prevalence":   float(sub[label_col].mean()) if n > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ── Result loading & lift computation ────────────────────────────────────────

def load_results(label_col: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / f"shareable_tabular_results_{label_col}.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found — run experiments first for '{label_col}'")
        return None
    df = pd.read_csv(path)
    df["label_column"] = label_col   # ensure column present even in older runs
    return df


def add_lift_columns(results: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Attach baseline_pr_auc (test-set prevalence) and pr_auc_lift to results.

    baseline_pr_auc = prevalence of positives in the test split for that label.
    pr_auc_lift     = test_pr_auc / baseline_pr_auc
                      (> 1 means the model beats the no-skill baseline).
    """
    test_prev = (
        stats[stats["split"] == "test"]
        .set_index("label_column")["prevalence"]
    )
    df = results.copy()
    df["baseline_pr_auc"] = df["label_column"].map(test_prev)
    df["pr_auc_lift"] = np.where(
        df["baseline_pr_auc"] > 0,
        df["test_pr_auc"] / df["baseline_pr_auc"],
        np.nan,
    )
    return df


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v*100:.1f}%" if pd.notna(v) else "n/a"


def _f3(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) and pd.notna(v) else str(v)


def _f2x(v) -> str:
    return f"{v:.2f}×" if isinstance(v, float) and pd.notna(v) else "n/a"


# ── Report sections ───────────────────────────────────────────────────────────

def section_endpoint_table() -> list[str]:
    lines = [
        "## Endpoint Definitions",
        "",
        "| Label column | Clinical definition |",
        "|---|---|",
    ]
    for col, desc in ENDPOINTS.items():
        lines.append(f"| `{col}` | {desc} |")
    lines.append("")
    return lines


def section_dataset_stats(stats: pd.DataFrame) -> list[str]:
    lines = [
        "## Dataset Statistics",
        "",
        "Rows with `label_join_status == 'coord_match'`, valid label, valid split.",
        "",
        "| Label column | Split | Rows | Patients | Positives | Prevalence |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in stats.iterrows():
        lines.append(
            f"| `{r['label_column']}` | {r['split']} "
            f"| {r['n_rows']:,} | {r['n_patients']:,} "
            f"| {r['n_positives']:,} | {_pct(r['prevalence'])} |"
        )
    lines.append("")

    # Δ positives between the two endpoints (test split)
    if set(ENDPOINTS).issubset(stats["label_column"].unique()):
        test = stats[stats["split"] == "test"].set_index("label_column")
        cols = list(ENDPOINTS)
        n0 = int(test.loc[cols[0], "n_positives"])
        n1 = int(test.loc[cols[1], "n_positives"])
        p0 = test.loc[cols[0], "prevalence"]
        p1 = test.loc[cols[1], "prevalence"]
        lines += [
            f"> **Test-set difference:** switching from `{cols[0]}` to `{cols[1]}` "
            f"removes {n0 - n1:,} positive labels "
            f"({_pct(p0)} → {_pct(p1)} prevalence).",
            "",
        ]
    return lines


def section_model_results(results: pd.DataFrame) -> list[str]:
    if results.empty:
        return [
            "## Model Results",
            "",
            "_No result files found. Run `src/run_shareable_tabular_experiments.py`",
            "with each `--label-column` option first._",
            "",
        ]

    lines = [
        "## Model Results by Endpoint",
        "",
        "- **baseline_pr_auc** = positive prevalence in the test split  ",
        "- **pr_auc_lift** = `test_pr_auc / baseline_pr_auc`  ",
        "  (1.0 = no-skill baseline; higher is better)",
        "",
    ]

    display_cols = [
        "experiment", "model",
        "test_roc_auc", "test_pr_auc", "baseline_pr_auc", "pr_auc_lift",
        "test_sensitivity", "test_specificity", "test_f1",
    ]

    for label_col, desc in ENDPOINTS.items():
        sub = results[results["label_column"] == label_col]
        lines += [f"### `{label_col}` — {desc}", ""]
        if sub.empty:
            lines += ["_No results for this endpoint._", ""]
            continue

        for mode in sub["mode"].unique():
            msub = sub[sub["mode"] == mode].sort_values(
                ["experiment", "test_roc_auc"], ascending=[True, False]
            )
            lines += [
                f"#### {mode}",
                "",
                "| " + " | ".join(display_cols) + " |",
                "|" + "---|" * len(display_cols),
            ]
            for _, r in msub.iterrows():
                cells = []
                for c in display_cols:
                    v = r.get(c, np.nan)
                    if c == "pr_auc_lift":
                        cells.append(_f2x(v))
                    else:
                        cells.append(_f3(v))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    return lines


def section_side_by_side(results: pd.DataFrame) -> list[str]:
    if results.empty:
        return []

    ep_cols = list(ENDPOINTS)

    lines = [
        "## Side-by-side: Best Model per Experiment",
        "",
        "Best model selected by `test_roc_auc` within each (endpoint, mode, experiment).",
        "",
    ]

    # Header: experiment | mode | model | GG2+ ROC | GG2+ PR-lift | GG3+ ROC | GG3+ PR-lift
    header = (
        "| experiment | mode | best model | "
        + " | ".join(f"`{c}` ROC-AUC" for c in ep_cols)
        + " | "
        + " | ".join(f"`{c}` PR-lift" for c in ep_cols)
        + " |"
    )
    sep = "|" + "---|" * (4 + 2 * len(ep_cols))
    lines += [header, sep]

    pairs = (
        results[["mode", "experiment"]]
        .drop_duplicates()
        .sort_values(["mode", "experiment"])
    )
    for _, pair in pairs.iterrows():
        mode_name = pair["mode"]
        exp_name  = pair["experiment"]

        roc_cells  = []
        lift_cells = []
        best_models: list[str] = []

        for label_col in ep_cols:
            sub = results[
                (results["label_column"] == label_col)
                & (results["mode"] == mode_name)
                & (results["experiment"] == exp_name)
            ]
            if sub.empty:
                roc_cells.append("n/a")
                lift_cells.append("n/a")
            else:
                best = sub.loc[sub["test_roc_auc"].idxmax()]
                best_models.append(best["model"])
                roc_cells.append(_f3(best["test_roc_auc"]))
                lift_cells.append(_f2x(best.get("pr_auc_lift", np.nan)))

        unique_models = list(dict.fromkeys(best_models))  # deduplicate, preserve order
        model_str = " / ".join(unique_models) if unique_models else "n/a"

        lines.append(
            "| " + " | ".join(
                [exp_name, mode_name, model_str] + roc_cells + lift_cells
            ) + " |"
        )

    lines.append("")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Endpoint comparison ===\n")

    print(f"Loading dataset: {DATA_PATH.relative_to(REPO_ROOT)} …")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  {len(df_raw):,} rows, {len(df_raw.columns)} columns\n")

    # Dataset statistics
    stat_frames: list[pd.DataFrame] = []
    for label_col in ENDPOINTS:
        if label_col not in df_raw.columns:
            print(f"  [warning] column '{label_col}' not in dataset — skipping stats")
            continue
        stat_frames.append(dataset_stats(df_raw, label_col))

    stats = pd.concat(stat_frames, ignore_index=True) if stat_frames else pd.DataFrame()

    # Print quick summary
    if not stats.empty:
        print("Dataset statistics (test split):")
        test_stats = stats[stats["split"] == "test"]
        for _, r in test_stats.iterrows():
            print(
                f"  {r['label_column']:<35s}  "
                f"n={r['n_rows']:>5,}  "
                f"patients={r['n_patients']:>4,}  "
                f"positives={r['n_positives']:>4,}  "
                f"prevalence={_pct(r['prevalence'])}"
            )
        print()

    # Model results
    result_frames: list[pd.DataFrame] = []
    for label_col in ENDPOINTS:
        df_res = load_results(label_col)
        if df_res is not None:
            result_frames.append(df_res)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()

    if not results.empty and not stats.empty:
        results = add_lift_columns(results, stats)

    # Save CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "endpoint_comparison.csv"
    if not results.empty:
        results.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")
    else:
        stats.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}  (dataset stats only)")

    # Save Markdown
    md_lines: list[str] = [
        "# Endpoint Comparison Report",
        "",
        "Comparing two binary classification targets on the same biopsy cores.",
        "",
    ]
    md_lines += section_endpoint_table()
    md_lines += section_dataset_stats(stats)
    md_lines += section_model_results(results)
    md_lines += section_side_by_side(results)

    md_path = REPORTS_DIR / "endpoint_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
