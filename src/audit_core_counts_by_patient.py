"""
Audit whether biopsy core counts per patient may influence patient-level
performance and labels.

For each endpoint and split this script reports:
  - Per-patient core count distribution (n, mean, std, median, IQR, min, max)
  - Core count comparison between positive and negative patients
  - Spearman correlations: n_cores vs patient_label / n_positive_cores /
    positive_core_fraction
  - Cross-split comparison of core-count distributions

Usage
-----
    python src/audit_core_counts_by_patient.py

Outputs
-------
    reports/core_counts_by_patient_audit.csv
    reports/core_counts_by_patient_audit.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

PATIENT_COL  = "patient_number"
SPLIT_COL    = "split"
VALID_SPLITS = ("train", "val", "test")

LABEL_COLS = ["binary_label_int", "binary_label_gg3plus_int"]
LABEL_DESC = {
    "binary_label_int":         "GG2+ / csPCa (Gleason ≥ 3+4=7)",
    "binary_label_gg3plus_int": "GG3+ / high-grade (Gleason ≥ 4+3=7)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v, dec: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(int(v)) if isinstance(v, (int, np.integer)) else str(v)


def _pval(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def dist_stats(series: pd.Series) -> dict:
    s = series.dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return dict(
        n=int(len(s)),
        mean=float(s.mean()),
        std=float(s.std()),
        median=float(s.median()),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        min=float(s.min()),
        max=float(s.max()),
    )


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 5:
        return np.nan, np.nan
    r, p = stats.spearmanr(x[valid], y[valid])
    return float(r), float(p)


# ── Per-patient summary ───────────────────────────────────────────────────────

def build_patient_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """One row per patient with aggregated core-count statistics."""
    agg = (
        df.groupby(PATIENT_COL)
        .agg(
            split=(SPLIT_COL, "first"),
            n_cores=(label_col, "count"),
            n_positive_cores=(label_col, "sum"),
        )
        .reset_index()
    )
    agg["patient_label"]          = (agg["n_positive_cores"] > 0).astype(int)
    agg["positive_core_fraction"] = agg["n_positive_cores"] / agg["n_cores"]
    return agg


# ── Report sections ───────────────────────────────────────────────────────────

def section_distribution(
    pat: pd.DataFrame,
    label_col: str,
) -> tuple[list[str], list[dict]]:
    desc  = LABEL_DESC[label_col]
    lines = [
        f"## Core-count distribution — `{label_col}` ({desc})",
        "",
        "One row per split. Statistics are over per-patient core counts.",
        "",
        "| Split | Patients | Mean | Std | Median | IQR | Min | Max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    csv_rows: list[dict] = []

    splits_order = list(VALID_SPLITS) + ["all"]
    for split in splits_order:
        sub = pat if split == "all" else pat[pat["split"] == split]
        st  = dist_stats(sub["n_cores"])
        lines.append(
            f"| {split} "
            f"| {st['n']:,} "
            f"| {_f(st['mean'])} "
            f"| {_f(st['std'])} "
            f"| {_f(st['median'])} "
            f"| {_f(st['iqr'])} "
            f"| {_f(st['min'], 0)} "
            f"| {_f(st['max'], 0)} |"
        )
        csv_rows.append(dict(
            section="distribution", label_col=label_col, split=split, **st
        ))

    lines.append("")
    return lines, csv_rows


def section_positive_vs_negative(
    pat: pd.DataFrame,
    label_col: str,
) -> tuple[list[str], list[dict]]:
    lines = [
        "## Core counts: positive vs negative patients",
        "",
        "| Split | Group | N patients | Mean n_cores | Median n_cores |",
        "|---|---|---|---|---|",
    ]
    csv_rows: list[dict] = []

    splits_order = list(VALID_SPLITS) + ["all"]
    for split in splits_order:
        sub = pat if split == "all" else pat[pat["split"] == split]
        for lbl, grp_name in [(1, "positive"), (0, "negative")]:
            grp = sub[sub["patient_label"] == lbl]["n_cores"]
            n   = len(grp)
            mean_c   = float(grp.mean())   if n > 0 else np.nan
            median_c = float(grp.median()) if n > 0 else np.nan
            lines.append(
                f"| {split} | {grp_name} "
                f"| {n:,} "
                f"| {_f(mean_c)} "
                f"| {_f(median_c)} |"
            )
            csv_rows.append(dict(
                section="pos_vs_neg", label_col=label_col, split=split,
                group=grp_name, n_patients=n,
                mean_n_cores=mean_c, median_n_cores=median_c,
            ))

    lines.append("")
    return lines, csv_rows


def section_correlations(
    pat: pd.DataFrame,
    label_col: str,
) -> tuple[list[str], list[dict]]:
    lines = [
        "## Spearman correlations with n_cores",
        "",
        "Computed over all patients (all splits combined).",
        "",
        "| Target | Spearman ρ | p-value | Interpretation |",
        "|---|---|---|---|",
    ]
    csv_rows: list[dict] = []

    targets = [
        ("patient_label",          "patient label (0/1)"),
        ("n_positive_cores",       "number of positive cores"),
        ("positive_core_fraction", "fraction of positive cores"),
    ]
    for col, col_desc in targets:
        r, p = spearman(pat["n_cores"], pat[col])
        if np.isnan(r):
            interp = "n/a"
        elif abs(r) < 0.10:
            interp = "negligible"
        elif abs(r) < 0.30:
            interp = "weak"
        elif abs(r) < 0.50:
            interp = "moderate"
        else:
            interp = "strong"
        sign = "positive" if r >= 0 else "negative"
        interp_str = f"{interp} {sign}" if not np.isnan(r) else "n/a"
        lines.append(
            f"| {col_desc} | {_f(r, 3)} | {_pval(p)} | {interp_str} |"
        )
        csv_rows.append(dict(
            section="correlation", label_col=label_col,
            target=col, spearman_r=r, spearman_p=p,
        ))

    lines.append("")
    return lines, csv_rows


def section_cross_split(
    pat: pd.DataFrame,
    label_col: str,
) -> tuple[list[str], list[dict]]:
    """Kruskal-Wallis test across splits and pairwise Mann-Whitney U."""
    groups = [
        pat[pat["split"] == s]["n_cores"].dropna().values
        for s in VALID_SPLITS
    ]

    if all(len(g) > 0 for g in groups):
        kw_stat, kw_p = stats.kruskal(*groups)
    else:
        kw_stat, kw_p = np.nan, np.nan

    lines = [
        "## Cross-split core-count comparison",
        "",
        "Kruskal-Wallis test for differences across train / val / test.",
        "",
        f"Kruskal-Wallis H = {_f(kw_stat, 3)},  p = {_pval(kw_p)}",
        "",
    ]

    if not np.isnan(kw_p) and kw_p < 0.05:
        lines.append(
            "> **Significant difference detected.** Core counts are not identically "
            "distributed across splits — this could bias patient-level aggregation "
            "(e.g. max-prob favours patients with more cores)."
        )
    else:
        lines.append(
            "> No significant difference across splits (p ≥ 0.05). "
            "Core count distributions appear comparable."
        )
    lines.append("")

    # Pairwise Mann-Whitney U
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    lines += [
        "Pairwise Mann-Whitney U (two-sided):",
        "",
        "| Pair | U statistic | p-value |",
        "|---|---|---|",
    ]
    csv_rows: list[dict] = [dict(
        section="cross_split", label_col=label_col,
        test="kruskal_wallis", stat=kw_stat, p_value=kw_p,
        split_a="all", split_b="all",
    )]
    split_data = {s: pat[pat["split"] == s]["n_cores"].dropna().values
                  for s in VALID_SPLITS}
    for a, b in pairs:
        ga, gb = split_data[a], split_data[b]
        if len(ga) > 0 and len(gb) > 0:
            u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            u, p = float(u), float(p)
        else:
            u, p = np.nan, np.nan
        lines.append(f"| {a} vs {b} | {_f(u, 1)} | {_pval(p)} |")
        csv_rows.append(dict(
            section="cross_split", label_col=label_col,
            test="mann_whitney_u", stat=u, p_value=p,
            split_a=a, split_b=b,
        ))

    lines.append("")
    return lines, csv_rows


def section_leakage_note(pat: pd.DataFrame, label_col: str) -> list[str]:
    """
    Flag whether positive patients systematically have more cores — this
    could cause max-prob and top3-mean aggregation to favour them spuriously.
    """
    r, p = spearman(pat["n_cores"], pat["patient_label"])
    lines = ["## Potential bias note", ""]
    if not np.isnan(r) and abs(r) >= 0.10 and p < 0.05:
        lines += [
            f"> **Weak but statistically significant correlation** (ρ = {_f(r, 3)}, p = {_pval(p)}) "
            f"between n_cores and patient label.  ",
            "> Positive patients tend to have slightly more cores. "
            "Aggregation methods that depend on core count (max-prob, top-3 mean) "
            "may be mildly influenced by this imbalance.",
            "",
        ]
    else:
        lines += [
            f"> Weak or non-significant correlation between n_cores and patient label "
            f"(ρ = {_f(r, 3)}, p = {_pval(p)}).  ",
            "> Core-count imbalance between positive and negative patients is unlikely "
            "to be a major confound.",
            "",
        ]
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Core counts by patient audit ===\n")

    raw = pd.read_csv(DATA_PATH)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_csv_rows: list[dict] = []
    all_md_lines: list[str]  = [
        "# Core Counts by Patient Audit",
        "",
        "Checks whether the number of biopsy cores per patient is balanced across  ",
        "splits and label groups, and whether it may confound patient-level aggregation.",
        "",
    ]

    for label_col in LABEL_COLS:
        desc = LABEL_DESC[label_col]
        print(f"Label: {label_col}")

        mask = (
            (raw["label_join_status"] == "coord_match")
            & raw[label_col].notna()
            & raw[SPLIT_COL].isin(VALID_SPLITS)
        )
        df = raw[mask].copy()
        df[label_col] = df[label_col].astype(int)

        pat = build_patient_df(df, label_col)

        n_pat = len(pat)
        n_pos = int((pat["patient_label"] == 1).sum())
        print(f"  {n_pat:,} patients, {n_pos:,} positive ({n_pos/n_pat*100:.1f}%)")
        print(f"  n_cores: median={pat['n_cores'].median():.0f}  "
              f"mean={pat['n_cores'].mean():.1f}  "
              f"range={pat['n_cores'].min():.0f}–{pat['n_cores'].max():.0f}")

        all_md_lines += [f"# Endpoint: `{label_col}` — {desc}", ""]

        sec, rows = section_distribution(pat, label_col)
        all_md_lines += sec
        all_csv_rows += rows

        sec, rows = section_positive_vs_negative(pat, label_col)
        all_md_lines += sec
        all_csv_rows += rows

        sec, rows = section_correlations(pat, label_col)
        all_md_lines += sec
        all_csv_rows += rows

        sec, rows = section_cross_split(pat, label_col)
        all_md_lines += sec
        all_csv_rows += rows

        all_md_lines += section_leakage_note(pat, label_col)

        print()

    csv_path = REPORTS_DIR / "core_counts_by_patient_audit.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    md_path = REPORTS_DIR / "core_counts_by_patient_audit.md"
    md_path.write_text("\n".join(all_md_lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
