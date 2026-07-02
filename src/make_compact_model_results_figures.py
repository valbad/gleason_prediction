"""
Publication-style result figures for the compact target_geometry_plus_clinical model.

Figure 2: Held-out ROC-AUC and PR-AUC with 95% bootstrap CIs.
Figure 3: Risk stratification enrichment (prevalence in top 5/10/20% of scored cores).

All data are read from existing CSVs — no models are re-run.

Inputs
------
reports/minimal_vs_full_model_comparison.csv
reports/minimal_vs_full_risk_stratification.csv

Outputs
-------
figures/figure2_compact_model_performance.png/.pdf
figures/figure3_risk_stratification_enrichment.png/.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PERF_CSV  = REPORTS_DIR / "minimal_vs_full_model_comparison.csv"
STRAT_CSV = REPORTS_DIR / "minimal_vs_full_risk_stratification.csv"

# ── Filter constants ──────────────────────────────────────────────────────────

COMPACT_FS    = "target_geometry_plus_clinical"
CENTRAL_MODEL = "logistic_regression"

ENDPOINT_ORDER = ["binary_label_int", "binary_label_gg3plus_int"]
ENDPOINT_LABELS = ["GG2+ / csPCa", "GG3+ / high-grade"]

# ── Shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":     "sans-serif",
    "axes.facecolor":  "white",
    "figure.facecolor":"white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_ROC = "#2c5f8a"   # blue  — ROC-AUC
C_PR  = "#1a7a3c"   # green — PR-AUC

# Enrichment palette: [baseline_gray, top-20% light, top-10% medium, top-5% dark]
STRAT_PAL = {
    "binary_label_int":         ["#c0c0c0", "#7fb3d9", "#2c5f8a", "#1a3f5e"],
    "binary_label_gg3plus_int": ["#c0c0c0", "#f0a07a", "#b5451b", "#7a2e0e"],
}


# ── Figure 2: compact model performance ──────────────────────────────────────

def make_figure2(df: pd.DataFrame) -> None:
    sub = (
        df[(df["feature_set"] == COMPACT_FS) & (df["model"] == CENTRAL_MODEL)]
        .set_index("label_col")
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bar_w  = 0.30
    offset = 0.18          # distance from group centre to each bar
    x = np.arange(len(ENDPOINT_ORDER))

    x_roc = x - offset
    x_pr  = x + offset

    for i, (lc, ep_label) in enumerate(zip(ENDPOINT_ORDER, ENDPOINT_LABELS)):
        row = sub.loc[lc]

        roc    = float(row["test_roc_auc"])
        roc_lo = roc - float(row["roc_auc_ci_low"])
        roc_hi = float(row["roc_auc_ci_high"]) - roc

        pr     = float(row["test_pr_auc"])
        pr_lo  = pr - float(row["pr_auc_ci_low"])
        pr_hi  = float(row["pr_auc_ci_high"]) - pr

        kw_label_roc = {"label": "ROC-AUC"} if i == 0 else {}
        kw_label_pr  = {"label": "PR-AUC"}  if i == 0 else {}

        # ROC bar
        ax.bar(x_roc[i], roc, width=bar_w, color=C_ROC, zorder=3, **kw_label_roc)
        ax.errorbar(x_roc[i], roc, yerr=[[roc_lo], [roc_hi]],
                    fmt="none", color="#222222", capsize=4, lw=1.2, zorder=4)
        ax.text(x_roc[i], roc + roc_hi + 0.018,
                f"{roc:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_ROC,
                fontweight="bold")

        # PR bar
        ax.bar(x_pr[i], pr, width=bar_w, color=C_PR, zorder=3, **kw_label_pr)
        ax.errorbar(x_pr[i], pr, yerr=[[pr_lo], [pr_hi]],
                    fmt="none", color="#222222", capsize=4, lw=1.2, zorder=4)
        ax.text(x_pr[i], pr + pr_hi + 0.018,
                f"{pr:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_PR,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ENDPOINT_LABELS, fontsize=10.5)
    ax.set_ylabel("AUC — held-out test set", fontsize=10)
    ax.set_ylim(0, 1.09)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.set_title("Compact model held-out performance",
                 fontsize=12, fontweight="bold", pad=10)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.92,
              edgecolor="#cccccc", handlelength=1.4)

    fig.text(
        0.5, 0.01,
        "Target geometry + clinical, logistic regression; patient-level bootstrap CIs.  "
        "PR-AUC baseline equals endpoint prevalence; ROC-AUC chance level is 0.5.",
        ha="center", fontsize=7.5, color="#777777",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    _save(fig, "figure2_compact_model_performance")


# ── Figure 3: risk stratification enrichment ──────────────────────────────────

def make_figure3(df: pd.DataFrame) -> None:
    sub = (
        df[(df["feature_set"] == COMPACT_FS) & (df["model"] == CENTRAL_MODEL)]
        .set_index("label_col")
    )

    # Display order: least selective → most selective
    group_labels = ["Baseline", "Top 20%", "Top 10%", "Top 5%"]
    value_cols   = [
        "prevalence",
        "top20pct_prevalence",
        "top10pct_prevalence",
        "top5pct_prevalence",
    ]
    x = np.arange(len(group_labels))
    bar_w = 0.58

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))

    for ax, lc, ep_label in zip(axes, ENDPOINT_ORDER, ENDPOINT_LABELS):
        row    = sub.loc[lc]
        values = [float(row[c]) for c in value_cols]
        colors = STRAT_PAL[lc]

        ax.bar(x, values, width=bar_w, color=colors,
               edgecolor="white", linewidth=0.6, zorder=3)

        # Value labels above each bar
        for xi, v in zip(x, values):
            ax.text(xi, v + 0.003, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=9.5)

        # Baseline dashed line
        baseline = values[0]
        ax.axhline(baseline, color="#888888", lw=0.9, ls="--", zorder=2)

        # Enrichment label above the top-5% bar (index 3, rightmost)
        enrichment_5pct = values[3] / baseline
        top5_height = values[3]
        # Place above the percentage label already drawn at v + 0.003
        ax.text(x[3], top5_height + 0.003,
                "",   # percentage label drawn in the loop above
                ha="center", va="bottom")
        ax.annotate(
            f"{enrichment_5pct:.1f}× baseline",
            xy=(x[3], top5_height),
            xytext=(x[3], top5_height * 1.28),
            ha="center", va="bottom",
            fontsize=8.5, color="#333333", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7,
                            shrinkA=0, shrinkB=2),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=9.5)
        ax.set_ylabel("Positive core prevalence", fontsize=9.5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
        )
        # Extra headroom for the enrichment label above the top-5% bar
        ax.set_ylim(0, max(values) * 1.55)
        ax.set_title(ep_label, fontsize=11, fontweight="bold")

    fig.suptitle("Risk enrichment among highest-scored biopsy cores",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.text(
        0.5, -0.03,
        "Held-out test split; highest-scored cores ranked by predicted probability.",
        ha="center", fontsize=7.5, color="#777777",
    )

    fig.tight_layout()

    _save(fig, "figure3_risk_stratification_enrichment")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: str) -> None:
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png.relative_to(REPO_ROOT)}")
    print(f"Saved -> {pdf.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    perf_df  = pd.read_csv(PERF_CSV)
    strat_df = pd.read_csv(STRAT_CSV)

    make_figure2(perf_df)
    make_figure3(strat_df)

    print("Done.")


if __name__ == "__main__":
    main()
