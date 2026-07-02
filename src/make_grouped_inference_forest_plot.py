"""
Publication-style forest plot for grouped core-level inference (cluster-robust LR).

Inputs
------
reports/grouped_core_level_inference_cluster_robust.csv

Outputs
-------
figures/figure4_grouped_inference_forest_plot.png/.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CR_CSV = REPORTS_DIR / "grouped_core_level_inference_cluster_robust.csv"

# ── Feature ordering and display names ───────────────────────────────────────

FEATURE_ORDER = [
    "distance_midpoint_to_target_surface_mm",
    "approximate_fraction_of_centerline_inside_target",
    "distance_midpoint_to_target_centroid_mm",
    "trajectory_intersects_target",
    "log_psa_ng_ml",
    "psa_ng_ml",
    "prostate_volume_cc",
    "psa_density",
]

DISPLAY_NAMES = {
    "distance_midpoint_to_target_surface_mm":           "Distance to target surface",
    "approximate_fraction_of_centerline_inside_target": "Fraction inside target",
    "distance_midpoint_to_target_centroid_mm":          "Distance to target centroid",
    "trajectory_intersects_target":                     "Trajectory intersects target",
    "log_psa_ng_ml":                                    "log(PSA)",
    "psa_ng_ml":                                        "PSA",
    "prostate_volume_cc":                               "Prostate volume",
    "psa_density":                                      "PSA density",
}

CLINICAL_START = 4   # index in FEATURE_ORDER where clinical features begin

ENDPOINT_ORDER  = ["binary_label_int", "binary_label_gg3plus_int"]
ENDPOINT_LABELS = ["GG2+ / csPCa", "GG3+ / high-grade"]

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
})

C_SIG   = "#2c5f8a"   # blue  — significant (p < 0.05)
C_NSIG  = "#999999"   # gray  — not significant
P_THRESH = 0.05

N = len(FEATURE_ORDER)
# Feature at index 0 (distance to surface) sits at y = N-1 (top).
# Feature at index N-1 (PSA density) sits at y = 0 (bottom).
Y_POS = {feat: N - 1 - i for i, feat in enumerate(FEATURE_ORDER)}
Y_MIN = -0.6
Y_MAX = N - 0.4   # = 7.6


def _data_y_to_axes(y_data: float) -> float:
    """Convert a data-space y position to axes-fraction (for transAxes labels)."""
    return (y_data - Y_MIN) / (Y_MAX - Y_MIN)


def _draw_panel(ax: plt.Axes, sub: pd.DataFrame, ep_data: pd.DataFrame) -> None:
    """Draw one panel of the forest plot onto *ax*."""

    # Light shading behind the geometry features block
    geo_y_lo = Y_POS[FEATURE_ORDER[CLINICAL_START - 1]] - 0.45
    geo_y_hi = Y_POS[FEATURE_ORDER[0]] + 0.45
    ax.axhspan(geo_y_lo, geo_y_hi, color="#eef2f8", alpha=1.0, zorder=0)

    # Separator between geometry and clinical groups
    sep_y = (
        Y_POS[FEATURE_ORDER[CLINICAL_START - 1]]
        + Y_POS[FEATURE_ORDER[CLINICAL_START]]
    ) / 2.0
    ax.axhline(sep_y, color="#aaaaaa", lw=0.9, ls="-", zorder=1)

    # Reference line at OR = 1
    ax.axvline(1.0, color="#555555", lw=0.9, ls="--", zorder=1)

    # Feature rows
    for feat in FEATURE_ORDER:
        if feat not in sub.index:
            continue
        row    = sub.loc[feat]
        y      = Y_POS[feat]
        or_val = float(row["or"])
        ci_lo  = float(row["ci_lo_or"])
        ci_hi  = float(row["ci_hi_or"])
        p_val  = float(row["p_value"])

        sig    = p_val < P_THRESH
        color  = C_SIG  if sig else C_NSIG
        marker = "D"    if sig else "o"
        msz    = 6.5    if sig else 5.5
        mfc    = color  if sig else "white"

        # Horizontal CI bar
        ax.plot([ci_lo, ci_hi], [y, y],
                color=color, lw=1.8, solid_capstyle="butt",
                alpha=0.85, zorder=2)

        # Point estimate
        ax.plot(or_val, y,
                marker=marker,
                color=color,
                markersize=msz,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=1.5,
                linestyle="none",
                zorder=3)

    # Log x-axis
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (95% CI)", fontsize=9.5)
    ax.xaxis.set_major_locator(
        mticker.LogLocator(base=10.0, subs=[1.0, 2.0, 3.0, 5.0], numticks=8)
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # x limits: derived per endpoint from the full endpoint data
    x_lo = ep_data["ci_lo_or"].min() * 0.72
    x_hi = ep_data["ci_hi_or"].max() * 1.40
    ax.set_xlim(x_lo, x_hi)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.tick_params(axis="x", labelsize=8.5)


def make_forest_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for ax, ep, ep_label in zip(axes, ENDPOINT_ORDER, ENDPOINT_LABELS):
        sub     = df[df["endpoint"] == ep].set_index("feature")
        ep_data = df[df["endpoint"] == ep]
        _draw_panel(ax, sub, ep_data)
        ax.set_title(ep_label, fontsize=11, fontweight="bold", pad=9)

    # y-axis tick labels on the left panel only (sharey suppresses duplicates)
    axes[0].set_yticks(list(Y_POS.values()))
    axes[0].set_yticklabels(
        [DISPLAY_NAMES[f] for f in FEATURE_ORDER],
        fontsize=9.5,
    )
    axes[0].tick_params(axis="y", length=0, pad=6)
    axes[1].tick_params(axis="y", length=0)

    # Section labels ("Geometry" / "Clinical") to the right of each panel
    geo_mid_y  = (Y_POS[FEATURE_ORDER[0]] + Y_POS[FEATURE_ORDER[CLINICAL_START - 1]]) / 2
    clin_mid_y = (Y_POS[FEATURE_ORDER[CLINICAL_START]] + Y_POS[FEATURE_ORDER[-1]]) / 2

    for ax in axes:
        ax.text(
            1.03, _data_y_to_axes(geo_mid_y),
            "Geometry",
            transform=ax.transAxes,
            fontsize=8, color="#3a5f7a",
            va="center", ha="left",
            style="italic", clip_on=False,
        )
        ax.text(
            1.03, _data_y_to_axes(clin_mid_y),
            "Clinical",
            transform=ax.transAxes,
            fontsize=8, color="#666666",
            va="center", ha="left",
            style="italic", clip_on=False,
        )

    # Legend
    h_sig = mlines.Line2D(
        [], [], marker="D", color=C_SIG, markersize=6.5,
        linestyle="-", lw=1.8, label="p < 0.05",
    )
    h_nsig = mlines.Line2D(
        [], [], marker="o", color=C_NSIG, markersize=5.5,
        linestyle="-", lw=1.8,
        markerfacecolor="white", markeredgecolor=C_NSIG,
        label="p ≥ 0.05",
    )
    axes[1].legend(
        handles=[h_sig, h_nsig],
        fontsize=8.5, loc="lower right",
        framealpha=0.9, edgecolor="#cccccc",
    )

    # Figure title and caption
    fig.suptitle(
        "Cluster-robust associations for compact model features",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "Core-level logistic regression with standard errors clustered by patient.  "
        "Continuous predictors are standardised; trajectory_intersects_target is binary.",
        ha="center", fontsize=7.5, color="#777777",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "figure4_grouped_inference_forest_plot")


def _save(fig: plt.Figure, stem: str) -> None:
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png.relative_to(REPO_ROOT)}")
    print(f"Saved -> {pdf.relative_to(REPO_ROOT)}")


def main() -> None:
    df = pd.read_csv(CR_CSV)
    make_forest_plot(df)
    print("Done.")


if __name__ == "__main__":
    main()
