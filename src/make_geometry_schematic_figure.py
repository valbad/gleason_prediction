"""
Schematic figure of target-relative biopsy geometry features.

This is a conceptual illustration for paper Figure 1.
It does NOT represent patient data or real imaging.
All coordinates are chosen for visual clarity.

Output
------
figures/figure1_geometry_schematic.png
figures/figure1_geometry_schematic.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# ── Output paths ──────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

OUT_PNG = FIGURES_DIR / "figure1_geometry_schematic.png"
OUT_PDF = FIGURES_DIR / "figure1_geometry_schematic.pdf"

# ── Schematic coordinates (arbitrary units) ───────────────────────────────────

PROSTATE_CENTER = (0.0, 0.0)
PROSTATE_W = 5.6
PROSTATE_H = 4.2

TARGET_CENTER = (-0.9, 0.7)
TARGET_W = 1.8
TARGET_H = 1.3
TARGET_ANGLE = 20.0

NEEDLE_START = np.array([-3.2, -1.4])
NEEDLE_END   = np.array([1.6,  2.1])
NEEDLE_MID   = (NEEDLE_START + NEEDLE_END) / 2

TARGET_C = np.array(TARGET_CENTER)

# Intersection of needle line with target ellipse (chosen visually)
INTERSECT_ENTER = np.array([-1.76,  0.28])
INTERSECT_EXIT  = np.array([-0.08,  1.28])

# Closest point on target surface to biopsy midpoint (chosen visually)
SURFACE_CLOSEST = np.array([-1.20, -0.08])


def _draw_double_arrow(ax, p1, p2, color, lw=1.2, ls="-"):
    """Draw a bidirectional arrow between two points using two annotate calls."""
    ax.annotate(
        "",
        xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        linestyle=ls, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=p1, xytext=p2,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        linestyle=ls, shrinkA=0, shrinkB=0),
        zorder=4,
    )


def make_figure() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-3.6, 3.2)
    ax.set_ylim(-2.8, 2.8)
    ax.axis("off")

    # ── Prostate ──────────────────────────────────────────────────────────────
    prostate = Ellipse(
        xy=PROSTATE_CENTER, width=PROSTATE_W, height=PROSTATE_H, angle=0,
        linewidth=1.8, edgecolor="#4a4a4a", facecolor="#f0ece4", zorder=1,
    )
    ax.add_patch(prostate)
    ax.text(0.0, -2.35, "Prostate", ha="center", va="center",
            fontsize=10, color="#4a4a4a", style="italic")

    # ── Target lesion ─────────────────────────────────────────────────────────
    target = Ellipse(
        xy=TARGET_CENTER, width=TARGET_W, height=TARGET_H, angle=TARGET_ANGLE,
        linewidth=1.5, edgecolor="#b5451b", facecolor="#f7c5b0", zorder=2,
    )
    ax.add_patch(target)
    ax.text(TARGET_CENTER[0] - 0.05, TARGET_CENTER[1] + 1.05,
            "MRI target", ha="center", va="bottom",
            fontsize=9, color="#b5451b")

    # ── Full needle (dashed) ──────────────────────────────────────────────────
    ax.plot(
        [NEEDLE_START[0], NEEDLE_END[0]],
        [NEEDLE_START[1], NEEDLE_END[1]],
        color="#2c5f8a", linewidth=1.2, linestyle="--", zorder=3,
    )

    # ── Segment inside target (thick solid) ───────────────────────────────────
    ax.plot(
        [INTERSECT_ENTER[0], INTERSECT_EXIT[0]],
        [INTERSECT_ENTER[1], INTERSECT_EXIT[1]],
        color="#2c5f8a", linewidth=5, linestyle="-",
        solid_capstyle="round", zorder=4,
    )

    # Label: "trajectory intersects target"
    ax.annotate(
        "trajectory\nintersects target",
        xy=INTERSECT_ENTER,
        xytext=(-2.85, 1.05),
        fontsize=8, color="#2c5f8a", ha="center",
        arrowprops=dict(arrowstyle="-", color="#2c5f8a", lw=0.8,
                        shrinkA=0, shrinkB=0),
    )

    # Label: "fraction of trajectory inside target"
    frac_mid = (INTERSECT_ENTER + INTERSECT_EXIT) / 2
    ax.annotate(
        "fraction of trajectory\ninside target",
        xy=frac_mid,
        xytext=(frac_mid[0] + 1.4, frac_mid[1] - 0.9),
        fontsize=8, color="#2c5f8a", ha="left",
        arrowprops=dict(arrowstyle="->", color="#2c5f8a", lw=0.9,
                        shrinkA=0, shrinkB=3),
    )

    # ── Biopsy midpoint ───────────────────────────────────────────────────────
    ax.plot(*NEEDLE_MID, "o", color="#2c5f8a", markersize=7, zorder=5)
    ax.text(NEEDLE_MID[0] + 0.12, NEEDLE_MID[1] - 0.22,
            "biopsy\nmidpoint", fontsize=8.5, color="#2c5f8a",
            ha="left", va="top")

    # ── Target centroid ───────────────────────────────────────────────────────
    ax.plot(*TARGET_C, "+", color="#b5451b", markersize=11,
            markeredgewidth=2.2, zorder=5)
    ax.text(TARGET_C[0] + 0.14, TARGET_C[1] + 0.14,
            "target\ncentroid", fontsize=8.5, color="#b5451b",
            ha="left", va="bottom")

    # ── Distance to target centroid (dashed bidirectional) ────────────────────
    _draw_double_arrow(ax, NEEDLE_MID, TARGET_C, color="#666666", lw=1.1, ls="--")
    centroid_label_mid = (NEEDLE_MID + TARGET_C) / 2
    ax.text(centroid_label_mid[0] + 0.44, centroid_label_mid[1] + 0.05,
            "distance to\ntarget centroid", fontsize=8, color="#555555",
            ha="left", va="center")

    # ── Distance to target surface (solid bidirectional, green) ───────────────
    _draw_double_arrow(ax, NEEDLE_MID, SURFACE_CLOSEST, color="#1a7a3c", lw=1.5)
    ax.plot(*SURFACE_CLOSEST, "o", color="#1a7a3c", markersize=5, zorder=5)
    surface_label_mid = (NEEDLE_MID + SURFACE_CLOSEST) / 2
    ax.text(surface_label_mid[0] - 0.08, surface_label_mid[1] - 0.32,
            "distance to\ntarget surface", fontsize=8, color="#1a7a3c",
            ha="center", va="top")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#f0ece4", edgecolor="#4a4a4a",
                       linewidth=1.2, label="Prostate"),
        mpatches.Patch(facecolor="#f7c5b0", edgecolor="#b5451b",
                       linewidth=1.2, label="MRI-defined target"),
        Line2D([0], [0], color="#2c5f8a", lw=1.2, linestyle="--",
               label="Needle trajectory (outside target)"),
        Line2D([0], [0], color="#2c5f8a", lw=4.5, linestyle="-",
               label="Needle trajectory (inside target)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.92, edgecolor="#cccccc")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title("Target-relative biopsy geometry",
                 fontsize=13, fontweight="bold", pad=10)

    # ── Caption note ──────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             "Schematic only — not patient data. 2D axial view; coordinates are illustrative.",
             ha="center", fontsize=7.5, color="#888888")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"Saved -> {OUT_PDF.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    make_figure()
