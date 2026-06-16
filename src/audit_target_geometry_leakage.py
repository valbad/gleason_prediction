"""
Audit whether target geometry features may leak label information.

The target geometry features (target_mesh_available, distances, trajectory
intersection, fraction inside target) are derived from MRI-defined lesion
meshes. This audit checks:

  1. Feature-set hygiene   — no pathology column is used as a feature.
  2. Split integrity       — patients do not overlap across train/val/test.
  3. Descriptive stats     — distributions of each feature by label (0 vs 1).
  4. Positive-rate tables  — how often label=1 within each geometric stratum.
  5. target_mesh_available — whether having a target at all is associated with
                             the label (potential confound or selection bias).

The script does NOT prove or disprove clinical causality; it surfaces whether
the feature distributions look plausible (features encoding biopsy geometry
relative to a suspicious lesion defined before pathology) or suspicious (e.g.
target_mesh_available being a near-perfect proxy for the label).

Usage
-----
    python src/audit_target_geometry_leakage.py

Outputs
-------
    reports/target_geometry_leakage_audit.md
    reports/target_geometry_leakage_audit.csv   (tidy rows for every statistic)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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

# Exact feature list from run_shareable_tabular_experiments.py — authoritative.
TARGET_GEOMETRY_FEATURES = [
    "target_mesh_available",
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
]

# All columns that encode pathology or the label — none must appear in features.
PATHOLOGY_COLS = [
    "pathology_hint_from_filename",
    "pathology_label",
    "pathology_label_int",
    "binary_label",
    "binary_label_int",
    "binary_label_gg3plus_int",
    "core_label",
    "primary_gleason",
    "secondary_gleason",
    "cancer_length_mm",
    "pct_cancer_in_core",
]

# Continuous / ordered features for descriptive stats (excludes the flag).
STATS_FEATURES = [
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
]

# Returns (condition_name, valid_mask, met_mask) triples:
#   valid_mask — rows where the underlying feature is not NaN
#   met_mask   — subset of valid_mask where the condition is True
# This 3-tuple design is necessary because pandas float comparisons return
# False (not NaN) for NaN inputs, making it impossible to separate
# "condition not met" from "feature missing" with a single boolean mask.
def _conditions(df: pd.DataFrame):
    tit  = pd.to_numeric(df["trajectory_intersects_target"],                    errors="coerce")
    frac = pd.to_numeric(df["approximate_fraction_of_centerline_inside_target"], errors="coerce")
    surf = pd.to_numeric(df["distance_midpoint_to_target_surface_mm"],           errors="coerce")

    v_tit  = tit.notna()
    v_frac = frac.notna()
    v_surf = surf.notna()

    return [
        ("trajectory_intersects_target == True",               v_tit,  v_tit  & (tit == 1)),
        ("trajectory_intersects_target == False",              v_tit,  v_tit  & (tit == 0)),
        ("fraction_inside_target > 0",                         v_frac, v_frac & (frac > 0)),
        ("fraction_inside_target > 0.25",                      v_frac, v_frac & (frac > 0.25)),
        ("fraction_inside_target > 0.5",                       v_frac, v_frac & (frac > 0.5)),
        ("distance_to_target_surface_mm <= 0 (inside target)", v_surf, v_surf & (surf <= 0)),
        ("distance_to_target_surface_mm <= 2 mm",              v_surf, v_surf & (surf <= 2)),
        ("distance_to_target_surface_mm <= 5 mm",              v_surf, v_surf & (surf <= 5)),
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_numeric_01(series: pd.Series) -> pd.Series:
    """Coerce True/False strings and ints to numeric 0/1."""
    truthy = {True, "True", "TRUE", "true", 1, "1", 1.0}
    falsy  = {False, "False", "FALSE", "false", 0, "0", 0.0}
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[series.isin(truthy)] = 1.0
    out[series.isin(falsy)]  = 0.0
    return out


def load_filtered(label_col: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df_raw["label_join_status"] == "coord_match")
        & df_raw[label_col].notna()
        & df_raw[SPLIT_COL].isin(VALID_SPLITS)
    )
    df = df_raw[mask].copy()
    df[label_col] = df[label_col].astype(int)
    # Normalise boolean-encoded columns to numeric
    for col in ["trajectory_intersects_target", "target_mesh_available"]:
        if col in df.columns:
            df[col] = _to_numeric_01(df[col])
    return df.reset_index(drop=True)


def _fmt(v, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(v)


def _pct(n: int, total: int) -> str:
    return f"{n/total*100:.1f}%" if total > 0 else "n/a"


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return dict(n=0, mean=np.nan, std=np.nan,
                    p25=np.nan, median=np.nan, p75=np.nan,
                    min=np.nan, max=np.nan)
    return dict(
        n=int(len(s)),
        mean=float(s.mean()),
        std=float(s.std()),
        p25=float(s.quantile(0.25)),
        median=float(s.median()),
        p75=float(s.quantile(0.75)),
        min=float(s.min()),
        max=float(s.max()),
    )


# ── Report builders ───────────────────────────────────────────────────────────

def section_feature_hygiene(df_columns: list[str]) -> tuple[list[str], list[dict]]:
    """
    1. Confirm the exact features used.
    2. Confirm no pathology column overlaps.
    """
    lines = [
        "## 1. Feature-set Hygiene",
        "",
        "### 1a. Target geometry features (from `run_shareable_tabular_experiments.py`)",
        "",
        "| # | Column | Present in dataset |",
        "|---|---|---|",
    ]
    csv_rows: list[dict] = []
    all_present = True
    for i, col in enumerate(TARGET_GEOMETRY_FEATURES, 1):
        present = col in df_columns
        if not present:
            all_present = False
        lines.append(f"| {i} | `{col}` | {'✓' if present else '✗ MISSING'} |")
        csv_rows.append(dict(section="feature_hygiene", check=col, result="present" if present else "MISSING"))
    lines.append("")

    # Overlap with pathology columns
    overlap = [c for c in TARGET_GEOMETRY_FEATURES if c in PATHOLOGY_COLS]
    overlap_result = "NONE — OK" if not overlap else f"OVERLAP: {overlap}"
    lines += [
        "### 1b. Overlap with pathology / label columns",
        "",
        f"Pathology columns checked: {len(PATHOLOGY_COLS)}  ",
        f"Overlap with target geometry features: **{overlap_result}**",
        "",
    ]
    csv_rows.append(dict(section="feature_hygiene", check="pathology_overlap",
                         result=overlap_result))

    status = "PASS" if all_present and not overlap else "FAIL"
    lines += [f"> **Section result: {status}**", ""]
    csv_rows.append(dict(section="feature_hygiene", check="overall", result=status))
    return lines, csv_rows


def section_split_integrity(df_raw: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Verify no patient_number appears in more than one split."""
    lines = [
        "## 2. Patient-level Split Integrity",
        "",
        "Each `patient_number` must appear in exactly one of train / val / test.",
        "",
    ]
    csv_rows: list[dict] = []

    # Use binary_label_int to define the base filter (most rows; gg3plus same patients)
    base = df_raw[
        (df_raw["label_join_status"] == "coord_match")
        & df_raw["binary_label_int"].notna()
        & df_raw[SPLIT_COL].isin(VALID_SPLITS)
    ]

    split_patients: dict[str, set] = {
        s: set(base[base[SPLIT_COL] == s][PATIENT_COL].unique())
        for s in VALID_SPLITS
    }

    lines += [
        "| Split | Unique patients |",
        "|---|---|",
    ]
    for s, pts in split_patients.items():
        lines.append(f"| {s} | {len(pts):,} |")
    lines.append("")

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val",   "test"),
    ]
    all_ok = True
    lines += ["| Pair | Shared patients | Result |", "|---|---|---|"]
    for a, b in pairs:
        shared = split_patients[a] & split_patients[b]
        ok = len(shared) == 0
        if not ok:
            all_ok = False
        result = "PASS" if ok else f"FAIL ({len(shared)} shared)"
        lines.append(f"| {a} / {b} | {len(shared)} | {result} |")
        csv_rows.append(dict(section="split_integrity", check=f"{a}_vs_{b}",
                             result=result))
    lines.append("")

    status = "PASS" if all_ok else "FAIL"
    lines += [f"> **Section result: {status}**", ""]
    csv_rows.append(dict(section="split_integrity", check="overall", result=status))
    return lines, csv_rows


def section_descriptive_stats(df: pd.DataFrame, label_col: str) -> tuple[list[str], list[dict]]:
    """Per-label descriptive statistics for continuous target-geometry features."""
    desc = LABEL_DESC[label_col]
    lines = [
        f"## 3. Descriptive Statistics — `{label_col}` ({desc})",
        "",
        "Statistics computed on coord_match rows with valid label and split.",
        "",
    ]
    csv_rows: list[dict] = []

    stat_header = "| Feature | Label | N | Mean | Std | P25 | Median | P75 | Min | Max |"
    stat_sep    = "|---|---|---|---|---|---|---|---|---|---|"
    lines += [stat_header, stat_sep]

    for feat in STATS_FEATURES:
        if feat not in df.columns:
            lines.append(f"| `{feat}` | — | _not in dataset_ | | | | | | | |")
            continue
        for lbl in [0, 1]:
            sub = df[df[label_col] == lbl][feat]
            st  = _stats(sub)
            lines.append(
                f"| `{feat}` | {lbl} | {st['n']:,} "
                f"| {_fmt(st['mean'])} | {_fmt(st['std'])} "
                f"| {_fmt(st['p25'])} | {_fmt(st['median'])} | {_fmt(st['p75'])} "
                f"| {_fmt(st['min'])} | {_fmt(st['max'])} |"
            )
            csv_rows.append(dict(
                section="descriptive_stats", label_col=label_col, feature=feat,
                label_value=lbl, **st
            ))
    lines.append("")
    return lines, csv_rows


def section_positive_rates(df: pd.DataFrame, label_col: str) -> tuple[list[str], list[dict]]:
    """Positive rate of the label within each geometric stratum."""
    desc = LABEL_DESC[label_col]
    lines = [
        f"## 4. Positive Rates by Geometric Condition — `{label_col}` ({desc})",
        "",
        f"For each condition, counts and positive rate of `{label_col}==1` are shown "
        "for rows where the feature is not NaN.",
        "",
        "| Condition | N (condition met) | N positive | Prevalence | "
        "N (condition not met) | N positive | Prevalence |",
        "|---|---|---|---|---|---|---|",
    ]
    csv_rows: list[dict] = []

    for cond_name, valid_mask, met_mask in _conditions(df):
        not_met_mask = valid_mask & ~met_mask

        sub_pos = df[met_mask]
        sub_neg = df[not_met_mask]

        n_pos_total = int(sub_pos[label_col].sum())
        n_neg_total = int(sub_neg[label_col].sum())

        lines.append(
            f"| {cond_name} "
            f"| {len(sub_pos):,} | {n_pos_total:,} | {_pct(n_pos_total, len(sub_pos))} "
            f"| {len(sub_neg):,} | {n_neg_total:,} | {_pct(n_neg_total, len(sub_neg))} |"
        )
        csv_rows.append(dict(
            section="positive_rates",
            label_col=label_col,
            condition=cond_name,
            n_condition_met=len(sub_pos),
            n_positive_condition_met=n_pos_total,
            prevalence_condition_met=n_pos_total / len(sub_pos) if len(sub_pos) > 0 else np.nan,
            n_condition_not_met=len(sub_neg),
            n_positive_condition_not_met=n_neg_total,
            prevalence_condition_not_met=n_neg_total / len(sub_neg) if len(sub_neg) > 0 else np.nan,
        ))
    lines.append("")
    return lines, csv_rows


def section_target_mesh(df: pd.DataFrame, label_col: str) -> tuple[list[str], list[dict]]:
    """
    Check whether target_mesh_available is associated with the label.

    A near-perfect association would be suspicious: it would mean that
    having an MRI target defined is almost synonymous with being cancer-positive,
    which would imply the target was defined *because* the biopsy was positive
    rather than being an independent MRI finding.
    """
    desc = LABEL_DESC[label_col]
    lines = [
        f"## 5. `target_mesh_available` Association — `{label_col}` ({desc})",
        "",
        "If MRI targets are defined independently of pathology (e.g. from PI-RADS ≥ 3",
        "lesions), `target_mesh_available` should have moderate but not near-perfect",
        "association with the label. A very high positive rate when mesh=True AND a",
        "near-zero rate when mesh=False would be a strong leakage signal.",
        "",
    ]
    csv_rows: list[dict] = []

    if "target_mesh_available" not in df.columns:
        lines += ["_Column not present in dataset._", ""]
        return lines, csv_rows

    col = "target_mesh_available"

    for mesh_val, label_name in [(1.0, "True (mesh present)"), (0.0, "False (no mesh)")]:
        sub = df[df[col] == mesh_val]
        n   = len(sub)
        n_pos = int(sub[label_col].sum()) if n > 0 else 0
        prev = n_pos / n if n > 0 else np.nan

        lines.append(f"**`target_mesh_available = {label_name}`**")
        lines.append("")
        lines += [
            f"- Rows: {n:,}",
            f"- Positives (`{label_col}==1`): {n_pos:,}",
            f"- Prevalence: {_pct(n_pos, n)}",
            "",
        ]
        csv_rows.append(dict(
            section="target_mesh_association",
            label_col=label_col,
            target_mesh_available=int(mesh_val),
            n_rows=n,
            n_positives=n_pos,
            prevalence=prev,
        ))

    # Also: among positives, what fraction have a mesh?
    n_lab_pos = int((df[label_col] == 1).sum())
    n_pos_with_mesh = int(((df[label_col] == 1) & (df[col] == 1.0)).sum())
    n_lab_neg = int((df[label_col] == 0).sum())
    n_neg_with_mesh = int(((df[label_col] == 0) & (df[col] == 1.0)).sum())

    lines += [
        "**Reverse view: mesh rate by label**",
        "",
        "| Label | N | Has target mesh | Mesh rate |",
        "|---|---|---|---|",
        f"| 0 (negative) | {n_lab_neg:,} | {n_neg_with_mesh:,} | {_pct(n_neg_with_mesh, n_lab_neg)} |",
        f"| 1 (positive) | {n_lab_pos:,} | {n_pos_with_mesh:,} | {_pct(n_pos_with_mesh, n_lab_pos)} |",
        "",
    ]
    csv_rows.append(dict(
        section="target_mesh_reverse",
        label_col=label_col,
        label_value=0,
        n_rows=n_lab_neg,
        n_with_mesh=n_neg_with_mesh,
        mesh_rate=n_neg_with_mesh / n_lab_neg if n_lab_neg > 0 else np.nan,
    ))
    csv_rows.append(dict(
        section="target_mesh_reverse",
        label_col=label_col,
        label_value=1,
        n_rows=n_lab_pos,
        n_with_mesh=n_pos_with_mesh,
        mesh_rate=n_pos_with_mesh / n_lab_pos if n_lab_pos > 0 else np.nan,
    ))

    # Interpretation hint
    mesh_diff = (n_pos_with_mesh / n_lab_pos if n_lab_pos > 0 else 0) - \
                (n_neg_with_mesh / n_lab_neg if n_lab_neg > 0 else 0)
    if abs(mesh_diff) > 0.3:
        interp = (
            "**Large difference (> 30 pp):** mesh availability is strongly associated "
            "with the label. This could reflect targeted biopsies or post-hoc target "
            "definition. The `target_geometry_only` experiment should be interpreted "
            "with caution."
        )
    elif abs(mesh_diff) > 0.1:
        interp = (
            "**Moderate difference (10–30 pp):** some association between mesh "
            "availability and the label. Target geometry features carry real signal "
            "but may partly reflect lesion pre-selection."
        )
    else:
        interp = (
            "**Small difference (< 10 pp):** mesh availability is similarly distributed "
            "across labels, consistent with targets being defined independently of "
            "pathology outcome."
        )
    lines += [f"> {interp}", ""]
    return lines, csv_rows


def write_csv(all_rows: list[dict], path: Path) -> None:
    pd.DataFrame(all_rows).to_csv(path, index=False)
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")


def write_md(sections: list[list[str]], path: Path) -> None:
    lines: list[str] = [
        "# Target Geometry Leakage Audit",
        "",
        "Assesses whether target-derived geometric features encode legitimate",
        "MRI-based predictors or may leak pathology label information.",
        "",
        "**Dataset:** `data/share/needle_features_v1.csv`  ",
        "**Filter:** `label_join_status == 'coord_match'`, valid label, valid split  ",
        "**Labels audited:** `binary_label_int` (GG2+) and `binary_label_gg3plus_int` (GG3+)",
        "",
    ]
    for section_lines in sections:
        lines.extend(section_lines)
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Target geometry leakage audit ===\n")

    print(f"Loading {DATA_PATH.relative_to(REPO_ROOT)} …")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  {len(df_raw):,} rows, {len(df_raw.columns)} columns\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_sections: list[list[str]] = []
    all_csv_rows: list[dict]      = []

    # ── 1. Feature hygiene (dataset-level, no label needed) ───────────────
    sec, rows = section_feature_hygiene(df_raw.columns.tolist())
    all_sections.append(sec)
    all_csv_rows.extend(rows)

    # ── 2. Split integrity (dataset-level) ────────────────────────────────
    sec, rows = section_split_integrity(df_raw)
    all_sections.append(sec)
    all_csv_rows.extend(rows)

    # ── Per-label sections (3, 4, 5) ─────────────────────────────────────
    for label_col in LABEL_COLS:
        if label_col not in df_raw.columns:
            print(f"  [skip] '{label_col}' not in dataset")
            continue

        print(f"Auditing label: {label_col} …")
        df = load_filtered(label_col, df_raw)
        n_pos = int((df[label_col] == 1).sum())
        n_tot = len(df)
        print(f"  {n_tot:,} rows, {n_pos:,} positives ({n_pos/n_tot*100:.1f}%)")

        # 3. Descriptive stats
        sec, rows = section_descriptive_stats(df, label_col)
        all_sections.append(sec)
        all_csv_rows.extend(rows)

        # 4. Positive-rate tables
        sec, rows = section_positive_rates(df, label_col)
        all_sections.append(sec)
        all_csv_rows.extend(rows)

        # 5. target_mesh_available association
        sec, rows = section_target_mesh(df, label_col)
        all_sections.append(sec)
        all_csv_rows.extend(rows)

    print()

    # ── Interpretation guide ──────────────────────────────────────────────
    interp_section = [
        "## 6. Interpretation Guide",
        "",
        "| Signal | Plausible predictor | Leakage concern |",
        "|---|---|---|",
        "| `target_mesh_available` similarly prevalent in positive and negative cases | ✓ | |",
        "| `target_mesh_available` strongly associated with label | | ✓ |",
        "| `distance_to_target_surface_mm` lower for positives (shorter distance) | ✓ | |",
        "| `distance_to_target_surface_mm ≤ 0` covers nearly all positives | | ✓ |",
        "| `trajectory_intersects_target` higher rate in positives | ✓ (targeted biopsy) | |",
        "| `trajectory_intersects_target` = near-perfect label proxy | | ✓ |",
        "| `fraction_inside_target` graded association with label | ✓ | |",
        "| `fraction_inside_target > 0` = near-perfect label proxy | | ✓ |",
        "",
        "**Context:** In the TCIA Prostate-MRI-US-Biopsy dataset both systematic",
        "(non-targeted) and targeted (MRI-directed) biopsies are present. MRI targets",
        "are PI-RADS lesions defined *before* the biopsy, so proximity to the target",
        "is a legitimate predictor. However, if the dataset only provides a target mesh",
        "for patients who later tested positive, the geometry features would leak the label.",
        "The statistics above should be compared against that prior knowledge.",
        "",
    ]
    all_sections.append(interp_section)

    # ── Save outputs ──────────────────────────────────────────────────────
    write_csv(all_csv_rows, REPORTS_DIR / "target_geometry_leakage_audit.csv")
    write_md(all_sections,  REPORTS_DIR / "target_geometry_leakage_audit.md")


if __name__ == "__main__":
    main()
