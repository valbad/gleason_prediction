"""
audit_extended_feature_opportunities.py

Goal
----
Audit which additional feature opportunities are available for a final
improvement sprint before manuscript freeze.

This is NOT a modelling script. No models are trained; no existing reports,
figures, or CSVs are modified.

Output
------
reports/extended_feature_opportunity_audit.md

Usage
-----
    python src/audit_extended_feature_opportunities.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"
OUT_PATH    = REPORTS_DIR / "extended_feature_opportunity_audit.md"

# ── Known feature groups (from run_shareable_tabular_experiments.py) ──────────

CLINICAL_FEATURES = [
    "psa_ng_ml",
    "log_psa_ng_ml",      # derived at load time
    "prostate_volume_cc",
    "psa_density",        # derived at load time
]

BIOPSY_GEOMETRY_FEATURES = [
    "core_length_mm",
    "n_centerline_voxels",
    "n_tube_voxels",
    "midpoint_inside_prostate",
    "distance_midpoint_to_prostate_surface_mm",
    "approximate_fraction_of_centerline_inside_prostate",
]

TARGET_GEOMETRY_FEATURES = [
    "target_mesh_available",
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
]

CENTERLINE_INTENSITY_FEATURES = [
    "centerline_intensity_mean",
    "centerline_intensity_std",
    "centerline_intensity_p25",
    "centerline_intensity_p75",
]

ENDPOINT_COLS = ["binary_label_int", "binary_label_gg3plus_int"]

META_COLS = [
    "patient_number", "split", "label_join_status",
    "extraction_status",
]

KNOWN_COLS = set(
    CLINICAL_FEATURES
    + BIOPSY_GEOMETRY_FEATURES
    + TARGET_GEOMETRY_FEATURES
    + CENTERLINE_INTENSITY_FEATURES
    + ENDPOINT_COLS
    + META_COLS
    + ["log_psa_ng_ml", "psa_density"]   # derived, may or may not be in CSV
)

# Keyword patterns for column-name scanning
INTENSITY_PATTERNS = re.compile(
    r"intensity|mean|median|std|percentile|p\d{2,3}|radiomics|texture|"
    r"entropy|t2|adc|dwi|patch|voxel|signal|histogram",
    re.IGNORECASE,
)
PRIOR_BX_PATTERNS = re.compile(
    r"prior|previous|history|prev_|past|repeat|baseline|"
    r"target_prev|prior_pos|prior_grade|prior_cancer",
    re.IGNORECASE,
)
TARGET_ID_PATTERNS = re.compile(
    r"target_id|lesion_id|target_number|lesion_number|target_name|"
    r"target_label|mri_target|roi_id|region_id",
    re.IGNORECASE,
)
SIZE_PATTERNS = re.compile(
    r"target_volume|lesion_volume|target_size|lesion_size|target_radius|"
    r"mesh_volume|mesh_area|target_longest_axis|roi_volume",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _miss(s: pd.Series) -> str:
    n = s.isna().sum()
    pct = 100 * n / len(s)
    return f"{n} ({pct:.1f}%)"


def _nuniq(s: pd.Series) -> int:
    return s.nunique(dropna=True)


def _computable(col: str, df: pd.DataFrame) -> tuple[bool, str]:
    """Return (computable, reason/note) for a derived feature."""
    exists = col in df.columns
    return exists, ("in CSV" if exists else "not in CSV")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load raw CSV (no filters) ─────────────────────────────────────────────
    raw = pd.read_csv(DATA_PATH)
    n_raw = len(raw)
    all_cols = raw.columns.tolist()

    # ── Apply standard row filter ─────────────────────────────────────────────
    df = raw[
        (raw["label_join_status"] == "coord_match")
        & raw["binary_label_int"].notna()
        & raw["split"].isin({"train", "val", "test"})
    ].copy()
    n_filtered = len(df)

    # Derive log_psa and psa_density if not already present
    if "log_psa_ng_ml" not in df.columns and "psa_ng_ml" in df.columns:
        df["log_psa_ng_ml"] = np.log1p(df["psa_ng_ml"])
    if "psa_density" not in df.columns and "psa_ng_ml" in df.columns and "prostate_volume_cc" in df.columns:
        valid = df["prostate_volume_cc"].notna() & (df["prostate_volume_cc"] > 0)
        df["psa_density"] = np.where(
            valid & df["psa_ng_ml"].notna(),
            df["psa_ng_ml"] / df["prostate_volume_cc"],
            np.nan,
        )

    n_patients = df["patient_number"].nunique()
    n_per_split = df.groupby("split")["patient_number"].nunique()

    # ── Categorise all columns ────────────────────────────────────────────────
    unknown_cols = [c for c in all_cols if c not in KNOWN_COLS]
    intensity_cols = [c for c in all_cols if INTENSITY_PATTERNS.search(c) and c not in KNOWN_COLS]
    prior_bx_cols  = [c for c in all_cols if PRIOR_BX_PATTERNS.search(c)]
    target_id_cols = [c for c in all_cols if TARGET_ID_PATTERNS.search(c)]
    size_cols      = [c for c in all_cols if SIZE_PATTERNS.search(c)]

    # ── Extraction-status subset ──────────────────────────────────────────────
    if "extraction_status" in df.columns:
        n_mode_b = (df["extraction_status"] == "ok").sum()
        mode_b_pct = 100 * n_mode_b / n_filtered
    else:
        n_mode_b = None
        mode_b_pct = None

    # ── Patient/target grouping ───────────────────────────────────────────────
    cores_per_patient = df.groupby("patient_number").size()
    if target_id_cols:
        tid = target_id_cols[0]
        targets_per_patient = df.groupby("patient_number")[tid].nunique()
        cores_per_target = df.groupby(tid).size() if tid in df.columns else None
    else:
        targets_per_patient = None
        cores_per_target = None

    # ── GG2+ / GG3+ positivity by split ──────────────────────────────────────
    gg2_pos = df.groupby("split")["binary_label_int"].agg(["sum", "count"])
    gg2_pos["prev"] = gg2_pos["sum"] / gg2_pos["count"]
    gg3_pos = None
    if "binary_label_gg3plus_int" in df.columns:
        gg3_pos = df.groupby("split")["binary_label_gg3plus_int"].agg(["sum", "count"])
        gg3_pos["prev"] = gg3_pos["sum"] / gg3_pos["count"]

    # ── New feature opportunity checks ───────────────────────────────────────

    # Signed distance to target surface
    signed_ok = "distance_midpoint_to_target_surface_mm" in df.columns and \
                "midpoint_inside_prostate" in df.columns

    # Core centreline length in mm (already have n_centerline_voxels + core_length_mm)
    centreline_mm_ok = "core_length_mm" in df.columns and "n_centerline_voxels" in df.columns

    # Target-size normalisation (need target volume or radius)
    target_vol_col = next((c for c in size_cols if c in df.columns), None)
    norm_by_size_ok = "distance_midpoint_to_target_surface_mm" in df.columns and target_vol_col is not None

    # Core rank/position within target (need target_id)
    rank_ok = bool(target_id_cols) and "distance_midpoint_to_target_surface_mm" in df.columns

    # n_cores per target
    n_cores_target_ok = bool(target_id_cols)

    # n_cores per patient
    n_cores_patient_ok = "patient_number" in df.columns

    # Interaction: distance_to_surface × psa_density
    inter_ok = ("distance_midpoint_to_target_surface_mm" in df.columns
                and ("psa_density" in df.columns or "psa_ng_ml" in df.columns))

    # ── Build the report ─────────────────────────────────────────────────────
    lines: list[str] = []
    W = lines.append

    W("# Extended Feature Opportunity Audit")
    W("")
    W(f"**Branch:** `feat/geometry-clinical-baselines`")
    W(f"**Script:** `src/audit_extended_feature_opportunities.py`")
    W(f"**Data:** `{DATA_PATH.relative_to(REPO_ROOT)}`")
    W(f"**Generated automatically. Do not edit by hand.**")
    W("")
    W("---")
    W("")

    # ── Section 1: Dataset basics ─────────────────────────────────────────────
    W("## 1. Dataset basics")
    W("")
    W(f"| Item | Value |")
    W(f"|------|-------|")
    W(f"| Raw rows (before filter) | {n_raw:,} |")
    W(f"| Rows after standard filter | {n_filtered:,} |")
    W(f"| Unique patients (filtered) | {n_patients} |")
    for sp in ["train", "val", "test"]:
        np_ = int(n_per_split.get(sp, 0))
        nc_ = int((df["split"] == sp).sum())
        W(f"| {sp.capitalize()} patients / cores | {np_} / {nc_:,} |")
    W(f"| Total columns (raw CSV) | {len(all_cols)} |")
    W(f"| Unknown / unclassified columns | {len(unknown_cols)} |")
    W("")

    W("**Endpoint prevalences (coord_match, valid split):**")
    W("")
    W("| Split | GG2+ positives | GG2+ prevalence | GG3+ positives | GG3+ prevalence |")
    W("|-------|---------------|-----------------|---------------|-----------------|")
    for sp in ["train", "val", "test"]:
        g2n = int(gg2_pos.loc[sp, "sum"]) if sp in gg2_pos.index else "—"
        g2p = f"{gg2_pos.loc[sp, 'prev']*100:.1f}%" if sp in gg2_pos.index else "—"
        if gg3_pos is not None:
            g3n = int(gg3_pos.loc[sp, "sum"]) if sp in gg3_pos.index else "—"
            g3p = f"{gg3_pos.loc[sp, 'prev']*100:.1f}%" if sp in gg3_pos.index else "—"
        else:
            g3n, g3p = "—", "—"
        W(f"| {sp.capitalize()} | {g2n} | {g2p} | {g3n} | {g3p} |")
    W("")

    W("**Label / split missingness (raw CSV):**")
    W("")
    for col in ["label_join_status", "split", "binary_label_int", "binary_label_gg3plus_int", "extraction_status"]:
        if col in raw.columns:
            W(f"- `{col}`: {_miss(raw[col])} missing")
    W("")

    # ── Section 2: Existing feature columns ───────────────────────────────────
    W("## 2. Existing feature columns")
    W("")

    def _feature_table(cols: list[str], header: str) -> None:
        W(f"### {header}")
        W("")
        present = [c for c in cols if c in df.columns]
        absent  = [c for c in cols if c not in df.columns]
        if present:
            W("| Column | Missing (coord_match) | Unique values |")
            W("|--------|-----------------------|---------------|")
            for c in present:
                W(f"| `{c}` | {_miss(df[c])} | {_nuniq(df[c])} |")
        if absent:
            W(f"\n*Not in CSV:* {', '.join(f'`{c}`' for c in absent)}")
        W("")

    _feature_table(CLINICAL_FEATURES, "Clinical features")
    _feature_table(BIOPSY_GEOMETRY_FEATURES, "Biopsy / prostate geometry features")
    _feature_table(TARGET_GEOMETRY_FEATURES, "Target geometry features")
    _feature_table(CENTERLINE_INTENSITY_FEATURES, "Centreline intensity features (MODE B subset)")

    if n_mode_b is not None:
        W(f"**MODE B subset** (`extraction_status == 'ok'`): {n_mode_b:,} rows ({mode_b_pct:.1f}% of filtered).")
        W("")

    # Unknown / unclassified columns
    W("### Unclassified columns (not in any known feature group)")
    W("")
    if unknown_cols:
        W("| Column | dtype | Missing (raw) | Sample values |")
        W("|--------|-------|--------------|---------------|")
        for c in sorted(unknown_cols):
            dtype = str(raw[c].dtype)
            miss  = _miss(raw[c])
            sample = raw[c].dropna().astype(str).unique()[:3]
            sample_str = "; ".join(sample) if len(sample) > 0 else "all NaN"
            W(f"| `{c}` | {dtype} | {miss} | {sample_str} |")
    else:
        W("*No unclassified columns.*")
    W("")

    # ── Section 3: Potential new geometry features ─────────────────────────────
    W("## 3. Potential new geometry features")
    W("")
    W("| Candidate feature | Computable? | Source columns | Missingness concern | Leakage risk | Scientific usefulness |")
    W("|------------------|------------|---------------|--------------------|--------------|-----------------------|")

    # 1. Signed distance
    sd_src = "distance_midpoint_to_target_surface_mm + trajectory_intersects_target (sign)"
    sd_miss = "Inherits from distance column; ~0% missing after coord_match filter"
    W(f"| Signed distance to target surface | {'Yes' if signed_ok else 'No — missing source'} | `{sd_src}` | {sd_miss} | None | **High** — distinguishes inside vs outside target more cleanly than unsigned distance |")

    # 2. Centreline length in mm (from n_centerline_voxels × voxel pitch if pitch available)
    cl_src = "n_centerline_voxels × voxel_pitch_mm (if pitch available)"
    cl_note = "Redundant with core_length_mm if pitch is constant across cores; useful only if voxel pitch varies"
    W(f"| Centreline inside target in mm | {'Possibly' if centreline_mm_ok else 'No'} | {cl_src} | Check pitch availability | None | Medium — adds interpretable units but likely redundant with fraction_inside |")

    # 3. Distance normalised by target size
    ns_src = f"`distance_midpoint_to_target_surface_mm` / `{target_vol_col or 'target_radius [NOT AVAILABLE]'}`"
    ns_note = "target_radius would require target volume → radius = (3V/4π)^(1/3)"
    W(f"| Distance normalised by target radius | {'Yes' if norm_by_size_ok else 'No — target size column not found'} | {ns_src} | {ns_note} | None if size comes from MRI (pre-biopsy) | **Medium** — controls for heterogeneous lesion size; requires target volume column |")

    # 4. Target volume / equivalent radius
    tv_col = target_vol_col or "NOT FOUND"
    W(f"| Target volume or equivalent radius | {'Yes — `' + tv_col + '`' if target_vol_col else 'No — no target size column found'} | `{tv_col}` | Check missingness | Low — MRI pre-biopsy measurement | **High if available** — major confounder (larger targets easier to hit) |")

    # 5. Core position/rank within target (by distance to centroid)
    rank_src = "rank of distance_midpoint_to_target_centroid_mm within target_id group"
    W(f"| Core rank within target | {'Yes — target_id found' if rank_ok else 'No — no target_id column found'} | target_id + distance_midpoint_to_target_centroid_mm | Inherits from source columns | None | Medium — captures systematic spatial sampling pattern |")

    # 6. N cores per target
    W(f"| N cores per target | {'Yes — target_id found' if n_cores_target_ok else 'No — no target_id column found'} | target_id | None | Possible soft leakage if used as predictor (counts are fixed once biopsy is done) | Low — informative about sampling intensity per lesion |")

    # 7. N cores per patient
    W(f"| N cores per patient | Yes | patient_number | None | Possible soft leakage (correlated with patient complexity) | Low — known confounder at patient level |")

    # 8. Interaction: distance × psa_density
    W(f"| distance_to_surface × psa_density | {'Yes' if inter_ok else 'No'} | distance_midpoint_to_target_surface_mm, psa_density | None if both columns available | None | Low — interaction hard to interpret; risk of overfitting in small test set |")
    W("")

    # ── Section 4: Patient/target grouping structure ───────────────────────────
    W("## 4. Patient / target grouping structure")
    W("")
    W(f"**Cores per patient (coord_match, valid split):**")
    W(f"- Min: {int(cores_per_patient.min())} | Median: {int(cores_per_patient.median())} | "
       f"Mean: {cores_per_patient.mean():.1f} | Max: {int(cores_per_patient.max())}")
    W("")

    if target_id_cols:
        W(f"**Target ID columns found:** {', '.join(f'`{c}`' for c in target_id_cols)}")
        tid = target_id_cols[0]
        if tid in df.columns:
            W(f"- Unique targets ({tid}): {df[tid].nunique()}")
            tpp = targets_per_patient.describe()
            W(f"- Targets per patient: min {int(tpp['min'])} / median {int(tpp['50%'])} / max {int(tpp['max'])}")
            if cores_per_target is not None:
                cpt = cores_per_target.describe()
                W(f"- Cores per target: min {int(cpt['min'])} / median {int(cpt['50%'])} / max {int(cpt['max'])}")
        W("")
        W("**Positivity rate by target:**")
        if tid in df.columns:
            tgt_pos = df.groupby(tid)["binary_label_int"].agg(["sum", "count", "mean"])
            tgt_pos.columns = ["n_pos_gg2", "n_cores", "prev_gg2"]
            W(f"- Targets with ≥1 positive core (GG2+): "
               f"{(tgt_pos['prev_gg2'] > 0).sum()} / {len(tgt_pos)} "
               f"({100*(tgt_pos['prev_gg2']>0).mean():.1f}%)")
            W(f"- Median GG2+ positivity per target: {tgt_pos['prev_gg2'].median()*100:.1f}%")
    else:
        W("**No target_id column found.** Cores cannot be grouped by MRI lesion.")
        W("")
        W("*Implication:* Target-level features (cores per target, positivity per target, "
           "target volume normalisation) cannot be computed from the current CSV without "
           "re-joining the original MRI metadata.")
    W("")

    # ── Section 5: Image / radiomics feasibility ──────────────────────────────
    W("## 5. Image / radiomics feasibility")
    W("")
    if intensity_cols:
        W(f"**Intensity-like columns found ({len(intensity_cols)}):**")
        W("")
        W("| Column | dtype | Missing (raw) |")
        W("|--------|-------|--------------|")
        for c in intensity_cols:
            W(f"| `{c}` | {str(raw[c].dtype)} | {_miss(raw[c])} |")
        W("")
        W("These columns may allow first-order radiomics experiments *without* reprocessing "
           "raw DICOMs, subject to the MODE B availability restriction.")
    else:
        W("**No intensity / radiomics columns found beyond the four centreline intensity features "
           "already in CENTERLINE_INTENSITY_FEATURES.**")
        W("")
        W("First-order radiomics from MRI patches (T2, ADC, DWI histogram statistics) are "
           "**not available** in the current shareable CSV. Testing them would require extracting "
           "voxel values from raw DICOM series, which is out of scope for this sprint.")
    W("")

    # ── Section 6: Prior biopsy / history feasibility ─────────────────────────
    W("## 6. Prior biopsy / history feasibility")
    W("")
    if prior_bx_cols:
        W(f"**Prior biopsy / history columns found ({len(prior_bx_cols)}):**")
        W("")
        W("| Column | dtype | Missing (raw) | Sample values |")
        W("|--------|-------|--------------|---------------|")
        for c in prior_bx_cols:
            dtype = str(raw[c].dtype)
            miss  = _miss(raw[c])
            sample = raw[c].dropna().astype(str).unique()[:3]
            sample_str = "; ".join(sample) if len(sample) > 0 else "all NaN"
            W(f"| `{c}` | {dtype} | {miss} | {sample_str} |")
        W("")
        W("**Leakage assessment:** If a 'prior positive target' flag is recorded at biopsy "
           "time (i.e., whether the same lesion was positive at a previous session), it is "
           "clinically legitimate — this is information available before the core outcome is "
           "known. Flags that encode *the same session's outcome* would be leakage.")
    else:
        W("**No prior biopsy / history columns found** matching keyword patterns "
           "(prior, previous, history, repeat, prior_pos, prior_grade, prior_cancer).")
        W("")
        W("*Note:* The session summary states that 7,649 cores target 'prior positive lesions' "
           "in the full dataset. If a `target_type` or `target_label` column encodes this "
           "(e.g., 'AS' for active surveillance / prior positive), it may serve as a legitimate "
           "clinical feature. Check unclassified columns (§2) for such a flag.")
    W("")

    # ── Section 7: Ranked recommendations ────────────────────────────────────
    W("## 7. Ranked feature recommendations")
    W("")
    W("### A. Low-cost, high-value — test now")
    W("")
    W("1. **Signed distance to target surface.**  "
       "Convert `distance_midpoint_to_target_surface_mm` to signed: negative if needle midpoint "
       "is outside the target (use `trajectory_intersects_target == 0` as a proxy for sign, "
       "or derive from `approximate_fraction_of_centerline_inside_target == 0`).  "
       "Expected gain: separates 'close but outside' from 'close and inside' — the current "
       "unsigned distance cannot distinguish these.")
    W("")
    W("2. **Target volume (if found as a column).**  "
       "Check §4 and §2 for a size column (volume, longest axis). If available: "
       "(a) use as a standalone clinical covariate (larger target → easier to hit); "
       "(b) use to normalise distance features. No leakage risk — MRI measurement precedes biopsy.")
    W("")
    W("3. **N cores per patient (aggregate-level covariate).**  "
       "Already computable from `patient_number`. "
       "Flag as a *patient-level* covariate only; not to be used as a core-level predictor "
       "without proper hierarchical modelling, as it could encode selection bias "
       "(sicker patients may receive more cores).")
    W("")

    W("### B. Medium-cost — test only if target_id is available")
    W("")
    W("1. **Target-level features** (cores per target, target positivity from training set, "
       "target-level summary of distance features) — require a target_id column.  "
       "If target_id is absent from the current CSV, these require re-joining MRI metadata.  "
       "Do not attempt without a target_id column in the CSV.")
    W("")
    W("2. **N cores per target** — same requirement.")
    W("")
    W("3. **Core rank within target** (by distance to centroid, within target_id group) — "
       "conceptually clean but requires target_id.")
    W("")

    W("### C. Do not test now")
    W("")
    W("1. **Centreline intensity features (MODE B subset).**  "
       f"Available for only {n_mode_b:,} cores ({mode_b_pct:.1f}%) — "
       if n_mode_b is not None else "Availability unknown — "
       "restricts the test set to a non-representative subset. "
       "These features were already evaluated in MODE B experiments and did not "
       "substantially improve over geometry-only models. Do not retest without new data.")
    W("")
    W("2. **Interaction terms** (distance × psa_density etc.).  "
       "Risk of overfitting on a 120-patient test set; no strong scientific prior; "
       "collinearity between distance features and PSA features already documented.")
    W("")
    W("3. **CNN / DL patch features.**  "
       "Require DICOM extraction, GPU infrastructure, and a substantially larger labelled "
       "cohort for reliable evaluation. Out of scope before manuscript freeze.")
    W("")
    W("4. **PSA density non-linearity** (spline, binned regression).  "
       "Association model results already include psa_density as a linear term; "
       "non-linearity is Discussion §D9 future-work item, not a current-sprint feature.")
    W("")

    # ── Write output ──────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit written to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
