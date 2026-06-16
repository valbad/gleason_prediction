"""
Dataset audit — reproducible summary of the Gleason-prediction data pipeline.

Walks through every stage of the pipeline and reports counts at each step:

  1. Raw TCIA biopsy spreadsheet (data/raw/TCIA-Biopsy-Data_2020-07-14.xlsx)
  2. Linked manifest (data/manifest.csv, produced by build_manifest.py)
  3. Contamination filter (data/cores_to_drop_contamination.csv)
  4. ROI extraction outcome (data/extraction_report.csv, from extract_rois.py)
  5. Train / validation / test split (as used by notebooks/04 and 06)
  6. Availability of auxiliary clinical variables (PSA, prostate volume,
     "Target Data")

Every section degrades gracefully: if an input file is missing, the
corresponding rows are reported as "n/a" with an explanatory note instead
of raising an error, so the audit can be run at any point in the pipeline.

Outputs
-------
reports/dataset_audit.csv   long-format table: section, metric, value, details
reports/dataset_audit.md    the same information as a Markdown report

Usage
-----
    python src/audit_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

MANIFEST          = DATA_DIR / "manifest.csv"
EXTRACTION_REPORT = DATA_DIR / "extraction_report.csv"
DROP_LIST         = DATA_DIR / "cores_to_drop_contamination.csv"
TEST_SUBJECTS     = DATA_DIR / "test_subjects.csv"
BIOPSY_XLSX       = DATA_DIR / "raw" / "TCIA-Biopsy-Data_2020-07-14.xlsx"

# Must match build_manifest.py: needles shorter than this are dropped as
# degenerate (tip == base).
MIN_NEEDLE_LENGTH_MM = 0.1

# Must match notebooks/04_baseline.ipynb / 06_dl_gnn_final.ipynb so the audit
# reflects the split actually used for model evaluation.
TEST_SIZE    = 0.15
N_CV_FOLDS   = 5
RANDOM_STATE = 42

GG_LOW  = {"Benign", "GG1", "GG2"}   # label = 0 (not clinically significant)
GG_HIGH = {"GG3", "GG4", "GG5"}      # label = 1 (clinically significant)

COORD_COLS = [
    "tip_x_mri", "tip_y_mri", "tip_z_mri",
    "base_x_mri", "base_y_mri", "base_z_mri",
]

# Maps raw TCIA spreadsheet column names -> internal names used by
# build_manifest.py. Columns not listed here are kept under their
# original spreadsheet name.
RAW_COLUMN_MAP = {
    "Patient Number": "subject_id",
    "Series Instance UID (MRI)": "series_uid_mri",
    "Series Instance UID (US)": "series_uid_us",
    "Bx Tip X (MRI Coord)": "tip_x_mri",
    "Bx Tip Y (MRI Coord)": "tip_y_mri",
    "Bx Tip Z (MRI Coord)": "tip_z_mri",
    "Bx Base X (MRI Coord)": "base_x_mri",
    "Bx Base Y (MRI Coord)": "base_y_mri",
    "Bx Base Z (MRI Coord)": "base_z_mri",
    "Primary Gleason": "primary_gleason",
    "Secondary Gleason": "secondary_gleason",
    "Cancer Length (mm)": "cancer_length_mm",
    "% Cancer in Core": "pct_cancer",
    "Core Label": "core_label",
    "PSA (ng/mL)": "psa",
    "Prostate Volume (CC)": "prostate_volume_cc",
}


# ── Small helpers ────────────────────────────────────────────────────────────

def grade_group(row):
    """Gleason Grade Group assignment — mirrors build_manifest.load_biopsy_spreadsheet."""
    pg, sg = row.get("primary_gleason"), row.get("secondary_gleason")
    if pd.isna(pg) or pd.isna(sg):
        return "Benign"
    total = pg + sg
    if pg == 3 and sg == 3:
        return "GG1"
    elif pg == 3 and sg == 4:
        return "GG2"
    elif pg == 4 and sg == 3:
        return "GG3"
    elif total == 8:
        return "GG4"
    else:  # total >= 9
        return "GG5"


def safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name], or an all-NaN series if the column is absent."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def pct(n, total) -> str:
    if not total:
        return "n/a"
    return f"{100 * n / total:.1f}%"


def try_read_csv(path: Path, **kwargs):
    if not path.exists():
        print(f"  (not found: {path.relative_to(REPO_ROOT)})")
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"  Warning: failed to read {path.relative_to(REPO_ROOT)}: {e}")
        return None


def try_read_excel(path: Path, **kwargs):
    if not path.exists():
        print(f"  (not found: {path.relative_to(REPO_ROOT)})")
        return None
    try:
        return pd.read_excel(path, **kwargs)
    except Exception as e:
        print(f"  Warning: failed to read {path.relative_to(REPO_ROOT)}: {e}")
        return None


# ── Report rows ─────────────────────────────────────────────────────────────

REPORT_ROWS = []


def add(section: str, metric: str, value, details: str = ""):
    REPORT_ROWS.append({"section": section, "metric": metric, "value": value, "details": details})


# ── 1. Raw biopsy spreadsheet ───────────────────────────────────────────────

def audit_raw_spreadsheet():
    print("=== 1. Raw biopsy spreadsheet ===")
    raw_df = try_read_excel(BIOPSY_XLSX)

    if raw_df is None:
        note = f"file not found: data/raw/{BIOPSY_XLSX.name}"
        add("1. Raw biopsy data", "Raw biopsy cores", "n/a", note)
        add("1. Raw biopsy data", "Cores with valid MRI coordinates", "n/a", note)
        add("1. Raw biopsy data", "Cores with valid Gleason label", "n/a", note)
        add("1. Raw biopsy data", "Cores with valid MRI SeriesInstanceUID", "n/a", note)
        add("1. Raw biopsy data", "Number of patients (raw)", "n/a", note)
        add("5. Clinical variable availability", "PSA available", "n/a", note)
        add("5. Clinical variable availability", "Prostate volume available", "n/a", note)
        add("5. Clinical variable availability", "Target Data available", "n/a", note)
        return None

    raw_df = raw_df.rename(columns=RAW_COLUMN_MAP)
    n_raw = len(raw_df)
    add("1. Raw biopsy data", "Raw biopsy cores", n_raw)

    # Valid MRI coordinates: tip + base fully specified and the needle has
    # non-zero length (same threshold used by build_manifest.py).
    coords = pd.concat([safe_col(raw_df, c) for c in COORD_COLS], axis=1)
    coords.columns = COORD_COLS
    have_coords = coords.notna().all(axis=1)
    dx = coords["tip_x_mri"] - coords["base_x_mri"]
    dy = coords["tip_y_mri"] - coords["base_y_mri"]
    dz = coords["tip_z_mri"] - coords["base_z_mri"]
    needle_len = np.sqrt(dx**2 + dy**2 + dz**2)
    valid_coords = have_coords & (needle_len >= MIN_NEEDLE_LENGTH_MM)
    n_valid_coords = int(valid_coords.sum())
    add("1. Raw biopsy data", "Cores with valid MRI coordinates", n_valid_coords,
        f"{pct(n_valid_coords, n_raw)} of raw cores "
        f"(tip/base non-null, needle length >= {MIN_NEEDLE_LENGTH_MM} mm)")

    # Valid Gleason label: grade_group() always returns a value, so report
    # the total plus a breakdown of scored vs. benign-by-absence-of-score.
    raw_df["gleason_grade"] = raw_df.apply(grade_group, axis=1)
    has_score = safe_col(raw_df, "primary_gleason").notna() & safe_col(raw_df, "secondary_gleason").notna()
    n_valid_gleason = int(raw_df["gleason_grade"].notna().sum())
    add("1. Raw biopsy data", "Cores with valid Gleason label", n_valid_gleason,
        f"{pct(n_valid_gleason, n_raw)}; {int(has_score.sum())} with explicit "
        f"Primary/Secondary Gleason score, {int((~has_score).sum())} labeled Benign (no score reported)")

    # Valid MRI SeriesInstanceUID
    n_valid_uid = int(safe_col(raw_df, "series_uid_mri").notna().sum())
    add("1. Raw biopsy data", "Cores with valid MRI SeriesInstanceUID", n_valid_uid,
        f"{pct(n_valid_uid, n_raw)} of raw cores")

    # Patients
    n_patients_raw = int(safe_col(raw_df, "subject_id").nunique())
    add("1. Raw biopsy data", "Number of patients (raw)", n_patients_raw)

    # GG distribution on raw spreadsheet
    n_low = int(raw_df["gleason_grade"].isin(GG_LOW).sum())
    n_high = int(raw_df["gleason_grade"].isin(GG_HIGH).sum())
    add("2. Gleason grade distribution (raw)", "GG0-2 cores (label=0)", n_low, pct(n_low, n_raw))
    add("2. Gleason grade distribution (raw)", "GG3+ cores (label=1)", n_high, pct(n_high, n_raw))
    for gg, cnt in raw_df["gleason_grade"].value_counts().sort_index().items():
        add("2. Gleason grade distribution (raw)", f"  {gg}", int(cnt), pct(int(cnt), n_raw))

    # PSA / prostate volume / "Target Data" availability
    if "psa" in raw_df.columns:
        n_psa = int(raw_df["psa"].notna().sum())
        add("5. Clinical variable availability", "PSA available", n_psa, f"{pct(n_psa, n_raw)} of raw cores")
    else:
        add("5. Clinical variable availability", "PSA available", "n/a", "column not found in spreadsheet")

    if "prostate_volume_cc" in raw_df.columns:
        n_vol = int(raw_df["prostate_volume_cc"].notna().sum())
        add("5. Clinical variable availability", "Prostate volume available", n_vol, f"{pct(n_vol, n_raw)} of raw cores")
    else:
        add("5. Clinical variable availability", "Prostate volume available", "n/a", "column not found in spreadsheet")

    audit_target_data(raw_df, n_raw)

    return raw_df


def audit_target_data(raw_df: pd.DataFrame, n_raw: int):
    """Report availability of any "Target Data"-like column or sheet, if present."""
    target_cols = [c for c in raw_df.columns if "target" in str(c).lower()]

    target_sheets = []
    try:
        xls = pd.ExcelFile(BIOPSY_XLSX)
        target_sheets = [s for s in xls.sheet_names if "target" in s.lower()]
    except Exception as e:
        print(f"  Warning: could not list sheets in {BIOPSY_XLSX.name}: {e}")

    if not target_cols and not target_sheets:
        add("5. Clinical variable availability", "Target Data available", "Not present",
            "no column or sheet matching 'target' found in the biopsy spreadsheet")
        return

    for col in target_cols:
        n_present = int(raw_df[col].notna().sum())
        add("5. Clinical variable availability", f"Target Data: '{col}' available", n_present,
            f"{pct(n_present, n_raw)} of raw cores")

    if target_sheets:
        add("5. Clinical variable availability", "Target Data sheet(s) found", ", ".join(target_sheets),
            "present in workbook but not loaded by build_manifest.py (only the first sheet is read)")


# ── 2. Manifest (build_manifest.py output) ──────────────────────────────────

def audit_manifest():
    print("=== 2. Manifest (data/manifest.csv) ===")
    manifest_df = try_read_csv(MANIFEST)

    if manifest_df is None:
        note = "file not found: data/manifest.csv (run src/build_manifest.py)"
        add("3. Manifest (build_manifest.py)", "Cores in manifest.csv", "n/a", note)
        add("3. Manifest (build_manifest.py)", "Cores linked to a DICOM series", "n/a", note)
        add("3. Manifest (build_manifest.py)", "Number of patients (manifest)", "n/a", note)
        add("4. Gleason grade distribution (manifest)", "GG0-2 cores (label=0)", "n/a", note)
        add("4. Gleason grade distribution (manifest)", "GG3+ cores (label=1)", "n/a", note)
        return None

    n_manifest = len(manifest_df)
    add("3. Manifest (build_manifest.py)", "Cores in manifest.csv", n_manifest)

    if "mri_dicom_path" in manifest_df.columns:
        n_linked = int(manifest_df["mri_dicom_path"].notna().sum())
        add("3. Manifest (build_manifest.py)", "Cores linked to a DICOM series", n_linked,
            f"{pct(n_linked, n_manifest)} of manifest cores")
    else:
        add("3. Manifest (build_manifest.py)", "Cores linked to a DICOM series", "n/a",
            "column 'mri_dicom_path' not found in manifest.csv")

    if "subject_id" in manifest_df.columns:
        n_patients = int(manifest_df["subject_id"].nunique())
        add("3. Manifest (build_manifest.py)", "Number of patients (manifest)", n_patients)

    if "label" in manifest_df.columns:
        n_low = int((manifest_df["label"] == 0).sum())
        n_high = int((manifest_df["label"] == 1).sum())
        add("4. Gleason grade distribution (manifest)", "GG0-2 cores (label=0)", n_low, pct(n_low, n_manifest))
        add("4. Gleason grade distribution (manifest)", "GG3+ cores (label=1)", n_high, pct(n_high, n_manifest))
    else:
        add("4. Gleason grade distribution (manifest)", "GG0-2 cores (label=0)", "n/a", "column 'label' not found")
        add("4. Gleason grade distribution (manifest)", "GG3+ cores (label=1)", "n/a", "column 'label' not found")

    # Fall back to manifest.csv for PSA / prostate volume availability if the
    # raw spreadsheet wasn't available — manifest.csv carries both columns
    # through from build_manifest.py, just restricted to linked cores.
    if not any(r["section"] == "5. Clinical variable availability" and r["metric"] == "PSA available"
               for r in REPORT_ROWS):
        if "psa" in manifest_df.columns:
            n_psa = int(manifest_df["psa"].notna().sum())
            add("5. Clinical variable availability", "PSA available", n_psa,
                f"{pct(n_psa, n_manifest)} of manifest cores (raw spreadsheet unavailable)")
        if "prostate_volume_cc" in manifest_df.columns:
            n_vol = int(manifest_df["prostate_volume_cc"].notna().sum())
            add("5. Clinical variable availability", "Prostate volume available", n_vol,
                f"{pct(n_vol, n_manifest)} of manifest cores (raw spreadsheet unavailable)")

    return manifest_df


# ── 3. Contamination filter + extraction report ─────────────────────────────

def audit_contamination_and_extraction(manifest_df):
    print("=== 3. Contamination filter & extraction report ===")
    if manifest_df is None:
        add("6. Contamination filter", "Cores removed (contamination)", "n/a", "manifest.csv not found")
        add("7. Final dataset", "Final usable cores", "n/a", "manifest.csv not found")
        return None

    n_manifest = len(manifest_df)
    drop_df = try_read_csv(DROP_LIST)

    if drop_df is not None and "core_id" in drop_df.columns:
        drop_ids = set(drop_df["core_id"])
        is_dropped = manifest_df["core_id"].isin(drop_ids)
        n_dropped = int(is_dropped.sum())
        kept_df = manifest_df[~is_dropped].copy()
        add("6. Contamination filter", "Cores removed (contamination)", n_dropped,
            f"{pct(n_dropped, n_manifest)} of manifest cores")
    else:
        n_dropped = 0
        kept_df = manifest_df.copy()
        add("6. Contamination filter", "Cores removed (contamination)", "n/a",
            "file not found: data/cores_to_drop_contamination.csv")

    add("6. Contamination filter", "Cores after contamination filter", len(kept_df),
        f"{pct(len(kept_df), n_manifest)} of manifest cores")

    extraction_df = try_read_csv(EXTRACTION_REPORT)
    if extraction_df is not None and {"core_id", "status"}.issubset(extraction_df.columns):
        ok_ids = set(extraction_df.loc[extraction_df["status"] == "ok", "core_id"])
        final_df = kept_df[kept_df["core_id"].isin(ok_ids)].copy()
        n_failed_extraction = int(kept_df["core_id"].isin(set(extraction_df["core_id"]) - ok_ids).sum())
        add("7. Final dataset", "Cores that failed ROI extraction", n_failed_extraction,
            f"{pct(n_failed_extraction, len(kept_df))} of contamination-filtered cores")
        add("7. Final dataset", "Final usable cores", len(final_df),
            f"{pct(len(final_df), n_manifest)} of manifest cores; "
            f"contamination-filtered AND successfully extracted (status == 'ok')")
    else:
        final_df = kept_df
        add("7. Final dataset", "Final usable cores", len(final_df),
            "extraction_report.csv not found — equals contamination-filtered count "
            "(extraction status not applied)")

    if "label" in final_df.columns:
        n_low = int((final_df["label"] == 0).sum())
        n_high = int((final_df["label"] == 1).sum())
        add("7. Final dataset", "  GG0-2 cores (label=0)", n_low, pct(n_low, len(final_df)))
        add("7. Final dataset", "  GG3+ cores (label=1)", n_high, pct(n_high, len(final_df)))
    if "subject_id" in final_df.columns:
        add("7. Final dataset", "  Number of patients", int(final_df["subject_id"].nunique()))

    return final_df


# ── 4. Train / validation / test split ──────────────────────────────────────

def audit_split(final_df):
    print("=== 4. Train / validation / test split ===")
    if final_df is None or final_df.empty:
        add("8. Train/validation/test split", "split", "n/a", "no usable cores available")
        return

    required = {"subject_id", "label", "core_id"}
    if not required.issubset(final_df.columns):
        add("8. Train/validation/test split", "split", "n/a",
            f"required columns missing from manifest: {sorted(required - set(final_df.columns))}")
        return

    try:
        from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    except ImportError:
        add("8. Train/validation/test split", "split", "n/a", "scikit-learn not available")
        return

    df = final_df.reset_index(drop=True)

    # Subject-level positive flag for stratified test-set splitting.
    subj_label = df.groupby("subject_id")["label"].max().reset_index()
    subjects = subj_label["subject_id"].values
    subj_pos = subj_label["label"].values

    if TEST_SUBJECTS.exists():
        test_subjs = set(pd.read_csv(TEST_SUBJECTS, header=None)[0].tolist())
        split_source = "data/test_subjects.csv (fixed split saved by notebooks/04_baseline.ipynb)"
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        _, test_idx = next(splitter.split(subjects, subj_pos, groups=subjects))
        test_subjs = set(subjects[test_idx])
        split_source = (f"recomputed with GroupShuffleSplit(test_size={TEST_SIZE}, "
                         f"random_state={RANDOM_STATE}) — data/test_subjects.csv not found")

    df["split"] = df["subject_id"].apply(lambda s: "test" if s in test_subjs else "trainval")
    add("8. Train/validation/test split", "split source", split_source)

    trainval_df = df[df["split"] == "trainval"].reset_index(drop=True)
    test_df = df[df["split"] == "test"]

    # Trainval is evaluated with 5-fold StratifiedGroupKFold CV rather than a
    # single fixed train/val split. Fold 0 is reported as a representative
    # "validation" split; all folds are similar in size by construction.
    cv = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(cv.split(trainval_df["core_id"], trainval_df["label"], groups=trainval_df["subject_id"]))
    train_df = trainval_df.iloc[tr_idx]
    val_df = trainval_df.iloc[va_idx]
    add("8. Train/validation/test split", "validation set",
        f"fold 0 of {N_CV_FOLDS}-fold StratifiedGroupKFold(random_state={RANDOM_STATE}) over trainval",
        "all folds are similarly sized; 'train' below is the remaining 4 folds")

    n_total_cores = len(df)
    n_total_subjs = df["subject_id"].nunique()

    for name, sub in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        n_cores = len(sub)
        n_subjs = sub["subject_id"].nunique()
        n_pos = int((sub["label"] == 1).sum())
        add("8. Train/validation/test split", f"{name} — cores", n_cores,
            f"{pct(n_cores, n_total_cores)} of cores; GG3+ rate {pct(n_pos, n_cores)}")
        add("8. Train/validation/test split", f"{name} — patients", n_subjs,
            f"{pct(n_subjs, n_total_subjs)} of patients")


# ── Output writers ───────────────────────────────────────────────────────────

def write_csv(path: Path):
    df = pd.DataFrame(REPORT_ROWS, columns=["section", "metric", "value", "details"])
    df.to_csv(path, index=False)
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")


def write_markdown(path: Path):
    lines = ["# Dataset Audit Report", "", "Generated by `src/audit_dataset.py`.", ""]

    current_section = None
    for row in REPORT_ROWS:
        if row["section"] != current_section:
            current_section = row["section"]
            lines += ["", f"## {current_section}", "", "| Metric | Value | Details |", "|---|---|---|"]
        metric = str(row["metric"]).replace("|", "\\|")
        value = str(row["value"]).replace("|", "\\|")
        details = str(row["details"]).replace("|", "\\|")
        lines.append(f"| {metric} | {value} | {details} |")

    path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Dataset audit ===\n")

    audit_raw_spreadsheet()
    print()
    manifest_df = audit_manifest()
    print()
    final_df = audit_contamination_and_extraction(manifest_df)
    print()
    audit_split(final_df)
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS_DIR / "dataset_audit.csv")
    write_markdown(REPORTS_DIR / "dataset_audit.md")

    print("\n=== Summary ===")
    for row in REPORT_ROWS:
        print(f"  [{row['section']}] {row['metric']}: {row['value']}"
              + (f"  ({row['details']})" if row["details"] else ""))
