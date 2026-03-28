"""
Step 1 — Build manifest.csv

For each biopsy core in the TCIA spreadsheet:
  - Assign a Gleason grade label (Benign, GG1–GG5)
  - Link to the MRI DICOM series folder via SeriesInstanceUID
  - Keep MRI LPS coordinates of needle tip and base

Outputs:
  data/dicom_index.csv   — series_uid → dicom_series_path (one row per MRI series)
  data/manifest.csv      — one row per biopsy core, fully linked
"""

import os
import re
import pandas as pd
import pydicom
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DICOM_ROOT = DATA_DIR / "dicom" / "manifest-1774220151712" / "Prostate-MRI-US-Biopsy"
BIOPSY_XLSX = DATA_DIR / "raw" / "TCIA-Biopsy-Data_2020-07-14.xlsx"


# ── 1. Load the spreadsheet ────────────────────────────────────────────────────

def load_biopsy_spreadsheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Assign Gleason Grade Group label
    def grade_group(row):
        pg = row["Primary Gleason"]
        sg = row["Secondary Gleason"]
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

    df["gleason_grade"] = df.apply(grade_group, axis=1)

    # Rename for convenience
    df = df.rename(columns={
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
    })

    keep_cols = [
        "subject_id", "series_uid_mri", "series_uid_us",
        "tip_x_mri", "tip_y_mri", "tip_z_mri",
        "base_x_mri", "base_y_mri", "base_z_mri",
        "primary_gleason", "secondary_gleason", "gleason_grade",
        "cancer_length_mm", "pct_cancer", "core_label",
        "psa", "prostate_volume_cc",
    ]
    return df[keep_cols]


# ── 2. Build DICOM series index ────────────────────────────────────────────────

def build_dicom_index(dicom_root: Path, cache_path: Path) -> pd.DataFrame:
    """
    Walk all series directories under dicom_root, read one DICOM file per series
    to extract SeriesInstanceUID, and return a DataFrame mapping UID → path.
    Results are cached to cache_path so this only runs once.
    """
    if cache_path.exists():
        print(f"Loading cached DICOM index from {cache_path}")
        return pd.read_csv(cache_path)

    print("Building DICOM index (reads one DICOM per series — runs once)...")
    records = []

    # Depth: subject / study / series / *.dcm
    subject_dirs = sorted([d for d in dicom_root.iterdir() if d.is_dir()])
    for subject_dir in tqdm(subject_dirs, desc="Subjects"):
        subject_id = subject_dir.name  # e.g. Prostate-MRI-US-Biopsy-0001
        for study_dir in subject_dir.iterdir():
            if not study_dir.is_dir():
                continue
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                dcm_files = sorted(series_dir.glob("*.dcm"))
                if not dcm_files:
                    continue
                try:
                    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                    records.append({
                        "subject_id": subject_id,
                        "series_uid": str(ds.SeriesInstanceUID),
                        "modality": getattr(ds, "SeriesDescription", ""),
                        "dicom_series_path": str(series_dir),
                        "n_slices": len(dcm_files),
                    })
                except Exception as e:
                    print(f"  Warning: could not read {dcm_files[0]}: {e}")

    index_df = pd.DataFrame(records)
    index_df.to_csv(cache_path, index=False)
    print(f"Saved DICOM index ({len(index_df)} series) → {cache_path}")
    return index_df


# ── 3. Link and filter ─────────────────────────────────────────────────────────

def build_manifest(biopsy_df: pd.DataFrame, dicom_index: pd.DataFrame) -> pd.DataFrame:
    """
    Join biopsy cores to their MRI DICOM series path.
    Only keep cores that have a non-null MRI series UID present in the DICOM index.

    Clinical task (from collaborating clinician):
      label=0  →  GG0 (Benign) + GG1 (3+3) + GG2 (3+4)   — not clinically significant
      label=1  →  GG3 (4+3) + GG4 + GG5                   — clinically significant cancer
    """
    # Keep only T2 MR series in the index
    mr_index = dicom_index[dicom_index["modality"].str.contains("t2", case=False, na=False)].copy()
    mr_index = mr_index.rename(columns={
        "series_uid": "series_uid_mri",
        "dicom_series_path": "mri_dicom_path",
        "n_slices": "mri_n_slices",
    })[["series_uid_mri", "mri_dicom_path", "mri_n_slices"]]

    n_total = len(biopsy_df)

    # Drop cores without MRI series UID
    biopsy_df = biopsy_df.dropna(subset=["series_uid_mri"])
    print(f"Cores with MRI series UID: {len(biopsy_df)} / {n_total}")

    # Drop zero-length needles (tip == base)
    dx = biopsy_df["tip_x_mri"] - biopsy_df["base_x_mri"]
    dy = biopsy_df["tip_y_mri"] - biopsy_df["base_y_mri"]
    dz = biopsy_df["tip_z_mri"] - biopsy_df["base_z_mri"]
    needle_len = (dx**2 + dy**2 + dz**2) ** 0.5
    valid = needle_len >= 0.1
    print(f"Zero-length needles dropped: {(~valid).sum()}")
    biopsy_df = biopsy_df[valid].copy()
    biopsy_df["needle_length_mm"] = needle_len[valid]

    # Binary clinical label: GG3+ = 1 (clinically significant), else = 0
    biopsy_df["label"] = biopsy_df["gleason_grade"].isin(["GG3", "GG4", "GG5"]).astype(int)

    # Join to DICOM paths
    manifest = biopsy_df.merge(mr_index, on="series_uid_mri", how="left")
    n_linked = manifest["mri_dicom_path"].notna().sum()
    print(f"Cores linked to a DICOM path: {n_linked} / {len(manifest)}")

    manifest = manifest.reset_index(drop=True)
    manifest.insert(0, "core_id", manifest.index)

    return manifest


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 1: Building manifest ===\n")

    biopsy_df = load_biopsy_spreadsheet(BIOPSY_XLSX)
    print(f"Spreadsheet loaded: {len(biopsy_df)} cores, {biopsy_df['subject_id'].nunique()} subjects\n")

    dicom_index = build_dicom_index(DICOM_ROOT, DATA_DIR / "dicom_index.csv")
    print(f"DICOM index: {len(dicom_index)} series across "
          f"{dicom_index['subject_id'].nunique()} subjects\n")

    manifest = build_manifest(biopsy_df, dicom_index)

    out_path = DATA_DIR / "manifest.csv"
    manifest.to_csv(out_path, index=False)
    print(f"\nManifest saved → {out_path}")
    print(f"  Total cores   : {len(manifest)}")
    print(f"  Linked cores  : {manifest['mri_dicom_path'].notna().sum()}")
    print(f"  Subjects      : {manifest['subject_id'].nunique()}")
    print(f"\nGleason grade distribution:")
    print(manifest["gleason_grade"].value_counts().to_string())
    print(f"\nBinary label distribution (GG3+ vs rest):")
    print(manifest["label"].value_counts().rename({0: "label=0 (GG0-2)", 1: "label=1 (GG3+)"}).to_string())
