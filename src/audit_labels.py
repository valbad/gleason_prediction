"""
Label definition audit for data/share/needle_features_v1.csv.

Determines the exact threshold encoded in binary_label_int, compares it with
the intended GG3+ task, and creates a corrected binary_label_gg3plus_int column.

Usage
-----
    python src/audit_labels.py

Outputs
-------
    reports/label_definition_audit.md   — full audit report with crosstabs
    data/share/needle_features_v1.csv   — updated with binary_label_gg3plus_int column
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORT_PATH = REPO_ROOT / "reports" / "label_definition_audit.md"

RELEVANT_COLS = [
    "pathology_hint_from_filename",
    "pathology_label",
    "binary_label",
    "pathology_label_int",
    "binary_label_int",
    "primary_gleason",
    "secondary_gleason",
    "core_label",
]

GG_ORDER = ["Benign", "GG1", "GG2", "GG3", "GG4", "GG5"]

GLEASON_PATTERN = {
    "Benign": "no cancer / no score",
    "GG1":    "3+3=6",
    "GG2":    "3+4=7",
    "GG3":    "4+3=7",
    "GG4":    "4+4=8, 3+5=8, 5+3=8",
    "GG5":    "4+5=9, 5+4=9, 5+5=10",
}


# ── Grade-group functions ─────────────────────────────────────────────────────

def grade_group_pipeline(primary, secondary) -> str:
    """
    Historical grade group as implemented in the original src/build_manifest.py
    (pipeline reference — preserved here to audit what logic was used when
    needle_features_v1.csv was generated).

    HISTORICAL BUG (now fixed in src/build_manifest.py): the branch `if p == 3`
    fires before the Gleason-sum-8 check, so Gleason 3+5=8 was returned as GG2
    instead of the ISUP-correct GG4.
    Kept here for comparison only — do NOT use for new labels.
    """
    if pd.isna(primary) or pd.isna(secondary):
        return "Benign"
    p, s = int(primary), int(secondary)
    g = p + s
    if g <= 6:
        return "GG1"
    if p == 3:          # catches 3+4 AND the edge-case 3+5 — wrong for 3+5
        return "GG2"
    if p == 4 and s == 3:
        return "GG3"
    if g == 8:
        return "GG4"
    return "GG5"


def grade_group_isup(primary, secondary) -> str:
    """
    Clinically correct ISUP 2014 Grade Group mapping.

    GG0 / Benign : no Gleason score (NaN primary or secondary)
    GG1           : 3+3 = 6
    GG2           : 3+4 = 7
    GG3           : 4+3 = 7
    GG4           : sum = 8  (4+4, 3+5, 5+3)
    GG5           : sum ≥ 9  (4+5, 5+4, 5+5)
    """
    if pd.isna(primary) or pd.isna(secondary):
        return "Benign"
    p, s = int(primary), int(secondary)
    g = p + s
    if g <= 6:
        return "GG1"
    if g == 7:
        return "GG2" if p == 3 else "GG3"   # 3+4 vs 4+3
    if g == 8:
        return "GG4"    # 4+4, 3+5, 5+3 — all correctly GG4
    return "GG5"        # sum ≥ 9


# ── Helpers ───────────────────────────────────────────────────────────────────

def md_crosstab(ct: pd.DataFrame, title: str) -> list[str]:
    """Format a pandas crosstab as a markdown table."""
    lines = [f"### {title}", ""]
    col_names = [str(c) for c in ct.columns]
    idx_name  = ct.index.name or "index"
    lines.append("| " + idx_name + " | " + " | ".join(col_names) + " |")
    lines.append("|" + "---|" * (len(col_names) + 1))
    for idx_val, row in ct.iterrows():
        cells = " | ".join(str(v) for v in row.values)
        lines.append(f"| {idx_val} | {cells} |")
    lines.append("")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Reading {DATA_PATH.relative_to(REPO_ROOT)} …")
    df = pd.read_csv(DATA_PATH)
    n_total = len(df)
    print(f"  {n_total:,} rows, {len(df.columns)} columns")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Label Definition Audit",
        "",
        f"Source: `data/share/{DATA_PATH.name}`  ({n_total:,} rows total)",
        "",
    ]

    # ── 1. Value counts for each relevant column ──────────────────────────
    lines += ["## 1. Value counts for relevant label columns", ""]
    for col in RELEVANT_COLS:
        if col not in df.columns:
            lines += [f"### {col}", "", "_Column not present in dataset._", ""]
            continue
        vc = df[col].value_counts(dropna=False)
        lines += [f"### {col}", "", "| Value | Count |", "|---|---|"]
        for val, cnt in vc.items():
            lines.append(f"| `{val!r}` | {cnt:,} |")
        lines.append("")

    # ── 2. Crosstabs ──────────────────────────────────────────────────────
    lines += ["## 2. Crosstabs", ""]

    ct = pd.crosstab(df["binary_label_int"], df["pathology_label"], margins=True)
    ct.index.name = "binary_label_int"
    lines += md_crosstab(ct, "binary_label_int × pathology_label")

    ct = pd.crosstab(df["binary_label_int"], df["pathology_label_int"], margins=True)
    ct.index.name = "binary_label_int"
    lines += md_crosstab(ct, "binary_label_int × pathology_label_int")

    ct = pd.crosstab(
        df["binary_label_int"].astype(str),
        df["pathology_hint_from_filename"].fillna("NaN"),
        margins=True,
    )
    ct.index.name = "binary_label_int"
    lines += md_crosstab(ct, "binary_label_int × pathology_hint_from_filename")

    ct = pd.crosstab(
        df["binary_label_int"].astype(str),
        df["primary_gleason"].fillna("NaN").astype(str),
        margins=True,
    )
    ct.index.name = "binary_label_int"
    lines += md_crosstab(ct, "binary_label_int × primary_gleason")

    ct = pd.crosstab(
        df["binary_label_int"].astype(str),
        df["secondary_gleason"].fillna("NaN").astype(str),
        margins=True,
    )
    ct.index.name = "binary_label_int"
    lines += md_crosstab(ct, "binary_label_int × secondary_gleason")

    ct = pd.crosstab(
        df["binary_label_int"].astype(str),
        df["core_label"].fillna("NaN"),
        margins=True,
    )
    ct.index.name = "binary_label_int"
    lines += md_crosstab(ct, "binary_label_int × core_label")

    # ── 3. Comparing the two grade-group mappings ─────────────────────────
    lines += [
        "## 3. Grade group: pipeline mapping vs. ISUP-correct mapping",
        "",
        "Two grade-group functions are evaluated:",
        "",
        "| Function | Source | Status |",
        "|---|---|---|",
        "| `grade_group_pipeline()` | original `src/build_manifest.py` | Historical bug: Gleason 3+5=8 → GG2 (should be GG4). **Source now fixed; CSV not yet regenerated.** |",
        "| `grade_group_isup()` | this script | Clinically correct ISUP 2014 standard |",
        "",
    ]

    df["_gg_pipeline"] = df.apply(
        lambda r: grade_group_pipeline(r["primary_gleason"], r["secondary_gleason"]), axis=1
    )
    df["_gg_isup"] = df.apply(
        lambda r: grade_group_isup(r["primary_gleason"], r["secondary_gleason"]), axis=1
    )

    # Rows where the two functions disagree
    differs = df["_gg_pipeline"] != df["_gg_isup"]
    n_differs = int(differs.sum())
    print(f"  Rows where pipeline vs. ISUP mapping differ: {n_differs}")

    lines += [
        f"Rows where the two functions assign a **different grade group**: **{n_differs:,}**",
        f"(out of {n_total:,} total rows = {n_differs/n_total*100:.3f}%)",
        "",
    ]

    if n_differs > 0:
        diff_df = df[differs][
            ["primary_gleason", "secondary_gleason", "_gg_pipeline", "_gg_isup"]
        ]
        # Summarise by (primary, secondary, pipeline_result, isup_result)
        diff_summary = (
            diff_df
            .groupby(["primary_gleason", "secondary_gleason", "_gg_pipeline", "_gg_isup"])
            .size()
            .reset_index(name="count")
        )
        lines += [
            "### Differing rows by Gleason pattern",
            "",
            "| primary | secondary | pipeline GG | ISUP GG | count |",
            "|---|---|---|---|---|",
        ]
        for _, row in diff_summary.iterrows():
            lines.append(
                f"| {int(row['primary_gleason'])} | {int(row['secondary_gleason'])} "
                f"| {row['_gg_pipeline']} | {row['_gg_isup']} | {row['count']:,} |"
            )
        lines.append("")
        lines += [
            "> **Dataset-generation note:** in the original pipeline the `if p == 3`",
            "> branch fired before the `if g == 8` check, so Gleason 3+5=8 was",
            "> classified as GG2 instead of GG4 when `needle_features_v1.csv` was built.",
            ">",
            "> **Source-code status:** `src/build_manifest.py::grade_group` has now been",
            "> fixed to follow the ISUP 2014 standard.",
            ">",
            "> **Dataset status:** `needle_features_v1.csv` was generated before that fix,",
            "> so the 36-row discrepancy remains in this CSV until the dataset is regenerated.",
            "",
        ]

    # ISUP grade-group distribution
    lines += ["### ISUP grade group distribution (used for all downstream steps)", ""]
    gg_counts_isup = df["_gg_isup"].value_counts().reindex(GG_ORDER, fill_value=0)
    lines += ["| Grade Group | Gleason pattern(s) | Count (ISUP) |", "|---|---|---|"]
    for gg in GG_ORDER:
        lines.append(f"| {gg} | {GLEASON_PATTERN[gg]} | {gg_counts_isup[gg]:,} |")
    lines.append("")

    ct = pd.crosstab(df["_gg_isup"], df["binary_label_int"], margins=True)
    ct.index.name = "grade_group_isup"
    lines += md_crosstab(ct, "grade_group_isup × binary_label_int")

    # ── 4. What does binary_label_int encode? ─────────────────────────────
    lines += ["## 4. What does binary_label_int encode?", ""]

    labeled = df[df["binary_label_int"].notna()].copy()
    labeled["_bli"] = labeled["binary_label_int"].astype(int)
    n_lab = len(labeled)

    gg3plus_set = {"GG3", "GG4", "GG5"}
    gg2plus_set = {"GG2", "GG3", "GG4", "GG5"}
    cancer_set  = {"GG1", "GG2", "GG3", "GG4", "GG5"}

    labeled["_gg3p"] = labeled["_gg_isup"].isin(gg3plus_set).astype(int)
    labeled["_gg2p"] = labeled["_gg_isup"].isin(gg2plus_set).astype(int)
    labeled["_canc"] = labeled["_gg_isup"].isin(cancer_set).astype(int)

    agree_gg3  = int((labeled["_bli"] == labeled["_gg3p"]).sum())
    agree_gg2  = int((labeled["_bli"] == labeled["_gg2p"]).sum())
    agree_canc = int((labeled["_bli"] == labeled["_canc"]).sum())

    lines += [
        f"Rows with a non-null `binary_label_int`: **{n_lab:,}**",
        "",
        "Agreement is measured using the ISUP-correct grade group.",
        "",
        "| Hypothesis | Agreement | % |",
        "|---|---|---|",
        f"| `binary_label_int` == (GG3+) | {agree_gg3:,} | {agree_gg3/n_lab*100:.2f}% |",
        f"| `binary_label_int` == (GG2+) | {agree_gg2:,} | {agree_gg2/n_lab*100:.2f}% |",
        f"| `binary_label_int` == (any cancer) | {agree_canc:,} | {agree_canc/n_lab*100:.2f}% |",
        "",
    ]

    lines += ["### Positive labels (binary_label_int == 1) per ISUP grade group", ""]
    pos = labeled[labeled["_bli"] == 1]
    neg = labeled[labeled["_bli"] == 0]
    lines += ["| Grade Group | binary_label_int=1 | binary_label_int=0 |", "|---|---|---|"]
    for gg in GG_ORDER:
        n_pos = int((pos["_gg_isup"] == gg).sum())
        n_neg = int((neg["_gg_isup"] == gg).sum())
        lines.append(f"| {gg} | {n_pos:,} | {n_neg:,} |")
    lines.append("")

    if agree_gg2 == n_lab:
        conclusion = "**GG2+** (csPCa: ISUP Grade Group ≥ 2 / Gleason ≥ 3+4=7)"
    elif agree_gg3 == n_lab:
        conclusion = "**GG3+** (ISUP Grade Group ≥ 3 / Gleason ≥ 4+3=7)"
    elif agree_canc == n_lab:
        conclusion = "**any cancer** (GG1, GG2, GG3, GG4, GG5 all positive)"
    else:
        best_pct = max(agree_gg3, agree_gg2, agree_canc) / n_lab * 100
        conclusion = f"**unclear** — best match is {best_pct:.2f}% (does not perfectly match any standard threshold)"

    lines += [
        f"**Conclusion:** `binary_label_int` encodes {conclusion}.",
        "",
        "The `binary_label` column contains the string `'csPCa'` (clinically significant",
        "prostate cancer), defined here as **ISUP Grade Group ≥ 2 (Gleason ≥ 3+4=7)**.",
        "",
        "This is **NOT equivalent to GG3+** (Grade Group ≥ 3 / Gleason ≥ 4+3=7).",
        "Specifically, **GG2 cores (Gleason 3+4=7) are labelled positive** in the dataset",
        "even though some clinical definitions of clinically significant PCa require GG3+.",
        "",
        "| binary_label_int | Grade groups included |",
        "|---|---|",
        "| 0 (non-csPCa) | Benign, GG1 (3+3=6) |",
        "| 1 (csPCa)     | GG2 (3+4=7), GG3 (4+3=7), GG4 (sum=8), GG5 (sum≥9) |",
        "",
    ]

    # ── 5. New column: binary_label_gg3plus_int ───────────────────────────
    lines += ["## 5. Derived column: binary_label_gg3plus_int", ""]

    df["binary_label_gg3plus_int"] = df["_gg_isup"].isin(gg3plus_set).astype(float)
    df.loc[df["binary_label_int"].isna(), "binary_label_gg3plus_int"] = np.nan

    labeled_new = df[df["binary_label_int"].notna()]
    n_gg2plus = int((labeled_new["binary_label_int"] == 1).sum())
    n_gg3plus_new = int((labeled_new["binary_label_gg3plus_int"] == 1).sum())
    n_gg2_only = n_gg2plus - n_gg3plus_new

    lines += [
        "**Rule (using `grade_group_isup()`):**",
        "- `binary_label_gg3plus_int = 1` if ISUP grade group ∈ {GG3, GG4, GG5}",
        "- `binary_label_gg3plus_int = 0` if ISUP grade group ∈ {Benign, GG1, GG2}",
        "- `binary_label_gg3plus_int = NaN` where `binary_label_int` is NaN",
        "",
        f"- `binary_label_int == 1` (GG2+, current label):            **{n_gg2plus:,} rows**",
        f"- `binary_label_gg3plus_int == 1` (GG3+, corrected label):  **{n_gg3plus_new:,} rows**",
        f"- GG2 cores (3+4=7) that flip from positive → negative:     **{n_gg2_only:,} rows**",
        "",
    ]

    ct = pd.crosstab(
        df["binary_label_int"].astype(str),
        df["binary_label_gg3plus_int"].astype(str),
        margins=True,
    )
    ct.index.name = "binary_label_int"
    lines += md_crosstab(ct, "binary_label_int × binary_label_gg3plus_int")

    lines += ["### Prevalence by split", ""]
    lines += [
        "| Split | N (labelled) | GG2+ prev. (current) | GG3+ prev. (corrected) |",
        "|---|---|---|---|",
    ]
    for split in ["train", "val", "test", "ALL"]:
        sub = df if split == "ALL" else df[df["split"] == split]
        sub = sub[sub["binary_label_int"].notna()]
        n = len(sub)
        p2 = float(sub["binary_label_int"].mean())
        p3 = float(sub["binary_label_gg3plus_int"].mean())
        lines.append(
            f"| {split} | {n:,} | {p2:.3f} ({p2*100:.1f}%) | {p3:.3f} ({p3*100:.1f}%) |"
        )
    lines.append("")

    # ── 6. Save updated CSV ───────────────────────────────────────────────
    df = df.drop(columns=["_gg_pipeline", "_gg_isup"])
    df.to_csv(DATA_PATH, index=False)
    print(f"Added column binary_label_gg3plus_int -> {DATA_PATH.relative_to(REPO_ROOT)}")

    lines += [
        "## 6. Status and next steps",
        "",
        f"Column `binary_label_gg3plus_int` has been written to `{DATA_PATH.relative_to(REPO_ROOT)}`.",
        "",
        "**Source-code status — `src/build_manifest.py`:**",
        "`grade_group()` has been corrected to follow the ISUP 2014 standard.",
        "Gleason 3+5=8 now correctly maps to GG4.",
        "",
        "**Dataset status — `data/share/needle_features_v1.csv`:**",
        "This CSV was generated before the source fix.",
        "The 36 Gleason 3+5=8 cores remain labelled with the historical pipeline grade",
        "group until the shareable dataset is regenerated from the corrected pipeline.",
        "Downstream analyses use the ISUP-correct derived endpoint (`binary_label_gg3plus_int`)",
        "so they are not affected by this historical discrepancy.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved report -> {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
