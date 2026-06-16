# Label Definition Audit

Source: `data/share/needle_features_v1.csv`  (18,176 rows total)

## 1. Value counts for relevant label columns

### pathology_hint_from_filename

| Value | Count |
|---|---|
| `'Benign'` | 13,944 |
| `'GG1'` | 2,226 |
| `'GG2+'` | 2,006 |

### pathology_label

| Value | Count |
|---|---|
| `'benign'` | 12,990 |
| `'gleason_3_3'` | 2,115 |
| `'gleason_ge_3_4'` | 1,887 |
| `nan` | 1,184 |

### binary_label

| Value | Count |
|---|---|
| `'non-csPCa'` | 15,105 |
| `'csPCa'` | 1,887 |
| `nan` | 1,184 |

### pathology_label_int

| Value | Count |
|---|---|
| `0.0` | 12,990 |
| `1.0` | 2,115 |
| `2.0` | 1,887 |
| `nan` | 1,184 |

### binary_label_int

| Value | Count |
|---|---|
| `0.0` | 15,105 |
| `1.0` | 1,887 |
| `nan` | 1,184 |

### primary_gleason

| Value | Count |
|---|---|
| `nan` | 14,174 |
| `3.0` | 3,391 |
| `4.0` | 574 |
| `5.0` | 37 |

### secondary_gleason

| Value | Count |
|---|---|
| `nan` | 14,174 |
| `3.0` | 2,446 |
| `4.0` | 1,420 |
| `5.0` | 136 |

### core_label

| Value | Count |
|---|---|
| `'TARGET OR PRIOR POSITIVE'` | 7,649 |
| `nan` | 1,184 |
| `'LEFT APEX'` | 927 |
| `'RIGHT APEX'` | 899 |
| `'RIGHT BASE'` | 790 |
| `'LEFT MID'` | 774 |
| `'LEFT BASE'` | 769 |
| `'RIGHT MID'` | 767 |
| `'LEFT LATERAL BASE'` | 749 |
| `'RIGHT LATERAL BASE'` | 739 |
| `'RIGHT LATERAL MID'` | 739 |
| `'LEFT LATERAL APEX'` | 736 |
| `'RIGHT LATERAL APEX'` | 727 |
| `'LEFT LATERAL MID'` | 726 |
| `'EXTRA'` | 1 |

## 2. Crosstabs

### binary_label_int × pathology_label

| binary_label_int | benign | gleason_3_3 | gleason_ge_3_4 | All |
|---|---|---|---|---|
| 0.0 | 12990 | 2115 | 0 | 15105 |
| 1.0 | 0 | 0 | 1887 | 1887 |
| All | 12990 | 2115 | 1887 | 16992 |

### binary_label_int × pathology_label_int

| binary_label_int | 0.0 | 1.0 | 2.0 | All |
|---|---|---|---|---|
| 0.0 | 12990 | 2115 | 0 | 15105 |
| 1.0 | 0 | 0 | 1887 | 1887 |
| All | 12990 | 2115 | 1887 | 16992 |

### binary_label_int × pathology_hint_from_filename

| binary_label_int | Benign | GG1 | GG2+ | All |
|---|---|---|---|---|
| 0.0 | 12990 | 2114 | 1 | 15105 |
| 1.0 | 3 | 0 | 1884 | 1887 |
| All | 12993 | 2114 | 1885 | 16992 |

### binary_label_int × primary_gleason

| binary_label_int | 3.0 | 4.0 | 5.0 | NaN | All |
|---|---|---|---|---|---|
| 0.0 | 2115 | 0 | 0 | 12990 | 15105 |
| 1.0 | 1276 | 574 | 37 | 0 | 1887 |
| All | 3391 | 574 | 37 | 12990 | 16992 |

### binary_label_int × secondary_gleason

| binary_label_int | 3.0 | 4.0 | 5.0 | NaN | All |
|---|---|---|---|---|---|
| 0.0 | 2115 | 0 | 0 | 12990 | 15105 |
| 1.0 | 331 | 1420 | 136 | 0 | 1887 |
| All | 2446 | 1420 | 136 | 12990 | 16992 |

### binary_label_int × core_label

| binary_label_int | EXTRA | LEFT APEX | LEFT BASE | LEFT LATERAL APEX | LEFT LATERAL BASE | LEFT LATERAL MID | LEFT MID | RIGHT APEX | RIGHT BASE | RIGHT LATERAL APEX | RIGHT LATERAL BASE | RIGHT LATERAL MID | RIGHT MID | TARGET OR PRIOR POSITIVE | All |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1 | 884 | 723 | 699 | 714 | 679 | 723 | 865 | 757 | 695 | 704 | 703 | 722 | 6236 | 15105 |
| 1.0 | 0 | 43 | 46 | 37 | 35 | 47 | 51 | 34 | 33 | 32 | 35 | 36 | 45 | 1413 | 1887 |
| All | 1 | 927 | 769 | 736 | 749 | 726 | 774 | 899 | 790 | 727 | 739 | 739 | 767 | 7649 | 16992 |

## 3. Grade group: pipeline mapping vs. ISUP-correct mapping

Two grade-group functions are evaluated:

| Function | Source | Known issue |
|---|---|---|
| `grade_group_pipeline()` | `src/build_manifest.py` | Gleason 3+5=8 → GG2 (wrong; should be GG4) |
| `grade_group_isup()` | this script | Clinically correct ISUP 2014 standard |

Rows where the two functions assign a **different grade group**: **36**
(out of 18,176 total rows = 0.198%)

### Differing rows by Gleason pattern

| primary | secondary | pipeline GG | ISUP GG | count |
|---|---|---|---|---|
| 3 | 5 | GG2 | GG4 | 36 |

> **Note for `build_manifest.py`:** the `if p == 3` branch fires before
> the `if g == 8` check, so Gleason 3+5=8 is classified as GG2 instead
> of GG4. `build_manifest.py` is **not modified here** — the discrepancy
> is documented for a future fix.

### ISUP grade group distribution (used for all downstream steps)

| Grade Group | Gleason pattern(s) | Count (ISUP) |
|---|---|---|
| Benign | no cancer / no score | 14,174 |
| GG1 | 3+3=6 | 2,115 |
| GG2 | 3+4=7 | 1,240 |
| GG3 | 4+3=7 | 329 |
| GG4 | 4+4=8, 3+5=8, 5+3=8 | 186 |
| GG5 | 4+5=9, 5+4=9, 5+5=10 | 132 |

### grade_group_isup × binary_label_int

| grade_group_isup | 0.0 | 1.0 | All |
|---|---|---|---|
| Benign | 12990 | 0 | 12990 |
| GG1 | 2115 | 0 | 2115 |
| GG2 | 0 | 1240 | 1240 |
| GG3 | 0 | 329 | 329 |
| GG4 | 0 | 186 | 186 |
| GG5 | 0 | 132 | 132 |
| All | 15105 | 1887 | 16992 |

## 4. What does binary_label_int encode?

Rows with a non-null `binary_label_int`: **16,992**

Agreement is measured using the ISUP-correct grade group.

| Hypothesis | Agreement | % |
|---|---|---|
| `binary_label_int` == (GG3+) | 15,752 | 92.70% |
| `binary_label_int` == (GG2+) | 16,992 | 100.00% |
| `binary_label_int` == (any cancer) | 14,877 | 87.55% |

### Positive labels (binary_label_int == 1) per ISUP grade group

| Grade Group | binary_label_int=1 | binary_label_int=0 |
|---|---|---|
| Benign | 0 | 12,990 |
| GG1 | 0 | 2,115 |
| GG2 | 1,240 | 0 |
| GG3 | 329 | 0 |
| GG4 | 186 | 0 |
| GG5 | 132 | 0 |

**Conclusion:** `binary_label_int` encodes **GG2+** (csPCa: ISUP Grade Group ≥ 2 / Gleason ≥ 3+4=7).

The `binary_label` column contains the string `'csPCa'` (clinically significant
prostate cancer), defined here as **ISUP Grade Group ≥ 2 (Gleason ≥ 3+4=7)**.

This is **NOT equivalent to GG3+** (Grade Group ≥ 3 / Gleason ≥ 4+3=7).
Specifically, **GG2 cores (Gleason 3+4=7) are labelled positive** in the dataset
even though some clinical definitions of clinically significant PCa require GG3+.

| binary_label_int | Grade groups included |
|---|---|
| 0 (non-csPCa) | Benign, GG1 (3+3=6) |
| 1 (csPCa)     | GG2 (3+4=7), GG3 (4+3=7), GG4 (sum=8), GG5 (sum≥9) |

## 5. Derived column: binary_label_gg3plus_int

**Rule (using `grade_group_isup()`):**
- `binary_label_gg3plus_int = 1` if ISUP grade group ∈ {GG3, GG4, GG5}
- `binary_label_gg3plus_int = 0` if ISUP grade group ∈ {Benign, GG1, GG2}
- `binary_label_gg3plus_int = NaN` where `binary_label_int` is NaN

- `binary_label_int == 1` (GG2+, current label):            **1,887 rows**
- `binary_label_gg3plus_int == 1` (GG3+, corrected label):  **647 rows**
- GG2 cores (3+4=7) that flip from positive → negative:     **1,240 rows**

### binary_label_int × binary_label_gg3plus_int

| binary_label_int | 0.0 | 1.0 | All |
|---|---|---|---|
| 0.0 | 15105 | 0 | 15105 |
| 1.0 | 1240 | 647 | 1887 |
| All | 16345 | 647 | 16992 |

### Prevalence by split

| Split | N (labelled) | GG2+ prev. (current) | GG3+ prev. (corrected) |
|---|---|---|---|
| train | 11,827 | 0.115 (11.5%) | 0.040 (4.0%) |
| val | 2,748 | 0.099 (9.9%) | 0.029 (2.9%) |
| test | 2,417 | 0.107 (10.7%) | 0.038 (3.8%) |
| ALL | 16,992 | 0.111 (11.1%) | 0.038 (3.8%) |

## 6. Next steps

Column `binary_label_gg3plus_int` has been written to `data/share/needle_features_v1.csv`.

**Pending fix — `src/build_manifest.py`:**
The `grade_group()` function there misclassifies Gleason 3+5=8 as GG2.
It should be updated to use the ISUP-correct logic (`grade_group_isup` above).
This is left for a separate commit to avoid touching the pipeline mid-audit.

**Action required in `src/run_shareable_tabular_experiments.py`:**
Change `TARGET_COL = 'binary_label_int'` to `TARGET_COL = 'binary_label_gg3plus_int'`
to train and evaluate models on the GG3+ task.

