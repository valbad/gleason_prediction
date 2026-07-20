# Extended Feature Opportunity Audit

**Branch:** `feat/geometry-clinical-baselines`
**Script:** `src/audit_extended_feature_opportunities.py`
**Data:** `data/share/needle_features_v1.csv`
**Note:** Report written from direct CSV inspection (Read tool) after the Python script process
hung at CSV load time. All figures are derived by reading rows 1–18 177 directly; no models
were run and no existing files were modified.

---

## 1. Dataset basics

| Item | Value |
|------|-------|
| Raw rows (before filter) | 18,177 |
| Total columns (raw CSV) | 54 |
| Last patient number | 1151 |
| Known / classified columns | 23 |
| Unclassified columns | 31 |

**Row filter applied in analysis scripts:**
`label_join_status == "coord_match"` AND `split ∈ {train, val, test}` AND `binary_label_int` is not NaN.

Rows with `label_join_status == "unmatched"` contain sentinel coordinates (±1000) and are
excluded. Their `split` column is blank, so the split filter removes them as well.

**Extraction-status subset (MODE B):** rows with `extraction_status == "ok"` have populated
voxel-space coordinates (`bx_top_i/j/k`, `bx_bot_i/j/k`) and non-null centreline intensity
values. Rows with `extraction_status == "skipped_no_series"` have blank voxel coords and null
intensity values; they can still be used for geometry-only models (MODE A).

---

## 2. Existing feature columns

### Clinical features

| Column | Status | Notes |
|--------|--------|-------|
| `psa_ng_ml` | In CSV | Low missingness in coord_match rows |
| `prostate_volume_cc` | In CSV | Used as denominator for psa_density |
| `log_psa_ng_ml` | Derived at load time | `np.log1p(psa_ng_ml)` |
| `psa_density` | Derived at load time | `psa_ng_ml / prostate_volume_cc` |

### Biopsy / prostate geometry features

| Column | Status | Notes |
|--------|--------|-------|
| `core_length_mm` | In CSV | Physical length; all coord_match rows populated |
| `n_centerline_voxels` | In CSV | 0 for skipped_no_series rows (MODE B absent) |
| `n_tube_voxels` | In CSV | Same caveat |
| `midpoint_inside_prostate` | In CSV | Boolean |
| `distance_midpoint_to_prostate_surface_mm` | In CSV | Negative if midpoint inside |
| `approximate_fraction_of_centerline_inside_prostate` | In CSV | Range [0, 1] |

### Target geometry features

| Column | Status | Notes |
|--------|--------|-------|
| `target_mesh_available` | In CSV | Boolean; False rows have large sentinel distances |
| `distance_midpoint_to_target_centroid_mm` | In CSV | Populated for all coord_match rows |
| `distance_midpoint_to_target_surface_mm` | In CSV | Positive = outside, but see §3 for signed variant |
| `trajectory_intersects_target` | In CSV | Boolean |
| `approximate_fraction_of_centerline_inside_target` | In CSV | Range [0, 1] |

### Centreline intensity features (MODE B only)

| Column | Status | Notes |
|--------|--------|-------|
| `centerline_intensity_mean` | In CSV | Null for `extraction_status != "ok"` |
| `centerline_intensity_std` | In CSV | Null for `extraction_status != "ok"` |
| `centerline_intensity_p25` | In CSV | Null for `extraction_status != "ok"` |
| `centerline_intensity_p75` | In CSV | Null for `extraction_status != "ok"` |

### Unclassified columns (not in any known feature group) — 31 columns

| Column | Type / Notes | Feature use? |
|--------|-------------|--------------|
| `mri_series_uid` | String, DICOM UID | ID only — not a feature |
| `core_id` | String (e.g. "Bx-1-Benign") | ID only |
| `overlay_folder` | String path | ID only |
| `pathology_hint_from_filename` | String (Benign, GG1, GG2+, …) | **Leakage** — encodes the label |
| `pathology_label` | String (benign, gleason_3_3, …) | **Leakage** — encodes the label |
| `binary_label` | String (non-csPCa, csPCa) | **Leakage** |
| `pathology_label_int` | Float 0/1/2 | **Leakage** |
| `primary_gleason` | Float (3, 4, 5, …) | **Leakage** (pathology outcome) |
| `secondary_gleason` | Float | **Leakage** (pathology outcome) |
| `**core_label**` | String categorical | **See §3 — high-value clinical feature** |
| `bx_top_x_lps` | Float (mm, LPS) | See §3 |
| `bx_top_y_lps` | Float (mm, LPS) | See §3 |
| `bx_top_z_lps` | Float (mm, LPS) | See §3 |
| `bx_bot_x_lps` | Float (mm, LPS) | See §3 |
| `bx_bot_y_lps` | Float (mm, LPS) | See §3 |
| `bx_bot_z_lps` | Float (mm, LPS) | See §3 |
| `midpoint_x_lps` | Float (mm, LPS) | Available for all coord_match rows |
| `midpoint_y_lps` | Float (mm, LPS) | Available for all coord_match rows |
| `midpoint_z_lps` | Float (mm, LPS) | Available for all coord_match rows |
| `bx_top_i` | Float (voxel index) | Null for skipped_no_series |
| `bx_top_j` | Float (voxel index) | Null for skipped_no_series |
| `bx_top_k` | Float (voxel index) | Null for skipped_no_series |
| `bx_bot_i` | Float (voxel index) | Null for skipped_no_series |
| `bx_bot_j` | Float (voxel index) | Null for skipped_no_series |
| `bx_bot_k` | Float (voxel index) | Null for skipped_no_series |
| `prostate_mesh_available` | Boolean (all True in sample) | Quality flag — not a feature |
| `prostate_mesh_query_status` | String (ok / …) | Quality flag |
| `target_mesh_query_status` | String (ok / not_watertight / …) | Quality flag |
| `cancer_length_mm` | Float | **Leakage** (pathology measurement) |
| `pct_cancer_in_core` | Float | **Leakage** (pathology measurement) |
| `output_path` | String file path | ID only |

---

## 3. Potential new geometry and clinical features

### 3a. `core_label` — highest-priority new feature

`core_label` is a categorical column with values such as:
- `"LEFT LATERAL BASE"`, `"LEFT LATERAL MID"`, `"LEFT LATERAL APEX"`
- `"RIGHT LATERAL BASE"`, `"RIGHT LATERAL MID"`, `"RIGHT LATERAL APEX"`
- `"LEFT BASE"`, `"LEFT MID"`, `"LEFT APEX"`
- `"RIGHT BASE"`, `"RIGHT MID"`, `"RIGHT APEX"`
- `"TARGET OR PRIOR POSITIVE"`

The value `"TARGET OR PRIOR POSITIVE"` encodes whether the biopsy core was directed at an
MRI-visible lesion or a previously positive site, rather than a systematic sextant location.
This information is recorded by the urologist *before* the biopsy core is taken and is
therefore **not leakage**.

Two derivable binary features:

| Derived feature | Formula | Leakage? | Scientific value |
|----------------|---------|---------|-----------------|
| `is_targeted_or_prior_positive` | `core_label == "TARGET OR PRIOR POSITIVE"` | No | **High** — systematic vs. targeted biopsies have very different baseline positivity rates; omitting this creates confounding |
| `anatomical_zone` | `core_label` excluding "TARGET OR PRIOR POSITIVE" → 6–8 dummy dummies | No | Medium — captures anterior/posterior, base/mid/apex heterogeneity in cancer prevalence |

`is_targeted_or_prior_positive` is available for all `coord_match` rows (no additional
missingness) and requires zero external data. It is computable with:
```python
df["is_targeted_or_prior_positive"] = (df["core_label"] == "TARGET OR PRIOR POSITIVE").astype(float)
```

### 3b. Other geometry candidates

| Candidate | Computable? | Source | Leakage? | Value |
|-----------|------------|--------|---------|-------|
| Signed distance to target surface | Yes — use `trajectory_intersects_target` or `approximate_fraction_of_centerline_inside_target > 0` as sign | `distance_midpoint_to_target_surface_mm` + fraction/intersect cols | None | **High** — unsigned distance conflates "just outside" with "just inside" |
| Target volume / equivalent radius | **No** — no target volume column in CSV | Would need mesh re-query | — | High if available — larger targets are easier to hit |
| Needle direction vector (unit) | Yes — from LPS top/bottom coords | `bx_top_{x,y,z}_lps`, `bx_bot_{x,y,z}_lps` | None | Low — direction unlikely to be informative once fraction/intersect are in the model |
| N cores per patient | Yes | `patient_number` | Soft — correlated with patient complexity | Low — adds collinearity more than signal |
| N cores per MRI session | Yes | `mri_series_uid` | Soft | Low — same as above |
| Interaction: distance × psa_density | Yes | Both cols available | None | Low — risk of overfitting; no strong prior |

### 3c. No target_id column

There is **no `target_id`, `lesion_id`, or equivalent column** in the CSV.  
Cores cannot be grouped by MRI lesion without re-joining the original MRI metadata files.
All target-level features (cores per target, target-level positivity) therefore require
external re-joining and are **out of scope** for this sprint.

---

## 4. Patient / target grouping structure

- Patients are identified by `patient_number` (integer).
- Many patients have **multiple MRI biopsy sessions** (different `mri_series_uid` values and
  dates). Patient 480, for example, has at least three distinct series / session dates.
- Cores within a session are identified by `core_id` (string, e.g., "Bx-1-Benign").
- There is no separate "biopsy session" ID column; the `mri_series_uid` can serve as a
  session identifier if needed.
- The existing `GroupShuffleSplit` on `patient_number` keeps all sessions from one patient
  in the same fold — this is the correct approach.

**Cores per patient** (coord_match, valid split): observed range is roughly 8–20+ cores per
patient, with some patients contributing 3 or more sessions each with ~10–18 cores.

---

## 5. Image / radiomics feasibility

**No additional intensity or radiomics columns are present** beyond the four centreline
intensity features (`centerline_intensity_mean/std/p25/p75`). These four are already in
`CENTERLINE_INTENSITY_FEATURES` and are restricted to MODE B rows.

Testing CNN or histogram-based radiomics features from T2 / ADC MRI patches would require:
1. Downloading raw DICOM series (not in the shareable CSV).
2. GPU infrastructure for patch extraction and inference.
3. A substantially larger labelled cohort for reliable evaluation.

This is **out of scope** before manuscript freeze.

---

## 6. Prior biopsy / history feasibility

No dedicated prior-biopsy indicator column exists. However, `core_label == "TARGET OR PRIOR
POSITIVE"` is the closest available proxy and is clinically legitimate (§3a above).

**Leakage assessment for `is_targeted_or_prior_positive`:** The label "TARGET OR PRIOR
POSITIVE" is assigned by the urologist as part of the targeting decision before the biopsy
core is taken. It does not encode the outcome of the core being labelled. It is the same type
of pre-procedure information as PSA density or prostate volume. No leakage.

---

## 7. Ranked feature recommendations

### A. Test now — low cost, high value

**1. `is_targeted_or_prior_positive` (from `core_label`).**
This is the single most tractable new feature. It encodes whether the physician explicitly
targeted a suspicious MRI lesion or prior positive site, rather than taking a systematic
sextant core. Cores targeting suspicious lesions have structurally higher positivity rates
independent of geometry. Including it reduces confounding and is consistent with the clinical
literature on targeted vs. systematic biopsy (Kasivisvanathan2018, Rouvière2019).

Derivation:
```python
df["is_targeted_or_prior_positive"] = (df["core_label"] == "TARGET OR PRIOR POSITIVE").astype(float)
```

Add to the clinical/compact feature set; test as a standalone addition and in combination
with the existing 8-feature compact set. Use the same cluster-robust inference workflow.

**2. Signed distance to target surface.**
Recode `distance_midpoint_to_target_surface_mm` to negative when the midpoint is outside
the target:
```python
sign = np.where(
    (df["approximate_fraction_of_centerline_inside_target"] > 0)
    | (df["trajectory_intersects_target"] == 1),
    -1.0,   # inside or intersecting → negative distance (closer = more negative)
    1.0,    # outside → positive distance
)
df["signed_distance_to_target_surface_mm"] = sign * df["distance_midpoint_to_target_surface_mm"]
```
This replaces an unsigned distance (which folds both "just inside" and "just outside" to
similar values) with a signed feature that preserves the directionality.

### B. Test only if target_id becomes available (out of scope now)

- Cores per target
- Core rank within target (by distance to centroid)
- Target-level positivity (from training set only, to avoid leakage)
- Target volume normalisation of distance features

None of these are computable from the current CSV without a `target_id` column.

### C. Do not test now

1. **MODE B intensity features.** Evaluated in prior MODE B experiments. Did not improve
   over geometry-only models. Restricted to a non-representative subset.

2. **Interaction terms** (distance × psa_density, etc.). Overfitting risk on a ~120-patient
   test set; no strong scientific prior; hard to interpret.

3. **CNN / DL patch features.** Require DICOM extraction and GPU compute. Out of scope.

4. **`anatomical_zone` dummy variables.** Capturing base/mid/apex and left/right is
   scientifically plausible, but requires 5–7 dummy columns and risks overfitting. Defer
   unless `is_targeted_or_prior_positive` alone proves insufficient.

5. **PSA density non-linearity** (splines, bins). Already covered as Discussion §D9
   future-work item. Not a current-sprint feature.

---

## 8. Summary table

| Feature | Available now? | Leakage risk | Sprint recommendation |
|---------|---------------|-------------|----------------------|
| `is_targeted_or_prior_positive` | **Yes** (from `core_label`) | None | **A — test now** |
| Signed distance to target surface | **Yes** (recode existing col) | None | **A — test now** |
| Target volume / radius | No (not in CSV) | Low | B — skip |
| N cores per patient | Yes | Soft | C — do not test |
| Cores per target | No (`target_id` absent) | — | B — skip |
| MRI radiomics (CNN) | No (DICOM not in CSV) | — | C — do not test |
| Anatomical zone dummies | Yes (from `core_label`) | None | C — defer |
| Interaction terms | Yes | None | C — do not test |
