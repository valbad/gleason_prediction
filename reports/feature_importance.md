# Feature Importance — Permutation Analysis

Feature set: `all_geometry_no_availability_plus_clinical`  
(biopsy geometry ×6 + target geometry no availability ×4 + clinical ×4 = 14 features)  
Permutation repeats: 50 · random seed: 42

**Importance** = mean decrease in test metric when a single feature is randomly shuffled.  
Positive values indicate a useful feature; near-zero or negative values suggest no reliable contribution.  
Each section shows the same features sorted first by ROC-AUC drop, then by PR-AUC drop.

> **Correlation caveat:** permutation importance should be interpreted cautiously when features are correlated; importance may be shared or masked among correlated predictors. The four clinical features (psa_ng_ml, log_psa_ng_ml, prostate_volume_cc, psa_density) are correlated — any single feature may appear unimportant while the group as a whole contributes meaningful signal. See the per-section interpretation for guidance.

---

## GG2+ / csPCa — logistic_regression

Baseline test ROC-AUC: **0.753**  ·  Baseline test PR-AUC: **0.319**

### Sorted by ROC-AUC importance

| # | Feature | ROC-AUC drop mean | ± std | PR-AUC drop mean | ± std |
|---|---|---|---|---|---|
| 1 | `distance_midpoint_to_target_surface_mm` | 0.0888 | ±0.0123 | 0.0679 | ±0.0144 |
| 2 | `log_psa_ng_ml` | 0.0582 | ±0.0087 | 0.1189 | ±0.0104 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0157 | ±0.0045 | 0.0397 | ±0.0120 |
| 4 | `prostate_volume_cc` | 0.0143 | ±0.0039 | 0.0140 | ±0.0059 |
| 5 | `n_tube_voxels` | 0.0122 | ±0.0020 | 0.0134 | ±0.0053 |
| 6 | `distance_midpoint_to_target_centroid_mm` | 0.0072 | ±0.0021 | 0.0062 | ±0.0044 |
| 7 | `core_length_mm` | 0.0060 | ±0.0022 | 0.0047 | ±0.0049 |
| 8 | `distance_midpoint_to_prostate_surface_mm` | 0.0005 | ±0.0018 | -0.0010 | ±0.0034 |
| 9 | `approximate_fraction_of_centerline_inside_prostate` | 0.0002 | ±0.0001 | 0.0000 | ±0.0008 |
| 10 | `midpoint_inside_prostate` | 0.0001 | ±0.0001 | 0.0001 | ±0.0004 |
| 11 | `psa_ng_ml` | 0.0001 | ±0.0000 | -0.0021 | ±0.0004 |
| 12 | `psa_density` | -0.0002 | ±0.0003 | -0.0010 | ±0.0012 |
| 13 | `trajectory_intersects_target` | -0.0005 | ±0.0005 | -0.0004 | ±0.0016 |
| 14 | `n_centerline_voxels` | -0.0041 | ±0.0029 | -0.0043 | ±0.0054 |

### Sorted by PR-AUC importance

| # | Feature | PR-AUC drop mean | ± std | ROC-AUC drop mean | ± std |
|---|---|---|---|---|---|
| 1 | `log_psa_ng_ml` | 0.1189 | ±0.0104 | 0.0582 | ±0.0087 |
| 2 | `distance_midpoint_to_target_surface_mm` | 0.0679 | ±0.0144 | 0.0888 | ±0.0123 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0397 | ±0.0120 | 0.0157 | ±0.0045 |
| 4 | `prostate_volume_cc` | 0.0140 | ±0.0059 | 0.0143 | ±0.0039 |
| 5 | `n_tube_voxels` | 0.0134 | ±0.0053 | 0.0122 | ±0.0020 |
| 6 | `distance_midpoint_to_target_centroid_mm` | 0.0062 | ±0.0044 | 0.0072 | ±0.0021 |
| 7 | `core_length_mm` | 0.0047 | ±0.0049 | 0.0060 | ±0.0022 |
| 8 | `midpoint_inside_prostate` | 0.0001 | ±0.0004 | 0.0001 | ±0.0001 |
| 9 | `approximate_fraction_of_centerline_inside_prostate` | 0.0000 | ±0.0008 | 0.0002 | ±0.0001 |
| 10 | `trajectory_intersects_target` | -0.0004 | ±0.0016 | -0.0005 | ±0.0005 |
| 11 | `psa_density` | -0.0010 | ±0.0012 | -0.0002 | ±0.0003 |
| 12 | `distance_midpoint_to_prostate_surface_mm` | -0.0010 | ±0.0034 | 0.0005 | ±0.0018 |
| 13 | `psa_ng_ml` | -0.0021 | ±0.0004 | 0.0001 | ±0.0000 |
| 14 | `n_centerline_voxels` | -0.0043 | ±0.0054 | -0.0041 | ±0.0029 |

### Interpretation

- **Target geometry dominates**: target-trajectory features contribute ~55% of the total positive ROC-AUC importance, consistent with proximity to the MRI lesion being the primary predictive signal.
- **PSA-related variables contribute meaningfully** (`log_psa_ng_ml` ranked #2, ROC-AUC drop = 0.0582). Clinical signal is present, though importance may be distributed across correlated features (psa_ng_ml, log_psa_ng_ml, psa_density). Low individual importance for any one of these should not be interpreted as absence of clinical value.
- **Prostate/biopsy geometry adds signal** (best feature `n_tube_voxels`, ROC-AUC drop = 0.0122), indicating that needle placement relative to the prostate adds information beyond target proximity.
- **Stability: good** — for all top-5 features, std < mean importance, indicating the importance rankings are stable across permutation repeats.

---

## GG2+ / csPCa — xgboost

Baseline test ROC-AUC: **0.758**  ·  Baseline test PR-AUC: **0.327**

### Sorted by ROC-AUC importance

| # | Feature | ROC-AUC drop mean | ± std | PR-AUC drop mean | ± std |
|---|---|---|---|---|---|
| 1 | `distance_midpoint_to_target_surface_mm` | 0.0887 | ±0.0104 | 0.0819 | ±0.0143 |
| 2 | `psa_density` | 0.0457 | ±0.0076 | 0.0940 | ±0.0144 |
| 3 | `distance_midpoint_to_target_centroid_mm` | 0.0137 | ±0.0040 | 0.0125 | ±0.0098 |
| 4 | `core_length_mm` | 0.0075 | ±0.0032 | 0.0047 | ±0.0037 |
| 5 | `approximate_fraction_of_centerline_inside_target` | 0.0051 | ±0.0028 | 0.0083 | ±0.0080 |
| 6 | `psa_ng_ml` | 0.0038 | ±0.0034 | 0.0239 | ±0.0094 |
| 7 | `distance_midpoint_to_prostate_surface_mm` | 0.0010 | ±0.0023 | 0.0028 | ±0.0031 |
| 8 | `approximate_fraction_of_centerline_inside_prostate` | 0.0001 | ±0.0003 | 0.0007 | ±0.0008 |
| 9 | `midpoint_inside_prostate` | 0.0000 | ±0.0000 | 0.0000 | ±0.0000 |
| 10 | `trajectory_intersects_target` | 0.0000 | ±0.0000 | 0.0000 | ±0.0000 |
| 11 | `log_psa_ng_ml` | 0.0000 | ±0.0000 | 0.0000 | ±0.0000 |
| 12 | `n_tube_voxels` | -0.0001 | ±0.0006 | -0.0005 | ±0.0015 |
| 13 | `prostate_volume_cc` | -0.0004 | ±0.0024 | 0.0077 | ±0.0055 |
| 14 | `n_centerline_voxels` | -0.0008 | ±0.0005 | -0.0003 | ±0.0015 |

### Sorted by PR-AUC importance

| # | Feature | PR-AUC drop mean | ± std | ROC-AUC drop mean | ± std |
|---|---|---|---|---|---|
| 1 | `psa_density` | 0.0940 | ±0.0144 | 0.0457 | ±0.0076 |
| 2 | `distance_midpoint_to_target_surface_mm` | 0.0819 | ±0.0143 | 0.0887 | ±0.0104 |
| 3 | `psa_ng_ml` | 0.0239 | ±0.0094 | 0.0038 | ±0.0034 |
| 4 | `distance_midpoint_to_target_centroid_mm` | 0.0125 | ±0.0098 | 0.0137 | ±0.0040 |
| 5 | `approximate_fraction_of_centerline_inside_target` | 0.0083 | ±0.0080 | 0.0051 | ±0.0028 |
| 6 | `prostate_volume_cc` | 0.0077 | ±0.0055 | -0.0004 | ±0.0024 |
| 7 | `core_length_mm` | 0.0047 | ±0.0037 | 0.0075 | ±0.0032 |
| 8 | `distance_midpoint_to_prostate_surface_mm` | 0.0028 | ±0.0031 | 0.0010 | ±0.0023 |
| 9 | `approximate_fraction_of_centerline_inside_prostate` | 0.0007 | ±0.0008 | 0.0001 | ±0.0003 |
| 10 | `midpoint_inside_prostate` | 0.0000 | ±0.0000 | 0.0000 | ±0.0000 |
| 11 | `trajectory_intersects_target` | 0.0000 | ±0.0000 | 0.0000 | ±0.0000 |
| 12 | `log_psa_ng_ml` | 0.0000 | ±0.0000 | 0.0000 | ±0.0000 |
| 13 | `n_centerline_voxels` | -0.0003 | ±0.0015 | -0.0008 | ±0.0005 |
| 14 | `n_tube_voxels` | -0.0005 | ±0.0015 | -0.0001 | ±0.0006 |

### Interpretation

- **Target geometry dominates**: target-trajectory features contribute ~65% of the total positive ROC-AUC importance, consistent with proximity to the MRI lesion being the primary predictive signal.
- **PSA-related variables contribute meaningfully** (`psa_density` ranked #2, ROC-AUC drop = 0.0457). Clinical signal is present, though importance may be distributed across correlated features (psa_ng_ml, log_psa_ng_ml, psa_density). Low individual importance for any one of these should not be interpreted as absence of clinical value.
- **Prostate/biopsy geometry adds signal** (best feature `core_length_mm`, ROC-AUC drop = 0.0075), indicating that needle placement relative to the prostate adds information beyond target proximity.
- **Stability: good** — for all top-5 features, std < mean importance, indicating the importance rankings are stable across permutation repeats.

---

## GG3+ / high-grade — logistic_regression

Baseline test ROC-AUC: **0.829**  ·  Baseline test PR-AUC: **0.265**

### Sorted by ROC-AUC importance

| # | Feature | ROC-AUC drop mean | ± std | PR-AUC drop mean | ± std |
|---|---|---|---|---|---|
| 1 | `distance_midpoint_to_target_surface_mm` | 0.1431 | ±0.0201 | 0.0999 | ±0.0196 |
| 2 | `log_psa_ng_ml` | 0.1062 | ±0.0198 | 0.1818 | ±0.0103 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0159 | ±0.0048 | 0.0444 | ±0.0144 |
| 4 | `prostate_volume_cc` | 0.0111 | ±0.0041 | -0.0161 | ±0.0098 |
| 5 | `core_length_mm` | 0.0082 | ±0.0042 | -0.0017 | ±0.0142 |
| 6 | `psa_ng_ml` | 0.0025 | ±0.0015 | -0.0085 | ±0.0036 |
| 7 | `distance_midpoint_to_prostate_surface_mm` | 0.0006 | ±0.0005 | 0.0023 | ±0.0026 |
| 8 | `midpoint_inside_prostate` | 0.0002 | ±0.0004 | 0.0006 | ±0.0013 |
| 9 | `n_centerline_voxels` | -0.0003 | ±0.0001 | 0.0013 | ±0.0026 |
| 10 | `approximate_fraction_of_centerline_inside_prostate` | -0.0003 | ±0.0005 | 0.0027 | ±0.0035 |
| 11 | `distance_midpoint_to_target_centroid_mm` | -0.0017 | ±0.0007 | 0.0023 | ±0.0032 |
| 12 | `psa_density` | -0.0024 | ±0.0023 | -0.0182 | ±0.0078 |
| 13 | `trajectory_intersects_target` | -0.0034 | ±0.0014 | -0.0014 | ±0.0056 |
| 14 | `n_tube_voxels` | -0.0050 | ±0.0015 | 0.0006 | ±0.0054 |

### Sorted by PR-AUC importance

| # | Feature | PR-AUC drop mean | ± std | ROC-AUC drop mean | ± std |
|---|---|---|---|---|---|
| 1 | `log_psa_ng_ml` | 0.1818 | ±0.0103 | 0.1062 | ±0.0198 |
| 2 | `distance_midpoint_to_target_surface_mm` | 0.0999 | ±0.0196 | 0.1431 | ±0.0201 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0444 | ±0.0144 | 0.0159 | ±0.0048 |
| 4 | `approximate_fraction_of_centerline_inside_prostate` | 0.0027 | ±0.0035 | -0.0003 | ±0.0005 |
| 5 | `distance_midpoint_to_prostate_surface_mm` | 0.0023 | ±0.0026 | 0.0006 | ±0.0005 |
| 6 | `distance_midpoint_to_target_centroid_mm` | 0.0023 | ±0.0032 | -0.0017 | ±0.0007 |
| 7 | `n_centerline_voxels` | 0.0013 | ±0.0026 | -0.0003 | ±0.0001 |
| 8 | `n_tube_voxels` | 0.0006 | ±0.0054 | -0.0050 | ±0.0015 |
| 9 | `midpoint_inside_prostate` | 0.0006 | ±0.0013 | 0.0002 | ±0.0004 |
| 10 | `trajectory_intersects_target` | -0.0014 | ±0.0056 | -0.0034 | ±0.0014 |
| 11 | `core_length_mm` | -0.0017 | ±0.0142 | 0.0082 | ±0.0042 |
| 12 | `psa_ng_ml` | -0.0085 | ±0.0036 | 0.0025 | ±0.0015 |
| 13 | `prostate_volume_cc` | -0.0161 | ±0.0098 | 0.0111 | ±0.0041 |
| 14 | `psa_density` | -0.0182 | ±0.0078 | -0.0024 | ±0.0023 |

### Interpretation

- **Target geometry dominates**: target-trajectory features contribute ~55% of the total positive ROC-AUC importance, consistent with proximity to the MRI lesion being the primary predictive signal.
- **PSA-related variables contribute meaningfully** (`log_psa_ng_ml` ranked #2, ROC-AUC drop = 0.1062). Clinical signal is present, though importance may be distributed across correlated features (psa_ng_ml, log_psa_ng_ml, psa_density). Low individual importance for any one of these should not be interpreted as absence of clinical value.
- **Prostate/biopsy geometry adds signal** (best feature `core_length_mm`, ROC-AUC drop = 0.0082), indicating that needle placement relative to the prostate adds information beyond target proximity.
- **Stability: good** — for all top-5 features, std < mean importance, indicating the importance rankings are stable across permutation repeats.

