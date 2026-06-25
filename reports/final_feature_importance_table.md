# Feature Importance — Permutation Analysis (Summary)

Feature set: `all_geometry_no_availability_plus_clinical` (biopsy geometry ×6 + target geometry no availability ×4 + clinical ×4).  
**Importance** = mean decrease in test metric when a feature is permuted.  
Positive = feature helps; near-zero or negative = no reliable contribution.

> **Correlation caveat:** importance may be shared or masked among correlated predictors. The clinical features (psa_ng_ml, log_psa_ng_ml, prostate_volume_cc, psa_density) are correlated — low individual importance does not imply absence of clinical value.

---

## GG2+ / csPCa — logistic_regression

Baseline test ROC-AUC: **0.753**  ·  Baseline test PR-AUC: **0.319**

### Top 5 features by ROC-AUC importance

| # | Feature | ROC-AUC drop | ± std | PR-AUC drop | ± std |
|---|---|---|---|---|---|
| 1 | `distance_midpoint_to_target_surface_mm` | 0.0888 | ±0.0123 | 0.0679 | ±0.0144 |
| 2 | `log_psa_ng_ml` | 0.0582 | ±0.0087 | 0.1189 | ±0.0104 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0157 | ±0.0045 | 0.0397 | ±0.0120 |
| 4 | `prostate_volume_cc` | 0.0143 | ±0.0039 | 0.0140 | ±0.0059 |
| 5 | `n_tube_voxels` | 0.0122 | ±0.0020 | 0.0134 | ±0.0053 |

### Top 5 features by PR-AUC importance

| # | Feature | PR-AUC drop | ± std | ROC-AUC drop | ± std |
|---|---|---|---|---|---|
| 1 | `log_psa_ng_ml` | 0.1189 | ±0.0104 | 0.0582 | ±0.0087 |
| 2 | `distance_midpoint_to_target_surface_mm` | 0.0679 | ±0.0144 | 0.0888 | ±0.0123 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0397 | ±0.0120 | 0.0157 | ±0.0045 |
| 4 | `prostate_volume_cc` | 0.0140 | ±0.0059 | 0.0143 | ±0.0039 |
| 5 | `n_tube_voxels` | 0.0134 | ±0.0053 | 0.0122 | ±0.0020 |

**Interpretation:** Target geometry dominates (~55% of positive ROC-AUC importance). PSA-related variables contribute meaningfully (best PSA feature ranked #2); importance may be distributed across correlated clinical features. Prostate/biopsy geometry adds signal (max drop = 0.0122). Results stable (std < mean for all top-5 features).

---

## GG2+ / csPCa — xgboost

Baseline test ROC-AUC: **0.758**  ·  Baseline test PR-AUC: **0.327**

### Top 5 features by ROC-AUC importance

| # | Feature | ROC-AUC drop | ± std | PR-AUC drop | ± std |
|---|---|---|---|---|---|
| 1 | `distance_midpoint_to_target_surface_mm` | 0.0887 | ±0.0104 | 0.0819 | ±0.0143 |
| 2 | `psa_density` | 0.0457 | ±0.0076 | 0.0940 | ±0.0144 |
| 3 | `distance_midpoint_to_target_centroid_mm` | 0.0137 | ±0.0040 | 0.0125 | ±0.0098 |
| 4 | `core_length_mm` | 0.0075 | ±0.0032 | 0.0047 | ±0.0037 |
| 5 | `approximate_fraction_of_centerline_inside_target` | 0.0051 | ±0.0028 | 0.0083 | ±0.0080 |

### Top 5 features by PR-AUC importance

| # | Feature | PR-AUC drop | ± std | ROC-AUC drop | ± std |
|---|---|---|---|---|---|
| 1 | `psa_density` | 0.0940 | ±0.0144 | 0.0457 | ±0.0076 |
| 2 | `distance_midpoint_to_target_surface_mm` | 0.0819 | ±0.0143 | 0.0887 | ±0.0104 |
| 3 | `psa_ng_ml` | 0.0239 | ±0.0094 | 0.0038 | ±0.0034 |
| 4 | `distance_midpoint_to_target_centroid_mm` | 0.0125 | ±0.0098 | 0.0137 | ±0.0040 |
| 5 | `approximate_fraction_of_centerline_inside_target` | 0.0083 | ±0.0080 | 0.0051 | ±0.0028 |

**Interpretation:** Target geometry dominates (~65% of positive ROC-AUC importance). PSA-related variables contribute meaningfully (best PSA feature ranked #2); importance may be distributed across correlated clinical features. Prostate/biopsy geometry adds signal (max drop = 0.0075). Results stable (std < mean for all top-5 features).

---

## GG3+ / high-grade — logistic_regression

Baseline test ROC-AUC: **0.829**  ·  Baseline test PR-AUC: **0.265**

### Top 5 features by ROC-AUC importance

| # | Feature | ROC-AUC drop | ± std | PR-AUC drop | ± std |
|---|---|---|---|---|---|
| 1 | `distance_midpoint_to_target_surface_mm` | 0.1431 | ±0.0201 | 0.0999 | ±0.0196 |
| 2 | `log_psa_ng_ml` | 0.1062 | ±0.0198 | 0.1818 | ±0.0103 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0159 | ±0.0048 | 0.0444 | ±0.0144 |
| 4 | `prostate_volume_cc` | 0.0111 | ±0.0041 | -0.0161 | ±0.0098 |
| 5 | `core_length_mm` | 0.0082 | ±0.0042 | -0.0017 | ±0.0142 |

### Top 5 features by PR-AUC importance

| # | Feature | PR-AUC drop | ± std | ROC-AUC drop | ± std |
|---|---|---|---|---|---|
| 1 | `log_psa_ng_ml` | 0.1818 | ±0.0103 | 0.1062 | ±0.0198 |
| 2 | `distance_midpoint_to_target_surface_mm` | 0.0999 | ±0.0196 | 0.1431 | ±0.0201 |
| 3 | `approximate_fraction_of_centerline_inside_target` | 0.0444 | ±0.0144 | 0.0159 | ±0.0048 |
| 4 | `approximate_fraction_of_centerline_inside_prostate` | 0.0027 | ±0.0035 | -0.0003 | ±0.0005 |
| 5 | `distance_midpoint_to_prostate_surface_mm` | 0.0023 | ±0.0026 | 0.0006 | ±0.0005 |

**Interpretation:** Target geometry dominates (~55% of positive ROC-AUC importance). PSA-related variables contribute meaningfully (best PSA feature ranked #2); importance may be distributed across correlated clinical features. Prostate/biopsy geometry adds signal (max drop = 0.0082). Results stable (std < mean for all top-5 features).

