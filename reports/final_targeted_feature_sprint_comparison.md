# Final Targeted Feature Sprint — Comparison

**Branch:** `feat/geometry-clinical-baselines`  
**Script:** `src/compare_final_targeted_feature_sprint.py`  
**Data:** `data/share/needle_features_v1.csv`

## Methodology

**Endpoints:** GG2+ / csPCa (`binary_label_int`) and GG3+ / high-grade (`binary_label_gg3plus_int`).  
**Models:** logistic regression (baseline, C=1), tuned L2/L1/Elastic Net (C tuned on val PR-AUC), HistGradientBoosting (sensitivity), XGBoost if installed (sensitivity).  
**Preprocessing:** numeric features: median imputation + StandardScaler; boolean/one-hot features: most_frequent imputation, no scaling.  
**Threshold selection:** Youden J maximised on validation split.  
**Bootstrap CIs:** 1,000 patient-level resamples (patients resampled with replacement, all their cores included; 2.5/97.5 percentiles).  
**Reference baseline:** `current_compact` (8 features: 4 target-geometry + 4 clinical).  
**Decision thresholds:** ΔPR-AUC ≥ +0.030 OR ΔROC-AUC ≥ +0.020 (stricter than extended sprint).  

**Excluded predictors (pathology leakage):**  
`cancer_length_mm`, `pct_cancer_in_core`, `primary_gleason`, `secondary_gleason`, `pathology_label`, `pathology_label_int`, `binary_label` and all endpoint columns — never used.

## Leakage Audit and Feature Inventory

### Excluded leakage columns (pathology outcomes — never used as predictors)

- `binary_label` — present in dataset, excluded
- `binary_label_gg3plus_int` — present in dataset, excluded
- `binary_label_int` — present in dataset, excluded
- `cancer_length_mm` — present in dataset, excluded
- `pathology_label` — present in dataset, excluded
- `pathology_label_int` — present in dataset, excluded
- `pct_cancer_in_core` — present in dataset, excluded
- `primary_gleason` — present in dataset, excluded
- `secondary_gleason` — present in dataset, excluded

### Excluded metadata / ID / split columns

- `core_id` (present)
- `extraction_status` (present)
- `filename` (absent)
- `label_join_status` (present)
- `midpoint_inside_prostate` (present)
- `path` (absent)
- `patient_number` (present)
- `split` (present)
- `target_mesh_available` (present)

### `core_label` distribution

**Leakage status:** `core_label` is a PRE-BIOPSY procedural/anatomical label (physician's targeting decision). It is NOT pathology leakage. However, it encodes clinical suspicion (MRI PI-RADS, prior session positivity) and is therefore labelled as PROCEDURAL/ANATOMICAL CONTEXT, not pure geometry.

**Missingness:** 0 / 16992 (0.0%) missing.

| Value | Count | % total |
|-------|-------|---------|
| TARGET OR PRIOR POSITIVE | 7649 | 45.0% |
| LEFT APEX | 927 | 5.5% |
| RIGHT APEX | 899 | 5.3% |
| RIGHT BASE | 790 | 4.6% |
| LEFT MID | 774 | 4.6% |
| LEFT BASE | 769 | 4.5% |
| RIGHT MID | 767 | 4.5% |
| LEFT LATERAL BASE | 749 | 4.4% |
| RIGHT LATERAL BASE | 739 | 4.3% |
| RIGHT LATERAL MID | 739 | 4.3% |
| LEFT LATERAL APEX | 736 | 4.3% |
| RIGHT LATERAL APEX | 727 | 4.3% |
| LEFT LATERAL MID | 726 | 4.3% |
| EXTRA | 1 | 0.0% |

**`core_label` by split:**

| Split | n_rows | n_targeted | targeted_pct |
|-------|--------|------------|--------------|
| train | 11827 | 5209 | 44.0% |
| val | 2748 | 1333 | 48.5% |
| test | 2417 | 1107 | 45.8% |

### Auto-detected image/intensity columns

**2 column(s) detected** (keyword match + ≥10% non-NaN):

- `n_centerline_voxels`: missingness 0.0%
- `n_tube_voxels`: missingness 0.0%

### Derived `core_label` one-hot columns

**14 one-hot columns:** `cl_extra`, `cl_left_apex`, `cl_left_base`, `cl_left_lateral_apex`, `cl_left_lateral_base`, `cl_left_lateral_mid`, `cl_left_mid`, `cl_right_apex`, `cl_right_base`, `cl_right_lateral_apex`, `cl_right_lateral_base`, `cl_right_lateral_mid`, `cl_right_mid`, `cl_target_or_prior_positive`

### Derived anatomical/procedural flag columns

**9 anatomical flag columns:** `cl_is_targeted_or_prior_positive`, `cl_is_systematic_core`, `cl_is_left`, `cl_is_right`, `cl_is_apex`, `cl_is_mid`, `cl_is_base`, `cl_is_lateral`, `cl_is_extra`

### Derived transformed geometry columns

**7 transformed geometry columns:** `geo_log1p_dist_surface`, `geo_log1p_dist_centroid`, `geo_surface_dist_per_core_len`, `geo_centroid_dist_per_core_len`, `geo_frac_inside_times_core_len`, `geo_outside_dist_not_intersecting`, `geo_inside_frac_when_intersecting`

### Missingness of new derived features (all splits)

| Column | missingness |
|--------|------------|
| `cl_extra` | 0.0% |
| `cl_left_apex` | 0.0% |
| `cl_left_base` | 0.0% |
| `cl_left_lateral_apex` | 0.0% |
| `cl_left_lateral_base` | 0.0% |
| `cl_left_lateral_mid` | 0.0% |
| `cl_left_mid` | 0.0% |
| `cl_right_apex` | 0.0% |
| `cl_right_base` | 0.0% |
| `cl_right_lateral_apex` | 0.0% |
| `cl_right_lateral_base` | 0.0% |
| `cl_right_lateral_mid` | 0.0% |
| `cl_right_mid` | 0.0% |
| `cl_target_or_prior_positive` | 0.0% |
| `cl_is_targeted_or_prior_positive` | 0.0% |
| `cl_is_systematic_core` | 0.0% |
| `cl_is_left` | 0.0% |
| `cl_is_right` | 0.0% |
| `cl_is_apex` | 0.0% |
| `cl_is_mid` | 0.0% |
| `cl_is_base` | 0.0% |
| `cl_is_lateral` | 0.0% |
| `cl_is_extra` | 0.0% |
| `geo_log1p_dist_surface` | 0.2% |
| `geo_log1p_dist_centroid` | 0.2% |
| `geo_surface_dist_per_core_len` | 0.2% |
| `geo_centroid_dist_per_core_len` | 0.2% |
| `geo_frac_inside_times_core_len` | 2.3% |
| `geo_outside_dist_not_intersecting` | 0.2% |
| `geo_inside_frac_when_intersecting` | 2.3% |

## Performance comparison

### GG2+ / csPCa

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train | n_val | n_test | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 8 | 0.754 | 0.699 | 0.807 | 0.327 | 0.230 | 0.461 | 3.050 | 0.463 | 0.703 | 0.706 | 0.223 | 0.339 | 11827 | 2748 | 2417 | 0.107 |
| current_compact | tuned_l2_logistic_regression | 8 | 0.754 | 0.699 | 0.807 | 0.327 | 0.230 | 0.461 | 3.051 | 0.463 | 0.703 | 0.707 | 0.223 | 0.339 | 11827 | 2748 | 2417 | 0.107 |
| current_compact | tuned_l1_logistic_regression | 8 | 0.754 | 0.699 | 0.807 | 0.324 | 0.228 | 0.460 | 3.022 | 0.468 | 0.707 | 0.711 | 0.227 | 0.343 | 11827 | 2748 | 2417 | 0.107 |
| current_compact | tuned_elasticnet_logistic_regression | 8 | 0.754 | 0.699 | 0.807 | 0.324 | 0.228 | 0.460 | 3.028 | 0.464 | 0.707 | 0.706 | 0.224 | 0.340 | 11827 | 2748 | 2417 | 0.107 |
| current_compact | hist_gradient_boosting | 8 | 0.745 | 0.682 | 0.800 | 0.295 | 0.210 | 0.406 | 2.752 | 0.107 | 0.591 | 0.770 | 0.236 | 0.337 | 11827 | 2748 | 2417 | 0.107 |
| current_compact | xgboost | 8 | 0.754 | 0.694 | 0.808 | 0.321 | 0.230 | 0.450 | 2.999 | 0.521 | 0.595 | 0.789 | 0.253 | 0.355 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_onehot | logistic_regression | 22 | 0.760 | 0.710 | 0.809 | 0.338 | 0.238 | 0.460 | 3.156 | 0.537 | 0.645 | 0.773 | 0.254 | 0.365 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_onehot | tuned_l2_logistic_regression | 22 | 0.760 | 0.710 | 0.809 | 0.339 | 0.238 | 0.461 | 3.163 | 0.537 | 0.645 | 0.773 | 0.255 | 0.365 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_onehot | tuned_l1_logistic_regression | 22 | 0.760 | 0.710 | 0.809 | 0.339 | 0.239 | 0.461 | 3.162 | 0.537 | 0.645 | 0.773 | 0.254 | 0.365 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_onehot | tuned_elasticnet_logistic_regression | 22 | 0.760 | 0.710 | 0.809 | 0.339 | 0.239 | 0.461 | 3.165 | 0.536 | 0.649 | 0.772 | 0.254 | 0.365 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_onehot | hist_gradient_boosting | 22 | 0.748 | 0.683 | 0.802 | 0.306 | 0.218 | 0.413 | 2.859 | 0.091 | 0.676 | 0.726 | 0.228 | 0.341 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_onehot | xgboost | 22 | 0.763 | 0.706 | 0.814 | 0.329 | 0.235 | 0.451 | 3.069 | 0.526 | 0.591 | 0.785 | 0.248 | 0.349 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_anatomical_core_label | logistic_regression | 17 | 0.760 | 0.711 | 0.810 | 0.338 | 0.238 | 0.460 | 3.152 | 0.534 | 0.649 | 0.771 | 0.253 | 0.364 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_anatomical_core_label | tuned_l2_logistic_regression | 17 | 0.760 | 0.711 | 0.810 | 0.338 | 0.238 | 0.460 | 3.152 | 0.534 | 0.649 | 0.772 | 0.255 | 0.366 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_anatomical_core_label | tuned_l1_logistic_regression | 17 | 0.760 | 0.711 | 0.810 | 0.338 | 0.238 | 0.461 | 3.158 | 0.535 | 0.649 | 0.772 | 0.255 | 0.366 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_anatomical_core_label | tuned_elasticnet_logistic_regression | 17 | 0.760 | 0.711 | 0.810 | 0.339 | 0.239 | 0.461 | 3.165 | 0.534 | 0.649 | 0.772 | 0.255 | 0.366 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_anatomical_core_label | hist_gradient_boosting | 17 | 0.757 | 0.695 | 0.810 | 0.312 | 0.220 | 0.429 | 2.907 | 0.067 | 0.734 | 0.658 | 0.205 | 0.320 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_anatomical_core_label | xgboost | 17 | 0.763 | 0.708 | 0.813 | 0.325 | 0.233 | 0.444 | 3.037 | 0.544 | 0.571 | 0.803 | 0.258 | 0.355 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_intensity | logistic_regression | 10 | 0.751 | 0.695 | 0.804 | 0.324 | 0.224 | 0.461 | 3.021 | 0.468 | 0.687 | 0.718 | 0.226 | 0.341 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_intensity | tuned_l2_logistic_regression | 10 | 0.751 | 0.695 | 0.804 | 0.324 | 0.224 | 0.461 | 3.022 | 0.468 | 0.687 | 0.719 | 0.227 | 0.341 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_intensity | tuned_l1_logistic_regression | 10 | 0.752 | 0.696 | 0.805 | 0.323 | 0.225 | 0.461 | 3.017 | 0.468 | 0.691 | 0.715 | 0.226 | 0.340 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_intensity | tuned_elasticnet_logistic_regression | 10 | 0.752 | 0.696 | 0.805 | 0.324 | 0.225 | 0.462 | 3.020 | 0.468 | 0.687 | 0.717 | 0.226 | 0.340 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_intensity | hist_gradient_boosting | 10 | 0.737 | 0.666 | 0.792 | 0.284 | 0.203 | 0.395 | 2.648 | 0.064 | 0.722 | 0.647 | 0.197 | 0.310 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_intensity | xgboost | 10 | 0.749 | 0.690 | 0.803 | 0.315 | 0.222 | 0.445 | 2.938 | 0.515 | 0.591 | 0.788 | 0.250 | 0.352 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_transformed_geometry | logistic_regression | 15 | 0.756 | 0.701 | 0.805 | 0.315 | 0.222 | 0.446 | 2.935 | 0.506 | 0.668 | 0.743 | 0.238 | 0.351 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_transformed_geometry | tuned_l2_logistic_regression | 15 | 0.756 | 0.701 | 0.805 | 0.314 | 0.221 | 0.445 | 2.930 | 0.505 | 0.668 | 0.743 | 0.238 | 0.351 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_transformed_geometry | tuned_l1_logistic_regression | 15 | 0.756 | 0.702 | 0.805 | 0.315 | 0.222 | 0.446 | 2.939 | 0.505 | 0.668 | 0.743 | 0.238 | 0.351 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_transformed_geometry | tuned_elasticnet_logistic_regression | 15 | 0.756 | 0.702 | 0.805 | 0.315 | 0.222 | 0.446 | 2.939 | 0.505 | 0.668 | 0.743 | 0.238 | 0.351 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_transformed_geometry | hist_gradient_boosting | 15 | 0.754 | 0.694 | 0.806 | 0.296 | 0.214 | 0.412 | 2.763 | 0.069 | 0.710 | 0.672 | 0.206 | 0.320 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_transformed_geometry | xgboost | 15 | 0.758 | 0.700 | 0.811 | 0.337 | 0.236 | 0.463 | 3.142 | 0.565 | 0.537 | 0.823 | 0.266 | 0.356 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_intensity | logistic_regression | 24 | 0.757 | 0.706 | 0.806 | 0.336 | 0.235 | 0.460 | 3.132 | 0.543 | 0.618 | 0.780 | 0.252 | 0.358 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_intensity | tuned_l2_logistic_regression | 24 | 0.757 | 0.706 | 0.807 | 0.337 | 0.237 | 0.462 | 3.149 | 0.529 | 0.633 | 0.770 | 0.248 | 0.357 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_intensity | tuned_l1_logistic_regression | 24 | 0.757 | 0.706 | 0.806 | 0.336 | 0.235 | 0.460 | 3.136 | 0.543 | 0.618 | 0.780 | 0.252 | 0.358 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_intensity | tuned_elasticnet_logistic_regression | 24 | 0.758 | 0.708 | 0.807 | 0.340 | 0.238 | 0.463 | 3.171 | 0.557 | 0.602 | 0.791 | 0.257 | 0.361 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_intensity | hist_gradient_boosting | 24 | 0.748 | 0.686 | 0.799 | 0.292 | 0.209 | 0.402 | 2.723 | 0.056 | 0.722 | 0.644 | 0.196 | 0.308 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_intensity | xgboost | 24 | 0.761 | 0.706 | 0.811 | 0.320 | 0.229 | 0.438 | 2.987 | 0.521 | 0.595 | 0.785 | 0.249 | 0.351 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_transformed_geo | logistic_regression | 29 | 0.761 | 0.710 | 0.808 | 0.327 | 0.229 | 0.449 | 3.051 | 0.532 | 0.653 | 0.762 | 0.248 | 0.359 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_transformed_geo | tuned_l2_logistic_regression | 29 | 0.761 | 0.710 | 0.808 | 0.326 | 0.229 | 0.446 | 3.041 | 0.530 | 0.656 | 0.761 | 0.248 | 0.360 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_transformed_geo | tuned_l1_logistic_regression | 29 | 0.763 | 0.713 | 0.810 | 0.338 | 0.242 | 0.458 | 3.150 | 0.563 | 0.610 | 0.789 | 0.257 | 0.362 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_transformed_geo | tuned_elasticnet_logistic_regression | 29 | 0.761 | 0.711 | 0.808 | 0.326 | 0.228 | 0.447 | 3.042 | 0.528 | 0.656 | 0.760 | 0.247 | 0.359 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_transformed_geo | hist_gradient_boosting | 29 | 0.759 | 0.699 | 0.810 | 0.287 | 0.211 | 0.389 | 2.675 | 0.079 | 0.718 | 0.696 | 0.221 | 0.338 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_core_label_plus_transformed_geo | xgboost | 29 | 0.765 | 0.709 | 0.813 | 0.332 | 0.237 | 0.450 | 3.101 | 0.552 | 0.564 | 0.805 | 0.258 | 0.354 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_all_low_cost_features | logistic_regression | 40 | 0.758 | 0.709 | 0.805 | 0.324 | 0.226 | 0.444 | 3.022 | 0.535 | 0.633 | 0.769 | 0.247 | 0.356 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_all_low_cost_features | tuned_l2_logistic_regression | 40 | 0.760 | 0.710 | 0.807 | 0.335 | 0.235 | 0.460 | 3.125 | 0.579 | 0.575 | 0.807 | 0.263 | 0.361 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_all_low_cost_features | tuned_l1_logistic_regression | 40 | 0.758 | 0.708 | 0.806 | 0.322 | 0.224 | 0.442 | 3.005 | 0.532 | 0.641 | 0.767 | 0.249 | 0.358 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_all_low_cost_features | tuned_elasticnet_logistic_regression | 40 | 0.758 | 0.708 | 0.806 | 0.323 | 0.226 | 0.442 | 3.011 | 0.532 | 0.641 | 0.767 | 0.249 | 0.358 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_all_low_cost_features | hist_gradient_boosting | 40 | 0.757 | 0.695 | 0.808 | 0.298 | 0.214 | 0.409 | 2.779 | 0.128 | 0.579 | 0.788 | 0.247 | 0.346 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_all_low_cost_features | xgboost | 40 | 0.764 | 0.710 | 0.813 | 0.327 | 0.233 | 0.448 | 3.055 | 0.537 | 0.575 | 0.799 | 0.256 | 0.354 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_core_label | logistic_regression | 18 | 0.742 | 0.690 | 0.788 | 0.340 | 0.236 | 0.456 | 3.177 | 0.599 | 0.502 | 0.810 | 0.241 | 0.326 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_core_label | tuned_l2_logistic_regression | 18 | 0.742 | 0.690 | 0.788 | 0.340 | 0.236 | 0.456 | 3.175 | 0.599 | 0.502 | 0.810 | 0.241 | 0.326 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_core_label | tuned_l1_logistic_regression | 18 | 0.743 | 0.694 | 0.789 | 0.340 | 0.235 | 0.451 | 3.170 | 0.592 | 0.498 | 0.810 | 0.240 | 0.324 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_core_label | tuned_elasticnet_logistic_regression | 18 | 0.742 | 0.690 | 0.788 | 0.340 | 0.234 | 0.455 | 3.175 | 0.594 | 0.498 | 0.810 | 0.239 | 0.323 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_core_label | hist_gradient_boosting | 18 | 0.717 | 0.657 | 0.770 | 0.281 | 0.205 | 0.370 | 2.622 | 0.076 | 0.606 | 0.670 | 0.181 | 0.278 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_core_label | xgboost | 18 | 0.732 | 0.682 | 0.779 | 0.310 | 0.222 | 0.415 | 2.894 | 0.549 | 0.483 | 0.787 | 0.214 | 0.297 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_core_label | logistic_regression | 18 | 0.734 | 0.680 | 0.779 | 0.227 | 0.163 | 0.317 | 2.114 | 0.545 | 0.653 | 0.698 | 0.206 | 0.313 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_core_label | tuned_l2_logistic_regression | 18 | 0.734 | 0.680 | 0.779 | 0.226 | 0.163 | 0.317 | 2.113 | 0.546 | 0.649 | 0.698 | 0.205 | 0.312 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_core_label | tuned_l1_logistic_regression | 18 | 0.734 | 0.680 | 0.779 | 0.227 | 0.163 | 0.317 | 2.114 | 0.549 | 0.645 | 0.702 | 0.206 | 0.312 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_core_label | tuned_elasticnet_logistic_regression | 18 | 0.734 | 0.680 | 0.779 | 0.226 | 0.163 | 0.317 | 2.114 | 0.549 | 0.645 | 0.702 | 0.206 | 0.312 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_core_label | hist_gradient_boosting | 18 | 0.727 | 0.679 | 0.768 | 0.218 | 0.165 | 0.297 | 2.035 | 0.122 | 0.672 | 0.672 | 0.198 | 0.305 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_core_label | xgboost | 18 | 0.736 | 0.688 | 0.779 | 0.228 | 0.168 | 0.314 | 2.126 | 0.526 | 0.699 | 0.675 | 0.205 | 0.317 | 11827 | 2748 | 2417 | 0.107 |

### GG3+ / high-grade

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train | n_val | n_test | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 8 | 0.828 | 0.751 | 0.892 | 0.269 | 0.094 | 0.516 | 7.070 | 0.400 | 0.793 | 0.712 | 0.098 | 0.175 | 11827 | 2748 | 2417 | 0.038 |
| current_compact | tuned_l2_logistic_regression | 8 | 0.828 | 0.750 | 0.892 | 0.267 | 0.094 | 0.516 | 7.021 | 0.400 | 0.793 | 0.711 | 0.098 | 0.175 | 11827 | 2748 | 2417 | 0.038 |
| current_compact | tuned_l1_logistic_regression | 8 | 0.828 | 0.751 | 0.892 | 0.269 | 0.094 | 0.516 | 7.069 | 0.400 | 0.793 | 0.712 | 0.098 | 0.175 | 11827 | 2748 | 2417 | 0.038 |
| current_compact | tuned_elasticnet_logistic_regression | 8 | 0.828 | 0.751 | 0.892 | 0.267 | 0.094 | 0.516 | 7.020 | 0.400 | 0.793 | 0.712 | 0.098 | 0.175 | 11827 | 2748 | 2417 | 0.038 |
| current_compact | hist_gradient_boosting | 8 | 0.808 | 0.717 | 0.881 | 0.255 | 0.083 | 0.469 | 6.693 | 0.018 | 0.663 | 0.768 | 0.101 | 0.176 | 11827 | 2748 | 2417 | 0.038 |
| current_compact | xgboost | 8 | 0.817 | 0.732 | 0.884 | 0.248 | 0.093 | 0.459 | 6.508 | 0.353 | 0.674 | 0.750 | 0.096 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_onehot | logistic_regression | 22 | 0.833 | 0.757 | 0.896 | 0.284 | 0.097 | 0.528 | 7.453 | 0.407 | 0.772 | 0.724 | 0.100 | 0.176 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_onehot | tuned_l2_logistic_regression | 22 | 0.833 | 0.757 | 0.896 | 0.284 | 0.098 | 0.527 | 7.455 | 0.410 | 0.772 | 0.727 | 0.101 | 0.178 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_onehot | tuned_l1_logistic_regression | 22 | 0.829 | 0.752 | 0.892 | 0.287 | 0.095 | 0.527 | 7.538 | 0.379 | 0.815 | 0.680 | 0.092 | 0.165 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_onehot | tuned_elasticnet_logistic_regression | 22 | 0.832 | 0.756 | 0.894 | 0.304 | 0.095 | 0.532 | 7.988 | 0.361 | 0.815 | 0.671 | 0.089 | 0.161 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_onehot | hist_gradient_boosting | 22 | 0.813 | 0.727 | 0.881 | 0.247 | 0.079 | 0.455 | 6.492 | 0.007 | 0.815 | 0.587 | 0.072 | 0.133 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_onehot | xgboost | 22 | 0.819 | 0.737 | 0.886 | 0.241 | 0.092 | 0.431 | 6.341 | 0.246 | 0.783 | 0.672 | 0.086 | 0.155 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_anatomical_core_label | logistic_regression | 17 | 0.831 | 0.755 | 0.895 | 0.281 | 0.096 | 0.526 | 7.379 | 0.408 | 0.783 | 0.727 | 0.102 | 0.180 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_anatomical_core_label | tuned_l2_logistic_regression | 17 | 0.831 | 0.755 | 0.895 | 0.281 | 0.096 | 0.525 | 7.376 | 0.407 | 0.783 | 0.726 | 0.102 | 0.180 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_anatomical_core_label | tuned_l1_logistic_regression | 17 | 0.830 | 0.753 | 0.892 | 0.284 | 0.095 | 0.527 | 7.457 | 0.397 | 0.772 | 0.699 | 0.092 | 0.165 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_anatomical_core_label | tuned_elasticnet_logistic_regression | 17 | 0.833 | 0.758 | 0.896 | 0.305 | 0.097 | 0.535 | 8.017 | 0.425 | 0.761 | 0.738 | 0.103 | 0.182 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_anatomical_core_label | hist_gradient_boosting | 17 | 0.818 | 0.735 | 0.885 | 0.267 | 0.084 | 0.483 | 7.016 | 0.030 | 0.620 | 0.815 | 0.117 | 0.197 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_anatomical_core_label | xgboost | 17 | 0.820 | 0.740 | 0.885 | 0.228 | 0.087 | 0.413 | 5.984 | 0.214 | 0.804 | 0.653 | 0.084 | 0.152 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_intensity | logistic_regression | 10 | 0.822 | 0.741 | 0.890 | 0.274 | 0.093 | 0.523 | 7.207 | 0.410 | 0.761 | 0.732 | 0.101 | 0.178 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_intensity | tuned_l2_logistic_regression | 10 | 0.822 | 0.741 | 0.890 | 0.275 | 0.094 | 0.524 | 7.212 | 0.411 | 0.761 | 0.732 | 0.101 | 0.179 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_intensity | tuned_l1_logistic_regression | 10 | 0.826 | 0.747 | 0.890 | 0.282 | 0.094 | 0.535 | 7.422 | 0.383 | 0.815 | 0.687 | 0.093 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_intensity | tuned_elasticnet_logistic_regression | 10 | 0.822 | 0.741 | 0.890 | 0.274 | 0.093 | 0.523 | 7.208 | 0.410 | 0.761 | 0.732 | 0.101 | 0.178 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_intensity | hist_gradient_boosting | 10 | 0.792 | 0.694 | 0.872 | 0.243 | 0.078 | 0.457 | 6.382 | 0.009 | 0.707 | 0.641 | 0.072 | 0.131 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_intensity | xgboost | 10 | 0.798 | 0.697 | 0.880 | 0.245 | 0.089 | 0.471 | 6.437 | 0.355 | 0.641 | 0.754 | 0.094 | 0.163 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_transformed_geometry | logistic_regression | 15 | 0.825 | 0.748 | 0.889 | 0.256 | 0.095 | 0.481 | 6.722 | 0.400 | 0.783 | 0.702 | 0.094 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_transformed_geometry | tuned_l2_logistic_regression | 15 | 0.825 | 0.747 | 0.889 | 0.255 | 0.094 | 0.480 | 6.701 | 0.401 | 0.783 | 0.704 | 0.095 | 0.169 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_transformed_geometry | tuned_l1_logistic_regression | 15 | 0.825 | 0.746 | 0.888 | 0.253 | 0.093 | 0.479 | 6.657 | 0.415 | 0.772 | 0.719 | 0.098 | 0.174 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_transformed_geometry | tuned_elasticnet_logistic_regression | 15 | 0.825 | 0.746 | 0.889 | 0.255 | 0.093 | 0.483 | 6.707 | 0.416 | 0.772 | 0.719 | 0.098 | 0.174 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_transformed_geometry | hist_gradient_boosting | 15 | 0.811 | 0.727 | 0.877 | 0.252 | 0.076 | 0.467 | 6.610 | 0.011 | 0.783 | 0.654 | 0.082 | 0.149 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_transformed_geometry | xgboost | 15 | 0.820 | 0.739 | 0.885 | 0.293 | 0.100 | 0.494 | 7.698 | 0.344 | 0.685 | 0.742 | 0.095 | 0.167 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_intensity | logistic_regression | 24 | 0.827 | 0.749 | 0.894 | 0.291 | 0.095 | 0.532 | 7.646 | 0.415 | 0.728 | 0.737 | 0.099 | 0.174 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_intensity | tuned_l2_logistic_regression | 24 | 0.827 | 0.749 | 0.894 | 0.291 | 0.096 | 0.533 | 7.658 | 0.410 | 0.750 | 0.732 | 0.100 | 0.176 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_intensity | tuned_l1_logistic_regression | 24 | 0.826 | 0.748 | 0.890 | 0.288 | 0.095 | 0.534 | 7.575 | 0.405 | 0.772 | 0.711 | 0.096 | 0.170 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_intensity | tuned_elasticnet_logistic_regression | 24 | 0.827 | 0.748 | 0.891 | 0.309 | 0.096 | 0.538 | 8.119 | 0.368 | 0.815 | 0.681 | 0.092 | 0.165 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_intensity | hist_gradient_boosting | 24 | 0.795 | 0.700 | 0.876 | 0.258 | 0.080 | 0.475 | 6.770 | 0.010 | 0.739 | 0.650 | 0.077 | 0.140 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_intensity | xgboost | 24 | 0.806 | 0.712 | 0.883 | 0.262 | 0.094 | 0.453 | 6.886 | 0.280 | 0.696 | 0.698 | 0.084 | 0.149 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_transformed_geo | logistic_regression | 29 | 0.829 | 0.751 | 0.892 | 0.268 | 0.095 | 0.485 | 7.052 | 0.393 | 0.793 | 0.698 | 0.094 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_transformed_geo | tuned_l2_logistic_regression | 29 | 0.828 | 0.749 | 0.891 | 0.270 | 0.093 | 0.483 | 7.093 | 0.420 | 0.750 | 0.724 | 0.097 | 0.172 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_transformed_geo | tuned_l1_logistic_regression | 29 | 0.828 | 0.750 | 0.891 | 0.270 | 0.094 | 0.483 | 7.092 | 0.422 | 0.750 | 0.725 | 0.097 | 0.172 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_transformed_geo | tuned_elasticnet_logistic_regression | 29 | 0.829 | 0.750 | 0.892 | 0.270 | 0.093 | 0.485 | 7.085 | 0.422 | 0.750 | 0.725 | 0.097 | 0.172 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_transformed_geo | hist_gradient_boosting | 29 | 0.811 | 0.729 | 0.878 | 0.250 | 0.075 | 0.469 | 6.558 | 0.010 | 0.826 | 0.611 | 0.077 | 0.142 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_core_label_plus_transformed_geo | xgboost | 29 | 0.822 | 0.744 | 0.886 | 0.254 | 0.094 | 0.446 | 6.666 | 0.364 | 0.674 | 0.762 | 0.101 | 0.175 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_all_low_cost_features | logistic_regression | 40 | 0.824 | 0.745 | 0.890 | 0.268 | 0.092 | 0.485 | 7.047 | 0.409 | 0.739 | 0.720 | 0.094 | 0.167 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_all_low_cost_features | tuned_l2_logistic_regression | 40 | 0.823 | 0.744 | 0.889 | 0.270 | 0.092 | 0.488 | 7.105 | 0.407 | 0.750 | 0.717 | 0.095 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_all_low_cost_features | tuned_l1_logistic_regression | 40 | 0.823 | 0.744 | 0.889 | 0.270 | 0.092 | 0.489 | 7.097 | 0.408 | 0.750 | 0.719 | 0.096 | 0.170 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_all_low_cost_features | tuned_elasticnet_logistic_regression | 40 | 0.824 | 0.744 | 0.889 | 0.272 | 0.092 | 0.489 | 7.151 | 0.409 | 0.739 | 0.720 | 0.094 | 0.167 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_all_low_cost_features | hist_gradient_boosting | 40 | 0.797 | 0.705 | 0.870 | 0.243 | 0.073 | 0.455 | 6.384 | 0.019 | 0.630 | 0.767 | 0.097 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_all_low_cost_features | xgboost | 40 | 0.811 | 0.722 | 0.885 | 0.299 | 0.094 | 0.490 | 7.853 | 0.376 | 0.652 | 0.765 | 0.099 | 0.172 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_core_label | logistic_regression | 18 | 0.794 | 0.701 | 0.868 | 0.292 | 0.082 | 0.508 | 7.670 | 0.420 | 0.717 | 0.707 | 0.088 | 0.157 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_core_label | tuned_l2_logistic_regression | 18 | 0.794 | 0.700 | 0.867 | 0.291 | 0.082 | 0.507 | 7.647 | 0.419 | 0.717 | 0.707 | 0.088 | 0.157 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_core_label | tuned_l1_logistic_regression | 18 | 0.787 | 0.694 | 0.863 | 0.286 | 0.078 | 0.501 | 7.520 | 0.434 | 0.685 | 0.714 | 0.087 | 0.154 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_core_label | tuned_elasticnet_logistic_regression | 18 | 0.790 | 0.698 | 0.864 | 0.289 | 0.080 | 0.504 | 7.585 | 0.421 | 0.707 | 0.714 | 0.089 | 0.158 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_core_label | hist_gradient_boosting | 18 | 0.756 | 0.654 | 0.841 | 0.267 | 0.073 | 0.479 | 7.026 | 0.043 | 0.489 | 0.860 | 0.121 | 0.194 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_core_label | xgboost | 18 | 0.780 | 0.683 | 0.859 | 0.242 | 0.078 | 0.429 | 6.358 | 0.750 | 0.359 | 0.934 | 0.177 | 0.237 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_core_label | logistic_regression | 18 | 0.808 | 0.736 | 0.864 | 0.140 | 0.058 | 0.285 | 3.675 | 0.491 | 0.804 | 0.662 | 0.086 | 0.155 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_core_label | tuned_l2_logistic_regression | 18 | 0.808 | 0.736 | 0.864 | 0.140 | 0.058 | 0.284 | 3.677 | 0.491 | 0.804 | 0.661 | 0.086 | 0.155 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_core_label | tuned_l1_logistic_regression | 18 | 0.788 | 0.709 | 0.853 | 0.130 | 0.053 | 0.276 | 3.422 | 0.350 | 0.761 | 0.649 | 0.079 | 0.143 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_core_label | tuned_elasticnet_logistic_regression | 18 | 0.805 | 0.732 | 0.863 | 0.138 | 0.057 | 0.283 | 3.628 | 0.480 | 0.804 | 0.662 | 0.086 | 0.155 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_core_label | hist_gradient_boosting | 18 | 0.783 | 0.715 | 0.838 | 0.111 | 0.054 | 0.217 | 2.914 | 0.017 | 0.946 | 0.489 | 0.068 | 0.127 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_core_label | xgboost | 18 | 0.801 | 0.727 | 0.860 | 0.130 | 0.056 | 0.276 | 3.409 | 0.487 | 0.728 | 0.722 | 0.094 | 0.166 | 11827 | 2748 | 2417 | 0.038 |

## Delta vs compact baseline (logistic regression)

### GG2+ / csPCa

Delta values = feature set − `current_compact` (LR, test split). Positive = better.

Compact baseline: ROC-AUC=0.754, PR-AUC=0.327, top5% prev=0.504, top10% capture=0.320

| Feature set | n_feat | ROC-AUC | ΔROC | PR-AUC | ΔPR | Δtop5%prev | Δtop10%cap | Threshold met? |
|-------------|--------|---------|------|--------|-----|-----------|-----------|---------------|
| `current_compact` | 8 | 0.754 | +0.000 | 0.327 | +0.000 | +0.000 | +0.000 | No |
| `compact_plus_core_label_onehot` | 22 | 0.760 | +0.005 | 0.338 | +0.011 | -0.017 | -0.012 | No |
| `compact_plus_anatomical_core_label` | 17 | 0.760 | +0.006 | 0.338 | +0.011 | -0.017 | -0.015 | No |
| `compact_plus_intensity` | 10 | 0.751 | -0.003 | 0.324 | -0.003 | +0.000 | +0.004 | No |
| `compact_plus_transformed_geometry` | 15 | 0.756 | +0.002 | 0.315 | -0.012 | -0.033 | +0.004 | No |
| `compact_plus_core_label_plus_intensity` | 24 | 0.757 | +0.003 | 0.336 | +0.009 | -0.025 | -0.008 | No |
| `compact_plus_core_label_plus_transformed_geo` | 29 | 0.761 | +0.007 | 0.327 | +0.000 | -0.041 | -0.012 | No |
| `compact_plus_all_low_cost_features` | 40 | 0.758 | +0.004 | 0.324 | -0.003 | -0.041 | -0.008 | No |
| `clinical_plus_core_label` | 18 | 0.742 | -0.012 | 0.340 | +0.014 | -0.008 | -0.023 | No |
| `target_geometry_plus_core_label` | 18 | 0.734 | -0.020 | 0.227 | -0.100 | -0.240 | -0.069 | No |

### GG3+ / high-grade

Delta values = feature set − `current_compact` (LR, test split). Positive = better.

Compact baseline: ROC-AUC=0.828, PR-AUC=0.269, top5% prev=0.314, top10% capture=0.500

| Feature set | n_feat | ROC-AUC | ΔROC | PR-AUC | ΔPR | Δtop5%prev | Δtop10%cap | Threshold met? |
|-------------|--------|---------|------|--------|-----|-----------|-----------|---------------|
| `current_compact` | 8 | 0.828 | +0.000 | 0.269 | +0.000 | +0.000 | +0.000 | No |
| `compact_plus_core_label_onehot` | 22 | 0.833 | +0.005 | 0.284 | +0.015 | +0.000 | +0.000 | No |
| `compact_plus_anatomical_core_label` | 17 | 0.831 | +0.003 | 0.281 | +0.012 | +0.000 | +0.011 | No |
| `compact_plus_intensity` | 10 | 0.822 | -0.006 | 0.274 | +0.005 | +0.008 | +0.011 | No |
| `compact_plus_transformed_geometry` | 15 | 0.825 | -0.003 | 0.256 | -0.013 | +0.000 | +0.000 | No |
| `compact_plus_core_label_plus_intensity` | 24 | 0.827 | -0.001 | 0.291 | +0.022 | +0.008 | +0.011 | No |
| `compact_plus_core_label_plus_transformed_geo` | 29 | 0.829 | +0.001 | 0.268 | -0.001 | +0.000 | -0.011 | No |
| `compact_plus_all_low_cost_features` | 40 | 0.824 | -0.004 | 0.268 | -0.001 | +0.000 | +0.000 | No |
| `clinical_plus_core_label` | 18 | 0.794 | -0.034 | 0.292 | +0.023 | -0.025 | -0.054 | No |
| `target_geometry_plus_core_label` | 18 | 0.808 | -0.020 | 0.140 | -0.129 | -0.157 | -0.163 | No |

