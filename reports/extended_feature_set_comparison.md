# Extended Feature Set Comparison

**Branch:** `feat/geometry-clinical-baselines`  
**Script:** `src/compare_extended_feature_sets.py`  
**Data:** `data/share/needle_features_v1.csv`

## Methodology

**Endpoints:** GG2+ / csPCa (`binary_label_int`) and GG3+ / high-grade (`binary_label_gg3plus_int`).  
**Models:** logistic regression (class-balanced, standardised features), HistGradientBoosting, and XGBoost (if installed).  
**Threshold selection:** Youden J maximised on the validation split.  
**Bootstrap CIs:** 1,000 patient-level resamples of the test set.  
**Reference baseline:** `current_compact` (8-feature compact model: 4 target-geometry + 4 clinical).  
**Delta convention:** extended − baseline; positive = extension outperforms.  

**Excluded predictors (pathology leakage):**
`cancer_length_mm`, `pct_cancer_in_core`, `primary_gleason`, `secondary_gleason` — all measured after core processing; never used.

## Feature Audit: `core_label` and `is_targeted_or_prior_positive`

### Critical interpretation notes

**`is_targeted_or_prior_positive`** is derived from `core_label == "TARGET OR PRIOR POSITIVE"`. It encodes whether the urologist directed
this biopsy core at an MRI-visible suspicious lesion or a previously positive site, rather than a systematic sextant location.  This information is recorded **before** the pathology outcome of the current core is known.

This is **not** pathology leakage in the strict sense.  However, it is a *procedural targeting-context* feature, not a geometry or clinical feature.  Including it encodes clinical suspicion (the radiologist's / urologist's belief that this site is suspicious), which may partially proxy for the MRI PI-RADS score or prior-session pathology.  Any model including this feature must be reported as a separate **procedural-context model**, not as an extension of the pure geometry-clinical compact model.

**Excluded pathology outcomes (never used as predictors):**
- `cancer_length_mm` — millimetres of cancer in core (pathology result)
- `pct_cancer_in_core` — percentage cancer (pathology result)
- `primary_gleason` / `secondary_gleason` — Gleason grades (pathology result)

**`core_label` missingness (coord_match, valid split):** 0 / 16992 (0.0%) missing.

**`core_label` value distribution (all rows, all splits):**

| Value | Count | % of total |
|-------|-------|-----------|
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

**GG2+ prevalence by `is_targeted_or_prior_positive` (test split):**

- **Targeted / prior-positive** (n=1107): GG2+ prevalence = 18.0%
- **Systematic sextant** (n=1310): GG2+ prevalence = 4.6%

**GG3+ prevalence by `is_targeted_or_prior_positive` (test split):**

- **Targeted / prior-positive** (n=1107): GG3+ prevalence = 6.8%
- **Systematic sextant** (n=1310): GG3+ prevalence = 1.3%

### Signed-distance derivation

**`signed_distance_midpoint_to_target_surface_mm`** is a deterministic recoding of `distance_midpoint_to_target_surface_mm`:
- **Negative** when `approximate_fraction_of_centerline_inside_target > 0` (needle partially inside the target mesh)
- **Positive** otherwise (needle fully outside the target mesh)

This is not a new measurement; it adds sign information that the unsigned distance discards, at the cost of one additional assumption about the sign convention.

### Interaction features

Four interaction terms are computed as simple products:

| Column | Formula |
|--------|---------|
| `inter_signed_dist_psa_density` | signed_dist × psa_density |
| `inter_frac_inside_psa_density` | frac_inside_target × psa_density |
| `inter_signed_dist_log_psa` | signed_dist × log_psa |
| `inter_frac_inside_log_psa` | frac_inside_target × log_psa |

These interaction terms capture potential PSA-severity modulation of needle-target proximity effects.  They increase model complexity and risk of overfitting on a small test set (~120 patients); they are included as exploratory sensitivity analyses only.

## Performance comparison

### GG2+ / csPCa

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train | n_val | n_test | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 8 | 0.754 | 0.699 | 0.807 | 0.327 | 0.230 | 0.461 | 3.050 | 0.463 | 0.703 | 0.706 | 0.223 | 0.339 | 11827 | 2748 | 2417 | 0.107 |
| current_compact | hist_gradient_boosting | 8 | 0.745 | 0.682 | 0.800 | 0.295 | 0.210 | 0.406 | 2.752 | 0.107 | 0.591 | 0.770 | 0.236 | 0.337 | 11827 | 2748 | 2417 | 0.107 |
| current_compact | xgboost | 8 | 0.754 | 0.694 | 0.808 | 0.321 | 0.230 | 0.450 | 2.999 | 0.521 | 0.595 | 0.789 | 0.253 | 0.355 | 11827 | 2748 | 2417 | 0.107 |
| compact_with_signed_distance | logistic_regression | 8 | 0.755 | 0.700 | 0.808 | 0.329 | 0.230 | 0.467 | 3.069 | 0.472 | 0.714 | 0.708 | 0.227 | 0.345 | 11827 | 2748 | 2417 | 0.107 |
| compact_with_signed_distance | hist_gradient_boosting | 8 | 0.737 | 0.671 | 0.795 | 0.282 | 0.201 | 0.389 | 2.630 | 0.089 | 0.668 | 0.723 | 0.225 | 0.336 | 11827 | 2748 | 2417 | 0.107 |
| compact_with_signed_distance | xgboost | 8 | 0.752 | 0.690 | 0.806 | 0.318 | 0.228 | 0.442 | 2.964 | 0.491 | 0.606 | 0.775 | 0.245 | 0.349 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_targeting_context | logistic_regression | 9 | 0.762 | 0.712 | 0.811 | 0.340 | 0.240 | 0.461 | 3.173 | 0.538 | 0.645 | 0.773 | 0.254 | 0.365 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_targeting_context | hist_gradient_boosting | 9 | 0.756 | 0.693 | 0.807 | 0.301 | 0.217 | 0.409 | 2.808 | 0.060 | 0.761 | 0.653 | 0.208 | 0.327 | 11827 | 2748 | 2417 | 0.107 |
| compact_plus_targeting_context | xgboost | 9 | 0.760 | 0.703 | 0.810 | 0.321 | 0.230 | 0.441 | 2.993 | 0.499 | 0.614 | 0.769 | 0.242 | 0.347 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_targeting_context | logistic_regression | 9 | 0.762 | 0.712 | 0.810 | 0.346 | 0.239 | 0.469 | 3.225 | 0.484 | 0.680 | 0.720 | 0.225 | 0.338 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_targeting_context | hist_gradient_boosting | 9 | 0.750 | 0.684 | 0.806 | 0.297 | 0.217 | 0.402 | 2.768 | 0.066 | 0.737 | 0.671 | 0.212 | 0.329 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_targeting_context | xgboost | 9 | 0.760 | 0.704 | 0.809 | 0.325 | 0.232 | 0.443 | 3.032 | 0.478 | 0.645 | 0.753 | 0.239 | 0.348 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_interactions | logistic_regression | 12 | 0.755 | 0.702 | 0.808 | 0.326 | 0.227 | 0.458 | 3.040 | 0.473 | 0.691 | 0.718 | 0.227 | 0.342 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_interactions | hist_gradient_boosting | 12 | 0.737 | 0.675 | 0.792 | 0.280 | 0.203 | 0.399 | 2.616 | 0.093 | 0.622 | 0.739 | 0.222 | 0.327 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_interactions | xgboost | 12 | 0.754 | 0.693 | 0.807 | 0.333 | 0.233 | 0.455 | 3.106 | 0.477 | 0.622 | 0.760 | 0.237 | 0.343 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_targeting_plus_inter | logistic_regression | 13 | 0.763 | 0.714 | 0.811 | 0.348 | 0.242 | 0.471 | 3.245 | 0.481 | 0.680 | 0.727 | 0.230 | 0.344 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_targeting_plus_inter | hist_gradient_boosting | 13 | 0.754 | 0.696 | 0.804 | 0.302 | 0.218 | 0.419 | 2.820 | 0.085 | 0.672 | 0.722 | 0.225 | 0.337 | 11827 | 2748 | 2417 | 0.107 |
| compact_signed_plus_targeting_plus_inter | xgboost | 13 | 0.762 | 0.707 | 0.811 | 0.327 | 0.232 | 0.447 | 3.052 | 0.501 | 0.618 | 0.772 | 0.245 | 0.351 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_targeting_context | logistic_regression | 5 | 0.743 | 0.693 | 0.789 | 0.343 | 0.238 | 0.455 | 3.199 | 0.589 | 0.514 | 0.803 | 0.238 | 0.325 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_targeting_context | hist_gradient_boosting | 5 | 0.727 | 0.670 | 0.780 | 0.282 | 0.204 | 0.372 | 2.636 | 0.086 | 0.664 | 0.713 | 0.217 | 0.327 | 11827 | 2748 | 2417 | 0.107 |
| clinical_plus_targeting_context | xgboost | 5 | 0.727 | 0.674 | 0.775 | 0.303 | 0.214 | 0.407 | 2.824 | 0.474 | 0.602 | 0.715 | 0.202 | 0.303 | 11827 | 2748 | 2417 | 0.107 |
| targeting_context_only | logistic_regression | 1 | 0.674 | 0.627 | 0.715 | 0.163 | 0.129 | 0.205 | 1.521 | 0.649 | 0.768 | 0.579 | 0.180 | 0.291 | 11827 | 2748 | 2417 | 0.107 |
| targeting_context_only | hist_gradient_boosting | 1 | 0.674 | 0.627 | 0.715 | 0.163 | 0.129 | 0.205 | 1.521 | 0.193 | 0.768 | 0.579 | 0.180 | 0.291 | 11827 | 2748 | 2417 | 0.107 |
| targeting_context_only | xgboost | 1 | 0.674 | 0.627 | 0.715 | 0.163 | 0.129 | 0.205 | 1.521 | 0.649 | 0.768 | 0.579 | 0.180 | 0.291 | 11827 | 2748 | 2417 | 0.107 |

### GG3+ / high-grade

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train | n_val | n_test | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 8 | 0.828 | 0.751 | 0.892 | 0.269 | 0.094 | 0.516 | 7.070 | 0.400 | 0.793 | 0.712 | 0.098 | 0.175 | 11827 | 2748 | 2417 | 0.038 |
| current_compact | hist_gradient_boosting | 8 | 0.808 | 0.717 | 0.881 | 0.255 | 0.083 | 0.469 | 6.693 | 0.018 | 0.663 | 0.768 | 0.101 | 0.176 | 11827 | 2748 | 2417 | 0.038 |
| current_compact | xgboost | 8 | 0.817 | 0.732 | 0.884 | 0.248 | 0.093 | 0.459 | 6.508 | 0.353 | 0.674 | 0.750 | 0.096 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| compact_with_signed_distance | logistic_regression | 8 | 0.821 | 0.740 | 0.888 | 0.271 | 0.095 | 0.523 | 7.128 | 0.387 | 0.793 | 0.690 | 0.092 | 0.165 | 11827 | 2748 | 2417 | 0.038 |
| compact_with_signed_distance | hist_gradient_boosting | 8 | 0.813 | 0.731 | 0.882 | 0.261 | 0.082 | 0.479 | 6.848 | 0.007 | 0.859 | 0.593 | 0.077 | 0.141 | 11827 | 2748 | 2417 | 0.038 |
| compact_with_signed_distance | xgboost | 8 | 0.817 | 0.735 | 0.884 | 0.276 | 0.092 | 0.470 | 7.262 | 0.307 | 0.717 | 0.718 | 0.091 | 0.162 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_targeting_context | logistic_regression | 9 | 0.829 | 0.752 | 0.892 | 0.286 | 0.093 | 0.523 | 7.527 | 0.401 | 0.772 | 0.717 | 0.098 | 0.173 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_targeting_context | hist_gradient_boosting | 9 | 0.822 | 0.740 | 0.887 | 0.267 | 0.088 | 0.485 | 7.017 | 0.044 | 0.576 | 0.877 | 0.156 | 0.245 | 11827 | 2748 | 2417 | 0.038 |
| compact_plus_targeting_context | xgboost | 9 | 0.812 | 0.727 | 0.882 | 0.261 | 0.090 | 0.461 | 6.854 | 0.327 | 0.685 | 0.727 | 0.090 | 0.160 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_targeting_context | logistic_regression | 9 | 0.821 | 0.740 | 0.886 | 0.276 | 0.093 | 0.526 | 7.262 | 0.392 | 0.783 | 0.698 | 0.093 | 0.166 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_targeting_context | hist_gradient_boosting | 9 | 0.801 | 0.709 | 0.875 | 0.238 | 0.074 | 0.454 | 6.255 | 0.011 | 0.772 | 0.627 | 0.076 | 0.138 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_targeting_context | xgboost | 9 | 0.813 | 0.730 | 0.880 | 0.251 | 0.087 | 0.445 | 6.596 | 0.318 | 0.696 | 0.724 | 0.091 | 0.161 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_interactions | logistic_regression | 12 | 0.818 | 0.736 | 0.885 | 0.225 | 0.088 | 0.463 | 5.914 | 0.381 | 0.783 | 0.689 | 0.090 | 0.162 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_interactions | hist_gradient_boosting | 12 | 0.816 | 0.728 | 0.885 | 0.236 | 0.077 | 0.449 | 6.199 | 0.007 | 0.848 | 0.619 | 0.081 | 0.148 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_interactions | xgboost | 12 | 0.810 | 0.727 | 0.878 | 0.258 | 0.087 | 0.456 | 6.785 | 0.345 | 0.674 | 0.738 | 0.092 | 0.162 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_targeting_plus_inter | logistic_regression | 13 | 0.818 | 0.736 | 0.884 | 0.239 | 0.090 | 0.490 | 6.281 | 0.407 | 0.772 | 0.716 | 0.097 | 0.173 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_targeting_plus_inter | hist_gradient_boosting | 13 | 0.807 | 0.720 | 0.879 | 0.244 | 0.077 | 0.462 | 6.406 | 0.012 | 0.717 | 0.694 | 0.085 | 0.152 | 11827 | 2748 | 2417 | 0.038 |
| compact_signed_plus_targeting_plus_inter | xgboost | 13 | 0.811 | 0.727 | 0.878 | 0.281 | 0.088 | 0.494 | 7.386 | 0.272 | 0.696 | 0.691 | 0.082 | 0.146 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_targeting_context | logistic_regression | 5 | 0.789 | 0.696 | 0.862 | 0.289 | 0.079 | 0.505 | 7.592 | 0.420 | 0.696 | 0.714 | 0.088 | 0.156 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_targeting_context | hist_gradient_boosting | 5 | 0.743 | 0.635 | 0.833 | 0.240 | 0.069 | 0.432 | 6.299 | 0.018 | 0.511 | 0.790 | 0.088 | 0.150 | 11827 | 2748 | 2417 | 0.038 |
| clinical_plus_targeting_context | xgboost | 5 | 0.771 | 0.670 | 0.853 | 0.284 | 0.080 | 0.500 | 7.452 | 0.315 | 0.728 | 0.683 | 0.083 | 0.150 | 11827 | 2748 | 2417 | 0.038 |
| targeting_context_only | logistic_regression | 1 | 0.686 | 0.612 | 0.744 | 0.062 | 0.039 | 0.096 | 1.636 | 0.618 | 0.815 | 0.556 | 0.068 | 0.125 | 11827 | 2748 | 2417 | 0.038 |
| targeting_context_only | hist_gradient_boosting | 1 | 0.686 | 0.612 | 0.744 | 0.062 | 0.039 | 0.096 | 1.636 | 0.063 | 0.815 | 0.556 | 0.068 | 0.125 | 11827 | 2748 | 2417 | 0.038 |
| targeting_context_only | xgboost | 1 | 0.686 | 0.612 | 0.744 | 0.062 | 0.039 | 0.096 | 1.636 | 0.618 | 0.815 | 0.556 | 0.068 | 0.125 | 11827 | 2748 | 2417 | 0.038 |

## Delta vs compact baseline (logistic regression)

### GG2+ / csPCa

Delta values = extended − current_compact (logistic regression, test split).  Positive = extension outperforms compact baseline.

| Feature set | n_feat | ROC-AUC | ΔROC | PR-AUC | ΔPR | top5% prev | Δtop5% | top10% cap | Δtop10% |
|-------------|--------|---------|------|--------|-----|---------|--------|-----------|---------|
| `current_compact` | 8 | 0.754 | +0.000 | 0.327 | +0.000 | — | — | — | — |
| `compact_with_signed_distance` | 8 | 0.755 | +0.001 | 0.329 | +0.002 | — | — | — | — |
| `compact_plus_targeting_context` | 9 | 0.762 | +0.008 | 0.340 | +0.013 | — | — | — | — |
| `compact_signed_plus_targeting_context` | 9 | 0.762 | +0.008 | 0.346 | +0.019 | — | — | — | — |
| `compact_signed_plus_interactions` | 12 | 0.755 | +0.001 | 0.326 | -0.001 | — | — | — | — |
| `compact_signed_plus_targeting_plus_inter` | 13 | 0.763 | +0.009 | 0.348 | +0.021 | — | — | — | — |
| `clinical_plus_targeting_context` | 5 | 0.743 | -0.011 | 0.343 | +0.016 | — | — | — | — |
| `targeting_context_only` | 1 | 0.674 | -0.080 | 0.163 | -0.164 | — | — | — | — |

### GG3+ / high-grade

Delta values = extended − current_compact (logistic regression, test split).  Positive = extension outperforms compact baseline.

| Feature set | n_feat | ROC-AUC | ΔROC | PR-AUC | ΔPR | top5% prev | Δtop5% | top10% cap | Δtop10% |
|-------------|--------|---------|------|--------|-----|---------|--------|-----------|---------|
| `current_compact` | 8 | 0.828 | +0.000 | 0.269 | +0.000 | — | — | — | — |
| `compact_with_signed_distance` | 8 | 0.821 | -0.007 | 0.271 | +0.002 | — | — | — | — |
| `compact_plus_targeting_context` | 9 | 0.829 | +0.001 | 0.286 | +0.017 | — | — | — | — |
| `compact_signed_plus_targeting_context` | 9 | 0.821 | -0.008 | 0.276 | +0.007 | — | — | — | — |
| `compact_signed_plus_interactions` | 12 | 0.818 | -0.010 | 0.225 | -0.044 | — | — | — | — |
| `compact_signed_plus_targeting_plus_inter` | 13 | 0.818 | -0.010 | 0.239 | -0.030 | — | — | — | — |
| `clinical_plus_targeting_context` | 5 | 0.789 | -0.039 | 0.289 | +0.020 | — | — | — | — |
| `targeting_context_only` | 1 | 0.686 | -0.143 | 0.062 | -0.207 | — | — | — | — |

