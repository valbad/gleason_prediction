# Minimal vs Full Model Comparison

## Methodology

**Data:** `data/share/needle_features_v1.csv`, filtered to `label_join_status == "coord_match"` with pre-assigned `split` column.  
**Endpoints:** GG2+ / csPCa (`binary_label_int`) and GG3+ / high-grade (`binary_label_gg3plus_int`).  
**Models:** logistic regression (class-balanced, standardised features) and HistGradientBoosting; XGBoost included when installed.  
**Threshold selection:** Youden J maximised on the validation split.  
**Bootstrap CIs:** 1,000 patient-level resamples of the test set (patients resampled with replacement, all their cores included; 2.5/97.5 percentiles reported).  
**Reference feature set:** `all_geometry_no_availability_plus_clinical` (biopsy/prostate geometry ×6 + target geometry no availability ×4 + clinical ×4).  
`target_mesh_available` is excluded from all feature sets (conservative ablation; mesh-absent cores are rare but have elevated positivity).

## Performance

### GG2+ / csPCa

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train | n_val | n_test | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.650 | 0.579 | 0.720 | 0.222 | 0.150 | 0.326 | 2.071 | 0.562 | 0.363 | 0.850 | 0.225 | 0.278 | 11827 | 2748 | 2417 | 0.107 |
| clinical_only | hist_gradient_boosting | 4 | 0.600 | 0.519 | 0.675 | 0.208 | 0.126 | 0.331 | 1.941 | 0.138 | 0.378 | 0.776 | 0.169 | 0.233 | 11827 | 2748 | 2417 | 0.107 |
| clinical_only | xgboost | 4 | 0.625 | 0.551 | 0.695 | 0.240 | 0.136 | 0.360 | 2.240 | 0.615 | 0.324 | 0.863 | 0.222 | 0.263 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_only | logistic_regression | 4 | 0.730 | 0.674 | 0.777 | 0.224 | 0.162 | 0.318 | 2.090 | 0.494 | 0.707 | 0.653 | 0.197 | 0.308 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_only | hist_gradient_boosting | 4 | 0.717 | 0.664 | 0.762 | 0.218 | 0.164 | 0.300 | 2.033 | 0.132 | 0.622 | 0.705 | 0.202 | 0.305 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_only | xgboost | 4 | 0.726 | 0.674 | 0.771 | 0.224 | 0.166 | 0.312 | 2.092 | 0.528 | 0.660 | 0.677 | 0.197 | 0.304 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_clinical | logistic_regression | 8 | 0.754 | 0.699 | 0.807 | 0.327 | 0.230 | 0.461 | 3.050 | 0.463 | 0.703 | 0.706 | 0.223 | 0.339 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_clinical | hist_gradient_boosting | 8 | 0.745 | 0.682 | 0.800 | 0.295 | 0.210 | 0.406 | 2.752 | 0.107 | 0.591 | 0.770 | 0.236 | 0.337 | 11827 | 2748 | 2417 | 0.107 |
| target_geometry_plus_clinical | xgboost | 8 | 0.754 | 0.694 | 0.808 | 0.321 | 0.230 | 0.450 | 2.999 | 0.521 | 0.595 | 0.789 | 0.253 | 0.355 | 11827 | 2748 | 2417 | 0.107 |
| biopsy_prostate_geometry_plus_clinical | logistic_regression | 10 | 0.645 | 0.574 | 0.715 | 0.207 | 0.142 | 0.322 | 1.936 | 0.461 | 0.533 | 0.649 | 0.154 | 0.239 | 11827 | 2748 | 2417 | 0.107 |
| biopsy_prostate_geometry_plus_clinical | hist_gradient_boosting | 10 | 0.630 | 0.554 | 0.698 | 0.238 | 0.136 | 0.363 | 2.219 | 0.126 | 0.425 | 0.736 | 0.162 | 0.234 | 11827 | 2748 | 2417 | 0.107 |
| biopsy_prostate_geometry_plus_clinical | xgboost | 10 | 0.644 | 0.568 | 0.713 | 0.254 | 0.146 | 0.378 | 2.371 | 0.487 | 0.429 | 0.736 | 0.163 | 0.236 | 11827 | 2748 | 2417 | 0.107 |
| all_geometry_no_clinical | logistic_regression | 10 | 0.723 | 0.669 | 0.774 | 0.217 | 0.158 | 0.308 | 2.021 | 0.458 | 0.741 | 0.609 | 0.185 | 0.297 | 11827 | 2748 | 2417 | 0.107 |
| all_geometry_no_clinical | hist_gradient_boosting | 10 | 0.714 | 0.663 | 0.759 | 0.218 | 0.164 | 0.293 | 2.030 | 0.118 | 0.637 | 0.683 | 0.194 | 0.298 | 11827 | 2748 | 2417 | 0.107 |
| all_geometry_no_clinical | xgboost | 10 | 0.721 | 0.670 | 0.767 | 0.223 | 0.167 | 0.304 | 2.082 | 0.456 | 0.718 | 0.614 | 0.183 | 0.291 | 11827 | 2748 | 2417 | 0.107 |
| all_geometry_no_availability_plus_clinical | logistic_regression | 14 | 0.753 | 0.698 | 0.804 | 0.319 | 0.225 | 0.449 | 2.981 | 0.520 | 0.645 | 0.759 | 0.243 | 0.353 | 11827 | 2748 | 2417 | 0.107 |
| all_geometry_no_availability_plus_clinical | hist_gradient_boosting | 14 | 0.736 | 0.675 | 0.790 | 0.298 | 0.212 | 0.411 | 2.778 | 0.096 | 0.614 | 0.740 | 0.221 | 0.324 | 11827 | 2748 | 2417 | 0.107 |
| all_geometry_no_availability_plus_clinical | xgboost | 14 | 0.758 | 0.699 | 0.811 | 0.327 | 0.233 | 0.447 | 3.052 | 0.547 | 0.552 | 0.803 | 0.251 | 0.345 | 11827 | 2748 | 2417 | 0.107 |

### GG3+ / high-grade

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train | n_val | n_test | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.716 | 0.582 | 0.819 | 0.156 | 0.059 | 0.334 | 4.101 | 0.470 | 0.533 | 0.791 | 0.091 | 0.156 | 11827 | 2748 | 2417 | 0.038 |
| clinical_only | hist_gradient_boosting | 4 | 0.710 | 0.585 | 0.809 | 0.188 | 0.051 | 0.370 | 4.931 | 0.008 | 0.620 | 0.625 | 0.061 | 0.112 | 11827 | 2748 | 2417 | 0.038 |
| clinical_only | xgboost | 4 | 0.737 | 0.618 | 0.833 | 0.152 | 0.054 | 0.368 | 3.997 | 0.571 | 0.446 | 0.871 | 0.121 | 0.190 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_only | logistic_regression | 4 | 0.806 | 0.734 | 0.864 | 0.138 | 0.057 | 0.284 | 3.636 | 0.500 | 0.772 | 0.680 | 0.087 | 0.156 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_only | hist_gradient_boosting | 4 | 0.778 | 0.707 | 0.835 | 0.114 | 0.052 | 0.232 | 2.987 | 0.019 | 0.935 | 0.504 | 0.069 | 0.129 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_only | xgboost | 4 | 0.795 | 0.720 | 0.855 | 0.120 | 0.054 | 0.249 | 3.149 | 0.459 | 0.783 | 0.669 | 0.086 | 0.154 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_clinical | logistic_regression | 8 | 0.828 | 0.751 | 0.892 | 0.269 | 0.094 | 0.516 | 7.070 | 0.400 | 0.793 | 0.712 | 0.098 | 0.175 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_clinical | hist_gradient_boosting | 8 | 0.808 | 0.717 | 0.881 | 0.255 | 0.083 | 0.469 | 6.693 | 0.018 | 0.663 | 0.768 | 0.101 | 0.176 | 11827 | 2748 | 2417 | 0.038 |
| target_geometry_plus_clinical | xgboost | 8 | 0.817 | 0.732 | 0.884 | 0.248 | 0.093 | 0.459 | 6.508 | 0.353 | 0.674 | 0.750 | 0.096 | 0.168 | 11827 | 2748 | 2417 | 0.038 |
| biopsy_prostate_geometry_plus_clinical | logistic_regression | 10 | 0.720 | 0.591 | 0.823 | 0.154 | 0.057 | 0.362 | 4.036 | 0.496 | 0.511 | 0.806 | 0.095 | 0.160 | 11827 | 2748 | 2417 | 0.038 |
| biopsy_prostate_geometry_plus_clinical | hist_gradient_boosting | 10 | 0.735 | 0.619 | 0.834 | 0.194 | 0.053 | 0.384 | 5.106 | 0.037 | 0.511 | 0.849 | 0.118 | 0.192 | 11827 | 2748 | 2417 | 0.038 |
| biopsy_prostate_geometry_plus_clinical | xgboost | 10 | 0.724 | 0.585 | 0.830 | 0.180 | 0.049 | 0.404 | 4.723 | 0.502 | 0.478 | 0.836 | 0.104 | 0.170 | 11827 | 2748 | 2417 | 0.038 |
| all_geometry_no_clinical | logistic_regression | 10 | 0.809 | 0.738 | 0.866 | 0.127 | 0.058 | 0.261 | 3.336 | 0.464 | 0.870 | 0.633 | 0.086 | 0.156 | 11827 | 2748 | 2417 | 0.038 |
| all_geometry_no_clinical | hist_gradient_boosting | 10 | 0.779 | 0.712 | 0.833 | 0.104 | 0.050 | 0.217 | 2.730 | 0.017 | 0.946 | 0.455 | 0.064 | 0.120 | 11827 | 2748 | 2417 | 0.038 |
| all_geometry_no_clinical | xgboost | 10 | 0.778 | 0.697 | 0.848 | 0.131 | 0.052 | 0.284 | 3.447 | 0.373 | 0.848 | 0.572 | 0.073 | 0.134 | 11827 | 2748 | 2417 | 0.038 |
| all_geometry_no_availability_plus_clinical | logistic_regression | 14 | 0.829 | 0.752 | 0.889 | 0.265 | 0.096 | 0.495 | 6.961 | 0.342 | 0.815 | 0.657 | 0.086 | 0.156 | 11827 | 2748 | 2417 | 0.038 |
| all_geometry_no_availability_plus_clinical | hist_gradient_boosting | 14 | 0.796 | 0.698 | 0.876 | 0.271 | 0.080 | 0.482 | 7.118 | 0.014 | 0.707 | 0.705 | 0.087 | 0.154 | 11827 | 2748 | 2417 | 0.038 |
| all_geometry_no_availability_plus_clinical | xgboost | 14 | 0.812 | 0.723 | 0.884 | 0.261 | 0.091 | 0.480 | 6.860 | 0.339 | 0.707 | 0.745 | 0.099 | 0.174 | 11827 | 2748 | 2417 | 0.038 |

## Interpretation

### GG2+ / csPCa

**Signal capture (target geometry + clinical vs full model):** `target_geometry_plus_clinical` (mean ROC-AUC 0.751) captures most of the full model signal (mean ROC-AUC 0.749; ΔROC = -0.002, ΔPR = 0.000). The compact model is preferred on parsimony grounds.

**Biopsy/prostate geometry incremental value:** Adding biopsy/prostate geometry to `target_geometry_plus_clinical` yields limited gain (ΔROC = -0.002, ΔPR = 0.000 for `all_geometry_no_availability_plus_clinical` vs `target_geometry_plus_clinical`). Target-relative geometry carries most of the geometric signal; the compact model may be preferred on parsimony grounds.

**Geometry-only increment (without clinical features):** `all_geometry_no_clinical` adds limited gain over `target_geometry_only` (ΔROC = -0.005, ΔPR = -0.003), consistent with target-relative geometry driving most of the geometric signal.

**Note on `biopsy_prostate_geometry_plus_clinical`:** This feature set (non-target biopsy/prostate geometry + clinical, mean ROC-AUC 0.640) serves as a negative-control comparison rather than a direct incremental-value test: it confirms that target-relative geometry, not biopsy/prostate geometry alone, drives model performance.

### GG3+ / high-grade

**Signal capture (target geometry + clinical vs full model):** `target_geometry_plus_clinical` (mean ROC-AUC 0.818) captures most of the full model signal (mean ROC-AUC 0.813; ΔROC = -0.005, ΔPR = 0.008). The compact model is preferred on parsimony grounds.

**Biopsy/prostate geometry incremental value:** Adding biopsy/prostate geometry to `target_geometry_plus_clinical` yields limited gain (ΔROC = -0.005, ΔPR = 0.008 for `all_geometry_no_availability_plus_clinical` vs `target_geometry_plus_clinical`). Target-relative geometry carries most of the geometric signal; the compact model may be preferred on parsimony grounds.

**Geometry-only increment (without clinical features):** `all_geometry_no_clinical` adds limited gain over `target_geometry_only` (ΔROC = -0.004, ΔPR = -0.003), consistent with target-relative geometry driving most of the geometric signal.

**Note on `biopsy_prostate_geometry_plus_clinical`:** This feature set (non-target biopsy/prostate geometry + clinical, mean ROC-AUC 0.726) serves as a negative-control comparison rather than a direct incremental-value test: it confirms that target-relative geometry, not biopsy/prostate geometry alone, drives model performance.

### Consistency across endpoints

Although the full feature set (`all_geometry_no_availability_plus_clinical`) occasionally gives the highest raw ROC-AUC, `target_geometry_plus_clinical` remains within the predefined equivalence margin for both endpoints (GG2+: ΔROC = -0.002, ΔPR = 0.000; GG3+: ΔROC = -0.005, ΔPR = 0.008). Therefore, the compact model is the preferred central model on parsimony and interpretability grounds, while the full geometry-clinical model should be reported as a sensitivity/reference analysis.

