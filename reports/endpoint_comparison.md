# Endpoint Comparison Report

Comparing two binary classification targets on the same biopsy cores.

## Endpoint Definitions

| Label column | Clinical definition |
|---|---|
| `binary_label_int` | GG2+ / csPCa (Gleason ≥ 3+4=7) |
| `binary_label_gg3plus_int` | GG3+ / high-grade (Gleason ≥ 4+3=7) |

## Dataset Statistics

Rows with `label_join_status == 'coord_match'`, valid label, valid split.

| Label column | Split | Rows | Patients | Positives | Prevalence |
|---|---|---|---|---|---|
| `binary_label_int` | train | 11,827 | 557 | 1,356 | 11.5% |
| `binary_label_int` | val | 2,748 | 119 | 272 | 9.9% |
| `binary_label_int` | test | 2,417 | 120 | 259 | 10.7% |
| `binary_label_int` | ALL | 16,992 | 796 | 1,887 | 11.1% |
| `binary_label_gg3plus_int` | train | 11,827 | 557 | 475 | 4.0% |
| `binary_label_gg3plus_int` | val | 2,748 | 119 | 80 | 2.9% |
| `binary_label_gg3plus_int` | test | 2,417 | 120 | 92 | 3.8% |
| `binary_label_gg3plus_int` | ALL | 16,992 | 796 | 647 | 3.8% |

> **Test-set difference:** switching from `binary_label_int` to `binary_label_gg3plus_int` removes 167 positive labels (10.7% → 3.8% prevalence).

## Model Results by Endpoint

- **baseline_pr_auc** = positive prevalence in the test split  
- **pr_auc_lift** = `test_pr_auc / baseline_pr_auc`  
  (1.0 = no-skill baseline; higher is better)

### `binary_label_int` — GG2+ / csPCa (Gleason ≥ 3+4=7)

#### full_geometry_dataset

| experiment | model | test_roc_auc | test_pr_auc | baseline_pr_auc | pr_auc_lift | test_sensitivity | test_specificity | test_f1 |
|---|---|---|---|---|---|---|---|---|
| all_geometry | logistic_regression | 0.724 | 0.217 | 0.107 | 2.03× | 0.741 | 0.610 | 0.297 |
| all_geometry | xgboost | 0.722 | 0.224 | 0.107 | 2.09× | 0.710 | 0.633 | 0.298 |
| all_geometry | hist_gradient_boosting | 0.714 | 0.218 | 0.107 | 2.03× | 0.637 | 0.683 | 0.298 |
| all_geometry_plus_clinical | xgboost | 0.758 | 0.327 | 0.107 | 3.05× | 0.552 | 0.803 | 0.345 |
| all_geometry_plus_clinical | logistic_regression | 0.753 | 0.321 | 0.107 | 2.99× | 0.629 | 0.764 | 0.350 |
| all_geometry_plus_clinical | hist_gradient_boosting | 0.736 | 0.298 | 0.107 | 2.78× | 0.614 | 0.740 | 0.324 |
| biopsy_geometry_only | xgboost | 0.515 | 0.116 | 0.107 | 1.08× | 0.595 | 0.392 | 0.179 |
| biopsy_geometry_only | hist_gradient_boosting | 0.504 | 0.111 | 0.107 | 1.04× | 0.649 | 0.359 | 0.186 |
| biopsy_geometry_only | logistic_regression | 0.459 | 0.094 | 0.107 | 0.88× | 0.803 | 0.163 | 0.183 |
| clinical_only | logistic_regression | 0.650 | 0.222 | 0.107 | 2.07× | 0.363 | 0.850 | 0.278 |
| clinical_only | xgboost | 0.625 | 0.240 | 0.107 | 2.24× | 0.324 | 0.863 | 0.263 |
| clinical_only | hist_gradient_boosting | 0.600 | 0.208 | 0.107 | 1.94× | 0.378 | 0.776 | 0.233 |
| target_geometry_only | logistic_regression | 0.730 | 0.224 | 0.107 | 2.09× | 0.664 | 0.681 | 0.307 |
| target_geometry_only | xgboost | 0.726 | 0.223 | 0.107 | 2.08× | 0.707 | 0.664 | 0.314 |
| target_geometry_only | hist_gradient_boosting | 0.717 | 0.218 | 0.107 | 2.03× | 0.622 | 0.705 | 0.305 |

#### extracted_voxel_subset

| experiment | model | test_roc_auc | test_pr_auc | baseline_pr_auc | pr_auc_lift | test_sensitivity | test_specificity | test_f1 |
|---|---|---|---|---|---|---|---|---|
| all_geometry | logistic_regression | 0.749 | 0.365 | 0.107 | 3.41× | 0.660 | 0.800 | 0.483 |
| all_geometry | xgboost | 0.692 | 0.332 | 0.107 | 3.09× | 0.774 | 0.498 | 0.346 |
| all_geometry | hist_gradient_boosting | 0.628 | 0.301 | 0.107 | 2.81× | 0.811 | 0.375 | 0.314 |
| centerline_plus_geometry | logistic_regression | 0.749 | 0.365 | 0.107 | 3.41× | 0.660 | 0.800 | 0.483 |
| centerline_plus_geometry | xgboost | 0.692 | 0.332 | 0.107 | 3.09× | 0.774 | 0.498 | 0.346 |
| centerline_plus_geometry | hist_gradient_boosting | 0.628 | 0.301 | 0.107 | 2.81× | 0.811 | 0.375 | 0.314 |
| centerline_plus_geometry_plus_clinical | xgboost | 0.701 | 0.345 | 0.107 | 3.22× | 0.302 | 0.954 | 0.390 |
| centerline_plus_geometry_plus_clinical | logistic_regression | 0.697 | 0.340 | 0.107 | 3.17× | 0.604 | 0.723 | 0.390 |
| centerline_plus_geometry_plus_clinical | hist_gradient_boosting | 0.673 | 0.332 | 0.107 | 3.10× | 0.245 | 0.933 | 0.306 |
| clinical_only | hist_gradient_boosting | 0.589 | 0.199 | 0.107 | 1.85× | 0.472 | 0.618 | 0.267 |
| clinical_only | xgboost | 0.577 | 0.202 | 0.107 | 1.89× | 0.264 | 0.814 | 0.233 |
| clinical_only | logistic_regression | 0.528 | 0.187 | 0.107 | 1.75× | 0.075 | 0.958 | 0.116 |
| target_geometry_only | logistic_regression | 0.757 | 0.363 | 0.107 | 3.39× | 0.717 | 0.705 | 0.434 |
| target_geometry_only | xgboost | 0.747 | 0.332 | 0.107 | 3.10× | 0.811 | 0.642 | 0.434 |
| target_geometry_only | hist_gradient_boosting | 0.696 | 0.322 | 0.107 | 3.00× | 0.566 | 0.733 | 0.377 |

### `binary_label_gg3plus_int` — GG3+ / high-grade (Gleason ≥ 4+3=7)

#### full_geometry_dataset

| experiment | model | test_roc_auc | test_pr_auc | baseline_pr_auc | pr_auc_lift | test_sensitivity | test_specificity | test_f1 |
|---|---|---|---|---|---|---|---|---|
| all_geometry | logistic_regression | 0.812 | 0.131 | 0.038 | 3.44× | 0.859 | 0.633 | 0.154 |
| all_geometry | xgboost | 0.779 | 0.133 | 0.038 | 3.51× | 0.880 | 0.568 | 0.138 |
| all_geometry | hist_gradient_boosting | 0.779 | 0.104 | 0.038 | 2.73× | 0.946 | 0.455 | 0.120 |
| all_geometry_plus_clinical | logistic_regression | 0.830 | 0.266 | 0.038 | 7.00× | 0.793 | 0.681 | 0.161 |
| all_geometry_plus_clinical | xgboost | 0.812 | 0.261 | 0.038 | 6.86× | 0.707 | 0.745 | 0.174 |
| all_geometry_plus_clinical | hist_gradient_boosting | 0.796 | 0.271 | 0.038 | 7.12× | 0.707 | 0.705 | 0.154 |
| biopsy_geometry_only | xgboost | 0.532 | 0.039 | 0.038 | 1.01× | 0.848 | 0.272 | 0.084 |
| biopsy_geometry_only | hist_gradient_boosting | 0.510 | 0.044 | 0.038 | 1.15× | 0.815 | 0.222 | 0.076 |
| biopsy_geometry_only | logistic_regression | 0.502 | 0.041 | 0.038 | 1.07× | 0.880 | 0.182 | 0.078 |
| clinical_only | xgboost | 0.737 | 0.152 | 0.038 | 4.00× | 0.446 | 0.871 | 0.190 |
| clinical_only | logistic_regression | 0.716 | 0.156 | 0.038 | 4.10× | 0.533 | 0.791 | 0.156 |
| clinical_only | hist_gradient_boosting | 0.710 | 0.188 | 0.038 | 4.93× | 0.620 | 0.625 | 0.112 |
| target_geometry_only | logistic_regression | 0.808 | 0.139 | 0.038 | 3.66× | 0.772 | 0.693 | 0.162 |
| target_geometry_only | xgboost | 0.796 | 0.122 | 0.038 | 3.21× | 0.739 | 0.702 | 0.159 |
| target_geometry_only | hist_gradient_boosting | 0.778 | 0.114 | 0.038 | 2.99× | 0.935 | 0.504 | 0.129 |

#### extracted_voxel_subset

| experiment | model | test_roc_auc | test_pr_auc | baseline_pr_auc | pr_auc_lift | test_sensitivity | test_specificity | test_f1 |
|---|---|---|---|---|---|---|---|---|
| all_geometry | logistic_regression | 0.843 | 0.153 | 0.038 | 4.02× | 0.588 | 0.866 | 0.286 |
| all_geometry | hist_gradient_boosting | 0.706 | 0.088 | 0.038 | 2.32× | 0.882 | 0.530 | 0.164 |
| all_geometry | xgboost | 0.672 | 0.077 | 0.038 | 2.04× | 0.824 | 0.583 | 0.170 |
| centerline_plus_geometry | logistic_regression | 0.843 | 0.153 | 0.038 | 4.02× | 0.588 | 0.866 | 0.286 |
| centerline_plus_geometry | hist_gradient_boosting | 0.706 | 0.088 | 0.038 | 2.32× | 0.882 | 0.530 | 0.164 |
| centerline_plus_geometry | xgboost | 0.672 | 0.077 | 0.038 | 2.04× | 0.824 | 0.583 | 0.170 |
| centerline_plus_geometry_plus_clinical | logistic_regression | 0.756 | 0.131 | 0.038 | 3.45× | 0.235 | 0.953 | 0.222 |
| centerline_plus_geometry_plus_clinical | xgboost | 0.647 | 0.196 | 0.038 | 5.15× | 0.235 | 0.950 | 0.216 |
| centerline_plus_geometry_plus_clinical | hist_gradient_boosting | 0.547 | 0.177 | 0.038 | 4.65× | 0.235 | 0.900 | 0.151 |
| clinical_only | hist_gradient_boosting | 0.535 | 0.066 | 0.038 | 1.73× | 0.235 | 0.857 | 0.119 |
| clinical_only | xgboost | 0.490 | 0.081 | 0.038 | 2.12× | 0.235 | 0.857 | 0.119 |
| clinical_only | logistic_regression | 0.438 | 0.058 | 0.038 | 1.53× | 0.176 | 0.910 | 0.122 |
| target_geometry_only | logistic_regression | 0.795 | 0.142 | 0.038 | 3.74× | 0.588 | 0.748 | 0.185 |
| target_geometry_only | hist_gradient_boosting | 0.649 | 0.086 | 0.038 | 2.26× | 0.412 | 0.698 | 0.116 |
| target_geometry_only | xgboost | 0.647 | 0.086 | 0.038 | 2.27× | 0.765 | 0.417 | 0.120 |

## Side-by-side: Best Model per Experiment

Best model selected by `test_roc_auc` within each (endpoint, mode, experiment).

| experiment | mode | best model | `binary_label_int` ROC-AUC | `binary_label_gg3plus_int` ROC-AUC | `binary_label_int` PR-lift | `binary_label_gg3plus_int` PR-lift |
|---|---|---|---|---|---|---|---|
| all_geometry | extracted_voxel_subset | logistic_regression | 0.749 | 0.843 | 3.41× | 4.02× |
| centerline_plus_geometry | extracted_voxel_subset | logistic_regression | 0.749 | 0.843 | 3.41× | 4.02× |
| centerline_plus_geometry_plus_clinical | extracted_voxel_subset | xgboost / logistic_regression | 0.701 | 0.756 | 3.22× | 3.45× |
| clinical_only | extracted_voxel_subset | hist_gradient_boosting | 0.589 | 0.535 | 1.85× | 1.73× |
| target_geometry_only | extracted_voxel_subset | logistic_regression | 0.757 | 0.795 | 3.39× | 3.74× |
| all_geometry | full_geometry_dataset | logistic_regression | 0.724 | 0.812 | 2.03× | 3.44× |
| all_geometry_plus_clinical | full_geometry_dataset | xgboost / logistic_regression | 0.758 | 0.830 | 3.05× | 7.00× |
| biopsy_geometry_only | full_geometry_dataset | xgboost | 0.515 | 0.532 | 1.08× | 1.01× |
| clinical_only | full_geometry_dataset | logistic_regression / xgboost | 0.650 | 0.737 | 2.07× | 4.00× |
| target_geometry_only | full_geometry_dataset | logistic_regression | 0.730 | 0.808 | 2.09× | 3.66× |

