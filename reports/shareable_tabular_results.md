# Shareable Tabular Experiment Results


## full_geometry_dataset

| experiment | model | n_features | test_roc_auc | test_pr_auc | test_sensitivity | test_specificity | test_precision | test_recall | test_f1 | threshold_youden_val |
|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.650 | 0.222 | 0.363 | 0.850 | 0.225 | 0.363 | 0.278 | 0.562 |
| clinical_only | hist_gradient_boosting | 4 | 0.600 | 0.208 | 0.378 | 0.776 | 0.169 | 0.378 | 0.233 | 0.138 |
| clinical_only | xgboost | 4 | 0.625 | 0.240 | 0.324 | 0.863 | 0.222 | 0.324 | 0.263 | 0.615 |
| biopsy_geometry_only | logistic_regression | 6 | 0.459 | 0.094 | 0.803 | 0.163 | 0.103 | 0.803 | 0.183 | 0.464 |
| biopsy_geometry_only | hist_gradient_boosting | 6 | 0.504 | 0.111 | 0.649 | 0.359 | 0.108 | 0.649 | 0.186 | 0.107 |
| biopsy_geometry_only | xgboost | 6 | 0.515 | 0.116 | 0.595 | 0.392 | 0.105 | 0.595 | 0.179 | 0.472 |
| target_geometry_only | logistic_regression | 5 | 0.730 | 0.224 | 0.664 | 0.681 | 0.200 | 0.664 | 0.307 | 0.510 |
| target_geometry_only | hist_gradient_boosting | 5 | 0.717 | 0.218 | 0.622 | 0.705 | 0.202 | 0.622 | 0.305 | 0.132 |
| target_geometry_only | xgboost | 5 | 0.726 | 0.223 | 0.707 | 0.664 | 0.202 | 0.707 | 0.314 | 0.510 |
| all_geometry | logistic_regression | 11 | 0.724 | 0.217 | 0.741 | 0.610 | 0.186 | 0.741 | 0.297 | 0.453 |
| all_geometry | hist_gradient_boosting | 11 | 0.714 | 0.218 | 0.637 | 0.683 | 0.194 | 0.637 | 0.298 | 0.118 |
| all_geometry | xgboost | 11 | 0.722 | 0.224 | 0.710 | 0.633 | 0.189 | 0.710 | 0.298 | 0.472 |
| all_geometry_plus_clinical | logistic_regression | 15 | 0.753 | 0.321 | 0.629 | 0.764 | 0.243 | 0.629 | 0.350 | 0.530 |
| all_geometry_plus_clinical | hist_gradient_boosting | 15 | 0.736 | 0.298 | 0.614 | 0.740 | 0.221 | 0.614 | 0.324 | 0.096 |
| all_geometry_plus_clinical | xgboost | 15 | 0.758 | 0.327 | 0.552 | 0.803 | 0.251 | 0.552 | 0.345 | 0.547 |

## extracted_voxel_subset

| experiment | model | n_features | test_roc_auc | test_pr_auc | test_sensitivity | test_specificity | test_precision | test_recall | test_f1 | threshold_youden_val |
|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.528 | 0.187 | 0.075 | 0.958 | 0.250 | 0.075 | 0.116 | 0.812 |
| clinical_only | hist_gradient_boosting | 4 | 0.589 | 0.199 | 0.472 | 0.618 | 0.187 | 0.472 | 0.267 | 0.052 |
| clinical_only | xgboost | 4 | 0.577 | 0.202 | 0.264 | 0.814 | 0.209 | 0.264 | 0.233 | 0.654 |
| target_geometry_only | logistic_regression | 5 | 0.757 | 0.363 | 0.717 | 0.705 | 0.311 | 0.717 | 0.434 | 0.478 |
| target_geometry_only | hist_gradient_boosting | 5 | 0.696 | 0.322 | 0.566 | 0.733 | 0.283 | 0.566 | 0.377 | 0.035 |
| target_geometry_only | xgboost | 5 | 0.747 | 0.332 | 0.811 | 0.642 | 0.297 | 0.811 | 0.434 | 0.283 |
| all_geometry | logistic_regression | 11 | 0.749 | 0.365 | 0.660 | 0.800 | 0.380 | 0.660 | 0.483 | 0.523 |
| all_geometry | hist_gradient_boosting | 11 | 0.628 | 0.301 | 0.811 | 0.375 | 0.195 | 0.811 | 0.314 | 0.001 |
| all_geometry | xgboost | 11 | 0.692 | 0.332 | 0.774 | 0.498 | 0.223 | 0.774 | 0.346 | 0.177 |
| centerline_plus_geometry | logistic_regression | 11 | 0.749 | 0.365 | 0.660 | 0.800 | 0.380 | 0.660 | 0.483 | 0.523 |
| centerline_plus_geometry | hist_gradient_boosting | 11 | 0.628 | 0.301 | 0.811 | 0.375 | 0.195 | 0.811 | 0.314 | 0.001 |
| centerline_plus_geometry | xgboost | 11 | 0.692 | 0.332 | 0.774 | 0.498 | 0.223 | 0.774 | 0.346 | 0.177 |
| centerline_plus_geometry_plus_clinical | logistic_regression | 15 | 0.697 | 0.340 | 0.604 | 0.723 | 0.288 | 0.604 | 0.390 | 0.434 |
| centerline_plus_geometry_plus_clinical | hist_gradient_boosting | 15 | 0.673 | 0.332 | 0.245 | 0.933 | 0.406 | 0.245 | 0.306 | 0.156 |
| centerline_plus_geometry_plus_clinical | xgboost | 15 | 0.701 | 0.345 | 0.302 | 0.954 | 0.552 | 0.302 | 0.390 | 0.596 |
