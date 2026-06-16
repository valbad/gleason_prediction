# Shareable Tabular Experiment Results

Label column: `binary_label_gg3plus_int`


## full_geometry_dataset

| experiment | model | n_features | test_roc_auc | test_pr_auc | test_sensitivity | test_specificity | test_precision | test_recall | test_f1 | threshold_youden_val |
|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.716 | 0.156 | 0.533 | 0.791 | 0.091 | 0.533 | 0.156 | 0.470 |
| clinical_only | hist_gradient_boosting | 4 | 0.710 | 0.188 | 0.620 | 0.625 | 0.061 | 0.620 | 0.112 | 0.008 |
| clinical_only | xgboost | 4 | 0.737 | 0.152 | 0.446 | 0.871 | 0.121 | 0.446 | 0.190 | 0.571 |
| biopsy_geometry_only | logistic_regression | 6 | 0.502 | 0.041 | 0.880 | 0.182 | 0.041 | 0.880 | 0.078 | 0.477 |
| biopsy_geometry_only | hist_gradient_boosting | 6 | 0.510 | 0.044 | 0.815 | 0.222 | 0.040 | 0.815 | 0.076 | 0.032 |
| biopsy_geometry_only | xgboost | 6 | 0.532 | 0.039 | 0.848 | 0.272 | 0.044 | 0.848 | 0.084 | 0.389 |
| target_geometry_only | logistic_regression | 5 | 0.808 | 0.139 | 0.772 | 0.693 | 0.090 | 0.772 | 0.162 | 0.494 |
| target_geometry_only | hist_gradient_boosting | 5 | 0.778 | 0.114 | 0.935 | 0.504 | 0.069 | 0.935 | 0.129 | 0.019 |
| target_geometry_only | xgboost | 5 | 0.796 | 0.122 | 0.739 | 0.702 | 0.089 | 0.739 | 0.159 | 0.477 |
| target_geometry_no_availability | logistic_regression | 4 | 0.806 | 0.138 | 0.772 | 0.680 | 0.087 | 0.772 | 0.156 | 0.500 |
| target_geometry_no_availability | hist_gradient_boosting | 4 | 0.778 | 0.114 | 0.935 | 0.504 | 0.069 | 0.935 | 0.129 | 0.019 |
| target_geometry_no_availability | xgboost | 4 | 0.795 | 0.120 | 0.783 | 0.669 | 0.086 | 0.783 | 0.154 | 0.459 |
| all_geometry | logistic_regression | 11 | 0.812 | 0.131 | 0.859 | 0.633 | 0.085 | 0.859 | 0.154 | 0.453 |
| all_geometry | hist_gradient_boosting | 11 | 0.779 | 0.104 | 0.946 | 0.455 | 0.064 | 0.946 | 0.120 | 0.017 |
| all_geometry | xgboost | 11 | 0.779 | 0.133 | 0.880 | 0.568 | 0.075 | 0.880 | 0.138 | 0.351 |
| all_geometry_no_availability | logistic_regression | 10 | 0.809 | 0.127 | 0.870 | 0.633 | 0.086 | 0.870 | 0.156 | 0.464 |
| all_geometry_no_availability | hist_gradient_boosting | 10 | 0.779 | 0.104 | 0.946 | 0.455 | 0.064 | 0.946 | 0.120 | 0.017 |
| all_geometry_no_availability | xgboost | 10 | 0.778 | 0.131 | 0.848 | 0.572 | 0.073 | 0.848 | 0.134 | 0.373 |
| all_geometry_plus_clinical | logistic_regression | 15 | 0.830 | 0.266 | 0.793 | 0.681 | 0.090 | 0.793 | 0.161 | 0.370 |
| all_geometry_plus_clinical | hist_gradient_boosting | 15 | 0.796 | 0.271 | 0.707 | 0.705 | 0.087 | 0.707 | 0.154 | 0.014 |
| all_geometry_plus_clinical | xgboost | 15 | 0.812 | 0.261 | 0.707 | 0.745 | 0.099 | 0.707 | 0.174 | 0.339 |
| all_geometry_no_availability_plus_clinical | logistic_regression | 14 | 0.829 | 0.265 | 0.815 | 0.657 | 0.086 | 0.815 | 0.156 | 0.342 |
| all_geometry_no_availability_plus_clinical | hist_gradient_boosting | 14 | 0.796 | 0.271 | 0.707 | 0.705 | 0.087 | 0.707 | 0.154 | 0.014 |
| all_geometry_no_availability_plus_clinical | xgboost | 14 | 0.812 | 0.261 | 0.707 | 0.745 | 0.099 | 0.707 | 0.174 | 0.339 |

## extracted_voxel_subset

| experiment | model | n_features | test_roc_auc | test_pr_auc | test_sensitivity | test_specificity | test_precision | test_recall | test_f1 | threshold_youden_val |
|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.438 | 0.058 | 0.176 | 0.910 | 0.094 | 0.176 | 0.122 | 0.817 |
| clinical_only | hist_gradient_boosting | 4 | 0.535 | 0.066 | 0.235 | 0.857 | 0.080 | 0.235 | 0.119 | 0.005 |
| clinical_only | xgboost | 4 | 0.490 | 0.081 | 0.235 | 0.857 | 0.080 | 0.235 | 0.119 | 0.446 |
| target_geometry_only | logistic_regression | 5 | 0.795 | 0.142 | 0.588 | 0.748 | 0.110 | 0.588 | 0.185 | 0.510 |
| target_geometry_only | hist_gradient_boosting | 5 | 0.649 | 0.086 | 0.412 | 0.698 | 0.067 | 0.412 | 0.116 | 0.002 |
| target_geometry_only | xgboost | 5 | 0.647 | 0.086 | 0.765 | 0.417 | 0.065 | 0.765 | 0.120 | 0.037 |
| target_geometry_no_availability | logistic_regression | 4 | 0.795 | 0.142 | 0.588 | 0.748 | 0.110 | 0.588 | 0.185 | 0.510 |
| target_geometry_no_availability | hist_gradient_boosting | 4 | 0.649 | 0.086 | 0.412 | 0.698 | 0.067 | 0.412 | 0.116 | 0.002 |
| target_geometry_no_availability | xgboost | 4 | 0.647 | 0.086 | 0.765 | 0.417 | 0.065 | 0.765 | 0.120 | 0.037 |
| all_geometry | logistic_regression | 11 | 0.843 | 0.153 | 0.588 | 0.866 | 0.189 | 0.588 | 0.286 | 0.580 |
| all_geometry | hist_gradient_boosting | 11 | 0.706 | 0.088 | 0.882 | 0.530 | 0.090 | 0.882 | 0.164 | 0.000 |
| all_geometry | xgboost | 11 | 0.672 | 0.077 | 0.824 | 0.583 | 0.095 | 0.824 | 0.170 | 0.068 |
| all_geometry_no_availability | logistic_regression | 10 | 0.844 | 0.154 | 0.588 | 0.866 | 0.189 | 0.588 | 0.286 | 0.580 |
| all_geometry_no_availability | hist_gradient_boosting | 10 | 0.706 | 0.088 | 0.882 | 0.530 | 0.090 | 0.882 | 0.164 | 0.000 |
| all_geometry_no_availability | xgboost | 10 | 0.672 | 0.077 | 0.824 | 0.583 | 0.095 | 0.824 | 0.170 | 0.068 |
| centerline_plus_geometry | logistic_regression | 11 | 0.843 | 0.153 | 0.588 | 0.866 | 0.189 | 0.588 | 0.286 | 0.580 |
| centerline_plus_geometry | hist_gradient_boosting | 11 | 0.706 | 0.088 | 0.882 | 0.530 | 0.090 | 0.882 | 0.164 | 0.000 |
| centerline_plus_geometry | xgboost | 11 | 0.672 | 0.077 | 0.824 | 0.583 | 0.095 | 0.824 | 0.170 | 0.068 |
| centerline_plus_geometry_plus_clinical | logistic_regression | 15 | 0.756 | 0.131 | 0.235 | 0.953 | 0.211 | 0.235 | 0.222 | 0.763 |
| centerline_plus_geometry_plus_clinical | hist_gradient_boosting | 15 | 0.547 | 0.177 | 0.235 | 0.900 | 0.111 | 0.235 | 0.151 | 0.018 |
| centerline_plus_geometry_plus_clinical | xgboost | 15 | 0.647 | 0.196 | 0.235 | 0.950 | 0.200 | 0.235 | 0.216 | 0.359 |
| centerline_plus_geometry_no_availability_plus_clinical | logistic_regression | 14 | 0.757 | 0.132 | 0.235 | 0.953 | 0.211 | 0.235 | 0.222 | 0.762 |
| centerline_plus_geometry_no_availability_plus_clinical | hist_gradient_boosting | 14 | 0.547 | 0.177 | 0.235 | 0.900 | 0.111 | 0.235 | 0.151 | 0.018 |
| centerline_plus_geometry_no_availability_plus_clinical | xgboost | 14 | 0.647 | 0.196 | 0.235 | 0.950 | 0.200 | 0.235 | 0.216 | 0.359 |
