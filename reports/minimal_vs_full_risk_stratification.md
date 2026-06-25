# Risk Stratification

**Capture rate** = fraction of all test-set positives in the top-N% of scored cores.  
**Prevalence** in top-N% = positive rate among the top-N% highest-probability cores.  
**Baseline prevalence** = overall positive rate in the test set.

## GG2+ / csPCa

| feature_set | model | prevalence | top5pct_prevalence | top5pct_capture_rate | top10pct_prevalence | top10pct_capture_rate | top20pct_prevalence | top20pct_capture_rate |
|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 0.107 | 0.322 | 0.151 | 0.314 | 0.293 | 0.194 | 0.363 |
| clinical_only | hist_gradient_boosting | 0.107 | 0.314 | 0.147 | 0.260 | 0.243 | 0.174 | 0.324 |
| clinical_only | xgboost | 0.107 | 0.322 | 0.151 | 0.273 | 0.255 | 0.200 | 0.375 |
| target_geometry_only | logistic_regression | 0.107 | 0.264 | 0.124 | 0.256 | 0.239 | 0.246 | 0.459 |
| target_geometry_only | hist_gradient_boosting | 0.107 | 0.306 | 0.143 | 0.256 | 0.239 | 0.233 | 0.436 |
| target_geometry_only | xgboost | 0.107 | 0.264 | 0.124 | 0.273 | 0.255 | 0.231 | 0.432 |
| target_geometry_plus_clinical | logistic_regression | 0.107 | 0.504 | 0.236 | 0.343 | 0.320 | 0.269 | 0.502 |
| target_geometry_plus_clinical | hist_gradient_boosting | 0.107 | 0.463 | 0.216 | 0.318 | 0.297 | 0.258 | 0.483 |
| target_geometry_plus_clinical | xgboost | 0.107 | 0.496 | 0.232 | 0.343 | 0.320 | 0.264 | 0.494 |
| biopsy_prostate_geometry_plus_clinical | logistic_regression | 0.107 | 0.331 | 0.154 | 0.306 | 0.286 | 0.190 | 0.355 |
| biopsy_prostate_geometry_plus_clinical | hist_gradient_boosting | 0.107 | 0.380 | 0.178 | 0.277 | 0.259 | 0.196 | 0.367 |
| biopsy_prostate_geometry_plus_clinical | xgboost | 0.107 | 0.388 | 0.181 | 0.281 | 0.263 | 0.200 | 0.375 |
| all_geometry_no_clinical | logistic_regression | 0.107 | 0.248 | 0.116 | 0.281 | 0.263 | 0.238 | 0.444 |
| all_geometry_no_clinical | hist_gradient_boosting | 0.107 | 0.306 | 0.143 | 0.260 | 0.243 | 0.233 | 0.436 |
| all_geometry_no_clinical | xgboost | 0.107 | 0.256 | 0.120 | 0.264 | 0.247 | 0.244 | 0.456 |
| all_geometry_no_availability_plus_clinical | logistic_regression | 0.107 | 0.479 | 0.224 | 0.351 | 0.328 | 0.277 | 0.517 |
| all_geometry_no_availability_plus_clinical | hist_gradient_boosting | 0.107 | 0.488 | 0.228 | 0.351 | 0.328 | 0.252 | 0.471 |
| all_geometry_no_availability_plus_clinical | xgboost | 0.107 | 0.488 | 0.228 | 0.360 | 0.336 | 0.277 | 0.517 |

## GG3+ / high-grade

| feature_set | model | prevalence | top5pct_prevalence | top5pct_capture_rate | top10pct_prevalence | top10pct_capture_rate | top20pct_prevalence | top20pct_capture_rate |
|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 0.038 | 0.207 | 0.272 | 0.182 | 0.478 | 0.099 | 0.522 |
| clinical_only | hist_gradient_boosting | 0.038 | 0.240 | 0.315 | 0.149 | 0.391 | 0.091 | 0.478 |
| clinical_only | xgboost | 0.038 | 0.223 | 0.293 | 0.169 | 0.446 | 0.097 | 0.511 |
| target_geometry_only | logistic_regression | 0.038 | 0.157 | 0.207 | 0.120 | 0.315 | 0.112 | 0.587 |
| target_geometry_only | hist_gradient_boosting | 0.038 | 0.140 | 0.185 | 0.112 | 0.293 | 0.101 | 0.533 |
| target_geometry_only | xgboost | 0.038 | 0.174 | 0.228 | 0.128 | 0.337 | 0.097 | 0.511 |
| target_geometry_plus_clinical | logistic_regression | 0.038 | 0.314 | 0.413 | 0.190 | 0.500 | 0.110 | 0.576 |
| target_geometry_plus_clinical | hist_gradient_boosting | 0.038 | 0.248 | 0.326 | 0.190 | 0.500 | 0.116 | 0.609 |
| target_geometry_plus_clinical | xgboost | 0.038 | 0.331 | 0.435 | 0.190 | 0.500 | 0.110 | 0.576 |
| biopsy_prostate_geometry_plus_clinical | logistic_regression | 0.038 | 0.207 | 0.272 | 0.186 | 0.489 | 0.097 | 0.511 |
| biopsy_prostate_geometry_plus_clinical | hist_gradient_boosting | 0.038 | 0.215 | 0.283 | 0.153 | 0.402 | 0.101 | 0.533 |
| biopsy_prostate_geometry_plus_clinical | xgboost | 0.038 | 0.231 | 0.304 | 0.153 | 0.402 | 0.099 | 0.522 |
| all_geometry_no_clinical | logistic_regression | 0.038 | 0.149 | 0.196 | 0.128 | 0.337 | 0.114 | 0.598 |
| all_geometry_no_clinical | hist_gradient_boosting | 0.038 | 0.124 | 0.163 | 0.116 | 0.304 | 0.097 | 0.511 |
| all_geometry_no_clinical | xgboost | 0.038 | 0.165 | 0.217 | 0.132 | 0.348 | 0.099 | 0.522 |
| all_geometry_no_availability_plus_clinical | logistic_regression | 0.038 | 0.306 | 0.402 | 0.186 | 0.489 | 0.118 | 0.620 |
| all_geometry_no_availability_plus_clinical | hist_gradient_boosting | 0.038 | 0.264 | 0.348 | 0.174 | 0.457 | 0.112 | 0.587 |
| all_geometry_no_availability_plus_clinical | xgboost | 0.038 | 0.298 | 0.391 | 0.194 | 0.511 | 0.112 | 0.587 |

