# Final Targeted Feature Sprint — Risk Stratification

**Capture rate** = fraction of all test positives in the top-N% scored cores.  
**Prevalence** in top-N% = positive rate among top-N% highest-probability cores.  
**Enrichment** = top-N% prevalence / baseline prevalence.  
**Baseline:** overall positive rate in the test set.

**Note:** feature sets containing `core_label` are PROCEDURAL/ANATOMICAL CONTEXT models. Their risk-stratification gains may reflect the structural positivity difference between targeted and systematic cores rather than geometric signal.

## GG2+ / csPCa

| feature_set | model | prevalence | top5pct_prevalence | top5pct_capture_rate | top5pct_enrichment | top10pct_prevalence | top10pct_capture_rate | top10pct_enrichment | top20pct_prevalence | top20pct_capture_rate | top20pct_enrichment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 0.107 | 0.504 | 0.236 | 4.705 | 0.343 | 0.320 | 3.201 | 0.269 | 0.502 | 2.507 |
| current_compact | tuned_l2_logistic_regression | 0.107 | 0.504 | 0.236 | 4.705 | 0.343 | 0.320 | 3.201 | 0.269 | 0.502 | 2.507 |
| current_compact | tuned_l1_logistic_regression | 0.107 | 0.496 | 0.232 | 4.627 | 0.339 | 0.317 | 3.162 | 0.269 | 0.502 | 2.507 |
| current_compact | tuned_elasticnet_logistic_regression | 0.107 | 0.496 | 0.232 | 4.627 | 0.339 | 0.317 | 3.162 | 0.269 | 0.502 | 2.507 |
| current_compact | hist_gradient_boosting | 0.107 | 0.463 | 0.216 | 4.319 | 0.318 | 0.297 | 2.969 | 0.258 | 0.483 | 2.410 |
| current_compact | xgboost | 0.107 | 0.496 | 0.232 | 4.627 | 0.343 | 0.320 | 3.201 | 0.264 | 0.494 | 2.468 |
| compact_plus_core_label_onehot | logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.331 | 0.309 | 3.085 | 0.275 | 0.514 | 2.564 |
| compact_plus_core_label_onehot | tuned_l2_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.331 | 0.309 | 3.085 | 0.277 | 0.517 | 2.584 |
| compact_plus_core_label_onehot | tuned_l1_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.331 | 0.309 | 3.085 | 0.277 | 0.517 | 2.584 |
| compact_plus_core_label_onehot | tuned_elasticnet_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.331 | 0.309 | 3.085 | 0.277 | 0.517 | 2.584 |
| compact_plus_core_label_onehot | hist_gradient_boosting | 0.107 | 0.471 | 0.220 | 4.396 | 0.314 | 0.293 | 2.931 | 0.267 | 0.498 | 2.487 |
| compact_plus_core_label_onehot | xgboost | 0.107 | 0.479 | 0.224 | 4.473 | 0.355 | 0.332 | 3.316 | 0.273 | 0.510 | 2.545 |
| compact_plus_anatomical_core_label | logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.326 | 0.305 | 3.046 | 0.275 | 0.514 | 2.564 |
| compact_plus_anatomical_core_label | tuned_l2_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.326 | 0.305 | 3.046 | 0.277 | 0.517 | 2.584 |
| compact_plus_anatomical_core_label | tuned_l1_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.326 | 0.305 | 3.046 | 0.277 | 0.517 | 2.584 |
| compact_plus_anatomical_core_label | tuned_elasticnet_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.326 | 0.305 | 3.046 | 0.277 | 0.517 | 2.584 |
| compact_plus_anatomical_core_label | hist_gradient_boosting | 0.107 | 0.488 | 0.228 | 4.550 | 0.343 | 0.320 | 3.201 | 0.271 | 0.506 | 2.526 |
| compact_plus_anatomical_core_label | xgboost | 0.107 | 0.471 | 0.220 | 4.396 | 0.351 | 0.328 | 3.278 | 0.273 | 0.510 | 2.545 |
| compact_plus_intensity | logistic_regression | 0.107 | 0.504 | 0.236 | 4.705 | 0.347 | 0.324 | 3.239 | 0.269 | 0.502 | 2.507 |
| compact_plus_intensity | tuned_l2_logistic_regression | 0.107 | 0.504 | 0.236 | 4.705 | 0.343 | 0.320 | 3.201 | 0.269 | 0.502 | 2.507 |
| compact_plus_intensity | tuned_l1_logistic_regression | 0.107 | 0.496 | 0.232 | 4.627 | 0.343 | 0.320 | 3.201 | 0.271 | 0.506 | 2.526 |
| compact_plus_intensity | tuned_elasticnet_logistic_regression | 0.107 | 0.504 | 0.236 | 4.705 | 0.343 | 0.320 | 3.201 | 0.271 | 0.506 | 2.526 |
| compact_plus_intensity | hist_gradient_boosting | 0.107 | 0.455 | 0.212 | 4.242 | 0.326 | 0.305 | 3.046 | 0.258 | 0.483 | 2.410 |
| compact_plus_intensity | xgboost | 0.107 | 0.496 | 0.232 | 4.627 | 0.331 | 0.309 | 3.085 | 0.260 | 0.486 | 2.429 |
| compact_plus_transformed_geometry | logistic_regression | 0.107 | 0.471 | 0.220 | 4.396 | 0.347 | 0.324 | 3.239 | 0.267 | 0.498 | 2.487 |
| compact_plus_transformed_geometry | tuned_l2_logistic_regression | 0.107 | 0.471 | 0.220 | 4.396 | 0.347 | 0.324 | 3.239 | 0.260 | 0.486 | 2.429 |
| compact_plus_transformed_geometry | tuned_l1_logistic_regression | 0.107 | 0.471 | 0.220 | 4.396 | 0.347 | 0.324 | 3.239 | 0.267 | 0.498 | 2.487 |
| compact_plus_transformed_geometry | tuned_elasticnet_logistic_regression | 0.107 | 0.471 | 0.220 | 4.396 | 0.347 | 0.324 | 3.239 | 0.267 | 0.498 | 2.487 |
| compact_plus_transformed_geometry | hist_gradient_boosting | 0.107 | 0.455 | 0.212 | 4.242 | 0.335 | 0.313 | 3.124 | 0.289 | 0.541 | 2.699 |
| compact_plus_transformed_geometry | xgboost | 0.107 | 0.496 | 0.232 | 4.627 | 0.343 | 0.320 | 3.201 | 0.271 | 0.506 | 2.526 |
| compact_plus_core_label_plus_intensity | logistic_regression | 0.107 | 0.479 | 0.224 | 4.473 | 0.335 | 0.313 | 3.124 | 0.277 | 0.517 | 2.584 |
| compact_plus_core_label_plus_intensity | tuned_l2_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.331 | 0.309 | 3.085 | 0.275 | 0.514 | 2.564 |
| compact_plus_core_label_plus_intensity | tuned_l1_logistic_regression | 0.107 | 0.479 | 0.224 | 4.473 | 0.335 | 0.313 | 3.124 | 0.277 | 0.517 | 2.584 |
| compact_plus_core_label_plus_intensity | tuned_elasticnet_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.331 | 0.309 | 3.085 | 0.275 | 0.514 | 2.564 |
| compact_plus_core_label_plus_intensity | hist_gradient_boosting | 0.107 | 0.455 | 0.212 | 4.242 | 0.306 | 0.286 | 2.854 | 0.258 | 0.483 | 2.410 |
| compact_plus_core_label_plus_intensity | xgboost | 0.107 | 0.455 | 0.212 | 4.242 | 0.351 | 0.328 | 3.278 | 0.267 | 0.498 | 2.487 |
| compact_plus_core_label_plus_transformed_geo | logistic_regression | 0.107 | 0.463 | 0.216 | 4.319 | 0.331 | 0.309 | 3.085 | 0.260 | 0.486 | 2.429 |
| compact_plus_core_label_plus_transformed_geo | tuned_l2_logistic_regression | 0.107 | 0.463 | 0.216 | 4.319 | 0.322 | 0.301 | 3.008 | 0.260 | 0.486 | 2.429 |
| compact_plus_core_label_plus_transformed_geo | tuned_l1_logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.335 | 0.313 | 3.124 | 0.281 | 0.525 | 2.622 |
| compact_plus_core_label_plus_transformed_geo | tuned_elasticnet_logistic_regression | 0.107 | 0.463 | 0.216 | 4.319 | 0.322 | 0.301 | 3.008 | 0.260 | 0.486 | 2.429 |
| compact_plus_core_label_plus_transformed_geo | hist_gradient_boosting | 0.107 | 0.438 | 0.205 | 4.088 | 0.326 | 0.305 | 3.046 | 0.260 | 0.486 | 2.429 |
| compact_plus_core_label_plus_transformed_geo | xgboost | 0.107 | 0.479 | 0.224 | 4.473 | 0.355 | 0.332 | 3.316 | 0.275 | 0.514 | 2.564 |
| compact_plus_all_low_cost_features | logistic_regression | 0.107 | 0.463 | 0.216 | 4.319 | 0.335 | 0.313 | 3.124 | 0.258 | 0.483 | 2.410 |
| compact_plus_all_low_cost_features | tuned_l2_logistic_regression | 0.107 | 0.471 | 0.220 | 4.396 | 0.335 | 0.313 | 3.124 | 0.273 | 0.510 | 2.545 |
| compact_plus_all_low_cost_features | tuned_l1_logistic_regression | 0.107 | 0.463 | 0.216 | 4.319 | 0.335 | 0.313 | 3.124 | 0.258 | 0.483 | 2.410 |
| compact_plus_all_low_cost_features | tuned_elasticnet_logistic_regression | 0.107 | 0.463 | 0.216 | 4.319 | 0.335 | 0.313 | 3.124 | 0.256 | 0.479 | 2.391 |
| compact_plus_all_low_cost_features | hist_gradient_boosting | 0.107 | 0.430 | 0.201 | 4.010 | 0.322 | 0.301 | 3.008 | 0.279 | 0.521 | 2.603 |
| compact_plus_all_low_cost_features | xgboost | 0.107 | 0.479 | 0.224 | 4.473 | 0.347 | 0.324 | 3.239 | 0.273 | 0.510 | 2.545 |
| clinical_plus_core_label | logistic_regression | 0.107 | 0.496 | 0.232 | 4.627 | 0.318 | 0.297 | 2.969 | 0.236 | 0.440 | 2.198 |
| clinical_plus_core_label | tuned_l2_logistic_regression | 0.107 | 0.496 | 0.232 | 4.627 | 0.318 | 0.297 | 2.969 | 0.236 | 0.440 | 2.198 |
| clinical_plus_core_label | tuned_l1_logistic_regression | 0.107 | 0.512 | 0.239 | 4.782 | 0.318 | 0.297 | 2.969 | 0.233 | 0.436 | 2.179 |
| clinical_plus_core_label | tuned_elasticnet_logistic_regression | 0.107 | 0.512 | 0.239 | 4.782 | 0.318 | 0.297 | 2.969 | 0.233 | 0.436 | 2.179 |
| clinical_plus_core_label | hist_gradient_boosting | 0.107 | 0.438 | 0.205 | 4.088 | 0.318 | 0.297 | 2.969 | 0.225 | 0.421 | 2.102 |
| clinical_plus_core_label | xgboost | 0.107 | 0.438 | 0.205 | 4.088 | 0.306 | 0.286 | 2.854 | 0.229 | 0.429 | 2.140 |
| target_geometry_plus_core_label | logistic_regression | 0.107 | 0.264 | 0.124 | 2.468 | 0.269 | 0.251 | 2.507 | 0.244 | 0.456 | 2.275 |
| target_geometry_plus_core_label | tuned_l2_logistic_regression | 0.107 | 0.264 | 0.124 | 2.468 | 0.269 | 0.251 | 2.507 | 0.246 | 0.459 | 2.294 |
| target_geometry_plus_core_label | tuned_l1_logistic_regression | 0.107 | 0.264 | 0.124 | 2.468 | 0.269 | 0.251 | 2.507 | 0.244 | 0.456 | 2.275 |
| target_geometry_plus_core_label | tuned_elasticnet_logistic_regression | 0.107 | 0.264 | 0.124 | 2.468 | 0.269 | 0.251 | 2.507 | 0.244 | 0.456 | 2.275 |
| target_geometry_plus_core_label | hist_gradient_boosting | 0.107 | 0.289 | 0.135 | 2.699 | 0.260 | 0.243 | 2.429 | 0.233 | 0.436 | 2.179 |
| target_geometry_plus_core_label | xgboost | 0.107 | 0.264 | 0.124 | 2.468 | 0.269 | 0.251 | 2.507 | 0.238 | 0.444 | 2.217 |

## GG3+ / high-grade

| feature_set | model | prevalence | top5pct_prevalence | top5pct_capture_rate | top5pct_enrichment | top10pct_prevalence | top10pct_capture_rate | top10pct_enrichment | top20pct_prevalence | top20pct_capture_rate | top20pct_enrichment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| current_compact | tuned_l2_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| current_compact | tuned_l1_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| current_compact | tuned_elasticnet_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| current_compact | hist_gradient_boosting | 0.038 | 0.248 | 0.326 | 6.514 | 0.190 | 0.500 | 4.994 | 0.116 | 0.609 | 3.040 |
| current_compact | xgboost | 0.038 | 0.331 | 0.435 | 8.685 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| compact_plus_core_label_onehot | logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| compact_plus_core_label_onehot | tuned_l2_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| compact_plus_core_label_onehot | tuned_l1_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.190 | 0.500 | 4.994 | 0.114 | 0.598 | 2.985 |
| compact_plus_core_label_onehot | tuned_elasticnet_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.194 | 0.511 | 5.102 | 0.116 | 0.609 | 3.040 |
| compact_plus_core_label_onehot | hist_gradient_boosting | 0.038 | 0.264 | 0.348 | 6.948 | 0.182 | 0.478 | 4.777 | 0.116 | 0.609 | 3.040 |
| compact_plus_core_label_onehot | xgboost | 0.038 | 0.306 | 0.402 | 8.034 | 0.182 | 0.478 | 4.777 | 0.112 | 0.587 | 2.931 |
| compact_plus_anatomical_core_label | logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.194 | 0.511 | 5.102 | 0.107 | 0.565 | 2.823 |
| compact_plus_anatomical_core_label | tuned_l2_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.194 | 0.511 | 5.102 | 0.107 | 0.565 | 2.823 |
| compact_plus_anatomical_core_label | tuned_l1_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.190 | 0.500 | 4.994 | 0.116 | 0.609 | 3.040 |
| compact_plus_anatomical_core_label | tuned_elasticnet_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.194 | 0.511 | 5.102 | 0.116 | 0.609 | 3.040 |
| compact_plus_anatomical_core_label | hist_gradient_boosting | 0.038 | 0.264 | 0.348 | 6.948 | 0.178 | 0.467 | 4.668 | 0.116 | 0.609 | 3.040 |
| compact_plus_anatomical_core_label | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.178 | 0.467 | 4.668 | 0.114 | 0.598 | 2.985 |
| compact_plus_intensity | logistic_regression | 0.038 | 0.322 | 0.424 | 8.468 | 0.194 | 0.511 | 5.102 | 0.105 | 0.554 | 2.768 |
| compact_plus_intensity | tuned_l2_logistic_regression | 0.038 | 0.322 | 0.424 | 8.468 | 0.194 | 0.511 | 5.102 | 0.105 | 0.554 | 2.768 |
| compact_plus_intensity | tuned_l1_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.190 | 0.500 | 4.994 | 0.112 | 0.587 | 2.931 |
| compact_plus_intensity | tuned_elasticnet_logistic_regression | 0.038 | 0.322 | 0.424 | 8.468 | 0.194 | 0.511 | 5.102 | 0.103 | 0.543 | 2.714 |
| compact_plus_intensity | hist_gradient_boosting | 0.038 | 0.223 | 0.293 | 5.862 | 0.186 | 0.489 | 4.885 | 0.118 | 0.620 | 3.094 |
| compact_plus_intensity | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.194 | 0.511 | 5.102 | 0.107 | 0.565 | 2.823 |
| compact_plus_transformed_geometry | logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.114 | 0.598 | 2.985 |
| compact_plus_transformed_geometry | tuned_l2_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.190 | 0.500 | 4.994 | 0.114 | 0.598 | 2.985 |
| compact_plus_transformed_geometry | tuned_l1_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.186 | 0.489 | 4.885 | 0.114 | 0.598 | 2.985 |
| compact_plus_transformed_geometry | tuned_elasticnet_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.186 | 0.489 | 4.885 | 0.114 | 0.598 | 2.985 |
| compact_plus_transformed_geometry | hist_gradient_boosting | 0.038 | 0.256 | 0.337 | 6.731 | 0.169 | 0.446 | 4.451 | 0.124 | 0.652 | 3.257 |
| compact_plus_transformed_geometry | xgboost | 0.038 | 0.322 | 0.424 | 8.468 | 0.186 | 0.489 | 4.885 | 0.118 | 0.620 | 3.094 |
| compact_plus_core_label_plus_intensity | logistic_regression | 0.038 | 0.322 | 0.424 | 8.468 | 0.194 | 0.511 | 5.102 | 0.103 | 0.543 | 2.714 |
| compact_plus_core_label_plus_intensity | tuned_l2_logistic_regression | 0.038 | 0.322 | 0.424 | 8.468 | 0.194 | 0.511 | 5.102 | 0.105 | 0.554 | 2.768 |
| compact_plus_core_label_plus_intensity | tuned_l1_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.190 | 0.500 | 4.994 | 0.112 | 0.587 | 2.931 |
| compact_plus_core_label_plus_intensity | tuned_elasticnet_logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.194 | 0.511 | 5.102 | 0.110 | 0.576 | 2.877 |
| compact_plus_core_label_plus_intensity | hist_gradient_boosting | 0.038 | 0.256 | 0.337 | 6.731 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| compact_plus_core_label_plus_intensity | xgboost | 0.038 | 0.322 | 0.424 | 8.468 | 0.194 | 0.511 | 5.102 | 0.112 | 0.587 | 2.931 |
| compact_plus_core_label_plus_transformed_geo | logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.186 | 0.489 | 4.885 | 0.116 | 0.609 | 3.040 |
| compact_plus_core_label_plus_transformed_geo | tuned_l2_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.186 | 0.489 | 4.885 | 0.112 | 0.587 | 2.931 |
| compact_plus_core_label_plus_transformed_geo | tuned_l1_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.186 | 0.489 | 4.885 | 0.112 | 0.587 | 2.931 |
| compact_plus_core_label_plus_transformed_geo | tuned_elasticnet_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.186 | 0.489 | 4.885 | 0.112 | 0.587 | 2.931 |
| compact_plus_core_label_plus_transformed_geo | hist_gradient_boosting | 0.038 | 0.248 | 0.326 | 6.514 | 0.174 | 0.457 | 4.560 | 0.118 | 0.620 | 3.094 |
| compact_plus_core_label_plus_transformed_geo | xgboost | 0.038 | 0.322 | 0.424 | 8.468 | 0.190 | 0.500 | 4.994 | 0.116 | 0.609 | 3.040 |
| compact_plus_all_low_cost_features | logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.112 | 0.587 | 2.931 |
| compact_plus_all_low_cost_features | tuned_l2_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.186 | 0.489 | 4.885 | 0.107 | 0.565 | 2.823 |
| compact_plus_all_low_cost_features | tuned_l1_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.186 | 0.489 | 4.885 | 0.107 | 0.565 | 2.823 |
| compact_plus_all_low_cost_features | tuned_elasticnet_logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.107 | 0.565 | 2.823 |
| compact_plus_all_low_cost_features | hist_gradient_boosting | 0.038 | 0.264 | 0.348 | 6.948 | 0.165 | 0.435 | 4.342 | 0.114 | 0.598 | 2.985 |
| compact_plus_all_low_cost_features | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.182 | 0.478 | 4.777 | 0.112 | 0.587 | 2.931 |
| clinical_plus_core_label | logistic_regression | 0.038 | 0.289 | 0.380 | 7.599 | 0.169 | 0.446 | 4.451 | 0.101 | 0.533 | 2.660 |
| clinical_plus_core_label | tuned_l2_logistic_regression | 0.038 | 0.298 | 0.391 | 7.816 | 0.161 | 0.424 | 4.234 | 0.101 | 0.533 | 2.660 |
| clinical_plus_core_label | tuned_l1_logistic_regression | 0.038 | 0.298 | 0.391 | 7.816 | 0.161 | 0.424 | 4.234 | 0.097 | 0.511 | 2.551 |
| clinical_plus_core_label | tuned_elasticnet_logistic_regression | 0.038 | 0.298 | 0.391 | 7.816 | 0.161 | 0.424 | 4.234 | 0.097 | 0.511 | 2.551 |
| clinical_plus_core_label | hist_gradient_boosting | 0.038 | 0.231 | 0.304 | 6.079 | 0.157 | 0.413 | 4.125 | 0.097 | 0.511 | 2.551 |
| clinical_plus_core_label | xgboost | 0.038 | 0.273 | 0.359 | 7.165 | 0.145 | 0.380 | 3.800 | 0.112 | 0.587 | 2.931 |
| target_geometry_plus_core_label | logistic_regression | 0.038 | 0.157 | 0.207 | 4.125 | 0.128 | 0.337 | 3.365 | 0.107 | 0.565 | 2.823 |
| target_geometry_plus_core_label | tuned_l2_logistic_regression | 0.038 | 0.157 | 0.207 | 4.125 | 0.132 | 0.348 | 3.474 | 0.107 | 0.565 | 2.823 |
| target_geometry_plus_core_label | tuned_l1_logistic_regression | 0.038 | 0.182 | 0.239 | 4.777 | 0.128 | 0.337 | 3.365 | 0.101 | 0.533 | 2.660 |
| target_geometry_plus_core_label | tuned_elasticnet_logistic_regression | 0.038 | 0.157 | 0.207 | 4.125 | 0.128 | 0.337 | 3.365 | 0.107 | 0.565 | 2.823 |
| target_geometry_plus_core_label | hist_gradient_boosting | 0.038 | 0.132 | 0.174 | 3.474 | 0.107 | 0.283 | 2.823 | 0.107 | 0.565 | 2.823 |
| target_geometry_plus_core_label | xgboost | 0.038 | 0.182 | 0.239 | 4.777 | 0.136 | 0.359 | 3.583 | 0.103 | 0.543 | 2.714 |

