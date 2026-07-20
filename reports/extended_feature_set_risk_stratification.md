# Extended Feature Set — Risk Stratification

**Capture rate** = fraction of all test positives in the top-N% scored cores.  
**Prevalence** in top-N% = positive rate among top-N% highest-probability cores.  
**Enrichment** = top-N% prevalence / baseline prevalence.  
**Baseline:** overall positive rate in the test set.

**Note:** top-N% risk stratification is most interpretable for models with continuous predicted scores. For `targeting_context_only`, many cores have tied predicted probabilities because the model uses a single binary feature; top-N% prevalence and capture rates for this row are order-dependent and should not be interpreted.

## GG2+ / csPCa

| feature_set | model | prevalence | top5pct_prevalence | top5pct_capture_rate | top5pct_enrichment | top10pct_prevalence | top10pct_capture_rate | top10pct_enrichment | top20pct_prevalence | top20pct_capture_rate | top20pct_enrichment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 0.107 | 0.504 | 0.236 | 4.705 | 0.343 | 0.320 | 3.201 | 0.269 | 0.502 | 2.507 |
| current_compact | hist_gradient_boosting | 0.107 | 0.463 | 0.216 | 4.319 | 0.318 | 0.297 | 2.969 | 0.258 | 0.483 | 2.410 |
| current_compact | xgboost | 0.107 | 0.496 | 0.232 | 4.627 | 0.343 | 0.320 | 3.201 | 0.264 | 0.494 | 2.468 |
| compact_with_signed_distance | logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.343 | 0.320 | 3.201 | 0.273 | 0.510 | 2.545 |
| compact_with_signed_distance | hist_gradient_boosting | 0.107 | 0.430 | 0.201 | 4.010 | 0.310 | 0.290 | 2.892 | 0.258 | 0.483 | 2.410 |
| compact_with_signed_distance | xgboost | 0.107 | 0.471 | 0.220 | 4.396 | 0.351 | 0.328 | 3.278 | 0.267 | 0.498 | 2.487 |
| compact_plus_targeting_context | logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.326 | 0.305 | 3.046 | 0.277 | 0.517 | 2.584 |
| compact_plus_targeting_context | hist_gradient_boosting | 0.107 | 0.446 | 0.208 | 4.165 | 0.322 | 0.301 | 3.008 | 0.258 | 0.483 | 2.410 |
| compact_plus_targeting_context | xgboost | 0.107 | 0.446 | 0.208 | 4.165 | 0.335 | 0.313 | 3.124 | 0.271 | 0.506 | 2.526 |
| compact_signed_plus_targeting_context | logistic_regression | 0.107 | 0.479 | 0.224 | 4.473 | 0.331 | 0.309 | 3.085 | 0.264 | 0.494 | 2.468 |
| compact_signed_plus_targeting_context | hist_gradient_boosting | 0.107 | 0.421 | 0.197 | 3.933 | 0.331 | 0.309 | 3.085 | 0.256 | 0.479 | 2.391 |
| compact_signed_plus_targeting_context | xgboost | 0.107 | 0.455 | 0.212 | 4.242 | 0.347 | 0.324 | 3.239 | 0.267 | 0.498 | 2.487 |
| compact_signed_plus_interactions | logistic_regression | 0.107 | 0.488 | 0.228 | 4.550 | 0.351 | 0.328 | 3.278 | 0.273 | 0.510 | 2.545 |
| compact_signed_plus_interactions | hist_gradient_boosting | 0.107 | 0.405 | 0.189 | 3.779 | 0.306 | 0.286 | 2.854 | 0.250 | 0.467 | 2.333 |
| compact_signed_plus_interactions | xgboost | 0.107 | 0.488 | 0.228 | 4.550 | 0.360 | 0.336 | 3.355 | 0.273 | 0.510 | 2.545 |
| compact_signed_plus_targeting_plus_inter | logistic_regression | 0.107 | 0.479 | 0.224 | 4.473 | 0.335 | 0.313 | 3.124 | 0.269 | 0.502 | 2.507 |
| compact_signed_plus_targeting_plus_inter | hist_gradient_boosting | 0.107 | 0.463 | 0.216 | 4.319 | 0.322 | 0.301 | 3.008 | 0.246 | 0.459 | 2.294 |
| compact_signed_plus_targeting_plus_inter | xgboost | 0.107 | 0.471 | 0.220 | 4.396 | 0.347 | 0.324 | 3.239 | 0.267 | 0.498 | 2.487 |
| clinical_plus_targeting_context | logistic_regression | 0.107 | 0.512 | 0.239 | 4.782 | 0.318 | 0.297 | 2.969 | 0.236 | 0.440 | 2.198 |
| clinical_plus_targeting_context | hist_gradient_boosting | 0.107 | 0.438 | 0.205 | 4.088 | 0.314 | 0.293 | 2.931 | 0.240 | 0.448 | 2.237 |
| clinical_plus_targeting_context | xgboost | 0.107 | 0.446 | 0.208 | 4.165 | 0.302 | 0.282 | 2.815 | 0.219 | 0.409 | 2.044 |
| targeting_context_only | logistic_regression | 0.107 | 0.017 | 0.008 | 0.154 | 0.070 | 0.066 | 0.656 | 0.122 | 0.228 | 1.138 |
| targeting_context_only | hist_gradient_boosting | 0.107 | 0.017 | 0.008 | 0.154 | 0.070 | 0.066 | 0.656 | 0.122 | 0.228 | 1.138 |
| targeting_context_only | xgboost | 0.107 | 0.017 | 0.008 | 0.154 | 0.070 | 0.066 | 0.656 | 0.122 | 0.228 | 1.138 |

## GG3+ / high-grade

| feature_set | model | prevalence | top5pct_prevalence | top5pct_capture_rate | top5pct_enrichment | top10pct_prevalence | top10pct_capture_rate | top10pct_enrichment | top20pct_prevalence | top20pct_capture_rate | top20pct_enrichment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current_compact | logistic_regression | 0.038 | 0.314 | 0.413 | 8.251 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| current_compact | hist_gradient_boosting | 0.038 | 0.248 | 0.326 | 6.514 | 0.190 | 0.500 | 4.994 | 0.116 | 0.609 | 3.040 |
| current_compact | xgboost | 0.038 | 0.331 | 0.435 | 8.685 | 0.190 | 0.500 | 4.994 | 0.110 | 0.576 | 2.877 |
| compact_with_signed_distance | logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.182 | 0.478 | 4.777 | 0.105 | 0.554 | 2.768 |
| compact_with_signed_distance | hist_gradient_boosting | 0.038 | 0.264 | 0.348 | 6.948 | 0.169 | 0.446 | 4.451 | 0.114 | 0.598 | 2.985 |
| compact_with_signed_distance | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.190 | 0.500 | 4.994 | 0.112 | 0.587 | 2.931 |
| compact_plus_targeting_context | logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.194 | 0.511 | 5.102 | 0.110 | 0.576 | 2.877 |
| compact_plus_targeting_context | hist_gradient_boosting | 0.038 | 0.281 | 0.370 | 7.382 | 0.194 | 0.511 | 5.102 | 0.118 | 0.620 | 3.094 |
| compact_plus_targeting_context | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.182 | 0.478 | 4.777 | 0.107 | 0.565 | 2.823 |
| compact_signed_plus_targeting_context | logistic_regression | 0.038 | 0.298 | 0.391 | 7.816 | 0.182 | 0.478 | 4.777 | 0.103 | 0.543 | 2.714 |
| compact_signed_plus_targeting_context | hist_gradient_boosting | 0.038 | 0.264 | 0.348 | 6.948 | 0.165 | 0.435 | 4.342 | 0.118 | 0.620 | 3.094 |
| compact_signed_plus_targeting_context | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.182 | 0.478 | 4.777 | 0.112 | 0.587 | 2.931 |
| compact_signed_plus_interactions | logistic_regression | 0.038 | 0.306 | 0.402 | 8.034 | 0.182 | 0.478 | 4.777 | 0.105 | 0.554 | 2.768 |
| compact_signed_plus_interactions | hist_gradient_boosting | 0.038 | 0.223 | 0.293 | 5.862 | 0.174 | 0.457 | 4.560 | 0.118 | 0.620 | 3.094 |
| compact_signed_plus_interactions | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.182 | 0.478 | 4.777 | 0.110 | 0.576 | 2.877 |
| compact_signed_plus_targeting_plus_inter | logistic_regression | 0.038 | 0.298 | 0.391 | 7.816 | 0.182 | 0.478 | 4.777 | 0.105 | 0.554 | 2.768 |
| compact_signed_plus_targeting_plus_inter | hist_gradient_boosting | 0.038 | 0.289 | 0.380 | 7.599 | 0.178 | 0.467 | 4.668 | 0.110 | 0.576 | 2.877 |
| compact_signed_plus_targeting_plus_inter | xgboost | 0.038 | 0.298 | 0.391 | 7.816 | 0.186 | 0.489 | 4.885 | 0.110 | 0.576 | 2.877 |
| clinical_plus_targeting_context | logistic_regression | 0.038 | 0.298 | 0.391 | 7.816 | 0.161 | 0.424 | 4.234 | 0.101 | 0.533 | 2.660 |
| clinical_plus_targeting_context | hist_gradient_boosting | 0.038 | 0.248 | 0.326 | 6.514 | 0.157 | 0.413 | 4.125 | 0.097 | 0.511 | 2.551 |
| clinical_plus_targeting_context | xgboost | 0.038 | 0.273 | 0.359 | 7.165 | 0.145 | 0.380 | 3.800 | 0.116 | 0.609 | 3.040 |
| targeting_context_only | logistic_regression | 0.038 | 0.008 | 0.011 | 0.217 | 0.033 | 0.087 | 0.868 | 0.058 | 0.304 | 1.520 |
| targeting_context_only | hist_gradient_boosting | 0.038 | 0.008 | 0.011 | 0.217 | 0.033 | 0.087 | 0.868 | 0.058 | 0.304 | 1.520 |
| targeting_context_only | xgboost | 0.038 | 0.008 | 0.011 | 0.217 | 0.033 | 0.087 | 0.868 | 0.058 | 0.304 | 1.520 |

