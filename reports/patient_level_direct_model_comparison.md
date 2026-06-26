# Direct Patient-Level Model Comparison

## 1. Methodology

Core-level features are aggregated to one row per patient before model fitting. The patient label is positive if any core is positive for the endpoint. Models are trained on train-split patients, threshold is selected on val-split patients (Youden J), and test-split patients are evaluated once.  
**Bootstrap CIs:** 1,000 patient resamples of the test set with replacement (2.5/97.5 percentiles).  
`target_mesh_available` is excluded from all feature sets.  
`n_cores` is computed for auditing but excluded from the main predictive feature sets, because biopsy count may encode sampling/procedural intensity rather than anatomy or biology.

## 2. Patient-Level Dataset Summary

### GG2+ / csPCa

| Split | Patients | Positive | Prevalence |
|---|---|---|---|
| train | 557 | 324 | 0.582 |
| val | 119 | 69 | 0.580 |
| test | 120 | 70 | 0.583 |

### GG3+ / high-grade

| Split | Patients | Positive | Prevalence |
|---|---|---|---|
| train | 557 | 145 | 0.260 |
| val | 119 | 34 | 0.286 |
| test | 120 | 31 | 0.258 |

## 3. Main Performance Table

### GG2+ / csPCa

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train_patients | n_val_patients | n_test_patients | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.613 | 0.515 | 0.716 | 0.697 | 0.604 | 0.816 | 1.195 | 0.434 | 0.643 | 0.560 | 0.672 | 0.657 | 557 | 119 | 120 | 0.583 |
| clinical_only | hist_gradient_boosting | 4 | 0.558 | 0.456 | 0.660 | 0.679 | 0.576 | 0.782 | 1.164 | 0.944 | 0.171 | 0.940 | 0.800 | 0.282 | 557 | 119 | 120 | 0.583 |
| clinical_only | xgboost | 4 | 0.606 | 0.504 | 0.706 | 0.716 | 0.621 | 0.814 | 1.228 | 0.366 | 0.671 | 0.400 | 0.610 | 0.639 | 557 | 119 | 120 | 0.583 |
| target_geometry_aggregates_only | logistic_regression | 12 | 0.502 | 0.394 | 0.610 | 0.589 | 0.493 | 0.728 | 1.009 | 0.515 | 0.414 | 0.540 | 0.558 | 0.475 | 557 | 119 | 120 | 0.583 |
| target_geometry_aggregates_only | hist_gradient_boosting | 12 | 0.462 | 0.351 | 0.570 | 0.577 | 0.471 | 0.705 | 0.989 | 0.305 | 0.729 | 0.260 | 0.580 | 0.646 | 557 | 119 | 120 | 0.583 |
| target_geometry_aggregates_only | xgboost | 12 | 0.447 | 0.336 | 0.561 | 0.556 | 0.465 | 0.694 | 0.953 | 0.539 | 0.443 | 0.440 | 0.525 | 0.481 | 557 | 119 | 120 | 0.583 |
| target_geometry_aggregates_plus_clinical | logistic_regression | 16 | 0.592 | 0.495 | 0.690 | 0.706 | 0.614 | 0.820 | 1.210 | 0.471 | 0.529 | 0.620 | 0.661 | 0.587 | 557 | 119 | 120 | 0.583 |
| target_geometry_aggregates_plus_clinical | hist_gradient_boosting | 16 | 0.582 | 0.477 | 0.679 | 0.700 | 0.602 | 0.814 | 1.201 | 0.834 | 0.400 | 0.880 | 0.824 | 0.538 | 557 | 119 | 120 | 0.583 |
| target_geometry_aggregates_plus_clinical | xgboost | 16 | 0.579 | 0.471 | 0.681 | 0.697 | 0.600 | 0.809 | 1.196 | 0.625 | 0.429 | 0.760 | 0.714 | 0.536 | 557 | 119 | 120 | 0.583 |
| all_geometry_aggregates_plus_clinical | logistic_regression | 22 | 0.582 | 0.481 | 0.684 | 0.676 | 0.579 | 0.796 | 1.159 | 0.313 | 0.771 | 0.240 | 0.587 | 0.667 | 557 | 119 | 120 | 0.583 |
| all_geometry_aggregates_plus_clinical | hist_gradient_boosting | 22 | 0.613 | 0.508 | 0.711 | 0.722 | 0.622 | 0.828 | 1.237 | 0.285 | 0.714 | 0.380 | 0.617 | 0.662 | 557 | 119 | 120 | 0.583 |
| all_geometry_aggregates_plus_clinical | xgboost | 22 | 0.595 | 0.485 | 0.698 | 0.700 | 0.602 | 0.815 | 1.201 | 0.376 | 0.671 | 0.440 | 0.627 | 0.648 | 557 | 119 | 120 | 0.583 |

### GG3+ / high-grade

| feature_set | model | n_features | test_roc_auc | roc_auc_ci_low | roc_auc_ci_high | test_pr_auc | pr_auc_ci_low | pr_auc_ci_high | pr_lift | threshold | test_sensitivity | test_specificity | test_precision | test_f1 | n_train_patients | n_val_patients | n_test_patients | test_prevalence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clinical_only | logistic_regression | 4 | 0.656 | 0.535 | 0.778 | 0.467 | 0.323 | 0.671 | 1.809 | 0.564 | 0.452 | 0.876 | 0.560 | 0.500 | 557 | 119 | 120 | 0.258 |
| clinical_only | hist_gradient_boosting | 4 | 0.696 | 0.582 | 0.807 | 0.436 | 0.308 | 0.633 | 1.689 | 0.118 | 0.742 | 0.584 | 0.383 | 0.505 | 557 | 119 | 120 | 0.258 |
| clinical_only | xgboost | 4 | 0.656 | 0.535 | 0.772 | 0.442 | 0.300 | 0.643 | 1.712 | 0.398 | 0.516 | 0.640 | 0.333 | 0.405 | 557 | 119 | 120 | 0.258 |
| target_geometry_aggregates_only | logistic_regression | 12 | 0.618 | 0.517 | 0.724 | 0.361 | 0.238 | 0.516 | 1.397 | 0.445 | 0.613 | 0.551 | 0.322 | 0.422 | 557 | 119 | 120 | 0.258 |
| target_geometry_aggregates_only | hist_gradient_boosting | 12 | 0.580 | 0.455 | 0.689 | 0.337 | 0.227 | 0.506 | 1.306 | 0.293 | 0.290 | 0.787 | 0.321 | 0.305 | 557 | 119 | 120 | 0.258 |
| target_geometry_aggregates_only | xgboost | 12 | 0.624 | 0.512 | 0.725 | 0.344 | 0.239 | 0.508 | 1.333 | 0.476 | 0.355 | 0.730 | 0.314 | 0.333 | 557 | 119 | 120 | 0.258 |
| target_geometry_aggregates_plus_clinical | logistic_regression | 16 | 0.664 | 0.540 | 0.781 | 0.460 | 0.316 | 0.669 | 1.782 | 0.378 | 0.677 | 0.596 | 0.368 | 0.477 | 557 | 119 | 120 | 0.258 |
| target_geometry_aggregates_plus_clinical | hist_gradient_boosting | 16 | 0.669 | 0.545 | 0.789 | 0.470 | 0.322 | 0.671 | 1.818 | 0.219 | 0.516 | 0.820 | 0.500 | 0.508 | 557 | 119 | 120 | 0.258 |
| target_geometry_aggregates_plus_clinical | xgboost | 16 | 0.678 | 0.561 | 0.795 | 0.467 | 0.320 | 0.660 | 1.808 | 0.383 | 0.581 | 0.753 | 0.450 | 0.507 | 557 | 119 | 120 | 0.258 |
| all_geometry_aggregates_plus_clinical | logistic_regression | 22 | 0.657 | 0.542 | 0.775 | 0.428 | 0.296 | 0.632 | 1.658 | 0.354 | 0.710 | 0.517 | 0.338 | 0.458 | 557 | 119 | 120 | 0.258 |
| all_geometry_aggregates_plus_clinical | hist_gradient_boosting | 22 | 0.706 | 0.583 | 0.823 | 0.498 | 0.357 | 0.698 | 1.927 | 0.034 | 0.742 | 0.551 | 0.365 | 0.489 | 557 | 119 | 120 | 0.258 |
| all_geometry_aggregates_plus_clinical | xgboost | 22 | 0.654 | 0.535 | 0.770 | 0.438 | 0.301 | 0.636 | 1.695 | 0.552 | 0.323 | 0.865 | 0.455 | 0.377 | 557 | 119 | 120 | 0.258 |

## 4. Best Direct Patient-Level Model per Endpoint

**GG2+ / csPCa:** `clinical_only` / `logistic_regression` — ROC-AUC 0.613 [0.515–0.716], PR-AUC 0.697 [0.604–0.816]

**GG3+ / high-grade:** `all_geometry_aggregates_plus_clinical` / `hist_gradient_boosting` — ROC-AUC 0.706 [0.583–0.823], PR-AUC 0.498 [0.357–0.698]

## 5. Comparison Against Previous Naive Core-Score Aggregation

| Endpoint | Approach | Feature set / model | ROC-AUC | PR-AUC | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|---|
| GG2+ / csPCa | Naive aggregation | XGBoost mean prob | 0.617 | 0.695 | — | — |
| GG2+ / csPCa | Direct patient-level (best) | `clinical_only` / `logistic_regression` | 0.613 | 0.697 | -0.004 | +0.002 |
| GG3+ / high-grade | Naive aggregation | Logistic Regression mean prob | 0.682 | 0.512 | — | — |
| GG3+ / high-grade | Direct patient-level (best) | `all_geometry_aggregates_plus_clinical` / `hist_gradient_boosting` | 0.706 | 0.498 | +0.024 | -0.014 |

## 6. Interpretation

### GG2+ / csPCa

**Direct vs naive aggregation:** Direct patient-level modelling and naive aggregation perform comparably (best direct: `clinical_only` / `logistic_regression`, ROC-AUC 0.613; naive: XGBoost mean prob, ROC-AUC 0.617; ΔROC = -0.004, ΔPR = +0.002).

**Target geometry aggregates vs clinical only:** Target geometry aggregates add limited incremental value over clinical features alone at the patient level (mean ROC-AUC 0.584 vs 0.592; ΔROC = -0.008).

**Patient-level vs core-level performance:** Patient-level prediction remains weaker than the best core-level model (direct patient-level ROC-AUC 0.613 vs core-level 0.758; ΔROC = -0.145). Aggregating to the patient level dilutes the core-level geometric signal.

### GG3+ / high-grade

**Direct vs naive aggregation:** Direct patient-level modelling (`all_geometry_aggregates_plus_clinical` / `hist_gradient_boosting`, ROC-AUC 0.706) improves over naive core-score aggregation (Logistic Regression mean prob, ROC-AUC 0.682; ΔROC = +0.024, ΔPR = -0.014).

**Target geometry aggregates vs clinical only:** Target geometry aggregates add limited incremental value over clinical features alone at the patient level (mean ROC-AUC 0.670 vs 0.670; ΔROC = 0.001).

**Patient-level vs core-level performance:** Patient-level prediction remains weaker than the best core-level model (direct patient-level ROC-AUC 0.706 vs core-level 0.829; ΔROC = -0.123). Aggregating to the patient level dilutes the core-level geometric signal.

### Implication for paper framing

Direct patient-level modelling shows some improvement over naive aggregation for at least one endpoint. However, patient prevalence is substantially higher than core prevalence, many patients have one positive core among many negatives, and the core-level geometry signal is diluted when summarised to a single patient-level score. The paper should report direct patient-level models as a secondary or exploratory analysis, while keeping core-level risk stratification as the primary result where the geometric signal is strongest.

