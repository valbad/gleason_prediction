# Risk Stratification Analysis

Feature set: **all_geometry_no_availability_plus_clinical**  
(`target_mesh_available` excluded by design)

Probabilities are predicted on the test split.  No hard threshold is applied.  
Decile 10 = highest predicted risk; decile 1 = lowest.

## binary_label_int — GG2+ / csPCa (Gleason ≥ 3+4=7)

- Test cores: 2,417
- Test positives: 259 (10.7%)
- Features: 14 (all_geometry_no_availability_plus_clinical)

### logistic_regression

Test ROC-AUC = **0.753**  ·  PR-AUC = **0.319**  ·  baseline PR-AUC = **10.7%** (no-skill)

#### Risk deciles (10 = highest risk)

| Decile | Risk tier | Cores | Positives | Observed prevalence | Mean predicted prob | Min | Max |
|---|---|---|---|---|---|---|---|
| 10 | highest | 241 | 85 | 35.3% | 0.826 | 0.736 | 0.970 |
| 9 |  | 242 | 49 | 20.2% | 0.669 | 0.614 | 0.734 |
| 8 |  | 242 | 35 | 14.5% | 0.561 | 0.503 | 0.613 |
| 7 |  | 241 | 22 | 9.1% | 0.453 | 0.406 | 0.502 |
| 6 |  | 242 | 17 | 7.0% | 0.360 | 0.321 | 0.406 |
| 5 |  | 242 | 18 | 7.4% | 0.279 | 0.239 | 0.321 |
| 4 |  | 241 | 9 | 3.7% | 0.207 | 0.178 | 0.239 |
| 3 |  | 242 | 9 | 3.7% | 0.148 | 0.120 | 0.178 |
| 2 |  | 242 | 7 | 2.9% | 0.096 | 0.072 | 0.120 |
| 1 | lowest | 242 | 8 | 3.3% | 0.043 | 0.006 | 0.072 |

#### Positive capture in top highest-risk cores

| Top % of cores | Cores in top | Positives captured | Total positives | Capture rate |
|---|---|---|---|---|
| 5.0% | 120 | 57 | 259 | 22.0% |
| 10.0% | 241 | 85 | 259 | 32.8% |
| 20.0% | 483 | 134 | 259 | 51.7% |

#### Calibration (predicted vs observed prevalence)

| Predicted prob bin | Cores | Positives | Observed prevalence | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1) | 375 | 11 | 2.9% | 0.058 |
| [0.1, 0.2) | 438 | 20 | 4.6% | 0.146 |
| [0.2, 0.3) | 338 | 15 | 4.4% | 0.246 |
| [0.3, 0.4) | 287 | 21 | 7.3% | 0.348 |
| [0.4, 0.5) | 252 | 23 | 9.1% | 0.450 |
| [0.5, 0.6) | 211 | 28 | 13.3% | 0.553 |
| [0.6, 0.7) | 216 | 48 | 22.2% | 0.647 |
| [0.7, 0.8) | 162 | 32 | 19.8% | 0.747 |
| [0.8, 0.9) | 105 | 40 | 38.1% | 0.848 |
| [0.9, 1.0) | 33 | 21 | 63.6% | 0.947 |

### hist_gradient_boosting

Test ROC-AUC = **0.736**  ·  PR-AUC = **0.298**  ·  baseline PR-AUC = **10.7%** (no-skill)

#### Risk deciles (10 = highest risk)

| Decile | Risk tier | Cores | Positives | Observed prevalence | Mean predicted prob | Min | Max |
|---|---|---|---|---|---|---|---|
| 10 | highest | 241 | 85 | 35.3% | 0.492 | 0.274 | 0.957 |
| 9 |  | 242 | 36 | 14.9% | 0.209 | 0.158 | 0.274 |
| 8 |  | 242 | 38 | 15.7% | 0.122 | 0.095 | 0.158 |
| 7 |  | 241 | 20 | 8.3% | 0.076 | 0.060 | 0.095 |
| 6 |  | 242 | 26 | 10.7% | 0.049 | 0.039 | 0.059 |
| 5 |  | 242 | 16 | 6.6% | 0.031 | 0.025 | 0.039 |
| 4 |  | 241 | 13 | 5.4% | 0.021 | 0.016 | 0.025 |
| 3 |  | 242 | 8 | 3.3% | 0.013 | 0.010 | 0.016 |
| 2 |  | 242 | 8 | 3.3% | 0.007 | 0.005 | 0.010 |
| 1 | lowest | 242 | 9 | 3.7% | 0.003 | 0.001 | 0.005 |

#### Positive capture in top highest-risk cores

| Top % of cores | Cores in top | Positives captured | Total positives | Capture rate |
|---|---|---|---|---|
| 5.0% | 120 | 59 | 259 | 22.8% |
| 10.0% | 241 | 85 | 259 | 32.8% |
| 20.0% | 483 | 121 | 259 | 46.7% |

#### Calibration (predicted vs observed prevalence)

| Predicted prob bin | Cores | Positives | Observed prevalence | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1) | 1716 | 102 | 5.9% | 0.030 |
| [0.1, 0.2) | 329 | 50 | 15.2% | 0.143 |
| [0.2, 0.3) | 159 | 28 | 17.6% | 0.244 |
| [0.3, 0.4) | 76 | 15 | 19.7% | 0.345 |
| [0.4, 0.5) | 38 | 16 | 42.1% | 0.444 |
| [0.5, 0.6) | 31 | 10 | 32.3% | 0.546 |
| [0.6, 0.7) | 30 | 18 | 60.0% | 0.644 |
| [0.7, 0.8) | 18 | 10 | 55.6% | 0.755 |
| [0.8, 0.9) | 12 | 7 | 58.3% | 0.848 |
| [0.9, 1.0) | 8 | 3 | 37.5% | 0.928 |

### xgboost

Test ROC-AUC = **0.758**  ·  PR-AUC = **0.327**  ·  baseline PR-AUC = **10.7%** (no-skill)

#### Risk deciles (10 = highest risk)

| Decile | Risk tier | Cores | Positives | Observed prevalence | Mean predicted prob | Min | Max |
|---|---|---|---|---|---|---|---|
| 10 | highest | 241 | 87 | 36.1% | 0.827 | 0.719 | 0.969 |
| 9 |  | 242 | 47 | 19.4% | 0.656 | 0.589 | 0.718 |
| 8 |  | 242 | 39 | 16.1% | 0.525 | 0.462 | 0.589 |
| 7 |  | 241 | 25 | 10.4% | 0.400 | 0.341 | 0.460 |
| 6 |  | 242 | 14 | 5.8% | 0.302 | 0.267 | 0.341 |
| 5 |  | 242 | 15 | 6.2% | 0.234 | 0.205 | 0.267 |
| 4 |  | 241 | 6 | 2.5% | 0.170 | 0.141 | 0.205 |
| 3 |  | 242 | 4 | 1.7% | 0.121 | 0.102 | 0.141 |
| 2 |  | 242 | 11 | 4.5% | 0.085 | 0.067 | 0.102 |
| 1 | lowest | 242 | 11 | 4.5% | 0.042 | 0.011 | 0.067 |

#### Positive capture in top highest-risk cores

| Top % of cores | Cores in top | Positives captured | Total positives | Capture rate |
|---|---|---|---|---|
| 5.0% | 120 | 59 | 259 | 22.8% |
| 10.0% | 241 | 87 | 259 | 33.6% |
| 20.0% | 483 | 134 | 259 | 51.7% |

#### Calibration (predicted vs observed prevalence)

| Predicted prob bin | Cores | Positives | Observed prevalence | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1) | 473 | 20 | 4.2% | 0.063 |
| [0.1, 0.2) | 481 | 12 | 2.5% | 0.143 |
| [0.2, 0.3) | 369 | 21 | 5.7% | 0.248 |
| [0.3, 0.4) | 252 | 21 | 8.3% | 0.344 |
| [0.4, 0.5) | 190 | 23 | 12.1% | 0.450 |
| [0.5, 0.6) | 191 | 35 | 18.3% | 0.551 |
| [0.6, 0.7) | 182 | 32 | 17.6% | 0.652 |
| [0.7, 0.8) | 140 | 33 | 23.6% | 0.741 |
| [0.8, 0.9) | 85 | 32 | 37.6% | 0.846 |
| [0.9, 1.0) | 54 | 30 | 55.6% | 0.937 |

## binary_label_gg3plus_int — GG3+ / high-grade (Gleason ≥ 4+3=7)

- Test cores: 2,417
- Test positives: 92 (3.8%)
- Features: 14 (all_geometry_no_availability_plus_clinical)

### logistic_regression

Test ROC-AUC = **0.829**  ·  PR-AUC = **0.265**  ·  baseline PR-AUC = **3.8%** (no-skill)

#### Risk deciles (10 = highest risk)

| Decile | Risk tier | Cores | Positives | Observed prevalence | Mean predicted prob | Min | Max |
|---|---|---|---|---|---|---|---|
| 10 | highest | 241 | 45 | 18.7% | 0.808 | 0.664 | 0.981 |
| 9 |  | 242 | 12 | 5.0% | 0.594 | 0.524 | 0.664 |
| 8 |  | 242 | 13 | 5.4% | 0.470 | 0.415 | 0.524 |
| 7 |  | 241 | 7 | 2.9% | 0.357 | 0.305 | 0.415 |
| 6 |  | 242 | 7 | 2.9% | 0.258 | 0.217 | 0.305 |
| 5 |  | 242 | 6 | 2.5% | 0.187 | 0.157 | 0.217 |
| 4 |  | 241 | 1 | 0.4% | 0.130 | 0.103 | 0.157 |
| 3 |  | 242 | 1 | 0.4% | 0.084 | 0.065 | 0.103 |
| 2 |  | 242 | 0 | 0.0% | 0.048 | 0.033 | 0.065 |
| 1 | lowest | 242 | 0 | 0.0% | 0.018 | 0.003 | 0.033 |

#### Positive capture in top highest-risk cores

| Top % of cores | Cores in top | Positives captured | Total positives | Capture rate |
|---|---|---|---|---|
| 5.0% | 120 | 37 | 92 | 40.2% |
| 10.0% | 241 | 45 | 92 | 48.9% |
| 20.0% | 483 | 57 | 92 | 62.0% |

#### Calibration (predicted vs observed prevalence)

| Predicted prob bin | Cores | Positives | Observed prevalence | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1) | 698 | 1 | 0.1% | 0.048 |
| [0.1, 0.2) | 444 | 7 | 1.6% | 0.147 |
| [0.2, 0.3) | 302 | 7 | 2.3% | 0.246 |
| [0.3, 0.4) | 214 | 7 | 3.3% | 0.347 |
| [0.4, 0.5) | 217 | 11 | 5.1% | 0.449 |
| [0.5, 0.6) | 190 | 10 | 5.3% | 0.546 |
| [0.6, 0.7) | 149 | 5 | 3.4% | 0.644 |
| [0.7, 0.8) | 86 | 7 | 8.1% | 0.753 |
| [0.8, 0.9) | 71 | 15 | 21.1% | 0.849 |
| [0.9, 1.0) | 46 | 22 | 47.8% | 0.953 |

