# Probability Calibration Analysis

Feature set: `all_geometry_no_availability_plus_clinical`  
(target_mesh_available excluded).  
Recalibration: logistic Platt scaling fitted on val-split logit(predicted probabilities).

**Calibration slope** b ≈ 1 → well-spread; b < 1 → over-confident; b > 1 → under-confident.  
**ECE** = expected calibration error (10 equal-width bins); lower is better.  
**Brier skill score** = 1 − Brier / Brier_null; higher is better.

## binary_label_int — GG2+ / csPCa (Gleason ≥ 3+4=7)

- Test cores: 2,417  ·  Prevalence: **10.7%**

### logistic_regression

#### Calibration metrics summary

| Metric | Before recalibration | After recalibration |
|---|---|---|
| Brier score | 0.177 | 0.085 |
| Brier skill score | -0.850 | 0.108 |
| Calibration intercept | -1.939 | 0.185 |
| Calibration slope | 0.736 | 0.966 |
| ECE (10 bins) | 0.257 | 0.020 |

Recalibration model fitted on val split:  
logit(p_recal) = -2.197 + 0.762 × logit(p_base)

#### Calibration bins — before recalibration

| Bin | Cores | Positives | Observed prev | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1] | 375 | 11 | 2.9% | 0.058 |
| [0.1, 0.2] | 438 | 20 | 4.6% | 0.146 |
| [0.2, 0.3] | 338 | 15 | 4.4% | 0.246 |
| [0.3, 0.4] | 287 | 21 | 7.3% | 0.348 |
| [0.4, 0.5] | 252 | 23 | 9.1% | 0.450 |
| [0.5, 0.6] | 211 | 28 | 13.3% | 0.553 |
| [0.6, 0.7] | 216 | 48 | 22.2% | 0.647 |
| [0.7, 0.8] | 162 | 32 | 19.8% | 0.747 |
| [0.8, 0.9] | 105 | 40 | 38.1% | 0.848 |
| [0.9, 1.0] | 33 | 21 | 63.6% | 0.947 |

#### Calibration bins — after recalibration

| Bin | Cores | Positives | Observed prev | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1] | 1690 | 90 | 5.3% | 0.043 |
| [0.1, 0.2] | 505 | 88 | 17.4% | 0.142 |
| [0.2, 0.3] | 146 | 39 | 26.7% | 0.240 |
| [0.3, 0.4] | 45 | 23 | 51.1% | 0.335 |
| [0.4, 0.5] | 11 | 6 | 54.5% | 0.461 |
| [0.5, 0.6] | 18 | 12 | 66.7% | 0.544 |
| [0.6, 0.7] | 2 | 1 | 50.0% | 0.611 |
| [0.7, 0.8] | 0 | 0 | n/a | n/a |
| [0.8, 0.9] | 0 | 0 | n/a | n/a |
| [0.9, 1.0] | 0 | 0 | n/a | n/a |

**Interpretation:**

The base model is **over-confident**: its probability spread is too extreme (calibration slope = 0.736).  ECE (0.257) is high (> 0.10), indicating significant miscalibration.

Recalibration reduces ECE from 0.257 to 0.020 (−0.237), indicating the raw probabilities were shifted relative to the true positive rate.

For this endpoint (`binary_label_int`, prevalence ≈ 10.7%), the model is primarily useful as a **ranking tool**. Recalibration is recommended before reporting absolute risk values.

### xgboost

#### Calibration metrics summary

| Metric | Before recalibration | After recalibration |
|---|---|---|
| Brier score | 0.162 | 0.085 |
| Brier skill score | -0.690 | 0.116 |
| Calibration intercept | -1.854 | 0.209 |
| Calibration slope | 0.706 | 0.977 |
| ECE (10 bins) | 0.229 | 0.020 |

Recalibration model fitted on val split:  
logit(p_recal) = -2.112 + 0.723 × logit(p_base)

#### Calibration bins — before recalibration

| Bin | Cores | Positives | Observed prev | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1] | 473 | 20 | 4.2% | 0.063 |
| [0.1, 0.2] | 481 | 12 | 2.5% | 0.143 |
| [0.2, 0.3] | 369 | 21 | 5.7% | 0.248 |
| [0.3, 0.4] | 252 | 21 | 8.3% | 0.344 |
| [0.4, 0.5] | 190 | 23 | 12.1% | 0.450 |
| [0.5, 0.6] | 191 | 35 | 18.3% | 0.551 |
| [0.6, 0.7] | 182 | 32 | 17.6% | 0.652 |
| [0.7, 0.8] | 140 | 33 | 23.6% | 0.741 |
| [0.8, 0.9] | 85 | 32 | 37.6% | 0.846 |
| [0.9, 1.0] | 54 | 30 | 55.6% | 0.937 |

#### Calibration bins — after recalibration

| Bin | Cores | Positives | Observed prev | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1] | 1715 | 92 | 5.4% | 0.043 |
| [0.1, 0.2] | 488 | 84 | 17.2% | 0.144 |
| [0.2, 0.3] | 124 | 35 | 28.2% | 0.240 |
| [0.3, 0.4] | 52 | 24 | 46.2% | 0.350 |
| [0.4, 0.5] | 15 | 10 | 66.7% | 0.451 |
| [0.5, 0.6] | 23 | 14 | 60.9% | 0.549 |
| [0.6, 0.7] | 0 | 0 | n/a | n/a |
| [0.7, 0.8] | 0 | 0 | n/a | n/a |
| [0.8, 0.9] | 0 | 0 | n/a | n/a |
| [0.9, 1.0] | 0 | 0 | n/a | n/a |

**Interpretation:**

The base model is **over-confident**: its probability spread is too extreme (calibration slope = 0.706).  ECE (0.229) is high (> 0.10), indicating significant miscalibration.

Recalibration reduces ECE from 0.229 to 0.020 (−0.209), indicating the raw probabilities were shifted relative to the true positive rate.

For this endpoint (`binary_label_int`, prevalence ≈ 10.7%), the model is primarily useful as a **ranking tool**. Recalibration is recommended before reporting absolute risk values.

### Cross-model calibration summary

| Model | Brier (before→after) | ECE (before→after) | Slope (before→after) |
|---|---|---|---|
| logistic_regression | 0.177 → 0.085 | 0.257 → 0.020 | 0.736 → 0.966 |
| xgboost | 0.162 → 0.085 | 0.229 → 0.020 | 0.706 → 0.977 |

## binary_label_gg3plus_int — GG3+ / high-grade (Gleason ≥ 4+3=7)

- Test cores: 2,417  ·  Prevalence: **3.8%**

### logistic_regression

#### Calibration metrics summary

| Metric | Before recalibration | After recalibration |
|---|---|---|
| Brier score | 0.139 | 0.034 |
| Brier skill score | -2.801 | 0.084 |
| Calibration intercept | -3.021 | 2.076 |
| Calibration slope | 0.880 | 1.511 |
| ECE (10 bins) | 0.257 | 0.013 |

Recalibration model fitted on val split:  
logit(p_recal) = -3.372 + 0.582 × logit(p_base)

#### Calibration bins — before recalibration

| Bin | Cores | Positives | Observed prev | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1] | 698 | 1 | 0.1% | 0.048 |
| [0.1, 0.2] | 444 | 7 | 1.6% | 0.147 |
| [0.2, 0.3] | 302 | 7 | 2.3% | 0.246 |
| [0.3, 0.4] | 214 | 7 | 3.3% | 0.347 |
| [0.4, 0.5] | 217 | 11 | 5.1% | 0.449 |
| [0.5, 0.6] | 190 | 10 | 5.3% | 0.546 |
| [0.6, 0.7] | 149 | 5 | 3.4% | 0.644 |
| [0.7, 0.8] | 86 | 7 | 8.1% | 0.753 |
| [0.8, 0.9] | 71 | 15 | 21.1% | 0.849 |
| [0.9, 1.0] | 46 | 22 | 47.8% | 0.953 |

#### Calibration bins — after recalibration

| Bin | Cores | Positives | Observed prev | Mean predicted |
|---|---|---|---|---|
| [0.0, 0.1] | 2362 | 68 | 2.9% | 0.021 |
| [0.1, 0.2] | 37 | 15 | 40.5% | 0.140 |
| [0.2, 0.3] | 18 | 9 | 50.0% | 0.223 |
| [0.3, 0.4] | 0 | 0 | n/a | n/a |
| [0.4, 0.5] | 0 | 0 | n/a | n/a |
| [0.5, 0.6] | 0 | 0 | n/a | n/a |
| [0.6, 0.7] | 0 | 0 | n/a | n/a |
| [0.7, 0.8] | 0 | 0 | n/a | n/a |
| [0.8, 0.9] | 0 | 0 | n/a | n/a |
| [0.9, 1.0] | 0 | 0 | n/a | n/a |

**Interpretation:**

The base model slope is near 1 (0.880), suggesting reasonable spread.  ECE (0.257) is high (> 0.10), indicating significant miscalibration.

Recalibration reduces ECE from 0.257 to 0.013 (−0.244), indicating the raw probabilities were shifted relative to the true positive rate.

> **Caution:** After recalibration, most predictions are concentrated in the lowest probability bin, which improves average calibration metrics but suggests that absolute risk estimates remain conservative and should be interpreted cautiously.

For this endpoint (`binary_label_gg3plus_int`, prevalence ≈ 3.8%), the model is primarily useful as a **ranking tool**. Recalibration is recommended before reporting absolute risk values.

### Cross-model calibration summary

| Model | Brier (before→after) | ECE (before→after) | Slope (before→after) |
|---|---|---|---|
| logistic_regression | 0.139 → 0.034 | 0.257 → 0.013 | 0.880 → 1.511 |

