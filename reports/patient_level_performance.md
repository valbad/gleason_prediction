# Patient-level Performance Analysis

Core-level predicted probabilities are aggregated to the patient level.
A patient is **positive** if at least one of their cores is positive for the endpoint.

**Core-level Youden-J threshold** is selected on the val split and used for  
`prop_above_threshold`.  A **patient-level Youden-J threshold** is also selected  
for each aggregation method on val-split patient scores — this drives binary metrics.

Decile 10 = highest predicted risk; decile 1 = lowest.

## binary_label_int — GG2+ / csPCa (Gleason ≥ 3+4=7)

- Test patients: **120**
- Positive test patients: **70** (58.3%)
- Val patients: 119

### logistic_regression

Core-level Youden-J threshold (val): **0.520**

#### Performance by aggregation method

| Aggregation | Threshold | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|
| Max prob | 0.670 | 0.559 | 0.682 | 0.614 | 0.480 | 0.619 | 15.7% | 27.1% |
| Top-3 mean | 0.626 | 0.560 | 0.683 | 0.657 | 0.420 | 0.634 | 14.3% | 28.6% |
| Mean prob | 0.286 | 0.578 | 0.674 | 0.729 | 0.420 | 0.680 | 12.9% | 25.7% |
| Prop > thr | 0.241 | 0.561 | 0.657 | 0.600 | 0.500 | 0.613 | 11.4% | 24.3% |

<details><summary>Risk deciles — Max prob</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 11 | 91.7% | 0.923 |
| 9 | 12 | 8 | 66.7% | 0.862 |
| 8 | 12 | 4 | 33.3% | 0.799 |
| 7 | 12 | 6 | 50.0% | 0.754 |
| 6 | 12 | 6 | 50.0% | 0.716 |
| 5 | 12 | 10 | 83.3% | 0.679 |
| 4 | 12 | 7 | 58.3% | 0.630 |
| 3 | 12 | 5 | 41.7% | 0.569 |
| 2 | 12 | 5 | 41.7% | 0.483 |
| 1 ← lowest risk | 12 | 8 | 66.7% | 0.372 |

</details>

<details><summary>Risk deciles — Top-3 mean</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 10 | 83.3% | 0.908 |
| 9 | 12 | 10 | 83.3% | 0.839 |
| 8 | 12 | 3 | 25.0% | 0.778 |
| 7 | 12 | 6 | 50.0% | 0.734 |
| 6 | 12 | 7 | 58.3% | 0.687 |
| 5 | 12 | 8 | 66.7% | 0.649 |
| 4 | 12 | 9 | 75.0% | 0.602 |
| 3 | 12 | 3 | 25.0% | 0.536 |
| 2 | 12 | 5 | 41.7% | 0.450 |
| 1 ← lowest risk | 12 | 9 | 75.0% | 0.334 |

</details>

<details><summary>Risk deciles — Mean prob</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 9 | 75.0% | 0.778 |
| 9 | 12 | 9 | 75.0% | 0.585 |
| 8 | 12 | 5 | 41.7% | 0.482 |
| 7 | 12 | 8 | 66.7% | 0.418 |
| 6 | 12 | 6 | 50.0% | 0.384 |
| 5 | 12 | 9 | 75.0% | 0.341 |
| 4 | 12 | 6 | 50.0% | 0.294 |
| 3 | 12 | 6 | 50.0% | 0.260 |
| 2 | 12 | 4 | 33.3% | 0.227 |
| 1 ← lowest risk | 12 | 8 | 66.7% | 0.141 |

</details>

<details><summary>Risk deciles — Prop > thr</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 8 | 66.7% | 0.934 |
| 9 | 12 | 9 | 75.0% | 0.627 |
| 8 | 12 | 8 | 66.7% | 0.448 |
| 7 | 12 | 6 | 50.0% | 0.368 |
| 6 | 12 | 6 | 50.0% | 0.306 |
| 5 | 12 | 8 | 66.7% | 0.246 |
| 4 | 12 | 6 | 50.0% | 0.189 |
| 3 | 12 | 6 | 50.0% | 0.100 |
| 2 | 12 | 6 | 50.0% | 0.003 |
| 1 ← lowest risk | 12 | 7 | 58.3% | 0.000 |

</details>

### xgboost

Core-level Youden-J threshold (val): **0.547**

#### Performance by aggregation method

| Aggregation | Threshold | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|
| Max prob | 0.612 | 0.592 | 0.686 | 0.671 | 0.420 | 0.644 | 14.3% | 27.1% |
| Top-3 mean | 0.582 | 0.604 | 0.704 | 0.686 | 0.460 | 0.662 | 12.9% | 25.7% |
| Mean prob | 0.345 | 0.617 | 0.695 | 0.571 | 0.660 | 0.630 | 12.9% | 24.3% |
| Prop > thr | 0.158 | 0.595 | 0.670 | 0.657 | 0.560 | 0.667 | 12.9% | 24.3% |

<details><summary>Risk deciles — Max prob</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 10 | 83.3% | 0.936 |
| 9 | 12 | 9 | 75.0% | 0.855 |
| 8 | 12 | 6 | 50.0% | 0.791 |
| 7 | 12 | 7 | 58.3% | 0.727 |
| 6 | 12 | 6 | 50.0% | 0.693 |
| 5 | 12 | 9 | 75.0% | 0.651 |
| 4 | 12 | 4 | 33.3% | 0.593 |
| 3 | 12 | 6 | 50.0% | 0.521 |
| 2 | 12 | 5 | 41.7% | 0.461 |
| 1 ← lowest risk | 12 | 8 | 66.7% | 0.281 |

</details>

<details><summary>Risk deciles — Top-3 mean</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 9 | 75.0% | 0.923 |
| 9 | 12 | 9 | 75.0% | 0.834 |
| 8 | 12 | 8 | 66.7% | 0.770 |
| 7 | 12 | 8 | 66.7% | 0.708 |
| 6 | 12 | 4 | 33.3% | 0.666 |
| 5 | 12 | 8 | 66.7% | 0.615 |
| 4 | 12 | 6 | 50.0% | 0.549 |
| 3 | 12 | 5 | 41.7% | 0.473 |
| 2 | 12 | 5 | 41.7% | 0.398 |
| 1 ← lowest risk | 12 | 8 | 66.7% | 0.261 |

</details>

<details><summary>Risk deciles — Mean prob</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 9 | 75.0% | 0.788 |
| 9 | 12 | 8 | 66.7% | 0.560 |
| 8 | 12 | 7 | 58.3% | 0.452 |
| 7 | 12 | 10 | 83.3% | 0.388 |
| 6 | 12 | 8 | 66.7% | 0.353 |
| 5 | 12 | 6 | 50.0% | 0.315 |
| 4 | 12 | 6 | 50.0% | 0.266 |
| 3 | 12 | 4 | 33.3% | 0.228 |
| 2 | 12 | 5 | 41.7% | 0.182 |
| 1 ← lowest risk | 12 | 7 | 58.3% | 0.113 |

</details>

<details><summary>Risk deciles — Prop > thr</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 9 | 75.0% | 0.908 |
| 9 | 12 | 8 | 66.7% | 0.547 |
| 8 | 12 | 7 | 58.3% | 0.385 |
| 7 | 12 | 8 | 66.7% | 0.320 |
| 6 | 12 | 8 | 66.7% | 0.260 |
| 5 | 12 | 6 | 50.0% | 0.183 |
| 4 | 12 | 5 | 41.7% | 0.084 |
| 3 | 12 | 6 | 50.0% | 0.000 |
| 2 | 12 | 5 | 41.7% | 0.000 |
| 1 ← lowest risk | 12 | 8 | 66.7% | 0.000 |

</details>

## binary_label_gg3plus_int — GG3+ / high-grade (Gleason ≥ 4+3=7)

- Test patients: **120**
- Positive test patients: **31** (25.8%)
- Val patients: 119

### logistic_regression

Core-level Youden-J threshold (val): **0.342**

#### Performance by aggregation method

| Aggregation | Threshold | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|
| Max prob | 0.674 | 0.678 | 0.496 | 0.516 | 0.674 | 0.421 | 25.8% | 41.9% |
| Top-3 mean | 0.507 | 0.679 | 0.492 | 0.774 | 0.472 | 0.471 | 25.8% | 45.2% |
| Mean prob | 0.316 | 0.682 | 0.512 | 0.645 | 0.629 | 0.476 | 22.6% | 38.7% |
| Prop > thr | 0.450 | 0.678 | 0.467 | 0.645 | 0.663 | 0.494 | 22.6% | 35.5% |

<details><summary>Risk deciles — Max prob</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 8 | 66.7% | 0.928 |
| 9 | 12 | 5 | 41.7% | 0.823 |
| 8 | 12 | 2 | 16.7% | 0.749 |
| 7 | 12 | 1 | 8.3% | 0.677 |
| 6 | 12 | 3 | 25.0% | 0.634 |
| 5 | 12 | 5 | 41.7% | 0.561 |
| 4 | 12 | 2 | 16.7% | 0.501 |
| 3 | 12 | 2 | 16.7% | 0.418 |
| 2 | 12 | 1 | 8.3% | 0.304 |
| 1 ← lowest risk | 12 | 2 | 16.7% | 0.204 |

</details>

<details><summary>Risk deciles — Top-3 mean</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 8 | 66.7% | 0.915 |
| 9 | 12 | 6 | 50.0% | 0.805 |
| 8 | 12 | 1 | 8.3% | 0.719 |
| 7 | 12 | 1 | 8.3% | 0.648 |
| 6 | 12 | 3 | 25.0% | 0.600 |
| 5 | 12 | 5 | 41.7% | 0.532 |
| 4 | 12 | 2 | 16.7% | 0.462 |
| 3 | 12 | 2 | 16.7% | 0.389 |
| 2 | 12 | 1 | 8.3% | 0.283 |
| 1 ← lowest risk | 12 | 2 | 16.7% | 0.183 |

</details>

<details><summary>Risk deciles — Mean prob</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 7 | 58.3% | 0.787 |
| 9 | 12 | 5 | 41.7% | 0.554 |
| 8 | 12 | 5 | 41.7% | 0.429 |
| 7 | 12 | 2 | 16.7% | 0.359 |
| 6 | 12 | 1 | 8.3% | 0.313 |
| 5 | 12 | 2 | 16.7% | 0.240 |
| 4 | 12 | 3 | 25.0% | 0.204 |
| 3 | 12 | 3 | 25.0% | 0.170 |
| 2 | 12 | 1 | 8.3% | 0.132 |
| 1 ← lowest risk | 12 | 2 | 16.7% | 0.078 |

</details>

<details><summary>Risk deciles — Prop > thr</summary>

| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |
|---|---|---|---|---|
| 10 ← highest risk | 12 | 7 | 58.3% | 0.990 |
| 9 | 12 | 4 | 33.3% | 0.836 |
| 8 | 12 | 6 | 50.0% | 0.632 |
| 7 | 12 | 2 | 16.7% | 0.520 |
| 6 | 12 | 1 | 8.3% | 0.394 |
| 5 | 12 | 1 | 8.3% | 0.288 |
| 4 | 12 | 4 | 33.3% | 0.238 |
| 3 | 12 | 3 | 25.0% | 0.153 |
| 2 | 12 | 3 | 25.0% | 0.004 |
| 1 ← lowest risk | 12 | 0 | 0.0% | 0.000 |

</details>

