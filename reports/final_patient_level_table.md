# Patient-level Performance

Core-level predicted probabilities aggregated to patient level.  
A patient is **positive** if at least one core is positive for the endpoint.  
**Best aggregation** selected by patient-level ROC-AUC on the test split.

## Best aggregation per endpoint and model

| Endpoint | Model | Best aggregation | Prevalence | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|---|---|
| GG3+ / high-grade | logistic_regression | Mean prob | 25.8% | 0.682 | 0.512 | 0.645 | 0.629 | 0.476 | 22.6% | 38.7% |
| GG2+ / csPCa | logistic_regression | Mean prob | 58.3% | 0.578 | 0.674 | 0.729 | 0.420 | 0.680 | 12.9% | 25.7% |
| GG2+ / csPCa | xgboost | Mean prob | 58.3% | 0.617 | 0.695 | 0.571 | 0.660 | 0.630 | 12.9% | 24.3% |

## All aggregation methods

### GG3+ / high-grade — logistic_regression

| Endpoint | Model | Aggregation | Prevalence | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|---|---|
| GG3+ / high-grade | logistic_regression | Mean prob | 25.8% | 0.682 | 0.512 | 0.645 | 0.629 | 0.476 | 22.6% | 38.7% |
| GG3+ / high-grade | logistic_regression | Top-3 mean | 25.8% | 0.679 | 0.492 | 0.774 | 0.472 | 0.471 | 25.8% | 45.2% |
| GG3+ / high-grade | logistic_regression | Prop > thr | 25.8% | 0.678 | 0.467 | 0.645 | 0.663 | 0.494 | 22.6% | 35.5% |
| GG3+ / high-grade | logistic_regression | Max prob | 25.8% | 0.678 | 0.496 | 0.516 | 0.674 | 0.421 | 25.8% | 41.9% |

### GG2+ / csPCa — logistic_regression

| Endpoint | Model | Aggregation | Prevalence | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | logistic_regression | Mean prob | 58.3% | 0.578 | 0.674 | 0.729 | 0.420 | 0.680 | 12.9% | 25.7% |
| GG2+ / csPCa | logistic_regression | Prop > thr | 58.3% | 0.561 | 0.657 | 0.600 | 0.500 | 0.613 | 11.4% | 24.3% |
| GG2+ / csPCa | logistic_regression | Top-3 mean | 58.3% | 0.560 | 0.683 | 0.657 | 0.420 | 0.634 | 14.3% | 28.6% |
| GG2+ / csPCa | logistic_regression | Max prob | 58.3% | 0.559 | 0.682 | 0.614 | 0.480 | 0.619 | 15.7% | 27.1% |

### GG2+ / csPCa — xgboost

| Endpoint | Model | Aggregation | Prevalence | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | xgboost | Mean prob | 58.3% | 0.617 | 0.695 | 0.571 | 0.660 | 0.630 | 12.9% | 24.3% |
| GG2+ / csPCa | xgboost | Top-3 mean | 58.3% | 0.604 | 0.704 | 0.686 | 0.460 | 0.662 | 12.9% | 25.7% |
| GG2+ / csPCa | xgboost | Prop > thr | 58.3% | 0.595 | 0.670 | 0.657 | 0.560 | 0.667 | 12.9% | 24.3% |
| GG2+ / csPCa | xgboost | Max prob | 58.3% | 0.592 | 0.686 | 0.671 | 0.420 | 0.644 | 14.3% | 27.1% |

