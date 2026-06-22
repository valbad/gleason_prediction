# Main Model Performance

Mode: `full_geometry_dataset`. Best model selected by test ROC-AUC within each (endpoint, experiment) group.  
95% CI from patient-level bootstrap (omitted if experiments were run with `--no-bootstrap`).

| Endpoint | Prevalence | Experiment | Best model | ROC-AUC [95% CI] | PR-AUC [95% CI] | PR-lift | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | 10.7% | target_geometry_no_availability | logistic_regression | 0.730 [0.669 – 0.771] | 0.224 [0.163 – 0.320] | 2.09 | 0.707 | 0.653 | 0.197 | 0.308 |
| GG3+ / high-grade | 3.8% | target_geometry_no_availability | logistic_regression | 0.806 [0.736 – 0.868] | 0.138 [0.054 – 0.287] | 3.64 | 0.772 | 0.680 | 0.087 | 0.156 |
| GG2+ / csPCa | 10.7% | all_geometry_no_availability_plus_clinical | xgboost | 0.758 [0.692 – 0.804] | 0.327 [0.233 – 0.437] | 3.05 | 0.552 | 0.803 | 0.251 | 0.345 |
| GG3+ / high-grade | 3.8% | all_geometry_no_availability_plus_clinical | logistic_regression | 0.829 [0.760 – 0.888] | 0.265 [0.106 – 0.502] | 6.96 | 0.815 | 0.657 | 0.086 | 0.156 |

> **PR-lift** = PR-AUC / test prevalence. Values > 1 indicate the model beats the no-skill baseline.

