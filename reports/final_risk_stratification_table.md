# Risk Stratification Summary

Best model per endpoint (by test ROC-AUC) from `analyze_risk_stratification.py`.  
Feature set: `all_geometry_no_availability_plus_clinical` (target_mesh_available excluded).  
**Decile 10** = highest-risk 10% of test cores.  
**Capture top N%** = fraction of all positive cores found in the top N% by predicted risk.

| Endpoint | Model | ROC-AUC | PR-AUC | Baseline prevalence | Decile 10 prevalence | Capture top 5% | Capture top 10% | Capture top 20% |
|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | xgboost | 0.758 | 0.327 | 10.7% | 36.1% | 22.8% | 33.6% | 51.7% |
| GG3+ / high-grade | logistic_regression | 0.829 | 0.265 | 3.8% | 18.7% | 40.2% | 48.9% | 62.0% |

