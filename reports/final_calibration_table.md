# Calibration Analysis Summary

Calibration metrics before and after logistic recalibration (Platt scaling  
on val-split logit(predicted probabilities)).  
**BSS** = Brier skill score (1 − Brier / Brier_null); higher is better; < 0 = worse than naive.  
**ECE** = expected calibration error (10 equal-width bins); lower is better.  
**Slope** b ≈ 1 → well-spread; b < 1 → over-confident; b > 1 → under-confident.

| Endpoint | Model | Brier (before) | Brier (after) | BSS (before) | BSS (after) | ECE (before) | ECE (after) | Slope (before) | Slope (after) | Intercept (before) | Intercept (after) | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | logistic_regression | 0.177 | 0.085 | -0.850 | 0.108 | 0.257 | 0.020 | 0.736 | 0.966 | -1.939 | 0.185 | Recalibration improves calibration. Raw probabilities poor; recalibrated probabilities usable with caution. Discrimination / ranking should be interpreted separately from absolute risk calibration. |
| GG2+ / csPCa | xgboost | 0.162 | 0.085 | -0.690 | 0.116 | 0.229 | 0.020 | 0.706 | 0.977 | -1.854 | 0.209 | Recalibration improves calibration. Raw probabilities poor; recalibrated probabilities usable with caution. Discrimination / ranking should be interpreted separately from absolute risk calibration. |
| GG3+ / high-grade | logistic_regression | 0.139 | 0.034 | -2.801 | 0.084 | 0.257 | 0.013 | 0.880 | 1.511 | -3.021 | 2.076 | Recalibration improves calibration. Raw probabilities poor; recalibrated probabilities usable with caution. After recalibration, predictions concentrate in the lowest probability bin; absolute risk estimates remain conservative. Discrimination / ranking should be interpreted separately from absolute risk calibration. |

