# Delta Table: Feature Set Comparison vs Full Model

**Reference:** `all_geometry_no_availability_plus_clinical` (14 features).  
ΔROC-AUC and ΔPR-AUC = (this feature set) − (reference).  
Negative values indicate underperformance relative to the reference.

### GG2+ / csPCa

#### logistic_regression

| Feature set | n_feat | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| `clinical_only` | 4 | -0.103 | -0.097 |
| `target_geometry_only` | 4 | -0.023 | -0.095 |
| `target_geometry_plus_clinical` | 8 | +0.001 | +0.007 |
| `biopsy_prostate_geometry_plus_clinical` | 10 | -0.108 | -0.112 |
| `all_geometry_no_clinical` | 10 | -0.030 | -0.103 |
| `all_geometry_no_availability_plus_clinical` | 14 | +0.000 | +0.000 |

#### hist_gradient_boosting

| Feature set | n_feat | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| `clinical_only` | 4 | -0.136 | -0.090 |
| `target_geometry_only` | 4 | -0.018 | -0.080 |
| `target_geometry_plus_clinical` | 8 | +0.009 | -0.003 |
| `biopsy_prostate_geometry_plus_clinical` | 10 | -0.106 | -0.060 |
| `all_geometry_no_clinical` | 10 | -0.022 | -0.080 |
| `all_geometry_no_availability_plus_clinical` | 14 | +0.000 | +0.000 |

#### xgboost

| Feature set | n_feat | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| `clinical_only` | 4 | -0.133 | -0.087 |
| `target_geometry_only` | 4 | -0.032 | -0.103 |
| `target_geometry_plus_clinical` | 8 | -0.004 | -0.006 |
| `biopsy_prostate_geometry_plus_clinical` | 10 | -0.114 | -0.073 |
| `all_geometry_no_clinical` | 10 | -0.037 | -0.104 |
| `all_geometry_no_availability_plus_clinical` | 14 | +0.000 | +0.000 |

### GG3+ / high-grade

#### logistic_regression

| Feature set | n_feat | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| `clinical_only` | 4 | -0.113 | -0.109 |
| `target_geometry_only` | 4 | -0.023 | -0.127 |
| `target_geometry_plus_clinical` | 8 | -0.001 | +0.004 |
| `biopsy_prostate_geometry_plus_clinical` | 10 | -0.109 | -0.111 |
| `all_geometry_no_clinical` | 10 | -0.020 | -0.138 |
| `all_geometry_no_availability_plus_clinical` | 14 | +0.000 | +0.000 |

#### hist_gradient_boosting

| Feature set | n_feat | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| `clinical_only` | 4 | -0.087 | -0.083 |
| `target_geometry_only` | 4 | -0.018 | -0.157 |
| `target_geometry_plus_clinical` | 8 | +0.012 | -0.016 |
| `biopsy_prostate_geometry_plus_clinical` | 10 | -0.062 | -0.077 |
| `all_geometry_no_clinical` | 10 | -0.017 | -0.167 |
| `all_geometry_no_availability_plus_clinical` | 14 | +0.000 | +0.000 |

#### xgboost

| Feature set | n_feat | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| `clinical_only` | 4 | -0.075 | -0.109 |
| `target_geometry_only` | 4 | -0.017 | -0.141 |
| `target_geometry_plus_clinical` | 8 | +0.004 | -0.013 |
| `biopsy_prostate_geometry_plus_clinical` | 10 | -0.089 | -0.081 |
| `all_geometry_no_clinical` | 10 | -0.035 | -0.130 |
| `all_geometry_no_availability_plus_clinical` | 14 | +0.000 | +0.000 |

