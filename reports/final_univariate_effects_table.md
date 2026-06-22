# Univariate Feature Effects

Univariate logistic regression (LR) fit on the train split, evaluated on the test split.  
**LR coef** is standardised for numeric features (1 unit = 1 σ) and natural 0→1 for booleans.  
**Dir** ▲ = higher feature value → higher predicted probability of label = 1.  
**Spearman ρ** computed on all filtered rows (train + val + test).

## GG2+ / csPCa (`binary_label_int`)

| Endpoint | Feature | Boolean? | Dir | LR coef (std) | ROC-AUC | PR-AUC | PR-lift | Spearman ρ | p-value |
|---|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | trajectory_intersects_target | True | ▲ | 1.4461 | 0.670 | 0.173 | 1.62 | 0.244 | < 0.001 |
| GG2+ / csPCa | distance_midpoint_to_target_surface_mm | False | ▼ | -1.1315 | 0.712 | 0.192 | 1.79 | -0.244 | < 0.001 |
| GG2+ / csPCa | distance_midpoint_to_target_centroid_mm | False | ▼ | -0.9196 | 0.703 | 0.180 | 1.68 | -0.222 | < 0.001 |
| GG2+ / csPCa | psa_density | False | ▲ | 0.8453 | 0.658 | 0.213 | 1.99 | 0.215 | < 0.001 |
| GG2+ / csPCa | log_psa_ng_ml | False | ▲ | 0.6697 | 0.622 | 0.236 | 2.20 | 0.171 | < 0.001 |
| GG2+ / csPCa | approximate_fraction_of_centerline_inside_target | False | ▲ | 0.6585 | 0.680 | 0.196 | 1.82 | 0.266 | < 0.001 |

## GG3+ / high-grade (`binary_label_gg3plus_int`)

| Endpoint | Feature | Boolean? | Dir | LR coef (std) | ROC-AUC | PR-AUC | PR-lift | Spearman ρ | p-value |
|---|---|---|---|---|---|---|---|---|---|
| GG3+ / high-grade | trajectory_intersects_target | True | ▲ | 1.3162 | 0.725 | 0.079 | 2.07 | 0.158 | < 0.001 |
| GG3+ / high-grade | distance_midpoint_to_target_surface_mm | False | ▼ | -1.1782 | 0.776 | 0.099 | 2.60 | -0.150 | < 0.001 |
| GG3+ / high-grade | psa_density | False | ▲ | 1.1282 | 0.736 | 0.144 | 3.79 | 0.185 | < 0.001 |
| GG3+ / high-grade | log_psa_ng_ml | False | ▲ | 1.0717 | 0.701 | 0.245 | 6.44 | 0.171 | < 0.001 |
| GG3+ / high-grade | distance_midpoint_to_target_centroid_mm | False | ▼ | -0.7914 | 0.741 | 0.077 | 2.01 | -0.120 | < 0.001 |
| GG3+ / high-grade | approximate_fraction_of_centerline_inside_target | False | ▲ | 0.5848 | 0.750 | 0.109 | 2.87 | 0.175 | < 0.001 |

