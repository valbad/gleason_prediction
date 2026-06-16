# Univariate Geometry & Clinical Feature Effects

Marginal associations between individual features and each binary label.
**Not** a multivariate model: interactions and confounding are not captured.

**Coefficient** = standardised log-odds per 1 σ (numeric features, z-scored  
before LR) or per 0→1 transition (boolean features, unscaled).  
**ROC-AUC / PR-AUC** evaluated on the test split (threshold-free).  
**Spearman ρ** computed on all filtered rows (train + val + test).

## binary_label_int — GG2+ / csPCa (Gleason ≥ 3+4=7)

Test-set prevalence: **0.107**  
(train n = 11,827  · test n = 2,417)

| Feature | Scaled? | Spearman ρ | Spearman p | Coef (std LR) | Dir | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| `trajectory_intersects_target` | 0/1 flag | 0.244 | 0.000 | 1.446 | ▲ | 0.670 | 0.173 |
| `distance_midpoint_to_target_surface_mm` | std-scaled | -0.244 | 0.000 | -1.132 | ▼ | 0.712 | 0.192 |
| `distance_midpoint_to_target_centroid_mm` | std-scaled | -0.222 | 0.000 | -0.920 | ▼ | 0.703 | 0.180 |
| `psa_density` | std-scaled | 0.215 | 0.000 | 0.845 | ▲ | 0.658 | 0.213 |
| `log_psa_ng_ml` | std-scaled | 0.171 | 0.000 | 0.670 | ▲ | 0.622 | 0.236 |
| `approximate_fraction_of_centerline_inside_target` | std-scaled | 0.266 | 0.000 | 0.659 | ▲ | 0.680 | 0.196 |
| `prostate_volume_cc` | std-scaled | -0.075 | 0.000 | -0.208 | ▼ | 0.558 | 0.122 |
| `midpoint_inside_prostate` | 0/1 flag | -0.000 | 0.996 | 0.059 | ▲ | 0.498 | 0.107 |
| `core_length_mm` | std-scaled | -0.004 | 0.587 | -0.012 | ▼ | 0.500 | 0.112 |

**Strongest positive associations** (▲ → higher probability of label = 1):

- `trajectory_intersects_target`: coef = 1.446, ROC-AUC = 0.670, Spearman ρ = 0.244
- `psa_density`: coef = 0.845, ROC-AUC = 0.658, Spearman ρ = 0.215
- `log_psa_ng_ml`: coef = 0.670, ROC-AUC = 0.622, Spearman ρ = 0.171

**Strongest negative associations** (▼ → lower probability of label = 1):

- `distance_midpoint_to_target_surface_mm`: coef = -1.132, ROC-AUC = 0.712, Spearman ρ = -0.244
- `distance_midpoint_to_target_centroid_mm`: coef = -0.920, ROC-AUC = 0.703, Spearman ρ = -0.222
- `prostate_volume_cc`: coef = -0.208, ROC-AUC = 0.558, Spearman ρ = -0.075

## binary_label_gg3plus_int — GG3+ / high-grade (Gleason ≥ 4+3=7)

Test-set prevalence: **0.038**  
(train n = 11,827  · test n = 2,417)

| Feature | Scaled? | Spearman ρ | Spearman p | Coef (std LR) | Dir | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| `trajectory_intersects_target` | 0/1 flag | 0.158 | 0.000 | 1.316 | ▲ | 0.725 | 0.079 |
| `distance_midpoint_to_target_surface_mm` | std-scaled | -0.150 | 0.000 | -1.178 | ▼ | 0.776 | 0.099 |
| `psa_density` | std-scaled | 0.185 | 0.000 | 1.128 | ▲ | 0.736 | 0.144 |
| `log_psa_ng_ml` | std-scaled | 0.171 | 0.000 | 1.072 | ▲ | 0.701 | 0.245 |
| `distance_midpoint_to_target_centroid_mm` | std-scaled | -0.120 | 0.000 | -0.791 | ▼ | 0.741 | 0.077 |
| `approximate_fraction_of_centerline_inside_target` | std-scaled | 0.175 | 0.000 | 0.585 | ▲ | 0.750 | 0.109 |
| `midpoint_inside_prostate` | 0/1 flag | 0.009 | 0.219 | 0.216 | ▲ | 0.509 | 0.039 |
| `prostate_volume_cc` | std-scaled | -0.024 | 0.001 | -0.118 | ▼ | 0.543 | 0.042 |
| `core_length_mm` | std-scaled | 0.016 | 0.042 | 0.092 | ▲ | 0.505 | 0.038 |

**Strongest positive associations** (▲ → higher probability of label = 1):

- `trajectory_intersects_target`: coef = 1.316, ROC-AUC = 0.725, Spearman ρ = 0.158
- `psa_density`: coef = 1.128, ROC-AUC = 0.736, Spearman ρ = 0.185
- `log_psa_ng_ml`: coef = 1.072, ROC-AUC = 0.701, Spearman ρ = 0.171

**Strongest negative associations** (▼ → lower probability of label = 1):

- `distance_midpoint_to_target_surface_mm`: coef = -1.178, ROC-AUC = 0.776, Spearman ρ = -0.150
- `distance_midpoint_to_target_centroid_mm`: coef = -0.791, ROC-AUC = 0.741, Spearman ρ = -0.120
- `prostate_volume_cc`: coef = -0.118, ROC-AUC = 0.543, Spearman ρ = -0.024

