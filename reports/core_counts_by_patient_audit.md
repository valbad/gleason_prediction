# Core Counts by Patient Audit

Checks whether the number of biopsy cores per patient is balanced across  
splits and label groups, and whether it may confound patient-level aggregation.

# Endpoint: `binary_label_int` — GG2+ / csPCa (Gleason ≥ 3+4=7)

## Core-count distribution — `binary_label_int` (GG2+ / csPCa (Gleason ≥ 3+4=7))

One row per split. Statistics are over per-patient core counts.

| Split | Patients | Mean | Std | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|---|
| train | 557 | 21.23 | 12.12 | 17.00 | 7.00 | 3 | 83 |
| val | 119 | 23.09 | 13.99 | 17.00 | 14.50 | 4 | 78 |
| test | 120 | 20.14 | 9.55 | 17.00 | 6.00 | 6 | 57 |
| all | 796 | 21.35 | 12.09 | 17.00 | 8.00 | 3 | 83 |

## Core counts: positive vs negative patients

| Split | Group | N patients | Mean n_cores | Median n_cores |
|---|---|---|---|---|
| train | positive | 324 | 22.59 | 17.00 |
| train | negative | 233 | 19.34 | 17.00 |
| val | positive | 69 | 25.49 | 18.00 |
| val | negative | 50 | 19.78 | 16.00 |
| test | positive | 70 | 20.36 | 17.00 |
| test | negative | 50 | 19.84 | 17.00 |
| all | positive | 463 | 22.69 | 17.00 |
| all | negative | 333 | 19.48 | 17.00 |

## Spearman correlations with n_cores

Computed over all patients (all splits combined).

| Target | Spearman ρ | p-value | Interpretation |
|---|---|---|---|
| patient label (0/1) | 0.101 | 0.004 | weak positive |
| number of positive cores | 0.097 | 0.006 | negligible positive |
| fraction of positive cores | -0.087 | 0.014 | negligible negative |

## Cross-split core-count comparison

Kruskal-Wallis test for differences across train / val / test.

Kruskal-Wallis H = 0.519,  p = 0.772

> No significant difference across splits (p ≥ 0.05). Core count distributions appear comparable.

Pairwise Mann-Whitney U (two-sided):

| Pair | U statistic | p-value |
|---|---|---|
| train vs val | 31878.0 | 0.512 |
| train vs test | 33697.5 | 0.886 |
| val vs test | 7497.5 | 0.502 |

## Potential bias note

> **Weak but statistically significant correlation** (ρ = 0.101, p = 0.004) between n_cores and patient label.  
> Positive patients tend to have slightly more cores. Aggregation methods that depend on core count (max-prob, top-3 mean) may be mildly influenced by this imbalance.

# Endpoint: `binary_label_gg3plus_int` — GG3+ / high-grade (Gleason ≥ 4+3=7)

## Core-count distribution — `binary_label_gg3plus_int` (GG3+ / high-grade (Gleason ≥ 4+3=7))

One row per split. Statistics are over per-patient core counts.

| Split | Patients | Mean | Std | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|---|
| train | 557 | 21.23 | 12.12 | 17.00 | 7.00 | 3 | 83 |
| val | 119 | 23.09 | 13.99 | 17.00 | 14.50 | 4 | 78 |
| test | 120 | 20.14 | 9.55 | 17.00 | 6.00 | 6 | 57 |
| all | 796 | 21.35 | 12.09 | 17.00 | 8.00 | 3 | 83 |

## Core counts: positive vs negative patients

| Split | Group | N patients | Mean n_cores | Median n_cores |
|---|---|---|---|---|
| train | positive | 145 | 20.14 | 17.00 |
| train | negative | 412 | 21.62 | 17.00 |
| val | positive | 34 | 22.03 | 17.00 |
| val | negative | 85 | 23.52 | 17.00 |
| test | positive | 31 | 20.45 | 17.00 |
| test | negative | 89 | 20.03 | 17.00 |
| all | positive | 210 | 20.50 | 17.00 |
| all | negative | 586 | 21.65 | 17.00 |

## Spearman correlations with n_cores

Computed over all patients (all splits combined).

| Target | Spearman ρ | p-value | Interpretation |
|---|---|---|---|
| patient label (0/1) | -0.050 | 0.158 | negligible negative |
| number of positive cores | -0.065 | 0.067 | negligible negative |
| fraction of positive cores | -0.100 | 0.005 | negligible negative |

## Cross-split core-count comparison

Kruskal-Wallis test for differences across train / val / test.

Kruskal-Wallis H = 0.519,  p = 0.772

> No significant difference across splits (p ≥ 0.05). Core count distributions appear comparable.

Pairwise Mann-Whitney U (two-sided):

| Pair | U statistic | p-value |
|---|---|---|
| train vs val | 31878.0 | 0.512 |
| train vs test | 33697.5 | 0.886 |
| val vs test | 7497.5 | 0.502 |

## Potential bias note

> Weak or non-significant correlation between n_cores and patient label (ρ = -0.050, p = 0.158).  
> Core-count imbalance between positive and negative patients is unlikely to be a major confound.

