# Grouped Core-Level Inference Analysis

## 1. Methodology

**Goal:** Assess whether the compact `target_geometry_plus_clinical` associations remain present after accounting for intra-patient correlation. This is an inference / association analysis — not a predictive-performance experiment.

**Data:** `data/share/needle_features_v1.csv`, filtered to `label_join_status == "coord_match"` and `split ∈ {train, val, test}`. All valid splits are pooled to maximise statistical power.

**Feature set:** `target_geometry_plus_clinical` (8 features). Clinical variables (`psa_ng_ml`, `log_psa_ng_ml`, `prostate_volume_cc`, `psa_density`) are patient-level measurements replicated to each core — they are not independent core-level observations.

**Preprocessing:** Missing values are median-imputed over all pooled rows. Continuous predictors are z-score standardised (one unit = one standard deviation) so that coefficients are on a comparable scale. `trajectory_intersects_target` is a binary (0/1) indicator and is kept un-standardised; its coefficient represents the log-odds difference between cores that do and do not intersect the target.

**Model 1 — Cluster-robust logistic regression:** `statsmodels.GLM` (Binomial, logit link) with `cov_type='cluster'` clustered by `patient_number`. This yields asymptotically valid standard errors and p-values that account for intra-patient core correlation without imposing a parametric correlation structure.

**Model 2 — GEE:** Binomial GEE with exchangeable working correlation (`statsmodels.genmod.GEE`), grouped by `patient_number`. GEE makes weaker parametric assumptions than mixed models; the exchangeable structure assumes a common intra-patient correlation across core pairs.

**Significance threshold:** α = 0.05 (two-tailed). Stars: \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001.

## 2. Dataset Summary

### GG2+ / csPCa

- Total rows (all splits): **16,992**
- Unique patients: **796**
- Positive cores: **1,887** (11.1%)

| Split | Rows | Patients | Positives | Prevalence |
|---|---|---|---|---|
| train | 11,827 | 557 | 1,356 | 11.5% |
| val | 2,748 | 119 | 272 | 9.9% |
| test | 2,417 | 120 | 259 | 10.7% |

### GG3+ / high-grade

- Total rows (all splits): **16,992**
- Unique patients: **796**
- Positive cores: **647** (3.8%)

| Split | Rows | Patients | Positives | Prevalence |
|---|---|---|---|---|
| train | 11,827 | 557 | 475 | 4.0% |
| val | 2,748 | 119 | 80 | 2.9% |
| test | 2,417 | 120 | 92 | 3.8% |

## 3. Cluster-Robust Logistic Regression

> **Standardisation note:** coefficients for continuous features are in units of one standard deviation. `trajectory_intersects_target` is binary (0/1); its coefficient is the log-odds change from non-intersecting to intersecting cores.

### GG2+ / csPCa

| Feature | Coef | SE | z | p | OR | 95% CI coef | 95% CI OR | Std? |
|---|---|---|---|---|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | -0.128 | 0.153 | -0.833 | 0.4047 | 0.880 | [-0.428, 0.173] | [0.652,  1.189] | yes |
| `distance_midpoint_to_target_surface_mm` | -0.758 | 0.162 | -4.677 | 0.0000*** | 0.468 | [-1.076, -0.441] | [0.341,  0.644] | yes |
| `trajectory_intersects_target` | -0.065 | 0.118 | -0.554 | 0.5799 | 0.937 | [-0.296, 0.166] | [0.744,  1.180] | no (0/1) |
| `approximate_fraction_of_centerline_inside_target` | 0.286 | 0.049 | 5.840 | 0.0000*** | 1.331 | [0.190, 0.381] | [1.209,  1.464] | yes |
| `psa_ng_ml` | 0.113 | 0.074 | 1.521 | 0.1281 | 1.120 | [-0.033, 0.259] | [0.968,  1.296] | yes |
| `log_psa_ng_ml` | 0.598 | 0.095 | 6.284 | 0.0000*** | 1.818 | [0.411, 0.784] | [1.509,  2.191] | yes |
| `prostate_volume_cc` | -0.383 | 0.089 | -4.318 | 0.0000*** | 0.681 | [-0.558, -0.209] | [0.573,  0.811] | yes |
| `psa_density` | -0.051 | 0.094 | -0.535 | 0.5927 | 0.951 | [-0.236, 0.135] | [0.790,  1.144] | yes |

### GG3+ / high-grade

| Feature | Coef | SE | z | p | OR | 95% CI coef | 95% CI OR | Std? |
|---|---|---|---|---|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | 0.055 | 0.204 | 0.269 | 0.7877 | 1.056 | [-0.344, 0.454] | [0.709,  1.575] | yes |
| `distance_midpoint_to_target_surface_mm` | -1.129 | 0.220 | -5.131 | 0.0000*** | 0.323 | [-1.560, -0.698] | [0.210,  0.498] | yes |
| `trajectory_intersects_target` | -0.095 | 0.186 | -0.511 | 0.6097 | 0.910 | [-0.459, 0.269] | [0.632,  1.309] | no (0/1) |
| `approximate_fraction_of_centerline_inside_target` | 0.204 | 0.074 | 2.761 | 0.0058** | 1.227 | [0.059, 0.350] | [1.061,  1.418] | yes |
| `psa_ng_ml` | 0.055 | 0.090 | 0.610 | 0.5420 | 1.056 | [-0.122, 0.231] | [0.886,  1.260] | yes |
| `log_psa_ng_ml` | 0.997 | 0.167 | 5.972 | 0.0000*** | 2.710 | [0.670, 1.324] | [1.954,  3.758] | yes |
| `prostate_volume_cc` | -0.360 | 0.126 | -2.848 | 0.0044** | 0.698 | [-0.607, -0.112] | [0.545,  0.894] | yes |
| `psa_density` | -0.063 | 0.113 | -0.561 | 0.5750 | 0.939 | [-0.284, 0.158] | [0.753,  1.171] | yes |

## 4. GEE Results (Optional)

> **Standardisation note:** same as Section 3.

### GG2+ / csPCa

| Feature | Coef | SE | z | p | OR | 95% CI coef | 95% CI OR | Std? |
|---|---|---|---|---|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | -0.355 | 0.143 | -2.490 | 0.0128* | 0.701 | [-0.634, -0.076] | [0.530,  0.927] | yes |
| `distance_midpoint_to_target_surface_mm` | -0.542 | 0.141 | -3.854 | 0.0001*** | 0.581 | [-0.818, -0.267] | [0.441,  0.766] | yes |
| `trajectory_intersects_target` | -0.025 | 0.106 | -0.238 | 0.8123 | 0.975 | [-0.233, 0.182] | [0.792,  1.200] | no (0/1) |
| `approximate_fraction_of_centerline_inside_target` | 0.273 | 0.045 | 6.016 | 0.0000*** | 1.314 | [0.184, 0.362] | [1.202,  1.436] | yes |
| `psa_ng_ml` | 0.146 | 0.081 | 1.802 | 0.0716 | 1.157 | [-0.013, 0.304] | [0.987,  1.356] | yes |
| `log_psa_ng_ml` | 0.708 | 0.098 | 7.260 | 0.0000*** | 2.031 | [0.517, 0.900] | [1.677,  2.459] | yes |
| `prostate_volume_cc` | -0.528 | 0.111 | -4.739 | 0.0000*** | 0.590 | [-0.746, -0.310] | [0.474,  0.734] | yes |
| `psa_density` | -0.098 | 0.094 | -1.038 | 0.2992 | 0.907 | [-0.283, 0.087] | [0.753,  1.091] | yes |

### GG3+ / high-grade

| Feature | Coef | SE | z | p | OR | 95% CI coef | 95% CI OR | Std? |
|---|---|---|---|---|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | -0.194 | 0.169 | -1.150 | 0.2500 | 0.823 | [-0.526, 0.137] | [0.591,  1.147] | yes |
| `distance_midpoint_to_target_surface_mm` | -0.818 | 0.182 | -4.491 | 0.0000*** | 0.441 | [-1.175, -0.461] | [0.309,  0.631] | yes |
| `trajectory_intersects_target` | -0.034 | 0.148 | -0.227 | 0.8201 | 0.967 | [-0.324, 0.256] | [0.723,  1.292] | no (0/1) |
| `approximate_fraction_of_centerline_inside_target` | 0.177 | 0.063 | 2.821 | 0.0048** | 1.193 | [0.054, 0.299] | [1.055,  1.349] | yes |
| `psa_ng_ml` | 0.084 | 0.093 | 0.905 | 0.3657 | 1.087 | [-0.098, 0.265] | [0.907,  1.304] | yes |
| `log_psa_ng_ml` | 0.961 | 0.159 | 6.052 | 0.0000*** | 2.615 | [0.650, 1.273] | [1.916,  3.570] | yes |
| `prostate_volume_cc` | -0.418 | 0.137 | -3.058 | 0.0022** | 0.658 | [-0.686, -0.150] | [0.503,  0.861] | yes |
| `psa_density` | -0.063 | 0.108 | -0.587 | 0.5573 | 0.939 | [-0.274, 0.148] | [0.760,  1.159] | yes |

## 5. Predictor Correlation and Collinearity

> **Warning:** `psa_ng_ml`, `log_psa_ng_ml`, and `psa_density` are strongly correlated by construction. Joint coefficient estimates for these features may be numerically unstable; individual coefficient magnitudes should not be used to rank PSA-related clinical importance.

### GG2+ / csPCa

#### Pairwise Spearman correlations

| Feature 1 | Feature 2 | Spearman ρ | p-value |
|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | `distance_midpoint_to_target_surface_mm` | 0.942 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `trajectory_intersects_target` | -0.697 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `approximate_fraction_of_centerline_inside_target` | -0.689 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `psa_ng_ml` | 0.057 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `log_psa_ng_ml` | 0.057 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `prostate_volume_cc` | 0.154 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `psa_density` | -0.035 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `trajectory_intersects_target` | -0.719 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `approximate_fraction_of_centerline_inside_target` | -0.708 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `psa_ng_ml` | -0.003 | 0.7050 |
| `distance_midpoint_to_target_surface_mm` | `log_psa_ng_ml` | -0.003 | 0.7050 |
| `distance_midpoint_to_target_surface_mm` | `prostate_volume_cc` | 0.140 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `psa_density` | -0.083 | 0.0000*** |
| `trajectory_intersects_target` | `approximate_fraction_of_centerline_inside_target` | 0.979 | 0.0000*** |
| `trajectory_intersects_target` | `psa_ng_ml` | 0.050 | 0.0000*** |
| `trajectory_intersects_target` | `log_psa_ng_ml` | 0.050 | 0.0000*** |
| `trajectory_intersects_target` | `prostate_volume_cc` | -0.033 | 0.0000*** |
| `trajectory_intersects_target` | `psa_density` | 0.062 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `psa_ng_ml` | 0.072 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `log_psa_ng_ml` | 0.072 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `prostate_volume_cc` | -0.032 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `psa_density` | 0.082 | 0.0000*** |
| `psa_ng_ml` | `log_psa_ng_ml` | 1.000 | 0.0000*** |
| `psa_ng_ml` | `prostate_volume_cc` | 0.247 | 0.0000*** |
| `psa_ng_ml` | `psa_density` | 0.777 | 0.0000*** |
| `log_psa_ng_ml` | `prostate_volume_cc` | 0.247 | 0.0000*** |
| `log_psa_ng_ml` | `psa_density` | 0.777 | 0.0000*** |
| `prostate_volume_cc` | `psa_density` | -0.336 | 0.0000*** |

#### Variance Inflation Factors (VIF)

| Feature | VIF |
|---|---|
| `distance_midpoint_to_target_centroid_mm` | 15.160 ← high |
| `distance_midpoint_to_target_surface_mm` | 14.600 ← high |
| `trajectory_intersects_target` | 3.320 |
| `approximate_fraction_of_centerline_inside_target` | 2.720 |
| `psa_ng_ml` | 4.930 |
| `log_psa_ng_ml` | 3.120 |
| `prostate_volume_cc` | 1.950 |
| `psa_density` | 5.120 ← moderate |

### GG3+ / high-grade

#### Pairwise Spearman correlations

| Feature 1 | Feature 2 | Spearman ρ | p-value |
|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | `distance_midpoint_to_target_surface_mm` | 0.942 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `trajectory_intersects_target` | -0.697 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `approximate_fraction_of_centerline_inside_target` | -0.689 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `psa_ng_ml` | 0.057 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `log_psa_ng_ml` | 0.057 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `prostate_volume_cc` | 0.154 | 0.0000*** |
| `distance_midpoint_to_target_centroid_mm` | `psa_density` | -0.035 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `trajectory_intersects_target` | -0.719 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `approximate_fraction_of_centerline_inside_target` | -0.708 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `psa_ng_ml` | -0.003 | 0.7050 |
| `distance_midpoint_to_target_surface_mm` | `log_psa_ng_ml` | -0.003 | 0.7050 |
| `distance_midpoint_to_target_surface_mm` | `prostate_volume_cc` | 0.140 | 0.0000*** |
| `distance_midpoint_to_target_surface_mm` | `psa_density` | -0.083 | 0.0000*** |
| `trajectory_intersects_target` | `approximate_fraction_of_centerline_inside_target` | 0.979 | 0.0000*** |
| `trajectory_intersects_target` | `psa_ng_ml` | 0.050 | 0.0000*** |
| `trajectory_intersects_target` | `log_psa_ng_ml` | 0.050 | 0.0000*** |
| `trajectory_intersects_target` | `prostate_volume_cc` | -0.033 | 0.0000*** |
| `trajectory_intersects_target` | `psa_density` | 0.062 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `psa_ng_ml` | 0.072 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `log_psa_ng_ml` | 0.072 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `prostate_volume_cc` | -0.032 | 0.0000*** |
| `approximate_fraction_of_centerline_inside_target` | `psa_density` | 0.082 | 0.0000*** |
| `psa_ng_ml` | `log_psa_ng_ml` | 1.000 | 0.0000*** |
| `psa_ng_ml` | `prostate_volume_cc` | 0.247 | 0.0000*** |
| `psa_ng_ml` | `psa_density` | 0.777 | 0.0000*** |
| `log_psa_ng_ml` | `prostate_volume_cc` | 0.247 | 0.0000*** |
| `log_psa_ng_ml` | `psa_density` | 0.777 | 0.0000*** |
| `prostate_volume_cc` | `psa_density` | -0.336 | 0.0000*** |

#### Variance Inflation Factors (VIF)

| Feature | VIF |
|---|---|
| `distance_midpoint_to_target_centroid_mm` | 15.160 ← high |
| `distance_midpoint_to_target_surface_mm` | 14.600 ← high |
| `trajectory_intersects_target` | 3.320 |
| `approximate_fraction_of_centerline_inside_target` | 2.720 |
| `psa_ng_ml` | 4.930 |
| `log_psa_ng_ml` | 3.120 |
| `prostate_volume_cc` | 1.950 |
| `psa_density` | 5.120 ← moderate |

## 6. Interpretation

### GG2+ / csPCa

**Distance to target surface:** `distance_midpoint_to_target_surface_mm` is negatively associated with the endpoint and remains statistically significant after accounting for intra-patient clustering (coef = -0.758, OR = 0.468, p = 0.0000***, cluster-robust SE).

**Target-relative geometry:** 2 of 4 target-geometry features are individually significant (p < 0.05) after cluster-robust adjustment, supporting the interpretability of `target_geometry_plus_clinical` after accounting for patient-level correlation.

**Note on `trajectory_intersects_target`:** `trajectory_intersects_target` is not individually significant once continuous distance and fraction-inside features are included, likely because it is highly redundant with `approximate_fraction_of_centerline_inside_target`.

**PSA-related variables:** `psa_ng_ml`, `log_psa_ng_ml`, and `psa_density` are correlated clinical measurements (see Section 5). Individual coefficient estimates for these features can be numerically unstable and should not be used to rank clinical importance. The group of PSA-related variables is informative, but no single coefficient alone is interpretable in isolation.

**GEE concordance:** The GEE estimate for `distance_midpoint_to_target_surface_mm` (coef = -0.542, p = 0.0001***) is concordant with the cluster-robust estimate.

### GG3+ / high-grade

**Distance to target surface:** `distance_midpoint_to_target_surface_mm` is negatively associated with the endpoint and remains statistically significant after accounting for intra-patient clustering (coef = -1.129, OR = 0.323, p = 0.0000***, cluster-robust SE).

**Target-relative geometry:** 2 of 4 target-geometry features are individually significant (p < 0.05) after cluster-robust adjustment, supporting the interpretability of `target_geometry_plus_clinical` after accounting for patient-level correlation.

**Note on `trajectory_intersects_target`:** `trajectory_intersects_target` is not individually significant once continuous distance and fraction-inside features are included, likely because it is highly redundant with `approximate_fraction_of_centerline_inside_target`.

**PSA-related variables:** `psa_ng_ml`, `log_psa_ng_ml`, and `psa_density` are correlated clinical measurements (see Section 5). Individual coefficient estimates for these features can be numerically unstable and should not be used to rank clinical importance. The group of PSA-related variables is informative, but no single coefficient alone is interpretable in isolation.

**GEE concordance:** The GEE estimate for `distance_midpoint_to_target_surface_mm` (coef = -0.818, p = 0.0000***) is concordant with the cluster-robust estimate.

### Support for the compact `target_geometry_plus_clinical` model

The cluster-robust analysis tests whether target-relative geometry associations survive intra-patient correlation adjustment. Statistically significant geometry associations after this adjustment support the use of the compact feature set as the central interpretable association model, while predictive claims should remain based on the held-out performance analyses. Clinical (PSA-related) variables contribute population-level discriminative value but their individual coefficients should be reported with appropriate collinearity caveats.

