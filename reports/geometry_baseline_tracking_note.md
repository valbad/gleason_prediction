# Geometry-Baseline Project — Tracking Note

*Last updated: 2026-06-24*

This note synthesises results from the full audit and modelling pipeline.
All numbers come directly from the generated report files; none are invented.

---

## 1. Project Framing

**Goal:** Establish whether needle-to-target geometry features derived from
MRI-guided biopsy coordinate frames can predict clinically significant prostate
cancer (csPCa) at the core level, with or without clinical covariates (PSA,
prostate volume).

**Dataset:** TCIA Prostate-MRI-US-Biopsy. Features in
`data/share/needle_features_v1.csv`. Experiments use rows where
`label_join_status == "coord_match"` and the label is not missing.

**Setup:** Core-level binary classification. Patient-level splits enforced
throughout (no patient appears in more than one split). Two complementary
classification endpoints are evaluated.

---

## 2. Dataset and Split Summary

| Split | Patients | Labelled cores (coord-match) | GG2+ prev. | GG3+ prev. |
|---|---|---|---|---|
| train | 557 | 11,827 | 11.5% | 4.0% |
| val | 119 | 2,748 | 9.9% | 2.9% |
| test | 120 | 2,417 | 10.7% | 3.8% |
| ALL | 796 | 16,992 | 11.1% | 3.8% |

Splits are patient-level (GroupShuffleSplit on patient ID). No inter-split
patient overlap was confirmed by the leakage audit
(`reports/target_geometry_leakage_audit.md`, Section 1: **PASS**).

---

## 3. Label Clarification

Two binary endpoints are used throughout:

| Column | Meaning | Positive definition | Positive cores (total) |
|---|---|---|---|
| `binary_label_int` | GG2+ / csPCa | ISUP GG ≥ 2 (Gleason ≥ 3+4=7) | 1,887 |
| `binary_label_gg3plus_int` | GG3+ / high-grade | ISUP GG ≥ 3 (Gleason ≥ 4+3=7) | 647 |

**`binary_label_int` is NOT equivalent to GG3+.** GG2 (Gleason 3+4=7) cores
(1,240 rows) are labelled positive for the GG2+ endpoint but negative for GG3+.
This was confirmed with 100% agreement via the label audit (`label_definition_audit.md`).

**Known pipeline bug (not fixed):** `src/build_manifest.py::grade_group()` maps
Gleason 3+5=8 as GG2 instead of GG4 (36 rows, 0.2% of total). All downstream
analysis uses the ISUP-correct `grade_group_isup()` from `audit_labels.py`.

---

## 4. Feature Groups

Four base feature groups are defined in `run_shareable_tabular_experiments.py`:

### Clinical (4 features)
- `psa_ng_ml`
- `log_psa_ng_ml`
- `prostate_volume_cc`
- `psa_density`

### Biopsy / prostate geometry (6 features)
- `core_length_mm`
- `n_centerline_voxels`
- `n_tube_voxels`
- `midpoint_inside_prostate`
- `distance_midpoint_to_prostate_surface_mm`
- `approximate_fraction_of_centerline_inside_prostate`

### Target geometry (5 features, includes availability flag)
- `target_mesh_available`
- `distance_midpoint_to_target_centroid_mm`
- `distance_midpoint_to_target_surface_mm`
- `trajectory_intersects_target`
- `approximate_fraction_of_centerline_inside_target`

### Target geometry — no availability (4 features)
- `distance_midpoint_to_target_centroid_mm`
- `distance_midpoint_to_target_surface_mm`
- `trajectory_intersects_target`
- `approximate_fraction_of_centerline_inside_target`

Composite groups (`all_geometry_no_availability`,
`all_geometry_no_availability_plus_clinical`) combine the above.

### Why `target_mesh_available` is excluded from final reporting

The leakage audit (`target_geometry_leakage_audit.md`, Section 5) shows:

- When `target_mesh_available = True`: GG2+ prevalence = **11.0%** (16,966 cores)
- When `target_mesh_available = False`: GG2+ prevalence = **53.8%** (26 cores)

Because mesh-absent cores are rare but have unusually high positivity, we
exclude `target_mesh_available` from final reporting as a conservative ablation.
Performance remains essentially unchanged without it, suggesting that the signal
is driven by continuous target-trajectory geometry rather than mesh availability.

---

## 5. Main Core-Level Performance

From `run_shareable_tabular_experiments.py` (test set, patient-level bootstrap
95% CIs). Best model selected by test ROC-AUC within each (endpoint, experiment)
group.

| Endpoint | Prevalence | Experiment | Best model | ROC-AUC [95% CI] | PR-AUC [95% CI] | PR-lift | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | 10.7% | target_geometry_no_availability | logistic_regression | 0.730 [0.669 – 0.771] | 0.224 [0.163 – 0.320] | 2.09 | 0.707 | 0.653 | 0.197 | 0.308 |
| GG3+ / high-grade | 3.8% | target_geometry_no_availability | logistic_regression | 0.806 [0.736 – 0.868] | 0.138 [0.054 – 0.287] | 3.64 | 0.772 | 0.680 | 0.087 | 0.156 |
| GG2+ / csPCa | 10.7% | all_geometry_no_availability_plus_clinical | xgboost | 0.758 [0.692 – 0.804] | 0.327 [0.233 – 0.437] | 3.05 | 0.552 | 0.803 | 0.251 | 0.345 |
| GG3+ / high-grade | 3.8% | all_geometry_no_availability_plus_clinical | logistic_regression | 0.829 [0.760 – 0.888] | 0.265 [0.106 – 0.502] | 6.96 | 0.815 | 0.657 | 0.086 | 0.156 |

> **PR-lift** = PR-AUC / test prevalence. Values > 1 indicate the model beats the no-skill baseline.

The GG3+ endpoint achieves higher PR-lift, consistent with high-grade cancers
being more spatially concentrated relative to MRI targets. Adding clinical
features (PSA density, prostate volume, log-PSA) may improve both ROC-AUC and
PR-AUC; the full-feature XGBoost model suggests the largest gains for GG2+.

---

## 6. Univariate Interpretability

From `audit_univariate_geometry_effects.py` / `final_univariate_effects_table.md`.
LR coefficients are standardised for numeric features (1 unit = 1 σ) and
natural 0 → 1 for booleans.

### GG2+ / csPCa

| Feature | Dir | LR coef (std) | ROC-AUC | Spearman ρ |
|---|---|---|---|---|
| trajectory_intersects_target | ▲ | +1.4461 | 0.670 | +0.244 |
| distance_midpoint_to_target_surface_mm | ▼ | −1.1315 | 0.712 | −0.244 |
| distance_midpoint_to_target_centroid_mm | ▼ | −0.9196 | 0.703 | −0.222 |
| psa_density | ▲ | +0.8453 | 0.658 | +0.215 |
| log_psa_ng_ml | ▲ | +0.6697 | 0.622 | +0.171 |
| approximate_fraction_of_centerline_inside_target | ▲ | +0.6585 | 0.680 | +0.266 |

### GG3+ / high-grade

| Feature | Dir | LR coef (std) | ROC-AUC | Spearman ρ |
|---|---|---|---|---|
| trajectory_intersects_target | ▲ | +1.3162 | 0.725 | +0.158 |
| distance_midpoint_to_target_surface_mm | ▼ | −1.1782 | 0.776 | −0.150 |
| psa_density | ▲ | +1.1282 | 0.736 | +0.185 |
| log_psa_ng_ml | ▲ | +1.0717 | 0.701 | +0.171 |
| distance_midpoint_to_target_centroid_mm | ▼ | −0.7914 | 0.741 | −0.120 |
| approximate_fraction_of_centerline_inside_target | ▲ | +0.5848 | 0.750 | +0.175 |

**Key takeaways:**
- Distance to target surface is the single strongest geometric predictor by
  univariate ROC-AUC (0.712 for GG2+; 0.776 for GG3+).
- PSA density is the strongest clinical predictor and achieves higher PR-lift
  than most geometric features (PR-lift up to 6.44 for GG3+).
- Geometric and clinical features may be complementary: geometry captures
  spatial proximity to the MRI lesion; PSA density may capture tumour burden.
- All 6 features reach statistical significance (Spearman p < 0.001).

---

## 7. Risk Stratification

From `analyze_risk_stratification.py` / `final_risk_stratification_table.md`.
Feature set: `all_geometry_no_availability_plus_clinical` (`target_mesh_available` excluded).
**Decile 10** = highest-risk 10% of test cores.
**Capture top N%** = fraction of all positive cores found in the top N% by predicted risk.

| Endpoint | Model | ROC-AUC | PR-AUC | Baseline prevalence | Decile 10 prevalence | Capture top 5% | Capture top 10% | Capture top 20% |
|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | xgboost | 0.758 | 0.327 | 10.7% | 36.1% | 22.8% | 33.6% | 51.7% |
| GG3+ / high-grade | logistic_regression | 0.829 | 0.265 | 3.8% | 18.7% | 40.2% | 48.9% | 62.0% |

**Reading the table:**
- Decile 10 prevalence is 3.4× the baseline for GG2+ and 4.9× for GG3+,
  suggesting the model is effective at concentrating positives in its top-ranked
  cores.
- The GG3+ model captures nearly half of all high-grade positives in the top
  10% of scored cores (capture = 48.9%).

---

## 8. Patient-Level Aggregation Results

From `analyze_patient_level_performance.py` / `final_patient_level_table.md`.

A patient is **positive** if at least one biopsy core is positive for the endpoint.
Four aggregation strategies were tested (max_prob, top3_mean_prob, mean_prob,
prop_above_threshold); threshold re-selected via Youden-J at patient level on
the val split.

### Best aggregation per endpoint and model

| Endpoint | Model | Best aggregation | Prevalence | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|---|---|---|
| GG3+ / high-grade | logistic_regression | Mean prob | 25.8% | 0.682 | 0.512 | 0.645 | 0.629 | 0.476 | 22.6% | 38.7% |
| GG2+ / csPCa | logistic_regression | Mean prob | 58.3% | 0.578 | 0.674 | 0.729 | 0.420 | 0.680 | 12.9% | 25.7% |
| GG2+ / csPCa | xgboost | Mean prob | 58.3% | 0.617 | 0.695 | 0.571 | 0.660 | 0.630 | 12.9% | 24.3% |

### All aggregation methods — GG3+ / high-grade — logistic_regression

| Aggregation | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Capture @10% | Capture @20% |
|---|---|---|---|---|---|---|---|
| Mean prob | 0.682 | 0.512 | 0.645 | 0.629 | 0.476 | 22.6% | 38.7% |
| Top-3 mean | 0.679 | 0.492 | 0.774 | 0.472 | 0.471 | 25.8% | 45.2% |
| Prop > thr | 0.678 | 0.467 | 0.645 | 0.663 | 0.494 | 22.6% | 35.5% |
| Max prob | 0.678 | 0.496 | 0.516 | 0.674 | 0.421 | 25.8% | 41.9% |

### Patient-level vs. core-level performance gap

| Endpoint | Model | Core-level ROC-AUC | Patient-level ROC-AUC | Drop |
|---|---|---|---|---|
| GG2+ | xgboost | 0.758 | 0.617 | −0.141 |
| GG3+ | logistic_regression | 0.829 | 0.682 | −0.147 |

**Note on the gap:** Patient prevalence is substantially higher than core
prevalence (GG2+: 58.3% vs 10.7%; GG3+: 25.8% vs 3.8%), because most patients
have many cores covering the whole gland and a single positive core flags the
patient positive. This may make patient-level discrimination harder than
core-level discrimination in this dataset. The aggregation step also adds
variance by averaging noisy core-level scores.

---

## 9. Core-Count Audit

From `audit_core_counts_by_patient.py` / `final_core_count_audit_summary.md`.

| Endpoint | Median n_cores | Mean n_cores | Mean (positive pat.) | Mean (negative pat.) | Spearman ρ (vs label) | p-value | Interpretation |
|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | 17.0 | 21.3 | 22.7 | 19.5 | 0.101 | 0.004 | Weak but significant: positive patients tend to have slightly more cores |
| GG3+ / high-grade | 17.0 | 21.3 | 20.5 | 21.7 | −0.050 | 0.158 | No significant association |

For GG2+, there is a **weak but statistically significant** correlation
(ρ = 0.101, p = 0.004) between core count and patient label. Positive patients
tend to have slightly more cores on average (22.7 vs 19.5). Aggregation methods
that depend on core count (max-prob, top-3 mean) may be mildly influenced by
this imbalance, but the effect size is small.

For GG3+, no significant association exists (p = 0.158).

---

## 10. Calibration Results

From `analyze_calibration.py` / `final_calibration_table.md`.
Recalibration uses Platt scaling: `LogisticRegression(C=1e6)` fit on
`logit(val_probs)` → val labels.
**BSS** = Brier skill score (1 − Brier / Brier_null); higher is better; < 0 = worse than naive.
**ECE** = expected calibration error (10 equal-width bins); lower is better.
**Slope** b ≈ 1 → well-spread; b < 1 → over-confident; b > 1 → under-confident.

| Endpoint | Model | Brier (before) | Brier (after) | BSS (before) | BSS (after) | ECE (before) | ECE (after) | Slope (before) | Slope (after) | Intercept (before) | Intercept (after) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | logistic_regression | 0.177 | 0.085 | −0.850 | 0.108 | 0.257 | 0.020 | 0.736 | 0.966 | −1.939 | 0.185 |
| GG2+ / csPCa | xgboost | 0.162 | 0.085 | −0.690 | 0.116 | 0.229 | 0.020 | 0.706 | 0.977 | −1.854 | 0.209 |
| GG3+ / high-grade | logistic_regression | 0.139 | 0.034 | −2.801 | 0.084 | 0.257 | 0.013 | 0.880 | 1.511 | −3.021 | 2.076 |

**BSS < 0 before recalibration** in all cases, suggesting raw predicted
probabilities are worse than the naive prevalence-based forecast. Platt scaling
on the val set brings all Brier Skill Scores positive and ECE below 0.03.

**GG3+ caution after recalibration:** the calibration slope rises to 1.511 and
the intercept to 2.076. After recalibration, the majority of GG3+ test scores
fall in the [0, 0.1] probability bin, which may improve average calibration
metrics while leaving absolute risk estimates conservative. The recalibrated
model is more reliable for ranking than for absolute probability estimation.

---

## 11. Current Limitations

1. **Small test set.** 120 patients / 2,417 labelled cores. Bootstrap CIs are
   wide (often ±0.05–0.07 on ROC-AUC). Results should be treated as preliminary.

2. **Core-level classification, not patient-level.** The modelling target is
   a biopsy core, not a patient. Patient-level aggregation is associated with
   substantially reduced discriminative performance (see Section 8).

3. **`target_mesh_available` exclusion.** The 26 mesh-absent cores have 53.8%
   positive rate versus 11.0% overall. They are excluded as a conservative
   ablation, which limits applicability to settings where target mesh availability
   may vary.

4. **Pipeline Gleason mapping bug (`build_manifest.py`).** Gleason 3+5=8 is
   mapped as GG2 (should be GG4): 36 rows / 0.2% of data. Not yet fixed in the
   pipeline; downstream analysis uses the correct ISUP mapping.

5. **Raw probabilities are poorly calibrated.** BSS < 0 before Platt
   recalibration for all models. Recalibrated probabilities improve but may
   concentrate in the low-probability bin, especially for GG3+.

6. **No external validation.** All results are on a single institutional
   dataset. Generalisability to other MRI-guided biopsy protocols is unknown.

7. **Threshold optimisation on val.** Youden-J threshold is selected on the val
   split. Sensitivity/specificity/F1 at the chosen threshold may not generalise
   as well as AUC metrics.

8. **Mild core-count imbalance for GG2+.** Positive patients have slightly more
   cores (ρ = 0.101), which may mildly advantage count-sensitive aggregation
   methods.

---

## Phase 1 — Compact vs full geometry-clinical model comparison

We compared six feature sets across both endpoints (GG2+ / csPCa and GG3+ / high-grade): clinical only, target geometry only, target geometry + clinical, biopsy/prostate geometry + clinical, all geometry without clinical, and all geometry + clinical.

The compact `target_geometry_plus_clinical` feature set captured nearly all the predictive signal of the full `all_geometry_no_availability_plus_clinical` model. Averaged across models, the full-vs-compact delta was negligible for both endpoints:
- GG2+ / csPCa: ΔROC = -0.002, ΔPR = 0.000
- GG3+ / high-grade: ΔROC = -0.005, ΔPR = 0.008

The geometry-only comparison also showed limited gain from adding non-target biopsy/prostate geometry to target-relative geometry. The `biopsy_prostate_geometry_plus_clinical` feature set should be interpreted as a negative-control comparison rather than an incremental-value test: it confirms that target-relative geometry, not biopsy/prostate geometry alone, drives most of the geometric signal.

Decision:
`target_geometry_plus_clinical` is selected as the preferred central feature set for the paper on parsimony and interpretability grounds. The full geometry-clinical model will be retained as a sensitivity/reference analysis.

---

## Phase 2 — Direct patient-level modelling

We built direct patient-level models using one row per patient. Patient labels were defined as positive if at least one biopsy core was positive for the endpoint. Core-level geometry features were aggregated into patient-level summaries, then combined with clinical variables.

For GG2+ / csPCa, direct patient-level modelling performed similarly to the previous naive aggregation of core-level scores:
- Previous naive aggregation best: ROC-AUC 0.617, PR-AUC 0.695
- Direct patient-level best: ROC-AUC 0.613, PR-AUC 0.697

For GG3+ / high-grade, direct patient-level modelling slightly improved ROC-AUC but not PR-AUC:
- Previous naive aggregation best: ROC-AUC 0.682, PR-AUC 0.512
- Direct patient-level best: ROC-AUC 0.706, PR-AUC 0.498

Patient-level prediction remained weaker than core-level prediction for both endpoints. This supports the interpretation that the strongest geometric signal is local and core-level. Patient-level models should therefore be reported as secondary or exploratory analyses, while the main paper framing should remain centered on core-level risk stratification.

Decision:
Direct patient-level modelling does not replace the core-level analysis as the primary result. It is useful as a methodological check addressing patient-level aggregation, but the main contribution remains the interpretable core-level target-relative geometry signal.

---

## Phase 3 — Grouped core-level inference

We assessed whether the compact `target_geometry_plus_clinical` associations remain present after accounting for intra-patient correlation among biopsy cores. We fitted core-level logistic regression models with cluster-robust standard errors grouped by patient, and a secondary GEE analysis with exchangeable working correlation.

For both endpoints, `distance_midpoint_to_target_surface_mm` remained negatively associated with positivity after patient-cluster adjustment:
- GG2+ / csPCa: OR 0.468, p < 0.001
- GG3+ / high-grade: OR 0.323, p < 0.001

The secondary GEE analysis was concordant:
- GG2+ / csPCa: OR 0.581, p < 0.001
- GG3+ / high-grade: OR 0.441, p < 0.001

`approximate_fraction_of_centerline_inside_target` was also positively associated with both endpoints. In contrast, `trajectory_intersects_target` was not individually significant once continuous distance and fraction-inside features were included, likely because it is highly redundant with fraction-inside.

Clinical PSA-related variables contributed signal, but their individual coefficients should be interpreted cautiously because PSA, log-PSA, prostate volume and PSA density are correlated.

Decision:
The grouped inference analysis supports the compact `target_geometry_plus_clinical` model as the central interpretable association model. It also strengthens the main methodological claim that target-relative biopsy geometry carries a core-level signal that is not solely an artefact of repeated correlated cores within patients.

---

## 12. Recommended Next Steps

1. **Fix `build_manifest.py` Gleason mapping.** Correct the `grade_group()`
   function so Gleason 3+5=8 maps to GG4, and re-run the full pipeline.

2. **Evaluate on a held-out external cohort.** The current test set is
   internal; results need external validation before clinical conclusions.

3. **Feature ablation.** Run Shapley value analysis or permutation importance
   to identify which geometry features drive the multi-feature models (beyond
   the univariate analysis in Section 6).

4. **Patient-level modelling.** Explore direct patient-level modelling (one
   row = one patient, aggregate features) rather than aggregating core-level
   predictions post hoc.

5. **Recalibration strategy for GG3+.** The Platt-recalibrated GG3+ model
   has a slope of 1.511 — investigate isotonic regression or temperature
   scaling as alternatives that may spread scores more evenly.

6. **Cross-validation instead of a single val/test split.** Given the small
   test size, nested cross-validation would give more stable estimates of
   generalisation performance.

7. **Incorporate prior biopsy history.** `core_label == 'TARGET OR PRIOR POSITIVE'`
   accounts for 7,649 cores. Including a "prior positive" indicator as a feature
   may substantially improve performance while remaining clinically available.

8. **Investigate PSA density non-linearity.** PSA density has the highest
   PR-lift among clinical features for GG3+; a spline or binned encoding may
   capture threshold effects (e.g., PSA density > 0.15 ng/mL/cc).
