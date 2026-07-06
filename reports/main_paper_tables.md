# Main Paper Tables

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-07-03  
**Status:** Draft manuscript tables. Values are copied from existing reports; no new model fitting or recomputation was performed.

**Sources used:**
- `reports/grouped_core_level_inference.md` (Tables 1, 5)
- `reports/final_core_count_audit_summary.md` (Table 1 note)
- `reports/minimal_vs_full_model_comparison.md` (Tables 2, 3)
- `reports/minimal_vs_full_delta_table.md` (Table 3)
- `reports/minimal_vs_full_risk_stratification.md` (Table 4)
- `reports/grouped_core_level_inference_cluster_robust.csv` (Table 5)
- `reports/patient_level_direct_model_comparison.md` (Table 6)

---

## Table 1 — Dataset overview

| Split | Patients | Cores | GG2+ positive | GG2+ prevalence | GG3+ positive | GG3+ prevalence |
|-------|----------|-------|--------------|-----------------|--------------|-----------------|
| Train | 557 | 11,827 | 1,356 | 11.5% | 475 | 4.0% |
| Validation | 119 | 2,748 | 272 | 9.9% | 80 | 2.9% |
| Test | 120 | 2,417 | 259 | 10.7% | 92 | 3.8% |
| **Total** | **796** | **16,992** | **1,887** | **11.1%** | **647** | **3.8%** |

**Definitions.**
- GG2+ / csPCa = clinically significant prostate cancer, ISUP Grade Group ≥ 2 (Gleason ≥ 3+4=7).
- GG3+ / high-grade = ISUP Grade Group ≥ 3 (Gleason ≥ 4+3=7).
- Splits are patient-disjoint (GroupShuffleSplit); all cores from one patient are assigned to the same split.
- Rows are filtered to `label_join_status == "coord_match"`.
- Median cores per patient: 17 (mean 21.3); positive patients have slightly more cores than negative patients for GG2+ (Spearman ρ = 0.10, p = 0.004) but not for GG3+ (ρ = −0.05, p = 0.16); see `reports/final_core_count_audit_summary.md`.

*Source: `reports/grouped_core_level_inference.md` Section 2.*

---

## Table 2 — Compact model held-out predictive performance

Feature set: `target_geometry_plus_clinical` (8 features: 4 target-relative geometry + 4 clinical).  
Model: logistic regression. Evaluation: held-out test split.

| Endpoint | ROC-AUC [95% CI] | PR-AUC [95% CI] | PR-lift | Sensitivity | Specificity | Precision | F1 |
|----------|-----------------|-----------------|---------|-------------|-------------|-----------|-----|
| GG2+ / csPCa | 0.754 [0.699–0.807] | 0.327 [0.230–0.461] | 3.05 | 0.703 | 0.706 | 0.223 | 0.339 |
| GG3+ / high-grade | 0.828 [0.751–0.892] | 0.269 [0.094–0.516] | 7.07 | 0.793 | 0.712 | 0.098 | 0.175 |

**Notes.**
- 95% CIs are from 1,000 patient-level bootstrap resamples of the test set (patients resampled with replacement; all cores from each resampled patient are included).
- Classification thresholds are selected by maximising Youden J on the validation split.
- PR-AUC baseline equals the test-split endpoint prevalence (10.7% for GG2+; 3.8% for GG3+); PR-lift = PR-AUC / prevalence.
- The full 14-feature model (`all_geometry_no_availability_plus_clinical`) is reported as a sensitivity reference in Table 3.

*Source: `reports/minimal_vs_full_model_comparison.md`.*

---

## Table 3 — Compact versus full model comparison

Δ = full − compact (positive = full outperforms compact; negative = compact outperforms full).  
Reference (full) feature set: `all_geometry_no_availability_plus_clinical` (14 features).  
Compact feature set: `target_geometry_plus_clinical` (8 features).  
Δ values are means across logistic regression, HistGradientBoosting, and XGBoost.  
Predefined practical equivalence margins: |ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015.

| Endpoint | Compact LR ROC-AUC | Full LR ROC-AUC | Mean ΔROC (full − compact) | Compact LR PR-AUC | Full LR PR-AUC | Mean ΔPR (full − compact) | Within margins? |
|----------|--------------------|-----------------|---------------------------|-------------------|----------------|--------------------------|-----------------|
| GG2+ / csPCa | 0.754 | 0.753 | −0.002 | 0.327 | 0.319 | 0.000 | Yes |
| GG3+ / high-grade | 0.828 | 0.829 | −0.005 | 0.269 | 0.265 | +0.008 | Yes |

**Notes.**
- LR values are from the test split; mean Δ values are averaged across all three model families (LR, HGB, XGBoost) using the full vs compact contrast.
- A negative mean Δ (full − compact) indicates the compact model has higher mean performance than the full model on that metric.
- The compact model is preferred on parsimony and interpretability grounds; the full model is retained as a sensitivity/reference analysis.
- These margins are predefined practical thresholds, not a formal statistical equivalence test.

*Sources: `reports/minimal_vs_full_model_comparison.md` (LR values); `reports/minimal_vs_full_delta_table.md` (mean Δ values).*

---

## Table 4 — Risk stratification enrichment

Feature set: `target_geometry_plus_clinical`, logistic regression. Test split only.  
Cores are ranked by predicted probability from highest to lowest.

| Endpoint | Baseline prevalence | Top 20% prevalence | Top 10% prevalence | Top 5% prevalence | Top 5% enrichment | Top 10% capture rate | Top 20% capture rate |
|----------|--------------------|--------------------|--------------------|--------------------|-------------------|----------------------|----------------------|
| GG2+ / csPCa | 10.7% | 26.9% | 34.3% | 50.4% | 4.7× | 32.0% | 50.2% |
| GG3+ / high-grade | 3.8% | 11.0% | 19.0% | 31.4% | 8.3× | 50.0% | 57.6% |

**Notes.**
- Baseline prevalence = overall positive rate in the test set.
- Top-N% prevalence = positive core rate among the highest-scoring N% of test cores.
- Top-N% capture rate = fraction of all positive test cores recovered in the top-N%.
- Enrichment = (top-5% prevalence) / (baseline prevalence).
- These figures characterise model-guided ranking; they do not reflect calibrated absolute cancer risk and should not be used for threshold-based clinical decisions without external validation and recalibration.

*Source: `reports/minimal_vs_full_risk_stratification.md`.*

---

## Table 5 — Grouped inference: cluster-robust odds ratios

Core-level logistic regression with standard errors clustered by patient.  
All valid splits pooled (16,992 cores, 796 patients) to maximise statistical power.  
Stars: \*\*\* p < 0.001; \*\* p < 0.01; \* p < 0.05.

| Feature | GG2+ OR [95% CI] | GG2+ p | GG3+ OR [95% CI] | GG3+ p |
|---------|-----------------|--------|-----------------|--------|
| Distance to target surface | 0.468 [0.341–0.644] | < 0.001 \*\*\* | 0.323 [0.210–0.498] | < 0.001 \*\*\* |
| Fraction inside target | 1.331 [1.209–1.464] | < 0.001 \*\*\* | 1.227 [1.061–1.418] | 0.006 \*\* |
| Distance to target centroid | 0.880 [0.652–1.189] | 0.40 | 1.056 [0.709–1.575] | 0.79 |
| Trajectory intersects target | 0.937 [0.744–1.180] | 0.58 | 0.910 [0.632–1.309] | 0.61 |
| log(PSA) | 1.818 [1.509–2.191] | < 0.001 \*\*\* | 2.710 [1.954–3.758] | < 0.001 \*\*\* |
| PSA | 1.120 [0.968–1.296] | 0.13 | 1.056 [0.886–1.260] | 0.54 |
| Prostate volume | 0.681 [0.573–0.811] | < 0.001 \*\*\* | 0.698 [0.545–0.894] | 0.004 \*\* |
| PSA density | 0.951 [0.790–1.144] | 0.59 | 0.939 [0.753–1.171] | 0.58 |

**Notes.**
- Continuous predictors are z-score standardised (1 SD unit); ORs are per one standard deviation.
- `Trajectory intersects target` is a binary (0/1) indicator; its OR is the log-odds change from non-intersecting to intersecting cores.
- Collinearity is present among both geometry and clinical predictors: distance-to-surface and distance-to-centroid are strongly correlated; PSA and log(PSA) are monotonically related; PSA density is a function of PSA and prostate volume. Individual coefficients for correlated predictors should be interpreted cautiously and are not standalone feature-importance rankings.
- GEE (exchangeable working correlation) results are concordant; see `reports/grouped_core_level_inference.md` Section 4.
- This is an association/inference analysis, not a held-out predictive-performance analysis.

*Sources: `reports/grouped_core_level_inference.md` Section 3; `reports/grouped_core_level_inference_cluster_robust.csv`.*

---

## Table 6 — Patient-level secondary analysis

Patient labels: positive if at least one biopsy core is positive for the endpoint.  
Patient-level prevalence is consequently higher than core-level (GG2+: 58.3%; GG3+: 25.8% on the test split).  
Δ = direct − naive (positive = direct model outperforms naive aggregation).

| Endpoint | Naive aggregation (model / approach) | Naive ROC-AUC | Naive PR-AUC | Best direct model (feature set / model) | Direct ROC-AUC | Direct PR-AUC | ΔROC | ΔPR | Interpretation |
|----------|--------------------------------------|--------------|-------------|----------------------------------------|---------------|--------------|------|-----|----------------|
| GG2+ / csPCa | XGBoost, mean predicted prob | 0.617 | 0.695 | `clinical_only` / LR | 0.613 | 0.697 | −0.004 | +0.002 | Direct ≈ naive; both well below core-level ROC-AUC 0.754 |
| GG3+ / high-grade | LR, mean predicted prob | 0.682 | 0.512 | `all_geometry_aggregates_plus_clinical` / HGB | 0.706 | 0.498 | +0.024 | −0.014 | Direct modestly improves ROC but reduces PR; still below core-level ROC-AUC 0.828 |

**Notes.**
- Naive aggregation: core-level predicted probabilities from the best core-level model, averaged across all cores per patient.
- Direct patient-level models: one row per patient, using aggregated summary features (mean, max, proportion above threshold across cores) plus clinical features; trained and threshold-selected on patient-level splits.
- `n_cores` is excluded from patient-level feature sets because biopsy core count may encode sampling intensity rather than anatomy.
- Patient-level performance remains below core-level performance for both endpoints, consistent with core-level modelling being the primary analytical unit.
- This analysis is secondary; the core-level compact model results (Tables 2, 4) are the primary findings.

*Sources: `reports/patient_level_direct_model_comparison.md` Sections 4–5; `reports/final_patient_level_table.md`.*

---

## Consistency checks and TODOs

### Verified

- **Table 1 totals** cross-check: 557 + 119 + 120 = 796 patients ✓; 11,827 + 2,748 + 2,417 = 16,992 cores ✓; 1,356 + 272 + 259 = 1,887 GG2+ positives ✓; 475 + 80 + 92 = 647 GG3+ positives ✓.
- **Table 2 PR-lift** check: 0.327 / 0.107 ≈ 3.06 ≈ 3.05 ✓; 0.269 / 0.038 ≈ 7.08 ≈ 7.07 ✓.
- **Table 3 sign convention**: the interpretation section of `minimal_vs_full_model_comparison.md` uses Δ = full − compact; the delta table uses Δ = compact − full. These are sign-consistent. Mean Δ (full − compact): GG2+ ΔROC ≈ −0.002 (compact above full), ΔPR ≈ 0.000; GG3+ ΔROC ≈ −0.005 (compact above full), ΔPR ≈ +0.008 (full above compact). All within predefined practical equivalence margins.
- **Table 4 enrichment**: 0.504 / 0.107 ≈ 4.71 → reported as 4.7× ✓; 0.314 / 0.038 ≈ 8.26 → reported as 8.3× ✓.
- **Table 5 feature ordering** matches the specification in `reports/final_paper_artifact_manifest.md` and `src/make_grouped_inference_forest_plot.py`.
- **Table 6 Δ values** match `reports/patient_level_direct_model_comparison.md` Section 5 comparison table verbatim.

### TODOs / ambiguities

- **Table 1**: `reports/final_core_count_audit_summary.md` does not include split-level breakdown; split-level numbers are taken from `reports/grouped_core_level_inference.md` Section 2. TODO: verify that the same dataset and filter are used across both reports.
- **Table 2**: PR-AUC CI for GG3+ is wide ([0.094–0.516]), reflecting the small number of positive cores in the test split (n = 92). This should be flagged in the manuscript.
- **Table 3**: The full-vs-compact mean Δ values in `reports/minimal_vs_full_model_comparison.md` are stated in prose (not as a computed table). TODO: confirm these values by recomputing from the delta table if any doubt arises.
- **Table 6**: The "best direct patient-level model" differs by endpoint (clinical-only LR for GG2+; all-geometry-plus-clinical HGB for GG3+). This inconsistency should be acknowledged in the manuscript: the model selection is data-driven and the small test-set patient count (n = 120) makes selection unstable.
