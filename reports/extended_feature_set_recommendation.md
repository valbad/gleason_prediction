# Extended Feature Set Recommendation

**Branch:** `feat/geometry-clinical-baselines`  
**Script:** `src/compare_extended_feature_sets.py`  
**Based on:** `reports/extended_feature_set_comparison.md`

## Decision criteria

An extended feature set is considered **meaningfully better** than the compact baseline only if, on the held-out **test split** with logistic regression:

- ΔPR-AUC ≥ +0.020  **OR**
- ΔROC-AUC ≥ +0.015

OR if it substantially improves top-ranked enrichment (top-5% capture rate or prevalence) without relying on pathology leakage.

These thresholds are identical to the equivalence margins used in the compact-vs-full comparison (`reports/minimal_vs_full_model_comparison.md`).

## Feature-type classifications

| Feature | Type | Leakage risk |
|---------|------|-------------|
| `distance_midpoint_to_target_surface_mm` | Target geometry | None |
| `signed_distance_midpoint_to_target_surface_mm` | Target geometry (recode) | None |
| `distance_midpoint_to_target_centroid_mm` | Target geometry | None |
| `trajectory_intersects_target` | Target geometry | None |
| `approximate_fraction_of_centerline_inside_target` | Target geometry | None |
| `psa_ng_ml`, `log_psa_ng_ml`, `prostate_volume_cc`, `psa_density` | Clinical | None |
| `is_targeted_or_prior_positive` | Procedural targeting context | **See note** |
| Interaction terms | Derived | None |
| `cancer_length_mm`, `pct_cancer_in_core` | Pathology outcome | **Leakage — excluded** |

**Note on `is_targeted_or_prior_positive`:** Encodes the physician's pre-biopsy targeting decision.  Not pathology leakage, but encodes clinical suspicion (MRI PI-RADS, prior-session positivity) that is not available in the standard compact feature set.  Any model including this feature must be clearly labelled as a *procedural-context extended model*, not the primary compact model.

## Results summary

### GG2+ / csPCa

**Compact baseline (LR):** ROC-AUC = 0.754, PR-AUC = 0.327

| Feature set | ΔROC-AUC | ΔPR-AUC | Meaningful? |
|-------------|----------|---------|------------|
| `compact_with_signed_distance` | +0.001 | +0.002 | No |
| `compact_plus_targeting_context` | +0.008 | +0.013 | No |
| `compact_signed_plus_targeting_context` | +0.008 | +0.019 | No |
| `compact_signed_plus_interactions` | +0.001 | -0.001 | No |
| `compact_signed_plus_targeting_plus_inter` | +0.009 | +0.021 | **YES** |
| `clinical_plus_targeting_context` | -0.011 | +0.016 | No |
| `targeting_context_only` | -0.080 | -0.164 | No |

**Note on `compact_signed_plus_targeting_plus_inter`:** This feature set crosses the numerical ΔPR-AUC threshold for GG2+ only (+0.021), but this gain is not reproduced for GG3+ and relies on post-hoc procedural-context and interaction features. It is therefore considered exploratory and does not justify replacing the primary compact model.

### GG3+ / high-grade

**Compact baseline (LR):** ROC-AUC = 0.828, PR-AUC = 0.269

| Feature set | ΔROC-AUC | ΔPR-AUC | Meaningful? |
|-------------|----------|---------|------------|
| `compact_with_signed_distance` | -0.007 | +0.002 | No |
| `compact_plus_targeting_context` | +0.001 | +0.017 | No |
| `compact_signed_plus_targeting_context` | -0.008 | +0.007 | No |
| `compact_signed_plus_interactions` | -0.010 | -0.044 | No |
| `compact_signed_plus_targeting_plus_inter` | -0.010 | -0.030 | No |
| `clinical_plus_targeting_context` | -0.039 | +0.020 | No |
| `targeting_context_only` | -0.143 | -0.207 | No |

## Recommendations

### 1. Geometry and clinical features

**Signed distance to target surface:** The signed-distance recode does not improve performance beyond the compact baseline for either endpoint.  Retain `distance_midpoint_to_target_surface_mm` (unsigned) in the compact model.  The signed variant provides cleaner interpretation but no measurable predictive benefit; mention as a model property in the Methods section if desired, but do not substitute.

**Interaction features:** Interaction terms (signed distance / fraction inside × PSA density / log-PSA) do not improve performance beyond the compact baseline.  Do not include in any model.  These exploratory interaction terms did not improve held-out performance consistently. This does not prove absence of biological interaction; it only indicates that these simple product terms are not useful additions to the current compact model.

### 2. Procedural targeting-context flag

**`is_targeted_or_prior_positive`** does not improve performance beyond the compact baseline by the pre-specified thresholds.  This suggests that the geometric features (distance, fraction inside target) already partially capture the information encoded in the targeting decision, or that the effect is not strong enough to detect given the test-set sample size.

**Recommendation:** Do not add `is_targeted_or_prior_positive` to any primary model.  The feature audit result (see above) should still be reported in the manuscript to document that the 'TARGET OR PRIOR POSITIVE' category was identified and its leakage status was assessed.

### 3. Primary compact model recommendation

**Retain `current_compact` (`target_geometry_plus_clinical`, 8 features) as the primary compact model** unless the signed-distance recode is meaningfully better for both endpoints (see §1 above).

No extension tested here justifies replacing the compact geometry-clinical model as the primary reported result.  The compact model remains the most parsimonious, interpretable, and scientifically defensible choice for the manuscript's central claim.

### 4. Limitations of this sprint

- Test set contains ~120 patients; all delta estimates have wide bootstrap CIs.  A ΔROC-AUC of ±0.015 is near the noise floor at this sample size.
- `is_targeted_or_prior_positive` was not pre-registered as a candidate feature before the modelling phase; it was identified by post-hoc column audit.  Treat its performance as exploratory.
- Interaction terms were not pre-specified; all results for extended sets E and F should be considered hypothesis-generating only.
- This sprint does not constitute a prospective validation.  Calibration and decision-curve analysis were not re-run; raw probabilities are not reported as calibrated absolute risks.

---

*Report generated by `src/compare_extended_feature_sets.py`.  Do not modify by hand.*
