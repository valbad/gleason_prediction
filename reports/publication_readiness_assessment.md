# Publication Readiness Assessment

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-06-10  
**Status:** Go/no-go decision based on completed analyses. Do not rerun models. Do not modify existing reports or CSVs.

---

## 1. Executive Decision

**GO — with cautious framing.**

The compact geometry-clinical model is ready to write up as a completed study with stated limitations. The evidence is rigorous given the data available (patient-disjoint evaluation, cluster-robust inference, PR-AUC primary metric, equivalence test, bootstrapped CIs), the discriminative signal is genuine and interpretable, and the dataset is public (TCIA). A single-site retrospective study with 796 patients and no external validation is publishable in this field; the framing must foreground these as primary limitations, not bury them. Do not claim clinical deployability. Do not claim the model is calibrated for clinical use without Platt recalibration and a prospective cohort. Do not claim generalisability.

---

## 2. Main Scientific Result

Core-level cancer prediction from needle-to-target geometry and four clinical variables (log-PSA, PSA density, prostate volume, age) yields clinically interpretable discrimination on the TCIA Prostate-MRI-US-Biopsy dataset. On an independent held-out test split (120 patients, 2,417 cores) with patient-disjoint splits, the compact 8-feature logistic regression achieves ROC-AUC 0.754 [0.699–0.807] and PR-AUC 0.327 [0.230–0.461] for GG2+ (csPCa, ISUP ≥ 2), and ROC-AUC 0.828 [0.751–0.892] and PR-AUC 0.269 [0.094–0.516] for GG3+ (high-grade, ISUP ≥ 3), with PR-lifts of 3.05× and 7.07× over the base prevalence respectively. The model concentrates the top-5% highest-scored cores at 4.7× enrichment for GG2+ and 8.3× enrichment for GG3+, capturing 32.0% and 50.0% of all test positives in the top-10%. Distance to target surface (cluster-robust OR 0.468 for GG2+, 0.323 for GG3+, both p < 0.001) and fraction of centerline inside target (OR 1.331, 1.227) are the strongest geometry predictors; log-PSA and PSA density are the strongest clinical predictors. A pre-specified equivalence test confirms that a 14-feature full model is not meaningfully better than the 8-feature compact model for either endpoint (|ΔROC| < 0.010, |ΔPR| < 0.015), supporting the compact model as the primary result.

---

## 3. Why the Result Is Publishable

1. **Patient-disjoint evaluation.** Train/val/test splits use GroupShuffleSplit on patient ID. No core from a test patient appears in training. This is a minimum bar for credible prediction model evaluation; many published studies in this space do not meet it.

2. **Appropriate primary metric.** PR-AUC is used as a co-primary metric alongside ROC-AUC. Cancer cores are 10.7% (GG2+) and 3.8% (GG3+) of all test cores; ROC-AUC alone would be optimistic. PR-lift (×3.05, ×7.07) provides an additional interpretable summary that reviewers can interrogate.

3. **Cluster-robust inference.** Coefficient standard errors are computed with cluster-robust sandwich estimators (statsmodels GLM with cov_type="cluster"), clustering on patient ID to account for within-patient correlation across cores. A GEE concordance check (exchangeable working correlation) confirms results. This is methodologically correct and explicitly stated.

4. **Bootstrapped CIs at the patient level.** All test-set performance metrics carry 95% CIs from 1,000 patient-level bootstrap resamples (resample patients, then pool their cores). CIs are wide for GG3+ PR-AUC [0.094–0.516], which is stated explicitly and does not invalidate the result; it calibrates the claim.

5. **Public dataset with a citable acquisition protocol.** The TCIA Prostate-MRI-US-Biopsy dataset is publicly available and its acquisition system (Artemis MRI-US fusion robot) is described in a peer-reviewed reference (Sonn 2013). Results are reproducible given the code.

6. **Pre-specified equivalence test (compact vs full).** The compact model is not simply chosen by post-hoc inspection. The equivalence margins (|ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015) were specified before comparing the two feature sets. Both models fall within these margins for both endpoints, so the choice of the compact model is statistically supported, not arbitrary.

7. **Calibration assessment with acknowledged limitations.** Raw logistic regression probabilities are poorly calibrated (Brier skill score negative relative to a prevalence-based model before recalibration). Platt recalibration yields usable calibration (BSS +0.108/+0.084 for GG2+/GG3+; ECE ≈ 0.020). This is reported transparently; the paper cannot claim that raw predicted probabilities are valid absolute risk estimates, and it does not need to.

---

## 4. What the Result Is Not

- **Not clinically deployable.** No prospective validation. No regulatory pathway. No decision-curve analysis demonstrating net benefit at any threshold. These must be stated as future work, not deferred limitations buried in a final paragraph.
- **Not externally validated.** All results are from a single institution (UCLA/Artemis system). Whether distance-to-surface geometry is predictive in a different fusion system, a different ultrasound platform, or a different registration pipeline is unknown. The claim is limited to this dataset.
- **Not calibrated for clinical risk communication.** Raw predicted probabilities should not be reported as "probability of cancer." Platt-recalibrated probabilities are acceptable for research use, with appropriate CI framing.
- **Not a deep learning model.** The model is interpretable logistic regression. This is a strength (sparse features, meaningful coefficients, no black box), not a limitation — but reviewers may expect DL. Frame the absence of DL as a deliberate design choice (interpretability, sample size, generalisability), not a gap.
- **Not a patient-level triage tool.** Patient-level models underperform core-level models in this dataset (ROC-AUC 0.613 vs 0.754 for GG2+; 0.706 vs 0.828 for GG3+). The contribution is at the core level, not the patient level. Do not conflate the two.
- **Not a causal or mechanistic model.** OR estimates from logistic regression are associative. Distance to target is the strongest predictor, which is biologically interpretable (nearer cores more likely to sample the lesion), but confounders exist and are not ruled out.

---

## 5. Strength of Evidence Table

| Claim | Evidence | Confidence | Caveats |
|-------|----------|-----------|---------|
| Needle-to-target geometry is predictive of biopsy cancer status | Test ROC-AUC 0.754 (GG2+), 0.828 (GG3+); distance-to-surface OR 0.468–0.323, both p < 0.001 | **High** within this dataset | Single institution, one fusion system |
| The compact 8-feature model captures most of the predictive signal | Within equivalence margins vs 14-feature full model for both endpoints | **High** | Defined on this dataset; may differ elsewhere |
| Distance to target surface is the strongest single geometry predictor | Largest significant OR in cluster-robust coefficient table; both endpoints *** | **High** | Correlation with other geometry features not fully deconfounded |
| Risk stratification is interpretable | Top-5% enrichment 4.7× (GG2+), 8.3× (GG3+); top-10% captures 32%, 50% of positives | **High** | Enrichment depends on prevalence in this dataset |
| Patient-level ROC-AUC is lower than core-level | 0.613 vs 0.754 (GG2+); 0.706 vs 0.828 (GG3+) | **High** | Expected from aggregation; not a flaw |
| Raw logistic probabilities are not calibrated | BSS negative before recalibration; ECE ~0.020 after Platt scaling | **High** | Platt-scaled probabilities usable with caution |
| Extended features (targeting context, interactions) do not replace compact model | No extension exceeds both Δ-thresholds for both endpoints | **Moderate** | Test set ~120 patients; Δ ≤ 0.015 near noise floor; `compact_signed_plus_targeting_plus_inter` crosses ΔPR for GG2+ only |
| Novelty is plausible: no prior core-level outcome prediction from needle geometry | Systematic literature review; 15 references verified; Gayo 2022 is closest prior work (needle geometry + RL, not outcome prediction) | **Moderate** | 3 literature gaps remain (mpMRI guideline; per-core yield variability; registration error study); a targeted PubMed search may surface closer prior work |

---

## 6. Extended Sprint Conclusion

The extended feature sprint (7 candidate feature sets vs compact baseline) returned a null result. No extension meaningfully and consistently improves performance:

- **Signed distance recode:** Negligible delta for both endpoints. The unsigned distance variable is retained.
- **Procedural targeting context (`is_targeted_or_prior_positive`):** Does not cross either Δ-threshold as a standalone addition. The geometry features likely encode much of the same information.
- **Interaction terms (geometry × PSA):** No consistent improvement; cannot detect interaction effects at this sample size.
- **`compact_signed_plus_targeting_plus_inter`:** Crosses ΔPR-AUC threshold for GG2+ only (+0.021), but the gain is not reproduced for GG3+ (ΔPR −0.030). This is treated as exploratory; it does not replace the primary compact model.

The sprint provides evidence that the compact model is not clearly under-specified. Report the null result as a robustness check, not a failure.

---

## 7. Remaining Work

### Tier A — Required before first submission

| Task | Rationale |
|------|-----------|
| Fill 3 citation gaps in `reports/citation_integration_plan.md` (mpMRI guideline/review sentence 2; per-core yield variability sentence 4; registration/targeting error study sentence 5) | Intro §P1 has TODO placeholders that must be resolved before submission |
| Write a complete manuscript draft (intro, methods, results, discussion, conclusion) | The outline and tables exist; the draft does not |
| Produce final figure set with captions | Figures are described in `reports/figure_and_table_captions.md`; confirm files exist and captions are complete |
| Add Platt-recalibration step to reported pipeline | Raw probabilities cannot be the outputs of the reported model; recalibrated probabilities need an explicit methods paragraph |

### Tier B — Strongly recommended before submission

| Task | Rationale |
|------|-----------|
| Decision-curve analysis (DCA) | Reviewers at clinical journals will ask. DCA at ISUP ≥ 2 threshold is straightforward with the existing test-split predictions. Vickers 2006 is already in the reference list |
| TRIPOD and TRIPOD+AI checklist completion | Collins 2015, Collins 2024 already in reference list; checklist ensures no reporting items are missed |
| PROBAST bias assessment | Wolff 2019 already in reference list; required by journals that follow PROBAST |
| Sensitivity analysis: GEE as primary (not concordance check) | Confirm core claims are robust to inference method choice |

### Tier C — Desirable but not blocking

| Task | Rationale |
|------|-----------|
| External validation on a second fusion-biopsy dataset | No comparable public dataset with needle coordinates identified; mention as a future study |
| Prospective integration into the Artemis workflow | Clinical research protocol, not a modelling task |
| Subgroup analyses (PI-RADS category, lesion size, prior-biopsy status) | Hypothesis-generating; sample size limits power; add as supplementary if reviewers request |
| Feature importance stability across bootstrap resamples | Supplementary material for methods-oriented reviewers |

---

## 8. Recommended Journal Framing

The work is a prediction model study using a public dataset, compact interpretable model, and standard ML evaluation. Appropriate venues in rough order of fit:

| Category | Examples | Fit |
|----------|----------|-----|
| Urological oncology, clinical focus | *European Urology*, *Journal of Urology*, *Prostate Cancer and Prostatic Diseases* | Best fit if DCA is added (Tier B); these journals value clinical impact and prospective framing. Without DCA, reviewers will push back. |
| Medical informatics / clinical prediction models | *BMC Medical Informatics and Decision Making*, *JAMIA*, *Journal of Biomedical Informatics* | Good fit for the methodological contribution (cluster-robust inference, PR-AUC, equivalence test). Less pressure on DCA. |
| Radiology / imaging informatics | *European Radiology*, *Radiology: Artificial Intelligence*, *European Journal of Radiology* | Fit if framing emphasises MRI-US fusion context and the TCIA dataset. |
| Surgical robotics / intervention | *International Journal of Medical Robotics and Computer Assisted Surgery* | Narrow fit; the Artemis robot is the data source but the contribution is statistical, not robotic. |
| Open-access preprint first | *medRxiv* | Consider depositing before journal submission to establish priority given the novelty claim. |

**Recommended target:** medical informatics or urological oncology journal. Add DCA (Tier B) before submitting to a clinical journal; a methods journal is reachable without it.

---

## 9. Recommended Next Action

**Stop new modelling.** The model is complete. The extended sprint returned a null result and is documented. No further feature engineering or model comparison is needed before first submission.

**Immediate next step:** Resolve the three citation gaps in `reports/citation_integration_plan.md` §P1 (mpMRI guideline, per-core yield variability reference, registration/targeting error study), then write a complete first draft of the manuscript using `reports/manuscript_outline_full.md` as the scaffold. The tables, captions, calibration results, risk-stratification results, coefficient tables, and extended-sprint null result are all ready to be dropped in. The draft can be produced without any new analysis.

---

*Sources: `reports/main_paper_tables.md`, `reports/grouped_core_level_inference.md`, `reports/final_calibration_table.md`, `reports/patient_level_direct_model_comparison.md`, `reports/extended_feature_set_comparison.md`, `reports/extended_feature_set_recommendation.md`, `reports/extended_feature_set_risk_stratification.md`, `reports/literature_review_findings.md`, `reports/citation_integration_plan.md`. No models were rerun and no numeric values were invented for this report.*
