# Paper Skeleton and Results Narrative

*Synthesised from existing reports as of 2026-07-01. All numbers are taken verbatim from generated report files.*

---

## 1. Proposed Working Title

1. **"Needle-to-target geometry predicts biopsy core positivity in MRI-guided prostate biopsy: a compact interpretable model"**
2. **"Target-relative spatial features of MRI-guided biopsy cores are associated with clinically significant prostate cancer: a core-level geometry analysis"**
3. **"Core-level biopsy trajectory geometry as an interpretable predictor of cancer grade in MRI-US fusion prostate biopsy"**

Avoid titles that imply a clinical decision-support system or deployment readiness. Emphasise the spatial/geometric framing, the MRI-defined target, and the interpretable core-level nature of the approach.

---

## 2. Central Claim

Four claims together form the paper's argument:

1. **Core-level signal exists.** Needle-to-target spatial geometry — principally the distance from the biopsy core midpoint to the target surface and the fraction of the core centerline inside the target — is associated with biopsy positivity for both clinically significant cancer (GG2+) and high-grade cancer (GG3+), at the individual core level. This holds across multiple model classes (logistic regression, gradient boosting) and is consistent with the biology: cores closer to or more deeply inside the lesion are more likely to sample it.

2. **A compact model captures most of the signal.** The eight-feature `target_geometry_plus_clinical` model (four target-geometry features + four PSA-related clinical features) performs within the pre-defined equivalence margin of the fourteen-feature full model across both endpoints (GG2+: ΔROC = −0.002, ΔPR = 0.000; GG3+: ΔROC = −0.005, ΔPR = +0.008). Non-target biopsy/prostate geometry adds negligible discriminative value on top of target-relative geometry.

3. **Patient-level prediction is harder.** Aggregating core-level scores or building direct patient-level models reduces discriminative performance by approximately 0.14 ROC-AUC compared to the core-level models (GG2+: 0.758 → 0.617; GG3+: 0.829 → 0.682). This is partly a consequence of the high patient-level prevalence (GG2+: 58.3%; GG3+: 25.8%) relative to core prevalence (10.7% / 3.8%), which compresses the discrimination problem.

4. **The core-level association is not solely an artefact of patient clustering.** Cluster-robust logistic regression (SE clustered by patient) confirms that `distance_midpoint_to_target_surface_mm` remains individually significant for both GG2+ (OR = 0.468, p < 0.001) and GG3+ (OR = 0.323, p < 0.001). A secondary GEE analysis with exchangeable working correlation is concordant. The association survives intra-patient correlation adjustment.

---

## 3. Main Paper Framing

### Why this should not be framed as a CNN/GNN performance paper

The proposed paper should not be framed around the earlier CNN/GNN experiments. The current central contribution uses tabular geometric features derived from biopsy and target coordinates rather than raw MRI image patches or deep neural representations. Foregrounding the earlier deep learning experiments would misrepresent the current contribution and obscure it.

Attempting to maximise ROC-AUC on a small single-cohort test set (120 patients, 2,417 cores) would also lead to unstable estimates with wide bootstrap CIs (±0.05–0.07), making any performance claim fragile.

### Why the stronger framing is methodological and interpretable

The genuine contribution is the representation choice: encoding each biopsy core as a point in a needle-aligned coordinate frame relative to the MRI-defined target mesh, then extracting scalar geometric features (surface distance, centroid distance, fraction inside, trajectory intersection). This representation is:

- **Clinically motivated.** Proximity to the target directly reflects how accurately the needle reached the lesion the radiologist identified. This is a first-order physical explanation of biopsy yield.
- **Transparent.** The features have units (millimetres, fractions) and directly interpretable directions (closer to surface → higher cancer probability). This contrasts with latent image features from a deep network.
- **Compact and reproducible.** Eight features are enough to capture most of the discriminative signal. The model can be inspected, audited, and re-run on any MRI-guided biopsy dataset that records target mesh coordinates.
- **Methodologically honest.** The paper quantifies core-level ranking and risk stratification performance, acknowledges calibration limitations, accounts for intra-patient dependence, and reports patient-level performance as a secondary check.

The framing should be: *we show that explicit spatial representation of needle-to-target geometry, combined with standard clinical features, provides an interpretable and effective core-level risk stratification approach, and we characterise the conditions under which this signal is and is not reliable.*

---

## 4. Main Analyses to Include in the Paper

### 4.1 Dataset and endpoint audit

**Purpose:** Establish the sample, splits, label definitions, and known data quality issues before any modelling.

**Main result:** 796 patients, 16,992 labelled cores (coord-match). GG2+ core prevalence 11.1%; GG3+ core prevalence 3.8%. Two pipeline caveats: (a) `target_mesh_available = False` for 26 cores with 53.8% positivity — excluded as conservative ablation; (b) 36 cores (0.2%) have Gleason 3+5=8 mismapped in `build_manifest.py` — identified and handled in downstream analysis.

**Why main paper:** Ground the reader. A dataset audit is necessary in any methods paper to establish reproducibility and flag caveats early.

**Suggested table/figure:** Table 1 — dataset split and endpoint summary (patients, cores, prevalence by split and endpoint). Inline text for the two caveats.

---

### 4.2 Core-level compact model performance

**Purpose:** Report the main predictive performance of the compact `target_geometry_plus_clinical` model.

**Main result:**
- GG2+: LR ROC-AUC 0.754 [0.699–0.807], XGBoost ROC-AUC 0.754 [0.694–0.808]; PR-AUC ~0.32; PR-lift ~3.0.
- GG3+: LR ROC-AUC 0.828 [0.751–0.892]; PR-AUC 0.269; PR-lift 7.1.
- Both endpoints outperform clinical-only (GG2+ LR: 0.650; GG3+ LR: 0.716) and target-geometry-only (GG2+ LR: 0.730; GG3+ LR: 0.806) baselines.

**Why main paper:** This is the primary held-out predictive-performance result.

**Suggested table/figure:** Table 2 — compact model ROC-AUC, PR-AUC, PR-lift, sensitivity, specificity, F1 with 95% bootstrap CIs for both endpoints. Figure 1 — ROC and PR curves for LR and XGBoost compact models, both endpoints.

---

### 4.3 Minimal vs full feature-set comparison

**Purpose:** Justify the compact model choice by showing the full model adds negligible value.

**Main result:** ΔROC = −0.002, ΔPR = 0.000 (GG2+); ΔROC = −0.005, ΔPR = +0.008 (GG3+). Both within the pre-defined equivalence margin (|ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015). Biopsy/prostate geometry without target geometry (negative-control set) performs near clinical-only level (GG2+ mean ROC-AUC 0.640; GG3+ 0.726), confirming target-relative geometry drives the signal.

**Why main paper:** This is a direct methodological contribution — it tells the reader which features matter and why parsimony is justified.

**Suggested table/figure:** Table 3 — mean ROC-AUC and PR-AUC by feature set across models, with ΔROC and ΔPR vs. reference. Figure 2 — barplot or delta forest plot of mean ROC-AUC by feature set for both endpoints.

---

### 4.4 Risk stratification / enrichment

**Purpose:** Translate discriminative performance into a clinically relevant framing: how well does the model concentrate positives in the highest-ranked cores?

**Main result (compact model, logistic regression):**
- GG2+: top 5% prevalence 50.4% (vs 10.7% baseline; 4.7× enrichment), top 10% captures 32.0% of all positives.
- GG3+: top 5% prevalence 31.4% (vs 3.8% baseline; 8.3× enrichment), top 10% captures 50.0% of all positives.

**Why main paper:** Risk stratification framing is more clinically accessible than ROC-AUC. On the held-out test split, the compact model ranks nearly half of high-grade positive cores within the top decile of predicted scores.

**Suggested table/figure:** Table 4 — top 5/10/20% prevalence and capture rate for compact model, both endpoints, with full model as sensitivity analysis. Figure 3 — enrichment plot (capture rate vs. fraction of cores examined), compact vs. full model.

---

### 4.5 Grouped core-level inference

**Purpose:** Test whether the core-level associations are robust to intra-patient correlation, not merely artefacts of repeated observations per patient.

**Main result:** `distance_midpoint_to_target_surface_mm` remains individually significant after patient-cluster adjustment for both endpoints. Cluster-robust OR: 0.468 (GG2+, p < 0.001), 0.323 (GG3+, p < 0.001). GEE concordant: OR 0.581 (GG2+, p < 0.001), 0.441 (GG3+, p < 0.001). `trajectory_intersects_target` is not individually significant once continuous geometry features are included, consistent with it being highly redundant with `approximate_fraction_of_centerline_inside_target` (Spearman ρ = 0.979).

**Why main paper:** Without this analysis, a reviewer could legitimately object that all results are inflated by intra-patient correlation (many cores per patient → many pseudo-replicated observations). The cluster-robust and GEE analyses directly address this concern and belong in the main body.

**Suggested table/figure:** Table 5 — cluster-robust odds ratios, 95% CIs, p-values for the compact feature set, both endpoints. Figure 4 — forest plot of cluster-robust ORs (compact model) for both endpoints simultaneously.

---

### 4.6 Patient-level direct modelling (secondary analysis)

**Purpose:** Report that patient-level prediction is harder, situate this as a secondary methodological check rather than the primary result.

**Main result (direct patient-level models):** GG2+ best: clinical-only LR, ROC-AUC 0.613 [0.515–0.716]; GG3+ best: all-geometry + clinical HGB, ROC-AUC 0.706 [0.583–0.823]. Both substantially below core-level equivalents (ΔROC ≈ −0.14). Target geometry aggregates add minimal value over clinical features at patient level.

**Why main paper (as secondary analysis):** A reviewer or reader will ask "but what about patient-level prediction?" Reporting it as a secondary analysis — with clear explanation of why it is harder — pre-empts this question and is methodologically responsible.

**Suggested table/figure:** Table 6 — patient-level direct model results vs. core-level reference, both endpoints. Can be condensed to a 4-row comparison table. No separate figure needed; reference the core-level vs. patient-level gap in text.

---

### 4.7 Calibration caveat

**Purpose:** Honestly report that raw probabilities are poorly calibrated and that Platt recalibration partially helps, but the GG3+ model remains conservative after recalibration.

**Main result:** BSS < 0 for all raw models (raw probabilities worse than prevalence-based naive prediction). After Platt scaling: BSS 0.108 (GG2+ XGBoost), 0.084 (GG3+ LR). ECE < 0.03 post-recalibration. GG3+ recalibration slope = 1.511, indicating scores concentrate at the low end; the model should be treated as a ranking tool, not an absolute risk estimator.

**Why main paper:** Any paper reporting probabilities must address calibration. Failing to do so would be a significant methodological omission. This is brief (2–3 sentences + summary table in supplement).

**Suggested table/figure:** Inline text noting calibration caveat; Table S4 (supplement) for full calibration numbers. Optionally, one calibration curve figure in the supplement.

---

## 5. Supplementary Analyses

| Analysis | Rationale for supplement |
|---|---|
| Original image/CNN/GNN baselines (prior project stage) | These predate the geometry framing; including in main paper would mix two different methodological contributions and risk burying the current one. Include as context in supplement or related work. |
| `target_mesh_available` audit | Supports the exclusion decision; one sentence in Methods + Table S1 in supplement. |
| Detailed univariate feature effects (standardised LR coef, Spearman ρ per feature) | Informative for interpretability but secondary to the joint model performance. Table S2. |
| Permutation feature importance (by model and endpoint) | Supplements the univariate table; confirms distance-to-surface importance. Table S3. |
| Core-count audit (ρ = 0.101 GG2+, no association GG3+) | Supports that core-count imbalance is small and does not drive results. One sentence in Limitations + Table S5. |
| Full calibration tables (Brier, BSS, ECE, slope, intercept before/after) | Too granular for main paper body. Table S4. |
| Full VIF / Spearman correlation matrices (8×8) | Supports PSA collinearity discussion; go in supplement alongside inference section. Table S6. |
| Full patient-level aggregation variants (max prob, top-3 mean, prop above threshold) | Supporting detail for the patient-level secondary analysis. Table S7. |
| Full 6-feature-set × model × endpoint performance table | The full `minimal_vs_full_model_comparison.md` table has 18 rows per endpoint; a condensed version (mean across models) belongs in the main paper. Full table in Table S8. |

---

## 6. Proposed Tables

| Table ID | Main / Supp | Content | Source report | Key message |
|---|---|---|---|---|
| Table 1 | Main | Dataset split summary: patients, labelled cores, GG2+ / GG3+ prevalence by split | `geometry_baseline_tracking_note.md` §2 | Dataset is moderately sized; class imbalance is substantial especially for GG3+ |
| Table 2 | Main | Compact model (`target_geometry_plus_clinical`) performance: ROC-AUC [95% CI], PR-AUC [95% CI], PR-lift, sensitivity, specificity, F1 — LR and XGBoost, both endpoints | `minimal_vs_full_model_comparison.md` | Compact 8-feature model achieves ROC-AUC 0.75–0.83 and PR-lift 3–7× across endpoints |
| Table 3 | Main | Feature-set comparison (6 sets): mean ROC-AUC across models, ΔROC and ΔPR vs. reference, both endpoints | `minimal_vs_full_model_comparison.md` / `minimal_vs_full_delta_table.md` | Target geometry + clinical captures full-model signal; biopsy/prostate geometry without target adds little |
| Table 4 | Main | Risk stratification: top 5/10/20% prevalence and capture rate — compact model vs. full model, both endpoints | `minimal_vs_full_risk_stratification.md` / `final_risk_stratification_table.md` | GG3+ model captures 50% of all high-grade positives in the top 10% of scored cores |
| Table 5 | Main | Cluster-robust odds ratios (compact feature set, all 8 features): coef, OR, 95% CI, p-value — both endpoints | `grouped_core_level_inference_cluster_robust.csv` / `grouped_core_level_inference.md` §3 | Distance to target surface: OR 0.47 (GG2+), 0.32 (GG3+) after patient-cluster adjustment |
| Table 6 | Main | Patient-level direct models vs. core-level reference: ROC-AUC, PR-AUC for best model per endpoint | `patient_level_direct_model_comparison.md` | Patient-level ROC-AUC is ~0.14 below core-level; core-level is the primary framing |
| Table S1 | Supp | `target_mesh_available` audit: prevalence by mesh availability | `geometry_baseline_tracking_note.md` §4 | Mesh-absent cores have 53.8% positivity; conservative to exclude |
| Table S2 | Supp | Univariate feature effects: standardised LR coef, univariate ROC-AUC, Spearman ρ | `final_univariate_effects_table.md` | Distance to surface is the single strongest geometric predictor |
| Table S3 | Supp | Permutation feature importance (mean drop in ROC-AUC per feature, by model and endpoint) | `final_feature_importance_table.md` | Consistent with univariate effects; geometry features rank above most clinical features for GG3+ |
| Table S4 | Supp | Full calibration results: Brier, BSS, ECE, slope, intercept before/after Platt scaling | `final_calibration_table.md` | Raw probs are poorly calibrated; Platt scaling helps but GG3+ model remains conservative |
| Table S5 | Supp | Core-count audit: Spearman ρ between n_cores and patient label | `final_core_count_audit_summary.md` | Weak positive association for GG2+ (ρ = 0.101); none for GG3+ |
| Table S6 | Supp | Pairwise Spearman correlations and VIF for compact feature set | `grouped_core_level_inference_correlation.csv` / `grouped_core_level_inference.md` §5 | Geometry features correlated among themselves; PSA group strongly correlated |
| Table S7 | Supp | Full patient-level aggregation variants (all 4 strategies × models × endpoints) | `final_patient_level_table.md` | All aggregation strategies give similar patient-level performance |
| Table S8 | Supp | Full 6 feature-set × 3 model × 2 endpoint performance table | `minimal_vs_full_model_comparison.md` | Complete ablation record |

---

## 7. Proposed Figures

| Figure ID | Main / Supp | Content | Source data / report | Key message |
|---|---|---|---|---|
| Figure 1 | Main | Schematic of biopsy trajectory geometry: prostate outline, target mesh, needle path, extracted scalar features annotated | Conceptual (not data-driven; to be drawn) | Motivates the feature representation; clarifies what distance-to-surface means anatomically |
| Figure 2 | Main | ROC and PR curves for compact model (LR + XGBoost), both endpoints on separate panels | `minimal_vs_full_model_comparison.md` predictions | Visualises discriminative performance; PR curve is important given class imbalance |
| Figure 3 | Main | Barplot of ROC-AUC by feature set (6 sets × 2 endpoints), horizontal bars with model-mean ± range or CI | `minimal_vs_full_model_comparison.md` | Compact model is nearly indistinguishable from full model; target geometry drives the signal |
| Figure 4 | Main | Risk-stratification enrichment: capture rate vs. % of cores examined (compact model vs. clinical-only), both endpoints | `minimal_vs_full_risk_stratification.md` | Compact model substantially enriches for positives vs. clinical-only baseline |
| Figure 5 | Main | Forest plot of cluster-robust ORs for compact feature set (8 features), both endpoints side by side | `grouped_core_level_inference_cluster_robust.csv` | Distance to surface has the largest and most consistent negative association; trajectory_intersects_target is redundant |
| Figure S1 | Supp | Top-5/10/20% enrichment barplot comparing all 6 feature sets | `minimal_vs_full_risk_stratification.md` | Full feature-set comparison of enrichment |
| Figure S2 | Supp | Compact vs. full model delta: ΔROC and ΔPR per model per endpoint, with equivalence bands shown | `minimal_vs_full_delta_table.md` | Visualises that compact model is within equivalence margin |
| Figure S3 | Supp | Calibration curves (reliability diagrams) before and after Platt scaling, both endpoints | `final_calibration_table.md` / recalibrated predictions | Shows calibration improvement and residual GG3+ conservatism |
| Figure S4 | Supp | Patient-level vs. core-level ROC-AUC comparison (bar or dot plot, both endpoints) | `patient_level_direct_model_comparison.md` / tracking note §8 | Quantifies the core-to-patient performance gap |

---

## 8. Results Narrative Draft

### Dataset and endpoints

We analysed 796 patients from the TCIA Prostate-MRI-US-Biopsy dataset, yielding 16,992 biopsy cores with confirmed coordinate-match labels after quality filtering. Patient-stratified train/val/test splits were enforced throughout (557 / 119 / 120 patients). Two binary endpoints were evaluated: GG2+ (ISUP Grade Group ≥ 2, Gleason ≥ 3+4=7; core prevalence 11.1%) and GG3+ (ISUP Grade Group ≥ 3, Gleason ≥ 4+3=7; core prevalence 3.8%). The 26 cores without a target mesh reconstruction were excluded as a conservative ablation (53.8% positivity in this subset versus 11.0% overall; see Supplement).

### Core-level geometry-clinical prediction

We extracted four target-relative geometry features for each biopsy core — distance from the core midpoint to the target centroid, distance to the target surface, a binary indicator of trajectory-target intersection, and the fraction of the core centerline inside the target volume — and combined them with four standard clinical features (PSA, log-PSA, prostate volume, PSA density). On the held-out test set, the compact eight-feature `target_geometry_plus_clinical` logistic regression model achieved ROC-AUC 0.754 [95% CI 0.699–0.807] for GG2+ and 0.828 [0.751–0.892] for GG3+, with PR-AUC of 0.327 and 0.269 respectively (PR-lift 3.1× and 7.1× over baseline). Gradient boosting gave comparable performance. Both endpoints substantially outperformed clinical-only baselines (GG2+: 0.650; GG3+: 0.716) and geometry-only baselines (GG2+: 0.730; GG3+: 0.806), with the combined model capturing complementary signal from each feature group.

### Compact model selection

We compared six feature sets, ranging from clinical-only (4 features) to the full geometry-clinical model (14 features). The compact `target_geometry_plus_clinical` model (8 features) performed within a pre-defined equivalence margin of the full model on both endpoints (GG2+: ΔROC = −0.002, ΔPR = 0.000; GG3+: ΔROC = −0.005, ΔPR = +0.008; margin |ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015). Adding non-target biopsy and prostate geometry to the compact model yielded no meaningful gain. A negative-control set restricted to non-target biopsy/prostate geometry plus clinical features (no target-relative features) performed near the clinical-only level (GG2+: 0.640; GG3+: 0.726), confirming that target-relative geometry — not biopsy geometry in general — drives the predictive signal. The compact model was therefore selected as the primary model on parsimony and interpretability grounds; the full model is retained as a sensitivity analysis.

### Core-level risk stratification

Applying the compact logistic regression model to the test set, the top 5% of cores by predicted risk contained 50.4% positive cores for GG2+ (enrichment 4.7×) and 31.4% for GG3+ (enrichment 8.3×). The top 10% of scored cores captured 32.0% of all GG2+ positives and 50.0% of all GG3+ positives. The GG3+ model, despite the lower absolute PR-AUC, achieves particularly high enrichment at low false-positive fractions, which may be clinically relevant for identifying high-grade lesions when cores are examined in rank order.

### Patient-level modelling

We assessed patient-level performance via two approaches: (i) aggregating core-level predicted probabilities (mean, top-3 mean, proportion above threshold, max) to a single patient-level score, and (ii) building direct patient-level models with aggregated geometric feature summaries. Both approaches performed substantially below the core-level models: the best patient-level direct model achieved ROC-AUC 0.613 (GG2+) and 0.706 (GG3+), compared to core-level equivalents of 0.758 and 0.829. This gap reflects in part the structural difficulty of the patient-level task: patient prevalence is substantially higher than core prevalence (GG2+: 58.3% vs. 10.7%; GG3+: 25.8% vs. 3.8%), and the geometric signal is local and core-specific. Patient-level prediction is reported as a secondary analysis; core-level risk stratification remains the primary result.

### Grouped inference and patient clustering

To assess whether core-level associations could be attributed to repeated correlated observations within patients, we fitted cluster-robust logistic regression models (standard errors clustered by patient) and secondary Binomial GEE models with exchangeable working correlation. In both analyses, `distance_midpoint_to_target_surface_mm` was independently and significantly associated with core positivity after cluster adjustment: cluster-robust OR 0.468 (95% CI 0.341–0.644, p < 0.001) for GG2+ and OR 0.323 (0.210–0.498, p < 0.001) for GG3+. GEE estimates were concordant (GG2+: OR 0.581; GG3+: OR 0.441). `approximate_fraction_of_centerline_inside_target` was positively associated with both endpoints. `trajectory_intersects_target` was not individually significant once continuous geometry features were included, which is expected given its high redundancy with fraction-inside (Spearman ρ = 0.979). These results indicate that the core-level geometric signal is not solely an artefact of intra-patient correlation; it reflects a genuine spatial association between needle-target proximity and cancer detection.

### Calibration and interpretation caveats

Raw predicted probabilities were poorly calibrated before recalibration (BSS < 0 for all models; ECE 0.23–0.26). After Platt scaling on the validation set, calibration improved substantially (BSS 0.08–0.12; ECE < 0.03). For GG3+, the recalibrated slope of 1.511 and high intercept indicate that post-recalibration scores concentrate in the low-probability range; the model should be interpreted as a ranking and risk-stratification tool rather than an absolute probability estimator. No external recalibration was performed; calibration should be re-assessed if the model is applied to a different cohort.

---

## 9. Discussion Narrative

**Main finding.** The geometric relationship between a biopsy needle trajectory and the MRI-defined target lesion is meaningfully associated with whether that core will be positive for clinically significant or high-grade prostate cancer. This finding is not surprising conceptually — cores that intersect or remain close to the target are simply more likely to sample it — but quantifying this relationship with a compact, transparent, interpretable model and validating it against intra-patient correlation is a methodological contribution.

**Why target-relative geometry is clinically meaningful.** In MRI-US fusion prostate biopsy, the radiologist defines a target region on MRI and the interventionist plans needle trajectories to reach it. Quantifying *how well* each needle reached the target post hoc — using the needle coordinate frame and the registered target mesh — creates an audit of the biopsy procedure. Cores that were on-target are more likely to be positive; cores that missed the target contain the geometric signature of the miss. This has implications for quality control and for understanding why any individual biopsy was negative.

**Why the compact model is preferable.** With only eight features, the compact model is fully auditable, requires no feature engineering beyond registration and coordinate extraction, and captures essentially all of the discriminative signal of the fourteen-feature full model. The non-target biopsy geometry features (core length, tube voxel count, prostate surface distance) contribute noise rather than signal when target-relative features are already in the model. Parsimony matters in a clinical setting because simpler models are easier to validate, deploy, audit, and explain to clinicians.

**Why patient-level prediction remains harder.** Predicting whether a patient has cancer (which is the ultimate clinical question) from a collection of core-level predictions involves aggregation under high patient-level prevalence. Most patients with any positive core have cancer; many patients have only one positive core among 15–20 negative ones. This makes patient-level separation difficult and makes the core-level framing — ranking cores by risk to identify which specific cores are likely to contain cancer — a more natural unit of analysis for this geometry model.

**Relationship to MRI radiomics caution.** There is a well-documented concern in the MRI radiomics literature about models that appear predictive on small cohorts but fail to generalise. The current approach differs from radiomic feature extraction in that the features are geometric and physically interpretable (units are millimetres and fractions), not high-dimensional intensity statistics from a specific MRI sequence. This offers some robustness to acquisition heterogeneity, though external validation remains essential.

**Limitations.** See Section 10.

**Future work.** External validation on an independent MRI-guided biopsy cohort is the most important next step. Incorporating prior biopsy history (cores labelled as targeting prior positive lesions account for 7,649 rows and are not currently used as features) may substantially improve performance. PSA density non-linearity (threshold effects around 0.15 ng/mL/cc) warrants investigation. Cross-validation on the full dataset would give more stable performance estimates given the small test cohort.

---

## 10. Limitations to State Explicitly

1. **Retrospective, single-institution dataset.** All results come from the publicly available TCIA Prostate-MRI-US-Biopsy dataset. Selection and referral biases inherent to a single centre are unknown.

2. **No external validation.** The test set is drawn from the same institution and registration pipeline as the training data. Whether the geometric associations or the calibrated probabilities generalise to other MRI-US fusion biopsy systems, scanners, or targeting protocols is unknown.

3. **Biopsy fusion and coordinate noise.** MRI-US registration introduces geometric error. Coordinate noise in the needle path or target mesh boundary would attenuate the geometric signal; the measured associations are therefore likely lower bounds of the true anatomical relationship.

4. **Histology label noise.** The biopsy label is assigned to the entire core, not to a specific spatial location along the needle. Gleason grading has known inter-rater variability. The 36 Gleason 3+5=8 cores mismapped in `build_manifest.py` (0.2% of data) were identified but not corrected in the upstream pipeline.

5. **Repeated cores per patient.** Each patient contributes multiple cores. While the cluster-robust and GEE analyses confirm that the geometric association is not solely an artefact of this structure, the effective sample size is closer to 796 patients than to 16,992 cores for inferential purposes.

6. **Class imbalance, especially GG3+.** With 3.8% core prevalence for GG3+, even a test set of 2,417 cores contains only 92 positives. Bootstrap confidence intervals are correspondingly wide (PR-AUC CI widths of 0.3–0.4 for GG3+). Performance estimates should be treated as indicative, not definitive.

7. **MRI acquisition heterogeneity.** The dataset likely contains images from multiple MRI scanners, field strengths, and sequences. Target mesh quality and registration accuracy may vary across acquisitions in ways not captured in the available metadata.

8. **Clinical variables replicated at core level.** PSA, prostate volume, and PSA density are measured once per patient and replicated to all cores in the analysis. These features are not independent core-level observations. The cluster-robust analysis accounts for this at the inference level, but their coefficients in a joint multi-feature model are subject to collinearity (particularly PSA, log-PSA, and PSA density, which share strong Spearman correlations ≥ 0.78).

9. **No claim of calibrated clinical risk without further validation.** Raw probabilities are not well calibrated. After Platt recalibration on the validation set, the GG3+ model produces scores that concentrate in the low-probability range (slope 1.511). Absolute probability estimates should not be used as clinical risk scores without recalibration on a prospective or external cohort.

10. **Target geometry may partly reflect radiologist targeting, not purely lesion biology.** The MRI-defined target captures the radiologist's assessment of where a lesion is. If targets are biased toward certain lesion types (large, PI-RADS 4–5, specific zones), the geometric signal may partly reflect this selection rather than the geometry of any biopsy reaching any lesion. The model should be interpreted as learning to predict whether a core sampled a lesion as defined and targeted in this cohort, not whether any cancer is present at that core location.

---

## 11. Remaining Analyses Before Manuscript

### A — Essential before writing

| Analysis | Reason |
|---|---|
| **Final figure generation** (ROC/PR curves, feature-set barplot, enrichment plot, forest plot of ORs) | Figures are needed to complete the paper and currently do not exist as standalone files. This is the primary remaining concrete step. |
| **Confirm delta table numbers** against `minimal_vs_full_delta_table.md` for Table 3 in final format | Numbers are referenced in multiple places; a single clean source table is needed to avoid inconsistencies. |
| **Calibration curve plots** (reliability diagrams before/after Platt) | Needed to support the calibration caveat claim and for the supplement. |

### B — Useful but not essential

| Analysis | Reason |
|---|---|
| Feature importance figure (permutation importance, compact model, both endpoints) | Strengthens interpretability claims; useful for supplement but not critical path. |
| Full sensitivity table with GEE coefficients alongside cluster-robust (combined Table 5) | GEE results are in CSV; combining into one clean table for the main paper is valuable but straightforward. |
| PSA density non-linearity check (spline or binned regression) | Mentioned in Recommended Next Steps; would add to Discussion but is not required for current claims. |
| Cross-validation on full dataset | Would give more robust performance estimates; however, the current train/val/test split is pre-established and changing it at this stage would require rewriting many results sections. Lower priority for this paper. |

### C — Likely not worth doing now

| Analysis | Reason |
|---|---|
| Deep learning / CNN baselines on raw MRI patches | A different methodological paper; would require substantial infrastructure beyond current scope and would dilute the geometry framing. |
| Survival or time-to-progression modelling | No follow-up data available in the TCIA dataset. |
| Multi-centre pooling or federated learning | No access to additional cohorts at this stage. |
| SHAP or Shapley value analysis on tree models | Permutation importance is already computed; SHAP would be supplementary at best and adds engineering overhead. |
| Target mesh quality score as a feature | Interesting but requires additional mesh QC infrastructure not currently in the pipeline. |

---

## 12. Recommended Next Concrete Step

**Generate the five main figures.** All model results, risk stratification numbers, and inference ORs exist in CSV and Markdown reports. The only thing missing is figure code.

Priority order:

1. **Figure 1 (geometry schematic):** Draw manually or with a simple diagram script (matplotlib + anatomical patches). This is the most important explanatory figure in the paper.
2. **Figure 2 (ROC + PR curves):** Requires accessing stored test-set probability arrays or rerunning inference (one forward pass, no retraining).
3. **Figure 3 (feature-set barplot):** Numbers are in `minimal_vs_full_model_comparison.md`. A 10-line plotting script suffices.
4. **Figure 4 (enrichment/capture-rate plot):** Numbers are in `minimal_vs_full_risk_stratification.md`.
5. **Figure 5 (forest plot of cluster-robust ORs):** Numbers are in `grouped_core_level_inference_cluster_robust.csv`.

No new modelling is needed. All five figures can be generated from existing CSV/Markdown outputs without re-running any experiment.
