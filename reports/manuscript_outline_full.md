# Full Manuscript Outline

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-07-06  
**Status:** Outline draft. All numbers from committed reports. No new analysis.

---

## 1. Candidate titles

1. **"Target-relative needle geometry predicts biopsy core positivity in MRI-guided prostate biopsy: a compact interpretable model"**
2. **"Distance from needle trajectory to MRI-defined target lesion is a cluster-robust predictor of core-level prostate cancer grade in fusion biopsy"**
3. **"Core-level risk stratification using target-relative biopsy geometry and clinical features in MRI-US fusion prostate biopsy"**
4. **"Spatial needle-to-target geometry improves biopsy core positivity prediction beyond clinical features alone in MRI-targeted prostate biopsy"**
5. **"Interpretable compact modelling of target-relative biopsy geometry for core-level prostate cancer prediction: a grouped inference and risk stratification study"**

*Notes:* Title 1 (from `reports/manuscript_methods_results_draft.md`) is the most concise. Title 2 foregrounds the cluster-robust inference result. Title 3 emphasises risk stratification. Title 4 emphasises the incremental value framing. Title 5 is complete but long. Recommend shortlisting Title 1 or Title 3; finalise after target journal is decided.

---

## 2. One-sentence contribution

We show that four target-relative geometric features — derived from the spatial relationship between each biopsy needle trajectory and the MRI-defined target mesh — combined with four standard clinical features, form a compact eight-feature logistic regression model that achieves held-out ROC-AUC of 0.75 (GG2+) and 0.83 (GG3+) at the biopsy core level, performs within practical equivalence margins of a larger full-geometry model, and yields cluster-robust associations with core positivity that remain significant after accounting for intra-patient correlation, without making claims about calibrated absolute risk or clinical deployment readiness.

---

## 3. Provisional abstract

**Background.** In MRI-ultrasound fusion prostate biopsy, the spatial relationship between each needle trajectory and the MRI-defined target lesion varies considerably across cores and patients. Whether this target-relative geometry, quantified as interpretable scalar features, can predict biopsy core positivity for clinically significant and high-grade prostate cancer — beyond standard clinical variables — has not been systematically characterised at the core level with appropriate patient-clustering correction.

**Methods.** We analysed 16,992 coordinate-matched biopsy cores from 796 patients in the publicly available TCIA Prostate-MRI-US-Biopsy dataset, using patient-disjoint train/validation/test splits. Two binary endpoints were defined: GG2+ (ISUP Grade Group ≥ 2) and GG3+ (ISUP Grade Group ≥ 3). A compact feature set combining four target-relative geometry features (distance to target surface, distance to target centroid, fraction of centreline inside target, and trajectory-target intersection) with four clinical features (PSA, log-PSA, prostate volume, PSA density) was evaluated using logistic regression as the central model. Held-out predictive performance was assessed on a 2,417-core / 120-patient test set with patient-level bootstrap confidence intervals. A compact-versus-full model comparison used pre-specified practical equivalence margins. Core-level risk stratification was assessed by ranking test-set cores by predicted probability. Cluster-robust logistic regression and generalised estimating equations were used to assess associations after accounting for intra-patient core correlation.

**Results.** The compact model achieved ROC-AUC 0.754 (95% CI 0.699–0.807) for GG2+ and 0.828 (0.751–0.892) for GG3+, with PR-AUC of 0.327 and 0.269 respectively (PR-lift 3.1× and 7.1×). Performance was within practical equivalence margins of the full 14-feature model. The top 5% of scored cores contained 50.4% GG2+ positives (4.7× baseline) and 31.4% GG3+ positives (8.3× baseline). Cluster-robust analysis showed that distance to target surface remained associated with core positivity for both endpoints in multivariable patient-cluster-robust models (GG2+: OR 0.468, 95% CI 0.341–0.644, p < 0.001; GG3+: OR 0.323, 0.210–0.498, p < 0.001). Direct patient-level modelling achieved substantially lower ROC-AUC than core-level prediction (GG2+: 0.613 vs. 0.754; GG3+: 0.706 vs. 0.828).

**Conclusions.** Target-relative biopsy geometry provides interpretable, cluster-robust predictive signal for core-level cancer grade, with a compact eight-feature model capturing most of the discriminative performance of a larger feature set. Predicted probabilities should not be treated as calibrated absolute risk without external recalibration and validation.

---

## 4. Introduction skeleton

---

### Paragraph 1 — Prostate MRI-targeted biopsy and remaining uncertainty at the core level

**Purpose:** Establish the clinical context (MRI fusion biopsy) and motivate the problem (variable biopsy yield).

**Draft paragraph:**
Prostate cancer remains one of the most commonly diagnosed malignancies in men worldwide [CITATION NEEDED]. Multiparametric MRI has transformed prostate biopsy practice by enabling fusion-guided targeted sampling of suspicious lesions identified on imaging [CITATION NEEDED]. MRI-ultrasound (MRI-US) fusion biopsy improves the detection rate of clinically significant prostate cancer (ISUP Grade Group ≥ 2) compared with systematic biopsy in several settings [CITATION NEEDED], but biopsy yield remains variable across cores and patients [CITATION NEEDED]. Not every needle that targets a lesion successfully samples it: registration error, prostate motion, deformable fusion inaccuracies, and the three-dimensional geometry of the needle trajectory relative to a lesion's volume all contribute to this variability [CITATION NEEDED]. Understanding which individual biopsy cores are likely to be positive — and why — remains an open question at the core level.

**Notes / citations needed:**
- Prostate cancer incidence statistics (Siegel et al. / Ferlay et al.)
- MRI fusion biopsy overview (Siddiqui et al. 2015, Kasivisvanathan et al. 2018 / PRECISION trial)
- Variability in biopsy yield / missed lesions (Kasivisvanathan et al., Elkhoury et al.)
- MRI-US registration error literature

---

### Paragraph 2 — Why the spatial relationship between needle and target should carry information

**Purpose:** Motivate the geometric feature representation as a first-principles physical argument.

**Draft paragraph:**
The probability that a biopsy core samples a target lesion is governed, at least in part, by the three-dimensional spatial relationship between the needle trajectory and the lesion volume. A core whose needle midpoint lies within the target mesh boundary, or whose centreline traverses the target volume, should — all else equal — be more likely to contain cancer cells from that lesion than a core whose trajectory misses the target entirely. This spatial reasoning motivates representing each core by explicit geometric features derived from registered needle-to-target coordinates: the distance from the needle midpoint to the target surface, the distance to the target centroid, the fraction of the centreline inside the target volume, and a binary indicator of trajectory-target intersection. These features have direct physical units (millimetres, fractions), interpretable sign directions (closer to surface → higher cancer probability), and can be derived from any MRI-US fusion biopsy system that records both needle trajectories and MRI target mesh coordinates without requiring image patch extraction or learned representations.

**Notes / citations needed:**
- MRI-US registration and coordinate systems (CITATION NEEDED — registration pipeline paper)
- Concept of needle-target miss / sampling geometry (CITATION NEEDED)

---

### Paragraph 3 — Gap in the literature

**Purpose:** Identify what is missing in current work: most modelling uses images or patient-level risk, less on interpretable target-relative biopsy geometry at core level.

**Draft paragraph:**
Existing computational approaches to prostate biopsy outcome prediction fall into two broad categories. Image-based models — including convolutional neural networks applied to MRI patches, radiomics pipelines, or multi-instance learning frameworks — aim to extract predictive features from the MRI signal itself [CITATION NEEDED]. Patient-level clinical models use PSA, prostate volume, and biopsy history to estimate the probability that a patient will be found to have significant cancer on any biopsy [CITATION NEEDED]. Both categories share a common limitation: they do not explicitly model the geometric relationship between where each needle went and where the MRI-identified lesion was located. Core-level geometry-based approaches exist in concept [CITATION NEEDED] but have rarely been formalised into compact, interpretable predictive models with appropriate statistical correction for intra-patient correlation. To our knowledge, no published study has (i) constructed an interpretable geometric feature set from needle-to-target spatial coordinates, (ii) evaluated it against clinical-only and full-geometry reference sets on a held-out test split, and (iii) tested the resulting associations with cluster-robust inference accounting for the non-independence of multiple cores per patient. TODO: verify this novelty claim after the literature search.

**Notes / citations needed:**
- MRI radiomics for prostate cancer (Steuber et al., Litjens et al., CITATION NEEDED)
- Patient-level risk models (ERSPC risk calculator, Prostate Biopsy Collaborative Group, CITATION NEEDED)
- Core-level geometry studies if any (CITATION NEEDED — to be searched)

---

### Paragraph 4 — Study contribution

**Purpose:** State what this paper does, in concrete terms.

**Draft paragraph:**
In this study, we derive and evaluate a compact eight-feature model combining four target-relative geometry features and four clinical features (`target_geometry_plus_clinical`) for core-level prostate cancer prediction in MRI-US fusion biopsy. Using 16,992 coordinate-matched biopsy cores from 796 patients in the publicly available TCIA Prostate-MRI-US-Biopsy dataset, with patient-disjoint splits, we (i) evaluate held-out predictive performance against clinical-only and full-geometry reference feature sets, (ii) assess core-level risk stratification by ranking held-out test cores by predicted probability, (iii) test whether the geometric associations are robust to intra-patient core clustering using cluster-robust logistic regression and generalised estimating equations, and (iv) report patient-level modelling performance as a secondary analysis to situate core-level results in context. The compact model is compared against a 14-feature full-geometry reference set using pre-specified practical equivalence margins, with the full model retained as a sensitivity analysis. Throughout, we distinguish held-out predictive performance from calibrated absolute risk, and emphasise risk stratification and ranking rather than clinical decision-making.

**Notes:** No citations needed here beyond what is introduced in §Introduction.

---

### Paragraph 5 — Summary of aims

**Purpose:** State the four aims of the study explicitly for the reader.

**Draft paragraph:**
The specific aims of this study were: (1) to evaluate whether target-relative biopsy geometry features, combined with standard clinical variables, improve core-level cancer prediction beyond clinical features alone in a held-out test split; (2) to determine whether a compact eight-feature model captures most of the predictive signal of a broader fourteen-feature model; (3) to assess the risk-stratification performance of the compact model — the enrichment of cancer-positive cores among the highest-scored biopsy cores; and (4) to test whether core-level geometric associations with cancer grade are robust to intra-patient correlation using cluster-robust and GEE inference. A secondary aim was to characterise the limits of patient-level prediction achieved by aggregating or extending core-level models.

---

## 5. Methods section map

Full methods prose is in `reports/manuscript_methods_results_draft.md`. The table below maps each subsection to its corresponding figure or table.

| Methods subsection | Key content | Figure / table |
|--------------------|-------------|----------------|
| §2.1 Dataset and study unit | TCIA dataset; 16,992 coord-matched cores; 796 patients; patient-disjoint splits; two endpoints (GG2+, GG3+) | Table 1 |
| §2.2 Target-relative geometry features | Four geometry features defined; needle trajectory and MRI target mesh; mesh-availability audit | Figure 1 |
| §2.3 Feature sets | Compact (8) vs full (14); clinical features; target_mesh_available exclusion rationale | Tables 2–3 |
| §2.4 Predictive modelling and evaluation | Logistic regression (central); HGB/XGB (sensitivity); ROC/PR-AUC; bootstrap CIs; Youden-J threshold; PR-lift definition | Figure 2, Table 2 |
| §2.4 (cont.) Risk stratification | Ranking by predicted probability; top 5/10/20% prevalence and capture rate | Figure 3, Table 4 |
| §2.5 Compact vs full comparison | Δ = full − compact; |ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015 pragmatic margins; not a formal equivalence test | Table 3 |
| §2.6 Grouped core-level inference | Cluster-robust GLM (statsmodels, cov_type='cluster'); GEE sensitivity; all splits pooled; continuous standardised; trajectory binary; association ≠ prediction | Figure 4, Table 5 |
| §2.7 Patient-level secondary | Naive aggregation (mean prob) vs direct patient-level models; aggregated feature summaries; n_cores excluded | Table 6 |
| §2.8 Calibration | Brier score, BSS, ECE; Platt scaling; ranking device, not absolute risk | Supplementary / §3.7 in Results |

---

## 6. Results section map

Full results prose is in `reports/manuscript_methods_results_draft.md`. Each subsection below gives the core claim, supporting numbers, and a caution.

---

### §3.1 Dataset and endpoint prevalence → Table 1

**Core claim:** The dataset is moderately sized with substantial class imbalance, particularly for GG3+.

**Supporting numbers:** 16,992 cores; 796 patients; GG2+ prevalence 10.7% test / 11.1% overall; GG3+ prevalence 3.8% test / 3.8% overall; 557/119/120 patients per split.

**Caution:** Core-count is weakly associated with GG2+ patient positivity (ρ = 0.101), which may mildly favour max-prob and top-k aggregation methods at patient level; no association for GG3+.

---

### §3.2 Compact model held-out performance → Figure 2, Table 2

**Core claim:** The compact `target_geometry_plus_clinical` logistic regression model achieves strong discriminative performance on the held-out test set for both endpoints.

**Supporting numbers:** GG2+: ROC-AUC 0.754 [0.699–0.807], PR-AUC 0.327 [0.230–0.461], PR-lift 3.1×. GG3+: ROC-AUC 0.828 [0.751–0.892], PR-AUC 0.269 [0.094–0.516], PR-lift 7.1×. Clinical-only baselines: GG2+ 0.650, GG3+ 0.716. Target-geometry-only baselines: GG2+ 0.730, GG3+ 0.806.

**Caution:** GG3+ PR-AUC CI is wide (range 0.42) due to only 92 positive cores in the test set. Performance claims for GG3+ should be interpreted as indicative. No external validation.

---

### §3.3 Compact model achieves equivalent held-out performance to the full feature set → Table 3

**Core claim:** The 8-feature compact model performs within pre-specified practical margins of the 14-feature full model; the non-target biopsy/prostate geometry features add negligible value.

**Supporting numbers:** Mean Δ (full − compact): GG2+ ΔROC = −0.002, ΔPR = 0.000; GG3+ ΔROC = −0.005, ΔPR = +0.008. All |Δ| within margins (|ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015). Negative-control ablation (non-target geometry + clinical): GG2+ mean ROC-AUC 0.640, GG3+ 0.726.

**Caution:** These are pragmatic margins, not a formal statistical equivalence test. The full model is retained as a sensitivity/reference analysis.

---

### §3.4 Risk stratification among highest-scored cores → Figure 3, Table 4

**Core claim:** Ranking cores by predicted probability concentrates cancer-positive cores in the highest-scored deciles, with substantial enrichment over baseline.

**Supporting numbers:** GG2+: top-5% prevalence 50.4% (4.7× baseline 10.7%), top-10% captures 32.0% of all test-set positives. GG3+: top-5% prevalence 31.4% (8.3× baseline 3.8%), top-10% captures 50.0% of all high-grade test-set positives.

**Caution:** These are core-level ranking metrics on the held-out test set. They characterise model-guided ranking, not calibrated absolute risk. No threshold-based clinical decisions should be drawn.

---

### §3.5 Cluster-robust association analysis → Figure 4, Table 5

**Core claim:** The geometric associations survive patient-clustering correction; distance to target surface and fraction inside target are the most robust geometry predictors.

**Supporting numbers:** Distance to target surface: cluster-robust OR 0.468 (95% CI 0.341–0.644, p < 0.001) GG2+; OR 0.323 (0.210–0.498, p < 0.001) GG3+. GEE concordant: OR 0.581 GG2+, OR 0.441 GG3+. Fraction inside target: OR 1.331 (p < 0.001) GG2+; OR 1.227 (p = 0.006) GG3+. log(PSA): OR 1.818 GG2+, 2.710 GG3+. Prostate volume: OR 0.681 GG2+, 0.698 GG3+. Trajectory intersects target: p = 0.58 GG2+, p = 0.61 GG3+ (redundant with fraction inside, ρ = 0.979).

**Caution:** Association analysis pooling all splits; not a held-out predictive performance result. Collinear predictors (distance features: ρ = 0.942, VIF > 14; PSA group: ρ ≥ 0.78) cannot be ranked by individual coefficient magnitude.

---

### §3.6 Patient-level secondary analysis → Table 6

**Core claim:** Patient-level prediction is substantially harder than core-level prediction; direct patient-level models offer only marginal improvement over naive score aggregation.

**Supporting numbers:** GG2+: best direct model ROC-AUC 0.613 vs naive 0.617 (ΔROC = −0.004). GG3+: best direct model ROC-AUC 0.706 vs naive 0.682 (ΔROC = +0.024, ΔPR = −0.014). Core-level gap: ~0.14 ROC-AUC below compact model for both endpoints.

**Caution:** Patient-level prevalence is high (GG2+: 58.3%; GG3+: 25.8%), compressing discrimination. Small test cohort (120 patients) makes model selection unstable (best model differs by endpoint). Secondary analysis only.

---

### §3.7 Calibration summary → Supplementary

**Core claim:** Raw probabilities are poorly calibrated; Platt scaling partially corrects this, but GG3+ scores remain conservative post-recalibration.

**Supporting numbers:** BSS < 0 for all raw models. Post-Platt: BSS 0.108 (GG2+ LR), 0.084 (GG3+ LR); ECE < 0.03 for both. GG3+ recalibration slope = 1.511 (scores concentrated at low probabilities).

**Caution:** No external recalibration performed. Predicted probabilities should not be used as clinical risk scores without recalibration on a prospective or external cohort. Full calibration results in supplementary.

---

## 7. Discussion skeleton

---

### §D1 — Main finding: target-relative geometry is informative at the core level

**Draft paragraph:** The principal finding of this study is that the spatial relationship between a biopsy needle trajectory and the MRI-defined target lesion — specifically the distance from the needle midpoint to the target surface and the fraction of the centreline inside the target — remains significantly associated with biopsy core positivity in multivariable patient-cluster-robust models for both clinically significant (GG2+) and high-grade (GG3+) prostate cancer after accounting for intra-patient core correlation. Combined with four standard clinical features, these four geometric predictors form a compact model that achieves ROC-AUC of 0.754 for GG2+ and 0.828 for GG3+ on a held-out test set, substantially above clinical-only baselines and consistent across logistic regression and gradient boosting model families.

**Key evidence:** Held-out ROC-AUC from Table 2; cluster-robust ORs from Table 5; GEE concordance from grouped inference report.

**Limitations / caveats:** Single-cohort results; test set of 120 patients; external validation not performed.

---

### §D2 — Interpretation of the two dominant geometry features

**Draft paragraph:** Two of the four target-relative geometry features account for the majority of the geometric signal. Distance from the needle midpoint to the target surface (OR < 1 for both endpoints; stronger effect for GG3+) quantifies how closely the needle reached the lesion boundary: cores farther outside the target are less likely to contain cancer from the targeted lesion. Fraction of centreline inside the target (OR > 1) captures the length of the needle that traversed the target volume. Together, these two features operationalise biopsy targeting accuracy at the individual core level. Distance to target centroid, while correlated with distance to surface (Spearman ρ = 0.942), did not reach significance once surface distance was included, suggesting the two features carry largely redundant information despite their different geometric definitions. The binary trajectory-intersection indicator is near-perfectly rank-correlated with fraction inside (ρ = 0.979) and was also non-significant in multivariable analysis; this feature is retained in the compact set for completeness but its coefficient should not be interpreted as a separable contribution beyond fraction inside.

**Key evidence:** Table 5 ORs; VIF and Spearman correlations from grouped_core_level_inference.md §5.

**Limitations / caveats:** Collinearity prevents ranking individual feature importance from coefficients.

---

### §D3 — Why the compact model is preferable

**Draft paragraph:** The compact eight-feature model performed within pragmatic equivalence margins of the full 14-feature model across both endpoints and all three model families evaluated (mean Δ ROC ≤ 0.005 in magnitude; mean Δ PR ≤ 0.008). The non-target biopsy and prostate geometry features added to the full model — prostate surface distance, core length, tube voxel count — contributed noise rather than signal when target-relative features were already present. This is consistent with the hypothesis that the target lesion location, not the overall prostate geometry, is the primary spatial determinant of whether any given core samples cancer. A negative-control ablation restricted to non-target biopsy geometry plus clinical features performed near the clinical-only level (GG2+: 0.640; GG3+: 0.726), confirming this interpretation. The compact model is preferred for practical reasons: it requires fewer features, is fully auditable, and its coefficients have direct physical interpretations.

**Key evidence:** Table 3 delta comparison; minimal_vs_full_model_comparison.md interpretation section.

**Limitations / caveats:** Pragmatic margins, not a formal equivalence test; full model retained as sensitivity reference.

---

### §D4 — Core-level versus patient-level analytical unit

**Draft paragraph:** Patient-level prediction — whether a patient has cancer — is structurally harder than core-level prediction in this dataset. Patient-level prevalence is 58.3% for GG2+ and 25.8% for GG3+, substantially compressing the discrimination problem compared to core-level prevalences of 10.7% and 3.8%. Direct patient-level models and naive score aggregation both achieved ROC-AUC approximately 0.14 below core-level models for each endpoint. This gap is expected: the core-specific geometric signal is attenuated when averaged across all 15–20 cores per patient, many of which are off-target. The core is the natural unit of analysis for this geometric model. Patient-level prediction from core-level geometry summaries is a secondary analysis; it is not the primary clinical question addressed by this study, and it should not be used to infer that the model is adequate or inadequate for patient-level triage.

**Key evidence:** Table 6; patient_level_direct_model_comparison.md §6.

**Limitations / caveats:** Small patient-level test set (120 patients); model selection is unstable across endpoints.

---

### §D5 — Interpretability and the radiologist targeting circularity

**Draft paragraph:** An eight-feature logistic regression model with standardised continuous coefficients and interpretable units (millimetres, fractions) offers a level of transparency not achievable with high-dimensional image-based classifiers. The direction and approximate magnitude of each association can be communicated directly: a one-standard-deviation increase in distance to the target surface corresponds to lower odds of positivity (OR 0.468 for GG2+ and OR 0.323 for GG3+), meaning cores farther from the target surface are proportionally less likely to be positive. Conversely, moving closer to the target surface corresponds to the reciprocal increase in odds. This interpretability is practically valuable for procedure audit and quality assurance. However, an important limitation must be stated: the MRI-defined target mesh is drawn by the same radiologist who planned the biopsy. Cores close to the target are more likely to be positive partly because the radiologist identified that region as suspicious on MRI. The geometric signal therefore reflects the radiologist's targeting accuracy and the spatial correspondence between the MRI-visible lesion and the histological finding, not purely the geometry of any biopsy reaching any tissue location. This circularity limits the interpretation of these features as purely post-hoc spatial descriptors free from the clinical targeting process.

**Key evidence:** Table 5; discussion in paper_skeleton_and_results_narrative.md §10.

**Limitations / caveats:** Circularity inherent to the study design; cannot be fully corrected without a randomised targeting study.

---

### §D6 — Calibration and limits of the risk stratification framing

**Draft paragraph:** Raw predicted probabilities from the compact logistic regression model were poorly calibrated: Brier skill scores were negative for all models before recalibration, indicating that uncorrected probabilities performed worse than a prevalence-only predictor. After Platt scaling on the validation set, calibration improved substantially (BSS 0.108 for GG2+, 0.084 for GG3+; ECE < 0.03). Nevertheless, the GG3+ post-recalibration calibration slope of 1.511 indicates that predicted scores are compressed toward the low end of the probability scale; the model should be treated as a ranking instrument rather than an absolute risk estimator. In particular, the enrichment results (e.g., 50.4% positivity in the top 5% of GG2+ scored cores) characterise the model's discrimination and ranking ability, not the absolute probability that any given core is positive. These results should not be used for threshold-based clinical decision-making without recalibration on an external prospective cohort.

**Key evidence:** final_calibration_table.md; §3.7 in manuscript_methods_results_draft.md.

**Limitations / caveats:** No external recalibration; calibration may degrade in different acquisition settings.

---

### §D7 — Relation to deep learning and radiomics approaches

**Draft paragraph:** The approach presented here is deliberately distinct from image-based deep learning and radiomics pipelines that extract features from MRI intensity, texture, or patch content. The features used in this study are geometric and clinical scalars derived from registered coordinates; they require no convolutional network, no GPU infrastructure, and no image patch extraction. This has practical advantages: the features are less directly dependent on MRI intensity protocol than radiomics or image-patch models, provided target meshes and needle trajectories are available in a reliable common coordinate frame, the model is fully inspectable, and the bootstrap CIs on a 120-patient test set are already at the limits of what a single-cohort study can reliably estimate. Attempting to maximise ROC-AUC by adding high-dimensional MRI features on the same cohort would risk overfitting and would produce performance claims whose CIs would overlap substantially with the current results. The appropriate next step is external validation, not architectural complexity. Future work could investigate whether MRI-derived lesion characteristics (PI-RADS score, lesion volume, T2 heterogeneity) add value over and above the geometric model on a larger, multi-institutional cohort.

**Key evidence:** §3 of paper_skeleton_and_results_narrative.md; bootstrap CI width discussion.

**Limitations / caveats:** No CNN/GNN performance numbers in the current phase; not a comparative deep learning study.

---

### §D8 — Limitations (systematic list)

**Draft paragraph:** This study has several limitations that bound the interpretation of our results. First, all data come from a single publicly available retrospective cohort; selection and referral biases are unknown. Second, no external validation has been performed; generalisation to other MRI-US fusion systems, scanners, targeting protocols, or reader populations is uncertain. Third, MRI-US registration introduces geometric noise, which attenuates the measurable geometric signal; reported associations are likely lower bounds of the true relationship. Fourth, histological labels are assigned to the entire core, not to a specific spatial location; Gleason grading has known inter-rater variability. Fifth, the effective sample size for inferential purposes is closer to 796 patients than to 16,992 cores. Sixth, GG3+ results have wide confidence intervals (PR-AUC CI range 0.42) due to only 92 positive cores in the test set, and should be treated as indicative. Seventh, PSA-related features are patient-level values replicated to all cores; their joint model coefficients are subject to collinearity and should not be used to rank clinical feature importance. Finally, the MRI-defined target mesh is determined by the same radiologist who guided the biopsy, introducing circularity into the geometric signal.

**Key evidence:** paper_skeleton_and_results_narrative.md §10 (10 limitations).

---

### §D9 — Future work

**Draft paragraph:** The most important next step is external validation on an independent MRI-guided biopsy cohort, ideally using a different scanner, targeting system, or centre, to assess whether the geometric associations and held-out performance generalise. Prospective validation — in which the model's predictions are computed for each core at the time of biopsy and compared to subsequent histology — would additionally address the retrospective design limitation. Within the current dataset, incorporating prior biopsy positivity history (7,649 cores targeting prior positive lesions are available but not currently used as features) may substantially improve discrimination for repeat biopsy patients. Investigating PSA density non-linearity and potential interaction effects between geometry and PSA density could refine the clinical covariate contribution. Integration with MRI-derived features (PI-RADS score, lesion volume, radiomics) should be explored in a larger multi-institutional cohort where feature-model combinations can be properly evaluated without overfitting. A clinical utility study using decision curve analysis would be needed to assess whether the model's risk stratification adds value over clinician judgment in the context of actual biopsy triage decisions.

---

## 8. Figure and table placement plan

| Manuscript section | Figure / table | Purpose | Introductory sentence |
|--------------------|---------------|---------|----------------------|
| Methods §2.2 | **Figure 1** — Geometry schematic | Illustrate the four target-relative features and their spatial definitions | "Figure 1 illustrates the spatial relationship between a biopsy needle trajectory and the MRI-defined target mesh, and the four derived scalar features used in this study." |
| Results §3.2 | **Figure 2** — Compact model performance bars | Show ROC-AUC and PR-AUC with bootstrap CIs for both endpoints | "Figure 2 shows the held-out ROC-AUC and PR-AUC for the compact logistic regression model on the test split." |
| Results §3.4 | **Figure 3** — Risk stratification enrichment | Show positive-core prevalence in top-ranked groups | "Figure 3 shows the enrichment of positive biopsy cores among the top 5%, 10%, and 20% of test-set cores ranked by predicted probability." |
| Results §3.5 | **Figure 4** — Forest plot (cluster-robust ORs) | Show OR and CI for all eight compact features, both endpoints | "Figure 4 presents the cluster-robust odds ratios and 95% confidence intervals for all eight compact model features, for both endpoints." |
| Methods §2.1 / Results §3.1 | **Table 1** — Dataset overview | Establish sample size, splits, prevalence, endpoints | "Table 1 summarises the dataset splits, patient and core counts, and endpoint prevalences." |
| Results §3.2 | **Table 2** — Compact model performance | Report ROC/PR-AUC, CIs, PR-lift, sensitivity/specificity/F1 | "Table 2 reports the held-out predictive performance of the compact logistic regression model for both endpoints." |
| Results §3.3 | **Table 3** — Compact vs full comparison | Justify compact model selection | "Table 3 compares the compact and full feature-set models across model families and endpoints." |
| Results §3.4 | **Table 4** — Risk stratification | Positive-core prevalence and capture rate by decile | "Table 4 reports the positive-core prevalence and capture rate in the top-ranked biopsy core groups." |
| Results §3.5 | **Table 5** — Cluster-robust odds ratios | Report association model results | "Table 5 presents the cluster-robust odds ratios for all eight compact features, for GG2+ and GG3+ endpoints." |
| Results §3.6 | **Table 6** — Patient-level secondary | Situate patient-level performance relative to core-level | "Table 6 compares patient-level direct modelling and naive score aggregation with the core-level reference." |

---

## 9. Main claims and allowed wording

### A. Claims supported by current evidence

- The compact `target_geometry_plus_clinical` logistic regression model achieves ROC-AUC of 0.754 [0.699–0.807] (GG2+) and 0.828 [0.751–0.892] (GG3+) on the held-out test split.
- The compact model performs within pragmatic equivalence margins of the full 14-feature model (|ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015 for both endpoints, averaged across model families).
- Distance from needle midpoint to target surface remains significantly associated with core positivity in multivariable patient-cluster-robust models (GG2+: OR 0.468, p < 0.001; GG3+: OR 0.323, p < 0.001).
- The top 5% of test-set cores ranked by predicted probability have a positive rate of 50.4% (GG2+; 4.7× baseline) and 31.4% (GG3+; 8.3× baseline).
- Non-target biopsy/prostate geometry features add negligible predictive value over target-relative geometry.
- Patient-level prediction is substantially below core-level prediction for both endpoints (~0.14 ROC-AUC gap).
- Raw predicted probabilities are not well calibrated before Platt scaling; post-recalibration ECE is below 0.03.

### B. Claims not supported and must be avoided

- Do not claim clinical readiness, deployment suitability, or that the model should guide biopsy decisions.
- Do not claim calibrated absolute risk or that predicted probabilities represent a patient's probability of cancer without external recalibration.
- Do not claim that core-level observations are independent: intra-patient correlation is a structural feature of the dataset.
- Do not claim that target-relative geometry is independent of radiologist targeting: the circularity limitation applies.
- Do not frame the paper as a deep learning or CNN/GNN study.
- Do not overstate the equivalence result: "within pragmatic margins" is the correct phrase, not "statistically equivalent."
- Do not claim that the model generalises to other datasets, biopsy systems, or patient populations without external validation.
- Do not use the word "efficacy" to describe model performance; use "held-out predictive performance."
- Do not claim the GG3+ PR-AUC result is robust: the CI is wide (0.094–0.516) due to 92 test-set positives.

---

## 10. Remaining manuscript tasks

### A. Essential before first full draft

1. **Gather literature citations** for Introduction §§P1–P3 and Discussion §§D2–D3, D7–D8. At minimum: MRI fusion biopsy trials/reviews, patient-level risk models, and a search for existing core-level geometry papers. Mark all CITATION NEEDED placeholders.
2. **Decide target journal and format.** Journal choice determines word limits, abstract structure, number of main figures/tables, and supplementary policy. This should be decided before reformatting any tables or sections.
3. **Convert tables to journal style.** Tables 1–6 in `reports/main_paper_tables.md` are Markdown drafts; they need to be reformatted to match the journal's LaTeX/Word table style and numbering convention.
4. **Verify all numbers against source reports** before submission. A consistency pass reading Table 1–6 values against the source CSVs and Markdown reports is required. The TODO list in `reports/main_paper_tables.md` §"Consistency checks" identifies known items.
5. **Write the Limitations section** carefully from the 10-point list in `reports/paper_skeleton_and_results_narrative.md` §10 and Discussion §D8 above. Limitations require precise wording and should be finalised before journal submission.

### B. Useful before supervisor review

- **Add GEE coefficient table** as a supplementary table alongside cluster-robust results (data are in `reports/grouped_core_level_inference.md` §4).
- **Supplementary delta plot** (Supp Fig S1): compact vs full ΔROC/ΔPR per model family and endpoint; data in `reports/minimal_vs_full_delta_table.md`; script not yet written.
- **Supplementary calibration plot** (Supp Fig S2): reliability diagrams before/after Platt scaling; data in `reports/final_calibration_table.md`; script not yet written.
- **Supplementary patient-level figure** (Supp Fig S3): bar or dot plot of core-level vs patient-level ROC-AUC; data in `reports/patient_level_direct_model_comparison.md`.
- **Brief CNN/GNN historical context note** for Discussion §D7: if prior image-based results from the earlier project phase are available, a one-paragraph summary situating them as a different methodological question.

### C. Later / not now

- New model training of any kind (CNN, GNN, transformer, multi-instance learning).
- Radiomics feature extraction from MRI patches.
- PSA density non-linearity analysis (spline/binned regression): interesting for discussion but not a blocker.
- Cross-validation on the full dataset: would require restructuring many results sections.
- Clinical decision curve analysis: relevant only if a specific clinical use case is proposed by a clinician collaborator or a reviewer.
- External validation: requires access to an independent cohort not currently available.
- SHAP or Shapley value analysis: not essential given permutation importance is already computed.
- Survival or progression modelling: no follow-up data available in the TCIA dataset.
