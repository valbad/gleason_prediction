# Manuscript Methods and Results Draft

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-07-06  
**Status:** First draft. Numbers taken verbatim from committed reports. No new analysis.

---

## 1. Working title

Three candidate titles are proposed below. All emphasise the geometric framing and avoid deployment claims.

1. **"Target-relative needle geometry predicts biopsy core positivity in MRI-guided prostate biopsy: a compact interpretable model"**
2. **"Spatial relationship between needle trajectory and MRI-defined target is associated with clinically significant and high-grade prostate cancer detection at the biopsy core level"**
3. **"Distance from needle midpoint to MRI target surface is an interpretable and patient-cluster-robust predictor of biopsy core positivity in fusion prostate biopsy"**

*Preference notes:* Title 1 is the most concise and positions the paper as a methods contribution. Title 2 is more conservative (uses "associated with"). Title 3 foregrounds the specific most significant feature, which matches the results but may be overly narrow.

---

## 2. Methods draft

### 2.1 Dataset and study unit

We analysed data from the Prostate-MRI-US-Biopsy collection of The Cancer Imaging Archive (TCIA), a publicly available dataset of MRI-ultrasound fusion-guided prostate biopsies. The primary study unit was the individual biopsy core. Cores were included if they had a confirmed coordinate match between the biopsy trajectory record and the histopathology label (`label_join_status == "coord_match"`), yielding 16,992 cores from 796 patients. Patients were assigned to training, validation, and test splits using patient-stratified random assignment (GroupShuffleSplit), ensuring that all cores from a single patient were placed in the same split (train: 557 patients / 11,827 cores; validation: 119 / 2,748; test: 120 / 2,417). No patient contributed to more than one split.

Two binary endpoints were defined. **GG2+ / clinically significant prostate cancer (csPCa):** any biopsy core with ISUP Grade Group ≥ 2 (Gleason ≥ 3+4=7). **GG3+ / high-grade prostate cancer:** any core with ISUP Grade Group ≥ 3 (Gleason ≥ 4+3=7). Core-level positive prevalences were 11.1% for GG2+ and 3.8% for GG3+ across the full dataset, and 10.7% and 3.8% respectively in the held-out test set.

### 2.2 Target-relative biopsy geometry features

Each biopsy core is characterised by a needle trajectory, represented as a line segment in registered three-dimensional coordinates. The MRI acquisition produces a target mesh: a surface enclosing the lesion region identified by the reporting radiologist and used to plan the biopsy. Four target-relative geometry features were derived from the spatial relationship between the needle trajectory and this target mesh:

1. **Distance to target surface** (`distance_midpoint_to_target_surface_mm`): signed distance from the needle midpoint to the nearest point on the target mesh surface (negative inside, positive outside).
2. **Distance to target centroid** (`distance_midpoint_to_target_centroid_mm`): Euclidean distance from the needle midpoint to the centroid of the target mesh.
3. **Fraction of centreline inside target** (`approximate_fraction_of_centerline_inside_target`): proportion of the needle centreline, sampled at regular intervals, that lies inside the target mesh volume.
4. **Trajectory intersects target** (`trajectory_intersects_target`): binary indicator (0/1) of whether any point on the needle trajectory passes through the target mesh.

These four features form the *target-relative geometry* group (Figure 1). They are conceptually distinct from generic biopsy geometry features (core length, tube volume, prostate surface distance) that do not reference the target mesh location. A small subgroup of cores lacked usable target-mesh information and was audited separately. Because this subgroup was rare and had an unusually high GG2+ positivity rate, the `target_mesh_available` indicator was not used as a predictor in the final reporting analyses. Final model comparisons therefore focus on target-relative geometry and clinical features without using mesh availability itself as a feature.

### 2.3 Clinical and reference feature sets

Four standard clinical features — total PSA (ng/mL), log-transformed PSA, prostate volume (cc), and PSA density (PSA / volume, ng/mL/cc) — were included alongside target-relative geometry. Log-PSA was included explicitly to allow the model to capture non-linear PSA effects through a linear term. These features are patient-level measurements replicated to all cores from the same patient; they are not independent core-level observations.

The **compact feature set** (`target_geometry_plus_clinical`) combines the four target-relative geometry features with the four clinical features (8 features total) and was selected as the central model after the compact-versus-full comparison, on grounds of parsimony and interpretability. A broader **full reference feature set** (`all_geometry_no_availability_plus_clinical`, 14 features) additionally includes non-target biopsy geometry features: prostate surface distance, core length, tube voxel count, and related shape descriptors. The indicator `target_mesh_available` was excluded from the final reported feature sets because mesh availability is a dataset- and preprocessing-dependent variable, and the rare missing-mesh subgroup showed abnormal positivity. Including it would risk modelling availability artefacts rather than generalisable biopsy geometry.

### 2.4 Predictive modelling and evaluation

The central predictive model was **logistic regression** with balanced class weights and z-score standardised continuous features, selected for interpretability and stable behaviour under class imbalance. **HistGradientBoosting** and **XGBoost** were evaluated as non-linear sensitivity models; results are reported for these models where relevant to confirm that logistic regression conclusions are not an artefact of linearity.

Predictive performance was evaluated on the held-out test set using two primary metrics. **ROC-AUC** (area under the receiver operating characteristic curve) measures overall discriminative ability and is threshold-independent. **PR-AUC** (area under the precision-recall curve) is additionally reported because the positive-class prevalence is low (10.7% and 3.8%), and PR-AUC is more sensitive to performance among positive cases under class imbalance than ROC-AUC. **PR-lift** (PR-AUC divided by test-set prevalence) expresses PR-AUC as a multiple of the no-skill baseline.

**95% confidence intervals** for ROC-AUC and PR-AUC were computed by patient-level bootstrap resampling of the test set (1,000 resamples; patients resampled with replacement, all cores from each resampled patient included; 2.5th and 97.5th percentiles reported). This approach preserves intra-patient correlation in the CI calculation. Sensitivity, specificity, precision, and F1 are reported at the classification threshold maximising Youden J (sensitivity + specificity − 1) on the validation set.

**Risk stratification** was assessed by ranking all test-set cores by descending predicted probability and computing the positive-class prevalence and capture rate (fraction of all positives recovered) in the top 5%, 10%, and 20% of scored cores. This framing characterises the model's ability to concentrate positives among the highest-ranked cores. Predicted probabilities are used here as a *ranking* device only; they are not treated as calibrated absolute risk estimates (see §2.8).

### 2.5 Compact versus full feature-set comparison

To justify the use of the compact feature set as the primary model, we compared its held-out performance to the full 14-feature reference set and four intermediate ablation sets across three model families (logistic regression, HistGradientBoosting, XGBoost). Differences in mean ROC-AUC (ΔROC) and mean PR-AUC (ΔPR) were computed as (full) − (compact) and compared to pragmatic equivalence margins: |ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015. Positive Δ values indicate higher performance for the full feature set; negative values indicate higher performance for the compact feature set. These thresholds were used as pragmatic margins for interpreting whether the additional non-target geometry features changed held-out performance by a practically meaningful amount. This comparison is a practical judgement about parsimony, not a formal statistical equivalence test.

### 2.6 Grouped core-level inference

Because multiple biopsy cores are collected from each patient, core-level observations are not independent. To test whether the geometric associations are robust to this intra-patient clustering, we fitted two association models using all valid splits pooled (16,992 cores, 796 patients), maximising statistical power.

**Cluster-robust logistic regression:** A logistic regression model (Binomial family, logit link) was fitted using `statsmodels.GLM` with `cov_type='cluster'` and patient number as the grouping variable. This yields asymptotically valid standard errors and p-values that account for intra-patient core correlation without imposing a parametric correlation structure.

**GEE sensitivity analysis:** A binomial Generalised Estimating Equations model with exchangeable working correlation (`statsmodels.genmod.GEE`) was fitted using the same grouping variable as a secondary confirmation.

For both analyses, continuous predictors were z-score standardised (mean zero, unit standard deviation) prior to fitting, so that odds ratios are reported per one standard deviation. `trajectory_intersects_target` is a binary indicator and was kept un-standardised; its odds ratio represents the log-odds change between non-intersecting and intersecting cores. Both analyses are *association* analyses — they assess covariate-outcome relationships in the pooled dataset — and are distinct from the held-out predictive-performance analyses in §2.4.

### 2.7 Patient-level secondary analysis

A secondary analysis assessed whether prediction at the patient level — *does this patient have cancer?* — could be achieved from core-level geometric features. A patient was labelled positive for an endpoint if at least one biopsy core from that patient was positive; patient-level prevalences are consequently much higher than core-level prevalences (GG2+: 58.3%; GG3+: 25.8% on the test split).

Two approaches were compared: (i) *naive aggregation*, in which core-level predicted probabilities from the best core-level model are aggregated to the patient level using the mean across all cores; and (ii) *direct patient-level models*, in which aggregated patient-level feature summaries (means, maxima, proportions above threshold, calculated across all cores for each patient) are combined with clinical features and used to train patient-level logistic regression, HistGradientBoosting, and XGBoost classifiers. The number of biopsy cores per patient (`n_cores`) was excluded from patient-level feature sets because core count may reflect sampling intensity or patient selection rather than anatomy or biology. Patient-level performance is reported as a secondary analysis; core-level risk stratification remains the primary result.

### 2.8 Calibration and interpretation of probabilities

Predicted probabilities from all models were assessed for calibration using the Brier score, Brier skill score (BSS = 1 − Brier / Brier_null), and expected calibration error (ECE, 10 equal-width bins). Platt scaling (logistic recalibration of the model logit on validation-split predicted probabilities) was applied as a post-hoc recalibration step. Calibration results are reported in brief in the Results and in full in the supplementary material. Throughout this paper, predicted probabilities are used as *ranking scores* for risk stratification, not as calibrated absolute risk estimates. In particular, the raw (pre-recalibration) probabilities should not be interpreted as individual cancer probabilities without recalibration on an external cohort.

---

## 3. Results draft

### 3.1 Dataset and endpoint prevalence

A total of 16,992 biopsy cores from 796 patients met the inclusion criterion (Table 1). The GG2+ endpoint had a core-level positive rate of 11.1% overall (10.7% in the test split); the GG3+ endpoint had a positive rate of 3.8% in both the overall dataset and the test split. The class imbalance, particularly for GG3+, motivates reporting PR-AUC alongside ROC-AUC throughout. A weak but statistically significant positive correlation between number of cores per patient and GG2+ positivity was observed (Spearman ρ = 0.101, p = 0.004), reflecting that patients with larger or more suspicious lesions may receive more targeted cores; no such association was present for GG3+ (ρ = −0.050, p = 0.16).

### 3.2 Compact model held-out predictive performance

On the held-out test split (2,417 cores, 120 patients), the compact `target_geometry_plus_clinical` logistic regression model achieved an ROC-AUC of 0.754 (95% CI 0.699–0.807) for GG2+ and 0.828 (0.751–0.892) for GG3+ (Table 2, Figure 2). PR-AUC was 0.327 (95% CI 0.230–0.461) for GG2+, corresponding to a PR-lift of 3.1× over the no-skill baseline, and 0.269 (0.094–0.516) for GG3+, corresponding to a PR-lift of 7.1×. The substantially wider confidence interval for GG3+ PR-AUC (range 0.42) compared with GG2+ (range 0.23) reflects the smaller number of positive cores in the GG3+ test set (n = 92 vs. n = 259).

Both endpoints substantially outperformed clinical-only baselines (GG2+: ROC-AUC 0.650; GG3+: 0.716) and target-geometry-only baselines (GG2+: 0.730; GG3+: 0.806), demonstrating complementary contributions from the geometric and clinical feature groups.

### 3.3 Compact model achieves equivalent held-out performance to the full feature set

Comparing the compact model to the 14-feature full reference model across all three model families, mean differences were small in absolute terms and within the pre-specified practical equivalence margins for both endpoints (Table 3). For GG2+, the mean ΔROC was −0.002 (Δ = full − compact; negative means compact outperforms full), with a mean ΔPR of 0.000. For GG3+, the mean ΔROC was −0.005 and the mean ΔPR was +0.008 (full model marginally above compact on PR-AUC). All values satisfied |ΔROC| ≤ 0.010 and |ΔPR| ≤ 0.015. These margins are practical benchmarks, not formal statistical equivalence criteria. The full 14-feature model is retained as a sensitivity/reference analysis.

A negative-control ablation restricted to non-target biopsy and prostate geometry combined with clinical features — but excluding all target-relative geometry features — performed near the clinical-only level (mean ROC-AUC 0.640 for GG2+; 0.726 for GG3+), confirming that target-relative geometry, rather than biopsy geometry in general, drives the predictive signal in the compact model.

### 3.4 Risk stratification among highest-scored biopsy cores

Ranking test-set cores by descending predicted probability from the compact logistic regression model, substantial enrichment for positive cores was observed at all thresholds examined (Table 4, Figure 3). For GG2+, the top 5% of scored cores contained 50.4% positive cores (4.7× the baseline prevalence of 10.7%), the top 10% contained 34.3% positives, and the top 20% captured 50.2% of all positive cores in the test set. For GG3+, enrichment was even more pronounced: the top 5% of scored cores contained 31.4% positives (8.3× the baseline prevalence of 3.8%), and the top 10% of scored cores captured 50.0% of all high-grade positive cores. These results characterise the model's utility as a ranking tool for biopsy core prioritisation; they do not imply that the model's predicted probabilities represent calibrated absolute cancer risk at those thresholds.

### 3.5 Core-level geometric associations are robust to intra-patient correlation

Cluster-robust logistic regression and GEE analyses, pooling all splits (16,992 cores, 796 patients), confirmed that the target-relative geometry associations were not attributable to repeated correlated observations within patients (Table 5, Figure 4). Across both endpoints, **distance from the needle midpoint to the target surface** was the strongest and most consistent geometry predictor: cluster-robust OR 0.468 (95% CI 0.341–0.644, p < 0.001) for GG2+ and 0.323 (0.210–0.498, p < 0.001) for GG3+. GEE estimates were concordant (GG2+: OR 0.581, p < 0.001; GG3+: OR 0.441, p < 0.001). **Fraction of centreline inside target** was also independently significant for both endpoints (GG2+: OR 1.331, p < 0.001; GG3+: OR 1.227, p = 0.006). Among clinical features, **log(PSA)** showed the largest effect (GG2+: OR 1.818, p < 0.001; GG3+: OR 2.710, p < 0.001) and **prostate volume** was inversely associated with positivity (GG2+: OR 0.681, p < 0.001; GG3+: OR 0.698, p = 0.004).

**Trajectory intersects target** was not individually significant in any multivariable model for either endpoint (GG2+: p = 0.58; GG3+: p = 0.61), consistent with its near-perfect redundancy with fraction of centreline inside target (Spearman ρ = 0.979). Distance to target centroid was not independently significant once distance to target surface was included, reflecting the high collinearity between the two distance features (Spearman ρ = 0.942; VIFs of 15.2 and 14.6 respectively). Similarly, PSA and PSA density did not show independent significance at α = 0.05 in the presence of log(PSA), with which they are strongly correlated by construction. Individual coefficient magnitudes for correlated predictors should not be interpreted as standalone feature importance rankings.

### 3.6 Patient-level secondary analysis

Direct patient-level models and naive aggregation of core-level scores both performed substantially below their core-level counterparts (Table 6). For GG2+, the best direct patient-level model (clinical-only logistic regression) achieved ROC-AUC 0.613 (95% CI 0.515–0.716), nearly identical to naive aggregation of core-level predicted probabilities (ROC-AUC 0.617; ΔROC = −0.004). For GG3+, direct patient-level modelling using full geometry aggregates and clinical features (HistGradientBoosting) gave a modest ROC-AUC improvement over naive aggregation (0.706 vs. 0.682; ΔROC = +0.024) but with a corresponding PR-AUC reduction (0.498 vs. 0.512; ΔPR = −0.014). In both cases, patient-level ROC-AUC remained approximately 0.14 below the corresponding core-level estimate (GG2+: 0.613 vs. 0.754; GG3+: 0.706 vs. 0.828). This gap reflects the structural difficulty of patient-level prediction: patient-level prevalence is substantially higher than core-level prevalence (GG2+: 58.3% vs. 10.7%; GG3+: 25.8% vs. 3.8%), compressing the discrimination problem, and the core-specific spatial geometry signal is attenuated when averaged across all of a patient's cores. Core-level risk stratification is therefore the primary analytical result of this paper; patient-level prediction is reported as a secondary analysis.

### 3.7 Calibration and interpretation caveats

Raw predicted probabilities from all models were poorly calibrated prior to recalibration, with Brier skill scores below zero (raw BSS < 0 for all endpoint–model combinations), indicating that uncorrected predicted probabilities performed worse than a naive prevalence-based predictor. After Platt scaling on the validation set, calibration improved substantially: post-recalibration BSS 0.108 for GG2+ logistic regression and 0.084 for GG3+ logistic regression; expected calibration error (ECE) fell below 0.03 for both endpoints after recalibration. For GG3+, the post-recalibration calibration slope was 1.511, indicating that predicted scores are compressed toward low probabilities; the model should be treated as a ranking instrument rather than an absolute risk estimator. No external recalibration was performed; calibration characteristics should be re-assessed before applying these models to a different dataset or biopsy system.

---

## 4. Discussion hooks

The following points are identified as the key discussion topics for a later full draft. Each is a starting point, not a final argument.

- **Target-relative geometry as a mechanistic signal.** The finding that cores closer to the MRI-defined target surface have higher cancer probability is a direct physical consequence of accurate biopsy targeting. The compact model quantifies, in interpretable units (millimetres, fractions), how closely each needle reached the lesion the radiologist identified. This distinguishes the current approach from black-box image-based classifiers and provides a first-order mechanistic explanation of biopsy yield.

- **Circularity of radiologist target definition.** The MRI target used to compute geometry features is defined by the same radiologist whose biopsy procedure generated the cores. Cores near the target are more likely to be positive partly because the radiologist identified that area as suspicious. The geometric signal therefore reflects the conjunction of (i) the radiologist's targeting accuracy and (ii) the correspondence between the MRI-visible lesion and the histological finding. This circularity must be acknowledged as a limitation; the model should not be interpreted as learning core-level cancer biology independently of targeting.

- **Core-level versus patient-level analytical unit.** The core is the natural unit for geometric analysis because the geometry features are core-specific. Patient-level prediction aggregates away the local spatial signal. The ~0.14 ROC-AUC gap between core-level and patient-level performance is consistent with this interpretation. Future work on patient-level prediction should consider incorporating multiple targets and multi-target geometric summaries rather than simple mean aggregation.

- **Interpretability and clinical communication.** An eight-feature logistic regression model with standardised coefficients can be presented as a scorecard. The two geometry features that drive the result (distance to surface, fraction inside) have clear physical meanings that can be communicated to clinicians and patients. This is a practical advantage over latent-feature neural models.

- **Class imbalance and metric choice.** The 3.8% prevalence of GG3+ at the core level means that ROC-AUC can appear favourable while the model has poor positive predictive value. PR-AUC and PR-lift are the more informative metrics under these conditions. The PR-AUC CI for GG3+ (0.094–0.516) is wide enough to prevent strong performance claims; risk-stratification enrichment (8.3× at top 5%) provides an alternative framing that is robust to this uncertainty.

- **Calibration limitations of raw probabilities.** The poor raw calibration (BSS < 0) means the model outputs should not be used as clinical risk scores without recalibration. After Platt scaling, calibration is adequate for GG2+ but the GG3+ model remains conservative (slope 1.511). Clinicians should be informed that the model is a ranking tool, not an absolute risk calculator.

- **No external validation.** All results come from a single publicly available dataset processed with a single registration pipeline. Generalisability to other fusion biopsy systems, scanners, MRI sequences, targeting protocols, and reader populations is unknown. This is the most significant limitation for any translational interpretation of the results.

- **Why not a deep learning framing.** The models used are logistic regression and gradient boosting on tabular geometric and clinical scalars, not convolutional networks operating on image patches. Deep learning models trained on 120-patient test sets with wide bootstrap CIs would be unlikely to produce more reliable performance estimates; the strength of the current approach is interpretability and transparency, not maximum ROC-AUC. Prior CNN/GNN experiments predate the geometry framing and should be mentioned briefly as context, not foregrounded.

- **Possible extensions: MRI features and radiomics.** The geometry model could be extended by adding MRI-derived features: PI-RADS score, lesion volume, T2 signal characteristics, or full radiomics descriptors from the target ROI. Whether these add discriminative value over the geometric features — which already encode where the needle went relative to the lesion — is an open empirical question. Such an extension would require careful feature extraction and likely a larger cohort to be powered for comparison.

- **Possible extensions: prior biopsy history.** The dataset includes 7,649 cores targeting prior biopsy positive lesions; these are not used as features in the current analysis. A prior-positivity indicator or spatial distance from a prior positive core may substantially improve GG2+ prediction and is a natural next analysis step.
