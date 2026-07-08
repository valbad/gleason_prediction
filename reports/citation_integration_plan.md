# Citation Integration Plan

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-07-08  
**Status:** Plan only. No files modified. All citation keys from `reports/literature_review_findings.md`. Do not apply patches until remaining gaps in §6 are resolved.

---

## 1. Purpose

This file maps verified references to specific manuscript claims in `reports/manuscript_outline_full.md` and identifies the remaining literature gaps that must be resolved before the citation layer is complete. It is a working plan, not a final bibliography.

---

## 2. Verified references to use

| Citation key | Short label | Main use in manuscript | Sections where cited |
|-------------|------------|----------------------|---------------------|
| Sung2021 | GLOBOCAN 2020 | Prostate cancer global burden | Intro §P1 |
| Kasivisvanathan2018 | PRECISION trial | MRI-targeted biopsy improves csPCa detection | Intro §P1 |
| Rouvière2019 | MRI-FIRST trial | Second prospective MRI-targeted biopsy study | Intro §P1 |
| Sonn2013 | Artemis system | MRI-US fusion system with tracked needle coordinates (data source method paper) | Intro §P2; Methods §2.1 |
| TCIA2020 | TCIA dataset | Formal dataset citation | Intro §P2; Methods §2.1 |
| Collins2015 | TRIPOD | Reporting standard for clinical prediction models | Methods §2.4; Disc §D3, §D8 |
| Collins2024 | TRIPOD+AI | Updated reporting standard for ML prediction models | Methods §2.4; Disc §D8 |
| Wolff2019 | PROBAST | Risk-of-bias tool for prediction model studies | Disc §D8 |
| VanCalster2019 | Calibration / Van Calster | Calibration as the key limitation of prediction models | Disc §D6 |
| Saito2015 | PR-AUC (Saito) | Justification for PR-AUC in imbalanced datasets | Methods §2.4 |
| Davis2006 | PR/ROC curves (Davis) | Theoretical grounding for PR-AUC | Methods §2.4 |
| Liang1986 | GEE / Liang–Zeger | Canonical GEE / cluster-robust SE methodology | Methods §2.6 |
| Vickers2006 | DCA | Decision curve analysis as a future clinical utility step | Disc §D9 |
| Varoquaux2022 | ML failures / Varoquaux | Overfitting risk in small medical imaging cohorts; supports compact-model preference | Disc §D3, §D7 |
| Gayo2022 | RL needle placement | Closest prior work: needle geometry operationalised computationally; establishes that no prior work did core-level outcome prediction from geometry | Intro §P2, §P3; Disc §D9 novelty |

---

## 3. Introduction citation placement

### §P1 — Prostate MRI-targeted biopsy and remaining uncertainty at the core level

**Claims and assignments:**

| Sentence | Claim | Citation |
|----------|-------|---------|
| 1 | Prostate cancer is among the most commonly diagnosed malignancies worldwide | [Sung2021] |
| 2 | mpMRI has transformed biopsy practice | [TODO: EAU/NICE guideline or landmark MRI review] |
| 3 | MRI-targeted fusion biopsy improves detection of csPCa vs systematic biopsy | [Kasivisvanathan2018; Rouvière2019] |
| 4 | Biopsy yield remains variable across cores and patients | [Kasivisvanathan2018; TODO: per-core variability reference] |
| 5 | Registration error, prostate motion, and fusion inaccuracies contribute to variability | [TODO: registration/targeting error study — Hale et al. or equivalent] |

**Revised §P1 draft:**

> Prostate cancer remains one of the most commonly diagnosed malignancies in men worldwide [Sung2021]. Multiparametric MRI has transformed prostate biopsy practice by enabling fusion-guided targeted sampling of suspicious lesions identified on imaging [TODO: landmark MRI review or guideline]. MRI-ultrasound (MRI-US) fusion biopsy improves the detection rate of clinically significant prostate cancer (ISUP Grade Group ≥ 2) compared with systematic biopsy in several settings [Kasivisvanathan2018; Rouvière2019], but biopsy yield remains variable across cores and patients [Kasivisvanathan2018; TODO]. Not every needle that targets a lesion successfully samples it: registration error, prostate motion, deformable fusion inaccuracies, and the three-dimensional geometry of the needle trajectory relative to a lesion's volume all contribute to this variability [TODO: targeting/registration error study]. Understanding which individual biopsy cores are likely to be positive — and why — remains an open question at the core level.

**TODO remaining:** mpMRI pathway guideline/review (sentence 2); per-core yield variability reference (sentence 4); registration/targeting error study (sentence 5).

---

### §P2 — Why the spatial relationship between needle and target should carry information

**Claims and assignments:**

| Sentence | Claim | Citation |
|----------|-------|---------|
| Physical argument (sentences 1–3) | Spatial relationship should determine whether core samples cancer | No external citation required if framed as first-principles reasoning |
| Sentence 4 | MRI-US fusion systems record needle trajectories and target mesh coordinates | [Sonn2013; TCIA2020] |
| Final new sentence | Needle-to-target geometry has been operationalised computationally in biopsy planning | [Gayo2022] |

**Revised §P2 draft:**

> The probability that a biopsy core samples a target lesion is governed, at least in part, by the three-dimensional spatial relationship between the needle trajectory and the lesion volume. A core whose needle midpoint lies within the target mesh boundary, or whose centreline traverses the target volume, should — all else equal — be more likely to contain cancer cells from that lesion than a core whose trajectory misses the target entirely. This spatial reasoning motivates representing each core by explicit geometric features derived from registered needle-to-target coordinates: the distance from the needle midpoint to the target surface, the distance to the target centroid, the fraction of the centreline inside the target volume, and a binary indicator of trajectory-target intersection. These features have direct physical units (millimetres, fractions), interpretable sign directions (closer to surface → higher cancer probability), and can be derived from MRI-US fusion biopsy systems that record both needle trajectories and MRI target mesh coordinates [Sonn2013; TCIA2020]. Prior work has demonstrated that needle-to-target geometry can be operationalised computationally in a biopsy planning context [Gayo2022], motivating its use as a feature set for core-level outcome prediction.

**TODO remaining:** None for verified references. The new final sentence citing Gayo2022 is an addition to the current manuscript draft — include it when applying patches.

---

### §P3 — Gap in the literature

**Claims and assignments:**

| Sentence | Claim | Citation |
|----------|-------|---------|
| Image-based model category | CNN/radiomics/multi-instance models exist | [TODO: radiomics review; TODO: DL review] |
| Patient-level risk model category | ERSPC / PBCG-type calculators use PSA, volume, biopsy history | [TODO: Roobol/ERSPC; TODO: Ankerst/PBCG] |
| Gap: neither category uses needle-to-target geometry | Prior geometry work is in planning context, not prediction | [Gayo2022] |
| Novelty claim | No prior study has constructed this feature set + held-out evaluation + cluster-robust inference | [see §5 below] |

**Revised §P3 draft:**

> Existing computational approaches to prostate biopsy outcome prediction fall into two broad categories. Image-based models — including convolutional neural networks applied to MRI patches, radiomics pipelines, or multi-instance learning frameworks — aim to extract predictive features from the MRI signal itself [TODO: prostate MRI radiomics review; TODO: deep learning systematic review]. Patient-level clinical models use PSA, prostate volume, and biopsy history to estimate the probability that a patient will be found to have significant cancer on any biopsy [TODO: Roobol/ERSPC; TODO: Ankerst/PBCG]. Both categories share a common limitation: they do not explicitly model the geometric relationship between where each needle went and where the MRI-identified lesion was located. Computational work has addressed needle geometry in a biopsy planning context [Gayo2022], but this is distinct from predicting core-level cancer outcomes from the geometry of needles already placed. To our knowledge, and pending a final targeted literature search, no prior study has combined interpretable target-relative needle-to-lesion geometry features with clinical variables for held-out core-level prostate biopsy outcome prediction while also testing multivariable associations using patient-cluster-robust inference. TODO: verify this novelty claim after the final targeted search (see `reports/literature_review_targets.md` §5).

**TODO remaining:** Radiomics review (priority 3 from `literature_review_findings.md`); DL review; ERSPC calculator; PBCG calculator. Novelty claim must remain provisional until final targeted search is completed.

---

### §P4 — Study contribution

No external citations required. All citations introduced in §§P1–P3 should already be in scope by this point.

**Note:** The sentence "(i) evaluate held-out predictive performance against clinical-only and full-geometry reference feature sets" may optionally cite [Collins2015] or [Collins2024] to establish the held-out split paradigm as best practice — but this is not required in the contribution paragraph.

---

### §P5 — Summary of aims

No external citations required. Aims reference the study design directly; methodological citations belong in the Methods section.

---

## 4. Discussion citation placement

### §D1 — Main finding

**Proposed citations:** None required in the main finding paragraph itself. The cluster-robust OR values are internal results; [Liang1986] is already in Methods §2.6 where the method is described.

**Sentence guidance:**
- The statement that geometric associations "remain significantly associated with biopsy core positivity in multivariable patient-cluster-robust models" is a results claim, not a claim requiring external literature support.
- If a reviewer asks for a precedent for the cluster-robust approach, the Methods section citation [Liang1986] covers this.

**TODO:** None.

---

### §D2 — Interpretation of the two dominant geometry features

**Proposed citations:** None required. The interpretation rests on internal VIF and Spearman correlation results.

**Sentence guidance:**
- "did not reach significance once surface distance was included" — internal collinearity result, no external citation needed.
- "its coefficient should not be interpreted as a separable contribution beyond fraction inside" — could optionally cite [Collins2015] or [Steyerberg textbook — TODO] for parsimony guidance, but is not required.

**TODO:** None for verified references.

---

### §D3 — Why the compact model is preferable

**Proposed citations:** [Varoquaux2022; Collins2015 or Collins2024; TODO: Steyerberg textbook]

**Sentence guidance:**

> The compact model is preferred for practical reasons: it requires fewer features, is fully auditable, and its coefficients have direct physical interpretations. This parsimony is consistent with methodological guidance advocating against adding high-dimensional features to small cohorts where additional complexity risks overfitting without improving generalisation [Varoquaux2022]. Reporting standards for clinical prediction models also emphasise the value of parsimonious, interpretable designs that can be externally validated [Collins2015; Collins2024].

**TODO:** Steyerberg textbook for parsimony argument in clinical prediction models — useful but not essential.

---

### §D4 — Core-level versus patient-level analytical unit

**Proposed citations:** None required. All claims are internal analysis results.

**Sentence guidance:**
- The observation that patient-level prevalence compresses discrimination is a study-specific finding, not a claim requiring external support.
- If the Discussion mentions that core-level analyses require intra-patient clustering correction, [Liang1986] is already cited in Methods.

**TODO:** None.

---

### §D5 — Interpretability and the radiologist targeting circularity

**Proposed citations:** None required. The circularity argument is a structural feature of the study design, not a literature-supported claim.

**Sentence guidance:**
- The limitation that "the MRI-defined target mesh is drawn by the same radiologist who planned the biopsy" is a study design limitation stated directly — no external citation needed.
- If the reviewers ask for a reference establishing that radiologist-drawn ROIs carry PI-RADS information, a prostate MRI guideline could be cited here [TODO: PI-RADS v2.1 guideline if needed].

**TODO:** PI-RADS v2.1 guideline (low priority; only if a reviewer requests it).

---

### §D6 — Calibration and limits of the risk stratification framing

**Proposed citations:** [VanCalster2019; Collins2015 or Collins2024]

**Sentence guidance:**

> Raw predicted probabilities from the compact logistic regression model were poorly calibrated: Brier skill scores were negative for all models before recalibration, illustrating that discrimination and calibration are distinct properties of a prediction model [VanCalster2019]. After Platt scaling on the validation set, calibration improved substantially (BSS 0.108 for GG2+, 0.084 for GG3+; ECE < 0.03). The model should be treated as a ranking instrument rather than an absolute risk estimator. These results should not be used for threshold-based clinical decision-making without recalibration and external validation on an independent prospective cohort — a requirement highlighted by clinical prediction model reporting standards [Collins2024; Wolff2019].

**TODO:** None for verified references. Wolff2019 (PROBAST) is an optional addition here to reinforce the external validation message.

---

### §D7 — Relation to deep learning and radiomics approaches

**Proposed citations:** [Varoquaux2022; TODO: prostate MRI radiomics review; TODO: DL systematic review]

**Sentence guidance:**

> The approach presented here is deliberately distinct from image-based deep learning and radiomics pipelines that extract features from MRI intensity, texture, or patch content [TODO: prostate MRI radiomics review; TODO: DL systematic review]. The features used in this study are geometric and clinical scalars derived from registered coordinates; they require no convolutional network, no GPU infrastructure, and no image patch extraction. This has practical advantages: the features are less directly dependent on MRI intensity protocol than radiomics or image-patch models, provided target meshes and needle trajectories are available in a reliable common coordinate frame. Attempting to maximise ROC-AUC by adding high-dimensional MRI features on the same cohort would risk overfitting and would produce performance claims whose confidence intervals would overlap substantially with the current results — a well-documented failure mode in medical imaging machine learning [Varoquaux2022]. The appropriate next step is external validation, not architectural complexity.

**TODO:** Radiomics review (Stanzione2020 — verify PMID/DOI; priority 3 in `literature_review_findings.md`); DL systematic review (choose between Eur Urol Oncol 2024 PMID 39547898 or Life 2022 DOI 10.3390/life12101490 — verify before committing).

---

### §D8 — Limitations

**Proposed citations:** [Collins2015 or Collins2024; Wolff2019; Liang1986; TODO: Gleason inter-rater variability]

**Sentence guidance:**

> Second, no external validation has been performed; generalisation to other MRI-US fusion systems, scanners, targeting protocols, or reader populations is uncertain — a critical requirement for clinical prediction models [Collins2024; Wolff2019]. [...] Fourth, histological labels are assigned to the entire core, not to a specific spatial location; Gleason grading has known inter-rater variability [TODO: Allsbrook or Epstein — verify]. Fifth, the effective sample size for inferential purposes is closer to 796 patients than to 16,992 cores, because core-level observations within a patient are not independent [Liang1986].

**TODO:** Gleason inter-rater variability citation (priority 9 in `literature_review_findings.md`).

---

### §D9 — Future work

**Proposed citations:** [Vickers2006]

**Sentence guidance:**

> A clinical utility study using decision curve analysis [Vickers2006] would be needed to assess whether the model's risk stratification adds value over clinician judgment in the context of actual biopsy triage decisions.

**Note:** Cite [Vickers2006] only for the future-work sentence about DCA. Do not cite it in the main results discussion as if DCA was performed.

**TODO:** None for verified references.

---

## 5. Novelty claim update

### Current claim (from `reports/manuscript_outline_full.md` §4 Introduction §P3)

> "To our knowledge, and pending a final targeted literature search, no prior study has combined interpretable target-relative needle-to-lesion geometry features with clinical variables for held-out core-level prostate biopsy outcome prediction while also testing multivariable associations using patient-cluster-robust inference. TODO: verify this novelty claim after the literature search."

### Current status

**Cautious novelty likely.** Searches conducted in `reports/literature_review_findings.md` found no direct prior work. The closest work found is:

---

**Gayo et al. 2022** — *Strategising template-guided needle placement for MR-targeted prostate biopsy.* CaPTion Workshop, MICCAI 2022. DOI: 10.1007/978-3-031-17979-2_15. arXiv:2207.10784.

| Dimension | Gayo2022 | This study |
|-----------|---------|-----------|
| Input | Pre-procedure MRI target + biopsy template grid | Post-procedure needle trajectory coordinates + MRI target mesh |
| Task | Reinforcement learning to plan optimal needle positions before biopsy | Predict core-level cancer outcome from geometric features of placed needles |
| Features | Grid template positions relative to MRI target | Interpretable scalars: distance to surface, distance to centroid, fraction inside, intersection |
| Outcome | Optimised needle hit-rate and cancer core length | Binary cancer label (GG2+, GG3+) per biopsy core |
| Inference | No patient-level statistical inference | Cluster-robust logistic regression and GEE |
| Evaluation | Simulation hit-rate | Held-out test-set ROC-AUC, PR-AUC, risk stratification |

**Summary:** Gayo2022 demonstrates that needle-to-target geometry is computationally tractable in the biopsy context, but addresses a fundamentally different problem (pre-procedure planning vs post-procedure outcome prediction). It does not build an interpretable feature set from placed needle coordinates, does not evaluate core-level cancer prediction on a held-out test set, and does not use cluster-robust inference.

---

### Recommended final novelty wording

> "While prior computational work has addressed needle placement planning using biopsy-target geometry [Gayo2022], and existing clinical prediction models use patient-level and MRI-based features [TODO: ERSPC; TODO: DL/radiomics reviews], no prior study has, to our knowledge, constructed an interpretable feature set from core-level needle-to-target spatial coordinates, evaluated it against reference feature sets on a held-out test split, and tested multivariable associations using patient-cluster-robust inference."

**TODO: retain this wording only after final targeted search for tracked-biopsy geometry and core-level biopsy prediction papers.** Run the five remaining searches listed in `reports/literature_review_findings.md` §7 (priorities 1–2) before committing.

---

## 6. Remaining citation gaps

### Must resolve before full draft

| Gap | Why required | Search needed |
|-----|-------------|---------------|
| Prostate MRI radiomics review (priority 3) | Intro §P3 and Discussion §D7 — the claim that "image-based models exist" must be supported by a systematic review, not asserted | PubMed: Stanzione 2020 Eur J Radiol — confirm PMID and DOI directly |
| Deep learning prostate MRI systematic review | Same location as above | Choose one of: Eur Urol Oncol 2024 (PMID 39547898) or Life 2022 (DOI 10.3390/life12101490); verify full citation |
| Patient-level biopsy risk calculator — ERSPC (Roobol et al.) | Intro §P3 category 2 claim | PubMed: `Roobol ERSPC risk calculator prostate biopsy Eur Urol` |
| Patient-level biopsy risk calculator — PBCG (Ankerst et al.) | Intro §P3 category 2 (alternative to or alongside ERSPC) | PubMed: `Ankerst PBCG prostate biopsy collaborative group risk calculator` |
| Targeting/registration error study (Hale et al. or equivalent) | Intro §P1 sentence 5 — the claim that registration and motion errors contribute to sampling variability must be cited with quantitative data | PubMed: `prostate biopsy MRI ultrasound registration error targeting accuracy in vivo` |
| Novelty verification — final targeted search | Introduction §P3 novelty claim cannot be finalised without these searches | Five searches listed in `reports/literature_review_findings.md` §7 priorities 1–2 |

### Useful before supervisor review (not essential for first draft)

| Gap | Why useful | Search needed |
|-----|-----------|---------------|
| Steyerberg EW textbook (2nd ed., 2019) | Discussion §D3 parsimony argument and §D6 calibration framing | Springer catalogue: confirm 2nd edition details (ISBN) |
| mpMRI guideline or landmark MRI review (Intro §P1 sentence 2) | Supports claim that mpMRI "transformed biopsy practice" | EAU guideline 2023/2024 prostate cancer section; or a major MRI-targeted biopsy review article |
| Gleason inter-rater variability (Allsbrook/Epstein) | Discussion §D8 limitation on labelling reliability | PubMed: `Gleason grading interobserver variability prostate pathology` |
| PI-RADS v2.1 guideline | Optional anchor for Discussion §D5 circularity discussion | PI-RADS v2.1 Eur Urol 2019 — Turkbey et al. |
| Lambin et al. — Radiomics Quality Score | Discussion §D7 methodological critique of radiomics | PubMed: distinguish RQS tool paper from Nat Rev Clin Oncol 2017 bridge paper |

---

## 7. Exact manuscript patches recommended

The following are proposed edits to `reports/manuscript_outline_full.md`. Do not apply these patches until the must-resolve gaps in §6 are addressed.

---

### Patch 7.1 — Introduction §P1 draft paragraph

**File:** `reports/manuscript_outline_full.md`  
**Section:** `### Paragraph 1`, under `**Draft paragraph:**`

**Replace current paragraph** with:

> Prostate cancer remains one of the most commonly diagnosed malignancies in men worldwide [Sung2021]. Multiparametric MRI has transformed prostate biopsy practice by enabling fusion-guided targeted sampling of suspicious lesions identified on imaging [TODO: EAU/NICE guideline or landmark review]. MRI-ultrasound (MRI-US) fusion biopsy improves the detection rate of clinically significant prostate cancer (ISUP Grade Group ≥ 2) compared with systematic biopsy in several settings [Kasivisvanathan2018; Rouvière2019], but biopsy yield remains variable across cores and patients [Kasivisvanathan2018; TODO: per-core variability reference]. Not every needle that targets a lesion successfully samples it: registration error, prostate motion, deformable fusion inaccuracies, and the three-dimensional geometry of the needle trajectory relative to a lesion's volume all contribute to this variability [TODO: registration/targeting error study]. Understanding which individual biopsy cores are likely to be positive — and why — remains an open question at the core level.

**Remove** the existing Notes / citations needed list and replace with:
> **Remaining TODOs:** mpMRI pathway guideline sentence 2; per-core yield variability reference sentence 4; registration error study sentence 5. See `reports/citation_integration_plan.md` §6.

---

### Patch 7.2 — Introduction §P2 draft paragraph

**File:** `reports/manuscript_outline_full.md`  
**Section:** `### Paragraph 2`, under `**Draft paragraph:**`

**Append** the following sentence to the end of the paragraph (after "...without requiring image patch extraction or learned representations."):

> Prior work has demonstrated that needle-to-target geometry can be operationalised computationally in a biopsy planning context [Gayo2022], motivating its use as a predictive feature set for core-level outcome modelling.

**Replace** existing Notes / citations needed with:
> **Remaining TODOs:** Sentence 4 now cites [Sonn2013; TCIA2020]. Final sentence cites [Gayo2022]. No further citations needed for this paragraph.

---

### Patch 7.3 — Introduction §P3 draft paragraph

**File:** `reports/manuscript_outline_full.md`  
**Section:** `### Paragraph 3`, under `**Draft paragraph:**`

**Replace current paragraph** with:

> Existing computational approaches to prostate biopsy outcome prediction fall into two broad categories. Image-based models — including convolutional neural networks applied to MRI patches, radiomics pipelines, or multi-instance learning frameworks — aim to extract predictive features from the MRI signal itself [TODO: prostate MRI radiomics review; TODO: DL systematic review]. Patient-level clinical models use PSA, prostate volume, and biopsy history to estimate the probability that a patient will be found to have significant cancer on any biopsy [TODO: Roobol/ERSPC; TODO: Ankerst/PBCG]. Both categories share a common limitation: they do not explicitly model the geometric relationship between where each needle went and where the MRI-identified lesion was located. Computational work has addressed needle geometry in a biopsy planning context [Gayo2022], but this is distinct from predicting core-level cancer outcomes from the geometry of needles already placed. To our knowledge, and pending a final targeted literature search, no prior study has combined interpretable target-relative needle-to-lesion geometry features with clinical variables for held-out core-level prostate biopsy outcome prediction while also testing multivariable associations using patient-cluster-robust inference.

**Replace** existing Notes / citations needed with:
> **Remaining TODOs:** Radiomics review; DL review; ERSPC; PBCG. Final novelty sentence must be re-evaluated after completing searches listed in `reports/literature_review_findings.md` §7 priorities 1–2.

---

### Patch 7.4 — Discussion §D6 draft paragraph

**File:** `reports/manuscript_outline_full.md`  
**Section:** `### §D6`, under `**Draft paragraph:**`

**Replace current paragraph** with:

> Raw predicted probabilities from the compact logistic regression model were poorly calibrated: Brier skill scores were negative for all models before recalibration, illustrating a well-recognised gap between discrimination and calibration in clinical prediction models [VanCalster2019]. After Platt scaling on the validation set, calibration improved substantially (BSS 0.108 for GG2+, 0.084 for GG3+; ECE < 0.03). Nevertheless, the GG3+ post-recalibration calibration slope of 1.511 indicates that predicted scores are compressed toward the low end of the probability scale; the model should be treated as a ranking instrument rather than an absolute risk estimator. In particular, the enrichment results (e.g., 50.4% positivity in the top 5% of GG2+ scored cores) characterise the model's discrimination and ranking ability, not the absolute probability that any given core is positive. These results should not be used for threshold-based clinical decision-making without recalibration and external validation on an independent prospective cohort — a requirement emphasised by clinical prediction model reporting standards [Collins2024; Wolff2019].

---

### Patch 7.5 — Discussion §D7 draft paragraph

**File:** `reports/manuscript_outline_full.md`  
**Section:** `### §D7`, under `**Draft paragraph:**`

**Replace current paragraph** with:

> The approach presented here is deliberately distinct from image-based deep learning and radiomics pipelines that extract features from MRI intensity, texture, or patch content [TODO: prostate MRI radiomics review; TODO: DL systematic review]. The features used in this study are geometric and clinical scalars derived from registered coordinates; they require no convolutional network, no GPU infrastructure, and no image patch extraction. This has practical advantages: the features are less directly dependent on MRI intensity protocol than radiomics or image-patch models, provided target meshes and needle trajectories are available in a reliable common coordinate frame. Attempting to maximise ROC-AUC by adding high-dimensional MRI features on the same cohort would risk overfitting and would produce performance claims whose confidence intervals would overlap substantially with the current results — a failure mode well documented in medical imaging machine learning [Varoquaux2022]. The appropriate next step is external validation, not architectural complexity. Future work could investigate whether MRI-derived lesion characteristics (PI-RADS score, lesion volume, T2 heterogeneity) add value over and above the geometric model on a larger, multi-institutional cohort.

---

### Patch 7.6 — Discussion §D8 draft paragraph

**File:** `reports/manuscript_outline_full.md`  
**Section:** `### §D8`, under `**Draft paragraph:**`

**Replace current paragraph** with:

> This study has several limitations that bound the interpretation of our results. First, all data come from a single publicly available retrospective cohort; selection and referral biases are unknown. Second, no external validation has been performed; generalisation to other MRI-US fusion systems, scanners, targeting protocols, or reader populations is uncertain — a requirement explicitly prescribed by clinical prediction model reporting standards [Collins2024; Wolff2019]. Third, MRI-US registration introduces geometric noise, which attenuates the measurable geometric signal; reported associations are likely lower bounds of the true relationship. Fourth, histological labels are assigned to the entire core, not to a specific spatial location; Gleason grading has known inter-rater variability [TODO: Allsbrook/Epstein or equivalent]. Fifth, the effective sample size for inferential purposes is closer to 796 patients than to 16,992 cores, because cores from the same patient are not independent [Liang1986]. Sixth, GG3+ results have wide confidence intervals (PR-AUC CI range 0.42) due to only 92 positive cores in the test set, and should be treated as indicative. Seventh, PSA-related features are patient-level values replicated to all cores; their joint model coefficients are subject to collinearity and should not be used to rank clinical feature importance. Finally, the MRI-defined target mesh is determined by the same radiologist who guided the biopsy, introducing circularity into the geometric signal.

**Note:** The only internal change is adding [Collins2024; Wolff2019] at limitation 2 and [Liang1986] at limitation 5. TODO for limitation 4 Gleason grading citation remains.

---

### Patch 7.7 — Discussion §D9 future work sentence (DCA)

**File:** `reports/manuscript_outline_full.md`  
**Section:** `### §D9`, within the draft paragraph

**Replace current sentence:**
> A clinical utility study using decision curve analysis would be needed to assess whether the model's risk stratification adds value over clinician judgment in the context of actual biopsy triage decisions.

**With:**
> A clinical utility study using decision curve analysis [Vickers2006] would be needed to assess whether the model's risk stratification adds value over clinician judgment in the context of actual biopsy triage decisions.
