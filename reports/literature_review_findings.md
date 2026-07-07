# Literature Review Findings

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-07-07  
**Status:** First verified pass. Bibliographic details confirmed via PubMed, journal websites, and TCIA pages where noted. Verification method stated per entry. All unverified entries kept in §3.

---

## 1. Purpose

This file records verified and candidate sources for:
1. Filling all CITATION NEEDED placeholders in `reports/manuscript_outline_full.md` and `reports/manuscript_methods_results_draft.md`.
2. Establishing the novelty claim status for Introduction §P3.
3. Supporting Discussion limitations with real methodological references.

Sources consulted for this pass: `reports/literature_review_targets.md`, `reports/manuscript_outline_full.md`, plus web searches against PubMed, journal websites, and the TCIA collection page.

---

## 2. Verified anchor references

All entries below were confirmed against PubMed, a journal/publisher page, or an official archive. Method of verification is noted in the Status column.

| Citation key | Full reference | DOI / PMID | Why it matters | Manuscript location | Status |
|-------------|---------------|-----------|---------------|---------------------|--------|
| Sung2021 | Sung H, Ferlay J, Siegel RL, Laversanne M, Soerjomataram I, Jemal A, Bray F. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. *CA Cancer J Clin.* 2021;71(3):209–249. | DOI 10.3322/caac.21660 / PMID 33538338 | Prostate cancer incidence and burden — supports Intro §P1 sentence 1 | Intro §P1 | Verified via PubMed and Wiley |
| Kasivisvanathan2018 | Kasivisvanathan V, Rannikko AS, Borghi M, et al.; PRECISION Study Group Collaborators. MRI-Targeted or Standard Biopsy for Prostate-Cancer Diagnosis. *N Engl J Med.* 2018;378(19):1767–1777. | DOI 10.1056/NEJMoa1801993 / PMID 29552975 | PRECISION trial — canonical RCT for MRI-targeted vs systematic biopsy in biopsy-naive men | Intro §P1 | Verified via PubMed and NEJM |
| Rouviere2019 | Rouvière O, Puech P, Renard-Penna R, et al.; MRI-FIRST Investigators. Use of prostate systematic and targeted biopsy on the basis of multiparametric MRI in biopsy-naive patients (MRI-FIRST): a prospective, multicentre, paired diagnostic study. *Lancet Oncol.* 2019;20(1):100–109. | DOI 10.1016/S1470-2045(18)30569-2 / PMID 30470502 | Second major prospective trial; shows combined strategy needed | Intro §P1 | Verified via PubMed and Lancet Oncology |
| Sonn2013 | Sonn GA, Natarajan S, Margolis DJ, et al. Targeted biopsy in the detection of prostate cancer using an office based magnetic resonance ultrasound fusion device. *J Urol.* 2013;189(1):86–91. | DOI 10.1016/j.juro.2012.08.095 / PMID 23158413 | Clinical methods paper for the Artemis MRI-US fusion system used in the TCIA dataset | Methods §2.1 / Intro §P1 | Verified via PubMed |
| TCIA2020 | Natarajan S, Priester A, Margolis D, Huang J, Marks L. *Prostate MRI and Ultrasound With Pathology and Coordinates of Tracked Biopsy* (version 2) [Data set]. The Cancer Imaging Archive; 2020. | DOI 10.7937/TCIA.2020.A61IOC1A | Formal citation for the dataset used in this study | Methods §2.1 | Verified via TCIA collection page (cancerimagingarchive.net/collection/prostate-mri-us-biopsy/) |
| Collins2015 | Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis Or Diagnosis (TRIPOD): The TRIPOD Statement. *Ann Intern Med.* 2015;162(1):55–63. | DOI 10.7326/M14-0697 / PMID 25560714 | Reporting guideline for prediction model studies — Methods §2.4 and Discussion §D8 | Methods §2.4 / Discussion §D8 | Verified via PubMed and Annals |
| Collins2024 | Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378. | DOI 10.1136/bmj-2023-078378 / PMID 38626948 | Updated TRIPOD for ML models — applicable because we use logistic regression and gradient boosting | Methods §2.4 / Discussion §D8 | Verified via PubMed |
| Wolff2019 | Wolff RF, Moons KGM, Riley RD, et al.; PROBAST Group. PROBAST: A Tool to Assess the Risk of Bias and Applicability of Prediction Model Studies. *Ann Intern Med.* 2019;170(1):51–58. | DOI 10.7326/M18-1376 / PMID 30596875 | Risk-of-bias appraisal tool for prediction model studies | Discussion §D8 | Verified via PubMed and Annals |
| VanCalster2019 | Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med.* 2019;17(1):230. | DOI 10.1186/s12916-019-1466-7 / PMID 31842878 | Key calibration reference; directly supports the framing of our calibration results as ranking-focused | Discussion §D6 | Verified via PubMed and BMC Medicine |
| Saito2015 | Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLoS One.* 2015;10(3):e0118432. | DOI 10.1371/journal.pone.0118432 / PMID 25738806 | Primary justification for reporting PR-AUC alongside ROC-AUC given class imbalance (10.7% / 3.8% prevalence) | Methods §2.4 | Verified via PubMed and PLOS ONE |
| Davis2006 | Davis J, Goadrich M. The relationship between Precision-Recall and ROC curves. In: *Proceedings of the 23rd International Conference on Machine Learning (ICML 2006)*. Pittsburgh, PA; 2006:233–240. ACM. | DOI 10.1145/1143844.1143874 | Theoretical grounding for PR-AUC; often cited alongside Saito 2015 | Methods §2.4 | Verified via ACM Digital Library and dblp |
| Liang1986 | Liang KY, Zeger SL. Longitudinal data analysis using generalized linear models. *Biometrika.* 1986;73(1):13–22. | DOI 10.1093/biomet/73.1.13 (no PMID — published before PubMed indexing of statistics journals) | Canonical GEE paper; supports the cluster-robust inference methodology (statsmodels GEE) | Methods §2.6 | Verified via Oxford Academic |
| Vickers2006 | Vickers AJ, Elkin EB. Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. *Med Decis Making.* 2006;26(6):565–574. | DOI 10.1177/0272989X06295361 / PMID 17099194 | DCA methodology; supports Discussion §D9 future work recommendation | Discussion §D9 | Verified via PubMed and SAGE Journals |
| Varoquaux2022 | Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. *npj Digit Med.* 2022;5:48. | DOI 10.1038/s41746-022-00592-y / PMID 35413988 | Methodological critique of ML in medical imaging; supports claim that high-dimensional features require larger cohorts for reliable evaluation | Discussion §D7 | Verified via PubMed and Nature |
| Gayo2022 | Gayo IJ, Saeed SU, Barratt DC, Clarkson MJ, Hu Y. Strategising template-guided needle placement for MR-targeted prostate biopsy. In: *CaPTion 2022: Cancer Prevention Through Early Detection*. Lecture Notes in Computer Science, vol 13581. Springer, Cham; 2022. | DOI 10.1007/978-3-031-17979-2_15 / arXiv:2207.10784 | Closest prior work found using needle geometry in prostate biopsy context — but for RL-based placement planning, not core-level cancer prediction. Important for novelty claim verification. | Intro §P3 / Discussion §D9 novelty | Verified via SpringerLink and arXiv |

---

## 3. Candidate references still needing verification

| Candidate source | Search query used / to use | Why relevant | Manuscript location | What must be verified |
|-----------------|-----------------------------|-------------|---------------------|-----------------------|
| Stanzione A, Gambardella M, Cuocolo R, et al. Prostate MRI Radiomics: A Systematic Review and Radiomic Quality Score Assessment. *Eur J Radiol.* 2020;129:109095. | Direct PubMed search: `Stanzione prostate MRI radiomics systematic review radiomic quality score 2020` | Systematic review of prostate MRI radiomics; Discussion §D7 claim that image-based approaches exist and have been systematically reviewed | Intro §P3 / Discussion §D7 | Verify PMID and DOI. Title and journal confirmed in secondary citations but not retrieved directly from PubMed in this pass. |
| Roobol MJ et al. ERSPC risk calculator paper. *Eur Urol.* (year unknown) | `Roobol ERSPC risk calculator prostate cancer biopsy Eur Urol` | Patient-level pre-biopsy clinical risk model; reference for Intro §P3 category 2 | Intro §P3 | Confirm which specific ERSPC calculator paper to cite (multiple exist); verify year, volume, DOI, PMID. |
| Ankerst DP et al. Prostate Biopsy Collaborative Group (PBCG) risk calculator. (year unknown) | `Ankerst PBCG prostate biopsy collaborative group risk calculator` | Alternative patient-level pre-biopsy risk model; reference for Intro §P3 category 2 | Intro §P3 | Confirm full citation including journal, year, DOI, PMID. |
| Steyerberg EW. *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating.* 2nd ed. Springer; 2019. | Springer catalogue search | Textbook reference for parsimony, calibration, and model evaluation — Discussion §D3, §D6 | Discussion §D3 / §D6 | Confirm 2nd edition year (1st: 2009, 2nd: 2019); verify ISBN. Page references needed only if citing a specific claim. |
| Marks LS et al. MRI-US fusion biopsy review or technique paper | `Marks MRI ultrasound fusion prostate biopsy` | Possible Introduction reference for MRI-US fusion system description | Intro §P1–§P2 | Author, title, journal, year, DOI, PMID all unconfirmed. Search PubMed directly. |
| Hale GR et al. targeting error / registration error during MRI-US biopsy | `Hale prostate biopsy registration error MRI ultrasound targeting` | Targeting accuracy / registration error literature for Intro §P1 sentence 5 | Intro §P1 | Author, title, journal, year all unconfirmed. May refer to a different author. Search for "prostate MRI ultrasound registration error in vivo" on PubMed. |
| Cool DW et al. rigid vs deformable registration / cognitive fusion comparison | `Cool MRI ultrasound fusion prostate biopsy registration` | Registration comparison study for Intro §P1 | Intro §P1 | Author, title, journal, year all unconfirmed. Search PubMed. |
| Deep learning prostate MRI diagnostic accuracy systematic review (author TBD) | `deep learning prostate cancer MRI detection systematic review 2022 2023` | Discussion §D7 claim that DL approaches have been systematically reviewed | Discussion §D7 | Multiple systematic reviews exist (see search results in this pass). Choose one or two most cited; confirm full citations. Candidates: (a) review in *Life* 2022, DOI 10.3390/life12101490; (b) review in *Eur Urol Oncol* 2024, PMID 39547898. Verify details before committing. |
| Allsbrook WC Jr et al. or Epstein JI et al. — Gleason grading inter-rater variability | `Gleason grading interobserver variability prostate pathology` | Discussion §D8 limitation on histological label reliability | Discussion §D8 | Confirm which specific inter-rater variability paper to cite; multiple exist. Search: `"Gleason grade" "interobserver" OR "inter-rater" variability pathology prostate`. |
| Lambin P et al. Radiomics Quality Score (RQS). *Eur J Cancer.* 2017. | `Lambin radiomics quality score Eur J Cancer 2017` | Methodological critique supporting Discussion §D7; also used to appraise Stanzione review | Discussion §D7 | Confirm full citation: Lambin P, et al. Radiomics: the bridge between medical imaging and personalized medicine. *Nat Rev Clin Oncol.* 2017;14(12):749–762. — OR — the RQS scoring paper (may be a different Lambin et al. paper). Distinguish between the Radiomics bridge paper and the RQS scoring tool paper. |

---

## 4. Citation placement map

### Introduction §P1

1. **[Sung2021]** — prostate cancer incidence statistics.
2. **[Kasivisvanathan2018]** — MRI-targeted biopsy improves csPCa detection; biopsy yield variability.
3. **[Rouvière2019]** — second major prospective study; combined targeted + systematic strategy.
4. **[Hale et al. — TODO]** — registration error / targeting inaccuracy during MRI-US fusion biopsy.
5. **[Marks et al. — TODO]** — MRI-US fusion technique description (if not subsumed by Sonn2013 and TCIA2020).

### Introduction §P2

1. **[Sonn2013]** — description of Artemis MRI-US fusion system with tracked needle coordinates.
2. **[TCIA2020]** — formal citation for the dataset providing needle trajectories and target mesh coordinates.
3. **[Gayo2022]** — prior work using needle geometry in prostate biopsy context (RL planning, not prediction); demonstrates that needle trajectory coordinates have been operationalised computationally.
4. **[Hale et al. or Cool et al. — TODO]** — further support for the claim that needle-to-target spatial relationships are captured by MRI-US fusion systems.

### Introduction §P3

1. **[Stanzione2020 — TODO verify]** — systematic review of prostate MRI radiomics; supports "image-based models exist" category.
2. **[Deep DL review — TODO]** — systematic review of deep learning for prostate MRI.
3. **[Roobol ERSPC — TODO]** and/or **[Ankerst PBCG — TODO]** — patient-level clinical risk models.
4. **[Gayo2022]** — closest prior work using needle geometry; demonstrate how this study goes beyond it (core-level outcome prediction + cluster-robust inference).
5. **[TCIA2020]** — dataset context; searches of prior work using this dataset.

### Discussion §D2–§D3

1. **[VanCalster2019]** — broader point that calibration and parsimony matter in clinical prediction.
2. **[Steyerberg textbook — TODO verify edition]** — parsimony and interpretability in prediction model development.
3. **[Collins2015]** (TRIPOD) or **[Collins2024]** (TRIPOD+AI) — reporting standards encouraging parsimonious, interpretable models.
4. **[Varoquaux2022]** — methodological critique supporting the argument that adding high-dimensional features on small cohorts does not improve generalisation.

### Discussion §D6

1. **[VanCalster2019]** — calibration is the "Achilles heel"; directly supports §D6 framing.
2. **[Collins2015]** / **[Collins2024]** — TRIPOD/TRIPOD+AI: external validation required before clinical use.
3. **[Wolff2019]** (PROBAST) — prediction model risk-of-bias tool; relevant when discussing study limitations in calibration.
4. **[Vickers2006]** — decision curve analysis as the appropriate next step for clinical utility assessment.

### Discussion §D7

1. **[Stanzione2020 — TODO verify]** — radiomics reviews for prostate MRI.
2. **[DL systematic review — TODO choose]** — deep learning for prostate MRI.
3. **[Varoquaux2022]** — methodological failures and overfitting risk in medical imaging ML.
4. **[Lambin et al. — TODO distinguish papers]** — Radiomics Quality Score to contextualise quality issues.

### Discussion §D8–§D9

1. **[Collins2015]** / **[Collins2024]** — TRIPOD: external validation required.
2. **[Wolff2019]** — PROBAST: risk-of-bias assessment.
3. **[Allsbrook/Epstein Gleason variability — TODO]** — inter-rater variability in Gleason grading as a labelling limitation.
4. **[Liang1986]** — GEE / cluster-robust inference (already cited in Methods §2.6).
5. **[Vickers2006]** — DCA as future work direction.

---

## 5. Novelty claim status

**Claim under review:**
> "To our knowledge, no published study has constructed an interpretable target-relative needle-to-lesion geometry feature set, evaluated it on held-out core-level prostate biopsy outcomes, and tested associations with patient-cluster-robust inference."

**Current status: Cautious novelty likely**

The claim appears well-supported by the absence of such work in searches conducted in this pass, but a definitive conclusion requires the full search protocol in §5 of `reports/literature_review_targets.md`.

---

### Closest prior works found

| Paper | What it does | What it does not do | Relevance to novelty |
|-------|-------------|---------------------|---------------------|
| **Gayo et al. 2022** (MICCAI/CaPTion workshop) | Uses reinforcement learning to optimise template-guided needle positioning for MR-targeted prostate biopsy; operationalises needle-to-target geometry in the biopsy planning context. | Does not build a predictive model of core-level cancer outcome from existing needle trajectories; does not use a geometric feature set for classification; no cluster-robust inference; no held-out evaluation of cancer prediction. | Most relevant prior work found. Confirms that needle-trajectory-relative-to-MRI-target is a recognised geometric quantity, but shows a fundamentally different use case (planning vs prediction). |
| **Clinical biopsy core prediction studies** (several, PubMed) | Predict biopsy positivity using clinical variables (PI-RADS, PSA density, age, lesion volume, Likert score) at the lesion or patient level. | Do not use spatial needle-to-target geometry as a feature; do not operate at the individual core level with geometric predictors; do not apply cluster-robust inference. | Establishes the category of "biopsy outcome prediction" but uses different predictors and analytical units. |
| **Needle deflection/localization papers** (e.g., Esfandiari, Vrooijink, Mehrabian types) | Develop mechanical or neural models to predict needle deflection during insertion for guidance purposes. | Are engineering/navigation papers, not cancer prediction; do not relate needle geometry to core-level cancer outcomes; no patient-level inference. | Confirms needle trajectory is tracked; not in the prediction-from-geometry literature. |

### Searches completed

- PubMed: `"needle trajectory" OR "needle-to-target" prostate biopsy geometry distance "core level" cancer prediction OR outcome machine learning` → no direct hits matching our approach.
- PubMed: `"biopsy core" "target" "distance" prostate cancer positivity prediction logistic regression MRI fusion` → returned patient/lesion-level clinical prediction papers only.
- Google Scholar: `needle trajectory prostate biopsy core level cancer prediction geometry cluster robust` → returned Gayo 2022 (RL planning) and deflection papers.
- TCIA collection searches: no published study using the TCIA Prostate-MRI-US-Biopsy dataset with interpretable geometry features for core-level prediction found.

### Searches still needed before finalising novelty claim

1. PubMed: `"prostate biopsy" "core level" prediction geometry coordinate spatial`
2. PubMed: `"tracked biopsy" prostate "target hit" OR "target miss" prediction outcome`
3. Google Scholar: `"prostate biopsy core" "needle" "target" "distance" classification`
4. PubMed: `TCIA "Prostate-MRI-US-Biopsy" prediction model`
5. Semantic Scholar: keyword search "biopsy needle target geometry prostate cancer core"

### Recommended framing (provisional)

Use **Option B (cautious novelty)** until the remaining searches above are completed:

> "While prior computational work has addressed needle placement planning using biopsy-target geometry [Gayo et al. 2022] and has evaluated clinical and imaging predictors of per-lesion biopsy yield [CITATIONS], no published study has, to our knowledge, constructed an interpretable geometric feature set from core-level needle-to-target spatial coordinates, evaluated it against clinical-only and full-geometry reference feature sets on a held-out test split, and tested the resulting associations with patient-cluster-robust inference."

Upgrade to Option A (strong novelty) only after the remaining PubMed searches are completed and confirm no direct prior work.

---

## 6. Manuscript wording updates suggested

### Introduction §P1

**Current:** "[CITATION NEEDED]" ×5  
**Suggested:**
- Sentence 1: add [Sung2021]
- Sentence 2: add EAU or NICE guideline [TODO] or a landmark review
- Sentence 3: add [Kasivisvanathan2018] and [Rouvière2019]
- Sentence 4: [Kasivisvanathan2018] contains supporting data on per-patient variability; confirm whether it covers per-core variability or add a more specific reference
- Sentence 5: add [Hale et al. — TODO] for quantified registration/motion error

### Introduction §P2

**Current:** Two [CITATION NEEDED] blocks  
**Suggested:**
- Needle trajectory and coordinate tracking: cite [Sonn2013] for the Artemis system; [TCIA2020] as the data source
- Needle-to-target geometry operationalised computationally: cite [Gayo2022] as prior work showing this is feasible in a biopsy planning context

### Introduction §P3

**Current:** One [CITATION NEEDED] per category; novelty claim marked TODO  
**Suggested:**
- Category 1 (image-based): cite [Stanzione2020 — TODO verify] and a DL review [TODO choose]
- Category 2 (patient-level): cite [Roobol ERSPC — TODO] and/or [Ankerst PBCG — TODO]
- Novelty sentence: use cautious Option B wording (see §5 above) until remaining searches are complete

### Discussion §D6

**Current:** No specific calibration citation in outline paragraph text  
**Suggested:**
- Add [VanCalster2019] directly after "poorly calibrated" and after "ranking instrument rather than an absolute risk estimator"
- Add [Collins2024] or [Collins2015] after "without recalibration on an external prospective cohort"

### Discussion §D7

**Current:** No radiomics/DL review citations; relies on general assertion  
**Suggested:**
- Add [Stanzione2020 — TODO verify] after "radiomics pipelines"
- Add [DL review — TODO choose] after "deep learning and radiomics pipelines"
- Add [Varoquaux2022] after "would risk overfitting and would produce performance claims whose CIs would overlap substantially"

### Limitations (Discussion §D8)

**Current:** No methodological citations in most limitation bullets  
**Suggested:**
- Add [Collins2015] or [Collins2024] after limitation 2 (no external validation)
- Add [Wolff2019] after limitation 2 (PROBAST risk-of-bias framing)
- Add [Allsbrook/Epstein — TODO] after limitation 4 (Gleason grading inter-rater variability)
- Add [Liang1986] after limitation 5 (effective sample size closer to 796 patients than 16,992 cores)
- Confirm that PSA collinearity limitation is already supported by VIF analysis in the committed reports (no external citation needed)

---

## 7. Next searches (ordered by priority)

| Priority | Search | Goal |
|---------|--------|------|
| 1 | PubMed: `"prostate biopsy" "core level" geometry spatial coordinate prediction` | Complete novelty verification — most targeted search remaining |
| 2 | PubMed: `"TCIA" OR "Prostate-MRI-US-Biopsy" prediction model machine learning core` | Find any prior ML study using this specific dataset |
| 3 | PubMed: `Stanzione prostate MRI radiomics systematic review radiomic quality score` | Confirm PMID and DOI for Stanzione 2020 |
| 4 | PubMed: `Roobol ERSPC risk calculator prostate biopsy prediction` | Confirm ERSPC risk calculator citation (multiple papers exist; choose the canonical one) |
| 5 | PubMed: `Ankerst prostate biopsy collaborative group PBCG risk calculator` | Confirm PBCG risk calculator citation |
| 6 | PubMed: `Hale prostate MRI ultrasound registration error targeting accuracy` OR `prostate biopsy in vivo registration error measurement` | Find registration/targeting error measurement study for Intro §P1 |
| 7 | PubMed: `deep learning prostate cancer MRI clinically significant systematic review` — short-list the two most-cited reviews published 2020–2024 | Choose canonical DL review for Discussion §D7 |
| 8 | Springer catalogue or ISBN lookup: Steyerberg EW Clinical Prediction Models 2nd edition | Verify edition (2009 vs 2019), ISBN, publisher details |
| 9 | PubMed: `Allsbrook Epstein Gleason grading interobserver variability prostate` | Confirm Gleason inter-rater variability citation for Discussion §D8 |
| 10 | PubMed: `Lambin radiomics quality score RQS tool prostate OR cancer` — distinguish bridge paper (Nat Rev Clin Oncol 2017) from RQS tool paper | Confirm which Lambin paper to cite for Discussion §D7 |
