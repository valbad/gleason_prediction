# Targeted Literature Review Matrix

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-07-07  
**Status:** Planning document for the human literature pass. No new analysis. No invented references. Bibliographic details marked TODO unless confirmed in committed reports.

---

## 1. Purpose

This file is used to:

1. Locate specific papers to fill all CITATION NEEDED placeholders in `reports/manuscript_outline_full.md` and `reports/manuscript_methods_results_draft.md`.
2. Verify the novelty claim in Introduction §P3 before committing to it.
3. Avoid overclaiming by confirming that cited papers actually support the stated claim rather than a weaker or stronger version of it.
4. Ensure that Discussion limitations and Discussion §D7 are grounded in real methodological literature rather than general assertions.

Sources used: `reports/manuscript_outline_full.md`, `reports/manuscript_methods_results_draft.md`, `reports/figure_and_table_captions.md`, `reports/main_paper_tables.md`.

---

## 2. Citation needs by manuscript paragraph

### Introduction P1

| Manuscript location | Claim needing support | Type of citation needed | Candidate papers / search terms | Priority | Notes |
|--------------------|-----------------------|------------------------|--------------------------------|----------|-------|
| Intro §P1, sentence 1 | Prostate cancer is among the most commonly diagnosed malignancies in men worldwide | Incidence / burden statistics | Global cancer statistics (Siegel et al. CA Cancer J Clin — TODO year; Ferlay et al. GLOBOCAN — TODO year) | High | Use most recent available edition. Check whether global or national (US) statistics are more appropriate for the target journal. |
| Intro §P1, sentence 2 | Multiparametric MRI has transformed prostate biopsy practice | Review / guideline endorsement | EAU guideline on prostate cancer (TODO year); NICE guideline (TODO year); mpMRI in prostate biopsy pathway review (TODO) | High | Need a citation that specifically states mpMRI changed the biopsy pathway, not just that mpMRI is useful. |
| Intro §P1, sentence 3 | MRI-targeted fusion biopsy improves detection of clinically significant prostate cancer (GG2+) vs systematic biopsy | RCT or high-quality prospective study | PRECISION trial: Kasivisvanathan et al. NEJM 2018 — TODO verify full citation; MRI-FIRST trial (Rouvière et al. Lancet Oncol — TODO); 4M trial (TODO) | High | The PRECISION trial is the canonical RCT. Confirm whether the claim applies to targeted vs systematic alone or combined strategies. |
| Intro §P1, sentence 4 | Biopsy yield remains variable across cores and patients | Observational or epidemiological study | Kasivisvanathan et al. 2018 may have supporting data; Elkhoury et al. (TODO — verify if this reference is real); series documenting core-level heterogeneity (TODO) | Medium | Need to confirm a citation that specifically documents variability in per-core positivity, not just per-patient detection rates. |
| Intro §P1, sentence 5 | Registration error, prostate motion, and targeting inaccuracies contribute to sampling variability | Technical / engineering study | Hale et al. (TODO — see §4 below); Cool et al. (TODO); rigid vs deformable registration review (TODO); motion during TRUS fusion biopsy (TODO) | High | This is a mechanistic claim and must be supported by real measurement data, not just stated as intuitive. |

---

### Introduction P2

| Manuscript location | Claim needing support | Type of citation needed | Candidate papers / search terms | Priority | Notes |
|--------------------|-----------------------|------------------------|--------------------------------|----------|-------|
| Intro §P2, sentence 1–2 | The three-dimensional spatial relationship between needle trajectory and lesion volume should determine whether the core samples cancer | Physical / mechanistic argument; possibly no direct citation if stated as reasoning, but supporting tracking literature is needed | Biopsy needle placement geometry; targeted biopsy tracking; template-guided biopsy targeting accuracy (TODO) | Medium | This sentence is a physical argument. If framed as "it is reasonable to expect," a citation may not be required. If framed as "evidence suggests," a tracking study is needed. Confirm framing before finalising. |
| Intro §P2, sentence 3 | Registered needle-to-target coordinates are available in MRI-US fusion biopsy systems | Technical description | MRI-US fusion biopsy system descriptions (Uronav, BK Fusion, Artemis — TODO); TCIA dataset descriptor (TODO) | Medium | The TCIA dataset itself provides this; a data descriptor or methods paper for the dataset would serve as a direct citation. |
| Intro §P2 | Needle trajectory / target hit / miss geometry has been described | Technical literature | Search: "prostate biopsy target hit" OR "needle trajectory MRI" OR "biopsy targeting error" (TODO) | Medium | Need to confirm whether any published study has described needle-to-target geometry quantitatively. This feeds directly into the novelty claim in §P3. |

---

### Introduction P3

| Manuscript location | Claim needing support | Type of citation needed | Candidate papers / search terms | Priority | Notes |
|--------------------|-----------------------|------------------------|--------------------------------|----------|-------|
| Intro §P3, category 1 | Image-based models use MRI radiomics or deep learning for prostate cancer prediction | Review or meta-analysis | Stanzione et al. (TODO — verify if this is a real radiomics review of prostate MRI); Litjens et al. (TODO — may refer to a DL in medical imaging review, not prostate specifically); prostate MRI DL systematic review (TODO) | High | Need a review, not individual model papers, to substantiate the category claim. |
| Intro §P3, category 2 | Patient-level clinical models use PSA, volume, and biopsy history | Review or model description | ERSPC risk calculator (Roobol et al. — TODO verify); Prostate Biopsy Collaborative Group (PBCG) risk calculator (Ankerst et al. — TODO verify); clinical prediction model review (TODO) | High | Both ERSPC and PBCG calculators are well-known; verify full citations and confirm they fit the claim. |
| Intro §P3, gap claim | Neither category explicitly models needle-to-target geometry at core level | Absence of literature — requires search confirmation | Search: "prostate biopsy core level prediction", "biopsy needle target geometry", "tracked prostate biopsy core outcome", "MRI-US fusion biopsy core model" (TODO) | High | **This claim is flagged as TODO in the manuscript outline. Must be verified before submission. See §5 below.** |
| Intro §P3, novelty claim | To our knowledge, no published study has constructed interpretable target-relative geometry features, evaluated on held-out core-level outcomes, and tested with cluster-robust inference | Absence of literature / novelty assertion | See §5 (Novelty claim verification plan) | High | Do not finalise this wording until §5 search tasks are completed. |

---

### Discussion §D2–§D3

| Manuscript location | Claim needing support | Type of citation needed | Candidate papers / search terms | Priority | Notes |
|--------------------|-----------------------|------------------------|--------------------------------|----------|-------|
| Discussion §D2 | Distance to target surface and fraction inside target operationalise targeting accuracy | Technical framing — physical argument | Targeting accuracy studies; tracking studies; core-level sampling theory (TODO) | Medium | If framed as geometric reasoning no citation may be needed, but a biopsy targeting accuracy paper would strengthen this. |
| Discussion §D3 | Compact models are preferred for interpretability and parsimony in clinical prediction | Methodological guidance | Steyerberg et al. clinical prediction models textbook (TODO); TRIPOD (Collins et al. — TODO); commentary on overfitting in small cohorts (TODO) | Medium | TRIPOD specifically discusses parsimony vs complexity tradeoffs. Steyerberg's textbook may also cover this. |
| Discussion §D3 | Overfitting risk in small medical imaging cohorts | Methodological critique | Maier-Hein et al. bioRxiv/MICCAI warnings on medical image analysis generalisation (TODO); Varoquaux et al. (TODO); radiomics quality scoring (Lambin et al. — TODO) | Medium | These are methodological papers; confirm whether they apply to tabular models or are specifically about imaging. |

---

### Discussion §D6

| Manuscript location | Claim needing support | Type of citation needed | Candidate papers / search terms | Priority | Notes |
|--------------------|-----------------------|------------------------|--------------------------------|----------|-------|
| Discussion §D6 | Calibration matters for clinical prediction models | Methodological / authoritative | Van Calster et al. (TODO — multiple calibration papers; confirm relevant one); Steyerberg et al. calibration review (TODO) | High | Van Calster has published extensively on calibration in clinical prediction. Confirm the specific paper most relevant to the Brier score / ECE framing used in our calibration analysis. |
| Discussion §D6 | Discrimination and calibration are distinct concepts; ranking ≠ absolute risk | Methodological | Van Calster and Vickers (TODO); Pencina et al. (TODO) | Medium | This is a methodological distinction that is well-established but should be cited for a clinical journal audience. |
| Discussion §D6 | External validation required before clinical use of prediction models | Guideline / reporting standard | TRIPOD (Collins et al. — TODO); PROBAST (Wolff et al. — TODO); Moons et al. (TODO) | High | TRIPOD and PROBAST are the canonical reporting and appraisal tools for clinical prediction models. Both should be cited in the Limitations section. |

---

### Discussion §D7

| Manuscript location | Claim needing support | Type of citation needed | Candidate papers / search terms | Priority | Notes |
|--------------------|-----------------------|------------------------|--------------------------------|----------|-------|
| Discussion §D7 | Prostate MRI radiomics and deep learning models exist and have been systematically reviewed | Systematic review | Stanzione et al. (TODO — verify if this covers prostate MRI radiomics specifically); Steuber et al. (TODO — verify); DL prostate MRI diagnostic accuracy review (TODO) | High | A systematic review or meta-analysis is preferred over individual model papers for this claim. |
| Discussion §D7 | High-dimensional MRI features require larger validation cohorts | Methodological critique | Radiomics quality scoring (RQS) by Lambin et al. (TODO); Varoquaux et al. on generalisation (TODO); Maier-Hein et al. (TODO) | Medium | The claim is that our study should not add high-dimensional MRI features on n=796 patients — this needs methodological support, not just assertion. |
| Discussion §D7 | Our study is not a deep learning study; geometric scalars are a distinct approach | Conceptual framing | No direct citation needed if stated clearly. If contrasted with specific papers, those papers need citations. | Low | This is a framing statement. A citation is needed only if a specific paper is contrasted directly. |

---

### Discussion §D8–§D9

| Manuscript location | Claim needing support | Type of citation needed | Candidate papers / search terms | Priority | Notes |
|--------------------|-----------------------|------------------------|--------------------------------|----------|-------|
| Discussion §D8 | Single-cohort retrospective design is a standard limitation | Methodological | TRIPOD / PROBAST; clinical epidemiology textbook (TODO) | Medium | State as limitation; TRIPOD guidance is the natural citation. |
| Discussion §D8 | Gleason grading has known inter-rater variability | Pathology | Allsbrook et al. (TODO); Epstein et al. (TODO); ISUP 2016 consensus (TODO) | Medium | Confirm whether inter-rater variability in Gleason grading is already cited in methods. If so, reuse. |
| Discussion §D8 | Intra-patient correlation must be accounted for in core-level analyses | Statistical methodology | Cluster-robust SE references (Liang and Zeger 1986 GEE paper — TODO); Rogers and Williams sandwich SE (TODO); statsmodels documentation is not a citable source | High | Liang and Zeger 1986 is the canonical GEE paper. The cluster-robust Huber-White SE also has standard statistical citations. These belong in the Methods section (§2.6) as well. |
| Discussion §D8 | PR-AUC is preferred over ROC-AUC for imbalanced classification | Statistical methodology | Saito and Rehmsmeier (2015 PLOS ONE — TODO verify); Davis and Goadrich (2006 — TODO verify) | Medium | This citation may also belong in the Methods section (§2.4) alongside the metric definition. |
| Discussion §D9 | Decision curve analysis for clinical utility assessment | Methodology | Vickers and Elkin (2006 — TODO verify); Van Calster et al. DCA (TODO) | Low | Mentioned as a future direction; brief citation in a sentence is sufficient. |

---

## 3. Seed search queries

Queries formatted for PubMed / Google Scholar / Semantic Scholar. Use Boolean AND/OR/NOT, field tags where indicated.

---

### Group 1 — MRI-targeted biopsy trials and guidelines

1. `("MRI-targeted biopsy" OR "MRI-guided biopsy") AND "clinically significant prostate cancer" AND (randomized OR "randomised") AND (detection OR diagnosis)`
2. `PRECISION trial prostate MRI biopsy NEJM`
3. `("MRI-US fusion biopsy" OR "magnetic resonance ultrasound fusion biopsy") AND "systematic biopsy" AND (detection OR comparison) AND prostate`
4. `mpMRI prostate biopsy pathway guideline EAU OR NICE OR AUA 2020 2021 2022 2023`
5. `"targeted biopsy" "Grade Group" OR "Gleason" AND prostate AND "randomized controlled trial"`

---

### Group 2 — MRI-US fusion biopsy registration and targeting error

1. `"MRI-US fusion" prostate biopsy "registration error" OR "targeting error" OR "deformable registration"`
2. `"elastic registration" OR "deformable registration" prostate biopsy MRI ultrasound accuracy`
3. `prostate motion "MRI-ultrasound" OR "MR-TRUS" biopsy tracking`
4. `"targeting accuracy" prostate biopsy MRI "needle placement" OR "core sampling"`
5. `"prostate biopsy" "registration" "in vivo" accuracy measurement displacement`

---

### Group 3 — Tracked biopsy / needle trajectory / target hit geometry

1. `"prostate biopsy" "needle trajectory" OR "needle placement" "target" geometry OR distance OR coordinate`
2. `"tracked biopsy" OR "tracking" prostate biopsy "MRI target" OR "target lesion" hit OR miss`
3. `"biopsy core" "spatial" OR "geometric" "target" prostate cancer prediction`
4. `"target hit" OR "target miss" prostate biopsy needle MRI`
5. `"needle-to-target" OR "needle to target" prostate biopsy geometry OR distance`

---

### Group 4 — Prostate MRI radiomics and deep learning reviews

1. `prostate MRI radiomics systematic review OR meta-analysis 2019 2020 2021 2022 2023`
2. `deep learning prostate cancer MRI detection OR diagnosis systematic review`
3. `"prostate MRI" "radiomics" "cancer detection" OR "cancer prediction" review`
4. `"convolutional neural network" prostate MRI biopsy OR "Gleason grade" systematic review`
5. `"radiomics quality score" OR "RQS" prostate imaging review`

---

### Group 5 — Clinical prediction models / prostate biopsy risk calculators

1. `"prostate biopsy" "risk calculator" OR "risk model" PSA volume "clinical prediction" external validation`
2. `"ERSPC risk calculator" prostate cancer biopsy`
3. `"Prostate Biopsy Collaborative Group" risk model OR calculator`
4. `"clinical prediction model" prostate biopsy cancer detection PSA density`
5. `"prostate cancer" "risk stratification" "biopsy" "pre-biopsy" model prediction`

---

### Group 6 — Calibration and prediction model reporting standards

1. `"calibration" "clinical prediction model" "Brier score" OR "calibration plot" OR "expected calibration error"`
2. `Van Calster calibration "clinical prediction" model`
3. `TRIPOD "transparent reporting" "prediction model" development validation`
4. `PROBAST "prediction model risk of bias" prostate OR cancer`
5. `"discrimination" "calibration" "distinction" "absolute risk" prediction model clinical`

---

### Group 7 — Core-level prostate biopsy prediction / TCIA Prostate-MRI-US-Biopsy dataset

1. `"TCIA" OR "The Cancer Imaging Archive" "Prostate-MRI-US-Biopsy" dataset`
2. `"prostate biopsy" "core level" prediction model machine learning OR logistic regression`
3. `"biopsy core" "positivity" "prediction" "logistic regression" prostate MRI OR clinical`
4. `"MRI-ultrasound fusion biopsy" dataset "core-level" OR "per-core" prediction`
5. `"prostate biopsy" "intra-patient" OR "within-patient" "clustering" OR "correlation" "statistical model"`

---

## 4. Preliminary candidate sources to verify

All entries below are **candidates only**. Mark verified once the full bibliographic record has been confirmed from the primary source.

| Candidate source | Why relevant | Manuscript section | Need to verify? | Notes |
|-----------------|--------------|--------------------|-----------------|-------|
| Kasivisvanathan et al. NEJM 2018 — PRECISION trial | RCT showing MRI-targeted biopsy detects more csPCa and fewer low-grade cancers vs systematic biopsy | Intro §P1 | Yes — confirm full citation: authors, journal, volume, page, DOI | Well-known trial; confirm claim aligns with our endpoint definition (GG2+ = csPCa) |
| Rouvière et al. Lancet Oncol — MRI-FIRST trial | Multicentre RCT; targeted + systematic vs systematic alone | Intro §P1 | Yes — confirm year, volume, DOI | TODO: confirm this is the MRI-FIRST paper specifically |
| Marks et al. — MRI-US fusion biopsy review | Review of MRI-US fusion biopsy technique and clinical utility | Intro §P1 | Yes — exact title, journal, year unknown. Search "Marks MRI ultrasound fusion biopsy" | Candidate / verify bibliographic details |
| Cool et al. — rigid vs elastic registration / cognitive fusion | Comparison of targeting strategies in MRI-guided biopsy | Intro §P1 / §P2 | Yes — author, title, year unconfirmed | Candidate / verify bibliographic details |
| Hale et al. — registration error / deformable fusion | Measures of in vivo prostate motion or registration error during fusion biopsy | Intro §P1 / §P5 (Discussion §D8) | Yes — author, title, year unconfirmed | Candidate / verify bibliographic details; could be replaced by a different targeting error paper |
| Stanzione et al. — prostate MRI radiomics review | Systematic review of radiomics for prostate cancer on MRI | Intro §P3 / Discussion §D7 | Yes — verify this is a real published review, confirm journal and year | Candidate / verify bibliographic details |
| Litjens et al. — deep learning in medical imaging | General review of DL in medical imaging; possibly prostate-specific section | Intro §P3 / Discussion §D7 | Yes — this may be Litjens et al. IEEE TMI 2017 overview; confirm whether prostate biopsy is covered | TODO: determine whether a more prostate-specific DL review is preferable |
| Roobol et al. — ERSPC risk calculator | Patient-level pre-biopsy risk model using PSA, DRE, prior biopsy | Intro §P3 | Yes — confirm the specific ERSPC calculator paper; multiple papers exist | Candidate / verify bibliographic details |
| Ankerst et al. — PBCG risk calculator | Patient-level clinical risk model for prostate biopsy | Intro §P3 | Yes — confirm full citation (Prostate Biopsy Collaborative Group, year, journal) | Candidate / verify bibliographic details |
| TRIPOD — Collins et al. | Reporting guideline for clinical prediction models (development and validation) | Methods §2.4 / Discussion §D6, §D8 | Yes — confirm: Collins et al. Ann Intern Med 2015 is the likely citation | Well-known guideline; confirm BMJ vs Ann Intern Med version |
| PROBAST — Wolff et al. | Risk-of-bias tool for clinical prediction model studies | Discussion §D8 | Yes — confirm: Wolff et al. Ann Intern Med 2019 | Candidate / verify bibliographic details |
| Van Calster et al. — calibration papers | Methodological work on calibration in clinical prediction models | Discussion §D6 | Yes — multiple Van Calster papers exist; identify the most relevant one for Brier score / calibration curves | TODO: determine which Van Calster paper to cite (2016 Stat Med? 2019 BMJ?) |
| Steyerberg — Clinical Prediction Models textbook | Reference for parsimony, calibration, and model evaluation in clinical prediction | Discussion §D3, §D6 | Yes — Steyerberg EN, Springer 2009 or updated edition; confirm edition and page range if citing for specific claim | Standard reference; confirm edition |
| Saito and Rehmsmeier PLOS ONE 2015 | Justification for using PR-AUC over ROC-AUC for imbalanced classification | Methods §2.4 / Discussion §D8 | Yes — confirm: Saito T, Rehmsmeier M. PLoS ONE. 2015;10(3):e0118432 | Candidate / verify bibliographic details |
| Davis and Goadrich 2006 — precision-recall curves | Early work justifying PR-AUC for imbalanced data | Methods §2.4 | Yes — confirm: Davis J, Goadrich M. ICML 2006 | Candidate / verify bibliographic details; conference paper, not journal |
| Liang and Zeger 1986 — GEE | Canonical GEE paper used to support cluster-robust inference methodology | Methods §2.6 | Yes — confirm: Liang KY, Zeger SL. Biometrika. 1986;73(1):13–22 | Well-known; confirm citation details |
| TCIA Prostate-MRI-US-Biopsy data descriptor | Dataset paper for the public dataset used in this study | Methods §2.1 | Yes — search for a published data descriptor or cohort paper for this TCIA collection | TODO: confirm whether a peer-reviewed data descriptor paper exists for this specific TCIA collection |
| Lambin et al. — Radiomics Quality Score (RQS) | Methodological critique / scoring tool for radiomics study quality | Discussion §D7 | Yes — Lambin et al. Eur J Cancer 2017 or similar; confirm | Candidate / verify bibliographic details |
| Vickers and Elkin 2006 — Decision Curve Analysis | DCA methodology for assessing clinical utility of prediction models | Discussion §D9 | Yes — confirm: Vickers AJ, Elkin EB. Med Decis Making. 2006;26(6):565–574 | Candidate / verify bibliographic details |
| Any needle-to-target geometry / tracked prostate biopsy paper found in search | If found during Group 3 search, becomes the closest prior work — critical for novelty claim | Intro §P3 / Discussion | TODO — must be searched | If such a paper exists, it changes the novelty framing; see §5 |

---

## 5. Novelty claim verification plan

The current Introduction §P3 contains the following claim, marked TODO for verification:

> "To our knowledge, no published study has (i) constructed an interpretable geometric feature set from needle-to-target spatial coordinates, (ii) evaluated it against clinical-only and full-geometry reference sets on a held-out test split, and (iii) tested the resulting associations with cluster-robust inference accounting for the non-independence of multiple cores per patient."

This claim is **strong** and must be verified before submission. It touches three separable sub-claims. The following search tasks address each.

---

### Task 5.1 — Search for papers using biopsy-core geometry / needle-to-target distance as a predictor

**Search queries:**
- `"needle trajectory" OR "needle placement" "prostate biopsy" "target" "distance" OR "geometry" prediction`
- `"biopsy core" "target" "distance" prostate cancer grade OR Gleason prediction`
- `"tracked biopsy" "prostate" "target lesion" "prediction" OR "outcome"`
- `"MRI-US fusion" "biopsy" "spatial" OR "geometric" feature cancer detection`

**What to record:** author, year, journal, whether they use needle-to-target geometry as a feature, whether they have a held-out test set, whether they address intra-patient clustering.

---

### Task 5.2 — Search TCIA Prostate-MRI-US-Biopsy dataset papers

**Search queries:**
- `"Prostate-MRI-US-Biopsy" TCIA`
- `TCIA prostate biopsy dataset MRI ultrasound fusion prediction`
- `"National Cancer Institute" prostate MRI biopsy dataset machine learning`

**What to record:** whether prior published studies using this dataset have built core-level prediction models; what features they used.

---

### Task 5.3 — Search biopsy target miss / targeting accuracy literature

**Search queries:**
- `"target miss" OR "target hit" prostate biopsy MRI "per-core" OR "core-level"`
- `"targeting accuracy" "MRI-guided biopsy" "prostate" geometry distance`
- `"biopsy needle" "MRI target" "sampling" prostate hit OR miss rate`

**What to record:** whether any paper quantifies needle-to-target distance and links it to core positivity.

---

### Task 5.4 — Search radiomics papers that use biopsy coordinate labels

**Search queries:**
- `"prostate MRI" radiomics "biopsy core" label OR annotation coordinate`
- `"per-core" OR "core-level" prostate cancer prediction MRI features`

**What to record:** whether these papers use spatial coordinates of biopsy sites (which would be related to, but distinct from, our geometric features).

---

### Task 5.5 — Search needle placement optimisation / RL papers

**Search queries:**
- `"prostate biopsy" "needle placement" optimisation OR optimization reinforcement learning`
- `"biopsy planning" prostate MRI target lesion needle trajectory`

**What to record:** whether optimisation papers operationalise needle-to-target geometry in a way that overlaps with our approach.

---

### Novelty framing decision

After completing Tasks 5.1–5.5, choose one of the following framings:

| Option | Use when | Wording |
|--------|----------|---------|
| **A. Strong novelty** | No paper found using needle-to-target geometry as a predictor at core level with cluster-robust inference | "To our knowledge, no published study has…" (current wording, after verification) |
| **B. Cautious novelty** | Papers exist that use related spatial features or coordinate labels, but not interpretable geometric features evaluated on a held-out test split with cluster-robust inference | "While prior work has used biopsy coordinate labels [REF] or spatial proximity measures [REF], no published study has, to our knowledge, combined these into a compact interpretable feature set, evaluated it on a held-out test split, and tested associations with patient-cluster-robust inference." |
| **C. Contribution relative to prior geometry literature** | A direct prior work exists that is close to our approach | Cite the prior work, state how our contribution extends it (held-out evaluation, cluster-robust inference, compact vs full comparison, different feature formulation) |

Record the decision and update Introduction §P3 accordingly before manuscript submission.

---

## 6. Output checklist for the human literature pass

- [ ] Fill each CITATION NEEDED placeholder in `reports/manuscript_outline_full.md` (Introduction §§P1–P3; Discussion §§D2–D3, D6–D9).
- [ ] Fill each CITATION NEEDED placeholder in `reports/manuscript_methods_results_draft.md` (Methods §§2.1, 2.4, 2.6; Results §3.7).
- [ ] Add verified BibTeX or Zotero entries for all confirmed citations.
- [ ] Complete novelty verification tasks §§5.1–5.5 and record findings.
- [ ] Select novelty framing (A, B, or C from §5) and update Introduction §P3 wording.
- [ ] Identify 2–3 closest prior works to cite directly in Introduction §P3 or Discussion §D7, regardless of novelty outcome.
- [ ] Confirm TRIPOD and PROBAST are cited in Methods and in Discussion §D8 Limitations.
- [ ] Confirm GEE / cluster-robust SE citation in Methods §2.6.
- [ ] Confirm PR-AUC justification citation in Methods §2.4.
- [ ] Confirm TCIA dataset paper exists and cite in Methods §2.1.
- [ ] Confirm Gleason inter-rater variability citation if used in Discussion §D8.
- [ ] Remove or update all remaining CITATION NEEDED and TODO flags before journal submission.
- [ ] Do not cite weak or tangentially relevant papers for central claims (incidence statistics, PRECISION trial, calibration methodology).
- [ ] Do not fabricate DOIs, volume numbers, or page ranges. If uncertain, search PubMed and copy the official citation.
