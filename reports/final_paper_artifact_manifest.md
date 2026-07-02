# Final Paper Artifact Manifest

**Branch:** `feat/geometry-clinical-baselines`  
**Date:** 2026-07-02

---

## 1. Current project status

The geometry-clinical baseline phase is complete enough to support paper drafting. All core analyses are run and committed: core-level compact model evaluation (held-out ROC/PR-AUC, risk stratification), compact vs full model comparison, grouped core-level inference with cluster-robust standard errors and GEE, and patient-level direct modelling as a secondary analysis. Four figure scripts exist and four main-paper figures can be generated from committed CSVs without re-running any model. The paper skeleton and results narrative are drafted in `reports/paper_skeleton_and_results_narrative.md`. The remaining work before manuscript writing is copyediting, caption drafting, and table formatting — not new analysis.

---

## 2. Main-paper figures

| Figure | File(s) | Generating script | Main message | Status |
|--------|---------|-------------------|--------------|--------|
| Figure 1 — Geometry schematic | `figures/figure1_geometry_schematic.png` / `.pdf` | `src/make_geometry_schematic_figure.py` | Schematic of target-relative biopsy geometry features: distance to surface, fraction inside target, centroid distance, trajectory intersection. | Generated and committed |
| Figure 2 — Compact model performance | `figures/figure2_compact_model_performance.png` / `.pdf` | `src/make_compact_model_results_figures.py` | Held-out ROC-AUC and PR-AUC for `target_geometry_plus_clinical` logistic regression; patient-level bootstrap CIs. | Generated and committed |
| Figure 3 — Risk stratification enrichment | `figures/figure3_risk_stratification_enrichment.png` / `.pdf` | `src/make_compact_model_results_figures.py` | Highest-scored biopsy cores are enriched for positive cores relative to baseline prevalence (top-5% enrichment labeled above bars). | Generated and committed |
| Figure 4 — Forest plot | `figures/figure4_grouped_inference_forest_plot.png` / `.pdf` | `src/make_grouped_inference_forest_plot.py` | Cluster-robust odds ratios for all compact model features; both endpoints; geometry features shaded; significance markers. | Generated and committed |

**To verify figure reproducibility, rerun:**
```bash
python src/make_geometry_schematic_figure.py
python src/make_compact_model_results_figures.py
python src/make_grouped_inference_forest_plot.py
```

---

## 3. Candidate supplementary figures

| Candidate | Description | Status |
|-----------|-------------|--------|
| Supp Fig S1 — Compact vs full delta plot | Bar chart of ΔROC-AUC and ΔPR-AUC between compact and full model across endpoints and splits. Data in `reports/minimal_vs_full_delta_table.md`. | Candidate / optional — data exists, script not yet written |
| Supp Fig S2 — Calibration plot | Reliability/calibration curve for compact LR model on test split. Data in `reports/final_calibration_table.md`. | Candidate / optional — data exists, script not yet written |
| Supp Fig S3 — Patient-level model comparison | ROC-AUC comparison for patient-level aggregation approaches. Data in `reports/patient_level_direct_model_comparison.md`. | Candidate / optional — would replace or supplement the table |
| Supp Fig S4 — CNN/GNN historical comparison | Summary of prior CNN/GNN results from earlier project phase for context. | Do not generate now — belongs in discussion framing, not results |

---

## 4. Main-paper tables

| Table | Source report | Main message | Status |
|-------|--------------|--------------|--------|
| Table 1 — Dataset overview | `reports/final_core_count_audit_summary.md` | N cores, N patients, train/val/test split sizes, endpoint prevalences. | Generated |
| Table 2 — Compact model held-out performance | `reports/final_main_performance_table.md` | ROC-AUC and PR-AUC with 95% CIs for compact LR on GG2+ and GG3+. | Generated |
| Table 3 — Compact vs full model comparison | `reports/minimal_vs_full_model_comparison.md` + `reports/minimal_vs_full_delta_table.md` | |ΔROC| ≤ 0.010, |ΔPR| ≤ 0.015 equivalence margins met; 8-feature compact set ≈ 14-feature full model. | Generated |
| Table 4 — Risk stratification | `reports/final_risk_stratification_table.md` + `reports/minimal_vs_full_risk_stratification.md` | Prevalence in top 5/10/20% of scored cores vs baseline; enrichment ratios. | Generated |
| Table 5 — Grouped inference odds ratios | `reports/grouped_core_level_inference.md` + `reports/grouped_core_level_inference_cluster_robust.csv` | Cluster-robust OR and 95% CI for all compact features; both endpoints. Summary of significant associations (distance to surface, fraction inside, log PSA, prostate volume). | Generated |
| Table 6 — Patient-level secondary analysis | `reports/final_patient_level_table.md` + `reports/patient_level_direct_model_comparison.md` | Patient-level aggregation performance; context that patient-level is harder and secondary. | Generated |

---

## 5. Supplementary reports/tables

| Artifact | Purpose | Keep or optional | Notes |
|----------|---------|-----------------|-------|
| `reports/final_univariate_effects_table.md` | Univariate OR for each feature individually; complements multivariable Table 5. | Keep | Good for supplementary |
| `reports/final_feature_importance_table.md` | Feature importance rankings from models across feature sets. | Keep | Supplementary |
| `reports/final_calibration_table.md` | Calibration metrics (Brier score, ECE) for compact LR. | Keep | Supplementary — brief methods note suffices if no calibration figure |
| `reports/grouped_core_level_inference.md` | Full inference report (cluster-robust + GEE + VIF + correlations + interpretation). | Keep | Primary source for Table 5; useful as internal reference |
| `reports/geometry_baseline_tracking_note.md` | Phase-by-phase tracking note (Phases 1–3) documenting all methodological decisions. | Keep internal | Not for paper; reference document for authors |
| `reports/paper_skeleton_and_results_narrative.md` | Drafted 12-section paper outline with results narrative. | Keep internal | Working draft for Methods/Results/Discussion |
| `reports/minimal_vs_full_delta_table.md` | ΔROC / ΔPR comparison across all model × feature-set combinations. | Keep | Source for Table 3 and potential Supp Fig S1 |
| `reports/patient_level_direct_model_comparison.md` | Patient-level LR, RF, XGBoost across aggregation strategies. | Keep | Source for Table 6 |

---

## 6. Reproducibility map

| Script | Primary inputs | Primary outputs | Purpose |
|--------|---------------|-----------------|---------|
| `src/compare_minimal_vs_full_models.py` | Dataset CSV | `reports/minimal_vs_full_model_comparison.*`, `reports/minimal_vs_full_delta_table.*`, `reports/minimal_vs_full_risk_stratification.*` | Core-level compact vs full model comparison; equivalence test; risk stratification |
| `src/analyze_patient_level_direct_models.py` | Dataset CSV | `reports/patient_level_direct_model_comparison.*` | Patient-level aggregation models (secondary analysis) |
| `src/analyze_grouped_core_level_inference.py` | Dataset CSV | `reports/grouped_core_level_inference.md`, `reports/grouped_core_level_inference_cluster_robust.csv`, `reports/grouped_core_level_inference_gee.csv`, `reports/grouped_core_level_inference_vif.csv`, `reports/grouped_core_level_inference_correlations.csv` | Cluster-robust LR, GEE, VIF, Spearman correlations; patient-grouped inference |
| `src/make_geometry_schematic_figure.py` | None (deterministic) | `figures/figure1_geometry_schematic.png/.pdf` | Figure 1: conceptual schematic |
| `src/make_compact_model_results_figures.py` | `reports/minimal_vs_full_model_comparison.csv`, `reports/minimal_vs_full_risk_stratification.csv` | `figures/figure2_compact_model_performance.png/.pdf`, `figures/figure3_risk_stratification_enrichment.png/.pdf` | Figures 2–3: compact model performance and enrichment |
| `src/make_grouped_inference_forest_plot.py` | `reports/grouped_core_level_inference_cluster_robust.csv` | `figures/figure4_grouped_inference_forest_plot.png/.pdf` | Figure 4: forest plot |

---

## 7. Main vs supplementary decision

**Main paper:**
- Figure 1: geometry schematic (Methods)
- Figure 2: compact model ROC/PR performance (Results — primary)
- Figure 3: risk stratification enrichment (Results — clinical relevance)
- Figure 4: forest plot for grouped inference (Results — methodological robustness)
- Table 1: dataset overview (Methods/Results)
- Table 2: compact model performance (Results — primary)
- Table 3: compact vs full model equivalence (Results)
- Table 4: risk stratification (Results)
- Table 5: grouped inference ORs (Results)
- Table 6: patient-level secondary (Results — brief)

**Supplementary:**
- Supp Table S1: univariate effects (`final_univariate_effects_table.md`)
- Supp Table S2: feature importance (`final_feature_importance_table.md`)
- Supp Table S3: calibration metrics (`final_calibration_table.md`)
- Supp Table S4: full delta table (`minimal_vs_full_delta_table.md`)
- Supp Fig S1: compact vs full delta plot (candidate — if space permits)
- Supp Fig S2: calibration plot (candidate — if space permits)
- Supp Fig S3: patient-level comparison plot (candidate — replaces or accompanies Table 6)

**Not in paper:**
- `geometry_baseline_tracking_note.md` — internal reference only
- `paper_skeleton_and_results_narrative.md` — working draft only
- GEE and VIF detail tables — narrative summary in Methods suffices

---

## 8. Remaining gaps before writing

### A. Essential

1. **Figure captions (Figures 1–4):** Draft one clear, self-contained caption per figure. Currently no captions exist outside the narrative in `paper_skeleton_and_results_narrative.md`.
2. **Main table captions and column headers:** Tables 1–6 need short captions and standardised column names ready for the target journal format.
3. **Short Methods text:** Two paragraphs on (a) core-level modelling and feature set, and (b) grouped inference rationale and implementation. The skeleton narrative exists; it needs to be condensed into manuscript-style prose.
4. **Reproducibility check:** Confirm that running the three figure scripts and three analysis scripts from a clean state regenerates all committed artifacts. Document any missing seed or environment dependency.

### B. Useful but not essential

- **Supp Fig S1 — compact vs full delta plot:** Exists as table data; a figure would be clearer than the table for reviewers comparing across model families.
- **Supp Fig S2 — calibration curve:** Good practice for clinical-adjacent work; brief script from `final_calibration_table.md`.
- **Supp Fig S3 — patient-level comparison:** Makes Table 6 easier to read at a glance; patient-level section is short enough that a figure helps.
- **Cleaner CNN/GNN historical summary:** One table or figure summarising the prior CNN/GNN result (if available) for the Discussion framing; should not require re-training.

### C. Do not do now

- New CNN/GNN training or architecture search.
- Radiomics texture feature expansion.
- Multi-instance learning or deep patient-level models.
- External dataset validation (not yet available).
- Threshold optimisation or clinical decision analysis.
- Any analysis not directly supporting the compact geometry-clinical model framing.

---

## 9. Recommended next concrete action

**Draft final figure captions for Figures 1–4 and main table captions (Tables 1–6).**

One paragraph per figure (2–4 sentences: what the figure shows, what the key result is, what the take-away is). One sentence per table (what is shown, over what data subset). Write these captions into a new file: `reports/figure_and_table_captions.md`.
