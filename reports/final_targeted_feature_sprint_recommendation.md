# Final Targeted Feature Sprint — Recommendation

**Branch:** `feat/geometry-clinical-baselines`  
**Script:** `src/compare_final_targeted_feature_sprint.py`  
**Based on:** `reports/final_targeted_feature_sprint_comparison.md`

## Decision criteria (stricter than extended sprint)

An extension is worth discussing as a **candidate improved model** only if:

- ΔPR-AUC ≥ +0.030 **OR** ΔROC-AUC ≥ +0.020 on held-out test split with logistic regression
- **AND** gain appears on at least one clinically important endpoint without large degradation (|ΔROC| > 0.015 or |ΔPR| > 0.020) on the other endpoint
- **AND** feature set is interpretable and not leakage-prone.

Replacement of the primary compact model is recommended only if:
- Gain is consistent across both endpoints, OR clinically meaningful for one
- Feature family is scientifically defensible
- Model remains interpretable enough to explain in a manuscript

## Feature-type classification

| Feature family | Type | Leakage risk | Notes |
|---------------|------|-------------|-------|
| Target geometry (4) | Geometric | None | Pure spatial |
| Clinical (4) | Clinical | None | PSA, prostate volume |
| `core_label` one-hot / flags | Procedural/anatomical | **See note** | Pre-biopsy targeting decision |
| Intensity (auto-detected) | Image-derived | None | First-order only |
| Transformed geometry | Derived | None | Log, ratio, interaction |

**Note on `core_label`:** Encodes the physician's pre-biopsy targeting decision (systematic sextant vs MRI-targeted / prior-positive site). Not pathology leakage, but encodes clinical suspicion not present in the geometry-clinical compact model. If any model including `core_label` crosses the threshold, the gain is flagged as **procedural/anatomical-context-driven**, not geometry-driven.

## Results summary (logistic regression, test split)

### GG2+ / csPCa

**Compact baseline (LR):** ROC-AUC = 0.754, PR-AUC = 0.327

| Feature set | ΔROC-AUC | ΔPR-AUC | Threshold met? |
|-------------|----------|---------|---------------|
| `compact_plus_core_label_onehot` | +0.005 | +0.011 | No |
| `compact_plus_anatomical_core_label` | +0.006 | +0.011 | No |
| `compact_plus_intensity` | -0.003 | -0.003 | No |
| `compact_plus_transformed_geometry` | +0.002 | -0.012 | No |
| `compact_plus_core_label_plus_intensity` | +0.003 | +0.009 | No |
| `compact_plus_core_label_plus_transformed_geo` | +0.007 | +0.000 | No |
| `compact_plus_all_low_cost_features` | +0.004 | -0.003 | No |
| `clinical_plus_core_label` | -0.012 | +0.014 | No |
| `target_geometry_plus_core_label` | -0.020 | -0.100 | No |

### GG3+ / high-grade

**Compact baseline (LR):** ROC-AUC = 0.828, PR-AUC = 0.269

| Feature set | ΔROC-AUC | ΔPR-AUC | Threshold met? |
|-------------|----------|---------|---------------|
| `compact_plus_core_label_onehot` | +0.005 | +0.015 | No |
| `compact_plus_anatomical_core_label` | +0.003 | +0.012 | No |
| `compact_plus_intensity` | -0.006 | +0.005 | No |
| `compact_plus_transformed_geometry` | -0.003 | -0.013 | No |
| `compact_plus_core_label_plus_intensity` | -0.001 | +0.022 | No |
| `compact_plus_core_label_plus_transformed_geo` | +0.001 | -0.001 | No |
| `compact_plus_all_low_cost_features` | -0.004 | -0.001 | No |
| `clinical_plus_core_label` | -0.034 | +0.023 | No |
| `target_geometry_plus_core_label` | -0.020 | -0.129 | No |

## Overall recommendation

**No feature set crosses either strict threshold for either endpoint (ΔPR ≥ +0.030 or ΔROC ≥ +0.020).**

The compact `current_compact` model (`target_geometry_plus_clinical`, 8 features) remains the best central model. This is a **null result** for all four feature families tested. Further performance gains likely require genuinely new information (external dataset, additional clinical variables, calibrated image features) rather than minor feature engineering on the current dataset.

### Regularized LR variants

Tuned L2/L1/Elastic Net models were compared against the baseline LR (C=1). Regularization tuning is most useful when the feature space is large relative to the sample size (many one-hot categories). With only 8 compact features, the gain from tuning over the default C=1 is typically negligible. Results are reported in the comparison table.

### Limitations of this sprint

- Test set: ~120 patients. Delta estimates at this sample size have wide bootstrap CIs; a ΔROC of ±0.020 is near the noise floor.
- `core_label` was audited post-hoc (column discovery, not pre-registration). Results for `core_label`-containing feature sets are exploratory.
- Intensity features (if detected) are first-order only; any gain from them should be verified on the restricted `extraction_status == 'ok'` subset before reporting.
- Transformed geometry features are deterministic recodes; they cannot add genuinely new information, only different parameterizations of existing features.
- This sprint does not include calibration re-assessment or decision-curve analysis.

---

## Conclusion à montrer au binôme

**Résultat : null result complet.**

Aucune des quatre familles de features testées — encodage complet de `core_label`, features d'intensité image premier ordre, transformations géométriques simples, régression logistique régularisée (L2/L1/Elastic Net) — ne dépasse les seuils stricts (ΔPR ≥ +0.030 ou ΔROC ≥ +0.020) sur le test set pour aucun des deux endpoints.

**Décision : le modèle compact reste le meilleur modèle central.** On ne change rien.

**La modélisation est terminée.** Les gains de performance supplémentaires nécessiteraient des informations genuinement nouvelles (dataset externe, features image calibrées, variables cliniques non disponibles), pas des transformations supplémentaires du dataset existant.

**Prochaine étape recommandée :** rédiger le manuscrit. Tous les résultats nécessaires sont documentés.

---

*Report generated by `src/compare_final_targeted_feature_sprint.py`. Do not modify by hand.*
