# Core Count Audit Summary

Per-patient biopsy core counts, stratified by endpoint and label group.  
**Spearman ρ** measures the association between number of cores and the patient label  
(positive = at least one core positive). A significant positive correlation would  
mean positive patients systematically receive more cores, which could mildly  
advantage core-count-sensitive aggregation methods (max-prob, top-3 mean).

| Endpoint | Median n_cores | Mean n_cores | Mean (positive pat.) | Mean (negative pat.) | Spearman ρ (vs label) | p-value | Interpretation |
|---|---|---|---|---|---|---|---|
| GG2+ / csPCa | 17.0 | 21.3 | 22.7 | 19.5 | 0.101 | 0.004 | Weak but significant: positive patients tend to have slightly more cores |
| GG3+ / high-grade | 17.0 | 21.3 | 20.5 | 21.7 | -0.050 | 0.158 | No significant association |

