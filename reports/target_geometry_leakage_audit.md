# Target Geometry Leakage Audit

Assesses whether target-derived geometric features encode legitimate
MRI-based predictors or may leak pathology label information.

**Dataset:** `data/share/needle_features_v1.csv`  
**Filter:** `label_join_status == 'coord_match'`, valid label, valid split  
**Labels audited:** `binary_label_int` (GG2+) and `binary_label_gg3plus_int` (GG3+)

## 1. Feature-set Hygiene

### 1a. Target geometry features (from `run_shareable_tabular_experiments.py`)

| # | Column | Present in dataset |
|---|---|---|
| 1 | `target_mesh_available` | ✓ |
| 2 | `distance_midpoint_to_target_centroid_mm` | ✓ |
| 3 | `distance_midpoint_to_target_surface_mm` | ✓ |
| 4 | `trajectory_intersects_target` | ✓ |
| 5 | `approximate_fraction_of_centerline_inside_target` | ✓ |

### 1b. Overlap with pathology / label columns

Pathology columns checked: 11  
Overlap with target geometry features: **NONE — OK**

> **Section result: PASS**

## 2. Patient-level Split Integrity

Each `patient_number` must appear in exactly one of train / val / test.

| Split | Unique patients |
|---|---|
| train | 557 |
| val | 119 |
| test | 120 |

| Pair | Shared patients | Result |
|---|---|---|
| train / val | 0 | PASS |
| train / test | 0 | PASS |
| val / test | 0 | PASS |

> **Section result: PASS**

## 3. Descriptive Statistics — `binary_label_int` (GG2+ / csPCa (Gleason ≥ 3+4=7))

Statistics computed on coord_match rows with valid label and split.

| Feature | Label | N | Mean | Std | P25 | Median | P75 | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | 0 | 15,093 | 16.987 | 10.661 | 7.613 | 15.581 | 25.067 | 0.435 | 56.381 |
| `distance_midpoint_to_target_centroid_mm` | 1 | 1,873 | 9.764 | 7.394 | 4.657 | 7.372 | 12.192 | 0.624 | 48.305 |
| `distance_midpoint_to_target_surface_mm` | 0 | 15,093 | 12.016 | 10.084 | 2.628 | 10.130 | 19.292 | 0.001 | 52.665 |
| `distance_midpoint_to_target_surface_mm` | 1 | 1,873 | 4.770 | 6.260 | 0.931 | 2.384 | 5.635 | 0.000 | 46.294 |
| `trajectory_intersects_target` | 0 | 14,816 | 0.273 | 0.445 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| `trajectory_intersects_target` | 1 | 1,779 | 0.639 | 0.481 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| `approximate_fraction_of_centerline_inside_target` | 0 | 14,816 | 0.081 | 0.166 | 0.000 | 0.000 | 0.080 | 0.000 | 1.000 |
| `approximate_fraction_of_centerline_inside_target` | 1 | 1,779 | 0.253 | 0.271 | 0.000 | 0.180 | 0.420 | 0.000 | 1.000 |

## 4. Positive Rates by Geometric Condition — `binary_label_int` (GG2+ / csPCa (Gleason ≥ 3+4=7))

For each condition, counts and positive rate of `binary_label_int==1` are shown for rows where the feature is not NaN.

| Condition | N (condition met) | N positive | Prevalence | N (condition not met) | N positive | Prevalence |
|---|---|---|---|---|---|---|
| trajectory_intersects_target == True | 5,179 | 1,136 | 21.9% | 11,416 | 643 | 5.6% |
| trajectory_intersects_target == False | 11,416 | 643 | 5.6% | 5,179 | 1,136 | 21.9% |
| fraction_inside_target > 0 | 5,179 | 1,136 | 21.9% | 11,416 | 643 | 5.6% |
| fraction_inside_target > 0.25 | 2,889 | 762 | 26.4% | 13,706 | 1,017 | 7.4% |
| fraction_inside_target > 0.5 | 847 | 318 | 37.5% | 15,748 | 1,461 | 9.3% |
| distance_to_target_surface_mm <= 0 (inside target) | 0 | 0 | n/a | 16,966 | 1,873 | 11.0% |
| distance_to_target_surface_mm <= 2 mm | 4,047 | 854 | 21.1% | 12,919 | 1,019 | 7.9% |
| distance_to_target_surface_mm <= 5 mm | 6,536 | 1,355 | 20.7% | 10,430 | 518 | 5.0% |

## 5. `target_mesh_available` Association — `binary_label_int` (GG2+ / csPCa (Gleason ≥ 3+4=7))

If MRI targets are defined independently of pathology (e.g. from PI-RADS ≥ 3
lesions), `target_mesh_available` should have moderate but not near-perfect
association with the label. A very high positive rate when mesh=True AND a
near-zero rate when mesh=False would be a strong leakage signal.

**`target_mesh_available = True (mesh present)`**

- Rows: 16,966
- Positives (`binary_label_int==1`): 1,873
- Prevalence: 11.0%

**`target_mesh_available = False (no mesh)`**

- Rows: 26
- Positives (`binary_label_int==1`): 14
- Prevalence: 53.8%

**Reverse view: mesh rate by label**

| Label | N | Has target mesh | Mesh rate |
|---|---|---|---|
| 0 (negative) | 15,105 | 15,093 | 99.9% |
| 1 (positive) | 1,887 | 1,873 | 99.3% |

> **Small difference (< 10 pp):** mesh availability is similarly distributed across labels, consistent with targets being defined independently of pathology outcome.

## 3. Descriptive Statistics — `binary_label_gg3plus_int` (GG3+ / high-grade (Gleason ≥ 4+3=7))

Statistics computed on coord_match rows with valid label and split.

| Feature | Label | N | Mean | Std | P25 | Median | P75 | Min | Max |
|---|---|---|---|---|---|---|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | 0 | 16,333 | 16.438 | 10.639 | 7.115 | 14.626 | 24.479 | 0.435 | 56.381 |
| `distance_midpoint_to_target_centroid_mm` | 1 | 633 | 9.788 | 6.800 | 4.835 | 7.702 | 12.462 | 0.624 | 39.466 |
| `distance_midpoint_to_target_surface_mm` | 0 | 16,333 | 11.493 | 10.038 | 2.294 | 9.253 | 18.611 | 0.001 | 52.665 |
| `distance_midpoint_to_target_surface_mm` | 1 | 633 | 4.074 | 5.098 | 0.934 | 2.277 | 4.685 | 0.000 | 34.812 |
| `trajectory_intersects_target` | 0 | 16,014 | 0.298 | 0.457 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| `trajectory_intersects_target` | 1 | 581 | 0.695 | 0.461 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| `approximate_fraction_of_centerline_inside_target` | 0 | 16,014 | 0.093 | 0.180 | 0.000 | 0.000 | 0.120 | 0.000 | 1.000 |
| `approximate_fraction_of_centerline_inside_target` | 1 | 581 | 0.292 | 0.282 | 0.000 | 0.240 | 0.500 | 0.000 | 1.000 |

## 4. Positive Rates by Geometric Condition — `binary_label_gg3plus_int` (GG3+ / high-grade (Gleason ≥ 4+3=7))

For each condition, counts and positive rate of `binary_label_gg3plus_int==1` are shown for rows where the feature is not NaN.

| Condition | N (condition met) | N positive | Prevalence | N (condition not met) | N positive | Prevalence |
|---|---|---|---|---|---|---|
| trajectory_intersects_target == True | 5,179 | 404 | 7.8% | 11,416 | 177 | 1.6% |
| trajectory_intersects_target == False | 11,416 | 177 | 1.6% | 5,179 | 404 | 7.8% |
| fraction_inside_target > 0 | 5,179 | 404 | 7.8% | 11,416 | 177 | 1.6% |
| fraction_inside_target > 0.25 | 2,889 | 286 | 9.9% | 13,706 | 295 | 2.2% |
| fraction_inside_target > 0.5 | 847 | 128 | 15.1% | 15,748 | 453 | 2.9% |
| distance_to_target_surface_mm <= 0 (inside target) | 0 | 0 | n/a | 16,966 | 633 | 3.7% |
| distance_to_target_surface_mm <= 2 mm | 4,047 | 292 | 7.2% | 12,919 | 341 | 2.6% |
| distance_to_target_surface_mm <= 5 mm | 6,536 | 487 | 7.5% | 10,430 | 146 | 1.4% |

## 5. `target_mesh_available` Association — `binary_label_gg3plus_int` (GG3+ / high-grade (Gleason ≥ 4+3=7))

If MRI targets are defined independently of pathology (e.g. from PI-RADS ≥ 3
lesions), `target_mesh_available` should have moderate but not near-perfect
association with the label. A very high positive rate when mesh=True AND a
near-zero rate when mesh=False would be a strong leakage signal.

**`target_mesh_available = True (mesh present)`**

- Rows: 16,966
- Positives (`binary_label_gg3plus_int==1`): 633
- Prevalence: 3.7%

**`target_mesh_available = False (no mesh)`**

- Rows: 26
- Positives (`binary_label_gg3plus_int==1`): 14
- Prevalence: 53.8%

**Reverse view: mesh rate by label**

| Label | N | Has target mesh | Mesh rate |
|---|---|---|---|
| 0 (negative) | 16,345 | 16,333 | 99.9% |
| 1 (positive) | 647 | 633 | 97.8% |

> **Small difference (< 10 pp):** mesh availability is similarly distributed across labels, consistent with targets being defined independently of pathology outcome.

## 6. Interpretation Guide

| Signal | Plausible predictor | Leakage concern |
|---|---|---|
| `target_mesh_available` similarly prevalent in positive and negative cases | ✓ | |
| `target_mesh_available` strongly associated with label | | ✓ |
| `distance_to_target_surface_mm` lower for positives (shorter distance) | ✓ | |
| `distance_to_target_surface_mm ≤ 0` covers nearly all positives | | ✓ |
| `trajectory_intersects_target` higher rate in positives | ✓ (targeted biopsy) | |
| `trajectory_intersects_target` = near-perfect label proxy | | ✓ |
| `fraction_inside_target` graded association with label | ✓ | |
| `fraction_inside_target > 0` = near-perfect label proxy | | ✓ |

**Context:** In the TCIA Prostate-MRI-US-Biopsy dataset both systematic
(non-targeted) and targeted (MRI-directed) biopsies are present. MRI targets
are PI-RADS lesions defined *before* the biopsy, so proximity to the target
is a legitimate predictor. However, if the dataset only provides a target mesh
for patients who later tested positive, the geometry features would leak the label.
The statistics above should be compared against that prior knowledge.

