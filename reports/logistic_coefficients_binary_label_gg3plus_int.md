# Logistic Regression Coefficients

Coefficients are **standardised** for numeric features (one unit = one σ  
after `StandardScaler`) and in natural 0→1 units for boolean features.  
A **positive** coefficient increases the predicted probability of `label = 1`.

## full_geometry_dataset

### clinical_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 1.0434 | std-scaled |
| `psa_density` | ▲ 0.1506 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `prostate_volume_cc` | ▼ 0.3866 | std-scaled |
| `psa_ng_ml` | ▼ 0.0260 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 1.0434 | std-scaled |
| 2 | `psa_density` | ▲ 0.1506 | std-scaled |
| 3 | `psa_ng_ml` | ▼ 0.0260 | std-scaled |
| 4 | `prostate_volume_cc` | ▼ 0.3866 | std-scaled |

</details>

### biopsy_geometry_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `midpoint_inside_prostate` | ▲ 0.2887 | 0/1 flag |
| `n_centerline_voxels` | ▲ 0.2706 | std-scaled |
| `core_length_mm` | ▲ 0.0934 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0547 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0179 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `n_tube_voxels` | ▼ 0.2961 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `midpoint_inside_prostate` | ▲ 0.2887 | 0/1 flag |
| 2 | `n_centerline_voxels` | ▲ 0.2706 | std-scaled |
| 3 | `core_length_mm` | ▲ 0.0934 | std-scaled |
| 4 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0547 | std-scaled |
| 5 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0179 | std-scaled |
| 6 | `n_tube_voxels` | ▼ 0.2961 | std-scaled |

</details>

### target_geometry_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.2320 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4945 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `target_mesh_available` | ▼ 4.1692 | 0/1 flag |
| `distance_midpoint_to_target_surface_mm` | ▼ 2.1754 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.4033 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.2320 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4945 | std-scaled |
| 3 | `trajectory_intersects_target` | ▼ 0.4033 | 0/1 flag |
| 4 | `distance_midpoint_to_target_surface_mm` | ▼ 2.1754 | std-scaled |
| 5 | `target_mesh_available` | ▼ 4.1692 | 0/1 flag |

</details>

### target_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.2107 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4928 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 2.1760 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.5031 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.2107 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4928 | std-scaled |
| 3 | `trajectory_intersects_target` | ▼ 0.5031 | 0/1 flag |
| 4 | `distance_midpoint_to_target_surface_mm` | ▼ 2.1760 | std-scaled |

</details>

### all_geometry

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.2509 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.5264 | std-scaled |
| `core_length_mm` | ▲ 0.2293 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.1850 | 0/1 flag |
| `n_centerline_voxels` | ▲ 0.1732 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `target_mesh_available` | ▼ 3.9604 | 0/1 flag |
| `distance_midpoint_to_target_surface_mm` | ▼ 2.2296 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.4545 | 0/1 flag |
| `n_tube_voxels` | ▼ 0.2024 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.0272 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0061 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.2509 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.5264 | std-scaled |
| 3 | `core_length_mm` | ▲ 0.2293 | std-scaled |
| 4 | `midpoint_inside_prostate` | ▲ 0.1850 | 0/1 flag |
| 5 | `n_centerline_voxels` | ▲ 0.1732 | std-scaled |
| 6 | `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0061 | std-scaled |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.0272 | std-scaled |
| 8 | `n_tube_voxels` | ▼ 0.2024 | std-scaled |
| 9 | `trajectory_intersects_target` | ▼ 0.4545 | 0/1 flag |
| 10 | `distance_midpoint_to_target_surface_mm` | ▼ 2.2296 | std-scaled |
| 11 | `target_mesh_available` | ▼ 3.9604 | 0/1 flag |

</details>

### all_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.2379 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.5344 | std-scaled |
| `core_length_mm` | ▲ 0.2780 | std-scaled |
| `n_centerline_voxels` | ▲ 0.1522 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.1256 | 0/1 flag |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0499 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0253 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 2.2494 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.5654 | 0/1 flag |
| `n_tube_voxels` | ▼ 0.1959 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.2379 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.5344 | std-scaled |
| 3 | `core_length_mm` | ▲ 0.2780 | std-scaled |
| 4 | `n_centerline_voxels` | ▲ 0.1522 | std-scaled |
| 5 | `midpoint_inside_prostate` | ▲ 0.1256 | 0/1 flag |
| 6 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0499 | std-scaled |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0253 | std-scaled |
| 8 | `n_tube_voxels` | ▼ 0.1959 | std-scaled |
| 9 | `trajectory_intersects_target` | ▼ 0.5654 | 0/1 flag |
| 10 | `distance_midpoint_to_target_surface_mm` | ▼ 2.2494 | std-scaled |

</details>

### all_geometry_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 1.0358 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3225 | std-scaled |
| `core_length_mm` | ▲ 0.3110 | std-scaled |
| `psa_density` | ▲ 0.1818 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.1506 | 0/1 flag |
| `distance_midpoint_to_target_centroid_mm` | ▲ 0.0654 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0103 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0070 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `target_mesh_available` | ▼ 2.4792 | 0/1 flag |
| `distance_midpoint_to_target_surface_mm` | ▼ 1.0805 | std-scaled |
| `prostate_volume_cc` | ▼ 0.2669 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.2099 | 0/1 flag |
| `psa_ng_ml` | ▼ 0.1228 | std-scaled |
| `n_tube_voxels` | ▼ 0.1025 | std-scaled |
| `n_centerline_voxels` | ▼ 0.0085 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 1.0358 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3225 | std-scaled |
| 3 | `core_length_mm` | ▲ 0.3110 | std-scaled |
| 4 | `psa_density` | ▲ 0.1818 | std-scaled |
| 5 | `midpoint_inside_prostate` | ▲ 0.1506 | 0/1 flag |
| 6 | `distance_midpoint_to_target_centroid_mm` | ▲ 0.0654 | std-scaled |
| 7 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0103 | std-scaled |
| 8 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0070 | std-scaled |
| 9 | `n_centerline_voxels` | ▼ 0.0085 | std-scaled |
| 10 | `n_tube_voxels` | ▼ 0.1025 | std-scaled |
| 11 | `psa_ng_ml` | ▼ 0.1228 | std-scaled |
| 12 | `trajectory_intersects_target` | ▼ 0.2099 | 0/1 flag |
| 13 | `prostate_volume_cc` | ▼ 0.2669 | std-scaled |
| 14 | `distance_midpoint_to_target_surface_mm` | ▼ 1.0805 | std-scaled |
| 15 | `target_mesh_available` | ▼ 2.4792 | 0/1 flag |

</details>

### all_geometry_no_availability_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 1.0638 | std-scaled |
| `core_length_mm` | ▲ 0.3266 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3212 | std-scaled |
| `psa_density` | ▲ 0.1908 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.1149 | 0/1 flag |
| `distance_midpoint_to_target_centroid_mm` | ▲ 0.0411 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0298 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0266 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 1.0694 | std-scaled |
| `prostate_volume_cc` | ▼ 0.2705 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.2349 | 0/1 flag |
| `psa_ng_ml` | ▼ 0.1294 | std-scaled |
| `n_tube_voxels` | ▼ 0.1118 | std-scaled |
| `n_centerline_voxels` | ▼ 0.0061 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 1.0638 | std-scaled |
| 2 | `core_length_mm` | ▲ 0.3266 | std-scaled |
| 3 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3212 | std-scaled |
| 4 | `psa_density` | ▲ 0.1908 | std-scaled |
| 5 | `midpoint_inside_prostate` | ▲ 0.1149 | 0/1 flag |
| 6 | `distance_midpoint_to_target_centroid_mm` | ▲ 0.0411 | std-scaled |
| 7 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.0298 | std-scaled |
| 8 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0266 | std-scaled |
| 9 | `n_centerline_voxels` | ▼ 0.0061 | std-scaled |
| 10 | `n_tube_voxels` | ▼ 0.1118 | std-scaled |
| 11 | `psa_ng_ml` | ▼ 0.1294 | std-scaled |
| 12 | `trajectory_intersects_target` | ▼ 0.2349 | 0/1 flag |
| 13 | `prostate_volume_cc` | ▼ 0.2705 | std-scaled |
| 14 | `distance_midpoint_to_target_surface_mm` | ▼ 1.0694 | std-scaled |

</details>

## extracted_voxel_subset

### clinical_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 1.2517 | std-scaled |
| `psa_density` | ▲ 0.3345 | std-scaled |
| `prostate_volume_cc` | ▲ 0.0991 | std-scaled |
| `psa_ng_ml` | ▲ 0.0215 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 1.2517 | std-scaled |
| 2 | `psa_density` | ▲ 0.3345 | std-scaled |
| 3 | `prostate_volume_cc` | ▲ 0.0991 | std-scaled |
| 4 | `psa_ng_ml` | ▲ 0.0215 | std-scaled |

</details>

### target_geometry_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.9745 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.2863 | std-scaled |
| `target_mesh_available` | ▲ 0.0024 | 0/1 flag |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 3.3244 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6514 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.9745 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.2863 | std-scaled |
| 3 | `target_mesh_available` | ▲ 0.0024 | 0/1 flag |
| 4 | `trajectory_intersects_target` | ▼ 0.6514 | 0/1 flag |
| 5 | `distance_midpoint_to_target_surface_mm` | ▼ 3.3244 | std-scaled |

</details>

### target_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.9763 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.2864 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 3.3278 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6522 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.9763 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.2864 | std-scaled |
| 3 | `trajectory_intersects_target` | ▼ 0.6522 | 0/1 flag |
| 4 | `distance_midpoint_to_target_surface_mm` | ▼ 3.3278 | std-scaled |

</details>

### all_geometry

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 2.0136 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.6784 | 0/1 flag |
| `core_length_mm` | ▲ 0.5276 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3109 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 3.3774 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6612 | 0/1 flag |
| `n_tube_voxels` | ▼ 0.5749 | std-scaled |
| `n_centerline_voxels` | ▼ 0.4366 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.2191 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0493 | std-scaled |
| `target_mesh_available` | ▼ 0.0095 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 2.0136 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.6784 | 0/1 flag |
| 3 | `core_length_mm` | ▲ 0.5276 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3109 | std-scaled |
| 5 | `target_mesh_available` | ▼ 0.0095 | 0/1 flag |
| 6 | `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0493 | std-scaled |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.2191 | std-scaled |
| 8 | `n_centerline_voxels` | ▼ 0.4366 | std-scaled |
| 9 | `n_tube_voxels` | ▼ 0.5749 | std-scaled |
| 10 | `trajectory_intersects_target` | ▼ 0.6612 | 0/1 flag |
| 11 | `distance_midpoint_to_target_surface_mm` | ▼ 3.3774 | std-scaled |

</details>

### all_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 2.0137 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.6804 | 0/1 flag |
| `core_length_mm` | ▲ 0.5290 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3118 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 3.3794 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6651 | 0/1 flag |
| `n_tube_voxels` | ▼ 0.5761 | std-scaled |
| `n_centerline_voxels` | ▼ 0.4383 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.2197 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0497 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 2.0137 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.6804 | 0/1 flag |
| 3 | `core_length_mm` | ▲ 0.5290 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3118 | std-scaled |
| 5 | `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0497 | std-scaled |
| 6 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.2197 | std-scaled |
| 7 | `n_centerline_voxels` | ▼ 0.4383 | std-scaled |
| 8 | `n_tube_voxels` | ▼ 0.5761 | std-scaled |
| 9 | `trajectory_intersects_target` | ▼ 0.6651 | 0/1 flag |
| 10 | `distance_midpoint_to_target_surface_mm` | ▼ 3.3794 | std-scaled |

</details>

### centerline_plus_geometry

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 2.0136 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.6784 | 0/1 flag |
| `core_length_mm` | ▲ 0.5276 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3109 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 3.3774 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6612 | 0/1 flag |
| `n_tube_voxels` | ▼ 0.5749 | std-scaled |
| `n_centerline_voxels` | ▼ 0.4366 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.2191 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0493 | std-scaled |
| `target_mesh_available` | ▼ 0.0095 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 2.0136 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.6784 | 0/1 flag |
| 3 | `core_length_mm` | ▲ 0.5276 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3109 | std-scaled |
| 5 | `target_mesh_available` | ▼ 0.0095 | 0/1 flag |
| 6 | `distance_midpoint_to_prostate_surface_mm` | ▼ 0.0493 | std-scaled |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.2191 | std-scaled |
| 8 | `n_centerline_voxels` | ▼ 0.4366 | std-scaled |
| 9 | `n_tube_voxels` | ▼ 0.5749 | std-scaled |
| 10 | `trajectory_intersects_target` | ▼ 0.6612 | 0/1 flag |
| 11 | `distance_midpoint_to_target_surface_mm` | ▼ 3.3774 | std-scaled |

</details>

### centerline_plus_geometry_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 1.1211 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.6535 | 0/1 flag |
| `psa_density` | ▲ 0.2507 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1986 | std-scaled |
| `psa_ng_ml` | ▲ 0.1874 | std-scaled |
| `core_length_mm` | ▲ 0.1579 | std-scaled |
| `prostate_volume_cc` | ▲ 0.0970 | std-scaled |
| `target_mesh_available` | ▲ 0.0053 | 0/1 flag |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 1.2193 | std-scaled |
| `n_tube_voxels` | ▼ 0.4134 | std-scaled |
| `distance_midpoint_to_target_centroid_mm` | ▼ 0.3655 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.1803 | 0/1 flag |
| `n_centerline_voxels` | ▼ 0.1759 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.1098 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▼ 0.1065 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 1.1211 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.6535 | 0/1 flag |
| 3 | `psa_density` | ▲ 0.2507 | std-scaled |
| 4 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1986 | std-scaled |
| 5 | `psa_ng_ml` | ▲ 0.1874 | std-scaled |
| 6 | `core_length_mm` | ▲ 0.1579 | std-scaled |
| 7 | `prostate_volume_cc` | ▲ 0.0970 | std-scaled |
| 8 | `target_mesh_available` | ▲ 0.0053 | 0/1 flag |
| 9 | `approximate_fraction_of_centerline_inside_target` | ▼ 0.1065 | std-scaled |
| 10 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.1098 | std-scaled |
| 11 | `n_centerline_voxels` | ▼ 0.1759 | std-scaled |
| 12 | `trajectory_intersects_target` | ▼ 0.1803 | 0/1 flag |
| 13 | `distance_midpoint_to_target_centroid_mm` | ▼ 0.3655 | std-scaled |
| 14 | `n_tube_voxels` | ▼ 0.4134 | std-scaled |
| 15 | `distance_midpoint_to_target_surface_mm` | ▼ 1.2193 | std-scaled |

</details>

### centerline_plus_geometry_no_availability_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 1.1152 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.6598 | 0/1 flag |
| `psa_density` | ▲ 0.2526 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1980 | std-scaled |
| `psa_ng_ml` | ▲ 0.1886 | std-scaled |
| `core_length_mm` | ▲ 0.1573 | std-scaled |
| `prostate_volume_cc` | ▲ 0.0978 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 1.2279 | std-scaled |
| `n_tube_voxels` | ▼ 0.4232 | std-scaled |
| `distance_midpoint_to_target_centroid_mm` | ▼ 0.3597 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.1861 | 0/1 flag |
| `n_centerline_voxels` | ▼ 0.1720 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.1108 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▼ 0.1059 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 1.1152 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.6598 | 0/1 flag |
| 3 | `psa_density` | ▲ 0.2526 | std-scaled |
| 4 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1980 | std-scaled |
| 5 | `psa_ng_ml` | ▲ 0.1886 | std-scaled |
| 6 | `core_length_mm` | ▲ 0.1573 | std-scaled |
| 7 | `prostate_volume_cc` | ▲ 0.0978 | std-scaled |
| 8 | `approximate_fraction_of_centerline_inside_target` | ▼ 0.1059 | std-scaled |
| 9 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.1108 | std-scaled |
| 10 | `n_centerline_voxels` | ▼ 0.1720 | std-scaled |
| 11 | `trajectory_intersects_target` | ▼ 0.1861 | 0/1 flag |
| 12 | `distance_midpoint_to_target_centroid_mm` | ▼ 0.3597 | std-scaled |
| 13 | `n_tube_voxels` | ▼ 0.4232 | std-scaled |
| 14 | `distance_midpoint_to_target_surface_mm` | ▼ 1.2279 | std-scaled |

</details>

