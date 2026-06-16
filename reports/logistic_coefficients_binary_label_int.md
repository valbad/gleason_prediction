# Logistic Regression Coefficients

Coefficients are **standardised** for numeric features (one unit = one σ  
after `StandardScaler`) and in natural 0→1 units for boolean features.  
A **positive** coefficient increases the predicted probability of `label = 1`.

## full_geometry_dataset

### clinical_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 0.6679 | std-scaled |
| `psa_ng_ml` | ▲ 0.0683 | std-scaled |
| `psa_density` | ▲ 0.0650 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `prostate_volume_cc` | ▼ 0.4113 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 0.6679 | std-scaled |
| 2 | `psa_ng_ml` | ▲ 0.0683 | std-scaled |
| 3 | `psa_density` | ▲ 0.0650 | std-scaled |
| 4 | `prostate_volume_cc` | ▼ 0.4113 | std-scaled |

</details>

### biopsy_geometry_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `midpoint_inside_prostate` | ▲ 0.2977 | 0/1 flag |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1279 | std-scaled |
| `n_tube_voxels` | ▲ 0.1167 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0034 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `n_centerline_voxels` | ▼ 0.1558 | std-scaled |
| `core_length_mm` | ▼ 0.0099 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `midpoint_inside_prostate` | ▲ 0.2977 | 0/1 flag |
| 2 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1279 | std-scaled |
| 3 | `n_tube_voxels` | ▲ 0.1167 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0034 | std-scaled |
| 5 | `core_length_mm` | ▼ 0.0099 | std-scaled |
| 6 | `n_centerline_voxels` | ▼ 0.1558 | std-scaled |

</details>

### target_geometry_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 0.6190 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4290 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `target_mesh_available` | ▼ 3.2194 | 0/1 flag |
| `distance_midpoint_to_target_surface_mm` | ▼ 1.4916 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.1849 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 0.6190 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4290 | std-scaled |
| 3 | `trajectory_intersects_target` | ▼ 0.1849 | 0/1 flag |
| 4 | `distance_midpoint_to_target_surface_mm` | ▼ 1.4916 | std-scaled |
| 5 | `target_mesh_available` | ▼ 3.2194 | 0/1 flag |

</details>

### target_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 0.6192 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4289 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 1.5009 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.2171 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 0.6192 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4289 | std-scaled |
| 3 | `trajectory_intersects_target` | ▼ 0.2171 | 0/1 flag |
| 4 | `distance_midpoint_to_target_surface_mm` | ▼ 1.5009 | std-scaled |

</details>

### all_geometry

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 0.6096 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4615 | std-scaled |
| `n_tube_voxels` | ▲ 0.1836 | std-scaled |
| `core_length_mm` | ▲ 0.1196 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1140 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.0622 | 0/1 flag |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `target_mesh_available` | ▼ 3.0854 | 0/1 flag |
| `distance_midpoint_to_target_surface_mm` | ▼ 1.4942 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.2401 | 0/1 flag |
| `n_centerline_voxels` | ▼ 0.2154 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.0088 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 0.6096 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4615 | std-scaled |
| 3 | `n_tube_voxels` | ▲ 0.1836 | std-scaled |
| 4 | `core_length_mm` | ▲ 0.1196 | std-scaled |
| 5 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1140 | std-scaled |
| 6 | `midpoint_inside_prostate` | ▲ 0.0622 | 0/1 flag |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.0088 | std-scaled |
| 8 | `n_centerline_voxels` | ▼ 0.2154 | std-scaled |
| 9 | `trajectory_intersects_target` | ▼ 0.2401 | 0/1 flag |
| 10 | `distance_midpoint_to_target_surface_mm` | ▼ 1.4942 | std-scaled |
| 11 | `target_mesh_available` | ▼ 3.0854 | 0/1 flag |

</details>

### all_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 0.6084 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4645 | std-scaled |
| `n_tube_voxels` | ▲ 0.1813 | std-scaled |
| `core_length_mm` | ▲ 0.1365 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1297 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.0464 | 0/1 flag |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0059 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 1.5045 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.2757 | 0/1 flag |
| `n_centerline_voxels` | ▼ 0.2179 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 0.6084 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4645 | std-scaled |
| 3 | `n_tube_voxels` | ▲ 0.1813 | std-scaled |
| 4 | `core_length_mm` | ▲ 0.1365 | std-scaled |
| 5 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1297 | std-scaled |
| 6 | `midpoint_inside_prostate` | ▲ 0.0464 | 0/1 flag |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0059 | std-scaled |
| 8 | `n_centerline_voxels` | ▼ 0.2179 | std-scaled |
| 9 | `trajectory_intersects_target` | ▼ 0.2757 | 0/1 flag |
| 10 | `distance_midpoint_to_target_surface_mm` | ▼ 1.5045 | std-scaled |

</details>

### all_geometry_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 0.7229 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3471 | std-scaled |
| `core_length_mm` | ▲ 0.2001 | std-scaled |
| `n_tube_voxels` | ▲ 0.1809 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1195 | std-scaled |
| `psa_density` | ▲ 0.0353 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `target_mesh_available` | ▼ 1.9579 | 0/1 flag |
| `distance_midpoint_to_target_surface_mm` | ▼ 0.7462 | std-scaled |
| `prostate_volume_cc` | ▼ 0.3223 | std-scaled |
| `n_centerline_voxels` | ▼ 0.2143 | std-scaled |
| `distance_midpoint_to_target_centroid_mm` | ▼ 0.1576 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.0902 | 0/1 flag |
| `midpoint_inside_prostate` | ▼ 0.0289 | 0/1 flag |
| `psa_ng_ml` | ▼ 0.0049 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.0029 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 0.7229 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3471 | std-scaled |
| 3 | `core_length_mm` | ▲ 0.2001 | std-scaled |
| 4 | `n_tube_voxels` | ▲ 0.1809 | std-scaled |
| 5 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1195 | std-scaled |
| 6 | `psa_density` | ▲ 0.0353 | std-scaled |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▼ 0.0029 | std-scaled |
| 8 | `psa_ng_ml` | ▼ 0.0049 | std-scaled |
| 9 | `midpoint_inside_prostate` | ▼ 0.0289 | 0/1 flag |
| 10 | `trajectory_intersects_target` | ▼ 0.0902 | 0/1 flag |
| 11 | `distance_midpoint_to_target_centroid_mm` | ▼ 0.1576 | std-scaled |
| 12 | `n_centerline_voxels` | ▼ 0.2143 | std-scaled |
| 13 | `prostate_volume_cc` | ▼ 0.3223 | std-scaled |
| 14 | `distance_midpoint_to_target_surface_mm` | ▼ 0.7462 | std-scaled |
| 15 | `target_mesh_available` | ▼ 1.9579 | 0/1 flag |

</details>

### all_geometry_no_availability_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 0.7342 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3466 | std-scaled |
| `core_length_mm` | ▲ 0.2077 | std-scaled |
| `n_tube_voxels` | ▲ 0.1766 | std-scaled |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1286 | std-scaled |
| `psa_density` | ▲ 0.0358 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0069 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 0.7520 | std-scaled |
| `prostate_volume_cc` | ▼ 0.3269 | std-scaled |
| `n_centerline_voxels` | ▼ 0.2119 | std-scaled |
| `distance_midpoint_to_target_centroid_mm` | ▼ 0.1580 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.0983 | 0/1 flag |
| `midpoint_inside_prostate` | ▼ 0.0483 | 0/1 flag |
| `psa_ng_ml` | ▼ 0.0041 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 0.7342 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3466 | std-scaled |
| 3 | `core_length_mm` | ▲ 0.2077 | std-scaled |
| 4 | `n_tube_voxels` | ▲ 0.1766 | std-scaled |
| 5 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.1286 | std-scaled |
| 6 | `psa_density` | ▲ 0.0358 | std-scaled |
| 7 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.0069 | std-scaled |
| 8 | `psa_ng_ml` | ▼ 0.0041 | std-scaled |
| 9 | `midpoint_inside_prostate` | ▼ 0.0483 | 0/1 flag |
| 10 | `trajectory_intersects_target` | ▼ 0.0983 | 0/1 flag |
| 11 | `distance_midpoint_to_target_centroid_mm` | ▼ 0.1580 | std-scaled |
| 12 | `n_centerline_voxels` | ▼ 0.2119 | std-scaled |
| 13 | `prostate_volume_cc` | ▼ 0.3269 | std-scaled |
| 14 | `distance_midpoint_to_target_surface_mm` | ▼ 0.7520 | std-scaled |

</details>

## extracted_voxel_subset

### clinical_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 0.9665 | std-scaled |
| `psa_density` | ▲ 0.0465 | std-scaled |
| `psa_ng_ml` | ▲ 0.0243 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `prostate_volume_cc` | ▼ 0.1635 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 0.9665 | std-scaled |
| 2 | `psa_density` | ▲ 0.0465 | std-scaled |
| 3 | `psa_ng_ml` | ▲ 0.0243 | std-scaled |
| 4 | `prostate_volume_cc` | ▼ 0.1635 | std-scaled |

</details>

### target_geometry_only

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.3470 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4877 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 2.0847 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.4589 | 0/1 flag |
| `target_mesh_available` | ▼ 0.1187 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.3470 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4877 | std-scaled |
| 3 | `target_mesh_available` | ▼ 0.1187 | 0/1 flag |
| 4 | `trajectory_intersects_target` | ▼ 0.4589 | 0/1 flag |
| 5 | `distance_midpoint_to_target_surface_mm` | ▼ 2.0847 | std-scaled |

</details>

### target_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.3468 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.4870 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 2.0856 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.4583 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.3468 | std-scaled |
| 2 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.4870 | std-scaled |
| 3 | `trajectory_intersects_target` | ▼ 0.4583 | 0/1 flag |
| 4 | `distance_midpoint_to_target_surface_mm` | ▼ 2.0856 | std-scaled |

</details>

### all_geometry

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.3831 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.7767 | 0/1 flag |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.5289 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.5245 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.2628 | std-scaled |
| `n_tube_voxels` | ▲ 0.2049 | std-scaled |
| `core_length_mm` | ▲ 0.0645 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 2.1512 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6029 | 0/1 flag |
| `n_centerline_voxels` | ▼ 0.0587 | std-scaled |
| `target_mesh_available` | ▼ 0.0043 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.3831 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.7767 | 0/1 flag |
| 3 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.5289 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.5245 | std-scaled |
| 5 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.2628 | std-scaled |
| 6 | `n_tube_voxels` | ▲ 0.2049 | std-scaled |
| 7 | `core_length_mm` | ▲ 0.0645 | std-scaled |
| 8 | `target_mesh_available` | ▼ 0.0043 | 0/1 flag |
| 9 | `n_centerline_voxels` | ▼ 0.0587 | std-scaled |
| 10 | `trajectory_intersects_target` | ▼ 0.6029 | 0/1 flag |
| 11 | `distance_midpoint_to_target_surface_mm` | ▼ 2.1512 | std-scaled |

</details>

### all_geometry_no_availability

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.3855 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.7688 | 0/1 flag |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.5295 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.5247 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.2645 | std-scaled |
| `n_tube_voxels` | ▲ 0.2038 | std-scaled |
| `core_length_mm` | ▲ 0.0636 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 2.1538 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6038 | 0/1 flag |
| `n_centerline_voxels` | ▼ 0.0571 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.3855 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.7688 | 0/1 flag |
| 3 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.5295 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.5247 | std-scaled |
| 5 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.2645 | std-scaled |
| 6 | `n_tube_voxels` | ▲ 0.2038 | std-scaled |
| 7 | `core_length_mm` | ▲ 0.0636 | std-scaled |
| 8 | `n_centerline_voxels` | ▼ 0.0571 | std-scaled |
| 9 | `trajectory_intersects_target` | ▼ 0.6038 | 0/1 flag |
| 10 | `distance_midpoint_to_target_surface_mm` | ▼ 2.1538 | std-scaled |

</details>

### centerline_plus_geometry

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_centroid_mm` | ▲ 1.3831 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.7767 | 0/1 flag |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.5289 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.5245 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.2628 | std-scaled |
| `n_tube_voxels` | ▲ 0.2049 | std-scaled |
| `core_length_mm` | ▲ 0.0645 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 2.1512 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.6029 | 0/1 flag |
| `n_centerline_voxels` | ▼ 0.0587 | std-scaled |
| `target_mesh_available` | ▼ 0.0043 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `distance_midpoint_to_target_centroid_mm` | ▲ 1.3831 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.7767 | 0/1 flag |
| 3 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.5289 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.5245 | std-scaled |
| 5 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.2628 | std-scaled |
| 6 | `n_tube_voxels` | ▲ 0.2049 | std-scaled |
| 7 | `core_length_mm` | ▲ 0.0645 | std-scaled |
| 8 | `target_mesh_available` | ▼ 0.0043 | 0/1 flag |
| 9 | `n_centerline_voxels` | ▼ 0.0587 | std-scaled |
| 10 | `trajectory_intersects_target` | ▼ 0.6029 | 0/1 flag |
| 11 | `distance_midpoint_to_target_surface_mm` | ▼ 2.1512 | std-scaled |

</details>

### centerline_plus_geometry_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 0.8840 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.7502 | 0/1 flag |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.7232 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3259 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.3248 | std-scaled |
| `n_tube_voxels` | ▲ 0.1460 | std-scaled |
| `psa_ng_ml` | ▲ 0.1425 | std-scaled |
| `n_centerline_voxels` | ▲ 0.0777 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 0.6310 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.3935 | 0/1 flag |
| `distance_midpoint_to_target_centroid_mm` | ▼ 0.1047 | std-scaled |
| `core_length_mm` | ▼ 0.0653 | std-scaled |
| `psa_density` | ▼ 0.0386 | std-scaled |
| `prostate_volume_cc` | ▼ 0.0162 | std-scaled |
| `target_mesh_available` | ▼ 0.0014 | 0/1 flag |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 0.8840 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.7502 | 0/1 flag |
| 3 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.7232 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3259 | std-scaled |
| 5 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.3248 | std-scaled |
| 6 | `n_tube_voxels` | ▲ 0.1460 | std-scaled |
| 7 | `psa_ng_ml` | ▲ 0.1425 | std-scaled |
| 8 | `n_centerline_voxels` | ▲ 0.0777 | std-scaled |
| 9 | `target_mesh_available` | ▼ 0.0014 | 0/1 flag |
| 10 | `prostate_volume_cc` | ▼ 0.0162 | std-scaled |
| 11 | `psa_density` | ▼ 0.0386 | std-scaled |
| 12 | `core_length_mm` | ▼ 0.0653 | std-scaled |
| 13 | `distance_midpoint_to_target_centroid_mm` | ▼ 0.1047 | std-scaled |
| 14 | `trajectory_intersects_target` | ▼ 0.3935 | 0/1 flag |
| 15 | `distance_midpoint_to_target_surface_mm` | ▼ 0.6310 | std-scaled |

</details>

### centerline_plus_geometry_no_availability_plus_clinical

**Top positive coefficients** (→ higher probability of label = 1  ↑)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `log_psa_ng_ml` | ▲ 0.8849 | std-scaled |
| `midpoint_inside_prostate` | ▲ 0.7566 | 0/1 flag |
| `distance_midpoint_to_prostate_surface_mm` | ▲ 0.7229 | std-scaled |
| `approximate_fraction_of_centerline_inside_target` | ▲ 0.3248 | std-scaled |
| `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.3239 | std-scaled |
| `n_tube_voxels` | ▲ 0.1450 | std-scaled |
| `psa_ng_ml` | ▲ 0.1398 | std-scaled |
| `n_centerline_voxels` | ▲ 0.0794 | std-scaled |

**Top negative coefficients** (→ higher probability of label = 1  ↓)

| Feature | Coefficient | Scaled? |
|---|---|---|
| `distance_midpoint_to_target_surface_mm` | ▼ 0.6378 | std-scaled |
| `trajectory_intersects_target` | ▼ 0.3904 | 0/1 flag |
| `distance_midpoint_to_target_centroid_mm` | ▼ 0.0981 | std-scaled |
| `core_length_mm` | ▼ 0.0664 | std-scaled |
| `psa_density` | ▼ 0.0379 | std-scaled |
| `prostate_volume_cc` | ▼ 0.0158 | std-scaled |

<details><summary>All coefficients (ranked)</summary>

| Rank | Feature | Coefficient | Scaled? |
|---|---|---|---|
| 1 | `log_psa_ng_ml` | ▲ 0.8849 | std-scaled |
| 2 | `midpoint_inside_prostate` | ▲ 0.7566 | 0/1 flag |
| 3 | `distance_midpoint_to_prostate_surface_mm` | ▲ 0.7229 | std-scaled |
| 4 | `approximate_fraction_of_centerline_inside_target` | ▲ 0.3248 | std-scaled |
| 5 | `approximate_fraction_of_centerline_inside_prostate` | ▲ 0.3239 | std-scaled |
| 6 | `n_tube_voxels` | ▲ 0.1450 | std-scaled |
| 7 | `psa_ng_ml` | ▲ 0.1398 | std-scaled |
| 8 | `n_centerline_voxels` | ▲ 0.0794 | std-scaled |
| 9 | `prostate_volume_cc` | ▼ 0.0158 | std-scaled |
| 10 | `psa_density` | ▼ 0.0379 | std-scaled |
| 11 | `core_length_mm` | ▼ 0.0664 | std-scaled |
| 12 | `distance_midpoint_to_target_centroid_mm` | ▼ 0.0981 | std-scaled |
| 13 | `trajectory_intersects_target` | ▼ 0.3904 | 0/1 flag |
| 14 | `distance_midpoint_to_target_surface_mm` | ▼ 0.6378 | std-scaled |

</details>

