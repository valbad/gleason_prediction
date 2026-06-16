# Shareable Tabular Dataset Summary

Source: `data/share/needle_features_v1.csv`

- Total rows in file: 18,176
- Rows after common filters (label_join_status == 'coord_match', `binary_label_int` not null, split in ('train', 'val', 'test')): 16,992


## MODE A — full_geometry_dataset (no extraction_status filter)

- Total rows: 16,992
- Total patients: 796

| Split | Rows | Patients | GG3+ prevalence |
|---|---|---|---|
| train | 11,827 | 557 | 0.115 |
| val | 2,748 | 119 | 0.099 |
| test | 2,417 | 120 | 0.107 |

## MODE B — extracted_voxel_subset (extraction_status == 'ok')

- Total rows: 1,706
- Total patients: 81

| Split | Rows | Patients | GG3+ prevalence |
|---|---|---|---|
| train | 1,119 | 53 | 0.100 |
| val | 249 | 10 | 0.112 |
| test | 338 | 18 | 0.157 |
