# Gleason Grade Prediction from MRI-Guided Biopsy

Prostate cancer classification using T2-weighted MRI biopsy core data. Three approaches are compared: a classical ML baseline, a 3D CNN, and a Graph Neural Network (GNN), all trained on the public [TCIA Prostate-MRI-US-Biopsy](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/) dataset (~16,700 biopsy cores).

**Binary task:** clinically significant cancer (Gleason Grade Group 3+) vs. low-grade/benign (GG0–2).

---

## Pipeline Overview

```
Step 1  build_manifest.py          Build dataset manifest (labels + needle coordinates)
Step 2  02_coordinate_alignment    Validate DICOM ↔ voxel coordinate transforms
Step 3  extract_rois.py            Extract sparse cylindrical point clouds per core
Step 4  04_baseline.ipynb          Random Forest on hand-crafted features (~AUC 0.72)
Step 5a 05_dl_3dcnn.ipynb          3D CNN on resampled grid (~AUC 0.77)
Step 5b 06_dl_gnn_final.ipynb      GNN on native sparse point cloud (~AUC 0.80)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `pydicom`, `torch`, `torch_geometric`, `scikit-learn`, `pandas`, `numpy`.

**Data:** Download the TCIA dataset and place DICOM series under `data/dicom/` and the Excel spreadsheet at `data/raw/TCIA-Biopsy-Data_2020-07-14.xlsx`.

---

## Running the Pipeline

```bash
# 1. Build the manifest (one row per biopsy core, with labels and needle coords)
python src/build_manifest.py

# 2. Validate coordinate system (inspect visuals in notebook)
jupyter notebook notebooks/02_coordinate_alignment.ipynb

# 3. Extract sparse ROIs from MRI volumes
python src/extract_rois.py

# 4–5. Train and evaluate models (notebooks run independently)
jupyter notebook notebooks/04_baseline.ipynb
jupyter notebook notebooks/05_dl_3dcnn.ipynb
jupyter notebook notebooks/06_dl_gnn_final.ipynb
```

---

## Project Structure

```
gleason_prediction/
├── src/
│   ├── build_manifest.py        # Step 1: dataset manifest builder
│   ├── dicom_utils.py           # DICOM loading, LPS ↔ voxel transforms
│   └── extract_rois.py          # Step 3: cylindrical ROI extraction
├── notebooks/
│   ├── 02_coordinate_alignment.ipynb
│   ├── 04_baseline.ipynb
│   ├── 05_dl_3dcnn.ipynb
│   └── 06_dl_gnn_final.ipynb
├── data/                        # Git-ignored except metadata CSVs
│   ├── manifest.csv             # Main dataset (16,692 cores)
│   ├── dicom_index.csv          # series_uid → DICOM path cache
│   ├── extraction_report.csv    # Step 3 QA stats
│   ├── cores_to_drop_contamination.csv
│   ├── patches/                 # Sparse point clouds (.npz), one per core
│   ├── grids/                   # Precomputed 3D grids for CNN
│   └── edges/                   # Precomputed graph edges for GNN
└── requirements.txt
```

---

## Methods

### ROI Representation

Each biopsy core is represented as a **sparse cylindrical point cloud** in a needle-aligned coordinate frame (axes: w = axial along needle, u/v = radial). Voxels within 2 mm of the needle trajectory are extracted from the T2-weighted MRI volume, with intensities z-score normalized per scan. This yields compact ~9 KB `.npz` files and avoids including irrelevant tissue or neighboring needle signals.

### Step 4 — Random Forest Baseline

~20 hand-crafted features per core (intensity statistics, spatial extent, gradient magnitude). Evaluated with 5-fold stratified group cross-validation (splits at subject level to prevent data leakage).

### Step 5a — 3D CNN

Point clouds are resampled onto a fixed 3D grid (50 × 12 × 12 bins, covering 100 mm axially and ±2 mm radially). A residual 3D CNN with global average pooling classifies each grid. Augmentation: rotation around the needle axis and radial flips.

### Step 5b — GNN (EdgeConv / DGCNN-style)

Nodes are voxels; edges connect nearby voxels (radius graph). Node features: normalized axial position, radial coordinates, radial distance, intensity. EdgeConv layers (`MLP(cat(x_i, x_j − x_i))`) with residual connections and global max+mean pooling. Graph edges are precomputed once and cached. Augmentation is the same as for the CNN and leaves edge structure invariant.

All three models share identical train/validation/test splits (15% subjects held out for testing) for fair comparison.

---

## Gleason Grading Reference

| Grade Group | Gleason Score | Clinical Significance |
|-------------|--------------|----------------------|
| GG0 | — | Benign |
| GG1 | 3+3 = 6 | Low-grade |
| GG2 | 3+4 = 7 | Intermediate |
| GG3 | 4+3 = 7 | **Clinically significant** (positive class) |
| GG4 | 8 | **Clinically significant** |
| GG5 | 9–10 | **Clinically significant** |

---

## Citation

Dataset: TCIA Prostate-MRI-US-Biopsy (2020). The Cancer Imaging Archive.
