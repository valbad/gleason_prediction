"""
Step 3 — Sparse cylindrical ROI extraction in needle-aligned coordinates

For each biopsy core we extract only the voxels that lie inside the
biopsy cylinder (radius 2mm, tip → base + axial margin) and represent
them as a sparse point cloud in the needle's own coordinate frame:

    w  : position along the needle axis from tip  (mm)  [−margin, L + margin]
    u  : first radial dimension                   (mm)  [−r, r]
    v  : second radial dimension                  (mm)  [−r, r]
    I  : z-score normalised T2 intensity

Saved as {core_id}.npz with two arrays:
    coords     float32  (N, 3)  — (w, u, v) in mm
    intensity  float32  (N,)    — z-scored voxel value

This representation is:
  - Compact  (~9 KB/core vs ~35 KB for an AABB cube)
  - Free of irrelevant tissue and neighbouring needles
  - Directly usable for GNN (node positions + features) and
    non-DL (masked intensity statistics)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dicom_utils import load_dicom_volume, lps_to_voxel, voxel_size_mm

# ── Config ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR    = REPO_ROOT / "data"
PATCHES_DIR = DATA_DIR / "patches"
MANIFEST    = DATA_DIR / "manifest.csv"
DROP_LIST   = DATA_DIR / "cores_to_drop_contamination.csv"

RADIUS_MM       = 2.0  # cylinder radius
AXIAL_MARGIN_MM = 2.0  # extra mm beyond tip and base along needle axis


def needle_frame(tip_lps: np.ndarray, base_lps: np.ndarray) -> tuple:
    """
    Build an orthonormal frame aligned to the needle axis.

    Returns (w_hat, u_hat, v_hat):
        w_hat : unit vector from tip to base (axial direction)
        u_hat : first radial direction  (perpendicular to w_hat)
        v_hat : second radial direction (perpendicular to both)
    """
    axis = base_lps - tip_lps
    w_hat = axis / np.linalg.norm(axis)

    # Choose a reference vector that is not parallel to w_hat
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(w_hat, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u_hat = np.cross(w_hat, ref)
    u_hat = u_hat / np.linalg.norm(u_hat)
    v_hat = np.cross(w_hat, u_hat)
    return w_hat, u_hat, v_hat


def extract_cylinder_pointcloud(
    volume: np.ndarray,
    affine: np.ndarray,
    tip_lps: np.ndarray,
    base_lps: np.ndarray,
    radius_mm: float,
    axial_margin_mm: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract all voxels inside the biopsy cylinder as a sparse point cloud
    in needle-aligned coordinates.

    Returns:
        coords    : float32 (N, 3) — (w, u, v) in mm
        intensity : float32 (N,)   — z-scored voxel value
    Returns None if no voxels fall inside the cylinder.
    """
    n_slices, n_rows, n_cols = volume.shape
    length = float(np.linalg.norm(base_lps - tip_lps))
    w_hat, u_hat, v_hat = needle_frame(tip_lps, base_lps)

    affine_inv = np.linalg.inv(affine)
    vox_col, vox_row, vox_slice = voxel_size_mm(affine)

    # Compute AABB of the cylinder in voxel space for efficient candidate screening
    # — sample 8 corners of the cylinder bounding box in LPS, convert to voxel
    r = radius_mm
    L = length
    corners_lps = []
    for w_end in [-axial_margin_mm, L + axial_margin_mm]:
        for u_sign in [-r, r]:
            for v_sign in [-r, r]:
                p = tip_lps + w_end * w_hat + u_sign * u_hat + v_sign * v_hat
                corners_lps.append(p)
    corners_vox = np.array([lps_to_voxel(c, affine_inv) for c in corners_lps])
    # corners_vox columns: (col, row, slice)
    c_min = np.floor(corners_vox.min(axis=0)).astype(int)
    c_max = np.ceil(corners_vox.max(axis=0)).astype(int) + 1

    # Clip to volume
    c0 = int(np.clip(c_min[0], 0, n_cols   - 1))
    c1 = int(np.clip(c_max[0], 1, n_cols))
    r0 = int(np.clip(c_min[1], 0, n_rows   - 1))
    r1 = int(np.clip(c_max[1], 1, n_rows))
    s0 = int(np.clip(c_min[2], 0, n_slices - 1))
    s1 = int(np.clip(c_max[2], 1, n_slices))

    if c1 <= c0 or r1 <= r0 or s1 <= s0:
        return None

    # Build LPS coordinates for every candidate voxel (vectorised)
    cols   = np.arange(c0, c1)
    rows   = np.arange(r0, r1)
    slices = np.arange(s0, s1)
    # Grid: (n_rows_crop, n_cols_crop, n_slices_crop)
    I, J, K = np.meshgrid(cols, rows, slices, indexing='xy')
    ones = np.ones_like(I)
    vox_h = np.stack([I, J, K, ones], axis=-1).reshape(-1, 4)  # (M, 4)

    lps_pts = (affine @ vox_h.T).T[:, :3]  # (M, 3)

    # Project onto needle frame
    v_from_tip = lps_pts - tip_lps          # (M, 3)
    w_coords = v_from_tip @ w_hat            # axial
    u_coords = v_from_tip @ u_hat            # radial x
    v_coords = v_from_tip @ v_hat            # radial y
    radial   = np.sqrt(u_coords**2 + v_coords**2)

    # Cylinder mask
    inside = (
        (w_coords >= -axial_margin_mm) &
        (w_coords <= length + axial_margin_mm) &
        (radial   <= radius_mm)
    )

    if not inside.any():
        return None

    # Map back to voxel indices to read intensities
    vox_inside = vox_h[inside]  # (N, 4): (col, row, slice, 1)
    intensities = volume[
        vox_inside[:, 2],   # slice index
        vox_inside[:, 1],   # row index
        vox_inside[:, 0],   # col index
    ]

    coords = np.stack([w_coords[inside],
                       u_coords[inside],
                       v_coords[inside]], axis=1).astype(np.float32)

    return coords, intensities.astype(np.float32)


def process_series(
    series_path: Path,
    cores: pd.DataFrame,
    patches_dir: Path,
    radius_mm: float,
    axial_margin_mm: float,
) -> list[dict]:
    results = []

    try:
        volume, affine, _ = load_dicom_volume(series_path)
    except Exception as e:
        for _, row in cores.iterrows():
            results.append({'core_id': row.core_id, 'status': f'load_error: {e}'})
        return results

    # Z-score normalise once per volume
    mu  = float(volume.mean())
    std = float(volume.std()) + 1e-8
    volume = (volume - mu) / std

    for _, row in cores.iterrows():
        core_id  = int(row.core_id)
        tip_lps  = np.array([row.tip_x_mri,  row.tip_y_mri,  row.tip_z_mri],  dtype=float)
        base_lps = np.array([row.base_x_mri, row.base_y_mri, row.base_z_mri], dtype=float)

        result = extract_cylinder_pointcloud(
            volume, affine, tip_lps, base_lps, radius_mm, axial_margin_mm
        )
        if result is None:
            results.append({'core_id': core_id, 'status': 'empty_cylinder'})
            continue

        coords, intensity = result
        out_path = patches_dir / f"{core_id}.npz"
        np.savez_compressed(out_path, coords=coords, intensity=intensity)

        results.append({
            'core_id':        core_id,
            'status':         'ok',
            'n_voxels':       len(intensity),
            'intensity_mean': float(intensity.mean()),
            'intensity_std':  float(intensity.std()),
            'w_min':          float(coords[:, 0].min()),
            'w_max':          float(coords[:, 0].max()),
        })

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 3: Sparse Cylindrical ROI Extraction ===\n")

    manifest = pd.read_csv(MANIFEST)
    drop_ids = set(pd.read_csv(DROP_LIST)['core_id'].tolist())
    manifest = manifest[~manifest['core_id'].isin(drop_ids)].copy()

    print(f"Cores after contamination filter : {len(manifest):,}")
    print(f"  label=0 : {(manifest['label']==0).sum():,}")
    print(f"  label=1 : {(manifest['label']==1).sum():,}")
    print()

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    groups = manifest.groupby('series_uid_mri')
    print(f"Unique MRI series to load : {groups.ngroups}")
    print()

    all_results = []
    for series_uid, cores in tqdm(groups, desc="Series", total=groups.ngroups):
        series_path = Path(cores['mri_dicom_path'].iloc[0])
        res = process_series(
            series_path, cores, PATCHES_DIR, RADIUS_MM, AXIAL_MARGIN_MM
        )
        all_results.extend(res)

    report = pd.DataFrame(all_results)
    report.to_csv(DATA_DIR / "extraction_report.csv", index=False)

    ok     = report[report['status'] == 'ok']
    errors = report[report['status'] != 'ok']

    print(f"\n=== Extraction complete ===")
    print(f"  Successful : {len(ok):,}")
    print(f"  Failed     : {len(errors):,}")
    if len(errors):
        print(errors['status'].value_counts().to_string())
    print(f"\nVoxels per core (statistics):")
    print(ok['n_voxels'].describe().round(1).to_string())
    print(f"\nReport → data/extraction_report.csv")
