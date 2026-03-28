"""
DICOM loading and coordinate conversion utilities.

Validated in notebooks/02_coordinate_alignment.ipynb on subject 0179:
  - Voxel size: 0.664 × 0.664 × 1.500 mm
  - Round-trip LPS ↔ voxel error: machine epsilon
  - All needle endpoints land inside volume bounds
"""

import numpy as np
import pydicom
from pathlib import Path


def load_dicom_volume(series_dir: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load a DICOM series into a 3D numpy volume.

    Slices are sorted by position along the slice-normal direction.

    Returns
    -------
    volume   : float32 array of shape (n_slices, n_rows, n_cols)
    affine   : 4×4 float64 array mapping (col, row, slice) → LPS mm
    datasets : list of pydicom datasets, sorted by slice position
    """
    dcm_files = sorted(Path(series_dir).glob("*.dcm"))
    datasets  = [pydicom.dcmread(str(f)) for f in dcm_files]

    row_dir = np.array([float(x) for x in datasets[0].ImageOrientationPatient[:3]])
    col_dir = np.array([float(x) for x in datasets[0].ImageOrientationPatient[3:]])
    normal  = np.cross(row_dir, col_dir)
    normal  = normal / np.linalg.norm(normal)

    def slice_pos(ds):
        ipp = np.array([float(x) for x in ds.ImagePositionPatient])
        return float(np.dot(ipp, normal))

    datasets.sort(key=slice_pos)

    volume = np.stack([ds.pixel_array.astype(np.float32) for ds in datasets], axis=0)

    positions     = np.array([slice_pos(ds) for ds in datasets])
    slice_spacing = float(np.median(np.diff(positions)))

    ps          = datasets[0].PixelSpacing
    row_spacing = float(ps[0])   # mm between rows  → along col_dir
    col_spacing = float(ps[1])   # mm between cols  → along row_dir
    origin      = np.array([float(x) for x in datasets[0].ImagePositionPatient])

    affine = np.eye(4)
    affine[:3, 0] = row_dir * col_spacing    # Δcol  → LPS
    affine[:3, 1] = col_dir * row_spacing    # Δrow  → LPS
    affine[:3, 2] = normal  * slice_spacing  # Δslice → LPS
    affine[:3, 3] = origin

    return volume, affine, datasets


def load_dicom_metadata(series_dir: Path) -> tuple[np.ndarray, tuple[int,int,int]]:
    """
    Like load_dicom_volume but skips pixel loading.
    Returns (affine, (n_slices, n_rows, n_cols)).
    Useful when you only need the coordinate transform without loading the full volume.
    """
    dcm_files = sorted(Path(series_dir).glob("*.dcm"))
    datasets  = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in dcm_files]

    row_dir = np.array([float(x) for x in datasets[0].ImageOrientationPatient[:3]])
    col_dir = np.array([float(x) for x in datasets[0].ImageOrientationPatient[3:]])
    normal  = np.cross(row_dir, col_dir)
    normal  = normal / np.linalg.norm(normal)

    def slice_pos(ds):
        ipp = np.array([float(x) for x in ds.ImagePositionPatient])
        return float(np.dot(ipp, normal))

    datasets.sort(key=slice_pos)

    positions     = np.array([slice_pos(ds) for ds in datasets])
    slice_spacing = float(np.median(np.diff(positions)))

    ps          = datasets[0].PixelSpacing
    origin      = np.array([float(x) for x in datasets[0].ImagePositionPatient])

    affine = np.eye(4)
    affine[:3, 0] = row_dir * float(ps[1])
    affine[:3, 1] = col_dir * float(ps[0])
    affine[:3, 2] = normal  * slice_spacing
    affine[:3, 3] = origin

    shape = (len(datasets), int(datasets[0].Rows), int(datasets[0].Columns))
    return affine, shape


def lps_to_voxel(lps_point: np.ndarray, affine_inv: np.ndarray) -> np.ndarray:
    """
    Convert a point in LPS world coordinates (mm) to voxel coordinates (col, row, slice).

    Parameters
    ----------
    lps_point  : array-like of shape (3,)
    affine_inv : 4×4 inverse of the DICOM affine (from np.linalg.inv(affine))

    Returns
    -------
    (col, row, slice) as float64 — not yet rounded to integers
    """
    h = np.array([*lps_point, 1.0])
    return (affine_inv @ h)[:3]


def voxel_to_lps(col: float, row: float, slc: float, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates (col, row, slice) to LPS world coordinates (mm).
    """
    h = np.array([col, row, slc, 1.0])
    return (affine @ h)[:3]


def voxel_size_mm(affine: np.ndarray) -> tuple[float, float, float]:
    """Return (col_spacing, row_spacing, slice_spacing) in mm."""
    return (
        float(np.linalg.norm(affine[:3, 0])),
        float(np.linalg.norm(affine[:3, 1])),
        float(np.linalg.norm(affine[:3, 2])),
    )
