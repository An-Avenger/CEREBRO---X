"""
Generate synthetic NIfTI MRI test files for Cerebro X upload testing.
Creates 4 realistic test scans with different brain volume profiles.
"""
import sys, os
sys.path.insert(0, "src")

import numpy as np
import nibabel as nib
from pathlib import Path

OUT_DIR = Path("test_mri_samples")
OUT_DIR.mkdir(exist_ok=True)

def make_brain_volume(shape=(128, 128, 90), brain_fraction=0.75, noise_std=80):
    """Create a synthetic T1-weighted-like brain volume."""
    D, H, W = shape
    rng = np.random.default_rng(42)

    # Background (air) — near zero
    vol = rng.normal(0, 10, shape).astype(np.float32)
    vol = np.abs(vol)

    # Ellipsoid brain mask
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    cz, cy, cx = D/2, H/2, W/2
    rz, ry, rx = D*0.42*brain_fraction, H*0.48*brain_fraction, W*0.48*brain_fraction
    mask = ((z-cz)/rz)**2 + ((y-cy)/ry)**2 + ((x-cx)/rx)**2 <= 1.0

    # White matter core (brighter)
    wm_mask = ((z-cz)/(rz*0.65))**2 + ((y-cy)/(ry*0.65))**2 + ((x-cx)/(rx*0.65))**2 <= 1.0
    vol[wm_mask] = rng.normal(800, noise_std, wm_mask.sum()).astype(np.float32)
    vol[mask & ~wm_mask] = rng.normal(550, noise_std * 0.8, (mask & ~wm_mask).sum()).astype(np.float32)

    vol = np.clip(vol, 0, 1200)
    return vol, mask


profiles = [
    {
        "name":     "normal_young_subject.nii.gz",
        "label":    "Normal / Young (CDR 0) — large brain volume",
        "fraction": 0.82,
        "noise":    60,
    },
    {
        "name":     "mild_cognitive_impairment.nii.gz",
        "label":    "Mild Cognitive Impairment (CDR 0.5) — moderate atrophy",
        "fraction": 0.72,
        "noise":    80,
    },
    {
        "name":     "mild_alzheimers.nii.gz",
        "label":    "Mild Alzheimer's (CDR 1.0) — noticeable atrophy",
        "fraction": 0.62,
        "noise":    100,
    },
    {
        "name":     "moderate_alzheimers.nii.gz",
        "label":    "Moderate Alzheimer's (CDR 2.0) — severe atrophy",
        "fraction": 0.50,
        "noise":    120,
    },
]

affine = np.diag([1.5, 1.5, 1.5, 1.0])  # 1.5mm isotropic voxels

print("Generating synthetic NIfTI MRI test files...\n")
print(f"Output directory: {OUT_DIR.resolve()}\n")

for p in profiles:
    vol, mask = make_brain_volume(shape=(128, 128, 90), brain_fraction=p["fraction"], noise_std=p["noise"])
    img = nib.Nifti1Image(vol, affine=affine)
    # Set realistic NIfTI header fields
    hdr = img.header
    hdr.set_data_dtype(np.float32)
    hdr['pixdim'][1:4] = [1.5, 1.5, 1.5]
    hdr['xyzt_units'] = 2  # mm

    out_path = OUT_DIR / p["name"]
    nib.save(img, str(out_path))
    size_kb = out_path.stat().st_size // 1024
    brain_voxels = int(mask.sum())
    total_voxels = int(np.prod(vol.shape))
    nwbv_proxy = round(brain_voxels / total_voxels, 4)
    print(f"  ✓ {p['name']}")
    print(f"    Profile:      {p['label']}")
    print(f"    Shape:        128 x 128 x 90 voxels (1.5mm isotropic)")
    print(f"    nWBV proxy:   {nwbv_proxy}  (brain/total voxel ratio)")
    print(f"    File size:    {size_kb} KB")
    print()

print(f"Done! 4 files written to:  {OUT_DIR.resolve()}")
print()
print("Upload any of these to http://localhost:3000/predict")
print("  → Section 2: Imaging Modalities → '📤 Upload MRI (.nii)'")
