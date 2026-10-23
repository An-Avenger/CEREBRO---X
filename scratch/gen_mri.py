import sys
sys.path.insert(0, "src")
import numpy as np
import nibabel as nib
from pathlib import Path

OUT_DIR = Path("test_mri_samples")
OUT_DIR.mkdir(exist_ok=True)

def make_brain_volume(brain_fraction=0.75, noise_std=80, seed=42):
    shape = (128, 128, 90)
    D, H, W = shape
    rng = np.random.default_rng(seed)
    vol = np.abs(rng.normal(0, 10, shape).astype(np.float32))
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    cz, cy, cx = D / 2, H / 2, W / 2
    rz = D * 0.42 * brain_fraction
    ry = H * 0.48 * brain_fraction
    rx = W * 0.48 * brain_fraction
    mask = ((z-cz)/rz)**2 + ((y-cy)/ry)**2 + ((x-cx)/rx)**2 <= 1.0
    wm   = ((z-cz)/(rz*0.65))**2 + ((y-cy)/(ry*0.65))**2 + ((x-cx)/(rx*0.65))**2 <= 1.0
    vol[wm] = rng.normal(800, noise_std, int(wm.sum())).astype(np.float32)
    gm = mask & ~wm
    vol[gm] = rng.normal(550, noise_std * 0.8, int(gm.sum())).astype(np.float32)
    return np.clip(vol, 0, 1200), mask

profiles = [
    ("normal_young_subject.nii.gz",      0.82,  60, 1),
    ("mild_cognitive_impairment.nii.gz",  0.72,  80, 2),
    ("mild_alzheimers.nii.gz",            0.62, 100, 3),
    ("moderate_alzheimers.nii.gz",        0.50, 120, 4),
]
affine = np.diag([1.5, 1.5, 1.5, 1.0])

for name, fraction, noise, seed in profiles:
    vol, mask = make_brain_volume(brain_fraction=fraction, noise_std=noise, seed=seed)
    img = nib.Nifti1Image(vol, affine=affine)
    img.header.set_data_dtype(np.float32)
    img.header['pixdim'][1:4] = [1.5, 1.5, 1.5]
    img.header['xyzt_units'] = 2
    out = OUT_DIR / name
    nib.save(img, str(out))
    sz = out.stat().st_size // 1024
    nwbv = round(int(mask.sum()) / int(np.prod(vol.shape)), 4)
    print("SAVED:", name, "| nWBV~" + str(nwbv), "| " + str(sz) + " KB")

print("DONE - 4 files in", str(OUT_DIR.resolve()))
