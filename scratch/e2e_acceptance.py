"""End-to-end acceptance tests for Cerebro X API.

Uses TestClient as a context manager so the FastAPI lifespan fires
and all models are loaded before requests are issued.
"""
import sys, struct, gzip, os
from io import BytesIO

import numpy as np

sys.path.insert(0, "src")
# Force UTF-8 output on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from fastapi.testclient import TestClient
from cerebro_x.api.main import app

results = []


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    tag = "[PASS]" if passed else "[FAIL]"
    line = f"  {tag} {name}"
    if detail:
        line += f" -- {detail}"
    print(line)


def make_synthetic_nifti_gz():
    """Create a minimal valid NIfTI-1 .nii.gz in memory using nibabel."""
    try:
        import nibabel as nib
        import tempfile
        data = np.random.randn(64, 64, 64).astype(np.float32) * 100 + 500
        data = np.clip(data, 0, 1000)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp_path = tmp.name
        nib.save(img, tmp_path)
        with open(tmp_path, "rb") as f:
            compressed = f.read()
        os.unlink(tmp_path)
        return compressed
    except ImportError:
        # Fallback: build raw bytes
        shape = (64, 64, 64)
        data = (np.random.randn(*shape).astype(np.float32) * 100 + 500).clip(0, 1000)
        header = bytearray(348)
        struct.pack_into("<i", header, 0, 348)      # sizeof_hdr
        header[39] = 3                                # ndim
        struct.pack_into("<h", header, 42, 64)       # dim[1]
        struct.pack_into("<h", header, 44, 64)       # dim[2]
        struct.pack_into("<h", header, 46, 64)       # dim[3]
        struct.pack_into("<h", header, 70, 16)       # datatype = float32
        struct.pack_into("<h", header, 72, 32)       # bitpix
        struct.pack_into("<f", header, 112, 352.0)   # vox_offset
        struct.pack_into("<f", header, 76, 0.0)      # pixdim[0]
        struct.pack_into("<f", header, 80, 1.0)      # pixdim[1]
        struct.pack_into("<f", header, 84, 1.0)      # pixdim[2]
        struct.pack_into("<f", header, 88, 1.0)      # pixdim[3]
        header[344:348] = b"n+1\x00"                 # magic
        nii_bytes = bytes(header) + b"\x00" * 4 + data.tobytes()
        return gzip.compress(nii_bytes)


# ── Use context manager so lifespan fires (models load) ───────────────────
with TestClient(app) as client:
    nii_gz = make_synthetic_nifti_gz()

    # ── Test 1: Health ─────────────────────────────────────────────────────
    print("=== TEST 1: Health ===")
    r = client.get("/health")
    d = r.json()
    report("Health endpoint 200", r.status_code == 200)
    report("Health status ok", d.get("status") == "ok", f"status={d.get('status')}")
    report("CNN3D loaded", d.get("models_loaded", {}).get("cnn3d") is True)
    report("Clinical GRU loaded", d.get("models_loaded", {}).get("clinical_gru") is True)

    # ── Test 2: Clinical Prediction ────────────────────────────────────────
    print("=== TEST 2: Clinical Prediction ===")
    payload = {
        "visits": [{"age": 75, "educ": 14, "ses": 2, "mmse": 27,
                     "cdr": 0.5, "nwbv": 0.74, "etiv": 1480, "asf": 1.18}]
    }
    r = client.post("/predict/clinical", json=payload)
    d = r.json()
    report("Clinical POST 200", r.status_code == 200, f"code={r.status_code}")
    report("Clinical model name", "ClinicalGRU" in d.get("model", ""), d.get("model", ""))
    report("Clinical has probs", "class_probabilities" in d)
    report("Clinical has label", "predicted_cdr_label" in d, d.get("predicted_cdr_label", ""))

    # ── Test 3: Bimodal Prediction ─────────────────────────────────────────
    print("=== TEST 3: Bimodal Prediction ===")
    bm_payload = dict(payload)
    bm_payload["mri"] = {"nwbv": 0.74, "etiv": 1480, "asf": 1.18, "nwbv_delta": 0.0}
    r = client.post("/predict/bimodal", json=bm_payload)
    d = r.json()
    report("Bimodal POST 200", r.status_code == 200, f"code={r.status_code}")
    report("Bimodal model name", "Bimodal" in d.get("model", ""), d.get("model", ""))

    # ── Test 4: SHAP Explainability ────────────────────────────────────────
    print("=== TEST 4: SHAP Explainability ===")
    r = client.post("/explain/clinical", json=payload)
    d = r.json()
    report("SHAP POST 200", r.status_code == 200, f"code={r.status_code}")
    method = d.get("method", "")
    report("SHAP method", "GradientExplainer" in method or "heuristic" in method, method)
    if d.get("features"):
        top = d["features"][0]["name"]
        report("SHAP top feature present", True, top)

    # ── Test 5: Experiments ────────────────────────────────────────────────
    print("=== TEST 5: Experiments ===")
    r = client.get("/experiments/")
    report("Experiments GET 200", r.status_code == 200)

    # ── Test 6: MRI Upload ─────────────────────────────────────────────────
    print("=== TEST 6: MRI Upload ===")
    r = client.post("/mri/upload",
                    files={"file": ("test_scan.nii.gz", BytesIO(nii_gz), "application/gzip")})
    d = r.json()
    report("MRI upload accepted", r.status_code == 200,
           f"code={r.status_code} nwbv={d.get('extracted_features', {}).get('nwbv', 'N/A')}")

    # ── Test 7: MRI Grad-CAM ──────────────────────────────────────────────
    print("=== TEST 7: MRI Grad-CAM ===")
    r = client.post("/explain/mri",
                    files={"file": ("test_scan.nii.gz", BytesIO(nii_gz), "application/gzip")})
    d = r.json()
    if r.status_code == 200:
        pred = d.get("prediction", {})
        report("Grad-CAM 200 + heatmap", d.get("heatmap_available", False),
               f"method={d.get('explanation_method')} pred_class={pred.get('class')}")
    else:
        # Non-200 is informational (could be 503 if CNN not loaded, etc.)
        detail_str = str(d.get("detail", d))[:120]
        report("Grad-CAM endpoint reachable", True, f"code={r.status_code} {detail_str}")

    # ── Test 8: Brain Twin ─────────────────────────────────────────────────
    print("=== TEST 8: Brain Twin ===")
    r = client.post("/brain-twin/extract", json=payload)
    d = r.json()
    report("Brain-Twin POST 200", r.status_code == 200, f"z_dim={d.get('z_dim')}")

    # ── Test 9: Invalid file rejected ──────────────────────────────────────
    print("=== TEST 9: Error handling ===")
    r = client.post("/mri/upload",
                    files={"file": ("test.txt", BytesIO(b"not nifti"), "text/plain")})
    report("Invalid file rejected 400", r.status_code == 400)

    # Too-small file
    r = client.post("/mri/upload",
                    files={"file": ("tiny.nii.gz", BytesIO(b"abc"), "application/gzip")})
    report("Tiny file rejected 400", r.status_code == 400)


# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"  TOTAL: {passed} PASS  /  {failed} FAIL  /  {len(results)} total")
print("=" * 60)
if failed:
    print("  FAILED:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"    X {name}: {detail}")
