#!/usr/bin/env python
"""
scripts/gpu_smoke_test.py
---------------------------
GPU smoke test for the Cerebro-X ADNI MRI pipeline.

Verifies:
  1. Python environment and PyTorch installation
  2. CUDA availability and GPU identification
  3. NIfTI loading of one real MRI volume from the prepared dataset
  4. MRI preprocessing (resize, z-score)
  5. One model forward pass
  6. One training step (backward pass, gradient check)
  7. One validation step
  8. Loss and gradient finiteness
  9. GPU memory usage
  10. Checkpoint save and reload

Does NOT start full training.
Reports results and proposes the full-training command.

Usage:
    # With real ADNI data (after dcm2niix + label alignment):
    python scripts/gpu_smoke_test.py \\
        --manifest  data/raw/mri/ADNI/manifest_with_labels.csv \\
        --nifti-dir data/raw/mri/ADNI/nifti

    # CPU smoke test (no CUDA required):
    python scripts/gpu_smoke_test.py \\
        --manifest  data/raw/mri/ADNI/manifest_with_labels.csv \\
        --nifti-dir data/raw/mri/ADNI/nifti \\
        --no-gpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.imaging.adni_dataset import RealADNIMRIDataset
from cerebro_x.models.deep.cnn3d import MRICerebroNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gpu_smoke_test")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU smoke test for ADNI MRI pipeline.")
    parser.add_argument("--manifest",   required=True, help="manifest_with_labels.csv")
    parser.add_argument("--nifti-dir",  required=True, help="Root directory of NIfTI files")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail:
        msg += f": {detail}"
    logger.info(msg)
    return condition


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    results: dict[str, bool] = {}
    report: dict = {"checks": [], "errors": [], "warnings": []}

    logger.info("=" * 60)
    logger.info("Cerebro-X GPU Smoke Test")
    logger.info("=" * 60)

    # ── 1. Python and PyTorch ─────────────────────────────────────────────────
    import platform
    logger.info("\n[1] Environment")
    py_ok = check("Python version", True, platform.python_version())
    torch_ok = check("PyTorch version", True, torch.__version__)
    results["environment"] = py_ok and torch_ok
    report["checks"].append({"name": "environment", "passed": results["environment"]})

    try:
        import nibabel
        nib_ok = check("nibabel version", True, nibabel.__version__)
    except ImportError:
        nib_ok = check("nibabel", False, "NOT INSTALLED — run: pip install nibabel")
        report["errors"].append("nibabel not installed")
    results["nibabel"] = nib_ok

    # ── 2. CUDA check ─────────────────────────────────────────────────────────
    logger.info("\n[2] CUDA / GPU")
    if args.no_gpu:
        device = torch.device("cpu")
        cuda_ok = check("CUDA (skipped)", True, "CPU mode forced via --no-gpu")
        report["warnings"].append("Running in CPU mode. Training on CPU will be very slow.")
    else:
        cuda_available = torch.cuda.is_available()
        cuda_ok = check("CUDA available", cuda_available)
        if not cuda_available:
            logger.error(
                "CUDA not available. "
                "Use --no-gpu for CPU-only smoke test, or run on a GPU machine."
            )
            report["errors"].append("CUDA not available")
            sys.exit(1)
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("GPU detected", True, f"{gpu_name} ({vram_gb:.1f} GB VRAM)")
        report["gpu"] = {"name": gpu_name, "vram_gb": round(vram_gb, 2)}

    results["cuda"] = cuda_ok

    # ── 3. Load one real MRI ──────────────────────────────────────────────────
    logger.info("\n[3] Real MRI Loading")
    dataset = None
    vol_tensor = None
    try:
        dataset = RealADNIMRIDataset(
            manifest_path=args.manifest,
            nifti_base_dir=args.nifti_dir,
            augment=False,
        )
        n_ok = check("Dataset loaded", len(dataset) > 0, f"{len(dataset)} scans")
        results["dataset"] = n_ok

        if len(dataset) == 0:
            report["errors"].append("Dataset is empty")
            sys.exit(1)

        t_load = time.time()
        vol_tensor, label = dataset[0]
        load_time = time.time() - t_load
        shape_ok = check(
            "MRI volume shape", vol_tensor.shape == (1, 64, 64, 64),
            str(tuple(vol_tensor.shape))
        )
        finite_ok = check(
            "MRI volume finite", torch.isfinite(vol_tensor).all().item(),
            f"min={vol_tensor.min():.3f} max={vol_tensor.max():.3f}"
        )
        label_ok = check(
            "Label in range", 0 <= int(label) <= 3,
            f"label={int(label)}"
        )
        check("Load time", True, f"{load_time:.2f}s")
        results["mri_loading"] = shape_ok and finite_ok and label_ok

    except Exception as e:
        results["mri_loading"] = False
        report["errors"].append(f"MRI loading failed: {e}")
        check("MRI loading", False, str(e))

    # ── 4. Model forward pass ─────────────────────────────────────────────────
    logger.info("\n[4] Model Forward Pass")
    model = None
    logits = None
    try:
        model = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        check("Model instantiated", True, f"{n_params:,} parameters")

        if vol_tensor is not None:
            x = vol_tensor.unsqueeze(0).to(device)  # (1, 1, 64, 64, 64)
            model.eval()
            with torch.no_grad():
                t_fwd = time.time()
                logits = model(x)
                fwd_time = time.time() - t_fwd

            shape_ok = check(
                "Output shape", logits.shape == (1, 4),
                str(tuple(logits.shape))
            )
            finite_ok = check(
                "Output finite", torch.isfinite(logits).all().item(),
                f"logits={logits.detach().cpu().numpy().round(3)}"
            )
            check("Forward pass time", True, f"{fwd_time*1000:.1f} ms")
            results["forward_pass"] = shape_ok and finite_ok
        else:
            results["forward_pass"] = False

    except Exception as e:
        results["forward_pass"] = False
        report["errors"].append(f"Forward pass failed: {e}")
        check("Forward pass", False, str(e))

    # ── 5. Training step ──────────────────────────────────────────────────────
    logger.info("\n[5] Training Step (backward pass)")
    try:
        if model is not None and vol_tensor is not None:
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            x = vol_tensor.unsqueeze(0).to(device)
            y = torch.tensor([int(label)], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            loss_ok = check("Loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")

            # Check all gradients are finite
            grad_ok = all(
                p.grad is not None and torch.isfinite(p.grad).all().item()
                for p in model.parameters()
                if p.requires_grad
            )
            grad_ok = check("Gradients finite", grad_ok)

            optimizer.step()
            results["training_step"] = loss_ok and grad_ok
        else:
            results["training_step"] = False

    except Exception as e:
        results["training_step"] = False
        report["errors"].append(f"Training step failed: {e}")
        check("Training step", False, str(e))

    # ── 6. GPU memory ─────────────────────────────────────────────────────────
    if device.type == "cuda":
        logger.info("\n[6] GPU Memory")
        mem_alloc = torch.cuda.memory_allocated(0) / 1e6
        mem_reserved = torch.cuda.memory_reserved(0) / 1e6
        check("GPU memory allocated", True, f"{mem_alloc:.1f} MB")
        check("GPU memory reserved", True, f"{mem_reserved:.1f} MB")
        report["gpu_memory_mb"] = {"allocated": round(mem_alloc, 1), "reserved": round(mem_reserved, 1)}
    else:
        logger.info("\n[6] GPU Memory (CPU mode — skipped)")

    # ── 7. Checkpoint save and reload ─────────────────────────────────────────
    logger.info("\n[7] Checkpoint Save / Reload")
    try:
        if model is not None:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                tmp_path = Path(f.name)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {"embedding_dim": 64, "num_classes": 4},
                "epoch": 0,
            }
            torch.save(checkpoint, str(tmp_path))
            save_ok = check("Checkpoint saved", tmp_path.exists(), str(tmp_path))

            # Reload
            ckpt = torch.load(str(tmp_path), map_location="cpu")
            model2 = MRICerebroNet(in_channels=1, embedding_dim=64, num_classes=4)
            model2.load_state_dict(ckpt["model_state_dict"])
            reload_ok = check("Checkpoint reloaded", True)

            # Verify same output
            if vol_tensor is not None:
                model2.eval()
                with torch.no_grad():
                    out1 = model.cpu().eval()(vol_tensor.unsqueeze(0))
                    out2 = model2(vol_tensor.unsqueeze(0))
                same_ok = check(
                    "Reloaded model produces same output",
                    torch.allclose(out1, out2, atol=1e-5),
                )
            else:
                same_ok = True

            tmp_path.unlink(missing_ok=True)
            results["checkpoint"] = save_ok and reload_ok and same_ok
        else:
            results["checkpoint"] = False

    except Exception as e:
        results["checkpoint"] = False
        report["errors"].append(f"Checkpoint test failed: {e}")
        check("Checkpoint", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = all(results.values())
    for name, passed in results.items():
        logger.info("  %s  %s", PASS if passed else FAIL, name.replace("_", " ").title())

    report["results"] = results
    report["all_passed"] = all_passed

    # Save report
    smoke_report_path = Path("artifacts") / "smoke_test_report.json"
    smoke_report_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_report_path.write_text(json.dumps(report, indent=2))
    logger.info("Smoke test report: %s", smoke_report_path)

    if all_passed:
        logger.info("\n🎉 All checks passed! Ready for full training.")
        logger.info("\nProposed full-training command:")
        logger.info(
            "  python scripts/train_cnn3d.py \\\n"
            "    --manifest %s \\\n"
            "    --nifti-dir %s \\\n"
            "    --out-dir artifacts/EXP-MRI-ADNI-001 \\\n"
            "    --epochs 50 --batch-size 4 --seed 42 --augment",
            args.manifest, args.nifti_dir,
        )
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error("\n❌ Smoke test FAILED. Fix the following before training: %s", failed)
        if report["errors"]:
            for err in report["errors"]:
                logger.error("  Error: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
