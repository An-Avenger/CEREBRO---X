#!/usr/bin/env python
"""
scripts/audit_oasis2.py
-----------------------
Run the OASIS-2 dataset audit.

Usage:
    python scripts/audit_oasis2.py --config configs/datasets/oasis2_kaggle.yaml
    python scripts/audit_oasis2.py --csv data/raw/oasis_longitudinal.csv

On Kaggle, the CSV is discovered automatically.

Outputs:
    artifacts/AUDIT-001/
        audit_report.json
        plots/cdr_distribution.png
        plots/mmse_distribution.png
        plots/visit_distribution.png
        plots/missingness.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for local/Kaggle execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.oasis2.loader import load_oasis2, load_oasis2_auto
from cerebro_x.data.oasis2.audit import (
    run_audit,
    save_audit_report,
    plot_cdr_distribution,
    plot_mmse_distribution,
    plot_visit_distribution,
    plot_missingness,
)
from cerebro_x.data.provenance import build_provenance, save_provenance
from cerebro_x.utils.io import load_yaml, setup_logging, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OASIS-2 dataset audit for Cerebro X."
    )
    parser.add_argument(
        "--config",
        default="configs/datasets/oasis2_kaggle.yaml",
        help="Path to the dataset config YAML.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Direct path to oasis_longitudinal.csv (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/AUDIT-001",
        help="Directory to write audit outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("audit_oasis2")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    config = {}
    if Path(args.config).exists():
        config = load_yaml(args.config)
        logger.info("Loaded config: %s", args.config)
    else:
        logger.warning("Config not found at %s — using defaults.", args.config)

    # ── Load dataset ───────────────────────────────────────────────────────
    try:
        if args.csv:
            logger.info("Loading from --csv argument: %s", args.csv)
            df = load_oasis2(args.csv)
        else:
            df = load_oasis2_auto(config=config)
    except FileNotFoundError as e:
        logger.error(
            "\n%s\n\n"
            "AUDIT CANNOT RUN WITHOUT DATA.\n"
            "Place oasis_longitudinal.csv in data/raw/ or attach the Kaggle dataset.",
            e,
        )
        sys.exit(1)

    # ── Run audit ──────────────────────────────────────────────────────────
    logger.info("Running dataset audit...")
    report = run_audit(df)

    # Print summary to console
    print("\n" + "=" * 60)
    print("CEREBRO X — OASIS-2 DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"Rows:             {report.total_rows}")
    print(f"Columns:          {report.total_columns}")
    print(f"Unique subjects:  {report.unique_subjects}")
    print(f"Total pairs:      {sum(k * v for k, v in report.visits_per_subject.items()) - report.unique_subjects}")
    print(f"Subjects with changing CDR: {report.subjects_with_changing_cdr}")
    print(f"CDR distribution: {report.cdr_distribution}")
    print(f"MMSE missing:     {report.mmse_stats.get('missing', 'N/A')}")
    print(f"Columns with missing data: {list(report.missingness.keys())}")
    if report.warnings:
        print(f"\nWARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  [WARNING] {w}")
    else:
        print("\n[OK] All checks match verified dataset facts.")
    print("=" * 60 + "\n")

    # ── Save audit report ──────────────────────────────────────────────────
    save_audit_report(report, output_dir)

    # ── Generate plots ─────────────────────────────────────────────────────
    plot_cdr_distribution(df, plots_dir / "cdr_distribution.png")
    plot_mmse_distribution(df, plots_dir / "mmse_distribution.png")
    plot_visit_distribution(df, plots_dir / "visit_distribution.png")
    plot_missingness(df, plots_dir / "missingness.png")

    # ── Save provenance ────────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
    else:
        from cerebro_x.data.oasis2.loader import discover_dataset
        csv_path = discover_dataset()

    if csv_path and csv_path.exists():
        prov = build_provenance(
            dataset_id="oasis2_kaggle",
            file_path=csv_path,
            df=df,
            download_note=config.get("download_note", "Kaggle jboysen/mri-and-alzheimers"),
        )
        save_provenance(prov, output_dir / "provenance.json")
    else:
        logger.warning("Could not compute provenance — CSV path not available.")

    logger.info("Audit complete. Outputs at: %s", output_dir.resolve())
    print(f"Audit outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
