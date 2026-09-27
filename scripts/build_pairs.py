#!/usr/bin/env python
"""
scripts/build_pairs.py
----------------------
Build longitudinal next-visit pairs from the OASIS-2 CSV.

Usage:
    python scripts/build_pairs.py --config configs/base.yaml
    python scripts/build_pairs.py --csv data/raw/oasis_longitudinal.csv

Outputs:
    data/processed/next_visit_pairs.csv
    data/processed/pair_metadata.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cerebro_x.data.oasis2.loader import load_oasis2, load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs, save_pairs
from cerebro_x.utils.io import load_yaml, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build longitudinal next-visit pairs for Cerebro X."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to base config YAML.",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/oasis2_kaggle.yaml",
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Direct path to oasis_longitudinal.csv (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write pairs CSV and metadata.",
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
    logger = logging.getLogger("build_pairs")

    # ── Load configs ───────────────────────────────────────────────────────
    base_config = {}
    if Path(args.config).exists():
        base_config = load_yaml(args.config)

    dataset_config = {}
    if Path(args.dataset_config).exists():
        dataset_config = load_yaml(args.dataset_config)

    # ── Load dataset ───────────────────────────────────────────────────────
    try:
        if args.csv:
            df = load_oasis2(args.csv)
        else:
            df = load_oasis2_auto(config=dataset_config)
    except FileNotFoundError as e:
        logger.error(
            "\n%s\n\n"
            "PAIR CONSTRUCTION CANNOT RUN WITHOUT DATA.\n"
            "Run the audit first: python scripts/audit_oasis2.py",
            e,
        )
        sys.exit(1)

    # ── Build pairs ────────────────────────────────────────────────────────
    logger.info("Building longitudinal next-visit pairs...")
    pairs_df, meta = build_next_visit_pairs(df)

    # Console summary
    print("\n" + "=" * 60)
    print("CEREBRO X — LONGITUDINAL PAIR CONSTRUCTION SUMMARY")
    print("=" * 60)
    print(f"Input rows:          {meta.total_input_rows}")
    print(f"Input subjects:      {meta.unique_subjects_input}")
    print(f"Output pairs:        {meta.total_pairs}")
    print(f"Subjects with pairs: {meta.unique_subjects_with_pairs}")
    print(f"Pairs per subject:   {meta.pair_distribution}")
    print(f"next_CDR dist:       {meta.next_cdr_distribution}")
    print(f"next_MMSE missing:   {meta.next_mmse_stats.get('missing', 'N/A')}")
    print(f"Feature columns:     {len(meta.input_feature_columns)}")
    if meta.notes:
        print(f"Notes: {meta.notes}")
    print("=" * 60 + "\n")

    # ── Save outputs ───────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    save_pairs(pairs_df, meta, output_dir)

    logger.info("Done. Pairs written to: %s", (output_dir / "next_visit_pairs.csv").resolve())
    print(f"Pairs saved to: {(output_dir / 'next_visit_pairs.csv').resolve()}")


if __name__ == "__main__":
    main()
