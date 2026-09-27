"""OASIS-2 dataset loader.

Responsibilities:
- Discover oasis_longitudinal.csv on local or Kaggle file system
- Load and validate schema
- Return a clean DataFrame with verified column types
- Raise explicit, actionable errors for any schema violation

This module does NOT perform any modeling, splitting, or preprocessing.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from cerebro_x.data.schemas import (
    Oasis2Columns,
    validate_oasis2_columns,
    validate_oasis2_shape,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_dataset(
    search_roots: list[str | Path] | None = None,
    filename: str = "oasis_longitudinal.csv",
) -> Optional[Path]:
    """
    Search for oasis_longitudinal.csv across common locations.

    Checks (in order):
      1. Each path in search_roots
      2. /kaggle/input/** (recursive)
      3. data/raw/ (relative to CWD)

    Args:
        search_roots: Additional directories to search first.
        filename:     The target filename to locate.

    Returns:
        Path to the first match found, or None if not found.
    """
    candidates: list[str] = []

    # 1. User-supplied roots
    for root in (search_roots or []):
        candidates += glob.glob(str(Path(root) / "**" / filename), recursive=True)
        candidates += glob.glob(str(Path(root) / filename))

    # 2. Kaggle environment
    candidates += glob.glob(f"/kaggle/input/**/{filename}", recursive=True)

    # 3. Local data/raw relative to CWD
    local_path = Path("data") / "raw" / filename
    if local_path.exists():
        candidates.append(str(local_path))

    if not candidates:
        return None

    found = Path(candidates[0]).resolve()
    logger.info("Dataset discovered at: %s", found)
    return found


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_oasis2(
    path: str | Path,
    validate_shape: bool = True,
    expected_rows: int = 373,
    expected_cols: int = 15,
) -> pd.DataFrame:
    """
    Load the OASIS-2 longitudinal CSV and validate its schema.

    Args:
        path:           Absolute or relative path to oasis_longitudinal.csv.
        validate_shape: If True, warn when shape differs from verified facts.
        expected_rows:  Verified row count for the Kaggle dataset version.
        expected_cols:  Verified column count.

    Returns:
        DataFrame with correct dtypes and no schema violations.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"OASIS-2 CSV not found at: {path}\n\n"
            f"To fix this:\n"
            f"  Local:  Place oasis_longitudinal.csv in data/raw/\n"
            f"  Kaggle: Attach dataset 'jboysen/mri-and-alzheimers' to your notebook\n"
            f"  Then run: python scripts/audit_oasis2.py --config configs/datasets/oasis2_kaggle.yaml"
        )

    logger.info("Loading OASIS-2 from: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded shape: %s", df.shape)

    # Schema validation — raises on missing columns
    validate_oasis2_columns(list(df.columns))

    # Shape check — warns (does not raise) on unexpected shape
    if validate_shape:
        validate_oasis2_shape(df, expected_rows=expected_rows, expected_cols=expected_cols)

    # Cast known numeric columns (handle any read-in as object)
    numeric_cols = [
        Oasis2Columns.VISIT,
        Oasis2Columns.MR_DELAY,
        Oasis2Columns.AGE,
        Oasis2Columns.EDUC,
        Oasis2Columns.SES,
        Oasis2Columns.MMSE,
        Oasis2Columns.CDR,
        Oasis2Columns.ETIV,
        Oasis2Columns.NWBV,
        Oasis2Columns.ASF,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by subject then visit — this is required for longitudinal pair building
    df = df.sort_values(
        by=[Oasis2Columns.SUBJECT_ID, Oasis2Columns.VISIT],
        ascending=True,
    ).reset_index(drop=True)

    logger.info(
        "OASIS-2 loaded: %d rows, %d subjects, %d columns",
        len(df),
        df[Oasis2Columns.SUBJECT_ID].nunique(),
        len(df.columns),
    )
    return df


def load_oasis2_auto(
    config: dict | None = None,
    extra_search_roots: list[str] | None = None,
) -> pd.DataFrame:
    """
    Discover and load OASIS-2 automatically, using config for hints.

    Args:
        config:             Optional dataset config dict (from oasis2_kaggle.yaml).
        extra_search_roots: Additional directories to search.

    Returns:
        Loaded, validated DataFrame.

    Raises:
        FileNotFoundError: If no CSV is found anywhere.
    """
    search_roots: list[str | Path] = list(extra_search_roots or [])

    # Add config-specified path as a search hint
    if config and "local_path" in config:
        hint = Path(config["local_path"])
        if hint.exists():
            return load_oasis2(hint)
        # If the config path doesn't exist, add its parent as a search root
        search_roots.append(hint.parent)

    found = discover_dataset(search_roots=search_roots)
    if found is None:
        raise FileNotFoundError(
            "oasis_longitudinal.csv was not found in any of the search locations:\n"
            "  - config['local_path'] (if specified)\n"
            "  - /kaggle/input/** (Kaggle environment)\n"
            "  - data/raw/ (local)\n\n"
            "Please place the file at data/raw/oasis_longitudinal.csv "
            "or attach the Kaggle dataset 'jboysen/mri-and-alzheimers'."
        )

    return load_oasis2(found)
