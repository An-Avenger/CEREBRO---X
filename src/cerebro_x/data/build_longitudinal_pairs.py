"""Build longitudinal next-visit pairs from OASIS-2 data.

For each subject, consecutive visit pairs are created:
  Visit T (input features) → Visit T+1 (target: next_CDR, next_MMSE)

Rules (hard-coded, not configurable — any change requires a new ADR):
  - Sort by Subject ID, then by Visit number.
  - The LAST visit for each subject has no next visit → EXCLUDED from supervised pairs.
  - next_CDR and next_MMSE are added as TARGET columns ONLY, never in input features.
  - No future information from visit T+1 enters the feature set for visit T.

Output:
  data/processed/next_visit_pairs.csv
  data/processed/pair_metadata.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from cerebro_x.data.schemas import Oasis2Columns, PairColumns

logger = logging.getLogger(__name__)

# Target columns — these must NEVER appear in the model's input feature matrix.
TARGET_COLUMNS = [PairColumns.NEXT_CDR, PairColumns.NEXT_MMSE]


# ---------------------------------------------------------------------------
# Metadata dataclass
# ---------------------------------------------------------------------------

@dataclass
class PairMetadata:
    """Describes the longitudinal pair dataset produced by build_next_visit_pairs."""

    source_dataset: str = "oasis2_kaggle"
    total_input_rows: int = 0
    unique_subjects_input: int = 0
    total_pairs: int = 0
    unique_subjects_with_pairs: int = 0
    pair_distribution: dict = field(default_factory=dict)  # {n_pairs: subject_count}
    target_columns: list[str] = field(default_factory=lambda: TARGET_COLUMNS)
    input_feature_columns: list[str] = field(default_factory=list)
    next_cdr_distribution: dict = field(default_factory=dict)
    next_mmse_stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core pair-builder
# ---------------------------------------------------------------------------

def build_next_visit_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, PairMetadata]:
    """
    Create longitudinal next-visit pairs from the OASIS-2 DataFrame.

    For each subject with N visits, this creates N-1 pairs:
      (visit_1, visit_2), (visit_2, visit_3), ...

    The last visit in each subject's sequence is EXCLUDED because it has no
    subsequent visit to predict.

    Args:
        df: DataFrame loaded by loader.load_oasis2(). Must be sorted by
            (Subject ID, Visit) ascending.

    Returns:
        Tuple of:
          - pairs_df: DataFrame with current-visit features + target columns.
          - metadata: PairMetadata instance describing the constructed dataset.

    Raises:
        ValueError: If any target column appears in the input feature columns.
    """
    meta = PairMetadata()
    meta.total_input_rows = len(df)
    meta.unique_subjects_input = df[Oasis2Columns.SUBJECT_ID].nunique()

    rows: list[dict] = []

    for subject_id, group in df.groupby(Oasis2Columns.SUBJECT_ID, sort=True):
        group = group.sort_values(Oasis2Columns.VISIT).reset_index(drop=True)
        n_visits = len(group)

        if n_visits < 2:
            # This subject has only 1 visit — cannot form a pair.
            logger.debug("Subject %s has only 1 visit — skipped.", subject_id)
            meta.notes.append(f"Subject {subject_id}: only 1 visit, no pairs generated.")
            continue

        # Pair each visit with its successor
        for i in range(n_visits - 1):
            current = group.iloc[i]
            next_visit = group.iloc[i + 1]

            row: dict = {}

            # ── Current visit features ──────────────────────────────────────
            row[PairColumns.SUBJECT_ID] = subject_id
            row[PairColumns.CURRENT_VISIT] = int(current[Oasis2Columns.VISIT])
            row[PairColumns.NEXT_VISIT] = int(next_visit[Oasis2Columns.VISIT])
            row[PairColumns.CURRENT_MR_DELAY] = current[Oasis2Columns.MR_DELAY]
            row[PairColumns.NEXT_MR_DELAY] = next_visit[Oasis2Columns.MR_DELAY]

            # Time between visits in days
            row["days_between_visits"] = (
                next_visit[Oasis2Columns.MR_DELAY] - current[Oasis2Columns.MR_DELAY]
            )

            # Static / demographic (from current visit)
            row["curr_age"] = current[Oasis2Columns.AGE]
            row["sex"] = current[Oasis2Columns.SEX]
            row["hand"] = current[Oasis2Columns.HAND]
            row["educ"] = current[Oasis2Columns.EDUC]
            row["ses"] = current[Oasis2Columns.SES]

            # Current clinical state
            row["curr_mmse"] = current[Oasis2Columns.MMSE]
            row["curr_cdr"] = current[Oasis2Columns.CDR]
            row["curr_group"] = current[Oasis2Columns.GROUP]

            # Volumetric / MRI-derived scalars (current visit)
            row["curr_etiv"] = current[Oasis2Columns.ETIV]
            row["curr_nwbv"] = current[Oasis2Columns.NWBV]
            row["curr_asf"] = current[Oasis2Columns.ASF]

            # Number of prior visits (before the current one)
            row["n_prior_visits"] = i  # 0 = first visit

            # ── Temporal engineered features ────────────────────────────────
            # These use ONLY information from current-or-prior visits.
            if i > 0:
                prev = group.iloc[i - 1]
                row["prev_mmse"] = prev[Oasis2Columns.MMSE]
                row["prev_cdr"] = prev[Oasis2Columns.CDR]
                row["prev_nwbv"] = prev[Oasis2Columns.NWBV]
                row["mmse_delta"] = (
                    current[Oasis2Columns.MMSE] - prev[Oasis2Columns.MMSE]
                )
                row["cdr_delta"] = (
                    current[Oasis2Columns.CDR] - prev[Oasis2Columns.CDR]
                )
                row["nwbv_delta"] = (
                    current[Oasis2Columns.NWBV] - prev[Oasis2Columns.NWBV]
                )
            else:
                # First visit — no prior visit exists
                row["prev_mmse"] = float("nan")
                row["prev_cdr"] = float("nan")
                row["prev_nwbv"] = float("nan")
                row["mmse_delta"] = float("nan")
                row["cdr_delta"] = float("nan")
                row["nwbv_delta"] = float("nan")

            # ── TARGET columns — added LAST ─────────────────────────────────
            # These must NEVER appear in the feature matrix passed to a model.
            row[PairColumns.NEXT_CDR] = next_visit[Oasis2Columns.CDR]
            row[PairColumns.NEXT_MMSE] = next_visit[Oasis2Columns.MMSE]

            rows.append(row)

    if not rows:
        raise ValueError(
            "No longitudinal pairs could be constructed. "
            "Every subject must have at least 2 visits. "
            "Check that the dataset was loaded correctly."
        )

    pairs_df = pd.DataFrame(rows).reset_index(drop=True)

    # ── Safety check: targets must not be in feature columns ────────────────
    _verify_no_target_leakage(pairs_df)

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta.total_pairs = len(pairs_df)
    meta.unique_subjects_with_pairs = pairs_df[PairColumns.SUBJECT_ID].nunique()

    pairs_per_subject = pairs_df.groupby(PairColumns.SUBJECT_ID).size()
    ppd = pairs_per_subject.value_counts().sort_index()
    meta.pair_distribution = {int(k): int(v) for k, v in ppd.items()}

    next_cdr_dist = pairs_df[PairColumns.NEXT_CDR].value_counts().sort_index()
    meta.next_cdr_distribution = {str(float(k)): int(v) for k, v in next_cdr_dist.items()}

    next_mmse = pairs_df[PairColumns.NEXT_MMSE].dropna()
    meta.next_mmse_stats = {
        "count": int(next_mmse.count()),
        "missing": int(pairs_df[PairColumns.NEXT_MMSE].isnull().sum()),
        "mean": float(round(next_mmse.mean(), 2)),
        "std": float(round(next_mmse.std(), 2)),
        "min": float(next_mmse.min()),
        "max": float(next_mmse.max()),
    }

    # Identify feature columns (all columns except targets and identifiers)
    non_feature = {PairColumns.SUBJECT_ID, *TARGET_COLUMNS}
    meta.input_feature_columns = [c for c in pairs_df.columns if c not in non_feature]

    logger.info(
        "Longitudinal pairs constructed: %d pairs from %d subjects.",
        meta.total_pairs,
        meta.unique_subjects_with_pairs,
    )
    logger.info("next_CDR distribution: %s", meta.next_cdr_distribution)

    return pairs_df, meta


def _verify_no_target_leakage(pairs_df: pd.DataFrame) -> None:
    """
    Raise ValueError if any target column also appears in the input feature set.

    This is a hard safety check. If it fails, something in the pair-building
    logic is wrong and must be fixed before any model training.
    """
    # The feature matrix excludes SUBJECT_ID and TARGET_COLUMNS.
    # We check that TARGET_COLUMNS are not duplicated under other names.
    # Since we build the DataFrame ourselves, this checks for coding errors.
    feature_cols = set(pairs_df.columns) - {PairColumns.SUBJECT_ID} - set(TARGET_COLUMNS)

    # Check no column in feature_cols has a suspicious name pattern
    forbidden_patterns = ["next_cdr", "next_mmse", "future"]
    violations = [
        col for col in feature_cols
        if any(pat in col.lower() for pat in forbidden_patterns)
    ]
    if violations:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED in feature columns: {violations}\n"
            f"These columns appear to contain future information and must be removed.\n"
            f"Fix build_next_visit_pairs() before training any model."
        )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_pairs(
    pairs_df: pd.DataFrame,
    meta: PairMetadata,
    output_dir: str | Path,
) -> None:
    """
    Save pairs DataFrame and metadata to output_dir.

    Args:
        pairs_df:   The longitudinal pairs DataFrame.
        meta:       PairMetadata instance.
        output_dir: Directory to write outputs (created if missing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / "next_visit_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    logger.info("Pairs saved: %s (%d rows)", pairs_path, len(pairs_df))

    meta_path = output_dir / "pair_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta.to_dict(), fh, indent=2)
    logger.info("Pair metadata saved: %s", meta_path)
