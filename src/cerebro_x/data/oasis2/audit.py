"""OASIS-2 dataset audit.

Produces a comprehensive audit report of the loaded DataFrame:
- Shape and basic statistics
- Column types and missingness
- Subject and visit structure
- Group and CDR distributions
- MMSE statistics
- Progression indicators (CDR change across visits)
- Comparison to verified ground-truth facts

Run via: python scripts/audit_oasis2.py --config configs/datasets/oasis2_kaggle.yaml
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cerebro_x.data.schemas import Oasis2Columns, VerifiedDatasetFacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """Complete audit report for the OASIS-2 dataset."""

    # Basic shape
    total_rows: int = 0
    total_columns: int = 0
    unique_subjects: int = 0

    # Visit structure
    visits_per_subject: dict = field(default_factory=dict)   # {n_visits: count}
    visit_stats: dict = field(default_factory=dict)           # min, max, mean

    # Missingness (% missing per column)
    missingness: dict = field(default_factory=dict)

    # Distributions
    group_distribution: dict = field(default_factory=dict)
    cdr_distribution: dict = field(default_factory=dict)
    mmse_stats: dict = field(default_factory=dict)

    # Progression indicators
    subjects_with_changing_cdr: int = 0
    cdr_change_distribution: dict = field(default_factory=dict)  # change magnitude counts

    # Comparison to verified facts
    verified_comparison: dict = field(default_factory=dict)

    # Warnings raised during audit
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------

def run_audit(df: pd.DataFrame) -> AuditReport:
    """
    Run a complete audit on a loaded OASIS-2 DataFrame.

    Args:
        df: DataFrame loaded and validated by loader.load_oasis2().

    Returns:
        AuditReport with all computed fields.
    """
    report = AuditReport()
    expected = VerifiedDatasetFacts()

    # ── Shape ──────────────────────────────────────────────────────────────
    report.total_rows = len(df)
    report.total_columns = len(df.columns)
    report.unique_subjects = df[Oasis2Columns.SUBJECT_ID].nunique()

    # ── Missingness ────────────────────────────────────────────────────────
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    report.missingness = missing_pct[missing_pct > 0].to_dict()
    if not report.missingness:
        report.missingness["_note"] = "No missing values detected"

    # ── Visit structure ─────────────────────────────────────────────────────
    visit_counts = df.groupby(Oasis2Columns.SUBJECT_ID).size()
    vc_dist = visit_counts.value_counts().sort_index()
    report.visits_per_subject = {int(k): int(v) for k, v in vc_dist.items()}
    report.visit_stats = {
        "min": int(visit_counts.min()),
        "max": int(visit_counts.max()),
        "mean": float(round(visit_counts.mean(), 2)),
        "total_visits": int(visit_counts.sum()),
    }

    # ── Group distribution ──────────────────────────────────────────────────
    group_dist = df[Oasis2Columns.GROUP].value_counts()
    report.group_distribution = {str(k): int(v) for k, v in group_dist.items()}

    # ── CDR distribution ────────────────────────────────────────────────────
    cdr_dist = df[Oasis2Columns.CDR].value_counts().sort_index()
    report.cdr_distribution = {str(float(k)): int(v) for k, v in cdr_dist.items()}

    # ── MMSE statistics ─────────────────────────────────────────────────────
    mmse = df[Oasis2Columns.MMSE].dropna()
    report.mmse_stats = {
        "count": int(mmse.count()),
        "missing": int(df[Oasis2Columns.MMSE].isnull().sum()),
        "mean": float(round(mmse.mean(), 2)),
        "std": float(round(mmse.std(), 2)),
        "min": float(mmse.min()),
        "max": float(mmse.max()),
        "25th_pct": float(mmse.quantile(0.25)),
        "median": float(mmse.median()),
        "75th_pct": float(mmse.quantile(0.75)),
    }

    # ── CDR progression ─────────────────────────────────────────────────────
    def _cdr_changes(group: pd.DataFrame) -> bool:
        return group[Oasis2Columns.CDR].nunique() > 1

    subj_groups = df.groupby(Oasis2Columns.SUBJECT_ID)
    subjects_changing = subj_groups.filter(_cdr_changes)[Oasis2Columns.SUBJECT_ID].nunique()
    report.subjects_with_changing_cdr = int(subjects_changing)

    # CDR change magnitude across consecutive visits
    cdr_deltas: list[float] = []
    for _, grp in subj_groups:
        grp = grp.sort_values(Oasis2Columns.VISIT)
        deltas = grp[Oasis2Columns.CDR].diff().dropna()
        cdr_deltas.extend(deltas.tolist())
    delta_series = pd.Series(cdr_deltas)
    report.cdr_change_distribution = delta_series.value_counts().sort_index().to_dict()
    report.cdr_change_distribution = {
        str(float(k)): int(v) for k, v in report.cdr_change_distribution.items()
    }

    # ── Comparison to verified facts ─────────────────────────────────────────
    comparisons: dict[str, Any] = {}

    def _compare(key: str, actual, expected_val, tol: float = 0):
        match = (abs(actual - expected_val) <= tol) if isinstance(actual, (int, float)) else (actual == expected_val)
        comparisons[key] = {"actual": actual, "expected": expected_val, "match": match}
        if not match:
            report.warnings.append(
                f"MISMATCH — {key}: got {actual}, expected {expected_val}"
            )

    _compare("total_rows", report.total_rows, expected.total_rows)
    _compare("total_columns", report.total_columns, expected.total_columns)
    _compare("unique_subjects", report.unique_subjects, expected.unique_subjects)
    _compare(
        "subjects_with_changing_cdr",
        report.subjects_with_changing_cdr,
        expected.subjects_with_changing_cdr,
    )
    _compare("mmse_count", report.mmse_stats["count"], expected.mmse_count)

    report.verified_comparison = comparisons

    if report.warnings:
        logger.warning(
            "Audit found %d discrepancy(ies) vs. verified facts:\n%s",
            len(report.warnings),
            "\n".join(report.warnings),
        )
    else:
        logger.info("All audit checks passed — dataset matches verified facts.")

    return report


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_cdr_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    cdr_counts = df[Oasis2Columns.CDR].value_counts().sort_index()
    ax.bar([str(x) for x in cdr_counts.index], cdr_counts.values, color="#4A90D9")
    ax.set_xlabel("CDR Score")
    ax.set_ylabel("Record Count")
    ax.set_title("CDR Distribution — OASIS-2 (all visits)")
    for i, v in enumerate(cdr_counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("CDR distribution plot saved: %s", output_path)


def plot_mmse_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    mmse = df[Oasis2Columns.MMSE].dropna()
    ax.hist(mmse, bins=20, color="#5BAD72", edgecolor="white")
    ax.axvline(mmse.mean(), color="red", linestyle="--", label=f"Mean={mmse.mean():.1f}")
    ax.set_xlabel("MMSE Score")
    ax.set_ylabel("Frequency")
    ax.set_title("MMSE Distribution — OASIS-2 (all visits)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("MMSE distribution plot saved: %s", output_path)


def plot_visit_distribution(df: pd.DataFrame, output_path: Path) -> None:
    visit_counts = df.groupby(Oasis2Columns.SUBJECT_ID).size().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(x) for x in visit_counts.index], visit_counts.values, color="#E8884A")
    ax.set_xlabel("Number of Visits per Subject")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Visit Count Distribution — OASIS-2")
    for i, v in enumerate(visit_counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Visit distribution plot saved: %s", output_path)


def plot_missingness(df: pd.DataFrame, output_path: Path) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("No missing values — skipping missingness plot.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(missing.index.tolist(), (missing / len(df) * 100).tolist(), color="#C0504D")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Column Missingness — OASIS-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Missingness plot saved: %s", output_path)


# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------

def save_audit_report(report: AuditReport, output_dir: Path) -> None:
    """
    Save audit report JSON and all plots to output_dir.

    Args:
        report:     AuditReport instance.
        output_dir: Directory to write outputs (created if missing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    logger.info("Audit report saved: %s", report_path)
